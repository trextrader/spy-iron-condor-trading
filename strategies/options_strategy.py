# strategies/options_strategy.py - Iron Condor with MTF Intelligence + CondorBrain
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import datetime as dt
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligence.fuzzy_engine import get_fuzzy_consensus
from intelligence.regime_filter import check_liquidity_gate

# CondorBrain Neural Integration (Optional - falls back to rule-based if not available)
try:
    from intelligence.condor_brain import CondorBrainEngine, CondorSignal, HAS_MAMBA
    # Auto-discover any condor_brain model
    import glob
    _model_candidates = glob.glob("models/condor_brain*.pth")
    HAS_CONDOR_BRAIN = HAS_MAMBA and len(_model_candidates) > 0
except (ImportError, OSError):
    HAS_CONDOR_BRAIN = False
    CondorBrainEngine = None
    CondorSignal = None


from core.dto import OptionQuote, IronCondorLegs

# === Core Types ===
def lag_weighted_edge(edge: float, iv_conf: float, align_mode: str, cfg: Any) -> float:
    """
    Reduce VRP edge as options lag increases.

    If cfg.vrp_lag_weighting is False -> returns edge.

    Modes:
      - multiply: edge *= iv_conf * scale
      - subtract: edge -= (1 - iv_conf) * scale

    align_mode == "stale" => returns 0.0 (chain treated as empty under cutoff policies)
    """
    if not bool(getattr(cfg, "vrp_lag_weighting", True)):
        return edge
    scale = float(getattr(cfg, "vrp_lag_penalty_scale", 1.0))
    modew = str(getattr(cfg, "vrp_lag_weight_mode", "multiply"))
    iv_conf = float(iv_conf) if iv_conf is not None else 1.0
    iv_conf = max(0.0, min(1.0, iv_conf))

    if align_mode == "stale":
        return 0.0

    if modew == "subtract":
        return edge - (1.0 - iv_conf) * scale
    return edge * (iv_conf * scale)



class OptionsStrategy:
    """
    Base class / interface for options strategies.
    """
    def __init__(self, cfg: Any):
        self.cfg = cfg

    def alignment_policy(self) -> dict[str, Any]:
        """
        Strategy-level override for stale options handling.

        Return dict:
          - policy: "hard_cutoff" | "decay_only" | "decay_then_cutoff"
          - max_lag_sec: int | None (None => use config per-symbol/default)
          - allow_trade_without_chain: bool

        Defaults are conservative for multi-leg options strategies.
        """
        return {
            "policy": getattr(self.cfg, "lag_policy_default", "decay_then_cutoff"),
            "max_lag_sec": None,
            "allow_trade_without_chain": False,
        }

@dataclass
class PositionState:
    id: str
    legs: IronCondorLegs
    open_time: dt.datetime
    credit_received: float
    quantity: int
    adjustments_done: Dict[str, int]
    mtf_consensus: float = 0.5  # Store entry consensus

# === Pricing Utilities ===
def calc_spread_credit(short: OptionQuote, long: OptionQuote) -> float:
    return max(0.0, short.mid - long.mid)

def calc_condor_credit(legs: IronCondorLegs) -> float:
    call_credit = calc_spread_credit(legs.short_call, legs.long_call)
    put_credit = calc_spread_credit(legs.short_put, legs.long_put)
    return call_credit + put_credit

def max_loss(wing_width: float, total_credit: float) -> float:
    return max(0.0, wing_width - total_credit)

# === Strike Selection ===
def within_delta_band(opt: OptionQuote, target_low: float, target_high: float) -> bool:
    d = abs(opt.delta)
    return target_low <= d <= target_high

def pick_expiration(expirations: List[dt.date], cfg, today: dt.date) -> Optional[dt.date]:
    candidates = [e for e in expirations if cfg.dte_min <= (e - today).days <= cfg.dte_max]
    return sorted(candidates)[0] if candidates else None

def calculate_skew(chain: List[OptionQuote]) -> float:
    """
    Calculate 25-Delta Skew: (Put IV - Call IV)
    Positive => Puts are more expensive (Bearish Sentiment / Crash Protection)
    """
    puts_25 = [q for q in chain if not q.is_call]
    calls_25 = [q for q in chain if q.is_call]
    
    if not puts_25 or not calls_25:
        return 0.0
        
    p25 = min(puts_25, key=lambda q: abs(abs(q.delta) - 0.25))
    c25 = min(calls_25, key=lambda q: abs(abs(q.delta) - 0.25))
    
    return p25.iv - c25.iv

def nearest_by_delta(quotes: List[OptionQuote], is_call: bool, 
                     target_low: float, target_high: float,
                     atm_iv: float = None, market_skew: float = 0.0) -> Optional[OptionQuote]:
    """
    Select best strike by delta, penalizing those with 'bad' skew 
    (buying high IV or selling low IV relative to ATM).
    """
    candidates = [q for q in quotes if q.is_call == is_call 
                  and within_delta_band(q, target_low, target_high)]
    
    if not candidates:
        # Fallback: Find closest if within reasonable tolerance (e.g. +/- 0.05)
        # useful for sparse data backtests
        typed_quotes = [q for q in quotes if q.is_call == is_call]
        if typed_quotes:
            closest = min(typed_quotes, key=lambda q: min(abs(abs(q.delta) - target_low), abs(abs(q.delta) - target_high)))
            d = abs(closest.delta)
            # Accept if close enough (Backtest Data Patch)
            if (target_low - 0.05) <= d <= (target_high + 0.05):
                candidates = [closest]
            else:
                # Debugging: What deltas ARE available?
                all_deltas = [abs(q.delta) for q in typed_quotes]
                if len(all_deltas) > 0:
                     print(f"      [Debug] No { 'Call' if is_call else 'Put' } in [{target_low}, {target_high}]. Available abs(delta): {min(all_deltas):.4f} to {max(all_deltas):.4f}")
                return None
        else:
             return None

    center_delta = (target_low + target_high) / 2.0
    
    # Skew Logic: If ATM IV is provided, prefer selling strikes with IV > ATM (rich premium)
    best_candidate = None
    best_score = float('inf')
    
    for q in candidates:
        delta_dist = abs(abs(q.delta) - center_delta)
        
        # Skew Penalty: Penalize selling low IV
        iv_penalty = 0.0
        if atm_iv and q.iv < atm_iv:
            iv_penalty = (atm_iv - q.iv) * 100.0
            
        # Directional Skew Adjustment
        # If Market Skew is Positive (Puts > Calls), Selling Puts is lucrative but risky.
        # If we are selling Puts (is_call=False) and Skew is High (> 5%), 
        # we might favor lower delta (safer) to account for crash risk.
        skew_align_penalty = 0.0
        if not is_call and market_skew > 0.05:
            # High Put Skew: Penalize higher deltas (closer to money)
            if abs(q.delta) > center_delta: 
                skew_align_penalty = 2.0
                
        score = delta_dist + (iv_penalty * 0.1) + skew_align_penalty
        
        if score < best_score:
            best_score = score
            best_candidate = q
            
    return best_candidate

def pick_long_by_width(chain: List[OptionQuote], short_leg: OptionQuote, 
                       is_call: bool, wing_width: float) -> Optional[OptionQuote]:
    target_strike = short_leg.strike + wing_width if is_call else short_leg.strike - wing_width
    candidates = [q for q in chain if q.is_call == is_call 
                  and q.expiration == short_leg.expiration
                  and q.strike != short_leg.strike] # Ensure distinct strike
    
    # Enforce OTM direction (Calls > Short, Puts < Short)
    if is_call:
        candidates = [q for q in candidates if q.strike > short_leg.strike]
    else:
        candidates = [q for q in candidates if q.strike < short_leg.strike]
        
    if not candidates:
        return None
    return sorted(candidates, key=lambda q: abs(q.strike - target_strike))[0]

# === Wing Width Determination ===
def regime_wing_width(cfg, iv_rank: float, vix: float, regime: str = "NEUTRAL") -> float:
    """Dynamic wing width based on volatility regime"""
    base = cfg.wing_width_min
    
    # Regime-based overrides
    if regime == "VOLATILE":
        return cfg.wing_width_max
        
    if regime == "TRENDING" or regime == "TRENDING_WEAK":
        # In trending market, wide wings prevent early stopping on tested side?
        # Or actually, maybe we shouldn't trade. Assuming we trade, go wide.
        return min(cfg.wing_width_max, base + cfg.width_widen_increment)
        
    if regime == "RANGING":
        return base

    # Fallback VIX/IVR Logic
    high_vol = (iv_rank >= cfg.regime_iv_rank_widen) or (vix >= cfg.regime_vix_widen)
    if high_vol:
        return min(cfg.wing_width_max, base + cfg.width_widen_increment)
    return base

def optimize_wing_width(cfg, historical_credit_curve: Dict[float, float]) -> float:
    """Select optimal width from historical performance"""
    feasible = [(w, r) for w, r in historical_credit_curve.items() 
                if r >= cfg.min_credit_to_width 
                and cfg.wing_width_min <= w <= cfg.wing_width_max]
    if not feasible:
        return cfg.wing_width_min
    feasible.sort(key=lambda t: (-t[1], t[0]))  # Best ratio, then smallest width
    return feasible[0][0]

# === Build Condor ===
def build_condor(chain: List[OptionQuote], cfg, today: dt.date, 
                 spot: float, base_wing_width: float) -> Optional[IronCondorLegs]:
    """Construct iron condor from option chain"""
    # Estimate ATM IV and Skew
    atm_iv = None
    market_skew = calculate_skew(chain)
    
    calls = [q for q in chain if q.is_call]
    if calls:
        # Find closest to spot
        atm_call = min(calls, key=lambda q: abs(q.strike - spot))
        atm_iv = atm_call.iv
    
    short_call = nearest_by_delta(chain, True, cfg.target_short_delta_low, cfg.target_short_delta_high, atm_iv, market_skew)
    short_put = nearest_by_delta(chain, False, cfg.target_short_delta_low, cfg.target_short_delta_high, atm_iv, market_skew)
    
    if not short_call or not short_put:
        # Debug why it failed
        if not short_call: print(f"      [Build Fail] No Short Call in delta range [{cfg.target_short_delta_low}, {cfg.target_short_delta_high}]")
        if not short_put: print(f"      [Build Fail] No Short Put in delta range [{cfg.target_short_delta_low}, {cfg.target_short_delta_high}]")
        return None
    
    long_call = pick_long_by_width(chain, short_call, True, base_wing_width)
    long_put = pick_long_by_width(chain, short_put, False, base_wing_width)
    
    if not long_call or not long_put:
        if not long_call: print(f"      [Build Fail] No Long Call at width {base_wing_width} for Short Call {short_call.strike}")
        if not long_put: print(f"      [Build Fail] No Long Put at width {base_wing_width} for Short Put {short_put.strike}")
        return None
    
    # Calculate net credit and max loss
    # Credit = (short_call.mid + short_put.mid) - (long_call.mid + long_put.mid)
    net_credit = (short_call.mid + short_put.mid) - (long_call.mid + long_put.mid)
    
    # Max loss for iron condor = wing width - net credit
    call_width = abs(long_call.strike - short_call.strike)
    put_width = abs(short_put.strike - long_put.strike)
    wing_width = max(call_width, put_width)
    max_loss = (wing_width * 100) - (net_credit * 100)  # Per contract
    
    return IronCondorLegs(
        short_call=short_call,
        long_call=long_call,
        short_put=short_put,
        long_put=long_put,
        net_credit=net_credit,
        max_loss=max_loss
    )

# === Neural-Driven Build Condor (CondorBrain) ===
_condor_brain_engine = None

def get_condor_brain() -> Optional[Any]:
    """Lazy-load CondorBrain engine."""
    global _condor_brain_engine
    if not HAS_CONDOR_BRAIN:
        return None
    if _condor_brain_engine is None:
        try:
            _condor_brain_engine = CondorBrainEngine()
        except Exception as e:
            print(f"[CondorBrain] Failed to load: {e}")
            return None
    return _condor_brain_engine

def build_condor_neural(
    chain: List[OptionQuote], 
    cfg, 
    today: dt.date, 
    spot: float,
    features: np.ndarray = None
) -> Tuple[Optional[IronCondorLegs], Optional[Any]]:
    """
    Build Iron Condor using CondorBrain neural predictions.
    
    Falls back to rule-based build_condor if CondorBrain unavailable.
    
    Args:
        chain: List of OptionQuote objects
        cfg: Configuration
        today: Current date
        spot: Current spot price
        features: Pre-computed feature array (lookback, 24) for CondorBrain
        
    Returns:
        (IronCondorLegs, CondorSignal) or (None, None)
    """
    brain = get_condor_brain()
    
    if brain is None or features is None:
        # Fallback to rule-based
        legs = build_condor(chain, cfg, today, spot, cfg.wing_width_min)
        return legs, None
    
    # Get neural prediction
    try:
        signal: CondorSignal = brain.predict(features)
    except Exception as e:
        print(f"[CondorBrain] Prediction failed: {e}")
        legs = build_condor(chain, cfg, today, spot, cfg.wing_width_min)
        return legs, None
    
    # Check if signal is valid
    if not signal.is_valid_trade(min_confidence=0.5, min_pop=0.4):
        print(f"[CondorBrain] Signal rejected: conf={signal.confidence:.2f}, pop={signal.prob_profit:.2f}")
        return None, signal
    
    # Validate strikes against predicted price range
    call_safe, put_safe = signal.strikes_are_safe(spot)
    if not call_safe or not put_safe:
        print(f"[CondorBrain] Price trajectory breach risk: call_safe={call_safe}, put_safe={put_safe}")
        # Widen the offsets if trajectory predicts breach
        if not call_safe:
            signal.short_call_offset = min(5.0, signal.short_call_offset + 0.5)
        if not put_safe:
            signal.short_put_offset = min(5.0, signal.short_put_offset + 0.5)
    
    # Convert neural offsets to strikes
    neural_short_call = spot * (1 + signal.short_call_offset / 100)
    neural_short_put = spot * (1 - signal.short_put_offset / 100)
    neural_wing = signal.wing_width
    
    # Find actual chain strikes closest to neural targets
    calls = [q for q in chain if q.is_call]
    puts = [q for q in chain if not q.is_call]
    
    if not calls or not puts:
        print("[CondorBrain] Empty chain")
        return None, signal
    
    # Find closest strikes
    short_call = min(calls, key=lambda q: abs(q.strike - neural_short_call))
    short_put = min(puts, key=lambda q: abs(q.strike - neural_short_put))
    
    # Pick long legs by neural wing width
    long_call = pick_long_by_width(chain, short_call, True, neural_wing)
    long_put = pick_long_by_width(chain, short_put, False, neural_wing)
    
    if not long_call or not long_put:
        print(f"[CondorBrain] Could not find wings at width {neural_wing}")
        # Fall back to minimum width
        long_call = pick_long_by_width(chain, short_call, True, cfg.wing_width_min)
        long_put = pick_long_by_width(chain, short_put, False, cfg.wing_width_min)
        
    if not long_call or not long_put:
        return None, signal
    
    # Calculate credit and max loss
    net_credit = (short_call.mid + short_put.mid) - (long_call.mid + long_put.mid)
    
    call_width = abs(long_call.strike - short_call.strike)
    put_width = abs(short_put.strike - long_put.strike)
    wing_width = max(call_width, put_width)
    max_loss = (wing_width * 100) - (net_credit * 100)
    
    legs = IronCondorLegs(
        short_call=short_call,
        long_call=long_call,
        short_put=short_put,
        long_put=long_put,
        net_credit=net_credit,
        max_loss=max_loss
    )
    
    print(f"[CondorBrain] Neural IC: SC={short_call.strike}, SP={short_put.strike}, "
          f"width={neural_wing:.1f}, conf={signal.confidence:.2f}, pop={signal.prob_profit:.2f}")
    
    return legs, signal

# === Entry Filters ===
def can_enter_trade(broker, cfg) -> Tuple[bool, str]:
    """Check if we can open a new position"""
    ivr = broker.get_iv_rank(cfg.underlying)
    vix = broker.get_vix()
    
    if ivr < cfg.iv_rank_min:
        return False, f"Skip: IV Rank {ivr:.1f} < {cfg.iv_rank_min}"
    
    if vix > cfg.vix_threshold:
        return False, f"Skip: VIX {vix:.1f} > {cfg.vix_threshold}"
    
    acct = broker.get_account_metrics()
    positions = broker.get_open_positions(cfg.underlying)
    
    if len(positions) >= cfg.max_positions:
        return False, "Skip: max positions reached"
    
    positions_value = acct.get("positions_value", 0.0)
    equity = max(acct.get("equity", 1.0), 1.0)
    
    if positions_value / equity >= cfg.max_portfolio_alloc:
        return False, "Skip: portfolio allocation limit"
    
    return True, "OK"

def check_mtf_alignment(cfg, mtf_snapshot) -> Tuple[bool, str, float]:
    """Check if MTF consensus allows entry"""
    if not cfg.use_mtf_filter or mtf_snapshot is None:
        return True, "MTF filter disabled", 0.5
    
    consensus = get_fuzzy_consensus(mtf_snapshot)
    
    # Iron condors profit from range-bound markets (neutral consensus)
    if consensus < cfg.mtf_consensus_min or consensus > cfg.mtf_consensus_max:
        return False, f"Skip: MTF consensus {consensus:.2f} outside range [{cfg.mtf_consensus_min}, {cfg.mtf_consensus_max}]", consensus
    
    return True, f"MTF consensus {consensus:.2f} favorable for range", consensus

def calculate_breach_probability(delta: float) -> float:
    """
    Estimate probability of strike breach (ITM) at expiration.
    Approximation: Prob(ITM) ~= abs(Delta).
    For more complex models (Touch Prob), we typically multiply by 2.
    Here we focus on Expiration Prob.
    """
    return abs(delta)

def validate_pricing_and_risk(legs: IronCondorLegs, width: float, 
                              cfg, acct_equity: float) -> Tuple[bool, str, float]:
    """Validate credit and risk constraints"""
    credit = calc_condor_credit(legs)
    credit_ratio = credit / width
    
    if credit_ratio < cfg.min_credit_to_width:
        return False, f"Reject: credit ratio {credit_ratio:.2f} < {cfg.min_credit_to_width}", credit
    
    mloss = max_loss(width, credit)
    
    if mloss > cfg.max_account_risk_per_trade * acct_equity:
        return False, f"Reject: max loss ${mloss:.2f} exceeds per-trade risk", credit
        
    # Probabilistic Filter: Reject if leg delta is significantly drifted
    # Strategy picks 0.12-0.30. If we somehow got > 0.35, reject.
    prob_call = calculate_breach_probability(legs.short_call.delta)
    prob_put = calculate_breach_probability(legs.short_put.delta)
    
    max_prob = 0.35
    if prob_call > max_prob:
        return False, f"Reject: Call Delta {prob_call:.2f} > {max_prob} (High Breach Prob)", credit
    if prob_put > max_prob:
        return False, f"Reject: Put Delta {prob_put:.2f} > {max_prob} (High Breach Prob)", credit
    
    return True, "OK", credit

# === Main Entry Function ===
from core.dto import TradeDecision

def generate_trade_signal(broker, cfg, today: dt.date,
                            historical_credit_curve: Optional[Dict[float, float]] = None,
                            quantity: int = 1,
                            mtf_snapshot=None,
                            liquidity_data=None) -> Tuple[Optional[TradeDecision], Optional[IronCondorLegs], float, float]:
    """
    Pure signal generation logic.
    Returns: (TradeDecision, Legs, Credit, MTF_Consensus) or (None, None, 0.0, 0.0)
    """
    # 1. Check Portfolio Limits
    ok, msg = can_enter_trade(broker, cfg)
    if not ok:
        print(f"[Entry Blocked] {msg}")
        return None, None, 0.0, 0.0
    
    # 2. MTF Alignment Check
    mtf_ok, mtf_msg, consensus = check_mtf_alignment(cfg, mtf_snapshot)
    if not mtf_ok:
        print(f"[Entry Blocked] {mtf_msg}")
        return None, None, 0.0, 0.0
    
    # 3. Liquidity Gate
    if cfg.use_liquidity_gate and liquidity_data is not None:
        if not check_liquidity_gate(liquidity_data, spread_estimate=1.0):
            print(f"[Entry Blocked] Failed liquidity gate")
            return None, None, 0.0, 0.0
    
    # 4. Get Market Context
    spot = broker.get_spot(cfg.underlying)
    ivr = broker.get_iv_rank(cfg.underlying)
    vix = broker.get_vix()
    
    # 5. Determine Wing Width
    target_width = regime_wing_width(cfg, ivr, vix)
    
    if historical_credit_curve:
        optimized_width = optimize_wing_width(cfg, historical_credit_curve)
        target_width = max(target_width, optimized_width)  # Use wider of the two
    
    target_width = min(target_width, cfg.wing_width_max)
    
    # 6. Expiration Selection
    expirations = broker.get_expirations(cfg.underlying)
    exp = pick_expiration(expirations, cfg, today)
    if not exp:
        print("[Entry Blocked] No valid expiration in DTE window")
        return None, None, 0.0, 0.0
    
    # 7. Build Legs
    chain = broker.get_option_chain(cfg.underlying, exp)
    legs = build_condor(chain, cfg, today, spot, target_width)
    if not legs:
        print("[Entry Blocked] Unable to build condor legs")
        return None, None, 0.0, 0.0
    
    # 8. Validate Pricing & Risk
    acct = broker.get_account_metrics()
    equity = acct.get("equity", 0.0)
    valid, vmsg, credit = validate_pricing_and_risk(legs, target_width, cfg, equity)
    
    if not valid:
        print(f"[Entry Blocked] {vmsg}")
        return None, None, 0.0, 0.0
    
    # 9. Construct Decision (No Execution)
    decision = TradeDecision(
        symbol=cfg.underlying,
        should_trade=True,
        structure="iron_condor",
        bias="neutral",
        rationale={
            "mtf_consensus": consensus,
            "vix": vix,
            "ivr": ivr,
            "width": target_width,
            "credit": credit,
            "roi": (credit / target_width) if target_width > 0 else 0.0
        }
    )
    
    return decision, legs, credit, consensus

# === Position Management ===
def estimate_pnl(position: PositionState, current_quotes: IronCondorLegs) -> Tuple[float, float]:
    """Calculate unrealized P&L"""
    debit_call = max(0.0, current_quotes.long_call.mid - current_quotes.short_call.mid)
    debit_put = max(0.0, current_quotes.long_put.mid - current_quotes.short_put.mid)
    debit_total = debit_call + debit_put
    
    unreal = (position.credit_received - debit_total) * position.quantity
    max_prof = position.credit_received * position.quantity
    
    return unreal, max_prof

def breached_short_strike(spot: float, legs: IronCondorLegs) -> Optional[str]:
    """Check if price breached short strikes"""
    if spot >= legs.short_call.strike:
        return "call"
    if spot <= legs.short_put.strike:
        return "put"
    return None

def net_position_delta(legs: IronCondorLegs, quantity: int) -> float:
    """Calculate net portfolio delta"""
    net = (-legs.short_call.delta + legs.long_call.delta + 
           legs.short_put.delta - legs.long_put.delta) * quantity
    return net

def maybe_dynamic_delta_hedge(broker, cfg, pos: PositionState, current: IronCondorLegs) -> None:
    """Hedge portfolio delta if threshold exceeded"""
    nd = net_position_delta(current, pos.quantity)
    
    if abs(nd) >= cfg.dynamic_delta_hedge_threshold:
        units = int(round((abs(nd) / 0.01) * cfg.dynamic_delta_hedge_unit))
        if units > 0:
            side = "sell" if nd > 0 else "buy"
            broker.trade_shares(cfg.underlying, quantity=units, side=side)
            print(f"[Delta Hedge] {side.upper()} {units} shares (net delta: {nd:.3f})")

def manage_positions(broker, cfg, today: dt.date) -> None:
    """Manage open positions: exits, rolls, hedges"""
    positions = broker.get_open_positions(cfg.underlying)
    spot = broker.get_spot(cfg.underlying)
    
    for pos in positions:
        chain = broker.get_option_chain(cfg.underlying, pos.legs.short_call.expiration)
        
        # Match current quotes
        def match(q: OptionQuote) -> OptionQuote:
            matches = [x for x in chain if x.strike == q.strike and x.is_call == q.is_call]
            return matches[0] if matches else q
        
        current = IronCondorLegs(
            short_call=match(pos.legs.short_call),
            long_call=match(pos.legs.long_call),
            short_put=match(pos.legs.short_put),
            long_put=match(pos.legs.long_put)
        )
        
        # Calculate P&L
        unreal, max_prof = estimate_pnl(pos, current)
        
        # Exit Rule 1: Profit Target
        if unreal >= cfg.profit_take_pct * max_prof:
            print(f"[Exit] Profit target reached: ${unreal:.2f} (MTF entry: {pos.mtf_consensus:.2f})")
            broker.close_position(pos.id)
            continue
        
        # Exit Rule 2: DTE Threshold
        dte = (pos.legs.short_call.expiration - today).days
        if dte <= cfg.max_hold_days:
            print(f"[Exit] DTE threshold: {dte} days remaining")
            broker.close_position(pos.id)
            continue
        
        # Exit Rule 3: Stop Loss
        loss_threshold = -cfg.loss_close_multiple * pos.credit_received * pos.quantity
        if unreal <= loss_threshold:
            print(f"[Exit] Stop loss triggered: ${unreal:.2f}")
            broker.close_position(pos.id)
            continue
        
        # Adjustment: Roll Breached Side
        breach = breached_short_strike(spot, pos.legs)
        if breach and pos.adjustments_done[breach] == 0 and cfg.allow_one_adjustment_per_side:
            expirations = broker.get_expirations(cfg.underlying)
            later_exps = [e for e in expirations if e > pos.legs.short_call.expiration]
            
            if not later_exps:
                continue
            
            new_exp = later_exps[0]
            new_chain = broker.get_option_chain(cfg.underlying, new_exp)
            
            width_call = pos.legs.long_call.strike - pos.legs.short_call.strike
            width_put = pos.legs.short_put.strike - pos.legs.long_put.strike
            
            # Find new strikes
            def nearest_new(is_call: bool):
                center = (cfg.target_short_delta_low + cfg.target_short_delta_high) / 2.0
                candidates = [q for q in new_chain if q.is_call == is_call]
                if not candidates:
                    return None
                return sorted(candidates, key=lambda q: abs(abs(q.delta) - center))[0]
            
            if breach == "call":
                new_short = nearest_new(True)
                new_long = pick_long_by_width(new_chain, new_short, True, width_call) if new_short else None
            else:
                new_short = nearest_new(False)
                new_long = pick_long_by_width(new_chain, new_short, False, width_put) if new_short else None
            
            if new_short and new_long:
                addl_credit = max(0.0, new_short.mid - new_long.mid)
                print(f"[Roll] {breach.upper()} side breached, rolling for ${addl_credit:.2f} credit")
                broker.roll_vertical(pos.id, breach, new_short, new_long, limit_price=addl_credit)
        
        # Dynamic Delta Hedging
        maybe_dynamic_delta_hedge(broker, cfg, pos, current)

# === Legacy Interface ===
def run_strategy(broker, cfg, historical_credit_curve=None, quantity=1):
    """Main strategy runner for live mode"""
    today = dt.date.today()
    
    # 1. Generate Signal
    decision, legs, credit, consensus = generate_trade_signal(broker, cfg, today, historical_credit_curve, quantity)
    
    # 2. Execute if Valid
    if decision and decision.should_trade and legs:
        print(f"[Entering] {decision.symbol} {decision.structure} | Credit: ${credit:.2f} | MTF: {consensus:.2f}")
        order_id = broker.place_iron_condor(legs, quantity=quantity, limit_price=credit)
        
        if order_id:
            positions = broker.get_open_positions(cfg.underlying)
            if positions:
                positions[-1].mtf_consensus = consensus
                print(f"[Entered] Position {positions[-1].id} with credit ${positions[-1].credit_received:.2f}")
    
    manage_positions(broker, cfg, today)