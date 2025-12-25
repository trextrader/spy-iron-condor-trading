# options_strategy.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import datetime as dt

# Core types
@dataclass
class OptionQuote:
    symbol: str
    expiration: dt.date
    strike: float
    is_call: bool
    bid: float
    ask: float
    mid: float
    delta: float
    iv: float

@dataclass
class IronCondorLegs:
    short_call: OptionQuote
    long_call: OptionQuote
    short_put: OptionQuote
    long_put: OptionQuote

@dataclass
class PositionState:
    id: str
    legs: IronCondorLegs
    open_time: dt.datetime
    credit_received: float
    quantity: int
    adjustments_done: Dict[str, int]

# Utilities
def calc_spread_credit(short: OptionQuote, long: OptionQuote) -> float:
    return max(0.0, short.mid - long.mid)

def calc_condor_credit(legs: IronCondorLegs) -> float:
    return calc_spread_credit(legs.short_call, legs.long_call) + calc_spread_credit(legs.short_put, legs.long_put)

def max_loss(wing_width: float, total_credit: float) -> float:
    return max(0.0, wing_width - total_credit)

def within_delta_band(opt: OptionQuote, target_low: float, target_high: float) -> bool:
    d = abs(opt.delta)
    return target_low <= d <= target_high

def pick_expiration(expirations: List[dt.date], cfg, today: dt.date) -> Optional[dt.date]:
    candidates = [e for e in expirations if cfg.dte_min <= (e - today).days <= cfg.dte_max]
    return sorted(candidates)[0] if candidates else None

def nearest_by_delta(quotes: List[OptionQuote], is_call: bool, target_low: float, target_high: float) -> Optional[OptionQuote]:
    candidates = [q for q in quotes if q.is_call == is_call and within_delta_band(q, target_low, target_high)]
    if candidates:
        center = (target_low + target_high) / 2.0
        return sorted(candidates, key=lambda q: abs(abs(q.delta) - center))[0]
    return None

def pick_long_by_width(chain: List[OptionQuote], short_leg: OptionQuote, is_call: bool, wing_width: float) -> Optional[OptionQuote]:
    target_strike = short_leg.strike + wing_width if is_call else short_leg.strike - wing_width
    candidates = [q for q in chain if q.is_call == is_call and q.expiration == short_leg.expiration]
    if not candidates:
        return None
    return sorted(candidates, key=lambda q: abs(q.strike - target_strike))[0]

def build_condor(chain: List[OptionQuote], cfg, today: dt.date, spot: float, base_wing_width: float) -> Optional[IronCondorLegs]:
    short_call = nearest_by_delta(chain, True, cfg.target_short_delta_low, cfg.target_short_delta_high)
    short_put  = nearest_by_delta(chain, False, cfg.target_short_delta_low, cfg.target_short_delta_high)
    if not short_call or not short_put:
        return None
    long_call = pick_long_by_width(chain, short_call, True, base_wing_width)
    long_put  = pick_long_by_width(chain, short_put, False, base_wing_width)
    if not long_call or not long_put:
        return None
    return IronCondorLegs(short_call=short_call, long_call=long_call, short_put=short_put, long_put=long_put)

def regime_wing_width(cfg, iv_rank: float, vix: float) -> float:
    base = cfg.wing_width_min
    high_vol = (iv_rank >= cfg.regime_iv_rank_widen) or (vix >= cfg.regime_vix_widen)
    if high_vol:
        return min(cfg.wing_width_max, base + cfg.width_widen_increment)
    return base

def optimize_wing_width(cfg, historical_credit_curve: Dict[float, float]) -> float:
    feasible = [(w, r) for w, r in historical_credit_curve.items() if r >= cfg.min_credit_to_width and cfg.wing_width_min <= w <= cfg.wing_width_max]
    if not feasible:
        return cfg.wing_width_min
    feasible.sort(key=lambda t: (-t[1], t[0]))
    return feasible[0][0]

def can_enter_trade(broker, cfg) -> Tuple[bool, str]:
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
    if acct.get("positions_value", 0.0) / max(acct.get("equity", 1.0), 1.0) >= cfg.max_portfolio_alloc:
        return False, "Skip: portfolio allocation limit reached"
    return True, "OK"

def validate_pricing_and_risk(legs: IronCondorLegs, width: float, cfg, acct_equity: float) -> Tuple[bool, str, float]:
    credit = calc_condor_credit(legs)
    if credit < cfg.min_credit_to_width * width:
        return False, f"Reject: credit {credit:.2f} < {cfg.min_credit_to_width*width:.2f}", credit
    mloss = max_loss(width, credit)
    if mloss > cfg.max_account_risk_per_trade * acct_equity:
        return False, f"Reject: max loss {mloss:.2f} exceeds per-trade risk", credit
    return True, "OK", credit

def select_and_enter_condor(broker, cfg, today: dt.date,
                            historical_credit_curve: Optional[Dict[float, float]] = None,
                            quantity: int = 1):
    # 1. Check Portfolio limits (Max positions, etc.)
    ok, msg = can_enter_trade(broker, cfg)
    if not ok:
        print(f"Entry Blocked: {msg}")
        return None

    # 2. Get Market Context
    spot = broker.get_spot(cfg.underlying)
    ivr = broker.get_iv_rank(cfg.underlying)
    vix = broker.get_vix()
    
    # 3. Dynamic Wing Width Logic
    # Start with base width from config
    target_width = cfg.wing_width_min
    
    # Apply regime-based widening if VIX or IVR are high
    if vix > cfg.regime_vix_widen or ivr > cfg.regime_iv_rank_widen:
        target_width += cfg.width_widen_increment
        # Ensure we don't exceed the absolute max defined in config
        target_width = min(target_width, cfg.wing_width_max)
        print(f"High Volatility Detected (VIX: {vix:.2f}, IVR: {ivr:.2f}). "
              f"Widening wings to: ${target_width}")

    # 4. Expiration Selection
    expirations = broker.get_expirations(cfg.underlying)
    exp = pick_expiration(expirations, cfg, today)
    if not exp:
        print("No valid expiration in DTE window.")
        return None

    # 5. Build the Legs
    chain_poly = broker.get_option_chain(cfg.underlying, exp)
    legs = build_condor(chain_poly, cfg, today, spot, target_width)
    if not legs:
        print("Unable to build condor legs meeting delta requirements.")
        return None

    # 6. Pricing & Risk Validation (Credit-to-Width Filter)
    acct = broker.get_account_metrics()
    equity = acct.get("equity", 0.0)
    
    valid, vmsg, credit = validate_pricing_and_risk(legs, target_width, cfg, equity)
    
    # Explicit Credit-to-Width check
    credit_ratio = credit / target_width
    if credit_ratio < cfg.min_credit_to_width:
        print(f"Trade Skipped: Credit ratio {credit_ratio:.2f} is below minimum {cfg.min_credit_to_width}")
        return None

    if not valid:
        print(f"Risk Validation Failed: {vmsg}")
        return None

    # 7. Execution
    print(f"Entering Iron Condor on {cfg.underlying} | Width: {target_width} | Credit: ${credit:.2f}")
    order_id = broker.place_iron_condor(legs, quantity=quantity, limit_price=credit)
    
    if order_id:
        # Return the most recently opened position
        positions = broker.get_open_positions(cfg.underlying)
        return positions[-1] if positions else None
    
    return None

def estimate_pnl(position, current_quotes: IronCondorLegs) -> Tuple[float, float]:
    debit_call = max(0.0, current_quotes.long_call.mid - current_quotes.short_call.mid)
    debit_put  = max(0.0, current_quotes.long_put.mid - current_quotes.short_put.mid)
    debit_total = debit_call + debit_put
    unreal = (position.credit_received - debit_total) * position.quantity
    max_prof = position.credit_received * position.quantity
    return unreal, max_prof

def breached_short_strike(spot: float, legs: IronCondorLegs) -> Optional[str]:
    if spot >= legs.short_call.strike: return "call"
    if spot <= legs.short_put.strike:  return "put"
    return None

def net_position_delta(legs: IronCondorLegs, quantity: int) -> float:
    net = (-legs.short_call.delta + legs.long_call.delta + legs.short_put.delta - legs.long_put.delta) * quantity
    return net

def maybe_dynamic_delta_hedge(broker, cfg, pos, current: IronCondorLegs) -> None:
    nd = net_position_delta(current, pos.quantity)
    if abs(nd) >= cfg.dynamic_delta_hedge_threshold:
        units = int(round((abs(nd) / 0.01) * cfg.dynamic_delta_hedge_unit))
        if units > 0:
            side = "sell" if nd > 0 else "buy"
            broker.trade_shares("SPY", quantity=units, side=side)

def manage_positions(broker, cfg, today: dt.date) -> None:
    positions = broker.get_open_positions(cfg.underlying)
    spot = broker.get_spot(cfg.underlying)
    for pos in positions:
        chain = broker.get_option_chain(cfg.underlying, pos.legs.short_call.expiration)

        def match(q: OptionQuote) -> OptionQuote:
            matches = [x for x in chain if x.strike == q.strike and x.is_call == q.is_call]
            return matches[0]

        current = IronCondorLegs(
            short_call=match(pos.legs.short_call),
            long_call=match(pos.legs.long_call),
            short_put=match(pos.legs.short_put),
            long_put=match(pos.legs.long_put)
        )

        unreal, max_prof = estimate_pnl(pos, current)
        if unreal >= cfg.profit_take_pct * max_prof:
            broker.close_position(pos.id); continue

        dte = (pos.legs.short_call.expiration - today).days
        if dte <= cfg.max_hold_days:
            broker.close_position(pos.id); continue

        if unreal <= -cfg.loss_close_multiple * pos.credit_received * pos.quantity:
            broker.close_position(pos.id); continue

        breach = breached_short_strike(spot, pos.legs)
        if breach and pos.adjustments_done[breach] == 0 and cfg.allow_one_adjustment_per_side:
            expirations = broker.get_expirations(cfg.underlying)
            later_exps = [e for e in expirations if e > pos.legs.short_call.expiration]
            new_exp = later_exps[0] if later_exps else pos.legs.short_call.expiration
            new_chain = broker.get_option_chain(cfg.underlying, new_exp)
            width_call = pos.legs.long_call.strike - pos.legs.short_call.strike
            width_put  = pos.legs.short_put.strike - pos.legs.long_put.strike

            def nearest_new(is_call: bool):
                from math import inf
                center_low = cfg.target_short_delta_low
                center_high = cfg.target_short_delta_high
                candidates = [q for q in new_chain if q.is_call == is_call]
                if not candidates: return None
                center = (center_low + center_high)/2.0
                return sorted(candidates, key=lambda q: abs(abs(q.delta)-center))[0]

            if breach == "call":
                new_short = nearest_new(True)
                new_long  = pick_long_by_width(new_chain, new_short, True, width_call)
            else:
                new_short = nearest_new(False)
                new_long  = pick_long_by_width(new_chain, new_short, False, width_put)

            if new_short and new_long:
                addl_credit = max(0.0, new_short.mid - new_long.mid)
                broker.roll_vertical(pos.id, breach, new_short, new_long, limit_price=addl_credit)

        maybe_dynamic_delta_hedge(broker, cfg, pos, current)

def run_strategy(broker, cfg, historical_credit_curve=None, quantity=1):
    today = dt.date.today()
    pos = select_and_enter_condor(broker, cfg, today, historical_credit_curve, quantity)
    if pos:
        print(f"Entered position {pos.id} with credit {pos.credit_received:.2f}")
    manage_positions(broker, cfg, today)
