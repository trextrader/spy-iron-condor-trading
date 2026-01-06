# intelligence/fuzzy_engine.py
import numpy as np
from typing import Dict

# =============================================================================
# QUANTOR-MTFUZZ™: Fuzzy Position Sizing Logic
# =============================================================================

def compute_base_quantity(
    equity: float,
    max_loss_per_contract: float,
    risk_fraction: float = 0.02
) -> int:
    """
    Compute the hard ceiling on position size (Stage 1).
    q0 = floor((risk_fraction * equity) / max_loss_per_contract)
    """
    if equity <= 0.0:
        return 0

    if max_loss_per_contract <= 0.0:
        return 0

    max_risk = risk_fraction * equity
    q0 = int(max_risk // max_loss_per_contract)

    return max(q0, 0)


def compute_fuzzy_confidence(
    memberships: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute fuzzy confidence score Ft in [0, 1] (Stage 4).
    Ft = sum(w_j * mu_j)
    """
    confidence = 0.0

    for key, mu in memberships.items():
        w = weights.get(key, 0.0)
        confidence += w * mu

    # Hard clamp
    if confidence < 0.0: return 0.0
    if confidence > 1.0: return 1.0

    return confidence


def normalize_volatility(
    realized_vol: float,
    low_vol: float,
    high_vol: float
) -> float:
    """
    Normalize volatility into sigma_star in [0, 1] (Stage 5).
    Higher sigma_star = Higher Risk = Lower Size.
    """
    if high_vol <= low_vol:
        return 1.0

    sigma_star = (realized_vol - low_vol) / (high_vol - low_vol)

    if sigma_star < 0.0: return 0.0
    if sigma_star > 1.0: return 1.0

    return sigma_star


def compute_scaling_factor(
    confidence: float,
    volatility_penalty: float,
    min_scale: float = 0.0
) -> float:
    """
    Compute g(Ft, sigma_star) (Stage 6).
    g = Ft * (1 - sigma_star)
    """
    g = confidence * (1.0 - volatility_penalty)

    if g < min_scale: g = min_scale
    if g > 1.0: g = 1.0

    return g


def compute_position_size(
    equity: float,
    max_loss_per_contract: float,
    memberships: Dict[str, float],
    weights: Dict[str, float],
    realized_vol: float,
    low_vol: float,
    high_vol: float,
    risk_fraction: float = 0.02
) -> int:
    """
    Full sizing pipeline (Stage 7).
    """
    # Step 1: Hard ceiling
    q0 = compute_base_quantity(
        equity=equity,
        max_loss_per_contract=max_loss_per_contract,
        risk_fraction=risk_fraction
    )

    if q0 == 0:
        return 0

    # Step 2: Fuzzy confidence
    Ft = compute_fuzzy_confidence(
        memberships=memberships,
        weights=weights
    )

    # Step 3: Volatility penalty
    sigma_star = normalize_volatility(
        realized_vol=realized_vol,
        low_vol=low_vol,
        high_vol=high_vol
    )

    # Step 4: Scaling factor
    g = compute_scaling_factor(
        confidence=Ft,
        volatility_penalty=sigma_star
    )

    # Step 5: Final size
    q = int(q0 * g)
    
    # DEBUG LOG
    # print(f"[FuzzyDebug] Eq={equity:.0f} Risk={equity*risk_fraction:.0f} Loss={max_loss_per_contract:.0f} -> q0={q0} | Ft={Ft:.2f} Vol={realized_vol:.1f} Sig={sigma_star:.2f} -> g={g:.2f} => Final q={q}")
    
    if int(q0 * g) < 2 and q0 >= 2:
         print(f"[Debug] Rejected scaled_qty={int(q0*g)} from total_qty={q0}, confidence={Ft:.2f}, scaling={g:.2f}")

    return max(q, 0)

# =============================================================================
# Helper Functions for Membership Calculation
# =============================================================================

def calculate_mtf_membership(mtf_snapshot) -> float:
    """
    Calculates MTF Alignment score (0.0 to 1.0).
    Replaces old get_fuzzy_consensus.
    """
    if not mtf_snapshot:
        return 0.5
    
    # Weighting: Daily (50%), Hourly (35%), 5-Min (15%)
    weights = {'D': 0.50, '60': 0.35, '5': 0.15}
    total_score = 0
    total_weight = 0

    for tf, weight in weights.items():
        data = mtf_snapshot.get(tf)
        if data:
            # 1.0 = Bullish (Close > Open), 0.0 = Bearish
            # For Iron Condor, we want NEUTRAL (0.5).
            # But 'Membership' implies quality.
            # If we want alignment with NEUTRALITY:
            #   If Close ~ Open, score = 1.0 (Good for IC)
            #   If Trending, score = 0.0 (Bad for IC)
            
            # Let's align this with "MTF Alignment" where 1.0 = Favorable.
            # Favorable for IC = Neutral price action.
            
            close_price = data['close']
            open_price = data['open']
            
            # Absolute % change
            pct_change = abs(close_price - open_price) / open_price
            
            # Membership Function:
            # 0% change -> 1.0
            # >1% change -> 0.0
            score = max(0.0, 1.0 - (pct_change * 100.0))
            
            total_score += (score * weight)
            total_weight += weight

    if total_weight == 0:
        return 0.5
        
    return round(total_score / total_weight, 2)

def calculate_iv_membership(ivr: float) -> float:
    """
    Map IV Rank to favorability (0.0 to 1.0).
    Higher IVR is better for Short IC (selling premium).
    """
    # 0 IVR -> 0.0
    # 50 IVR -> 0.8
    # 100 IVR -> 1.0
    # Simple linear clip
    return min(1.0, ivr / 60.0)

def calculate_regime_membership(vix: float, threshold: float = 20.0) -> float:
    """
    Map VIX to regime stability.
    Lower VIX = Stable = 1.0
    High VIX = Unstable = 0.0
    """
    # If VIX <= 12 -> 1.0
    # If VIX >= threshold + 10 -> 0.0
    if vix <= 12.0: return 1.0
    
    # Linear decay
    # Range: 12 to 30 (approx 18 pts)
    # penalty = (vix - 12) / 18
    penalty = (vix - 12.0) / 18.0
    return max(0.0, 1.0 - penalty)


# =============================================================================
# ADVANCED TECHNICAL INDICATOR MEMBERSHIP FUNCTIONS
# =============================================================================

def calculate_rsi_membership(rsi: float, neutral_min: float = 40.0, neutral_max: float = 60.0) -> float:
    """
    RSI membership for Iron Condor favorability.
    
    Neutral RSI (40-60) = 1.0 (ideal for range-bound trading)
    Extremes (0-30, 70-100) = 0.0 (momentum breakout risk)
    """
    if rsi is None or np.isnan(rsi):
        return 0.3  # Conservative default
    
    # Perfect zone: RSI between 40-60
    if neutral_min <= rsi <= neutral_max:
        return 1.0
    
    # Below neutral zone: Linear decay
    elif rsi < neutral_min:
        return max(0.0, rsi / neutral_min)
    
    # Above neutral zone: Linear decay
    else:
        return max(0.0, (100 - rsi) / (100 - neutral_max))


def calculate_adx_membership(adx: float, threshold_low: float = 25.0, threshold_high: float = 40.0) -> float:
    """
    ADX membership for range-bound market detection.
    
    Low ADX (< 25) = 1.0 (weak trend, ideal for IC)
    High ADX (> 40) = 0.0 (strong trend, avoid IC)
    """
    if adx is None or np.isnan(adx):
        return 0.5  # Neutral default
    
    # Weak trend: Perfect for IC
    if adx <= threshold_low:
        return 1.0
    
    # Moderate trend: Linear decay
    elif adx <= threshold_high:
        return 1.0 - ((adx - threshold_low) / (threshold_high - threshold_low))
    
    # Strong trend: Avoid
    else:
        return 0.0


def calculate_bbands_membership(bb_position: float, bb_width: float = None, squeeze_threshold: float = 0.02) -> float:
    """
    Bollinger Bands membership for volatility regime.
    
    Favorable conditions:
    - Price in middle 60% of bands (position 0.2-0.8)
    - Compressed bands (width < 2%)
    """
    if bb_position is None or np.isnan(bb_position):
        return 0.5
    
    # Position score: Prefer middle (0.5 = perfect)
    position_score = max(0.0, 1.0 - abs(bb_position - 0.5) * 2)
    
    # Squeeze score: Prefer narrow bands
    if bb_width is None or np.isnan(bb_width):
        squeeze_score = 0.5
    else:
        squeeze_score = max(0.0, 1.0 - (bb_width / (squeeze_threshold * 2)))
    
    # Combined score: Position matters more (70/30 split)
    return (position_score * 0.7 + squeeze_score * 0.3)


def calculate_stoch_membership(stoch_k: float, neutral_min: float = 30.0, neutral_max: float = 70.0) -> float:
    """
    Stochastic oscillator membership (similar to RSI).
    
    Neutral %K (30-70) = 1.0
    Extremes = 0.0
    """
    if stoch_k is None or np.isnan(stoch_k):
        return 0.5
    
    if neutral_min <= stoch_k <= neutral_max:
        return 1.0
    elif stoch_k < neutral_min:
        return max(0.0, stoch_k / neutral_min)
    else:
        return max(0.0, (100 - stoch_k) / (100 - neutral_max))


def calculate_volume_membership(volume_ratio: float, min_ratio: float = 0.8) -> float:
    """
    Volume confirmation membership.
    
    High volume relative to average = better liquidity = 1.0
    Low volume = poor fills = 0.0
    """
    if volume_ratio is None or np.isnan(volume_ratio):
        return 0.5
    
    # Linear scale: 0 at ratio=0, 1.0 at ratio=min_ratio
    return min(1.0, max(0.0, volume_ratio / min_ratio))


def calculate_sma_distance_membership(sma_distance: float, max_distance: float = 0.02) -> float:
    """
    SMA distance membership (price near moving average = equilibrium).
    
    Price at SMA (distance = 0%) = 1.0
    Price > 2% from SMA = 0.0
    """
    if sma_distance is None or np.isnan(sma_distance):
        return 0.5
    
    abs_dist = abs(sma_distance)
    
    if abs_dist <= max_distance:
        return 1.0 - (abs_dist / max_distance)
    else:
        return 0.0


# =============================================================================
# EXIT INDICATOR FUNCTIONS
# =============================================================================

def calculate_atr_stop_multiplier(atr_pct: float, base_multiplier: float = 1.5) -> float:
    """
    Calculate dynamic stop-loss multiplier based on ATR.
    
    Low volatility (ATR < 0.5%) → Tight stop (1.0x)
    Medium volatility (ATR 0.5-2%) → Standard stop (1.5x)
    High volatility (ATR > 2%) → Wide stop (2.0x)
    """
    if atr_pct is None or np.isnan(atr_pct):
        return base_multiplier
    
    # Low volatility: Tighter stop
    if atr_pct <= 0.005:  # 0.5%
        return max(1.0, base_multiplier - 0.5)
    
    # High volatility: Wider stop
    elif atr_pct >= 0.02:  # 2.0%
        return min(2.5, base_multiplier + 0.5)
    
    # Medium volatility: Linear interpolation
    else:
        adjustment = ((atr_pct - 0.005) / 0.015) * 0.5
        return base_multiplier + adjustment


def check_bbands_breakout(bb_position: float, touch_threshold: float = 0.95) -> bool:
    """
    Check if price has touched Bollinger Bands (breakout signal).
    """
    if bb_position is None or np.isnan(bb_position):
        return False
    
    # Upper band touch
    if bb_position >= touch_threshold:
        return True
    
    # Lower band touch
    if bb_position <= (1.0 - touch_threshold):
        return True
    
    return False


# Backward Compatibility
get_fuzzy_consensus = calculate_mtf_membership

