"""
Signal primitives S001–S015
Exact canonical signatures – DO NOT MODIFY.
"""

from __future__ import annotations
import pandas as pd


def compute_macd_trend_signal(
    macd_norm: pd.Series,
    signal_norm: pd.Series,
    threshold: float = 0.0,
) -> dict:
    """
    S003 - MACD Crossover Signal (A2)

    Returns:
        {
            "trend_up": bool Series,
            "trend_down": bool Series,
            "reverse": bool Series,
        }
    """
    macd = macd_norm.fillna(0.0)
    sig = signal_norm.fillna(0.0)
    diff = macd - sig
    trend_up = diff > threshold
    trend_down = diff < -threshold
    prev_diff = diff.shift(1)
    reverse = (diff * prev_diff) < 0
    return {
        "trend_up": trend_up,
        "trend_down": trend_down,
        "bullish": trend_up,
        "bearish": trend_down,
        "reverse": reverse.fillna(False),
    }


def compute_band_squeeze_breakout_signal(
    bw_percentile: pd.Series = None,
    expansion_rate: pd.Series = None,
    breakout_score: pd.Series = None,
    pct_thresh: float = 0.05,
    expansion_thresh: float = 2.0,
    # Aliases
    bb_percentile: pd.Series = None,
    bw_expansion_rate: pd.Series = None,
) -> dict:
    """
    S002 – Squeeze → Breakout Signal (C1/E2)

    Returns:
        {
            "squeeze": bool Series,
            "breakout_long": bool Series,
            "breakout_short": bool Series,
        }
    """
    pct = bw_percentile if bw_percentile is not None else bb_percentile
    if pct is None: pct = pd.Series(50.0) # Fallback

    exp = expansion_rate if expansion_rate is not None else bw_expansion_rate
    if exp is None: exp = pd.Series(0.0) # Fallback
    
    # Try to infer index from whatever we have
    if hasattr(pct, 'index'):
        br = breakout_score if breakout_score is not None else pd.Series(0, index=pct.index)
    else:
        br = pd.Series(0) # Scalar fallback
        
    pct = pct.fillna(50.0)
    exp = exp.fillna(0.0)

    squeeze = pct <= pct_thresh
    expanding = exp > expansion_thresh

    breakout_long = (br > 0) & expanding
    breakout_short = (br < 0) & expanding

    return {
        "squeeze": squeeze,
        "breakout_long": breakout_long,
        "breakout_short": breakout_short,
    }


def compute_bb_breakout_signal(
    close: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
    breakout_score: pd.Series = None,
    threshold: float = 0.0,
) -> dict:
    """
    S001 - BB Breakout Signal (C1/D2)

    Returns:
        {
            "bullish": bool Series,
            "bearish": bool Series,
        }
    """
    if breakout_score is not None:
        score = breakout_score.fillna(0.0)
        bullish = score > threshold
        bearish = score < -threshold
    else:
        bullish = close > upper_band
        bearish = close < lower_band
    return {
        "bullish": bullish,
        "bearish": bearish,
        "breakout_long": bullish,
        "breakout_short": bearish,
    }


def compute_bb_reversion_signal(
    close: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
    buffer: float = 0.0,
) -> dict:
    """
    S002 - BB Reversion Signal (B2)

    Returns:
        {
            "bullish": bool Series,
            "bearish": bool Series,
        }
    """
    lower = lower_band * (1.0 + buffer)
    upper = upper_band * (1.0 - buffer)
    bullish = close <= lower
    bearish = close >= upper
    return {
        "bullish": bullish,
        "bearish": bearish,
        "reversion_long": bullish,
        "reversion_short": bearish,
    }


def compute_rsi_reversion_signal(
    rsi_dynamic: pd.Series,
    lower: float = 30.0,
    upper: float = 70.0,
) -> dict:
    """
    S003 – RSI Reversion Signal (B2/D1)

    Returns:
        {
            "reversion_long": bool Series,
            "reversion_short": bool Series,
        }
    """
    r = rsi_dynamic.fillna(50.0)
    reversion_long = r < lower
    reversion_short = r > upper
    return {"reversion_long": reversion_long, "reversion_short": reversion_short}


def compute_rsi_divergence_signal(
    close: pd.Series,
    rsi_dynamic: pd.Series,
    lookback: int = 5,
) -> dict:
    """
    S005 - RSI Divergence Signal (B2)

    Returns:
        {
            "bullish": bool Series,
            "bearish": bool Series,
        }
    """
    price_slope = close - close.shift(lookback)
    rsi_slope = rsi_dynamic - rsi_dynamic.shift(lookback)
    bullish = (price_slope < 0) & (rsi_slope > 0)
    bearish = (price_slope > 0) & (rsi_slope < 0)
    return {
        "bullish": bullish.fillna(False),
        "bearish": bearish.fillna(False),
        "divergence_long": bullish.fillna(False),
        "divergence_short": bearish.fillna(False),
    }


def compute_mtf_alignment_signal(
    mtf_consensus: pd.Series,
    min_abs: float = 0.5,
) -> dict:
    """
    S013 - Multi-Timeframe Alignment Signal (C1/E2)

    Returns:
        {
            "aligned_long": bool Series,
            "aligned_short": bool Series,
        }
    """
    c = mtf_consensus.fillna(0.0)
    aligned_long = (c >= min_abs)
    aligned_short = (c <= -min_abs)
    prev = c.shift(1)
    reverses = (c * prev) < 0
    return {
        "aligned_long": aligned_long,
        "aligned_short": aligned_short,
        "aligned_bullish": aligned_long,
        "aligned_bearish": aligned_short,
        "reverses": reverses.fillna(False),
    }


def compute_fuzzy_reversion_signal(
    fuzzy_reversion_11: pd.Series,
    entry_thresh: float,
    exit_thresh: float,
) -> dict:
    """
    S005 – Fuzzy Reversion Entry/Exit (B1)

    Returns:
        {
            "enter_reversion": bool Series,
            "exit_reversion": bool Series,
        }
    """
    mu = fuzzy_reversion_11.fillna(0.5)
    enter = mu >= entry_thresh
    exit_ = mu <= exit_thresh
    return {"enter_reversion": enter, "exit_reversion": exit_}


def compute_gap_event_signal(
    gap_risk_score: pd.Series,
    risk_override: pd.Series,
    warn_level: float,
) -> dict:
    """
    S006 – Gap Event Signal (E1 override)

    Returns:
        {
            "gap_warn": bool Series,
            "gap_force_exit": bool Series,
        }
    """
    g = gap_risk_score.fillna(0.0)
    ro = risk_override.fillna(False)
    gap_warn = g >= warn_level
    gap_force_exit = ro.astype(bool)
    return {"gap_warn": gap_warn, "gap_force_exit": gap_force_exit}


def compute_chaos_dampening_signal(
    chaos_membership: pd.Series = None,
    position_size_multiplier: pd.Series = None,
    chaos_warn: float = 0.8,
) -> dict:
    """
    S007 – Chaos Dampening Signal (E3)

    Returns:
        {
            "chaos_warn": bool Series,
            "size_mult": float Series,
        }
    """
    if chaos_membership is None:
        # Default: no chaos detected
        return {"chaos_warn": pd.Series([False]), "size_mult": pd.Series([1.0])}
    cm = chaos_membership.fillna(0.0)
    psm = position_size_multiplier.fillna(1.0) if position_size_multiplier is not None else pd.Series([1.0] * len(cm))
    warn = cm >= chaos_warn
    return {"chaos_warn": warn, "size_mult": psm}


def compute_chaos_detection_signal(
    chaos_membership: pd.Series,
    chaos_threshold: float = 0.8,
) -> dict:
    """
    S012 - Chaos Detection Signal (E3)

    Returns:
        {
            "chaos": bool Series
        }
    """
    cm = chaos_membership.fillna(0.0)
    chaos = cm >= chaos_threshold
    return {"chaos": chaos, "signal": chaos}


def compute_regime_shift_signal(
    regime_score: pd.Series,
    threshold_high: float,
    threshold_low: float,
) -> dict:
    """
    S008 – Regime Shift Signal (C2/C004)

    Returns:
        {
            "regime_bull": bool Series,
            "regime_bear": bool Series,
        }
    """
    rs = regime_score.fillna(0.0)
    regime_bull = rs >= threshold_high
    regime_bear = rs <= threshold_low
    return {"regime_bull": regime_bull, "regime_bear": regime_bear}


def compute_bb_squeeze_signal(
    bw_percentile: pd.Series,
    threshold: float = 0.05,
) -> dict:
    """
    S006 - BB Squeeze Signal (C1)

    Returns:
        {
            "squeeze": bool Series
        }
    """
    bw = bw_percentile.fillna(50.0)
    squeeze = bw <= threshold
    return {"squeeze": squeeze, "signal": squeeze}


def compute_volume_spike_signal(
    volume_ratio: pd.Series,
    threshold: float = 1.5,
) -> dict:
    """
    S008 - Volume Spike Signal (C1/D2)

    Returns:
        {
            "vol_spike": bool Series
        }
    """
    vr = volume_ratio.fillna(1.0)
    spike = vr >= threshold
    return {"vol_spike": spike, "signal": spike}


def compute_liquidity_exec_signal(
    exec_liquidity_ok: pd.Series,
) -> dict:
    """
    S009 – Execution-Ready Signal

    Returns:
        {
            "can_execute": bool Series
        }
    """
    ce = exec_liquidity_ok.fillna(False)
    return {"can_execute": ce}


def compute_trend_follow_entry_signal(
    trend_up: pd.Series,
    trend_down: pd.Series,
    trend_ok: pd.Series,
) -> dict:
    """
    S010 – Trend-Follow Entry (A1/A3)

    Returns:
        {
            "enter_trend_long": bool Series,
            "enter_trend_short": bool Series,
        }
    """
    tu = trend_up.fillna(False)
    td = trend_down.fillna(False)
    to = trend_ok.fillna(True)
    return {
        "enter_trend_long": (tu & to),
        "enter_trend_short": (td & to),
    }


def compute_reversion_vs_trend_conflict_signal(
    reversion_ok: pd.Series,
    trend_ok: pd.Series,
) -> dict:
    """
    S011 – Conflict Detector (Reversion vs Trend)

    Returns:
        {
            "conflict": bool Series
        }
    """
    r = reversion_ok.fillna(False)
    t = trend_ok.fillna(False)
    return {"conflict": (r & t)}


def compute_spread_block_signal(
    exec_allow: pd.Series,
) -> dict:
    """
    S012 – Spread Block Signal (E1)

    Returns:
        {
            "spread_block": bool Series
        }
    """
    ea = exec_allow.fillna(True)
    return {"spread_block": (~ea)}


def compute_gap_exit_signal(
    force_exit: pd.Series,
) -> dict:
    """
    S013 – Gap Force Exit Signal

    Returns:
        {
            "exit_due_to_gap": bool Series
        }
    """
    fe = force_exit.fillna(False)
    return {"exit_due_to_gap": fe}


def compute_swing_high_low_signal(
    high: pd.Series,
    low: pd.Series,
    window: int = 5,
) -> dict:
    """
    S014 - Swing High/Low Detection (B2)

    Returns:
        {
            "new_high": bool Series,
            "new_low": bool Series,
        }
    """
    rolling_high = high.rolling(window).max()
    rolling_low = low.rolling(window).min()
    new_high = high >= rolling_high
    new_low = low <= rolling_low
    return {
        "new_high": new_high.fillna(False),
        "new_low": new_low.fillna(False),
    }


def compute_size_adjustment_signal(
    size_mult: pd.Series,
) -> dict:
    """
    S014 – Position Size Adjustment Signal

    Returns:
        {
            "size_mult": float Series
        }
    """
    sm = size_mult.fillna(1.0)
    return {"size_mult": sm}


def compute_final_execution_signal(
    can_execute: pd.Series,
    spread_block: pd.Series,
    gap_force_exit: pd.Series,
) -> dict:
    """
    S015 – Final Execution Decision

    Returns:
        {
            "allow_open": bool Series,
            "force_exit": bool Series,
        }
    """
    ce = can_execute.fillna(False)
    sb = spread_block.fillna(False)
    gfe = gap_force_exit.fillna(False)

    allow_open = ce & (~sb) & (~gfe)
    force_exit = gfe
    return {"allow_open": allow_open, "force_exit": force_exit}


def compute_bandwidth_expansion_signal(
    expansion_rate: pd.Series,
    threshold: float = 0.0,
) -> dict:
    """
    S007 – Bandwidth Expansion Signal (A2 logic helper)
    
    Returns:
        {
            "is_expanding": bool Series
        }
    """
    er = expansion_rate.fillna(0.0)
    is_expanding = er > threshold
    return {
        "is_expanding": is_expanding,
        "val": is_expanding,
        "signal": is_expanding
    }


def compute_psar_flip_membership(
    psar: pd.Series = None,
    psar_trend: pd.Series = None,
    psar_reversion_mu: pd.Series = None,
) -> dict:
    """
    S0xx – PSAR Flip Membership (Reversion Logic)
    
    Returns:
        {
            "psar_bull": bool Series,
            "psar_bear": bool Series,
            "reversion_score": float Series
        }
    """
    pt = psar if psar is not None else psar_trend
    if pt is None:
        return {
            "psar_bull": pd.Series([False]),
            "psar_bear": pd.Series([False]),
            "reversion_score": pd.Series([0.5])
        }
        
    pt = pt.fillna(0.0)
    mu = psar_reversion_mu.fillna(0.5) if psar_reversion_mu is not None else pd.Series(0.5, index=pt.index)
    
    # 1.0 = Bullish trend, -1.0 = Bearish trend
    psar_bull = (pt > 0)
    psar_bear = (pt < 0)
    
    return {
        "psar_bull": psar_bull,
        "psar_bear": psar_bear,
        "reversion_score": mu
    }
