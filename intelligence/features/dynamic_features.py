"""
Dynamic Feature Computation for CondorBrain V2.1.

This module provides vectorized, regime-aware feature engineering functions.
All functions are designed for 1-minute OHLCV data with 10M+ row scalability.

Features computed:
- Curvature & volatility energy (geometric dynamics)
- Dynamic RSI (curvature-weighted)
- Dynamic Bollinger Bands (vol-energy scaled)
- Adaptive PSAR (ATR-conditioned acceleration)
- Adaptive ADX (cycle-adaptive smoothing)
- Dynamic Stochastic %K (vol-normalized)
- Consolidation/Breakout scores (regime events)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# =============================================================================
# CORE GEOMETRY: CURVATURE & VOLATILITY ENERGY
# =============================================================================

def compute_curvature_features(
    close: pd.Series,
    span: int = 64,
    alpha: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute curvature proxy and volatility energy.
    
    Curvature proxy = smoothed second difference of log returns, normalized by scale.
    Vol energy = log(1 + α|κ|) - monotone transform for gating other indicators.
    
    Args:
        close: Price series
        span: EMA span for smoothing
        alpha: Scaling factor for energy transform
        
    Returns:
        (kappa_proxy, vol_energy) - both as pd.Series
    """
    # Log returns
    r = np.log(close / close.shift(1))
    
    # First and second differences
    dr = r.diff()
    d2r = dr.diff()
    
    # Local scale (smoothed absolute first derivative)
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    
    # Curvature proxy: 2nd diff normalized by scale, then smoothed
    kappa_raw = d2r / scale
    kappa_proxy = kappa_raw.ewm(span=max(8, span // 4), adjust=False).mean()
    
    # Volatility energy: monotone map of |κ|
    vol_energy = np.log1p(alpha * kappa_proxy.abs())
    
    return kappa_proxy.astype(np.float32), vol_energy.astype(np.float32)


def compute_log_return(close: pd.Series) -> pd.Series:
    """Compute 1-bar log returns."""
    return np.log(close / close.shift(1)).astype(np.float32)


def compute_vol_ewma(close: pd.Series, span: int = 64) -> pd.Series:
    """Compute EWMA of squared log returns (realized volatility proxy)."""
    r = np.log(close / close.shift(1))
    var_ewma = (r ** 2).ewm(span=span, adjust=False).mean()
    vol_ewma = np.sqrt(var_ewma)
    return vol_ewma.astype(np.float32)


def compute_atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute ATR as percentage of close price.
    
    ATR_pct = ATR / close (normalized volatility measure).
    """
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    
    # Wilder smoothing (EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    atr_pct = atr / close
    
    return atr_pct.astype(np.float32)


# =============================================================================
# DYNAMIC RSI (Curvature-Weighted)
# =============================================================================

def compute_dynamic_rsi(
    close: pd.Series,
    vol_energy: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute curvature-weighted RSI.
    
    Unlike classic RSI, gains/losses are weighted by (1 + vol_energy).
    This makes RSI more responsive in high-volatility regimes.
    
    Args:
        close: Price series
        vol_energy: Volatility energy series (from compute_curvature_features)
        period: RSI lookback window
        
    Returns:
        rsi_dyn: Dynamic RSI in [0, 100]
    """
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    
    # Weights based on volatility energy
    weights = (1.0 + vol_energy).astype(float)
    
    # Weighted rolling sums
    weighted_gains = (gains * weights).rolling(period, min_periods=period).sum()
    weighted_losses = (losses * weights).rolling(period, min_periods=period).sum()
    weight_sum = weights.rolling(period, min_periods=period).sum() + 1e-12
    
    avg_gain = weighted_gains / weight_sum
    avg_loss = weighted_losses / weight_sum
    
    rs = avg_gain / (avg_loss + 1e-12)
    rsi_dyn = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi_dyn.astype(np.float32)


# =============================================================================
# DYNAMIC BOLLINGER BANDS (Vol-Energy Scaled)
# =============================================================================

def compute_dynamic_bollinger(
    close: pd.Series,
    vol_energy: pd.Series,
    window: int = 20,
    base_k: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute volatility-energy-modulated Bollinger Bands.
    
    Band width is scaled by (1 + vol_energy), widening during high-volatility regimes.
    
    Returns:
        (bb_mu_dyn, bb_sigma_dyn, bb_lower_dyn, bb_upper_dyn)
    """
    mu = close.rolling(window, min_periods=window).mean()
    sigma = close.rolling(window, min_periods=window).std(ddof=0)
    
    # Scale factor from volatility energy
    g = (1.0 + vol_energy).clip(lower=1.0)
    
    sigma_dyn = sigma * g
    upper = mu + base_k * sigma_dyn
    lower = mu - base_k * sigma_dyn
    
    return (
        mu.astype(np.float32),
        sigma_dyn.astype(np.float32),
        lower.astype(np.float32),
        upper.astype(np.float32),
    )


# =============================================================================
# ADAPTIVE PARABOLIC SAR (ATR-Conditioned)
# =============================================================================

def compute_dynamic_psar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_pct: pd.Series,
    af_start: float = 0.02,
    af_max: float = 0.20,
) -> pd.Series:
    """
    Compute ATR-adaptive Parabolic SAR.
    
    Output is normalized: (price - SAR) / (ATR * close), giving a centered signal.
    Positive = uptrend, Negative = downtrend.
    """
    n = len(close)
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()
    close_arr = close.to_numpy()
    atr_arr = atr_pct.to_numpy()
    
    # Reference ATR for scaling
    atr_ref = np.nanmean(atr_arr)
    if atr_ref <= 0 or np.isnan(atr_ref):
        atr_ref = 0.01
    
    sar = np.full(n, np.nan, dtype=np.float64)
    direction = 1  # 1 = uptrend, -1 = downtrend
    ep = high_arr[0]  # Extreme point
    sar[0] = low_arr[0]
    af = af_start
    
    for t in range(1, n):
        # Scale AF by relative ATR
        atr_ratio = atr_arr[t] / atr_ref if not np.isnan(atr_arr[t]) else 1.0
        af_step = af_start * atr_ratio
        af_cap = af_max * max(atr_ratio, 0.5)
        
        if direction == 1:  # Uptrend
            sar[t] = sar[t-1] + af * (ep - sar[t-1])
            
            # Check for reversal
            if close_arr[t] < sar[t]:
                direction = -1
                sar[t] = ep
                ep = low_arr[t]
                af = af_start
            else:
                # Update EP if new high
                if high_arr[t] > ep:
                    ep = high_arr[t]
                    af = min(af + af_step, af_cap)
                # SAR cannot be above recent lows
                if t >= 2:
                    sar[t] = min(sar[t], low_arr[t-1], low_arr[t-2])
                else:
                    sar[t] = min(sar[t], low_arr[t-1])
        else:  # Downtrend
            sar[t] = sar[t-1] - af * (sar[t-1] - ep)
            
            if close_arr[t] > sar[t]:
                direction = 1
                sar[t] = ep
                ep = high_arr[t]
                af = af_start
            else:
                if low_arr[t] < ep:
                    ep = low_arr[t]
                    af = min(af + af_step, af_cap)
                if t >= 2:
                    sar[t] = max(sar[t], high_arr[t-1], high_arr[t-2])
                else:
                    sar[t] = max(sar[t], high_arr[t-1])
    
    # Normalize: (price - SAR) / (ATR * price)
    atr_scale = atr_arr * close_arr + 1e-12
    psar_adaptive = (close_arr - sar) / atr_scale
    
    return pd.Series(psar_adaptive, index=close.index, dtype=np.float32)


# =============================================================================
# ADAPTIVE ADX (Simplified Cycle-Adaptive)
# =============================================================================

def compute_adaptive_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    vol_energy: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute ADX with optional volatility-adaptive smoothing.
    
    For simplicity, this uses classic Wilder ADX but can be modulated
    by vol_energy for regime-awareness.
    """
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    
    # Directional movement
    up_move = high - prev_high
    down_move = prev_low - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder smoothing
    alpha = 1 / period
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-12)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-12)
    
    # DX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    
    # ADX (smoothed DX)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    # Optionally modulate by vol_energy (amplify in high-vol regimes)
    if vol_energy is not None:
        adx = adx * (1 + 0.5 * vol_energy)
        adx = adx.clip(0, 100)
    
    return adx.astype(np.float32)


# =============================================================================
# DYNAMIC STOCHASTIC %K
# =============================================================================

def compute_dynamic_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    vol_energy: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.Series:
    """
    Compute volatility-normalized Stochastic %K.
    
    In high-volatility regimes, the oscillator is compressed toward 50
    to avoid false extremes.
    """
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    
    # Smooth with D period
    stoch_k = raw_k.rolling(d_period, min_periods=1).mean()
    
    # Volatility normalization: compress toward 50 in high-vol
    # stoch_dyn = 50 + (stoch_k - 50) / (1 + 0.5 * vol_energy)
    compression = 1 + 0.5 * vol_energy
    stoch_k_dyn = 50 + (stoch_k - 50) / compression
    
    return stoch_k_dyn.astype(np.float32)


# =============================================================================
# CONSOLIDATION & BREAKOUT SCORES
# =============================================================================

def compute_consolidation_breakout(
    close: pd.Series,
    bb_sigma_dyn: pd.Series,
    bb_upper_dyn: pd.Series,
    bb_lower_dyn: pd.Series,
    lookback: int = 64,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute consolidation and breakout scores.
    
    Consolidation Score: High (near 1) when price range is small relative to bands.
    Breakout Score: +1 if price > upper band, -1 if price < lower band, else 0.
    
    Returns:
        (consolidation_score, breakout_score)
    """
    # Rolling range
    hi = close.rolling(lookback, min_periods=lookback).max()
    lo = close.rolling(lookback, min_periods=lookback).min()
    price_range = (hi - lo).abs()
    
    # Mean band width
    mean_band_width = bb_sigma_dyn.rolling(lookback, min_periods=lookback).mean() + 1e-12
    
    # Consolidation score: inverse of range-to-width ratio
    ratio = price_range / mean_band_width
    consolidation_score = 1.0 / (1.0 + ratio)
    consolidation_score = consolidation_score.clip(0.0, 1.0)
    
    # Breakout score: +1 upper, -1 lower, 0 inside
    breakout_up = (close > bb_upper_dyn).astype(float)
    breakout_down = (close < bb_lower_dyn).astype(float)
    breakout_score = breakout_up - breakout_down
    
    return (
        consolidation_score.astype(np.float32),
        breakout_score.astype(np.float32),
    )


# =============================================================================
# MAIN ENTRY POINT: COMPUTE ALL DYNAMIC FEATURES
# =============================================================================

def compute_all_dynamic_features(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute all 16 dynamic features and add to DataFrame.
    
    This is the main entry point for feature engineering.
    
    Args:
        df: DataFrame with OHLCV columns
        close_col: Name of close price column
        high_col: Name of high price column
        low_col: Name of low price column
        inplace: If True, add columns to df; else return new DataFrame
        
    Returns:
        DataFrame with new columns:
        - log_return, vol_ewma, atr_pct
        - kappa_proxy, vol_energy
        - rsi_dyn, adx_adaptive, psar_adaptive
        - bb_mu_dyn, bb_sigma_dyn, bb_lower_dyn, bb_upper_dyn
        - stoch_k_dyn
        - consolidation_score, breakout_score
    """
    if not inplace:
        df = df.copy()
    
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    
    print("   Computing dynamic features...")
    
    # Core kinematics
    df["log_return"] = compute_log_return(close)
    df["vol_ewma"] = compute_vol_ewma(close)
    df["atr_pct"] = compute_atr_pct(high, low, close)
    
    # Curvature & energy
    kappa, vol_energy = compute_curvature_features(close)
    df["kappa_proxy"] = kappa
    df["vol_energy"] = vol_energy
    
    # Dynamic oscillators
    df["rsi_dyn"] = compute_dynamic_rsi(close, vol_energy)
    df["adx_adaptive"] = compute_adaptive_adx(high, low, close, vol_energy=vol_energy)
    df["psar_adaptive"] = compute_dynamic_psar(high, low, close, df["atr_pct"])
    
    # Dynamic Bollinger
    bb_mu, bb_sigma, bb_lower, bb_upper = compute_dynamic_bollinger(close, vol_energy)
    df["bb_mu_dyn"] = bb_mu
    df["bb_sigma_dyn"] = bb_sigma
    df["bb_lower_dyn"] = bb_lower
    df["bb_upper_dyn"] = bb_upper
    
    # Dynamic Stochastic
    df["stoch_k_dyn"] = compute_dynamic_stochastic(high, low, close, vol_energy)
    
    # Consolidation & Breakout
    consol, breakout = compute_consolidation_breakout(close, bb_sigma, bb_upper, bb_lower)
    df["consolidation_score"] = consol
    df["breakout_score"] = breakout
    
    print(f"   ✅ Added 16 dynamic features (15 new columns)")
    
    return df
