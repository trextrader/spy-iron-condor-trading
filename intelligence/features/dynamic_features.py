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
        - log_return, vol_ewma, ret_z, atr_pct
        - kappa_proxy, vol_energy
        - rsi_dyn, adx_adaptive, psar_adaptive
        - bb_mu_dyn, bb_sigma_dyn, bb_lower_dyn, bb_upper_dyn
        - stoch_k_dyn
        - consolidation_score, breakout_score
    """
    if not inplace:
        df = df.copy()
    
    # 🕵️ FORENSIC FIX: Data has ~100 rows per timestamp. 
    # Standard .rolling() will collapse on spot data because it sees the same value repeated.
    # We must compute spot features on a per-minute basis, then merge back.
    
    print("   Building bar-level frame for time-aware spot features...")
    time_col = 'dt' if 'dt' in df.columns else ('timestamp' if 'timestamp' in df.columns else None)
    if time_col is None:
        raise ValueError("Missing 'dt' or 'timestamp' column for time-aware grouping.")

    # 1. Extract unique bars
    bars = df.groupby(time_col, as_index=False).agg({
        close_col: 'first',
        high_col: 'first',
        low_col: 'first',
        'volume': 'first' if 'volume' in df.columns else 'first' # placeholder
    }).sort_values(time_col)
    
    close_bar = bars[close_col]
    high_bar = bars[high_col]
    low_bar = bars[low_col]
    vol_bar = bars['volume'] if 'volume' in bars.columns else pd.Series(0, index=bars.index)
    
    print("   Computing dynamic features on bar-frame (across time)...")
    
    # Core kinematics on BARS
    bars["log_return"] = compute_log_return(close_bar)
    bars["vol_ewma"] = compute_vol_ewma(close_bar)
    bars["ret_z"] = (bars["log_return"] / (bars["vol_ewma"] + 1e-12)).astype(np.float32)
    bars["atr_pct"] = compute_atr_pct(high_bar, low_bar, close_bar)
    
    # Curvature & energy
    kappa, vol_energy = compute_curvature_features(close_bar)
    bars["kappa_proxy"] = kappa
    bars["vol_energy"] = vol_energy
    
    # Oscillators
    bars["rsi_dyn"] = compute_dynamic_rsi(close_bar, vol_energy)
    bars["adx_adaptive"] = compute_adaptive_adx(high_bar, low_bar, close_bar, vol_energy=vol_energy)
    bars["psar_adaptive"] = compute_dynamic_psar(high_bar, low_bar, close_bar, bars["atr_pct"])
    
    # Bollinger
    bb_mu, bb_sigma, bb_lower, bb_upper = compute_dynamic_bollinger(close_bar, vol_energy)
    bars["bb_mu_dyn"] = bb_mu
    bars["bb_sigma_dyn"] = bb_sigma
    bars["bb_lower_dyn"] = bb_lower
    bars["bb_upper_dyn"] = bb_upper
    
    # Stochastic
    bars["stoch_k_dyn"] = compute_dynamic_stochastic(high_bar, low_bar, close_bar, vol_energy)
    
    # Consolidation
    consol, breakout = compute_consolidation_breakout(close_bar, bb_sigma, bb_upper, bb_lower)
    bars["consolidation_score"] = consol
    bars["breakout_score"] = breakout
    
    # 2. Map bar-features back to the full dataset (Memory Efficient Broadcast)
    spot_cols = [
        "log_return", "vol_ewma", "ret_z", "atr_pct", "kappa_proxy", "vol_energy",
        "rsi_dyn", "adx_adaptive", "psar_adaptive", "bb_mu_dyn", "bb_sigma_dyn",
        "bb_lower_dyn", "bb_upper_dyn", "stoch_k_dyn", "consolidation_score", "breakout_score"
    ]
    
    print(f"   Broadcasting {len(spot_cols)} spot features via Map (Memory Optimized)...")
    bars_indexed = bars.set_index(time_col)
    
    for col in spot_cols:
        df[col] = df[time_col].map(bars_indexed[col]).astype(np.float32)
    
    # Clean up bar frame to free memory
    del bars
    del bars_indexed
    
    # Spread Ratio (Keep row-specific if bid/ask exists per option, 
    # but here it likely comes from spot if OHLCV based)
    if "ask" in df.columns and "bid" in df.columns:
        mid = (df["ask"] + df["bid"]) / 2.0
        mid = mid.replace(0, 1.0) 
        df["spread_ratio"] = (df["ask"] - df["bid"]) / mid
    elif "spread_ratio" not in df.columns:
        df["spread_ratio"] = 0.0001
    
    print(f"   [OK] Added time-aligned dynamic features + spread_ratio")
    
    return df


# =============================================================================
# V2.2 PRIMITIVE FEATURES (14 CANONICAL PRIMITIVES)
# =============================================================================

def compute_all_primitive_features_v22(
    df: pd.DataFrame,
    close_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    spread_col: str = "spread_ratio",
    lag_minutes_col: str = "lag_minutes",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute all 20 V2.2 primitive features and add to DataFrame.

    This extends V2.1 (32 features) with 20 primitive outputs = 52 total.
    Must be called AFTER compute_all_dynamic_features().

    Args:
        df: DataFrame with V2.1 features already computed
        close_col, open_col, high_col, low_col, volume_col: Column names
        spread_col: Spread ratio column (for friction gate)
        lag_minutes_col: IV lag column (for IV confidence)
        inplace: If True, add columns to df
        
    Returns:
        DataFrame with 20 new primitive columns
    """
    from intelligence.primitives import (
        compute_dynamic_bollinger_bands,
        compute_bandwidth_percentile_and_expansion,
        compute_chaikin_money_flow,
        compute_directional_pressure,
        compute_spread_friction_ratio,
        compute_gap_risk_score,
        compute_iv_confidence,
        compute_mtf_consensus,
        compute_vol_normalized_macd,
        compute_vol_normalized_adx,
        compute_dynamic_rsi as prim_compute_dynamic_rsi,
        compute_psar_flip_membership,
        compute_chaos_membership,
        compute_fuzzy_reversion_score_11,
    )
    
    if not inplace:
        df = df.copy()
    
    close = df[close_col]
    open_ = df[open_col] if open_col in df.columns else close
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col] if volume_col in df.columns else pd.Series(0, index=df.index)

    print("   Computing V2.2 primitive features...")
    
    # === Get existing V2.1 features ===
    vol_ewma = df.get("vol_ewma", pd.Series(0.01, index=df.index))
    vol_energy = df.get("vol_energy", pd.Series(0, index=df.index))
    kappa_proxy = df.get("kappa_proxy", pd.Series(0, index=df.index))
    atr_pct = df.get("atr_pct", pd.Series(0.01, index=df.index))
    psar_adaptive = df.get("psar_adaptive", pd.Series(0, index=df.index))
    
    # === P001-P007: Bands, Microstructure ===
    
    # P002: Bandwidth percentile & expansion
    bandwidth = df.get("bb_sigma_dyn", close.rolling(20).std())
    df["bandwidth"] = bandwidth.astype(np.float32)  # Expose for Executor P002 input
    bw_result = compute_bandwidth_percentile_and_expansion(
        bandwidth=bandwidth,
        window=100,
        expansion_lookback=5,
    )
    df["bb_percentile"] = bw_result["bw_percentile"].astype(np.float32)
    df["bw_expansion_rate"] = bw_result["expansion_rate"].fillna(0).astype(np.float32)
    
    # P003: Chaikin Money Flow (replaces volume_ratio)
    cmf = compute_chaikin_money_flow(high, low, close, volume, window=20)
    df["cmf"] = cmf.fillna(0.0).astype(np.float32)

    # P003c: Directional Pressure (replaces bid/ask)
    pressure = compute_directional_pressure(open_, high, low, close)
    df["pressure_up"] = pressure["pressure_up"].astype(np.float32)
    df["pressure_down"] = pressure["pressure_down"].astype(np.float32)

    # P004: Spread friction (if spread available)
    # Use CMF scaled to [0.1, 10] for vol_ratio input (backward compat for friction calc)
    vol_ratio_scaled = ((cmf + 1) / 2 * 9.9 + 0.1).fillna(1.0)
    if spread_col in df.columns:
        spread = df[spread_col]
        event_flag = pd.Series(0.0, index=df.index)
        friction_result = compute_spread_friction_ratio(
            spread=spread,
            high=high,
            low=low,
            n=20,
            atr_norm=atr_pct,
            vol_ratio=vol_ratio_scaled,
            bandwidth=bandwidth,
            event_flag=event_flag,
            theta0=1.0, a=0.1, b=0.1, c=0.1, d=0.2,
            theta_min=0.5, theta_max=1.5,
        )
        df["friction_ratio"] = friction_result["friction_ratio"].fillna(0.5).astype(np.float32)
        df["exec_allow"] = friction_result["exec_allow"].fillna(1).astype(np.float32)
    else:
        df["friction_ratio"] = 0.5
        df["exec_allow"] = 1.0
    
    # P005: Gap risk score
    atr_spike = (atr_pct > atr_pct.rolling(20).mean() * 1.5).astype(float)
    bw_expansion = (df["bw_expansion_rate"] > 0.1).astype(float)
    late_day_flag = pd.Series(0.0, index=df.index)  # Placeholder
    gap_result = compute_gap_risk_score(
        event_flag=pd.Series(0.0, index=df.index),
        atr_spike=atr_spike,
        bw_expansion=bw_expansion,
        late_day_flag=late_day_flag,
        w_event=0.3, w_atr=0.3, w_bw=0.2, w_late=0.2,
        g_crit=0.7,
    )
    df["gap_risk_score"] = gap_result["gap_risk_score"].astype(np.float32)
    df["risk_override"] = gap_result["risk_override"].astype(np.float32)
    
    # P006: IV confidence
    if lag_minutes_col in df.columns:
        lag_mins = df[lag_minutes_col]
    else:
        lag_mins = pd.Series(0.0, index=df.index)  # Assume fresh data
    df["iv_confidence"] = compute_iv_confidence(lag_mins, lambda_decay=0.05).astype(np.float32)
    
    # P007: MTF consensus (stub - requires multi-TF data)
    # For now, use breakout_score as proxy
    signal_1m = df.get("breakout_score", pd.Series(0, index=df.index))
    df["mtf_consensus"] = compute_mtf_consensus(
        signal_1m=signal_1m,
        signal_5m=signal_1m.rolling(5).mean().fillna(0),
        signal_15m=signal_1m.rolling(15).mean().fillna(0),
        w_1m=0.2, w_5m=0.3, w_15m=0.5,
    ).astype(np.float32)
    
    # === M001-M004: Momentum ===
    
    # M001: Vol-normalized MACD
    macd_result = compute_vol_normalized_macd(
        close=close,
        fast=12, slow=26, signal=9,
        vol_ewma=vol_ewma,
    )
    df["macd_norm"] = macd_result["macd_norm"].fillna(0).astype(np.float32)
    df["macd_signal_norm"] = macd_result["signal_norm"].fillna(0).astype(np.float32)
    df["macd_histogram"] = (df["macd_norm"] - df["macd_signal_norm"]).astype(np.float32)
    
    # M002: Vol-normalized ADX with +DI/-DI
    adx_norm = compute_vol_normalized_adx(
        high=high, low=low, close=close,
        period=14, beta=0.15,
        volatility_energy=vol_energy,
    )
    # Compute +DI and -DI separately
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = tr.rolling(14).sum()
    df["plus_di"] = (100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / (tr_smooth + 1e-12)).fillna(25).astype(np.float32)
    df["minus_di"] = (100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / (tr_smooth + 1e-12)).fillna(25).astype(np.float32)
    
    # M004: PSAR flip membership
    psar_result = compute_psar_flip_membership(
        close=close,
        psar=close - psar_adaptive * atr_pct * close,  # Reconstruct PSAR from normalized value
    )
    df["psar_trend"] = psar_result["psar_trend"].astype(np.float32)
    df["psar_reversion_mu"] = psar_result["psar_reversion_membership"].astype(np.float32)
    
    # === T001-T002: Topology (Stubbed for V2.2) ===
    df["beta1_norm_stub"] = 0.0  # TDA deferred to V2.3
    
    # T002: Chaos membership (using vol_energy as proxy for beta1_gated)
    chaos_result = compute_chaos_membership(
        beta1_gated=vol_energy * 2,  # Scale vol_energy to approximate beta1 range
        chaos_threshold=2.0,
        relax_threshold=1.0,
    )
    df["chaos_membership"] = chaos_result["chaos_membership"].astype(np.float32)
    df["position_size_mult"] = chaos_result["position_size_multiplier"].astype(np.float32)
    
    # === F001: Fuzzy reversion score ===
    # Compute membership functions
    mu_mtf = df["mtf_consensus"].abs()
    mu_ivr = df.get("ivr", pd.Series(0.5, index=df.index))
    mu_vix = pd.Series(0.5, index=df.index)  # Placeholder
    mu_rsi = (df.get("rsi_dyn", pd.Series(50, index=df.index)) - 50).abs() / 50
    mu_stoch = (df.get("stoch_k_dyn", pd.Series(50, index=df.index)) - 50).abs() / 50
    mu_adx = 1 - df.get("adx_adaptive", pd.Series(25, index=df.index)) / 50
    mu_sma = pd.Series(0.5, index=df.index)  # Placeholder
    mu_psar = df["psar_reversion_mu"]
    mu_bb = pd.Series(0.5, index=df.index)  # Placeholder
    mu_bbsqueeze = 1 - df["bb_percentile"] / 100
    # CMF is [-1, 1], scale to [0, 1] for fuzzy membership
    mu_vol = (df["cmf"] + 1) / 2
    
    df["fuzzy_reversion_11"] = compute_fuzzy_reversion_score_11(
        mu_mtf=mu_mtf, mu_ivr=mu_ivr, mu_vix=mu_vix,
        mu_rsi=mu_rsi, mu_stoch=mu_stoch, mu_adx=mu_adx,
        mu_sma=mu_sma, mu_psar=mu_psar, mu_bb=mu_bb,
        mu_bbsqueeze=mu_bbsqueeze, mu_vol=mu_vol,
        w_mtf=0.25, w_ivr=0.15, w_vix=0.10, w_rsi=0.15, w_stoch=0.05,
        w_adx=0.05, w_sma=0.05, w_psar=0.10, w_bb=0.05, w_bbsqueeze=0.03, w_vol=0.02,
    ).astype(np.float32)
    
    print(f"   [OK] Added 20 V2.2 primitive features")
    
    return df

