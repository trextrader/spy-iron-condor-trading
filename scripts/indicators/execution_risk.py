"""
execution_risk.py — Execution & Risk Management Features
Columns: exec_allow, fuzzy_reversion_11, gap_risk_score, position_size_mult, risk_override, target_spot

All features are derived from market microstructure and volatility state.
"""
import numpy as np
import pandas as pd

def _curvature_energy(close: pd.Series, span: int = 64):
    r = np.log(close / close.shift(1))
    dr = r.diff()
    d2r = dr.diff()
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    kappa = (d2r / scale).ewm(span=max(8, span // 4), adjust=False).mean()
    return np.log1p(kappa.abs()).astype(np.float32)

def compute_exec_allow(spread: pd.Series, atr_pct: pd.Series, bandwidth: pd.Series,
                       theta0: float = 1.0, theta_min: float = 0.5, theta_max: float = 1.5) -> pd.Series:
    """
    Execution gate: 1 = safe to trade, 0 = blocked.
    Based on adaptive friction threshold.
    """
    hl_norm = atr_pct.rolling(20, min_periods=1).mean()
    friction = spread / (hl_norm + 1e-12)
    # Adaptive threshold via bandwidth
    theta = theta0 * (1 + 0.1 * bandwidth / (bandwidth.rolling(100, min_periods=1).median() + 1e-12))
    theta = theta.clip(theta_min, theta_max)
    return (friction < theta).astype(np.float32)

def compute_gap_risk_score(atr_pct: pd.Series, bw_expansion: pd.Series,
                           close: pd.Series) -> pd.DataFrame:
    """
    Gap risk: probability of adverse gap based on multiple signals.
    """
    atr_spike = (atr_pct > atr_pct.rolling(20, min_periods=1).mean() * 1.5).astype(float)
    bw_spike = (bw_expansion > 0.1).astype(float)
    
    # Composite gap risk score
    gap_risk = (0.4 * atr_spike + 0.4 * bw_spike + 0.2 * 0).clip(0, 1)  # 0.2 placeholder for event flag
    risk_override = (gap_risk > 0.7).astype(np.float32)
    
    return pd.DataFrame({
        'gap_risk_score': gap_risk.astype(np.float32),
        'risk_override': risk_override
    }, index=close.index)

def compute_target_spot(close: pd.Series, sma: pd.Series) -> pd.Series:
    """Target spot: SMA as the fair-value anchor for trade targeting."""
    return sma.astype(np.float32)

def compute_fuzzy_reversion(close: pd.Series, high: pd.Series, low: pd.Series,
                            volume: pd.Series) -> pd.Series:
    """
    11-input fuzzy reversion score.
    Aggregates multiple membership functions into a single reversion probability.
    """
    vol_energy = _curvature_energy(close)
    
    # RSI membership (divergence from 50)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    weights = (1.0 + vol_energy).astype(float)
    w_g = (gains * weights).rolling(14, min_periods=1).sum()
    w_l = (losses * weights).rolling(14, min_periods=1).sum()
    w_s = weights.rolling(14, min_periods=1).sum() + 1e-12
    rs = (w_g / w_s) / (w_l / w_s + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    mu_rsi = ((rsi - 50).abs() / 50).clip(0, 1)
    
    # Stochastic membership
    ll = low.rolling(14, min_periods=1).min()
    hh = high.rolling(14, min_periods=1).max()
    stoch = 100 * (close - ll) / (hh - ll + 1e-12)
    mu_stoch = ((stoch - 50).abs() / 50).clip(0, 1)
    
    # ADX membership (low ADX = trending is weak = reversion likely)
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    up = high.diff(); down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)
    alpha = 1/14
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / (tr_s + 1e-12)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / (tr_s + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    mu_adx = (1 - adx / 50).clip(0, 1)
    
    # BB squeeze membership  
    sigma = close.rolling(20, min_periods=1).std(ddof=0)
    g = (1.0 + vol_energy).clip(lower=1.0)
    bw = sigma * g
    bw_min = bw.rolling(100, min_periods=1).min()
    bw_max = bw.rolling(100, min_periods=1).max()
    bb_pct = (bw - bw_min) / (bw_max - bw_min + 1e-12)
    mu_bbsqueeze = (1 - bb_pct).clip(0, 1)
    
    # PSAR reversion membership
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    sma = close.rolling(20, min_periods=1).mean()
    dist_sma = ((close - sma) / (atr + 1e-12)).abs()
    mu_psar = np.exp(-dist_sma * 0.5).clip(0, 1)
    
    # CMF membership
    hl = high - low + 1e-12
    clv = (2 * close - low - high) / hl
    cmf = (clv * volume).rolling(20, min_periods=1).sum() / (volume.rolling(20, min_periods=1).sum() + 1e-12)
    mu_vol = ((cmf + 1) / 2).clip(0, 1)
    
    # Weighted fuzzy aggregation (11 inputs, some stubbed)
    score = (0.15 * mu_rsi + 0.05 * mu_stoch + 0.05 * mu_adx +
             0.10 * mu_psar + 0.03 * mu_bbsqueeze + 0.02 * mu_vol +
             0.25 * 0.5 +   # mu_mtf stub
             0.15 * 0.5 +   # mu_ivr stub
             0.10 * 0.5 +   # mu_vix stub
             0.05 * 0.5 +   # mu_sma stub
             0.05 * 0.5)    # mu_bb stub
    
    return score.clip(0, 1).astype(np.float32)

def compute_position_size_mult(close: pd.Series) -> pd.Series:
    """Position size multiplier based on chaos regime (vol_energy proxy)."""
    vol_energy = _curvature_energy(close)
    chaos_gate = (vol_energy * 2).clip(0, 5)  # Scale to beta1 range
    
    # High chaos → smaller position
    mult = np.where(chaos_gate > 2.0, 0.5, np.where(chaos_gate > 1.0, 0.75, 1.0))
    return pd.Series(mult, index=close.index).astype(np.float32)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    c, h, l = df['close'], df['high'], df['low']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    
    # ATR_pct
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_pct = tr.ewm(alpha=1/14, adjust=False).mean() / c
    
    # Spread
    spread = (df['ask'] - df['bid']) if 'ask' in df.columns and 'bid' in df.columns else pd.Series(0.01, index=df.index)
    
    # Bandwidth
    sigma = c.rolling(20, min_periods=1).std(ddof=0)
    ve = _curvature_energy(c)
    bw = sigma * (1.0 + ve).clip(lower=1.0)
    bw_exp = bw.pct_change(5).fillna(0)
    
    sma = c.rolling(20, min_periods=1).mean()
    
    df['exec_allow'] = compute_exec_allow(spread, atr_pct, bw)
    gap = compute_gap_risk_score(atr_pct, bw_exp, c)
    df['gap_risk_score'] = gap['gap_risk_score']
    df['risk_override'] = gap['risk_override']
    df['target_spot'] = compute_target_spot(c, sma)
    df['fuzzy_reversion_11'] = compute_fuzzy_reversion(c, h, l, vol)
    df['position_size_mult'] = compute_position_size_mult(c)
    
    df.to_csv(args.output, index=False)
    print(f"[OK] exec_allow, gap_risk_score, risk_override, target_spot, fuzzy_reversion_11, position_size_mult → {args.output}")
