"""
bb_dynamic.py — Volatility-Energy Modulated Bollinger Bands
Columns: bb_mu_dyn, bb_sigma_dyn, bb_lower_dyn, bb_upper_dyn, bb_percentile, bandwidth, bw_expansion_rate, stoch_k_dyn

Adaptive: Band width scales with vol_energy (curvature-derived).
  - Consolidation → bands squeeze tighter
  - Breakout/high-vol → bands widen dynamically
"""
import numpy as np
import pandas as pd

def _curvature_energy(close: pd.Series, span: int = 64):
    r = np.log(close / close.shift(1))
    dr = r.diff()
    d2r = dr.diff()
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    kappa = (d2r / scale).ewm(span=max(8, span // 4), adjust=False).mean()
    vol_energy = np.log1p(kappa.abs())
    return vol_energy.astype(np.float32)

def compute_bb_dynamic(close: pd.Series, high: pd.Series, low: pd.Series,
                       window: int = 20, base_k: float = 2.0, k_period: int = 14) -> pd.DataFrame:
    vol_energy = _curvature_energy(close)
    
    mu = close.rolling(window, min_periods=1).mean()
    sigma = close.rolling(window, min_periods=1).std(ddof=0)
    
    g = (1.0 + vol_energy).clip(lower=1.0)
    sigma_dyn = sigma * g
    upper = mu + base_k * sigma_dyn
    lower = mu - base_k * sigma_dyn
    
    # Bandwidth = sigma_dyn (already dynamic)
    bandwidth = sigma_dyn
    
    # BB Percentile: where current bandwidth sits in its rolling history
    bw_roll_min = bandwidth.rolling(100, min_periods=1).min()
    bw_roll_max = bandwidth.rolling(100, min_periods=1).max()
    bb_pct = 100 * (bandwidth - bw_roll_min) / (bw_roll_max - bw_roll_min + 1e-12)
    
    # BW Expansion Rate: 5-bar rate of change of bandwidth
    bw_expansion = bandwidth.pct_change(5).fillna(0)
    
    # Dynamic Stochastic %K (vol-compressed)
    lowest = low.rolling(k_period, min_periods=1).min()
    highest = high.rolling(k_period, min_periods=1).max()
    raw_k = 100 * (close - lowest) / (highest - lowest + 1e-12)
    stoch_k = raw_k.rolling(3, min_periods=1).mean()
    compression = 1 + 0.5 * vol_energy
    stoch_k_dyn = 50 + (stoch_k - 50) / compression
    
    result = pd.DataFrame(index=close.index)
    result['bb_mu_dyn'] = mu.astype(np.float32)
    result['bb_sigma_dyn'] = sigma_dyn.astype(np.float32)
    result['bb_lower_dyn'] = lower.astype(np.float32)
    result['bb_upper_dyn'] = upper.astype(np.float32)
    result['bb_percentile'] = bb_pct.clip(0, 100).astype(np.float32)
    result['bandwidth'] = bandwidth.astype(np.float32)
    result['bw_expansion_rate'] = bw_expansion.clip(-1, 1).astype(np.float32)
    result['stoch_k_dyn'] = stoch_k_dyn.clip(0, 100).astype(np.float32)
    
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    bb = compute_bb_dynamic(df['close'], df['high'], df['low'])
    for col in bb.columns:
        df[col] = bb[col]
    df.to_csv(args.output, index=False)
    print(f"[OK] 8 BB/Stoch columns → {args.output}")
