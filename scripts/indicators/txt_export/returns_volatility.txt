"""
returns_volatility.py — Returns & Volatility Features
Columns: atr_pct, kappa_proxy, log_return, ret_z, vol_energy, vol_ewma

These are the core kinematic/geometric features — all self-adjusting via curvature.
"""
import numpy as np
import pandas as pd

def compute_log_return(close: pd.Series) -> pd.Series:
    """1-bar log returns."""
    return np.log(close / close.shift(1)).astype(np.float32)

def compute_vol_ewma(close: pd.Series, span: int = 64) -> pd.Series:
    """EWMA of squared log returns = realized volatility proxy."""
    r = np.log(close / close.shift(1))
    return np.sqrt((r ** 2).ewm(span=span, adjust=False).mean()).astype(np.float32)

def compute_atr_pct(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 14) -> pd.Series:
    """ATR as percentage of close. Normalized volatility measure."""
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return (atr / close).astype(np.float32)

def compute_curvature_features(close: pd.Series, span: int = 64) -> pd.DataFrame:
    """
    Curvature proxy (κ) and volatility energy.
    κ = smoothed 2nd diff of log returns / local scale
    vol_energy = log(1 + |κ|) — monotone gate for all other indicators
    """
    r = np.log(close / close.shift(1))
    dr = r.diff()
    d2r = dr.diff()
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    kappa = (d2r / scale).ewm(span=max(8, span // 4), adjust=False).mean()
    vol_energy = np.log1p(kappa.abs())
    
    return pd.DataFrame({
        'kappa_proxy': kappa.astype(np.float32),
        'vol_energy': vol_energy.astype(np.float32)
    }, index=close.index)

def compute_ret_z(log_return: pd.Series, vol_ewma: pd.Series) -> pd.Series:
    """Standardized return: log_return / vol_ewma."""
    return (log_return / (vol_ewma + 1e-12)).astype(np.float32)

def compute_all_returns_volatility(high: pd.Series, low: pd.Series,
                                    close: pd.Series) -> pd.DataFrame:
    """Compute all 6 returns/volatility columns."""
    log_ret = compute_log_return(close)
    vol_ewma = compute_vol_ewma(close)
    atr_pct = compute_atr_pct(high, low, close)
    curv = compute_curvature_features(close)
    ret_z = compute_ret_z(log_ret, vol_ewma)
    
    result = pd.DataFrame(index=close.index)
    result['log_return'] = log_ret
    result['vol_ewma'] = vol_ewma
    result['atr_pct'] = atr_pct
    result['kappa_proxy'] = curv['kappa_proxy']
    result['vol_energy'] = curv['vol_energy']
    result['ret_z'] = ret_z
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    rv = compute_all_returns_volatility(df['high'], df['low'], df['close'])
    for col in rv.columns:
        df[col] = rv[col]
    df.to_csv(args.output, index=False)
    print(f"[OK] log_return, vol_ewma, atr_pct, kappa_proxy, vol_energy, ret_z → {args.output}")
