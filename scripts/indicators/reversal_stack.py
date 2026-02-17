"""
reversal_stack.py — Consolidation & Breakout Scoring
Columns: breakout_score, consolidation_score, spread_ratio

Adaptive via vol_energy-modulated dynamic bands.
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

def compute_reversal_stack(high: pd.Series, low: pd.Series, close: pd.Series,
                           lookback: int = 64) -> pd.DataFrame:
    vol_energy = _curvature_energy(close)
    
    # Dynamic Bollinger for reference
    mu = close.rolling(20, min_periods=1).mean()
    sigma = close.rolling(20, min_periods=1).std(ddof=0)
    g = (1.0 + vol_energy).clip(lower=1.0)
    sigma_dyn = sigma * g
    upper = mu + 2.0 * sigma_dyn
    lower = mu - 2.0 * sigma_dyn
    
    # Rolling price range
    hi = close.rolling(lookback, min_periods=1).max()
    lo = close.rolling(lookback, min_periods=1).min()
    price_range = (hi - lo).abs()
    
    # Mean band width
    mean_bw = sigma_dyn.rolling(lookback, min_periods=1).mean() + 1e-12
    
    # Consolidation: inverse of range-to-width
    ratio = price_range / mean_bw
    consolidation = (1.0 / (1.0 + ratio)).clip(0, 1)
    
    # Breakout: +1 above upper, -1 below lower, 0 inside
    breakout = (close > upper).astype(float) - (close < lower).astype(float)
    
    # Spread ratio
    mid = (high + low) / 2 + 1e-12
    spread_ratio = ((high - low) / mid).astype(np.float32)
    
    result = pd.DataFrame(index=close.index)
    result['consolidation_score'] = consolidation.astype(np.float32)
    result['breakout_score'] = breakout.astype(np.float32)
    result['spread_ratio'] = spread_ratio
    
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    rev = compute_reversal_stack(df['high'], df['low'], df['close'])
    for col in rev.columns:
        df[col] = rev[col]
    df.to_csv(args.output, index=False)
    print(f"[OK] consolidation_score, breakout_score, spread_ratio → {args.output}")
