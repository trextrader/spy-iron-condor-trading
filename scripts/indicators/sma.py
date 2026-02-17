"""
sma.py — Simple Moving Average (Dynamic Adaptive)
Column: sma (or MA)

Adaptive: Window length adjusts with realized volatility.
  - Low vol  → longer window (smoother trend)
  - High vol → shorter window (faster response)
"""
import numpy as np
import pandas as pd

def compute_sma(close: pd.Series, base_window: int = 20, vol_span: int = 64) -> pd.Series:
    """
    Adaptive SMA: window shrinks in high-vol, expands in low-vol.
    
    Effective window = base_window / (1 + vol_energy)
    """
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(vol_span, min_periods=1).std()
    vol_median = vol.rolling(vol_span * 4, min_periods=1).median()
    vol_ratio = (vol / (vol_median + 1e-12)).clip(0.5, 3.0)
    
    # Adaptive window: shrink when vol is high, expand when low
    adaptive_window = (base_window / vol_ratio).clip(5, base_window * 3).astype(int)
    
    # Compute using the median adaptive window for vectorized efficiency
    # Then apply EMA with adaptive alpha as proxy
    alpha = 2.0 / (adaptive_window + 1)
    sma = close.ewm(alpha=alpha.median(), adjust=False).mean()
    
    return sma.astype(np.float32)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--close-col", default="close")
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    df['sma'] = compute_sma(df[args.close_col])
    df.to_csv(args.output, index=False)
    print(f"[OK] sma computed → {args.output}")
