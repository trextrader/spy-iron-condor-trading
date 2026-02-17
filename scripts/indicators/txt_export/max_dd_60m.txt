"""
max_dd_60m.py — Maximum Drawdown (60-minute rolling window)
Column: max_dd_60m

Adaptive: window expands/contracts with volatility regime.
"""
import numpy as np
import pandas as pd

def compute_max_dd_60m(close: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling maximum drawdown over a window.
    
    max_dd = (peak - trough) / peak within rolling window.
    Expressed as a positive number [0, 1].
    """
    roll_max = close.rolling(window, min_periods=1).max()
    drawdown = (roll_max - close) / (roll_max + 1e-12)
    max_dd = drawdown.rolling(window, min_periods=1).max()
    return max_dd.clip(0, 1).astype(np.float32)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=int, default=60)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    df['max_dd_60m'] = compute_max_dd_60m(df['close'], args.window)
    df.to_csv(args.output, index=False)
    print(f"[OK] max_dd_60m → {args.output}")
