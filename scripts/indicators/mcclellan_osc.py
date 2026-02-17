"""
mcclellan_osc.py — McClellan Oscillator (Breadth Indicator)
Column: McClellanOsc

Adaptive: Uses EWMA-based advance-decline differential for breadth reading.
For single-instrument use (SPY), approximated via price advance/decline bars.
"""
import numpy as np
import pandas as pd

def compute_mcclellan(close: pd.Series, fast: int = 19, slow: int = 39) -> pd.Series:
    """
    McClellan Oscillator for single instrument.
    
    Uses advance-decline proxy: +1 if bar up, -1 if bar down.
    McClellan = EMA(AD, fast) - EMA(AD, slow)
    """
    delta = close.diff()
    ad = np.sign(delta).fillna(0)
    
    ema_fast = ad.ewm(span=fast, adjust=False).mean()
    ema_slow = ad.ewm(span=slow, adjust=False).mean()
    
    mcclellan = ema_fast - ema_slow
    return mcclellan.astype(np.float32)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    df['McClellanOsc'] = compute_mcclellan(df['close'])
    df.to_csv(args.output, index=False)
    print(f"[OK] McClellanOsc → {args.output}")
