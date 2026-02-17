"""
pivots_slopes.py — Pivot Point Detection & Structural Slopes
Columns: M1PivotHigh, M1PivotLow, M1Slope (prefix changes per timeframe)

Uses adaptive lookback for pivot detection based on ATR regime.
"""
import numpy as np
import pandas as pd

def compute_pivots_slopes(high: pd.Series, low: pd.Series, close: pd.Series,
                          prefix: str = "M1", base_lookback: int = 5) -> pd.DataFrame:
    """
    Adaptive pivot detection + structural slope.
    
    Pivots are local extrema within an ATR-adaptive lookback window.
    Slope = price change rate between consecutive pivots.
    """
    n = len(close)
    
    # ATR for adaptive lookback
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    atr_median = atr.rolling(200, min_periods=1).median()
    atr_ratio = (atr / (atr_median + 1e-12)).clip(0.5, 3.0)
    
    # Adaptive lookback: shorter in high vol, longer in low vol
    adaptive_lb = (base_lookback / atr_ratio).clip(2, base_lookback * 3).astype(int)
    
    h = high.values
    l = low.values
    c = close.values
    
    pivot_high = np.zeros(n, dtype=np.float32)
    pivot_low = np.zeros(n, dtype=np.float32)
    slope = np.zeros(n, dtype=np.float32)
    
    last_pivot_price = c[0]
    last_pivot_idx = 0
    
    for i in range(base_lookback, n - base_lookback):
        lb = min(adaptive_lb.iloc[i], i, n - i - 1)
        lb = max(lb, 2)
        
        # Check if local high pivot
        window_high = h[i - lb:i + lb + 1]
        if h[i] == np.max(window_high):
            pivot_high[i] = 1.0
            if i > last_pivot_idx:
                slope[i] = (h[i] - last_pivot_price) / (i - last_pivot_idx + 1e-12)
                last_pivot_price = h[i]
                last_pivot_idx = i
        
        # Check if local low pivot
        window_low = l[i - lb:i + lb + 1]
        if l[i] == np.min(window_low):
            pivot_low[i] = 1.0
            if i > last_pivot_idx:
                slope[i] = (l[i] - last_pivot_price) / (i - last_pivot_idx + 1e-12)
                last_pivot_price = l[i]
                last_pivot_idx = i
    
    # Forward-fill slope for non-pivot bars
    slope_series = pd.Series(slope, index=close.index).replace(0, np.nan).ffill().fillna(0)
    
    result = pd.DataFrame(index=close.index)
    result[f'{prefix}PivotHigh'] = pivot_high
    result[f'{prefix}PivotLow'] = pivot_low
    result[f'{prefix}Slope'] = slope_series.astype(np.float32)
    
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", default="M1", help="Timeframe prefix: M1, M5, M15, H1")
    p.add_argument("--lookback", type=int, default=5)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    pivots = compute_pivots_slopes(df['high'], df['low'], df['close'],
                                    prefix=args.prefix, base_lookback=args.lookback)
    for col in pivots.columns:
        df[col] = pivots[col]
    df.to_csv(args.output, index=False)
    print(f"[OK] {args.prefix}PivotHigh, {args.prefix}PivotLow, {args.prefix}Slope → {args.output}")
