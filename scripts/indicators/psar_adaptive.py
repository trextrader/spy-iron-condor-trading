"""
psar_adaptive.py — ATR-Adaptive Parabolic SAR
Columns: psar_mark, psar_adaptive, psar_trend, psar_reversion_mu

Adaptive: Acceleration factor scales with ATR ratio.
  - High vol → slower acceleration (wider SAR buffer)
  - Low vol  → faster acceleration (tighter trailing)
"""
import numpy as np
import pandas as pd

def compute_psar_full(high: pd.Series, low: pd.Series, close: pd.Series,
                      af_start: float = 0.04, af_max: float = 1.0) -> pd.DataFrame:
    """
    Returns DataFrame with columns: psar_mark, psar_adaptive, psar_trend, psar_reversion_mu
    """
    n = len(close)
    h, l, c = high.values, low.values, close.values
    
    # ATR for adaptive scaling
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    atr_ref = np.nanmean(atr)
    if atr_ref <= 0 or np.isnan(atr_ref): atr_ref = 0.01
    
    sar = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)  # 1=up, -1=down
    
    sar[0] = l[0]
    d = 1
    ep = h[0]
    af = af_start
    
    for t in range(1, n):
        atr_ratio = atr[t] / atr_ref if not np.isnan(atr[t]) else 1.0
        af_step = af_start * atr_ratio
        af_cap = af_max * max(atr_ratio, 0.5)
        
        if d == 1:
            sar[t] = sar[t-1] + af * (ep - sar[t-1])
            if c[t] < sar[t]:
                d = -1; sar[t] = ep; ep = l[t]; af = af_start
            else:
                if h[t] > ep: ep = h[t]; af = min(af + af_step, af_cap)
                if t >= 2: sar[t] = min(sar[t], l[t-1], l[t-2])
                else: sar[t] = min(sar[t], l[t-1])
        else:
            sar[t] = sar[t-1] - af * (sar[t-1] - ep)
            if c[t] > sar[t]:
                d = 1; sar[t] = ep; ep = h[t]; af = af_start
            else:
                if l[t] < ep: ep = l[t]; af = min(af + af_step, af_cap)
                if t >= 2: sar[t] = max(sar[t], h[t-1], h[t-2])
                else: sar[t] = max(sar[t], h[t-1])
        direction[t] = d
    
    atr_scale = atr * c + 1e-12
    
    result = pd.DataFrame(index=close.index)
    result['psar_mark'] = sar.astype(np.float32)
    result['psar_adaptive'] = ((c - sar) / atr_scale).astype(np.float32)
    result['psar_trend'] = direction.astype(np.float32)
    
    # Reversion membership: how close price is to SAR flip
    dist_to_sar = np.abs(c - sar) / (atr * c + 1e-12)
    result['psar_reversion_mu'] = np.exp(-dist_to_sar).astype(np.float32)
    
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    psar = compute_psar_full(df['high'], df['low'], df['close'])
    for col in psar.columns:
        df[col] = psar[col]
    df.to_csv(args.output, index=False)
    print(f"[OK] psar_mark, psar_adaptive, psar_trend, psar_reversion_mu → {args.output}")
