"""
dynamic_signals.py — Dynamic BB Break & PSAR Signals
Columns: M1_BBUpperBreak, M1_BBLowerBreak, M1_PSAR (prefix changes per TF)

These are event signals derived from the dynamic indicators.
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

def compute_dynamic_signals(high: pd.Series, low: pd.Series, close: pd.Series,
                            prefix: str = "M1", bb_window: int = 20) -> pd.DataFrame:
    """
    BB break signals and PSAR direction signal.
    
    BBUpperBreak = 1 when close > dynamic upper BB (breakout)
    BBLowerBreak = 1 when close < dynamic lower BB (breakdown)
    PSAR signal = +1 uptrend, -1 downtrend (from adaptive PSAR)
    """
    vol_energy = _curvature_energy(close)
    
    # Dynamic Bollinger
    mu = close.rolling(bb_window, min_periods=1).mean()
    sigma = close.rolling(bb_window, min_periods=1).std(ddof=0)
    g = (1.0 + vol_energy).clip(lower=1.0)
    sigma_dyn = sigma * g
    upper = mu + 2.0 * sigma_dyn
    lower = mu - 2.0 * sigma_dyn
    
    bb_upper_break = (close > upper).astype(np.float32)
    bb_lower_break = (close < lower).astype(np.float32)
    
    # PSAR direction (simplified adaptive)
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    atr_ref = atr.mean()
    
    n = len(close)
    h, l, c = high.values, low.values, close.values
    atr_v = atr.values
    
    sar = np.full(n, np.nan)
    d = 1; ep = h[0]; sar[0] = l[0]; af = 0.02
    direction = np.ones(n)
    
    for t in range(1, n):
        ar = atr_v[t] / (atr_ref + 1e-12) if not np.isnan(atr_v[t]) else 1.0
        af_step = 0.02 * ar
        af_cap = 0.20 * max(ar, 0.5)
        
        if d == 1:
            sar[t] = sar[t-1] + af * (ep - sar[t-1])
            if c[t] < sar[t]:
                d = -1; sar[t] = ep; ep = l[t]; af = 0.02
            else:
                if h[t] > ep: ep = h[t]; af = min(af + af_step, af_cap)
                sar[t] = min(sar[t], l[max(0, t-1)], l[max(0, t-2)]) if t >= 2 else min(sar[t], l[t-1])
        else:
            sar[t] = sar[t-1] - af * (sar[t-1] - ep)
            if c[t] > sar[t]:
                d = 1; sar[t] = ep; ep = h[t]; af = 0.02
            else:
                if l[t] < ep: ep = l[t]; af = min(af + af_step, af_cap)
                sar[t] = max(sar[t], h[max(0, t-1)], h[max(0, t-2)]) if t >= 2 else max(sar[t], h[t-1])
        direction[t] = d
    
    result = pd.DataFrame(index=close.index)
    result[f'{prefix}_BBUpperBreak'] = bb_upper_break
    result[f'{prefix}_BBLowerBreak'] = bb_lower_break
    result[f'{prefix}_PSAR'] = pd.Series(direction, index=close.index).astype(np.float32)
    
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", default="M1")
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    signals = compute_dynamic_signals(df['high'], df['low'], df['close'], prefix=args.prefix)
    for col in signals.columns:
        df[col] = signals[col]
    df.to_csv(args.output, index=False)
    print(f"[OK] {args.prefix}_BBUpperBreak, {args.prefix}_BBLowerBreak, {args.prefix}_PSAR → {args.output}")
