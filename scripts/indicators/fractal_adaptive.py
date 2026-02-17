"""
fractal_adaptive.py — Fractal Adaptive Indicators
Columns: Anchored_VWAP, adx_adaptive, rsi_dyn

All three are regime-aware via vol_energy gating.
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

def compute_rsi_dyn(close: pd.Series, period: int = 14, span: int = 64) -> pd.Series:
    """Curvature-weighted RSI: gains/losses weighted by (1 + vol_energy)."""
    vol_energy = _curvature_energy(close, span)
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    
    weights = (1.0 + vol_energy).astype(float)
    w_gains = (gains * weights).rolling(period, min_periods=1).sum()
    w_losses = (losses * weights).rolling(period, min_periods=1).sum()
    w_sum = weights.rolling(period, min_periods=1).sum() + 1e-12
    
    avg_g = w_gains / w_sum
    avg_l = w_losses / w_sum
    rs = avg_g / (avg_l + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(0, 100).astype(np.float32)

def compute_adx_adaptive(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 14, span: int = 64) -> pd.Series:
    """ADX with vol-energy amplification in high-vol regimes."""
    vol_energy = _curvature_energy(close, span)
    prev_c = close.shift(1)
    
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    up = high.diff()
    down = -low.diff()
    
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)
    
    alpha = 1/period
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / (tr_s + 1e-12)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / (tr_s + 1e-12)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    # Vol-energy amplification
    adx = adx * (1 + 0.5 * vol_energy)
    return adx.clip(0, 100).astype(np.float32)

def compute_anchored_vwap(close: pd.Series, volume: pd.Series, anchor_window: int = 390) -> pd.Series:
    """Rolling VWAP anchored to session window (default 390 = 1 trading day of M1)."""
    pv = close * volume
    cum_pv = pv.rolling(anchor_window, min_periods=1).sum()
    cum_v = volume.rolling(anchor_window, min_periods=1).sum()
    vwap = cum_pv / (cum_v + 1e-12)
    return vwap.astype(np.float32)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--anchor-window", type=int, default=390)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    df['rsi_dyn'] = compute_rsi_dyn(df['close'])
    df['adx_adaptive'] = compute_adx_adaptive(df['high'], df['low'], df['close'])
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    df['Anchored_VWAP'] = compute_anchored_vwap(df['close'], vol, args.anchor_window)
    df.to_csv(args.output, index=False)
    print(f"[OK] rsi_dyn, adx_adaptive, Anchored_VWAP → {args.output}")
