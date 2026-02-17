"""
volume_flow.py — Dynamic Volume & Money Flow Indicators
Columns: cmf, friction_ratio, OSC_Volume, pressure_down, pressure_up

All volume indicators are volatility-normalized to self-adjust across regimes.
"""
import numpy as np
import pandas as pd

def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Chaikin Money Flow — dynamic version.
    CLV = (2*close - low - high) / (high - low)
    CMF = SUM(CLV * volume, window) / SUM(volume, window)
    """
    hl_range = high - low + 1e-12
    clv = (2 * close - low - high) / hl_range
    clv_vol = clv * volume
    cmf = clv_vol.rolling(window, min_periods=1).sum() / (volume.rolling(window, min_periods=1).sum() + 1e-12)
    return cmf.clip(-1, 1).astype(np.float32)

def compute_friction_ratio(spread: pd.Series, high: pd.Series, low: pd.Series,
                           atr_pct: pd.Series, window: int = 20) -> pd.Series:
    """
    Spread friction: spread normalized by ATR-adjusted range.
    High friction = wide spread relative to volatility = low liquidity.
    """
    hl_range = (high - low).rolling(window, min_periods=1).mean()
    atr_norm = atr_pct.rolling(window, min_periods=1).mean() + 1e-12
    friction = spread / (hl_range * atr_norm + 1e-12)
    # Normalize to [0, 1] via sigmoid
    friction = 1 / (1 + np.exp(-friction))
    return friction.astype(np.float32)

def compute_osc_volume(volume: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """
    Volume Oscillator: (fast_EMA - slow_EMA) / slow_EMA.
    Dynamic: uses EMA for regime-responsiveness.
    """
    fast_ema = volume.ewm(span=fast, adjust=False).mean()
    slow_ema = volume.ewm(span=slow, adjust=False).mean()
    osc = (fast_ema - slow_ema) / (slow_ema + 1e-12)
    return osc.astype(np.float32)

def compute_pressure(open_: pd.Series, high: pd.Series,
                     low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Directional Pressure: buying vs selling pressure per bar.
    pressure_up = (close - low) / (high - low)
    pressure_down = (high - close) / (high - low)
    """
    hl = high - low + 1e-12
    p_up = ((close - low) / hl).clip(0, 1).astype(np.float32)
    p_down = ((high - close) / hl).clip(0, 1).astype(np.float32)
    return pd.DataFrame({'pressure_up': p_up, 'pressure_down': p_down}, index=close.index)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    
    df['cmf'] = compute_cmf(df['high'], df['low'], df['close'], vol)
    
    if 'bid' in df.columns and 'ask' in df.columns:
        spread = df['ask'] - df['bid']
    else:
        spread = pd.Series(0.01, index=df.index)
    
    # Need atr_pct
    prev_c = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - prev_c).abs(), (df['low'] - prev_c).abs()], axis=1).max(axis=1)
    atr_pct = tr.ewm(alpha=1/14, adjust=False).mean() / df['close']
    
    df['friction_ratio'] = compute_friction_ratio(spread, df['high'], df['low'], atr_pct)
    df['OSC_Volume'] = compute_osc_volume(vol)
    
    pressure = compute_pressure(df['open'], df['high'], df['low'], df['close'])
    df['pressure_up'] = pressure['pressure_up']
    df['pressure_down'] = pressure['pressure_down']
    
    df.to_csv(args.output, index=False)
    print(f"[OK] cmf, friction_ratio, OSC_Volume, pressure_up, pressure_down → {args.output}")
