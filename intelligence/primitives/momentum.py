# intelligence/primitives/momentum.py
"""
Momentum and signal primitives (M001-M004)
Exact canonical signatures - DO NOT MODIFY
"""

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Helper: exponential moving average"""
    return series.ewm(span=span, adjust=False).mean()


def compute_vol_normalized_macd(
    close: pd.Series,
    fast: int,
    slow: int,
    signal: int,
    vol_ewma: pd.Series,
) -> pd.DataFrame:
    """
    M001 - Volatility-Adaptive MACD (Rule A2 v2.0)

    Returns DataFrame with columns:
        ['macd', 'signal', 'hist', 'macd_norm', 'signal_norm']
    """
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig

    vol = vol_ewma.replace(0, np.nan)
    macd_norm = macd / vol
    sig_norm = sig / vol

    return pd.DataFrame(
        {
            "macd": macd,
            "signal": sig,
            "hist": hist,
            "macd_norm": macd_norm,
            "signal_norm": sig_norm,
        }
    )


def compute_vol_normalized_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    beta: float = 1.0,
    volatility_energy: pd.Series = None,
) -> pd.Series:
    """
    M002 - Volatility-Normalized ADX (Rules A1, A3)

    Returns Series: adx_normalized
    """
    if volatility_energy is None:
        volatility_energy = pd.Series(0, index=close.index)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr_smooth = tr.rolling(period).sum()
    plus_dm_smooth = plus_dm.rolling(period).sum()
    minus_dm_smooth = minus_dm.rolling(period).sum()

    plus_di = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()

    vol_energy = volatility_energy.fillna(0.0)
    adx_norm = adx / (1.0 + beta * vol_energy)

    return adx_norm


def compute_dynamic_rsi(
    close: pd.Series,
    period: int = 14,
    curvature_proxy: pd.Series = None,
    gamma: float = 1.0,
) -> pd.DataFrame:
    """
    M003 - Dynamic RSI (Rules A3, B1, B2, D1)

    Returns DataFrame with columns:
        ['rsi', 'rsi_dynamic']
    """
    if curvature_proxy is None:
        curvature_proxy = pd.Series(0, index=close.index)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    curv = curvature_proxy.fillna(0.0)
    rsi_dynamic = rsi * (1.0 + gamma * curv)

    return pd.DataFrame(
        {
            "rsi": rsi,
            "rsi_dynamic": rsi_dynamic,
        }
    )


def compute_psar_flip_membership(
    close: pd.Series,
    psar: pd.Series,
) -> pd.DataFrame:
    """
    M004 - PSAR Flip + Reversion Membership (Example PSAR rule, B1 v2.0)

    Returns DataFrame with columns:
        ['psar_trend', 'psar_reversion_membership']
    """
    # trend: +1 uptrend, -1 downtrend
    trend = np.where(psar < close, 1, -1)
    trend = pd.Series(trend, index=close.index)

    # reversion membership: 1 when PSAR suggests reversal vs current price
    # For mean reversion we want PSAR on opposite side of price
    reversion = np.where(
        ((psar > close) & (trend == 1)) | ((psar < close) & (trend == -1)),
        1.0,
        0.0,
    )
    reversion = pd.Series(reversion, index=close.index)

    return pd.DataFrame(
        {
            "psar_trend": trend,
            "psar_reversion_membership": reversion,
        }
    )
