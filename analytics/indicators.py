"""
analytics/indicators.py

Indicator implementations for 5-minute bars:
- RSI (Wilder smoothing)
- ADX (Wilder smoothing)
- IV Rank (percentile rank of IV over a rolling window)

Assumptions:
- bars is a pandas DataFrame with columns: 'high','low','close'
- index is time-ordered
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing (EMA with alpha = 1/period).

    Parameters
    ----------
    close : pd.Series
        Close prices (time-ordered).
    period : int
        RSI period.

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing = EMA(alpha=1/period)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(method="bfill").clip(0.0, 100.0)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ADX using Wilder's method (standard textbook ADX).

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC series (time-ordered).
    period : int
        ADX period.

    Returns
    -------
    pd.Series
        ADX values in [0, 100] typically.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = _true_range(high, low, close)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100.0 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0.0, np.nan))

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx.fillna(method="bfill").clip(lower=0.0, upper=100.0)


def iv_rank(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """
    IV Rank approximation as rolling percentile of current IV within window.

    Returns a 0-100 scale:
      IV Rank = 100 * (IV_t - min(IV_window)) / (max(IV_window) - min(IV_window))

    Parameters
    ----------
    iv_series : pd.Series
        Time series of IV values (e.g., ATM IV per bar/day).
    window : int
        Lookback window length (in observations). For 5-min bars,
        you likely want something like 78*20 (~1 month) or 78*60 (~3 months)
        depending on data volume.

    Returns
    -------
    pd.Series
        IV Rank values (0..100).
    """
    roll_min = iv_series.rolling(window).min()
    roll_max = iv_series.rolling(window).max()
    denom = (roll_max - roll_min).replace(0.0, np.nan)
    rank = 100.0 * (iv_series - roll_min) / denom
    return rank.fillna(method="bfill").clip(0.0, 100.0)


@dataclass
class IndicatorPack:
    """
    Convenience container for common indicators computed from bars and IV series.
    """
    adx: float
    rsi: float
    iv_rank: float

    @staticmethod
    def from_inputs(
        bars: pd.DataFrame,
        iv_atm_series: pd.Series,
        adx_period: int = 14,
        rsi_period: int = 14,
        iv_rank_window: int = 78 * 60,  # ~60 trading days of 5-min bars
    ) -> "IndicatorPack":
        adx_series = adx_wilder(bars["high"], bars["low"], bars["close"], period=adx_period)
        rsi_series = rsi_wilder(bars["close"], period=rsi_period)
        ivrank_series = iv_rank(iv_atm_series, window=iv_rank_window)

        return IndicatorPack(
            adx=float(adx_series.iloc[-1]),
            rsi=float(rsi_series.iloc[-1]),
            iv_rank=float(ivrank_series.iloc[-1]),
        )
