"""
intelligence/fuzzifier.py

Feature extraction + fuzzification into linguistic membership degrees.
Uses real indicators (ADX/RSI/IV Rank) from analytics module.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from core.config import StrategyConfig
from core.types import MarketSnapshot
from analytics.indicators import adx_wilder, rsi_wilder, iv_rank


@dataclass
class Fuzzifier:
    cfg: StrategyConfig

    def _atm_iv(self, chain: pd.DataFrame, spot: float) -> float:
        if chain is None or len(chain) == 0 or "strike" not in chain.columns:
            return np.nan
        if "iv" not in chain.columns:
            return np.nan
        idx = (chain["strike"] - spot).abs().idxmin()
        iv = chain.loc[idx, "iv"]
        try:
            return float(iv)
        except Exception:
            return np.nan

    def extract_features(self, snapshot: MarketSnapshot) -> dict[str, float]:
        bars: pd.DataFrame = snapshot.bars
        chain: pd.DataFrame = snapshot.option_chain

        # ADX/RSI from 5m bars
        adx_series = adx_wilder(bars["high"], bars["low"], bars["close"], period=14)
        rsi_series = rsi_wilder(bars["close"], period=14)

        adx_val = float(adx_series.iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])

        # IV Rank: build a rolling ATM IV series from history of snapshots if available.
        # For a mock provider, we approximate using current ATM IV only; rank needs history.
        # We'll derive a short rolling history using last M bars by reusing current chain slice ATM IV.
        M = int(getattr(self.cfg, "iv_rank_lookback_bars", 78 * 20))  # ~20 days of 5m bars by default
        atm_iv_now = self._atm_iv(chain, snapshot.spot)

        # Create a simple series of length=min(M,len(bars)) filled with atm_iv_now
        # (Replace this later with a stored IV series per bar when you have it.)
        n = min(M, len(bars))
        iv_series = pd.Series([atm_iv_now] * n)
        ivr = iv_rank(iv_series, window=max(2, min(n, 252))).iloc[-1]
        ivr_val = float(ivr) if np.isfinite(ivr) else 50.0

        return {"adx": adx_val, "rsi": rsi_val, "iv_rank": ivr_val}

    def fuzzify(self, features: dict[str, float]) -> dict[str, dict[str, float]]:
        """
        Convert crisp inputs into membership degrees.
        Returns nested dict like memberships['adx']['ranging'] = 0.7
        """
        adx = features["adx"]
        rsi = features["rsi"]
        ivr = features["iv_rank"]

        # Minimal deterministic memberships (replace with skfuzzy membership functions later)
        adx_ranging = max(0.0, min(1.0, (30.0 - adx) / 15.0))
        adx_trending = 1.0 - adx_ranging

        rsi_neutral = max(0.0, 1.0 - abs(rsi - 50.0) / 25.0)
        rsi_overbought = max(0.0, min(1.0, (rsi - 65.0) / 20.0))
        rsi_oversold = max(0.0, min(1.0, (35.0 - rsi) / 20.0))

        iv_low = max(0.0, min(1.0, (40.0 - ivr) / 20.0))
        iv_high = 1.0 - iv_low

        return {
            "adx": {"ranging": adx_ranging, "trending": adx_trending},
            "rsi": {"oversold": rsi_oversold, "neutral": rsi_neutral, "overbought": rsi_overbought},
            "iv_rank": {"low": iv_low, "high": iv_high},
        }
