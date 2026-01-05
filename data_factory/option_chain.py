"""
data_factory/option_chain.py

Mock OptionChainProvider for your synthetic marks format:
spy_options_marksa.csv header:
timestamp,date,symbol,option_symbol,strike,expiration,contract_type,bid,ask,last_price,
bid_size,ask_size,volume,open_interest,delta,gamma,theta,vega,implied_volatility

Returns per-timestamp slices and normalizes to canonical columns:
timestamp, symbol, option_symbol, strike, expiry, type, bid, ask, mid, last, volume, oi, delta, gamma, theta, vega, iv
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd


@dataclass
class OptionChainProvider:
    cfg: Any
    _df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        path = getattr(self.cfg, "options_chain_csv", None)
        if not path:
            raise ValueError("Config must include options_chain_csv for OptionChainProvider")

        df = pd.read_csv(path)

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Normalize expiration
        df["expiration"] = pd.to_datetime(df["expiration"], utc=True, errors="coerce")

        # Normalize type to "C"/"P"
        t = df["contract_type"].astype(str).str.lower()
        df["type"] = t.replace({"call": "C", "put": "P"}).str.upper()

        # Canonical columns
        out = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "symbol": df.get("symbol", "SPY"),
                "option_symbol": df.get("option_symbol"),
                "strike": pd.to_numeric(df["strike"], errors="coerce"),
                "expiry": df["expiration"],
                "type": df["type"],
                "bid": pd.to_numeric(df["bid"], errors="coerce"),
                "ask": pd.to_numeric(df["ask"], errors="coerce"),
                "last": pd.to_numeric(df.get("last_price"), errors="coerce"),
                "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
                "oi": pd.to_numeric(df.get("open_interest"), errors="coerce"),
                "delta": pd.to_numeric(df.get("delta"), errors="coerce"),
                "gamma": pd.to_numeric(df.get("gamma"), errors="coerce"),
                "theta": pd.to_numeric(df.get("theta"), errors="coerce"),
                "vega": pd.to_numeric(df.get("vega"), errors="coerce"),
                "iv": pd.to_numeric(df.get("implied_volatility"), errors="coerce"),
            }
        )

        # Mid price
        out["mid"] = (out["bid"].fillna(0.0) + out["ask"].fillna(0.0)) / 2.0

        # Clean
        out = out.dropna(subset=["strike"])
        self._df = out
        return out

    def get_chain(self, ts: Any, symbol: str) -> pd.DataFrame:
        df = self._load()

        # Exact timestamp match
        s = df[df["timestamp"] == ts]
        if not s.empty:
            return s

        # Nearest prior timestamp (common if spot bars and options timestamps don't align)
        prior = df[df["timestamp"] <= ts]
        if prior.empty:
            return df.iloc[0:0]
        last_ts = prior["timestamp"].iloc[-1]
        return df[df["timestamp"] == last_ts]
