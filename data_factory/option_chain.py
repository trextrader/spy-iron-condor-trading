"""
data_factory/option_chain.py

Mock OptionChainProvider for synthetic options marks file with columns:
timestamp,date,symbol,option_symbol,strike,expiration,contract_type,bid,ask,last_price,
bid_size,ask_size,volume,open_interest,delta,gamma,theta,vega,implied_volatility

Returns per-timestamp slices with ChainAlignment metadata.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd


@dataclass(frozen=True)
class ChainAlignment:
    chain: pd.DataFrame
    used_ts: Optional[pd.Timestamp]
    mode: str       # "exact" | "prior" | "stale" | "none"
    lag_sec: float
    iv_conf: float  # [0,1] confidence weight from lag decay


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

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        df["expiration"] = pd.to_datetime(df["expiration"], utc=True, errors="coerce")

        t = df["contract_type"].astype(str).str.lower()
        t = t.replace({"call": "C", "put": "P"})
        df["type"] = t.str.upper()

        out = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "symbol": df.get("symbol", "SPY"),
                "option_symbol": df.get("option_symbol"),
                "strike": pd.to_numeric(df["strike"], errors="coerce"),
                "expiry": df["expiration"],
                "type": df["type"],
                "bid": pd.to_numeric(df.get("bid"), errors="coerce"),
                "ask": pd.to_numeric(df.get("ask"), errors="coerce"),
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

        out["mid"] = (out["bid"].fillna(0.0) + out["ask"].fillna(0.0)) / 2.0
        out = out.dropna(subset=["strike"])
        self._df = out
        return out

    def _symbol_max_lag(self, symbol: str) -> int:
        mp = getattr(self.cfg, "max_option_lag_sec_by_symbol", None) or {}
        if symbol in mp:
            return int(mp[symbol])
        return int(getattr(self.cfg, "max_option_lag_sec", 600))

    def _decay_iv_conf(self, lag_sec: float) -> float:
        """
        Half-life exponential decay:
          iv_conf = 0.5 ** (lag_sec / iv_decay_half_life_sec)
        """
        hl = float(getattr(self.cfg, "iv_decay_half_life_sec", 300))
        if hl <= 0:
            return 1.0 if lag_sec <= 0 else 0.0
        if lag_sec <= 0:
            return 1.0
        return float(0.5 ** (lag_sec / hl))

    def get_chain(self, ts: Any, symbol: str) -> pd.DataFrame:
        """
        Backwards-compatible: return only chain DataFrame.
        For alignment metadata + iv_conf, use get_chain_with_meta().
        """
        aligned = self.get_chain_with_meta(ts, symbol)
        return aligned.chain

    def get_chain_with_meta(
        self,
        ts: Any,
        symbol: str,
        policy: Optional[str] = None,
        max_lag_sec: Optional[int] = None,
    ) -> ChainAlignment:
        """
        Align option snapshots to a spot timestamp.

        policy ∈ {"hard_cutoff","decay_only","decay_then_cutoff"}
        mode   ∈ {"exact","prior","stale","none"}
        """
        df = self._load()
        ts = pd.to_datetime(ts, utc=True)

        policy = policy or getattr(self.cfg, "lag_policy_default", "decay_then_cutoff")

        exact = df[df["timestamp"] == ts]
        if not exact.empty:
            return ChainAlignment(exact, ts, "exact", 0.0, 1.0)

        prior = df[df["timestamp"] <= ts]
        if prior.empty:
            return ChainAlignment(df.iloc[0:0], None, "none", float("nan"), 0.0)

        used_ts = pd.to_datetime(prior["timestamp"].iloc[-1], utc=True)
        lag_sec = float((ts - used_ts).total_seconds())
        iv_conf = self._decay_iv_conf(lag_sec)

        ml = int(max_lag_sec) if max_lag_sec is not None else self._symbol_max_lag(symbol)

        if policy == "hard_cutoff":
            if lag_sec > ml:
                return ChainAlignment(df.iloc[0:0], used_ts, "stale", lag_sec, 0.0)
            return ChainAlignment(df[df["timestamp"] == used_ts], used_ts, "prior", lag_sec, 1.0)

        if policy == "decay_only":
            return ChainAlignment(df[df["timestamp"] == used_ts], used_ts, "prior", lag_sec, iv_conf)

        # default: decay_then_cutoff
        if lag_sec > ml:
            return ChainAlignment(df.iloc[0:0], used_ts, "stale", lag_sec, 0.0)
        return ChainAlignment(df[df["timestamp"] == used_ts], used_ts, "prior", lag_sec, iv_conf)
