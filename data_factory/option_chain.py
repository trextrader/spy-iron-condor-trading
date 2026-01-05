"""
data_factory/option_chain.py

Mock OptionChainProvider for synthetic options marks file with columns:
timestamp,date,symbol,option_symbol,strike,expiration,contract_type,bid,ask,last_price,
bid_size,ask_size,volume,open_interest,delta,gamma,theta,vega,implied_volatility

Returns per-timestamp slices and normalizes to:
timestamp, symbol, option_symbol, strike, expiry, type, bid, ask, mid, last, volume, oi, delta, gamma, theta, vega, iv

Includes timestamp alignment diagnostics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
import numpy as np


@dataclass
class OptionChainProvider:
    cfg: Any
    _df: Optional[pd.DataFrame] = None
    
    # Alignment statistics
    _exact_matches: int = 0
    _fallback_prior: int = 0
    _lags_seconds: list = field(default_factory=list)
    _total_requests: int = 0

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
        t = t.replace({"call": "C", "put": "P"})
        df["type"] = t.str.upper()

        # Canonical columns
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

        # Mid price
        out["mid"] = (out["bid"].fillna(0.0) + out["ask"].fillna(0.0)) / 2.0

        # Clean
        out = out.dropna(subset=["strike"])
        self._df = out
        return out

    def get_chain(self, ts: Any, symbol: str) -> pd.DataFrame:
        df = self._load()
        self._total_requests += 1
        
        # Exact timestamp match
        s = df[df["timestamp"] == ts]
        if not s.empty:
            self._exact_matches += 1
            self._lags_seconds.append(0.0)
            return s

        # Nearest prior snapshot
        prior = df[df["timestamp"] <= ts]
        if prior.empty:
            return df.iloc[0:0]
        
        self._fallback_prior += 1
        last_ts = prior["timestamp"].iloc[-1]
        
        # Calculate lag in seconds
        lag_seconds = (pd.Timestamp(ts) - pd.Timestamp(last_ts)).total_seconds()
        self._lags_seconds.append(lag_seconds)
        
        return df[df["timestamp"] == last_ts]
    
    def print_alignment_stats(self) -> None:
        """Print timestamp alignment statistics."""
        if self._total_requests == 0:
            print("[ALIGN] No requests made yet")
            return
        
        exact_pct = 100.0 * self._exact_matches / self._total_requests
        fallback_pct = 100.0 * self._fallback_prior / self._total_requests
        
        lags = np.array(self._lags_seconds)
        median_lag = np.median(lags) if len(lags) > 0 else 0.0
        max_lag = np.max(lags) if len(lags) > 0 else 0.0
        
        print(f"\n[ALIGN] Timestamp Alignment Report:")
        print(f"  Total spot bars processed: {self._total_requests}")
        print(f"  Exact matches: {self._exact_matches} ({exact_pct:.1f}%)")
        print(f"  Nearest-prior fallback: {self._fallback_prior} ({fallback_pct:.1f}%)")
        print(f"  Median lag: {median_lag:.1f} seconds")
        print(f"  Max lag: {max_lag:.1f} seconds")
