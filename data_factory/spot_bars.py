"""
data_factory/spot_bars.py

Mock SpotBarsProvider that reads 1m/5m/15m SPY spot files.

Supports two spot formats:
A) timestamp, close, volume
   -> Synthesizes open/high/low from close.
B) timestamp, open, high, low, close, volume
   -> Uses OHLCV directly.

Chooses file based on cfg.bar_interval_minutes in {1,5,15}.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Any, Optional
import pandas as pd

from core.config import RunConfig


def _detect_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


@dataclass
class SpotBarsProvider:
    cfg: RunConfig

    def _spot_path(self) -> str:
        """
        Resolve path from cfg using:
          - cfg.spot_bars_csv (if provided) OR
          - cfg.spot_bars_csv_map (dict keyed by minutes)
        """
        direct = getattr(self.cfg, "spot_bars_csv", None)
        if direct:
            return direct

        m = int(getattr(self.cfg, "bar_interval_minutes", 5))
        mp = getattr(self.cfg, "spot_bars_csv_map", None)
        if not mp or m not in mp:
            raise ValueError(
                "Provide cfg.spot_bars_csv or cfg.spot_bars_csv_map={1:...,5:...,15:...}"
            )
        return mp[m]

    def _load_csv(self) -> pd.DataFrame:
        path = self._spot_path()
        df = pd.read_csv(path)

        ts_col = _detect_col(df, ["timestamp", "time", "datetime", "date"])
        if ts_col is None:
            raise ValueError(f"Could not detect timestamp column in {path}. Columns={list(df.columns)}")

        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)

        # detect close & volume always
        c = _detect_col(df, ["close", "c", "last"])
        v = _detect_col(df, ["volume", "vol", "v"])
        if c is None:
            raise ValueError(f"Missing close column in {path}. Columns={list(df.columns)}")
        if v is None:
            # allow missing volume; fill with 0
            df["__volume__"] = 0.0
            v = "__volume__"

        # detect OHLC if present
        o = _detect_col(df, ["open", "o"])
        h = _detect_col(df, ["high", "h"])
        l = _detect_col(df, ["low", "l"])

        close = df[c].astype(float)
        volume = df[v].astype(float)

        if o and h and l:
            out = pd.DataFrame(
                {
                    "open": df[o].astype(float),
                    "high": df[h].astype(float),
                    "low": df[l].astype(float),
                    "close": close,
                    "volume": volume,
                },
                index=df.index,
            )
            return out

        # close-only format -> synthesize OHLC
        # open = previous close (first open = first close)
        open_ = close.shift(1).fillna(close)
        # high/low = close (minimal synthetic)
        high = close
        low = close

        out = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=df.index,
        )
        return out

    def stream(self) -> Iterable[tuple[Any, str, Any]]:
        bars = self._load_csv()

        symbol = getattr(self.cfg, "symbol", None) or (self.cfg.symbols[0] if hasattr(self.cfg, 'symbols') and self.cfg.symbols else self.cfg.underlying)

        # Optional: automatically select overlap day (spot/options) if provided by cfg.auto_trace_day
        # Otherwise filter by cfg.trace_day_utc if set.
        day = getattr(self.cfg, "trace_day_utc", None)
        if day:
            day_start = pd.Timestamp(day).tz_localize("UTC")
            day_end = day_start + pd.Timedelta(days=1)
            bars = bars[(bars.index >= day_start) & (bars.index < day_end)]

        win = int(getattr(self.cfg, "bars_window", 500))

        for i in range(len(bars)):
            ts = bars.index[i]
            start = max(0, i - win + 1)
            yield ts, symbol, bars.iloc[start : i + 1]
