"""
data_factory/spot_bars.py

Mock SpotBarsProvider (30-minute OHLCV):
- Reads a CSV with at least timestamp, open, high, low, close, volume
- Streams rolling windows up to the current bar
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

    def _load_csv(self) -> pd.DataFrame:
        path = getattr(self.cfg, "spot_bars_csv", None)
        if not path:
            raise ValueError("RunConfig must include spot_bars_csv for SpotBarsProvider")

        df = pd.read_csv(path)

        ts_col = _detect_col(df, ["timestamp", "time", "datetime", "date"])
        if ts_col is None:
            raise ValueError(f"Could not detect timestamp column in {path}. Columns={list(df.columns)}")

        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)

        o = _detect_col(df, ["open", "o"])
        h = _detect_col(df, ["high", "h"])
        l = _detect_col(df, ["low", "l"])
        c = _detect_col(df, ["close", "c", "last"])
        v = _detect_col(df, ["volume", "vol", "v"])

        missing = [name for name, col in [("open", o), ("high", h), ("low", l), ("close", c), ("volume", v)] if col is None]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}. Columns={list(df.columns)}")

        out = pd.DataFrame(
            {
                "open": df[o].astype(float),
                "high": df[h].astype(float),
                "low": df[l].astype(float),
                "close": df[c].astype(float),
                "volume": df[v].astype(float),
            },
            index=df.index,
        )
        return out

    def stream(self) -> Iterable[tuple[Any, str, Any]]:
        bars = self._load_csv()

        symbol = getattr(self.cfg, "symbol", None) or (self.cfg.symbols[0] if hasattr(self.cfg, 'symbols') and self.cfg.symbols else self.cfg.underlying)

        # Optional single-day filter (UTC date string, e.g. "2024-01-02")
        day = getattr(self.cfg, "trace_day_utc", None)
        if day:
            day_start = pd.Timestamp(day).tz_localize("UTC")
            day_end = day_start + pd.Timedelta(days=1)
            bars = bars[(bars.index >= day_start) & (bars.index < day_end)]

        win = int(getattr(self.cfg, "bars_window", 200))  # 200x30m ~ 4 trading days

        for i in range(len(bars)):
            ts = bars.index[i]
            start = max(0, i - win + 1)
            window_df = bars.iloc[start : i + 1]
            yield ts, symbol, window_df
