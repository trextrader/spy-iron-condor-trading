"""
data_factory/aux_feeds.py

Mock AuxFeedsProvider:
- Provides prev_close/open for gap logic
- Placeholder for ES/VIX if you don't have them yet
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from core.config import RunConfig


@dataclass
class AuxFeedsProvider:
    cfg: RunConfig
    _prev_close: float | None = None
    _open_price: float | None = None
    _day_cached: str | None = None

    def seed_from_bars(self, bars_window) -> None:
        # Called opportunistically to set open/prev close once per day
        if bars_window is None or len(bars_window) == 0:
            return
        ts = bars_window.index[-1]
        day = ts.date().isoformat()
        if self._day_cached == day:
            return

        self._day_cached = day
        # best-effort: open = first bar open, prev close = first bar close (or prior day close if you feed it)
        self._open_price = float(bars_window["open"].iloc[0])
        self._prev_close = float(bars_window["close"].iloc[0])

    def get(self, ts: Any, symbol: str) -> dict[str, Any]:
        return {
            "vix": None,
            "es_price": None,
            "prev_close": self._prev_close,
            "open_price": self._open_price,
        }
