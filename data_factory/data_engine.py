"""
data_factory/data_engine.py

Streams MarketSnapshot objects by combining:
- spot bars window
- option chain slice
- aux feeds (gap inputs, etc.)

Includes AUTO-PICK overlap day feature:
If cfg.trace_day_utc is None and cfg.auto_pick_overlap_day is True,
it will choose the most recent UTC date that exists in BOTH spot and options.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import pandas as pd

from core.config import RunConfig
from core.types import MarketSnapshot
from data_factory.spot_bars import SpotBarsProvider
from data_factory.option_chain import OptionChainProvider
from data_factory.aux_feeds import AuxFeedsProvider


@dataclass
class DataEngine:
    cfg: RunConfig
    spot: SpotBarsProvider = None
    options: OptionChainProvider = None
    aux: AuxFeedsProvider = None

    def __post_init__(self) -> None:
        self.spot = self.spot or SpotBarsProvider(self.cfg)
        self.options = self.options or OptionChainProvider(self.cfg)
        self.aux = self.aux or AuxFeedsProvider(self.cfg)

        # Auto-pick overlap day if requested
        auto = bool(getattr(self.cfg, "auto_pick_overlap_day", False))
        if auto and getattr(self.cfg, "trace_day_utc", None) in (None, "", "auto"):
            self._auto_pick_overlap_day()

    def _auto_pick_overlap_day(self) -> None:
        """
        Find the most recent UTC date present in BOTH spot and option timestamps.
        Sets cfg.trace_day_utc = 'YYYY-MM-DD'.

        Raises:
            ValueError if no overlap exists.
        """
        # Load full spot bars (unfiltered)
        spot_df = self.spot._load_csv()  # expects datetime index (UTC)
        spot_days = set(spot_df.index.date)

        # Load full options marks (unfiltered)
        opt_df = self.options._load()    # expects 'timestamp' column (UTC)
        if "timestamp" not in opt_df.columns:
            raise ValueError("Options chain data missing 'timestamp' after normalization")
        opt_days = set(pd.to_datetime(opt_df["timestamp"], utc=True).dt.date)

        overlap = sorted(spot_days.intersection(opt_days))
        if not overlap:
            # Provide helpful bounds for debugging
            spot_min = spot_df.index.min()
            spot_max = spot_df.index.max()
            opt_min = pd.to_datetime(opt_df["timestamp"], utc=True).min()
            opt_max = pd.to_datetime(opt_df["timestamp"], utc=True).max()
            raise ValueError(
                "No overlapping UTC day between spot and options data.\n"
                f"Spot range:   {spot_min} -> {spot_max}\n"
                f"Options range: {opt_min} -> {opt_max}\n"
                "Fix: choose a trace day that exists in both, or point to matching datasets."
            )

        chosen = overlap[-1].isoformat()  # most recent overlap day
        try:
            setattr(self.cfg, "trace_day_utc", chosen)
        except Exception as e:
            raise ValueError(
                f"Could not set cfg.trace_day_utc (config may be frozen). "
                f"Chosen overlap day was {chosen}. "
                "Fix: add trace_day_utc field to RunConfig or make it mutable."
            ) from e

        print(f"[AUTO] trace_day_utc set to overlap day: {chosen}")

    def stream(self) -> Iterable[MarketSnapshot]:
        for ts, symbol, bars in self.spot.stream():
            # seed gap open/prev_close
            if hasattr(self.aux, "seed_from_bars"):
                self.aux.seed_from_bars(bars)

            chain = self.options.get_chain(ts, symbol)
            aux = self.aux.get(ts, symbol)

            yield MarketSnapshot(
                ts=ts,
                symbol=symbol,
                spot=float(bars["close"].iloc[-1]),
                bars=bars,
                option_chain=chain,
                vix=aux.get("vix"),
                es_price=aux.get("es_price"),
                prev_close=aux.get("prev_close"),
                open_price=aux.get("open_price"),
            )
