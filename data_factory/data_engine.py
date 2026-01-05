"""
data_factory/data_engine.py

Streams MarketSnapshot objects by combining:
- spot bars window
- option chain slice
- aux feeds (gap inputs, etc.)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

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
