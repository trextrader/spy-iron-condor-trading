"""
data_factory/data_engine.py

Streams MarketSnapshot objects by combining:
- spot bars window
- option chain slice
- aux feeds (gap inputs, etc.)

Includes:
- AUTO-PICK overlap day
- Lag-aware alignment with IV confidence decay
- Fail-fast mode
- Comprehensive diagnostics
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import pandas as pd
import numpy as np

from core.config import RunConfig
from core.dto import MarketSnapshot
from data_factory.spot_bars import SpotBarsProvider
from data_factory.option_chain import OptionChainProvider
from data_factory.aux_feeds import AuxFeedsProvider


@dataclass
class DataEngine:
    cfg: RunConfig
    spot: SpotBarsProvider = None
    options: OptionChainProvider = None
    aux: AuxFeedsProvider = None
    _diag_enabled: bool = False
    _diag_rows: list = None

    def __post_init__(self) -> None:
        self.spot = self.spot or SpotBarsProvider(self.cfg)
        self.options = self.options or OptionChainProvider(self.cfg)
        self.aux = self.aux or AuxFeedsProvider(self.cfg)

        self._diag_enabled = bool(getattr(self.cfg, "alignment_diagnostics", False))
        self._diag_rows = []

        auto = bool(getattr(self.cfg, "auto_pick_overlap_day", False))
        if auto and getattr(self.cfg, "trace_day_utc", None) in (None, "", "auto"):
            self._auto_pick_overlap_day()

    def _auto_pick_overlap_day(self) -> None:
        spot_df = self.spot._load_csv()
        spot_days = set(spot_df.index.date)

        opt_df = self.options._load()
        opt_days = set(pd.to_datetime(opt_df["timestamp"], utc=True).dt.date)

        overlap = sorted(spot_days.intersection(opt_days))
        if not overlap:
            spot_min, spot_max = spot_df.index.min(), spot_df.index.max()
            opt_min = pd.to_datetime(opt_df["timestamp"], utc=True).min()
            opt_max = pd.to_datetime(opt_df["timestamp"], utc=True).max()
            raise ValueError(
                "No overlapping UTC day between spot and options data.\n"
                f"Spot range:   {spot_min} -> {spot_max}\n"
                f"Options range:{opt_min} -> {opt_max}\n"
            )

        chosen = overlap[-1].isoformat()
        try:
            setattr(self.cfg, "trace_day_utc", chosen)
        except Exception as e:
            raise ValueError(
                f"Could not set cfg.trace_day_utc (config may be frozen). "
                f"Chosen overlap day was {chosen}. Add trace_day_utc to RunConfig or make config mutable."
            ) from e
        print(f"[AUTO] trace_day_utc set to overlap day: {chosen}")

    def _emit_alignment_summary(self) -> None:
        if not self._diag_enabled or not self._diag_rows:
            return
        df = pd.DataFrame(self._diag_rows)
        total = len(df)
        exact = int((df["mode"] == "exact").sum())
        prior = int((df["mode"] == "prior").sum())
        stale = int((df["mode"] == "stale").sum())
        none = int((df["mode"] == "none").sum())

        distinct_used_ts = df.loc[df["used_ts"].notna(), "used_ts"].nunique()

        lags = df.loc[df["mode"].isin(["exact", "prior"]), "lag_sec"].astype(float)
        ivc = df.loc[df["mode"].isin(["exact", "prior"]), "iv_conf"].astype(float)

        def _q(series, q):
            return float(np.quantile(series, q)) if len(series) else float("nan")

        mx_lag = float(np.max(lags)) if len(lags) else float("nan")
        mn_ivc = float(np.min(ivc)) if len(ivc) else float("nan")

        print("\n[ALIGNMENT DIAGNOSTICS]")
        print(f"  Total spot bars:                 {total}")
        print(f"  Exact match:                     {exact} ({exact/total:.1%})")
        print(f"  Fallback to prior snapshot:      {prior} ({prior/total:.1%})")
        print(f"  Stale (cutoff triggered):        {stale} ({stale/total:.1%})")
        print(f"  No options snapshot available:   {none} ({none/total:.1%})")
        print(f"  Distinct options timestamps used:{distinct_used_ts}")
        print(f"  Lag sec: median={_q(lags,0.5):.1f}  p90={_q(lags,0.90):.1f}  max={mx_lag:.1f}")
        print(f"  IV conf: median={_q(ivc,0.5):.3f}  p10={_q(ivc,0.10):.3f}  min={mn_ivc:.3f}")
        print("  Notes: 'stale' => lag exceeded max_option_lag_sec; chain treated as empty.\n")

    def stream(self) -> Iterable[MarketSnapshot]:
        total = 0
        stale = 0

        # Strategy-level override: stored by runner/engine as cfg._strategy_ref
        strategy = getattr(self.cfg, "_strategy_ref", None)
        strat_policy = strategy.alignment_policy() if strategy and hasattr(strategy, "alignment_policy") else {}
        policy = strat_policy.get("policy", getattr(self.cfg, "lag_policy_default", "decay_then_cutoff"))
        strat_max_lag = strat_policy.get("max_lag_sec", None)
        allow_no_chain = bool(strat_policy.get("allow_trade_without_chain", False))

        fail_fast_rate = float(getattr(self.cfg, "fail_fast_stale_rate", 1.0))
        fail_fast_min_bars = int(getattr(self.cfg, "fail_fast_min_bars", 50))

        try:
            for ts, symbol, bars in self.spot.stream():
                if hasattr(self.aux, "seed_from_bars"):
                    self.aux.seed_from_bars(bars)
                aux = self.aux.get(ts, symbol)

                used_ts = None
                mode = "unknown"
                lag_sec = None
                iv_conf = None

                if hasattr(self.options, "get_chain_with_meta"):
                    aligned = self.options.get_chain_with_meta(ts, symbol, policy=policy, max_lag_sec=strat_max_lag)
                    chain = aligned.chain
                    used_ts = aligned.used_ts
                    mode = aligned.mode
                    lag_sec = aligned.lag_sec
                    iv_conf = aligned.iv_conf
                else:
                    chain = self.options.get_chain(ts, symbol)

                total += 1
                if mode == "stale":
                    stale += 1

                if total >= fail_fast_min_bars:
                    stale_rate = (stale / total) if total else 0.0
                    if stale_rate > fail_fast_rate:
                        raise RuntimeError(
                            f"Fail-fast: stale_rate={stale_rate:.1%} > {fail_fast_rate:.1%} "
                            f"after {total} bars (stale={stale})."
                        )

                if (not allow_no_chain) and (chain is None or len(chain) == 0):
                    pass

                snap = MarketSnapshot(
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

                # attach metadata (works even if MarketSnapshot is older)
                try:
                    snap.option_used_ts = used_ts
                    snap.option_align_mode = mode
                    snap.option_lag_sec = lag_sec
                    snap.option_iv_conf = iv_conf
                except Exception:
                    pass

                if self._diag_enabled:
                    self._diag_rows.append(
                        {
                            "spot_ts": pd.to_datetime(ts, utc=True),
                            "used_ts": used_ts,
                            "mode": mode,
                            "lag_sec": float(lag_sec) if lag_sec is not None else float("nan"),
                            "iv_conf": float(iv_conf) if iv_conf is not None else float("nan"),
                            "chain_rows": int(len(chain)) if chain is not None else 0,
                            "symbol": symbol,
                        }
                    )

                yield snap
        finally:
            self._emit_alignment_summary()
