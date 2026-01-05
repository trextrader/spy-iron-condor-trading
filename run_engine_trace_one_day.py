"""
run_engine_trace_one_day.py

Runs TradingEngine using the real pipeline with mock data providers.
Prints a trace log for each snapshot and each stage of the pipeline.

Supports 1/5/15 minute timeframes via spot_bars_csv_map.
"""

from __future__ import annotations

from core.config import RunConfig
from core.engine import TradingEngine
from data_factory.data_engine import DataEngine
from core.types import MarketSnapshot


def main() -> int:
    # Create config
    cfg = RunConfig()

    # Pick one timeframe for the trace run: 1, 5, or 15
    tf = 5

    # Attach extra attributes for mock providers
    setattr(cfg, "symbol", "SPY")
    setattr(cfg, "bar_interval_minutes", tf)
    
    # Provide the file mapping for 1/5/15
    setattr(cfg, "spot_bars_csv_map", {
        1: "reports/SPY/SPY_1.csv",
        5: "reports/SPY/SPY_5.csv",
        15: "reports/SPY/SPY_15.csv",
    })

    # Point to your options marks file
    setattr(cfg, "options_chain_csv", "data/synthetic_options/spy_options_marks.csv")

    # Optional: constrain to a specific UTC day (must exist in BOTH spot and options)
    # If you omit this, it will run all timestamps available in the spot file.
    setattr(cfg, "trace_day_utc", "2025-07-03")

    # Rolling window length delivered to indicators/strategy
    setattr(cfg, "bars_window", 600)

    data = DataEngine(cfg)
    engine = TradingEngine(cfg=cfg, data=data)

    # ---- Trace wrappers ----
    original_stream = engine.data.stream

    def traced_stream():
        for snap in original_stream():
            assert isinstance(snap, MarketSnapshot)
            print(f"[SNAP] {snap.ts} {snap.symbol} spot={snap.spot:.2f} bars={len(snap.bars)} chain={len(snap.option_chain)}")
            yield snap

    engine.data.stream = traced_stream  # type: ignore

    orig_eval = engine.strategy.evaluate
    def traced_eval(snapshot):
        d = orig_eval(snapshot)
        print(f"[DEC ] trade={d.should_trade} bias={d.bias} rationale={list(d.rationale.keys())}")
        return d
    engine.strategy.evaluate = traced_eval  # type: ignore

    orig_size = engine.sizer.size
    def traced_size(decision, snapshot):
        s = orig_size(decision, snapshot)
        print(f"[SIZE] qty={s.contracts} conf={s.confidence:.3f}")
        return s
    engine.sizer.size = traced_size  # type: ignore

    orig_approve = engine.risk.approve
    def traced_approve(sized, snapshot):
        a = orig_approve(sized, snapshot)
        print(f"[RISK] ok={a.approved} reason={a.reason}")
        return a
    engine.risk.approve = traced_approve  # type: ignore

    orig_exec = engine.router.execute
    def traced_exec(plan, snapshot):
        if plan:
            print(f"[EXEC] orders={len(plan.orders)} meta={plan.metadata}")
        return orig_exec(plan, snapshot)
    engine.router.execute = traced_exec  # type: ignore

    print(f"[TRACE] Starting engine run (tf={tf}m)...")
    engine.run()
    print("[TRACE] Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
