"""
run_engine_trace_one_day.py

Runs TradingEngine using the real pipeline with mock data providers.
Prints a trace log for each snapshot and each stage of the pipeline.
"""

from __future__ import annotations

from core.config import RunConfig
from core.engine import TradingEngine
from data_factory.data_engine import DataEngine
from core.types import MarketSnapshot


def main() -> int:
    # Create config
    cfg = RunConfig()

    # --- REQUIRED: set these paths to your real files ---
    # Attach extra attributes for mock providers
    setattr(cfg, "symbol", "SPY")
    setattr(cfg, "trace_day_utc", "2025-07-03")  # Date that exists in BOTH datasets
    setattr(cfg, "bars_window", 200)             # Rolling bars window (200x30m ~ 4 days)
    setattr(cfg, "spot_bars_csv", "reports/SPY/SPY_5.csv")  # Your 5m bars (or SPY_30.csv for 30m)
    setattr(cfg, "options_chain_csv", "data/synthetic_options/spy_options_marks.csv")  # Your synthetic chain
    setattr(cfg, "iv_rank_lookback_bars", 78 * 20)

    data = DataEngine(cfg)
    engine = TradingEngine(cfg=cfg, data=data)

    # Wrap run loop with trace prints
    original_stream = engine.data.stream

    def traced_stream():
        for snap in original_stream():
            assert isinstance(snap, MarketSnapshot)
            print(f"[TRACE SNAP] {snap.ts} {snap.symbol} spot={snap.spot:.2f} bars={len(snap.bars)} chain={len(snap.option_chain)}")
            yield snap

    engine.data.stream = traced_stream  # type: ignore

    # Wrap strategy evaluate
    orig_eval = engine.strategy.evaluate
    def traced_eval(snapshot):
        decision = orig_eval(snapshot)
        print(f"[TRACE DEC] should_trade={decision.should_trade} bias={decision.bias} rationale_keys={list(decision.rationale.keys())}")
        return decision
    engine.strategy.evaluate = traced_eval  # type: ignore

    # Wrap sizing
    orig_size = engine.sizer.size
    def traced_size(decision, snapshot):
        sized = orig_size(decision, snapshot)
        print(f"[TRACE SIZE] contracts={sized.contracts} conf={sized.confidence:.3f}")
        return sized
    engine.sizer.size = traced_size  # type: ignore

    # Wrap risk approve
    orig_approve = engine.risk.approve
    def traced_approve(sized, snapshot):
        ap = orig_approve(sized, snapshot)
        print(f"[TRACE RISK] approved={ap.approved} reason={ap.reason}")
        return ap
    engine.risk.approve = traced_approve  # type: ignore

    # Wrap router execute
    orig_exec = engine.router.execute
    def traced_exec(plan, snapshot):
        if plan:
            print(f"[TRACE EXEC] orders={len(plan.orders)} meta={plan.metadata}")
        return orig_exec(plan, snapshot)
    engine.router.execute = traced_exec  # type: ignore

    print(f"[TRACE] Starting engine run...")
    engine.run()
    print("[TRACE] Engine run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
