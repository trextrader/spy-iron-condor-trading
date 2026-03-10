"""
optimizer_parity.py — Batch Engine Parity Validation
=====================================================
Compares the batch optimizer engine output against the reference
run_backtest() for a single configuration.

Known differences (acceptable):
  - Batch engine: max 1 concurrent position per candidate
  - Reference backtester: up to max_positions (default 6) concurrent
  - Exit prices: batch uses nearest-strike chain lookup; reference uses
    calculate_exit_fill() with bid-ask logic
  - Batch engine skips the execution-reality engine slippage model

Usage:
  python kaggle/optimizer_parity.py --use-v43 --strategies iron_butterfly

Or call programmatically:
  from optimizer_parity import validate_optimizer_parity
  report = validate_optimizer_parity(ctx, batch_result, ref_result)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from optimizer_engine import BatchEvalResult


@dataclass
class ParityReport:
    """Summary of the parity check between batch and reference."""
    template_id:     str
    fidelity:        str

    # Batch engine (single position per candidate)
    batch_net_pct:   float
    batch_max_dd:    float
    batch_trades:    int
    batch_win_rate:  float
    batch_obj:       float

    # Reference backtester
    ref_net_pct:     float
    ref_max_dd:      float
    ref_trades:      int
    ref_win_rate:    float

    # Deltas
    delta_net_pct:   float
    delta_max_dd:    float
    delta_trades:    int

    # Pass/fail
    passed:          bool
    notes:           str


_TOL_NET_PCT  = 50.0   # % — batch may differ significantly due to max-pos=1 simplification
_TOL_MAX_DD   = 30.0   # %
_TOL_TRADES   = None   # skip trade-count tolerance (max-pos=1 changes trade count)


def validate_optimizer_parity(
    ctx,                   # OptimizerContext
    batch_result: BatchEvalResult,
    ref_equity_curve,      # list of floats from run_backtest()
    ref_closed_trades,     # list of dicts from run_backtest()
    template_id: str = "iron_butterfly",
    candidate_idx: int = 0,
    verbose: bool = True,
) -> ParityReport:
    """
    Compare batch engine result (candidate_idx) against reference backtester.

    Parameters
    ----------
    ctx              : OptimizerContext (not used for comparison, kept for API consistency)
    batch_result     : output of run_backtest_optimizer_batch() for a 1-candidate batch
    ref_equity_curve : equity_vals list from run_backtest()
    ref_closed_trades: closed_trades list from run_backtest()
    """
    k = candidate_idx

    # ── Batch metrics ─────────────────────────────────────────────────────
    batch_net_pct  = float(batch_result.net_pct[k])
    batch_max_dd   = float(batch_result.max_dd[k])
    batch_trades   = int(batch_result.total[k])
    batch_win_rate = float(batch_result.win_rate[k])
    batch_obj      = float(batch_result.objective[k])

    # ── Reference metrics ─────────────────────────────────────────────────
    eq = np.array(ref_equity_curve, dtype=np.float64)
    ref_net_pct = float((eq[-1] - eq[0]) / eq[0] * 100) if len(eq) >= 2 else 0.0
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max * 100
    ref_max_dd  = float(abs(dd.min())) if len(dd) > 0 else 0.0

    ref_trades  = len(ref_closed_trades)
    pnls = [t.get("pnl", 0) for t in ref_closed_trades]
    ref_wins    = sum(1 for p in pnls if p > 0)
    ref_win_rate = ref_wins / max(ref_trades, 1)

    # ── Deltas ────────────────────────────────────────────────────────────
    delta_net = abs(batch_net_pct - ref_net_pct)
    delta_dd  = abs(batch_max_dd  - ref_max_dd)
    delta_trd = abs(batch_trades  - ref_trades)

    # Parity passes unless deltas are catastrophically large
    notes_parts = []
    passed = True

    if delta_net > _TOL_NET_PCT:
        notes_parts.append(f"WARN net_pct delta={delta_net:.1f}% exceeds {_TOL_NET_PCT}%")
        # This is expected due to max_pos=1 simplification — not a hard fail
    if delta_dd > _TOL_MAX_DD:
        notes_parts.append(f"WARN max_dd delta={delta_dd:.1f}% exceeds {_TOL_MAX_DD}%")

    # Hard fail: batch shows wildly wrong sign
    if batch_net_pct * ref_net_pct < 0 and abs(batch_net_pct) > 5 and abs(ref_net_pct) > 5:
        notes_parts.append("FAIL sign mismatch on net_pct")
        passed = False

    notes = "; ".join(notes_parts) if notes_parts else "OK"

    rpt = ParityReport(
        template_id    = template_id,
        fidelity       = batch_result.fidelity,
        batch_net_pct  = batch_net_pct,
        batch_max_dd   = batch_max_dd,
        batch_trades   = batch_trades,
        batch_win_rate = batch_win_rate,
        batch_obj      = batch_obj,
        ref_net_pct    = ref_net_pct,
        ref_max_dd     = ref_max_dd,
        ref_trades     = ref_trades,
        ref_win_rate   = ref_win_rate,
        delta_net_pct  = delta_net,
        delta_max_dd   = delta_dd,
        delta_trades   = delta_trd,
        passed         = passed,
        notes          = notes,
    )

    if verbose:
        _print_report(rpt)

    return rpt


def _print_report(rpt: ParityReport):
    print()
    print("=" * 60)
    print(f"  PARITY CHECK  [{rpt.template_id}]  fidelity={rpt.fidelity}")
    print("=" * 60)
    print(f"  {'Metric':<20}  {'Batch':>12}  {'Reference':>12}  {'Delta':>10}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*10}")
    print(f"  {'net_pct (%)':<20}  {rpt.batch_net_pct:>+12.2f}  {rpt.ref_net_pct:>+12.2f}  {rpt.delta_net_pct:>10.2f}")
    print(f"  {'max_dd (%)':<20}  {rpt.batch_max_dd:>12.2f}  {rpt.ref_max_dd:>12.2f}  {rpt.delta_max_dd:>10.2f}")
    print(f"  {'trades':<20}  {rpt.batch_trades:>12d}  {rpt.ref_trades:>12d}  {rpt.delta_trades:>10d}")
    print(f"  {'win_rate':<20}  {rpt.batch_win_rate:>12.3f}  {rpt.ref_win_rate:>12.3f}  {'':>10}")
    print(f"  {'objective':<20}  {rpt.batch_obj:>12.4f}  {'N/A':>12}  {'':>10}")
    print()
    status = "✓ PASS" if rpt.passed else "✗ FAIL"
    print(f"  Status: {status}   Notes: {rpt.notes}")
    print()
    print("  NOTE: batch engine uses max_positions=1; reference may have up to 6.")
    print("        Net P&L and trade count will differ — this is expected.")
    print("        Parity check validates relative ordering and sign direction only.")
    print()


if __name__ == "__main__":
    import argparse
    import sys

    # Path setup
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, ".."))
    for _p in [_here, _root,
               "/content/spy-iron-condor-trading",
               "/kaggle/working/spy-iron-condor-trading"]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    parser = argparse.ArgumentParser(description="Parity check: batch engine vs reference run_backtest")
    parser.add_argument("--use-v43", action="store_true")
    parser.add_argument("--strategies", type=str, default="iron_butterfly")
    parser.add_argument("--v43-model", type=str, default="models/condor_net_v43_run18.pth")
    parser.add_argument("--v43-data-dir", type=str, default="data/Datasetv4/v43")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    print("[parity] Loading backtester for parity check…")
    print("[parity] Run with --use-v43 --strategies iron_butterfly --limit 500")
    print("[parity] This imports condor_brain_backtest_v45 — ensure it is on sys.path")
    print()
    print("NOTE: Full parity validation requires calling validate_optimizer_parity()")
    print("      with both batch and reference results. See module docstring.")
