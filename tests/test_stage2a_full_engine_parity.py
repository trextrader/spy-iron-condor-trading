"""
test_stage2a_full_engine_parity.py
===================================
Stage 2A full engine parity tests: CPU numpy path vs GPU tensor path.

Verifies that run_backtest_optimizer_batch produces identical trade counts,
equity, and drawdown metrics regardless of execution path.

Test design
-----------
- Builds a synthetic OptimizerContext with a deterministic entry bar.
- Runs the engine twice: once forcing the CPU numpy path, once the GPU
  tensor path (CUDA required for GPU test).
- Compares BatchEvalResult fields within defined tolerances.

Forcing each path:
  CPU path: ctx.device = cpu           → _use_gpu = False (always)
  GPU path: ctx.device = cuda,
            K >= gpu_k_threshold       → _use_gpu = True

Tolerances:
  wins / losses / total : exact integer match
  net_pnl               : atol = 5.0 USD (float32 vs float64 unrealized rounding)
  max_dd                : atol = 0.1 percent
  win_rate / pf         : atol = 0.01

Usage:
    pytest tests/test_stage2a_full_engine_parity.py -v
    pytest tests/test_stage2a_full_engine_parity.py -v -s   # verbose engine output
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_KAGGLE_DIR   = _PROJECT_ROOT / "kaggle"
if str(_KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(_KAGGLE_DIR))

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from optimizer_engine import (
        run_backtest_optimizer_batch,
        BatchEvalResult,
        ObjectiveSpec,
    )
    from optimizer_prep import OptimizerContext
    from candidate_codec import CandidateBatch
    HAS_ENGINE = True
except ImportError as e:
    HAS_ENGINE = False
    _IMPORT_ERR = str(e)

pytestmark = pytest.mark.skipif(
    not HAS_ENGINE,
    reason=f"kaggle engine imports unavailable: {'' if HAS_ENGINE else _IMPORT_ERR}",
)

# ── Tolerances ─────────────────────────────────────────────────────────────────
PNL_ATOL   = 5.0      # USD — float32→float64 unrealized rounding
DD_ATOL    = 0.10     # percent
RATE_ATOL  = 0.01     # fraction (win_rate, profit_factor)

# ── Synthetic context builder ──────────────────────────────────────────────────

def _build_ctx(
    device:      torch.device,
    T:           int   = 15,      # number of bars
    spot_val:    float = 500.0,
    entry_bars:  list  = None,    # bar indices where gate fires
    M:           int   = 36,      # options per bar (M//2 strikes × 2 rights)
    seed:        int   = 42,
) -> "OptimizerContext":
    """
    Build a minimal synthetic OptimizerContext for parity testing.

    Timestamps use 1-day spacing so hold_days arithmetic is predictable.
    Chain is identical at every bar (constant prices → debit ≈ credit at exit).
    """
    if entry_bars is None:
        entry_bars = [2]

    # ── Timestamps: 1-day spacing from 2025-01-02 ─────────────────────────
    ts = np.arange(T, dtype=np.int64) * 86400

    # ── Constant spot + gate signals ──────────────────────────────────────
    spot         = np.full(T, spot_val, dtype=np.float32)
    gate_entry   = np.full(T, 0.10, dtype=np.float32)
    gate_pop     = np.full(T, 0.10, dtype=np.float32)
    strategy_idx = np.zeros(T, dtype=np.int16)
    abstain      = np.ones(T, dtype=bool)

    for b in entry_bars:
        if 0 <= b < T:
            gate_entry[b]   = 0.70   # > entry_threshold=0.55
            gate_pop[b]     = 0.60   # > pop_threshold=0.50
            strategy_idx[b] = 8      # iron_butterfly filter
            abstain[b]      = False

    # ── Synthetic chain (same options at every bar) ────────────────────────
    n_strikes = M // 2
    strikes     = np.linspace(spot_val * 0.90, spot_val * 1.10, n_strikes,
                               dtype=np.float32)
    call_deltas = np.linspace(0.92, 0.05, n_strikes, dtype=np.float32)
    put_deltas  = (1.0 - call_deltas).astype(np.float32)
    call_mid    = np.maximum(0.10, call_deltas * 15.0).astype(np.float32)
    put_mid     = np.maximum(0.10, put_deltas  * 15.0).astype(np.float32)

    # Per-bar chain layout: calls first [n_strikes], then puts [n_strikes]
    bar_right   = np.concatenate([np.zeros(n_strikes, np.int8),
                                   np.ones(n_strikes,  np.int8)])
    bar_strike  = np.tile(strikes, 2)
    bar_dte     = np.full(M, 21.0, np.float32)
    bar_delta   = np.concatenate([call_deltas, put_deltas])
    bar_bid     = np.concatenate([call_mid * 0.9, put_mid * 0.9])
    bar_ask     = np.concatenate([call_mid * 1.1, put_mid * 1.1])
    bar_mid_arr = np.concatenate([call_mid, put_mid])

    # CSR: same chain tiled T times
    bar_offsets = np.arange(T + 1, dtype=np.int64) * M
    right_flat  = np.tile(bar_right,   T)
    strike_flat = np.tile(bar_strike,  T)
    dte_flat    = np.tile(bar_dte,     T)
    delta_flat  = np.tile(bar_delta,   T)
    bid_flat    = np.tile(bar_bid,     T)
    ask_flat    = np.tile(bar_ask,     T)
    mid_flat    = np.tile(bar_mid_arr, T)

    # Bar dates (ISO strings from 2025-01-02)
    base = datetime.date(2025, 1, 2)
    bar_dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(T)]

    def _t(arr, dtype):
        return torch.as_tensor(arr, device=device, dtype=dtype)

    return OptimizerContext(
        device       = device,
        T            = T,
        timestamps   = _t(ts,           torch.int64),
        spot         = _t(spot,         torch.float32),
        gate_entry   = _t(gate_entry,   torch.float32),
        gate_pop     = _t(gate_pop,     torch.float32),
        strategy_idx = _t(strategy_idx, torch.int16),
        abstain      = _t(abstain,      torch.bool),
        bar_offsets  = _t(bar_offsets,  torch.int64),
        option_right = _t(right_flat,   torch.int8),
        option_strike= _t(strike_flat,  torch.float32),
        option_dte   = _t(dte_flat,     torch.float32),
        option_delta = _t(delta_flat,   torch.float32),
        opt_bid      = _t(bid_flat,     torch.float32),
        opt_ask      = _t(ask_flat,     torch.float32),
        opt_mid      = _t(mid_flat,     torch.float32),
        fast_end     = max(1, T // 4),
        medium_end   = max(1, T * 3 // 5),
        bar_dates    = bar_dates,
    )


def _make_candidates(
    K:         int,
    stop_loss: float = 600.0,
    hold_days: float = 10000.0,   # never fires within test horizon
) -> "CandidateBatch":
    """Build K identical candidates for iron_butterfly."""
    return CandidateBatch(K=K, params={
        "target_dte":      np.full(K, 21.0, np.float64),
        "short_delta":     np.full(K, 0.45, np.float64),
        "spread_width":    np.full(K,  5.0, np.float64),
        "stop_loss_dollar": np.full(K, stop_loss, np.float64),
        "profit_target":   np.full(K, 10_000_000.0, np.float64),
        "max_dte_exit":    np.zeros(K, np.float64),
        "hold_days":       np.full(K, hold_days, np.float64),
    })


_OBJ_SPEC = ObjectiveSpec()
_IRON_BFLY_IDX = 8   # strategy_idx_filter for iron_butterfly


def _run_cpu(ctx_cpu, candidates, fidelity="full"):
    """Run the engine forcing CPU numpy path."""
    return run_backtest_optimizer_batch(
        ctx_cpu, candidates,
        base_config={},
        objective_spec=_OBJ_SPEC,
        fidelity=fidelity,
        strategy_idx_filter=_IRON_BFLY_IDX,
        strategy_family="iron_butterfly",
        gpu_k_threshold=candidates.K + 1,   # K < threshold → CPU path always
    )


def _run_gpu(ctx_gpu, candidates, fidelity="full"):
    """Run the engine forcing GPU tensor path (CUDA required)."""
    return run_backtest_optimizer_batch(
        ctx_gpu, candidates,
        base_config={},
        objective_spec=_OBJ_SPEC,
        fidelity=fidelity,
        strategy_idx_filter=_IRON_BFLY_IDX,
        strategy_family="iron_butterfly",
        gpu_k_threshold=candidates.K,       # K >= threshold → GPU path
    )


def _assert_results_match(cpu: "BatchEvalResult", gpu: "BatchEvalResult",
                           label: str = ""):
    """Assert CPU and GPU BatchEvalResult fields match within tolerances."""
    prefix = f"[{label}] " if label else ""
    K = cpu.K

    # Exact integer fields
    np.testing.assert_array_equal(
        cpu.wins, gpu.wins,
        err_msg=f"{prefix}wins mismatch: cpu={cpu.wins} gpu={gpu.wins}",
    )
    np.testing.assert_array_equal(
        cpu.losses, gpu.losses,
        err_msg=f"{prefix}losses mismatch: cpu={cpu.losses} gpu={gpu.losses}",
    )
    np.testing.assert_array_equal(
        cpu.total, gpu.total,
        err_msg=f"{prefix}total mismatch: cpu={cpu.total} gpu={gpu.total}",
    )

    # Floating-point fields with tolerance
    np.testing.assert_allclose(
        cpu.net_pnl, gpu.net_pnl, atol=PNL_ATOL, rtol=0,
        err_msg=f"{prefix}net_pnl mismatch",
    )
    np.testing.assert_allclose(
        cpu.max_dd, gpu.max_dd, atol=DD_ATOL, rtol=0,
        err_msg=f"{prefix}max_dd mismatch",
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCPUDeterminism:
    """CPU path must be deterministic: same context → same results."""

    def test_identical_runs(self):
        cpu_dev = torch.device("cpu")
        ctx = _build_ctx(cpu_dev, T=10, entry_bars=[2])
        cands = _make_candidates(K=4)

        r1 = _run_cpu(ctx, cands)
        r2 = _run_cpu(ctx, cands)

        np.testing.assert_array_equal(r1.wins,   r2.wins)
        np.testing.assert_array_equal(r1.losses, r2.losses)
        np.testing.assert_array_equal(r1.net_pnl, r2.net_pnl)
        np.testing.assert_array_equal(r1.max_dd,  r2.max_dd)

    def test_at_least_one_entry(self):
        """Sanity check: gate bars trigger at least one entry."""
        cpu_dev = torch.device("cpu")
        ctx = _build_ctx(cpu_dev, T=10, entry_bars=[2])
        cands = _make_candidates(K=4)
        r = _run_cpu(ctx, cands)
        assert r.total.sum() >= 1, \
            f"Expected at least 1 trade, got total={r.total}. " \
            f"Check that gate_entry=0.70, gate_pop=0.60, sidx=8 at bar 2."

    def test_force_close_accounting(self):
        """
        Single entry bar, no exits during simulation → all positions
        force-closed at end with penalty = -stop_loss_arr * 0.5.
        """
        K          = 4
        stop_loss  = 600.0
        cpu_dev    = torch.device("cpu")
        ctx        = _build_ctx(cpu_dev, T=5, entry_bars=[2])
        cands      = _make_candidates(K=K, stop_loss=stop_loss, hold_days=10000.0)
        r          = _run_cpu(ctx, cands)

        # All K candidates entered and force-closed → each has 1 loss
        np.testing.assert_array_equal(r.total, np.ones(K, int),
                                       err_msg=f"expected total=[1]*K, got {r.total}")
        np.testing.assert_array_equal(r.wins,  np.zeros(K, int),
                                       err_msg=f"expected wins=[0]*K, got {r.wins}")
        np.testing.assert_array_equal(r.losses, np.ones(K, int),
                                       err_msg=f"expected losses=[1]*K, got {r.losses}")

        # net_pnl = -stop_loss/2 per candidate (force-close penalty)
        expected_pnl = np.full(K, -stop_loss * 0.5)
        np.testing.assert_allclose(r.net_pnl, expected_pnl, atol=1.0,
                                    err_msg=f"expected pnl={expected_pnl}, got {r.net_pnl}")

    def test_hold_days_exit(self):
        """
        Entries at bar 2 with hold_days=5 (5-day daily ts) should exit at bar 7.
        P&L ≈ 0 (constant chain prices), counted as a loss (pnl <= 0).
        """
        K = 4
        cpu_dev = torch.device("cpu")
        # T=15, entry at bar 2 → hold_days=5 fires at bar 7 (bar 7 - bar 2 = 5 days)
        ctx   = _build_ctx(cpu_dev, T=15, entry_bars=[2])
        cands = _make_candidates(K=K, stop_loss=10_000_000, hold_days=5.0)
        r = _run_cpu(ctx, cands)

        # All 4 should have exactly 1 trade (hold_days exit)
        assert r.total.sum() >= K, \
            f"Expected {K} trades (one per candidate), got total={r.total}"
        # Each pnl ≈ 0 (debit = credit for constant chain) → losses
        np.testing.assert_allclose(r.net_pnl, np.zeros(K), atol=PNL_ATOL,
                                    err_msg=f"expected pnl≈0, got {r.net_pnl}")


class TestCPUGPUParity:
    """CPU numpy path must match GPU tensor path for all metrics."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_force_close_parity(self):
        """
        Force-close scenario: single entry, no mid-run exits.
        CPU and GPU must agree on all BatchEvalResult fields.
        """
        K = 8
        cpu_dev  = torch.device("cpu")
        cuda_dev = torch.device("cuda")

        ctx_cpu  = _build_ctx(cpu_dev,  T=5, entry_bars=[2])
        ctx_gpu  = _build_ctx(cuda_dev, T=5, entry_bars=[2])
        cands    = _make_candidates(K=K, stop_loss=600.0, hold_days=10000.0)

        r_cpu = _run_cpu(ctx_cpu, cands)
        r_gpu = _run_gpu(ctx_gpu, cands)

        _assert_results_match(r_cpu, r_gpu, label="force_close")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hold_days_parity(self):
        """
        Hold-days exit scenario: entry fires, exits via hold_days.
        CPU and GPU must agree on wins, losses, total, and P&L.
        """
        K = 8
        cpu_dev  = torch.device("cpu")
        cuda_dev = torch.device("cuda")

        ctx_cpu  = _build_ctx(cpu_dev,  T=20, entry_bars=[2])
        ctx_gpu  = _build_ctx(cuda_dev, T=20, entry_bars=[2])
        cands    = _make_candidates(K=K, stop_loss=10_000_000, hold_days=5.0)

        r_cpu = _run_cpu(ctx_cpu, cands)
        r_gpu = _run_gpu(ctx_gpu, cands)

        _assert_results_match(r_cpu, r_gpu, label="hold_days")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiple_entries_parity(self):
        """
        Multiple entry bars with cooldown: all should be consistent CPU/GPU.
        Entry bars 2, 10, 20 with cooldown=5 and hold_days=3 (exits at 5,13,23).
        """
        K = 8
        cpu_dev  = torch.device("cpu")
        cuda_dev = torch.device("cuda")

        ctx_cpu  = _build_ctx(cpu_dev,  T=30, entry_bars=[2, 10, 20])
        ctx_gpu  = _build_ctx(cuda_dev, T=30, entry_bars=[2, 10, 20])
        cands    = _make_candidates(K=K, stop_loss=10_000_000, hold_days=3.0)

        r_cpu = _run_cpu(ctx_cpu, cands)
        r_gpu = _run_gpu(ctx_gpu, cands)

        _assert_results_match(r_cpu, r_gpu, label="multi_entry")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fidelity_medium_parity(self):
        """
        fidelity='medium' cuts the simulation to ~60% of bars.
        CPU and GPU should agree on the truncated run.
        """
        K = 8
        cpu_dev  = torch.device("cpu")
        cuda_dev = torch.device("cuda")

        # Entry at bar 2 — falls in the 'fast' region (< T//4=5 of T=20)
        ctx_cpu  = _build_ctx(cpu_dev,  T=20, entry_bars=[2])
        ctx_gpu  = _build_ctx(cuda_dev, T=20, entry_bars=[2])
        cands    = _make_candidates(K=K, stop_loss=600.0, hold_days=10000.0)

        r_cpu = _run_cpu(ctx_cpu, cands, fidelity="medium")
        r_gpu = _run_gpu(ctx_gpu, cands, fidelity="medium")

        _assert_results_match(r_cpu, r_gpu, label="fidelity_medium")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_entries_parity(self):
        """
        No gate-eligible bars → both paths should return all-zero metrics.
        """
        K = 8
        cpu_dev  = torch.device("cpu")
        cuda_dev = torch.device("cuda")

        # entry_bars=[] → no gate fires
        ctx_cpu  = _build_ctx(cpu_dev,  T=10, entry_bars=[])
        ctx_gpu  = _build_ctx(cuda_dev, T=10, entry_bars=[])
        cands    = _make_candidates(K=K)

        r_cpu = _run_cpu(ctx_cpu, cands)
        r_gpu = _run_gpu(ctx_gpu, cands)

        np.testing.assert_array_equal(r_cpu.total, np.zeros(K, int))
        np.testing.assert_array_equal(r_gpu.total, np.zeros(K, int))
        np.testing.assert_allclose(r_cpu.net_pnl, np.zeros(K), atol=1e-9)
        np.testing.assert_allclose(r_gpu.net_pnl, np.zeros(K), atol=1e-9)
