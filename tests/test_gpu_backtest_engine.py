"""
tests/test_gpu_backtest_engine.py
GPU multi-position backtest engine invariant tests.
GPU tests skipped if CUDA unavailable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kaggle'))

import numpy as np
import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
T_SMALL = 30


# ── Fixture helpers ──────────────────────────────────────────────────────────

def _make_ctx(T=T_SMALL, gate_bars=None, device=None):
    """Minimal synthetic OptimizerContext."""
    from optimizer_prep import OptimizerContext
    if device is None:
        device = torch.device('cpu')

    N = T * 4  # 4 options per bar
    bar_offsets = torch.arange(0, N + 4, 4, dtype=torch.int64)[:T + 1]

    gate_entry   = torch.zeros(T, dtype=torch.float32)
    gate_pop     = torch.zeros(T, dtype=torch.float32)
    strategy_idx = torch.full((T,), 99, dtype=torch.int16)
    abstain      = torch.ones(T, dtype=torch.bool)

    if gate_bars:
        for b in gate_bars:
            gate_entry[b]   = 0.80
            gate_pop[b]     = 0.70
            strategy_idx[b] = 8        # custom_multi_leg → iron_butterfly
            abstain[b]      = False

    right_cycle  = torch.tensor([0, 0, 1, 1], dtype=torch.int8).repeat(T)
    strike_cycle = torch.tensor([505.0, 510.0, 495.0, 490.0],
                                 dtype=torch.float32).repeat(T)
    dte_cycle    = torch.full((N,), 7.0,  dtype=torch.float32)
    delta_cycle  = torch.full((N,), 0.45, dtype=torch.float32)
    bid_cycle    = torch.full((N,), 2.0,  dtype=torch.float32)
    ask_cycle    = torch.full((N,), 2.4,  dtype=torch.float32)
    mid_cycle    = torch.full((N,), 2.2,  dtype=torch.float32)

    return OptimizerContext(
        device=device,
        T=T,
        timestamps=torch.arange(T, dtype=torch.int64) * 300,
        spot=torch.full((T,), 500.0, dtype=torch.float32),
        gate_entry=gate_entry.to(device),
        gate_pop=gate_pop.to(device),
        strategy_idx=strategy_idx.to(device),
        abstain=abstain.to(device),
        bar_offsets=bar_offsets.to(device),
        option_right=right_cycle.to(device),
        option_strike=strike_cycle.to(device),
        option_dte=dte_cycle.to(device),
        option_delta=delta_cycle.to(device),
        opt_bid=bid_cycle.to(device),
        opt_ask=ask_cycle.to(device),
        opt_mid=mid_cycle.to(device),
        fast_end=T // 4,
        medium_end=T // 2,
        bar_dates=['2025-01-01'] * T,
    )


def _make_configs():
    """Minimal strategy configs for test."""
    return {
        'iron_butterfly': {
            'template_id':      'iron_butterfly',
            'class_name':       'custom_multi_leg',
            'class_idx':        8,
            'target_dte':       7.0,
            'short_delta':      0.45,
            'spread_width':     5.0,
            'stop_loss_dollar': 400.0,
            'profit_target':    800.0,
            'hold_days':        7.0,
            'max_dte_exit':     0.0,
        },
    }


# ── Tests: family dispatch helpers (always runs) ─────────────────────────────

class TestFamilyHelpers:

    def test_class_to_engine_family(self):
        from gpu_backtest_engine import _get_engine_family
        assert _get_engine_family('single_call')      == 'short_call'
        assert _get_engine_family('custom_multi_leg') == 'iron_butterfly'
        assert _get_engine_family('iron_condor')      == 'iron_condor'
        assert _get_engine_family('unknown_xyz')      == 'iron_butterfly'

    def test_intrinsic_debit_short_call(self):
        """Short call: debit = max(0, spot - strike)."""
        from gpu_backtest_engine import _intrinsic_debit_vec, FC_SHORT_CALL
        spot    = torch.tensor(510.0)
        ss_call = torch.tensor([505.0, 495.0], dtype=torch.float32)
        ss_put  = torch.zeros(2)
        width   = torch.zeros(2)
        fc      = torch.tensor([FC_SHORT_CALL, FC_SHORT_CALL], dtype=torch.int8)
        mask    = torch.tensor([True, True])
        debit   = _intrinsic_debit_vec(spot, ss_call, ss_put, width, fc, mask)
        assert abs(float(debit[0]) - 5.0)  < 1e-4, f"expected 5.0,  got {float(debit[0])}"
        assert abs(float(debit[1]) - 15.0) < 1e-4, f"expected 15.0, got {float(debit[1])}"

    def test_intrinsic_debit_closed_slot_is_zero(self):
        from gpu_backtest_engine import _intrinsic_debit_vec, FC_IRON_BUTTERFLY
        spot  = torch.tensor(500.0)
        fc    = torch.tensor([FC_IRON_BUTTERFLY], dtype=torch.int8)
        mask  = torch.tensor([False])
        debit = _intrinsic_debit_vec(spot,
                                     torch.tensor([505.0]),
                                     torch.tensor([495.0]),
                                     torch.tensor([5.0]),
                                     fc, mask)
        assert float(debit[0]) == 0.0, "Closed slot must return 0"


# ── Tests: run_backtest_gpu (CUDA required) ──────────────────────────────────

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestRunBacktestGpu:

    def test_equity_curve_length_equals_T(self):
        from gpu_backtest_engine import run_backtest_gpu
        T   = T_SMALL
        ctx = _make_ctx(T=T, device=torch.device('cuda'))
        eq, trades = run_backtest_gpu(
            ctx, _make_configs(), None, torch.device('cuda'),
            max_positions=2, friday_closeout=False,
        )
        assert len(eq) == T, f"Expected {T} equity entries, got {len(eq)}"

    def test_no_trades_when_all_abstain(self):
        """All bars abstain → no trades, equity unchanged."""
        from gpu_backtest_engine import run_backtest_gpu, STARTING_EQUITY
        ctx = _make_ctx(T=T_SMALL, gate_bars=None, device=torch.device('cuda'))
        eq, trades = run_backtest_gpu(
            ctx, _make_configs(), None, torch.device('cuda'),
            max_positions=2, friday_closeout=False,
        )
        opens  = [e for e in trades if e['action'] == 'OPEN']
        closes = [e for e in trades if e['action'] == 'CLOSE']
        assert len(opens)  == 0, f"Expected 0 opens,  got {len(opens)}"
        assert len(closes) == 0, f"Expected 0 closes, got {len(closes)}"
        assert all(abs(e - STARTING_EQUITY) < 0.01 for e in eq), \
            "Equity must be flat when no trades"

    def test_trade_events_have_required_keys(self):
        """OPEN events must have action/idx/credit; CLOSE must have action/idx/pnl/pnl_pct."""
        from gpu_backtest_engine import run_backtest_gpu
        ctx = _make_ctx(T=50, gate_bars=[10, 20, 30], device=torch.device('cuda'))
        eq, trades = run_backtest_gpu(
            ctx, _make_configs(), None, torch.device('cuda'),
            max_positions=3, friday_closeout=False,
        )
        for e in trades:
            assert 'action' in e and 'idx' in e, f"Missing required keys: {e}"
            if e['action'] == 'OPEN':
                assert 'credit' in e, f"OPEN missing 'credit': {e}"
            if e['action'] == 'CLOSE':
                assert 'pnl' in e and 'pnl_pct' in e, f"CLOSE missing 'pnl'/'pnl_pct': {e}"

    def test_allowed_template_filter_respected(self):
        """allowed_template_ids={'nonexistent_template'} → no trades."""
        from gpu_backtest_engine import run_backtest_gpu
        ctx = _make_ctx(T=50, gate_bars=[10, 20], device=torch.device('cuda'))
        eq, trades = run_backtest_gpu(
            ctx, _make_configs(),
            allowed_template_ids={'nonexistent_template'},
            device=torch.device('cuda'),
            max_positions=2, friday_closeout=False,
        )
        opens = [e for e in trades if e['action'] == 'OPEN']
        assert len(opens) == 0, "No opens expected when no template matches filter"

    def test_existing_tests_still_pass(self):
        """Stage 3A compile parity tests unaffected."""
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'pytest',
             'tests/test_stage3a_compile_parity.py', '-v', '--tb=short'],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..'),
        )
        assert result.returncode == 0, (
            f"Stage 3A tests failed:\n{result.stdout[-2000:]}\n{result.stderr[-1000:]}")
