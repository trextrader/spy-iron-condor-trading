"""
test_stage3a_compile_parity.py — Stage 3A step_bar_gpu parity tests
====================================================================
Verifies that step_bar_gpu produces numerically identical results to the
original optimizer_engine.py GPU path.

Test classes
------------
TestStepBarGpuUnit          — unit tests for individual bar types (CPU)
TestStepBarGpuVsEngine      — full-run parity vs optimizer_engine (CPU forced)
TestCompileParity           — compiled vs eager parity (CUDA required)

Tolerances
----------
  net_pnl    atol=5 USD      (float64 arithmetic, tolerance for float32 chain prices)
  max_dd     atol=0.1 %
  wins/losses exact int match
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Path setup ──────────────────────────────────────────────────────────────────
_KAGGLE = Path(__file__).parent.parent / "kaggle"
if str(_KAGGLE) not in sys.path:
    sys.path.insert(0, str(_KAGGLE))

from optimizer_engine    import run_backtest_optimizer_batch, ObjectiveSpec, STARTING_EQUITY
from optimizer_prep      import OptimizerContext
from candidate_codec     import CandidateBatch
from optimizer_stage3a   import (
    BarState, init_bar_state, step_bar_gpu,
    make_compiled_step,
    FAMILY_TO_CODE, FAMILY_IRON_BUTTERFLY, FAMILY_IRON_CONDOR,
    FAMILY_SHORT_CALL, FAMILY_SHORT_PUT,
    CODE_TO_STR, _IC_MULTIPLIER,
)

_OBJ_SPEC = ObjectiveSpec()

# ── Shared fixtures ─────────────────────────────────────────────────────────────

def _build_ctx(
    device: torch.device,
    T: int = 40,
    spot_val: float = 500.0,
    entry_bars: list | None = None,
    M: int = 36,
    seed: int = 42,
) -> OptimizerContext:
    """Synthetic context with daily timestamps (deterministic hold_days)."""
    rng = np.random.default_rng(seed)
    base = datetime.date(2025, 1, 2)
    # Daily timestamps — 86400 s apart — so days_held = integer
    ts = np.array([
        int(datetime.datetime(
            *(base + datetime.timedelta(days=i)).timetuple()[:3],
            9, 30,
        ).timestamp())
        for i in range(T)
    ], dtype=np.int64)

    spot = np.full(T, spot_val, np.float32)
    gate_e   = np.full(T, 0.10, np.float32)
    gate_p   = np.full(T, 0.10, np.float32)
    sidx     = np.zeros(T, np.int16)
    abstain  = np.ones(T, bool)

    if entry_bars:
        for b in entry_bars:
            gate_e[b]  = 0.70
            gate_p[b]  = 0.60
            sidx[b]    = 8
            abstain[b] = False

    n_strikes = M // 2
    strikes   = np.linspace(spot_val * 0.88, spot_val * 1.12, n_strikes, dtype=np.float32)
    c_delta   = np.linspace(0.92, 0.05, n_strikes, dtype=np.float32)
    p_delta   = (1.0 - c_delta).astype(np.float32)
    c_mid     = np.maximum(0.10, c_delta * 15.0).astype(np.float32)
    p_mid     = np.maximum(0.10, p_delta * 15.0).astype(np.float32)

    bar_right  = np.concatenate([np.zeros(n_strikes, np.int8), np.ones(n_strikes, np.int8)])
    bar_strike = np.tile(strikes, 2)
    bar_dte    = np.full(M, 21.0, np.float32)
    bar_delta  = np.concatenate([c_delta, p_delta])
    bar_bid    = np.concatenate([c_mid * 0.9, p_mid * 0.9])
    bar_ask    = np.concatenate([c_mid * 1.1, p_mid * 1.1])

    offsets    = np.arange(T + 1, dtype=np.int64) * M
    right_flat = np.tile(bar_right,  T)
    stk_flat   = np.tile(bar_strike, T)
    dte_flat   = np.tile(bar_dte,    T)
    dlt_flat   = np.tile(bar_delta,  T)
    bid_flat   = np.tile(bar_bid,    T)
    ask_flat   = np.tile(bar_ask,    T)

    bar_dates  = [(base + datetime.timedelta(days=i)).isoformat() for i in range(T)]

    def _t(arr, dtype):
        return torch.as_tensor(arr, device=device, dtype=dtype)

    mid_flat = (bid_flat + ask_flat) / 2.0

    return OptimizerContext(
        device        = device,
        T             = T,
        timestamps    = _t(ts,         torch.int64),
        spot          = _t(spot,       torch.float32),
        gate_entry    = _t(gate_e,     torch.float32),
        gate_pop      = _t(gate_p,     torch.float32),
        strategy_idx  = _t(sidx,       torch.int16),
        abstain       = _t(abstain,    torch.bool),
        bar_offsets   = _t(offsets,    torch.int64),
        option_right  = _t(right_flat, torch.int8),
        option_strike = _t(stk_flat,   torch.float32),
        option_dte    = _t(dte_flat,   torch.float32),
        option_delta  = _t(dlt_flat,   torch.float32),
        opt_bid       = _t(bid_flat,   torch.float32),
        opt_ask       = _t(ask_flat,   torch.float32),
        opt_mid       = _t(mid_flat,   torch.float32),
        fast_end      = max(1, T // 4),
        medium_end    = max(1, T * 3 // 5),
        bar_dates     = bar_dates,
    )


def _make_candidates(K: int, stop_loss: float = 600.0,
                     hold_days: float = 7.0) -> CandidateBatch:
    return CandidateBatch(K=K, params={
        "target_dte":       np.full(K, 21.0),
        "short_delta":      np.full(K, 0.45),
        "spread_width":     np.full(K,  5.0),
        "stop_loss_dollar": np.full(K, stop_loss),
        "profit_target":    np.full(K, 1500.0),
        "max_dte_exit":     np.zeros(K),
        "hold_days":        np.full(K, hold_days),
    })


def _run_with_step_bar(
    ctx: OptimizerContext,
    candidates: CandidateBatch,
    *,
    family:       str = "iron_butterfly",
    gate_thresh:  float = 0.55,
    pop_thresh:   float = 0.50,
    sidx_filter:  int = 8,
    cooldown:     int = 5,
) -> dict:
    """
    Reference loop using step_bar_gpu for each bar.
    Mirrors run_backtest_optimizer_batch GPU path logic exactly.
    Returns dict of final numpy arrays matching BatchEvalResult fields.
    """
    K   = candidates.K
    dev = ctx.device
    fam = FAMILY_TO_CODE[family]

    # Pull small ctx arrays to numpy (same as optimizer_engine)
    spot_np       = ctx.spot.cpu().numpy()
    gate_e_np     = ctx.gate_entry.cpu().numpy()
    gate_p_np     = ctx.gate_pop.cpu().numpy()
    sidx_np       = ctx.strategy_idx.cpu().numpy()
    abstain_np    = ctx.abstain.cpu().numpy().astype(bool)
    offsets_np    = ctx.bar_offsets.cpu().numpy()
    ts_t          = torch.tensor(ctx.timestamps.cpu().numpy().astype(np.float64),
                                 dtype=torch.float64, device=dev)

    # Candidate param tensors on device
    p     = candidates.params
    td_t  = torch.from_numpy(p["target_dte"].astype(np.float32)).to(dev)
    sd_t  = torch.from_numpy(p["short_delta"].astype(np.float32)).to(dev)
    sw_t  = torch.from_numpy(p["spread_width"].astype(np.float32)).to(dev)
    sl_t  = torch.tensor(p["stop_loss_dollar"], dtype=torch.float64, device=dev)
    pt_t  = torch.tensor(p["profit_target"],    dtype=torch.float64, device=dev)
    mde_t = torch.tensor(p["max_dte_exit"],     dtype=torch.float64, device=dev)
    hd_t  = torch.tensor(p["hold_days"],        dtype=torch.float64, device=dev)

    # Chain GPU tensors (pinned to device)
    right_gpu  = ctx.option_right
    stk_gpu    = ctx.option_strike
    dte_gpu    = ctx.option_dte
    dlt_gpu    = ctx.option_delta
    bid_gpu    = ctx.opt_bid
    ask_gpu    = ctx.opt_ask

    state = init_bar_state(K, dev)

    T_end = ctx.T
    for i in range(T_end):
        s_off = int(offsets_np[i])
        e_off = int(offsets_np[i + 1])

        # Gate check (Python scalar — same as optimizer_engine)
        es   = float(gate_e_np[i])
        pop  = float(gate_p_np[i])
        sidx = int(sidx_np[i])
        abt  = bool(abstain_np[i])
        gate_ok = (es > gate_thresh) and (pop > pop_thresh) and \
                  (not abt) and (sidx == sidx_filter)

        # Chain slice — zero-copy views
        r_sl = right_gpu[s_off:e_off]
        s_sl = stk_gpu[s_off:e_off]
        d_sl = dte_gpu[s_off:e_off]
        da_sl = dlt_gpu[s_off:e_off]
        b_sl = bid_gpu[s_off:e_off]
        a_sl = ask_gpu[s_off:e_off]

        state = step_bar_gpu(
            state      = state,
            bar_idx    = i,
            spot       = float(spot_np[i]),
            ts_t       = ts_t,
            chain_right   = r_sl,
            chain_strike  = s_sl,
            chain_dte     = d_sl,
            chain_bid     = b_sl,
            chain_ask     = a_sl,
            chain_delta   = da_sl,
            td_t = td_t, sd_t = sd_t, sw_t = sw_t,
            sl_t = sl_t, pt_t = pt_t, mde_t = mde_t, hd_t = hd_t,
            gate_ok       = gate_ok,
            cooldown_bars = cooldown,
            family_code   = fam,
            K             = K,
        )

    # Force-close any remaining positions (mirror optimizer_engine logic)
    if state.open_mask.any():
        fc_pnl   = torch.where(state.open_mask, -sl_t.abs() * 0.5,
                               torch.zeros_like(state.equity))
        equity_t = torch.where(state.open_mask, state.equity + fc_pnl, state.equity)
        peak_t   = torch.maximum(state.peak, equity_t)
        fc_dd    = torch.where(peak_t > 0,
                               (peak_t - equity_t) / peak_t * 100.0,
                               torch.zeros_like(peak_t))
        max_dd_t = torch.maximum(state.max_dd, fc_dd)
        fc_w     = state.open_mask & (fc_pnl > 0)
        fc_l     = state.open_mask & ~(fc_pnl > 0)
        wins_t   = torch.where(fc_w, state.wins   + 1, state.wins)
        losses_t = torch.where(fc_l, state.losses + 1, state.losses)
        gw_t     = torch.where(fc_w, state.gross_win  + fc_pnl,       state.gross_win)
        gl_t     = torch.where(fc_l, state.gross_loss + fc_pnl.abs(), state.gross_loss)
    else:
        equity_t = state.equity
        max_dd_t = state.max_dd
        wins_t   = state.wins
        losses_t = state.losses
        gw_t     = state.gross_win
        gl_t     = state.gross_loss

    return {
        "equity":     equity_t.cpu().numpy(),
        "max_dd":     max_dd_t.cpu().numpy(),
        "wins":       wins_t.cpu().numpy(),
        "losses":     losses_t.cpu().numpy(),
        "gross_win":  gw_t.cpu().numpy(),
        "gross_loss": gl_t.cpu().numpy(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TestStepBarGpuUnit — CPU device, single-bar unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStepBarGpuUnit:
    """Unit tests for step_bar_gpu on CPU device (no CUDA required)."""

    def _make_state(self, K: int = 4) -> BarState:
        return init_bar_state(K, torch.device("cpu"))

    def _empty_chain(self, device):
        dtype_map = {
            "right":  torch.int8,
            "strike": torch.float32,
            "dte":    torch.float32,
            "bid":    torch.float32,
            "ask":    torch.float32,
            "delta":  torch.float32,
        }
        return {k: torch.zeros(0, dtype=v, device=device) for k, v in dtype_map.items()}

    def test_no_op_bar_no_gate_no_chain(self):
        """Bar with no gate and no chain leaves state unchanged."""
        K = 4
        state = self._make_state(K)
        ch = self._empty_chain(torch.device("cpu"))
        ts = torch.arange(100, dtype=torch.float64)
        p  = _make_candidates(K)
        dev = torch.device("cpu")

        new_state = step_bar_gpu(
            state=state, bar_idx=5, spot=500.0, ts_t=ts,
            chain_right=ch["right"], chain_strike=ch["strike"],
            chain_dte=ch["dte"], chain_bid=ch["bid"], chain_ask=ch["ask"],
            chain_delta=ch["delta"],
            td_t=torch.full((K,), 21.0),
            sd_t=torch.full((K,), 0.45),
            sw_t=torch.full((K,), 5.0),
            sl_t=torch.full((K,), 600.0, dtype=torch.float64),
            pt_t=torch.full((K,), 1500.0, dtype=torch.float64),
            mde_t=torch.zeros(K, dtype=torch.float64),
            hd_t=torch.full((K,), 7.0, dtype=torch.float64),
            gate_ok=False, cooldown_bars=5, family_code=FAMILY_IRON_BUTTERFLY, K=K,
        )

        # All state must be unchanged
        assert new_state.equity.tolist()    == state.equity.tolist()
        assert new_state.open_mask.tolist() == state.open_mask.tolist()
        assert new_state.wins.tolist()      == state.wins.tolist()
        assert new_state.losses.tolist()    == state.losses.tolist()
        assert new_state.max_dd.tolist()    == state.max_dd.tolist()

    def test_exit_stop_loss(self):
        """Position with large loss triggers stop-loss exit."""
        K = 2
        dev = torch.device("cpu")
        state = init_bar_state(K, dev)

        # Manually open a position for candidate 0.
        # Iron butterfly: short at 505 ±5 wings (long 510 call, long 500 put).
        # Credit = 1.0/share (small, so debit > credit when deep ITM).
        # At spot=600: intrinsic debit = (600-505)-(600-510) = 95-90 = 5 > credit 1
        # unrealized = (1.0 - 5.0) * 1 * 100 = -400, stop_loss = 1 → EXITS.
        state.open_mask[0]     = True
        state.entry_credit[0]  = 1.0      # $1/share credit
        state.entry_qty[0]     = 1
        state.entry_ss_call[0] = 505.0
        state.entry_ss_put[0]  = 505.0
        state.entry_width[0]   = 5.0
        state.entry_dte[0]     = 21.0
        state.entry_bar[0]     = 0

        # Timestamps: daily 86400 s spacing
        ts = torch.arange(100, dtype=torch.float64) * 86400.0

        # Empty chain → debit = NaN → intrinsic fallback used
        ch = self._empty_chain(dev)

        new_state = step_bar_gpu(
            state=state, bar_idx=3, spot=600.0, ts_t=ts,
            chain_right=ch["right"], chain_strike=ch["strike"],
            chain_dte=ch["dte"], chain_bid=ch["bid"], chain_ask=ch["ask"],
            chain_delta=ch["delta"],
            td_t=torch.full((K,), 21.0),
            sd_t=torch.full((K,), 0.45),
            sw_t=torch.full((K,), 5.0),
            sl_t=torch.full((K,), 1.0, dtype=torch.float64),   # $1 stop = any loss exits
            pt_t=torch.full((K,), 50000.0, dtype=torch.float64),
            mde_t=torch.zeros(K, dtype=torch.float64),
            hd_t=torch.full((K,), 99.0, dtype=torch.float64),
            gate_ok=False, cooldown_bars=5, family_code=FAMILY_IRON_BUTTERFLY, K=K,
        )

        # Candidate 0 should have exited (stop-loss hit: unrealized=-400 <= -1)
        assert not new_state.open_mask[0].item(), "Candidate 0 should be closed"
        assert not new_state.open_mask[1].item(), "Candidate 1 was never open"
        assert (new_state.wins[0].item() + new_state.losses[0].item()) == 1, \
            "Candidate 0 should have 1 trade"

    def test_entry_on_gate_ok(self):
        """With gate_ok=True and valid chain, entries are written to state."""
        K = 4
        dev = torch.device("cpu")
        # M=72 → 36 strikes per side, ~3.4pt spacing → tolerance 2pt satisfied for sw=5
        ctx = _build_ctx(dev, T=30, entry_bars=[5], M=72)

        s_off = int(ctx.bar_offsets[5].item())
        e_off = int(ctx.bar_offsets[6].item())

        ts = torch.tensor(ctx.timestamps.cpu().numpy().astype(np.float64),
                          dtype=torch.float64)
        state = init_bar_state(K, dev)
        p = _make_candidates(K)

        new_state = step_bar_gpu(
            state=state, bar_idx=5, spot=500.0, ts_t=ts,
            chain_right  = ctx.option_right[s_off:e_off],
            chain_strike = ctx.option_strike[s_off:e_off],
            chain_dte    = ctx.option_dte[s_off:e_off],
            chain_bid    = ctx.opt_bid[s_off:e_off],
            chain_ask    = ctx.opt_ask[s_off:e_off],
            chain_delta  = ctx.option_delta[s_off:e_off],
            td_t  = torch.from_numpy(p.params["target_dte"].astype(np.float32)),
            sd_t  = torch.from_numpy(p.params["short_delta"].astype(np.float32)),
            sw_t  = torch.from_numpy(p.params["spread_width"].astype(np.float32)),
            sl_t  = torch.tensor(p.params["stop_loss_dollar"],  dtype=torch.float64),
            pt_t  = torch.tensor(p.params["profit_target"],     dtype=torch.float64),
            mde_t = torch.tensor(p.params["max_dte_exit"],      dtype=torch.float64),
            hd_t  = torch.tensor(p.params["hold_days"],         dtype=torch.float64),
            gate_ok=True, cooldown_bars=5, family_code=FAMILY_IRON_BUTTERFLY, K=K,
        )

        # At least some candidates should have opened a position
        n_open = new_state.open_mask.sum().item()
        assert n_open > 0, "Expected ≥1 candidate to enter position"
        # Credit should be positive for open candidates
        open_credits = new_state.entry_credit[new_state.open_mask]
        assert (open_credits > 0).all(), "Open candidates must have positive credit"


# ══════════════════════════════════════════════════════════════════════════════
# TestStepBarGpuVsEngine — full-run parity vs optimizer_engine CPU path
# ══════════════════════════════════════════════════════════════════════════════

_CPU = torch.device("cpu")

class TestStepBarGpuVsEngine:
    """Compare step_bar_gpu loop vs optimizer_engine CPU path (forced by K threshold)."""

    def _compare(self, T: int, K: int, entry_bars: list,
                 stop_loss: float = 600.0, hold_days: float = 7.0,
                 family: str = "iron_butterfly", sidx_filter: int = 8):
        ctx   = _build_ctx(_CPU, T=T, entry_bars=entry_bars)
        cands = _make_candidates(K, stop_loss=stop_loss, hold_days=hold_days)

        # Reference: original engine, CPU path forced
        ref = run_backtest_optimizer_batch(
            ctx, cands,
            base_config={},
            objective_spec=_OBJ_SPEC,
            strategy_idx_filter=sidx_filter,
            strategy_family=family,
            gpu_k_threshold=K + 1,   # force CPU numpy path
        )

        # Candidate: step_bar_gpu loop
        cand = _run_with_step_bar(
            ctx, cands,
            family=family,
            sidx_filter=sidx_filter,
        )

        # Compare
        net_pnl_ref  = ref.net_pnl
        net_pnl_cand = cand["equity"] - STARTING_EQUITY

        np.testing.assert_array_equal(
            ref.wins, cand["wins"].astype(np.int32),
            err_msg="wins mismatch",
        )
        np.testing.assert_array_equal(
            ref.losses, cand["losses"].astype(np.int32),
            err_msg="losses mismatch",
        )
        np.testing.assert_allclose(
            net_pnl_cand, net_pnl_ref,
            atol=5.0, rtol=0,
            err_msg="net_pnl mismatch > 5 USD",
        )
        np.testing.assert_allclose(
            cand["max_dd"], ref.max_dd,
            atol=0.1, rtol=0,
            err_msg="max_dd mismatch > 0.1%",
        )

    def test_no_entries(self):
        """No gate signals → no trades → all-zero metrics on both paths."""
        self._compare(T=30, K=4, entry_bars=[])

    def test_single_entry_hold_days_exit(self):
        """Single entry at bar 5, exits via hold_days=3 (bar 8)."""
        self._compare(T=30, K=4, entry_bars=[5], hold_days=3.0)

    def test_multiple_entries(self):
        """Multiple entry signals, cooldown enforced."""
        self._compare(T=60, K=8, entry_bars=[5, 20, 45])

    def test_iron_condor_family(self):
        """Iron condor family parity."""
        self._compare(T=40, K=4, entry_bars=[5, 25], family="iron_condor",
                      sidx_filter=8)

    def test_short_call_family(self):
        """Short call family parity."""
        self._compare(T=40, K=4, entry_bars=[5, 25], family="short_call",
                      sidx_filter=8)

    def test_force_close_accounting(self):
        """Position open at end triggers force-close on both paths."""
        self._compare(T=10, K=4, entry_bars=[5], hold_days=99.0, stop_loss=600.0)

    def test_medium_fidelity(self):
        """Fidelity=medium truncation: use T=60, step_bar uses medium_end bars."""
        ctx   = _build_ctx(_CPU, T=60, entry_bars=[5, 30])
        cands = _make_candidates(K=4)

        ref = run_backtest_optimizer_batch(
            ctx, cands,
            base_config={},
            objective_spec=_OBJ_SPEC,
            strategy_idx_filter=8,
            strategy_family="iron_butterfly",
            fidelity="medium",
            gpu_k_threshold=5,  # CPU path
        )
        # step_bar_gpu loop for medium_end bars
        K = 4
        dev = _CPU
        fam = FAMILY_IRON_BUTTERFLY
        spot_np    = ctx.spot.cpu().numpy()
        gate_e_np  = ctx.gate_entry.cpu().numpy()
        gate_p_np  = ctx.gate_pop.cpu().numpy()
        sidx_np    = ctx.strategy_idx.cpu().numpy()
        abstain_np = ctx.abstain.cpu().numpy().astype(bool)
        offsets_np = ctx.bar_offsets.cpu().numpy()
        ts_t = torch.tensor(ctx.timestamps.cpu().numpy().astype(np.float64),
                            dtype=torch.float64)
        p  = cands.params
        td_t  = torch.from_numpy(p["target_dte"].astype(np.float32))
        sd_t  = torch.from_numpy(p["short_delta"].astype(np.float32))
        sw_t  = torch.from_numpy(p["spread_width"].astype(np.float32))
        sl_t  = torch.tensor(p["stop_loss_dollar"],  dtype=torch.float64)
        pt_t  = torch.tensor(p["profit_target"],     dtype=torch.float64)
        mde_t = torch.tensor(p["max_dte_exit"],      dtype=torch.float64)
        hd_t  = torch.tensor(p["hold_days"],         dtype=torch.float64)

        state = init_bar_state(K, dev)
        T_end = ctx.medium_end

        for i in range(T_end):
            s_off  = int(offsets_np[i])
            e_off  = int(offsets_np[i + 1])
            gate_ok = (
                float(gate_e_np[i]) > 0.55
                and float(gate_p_np[i]) > 0.50
                and not bool(abstain_np[i])
                and int(sidx_np[i]) == 8
            )
            state = step_bar_gpu(
                state=state, bar_idx=i, spot=float(spot_np[i]), ts_t=ts_t,
                chain_right   = ctx.option_right[s_off:e_off],
                chain_strike  = ctx.option_strike[s_off:e_off],
                chain_dte     = ctx.option_dte[s_off:e_off],
                chain_bid     = ctx.opt_bid[s_off:e_off],
                chain_ask     = ctx.opt_ask[s_off:e_off],
                chain_delta   = ctx.option_delta[s_off:e_off],
                td_t=td_t, sd_t=sd_t, sw_t=sw_t,
                sl_t=sl_t, pt_t=pt_t, mde_t=mde_t, hd_t=hd_t,
                gate_ok=gate_ok, cooldown_bars=5,
                family_code=fam, K=K,
            )

        # Force-close
        if state.open_mask.any():
            fc_pnl   = torch.where(state.open_mask, -sl_t.abs() * 0.5,
                                   torch.zeros_like(state.equity))
            equity_t = torch.where(state.open_mask, state.equity + fc_pnl, state.equity)
            peak_t   = torch.maximum(state.peak, equity_t)
            fc_dd    = torch.where(peak_t > 0,
                                   (peak_t - equity_t) / peak_t * 100.0,
                                   torch.zeros_like(peak_t))
            max_dd_t = torch.maximum(state.max_dd, fc_dd)
            wins_t   = torch.where(state.open_mask & (fc_pnl > 0),  state.wins + 1,   state.wins)
            losses_t = torch.where(state.open_mask & ~(fc_pnl > 0), state.losses + 1, state.losses)
        else:
            equity_t = state.equity
            max_dd_t = state.max_dd
            wins_t   = state.wins
            losses_t = state.losses

        net_pnl_cand = (equity_t - STARTING_EQUITY).cpu().numpy()
        np.testing.assert_allclose(
            net_pnl_cand, ref.net_pnl, atol=5.0, rtol=0,
        )
        np.testing.assert_allclose(
            max_dd_t.cpu().numpy(), ref.max_dd, atol=0.1, rtol=0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TestCompileParity — compiled vs eager parity (CUDA required)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCompileParity:
    """Verify torch.compile'd step_bar_gpu matches eager on CUDA."""

    def test_compiled_matches_eager_short_run(self):
        """T=100 run: compiled step_bar_gpu final metrics match eager."""
        dev   = torch.device("cuda")
        T     = 100
        K     = 32
        ctx   = _build_ctx(dev, T=T, entry_bars=list(range(0, T, 15)))
        cands = _make_candidates(K, stop_loss=600.0, hold_days=5.0)

        # Eager result
        eager_result  = _run_with_step_bar(ctx, cands, family="iron_butterfly")

        # Compiled result
        compiled_step = make_compiled_step()
        family_code   = FAMILY_IRON_BUTTERFLY
        family_str    = CODE_TO_STR[family_code]

        spot_np    = ctx.spot.cpu().numpy()
        gate_e_np  = ctx.gate_entry.cpu().numpy()
        gate_p_np  = ctx.gate_pop.cpu().numpy()
        sidx_np    = ctx.strategy_idx.cpu().numpy()
        abstain_np = ctx.abstain.cpu().numpy().astype(bool)
        offsets_np = ctx.bar_offsets.cpu().numpy()
        ts_t = torch.tensor(ctx.timestamps.cpu().numpy().astype(np.float64),
                            dtype=torch.float64, device=dev)
        p  = cands.params
        td_t  = torch.from_numpy(p["target_dte"].astype(np.float32)).to(dev)
        sd_t  = torch.from_numpy(p["short_delta"].astype(np.float32)).to(dev)
        sw_t  = torch.from_numpy(p["spread_width"].astype(np.float32)).to(dev)
        sl_t  = torch.tensor(p["stop_loss_dollar"],  dtype=torch.float64, device=dev)
        pt_t  = torch.tensor(p["profit_target"],     dtype=torch.float64, device=dev)
        mde_t = torch.tensor(p["max_dte_exit"],      dtype=torch.float64, device=dev)
        hd_t  = torch.tensor(p["hold_days"],         dtype=torch.float64, device=dev)

        state = init_bar_state(K, dev)

        for i in range(T):
            s_off   = int(offsets_np[i])
            e_off   = int(offsets_np[i + 1])
            gate_ok = (
                float(gate_e_np[i]) > 0.55
                and float(gate_p_np[i]) > 0.50
                and not bool(abstain_np[i])
                and int(sidx_np[i]) == 8
            )
            state = compiled_step(
                state=state, bar_idx=i, spot=float(spot_np[i]), ts_t=ts_t,
                chain_right   = ctx.option_right[s_off:e_off],
                chain_strike  = ctx.option_strike[s_off:e_off],
                chain_dte     = ctx.option_dte[s_off:e_off],
                chain_bid     = ctx.opt_bid[s_off:e_off],
                chain_ask     = ctx.opt_ask[s_off:e_off],
                chain_delta   = ctx.option_delta[s_off:e_off],
                td_t=td_t, sd_t=sd_t, sw_t=sw_t,
                sl_t=sl_t, pt_t=pt_t, mde_t=mde_t, hd_t=hd_t,
                gate_ok=gate_ok, cooldown_bars=5,
                family_code=family_code, K=K,
            )

        torch.cuda.synchronize()

        # Force-close
        if state.open_mask.any():
            fc_pnl   = torch.where(state.open_mask, -sl_t.abs() * 0.5,
                                   torch.zeros_like(state.equity))
            equity_t = torch.where(state.open_mask, state.equity + fc_pnl, state.equity)
            peak_t   = torch.maximum(state.peak, equity_t)
            fc_dd    = torch.where(peak_t > 0,
                                   (peak_t - equity_t) / peak_t * 100.0,
                                   torch.zeros_like(peak_t))
            max_dd_t = torch.maximum(state.max_dd, fc_dd)
            wins_t   = torch.where(state.open_mask & (fc_pnl > 0),  state.wins + 1,   state.wins)
            losses_t = torch.where(state.open_mask & ~(fc_pnl > 0), state.losses + 1, state.losses)
        else:
            equity_t = state.equity
            max_dd_t = state.max_dd
            wins_t   = state.wins
            losses_t = state.losses

        net_pnl_c = (equity_t - STARTING_EQUITY).cpu().numpy()

        np.testing.assert_array_equal(
            wins_t.cpu().numpy(),   eager_result["wins"],   err_msg="wins mismatch (compiled vs eager)")
        np.testing.assert_array_equal(
            losses_t.cpu().numpy(), eager_result["losses"], err_msg="losses mismatch (compiled vs eager)")
        np.testing.assert_allclose(
            net_pnl_c, eager_result["equity"] - STARTING_EQUITY,
            atol=5.0, rtol=0, err_msg="net_pnl mismatch (compiled vs eager)",
        )
        np.testing.assert_allclose(
            max_dd_t.cpu().numpy(), eager_result["max_dd"],
            atol=0.1, rtol=0, err_msg="max_dd mismatch (compiled vs eager)",
        )
