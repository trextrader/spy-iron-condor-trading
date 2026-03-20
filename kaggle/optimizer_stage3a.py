"""
optimizer_stage3a.py — Compile-friendly one-bar step function
=============================================================
Stage 3A of the GPU optimizer transition.

Extracts the GPU bar-step logic from optimizer_engine.py into a pure tensor
function designed for torch.compile compatibility.

Key design changes vs original optimizer_engine.py GPU path:
  - All .any()/.item() guards removed → no data-dependent Python branches
  - Verbose / diagnostic branches removed from hot path
  - Strategy family dispatched via static Python int (compile specialization)
  - gate_ok / has_chain passed as Python bools (compile specialization, not
    graph breaks because they are not derived from tensor values)
  - All state carried in BarState dataclass (torch.compile treats it as pytree)

Compile structure:
  - mark_to_market_gpu is computed OUTSIDE the compiled boundary (in the
    eager wrapper step_bar_gpu / _outer), so it never enters torch.compile
    tracing.
  - _step_bar_inner receives pre-computed raw_debit_t and is compiled cleanly.
  - Both make_compiled_step and make_compiled_step_fullgraph compile only
    _step_bar_inner; MtM performance equals Stage 2A.

Usage:
    from optimizer_stage3a import (
        BarState, init_bar_state, step_bar_gpu, make_compiled_step,
        FAMILY_IRON_BUTTERFLY, FAMILY_IRON_CONDOR,
        FAMILY_SHORT_CALL, FAMILY_SHORT_PUT,
        FAMILY_TO_CODE, CODE_TO_STR,
    )
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import gpu_strike_selector as _gss

# ── Strategy family codes (compile-static integer dispatchers) ─────────────────
FAMILY_IRON_BUTTERFLY = 0
FAMILY_IRON_CONDOR    = 1
FAMILY_SHORT_CALL     = 2
FAMILY_SHORT_PUT      = 3

FAMILY_TO_CODE: dict[str, int] = {
    "iron_butterfly": FAMILY_IRON_BUTTERFLY,
    "iron_condor":    FAMILY_IRON_CONDOR,
    "short_call":     FAMILY_SHORT_CALL,
    "short_put":      FAMILY_SHORT_PUT,
}
CODE_TO_STR: dict[int, str] = {v: k for k, v in FAMILY_TO_CODE.items()}

# ── Constants ──────────────────────────────────────────────────────────────────
_STARTING_EQUITY = 100_000.0
_IC_MULTIPLIER   = 100


# ── State container ────────────────────────────────────────────────────────────

@dataclass
class BarState:
    """All K GPU state tensors for one bar step."""
    equity:        torch.Tensor   # [K] float64
    peak:          torch.Tensor   # [K] float64
    max_dd:        torch.Tensor   # [K] float64
    open_mask:     torch.Tensor   # [K] bool
    entry_credit:  torch.Tensor   # [K] float64
    entry_qty:     torch.Tensor   # [K] int32
    entry_ss_call: torch.Tensor   # [K] float64
    entry_ss_put:  torch.Tensor   # [K] float64
    entry_width:   torch.Tensor   # [K] float64
    entry_bar:     torch.Tensor   # [K] int32
    entry_dte:     torch.Tensor   # [K] float64
    last_entry:    torch.Tensor   # [K] int32
    wins:          torch.Tensor   # [K] int32
    losses:        torch.Tensor   # [K] int32
    gross_win:     torch.Tensor   # [K] float64
    gross_loss:    torch.Tensor   # [K] float64


def init_bar_state(K: int, device: torch.device,
                   starting_equity: float = _STARTING_EQUITY) -> BarState:
    """Create a fresh zero-state BarState on device."""
    return BarState(
        equity        = torch.full((K,), starting_equity, dtype=torch.float64, device=device),
        peak          = torch.full((K,), starting_equity, dtype=torch.float64, device=device),
        max_dd        = torch.zeros(K, dtype=torch.float64, device=device),
        open_mask     = torch.zeros(K, dtype=torch.bool,    device=device),
        entry_credit  = torch.zeros(K, dtype=torch.float64, device=device),
        entry_qty     = torch.ones( K, dtype=torch.int32,   device=device),
        entry_ss_call = torch.zeros(K, dtype=torch.float64, device=device),
        entry_ss_put  = torch.zeros(K, dtype=torch.float64, device=device),
        entry_width   = torch.zeros(K, dtype=torch.float64, device=device),
        entry_bar     = torch.full((K,), -999, dtype=torch.int32, device=device),
        entry_dte     = torch.zeros(K, dtype=torch.float64, device=device),
        last_entry    = torch.full((K,), -999, dtype=torch.int32, device=device),
        wins          = torch.zeros(K, dtype=torch.int32,   device=device),
        losses        = torch.zeros(K, dtype=torch.int32,   device=device),
        gross_win     = torch.zeros(K, dtype=torch.float64, device=device),
        gross_loss    = torch.zeros(K, dtype=torch.float64, device=device),
    )


# ── Pure tensor one-bar step ───────────────────────────────────────────────────

def _step_bar_inner(
    state:         BarState,
    # ── Bar scalars ──────────────────────────────────────────────────────────
    bar_idx:       int,           # current bar index — dynamic int (safe with dynamic=True)
    spot_t:        torch.Tensor,  # [] float32  0-d tensor — avoids Python float specialization
    ts_t:          torch.Tensor,  # [T] float64 — full timestamp array (passed by ref)
    # ── Pre-computed MtM result (computed eagerly before entering compiled region)
    raw_debit_t:   torch.Tensor,  # [K] float32  pre-computed MtM debit
    # ── Pre-computed entry selection results (computed eagerly, never inside compiled region)
    entry_cred_t:  torch.Tensor,  # [K] float32  selected credit (zeros when gate_ok=False)
    entry_ssc_t:   torch.Tensor,  # [K] float32  short call strike
    entry_ssp_t:   torch.Tensor,  # [K] float32  short put strike
    entry_aw_t:    torch.Tensor,  # [K] float32  actual spread width
    entry_valid_t: torch.Tensor,  # [K] bool     entry valid flag
    entry_sdte_t:  torch.Tensor,  # [K] float32  actual DTE at entry
    # ── Candidate params [K] — fixed throughout run ───────────────────────────
    sl_t:          torch.Tensor,  # [K] float64  stop-loss dollar
    pt_t:          torch.Tensor,  # [K] float64  profit-target dollar
    mde_t:         torch.Tensor,  # [K] float64  max-DTE-exit
    hd_t:          torch.Tensor,  # [K] float64  hold days
    # ── Static config (Python scalars → compile specializations) ──────────────
    gate_ok:       bool,          # True when entry gate signals pass this bar
    cooldown_bars: int,           # minimum bars between entries per candidate
    family_code:   int,           # FAMILY_* constant — one version per family
    K:             int,           # candidate count — one version per K shape
    IC_MULTIPLIER: int = 100,
) -> BarState:
    """
    Core compileable one-bar step. raw_debit_t is a pre-computed parameter.

    Design contract
    ---------------
    - No .any() / .item() calls in this function (graph breaks removed).
    - No verbose / diagnostic output.
    - strategy_family dispatched via static Python int (compile specialization).
    - gate_ok / family_code / cooldown_bars / K are Python scalars → each unique
      combination is compiled as a separate specialization; no graph break.
    - raw_debit_t ([K] float32) is the MtM debit computed eagerly by the
      calling wrapper (step_bar_gpu or _outer) before entering this function.
      mark_to_market_gpu never enters the compiled region.
    - entry_*_t tensors are the entry selection results computed eagerly by the
      calling wrapper via select_entry_for_bar.  select_entry_for_bar (which
      contains argmin/any reductions) never enters the compiled region.

    Numerical parity
    ----------------
    Results are numerically identical to the original optimizer_engine.py GPU
    path, EXCEPT: this function always runs exits/entries unconditionally
    (original has .any() early-exit guards). When no positions are open and/or
    no candidates are eligible, torch.where ensures no state mutation occurs.
    """
    dev = state.equity.device
    zeros_f64 = torch.zeros(K, dtype=torch.float64, device=dev)
    zeros_f32 = torch.zeros(K, dtype=torch.float32, device=dev)

    # ── EXIT PHASE ─────────────────────────────────────────────────────────────
    # Runs unconditionally (no .any() guard).
    # When open_mask is all False: _dh_t=0, _unreal_t=0, exit masks all False.
    # State is unchanged via torch.where masking.

    # Days held + DTE remaining
    _eb_clamped  = state.entry_bar.clamp(min=0).to(torch.int64)
    _entry_ts_t  = ts_t[_eb_clamped]               # [K] float64  entry timestamps
    _ts_bar      = ts_t[bar_idx]                    # scalar float64
    _dh_t = torch.where(
        state.open_mask,
        (_ts_bar - _entry_ts_t) / 86400.0,
        zeros_f64,
    )
    _dtr_t    = torch.where(state.open_mask, state.entry_dte - _dh_t, zeros_f64)

    # Intrinsic value fallback (static dispatch on family_code — compile-static)
    _spot_t = spot_t
    _ssc_f  = state.entry_ss_call.float()
    _ssp_f  = state.entry_ss_put.float()
    _sw_f   = state.entry_width.float()
    if family_code == FAMILY_SHORT_CALL:
        _intr_t = torch.clamp(_spot_t - _ssc_f, min=0.0)
    elif family_code == FAMILY_SHORT_PUT:
        _intr_t = torch.clamp(_ssp_f - _spot_t, min=0.0)
    else:
        _ic   = torch.clamp(_spot_t - _ssc_f,             min=0.0)
        _ip   = torch.clamp(_ssp_f  - _spot_t,            min=0.0)
        _iwc  = torch.clamp(_spot_t - (_ssc_f + _sw_f),   min=0.0)
        _iwp  = torch.clamp((_ssp_f - _sw_f) - _spot_t,  min=0.0)
        _intr_t = _ic + _ip - _iwc - _iwp

    _debit_t = torch.where(torch.isfinite(raw_debit_t), raw_debit_t, _intr_t)

    # Unrealized P&L [K] float32
    _unreal_t = torch.where(
        state.open_mask,
        (state.entry_credit.float() - _debit_t) * state.entry_qty.float() * IC_MULTIPLIER,
        zeros_f32,
    )

    # Dynamic profit target
    _dpt_t = torch.where(
        pt_t > 0,
        pt_t.float(),
        state.entry_credit.float() * state.entry_qty.float() * IC_MULTIPLIER * 0.50,
    )

    # Drawdown update with mark-to-market equity (pre-exit)
    _eq_mark_t = state.equity + _unreal_t.double()
    peak_t     = torch.maximum(state.peak, _eq_mark_t)
    _dd_now_t  = torch.where(
        peak_t > 0,
        (peak_t - _eq_mark_t) / peak_t * 100.0,
        zeros_f64,
    )
    max_dd_t   = torch.maximum(state.max_dd, _dd_now_t)

    # Exit condition masks (device-side, no .any())
    _sl_hit_t = state.open_mask & (_unreal_t.double() <= -sl_t.abs())
    _pt_hit_t = state.open_mask & (_unreal_t.double() >= _dpt_t.double())
    _dte_ex_t = state.open_mask & (mde_t > 0) & (_dtr_t <= mde_t)
    _hd_ex_t  = state.open_mask & (_dh_t >= hd_t)
    _exp_ex_t = state.open_mask & (_dtr_t <= 0)
    _exit_t   = _sl_hit_t | _pt_hit_t | _dte_ex_t | _hd_ex_t | _exp_ex_t

    # Apply exits — no .any() guard; torch.where masks inactive candidates
    _pnl_t       = torch.where(_exit_t, _unreal_t.double(), zeros_f64)
    equity_t     = torch.where(_exit_t, state.equity + _pnl_t, state.equity)
    peak_t       = torch.maximum(peak_t, equity_t)
    _new_w_t     = _exit_t & (_pnl_t > 0)
    _new_l_t     = _exit_t & ~(_pnl_t > 0)
    wins_t       = torch.where(_new_w_t, state.wins    + 1, state.wins)
    losses_t     = torch.where(_new_l_t, state.losses  + 1, state.losses)
    gross_win_t  = torch.where(_new_w_t, state.gross_win  + _pnl_t,       state.gross_win)
    gross_loss_t = torch.where(_new_l_t, state.gross_loss + _pnl_t.abs(), state.gross_loss)
    open_mask_t  = state.open_mask & ~_exit_t

    # ── ENTRY PHASE ────────────────────────────────────────────────────────────
    # gate_ok is a Python bool → compile produces two specialized versions.
    # When False, entry state fields pass through unchanged — no compute wasted.
    if gate_ok:
        _bar_t = torch.full((K,), bar_idx, dtype=torch.int32, device=dev)

        # Cooldown + eligibility — no .any() guard
        _cool_t = (bar_idx - state.last_entry) >= cooldown_bars
        _elig_t = _cool_t & (~open_mask_t)

        # Entry selection pre-computed by outer wrapper (never inside compiled region)
        _cred_t = entry_cred_t
        _ssc_t  = entry_ssc_t
        _ssp_t  = entry_ssp_t
        _aw_t   = entry_aw_t
        _val_t  = entry_valid_t
        _sdte_t = entry_sdte_t

        _can_t = _elig_t & _val_t

        # Margin — static dispatch on family_code
        if family_code in (FAMILY_SHORT_CALL, FAMILY_SHORT_PUT):
            _marg_t = _cred_t.double() * IC_MULTIPLIER * 2.0
        else:
            _marg_t = torch.clamp(_aw_t.double() - _cred_t.double(), min=0.0) * IC_MULTIPLIER
        _marg_t  = torch.where(_can_t, _marg_t, zeros_f64)
        _capok_t = _can_t & (_marg_t <= equity_t * 0.40)

        # Apply entries — no .any() guard
        entry_credit_t  = torch.where(_capok_t, _cred_t.double(),  state.entry_credit)
        entry_qty_t     = torch.where(_capok_t,
                                      torch.ones(K, dtype=torch.int32, device=dev),
                                      state.entry_qty)
        entry_ss_call_t = torch.where(_capok_t, _ssc_t.double(),   state.entry_ss_call)
        entry_ss_put_t  = torch.where(_capok_t, _ssp_t.double(),   state.entry_ss_put)
        entry_width_t   = torch.where(_capok_t, _aw_t.double(),    state.entry_width)
        entry_bar_t     = torch.where(_capok_t, _bar_t,            state.entry_bar)
        entry_dte_t     = torch.where(_capok_t, _sdte_t.double(),  state.entry_dte)
        open_mask_t     = open_mask_t | _capok_t
        last_entry_t    = torch.where(_capok_t, _bar_t,            state.last_entry)
    else:
        # Pass state through unchanged — no entry compute
        entry_credit_t  = state.entry_credit
        entry_qty_t     = state.entry_qty
        entry_ss_call_t = state.entry_ss_call
        entry_ss_put_t  = state.entry_ss_put
        entry_width_t   = state.entry_width
        entry_bar_t     = state.entry_bar
        entry_dte_t     = state.entry_dte
        last_entry_t    = state.last_entry

    return BarState(
        equity        = equity_t,
        peak          = peak_t,
        max_dd        = max_dd_t,
        open_mask     = open_mask_t,
        entry_credit  = entry_credit_t,
        entry_qty     = entry_qty_t,
        entry_ss_call = entry_ss_call_t,
        entry_ss_put  = entry_ss_put_t,
        entry_width   = entry_width_t,
        entry_bar     = entry_bar_t,
        entry_dte     = entry_dte_t,
        last_entry    = last_entry_t,
        wins          = wins_t,
        losses        = losses_t,
        gross_win     = gross_win_t,
        gross_loss    = gross_loss_t,
    )


def step_bar_gpu(
    state:         BarState,
    *,
    bar_idx:       int,
    spot:          float,
    ts_t:          torch.Tensor,
    chain_right:   torch.Tensor,
    chain_strike:  torch.Tensor,
    chain_dte:     torch.Tensor,
    chain_bid:     torch.Tensor,
    chain_ask:     torch.Tensor,
    chain_delta:   torch.Tensor,
    td_t:          torch.Tensor,
    sd_t:          torch.Tensor,
    sw_t:          torch.Tensor,
    sl_t:          torch.Tensor,
    pt_t:          torch.Tensor,
    mde_t:         torch.Tensor,
    hd_t:          torch.Tensor,
    gate_ok:       bool,
    cooldown_bars: int,
    family_code:   int,
    K:             int,
    IC_MULTIPLIER: int = 100,
) -> BarState:
    """Public eager API. Computes MtM then delegates to _step_bar_inner."""
    dev        = state.equity.device
    zeros_f64  = torch.zeros(K, dtype=torch.float64, device=dev)
    family_str = CODE_TO_STR[family_code]
    spot_t     = torch.tensor(spot, dtype=torch.float32, device=dev)

    _eb_clamped = state.entry_bar.clamp(min=0).to(torch.int64)
    _ts_bar     = ts_t[bar_idx]
    _dh_t       = torch.where(
        state.open_mask,
        (_ts_bar - ts_t[_eb_clamped]) / 86400.0,
        zeros_f64,
    )
    _cur_dte_t = torch.where(state.open_mask, state.entry_dte - _dh_t, state.entry_dte)

    raw_debit_t = _gss.mark_to_market_gpu(
        chain_right, chain_strike, chain_dte, chain_bid, chain_ask,
        state.entry_ss_call, state.entry_ss_put, state.entry_width,
        state.open_mask, _cur_dte_t.float(), family_str,
        return_tensors=True,
    )

    if gate_ok:
        _g = _gss.select_entry_for_bar(
            chain_right, chain_strike, chain_dte, chain_delta,
            chain_bid, chain_ask, spot,
            td_t, sd_t, sw_t, family_str,
            return_tensors=True,
        )
        entry_cred_t  = _g["credit"]
        entry_ssc_t   = _g["ss_call"]
        entry_ssp_t   = _g["ss_put"]
        entry_aw_t    = _g["actual_width"]
        entry_valid_t = _g["valid"]
        entry_sdte_t  = _g["actual_dte"]
    else:
        entry_cred_t  = torch.zeros(K, dtype=torch.float32, device=dev)
        entry_ssc_t   = torch.zeros(K, dtype=torch.float32, device=dev)
        entry_ssp_t   = torch.zeros(K, dtype=torch.float32, device=dev)
        entry_aw_t    = torch.zeros(K, dtype=torch.float32, device=dev)
        entry_valid_t = torch.zeros(K, dtype=torch.bool,    device=dev)
        entry_sdte_t  = torch.zeros(K, dtype=torch.float32, device=dev)

    return _step_bar_inner(
        state,
        raw_debit_t=raw_debit_t,
        entry_cred_t=entry_cred_t, entry_ssc_t=entry_ssc_t, entry_ssp_t=entry_ssp_t,
        entry_aw_t=entry_aw_t, entry_valid_t=entry_valid_t, entry_sdte_t=entry_sdte_t,
        bar_idx=bar_idx, spot_t=spot_t, ts_t=ts_t,
        sl_t=sl_t, pt_t=pt_t, mde_t=mde_t, hd_t=hd_t,
        gate_ok=gate_ok, cooldown_bars=cooldown_bars,
        family_code=family_code, K=K, IC_MULTIPLIER=IC_MULTIPLIER,
    )


# ── torch.compile wrapper ──────────────────────────────────────────────────────

def make_compiled_step(mode: str = "default"):
    """
    Return a callable with the same interface as step_bar_gpu, where
    _step_bar_inner (MtM-free) is compiled via torch.compile.

    mark_to_market_gpu is called eagerly in a wrapper before the compiled
    inner function — it never enters the compiled region.  This eliminates
    the Triton misaligned-address crash caused by inductor fusing the
    float64→float32 _to_copy inside mark_to_market_gpu with the argmin/any
    reduction kernel when dynamic=True is used.
    """
    try:
        _compiled_inner = torch.compile(
            _step_bar_inner, mode=mode, fullgraph=False, dynamic=True
        )
    except Exception as exc:
        import warnings
        warnings.warn(
            f"torch.compile not available ({exc}); falling back to eager step_bar_gpu",
            RuntimeWarning,
            stacklevel=2,
        )
        return step_bar_gpu

    def _outer(
        state, *,
        bar_idx, spot, ts_t,
        chain_right, chain_strike, chain_dte, chain_bid, chain_ask, chain_delta,
        td_t, sd_t, sw_t, sl_t, pt_t, mde_t, hd_t,
        gate_ok, cooldown_bars, family_code, K, IC_MULTIPLIER=100,
    ):
        dev        = state.equity.device
        zeros_f64  = torch.zeros(K, dtype=torch.float64, device=dev)
        family_str = CODE_TO_STR[family_code]
        spot_t     = torch.tensor(spot, dtype=torch.float32, device=dev)

        _eb_clamped = state.entry_bar.clamp(min=0).to(torch.int64)
        _ts_bar     = ts_t[bar_idx]
        _dh_t       = torch.where(
            state.open_mask,
            (_ts_bar - ts_t[_eb_clamped]) / 86400.0,
            zeros_f64,
        )
        _cur_dte_t = torch.where(
            state.open_mask, state.entry_dte - _dh_t, state.entry_dte
        )
        raw_debit_t = _gss.mark_to_market_gpu(
            chain_right, chain_strike, chain_dte, chain_bid, chain_ask,
            state.entry_ss_call, state.entry_ss_put, state.entry_width,
            state.open_mask, _cur_dte_t.float(), family_str,
            return_tensors=True,
        )
        if gate_ok:
            _g = _gss.select_entry_for_bar(
                chain_right, chain_strike, chain_dte, chain_delta,
                chain_bid, chain_ask, spot,
                td_t, sd_t, sw_t, family_str,
                return_tensors=True,
            )
            entry_cred_t  = _g["credit"]
            entry_ssc_t   = _g["ss_call"]
            entry_ssp_t   = _g["ss_put"]
            entry_aw_t    = _g["actual_width"]
            entry_valid_t = _g["valid"]
            entry_sdte_t  = _g["actual_dte"]
        else:
            entry_cred_t  = torch.zeros(K, dtype=torch.float32, device=dev)
            entry_ssc_t   = torch.zeros(K, dtype=torch.float32, device=dev)
            entry_ssp_t   = torch.zeros(K, dtype=torch.float32, device=dev)
            entry_aw_t    = torch.zeros(K, dtype=torch.float32, device=dev)
            entry_valid_t = torch.zeros(K, dtype=torch.bool,    device=dev)
            entry_sdte_t  = torch.zeros(K, dtype=torch.float32, device=dev)
        return _compiled_inner(
            state,
            raw_debit_t=raw_debit_t,
            entry_cred_t=entry_cred_t, entry_ssc_t=entry_ssc_t, entry_ssp_t=entry_ssp_t,
            entry_aw_t=entry_aw_t, entry_valid_t=entry_valid_t, entry_sdte_t=entry_sdte_t,
            bar_idx=bar_idx, spot_t=spot_t, ts_t=ts_t,
            sl_t=sl_t, pt_t=pt_t, mde_t=mde_t, hd_t=hd_t,
            gate_ok=gate_ok, cooldown_bars=cooldown_bars,
            family_code=family_code, K=K, IC_MULTIPLIER=IC_MULTIPLIER,
        )

    return _outer


def make_compiled_step_fullgraph(mode: str = "default"):
    """
    fullgraph=True compilation of _step_bar_inner.

    With mark_to_market_gpu outside the compiled region there are no graph
    breaks remaining in _step_bar_inner.  fullgraph=True should compile
    cleanly.  Falls back to fullgraph=False on error.
    """
    try:
        _compiled_inner = torch.compile(
            _step_bar_inner, mode=mode, fullgraph=True, dynamic=True
        )
    except Exception as exc:
        import warnings
        warnings.warn(
            f"fullgraph=True failed ({exc}); falling back to fullgraph=False",
            RuntimeWarning,
            stacklevel=2,
        )
        return make_compiled_step(mode=mode)

    def _outer(
        state, *,
        bar_idx, spot, ts_t,
        chain_right, chain_strike, chain_dte, chain_bid, chain_ask, chain_delta,
        td_t, sd_t, sw_t, sl_t, pt_t, mde_t, hd_t,
        gate_ok, cooldown_bars, family_code, K, IC_MULTIPLIER=100,
    ):
        dev        = state.equity.device
        zeros_f64  = torch.zeros(K, dtype=torch.float64, device=dev)
        family_str = CODE_TO_STR[family_code]
        spot_t     = torch.tensor(spot, dtype=torch.float32, device=dev)

        _eb_clamped = state.entry_bar.clamp(min=0).to(torch.int64)
        _ts_bar     = ts_t[bar_idx]
        _dh_t       = torch.where(
            state.open_mask,
            (_ts_bar - ts_t[_eb_clamped]) / 86400.0,
            zeros_f64,
        )
        _cur_dte_t = torch.where(
            state.open_mask, state.entry_dte - _dh_t, state.entry_dte
        )
        raw_debit_t = _gss.mark_to_market_gpu(
            chain_right, chain_strike, chain_dte, chain_bid, chain_ask,
            state.entry_ss_call, state.entry_ss_put, state.entry_width,
            state.open_mask, _cur_dte_t.float(), family_str,
            return_tensors=True,
        )
        if gate_ok:
            _g = _gss.select_entry_for_bar(
                chain_right, chain_strike, chain_dte, chain_delta,
                chain_bid, chain_ask, spot,
                td_t, sd_t, sw_t, family_str,
                return_tensors=True,
            )
            entry_cred_t  = _g["credit"]
            entry_ssc_t   = _g["ss_call"]
            entry_ssp_t   = _g["ss_put"]
            entry_aw_t    = _g["actual_width"]
            entry_valid_t = _g["valid"]
            entry_sdte_t  = _g["actual_dte"]
        else:
            entry_cred_t  = torch.zeros(K, dtype=torch.float32, device=dev)
            entry_ssc_t   = torch.zeros(K, dtype=torch.float32, device=dev)
            entry_ssp_t   = torch.zeros(K, dtype=torch.float32, device=dev)
            entry_aw_t    = torch.zeros(K, dtype=torch.float32, device=dev)
            entry_valid_t = torch.zeros(K, dtype=torch.bool,    device=dev)
            entry_sdte_t  = torch.zeros(K, dtype=torch.float32, device=dev)
        return _compiled_inner(
            state,
            raw_debit_t=raw_debit_t,
            entry_cred_t=entry_cred_t, entry_ssc_t=entry_ssc_t, entry_ssp_t=entry_ssp_t,
            entry_aw_t=entry_aw_t, entry_valid_t=entry_valid_t, entry_sdte_t=entry_sdte_t,
            bar_idx=bar_idx, spot_t=spot_t, ts_t=ts_t,
            sl_t=sl_t, pt_t=pt_t, mde_t=mde_t, hd_t=hd_t,
            gate_ok=gate_ok, cooldown_bars=cooldown_bars,
            family_code=family_code, K=K, IC_MULTIPLIER=IC_MULTIPLIER,
        )

    return _outer
