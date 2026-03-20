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

Remaining graph break (noted in stage3a_graph_break_audit.md):
  - mark_to_market_gpu:528  bool(open_t.any().item()) — requires .item()
  - This limits fullgraph=True but allows fullgraph=False compilation

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

def step_bar_gpu(
    state:         BarState,
    # ── Bar scalars ──────────────────────────────────────────────────────────
    bar_idx:       int,           # current bar index — compile specialization
    spot:          float,         # current spot price — compile specialization
    ts_t:          torch.Tensor,  # [T] float64 — full timestamp array (passed by ref)
    # ── Chain slice [M] (M=0 when no chain this bar) ─────────────────────────
    chain_right:   torch.Tensor,  # [M] int8
    chain_strike:  torch.Tensor,  # [M] float32
    chain_dte:     torch.Tensor,  # [M] float32
    chain_bid:     torch.Tensor,  # [M] float32
    chain_ask:     torch.Tensor,  # [M] float32
    chain_delta:   torch.Tensor,  # [M] float32 (may contain NaN)
    # ── Candidate params [K] — fixed throughout run ───────────────────────────
    td_t:          torch.Tensor,  # [K] float32  target DTE
    sd_t:          torch.Tensor,  # [K] float32  short delta
    sw_t:          torch.Tensor,  # [K] float32  spread width
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
    Pure tensor one-bar step. Compatible with torch.compile(fullgraph=False).

    Design contract
    ---------------
    - No .any() / .item() calls in this function (graph breaks removed).
    - No verbose / diagnostic output.
    - strategy_family dispatched via static Python int (compile specialization).
    - gate_ok / family_code / cooldown_bars / K are Python scalars → each unique
      combination is compiled as a separate specialization; no graph break.

    Remaining external graph break
    --------------------------------
    mark_to_market_gpu line 528: bool(open_t.any().item())
    This prevents fullgraph=True but allows fullgraph=False fused subgraphs.

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

    # Make chain slices contiguous.
    # Callers pass views with non-zero storage offsets (e.g. option_right[s:e]).
    # Non-contiguous inputs cause misaligned-address errors in Triton's autotuner
    # when compiled by torch.compile.  .contiguous() is a no-op if already aligned.
    chain_right  = chain_right.contiguous()
    chain_strike = chain_strike.contiguous()
    chain_dte    = chain_dte.contiguous()
    chain_bid    = chain_bid.contiguous()
    chain_ask    = chain_ask.contiguous()
    chain_delta  = chain_delta.contiguous()

    family_str = CODE_TO_STR[family_code]   # Python dict lookup — trace-time constant

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
    _cur_dte_t = torch.where(state.open_mask, state.entry_dte - _dh_t, state.entry_dte)

    # Mark-to-market via GPU chain lookup (returns [K] float32, NaN for misses)
    # Note: when M=0 (no chain), mark_to_market_gpu returns full NaN immediately.
    # When M>0, there is ONE graph break inside: bool(open_t.any().item()).
    _raw_debit_t = _gss.mark_to_market_gpu(
        chain_right, chain_strike, chain_dte, chain_bid, chain_ask,
        state.entry_ss_call, state.entry_ss_put, state.entry_width,
        state.open_mask, _cur_dte_t.float(), family_str,
        return_tensors=True,
    )

    # Intrinsic value fallback (static dispatch on family_code — compile-static)
    _spot_t = _raw_debit_t.new_tensor(spot)
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

    _debit_t = torch.where(torch.isfinite(_raw_debit_t), _raw_debit_t, _intr_t)

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

        # Structure selection for all K candidates
        # (When M=0 chain, valid=False for all K → no entries made)
        _g = _gss.select_entry_for_bar(
            chain_right, chain_strike, chain_dte, chain_delta,
            chain_bid, chain_ask, spot,
            td_t, sd_t, sw_t, family_str,
            return_tensors=True,
        )
        _cred_t = _g["credit"]        # [K] float32
        _ssc_t  = _g["ss_call"]
        _ssp_t  = _g["ss_put"]
        _aw_t   = _g["actual_width"]
        _val_t  = _g["valid"]         # [K] bool
        _sdte_t = _g["actual_dte"]

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


# ── torch.compile wrapper ──────────────────────────────────────────────────────

def make_compiled_step(mode: str = "default"):
    """
    Return a torch.compile'd version of step_bar_gpu if PyTorch >= 2.0.

    mode="default" is the correct choice for our use case:
    - Each bar passes a different slice of the options tensor (different memory
      addresses) → "reduce-overhead" / CUDA graph capture is INCOMPATIBLE because
      CUDA graphs require static input addresses between calls.
    - "default" uses inductor kernel fusion without CUDA graph capture — safe for
      variable-address inputs.

    dynamic=True is REQUIRED:
    - `spot` (float) and `bar_idx` (int) change every bar. Without dynamic=True,
      torch.compile creates a new specialised version for each unique value, hitting
      cache_size_limit (default 8) after 8 bars and falling back to eager mode.
      This caused 77-second warm-up costs and 0.21x regression in benchmarks.
    - dynamic=True treats these as symbolic values — a single compiled version
      handles all bar indices and spot prices.

    fullgraph=False allows the one remaining graph break in mark_to_market_gpu.

    Warm-up: first call triggers JIT compilation — expect a few seconds.
    Specialisations are keyed on (gate_ok, family_code, K) — stable per-run.
    """
    try:
        compiled = torch.compile(step_bar_gpu, mode=mode, fullgraph=False, dynamic=True)
        return compiled
    except Exception as exc:
        import warnings
        warnings.warn(
            f"torch.compile not available ({exc}); falling back to eager step_bar_gpu",
            RuntimeWarning,
            stacklevel=2,
        )
        return step_bar_gpu


def make_compiled_step_fullgraph(mode: str = "default"):
    """
    Attempt fullgraph=True compilation for step_bar_gpu.

    This will fail if mark_to_market_gpu still contains the .any().item() call.
    Intended for post-audit use once that guard is removed from mark_to_market_gpu.
    Falls back to fullgraph=False on failure.
    """
    try:
        compiled = torch.compile(step_bar_gpu, mode=mode, fullgraph=True, dynamic=True)
        return compiled
    except Exception as exc:
        import warnings
        warnings.warn(
            f"fullgraph=True failed ({exc}); falling back to fullgraph=False",
            RuntimeWarning,
            stacklevel=2,
        )
        return make_compiled_step(mode=mode)
