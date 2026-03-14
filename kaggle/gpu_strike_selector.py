"""
gpu_strike_selector.py
======================
Phase 1 GPU vectorization: replace per-candidate Python delta search with
batched CUDA tensor ops.

The hot path that was:
    for k in range(K):
        argmin(|delta[chain_slice] - target_delta[k]|)

becomes:
    diff = |abs_delta[M].unsqueeze(0) - target_delta[K].unsqueeze(1)|   # [K, M]
    idx  = diff.argmin(dim=1)                                            # [K]

Position tracking (state machine) stays as CPU numpy — unchanged.

Public API
----------
    nearest_delta_indices_bruteforce(chain_deltas, target_deltas, valid_mask)
    nearest_delta_indices_sorted(sorted_deltas, target_deltas)
    select_entry_for_bar(chain_right, chain_strike, chain_dte, chain_delta,
                         chain_bid, chain_ask, spot,
                         target_dte, short_delta, spread_width, family)
    validate_against_cpu(gpu_result, cpu_credit, cpu_ss_call, cpu_ss_put,
                         cpu_aw, cpu_valid, cpu_dte, atol)
    bench_gpu(fn, *args, warmup, iters)

All input tensors must already be on the same CUDA device.
All outputs are CPU numpy — [K] results are small, transfer is negligible.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch


# ── Core delta-search primitives ──────────────────────────────────────────────

def nearest_delta_indices_bruteforce(
    chain_deltas:  torch.Tensor,                   # [S] on device
    target_deltas: torch.Tensor,                   # [K] on device
    valid_mask:    Optional[torch.Tensor] = None,  # [S] or [K, S] bool
) -> torch.Tensor:
    """
    Return the index into chain_deltas nearest to each target_delta.

    Parameters
    ----------
    chain_deltas  : [S]    abs-normalised deltas already on device
    target_deltas : [K]    candidate targets, same device
    valid_mask    : [S] or [K, S]  False entries are treated as inf distance
                    Pass [K, S] when candidates target different expiry windows.

    Returns
    -------
    idx : [K] int64 — index into S axis.
    Note: if all entries in a row are inf (no valid option) argmin still returns
    an index (the first element). Callers MUST check feasibility separately via
    valid_mask.any().
    """
    S = chain_deltas.shape[0]
    K = target_deltas.shape[0]

    abs_d = torch.abs(chain_deltas)                                     # [S]
    diff  = torch.abs(abs_d.unsqueeze(0) - target_deltas.unsqueeze(1)) # [K, S]

    if valid_mask is not None:
        if valid_mask.dim() == 1:
            vm = valid_mask.unsqueeze(0).expand(K, S)
        else:
            vm = valid_mask                                             # [K, S]
        diff = torch.where(vm, diff, diff.new_full((K, S), float("inf")))

    return diff.argmin(dim=1)                                          # [K]


def nearest_delta_indices_sorted(
    sorted_deltas: torch.Tensor,   # [S] ascending, on device
    target_deltas: torch.Tensor,   # [K] on device
) -> torch.Tensor:
    """
    Binary-search nearest-delta lookup.  ~O(log S) per candidate vs O(S).

    Requires sorted_deltas to be monotonically increasing along S.
    Use only when the strike axis is presorted — otherwise use bruteforce.

    Returns
    -------
    idx : [K] int64
    """
    n   = sorted_deltas.numel()
    pos = torch.searchsorted(sorted_deltas.contiguous(),
                             target_deltas.contiguous())               # [K]
    left  = (pos - 1).clamp(0, n - 1)
    right = pos.clamp(0, n - 1)

    lv = sorted_deltas[left]
    rv = sorted_deltas[right]
    use_right = torch.abs(rv - target_deltas) < torch.abs(lv - target_deltas)
    return torch.where(use_right, right, left)                         # [K]


# ── Expiry helpers ─────────────────────────────────────────────────────────────

def _select_expiry(
    chain_dte:  torch.Tensor,   # [M]
    target_dte: torch.Tensor,   # [K]
    dte_window: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each candidate k, find the expiry in chain_dte closest to target_dte[k].

    Returns
    -------
    best_dte : [K]    — actual selected DTE from chain
    exp_mask : [K, M] — True where option m is on candidate k's best expiry
    """
    diff     = torch.abs(chain_dte.unsqueeze(0) - target_dte.unsqueeze(1))  # [K, M]
    best_idx = diff.argmin(dim=1)                                            # [K]
    best_dte = chain_dte[best_idx]                                           # [K]
    exp_mask = torch.abs(chain_dte.unsqueeze(0) - best_dte.unsqueeze(1)) < dte_window
    return best_dte, exp_mask                                                # [K], [K, M]


def _wing_lookup(
    chain_strike:   torch.Tensor,    # [M]
    chain_mid:      torch.Tensor,    # [M]
    leg_mask:       torch.Tensor,    # [K, M] valid options for this leg
    target_strike:  torch.Tensor,    # [K]
    spread_width:   torch.Tensor,    # [K]
    wing_tol_mult:  float = 0.40,
    abs_tol:        float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the wing leg nearest to target_strike[k] for each candidate.

    Returns (idx[K], mid[K], ok[K])  — ok = distance within tolerance.
    """
    K, M = leg_mask.shape
    dist = torch.where(
        leg_mask,
        torch.abs(chain_strike.unsqueeze(0) - target_strike.unsqueeze(1)),
        chain_strike.new_full((K, M), float("inf")),
    )
    idx      = dist.argmin(dim=1)                                          # [K]
    best_d   = dist.min(dim=1).values                                      # [K]
    tol      = torch.maximum(chain_strike.new_tensor(abs_tol),
                             spread_width * wing_tol_mult)
    ok       = best_d <= tol                                               # [K]
    mid      = chain_mid[idx]                                              # [K]
    return idx, mid, ok


# ── Main entry-point ───────────────────────────────────────────────────────────

def select_entry_for_bar(
    chain_right:  torch.Tensor,   # [M] int8  0=call 1=put, on device
    chain_strike: torch.Tensor,   # [M] float32
    chain_dte:    torch.Tensor,   # [M] float32
    chain_delta:  torch.Tensor,   # [M] float32 (may have NaN/inf)
    chain_bid:    torch.Tensor,   # [M] float32
    chain_ask:    torch.Tensor,   # [M] float32
    spot:         float,
    target_dte:   torch.Tensor,   # [K] float32 — on device
    short_delta:  torch.Tensor,   # [K] float32
    spread_width: torch.Tensor,   # [K] float32 (0 for single-leg)
    strategy_family: str,         # "iron_butterfly"|"iron_condor"|"short_call"|"short_put"
) -> Dict[str, np.ndarray]:
    """
    GPU-vectorized entry structure selection for K candidates at one bar.

    Replaces the four _find_best_structure_* functions in optimizer_engine.py.
    Returns a dict of [K] CPU numpy arrays with the same semantics as the CPU
    functions, so the state machine in run_backtest_optimizer_batch needs no
    changes.

    Keys
    ----
    credit        float32   net credit per share (NaN = invalid)
    ss_call       float32   short call strike
    ss_put        float32   short put strike (= ss_call for iron_butterfly)
    actual_width  float32   realised wing width in the chain
    valid         bool
    actual_dte    float32   DTE of the selected expiry
    """
    M      = chain_right.shape[0]
    K      = target_dte.shape[0]
    device = chain_right.device
    nan_K  = chain_right.new_full((K,), float("nan"), dtype=torch.float32)
    zero_K = chain_right.new_zeros(K, dtype=torch.float32)
    false_K = torch.zeros(K, dtype=torch.bool, device=device)

    _empty = {
        "credit": nan_K.cpu().numpy(), "ss_call": zero_K.cpu().numpy(),
        "ss_put": zero_K.cpu().numpy(), "actual_width": zero_K.cpu().numpy(),
        "valid": false_K.cpu().numpy(), "actual_dte": nan_K.cpu().numpy(),
    }

    if M == 0:
        return _empty

    mid_chain = (chain_bid + chain_ask) * 0.5          # [M]
    abs_delta = torch.abs(chain_delta)                  # [M] normalised
    delta_ok  = torch.isfinite(abs_delta)               # [M]

    # ── Expiry selection (shared for call/put at same expiry) ──────────────
    best_dte, exp_mask = _select_expiry(chain_dte, target_dte)  # [K], [K, M]

    call_right_m = (chain_right == 0)                           # [M]
    put_right_m  = (chain_right == 1)                           # [M]

    # Per-candidate call/put masks within selected expiry
    cm = call_right_m.unsqueeze(0) & exp_mask           # [K, M]
    pm = put_right_m.unsqueeze(0)  & exp_mask           # [K, M]

    # ── Short-leg selection ────────────────────────────────────────────────

    def _short_delta_search(
        side_mask: torch.Tensor,    # [K, M] valid options for this leg
        atm_spot_mult: float,       # 1.015 for calls, 0.985 for puts
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find short strike for K candidates.
        Returns (strike_idx[K], stk[K], mid[K]).
        """
        valid_d = side_mask & delta_ok.unsqueeze(0)         # [K, M] has finite delta

        # Primary: nearest abs-delta to target
        delta_idx = nearest_delta_indices_bruteforce(
            abs_delta, short_delta, valid_d
        )                                                    # [K]

        # Fallback: nearest strike to ATM proxy (when no valid delta in row)
        has_delta = valid_d.any(dim=1)                       # [K]
        atm_target = chain_right.new_tensor(spot * atm_spot_mult, dtype=torch.float32)
        atm_dist   = torch.where(
            side_mask,
            torch.abs(chain_strike.unsqueeze(0) - atm_target),
            chain_right.new_full((K, M), float("inf"), dtype=torch.float32),
        )
        atm_idx    = atm_dist.argmin(dim=1)                  # [K]

        idx = torch.where(has_delta, delta_idx, atm_idx)    # [K]
        stk = chain_strike[idx]                              # [K]
        mid = mid_chain[idx]                                 # [K]
        return idx, stk, mid

    # ── Strategy dispatch ──────────────────────────────────────────────────

    if strategy_family == "iron_butterfly":
        # ── 1. Short strike (ATM, same for call and put) ──────────────────
        has_calls = cm.any(dim=1)    # [K]
        has_puts  = pm.any(dim=1)    # [K]
        can_enter = has_calls & has_puts

        _, sc_stk, sc_mid = _short_delta_search(cm, 1.0)   # ATM, not OTM

        # ── 2. Short put at same strike ────────────────────────────────────
        _, sp_mid, sp_ok = _wing_lookup(
            chain_strike, mid_chain, pm, sc_stk, spread_width,
            wing_tol_mult=0.25, abs_tol=2.0,
        )

        # ── 3. Long call at sc_stk + sw ────────────────────────────────────
        lc_target = sc_stk + spread_width
        lc_idx, lc_mid, lc_ok = _wing_lookup(
            chain_strike, mid_chain, cm, lc_target, spread_width,
        )
        actual_lc_stk = chain_strike[lc_idx]

        # ── 4. Long put at sc_stk − sw ─────────────────────────────────────
        lp_target = sc_stk - spread_width
        lp_idx, lp_mid, lp_ok = _wing_lookup(
            chain_strike, mid_chain, pm, lp_target, spread_width,
        )
        actual_lp_stk = chain_strike[lp_idx]

        credit = (sc_mid + sp_mid) - (lc_mid + lp_mid)
        actual_width = torch.clamp(
            (actual_lc_stk - sc_stk + sc_stk - actual_lp_stk) / 2.0, min=0.0
        )
        min_credit = torch.maximum(
            chain_right.new_tensor(0.10, dtype=torch.float32),
            spread_width * 0.05,
        )
        valid = can_enter & sp_ok & lc_ok & lp_ok & (credit >= min_credit)

        return {
            "credit":       torch.where(valid, credit,       nan_K ).cpu().numpy(),
            "ss_call":      torch.where(valid, sc_stk,       zero_K).cpu().numpy(),
            "ss_put":       torch.where(valid, sc_stk,       zero_K).cpu().numpy(),
            "actual_width": torch.where(valid, actual_width, zero_K).cpu().numpy(),
            "valid":        valid.cpu().numpy(),
            "actual_dte":   torch.where(valid, best_dte,     nan_K ).cpu().numpy(),
        }

    elif strategy_family == "iron_condor":
        # ── Short call: OTM call at sd above spot ─────────────────────────
        has_calls = cm.any(dim=1)
        has_puts  = pm.any(dim=1)
        _, sc_stk, sc_mid = _short_delta_search(cm, 1.015)

        # ── Short put: OTM put at sd below spot ───────────────────────────
        _, sp_stk, sp_mid = _short_delta_search(pm, 0.985)

        ic_ok = sc_stk > sp_stk                             # short call above short put

        # ── Wings ─────────────────────────────────────────────────────────
        lc_idx, lc_mid, lc_ok = _wing_lookup(
            chain_strike, mid_chain, cm, sc_stk + spread_width, spread_width,
        )
        actual_lc_stk = chain_strike[lc_idx]

        lp_idx, lp_mid, lp_ok = _wing_lookup(
            chain_strike, mid_chain, pm, sp_stk - spread_width, spread_width,
        )
        actual_lp_stk = chain_strike[lp_idx]

        credit = (sc_mid + sp_mid) - (lc_mid + lp_mid)
        actual_width = torch.clamp(
            (actual_lc_stk - sc_stk + sp_stk - actual_lp_stk) / 2.0, min=0.0
        )
        min_credit = torch.maximum(
            chain_right.new_tensor(0.10, dtype=torch.float32),
            spread_width * 0.05,
        )
        valid = has_calls & has_puts & ic_ok & lc_ok & lp_ok & (credit >= min_credit)

        return {
            "credit":       torch.where(valid, credit,       nan_K ).cpu().numpy(),
            "ss_call":      torch.where(valid, sc_stk,       zero_K).cpu().numpy(),
            "ss_put":       torch.where(valid, sp_stk,       zero_K).cpu().numpy(),
            "actual_width": torch.where(valid, actual_width, zero_K).cpu().numpy(),
            "valid":        valid.cpu().numpy(),
            "actual_dte":   torch.where(valid, best_dte,     nan_K ).cpu().numpy(),
        }

    elif strategy_family == "short_call":
        has_calls = cm.any(dim=1)
        _, sc_stk, sc_mid = _short_delta_search(cm, 1.015)
        valid = has_calls & (sc_mid >= chain_right.new_tensor(0.05, dtype=torch.float32))

        return {
            "credit":       torch.where(valid, sc_mid,   nan_K ).cpu().numpy(),
            "ss_call":      torch.where(valid, sc_stk,   zero_K).cpu().numpy(),
            "ss_put":       torch.where(valid, sc_stk,   zero_K).cpu().numpy(),
            "actual_width": zero_K.cpu().numpy(),
            "valid":        valid.cpu().numpy(),
            "actual_dte":   torch.where(valid, best_dte, nan_K ).cpu().numpy(),
        }

    elif strategy_family == "short_put":
        has_puts = pm.any(dim=1)
        _, sp_stk, sp_mid = _short_delta_search(pm, 0.985)
        valid = has_puts & (sp_mid >= chain_right.new_tensor(0.05, dtype=torch.float32))

        return {
            "credit":       torch.where(valid, sp_mid,   nan_K ).cpu().numpy(),
            "ss_call":      torch.where(valid, sp_stk,   zero_K).cpu().numpy(),
            "ss_put":       torch.where(valid, sp_stk,   zero_K).cpu().numpy(),
            "actual_width": zero_K.cpu().numpy(),
            "valid":        valid.cpu().numpy(),
            "actual_dte":   torch.where(valid, best_dte, nan_K ).cpu().numpy(),
        }

    else:
        # Unknown family — fall back silently (iron_butterfly treatment)
        return select_entry_for_bar(
            chain_right, chain_strike, chain_dte, chain_delta,
            chain_bid, chain_ask, spot,
            target_dte, short_delta, spread_width,
            "iron_butterfly",
        )


# ── Mark-to-market (GPU) ───────────────────────────────────────────────────────

def mark_to_market_gpu(
    chain_right:  torch.Tensor,   # [M] on device
    chain_strike: torch.Tensor,   # [M]
    chain_dte:    torch.Tensor,   # [M]
    chain_bid:    torch.Tensor,   # [M]
    chain_ask:    torch.Tensor,   # [M]
    entry_ssc:    np.ndarray,     # [K] CPU — short call strike
    entry_ssp:    np.ndarray,     # [K] CPU — short put strike
    entry_sw:     np.ndarray,     # [K] CPU — spread width
    open_mask:    np.ndarray,     # [K] CPU bool
    entry_dte:    np.ndarray,     # [K] CPU — DTE at entry
    strategy_family: str,
) -> np.ndarray:
    """
    GPU-vectorized mark-to-market.  Returns debit_per_share[K] as CPU float32.
    NaN means chain lookup failed — caller should use intrinsic fallback.
    """
    M      = chain_right.shape[0]
    K      = len(entry_ssc)
    device = chain_right.device
    debit_out = np.full(K, np.nan, np.float32)

    if M == 0 or not open_mask.any():
        return debit_out

    mid   = (chain_bid + chain_ask) * 0.5
    open_t = torch.from_numpy(open_mask).to(device)

    # Candidate entry state → device (K small values, negligible transfer)
    ssc_t = torch.from_numpy(entry_ssc.astype(np.float32)).to(device)
    ssp_t = torch.from_numpy(entry_ssp.astype(np.float32)).to(device)
    sw_t  = torch.from_numpy(entry_sw.astype(np.float32)).to(device)
    dte_e = torch.from_numpy(entry_dte.astype(np.float32)).to(device)

    _, exp_mask = _select_expiry(chain_dte, dte_e)          # [K, M]
    cm = (chain_right == 0).unsqueeze(0) & exp_mask         # [K, M]
    pm = (chain_right == 1).unsqueeze(0) & exp_mask         # [K, M]

    def _find_mid(mask, target):
        dist = torch.where(
            mask,
            torch.abs(chain_strike.unsqueeze(0) - target.unsqueeze(1)),
            chain_strike.new_full((K, M), float("inf")),
        )
        idx   = dist.argmin(dim=1)
        ok    = dist.min(dim=1).values < 5.0
        return mid[idx], ok

    if strategy_family in ("short_call",):
        leg_m = cm
        sc_mid_v, sc_ok = _find_mid(leg_m, ssc_t)
        debit_t = torch.where(open_t & sc_ok, sc_mid_v,
                              torch.full((K,), float("nan"), device=chain_right.device))

    elif strategy_family in ("short_put",):
        leg_m = pm
        sp_mid_v, sp_ok = _find_mid(leg_m, ssp_t)
        debit_t = torch.where(open_t & sp_ok, sp_mid_v,
                              torch.full((K,), float("nan"), device=chain_right.device))

    else:   # iron_butterfly, iron_condor, generic 4-leg
        sc_mid_v, sc_ok = _find_mid(cm, ssc_t)
        sp_mid_v, sp_ok = _find_mid(pm, ssp_t)
        lc_mid_v, lc_ok = _find_mid(cm, ssc_t + sw_t)
        lp_mid_v, lp_ok = _find_mid(pm, ssp_t - sw_t)
        all_ok  = sc_ok & sp_ok & lc_ok & lp_ok
        debit_v = (sc_mid_v + sp_mid_v) - (lc_mid_v + lp_mid_v)
        debit_t = torch.where(open_t & all_ok, debit_v,
                              torch.full((K,), float("nan"), device=chain_right.device))

    return debit_t.float().cpu().numpy()


# ── Validation helper ─────────────────────────────────────────────────────────

def validate_against_cpu(
    gpu: Dict[str, np.ndarray],
    cpu_credit:  np.ndarray,
    cpu_ss_call: np.ndarray,
    cpu_ss_put:  np.ndarray,
    cpu_aw:      np.ndarray,
    cpu_valid:   np.ndarray,
    cpu_dte:     np.ndarray,
    atol: float = 1e-3,
) -> bool:
    """
    Compare GPU and CPU results for a single bar.
    Logs mismatches with full context.  Returns True if all close.
    """
    passed = True
    for k in range(len(cpu_valid)):
        if cpu_valid[k] != gpu["valid"][k]:
            print(f"  [validate k={k}] VALID MISMATCH: cpu={cpu_valid[k]} gpu={gpu['valid'][k]}")
            passed = False
            continue
        if not cpu_valid[k]:
            continue
        for name, cpu_val, gpu_val in [
            ("credit",  cpu_credit[k],  gpu["credit"][k]),
            ("ss_call", cpu_ss_call[k], gpu["ss_call"][k]),
            ("ss_put",  cpu_ss_put[k],  gpu["ss_put"][k]),
            ("aw",      cpu_aw[k],      gpu["actual_width"][k]),
            ("dte",     cpu_dte[k],     gpu["actual_dte"][k]),
        ]:
            if abs(float(cpu_val) - float(gpu_val)) > atol:
                print(f"  [validate k={k}] {name}: cpu={cpu_val:.4f}  gpu={gpu_val:.4f}  "
                      f"diff={abs(float(cpu_val)-float(gpu_val)):.6f}")
                passed = False
    return passed


# ── Benchmark helper ──────────────────────────────────────────────────────────

def bench_gpu(fn, *args, warmup: int = 10, iters: int = 100) -> float:
    """
    Returns mean wall time per call (seconds) with CUDA synchronization.
    Use CUDA sync so timings reflect real GPU completion, not just kernel launch.
    """
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ── Optional torch.compile wrapper ───────────────────────────────────────────
# Wrap the hot path AFTER confirming correctness.
# PyTorch 2.x can fuse kernels and reduce Python overhead on tensor-pure code.

def compile_selector():
    """
    Return a compiled version of select_entry_for_bar if torch.compile is
    available (PyTorch >= 2.0).  Falls back to the eager version silently.
    """
    try:
        return torch.compile(select_entry_for_bar)
    except Exception:
        return select_entry_for_bar
