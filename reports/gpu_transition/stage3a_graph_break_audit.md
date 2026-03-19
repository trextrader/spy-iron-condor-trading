# Stage 3A Graph-Break Audit

**Date:** 2026-03-18
**Scope:** optimizer_engine.py GPU hot path → optimizer_stage3a.py step_bar_gpu

---

## Summary

Stage 3A extracted the GPU bar-step logic into `optimizer_stage3a.py::step_bar_gpu`.
All data-dependent Python branches on tensor values (`.any()`, `.item()`) have been
removed from the new function. One graph break remains inside `mark_to_market_gpu`.

`torch.compile(fullgraph=False)` is safe and recommended.
`torch.compile(fullgraph=True)` requires one additional fix (see below).

---

## Graph Breaks Removed

The following `.any()` / `.item()` guards were present in the original
`optimizer_engine.py` GPU path and have been **eliminated** in `step_bar_gpu`:

| Location (original) | Guard | Impact |
|---|---|---|
| `optimizer_engine.py:864` | `if open_mask_t.any():` | graph break: exits skipped when no positions |
| `optimizer_engine.py:940` | `if _exit_t.any():` | graph break: exit updates skipped when no exits |
| `optimizer_engine.py:1174` | `if not _elig_t.any(): continue` | graph break: entry skipped when no eligible |
| `optimizer_engine.py:1191` | `if not _can_t.any(): continue` | graph break: entry skipped when no valid struct |
| `optimizer_engine.py:1203` | `if _capok_t.any():` | graph break: entry writes skipped when no capital |

**Replaced by:** unconditional tensor ops with `torch.where` masking. Numerical
output is identical — `torch.where(mask=False, ...)` returns the unchanged branch.

---

## Remaining Graph Break

| File | Line | Code | Impact |
|---|---|---|---|
| `gpu_strike_selector.py` | 528 | `bool(open_t.any().item())` | One graph break per bar when M>0 |

**Effect:** With `fullgraph=False`, torch.compile splits each bar into two subgraphs
at this point. The subgraph before the break (MtM setup) and after (intrinsic fallback
onward) are compiled separately. This still provides kernel fusion within each subgraph.

**Why it exists:** Fast-path skip when no positions open — avoids matrix ops over K×M.

**Fix path for fullgraph=True:**
Remove the `.any().item()` guard from `mark_to_market_gpu`. Replace with unconditional
execution masked by `open_mask`, mirroring the approach taken in `step_bar_gpu`. This
is a targeted one-line removal. The function already handles all-False `open_mask`
correctly via `torch.where` on `debit_t`.

---

## Compile Specializations

These Python-scalar conditions are **not graph breaks** — torch.compile generates a
separate compiled version per unique value:

| Parameter | Type | Unique values per run | Notes |
|---|---|---|---|
| `gate_ok` | `bool` | 2 (`True` / `False`) | 2 specializations, cached after warm-up |
| `family_code` | `int` | 1 (one family per run) | 1 specialization |
| `cooldown_bars` | `int` | 1 | 1 specialization |
| `K` | `int` | 1 | 1 specialization |
| `strategy_family` (inside selector/MtM) | `str` | 1 | Specialization within sub-call |

Total warm-up compilations: **2** (gate_ok=True first encounter + gate_ok=False first encounter).
Chain M may vary per bar → additional specializations on M-shape if M changes. In production
data, M is typically fixed per session.

---

## Compile Mode Recommendations

| Mode | Supported | Notes |
|---|---|---|
| `fullgraph=False` (default) | ✅ Yes | One graph break per bar in MtM. Safe. |
| `fullgraph=True` | ❌ Not yet | Blocked by `mark_to_market_gpu:528`. Fix: remove `.any().item()`. |
| `mode="reduce-overhead"` | ✅ Yes | Best for repeated same-shape calls (production use). |
| `mode="default"` | ✅ Yes | Better first-call latency, lower steady-state than reduce-overhead. |
| `mode="max-autotune"` | ⚠️ Slow warm-up | May improve steady-state for very large K. |

---

## Numerical Parity

`step_bar_gpu` is **numerically identical** to the original GPU path under identical inputs:

- Unconditional exits: same as guarded when mask is False (torch.where identity)
- Unconditional entries: same as guarded when _capok_t all-False
- Intrinsic fallback: same formula, same floating-point ops

Test coverage: `tests/test_stage3a_compile_parity.py`
- `TestStepBarGpuVsEngine` — 7 full-run parity tests vs original engine (CPU path forced)
- `TestCompileParity` — compiled vs eager on CUDA (tolerances: wins/losses exact, net_pnl ±5 USD, max_dd ±0.1%)

---

## Next Step: fullgraph=True Path

To enable fullgraph=True and eliminate the last graph break:

1. In `gpu_strike_selector.py::mark_to_market_gpu` line 528, replace:
   ```python
   if M == 0 or not bool(open_t.any().item()):
       return debit_out if return_tensors else debit_out.cpu().numpy()
   ```
   with:
   ```python
   if M == 0:
       return debit_out if return_tensors else debit_out.cpu().numpy()
   ```
   (Remove the `.any().item()` guard; the existing `torch.where(open_t & all_ok, ...)` already
   handles all-False open_mask by returning NaN for all K.)

2. Update parity tests to verify MtM still returns NaN for all-False open_mask.

3. Re-run benchmark with `fullgraph=True` to measure additional speedup from full kernel fusion.
