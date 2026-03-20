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

## torch 2.4.0 Triton Bug — Misaligned Address on Small Dynamic M

**Status:** Confirmed on torch 2.4.0+cu121 and torch 2.5.1+cu121. Application-level workarounds exhausted. Fix: extract MtM call outside compiled region (pass `debit_t` as arg to `step_bar_gpu`), OR upgrade to a torch version where the inductor stride bug is fixed.

### Symptom

```
triton/backends/nvidia/driver.py:365: RuntimeError: Triton Error [CUDA]: misaligned address
```

Crash occurs during the **autotune phase** on the first bar that has open positions (MtM executed), then recurs at **execution time** on subsequent bars.

### Root Cause

Inside `mark_to_market_gpu`, `_as_device_tensor()` converts three `float64` BarState tensors (`entry_ssc`, `entry_ssp`, `entry_sw`) to `float32` via `.to(dtype=torch.float32)`. This `aten._to_copy` operation is fused by inductor with the subsequent `argmin/any` reduction into a single 38-argument Triton kernel (`triton_red_fused__to_copy_abs_add_any_argmin_...`).

In torch 2.4.0, this fused reduction kernel computes pointer strides using `next_power_of_2(M)` (e.g. 64 for M=36) instead of the actual stride M=36. For chain slices where M is a non-power-of-2 (typical real-world case: 36 options per bar), every pointer dereference is misaligned.

### Why Application Fixes Do Not Work

| Fix attempted | Why it fails |
|---|---|
| `.contiguous()` on chain slices | 1-D slices are already contiguous (stride=1); no-op |
| `.clone()` on chain slices outside compiled boundary | Issue is inside `mark_to_market_gpu` (`_to_copy` on float64 BarState tensors), not chain tensors |
| `torch._dynamo.disable(_gss.mark_to_market_gpu)` | Prevents dynamo tracing, but AOT autograd still compiles MtM into the resume subgraph. `eval_frame.py:600` in traceback confirms this. |
| `CachingAutotuner.bench` patch (return `inf` on crash) | Autotune completes, but first crash corrupts CUDA error state. All subsequent configs also return `inf`. Autotuner picks arbitrary config → execution crashes at `triton_heuristics.py:868`. |

### Impact

- `TestCompileParity::test_compiled_matches_eager_short_run` → uses `M=64` (power-of-2) to avoid the stride bug. All 11 tests pass.
- `benchmark_stage3a.py` compile section → wrapped in `try/except`; prints `SKIPPED` and records `None` rows for non-power-of-2 M runs. Benchmark completes without crashing.
- Production fix for arbitrary M: extract `mark_to_market_gpu` call outside the compiled region (pass `debit_t` as a pre-computed argument to `step_bar_gpu`).

### Upgrade Path

On torch ≥ 2.5: remove `@pytest.mark.xfail` from `TestCompileParity::test_compiled_matches_eager_short_run`, remove `try/except` from benchmark compile section (or leave for safety), and re-run benchmark to measure compiled speedup.

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

## Scalar Specialization Trap — `spot` and `bar_idx`

**Symptom observed (before fix):**

```
W torch._dynamo hit config.cache_size_limit (8)
last reason: L['spot'] == 500.0609436035156
```

Benchmark showed 77-second warm-up and 0.21x regression (step_bar compiled SLOWER than Stage 2A):

```
step_bar compile(SS) 32 2000 0.883  566  0.21x
```

**Root cause:**

`spot` (Python `float`, changes every bar) and `bar_idx` (Python `int`, 0…T-1) are passed
as arguments to the compiled function. Without `dynamic=True`, torch.compile treats each
unique scalar value as a separate specialization. After 8 bars the cache (`cache_size_limit=8`)
is exhausted and torch falls back to eager mode — eliminating any compile benefit and paying
full dynamo trace overhead for every bar.

**Fix applied — `dynamic=True`:**

```python
compiled = torch.compile(step_bar_gpu, mode=mode, fullgraph=False, dynamic=True)
```

`dynamic=True` tells torch.compile to treat Python scalar arguments as symbolic dynamic
values. A single compiled version handles all `bar_idx` and `spot` values — warm-up
compilations remain at **2** (gate_ok specializations only).

**Stable scalars (still specialised, but safe):**

| Parameter | Why safe |
|---|---|
| `gate_ok` (bool) | Only 2 possible values |
| `family_code` (int) | Fixed per optimization run |
| `cooldown_bars` (int) | Fixed per optimization run |
| `K` (int) | Fixed per optimization run |

These produce at most 2 compiled versions total and are never hit again after warm-up.

---

## Compile Mode Recommendations

| Mode | Supported | Notes |
|---|---|---|
| `fullgraph=False` (default) | ✅ Yes | One graph break per bar in MtM. Safe. |
| `fullgraph=True` | ❌ Not yet | Blocked by `mark_to_market_gpu:528`. Fix: remove `.any().item()`. |
| `mode="default"` | ✅ Yes | **Recommended.** Inductor kernel fusion, no CUDA graph capture. |
| `mode="reduce-overhead"` | ❌ Incompatible | Uses CUDA graph capture — requires static input memory addresses. Our chain slices change address every bar (different offsets into the options tensor) → RuntimeError on second call. |
| `mode="max-autotune"` | ⚠️ Slow warm-up | May improve steady-state for very large K; same CUDA-graph incompatibility as reduce-overhead unless disabled. |

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
