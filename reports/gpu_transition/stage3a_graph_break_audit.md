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

## Triton Misaligned Address Bug — Resolution

**Status: RESOLVED** (2026-03-20). Root cause identified and fixed in `optimizer_stage3a.py`. No torch upgrade required.

### Symptom

```
triton/backends/nvidia/driver.py:365: RuntimeError: Triton Error [CUDA]: misaligned address
```

Crash during **autotune** on first bar with open positions, then at **execution time** on subsequent bars. Confirmed on torch 2.4.0+cu121 and torch 2.5.1+cu121.

### Root Cause

Inductor fuses any `float64→float32 _to_copy` operation with any `argmin/any` reduction in the same compiled graph into a single Triton kernel. That fused kernel computes pointer strides incorrectly for dynamic M, causing misaligned reads.

Two functions contained `float64→float32` casts + reductions:
1. `mark_to_market_gpu` — `_to_copy` on BarState tensors + `argmin` for strike lookup
2. `select_entry_for_bar` — `_to_copy` on delta/price tensors + `argmin/any` for best strike

Both were inside the compiled region. Either one was sufficient to trigger the crash.

### Failed Fixes (Documented for Reference)

| Fix attempted | Why it fails |
|---|---|
| `.contiguous()` on chain slices | Already contiguous; no-op |
| `.clone()` outside compiled boundary | Issue is inside the functions, not chain slices |
| `torch._dynamo.disable(mark_to_market_gpu)` | Prevents dynamo tracing but AOT autograd still compiles MtM into resume subgraph |
| `CachingAutotuner.bench` patch | First crash corrupts CUDA state; subsequent configs all `inf` → arbitrary config → crash at execution |
| Extract MtM only | Reduced crash frequency; revealed same bug in `select_entry_for_bar` |

### Definitive Fix

Move **both** `mark_to_market_gpu` and `select_entry_for_bar` outside the compiled region into the eager `_outer` wrapper. `_step_bar_inner` (the compiled function) then contains **zero reductions** — only pointwise, element-wise, and gather ops. Inductor has nothing to fuse.

```python
# In _outer (uncompiled):
raw_debit_t = _gss.mark_to_market_gpu(..., return_tensors=True)     # eager
entry_*     = _gss.select_entry_for_bar(..., return_tensors=True)   # eager

# _step_bar_inner (compiled) receives pre-computed tensors — no reductions inside
return _compiled_inner(state, raw_debit_t=raw_debit_t,
                       entry_cred_t=..., entry_ssc_t=..., ...)
```

### Result

- `TestCompileParity::test_compiled_matches_eager_short_run` → **PASSED** (no xfail, any M value)
- 11/11 tests pass on Tesla T4, torch 2.5.1+cu121
- `benchmark_stage3a.py` compile section runs without try/except

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

**Symptom (initial):**
```
W torch._dynamo hit config.cache_size_limit (8)
last reason: L['spot'] == 500.0609436035156
```
After fixing `spot`, the next specialization offender appeared:
```
W torch._dynamo hit config.cache_size_limit (8)
last reason: L['bar_idx'] == 0
```

Both caused cache exhaustion after 8 unique values → dynamo fallback to eager for all remaining bars.

**Root cause:**

`dynamic=True` in `torch.compile` treats tensor **shape** integers as dynamic but does NOT prevent specialization on Python scalar **function arguments** (float or int). `spot` changes every bar (T unique floats) and `bar_idx` increments every bar (T unique ints) — both saturate the 8-entry compile cache within the first 8 bars.

**Fix applied — 0-d tensors:**

Convert both scalars to 0-d tensors in all callers **before** passing to `_step_bar_inner`:

```python
# In step_bar_gpu / _outer (before calling _compiled_inner):
spot_t    = torch.tensor(spot,    dtype=torch.float32, device=dev)
bar_idx_t = torch.tensor(bar_idx, dtype=torch.int64,   device=dev)
```

Tensor arguments are never captured as Python value guards. A single compiled version handles all bars — warm-up compilations remain at **2** (gate_ok=True + gate_ok=False).

**Stable scalars (still specialised, but safe):**

| Parameter | Why safe |
|---|---|
| `gate_ok` (bool) | Only 2 possible values — both compiled at warm-up |
| `family_code` (int) | Fixed per optimization run |
| `cooldown_bars` (int) | Fixed per optimization run |
| `K` (int) | Fixed per optimization run |

**Result:** No scalar specialization warnings. Compile warm-up ~6s (K=32). Steady-state compile matches Stage 2A throughput.

---

## Compile Mode Recommendations

| Mode | Supported | Notes |
|---|---|---|
| `fullgraph=False` (default) | ✅ Yes | **Recommended.** Zero graph breaks in `_step_bar_inner`. |
| `fullgraph=True` | ✅ Yes | `_step_bar_inner` contains no graph breaks. `make_compiled_step_fullgraph()` available. |
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

## Final Benchmark Results (Tesla T4, torch 2.5.1+cu121, 2026-03-20)

```
Mode                       K      T   wall_s     bars/s   vs Stage2A   trades
Stage2A GPU eager         32    500    0.180       2783   (baseline)        0
step_bar eager            32    500    0.922        542        0.19x        0
step_bar compile(SS)      32    500    0.542        922        0.33x        0

Stage2A GPU eager        128    500    0.181       2769   (baseline)        0
step_bar compile(SS)     128    500    0.541        924        0.33x        0

Stage2A GPU eager         32   2000    3.870        517   (baseline)       32
step_bar eager            32   2000    5.345        374        0.72x       32
step_bar compile(SS)      32   2000    3.818        524        1.01x       32

Stage2A GPU eager        128   2000    3.926        509   (baseline)      128
step_bar compile(SS)     128   2000    3.812        525        1.03x      128
```

**Key observations:**
- T=2000 (production-length runs): compiled path matches Stage 2A throughput (1.01–1.03x)
- T=500 (short runs): Python per-bar overhead from eager MtM+entry calls dominates; 0.33x vs Stage2A
- Warm-up ~6s for first K value; subsequent K reuses cached compilation (~0.6s)
- No scalar specialization warnings; no Triton crashes; no xfail

## Stage 3A — Complete

All deliverables committed. Next stages: 3B (chunked execution) or 3C (Triton feasibility).
