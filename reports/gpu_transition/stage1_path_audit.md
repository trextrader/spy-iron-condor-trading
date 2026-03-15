# Stage 1 Path Audit

## Scope

Audit target:

- `kaggle/gpu_strike_selector.py`
- `kaggle/optimizer_engine.py`
- `kaggle/optimizer_prep.py`

This report documents the current GPU selector and GPU MtM integration points,
the remaining host/device bounce locations, and the behavior of the CPU
fallback path.

## Call Sites

GPU selector call site:

- `kaggle/optimizer_engine.py`
  - `run_backtest_optimizer_batch(...)`
  - inside the entry block, GPU path calls `_gss.select_entry_for_bar(...)`
  - current engine behavior still consumes NumPy outputs

GPU MtM call site:

- `kaggle/optimizer_engine.py`
  - `run_backtest_optimizer_batch(...)`
  - inside the open-position exit block, GPU path calls `_gss.mark_to_market_gpu(...)`
  - current engine behavior still consumes NumPy outputs

No other live call sites for these GPU APIs were found in the optimizer path.

## Current GPU/CPU Boundary

Device-resident context:

- `optimizer_prep.py` constructs the immutable optimizer context directly on the
  target device:
  - `timestamps`
  - `spot`
  - `gate_entry`
  - `gate_pop`
  - `strategy_idx`
  - `abstain`
  - `bar_offsets`
  - flattened option arrays (`option_right`, `option_strike`, `option_dte`,
    `option_delta`, `opt_bid`, `opt_ask`, `opt_mid`)

Current host mirrors:

- `optimizer_engine.py` always materializes CPU copies of the bar-major arrays
  (`spot`, `gate_entry`, `gate_pop`, `strategy_idx`, `abstain`, `bar_offsets`,
  `timestamps`) because the outer bar loop is still Python/NumPy.
- When `_use_gpu` is true, the engine still also materializes full CPU copies of
  all option arrays for intrinsic fallback and legacy CPU logic:
  - `right_np`
  - `strike_np`
  - `dte_np`
  - `delta_np`
  - `bid_np`
  - `ask_np`

This is the main remaining Stage 1 memory/bounce issue.

## `.cpu().numpy()` in the Hot Path

Inside `gpu_strike_selector.py`:

- `select_entry_for_bar(...)`
  - previously always returned `.cpu().numpy()`
  - now supports `return_tensors: bool = False`
  - default remains NumPy for compatibility
- `mark_to_market_gpu(...)`
  - previously always converted entry-state NumPy arrays to tensors and then
    returned `.cpu().numpy()`
  - now supports tensor inputs plus `return_tensors: bool = False`
  - default remains NumPy for compatibility

Inside `optimizer_engine.py`:

- GPU entry path still calls `select_entry_for_bar(...)` in legacy NumPy mode
- GPU MtM path still calls `mark_to_market_gpu(...)` in legacy NumPy mode
- full option arrays are still mirrored to CPU in `_use_gpu` mode

So the interface is now ready for Stage 1 cleanup, but the engine is not yet
consuming tensor outputs.

## Fallback Behavior

CPU fallback path is active when any of the following is true:

- `ctx.device.type != "cuda"`
- `K < gpu_k_threshold`

Fallback behavior:

- Entry selection uses the CPU `_find_best_structure_*` helpers
- MtM uses CPU `_mark_to_market(...)` or `_mark_to_market_single_leg(...)`
- intrinsic fallback arithmetic remains CPU/NumPy

This fallback path is still intact after the interface update because both new
GPU API flags default to legacy NumPy behavior.

## `gpu_k_threshold` Behavior

Current gate in `optimizer_engine.py`:

- `_use_gpu = (ctx.device.type == "cuda" and K >= gpu_k_threshold)`

Observed implications:

- below threshold: full CPU path
- at or above threshold: GPU selector and GPU MtM path are enabled
- threshold tuning remains hardware-dependent and is currently supplied via
  `--gputype`

## CUDA Unavailable Behavior

When CUDA is unavailable:

- `optimizer_prep.py` builds the context on CPU
- `_use_gpu` evaluates false
- the optimizer remains functional on the CPU path

No code changes were needed here.

## Stage 1 Findings

What is complete after this audit:

- GPU selector entry point identified
- GPU MtM entry point identified
- selector/MtM APIs now support either NumPy or direct tensor return
- basic shape/device assertions added to the selector layer

What still remains before Stage 1 can be considered complete:

- switch `optimizer_engine.py` GPU path to `return_tensors=True`
- stop converting open-position state into NumPy before GPU MtM
- stop unconditionally mirroring the full option arrays to CPU in `_use_gpu`
- add parity tests around selector and MtM output modes
- add benchmark harness and record results

## Recommended Next Patch

Smallest high-value next change:

1. update `optimizer_engine.py` GPU entry path to consume tensor outputs from
   `select_entry_for_bar(...)`
2. update `optimizer_engine.py` GPU MtM path to pass tensor entry-state and
   consume tensor debit outputs
3. keep final metric extraction on CPU only at the end of the run
