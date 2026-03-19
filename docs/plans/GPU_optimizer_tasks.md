# GPU Transition Plan — Remaining End of Stage 1 → Stage 2 → Stage 3

## Objective

Complete the CondorNet optimizer transition from a hybrid CPU/GPU backtester into a fully tensorized GPU simulation engine.

The implementation roadmap is divided into:

- Stage 1 remainder — finish and harden GPU strike/MtM infrastructure
- Stage 2 — tensorize the position state machine
- Stage 3 — reduce or eliminate Python bar-loop launch overhead via compilation / kernel fusion

Status legend:

- `✅` completed
- `🟡` started / in progress
- `[ ]` not started

---

## Status Update (2026-03-19)

- ✅ Stage 1 selector audit and interface cleanup landed.
- 🟡 GPU hot-path bounce reduction is underway: GPU MtM now supports `return_tensors=True`, unconditional CPU chain mirrors were reduced, and open-position copy-back is narrower.
- 🟡 Stage 2A foundations are partially in place: the engine still uses host state and a Python bar loop, but the optimizer now has MtM drawdown correction, `np_dd` logging, guarded ranking/audit output, and checkpointed BO progress.
- ✅ Operational support landed outside the original stage checklist: `--checkpoint` JSON resume, per-strategy BO phase/round persistence, `selection_forensics.py`, `scripts/repo_sync_audit.py`, and a `W/L` leaderboard column.
- [ ] Stage 1 parity harness and benchmark work remain open.
- [ ] Stage 2 tensorized state work has not started in earnest yet.

---

# Stage 1 — Remaining Work
## Goal
Finish and harden the current GPU-vectorized strike-selection layer so it is stable, parity-tested, and ready to serve as the substrate for Stage 2.

---

## 1.1 Audit current GPU selector path
- ✅ Identify every call site using the GPU strike selector
- ✅ Identify every call site using GPU mark-to-market logic
- ✅ Confirm which paths still force `.cpu().numpy()` conversion in the hot loop
- ✅ Confirm fallback paths for CPU mode still behave correctly
- ✅ Confirm behavior when `K < gpu_k_threshold`
- ✅ Confirm behavior when CUDA is unavailable

### Deliverable
- ✅ Write `reports/gpu_transition/stage1_path_audit.md`

---

## 1.2 Standardize selector interfaces
- ✅ Review `select_entry_for_bar(...)`
- ✅ Review `mark_to_market_gpu(...)`
- ✅ Add a consistent argument contract for device, dtype, and return mode
- ✅ Add optional `return_tensors: bool = False`
- ✅ Preserve current CPU/NumPy behavior as fallback compatibility mode
- 🟡 Add explicit shape assertions for all selector inputs and outputs

### Deliverable
- ✅ Selector functions support both:
  - ✅ legacy NumPy return mode
  - ✅ direct torch tensor return mode

---

## 1.3 Remove unnecessary host/device bouncing
- ✅ Trace all GPU outputs that are immediately converted back to CPU
- 🟡 Eliminate conversions inside the hot path where possible
- 🟡 Keep final metrics extraction on CPU only at end-of-run
- [ ] Keep CPU fallback available only when explicitly required

### Success criteria
- [ ] No unnecessary `.cpu().numpy()` inside per-bar GPU path
- [ ] No unnecessary `torch.from_numpy(...)` round-trips during bar processing

---

## 1.4 Harden shape + dtype guarantees
- [ ] Assert all option tensors use expected dtype
- [ ] Assert candidate parameter tensors use expected dtype
- [ ] Assert index tensors use integer dtype
- [ ] Assert boolean masks are `torch.bool`
- [ ] Standardize float precision policy:
  - [ ] `float32` default for throughput
  - [ ] optional debug mode for stricter checking
- 🟡 Add device consistency assertions

### Deliverable
- [ ] `utils/tensor_contracts.py` or equivalent helper assertions

---

## 1.5 Stage 1 parity tests
- [ ] Run CPU selector vs brute-force GPU selector parity tests
- [ ] Run sorted-search selector vs brute-force selector parity tests
- [ ] Test edge cases:
  - [ ] no valid options on bar
  - [ ] one valid option only
  - [ ] exact delta match
  - [ ] target delta outside available delta range
  - [ ] duplicate deltas
  - [ ] NaN or masked entries
- [ ] Confirm same selected strikes and credits under parity mode

### Deliverable
- [ ] `tests/test_gpu_strike_selector_parity.py`

---

## 1.6 Stage 1 performance benchmark harness
- [ ] Benchmark CPU strike selection
- [ ] Benchmark GPU brute-force selector
- [ ] Benchmark GPU sorted selector
- [ ] Benchmark GPU mark-to-market path
- [ ] Collect timings across:
  - [ ] small K
  - [ ] medium K
  - [ ] large K
  - [ ] representative S sizes
- [ ] Record GPU model, CUDA version, torch version

### Deliverable
- [ ] `reports/gpu_transition/stage1_benchmarks.json`
- [ ] `reports/gpu_transition/stage1_benchmarks.md`

---

## 1.7 Stage 1 completion gate
- [ ] Selector parity passes
- [ ] MtM parity passes
- [ ] GPU mode stable across test datasets
- [ ] No critical host/device round-trips remain in strike/MtM path
- [ ] Benchmark report written

### Milestone
- [ ] **Stage 1 officially complete**

---

# Stage 2 — Tensorized Position State Machine
## Goal
Replace remaining host-side NumPy/Python position-state mutation with tensor-native state updates, first for the current one-position-per-candidate design (`M=1`), then optionally generalize to multi-slot state (`M>1`).

---

# Stage 2A — Tensorize current `M=1` state machine
## Goal
Keep current behavior exactly the same while moving state arrays from CPU/NumPy to GPU/torch tensors.

---

## 2A.1 Inventory current state variables
- [ ] Enumerate all optimizer state arrays currently mutated in-loop
- [ ] Confirm shapes, dtypes, semantics, and initialization values

### Expected state inventory
- [ ] `equity`
- [ ] `peak`
- [ ] `max_dd`
- [ ] `open_mask`
- [ ] `entry_credit`
- [ ] `entry_qty`
- [ ] `entry_ss_call`
- [ ] `entry_ss_put`
- [ ] `entry_width`
- [ ] `entry_bar`
- [ ] `entry_dte_at_entry`
- [ ] `last_entry`
- [ ] `wins`
- [ ] `losses`
- [ ] `gross_win`
- [ ] `gross_loss`

### Deliverable
- [ ] `reports/gpu_transition/stage2a_state_inventory.md`

---

## 2A.2 Convert state initialization to torch tensors
- [ ] Replace NumPy initialization with torch initialization on target device
- [ ] Standardize state dtype policy
- [ ] Keep initialization deterministic
- [ ] Verify zero-state matches legacy engine

### Deliverable
- [ ] State tensors created directly on device

---

## 2A.3 Tensorize open-position mark-to-market flow
- 🟡 Feed open-position state directly into GPU MtM path
- 🟡 Return device tensors instead of CPU arrays when Stage 2A enabled
- [ ] Compute unrealized PnL as torch tensors
- [ ] Keep all arithmetic device-side

### Success criteria
- [ ] No NumPy MtM calculations inside hot path
- [ ] No CPU extraction until final reporting

---

## 2A.4 Tensorize exit-condition computation
- [ ] Convert exit-condition logic to torch boolean masks
- [ ] Compute days-held on device
- [ ] Compute DTE remaining on device
- [ ] Compute stop/target/expiration masks on device
- [ ] Merge masks into a single final exit mask
- [ ] Preserve current exit precedence rules exactly

### Deliverable
- [ ] `tests/test_stage2a_exit_mask_parity.py`

---

## 2A.5 Tensorize realized PnL updates
- [ ] Use exit mask to compute realized PnL device-side
- [ ] Update `equity`, `wins`, `losses`, `gross_win`, `gross_loss` using tensor ops
- [ ] Update drawdown state device-side
- [ ] Preserve current accounting definitions exactly

### Success criteria
- [ ] Per-candidate realized PnL matches legacy engine
- [ ] Max drawdown metrics match within tolerance

---

## 2A.6 Tensorize entry eligibility logic
- [ ] Compute entry eligibility masks on device
- [ ] Compute cooldown / `last_entry` constraints on device
- [ ] Compute capital / quantity constraints on device
- [ ] Apply gating masks using tensor logic only
- [ ] Write entry state directly into state tensors

### Deliverable
- [ ] `tests/test_stage2a_entry_mask_parity.py`

---

## 2A.7 Keep Python bar loop, remove host-state mutation
- ✅ Preserve outer `for t in T` loop temporarily
- [ ] Ensure all per-bar updates are tensor-native
- [ ] Prohibit new NumPy state mutation inside loop
- ✅ Prohibit Python dict state in hot path

### Milestone
- [ ] Hybrid loop remains, but state machine is tensorized

---

## 2A.8 Stage 2A parity harness
- [ ] Compare legacy engine vs tensorized `M=1` engine
- [ ] Verify parity for:
  - [ ] entries
  - [ ] exits
  - [ ] realized PnL
  - [ ] equity curves
  - [ ] drawdown
  - [ ] win/loss counts
- [ ] Test across representative strategies and years
- [ ] Define acceptable tolerance thresholds

### Deliverables
- [ ] `tests/test_stage2a_full_engine_parity.py`
- [ ] `reports/gpu_transition/stage2a_parity_summary.json`

---

## 2A.9 Stage 2A benchmark
- [ ] Benchmark legacy hybrid engine
- [ ] Benchmark Stage 2A tensorized state engine
- [ ] Compare:
  - [ ] bars/sec
  - [ ] candidates/sec
  - [ ] end-to-end wall time
  - [ ] GPU utilization
  - [ ] CPU utilization
- [ ] Run on T4 if available
- [ ] Prepare same harness for future H100/A100 comparison

### Milestone
- [ ] **Stage 2A complete**

---

# Stage 2B — Generalize to multi-slot tensor state (`M > 1`)
## Goal
Extend the state machine from one concurrent position per candidate to a fixed-slot tensor position book.

---

## 2B.1 Define tensor book schema
- [ ] Define `M` = max concurrent positions per candidate
- [ ] Create tensor schema:
  - [ ] `pos_active[K, M]`
  - [ ] `pos_credit[K, M]`
  - [ ] `pos_qty[K, M]`
  - [ ] `pos_ss_call[K, M]`
  - [ ] `pos_ss_put[K, M]`
  - [ ] `pos_width[K, M]`
  - [ ] `pos_entry_bar[K, M]`
  - [ ] `pos_entry_dte[K, M]`
- [ ] Define candidate-level aggregates separately from slot-level state

### Deliverable
- [ ] `reports/gpu_transition/stage2b_state_schema.md`

---

## 2B.2 Implement slot-wise exit masks
- [ ] Compute MtM for all active slots
- [ ] Compute exit masks over `[K, M]`
- [ ] Apply expiration / stop / target / manual exit rules slot-wise
- [ ] Reduce realized PnL from slot level to candidate level

### Deliverable
- [ ] `tests/test_stage2b_slot_exit_logic.py`

---

## 2B.3 Implement free-slot discovery
- [ ] Compute free-slot mask `~pos_active`
- [ ] Choose insertion slot per candidate
- [ ] Handle no-free-slot case explicitly
- [ ] Preserve deterministic tie behavior

### Deliverable
- [ ] `tests/test_stage2b_free_slot_selection.py`

---

## 2B.4 Implement slot scatter updates
- [ ] Write new entries into slot tensors via `scatter_` or equivalent indexed writes
- [ ] Update `pos_active`
- [ ] Write entry metadata
- [ ] Preserve candidate-level entry accounting

### Success criteria
- [ ] New slot write path is deterministic
- [ ] Candidate metrics remain correct when multiple slots are active

---

## 2B.5 Implement slot reductions
- [ ] Aggregate unrealized PnL across slots
- [ ] Aggregate realized PnL across slots
- [ ] Aggregate risk exposure across slots if needed
- [ ] Update candidate equity and performance statistics correctly

### Deliverable
- [ ] `tests/test_stage2b_slot_reductions.py`

---

## 2B.6 Preserve `M=1` compatibility
- [ ] Verify `M=1` multi-slot engine reproduces Stage 2A results
- [ ] Ensure multi-slot design does not break one-slot behavior
- [ ] Keep migration path reversible during validation

### Milestone
- [ ] **Stage 2B complete**

---

# Stage 3 — Reduce Python Bar-Loop Overhead
## Goal
Move from tensorized per-bar execution to compiled or fused execution that minimizes Python launch overhead and improves end-to-end GPU throughput.

---

# Stage 3A — Compile-friendly refactor
## Goal
Restructure the Stage 2 engine so the hot path is compatible with `torch.compile` and similar graph capture approaches.

---

## 3A.1 Isolate pure tensor step function
- [ ] Extract one-bar transition into a pure tensor function:
  - [ ] inputs: current state, bar tensors, candidate params
  - [ ] outputs: next state, per-bar metrics
- [ ] Remove side effects from hot step where possible
- [ ] Separate logging/debugging from compute path

### Deliverable
- [ ] `step_bar_tensorized(...)` or equivalent

---

## 3A.2 Remove graph-breaking constructs
- [ ] Eliminate hidden Python branching in tensor path
- [ ] Eliminate data-dependent Python object mutation
- [ ] Eliminate NumPy calls in compiled region
- [ ] Reduce shape polymorphism where practical
- [ ] Standardize tensor layouts for compile stability

### Deliverable
- [ ] `reports/gpu_transition/stage3a_graph_break_audit.md`

---

## 3A.3 Test `torch.compile`
- [ ] Compile selector path
- [ ] Compile MtM path
- [ ] Compile full per-bar step function
- [ ] Benchmark compile warm-up vs steady-state runtime
- [ ] Confirm numerical parity with eager mode

### Deliverable
- [ ] `tests/test_stage3a_compile_parity.py`
- [ ] `reports/gpu_transition/stage3a_compile_benchmark.md`

### Milestone
- [ ] **Stage 3A complete**

---

# Stage 3B — Scan / chunked execution over bars
## Goal
Reduce Python overhead further by processing bars in chunks or using a scan-like compiled structure.

---

## 3B.1 Design chunked execution model
- [ ] Define chunk size `B_chunk`
- [ ] Process `T` bars as chunks instead of one-bar Python launches
- [ ] Preserve exact state carry-forward across chunk boundaries
- [ ] Ensure deterministic chunked results

### Deliverable
- [ ] `reports/gpu_transition/stage3b_chunk_design.md`

---

## 3B.2 Implement chunk executor
- [ ] Build chunk runner around compiled step
- [ ] Maintain state persistence across chunks
- [ ] Benchmark chunk sizes for optimal throughput
- [ ] Compare eager per-bar vs compiled chunked execution

### Deliverable
- [ ] `tests/test_stage3b_chunked_parity.py`

---

## 3B.3 Profile GPU occupancy
- [ ] Measure kernel launch count
- [ ] Measure GPU utilization
- [ ] Measure memory throughput
- [ ] Identify remaining bottlenecks:
  - [ ] selector
  - [ ] MtM lookup
  - [ ] state update
  - [ ] reductions
- [ ] Record T4 profile baseline for future A100/H100 scaling

### Deliverable
- [ ] `reports/gpu_transition/stage3b_profile_summary.md`

### Milestone
- [ ] **Stage 3B complete**

---

# Stage 3C — Triton / custom kernel exploration
## Goal
Evaluate whether a custom fused kernel materially improves performance beyond compiled torch execution.

---

## 3C.1 Triton feasibility study
- [ ] Identify exact hot kernels still dominating time
- [ ] Confirm problem sizes justify Triton/custom-kernel investment
- [ ] Estimate complexity vs expected gain
- [ ] Decide whether to target:
  - [ ] strike selection only
  - [ ] MtM only
  - [ ] full state update
  - [ ] fused bar-step kernel

### Deliverable
- [ ] `reports/gpu_transition/stage3c_triton_feasibility.md`

---

## 3C.2 Prototype custom kernel
- [ ] Implement minimal prototype for highest-value hot operation
- [ ] Validate against torch eager / compiled reference
- [ ] Benchmark on representative datasets
- [ ] Confirm maintenance burden is acceptable

### Deliverable
- [ ] `tests/test_stage3c_kernel_parity.py`
- [ ] `reports/gpu_transition/stage3c_kernel_benchmark.md`

---

## 3C.3 Go / no-go decision
- [ ] Compare Triton/custom kernel vs compiled torch
- [ ] Approve only if performance gain is material and stable
- [ ] Otherwise retain compiled torch as preferred production path

### Milestone
- [ ] **Stage 3 complete**

---

# Cross-Cutting Tasks
## Determinism
- [ ] Fix seeds for all benchmark/parity runs
- [ ] Record torch/CUDA versions
- [ ] Record GPU hardware info
- [ ] Ensure consistent dtype policy across all stages

## Logging / Instrumentation
- [ ] Add timing blocks for selector, MtM, state update, reductions
- [ ] Add optional debug assertions mode
- [ ] Add optional device-memory telemetry snapshots

## Documentation
- [ ] Update architecture report after Stage 1 completion
- [ ] Update architecture report after Stage 2A completion
- [ ] Update architecture report after Stage 2B completion
- [ ] Update architecture report after Stage 3 completion

## Regression Protection
- [ ] Add CI-friendly reduced-size GPU/CPU parity tests
- [ ] Add smoke-test dataset for fast validation
- [ ] Add benchmark harness with fixed scenarios

---

# Final Acceptance Gates

## Gate A — Stage 1 complete
- [ ] GPU strike selection parity verified
- [ ] GPU MtM parity verified
- [ ] Interface standardized
- [ ] Benchmark report written

## Gate B — Stage 2A complete
- [ ] Current `M=1` state machine tensorized
- [ ] No host-state mutation in hot path
- [ ] End-to-end parity with legacy engine established
- [ ] Benchmark improvement documented

## Gate C — Stage 2B complete
- [ ] Multi-slot `[K, M]` position book implemented
- [ ] `M=1` compatibility preserved
- [ ] Slot-wise entry/exit/reduction logic verified

## Gate D — Stage 3 complete
- [ ] Compiled or fused execution path benchmarked
- [ ] Python overhead materially reduced
- [ ] Best production path selected
- [ ] Final performance report written

---

# Recommended Execution Order

1. [ ] Finish Stage 1 interface cleanup and parity harness
2. [ ] Implement Stage 2A tensorized `M=1` state
3. [ ] Lock Stage 2A parity
4. [ ] Benchmark Stage 2A
5. [ ] Decide whether multi-slot `M>1` is immediately required
6. [ ] If yes, implement Stage 2B
7. [ ] Refactor for compile-friendly Stage 3A
8. [ ] Benchmark chunked Stage 3B
9. [ ] Only then evaluate Triton/custom kernel Stage 3C

---

# Recommendation

For the current codebase, the highest-value next step is:

- [ ] **Complete Stage 1 cleanup**
- [ ] **Implement Stage 2A before Stage 2B**
- [ ] **Do not jump to custom kernels until Stage 2A parity is locked**

That sequence minimizes risk and gives the fastest real-world acceleration.
________________________________________
Recommended immediate next section to start on
I will with these first 10 checkboxes, in order:
1.	✅ Stage 1 path audit
2.	✅ Standardize selector interfaces
3.	✅ Add return_tensors mode
4.	🟡 Remove hot-path .cpu().numpy() conversions
5.	Build Stage 1 parity tests
6.	Benchmark current Stage 1
7.	Inventory current state arrays
8.	Convert state initialization to torch tensors
9.	Tensorize exit masks
10.	Tensorize entry logic

