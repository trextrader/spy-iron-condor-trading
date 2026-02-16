# CondorNet v4.3 — Master Implementation Plan

> [!IMPORTANT]
> **Status**: REVIEW VERSION — no code changes until approved.
> Changes already applied (from earlier session) are marked ✅.
> All remaining items are pending approval.

---

## Architectural Ground Truth

Before planning observability, here is the **actual subsystem mapping** based on
[condor_brain_net_v42.py](file:///C:/SPYOptionTrader_test/intelligence/condor_brain_net_v42.py):

| Subsystem | Role | Key Classes | Where in Forward |
|-----------|------|-------------|-----------------|
| **TFT Control Encoder** | Long-horizon feature attention, variable selection, control embedding production | `TFTControlEncoder` (Part 6) | `self.tft(x)` → produces `u` (batch, seq, d_control) |
| **Neural CDE + ETD-1** | Continuous state evolution, market physics, memory retention | `BlockMatrixA`, `BlockMatrixB`, `CDEResponseG`, `FullForcingD`, `etd1_kernel`, `condornet_master_step` (Parts 2–3) | Time loop: `x_k = F·x_{k-1} + φ₁·B·u + G·dX + D` |
| **Predicate Logic Engine** | Inequality gates, combinatorics, set/superset routing, regime dynamics | `CanonicalPredicateGates`, `PredicateSignature`, `PredicateSet`, `SuperSet`, `RelationalLogicLayer`, `RegimeCombinatoricsDynamics` (Parts 4–5) | Per-timestep: `p_k → z_pred → r_k update → superset gating` |
| **Pivot Encoder** | Structural market turn detection, geometric embedding | `PivotEncoder`, `PivotHead` (Part 7) | Pre-loop: `x_piv_enc = PivotEncoder(x_piv)` → concat to features |
| **FusionGate** | Multi-expert weighting (currently used for superset combination) | `FusionGate` (Part 7) | Post-loop: `super_gate → z_gated → output_head` |

> [!NOTE]
> CondorNet is **not** three literally separate networks. It is a **unified CDE** where: TFT produces the control signal `u_k`, the ETD-1 kernel governs state dynamics via `A_θ`, and CDE response `G_θ` injects market data. The "three-brain" decomposition is functional, not architectural — we observe which *functional subsystem* drives each decision, how they modularize, fuse, segregate tasks, and what drives those decisions.

---

## A0 — Training Loop Hook Order Specification

Every training batch executes in this exact order:

```
1. forward         → model(x_b, return_diagnostics=True)
2. loss            → criterion(outputs, targets)
3. backward        → loss.backward()
4. optimizer step  → optimizer.step()
5. deep_observe    → capture diagnostics, write JSON
6. checkpoint      → atomic_save (if triggered)
7. telemetry emit  → stream to GUI/dashboard
```

All hooks must respect this ordering. No diagnostic capture may occur before backward (gradients not yet available). No checkpoint may occur before deep_observe (to ensure the report covers the batch that triggered the save).

---

## TRACK A — INFRASTRUCTURE & TRAINING PIPELINE

### A1 — Fix Model Saving Behavior

| ID | Deliverable | Status |
|----|-------------|--------|
| A1.1 | Default save dir → `models/ckpts/` | ✅ Done |
| A1.2 | `--checkpoint-dir` CLI flag | ✅ Done |
| A1.3 | Atomic save: write to `.tmp`, validate, rename | ✅ Done |
| A1.4 | Save on every convergence snapshot | Pending |
| A1.5 | Rotate last N checkpoints (`--keep-ckpts N`, default 3) | ✅ Done |

### A2 — GPU Utilization Optimization

| ID | Deliverable | Status |
|----|-------------|--------|
| A2.1 | CLI flags: `--num-workers`, `--prefetch-factor`, `--persistent-workers` | ✅ Done |
| A2.2 | DataLoader: workers=4, pin_memory=True, persistent=True, prefetch=2 | ✅ Done (hardcoded) |
| A2.3 | Duplicate DataLoader block removed | ✅ Done |
| A2.4 | GPU memory auto-tuner (try 256→384→512, stop before OOM) | Pending |
| A2.5 | AMP scaler diagnostics (log scale factor, overflow count) | Pending |
| A2.6 | CPU bottleneck profiler (dataloader vs compute vs Python overhead) | Pending |
| A2.7 | **DataLoader health monitor** (batch shape checks, NaN/Inf detection, worker crash detection, timing) | Pending |

### A3 — Training Stability & Crash Recovery

| ID | Deliverable | Status |
|----|-------------|--------|
| A3.1 | Auto-resume from last checkpoint (`--resume /path/to/ckpt`) | Pending |
| A3.2 | Save full training state (optimizer, scaler, epoch, batch, scheduler) | Pending |
| A3.3 | Resume-safe logging (append to existing log, not overwrite) | Pending |
| A3.4 | Crash sentinel file (`.training_active`, deleted on clean exit) | Pending |

### A4 — Global Run Metadata

Every training run produces `reports/run_metadata.json` containing:

| Field | Source |
|-------|--------|
| `git_commit` | `git rev-parse HEAD` |
| `cli_args` | `vars(args)` |
| `dataset_hash` | SHA-256 of CSV header + row count + first/last 100 rows |
| `model_param_count` | `sum(p.numel() for p in model.parameters())` |
| `hardware` | GPU name, VRAM, num_workers, batch_size |
| `start_time` / `end_time` | ISO timestamps |
| `crash_sentinel_status` | Active / Clean |
| `schema_version` | `"v4.3.0"` |

### A5 — Global Run Summary

At end of training, produce `reports/run_summary.json`:

| Field | Content |
|-------|---------|
| `best_model_path` | Path to best .pth |
| `convergence_batch` | Batch where convergence was first triggered |
| `convergence_loss` | Loss at convergence |
| `total_training_time` | Wall-clock seconds |
| `gpu_utilization_stats` | Mean/max from profiler |
| `subsystem_specialization_summary` | Final TFT/CDE/ETD contribution ratios |
| `top_predicates` | Top 10 by activation frequency |
| `top_conflicts` | Top 5 subsystem disagreements |
| `drift_summary` | Spectral radius trend, parameter drift |

---

## TRACK B — COGNITIVE TELEMETRY LAYER (CTL)

### B1 — Batch-Level Cognitive Telemetry

**CLI**: `--deep-observe` + `--observe-every N`

| ID | Metric | Source in Model | Status |
|----|--------|-----------------|--------|
| B1.1 | Predicate activation map (which fired, which didn't) | `diag['predicates']` + `diag['gate_stats']` | ✅ Captured |
| B1.3 | SuperSet routing probabilities | `diag['superset_routing']` | ✅ Captured |
| B1.4 | Fuzzy gate variance | `components['fuzzy']` (already in loss) | ✅ In logs |
| B1.5 | Pattern entropy evolution | `components['pattern_ent']` (already in loss) | ✅ In logs |
| B1.6 | A-matrix drift (spectral radius, eigenvalues) | `diag['cde']['spectral_radius']` | ✅ Captured |
| B1.7 | B-matrix influence (`‖Bu‖ / ‖Ax‖` ratio) | Needs `B_theta.full_matrix()` + state norms | Pending |
| B1.8 | Internal state norms (h, v, m, r) | `diag['h']`, `diag['v']`, etc. | ✅ Captured |
| B1.9 | Gradient norms per module | `model.named_parameters()` after backward | Pending |
| B1.10 | "What changed this batch?" deltas (param drift) | Compare `state_dict` snapshots | Pending |
| B1.F | `deep_observe()` formatter + JSON writer hook | Training loop (post-backward, step 5 in hook order) | Pending |

> [!NOTE]
> B1.2 (Set membership distributions) moved to B2.5 — belongs with Taste Testing.

### B2 — Predicate & Set "Taste Testing" Logs

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| B2.1 | Per-batch: which predicates fired/ignored/reinforced/rejected | Compare `p_k > 0.5` across batches; track activation history |
| B2.2 | Per-predicate sentiment: "liked", "disliked", "resembles previous", "contradicts" | Correlate predicate activation with loss improvement/degradation |
| B2.3 | Per-set/superset: "helped reduce loss", "increased risk", "unstable" | Track set routing × loss component correlation |
| B2.4 | **Rejected logic trace** (moved from B4.2) | Track predicates that fired but were gated out by superset (`super_gate < 0.1`) |
| B2.5 | **Set membership distributions** (moved from B1.2) | `PredicateSet.forward()` weights |
| B2.6 | **Predicate Journal persistence** (JSONL) | Append per-batch predicate decisions to `reports/predicate_journal.jsonl` |
| B2.7 | **Predicate pruning rules** | Decay predicates with sustained low activation and negative sentiment |
| B2.8 | **Predicate merge on resume** | When resuming from checkpoint, reconcile existing journal with new run |

### B3 — Pivot-Aware Learning Telemetry

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| B3.1 | Pivot proximity per batch | `diag['pivot']['pivot_raw']` — last values of p_dist_prev, p_slope_prev |
| B3.2 | Pivot influence on predicates/sets/loss | Correlation: pivot_embed_norm × gate activations × loss components |
| B3.3 | Reversal learning logs: which predicates specialize in reversals | Track predicate activation patterns near pivot_high=1 / pivot_low=1 bars |
| B3.4 | **Pivot Influence Heatmap Generator** | Heatmap of pivot proximity × predicate activation × loss change |

### B4 — Decision Making Transparency

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| B4.1 | Reasoning trace: pattern → predicate → set → superset → direction | Chain: `x_b features` → `p_k` → `PredicateSet` → `SuperSet` → `output_head` |
| B4.2 | **Predicate Explanation Engine** | Mini-interpreter: "Predicate 17 fired because Feature_12 > 0.43 and resembles Predicate 3" |
| B4.3 | Memory update logs: what CDE state changed, what was reinforced | `‖x_k - x_{k-1}‖` per block (h, v, m, r) — large Δh = new market physics learned |
| B4.4 | **Memory Drift tracking** (moved from C4.3) | CDE spectral radius change + hidden state norm evolution — part of reasoning, not just specialization |
| B4.5 | **Logic Chain Graph persistence** (edge list + weights) | Save dominant logic chains to `reports/logic_graphs/` |
| B4.6 | **Logic Chain Graph pruning** | Remove chains with low frequency or negative reward after N epochs |
| B4.7 | **Logic Chain Graph visualization hooks** | Generate edge-list format compatible with Graphviz/NetworkX |

---

## TRACK C — THREE-BRAIN FUSION INTERPRETABILITY

### C1 — Subsystem Attribution

**How each functional subsystem contributed to the batch's output.**

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| C1.1 | TFT contribution: `‖u_k‖ / ‖x_k‖` | Ratio of control influence vs total state |
| C1.2 | CDE contribution: `‖G·dX‖ / ‖x_k‖` | Ratio of CDE response injection vs total state |
| C1.3 | ETD drift contribution: `‖F·x_{k-1}‖ / ‖x_k‖` | Ratio of A-matrix drift vs total state |
| C1.4 | Forcing contribution: `‖D(greeks,r,q)‖ / ‖x_k‖` | Ratio of fundamental forcing vs total state |
| C1.5 | "Winner" subsystem per batch | argmax of {TFT, CDE, ETD, Forcing} contributions |
| C1.6 | **Attribution normalization spec** | Clipping rules (min 1e-8), zero-norm handling (fallback to uniform), sequence aggregation (mean over last 5 timesteps) |

### C2 — Subsystem Specialization Maps

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| C2.1 | TFT: which features it attends to (variable selection weights) | Expose `TFTControlEncoder` attention weights in forward |
| C2.2 | CDE response: which market signals it absorbs most | `G·dX` decomposed by feature dimension — top-5 signal sources |
| C2.3 | ETD drift: memory retention vs forgetting | `cos(x_0, x_T)` — high = strong memory, low = regime shift |
| C2.4 | Predicate specialization: which predicates influence which subsystem | Gradient of each subsystem output w.r.t. predicate activations |

### C3 — Subsystem Conflict Resolution

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| C3.1 | Disagreement index | Sign disagreement between `F·x`, `B·u`, `G·dX` components |
| C3.2 | Conflict logs | When TFT control pushes opposite to CDE drift |
| C3.3 | Resolution mechanism | Which component "won" via superset gating |

### C4 — Subsystem Drift Over Epochs

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| C4.1 | Parameter drift per module | `‖W_epoch_n - W_epoch_{n-1}‖` for A, B, G, D, TFT |
| C4.2 | Activation drift | Running mean/var of each subsystem's output, tracked per epoch |
| C4.4 | Specialization drift | Subsystem contribution ratios over epochs |
| C4.5 | **Drift anomaly detection** | Thresholds + alerts when spectral radius, param drift, or activation drift exceeds safe bounds |
| C4.6 | **Specialization drift visualization** | Plot subsystem contribution ratios over epochs |

> [!NOTE]
> C4.3 (Memory Drift) moved to B4.4 — memory drift is part of reasoning/decision transparency, not just subsystem specialization.

---

## TRACK D — DEEP OBSERVATION, PERFORMANCE & DIAGNOSTICS

### D0 — Deep Observation Report Schema

Every deep-observe batch produces a JSON file following this **canonical 7-section format**:

```
reports/deep_observe/
    epoch_0001/
        batch_0100.json
        batch_0200.json
    epoch_0002/
        ...
```

Each file contains `"schema_version": "v4.3.0"` and these 7 sections:

| # | Section | Contents |
|---|---------|----------|
| 1 | **Batch Metadata** | epoch, batch_index, timestamp, seq_len, batch_size, model_state_hash |
| 2 | **Subsystem Attribution Summary** | TFT/CDE/ETD/Forcing contribution ratios, winner, strongest gradients, agreement/conflict index |
| 3 | **Predicate Reward Trajectory** | Top 5 positive reward, top 5 negative reward, newly fired, decayed/pruned predicates |
| 4 | **Pivot-Aware Reversal Influence** | Distance to nearest pivot, pivot influence on activation, reversal-specialist predicates |
| 5 | **Memory Drift & State Evolution** | CDE state norms (h,v,m,r), TFT attention entropy, state stability indicator |
| 6 | **Logic Chain Compression** | Dominant chain: "Pred 12 + Pred 44 → Set 7 → Superset 3 → Direction: Up" |
| 7 | **Batch Delta Summary** | Predicates added/removed, sets that changed membership, subsystem usage shifts, memory drift Δ |

**Rules:**
- No raw tensor dumps (summaries only: norms, means, top-k)
- No per-neuron logs unless `--debug` mode
- Output to **JSON files only** (never stdout — keeps training log clean)
- Supports chunking: one file per `--observe-every` batches

#### D0.2 — Validation Deep-Observe Mode

Disabled by default. Enabled via `--deep-observe-val`. Runs the same 7-section report on validation batches for debugging generalization gaps.

### D1 — Resource Profiling & Optimization

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| D1.1 | PyTorch Profiler instrumentation | `torch.profiler` for per-layer time + memory |
| D1.2 | Training speed logs | Wall-clock time per batch/epoch, peak memory |
| D1.3 | Multi-GPU training support | `DistributedDataParallel` verification |
| D1.4 | Data pipeline benchmarking | DataLoader timing vs compute timing |
| D1.5 | Mixed-precision evaluation | FP16/BF16 stability and throughput testing |

### D2 — Training Diagnostics & Logging

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| D2.1 | `TrainingDiagnostics` integration | Gate stats and metrics at runtime |
| D2.2 | TensorBoard logging | Losses, gradients, predicate activations, gating |
| D2.3 | Run summary report | Collate hyperparams, final metrics, convergence info |

### D3 — Interpretability Report Generation

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| D3.1 | Epoch-level metric summaries | Aggregate per-batch deep-observe data into epoch summaries |
| D3.2 | Combined interpretability report | Tables/plots of subsystem metrics over full training |

### D4 — Telemetry Integration

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| D4.1 | GUI telemetry integration | Stream key metrics to dashboard (via `init_emitter`) |
| D4.2 | Persistent logging | All logs (JSON) reliably written, recoverable after crashes |

### D5 — Batch Replay Tool

Given a batch index, replay the full forward pass and reproduce:
- Predicate activations, routing decisions, logic chain, memory deltas
- Useful for debugging specific batches that caused anomalies

### D6 — Model DNA Export

A single `model_dna.json` file capturing:
- Architecture (layer types, dimensions)
- Parameter counts per subsystem
- Subsystem boundaries
- Predicate journal summary
- Logic graph summary
- Model state hash (SHA-256 of sorted keys, shapes, means)
- Predicate signature hash (for cross-run comparison)

---

## Cross-Cutting Requirements

### Every JSON File Must Include
- `"schema_version": "v4.3.0"` — prevents future incompatibility
- `"model_state_hash"` — SHA-256 of sorted state_dict keys + shapes + means for drift detection
- `"predicate_signature_hash"` — for cross-run predicate comparison

---

## Verification Plan

### Automated
- Syntax check: `python -c "import intelligence.condor_train_net_v42"`
- Smoke test: `--deep-observe --observe-every 10 --epochs 1 --max-rows 5000`
- Verify checkpoint saves to `models/ckpts/`
- Verify deep-observe JSON files appear in `reports/deep_observe/`

### Manual
- User runs on Lightning AI T4 with `--batch-size 384`
- User monitors `nvidia-smi` for utilization
- User reviews deep observation JSON reports for interpretability

---

## Execution Order (5 Phases)

> [!IMPORTANT]
> Each patch will be reviewed individually before the next one is applied.

### Phase 1 — Stability & Safety
1. A1.4 (save on convergence snapshot)
2. A3.1–A3.4 (crash recovery)
3. A2.7 (DataLoader health monitor)
4. A4 (run metadata)

### Phase 2 — Deep Observability Core
5. D0 (deep observation report schema + JSON writer)
6. B1.F (`deep_observe()` hook)
7. B1.7, B1.9, B1.10 (B-matrix, gradient norms, param drift)
8. B4.3 (memory deltas)
9. B4.5 (logic graph persistence)

### Phase 3 — Predicate & Pivot Intelligence
10. B2.1–B2.5 (taste testing + rejected logic + set distributions)
11. B3.1–B3.4 (pivot-aware learning + heatmap)
12. B2.6–B2.8 (predicate journal persistence, pruning, merge)

### Phase 4 — Subsystem Intelligence
13. C1 (attribution + normalization spec)
14. C2 (specialization maps)
15. C3 (conflict resolution)
16. C4 (drift tracking + anomaly detection + visualization)

### Phase 5 — Reporting & Integration
17. D3 (interpretability reports)
18. D6 (Model DNA export)
19. A5 (run summary)
20. D4 (GUI telemetry)
21. D5 (batch replay)
22. B4.2, B4.6, B4.7 (predicate explanation engine, logic graph pruning/visualization)
