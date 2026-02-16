# CondorNet v4.3 — Master Task Checklist

## Track A — Infrastructure & Training Pipeline

### A0: Training Loop Hook Order
- [ ] A0.1 — Document and enforce hook execution order (forward → loss → backward → step → observe → checkpoint → emit)

### A1: Model Saving
- [x] A1.1 — Default save dir → `models/ckpts/`
- [x] A1.2 — `--checkpoint-dir` CLI flag
- [x] A1.3 — Atomic save (tmp → validate → rename)
- [ ] A1.4 — Save on convergence snapshot
- [x] A1.5 — Rotate last N checkpoints (`--keep-ckpts`)

### A2: GPU Optimization
- [x] A2.1 — CLI: `--num-workers`, `--prefetch-factor`, `--persistent-workers`
- [x] A2.2 — DataLoader hardcoded (workers=4, pin_memory=True, persistent=True, prefetch=2)
- [x] A2.3 — Duplicate DataLoader block removed
- [ ] A2.4 — GPU memory auto-tuner
- [ ] A2.5 — AMP scaler diagnostics
- [ ] A2.6 — CPU bottleneck profiler
- [ ] A2.7 — DataLoader health monitor (shape checks, NaN/Inf, timing, worker crash detection)

### A3: Crash Recovery
- [ ] A3.1 — Auto-resume from checkpoint (`--resume`)
- [ ] A3.2 — Full training state save (optimizer, scaler, epoch, batch, scheduler)
- [ ] A3.3 — Resume-safe logging (append, not overwrite)
- [ ] A3.4 — Crash sentinel file (`.training_active`)

### A4: Global Run Metadata
- [ ] A4.1 — Write `reports/run_metadata.json` at training start (git hash, CLI args, dataset hash, hardware, schema_version)

### A5: Global Run Summary
- [ ] A5.1 — Write `reports/run_summary.json` at training end (best model, convergence, timing, top predicates, drift)

---

## Track B — Cognitive Telemetry Layer

### B1: Batch-Level Telemetry
- [x] B1.1 — Predicate activation map (diag dict)
- [x] B1.3 — SuperSet routing (diag dict)
- [x] B1.4 — Fuzzy gate variance (in loss logs)
- [x] B1.5 — Pattern entropy (in loss logs)
- [x] B1.6 — A-matrix spectral (diag dict)
- [ ] B1.7 — B-matrix influence ratio
- [x] B1.8 — State norms h/v/m/r (diag dict)
- [ ] B1.9 — Gradient norms per module
- [ ] B1.10 — Param drift deltas
- [ ] B1.F — `deep_observe()` formatter + JSON writer hook

### B2: Predicate & Set Taste Testing
- [ ] B2.1 — Fired/ignored/reinforced/rejected per batch
- [ ] B2.2 — Per-predicate sentiment correlation
- [ ] B2.3 — Set/superset loss correlation
- [ ] B2.4 — Rejected logic trace (moved from B4.2)
- [ ] B2.5 — Set membership distributions (moved from B1.2)
- [ ] B2.6 — Predicate Journal persistence (JSONL)
- [ ] B2.7 — Predicate pruning rules
- [ ] B2.8 — Predicate merge on resume

### B3: Pivot-Aware Learning
- [ ] B3.1 — Pivot proximity per batch
- [ ] B3.2 — Pivot influence on predicates/sets/loss
- [ ] B3.3 — Reversal specialization tracking
- [ ] B3.4 — Pivot Influence Heatmap Generator

### B4: Decision Transparency
- [ ] B4.1 — Reasoning trace chain
- [ ] B4.2 — Predicate Explanation Engine
- [ ] B4.3 — Memory update logs (CDE state Δ)
- [ ] B4.4 — Memory Drift tracking (moved from C4.3)
- [ ] B4.5 — Logic Chain Graph persistence
- [ ] B4.6 — Logic Chain Graph pruning
- [ ] B4.7 — Logic Chain Graph visualization hooks

---

## Track C — Three-Brain Fusion Interpretability

### C1: Subsystem Attribution
- [ ] C1.1 — TFT contribution ratio
- [ ] C1.2 — CDE contribution ratio
- [ ] C1.3 — ETD drift contribution ratio
- [ ] C1.4 — Forcing contribution ratio
- [ ] C1.5 — Winner subsystem per batch
- [ ] C1.6 — Attribution normalization spec (clipping, zero-norm, sequence aggregation)

### C2: Specialization Maps
- [ ] C2.1 — TFT variable selection weights
- [ ] C2.2 — CDE signal absorption decomposition
- [ ] C2.3 — ETD memory retention score
- [ ] C2.4 — Predicate→subsystem gradient influence

### C3: Conflict Resolution
- [ ] C3.1 — Disagreement index
- [ ] C3.2 — Conflict logs
- [ ] C3.3 — Resolution mechanism trace

### C4: Epoch-Level Drift
- [ ] C4.1 — Parameter drift per module
- [ ] C4.2 — Activation drift
- [ ] C4.4 — Specialization drift
- [ ] C4.5 — Drift anomaly detection (thresholds + alerts)
- [ ] C4.6 — Specialization drift visualization

---

## Track D — Deep Observation, Performance & Diagnostics

### D0: Deep Observation Report Schema
- [ ] D0.1 — Define canonical 7-section JSON format (metadata, attribution, predicates, pivots, memory, logic chain, deltas)
- [ ] D0.2 — Validation deep-observe mode (`--deep-observe-val`)

### D1: Resource Profiling
- [ ] D1.1 — PyTorch Profiler instrumentation
- [ ] D1.2 — Training speed and memory logs
- [ ] D1.3 — Multi-GPU training support
- [ ] D1.4 — Data pipeline benchmarking
- [ ] D1.5 — Mixed-precision evaluation

### D2: Training Diagnostics & Logging
- [ ] D2.1 — `TrainingDiagnostics` integration for gate stats
- [ ] D2.2 — TensorBoard logging (losses, gradients, activations)
- [ ] D2.3 — Run summary report (hyperparams, metrics)

### D3: Interpretability Reporting
- [ ] D3.1 — Epoch-level metric summaries
- [ ] D3.2 — Combined interpretability report

### D4: Telemetry Integration
- [ ] D4.1 — Stream metrics to GUI/dashboard
- [ ] D4.2 — Persistent logging (stdout/JSON) across sessions

### D5: Batch Replay Tool
- [ ] D5.1 — Replay forward pass for any batch index with full diagnostics

### D6: Model DNA Export
- [ ] D6.1 — `model_dna.json` (architecture, params, boundaries, hashes)

---

## Execution Order (5 Phases)

### Phase 1 — Stability & Safety
1. A1.4 (convergence snapshot saves)
2. A3.1–A3.4 (crash recovery)
3. A2.7 (DataLoader health monitor)
4. A4 (run metadata)

### Phase 2 — Deep Observability Core
5. D0 (report schema + JSON writer)
6. B1.F (deep_observe hook)
7. B1.7, B1.9, B1.10 (B-matrix, grads, drift)
8. B4.3 (memory deltas)
9. B4.5 (logic graph persistence)

### Phase 3 — Predicate & Pivot Intelligence
10. B2.1–B2.5 (taste testing + rejected logic + set distributions)
11. B3.1–B3.4 (pivot learning + heatmap)
12. B2.6–B2.8 (predicate journal persistence, pruning, merge)

### Phase 4 — Subsystem Intelligence
13. C1 (attribution + normalization)
14. C2 (specialization maps)
15. C3 (conflict resolution)
16. C4 (drift + anomaly detection + visualization)

### Phase 5 — Reporting & Integration
17. D3 (interpretability reports)
18. D6 (Model DNA)
19. A5 (run summary)
20. D4 (GUI telemetry)
21. D5 (batch replay)
22. B4.2, B4.6, B4.7 (predicate explanations, logic graph pruning/viz)
