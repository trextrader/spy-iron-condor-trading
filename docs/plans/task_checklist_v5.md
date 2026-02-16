# CondorNet v5.0 — Master Task Checklist

## Track A — Infrastructure & Training Pipeline

### A1: Model Saving
- [x] A1.1 — Default save dir → `models/ckpts/`
- [x] A1.2 — `--checkpoint-dir` CLI flag
- [ ] A1.3 — Atomic save (tmp → validate → rename)
- [ ] A1.4 — Save on convergence snapshot
- [ ] A1.5 — Rotate last N checkpoints (`--keep-ckpts`)

### A2: GPU Optimization
- [x] A2.1 — CLI: `--num-workers`, `--prefetch-factor`, `--persistent-workers`
- [x] A2.2 — DataLoader defaults (workers=4, pin_memory=auto, prefetch=2)
- [x] A2.3 — Duplicate DataLoader block removed
- [ ] A2.4 — GPU memory auto-tuner
- [ ] A2.5 — AMP scaler diagnostics
- [ ] A2.6 — CPU bottleneck profiler

### A3: Crash Recovery
- [ ] A3.1 — Auto-resume from checkpoint
- [ ] A3.2 — Full training state save
- [ ] A3.3 — Resume-safe logging
- [ ] A3.4 — Crash sentinel file

## Track B — Cognitive Telemetry Layer

### B1: Batch-Level Telemetry
- [x] B1.1 — Predicate activation map (diag dict)
- [ ] B1.2 — Set membership distributions
- [x] B1.3 — SuperSet routing (diag dict)
- [x] B1.4 — Fuzzy gate variance (in loss logs)
- [x] B1.5 — Pattern entropy (in loss logs)
- [x] B1.6 — A-matrix spectral (diag dict)
- [ ] B1.7 — B-matrix influence ratio
- [x] B1.8 — State norms h/v/m/r (diag dict)
- [ ] B1.9 — Gradient norms per module
- [ ] B1.10 — Param drift deltas
- [ ] B1.F — `deep_observe()` formatter function + loop hook

### B2: Predicate Taste Testing
- [ ] B2.1 — Fired/ignored/reinforced/rejected per batch
- [ ] B2.2 — Per-predicate sentiment correlation
- [ ] B2.3 — Set/superset loss correlation

### B3: Pivot-Aware Learning
- [ ] B3.1 — Pivot proximity per batch
- [ ] B3.2 — Pivot influence on predicates/sets/loss
- [ ] B3.3 — Reversal specialization tracking

### B4: Decision Transparency
- [ ] B4.1 — Reasoning trace chain
- [ ] B4.2 — Rejected logic trace
- [ ] B4.3 — Memory update logs (CDE state Δ)

## Track C — Three-Brain Fusion Interpretability

### C1: Subsystem Attribution
- [ ] C1.1–C1.4 — TFT/CDE/ETD/Forcing contribution ratios
- [ ] C1.5 — Winner subsystem per batch

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
- [ ] C4.3 — Memory drift (spectral)
- [ ] C4.4 — Specialization drift

## Execution Order
1. A1.3–A1.5 (atomic saves)
2. A3.1–A3.4 (crash recovery)
3. B1.F (deep_observe formatter + loop hook)
4. B1.7, B1.9, B1.10 (remaining telemetry)
5. B2 (taste testing)
6. B3 (pivot learning)
7. B4 (decision traces)
8. C1 (subsystem attribution)
9. C2 (specialization)
10. C3 (conflict resolution)
11. C4 (drift tracking)
12. A2.4–A2.6 (GPU profiler, optional)
