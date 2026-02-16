# CondorNet v5.0 — Master Implementation Plan

> [!IMPORTANT]
> **Status**: REVIEW VERSION — no code changes until approved.
> Changes already applied (from earlier session) are marked ✅.
> All remaining items are pending approval.

---

## Architectural Ground Truth

Before planning observability, here is the **actual subsystem mapping** based on [condor_brain_net_v42.py](file:///C:/SPYOptionTrader_test/intelligence/condor_brain_net_v42.py):

| Subsystem | Role | Key Classes | Where in Forward |
|-----------|------|-------------|-----------------|
| **TFT Control Encoder** | Long-horizon feature attention, variable selection, control embedding production | `TFTControlEncoder` (Part 6) | `self.tft(x)` → produces `u` (batch, seq, d_control) |
| **Neural CDE + ETD-1** | Continuous state evolution, market physics, memory retention | `BlockMatrixA`, `BlockMatrixB`, `CDEResponseG`, `FullForcingD`, `etd1_kernel`, `condornet_master_step` (Parts 2–3) | Time loop: `x_k = F·x_{k-1} + φ₁·B·u + G·dX + D` |
| **Predicate Logic Engine** | Inequality gates, combinatorics, set/superset routing, regime dynamics | `CanonicalPredicateGates`, `PredicateSignature`, `PredicateSet`, `SuperSet`, `RelationalLogicLayer`, `RegimeCombinatoricsDynamics` (Parts 4–5) | Per-timestep: `p_k → z_pred → r_k update → superset gating` |
| **Pivot Encoder** | Structural market turn detection, geometric embedding | `PivotEncoder`, `PivotHead` (Part 7) | Pre-loop: `x_piv_enc = PivotEncoder(x_piv)` → concat to features |
| **FusionGate** | Multi-expert weighting (currently used for superset combination) | `FusionGate` (Part 7) | Post-loop: `super_gate → z_gated → output_head` |

> [!NOTE]
> CondorNet is **not** three literally separate networks. It is a **unified CDE** where: TFT produces the control signal `u_k`, the ETD-1 kernel governs state dynamics via `A_θ`, and CDE response `G_θ` injects market data. The "three brain" decomposition is functional, not architectural — we observe which *functional subsystem* drives each decision.

---

## TRACK A — INFRASTRUCTURE & TRAINING PIPELINE

### A1 — Fix Model Saving Behavior

| ID | Deliverable | Status |
|----|-------------|--------|
| A1.1 | Default save dir → `models/ckpts/` | ✅ Done |
| A1.2 | `--checkpoint-dir` CLI flag (was `--save-dir`) | ✅ Done (line 846) |
| A1.3 | Atomic save: write to `.tmp`, validate, rename | Pending |
| A1.4 | Save on every convergence snapshot | Pending |
| A1.5 | Rotate last N checkpoints (`--keep-ckpts N`, default 3) | Pending |

### A2 — GPU Utilization Optimization

| ID | Deliverable | Status |
|----|-------------|--------|
| A2.1 | CLI flags: `--num-workers`, `--prefetch-factor`, `--persistent-workers` | ✅ Done |
| A2.2 | DataLoader defaults: workers=4, pin_memory=auto, persistent=True, prefetch=2 | ✅ Done |
| A2.3 | Duplicate DataLoader block removed | ✅ Done |
| A2.4 | GPU memory auto-tuner (try 256→384→512, stop before OOM) | Pending |
| A2.5 | AMP scaler diagnostics (log scale factor, overflow count) | Pending |
| A2.6 | CPU bottleneck profiler (dataloader vs compute vs Python overhead) | Pending |

### A3 — Training Stability & Crash Recovery

| ID | Deliverable | Status |
|----|-------------|--------|
| A3.1 | Auto-resume from last checkpoint (`--resume /path/to/ckpt`) | Pending |
| A3.2 | Save full training state (optimizer, scaler, epoch, batch, scheduler) | Pending |
| A3.3 | Resume-safe logging (append to existing log, not overwrite) | Pending |
| A3.4 | Crash sentinel file (`.training_active`, deleted on clean exit) | Pending |

---

## TRACK B — COGNITIVE TELEMETRY LAYER (CTL)

### B1 — Batch-Level Cognitive Telemetry

**CLI**: `--deep-observe` + `--observe-every N`

| ID | Metric | Source in Model | Status |
|----|--------|-----------------|--------|
| B1.1 | Predicate activation map (which fired, which didn't) | `diag['predicates']` + `diag['gate_stats']` | ✅ Captured |
| B1.2 | Set membership distributions | `PredicateSet.forward()` weights | Pending |
| B1.3 | SuperSet routing probabilities | `diag['superset_routing']` | ✅ Captured |
| B1.4 | Fuzzy gate variance | `components['fuzzy']` (already in loss) | ✅ In logs |
| B1.5 | Pattern entropy evolution | `components['pattern_ent']` (already in loss) | ✅ In logs |
| B1.6 | A-matrix drift (spectral radius, eigenvalues) | `diag['cde']['spectral_radius']` | ✅ Captured |
| B1.7 | B-matrix influence (`‖Bu‖ / ‖Ax‖` ratio) | Needs `B_theta.full_matrix()` + state norms | Pending |
| B1.8 | Internal state norms (h, v, m, r) | `diag['h']`, `diag['v']`, etc. | ✅ Captured |
| B1.9 | Gradient norms per module | `model.named_parameters()` after backward | Pending |
| B1.10 | "What changed this batch?" deltas (param drift) | Compare `state_dict` snapshots | Pending |

### B2 — Predicate & Set "Taste Testing" Logs

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| B2.1 | Per-batch: which predicates fired/ignored/reinforced/rejected | Compare `p_k > 0.5` across batches; track activation history |
| B2.2 | Per-predicate sentiment: "liked", "disliked", "resembles previous", "contradicts" | Correlate predicate activation with loss improvement/degradation |
| B2.3 | Per-set/superset: "helped reduce loss", "increased risk", "unstable" | Track set routing × loss component correlation |

### B3 — Pivot-Aware Learning Telemetry

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| B3.1 | Pivot proximity per batch | `diag['pivot']['pivot_raw']` — last values of p_dist_prev, p_slope_prev |
| B3.2 | Pivot influence on predicates/sets/loss | Correlation: pivot_embed_norm × gate activations × loss components |
| B3.3 | Reversal learning logs: which predicates specialize in reversals | Track predicate activation patterns near pivot_high=1 / pivot_low=1 bars |

### B4 — Decision Making Transparency

| ID | Deliverable | Implementation |
|----|-------------|---------------|
| B4.1 | Reasoning trace: pattern → predicate → set → superset → direction | Chain: `x_b features` → `p_k` → `PredicateSet` → `SuperSet` → `output_head` |
| B4.2 | Rejected logic trace: "considered predicate A but rejected" | Track predicates that fired but were gated out by superset (`super_gate < 0.1`) |
| B4.3 | Memory update logs: what CDE state changed, what was reinforced | `‖x_k - x_{k-1}‖` per block (h, v, m, r) — large Δh = new market physics learned |

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
| C4.3 | Memory drift | CDE spectral radius change over epochs |
| C4.4 | Specialization drift | Subsystem contribution ratios over epochs |

---

## Output Format

All deep observation data is:
1. **Printed to stdout** (inside `[DEEP OBSERVE B{N}]` blocks)
2. **Logged to JSON** in `reports/deep_observe/` (one file per observed batch)
3. **Summarized per epoch** in the interpretability report

---

## Verification Plan

### Automated
- Syntax check: `python -c "import intelligence.condor_train_net_v42"`
- Smoke test: `--deep-observe --observe-every 10 --epochs 1 --max-rows 5000`
- Verify checkpoint saves to `models/ckpts/`

### Manual
- User runs on Lightning AI T4 with `--batch-size 384 --num-workers 4`
- User monitors `nvidia-smi` for utilization
- User reviews deep observation reports for interpretability

---

## Execution Order

> [!IMPORTANT]
> Each patch will be reviewed individually before the next one is applied.

1. **A1.3–A1.5**: Atomic saves + rotation
2. **A3.1–A3.4**: Crash recovery
3. **B1 formatter**: `deep_observe()` function + training loop hook
4. **B1.7, B1.9, B1.10**: B-matrix, gradient norms, param drift
5. **B2**: Predicate taste testing
6. **B3**: Pivot-aware telemetry
7. **B4**: Decision transparency traces
8. **C1**: Subsystem attribution
9. **C2**: Specialization maps (requires TFT attention exposure)
10. **C3**: Conflict resolution
11. **C4**: Epoch-level drift tracking
12. **A2.4–A2.6**: GPU auto-tuner + profiler (optional, can be deferred)
