# Exit Scaffold & SimExit Integration — CondorNet v4.3
### Framework: "Entries with Exits and Cognitive Holds, Predictive Analytic Folds"
*Snapshot → Closed-Loop Portfolio Controller*

---

## Phase 1 — Exit Head Scaffold (DONE — local, untested)

- [x] `condor_brain_net_v43.py` — Add `ExitHead` class (PART 8b)
- [x] `condor_brain_net_v43.py` — Add `exit_signal: torch.Tensor` to `CondorNetOutput` dataclass
- [x] `condor_brain_net_v43.py` — Add `exit_signal` to `to_dict()`
- [x] `condor_brain_net_v43.py` — Instantiate `self.exit_head = ExitHead(d_joint)` in `__init__`
- [x] `condor_brain_net_v43.py` — Compute `exit_signal = self.exit_head(joint_last)` in `forward()`
- [x] `condor_brain_net_v43.py` — Return `exit_signal` in `CondorNetOutput(...)` return block
- [x] `schema_v43.py` — Add `"exit_signal"` as col 8 to `TF_LABEL_NAMES`
- [x] `data_pipeline_v43.py` — Add `"exit_signal"` to `PIPELINE_LABEL_NAMES`
- [x] `data_pipeline_v43.py` — Add `df["exit_signal"] = 0.0` placeholder in `compute_multitask_labels()`
- [x] `condor_train_net_v43.py` — Add `(8, 'exit_signal')` to risk label loading loop
- [x] `condor_train_net_v43.py` — Add `exit_bce` as component 11 in `CondorLossV43.forward()`
- [x] `configs/loss_weights_v43.json` — Add `"exit_bce": 0.5`

---

## Phase 1 — Validation (COMPLETE ✓ — Lightning AI 2026-02-27)

- [x] Smoke test: `exit_signal shape torch.Size([2,1])`, values [0.5, 0.5] (zero-init confirmed)
- [x] `CondorNetOutput.exit_signal` field present; `to_dict()` includes it
- [x] `forward()` and `forward_compat()` both return correct shape
- [x] Committed + pushed; Lightning AI `git pull && python test_exit_scaffold.py` → ALL TESTS PASSED
- [ ] Run short training test to confirm `exit_bce` appears in loss component breakdown

---

## Phase 2 — SimExit Labeling Engine (IMPLEMENTED — awaiting Lightning AI test)

Replaces the `df["exit_signal"] = 0.0` placeholder with real optimal-exit labels
computed from historical trade trajectories.

- [x] Added `_bsm_call_vec`, `_bsm_put_vec`, `_ic_value_scalar`, `_ic_value_vec` BSM helpers
- [x] Implemented `compute_simexit_labels()` in `data_pipeline_v43.py`
      Per-day simulation: entry at first bar, V_exit vs V_hold oracle comparison
- [x] Replaced placeholder block with real `compute_simexit_labels()` call
- [x] Added `--simexit-epsilon` CLI arg (default 0.02) wired through `label_kwargs`
- [x] Added `simexit_epsilon` param to `compute_multitask_labels()` signature
- [x] Updated test script with Phase 2 tests (BSM helpers, label shape/range/distribution)
- [ ] Run `python test_exit_scaffold.py` on Lightning AI — verify ALL TESTS PASSED
- [ ] Re-run ETL pipeline: `python intelligence/data_pipeline_v43.py --force --simexit-epsilon 0.02`
- [ ] Validate label distribution in pipeline output (expect ~20-40% exit=1)
- [ ] Retrain CondorNet v4.3 with real exit supervision

---

## Phase 3 — Position State Vector (TODO)

Feed real trade state into ETD-1 memory (`u_t` enrichment per framework Section IX).

- [ ] Define `PositionStateVector` (11 features per framework Section III):
      `[PnL%, UnrealizedPnL$, CreditReceived, BarsHeld, DTERemaining,
        DeltaExposure, GammaExposure, ThetaDecay, IVChangeSinceEntry,
        HighWaterMark, MaxAdverseExcursion]`
- [ ] Add position state features to dataset: new input block `pos_state: [B, T, 11]`
- [ ] Extend `CondorNetV43` forward to accept `pos_state` and concat to `u_t` before ETD-1
- [ ] When idle: `pos_state = zeros` (no open position)
- [ ] Retrain with position state enriched inputs

---

## Phase 4 — Capital Constraint Engine (TODO)

Deterministic portfolio layer above model output (framework Section II).

- [ ] Implement `CapitalConstraintEngine` (standalone, no gradients):
      `C_max = alpha * L * B_t`
      `C_avail = C_max - sum(C_i for open positions)`
      Entry allowed iff `C_strategy <= C_avail`
- [ ] Integrate into `core/backtest_engine.py` as pre-entry gate
- [ ] Implement portfolio flatten rule: `if sum(UnrealizedPnL_i) > 0 → Close All`
- [ ] Wire portfolio VaR threshold override

---

## Phase 5 — Hard Exit Rule Taxonomy (TODO)

Deterministic guardrails that cannot be overridden by model output
(framework Sections IV.1, IV.3, IV.5).

- [ ] Max Loss Exit: `if PnL% <= -200% of credit → Exit` (non-learnable)
- [ ] Delta Violation Exit: `if |net_delta| > 0.30 → Exit`
- [ ] Capital Emergency Exit: `if (B_t - B_peak) / B_peak < -D_max → Exit`
- [ ] Pivot Containment Hard Stop: if price crosses predicted pivot AGAINST position → Exit
      (framework Section V.2 — "not a soft gate but a hard stop")
- [ ] Portfolio Flatten Override: integrate with Phase 4 flatten rule
- [ ] Wire all hard exits into `core/backtest_engine.py` exit stack

---

## Phase 6 — Production Exit Stack (TODO)

Full multi-stage exit stack (framework Section X):

```
Exit if:
  HardExit
  OR (p_exit > 0.70 AND NOT in Protected Hold Zone)
```

- [ ] Implement `ExitDecisionStack` in `intelligence/` or `core/`
- [ ] Integrate hold zones (pre-decay DTE>14, pivot containment, theta favorable)
- [ ] Wire `exit_signal` from model output into `ExitDecisionStack`
- [ ] Back-test the full stack vs baseline hard-rule-only exits
- [ ] Tune `p_exit` threshold (framework default: 0.70)

---

## Notes

- Phase 1 changes are **local only** — not committed or pushed
- `ExitHead` is zero-initialized → starts at neutral p=0.5; has no effect until Phase 2 labels are real
- Checkpoint resume uses `strict=False` — old checkpoints load cleanly, `exit_head.*` inits to zero
- Framework document: `docs/Entries with Exits and Cognitive Holds Predictive Analytic Folds.pdf`
