# Exit Scaffold & SimExit Integration — CondorNet v4.3
### Framework: "Entries with Exits and Cognitive Holds, Predictive Analytic Folds"
*Snapshot → Closed-Loop Portfolio Controller*

---

## Phase 1 — Exit Head Scaffold (COMPLETE ✓ — Lightning AI 2026-02-27)

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

## Phase 2 — SimExit Labeling Engine (COMPLETE ✓ — Lightning AI 2026-02-27)

Replaces the `df["exit_signal"] = 0.0` placeholder with real optimal-exit labels
computed from historical trade trajectories.

- [x] Added `_bsm_call_vec`, `_bsm_put_vec`, `_ic_value_scalar`, `_ic_value_vec` BSM helpers
- [x] Implemented `compute_simexit_labels()` in `data_pipeline_v43.py`
      Per-day simulation: entry at first bar, V_exit vs V_hold oracle comparison
- [x] Replaced placeholder block with real `compute_simexit_labels()` call
- [x] Added `--simexit-epsilon` CLI arg (default 0.02) wired through `label_kwargs`
- [x] Added `simexit_epsilon` param to `compute_multitask_labels()` signature
- [x] Updated test script with Phase 2 tests (BSM helpers, label shape/range/distribution)
- [x] Run `python test_exit_scaffold.py` on Lightning AI — ALL TESTS PASSED ✓
- [x] Re-run ETL pipeline: `python intelligence/data_pipeline_v43.py --force --simexit-epsilon 0.02`
- [x] Validate label distribution in pipeline output — 7,220 / 18,494 bars (39.0%) ✓
- [x] Retrain CondorNet v4.3 with real exit supervision
      Best epoch 41 | val=1.7748 | exit_bce=0.5690 (drop: 0.1099 from zero-init) ✓

---

## Phase 3 — Position State Vector (COMPLETE ✓ — 2026-02-28)

Feed real trade state into ETD-1 memory (`u_t` enrichment per framework Section IX).

- [x] Define `PositionStateVector` (11 features — `POS_STATE_NAMES` / `N_POS_STATE` in `schema_v43.py`):
      `[ps_pnl_pct, ps_credit_norm, ps_bars_held, ps_dte_frac, ps_delta_exp,
        ps_gamma_exp, ps_theta_pos, ps_iv_change, ps_high_water, ps_mae, ps_unrealized_norm]`
- [x] `data_pipeline_v43.py` — Added BSM Greek helpers (`_bsm_delta_call_vec`, `_bsm_gamma_vec`, `_bsm_theta_call_vec`)
- [x] `data_pipeline_v43.py` — Implemented `compute_posstate_features()`: per-day IC simulation, 11 features per bar
- [x] `data_pipeline_v43.py` — Wired call into `compute_multitask_labels()` (step 8) after simexit
- [x] `data_pipeline_v43.py` — Added `--no-posstate` CLI arg; `skip_posstate` param in `compute_multitask_labels()`
- [x] `condor_brain_net_v43.py` — Added `PosStateProjector` class (PART 8c): Linear(11→256)→Tanh, zero-init
- [x] `condor_brain_net_v43.py` — Added `self.pos_state_proj = PosStateProjector(11, d_tf_joint)` in `__init__`
- [x] `condor_brain_net_v43.py` — Added `pos_state: Optional[Tensor] = None` to `forward()` signature
- [x] `condor_brain_net_v43.py` — Injection: `tf_fused += pos_state_proj(pos_state)` (Step 2b, after pivot fusion)
- [x] `condor_train_net_v43.py` — Added `_load_pos_state()` helper; loads 11 ps_* cols from M5 CSV
- [x] `condor_train_net_v43.py` — `V43Dataset`: added `m5_pos_state` param; `__getitem__` returns `pos_state [T,11]`
- [x] `condor_train_net_v43.py` — `v43_collate_fn`: stacks `pos_state` → `[B, T, 11]`
- [x] `condor_train_net_v43.py` — `build_dataloaders_v43`: loads pos_state, slices train/val, passes to dataset
- [x] `condor_train_net_v43.py` — Training + validation forward calls pass `pos_state=pos_state_batch`
- [x] `condor_train_net_v43.py` — Added `--min-delta` arg (default 0.0); patience uses `val < best - min_delta`
- [ ] Re-run ETL pipeline: `python intelligence/data_pipeline_v43.py --force --simexit-epsilon 0.02`
- [ ] Validate 11 ps_* columns present in m5_dataset_v43_final.csv
- [ ] Retrain CondorNet v4.3 with position state enriched inputs

---

## Phase 4 — Capital Constraint Engine (COMPLETE ✓ — 2026-02-28)

Deterministic portfolio layer above model output (framework Section II).

- [x] Implement `CapitalConstraintEngine` (standalone, no gradients) in `intelligence/exit_stack.py`:
      `C_max = alpha * L * B_t`
      `C_avail = C_max - sum(C_i for open positions)`
      Entry allowed iff `C_strategy <= C_avail`
- [x] Integrate into `core/backtest_engine.py` as pre-entry gate (Phase 4 Capital Constraint Gate block)
- [x] Implement portfolio flatten rule: `if sum(UnrealizedPnL_i) > 0 → Close All` (via ExitDecisionStack)
- [x] Wire capital emergency: `(B_peak - B_t) / B_peak > d_max → emergency exit` (in HardExitRules)
- [x] Config params added to `core/config.py`: `capital_constraint_alpha`, `capital_constraint_L`, `exit_hard_max_dd_pct`

---

## Phase 5 — Hard Exit Rule Taxonomy (COMPLETE ✓ — 2026-02-28)

Deterministic guardrails that cannot be overridden by model output
(framework Sections IV.1, IV.3, IV.5).

- [x] Max Loss Exit: `if cost >= 2.0 × credit_received → Exit` (non-learnable) — `hard_max_loss`
- [x] Delta Violation Exit: `if |net_delta| > 0.30 → Exit` — `hard_delta_violation`
      Net IC delta computed from live chain each bar; falls back to entry delta if chain unavailable
- [x] Capital Emergency Exit: `if (B_peak - B_t) / B_peak > d_max → Exit` — `hard_capital_emergency`
- [x] Pivot Containment Hard Stop: if spot >= pivot_high → `hard_pivot_call_breach`;
      if spot <= pivot_low → `hard_pivot_put_breach`
      (framework Section V.2 — "not a soft gate but a hard stop")
      pivot_high/pivot_low = None-safe; activates when CondorNet pivot_pred_head wired to inference
- [x] Portfolio Flatten Override: `portfolio_flatten_triggered()` in `CapitalConstraintEngine`
- [x] Wire all hard exits into `core/backtest_engine.py` via `ExitDecisionStack.evaluate()`
- [x] Config params: `exit_hard_max_loss_mult`, `exit_hard_max_delta`, `exit_hard_max_dd_pct`

---

## Phase 6 — Production Exit Stack (COMPLETE ✓ — 2026-02-28)

Full multi-stage exit stack (framework Section X):

```
Exit if:
  HardExit
  OR PortfolioFlattenTriggered
  OR (p_exit > 0.70 AND NOT in Protected Hold Zone)
```

- [x] Implement `ExitDecisionStack` in `intelligence/exit_stack.py`
- [x] Integrate hold zones:
      - Pre-decay DTE > 14 bars remaining
      - Theta-favorable: ps_theta_pos > 0.005
      - High-water: ps_high_water > 0.80 (near peak value)
- [x] Wire `exit_signal` from model output into `ExitDecisionStack`:
      reads `row['exit_signal']` from `neural_forecasts` DataFrame if column present;
      falls back to p_exit=0.5 (neutral) until CondorNet v4.3 inference is wired
- [x] Wire `position_high_water` tracking in `IronCondorStrategy` for hold-zone computation
- [x] Wire `dte_remaining` from expiry date for pre-decay protection
- [x] Config params: `use_exit_stack`, `exit_stack_p_threshold`, `exit_stack_dte_protected`,
      `exit_stack_theta_floor`, `exit_stack_high_water_floor`
- [ ] Back-test the full stack vs baseline hard-rule-only exits  ← next step after Lightning AI validation
- [ ] Tune `p_exit` threshold via backtest sweep once exit_signal column is populated

---

## Notes

- All Phase 1 + Phase 2 changes committed and pushed to `main`
- All Phase 3 + Phase 4 + Phase 5 + Phase 6 changes committed and pushed to `main` (2026-02-28)
- `ExitHead` is zero-initialized → starts at neutral p=0.5; learns real timing after Phase 2 retraining
- Checkpoint resume uses `strict=False` — old checkpoints load cleanly, `exit_head.*` inits to zero
- M5 CSV: 102 columns (91 + 11 ps_* Phase 3 columns), 39% exit=1 (7,220 / 18,494 bars, 238 days, epsilon=0.02)
- Framework document: `docs/Entries with Exits and Cognitive Holds Predictive Analytic Folds.pdf`
- `exit_signal` → `neural_forecasts` wiring: column will auto-activate when CondorNet v4.3 inference
  is plumbed into `run_backtest_headless` (replaces current Mamba engine path)
- `pivot_high` / `pivot_low` → wiring: will auto-activate when `pivot_pred_head` output is plumbed
  into neural_forecasts; both are None-safe in the current implementation
