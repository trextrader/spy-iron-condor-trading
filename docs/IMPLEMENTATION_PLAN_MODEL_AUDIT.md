# Implementation Plan: End-to-End Model/Checkpoint/Docs Audit

**Version:** 1.0  
**Date:** 2026-01-26  
**Status:** Draft Plan  
**Owner:** Engineering / Quant Research

## 1) Purpose
Establish a strict, pen-test style verification layer for the full pipeline:
training → serialization → inference contract → backtest → live usage → docs.

## 2) Scope (Contracts)
1. **Feature Vector Contract**
   - Deterministic feature ordering and schema hash.
   - Feature parity across training/backtest/live builders.
2. **Target/Label Contract**
   - Target horizons and thresholds match documentation.
   - No lookahead leakage.
3. **Head Mapping Contract**
   - Output heads are versioned and keyed, not inferred by index.
4. **Checkpoint Contract**
   - Checkpoint includes feature_cols, input_dim, scaler stats, version, and run metadata.
5. **Interpretation Contract**
   - Backtest and live interpret outputs identically (thresholds, sizing, risk gates).
6. **Docs Contract**
   - Docs reflect code; CI fails on drift.
7. **Time & Session Contract**
   - Session open/close boundaries, DST shifts, holidays, and bar alignment rules are explicit.
   - Options chain timestamps align with spot bars.
8. **Execution & PnL Accounting Contract**
   - Slippage, fees, fills, multipliers, and rounding are identical across backtest/live.
9. **Risk Gate Precedence Contract**
   - Gate order and override behavior are documented and enforced.
10. **Determinism & Reproducibility Contract**
   - Seeds, library versions, and inference determinism are captured in run manifests.

## 3) Deliverables
### 3.1 Scripts (Smoke tests, fast gating)
- `scripts/model_tests/01_checkpoint_inventory.py`
- `scripts/model_tests/02_feature_schema_alignment.py`
- `scripts/model_tests/03_norm_stats_sanity.py`
- `scripts/model_tests/04_model_forward_contract.py`
- `scripts/model_tests/05_output_distribution.py`
- `scripts/model_tests/06_ruleset_execution_smoke.py`
- `scripts/model_tests/07_feature_pipeline_parity.py`
- `scripts/model_tests/08_live_trading_parity_sim.py`
- `scripts/model_tests/09_checkpoint_diff_epochs.py`
- `scripts/model_tests/10_decision_trace_schema_validate.py`
- `scripts/model_tests/11_factor_attribution_aggregate.py`
- `scripts/model_tests/12_docs_contract_sync_check.py`
- `scripts/model_tests/test_time_alignment_and_session.py`
- `scripts/model_tests/test_pnl_accounting_parity.py`
- `scripts/model_tests/test_model_autodiscovery_selection.py`

### 3.2 Audit tools (heavier, on-demand)
- `audit/export_learned_conditions.py` (attribution + surrogate rules)
- `audit/generate_decision_factor_attribution.py`
- `audit/generate_audit_packet.py`

### 3.3 Artifacts
- `artifacts/learned_conditions/*`
- `artifacts/audit/*`
- `artifacts/audit/run_manifest.json`
- `artifacts/audit/contract_snapshot.json`
- `artifacts/audit/golden_fixtures/*`

## 4) Workflow Integration
### 4.1 Training workflow
After each training run:
1. Export checkpoint metadata (feature_cols, input_dim, median/mad, version).
2. Run `audit/export_learned_conditions.py` on the new model.
3. Store outputs under `artifacts/learned_conditions/`.

### 4.2 Backtest workflow
Before backtest:
1. Run `scripts/model_tests/01_checkpoint_inventory.py`.
2. Run `scripts/model_tests/02_feature_schema_alignment.py`.
3. Run `scripts/model_tests/03_norm_stats_sanity.py`.
4. Run `scripts/model_tests/04_model_forward_contract.py`.
5. Run `scripts/model_tests/06_ruleset_execution_smoke.py`.
6. Run `scripts/model_tests/test_time_alignment_and_session.py`.
7. Run `scripts/model_tests/test_pnl_accounting_parity.py`.
8. Run `scripts/model_tests/test_model_autodiscovery_selection.py`.

### 4.3 Live workflow
Before demo/live:
1. Run all smoke tests.
2. Generate a `run_manifest.json` and attach to the session log.
3. Confirm backtest/live output contract match.
4. Confirm time/session alignment and execution accounting parity.

## 5) Phased Execution Plan
### Phase A — Smoke Test Harness (Immediate)
1. Wire smoke tests to run locally before backtest.
2. Gate PRs on smoke suite for any model-affecting change.
3. Add collapse-guard thresholds to output distribution test.

### Phase B — Audit Artifacts (Short-term)
1. Standardize learned-conditions exports for each model build.
2. Produce decision trace + attribution CSV.
3. Produce audit packet PDF for release candidates.

### Phase C — Docs Contract (Short-term)
1. Add `docs/PIPELINE_CONTRACT.md` and `docs/PIPELINE_CONTRACT.json`.
2. Create a CI check to compare docs vs code-derived contract.
3. Add `contract_snapshot.json` to artifacts for each release candidate.

### Phase D — Deployment Validation (Mid-term)
1. Dual-run inference (backtest vs live) on the same slice.
2. Confirm identical actions and risk gates.

## 6) Risks & Mitigations
- **Risk:** Feature drift from data changes.  
  **Mitigation:** Schema hash checks and NaN-fill policy enforcement.
- **Risk:** Output head misinterpretation.  
  **Mitigation:** Named output contract + forward contract test.
- **Risk:** Docs drift.  
  **Mitigation:** CI gate on contract diff.
- **Risk:** Time/session misalignment.  
  **Mitigation:** Dedicated time/session contract and tests.
- **Risk:** Execution accounting divergence.  
  **Mitigation:** Accounting parity test with golden fixtures.
- **Risk:** Model collapse.  
  **Mitigation:** Collapse guards in output distribution tests.

## 7) Acceptance Criteria
- Smoke tests pass on clean environment.
- Checkpoint metadata present and consistent.
- Output contract verified; backtest/live match.
- Audit packet generated for release candidates.
- Time/session alignment verified (no off-by-one or DST errors).
- Execution accounting parity verified (fees, slippage, multipliers).
- Collapse guards pass (non-degenerate outputs).
- Docs contract matches code-derived contract.
- run_manifest.json + contract_snapshot.json generated for each RC model.
