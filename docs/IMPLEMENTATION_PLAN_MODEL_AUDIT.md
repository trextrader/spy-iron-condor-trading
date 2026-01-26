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

## 3) Deliverables
### 3.1 Scripts (Smoke tests, fast gating)
- `scripts/model_tests/01_checkpoint_inventory.py`
- `scripts/model_tests/02_feature_schema_alignment.py`
- `scripts/model_tests/03_norm_stats_sanity.py`
- `scripts/model_tests/04_model_forward_contract.py`
- `scripts/model_tests/05_output_distribution.py`
- `scripts/model_tests/06_ruleset_execution_smoke.py`
- `scripts/model_tests/07_feature_pipeline_parity.py`

### 3.2 Audit tools (heavier, on-demand)
- `audit/export_learned_conditions.py` (attribution + surrogate rules)
- `audit/generate_decision_factor_attribution.py`
- `audit/generate_audit_packet.py`

### 3.3 Artifacts
- `artifacts/learned_conditions/*`
- `artifacts/audit/*`
- `artifacts/audit/run_manifest.json`

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

### 4.3 Live workflow
Before demo/live:
1. Run all smoke tests.
2. Generate a `run_manifest.json` and attach to the session log.
3. Confirm backtest/live output contract match.

## 5) Phased Execution Plan
### Phase A — Smoke Test Harness (Immediate)
1. Wire smoke tests to run locally before backtest.
2. Gate PRs on smoke suite for any model-affecting change.

### Phase B — Audit Artifacts (Short-term)
1. Standardize learned-conditions exports for each model build.
2. Produce decision trace + attribution CSV.
3. Produce audit packet PDF for release candidates.

### Phase C — Docs Contract (Short-term)
1. Add `docs/PIPELINE_CONTRACT.md` and `docs/PIPELINE_CONTRACT.json`.
2. Create a CI check to compare docs vs code-derived contract.

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

## 7) Acceptance Criteria
- Smoke tests pass on clean environment.
- Checkpoint metadata present and consistent.
- Output contract verified; backtest/live match.
- Audit packet generated for release candidates.
