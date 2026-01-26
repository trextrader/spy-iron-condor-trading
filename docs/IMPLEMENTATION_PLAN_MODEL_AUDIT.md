Implementation Plan: End-to-End Model/Checkpoint/Docs Audit

Version: 1.1
Date: 2026-01-26
Status: Draft Plan (Ready for Implementation)
Owner: Engineering / Quant Research

1) Purpose

Establish a strict, pen-test style verification layer for the full pipeline:
training -> serialization -> inference contract -> backtest -> live usage -> docs, with no silent drift allowed.

2) Scope (Contracts)

Feature Vector Contract

- Deterministic feature ordering and schema hash.
- Feature parity across training/backtest/live builders.

Target/Label Contract

- Target horizons and thresholds match documentation.
- No lookahead leakage (train/val split integrity, no future bleed).

Head Mapping Contract

- Output heads are versioned and keyed, not inferred by index.

Checkpoint Contract

- Checkpoint includes feature_cols, input_dim, scaler stats, version, and run metadata.

Interpretation Contract

- Backtest and live interpret outputs identically (thresholds, sizing, risk gates).

Docs Contract

- Docs reflect code; CI fails on drift.

Time & Session Contract

- Session open/close boundaries, DST shifts, holidays, and bar alignment rules are explicit.
- Options chain timestamps align with spot bars (explicit alignment policy).

Execution & PnL Accounting Contract

- Slippage, fees, fills, multipliers, and rounding are identical across backtest/live (or explicitly parameterized and logged).

Risk Gate Precedence Contract

- Gate order and override behavior are documented and enforced (including "block entry / allow exit-only" modes).

Determinism & Reproducibility Contract

- Seeds, library versions, and inference determinism are captured in run manifests.

Data Freshness & Staleness Contract

- Max tolerated staleness for spot bars, options chain, and greeks is enforced.
- Stale data forces no-entry / exit-only behavior and logs gate failures.

Instrument & Contract Selection Contract

- OCC parsing, strike rounding, expiry selection, and DTE filters are explicit and tested.

Multi-leg Execution Integrity Contract

- Partial fills, leg imbalance, and cancel/replace behavior are deterministic and safe.

Safety Controls / Kill Switch Contract

- Daily loss limits, consecutive-loss limits, anomaly halts, and staleness halts are enforced.

Backward Compatibility Contract

- Checkpoints declare compatibility with registry/head mapping/normalization versions and are rejected if incompatible.

3) Deliverables

3.1 Scripts (Smoke tests, fast gating)

- scripts/model_tests/01_checkpoint_inventory.py
- scripts/model_tests/02_feature_schema_alignment.py
- scripts/model_tests/03_norm_stats_sanity.py
- scripts/model_tests/04_model_forward_contract.py
- scripts/model_tests/05_output_distribution.py
- scripts/model_tests/06_ruleset_execution_smoke.py
- scripts/model_tests/07_feature_pipeline_parity.py
- scripts/model_tests/08_live_trading_parity_sim.py
- scripts/model_tests/09_checkpoint_diff_epochs.py
- scripts/model_tests/10_decision_trace_schema_validate.py
- scripts/model_tests/11_factor_attribution_aggregate.py
- scripts/model_tests/12_docs_contract_sync_check.py
- scripts/model_tests/test_time_alignment_and_session.py
- scripts/model_tests/test_pnl_accounting_parity.py
- scripts/model_tests/test_model_autodiscovery_selection.py
- scripts/model_tests/test_data_staleness_gate.py
- scripts/model_tests/test_occ_symbol_roundtrip.py
- scripts/model_tests/test_expiry_calendar_and_dte.py
- scripts/model_tests/test_multileg_fill_state_machine.py
- scripts/model_tests/test_kill_switch_triggers.py
- scripts/model_tests/test_checkpoint_version_compat.py

Required hard-fail rule:

- 05_output_distribution.py must implement collapse guards that fail the suite if outputs are constant / near-zero variance / identical rows / pathological action rate.

3.2 Audit tools (heavier, on-demand)

- audit/export_learned_conditions.py (attribution + surrogate rules)
- audit/generate_decision_factor_attribution.py
- audit/generate_audit_packet.py

3.3 Artifacts

- artifacts/learned_conditions/*
- artifacts/audit/*
- artifacts/audit/run_manifest.json
- artifacts/audit/contract_snapshot.json
- artifacts/audit/golden_fixtures/*

Mandatory for every release-candidate model:

- run_manifest.json (hashes + env)
- contract_snapshot.json (contract IDs + thresholds + gate precedence)
- decision_trace.jsonl + schema validation report
- decision_factor_attribution.csv

4) Workflow Integration

4.1 Training workflow (Definition of Done for a training run)

After each training run, the model is not considered "valid" until all steps pass:

- Export checkpoint metadata (feature_cols, input_dim, median/mad, version).
- Generate artifacts/audit/run_manifest.json (commit, env, dataset hash, checkpoint hash).
- Generate artifacts/audit/contract_snapshot.json (feature_schema_id, head mapping, thresholds, gate precedence).
- Run smoke suite: 01-07 + 04 + 05 (collapse guards).
- Run 08_live_trading_parity_sim.py on a fixed fixture slice.
- Run audit/export_learned_conditions.py and store outputs under artifacts/learned_conditions/.

4.2 Backtest workflow

Before any backtest run:

- 01_checkpoint_inventory.py
- 02_feature_schema_alignment.py
- 03_norm_stats_sanity.py
- 04_model_forward_contract.py
- 06_ruleset_execution_smoke.py
- test_time_alignment_and_session.py
- test_pnl_accounting_parity.py
- test_model_autodiscovery_selection.py
- 05_output_distribution.py (collapse guard must pass)

Run backtest with decision_trace.jsonl enabled.

4.3 Live workflow (demo/live readiness gate)

Before demo/live:

- Run all smoke tests (entire suite).
- Generate a run_manifest.json and attach to the session log.
- Confirm backtest/live output contract match (parity SIM).
- Confirm time/session alignment and execution accounting parity.
- Confirm data staleness gates enforce no-entry / exit-only behavior.
- Confirm kill switch triggers correctly under loss/anomaly/staleness conditions.
- Confirm multi-leg execution state machine handles partial fills safely.

5) Phased Execution Plan

Phase A - Smoke Test Harness (Immediate)

- Wire smoke tests to run locally before backtest.
- Gate PRs on smoke suite for any model-affecting change.
- Add collapse-guard thresholds to output distribution test (hard-fail).

Phase B - Audit Artifacts (Short-term)

- Standardize learned-conditions exports for each model build (always).
- Produce decision_trace.jsonl (ENTRY/EXIT/SIZING) for backtests and SIM.
- Produce decision_factor_attribution.csv scoped by ENTRY/EXIT/SIZING.
- Generate contract_snapshot.json for every RC model.
- [ ] **Contract Verification**: `scripts/verify_model_contract.py` must pass on any `./models/*.pth`.
- [ ] **Component Logging**: Future runs must log split losses (`L_diff`, `L_policy`, `L_feat`) to TensorBoard/Console.

## Phase 11: Audit & Accountability
- Generate audit packet PDF for release candidates: artifacts/audit/Accountability_Audit_Packet.pdf.

Phase C - Docs Contract (Short-term)

- Add docs/PIPELINE_CONTRACT.md + docs/PIPELINE_CONTRACT.json.
- Implement 12_docs_contract_sync_check.py to compare docs vs code-derived contract.
- CI fails if contract drift exists without updated docs in the same PR.

Phase D - Deployment Validation (Mid-term)

- Dual-run inference (backtest vs live interpreter) on the same slice.
- Confirm identical actions, gates, sizing, and event logs.
- Run demo account for >= 2 weeks with full decision tracing and daily review.
- If demo passes, promote to live with kill switch enabled and staleness gates enforced.

6) Risks & Mitigations

Risk: Feature drift from data changes.
Mitigation: schema hash checks + strict schema validation + registry-driven construction.

Risk: Output head misinterpretation.
Mitigation: named output contract + forward contract test + parity SIM.

Risk: Docs drift.
Mitigation: CI gate on contract diff.

Risk: Posterior collapse / constant outputs.
Mitigation: collapse guards + output distribution sanity checks + action-rate sanity.

Risk: Live feed staleness or timestamp skew.
Mitigation: staleness gates + time/session tests + exit-only mode.

Risk: Multi-leg execution partial fill risk.
Mitigation: deterministic state machine tests + kill switch + override logging.

Risk: Wrong model loaded.
Mitigation: auto-discovery selection test + checkpoint version compatibility.

7) Acceptance Criteria (Release Candidate Gate)

A model can be promoted to demo/live only if:

- Smoke suite passes on clean environment (CPU baseline).
- Checkpoint metadata is complete and consistent (feature_cols, input_dim, scaler stats, version, hashes).
- Feature schema parity holds across train/backtest/live; feature_schema_id matches everywhere.
- Forward output contract verified (keys, shapes, meanings).
- Backtest/live parity SIM passes with zero mismatches (or explicitly allow-listed mismatches).
- Time/session alignment passes (no off-by-one, DST/holiday safe).
- Execution/PnL accounting parity passes within tolerance.
- Staleness gate passes (stale -> no-entry/exit-only + logged).
- Kill switch tests pass (loss/anomaly/staleness halts).
- Multi-leg execution state machine passes (partial fills safe).
- Collapse guards pass (no constant outputs / nonsense action-rate).
- Docs contract matches code (CI green).
- Artifacts are generated:
  - run_manifest.json
  - contract_snapshot.json
  - decision_trace.jsonl + schema validation
  - decision_factor_attribution.csv
  - audit packet PDF for RC

CondorBrain Advanced Model Enhancements

(Keep your existing phase checklist exactly as-is; it is good. The only "must" is that Phase 11 items become gated by the Acceptance Criteria above.)

Repository Sync Addendum (2026-01-24)

(Keep as-is; it correctly defines doc precedence.)

Final "double-check" verdict

This plan now covers all critical grounds for:

- I/O correctness
- model creation correctness
- model utilization correctness (backtest + live real-time)
- execution + accounting parity
- governance / traceability readiness

If you want one last strengthening move: add a single line stating "All production decision mapping must live in one shared module imported by both backtest and live." That prevents duplication drift permanently.
