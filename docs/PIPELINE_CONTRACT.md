# Pipeline Contract (Canonical)

Version: 1.1
Date: 2026-01-26
Owner: Engineering / Quant Research
Status: Draft (Aligned to IMPLEMENTATION_PLAN_MODEL_AUDIT.md)

Purpose
Define the canonical contracts, gates, and artifacts that must hold across:
training -> serialization -> inference -> backtest -> live execution -> docs.

Contract Registry (IDs are stable)

C01 Feature Vector Contract
- Deterministic feature ordering and schema hash.
- Feature parity across training/backtest/live builders.

C02 Target/Label Contract
- Target horizons and thresholds match documentation.
- No lookahead leakage (train/val split integrity, no future bleed).

C03 Head Mapping Contract
- Output heads are versioned and keyed, not inferred by index.

C04 Checkpoint Contract
- Checkpoint includes feature_cols, input_dim, scaler stats, version, and run metadata.

C05 Interpretation Contract
- Backtest and live interpret outputs identically (thresholds, sizing, risk gates).

C06 Docs Contract
- Docs reflect code; CI fails on drift.

C07 Time & Session Contract
- Session open/close boundaries, DST shifts, holidays, and bar alignment rules are explicit.
- Options chain timestamps align with spot bars (explicit alignment policy).

C08 Execution & PnL Accounting Contract
- Slippage, fees, fills, multipliers, and rounding are identical across backtest/live
  (or explicitly parameterized and logged).

C09 Risk Gate Precedence Contract
- Gate order and override behavior are documented and enforced
  (including "block entry / allow exit-only" modes).

C10 Determinism & Reproducibility Contract
- Seeds, library versions, and inference determinism are captured in run manifests.

C11 Data Freshness & Staleness Contract
- Max tolerated staleness for spot bars, options chain, and greeks is enforced.
- Stale data forces no-entry / exit-only behavior and logs gate failures.

C12 Instrument & Contract Selection Contract
- OCC parsing, strike rounding, expiry selection, and DTE filters are explicit and tested.

C13 Multi-leg Execution Integrity Contract
- Partial fills, leg imbalance, and cancel/replace behavior are deterministic and safe.

C14 Safety Controls / Kill Switch Contract
- Daily loss limits, consecutive-loss limits, anomaly halts, and staleness halts are enforced.

C15 Backward Compatibility Contract
- Checkpoints declare compatibility with registry/head mapping/normalization versions
  and are rejected if incompatible.

Required Gates (Release Candidate)
- Contract hash parity (feature_schema_id) across train/backtest/live.
- Forward output contract verified (keys, shapes, meanings).
- Backtest/live parity SIM passes (no mismatches unless allow-listed).
- Time/session alignment passes (no off-by-one, DST/holiday safe).
- Execution/PnL accounting parity passes within tolerance.
- Staleness gate enforced (stale -> no-entry/exit-only + logged).
- Kill switch triggers correctly (loss/anomaly/staleness).
- Multi-leg execution state machine handles partial fills safely.
- Collapse guards pass (no constant outputs / pathological action-rate).
- Docs contract matches code (CI green).

Required Artifacts (Release Candidate)
- artifacts/audit/run_manifest.json
- artifacts/audit/contract_snapshot.json
- artifacts/learned_conditions/decision_trace.jsonl (+ schema validation report)
- artifacts/learned_conditions/decision_factor_attribution.csv
- artifacts/audit/Accountability_Audit_Packet.pdf

Test Suite Mapping (minimum)
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

Change Control
- This contract is authoritative. Any change to pipeline semantics must update:
  1) docs/PIPELINE_CONTRACT.md
  2) docs/PIPELINE_CONTRACT.json
  3) tests that enforce the change
