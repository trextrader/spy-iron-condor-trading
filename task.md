# CondorBrain Advanced Model Enhancements

## Phase 1: Indicator Modules
- [x] Create `intelligence/indicators/__init__.py`
- [x] Create `intelligence/indicators/manifold_volatility.py`
- [x] Create `intelligence/indicators/tda_signature.py`
- [x] Create `intelligence/indicators/policy_outputs.py`

## Phase 2: Core Enhancement Modules
- [x] Create `intelligence/condor_loss.py` (CompositeCondorLoss)
- [x] Create `intelligence/vol_gated_attn.py` (VolGatedAttn)
- [x] Create `intelligence/topk_moe.py` (TopKMoE)

## Phase 3: Architecture Integration
- [x] Modify `intelligence/condor_brain.py` to integrate new modules
- [x] Modify `intelligence/train_condor_brain.py` for composite loss

## Phase 4: Testing & Documentation
- [x] Create `tests/test_model_enhancements.py`
- [x] Update `docs/scientific_spec.md` with new mathematics (Section 13: Meta-Forecaster)
- [x] Update `CHANGELOG.md` with v2.2.0 release notes
- [x] Create `docs/architecture/enhanced_architecture.dot` diagram
- [x] Generate PNG from Graphviz diagram

## Phase 5: Training Verification
- [x] Training smoke test with composite loss (Colab A100, 2M rows)
- [x] Fix cuDNN GRU BF16 compatibility issues  
- [x] Fix TopKMoE `return_experts` null handling
- [x] Fix `sample_predictions` null experts handling
- [x] Create walkthrough document

## Phase 6: Inference & Production Integration
- [x] Audit repo for enhancement compatibility
- [x] Update `CondorBrainEngine` with enhancement flags
- [x] Add model auto-discovery (`_find_latest_model`)
- [x] Update `options_strategy.py` model detection
- [x] Complete 2-epoch training, saved 394MB model

## Phase 7: Backtesting Validation
- [x] Run backtest with 20-epoch CondorBrain model (Failed: Model Collapsed)
- [x] Retrain Production Model (Sequence Mode, 256-ctx, 3 Epochs)
- [x] Verify new model with Inference Demo script (Verified: Loaded Successfully Remote)
- [x] Verify strategy performance metrics (Pipeline Verified: Model outputs are collapsed/zeros)
- [ ] Generate equity curve and report

## Phase 8: Model Retraining (Fixing Collapse)
- [x] Create `kaggle/condor_brain_retrain_v2.py` (Higher LR, specific init)
- [x] Push to Repo
- [x] Instruction: Run on Kaggle
- [x] Monitor Training (Status: RESCUED | Epoch 1 Saved | Loss=0.15)
- [x] Download Model Artifacts (`condor_brain_retrain_e1.pth`)
- [x] Verify Inference (Local & Kaggle)
    - **Result**: Valid Signals (Offsets ~$1.98, Prob=50.6%) on Demo.
    - **Issue**: Backtest initially failed due to initialization bug.
- [x] Diagnose Normalization Stats (Result: Stats OK. Volume MAD ~1300 is expected).
- [x] Run Full Backtest (Result: **Technical Success**, **Scientific Failure**).
    - **Metrics**: Total Return: -8.15%, Sharpe: -2666, Signals: 999,745
    - **Diagnosis**: Posterior Collapse (Constant Output: Conf=0.584, all rows identical)
    - **Root Cause**: 1-epoch model is a "mean guesser" - insufficient training for sequence learning

## Phase 8.5: Path to Live (V2.1 Dynamic Feature Pipeline)
- [x] **Step 1A**: Create `intelligence/canonical_feature_registry.py`
- [x] **Step 1B**: Create `intelligence/features/dynamic_features.py`
- [x] **Step 2**: Update `kaggle/condor_brain_retrain_v2.py` (32 features)
- [x] **Step 3**: Update `kaggle/condor_brain_backtest.py` (schema-driven)
- [x] **Step 4**: Process 10M Row Dataset (Feature Engineering) (Mechanism Verified via on-the-fly & precompute)
- [/] **Step 5**: Retrain (5 Epochs, Staged Diffusion) (Iterating on visual feedback)
- [ ] **Step 6**: Verify & Paper Trade

## Phase 9: Advanced Model Enhancements (Completed)
- [x] **Diffusion Head**: Implement Generative Trajectory Refinement (`intelligence/generative/diffusion.py`)
- [x] **TDA Integration**: Verify `tda_signature.py` (Persistent Homology)
- [x] **Full Architecture**: Generate `docs/architecture/full_system_architecture.png`
- [x] **Documentation**: Update `README.md` and `scientific_spec.md`
- [x] **Diagram Audit**: Ensure all diagrams (including `optimization_pipeline`) are Dark Mode & V2.2 compliant
- [x] **LaTeX Fixes**: Repair broken math rendering in `README.md`

## Phase 10: Meta-Forecaster (Hybrid Ensemble)
- [x] Implement `MetaForecaster` class (Algorithm Spec: Methods 1-7, Epsilon Logic)
- [ ] Integrate Meta-Forecaster into Backtest Engine (PAUSED: Sync with Training)
- [x] Verify ensemble switching logic (Local Unit Test Passed)

## Phase 11: Model/Checkpoint/Docs Audit (Workflow Integration)
- [x] Add comprehensive test suite in `scripts/model_tests`
- [x] Add learned-conditions export pipeline (`audit/export_learned_conditions.py`)
- [ ] Add docs contract files: `docs/PIPELINE_CONTRACT.md` + `docs/PIPELINE_CONTRACT.json`
- [ ] Add schema validators in `audit/schema/*`
- [ ] Add CI gate to run smoke tests on PRs
- [ ] Add run manifest generation (`artifacts/audit/run_manifest.json`)
- [ ] Add audit packet generator (`artifacts/audit/Accountability_Audit_Packet.pdf`)
- [ ] Implement backtest/live output contract match test
- [ ] Integrate audit tools into training workflow
- [ ] Add time/session alignment contract + tests
- [ ] Add execution & PnL accounting parity tests
- [ ] Add risk gate precedence matrix + docs
- [ ] Add determinism/golden fixture bundle (`artifacts/audit/golden_fixtures`)
- [ ] Add collapse guards to output distribution tests
- [ ] Add model auto-discovery selection test
- [ ] Add data staleness gates + tests
- [ ] Add OCC/contract selection tests (expiry/DTE/strike rounding)
- [ ] Add multi-leg execution state machine tests
- [ ] Add kill switch safety controls + tests
- [ ] Add checkpoint compatibility/version tests
- [ ] Add data staleness gate contract + tests
- [ ] Add OCC symbol parsing + expiry calendar tests
- [ ] Add multi-leg execution state machine tests
- [ ] Add kill switch contract + tests
- [ ] Add checkpoint version compatibility test

## Training Log (v2.2 Production)
- **Epoch 1**:
    - Train: `L_pol=0.83`, `L_feat=574.2` (Converging).
    - Val: `L_pol=0.6877`, `L_feat=0.0277` (No Overfitting).
    - Saved: `condor_brain_seq_e1.pth` (394MB)
- [ ] Epoch 2
- [ ] Epoch 3
- [ ] Epoch 4 (Final)


---

## Repository Sync Addendum (2026-01-24)

This document is part of the synchronized documentation set. The authoritative engineering spec and audit references are:

- `docs/INTEGRATION_PLAN_MASTER.md`
- `docs/INTERFACE_CATALOG.md`

Key alignment requirements:
1. Feature schema selection by **name** (V2.2) only; no CSV order dependence.
2. Dataset column order differs across years; schema validation must be strict.
3. Model config metadata (layers/heads/input_dim) must match deployed checkpoints.

If this document conflicts with the master spec, the master spec governs implementation.
