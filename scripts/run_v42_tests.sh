#!/usr/bin/env bash
set -e

# ============================================================
#  CondorNet v4.2 — Full System Verification Test Runner
#  Auto-path-aware, verbose, stops on first failure
# ============================================================

# Resolve absolute path to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Test file lives in the same directory
TEST_FILE="${SCRIPT_DIR}/test_condornet_v42.py"

# Repo root is one level above /scripts
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure Python can import intelligence/*
export PYTHONPATH="${REPO_ROOT}"

echo ""
echo "============================================================"
echo "   CONDORNET v4.2 — FULL SYSTEM VERIFICATION SUITE"
echo "   Starting test run at: $(date)"
echo "   Script directory: ${SCRIPT_DIR}"
echo "   Repo root:        ${REPO_ROOT}"
echo "   Test file:        ${TEST_FILE}"
echo "============================================================"
echo ""

# Check pytest exists
if ! command -v pytest &> /dev/null
then
    echo "ERROR: pytest not found. Install with: pip install pytest pytest-xdist"
    exit 1
fi

# ------------------------------------------------------------
# 1. Dataset Integrity Tests
# ------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  [1/8] DATASET INTEGRITY TESTS"
echo "------------------------------------------------------------"
pytest -vv -s -x "${TEST_FILE}::test_feature_registry_matches_dataset_columns_v41"
pytest -vv -s -x "${TEST_FILE}::test_dataset_v41_basic_sanity"

# ------------------------------------------------------------
# 2. Model Dimension & Flow Tests
# ------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  [2/8] MODEL DIMENSION & FLOW TESTS"
echo "------------------------------------------------------------"
pytest -vv -s -x "${TEST_FILE}::test_model_input_dim_matches_registry"
pytest -vv -s -x "${TEST_FILE}::test_state_dim_matches_input_plus_latent"
pytest -vv -s -x "${TEST_FILE}::test_A_spectral_radius_at_init"

# ------------------------------------------------------------
# 3. Predicate → Set → Superset Tests
# ------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  [3/8] PREDICATE / SET / SUPERSET TESTS"
echo "------------------------------------------------------------"
pytest -vv -s -x "${TEST_FILE}::test_predicate_layer_not_saturated_and_has_entropy"
pytest -vv -s -x "${TEST_FILE}::test_sets_and_supersets_exist_and_have_diversity"

# ------------------------------------------------------------
# 4. Training Pipeline Tests
# ------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  [4/8] TRAINING PIPELINE TESTS"
echo "------------------------------------------------------------"
pytest -vv -s -x "${TEST_FILE}::test_dataloader_and_single_batch_forward"
pytest -vv -s -x "${TEST_FILE}::test_no_nan_after_full_forward_with_noise"

# ------------------------------------------------------------
# 5. Pivot System Tests (v4.2)
# ------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  [5/8] PIVOT SYSTEM TESTS (v4.2)"
echo "------------------------------------------------------------"
pytest -vv -s -x "${TEST_FILE}::test_pivot_columns_present_in_v42_if_dataset_exists"
pytest -vv -s -x "${TEST_FILE}::test_pivot_encoder_and_heads_exist_if_configured"
pytest -vv -s -x "${TEST_FILE}::test_mtf_pivot_consensus_logic_exposed_or_computable"

# ------------------------------------------------------------
# 6. Architecture Routing Tests (TFT / CDE / ETD)
# ------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  [6/8] ARCHITECTURE ROUTING TESTS"
echo "------------------------------------------------------------"
pytest -vv -s -x "${TEST_FILE}::test_architecture_exposes_core_modules"

# ------------------------------------------------------------
# 7. High-Level Safety Tests
# ------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  [7/8] HIGH-LEVEL SAFETY TESTS"
echo "------------------------------------------------------------"
pytest -vv -s -x "${TEST_FILE}::test_entropy_and_diversity_do_not_collapse_early"

# ------------------------------------------------------------
# 8. v4.2 Dataset Upgrade Tests (Optional)
# ------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "  [8/8] DATASET UPGRADE TESTS (v4.2)"
echo "------------------------------------------------------------"
pytest -vv -s -x "${TEST_FILE}::test_v42_dataset_if_present_is_superset_of_v41"

echo ""
echo "============================================================"
echo "   ALL TESTS COMPLETED SUCCESSFULLY"
echo "   Finished at: $(date)"
echo "============================================================"
echo ""
echo "   Finished at: $(date)"
echo "============================================================"
echo ""
