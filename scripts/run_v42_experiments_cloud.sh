#!/bin/bash
# ==============================================================================
# CONDORNET v4.2 — EMPIRICAL VALIDATION SUITE (CLOUD/T4)
# ==============================================================================
# This script runs 3 seeded training runs to verify:
# 1. Convergence stability (loss curves should mask)
# 2. A-matrix spectral properties (should be similar but distinct)
# 3. Predicate entropy (should not collapse)
#
# Usage:
#   ./scripts/run_v42_experiments_cloud.sh [path/to/data.csv]
#
# Default Data: data/Datasetv4/condornet_v41_FINAL.csv
# ==============================================================================

DATA_PATH=${1:-"data/Datasetv4/condornet_v41_FINAL.csv"}

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting CondorNet v4.2 Empirical Validation on T4 Hardware${NC}"
echo "Data Path: $DATA_PATH"
echo "Project Root: $(pwd)"

# Create output directory
mkdir -p models/v42_experiments

# Function to run training
run_training() {
    SEED=$1
    echo -e "\n${GREEN}>>> LAUNCHING SEED $SEED <<<${NC}"
    
    python intelligence/condor_train_net_v42.py \
        --local-data "$DATA_PATH" \
        --data-version v4.2 \
        --seed $SEED \
        --output "models/v42_experiments/condornet_v42_seed${SEED}.pth" \
        --batch-size 128 \
        --accum-steps 4 \
        --epochs 50 \
        --lookback 240 \
        --d-h 256 \
        --n-layers 2 \
        --n-predicates 512 \
        --n-sets 256 \
        --n-super-sets 128 \
        --lr 1e-4 \
        --checkpoint-every 5 \
        --save-diagnostics \
        --gui-telemetry lightai
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}>>> SEED $SEED COMPLETE <<<${NC}"
    else
        echo -e "${RED}>>> SEED $SEED FAILED <<<${NC}"
        exit 1
    fi
}

# Run 3 Independent Seeds
# Seed 42: Analysis Baseline
run_training 42

# Seed 101: Validation A
run_training 101

# Seed 999: Validation B
run_training 999

echo -e "\n${BLUE}================================================================${NC}"
echo -e "${GREEN}ALL EXPERIMENTS COMPLETED${NC}"
echo -e "${BLUE}================================================================${NC}"
echo "Outputs are in: models/v42_experiments/"
echo "To analyze results, run:"
echo "  python scripts/compare_eigen_spectra.py --dir models/v42_experiments"
