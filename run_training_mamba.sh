#!/bin/bash

# Training Script for CondorBrain V2.2 (Mamba-2 Architecture)
# Comparison Run: Identical settings to CDE run, but with --no-cde

DATA_PATH="/kaggle/working/spy-iron-condor-trading/data/mamba_institutional_2025_1m_v22.csv"
LOG_FILE="train_mamba2_v22.log"

echo "[START] Validating environment..."
if [ ! -f "$DATA_PATH" ]; then
    echo "[ERROR] Data file not found: $DATA_PATH"
    echo "Existing files in data/processed/:"
    ls -lh data/processed/
    exit 1
fi

echo "[OK] Found dataset."

# Create output directories explicitly
echo "[SETUP] Creating output directories..."
mkdir -p models
mkdir -p training_exports_mamba

echo "[RUN] Starting MAMBA-2 training (3M ROWS) in background..."

nohup python -u intelligence/train_condor_brain.py \
  --local-data "$DATA_PATH" \
  --output models/cb_mamba2_v22_3M.pth \
  --max-rows 3000000 \
  --use-predicate-discovery \
  --predicate-slots 2048 \
  --max-active-predicates 256 \
  --sparsity-weight 0.001 \
  --epochs 50 \
  --batch-size 128 \
  --d-model 256 \
  --layers 32 \
  --lr 1e-4 \
  --accum-steps 2 \
  --no-cde \
  --gpu-dataset \
  --grad-checkpoint \
  --no-plots \
  --early-stop \
  --patience 5 \
  --val-limit 200 \
  --tensorboard \
  --monitor \
  --monitor-every 1 \
  --save-on-batch-loss 1.0 \
  --export-epoch-plots \
  --export-dir training_exports_mamba \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "[DONE] Process started with PID: $PID"
echo "[INFO] Monitor logs with: tail -f $LOG_FILE"
