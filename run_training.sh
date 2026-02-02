#!/bin/bash

# Training Script for CondorBrain V2.2 (Lightning AI Optimized)
# This script ensures the command is not truncated and the data exists.

DATA_PATH="data/processed/mamba_institutional_2025_1m_v22.csv"
LOG_FILE="train_v22_3M.log"

echo "[START] Validating environment..."
if [ ! -f "$DATA_PATH" ]; then
    echo "[ERROR] Data file not found: $DATA_PATH"
    echo "Existing files in data/processed/:"
    ls -lh data/processed/
    exit 1
fi

echo "[OK] Found dataset."
echo "[RUN] Starting training (3M ROWS) in background..."

nohup python -u intelligence/train_condor_brain.py \
  --local-data "$DATA_PATH" \
  --output models/cb_cde_discovery_v22_3M.pth \
  --max-rows 3000000 \
  --use-predicate-discovery \
  --predicate-slots 2048 \
  --max-active-predicates 256 \
  --sparsity-weight 0.001 \
  --epochs 50 \
  --batch-size 64 \
  --d-model 256 \
  --layers 32 \
  --lr 1e-4 \
  --accum-steps 4 \
  --cde \
  --gpu-dataset \
  --grad-checkpoint \
  --no-plots \
  --batch-patience 10 \
  --early-stop \
  --patience 5 \
  --val-limit 200 \
  --tensorboard \
  --monitor \
  --save-on-batch-loss 1.0 \
  --export-epoch-plots \
  --export-dir training_exports \
  "$@" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "[DONE] Process started with PID: $PID"
echo "[INFO] Monitor logs with: tail -f $LOG_FILE"
