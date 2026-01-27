#!/bin/bash
# =============================================================================
# CondorBrain Training Script for Lightning AI (T4 GPU)
# 4 Experiments: 2 without diffusion, 2 with diffusion
# Using 100K rows from mamba_institutional_2024_1m_last 1mil.csv
# =============================================================================

# Configuration
DATA_PATH="data/processed/mamba_institutional_2024_1m_last 1mil.csv"
MAX_ROWS=100000
EPOCHS=50
BATCH_SIZE=128  # T4 has 16GB VRAM - conservative batch size
D_MODEL=512     # Smaller model for T4
LAYERS=16       # Fewer layers for T4
LR=1e-4
LOOKBACK=240
FEATURE_GROUP_DROPOUT=0.15  # Anti-collapse regularization

# Common flags
COMMON_FLAGS="--local-data \"${DATA_PATH}\" \
    --max-rows ${MAX_ROWS} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --d-model ${D_MODEL} \
    --layers ${LAYERS} \
    --lr ${LR} \
    --lookback ${LOOKBACK} \
    --feature-group-dropout ${FEATURE_GROUP_DROPOUT} \
    --vol-gated-attn \
    --topk-moe \
    --composite-loss \
    --monitor \
    --early-stop \
    --patience 10 \
    --val-every 2 \
    --tensorboard"

echo "=============================================="
echo "CondorBrain Training - Lightning AI T4 GPU"
echo "Data: ${DATA_PATH}"
echo "Max Rows: ${MAX_ROWS}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Model: d_model=${D_MODEL}, layers=${LAYERS}"
echo "Feature Group Dropout: ${FEATURE_GROUP_DROPOUT}"
echo "=============================================="

# =============================================================================
# RUN 1: No Diffusion, Seed 42
# =============================================================================
echo ""
echo "=== RUN 1/4: No Diffusion (Seed 42) ==="
python intelligence/train_condor_brain.py ${COMMON_FLAGS} \
    --output "models/condor_run1_nodiff_s42.pth" \
    --tb-logdir "runs/run1_nodiff_s42" \
    2>&1 | tee logs/run1_nodiff_s42.log

# =============================================================================
# RUN 2: No Diffusion, Seed 123
# =============================================================================
echo ""
echo "=== RUN 2/4: No Diffusion (Seed 123) ==="
python intelligence/train_condor_brain.py ${COMMON_FLAGS} \
    --output "models/condor_run2_nodiff_s123.pth" \
    --tb-logdir "runs/run2_nodiff_s123" \
    2>&1 | tee logs/run2_nodiff_s123.log

# =============================================================================
# RUN 3: With Diffusion, Seed 42
# =============================================================================
echo ""
echo "=== RUN 3/4: WITH Diffusion (Seed 42) ==="
python intelligence/train_condor_brain.py ${COMMON_FLAGS} \
    --diffusion \
    --diffusion-steps 50 \
    --output "models/condor_run3_diff_s42.pth" \
    --tb-logdir "runs/run3_diff_s42" \
    2>&1 | tee logs/run3_diff_s42.log

# =============================================================================
# RUN 4: With Diffusion, Seed 123
# =============================================================================
echo ""
echo "=== RUN 4/4: WITH Diffusion (Seed 123) ==="
python intelligence/train_condor_brain.py ${COMMON_FLAGS} \
    --diffusion \
    --diffusion-steps 50 \
    --output "models/condor_run4_diff_s123.pth" \
    --tb-logdir "runs/run4_diff_s123" \
    2>&1 | tee logs/run4_diff_s123.log

echo ""
echo "=============================================="
echo "All 4 runs complete!"
echo "Models saved to: models/"
echo "TensorBoard logs: runs/"
echo "Training logs: logs/"
echo "=============================================="
echo ""
echo "To view results:"
echo "  tensorboard --logdir=runs --port=6006"
echo ""
echo "To export learned conditions from best model:"
echo "  python audit/export_learned_conditions.py --model-path <best_model.pth>"
echo "=============================================="
