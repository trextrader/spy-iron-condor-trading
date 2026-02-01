# ⚡ Lightning AI / Cloud Processing Instructions

Follow these steps to run the modern **CondorBrain v2.2 (Neural CDE + Predicate Discovery)** on Lightning AI.

## 1. Fast-Track Training (3M+ Rows)

We use a specialized "Fast-Track" training loop designed for large datasets on T4/A10G GPUs. This system uses:
1.  **Batch-Level Checkpointing ("Ratchet")**: Saves the model *immediately* when batch loss drops below 1.0.
2.  **Short-Circuit Epochs**: If a new best model is found, the epoch terminates early to validate.
3.  **Validation Limits**: Checks only 200 batches (~10%) to provide rapid feedback.

**Command (Recommended):**
```bash
# 1. Ensure you have the latest code
git pull

# 2. Run the optimization script (Background Mode)
bash run_training.sh
```

**What `run_training.sh` does:**
*   Loads `data/processed/mamba_institutional_2025_1m_v22.csv` (3M rows).
*   Sets `--save-on-batch-loss 1.0` (The Magic Barrier).
*   Sets `--val-limit 200` to speed up validation.
*   Logs to `train_v22_3M.log`.

**Monitoring:**
```bash
# Watch the training in real-time
tail -f train_v22_3M.log
```

**Look for:**
*   `[SAVE] New Best Batch Loss: 0.99xx`
*   `⚡ [FAST-TRACK] Breaking epoch early...`

## 2. Manual Training (Custom Flags)

If you want to run manually without the script:

```bash
python intelligence/train_condor_brain.py \
    --local-data "data/processed/mamba_institutional_2025_1m_v22.csv" \
    --output models/cb_manual_v22.pth \
    --max-rows 3000000 \
    --cde \
    --use-predicate-discovery \
    --predicate-slots 2048 \
    --epochs 20 \
    --batch-size 128 \
    --accum-steps 2 \
    --save-on-batch-loss 1.0 \
    --val-limit 200
```

## 3. Verification (Backtest)

Once training produces a checkpoint (e.g., `models/batch_checkpoints/batch_loss_0.9992_e1_b72.pth`), verify it:

```bash
python kaggle/condor_brain_backtest_v2.py \
    --model "models/batch_checkpoints/batch_loss_0.9992_e1_b72.pth" \
    --input "data/processed/mamba_institutional_2025_1m_v22.csv" \
    --limit 10000
```
