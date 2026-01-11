# A100 GPU Training Sweep Strategy

## Objective
Systematically explore hyperparameter space on A100 GPU to find optimal CondorBrain configuration.

---

## Sweep Matrix

| Sweep | Epochs | Batch | d_model | Layers | LR | Est. Time | Purpose |
|-------|--------|-------|---------|--------|-----|-----------|---------|
| **1** | 50 | 256 | 512 | 16 | 1e-4 | ~1hr | Baseline viability check |
| **2** | 50 | 256 | 1024 | 32 | 1e-4 | ~2hr | Scale test |
| **3** | 100 | 512 | 1024 | 32 | 5e-5 | ~3hr | Production candidate |
| **4** | 100 | 512 | 1024 | 48 | 5e-5 | ~4hr | Deep exploration |
| **5** | 200 | 256 | 1024 | 32 | 1e-5 | ~6hr | Fine-tuning champion |

---

## Colab Commands

### Sweep 1: Quick Baseline
```bash
!python intelligence/train_condor_brain.py \
    --local-data data/processed/mamba_institutional_1m_targets.csv \
    --epochs 50 --batch-size 256 --d-model 512 --layers 16 --lr 1e-4
!cp models/condor_brain.pth models/condor_brain_sweep1.pth
```

### Sweep 2: Scale Up
```bash
!python intelligence/train_condor_brain.py \
    --local-data data/processed/mamba_institutional_1m_targets.csv \
    --epochs 50 --batch-size 256 --d-model 1024 --layers 32 --lr 1e-4
!cp models/condor_brain.pth models/condor_brain_sweep2.pth
```

### Sweep 3: Production
```bash
!python intelligence/train_condor_brain.py \
    --local-data data/processed/mamba_institutional_1m_targets.csv \
    --epochs 100 --batch-size 512 --d-model 1024 --layers 32 --lr 5e-5
!cp models/condor_brain.pth models/condor_brain_sweep3.pth
```

### Sweep 4: Deep Layers
```bash
!python intelligence/train_condor_brain.py \
    --local-data data/processed/mamba_institutional_1m_targets.csv \
    --epochs 100 --batch-size 512 --d-model 1024 --layers 48 --lr 5e-5
!cp models/condor_brain.pth models/condor_brain_sweep4.pth
```

### Sweep 5: Extended Fine-Tune
```bash
!python intelligence/train_condor_brain.py \
    --local-data data/processed/mamba_institutional_1m_targets.csv \
    --epochs 200 --batch-size 256 --d-model 1024 --layers 32 --lr 1e-5
!cp models/condor_brain.pth models/condor_brain_sweep5.pth
```

---

## Evaluation Metrics

After each sweep, evaluate:
1. **Val Loss** (lower = better)
2. **Strike Accuracy** (L1 on offset predictions)
3. **Probability Calibration** (Brier score)
4. **Regime Classification** (accuracy on IVR bins)

---

## A100 Optimization Tips

```python
# Memory optimization (add to training script)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Gradient checkpointing for deep models (48+ layers)
model.gradient_checkpointing_enable()
```
