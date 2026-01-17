# ============================================================
# CONDORBRAIN GPU BACKTEST - FINAL (FIXED + MULTI-GPU)
# ============================================================
print("🚀 Starting CondorBrain GPU Backtest (Fixed)...")

# --- IMPORTS ---
print("\n[1/6] Importing libraries...")
import sys
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
print("✅ Imports complete")

# --- GPU CHECK ---
print("\n[2/6] Checking GPU...")
device = torch.device('cuda')
print(f"   PyTorch: {torch.__version__}")
print(f"   GPU: {torch.cuda.get_device_name(0)}")
n_gpus = torch.cuda.device_count()
print(f"   GPUs available: {n_gpus}")

# --- LOAD 20-EPOCH MODEL ---
print("\n[3/6] Loading CondorBrain 20-epoch model...")
from intelligence.condor_brain import CondorBrain

model = CondorBrain(
    d_model=1024, n_layers=24, input_dim=24,
    use_vol_gated_attn=True, use_topk_moe=True,
    moe_n_experts=3, moe_k=1
).to(device)

if n_gpus > 1:
    print(f"   ⚡ Enabling DataParallel on {n_gpus} GPUs")
    model = nn.DataParallel(model)

MODEL_PATH = "/kaggle/input/condor-brain-weights-e20/condor_brain_e20_d1024_L24_lr1e04.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Handle DataParallel state dict prefix if needed, or normal load
if isinstance(model, nn.DataParallel):
    # If checkpoint wasn't saved with DataParallel but we use it now, or vice versa, handles it
    model.module.load_state_dict(checkpoint, strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# --- LOAD DATA ---
print("\n[4/6] Loading data...")
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
df = pd.read_csv(DATA_PATH)
print(f"   Rows: {len(df):,}")

FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 
                'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 
                'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 
                'psar', 'strike', 'target_spot', 'max_dd_60m']

X = df[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
median = np.median(X, axis=0, keepdims=True)
mad = np.median(np.abs(X - median), axis=0, keepdims=True) + 1e-8
X_norm = np.clip((X - median) / (1.4826 * mad), -10, 10)
print(f"✅ Features ready: {X_norm.shape}")

# --- GPU INFERENCE (OPTIMIZED + MULTI-GPU) ---
print("\n[5/6] Running Optimized GPU inference (500k samples)...")
SAMPLE_SIZE = 500_000
LOOKBACK = 240
BATCH_SIZE = 512  # Reduced to avoid OOM on T4

# Create tensor on GPU
print("   Creating GPU tensors...")
X_tensor = torch.tensor(X_norm[:SAMPLE_SIZE + LOOKBACK], device='cuda', dtype=torch.float32)

# Unfold trick + Transpose -> (Batch, Time, Features) for model
sequences_view = X_tensor.unfold(0, LOOKBACK, 1).transpose(1, 2)

all_preds = []

with torch.no_grad():
    for start in tqdm(range(0, SAMPLE_SIZE, BATCH_SIZE), desc="Inference"):
        end = min(start + BATCH_SIZE, SAMPLE_SIZE)
        
        # Slicing the view is fast
        batch_seqs = sequences_view[start:end].contiguous()
        
        # DataParallel handles splitting batch to GPUs automatically
        outputs, _, _ = model(batch_seqs)
        all_preds.append(outputs.cpu().numpy())

predictions = np.concatenate(all_preds)
print(f"✅ Predictions: {predictions.shape}")

# --- RESULTS ---
print("\n[6/6] Analyzing results...")
OUTPUT_COLS = ['call_offset', 'put_offset', 'wing_width', 'dte', 
               'prob_profit', 'expected_roi', 'max_loss', 'confidence']

pred_df = pd.DataFrame(predictions, columns=OUTPUT_COLS)
print(pred_df.describe())

print("\n🎉 BACKTEST COMPLETE!")
print(f"   Mean confidence: {pred_df['confidence'].mean():.4f}")
print(f"   Std confidence: {pred_df['confidence'].std():.4f}")
print(f"   High confidence signals (>0.7): {(pred_df['confidence'] > 0.7).sum():,}")
print(f"   Low confidence signals (<0.3): {(pred_df['confidence'] < 0.3).sum():,}")
