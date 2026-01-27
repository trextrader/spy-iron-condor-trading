# ============================================================
# CONDORBRAIN GPU BACKTEST - LEGACY 20-EPOCH (No Enhancements)
# ============================================================
print("🚀 Starting CondorBrain GPU Backtest (Legacy Mode)...")

# --- 0. FORCE GIT CLEAN ---
print("\n[0/6] Cleaning repository...")
!cd spy-iron-condor-trading && git fetch origin && git reset --hard origin/main
print("✅ Repo updated")

# --- 1. IMPORTS ---
print("\n[1/6] Importing libraries...")
import sys
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from audit.contract_snapshot import generate_contract_snapshot
print("✅ Imports complete")

# --- 2. CONFIG & GPU ---
print("\n[2/6] Setup...")
SAMPLE_SIZE = 20_000
LOOKBACK = 240
BATCH_SIZE = 1024

device = torch.device('cuda')
n_gpus = torch.cuda.device_count()
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   Count: {n_gpus}")

# --- 3. LOAD MODEL (LEGACY MODE) ---
print("\n[3/6] Loading CondorBrain 20-epoch model (Legacy Config)...")
from intelligence.condor_brain import CondorBrain

# DISABLE New Features for Old Model
model = CondorBrain(
    d_model=1024, n_layers=24, input_dim=24,
    use_vol_gated_attn=False,  # <--- DISABLED
    use_topk_moe=False,        # <--- DISABLED
    moe_n_experts=3, moe_k=1
).to(device)

if n_gpus > 1:
    print(f"   ⚡ Enabling DataParallel on {n_gpus} GPUs")
    model = nn.DataParallel(model)

MODEL_PATH = "/kaggle/input/condor-brain-weights-e20/condor_brain_e20_d1024_L24_lr1e04.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(checkpoint, strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
print(f"✅ Model loaded")
generate_contract_snapshot(
    os.path.join(os.getcwd(), "artifacts", "audit", "contract_snapshot.json"),
    os.getcwd(),
    checkpoint_path=MODEL_PATH,
    extra={"mode": "backtest_legacy"},
)

# --- 4. LOAD DATA ---
print("\n[4/6] Loading data...")
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
df = pd.read_csv(DATA_PATH)
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

# --- 5. INFERENCE ---
print("\n[5/6] Running Optimized GPU inference...")
X_tensor = torch.tensor(X_norm[:SAMPLE_SIZE + LOOKBACK], device='cuda', dtype=torch.float32)
sequences_view = X_tensor.unfold(0, LOOKBACK, 1).transpose(1, 2)

all_preds = []

with torch.no_grad():
    for start in tqdm(range(0, SAMPLE_SIZE, BATCH_SIZE), desc="Inference"):
        end = min(start + BATCH_SIZE, SAMPLE_SIZE)
        batch_seqs = sequences_view[start:end].contiguous()
        with torch.amp.autocast('cuda'):
            outputs, _, _ = model(batch_seqs) # Legacy mode: likely returns just outputs
            # Newer code might return 3 vals even if features disabled, 
            # but let's handle tuple return just in case
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        
        all_preds.append(outputs.float().cpu().numpy())

predictions = np.concatenate(all_preds)
print(f"✅ Predictions: {predictions.shape}")

# --- 6. RESULTS ---
print("\n[6/6] Analyzing results...")
OUTPUT_COLS = ['call_offset', 'put_offset', 'wing_width', 'dte', 
               'prob_profit', 'expected_roi', 'max_loss', 'confidence']

pred_df = pd.DataFrame(predictions, columns=OUTPUT_COLS)
print("\n📊 STATISTICS:")
print(pred_df.describe())

print("\n🎉 PERFORMANCE METRICS:")
print(f"   Mean confidence: {pred_df['confidence'].mean():.4f}")
print(f"   Std confidence: {pred_df['confidence'].std():.4f}")
