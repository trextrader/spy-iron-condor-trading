# ============================================================
# CONDORBRAIN RESCUE SCRIPT (Installs + Fast Train [1M Rows])
# ============================================================
print("🚀 STARTING RESCUE SEQUENCE (1M Rows)...")

# --- CELL 1: SETUP (Run this first) ---
import os
print("\n[1/2] Installing Dependencies & Syncing Repo...")
os.system("pip install causal-conv1d>=1.2.0 mamba-ssm")
os.system("pip install pandas-ta")

if not os.path.exists("spy-iron-condor-trading"):
    os.system("git clone https://github.com/YOUR_REPO/spy-iron-condor-trading.git") # Update URL if needed or use existing folder

!cd spy-iron-condor-trading && git fetch origin && git reset --hard origin/main
print("✅ Environment Ready")

# --- CELL 2: TRAINING ---
import sys
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from intelligence.condor_brain import CondorBrain

print("\n[2/2] Training Model (Safe Mode)...")

# --- FAST CONFIG ---
EPOCHS = 1          # Just get ONE saved epoch to be safe
BATCH_SIZE = 4096   
LR = 1e-4           # STABLE Rate
ROWS_TO_LOAD = 1_000_000  # <--- REDUCED for ~45 min run
LOOKBACK = 240

device = torch.device('cuda')
print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Load Data
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
df = pd.read_csv(DATA_PATH).iloc[-ROWS_TO_LOAD:]
print(f"   Loaded: {len(df):,} rows")

# Features & targets (Same as before)
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'strike', 'target_spot', 'max_dd_60m']
X = df[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
median = np.median(X, axis=0, keepdims=True); mad = np.median(np.abs(X - median), axis=0, keepdims=True) + 1e-8
X_norm = np.clip((X - median) / (1.4826 * mad), -10, 10)

returns_60m = np.roll(X[:, 3], -60) - X[:, 3]
y_target = np.zeros((len(X), 8), dtype=np.float32)
y_target[:, 0] = 2.0 + np.clip(returns_60m * 0.5, -1, 1); y_target[:, 1] = 2.0 - np.clip(returns_60m * 0.5, -1, 1)
y_target[:, 2] = 5.0 + np.clip(np.abs(returns_60m) * 20, 0, 5); y_target[:, 4] = 0.5 + np.clip(returns_60m * 0.1, -0.4, 0.4)
y_target[:, 7] = 0.3 + np.clip(np.abs(returns_60m), 0, 0.6)

# Train
split_idx = int(len(X) * 0.9)
train_loader = DataLoader(TensorDataset(torch.tensor(X_norm[:split_idx]), torch.tensor(y_target[:split_idx])), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_norm[split_idx:]), torch.tensor(y_target[split_idx:])), batch_size=BATCH_SIZE, shuffle=False)

model = CondorBrain(d_model=512, n_layers=12, input_dim=24, use_vol_gated_attn=True, use_topk_moe=True, moe_n_experts=3, moe_k=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss(); scaler = torch.amp.GradScaler('cuda')

# Loop
model.train()
pbar = tqdm(train_loader, desc=f"Ep 1/{EPOCHS}")
for data, target in pbar:
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        out = model(data.unsqueeze(1).to(device))
        loss = criterion(out[0] if isinstance(out, tuple) else out, target.to(device))
    scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update()
    pbar.set_postfix({'loss': loss.item()})

torch.save(model.state_dict(), "condor_brain_fast_e1.pth")
print("\n✅ SAVED: condor_brain_fast_e1.pth - DOWNLOAD NOW!")
