# ============================================================
# CONDORBRAIN PRODUCTION TRAINING - 5M ROWS (2 GPUs)
# ============================================================
print("🚀 Starting CondorBrain Production Training (5M Rows, 2 GPUs)...")

# --- 0. PREP & CLEAN ---
!cd spy-iron-condor-trading && git fetch origin && git reset --hard origin/main
print("✅ Repo synced")

import sys
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from intelligence.condor_brain import CondorBrain

# --- CONFIG ---
EPOCHS = 5
BATCH_SIZE = 1024   # Aggressive batch size for 2 GPUs
LR = 3e-4          # Slightly higher LR for large batch
ROWS_TO_LOAD = 5_000_000
LOOKBACK = 240

device = torch.device('cuda')
n_gpus = torch.cuda.device_count()
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   Count: {n_gpus}")

# --- 1. DATA LOADING & PREP ---
print(f"\n[1/4] Loading & Processing {ROWS_TO_LOAD:,} Rows...")
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"

# Load tail of dataset (most recent 5M rows)
df = pd.read_csv(DATA_PATH).iloc[-ROWS_TO_LOAD:]
print(f"   Shape: {df.shape}")

# Features
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 
                'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 
                'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 
                'psar', 'strike', 'target_spot', 'max_dd_60m']

X = df[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

# Robust Scaling
median = np.median(X, axis=0, keepdims=True)
mad = np.median(np.abs(X - median), axis=0, keepdims=True) + 1e-8
X_norm = np.clip((X - median) / (1.4826 * mad), -10, 10)

# --- REALISTIC TARGET GENERATION (Approximation) ---
# Since we don't have labeled 'best condor' columns, we approximate 
# targets based on future price action to learn directional bias & volatility.
# Target 0: Call Offset  (Higher if Bullish)
# Target 1: Put Offset   (Higher if Bearish)
# Target 2: Wing Width   (Higher if Volatile)
# ...
print("   Generating semi-realistic targets...")
returns_60m = np.roll(X[:, 3], -60) - X[:, 3]
vol_60m = pd.Series(returns_60m).rolling(60).std().fillna(0).values

y_target = np.zeros((len(X), 8), dtype=np.float32)

# Vectorized Target Logic
# Call Offset: 2.0 base + bullish shift
y_target[:, 0] = 2.0 + np.clip(returns_60m * 0.5, -1, 1)
# Put Offset: 2.0 base - bearish shift
y_target[:, 1] = 2.0 - np.clip(returns_60m * 0.5, -1, 1)
# Width: 5.0 base + volatility expansion
y_target[:, 2] = 5.0 + np.clip(vol_60m * 20, 0, 5)
# Prob Profit: 0.5 + momentum
y_target[:, 4] = 0.5 + np.clip(returns_60m * 0.1, -0.4, 0.4)
# Confidence: Higher on clear trends
y_target[:, 7] = 0.3 + np.clip(np.abs(returns_60m), 0, 0.6)

# Time Series Split (Last 10% for validation)
split_idx = int(len(X) * 0.9)
X_train = torch.tensor(X_norm[:split_idx], dtype=torch.float32)
y_train = torch.tensor(y_target[:split_idx], dtype=torch.float32)
X_val = torch.tensor(X_norm[split_idx:], dtype=torch.float32)
y_val = torch.tensor(y_target[split_idx:], dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# --- 2. MODEL SETUP ---
print("\n[2/4] Initializing CondorBrain (Production Config)...")
model = CondorBrain(
    d_model=1024,      # Full size
    n_layers=24,       # Deep network
    input_dim=24,
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1
).to(device)

if n_gpus > 1:
    print(f"   ⚡ Enabling DataParallel on {n_gpus} GPUs")
    model = nn.DataParallel(model)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
criterion = nn.MSELoss()

# --- 3. TRAINING LOOP ---
print("\n[3/4] Training Loop...")
scaler = torch.amp.GradScaler('cuda')
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Mamba requires Sequence Dim: [B, 1, F]
        data_seq = data.unsqueeze(1).to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # DataParallel might return tuple/tensor depending on version/wrap
            outputs = model(data_seq)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data_seq = data.unsqueeze(1).to(device)
            target = target.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(data_seq)
                if isinstance(outputs, tuple): outputs = outputs[0]
                val_loss += criterion(outputs, target).item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"   Epoch {epoch+1}: Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f}")
    
    # Checkpoint
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        save_path = f"condor_brain_e{epoch+1}_loss{avg_val_loss:.4f}.pth"
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        print(f"      💾 Saved Best Model: {save_path}")

print("✅ Training Complete.")
