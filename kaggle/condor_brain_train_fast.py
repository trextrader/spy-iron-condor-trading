# ============================================================
# CONDORBRAIN TRAINING - FAST VERSION (Single GPU, High Speed)
# ============================================================
print("🚀 Starting CondorBrain Training (FAST MODE)...")

# --- 0. PREP ---
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
from intelligence.condor_brain import CondorBrain

# --- CONFIG (Speed Optimized) ---
EPOCHS = 5
BATCH_SIZE = 4096   # Massive batch for T4 (since Seq Length=1)
LR = 5e-4          # Higher LR for larger batch
ROWS_TO_LOAD = 2_000_000  # 2M Rows (Enough for solid convergence)

device = torch.device('cuda')
print(f"   GPU: {torch.cuda.get_device_name(0)}")

# --- 1. DATA ---
print(f"\n[1/4] Loading {ROWS_TO_LOAD:,} Rows...")
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
df = pd.read_csv(DATA_PATH).iloc[-ROWS_TO_LOAD:]

FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 
                'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 
                'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 
                'psar', 'strike', 'target_spot', 'max_dd_60m']

X = df[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

median = np.median(X, axis=0, keepdims=True)
mad = np.median(np.abs(X - median), axis=0, keepdims=True) + 1e-8
X_norm = np.clip((X - median) / (1.4826 * mad), -10, 10)

# Targets (Synthetic/Proxy)
print("   Generating targets...")
returns_60m = np.roll(X[:, 3], -60) - X[:, 3]
y_target = np.zeros((len(X), 8), dtype=np.float32)
y_target[:, 0] = 2.0 + np.clip(returns_60m * 0.5, -1, 1)
y_target[:, 1] = 2.0 - np.clip(returns_60m * 0.5, -1, 1)
y_target[:, 2] = 5.0 + np.clip(np.abs(returns_60m) * 20, 0, 5)
y_target[:, 4] = 0.5 + np.clip(returns_60m * 0.1, -0.4, 0.4)
y_target[:, 7] = 0.3 + np.clip(np.abs(returns_60m), 0, 0.6)

# Split
split_idx = int(len(X) * 0.9)
X_train = torch.tensor(X_norm[:split_idx], dtype=torch.float32)
y_train = torch.tensor(y_target[:split_idx], dtype=torch.float32)
X_val = torch.tensor(X_norm[split_idx:], dtype=torch.float32)
y_val = torch.tensor(y_target[split_idx:], dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# --- 2. MODEL (Efficient) ---
print("\n[2/4] Initializing CondorBrain (Efficient Config)...")
model = CondorBrain(
    d_model=512,       # Good balance for T4
    n_layers=12,       # Sufficient depth
    input_dim=24,
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')

# --- 3. TRAIN ---
print(f"\n[3/4] Training for {EPOCHS} Epochs...")
for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
    
    for data, target in pbar:
        data_seq = data.unsqueeze(1).to(device) # [B, 1, F]
        target = target.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(data_seq)
            if isinstance(out, tuple): out = out[0]
            loss = criterion(out, target)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data_seq = data.unsqueeze(1).to(device)
            target = target.to(device)
            with torch.amp.autocast('cuda'):
                out = model(data_seq)
                if isinstance(out, tuple): out = out[0]
                val_loss += criterion(out, target).item()
    
    avg_val = val_loss / len(val_loader)
    print(f"   Epoch {epoch+1} Val Loss: {avg_val:.6f}")
    
    # Save
    torch.save(model.state_dict(), f"condor_brain_fast_e{epoch+1}.pth")

print("✅ DONE.")
