# ============================================================
# CONDORBRAIN GPU TRAINING - FRESH RUN
# ============================================================
print("🚀 Starting CondorBrain Training on Kaggle (GPU)...")

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
from intelligence.condor_brain import CondorBrain

# --- CONFIG ---
EPOCHS = 2          # Quick proof-of-concept (increase to 20 later)
BATCH_SIZE = 512    # Safe for T4
LOOKBACK = 240
LR = 1e-4

device = torch.device('cuda')
print(f"   GPU: {torch.cuda.get_device_name(0)}")

# --- 1. DATA LOADING & PREP ---
print("\n[1/4] Loading & Processing Data...")
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
df = pd.read_csv(DATA_PATH).iloc[-500_000:] # Train on last 500k rows for speed
print(f"   Training on: {len(df):,} rows")

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

# Targets (Synthetic for demo - in real training use actual future labels)
# For now, we will create dummy targets just to verify the MODEL LEARNS
# In a real run, you MUST have target columns (e.g. 'future_profit', 'best_wing', etc.)
# We will simulate targets based on simple future returns for this test
print("   Generating synthetic targets for training test...")
future_returns = np.roll(X[:, 3], -60) - X[:, 3] # 60m future return
y_target = np.zeros((len(X), 8), dtype=np.float32)
y_target[:, 0] = 2.0  # Call Offset
y_target[:, 1] = 2.0  # Put Offset
y_target[:, 2] = 5.0  # Width
y_target[:, 4] = (future_returns > 0).astype(np.float32) # Prob Profit proxy
y_target[:, 7] = 0.5  # Confidence

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y_target, dtype=torch.float32)

# Create Sequences (Naive loop for simplicity in demo, Optimized unfold is better but memory hungry for training)
# effectively slicing
train_data = TensorDataset(X_tensor[:-LOOKBACK], y_tensor[LOOKBACK:]) 
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# --- 2. MODEL SETUP ---
print("\n[2/4] Initializing CondorBrain...")
model = CondorBrain(
    d_model=512,      # Smaller width for speed
    n_layers=12,      # Fewer layers for speed
    input_dim=24,
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# --- 3. TRAINING LOOP ---
print("\n[3/4] Training Loop...")
model.train()
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Create sequence dimension on the fly [Batch, Time, Feat]
        # This is strictly a hack for the demo; normally you pre-slice
        # We will just reshape linear input to [B, 1, F] for this sanity check
        # Real training requires proper [B, T, F] sequences
        
        # ACTUALLY: Let's use a proper sequence slice.
        # But for speed in this test script, we treat Time=1.
        # Mamba requires Time dimension.
        data_seq = data.unsqueeze(1).to(device) # [B, 1, F]
        target = target.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # Output is (outputs, auxiliary_loss, etc.)
            outputs, aux_loss, _ = model(data_seq)
            loss = criterion(outputs, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    print(f"   Epoch {epoch+1} Avg Loss: {epoch_loss / len(train_loader):.6f}")

# --- 4. SAVE ---
print("\n[4/4] Saving Model...")
torch.save(model.state_dict(), "condor_brain_new_trained.pth")
print("✅ Saved 'condor_brain_new_trained.pth'")
