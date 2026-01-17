# ============================================================
# CONDORBRAIN PRODUCTION TRAINING - SEQUENCE MODE
# ============================================================
print("🚀 Starting CondorBrain Production Training (Sequence Mode + Forecasting)...")

# --- 0. PREP & CLEAN ---
!cd spy-iron-condor-trading && git fetch origin && git reset --hard origin/main
print("✅ Repo synced")

import sys
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from intelligence.condor_brain import CondorBrain

# --- CONFIG ---
EPOCHS = 4
BATCH_SIZE = 128    # Reduced for Sequence 256
LR = 1e-4
ROWS_TO_LOAD = 3_000_000
SEQ_LEN = 256       # Lookback: 256 bars
PREDICT_HORIZON = 32 # Forecast: 32 bars (Autoregressive)

device = torch.device('cuda')
n_gpus = torch.cuda.device_count()
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   Count: {n_gpus}")

# --- 1. DATA LOADING & PREP ---
print(f"\n[1/4] Loading & Processing {ROWS_TO_LOAD:,} Rows...")
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"

# Load tail of dataset
df = pd.read_csv(DATA_PATH).iloc[-ROWS_TO_LOAD:]
print(f"   Shape: {df.shape}")

# Features
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 
                'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 
                'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 
                'psar', 'strike', 'target_spot', 'max_dd_60m']

# Extract raw arrays for target generation
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
volumes = df['volume'].values

# Calculate Forecasting Targets (Next Step Features: r, rho, d, v)
# r_t = log(C_t / C_{t-1})
# rho_t = log(H_t / L_t)
# d_t = log(C_t / O_t)
# v_t = log(V_t / V_{t-1})
print("   Computing forecasting targets...")
eps = 1e-8
log_c = np.log(closes + eps)
log_v = np.log(volumes + 1.0)

# Shifted arrays for t+1
# We want to predict step t+1 given history 0..t
# Target at index t is the feature vector of t+1
# Let's align: X[t] -> Predicts Y[t+1]

# Compute features for ALL steps first
r = np.zeros_like(closes)
r[1:] = np.diff(log_c)

rho = np.log((highs + eps) / (lows + eps))
d = np.log((closes + eps) / (opens + eps))

v = np.zeros_like(volumes, dtype=float)
v[1:] = np.diff(log_v)

# Stack targets
# shape (N, 4)
# Y_feat[i] contains the market state 'change' that happened at step i
Y_feat = np.stack([r, rho, d, v], axis=1).astype(np.float32)

# X features
X = df[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

# Robust Scaling
median = np.median(X, axis=0, keepdims=True)
mad = np.median(np.abs(X - median), axis=0, keepdims=True) + 1e-8
X_norm = np.clip((X - median) / (1.4826 * mad), -10, 10)

# Policy Targets (Iron Condor Params) - Semi-realistic
print("   Generating semi-realistic policy targets...")
returns_60m = np.roll(X[:, 3], -60) - X[:, 3]
vol_60m = pd.Series(returns_60m).rolling(60).std().fillna(0).values

Y_policy = np.zeros((len(X), 8), dtype=np.float32)
Y_policy[:, 0] = 2.0 + np.clip(returns_60m * 0.5, -1, 1) # Call Offset
Y_policy[:, 1] = 2.0 - np.clip(returns_60m * 0.5, -1, 1) # Put Offset
Y_policy[:, 2] = 5.0 + np.clip(vol_60m * 20, 0, 5)       # Width
Y_policy[:, 4] = 0.5 + np.clip(returns_60m * 0.1, -0.4, 0.4) # Prob Profit
Y_policy[:, 7] = 0.3 + np.clip(np.abs(returns_60m), 0, 0.6)  # Confidence

class SequenceDataset(Dataset):
    def __init__(self, X, Y_policy, Y_feat, seq_len):
        self.X = X
        self.Y_policy = Y_policy
        self.Y_feat = Y_feat
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len - 1
        
    def __getitem__(self, idx):
        # Sequence input: t to t+seq_len
        x_seq = self.X[idx : idx + self.seq_len]
        
        # Policy Target: The parameters relevant for the trade placed at end of sequence
        y_pol = self.Y_policy[idx + self.seq_len - 1]
        
        # Forecasting Target: The market features of the NEXT step (t + seq_len)
        # We predict what happens immediately after the sequence ends
        y_next = self.Y_feat[idx + self.seq_len]
        
        return torch.tensor(x_seq), torch.tensor(y_pol), torch.tensor(y_next)

# Split
split_idx = int(len(X) * 0.9)
train_dataset = SequenceDataset(X_norm[:split_idx], Y_policy[:split_idx], Y_feat[:split_idx], SEQ_LEN)
val_dataset = SequenceDataset(X_norm[split_idx:], Y_policy[split_idx:], Y_feat[split_idx:], SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- 2. MODEL SETUP ---
print("\n[2/4] Initializing CondorBrain (Sequence Config)...")
model = CondorBrain(
    d_model=512,       # Reduced for sequence stability on T4
    n_layers=12,       # Reduced layer count
    input_dim=24,
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1
).to(device)

if n_gpus > 1:
    model = nn.DataParallel(model)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
criterion_policy = nn.MSELoss()
criterion_forecast = nn.HuberLoss() # More robust for market features

# --- 3. TRAINING LOOP ---
print("\n[3/4] Training Loop...")
scaler = torch.amp.GradScaler('cuda')
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_pol_loss = 0
    total_feat_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for x_seq, y_pol, y_next in pbar:
        x_seq, y_pol, y_next = x_seq.to(device), y_pol.to(device), y_next.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Forward input sequence
            # Returns: (outputs, regime, horizon, features, experts)
            res = model(x_seq, return_features=True)
            
            # Unpack (Handle DataParallel tuple return if needed, but safe here)
            # Tuple: (outputs, regime_logits, horizon_forecast, feature_pred, experts)
            outputs = res[0]
            feat_pred = res[3] # feature prediction head output
            
            # Loss Calculation
            loss_pol = criterion_policy(outputs, y_pol)
            loss_feat = criterion_forecast(feat_pred, y_next)
            
            # Combined Loss (Weighting forecasting task equally for Meta-Forecaster utility)
            loss = loss_pol + loss_feat
        
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        total_pol_loss += loss_pol.item()
        total_feat_loss += loss_feat.item()
        
        pbar.set_postfix({
            'L_pol': f"{loss_pol.item():.2f}", 
            'L_feat': f"{loss_feat.item():.2f}"
        })
    
    avg_loss = total_loss / len(train_loader)
    print(f"   Epoch {epoch+1} Train: Policy={total_pol_loss/len(train_loader):.4f} | Feat={total_feat_loss/len(train_loader):.4f}")
    
    # --- VALIDATION LOOP ---
    model.eval()
    val_pol_loss = 0
    val_feat_loss = 0
    sample_printed = False
    
    with torch.no_grad():
        for x_seq, y_pol, y_next in val_loader:
            x_seq, y_pol, y_next = x_seq.to(device), y_pol.to(device), y_next.to(device)
            
            with torch.amp.autocast('cuda'):
                res = model(x_seq, return_features=True)
                outputs = res[0]
                feat_pred = res[3]
                
                loss_pol = criterion_policy(outputs, y_pol)
                loss_feat = criterion_forecast(feat_pred, y_next)
                
                val_pol_loss += loss_pol.item()
                val_feat_loss += loss_feat.item()
                
            # Print ONE sample for visual confirmation
            if not sample_printed:
                print(f"\n   🔍 SAMPLE OUPUT (Epoch {epoch+1}):")
                print(f"      Policy (Call Off) : True={y_pol[0,0]:.2f} | Pred={outputs[0,0]:.2f}")
                print(f"      Policy (Width)    : True={y_pol[0,2]:.2f} | Pred={outputs[0,2]:.2f}")
                print(f"      Feat (Next Ret)   : True={y_next[0,0]:.6f} | Pred={feat_pred[0,0]:.6f}")
                print(f"      Feat (Next Vol)   : True={y_next[0,1]:.6f} | Pred={feat_pred[0,1]:.6f}")
                sample_printed = True

    avg_val_pol = val_pol_loss / len(val_loader)
    avg_val_feat = val_feat_loss / len(val_loader)
    print(f"   Epoch {epoch+1} Val  : Policy={avg_val_pol:.4f} | Feat={avg_val_feat:.4f}")
    
    # Save Checkpoint
    save_path = f"condor_brain_seq_e{epoch+1}.pth"
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"      💾 Saved: {save_path}")

print("✅ Sequence Mode Training Complete.")
