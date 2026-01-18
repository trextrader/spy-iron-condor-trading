import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

from intelligence.condor_brain import CondorBrain

print("🚀 Starting CondorBrain Sequence Training (Production Mode)...")

# --- CONFIG ---
EPOCHS = 10
BATCH_SIZE = 128
LR = 2e-4
WEIGHT_DECAY = 0.01

# v2.2 Hardware Config
device = torch.device('cuda')
rows_to_load = 5_000_000  # More data for production

# --- 1. DATA LOADING & PREP ---
print(f"[1/4] Loading & Processing {rows_to_load:,} Rows...")
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"

# Check if running locally (for verification)
if not os.path.exists(DATA_PATH):
    print("   ⚠️ Running in local verification mode (Mock Data)")
    df = pd.DataFrame(np.random.randn(10000, 24), columns=['col'+str(i) for i in range(24)])
    # Mock specific columns needed
    df['open'] = 100 + np.cumsum(np.random.randn(10000))
    df['high'] = df['open'] + abs(np.random.randn(10000))
    df['low'] = df['open'] - abs(np.random.randn(10000))
    df['close'] = df['open'] + np.random.randn(10000)*0.1
    df['volume'] = np.random.randint(100, 1000, 10000)
else:
    df = pd.read_csv(DATA_PATH).iloc[-rows_to_load:]

print(f"   Shape: {df.shape}")

FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 
                'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 
                'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 
                'psar', 'strike', 'target_spot', 'max_dd_60m']

# ------------------------------
# FEATURE & TARGET CONSTRUCTION
# ------------------------------
eps = 1e-8

close = df["close"].to_numpy(np.float32)
highs = df["high"].to_numpy(np.float32)
lows = df["low"].to_numpy(np.float32)
opens = df["open"].to_numpy(np.float32)
volumes = df["volume"].to_numpy(np.float32)

log_c = np.log(close + eps)
log_v = np.log(volumes + 1.0)

r = np.zeros_like(close, dtype=np.float32)
r[1:] = np.diff(log_c)
rho = np.log((highs + eps) / (lows + eps)).astype(np.float32)
d = np.log((close + eps) / (opens + eps)).astype(np.float32)

v = np.zeros_like(volumes, dtype=np.float32)
v[1:] = np.diff(log_v).astype(np.float32)

Y_feat = np.stack([r, rho, d, v], axis=1).astype(np.float32)

# X features
X = df[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

# ------------------------------
# TRAIN / VAL SPLIT (BEFORE SCALING)  ✅ prevents leakage
# ------------------------------
split_idx = int(len(X) * 0.9)
X_train = X[:split_idx]
X_val   = X[split_idx:]
Y_feat_train = Y_feat[:split_idx]
Y_feat_val   = Y_feat[split_idx:]

# ------------------------------
# ROBUST SCALING (FIT TRAIN ONLY) ✅ prevents leakage
# ------------------------------
median = np.median(X_train, axis=0, keepdims=True).astype(np.float32)
mad = (np.median(np.abs(X_train - median), axis=0, keepdims=True) + 1e-8).astype(np.float32)

def robust_norm(x):
    return np.clip((x - median) / (1.4826 * mad), -10.0, 10.0).astype(np.float32)

X_train_norm = robust_norm(X_train)
X_val_norm   = robust_norm(X_val)

# Policy Targets (Iron Condor Params) - Semi-realistic
print("   Generating semi-realistic policy targets...")

# ✅ NO WRAPAROUND: avoid np.roll leakage
future_close = np.empty_like(close)
future_close[:-60] = close[60:]
future_close[-60:] = np.nan
returns_60m = (future_close - close).astype(np.float32)
returns_60m = np.nan_to_num(returns_60m, nan=0.0)

vol_60m = (
    pd.Series(returns_60m)
    .rolling(60, min_periods=1)
    .std(ddof=0)
    .fillna(0.0)
    .to_numpy(np.float32)
)

Y_policy = np.zeros((len(X), 8), dtype=np.float32)
Y_policy[:, 0] = 2.0 + np.clip(returns_60m * 0.5, -1, 1) # Call Offset
Y_policy[:, 1] = 2.0 - np.clip(returns_60m * 0.5, -1, 1) # Put Offset
Y_policy[:, 2] = 5.0 + np.clip(vol_60m * 20, 0, 5)       # Width
Y_policy[:, 4] = 0.5 + np.clip(returns_60m * 0.1, -0.4, 0.4) # Prob Profit
Y_policy[:, 7] = 0.3 + np.clip(np.abs(returns_60m), 0, 0.6)  # Confidence

Y_policy_train = Y_policy[:split_idx]
Y_policy_val   = Y_policy[split_idx:]

class SequenceDataset(Dataset):
    def __init__(self, X, Y_policy, Y_feat, seq_len):
        # ✅ Fast: store as torch tensors once (no per-sample tensor creation)
        self.X = torch.from_numpy(X)
        self.Y_policy = torch.from_numpy(Y_policy)
        self.Y_feat = torch.from_numpy(Y_feat)
        self.seq_len = seq_len
        self.max_i = len(X) - seq_len - 1
        
    def __len__(self):
        return self.max_i
        
    def __getitem__(self, idx):
        # Sequence input: t to t+seq_len
        x_seq = self.X[idx : idx + self.seq_len]
        
        # Policy: at last timestep of seq
        y_pol = self.Y_policy[idx + self.seq_len - 1]
        
        # Feature: next-step (t+seq_len)
        y_next = self.Y_feat[idx + self.seq_len]
        
        return x_seq, y_pol, y_next

# Hyperparams
SEQ_LEN = 256

# Build datasets
train_dataset = SequenceDataset(X_train_norm, Y_policy_train, Y_feat_train, SEQ_LEN)
val_dataset   = SequenceDataset(X_val_norm,   Y_policy_val,   Y_feat_val,   SEQ_LEN)

num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

# Build model
print("\n[2/4] Initializing CondorBrain V2...")
model = CondorBrain(
    d_model=512,
    n_layers=12,
    input_dim=24,
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1
).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
             m.bias.data.fill_(0.01)

model.apply(init_weights)

if torch.cuda.device_count() > 1:
    print(f"   Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
criterion_policy = nn.MSELoss()
criterion_forecast = nn.HuberLoss() # Better for financial data

# --- 3. TRAINING LOOP ---
print("\n[3/4] Production Training Loop...")
scaler = torch.amp.GradScaler('cuda')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for x_seq, y_pol, y_next in pbar:
        x_seq = x_seq.to(device, non_blocking=True)
        y_pol = y_pol.to(device, non_blocking=True)
        y_next = y_next.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            # Forward input sequence
            # return_features=True gives use the next-step prediction
            res = model(x_seq, return_features=True)
            outputs = res[0]
            feat_pred = res[3] # Index 3 is feature_pred
            
            # Loss Components
            loss_pol = criterion_policy(outputs, y_pol)
            loss_feat = criterion_forecast(feat_pred, y_next) * 100000.0
            
            # Combined Loss
            loss = loss_pol + loss_feat

        # ✅ Stability guard: skip non-finite loss
        if not torch.isfinite(loss):
            pbar.write("⚠️  Non-finite loss encountered, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L_Pol': f"{loss_pol.item():.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    print(f"   Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
    
    # Validation Loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_seq, y_pol, y_next in val_loader:
            x_seq = x_seq.to(device, non_blocking=True)
            y_pol = y_pol.to(device, non_blocking=True)
            y_next = y_next.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                res = model(x_seq, return_features=True)
                outputs = res[0]
                feat_pred = res[3]
                
                loss_pol = criterion_policy(outputs, y_pol)
                loss_feat = criterion_forecast(feat_pred, y_next) * 100000.0
                loss = loss_pol + loss_feat
                
            val_loss += loss.item()
            
    avg_val = val_loss / len(val_loader)
    print(f"   Epoch {epoch+1} Val Loss: {avg_val:.4f}")
    
    # Save Checkpoint
    save_path = f"condor_brain_seq_e{epoch+1}.pth"
    torch.save(
        {
            "state_dict": (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            "feature_cols": FEATURE_COLS,
            "median": median.astype(np.float32),
            "mad": mad.astype(np.float32),
            "seq_len": SEQ_LEN,
            "input_dim": len(FEATURE_COLS),
            "use_diffusion": False,
            "diffusion_steps": 0,
        },
        save_path
    )
    print(f"      💾 Saved: {save_path}")

print("✅ Sequence Mode Training Complete.")
