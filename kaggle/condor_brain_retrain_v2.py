# ============================================================
# CONDORBRAIN RETRAINING V2 - FIXING POSTERIOR COLLAPSE
# ============================================================
print("🚀 Starting CondorBrain Retraining V2 (Shock Therapy)...")
'''
# --- 0. PREP & CLEAN ---
import os
print("🔄 Syncing Repo...")
os.system("cd spy-iron-condor-trading && git fetch origin && git reset --hard origin/main")
print("✅ Repo synced")
'''
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

# --- CONFIG (Shock Therapy) ---
EPOCHS = 10         # Increased from 4
BATCH_SIZE = 128
LR = 5e-4           # Increased from 1e-4
ROWS_TO_LOAD = 3_000_000
SEQ_LEN = 256
PREDICT_HORIZON = 32

# Optimization Flags
USE_DATAPARALLEL = True       # Re-enabling: Dual T4 is significantly faster (1.76 it/s vs 0.67 it/s)
DIFFUSION_WARMUP_EPOCHS = 2   # Keep this: Skipping diffusion on 2 GPUs will be max speed
DIFFUSION_STEPS_TRAIN = 50    # Default steps

device = torch.device('cuda')
print(f"   GPU: {torch.cuda.get_device_name(0)}")

# --- 1. DATA LOADING & PREP ---
print(f"\n[1/4] Loading & Processing {ROWS_TO_LOAD:,} Rows...")
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
df = pd.read_csv(DATA_PATH).iloc[-ROWS_TO_LOAD:]
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

Y_feat_np = np.stack([r, rho, d, v], axis=1).astype(np.float32)

# X features
X_np = df[FEATURE_COLS].values.astype(np.float32)
X_np = np.nan_to_num(X_np, nan=0.0, posinf=10.0, neginf=-10.0)

# ------------------------------
# TRAIN / VAL SPLIT
# ------------------------------
split_idx = int(len(X_np) * 0.9)
X_train_np = X_np[:split_idx]
X_val_np   = X_np[split_idx:]
Y_feat_train_np = Y_feat_np[:split_idx]
Y_feat_val_np   = Y_feat_np[split_idx:]

# ------------------------------
# SCALING (FIT TRAIN ONLY)
# ------------------------------
median = np.median(X_train_np, axis=0, keepdims=True).astype(np.float32)
mad = (np.median(np.abs(X_train_np - median), axis=0, keepdims=True) + 1e-8).astype(np.float32)

def robust_norm(x):
    return np.clip((x - median) / (1.4826 * mad), -10.0, 10.0).astype(np.float32)

X_train_np = robust_norm(X_train_np)
X_val_np   = robust_norm(X_val_np)

# POLICY TARGETS (NO WRAPAROUND)
# ------------------------------
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

Y_policy_np = np.zeros((len(X_np), 8), dtype=np.float32)
Y_policy_np[:, 0] = 2.0 + np.clip(returns_60m * 0.5, -1.0, 1.0) # Call Offset
Y_policy_np[:, 1] = 2.0 - np.clip(returns_60m * 0.5, -1.0, 1.0) # Put Offset
Y_policy_np[:, 2] = 5.0 + np.clip(vol_60m * 20, 0, 5)       # Width
Y_policy_np[:, 4] = 0.5 + np.clip(returns_60m * 0.1, -0.4, 0.4) # Prob Profit
Y_policy_np[:, 7] = 0.3 + np.clip(np.abs(returns_60m), 0.0, 0.6)  # Confidence

Y_policy_train_np = Y_policy_np[:split_idx]
Y_policy_val_np = Y_policy_np[split_idx:]

# ------------------------------
# FAST TORCH DATASET (NO TENSOR CREATION PER SAMPLE)
# ------------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, Y_policy, Y_feat, seq_len, horizon=32):
        self.X = torch.from_numpy(X)
        self.Y_policy = torch.from_numpy(Y_policy)
        self.Y_feat = torch.from_numpy(Y_feat)
        self.seq_len = seq_len
        self.horizon = horizon
        self.max_i = len(X) - seq_len - horizon - 1
        
    def __len__(self):
        return self.max_i
        
    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_pol = self.Y_policy[idx + self.seq_len - 1]
        y_next = self.Y_feat[idx + self.seq_len]
        y_traj = self.Y_feat[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x_seq, y_pol, y_next, y_traj

train_dataset = SequenceDataset(X_train_np, Y_policy_train_np, Y_feat_train_np, SEQ_LEN, PREDICT_HORIZON)
val_dataset = SequenceDataset(X_val_np, Y_policy_val_np, Y_feat_val_np, SEQ_LEN, PREDICT_HORIZON)

num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=num_workers, 
    pin_memory=True,
    persistent_workers=True, 
    prefetch_factor=4
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=num_workers, 
    pin_memory=True,
    persistent_workers=True, 
    prefetch_factor=4
)

# --- 2. MODEL SETUP ---
print("\n[2/4] Initializing CondorBrain V2...")
model = CondorBrain(
    d_model=512,
    n_layers=12,
    input_dim=24,
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1,
    use_diffusion=True,     # Enable Diffusion Head
    diffusion_steps=DIFFUSION_STEPS_TRAIN
).to(device)

# --- INITIALIZATION FIX ---
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
             m.bias.data.fill_(0.01) # Small bias to prevent dead neurons

print("   Applying Xavier Initialization...")
model.apply(init_weights)

if USE_DATAPARALLEL and torch.cuda.device_count() > 1:
    print(f"   Using {torch.cuda.device_count()} GPUs (DataParallel)")
    model = nn.DataParallel(model)
else:
    print("   Using Single GPU (USE_DATAPARALLEL=False)")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Scheduler: Cosine Annealing with Warmup/Restarts
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LR, 
    steps_per_epoch=len(train_loader), 
    epochs=EPOCHS,
    pct_start=0.1
)

criterion_policy = nn.MSELoss()
criterion_forecast = nn.HuberLoss()

# --- 3. TRAINING LOOP ---
print("\n[3/4] Retraining Loop (Balanced Loss)...")
scaler = torch.amp.GradScaler('cuda')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_pol_loss = 0
    total_feat_loss = 0
    total_diff_loss = 0
    
    # 🌟 Staged Training Logic 🌟
    use_diffusion = (epoch >= DIFFUSION_WARMUP_EPOCHS)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Diff={'ON' if use_diffusion else 'OFF'}]")
    
    for batch_idx, (x_seq, y_pol, y_next, y_traj) in enumerate(pbar):
        x_seq = x_seq.to(device, non_blocking=True)
        y_pol = y_pol.to(device, non_blocking=True)
        y_next = y_next.to(device, non_blocking=True)
        y_traj = y_traj.to(device, non_blocking=True) # (B, 32, 4)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            # Only pass diffusion target if we are past warmup
            traj_target = y_traj if use_diffusion else None
            
            # Pass diffusion_target (y_traj) to compute diffusion loss internally
            # return_features=True -> output index 3
            # use_diffusion=True -> output index 4
            res = model(
                x_seq, 
                return_features=True, 
                diffusion_target=traj_target
            )
            
            # Unpack results
            # res structure: [outputs, regime, horizon, features, diffusion, experts...]
            outputs = res[0]
            # regime = res[1]
            # horizon = res[2]
            feat_pred = res[3]  # Index 3 if return_features=True
            diff_loss_scalar = res[4] if use_diffusion else None # Index 4 if use_diffusion=True
            
            # 1. Policy Loss
            loss_pol = criterion_policy(outputs, y_pol)
            
            # 2. Feature Loss (Next Step)
            loss_feat = criterion_forecast(feat_pred, y_next) * 1000.0

            # 3. Diffusion Loss
            loss_diff = torch.tensor(0.0, device=device)
            if diff_loss_scalar is not None:
                # Handle DataParallel: If gathered loss is a vector (one per GPU), take mean
                if diff_loss_scalar.dim() > 0:
                    diff_loss_scalar = diff_loss_scalar.mean()
                loss_diff = diff_loss_scalar * 10.0 # Scale diffusion loss
            
            # Total Loss
            loss = (loss_pol * 2.0) + loss_feat + loss_diff 
        
        # Stability check
        if not torch.isfinite(loss):
             pbar.write(f"⚠️  Non-finite loss encountered at batch {batch_idx}, skipping...")
             optimizer.zero_grad(set_to_none=True)
             continue

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        total_pol_loss += loss_pol.item()
        total_feat_loss += loss_feat.item()
        total_diff_loss += loss_diff.item()
        
        pbar.set_postfix({
            'L_pol': f"{loss_pol.item():.4f}", 
            'L_feat': f"{loss_feat.item():.4f}",
            'L_diff': f"{loss_diff.item():.4f}"
        })
        
        # Periodic Logging (every 100 batches)
        if batch_idx % 100 == 0:
            pbar.write(f"   [Batch {batch_idx}] L_All={loss.item():.4f} | Pol={loss_pol.item():.4f} | Diff={loss_diff.item():.4f}")
    
    avg_pol = total_pol_loss / len(train_loader)
    avg_feat = total_feat_loss / len(train_loader)
    avg_diff = total_diff_loss / len(train_loader)
    
    print(f"   Epoch {epoch+1} Train: Policy={avg_pol:.4f} | Feat={avg_feat:.4f} | Diff={avg_diff:.4f}")
    
    # Save Checkpoint with Metadata
    save_path = f"condor_brain_retrain_e{epoch+1}.pth"
    
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        "state_dict": state_dict,
        "feature_cols": FEATURE_COLS,
        "median": median.astype(np.float32),
        "mad": mad.astype(np.float32),
        "seq_len": SEQ_LEN,
        "input_dim": len(FEATURE_COLS),
        "use_diffusion": True,
        "diffusion_steps": 50
    }
    
    torch.save(checkpoint, save_path)
    print(f"      💾 Saved: {save_path}")

print("✅ Retraining Complete.")
