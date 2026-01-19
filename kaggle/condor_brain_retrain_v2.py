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
from torch.utils.tensorboard import SummaryWriter
from intelligence.condor_brain import CondorBrain
from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V21, INPUT_DIM_V21, VERSION_V21,
    NAN_POLICY_V21, NORMALIZATION_POLICY_V21,
    apply_semantic_nan_fill,
)
from intelligence.features.dynamic_features import compute_all_dynamic_features

# 🔄 FIX STALE IMPORTS (Colab/Jupyter specific)
import sys
import importlib
if 'intelligence.training_monitor' in sys.modules:
    import intelligence.training_monitor
    importlib.reload(intelligence.training_monitor)
    print("🔄 Forcing reload of intelligence.training_monitor")

from intelligence.training_monitor import (
    TrainingMonitor, compute_val_head_losses, sample_predictions, MAIN_HEADS
)
import io
from PIL import Image
import matplotlib.pyplot as plt

# --- TRAINING CONFIG ---
# ROWS_TO_LOAD: Number of rows from end of dataset
#   100K rows ≈ 1K unique spot bars (~1 min compute)
#   300K rows ≈ 3K unique spot bars (~2 min compute)
#   1M rows ≈ 10K unique spot bars (~3 min compute)
#   3M rows ≈ 30K unique spot bars (~5 min compute)
#   Full dataset: ~10M rows ≈ 100K unique spot bars

ROWS_TO_LOAD = 100_000  # ⚡ CHANGE THIS TO SCALE TRAINING ⚡
EPOCHS = 2              # Quick test: 2, Full training: 10

# Derived estimate
# Derived estimate
estimated_spots = max(ROWS_TO_LOAD // 100, 100)  # ~100 options per spot bar
print(f"📊 Config: {ROWS_TO_LOAD:,} rows, {EPOCHS} epochs")

BATCH_SIZE = 128
LR = 1e-4  # Lowered from 5e-4 for stability
SEQ_LEN = 256
PREDICT_HORIZON = 32

# Optimization Flags
DIFFUSION_WARMUP_EPOCHS = 1  # Skip diffusion for first epoch
DIFFUSION_STEPS_TRAIN = 50

device = torch.device('cuda')
n_gpus = torch.cuda.device_count()
USE_DATAPARALLEL = (n_gpus > 1)  # Auto-detect: Kaggle dual T4 vs Colab single T4
print(f"   GPU: {torch.cuda.get_device_name(0)} x{n_gpus} (DataParallel={'ON' if USE_DATAPARALLEL else 'OFF'})")

from typing import List, Tuple
def _isfinite_np(x: np.ndarray) -> bool:
    return np.isfinite(x).all()

def _sanitize_np_to_nan(x: np.ndarray) -> np.ndarray:
    """
    Replace +/-inf with NaN so semantic fill can treat them as missing.
    """
    if not np.isfinite(x).all():
        x = x.copy()
        x[~np.isfinite(x)] = np.nan
    return x

def _check_finite_t(name: str, t: torch.Tensor) -> Tuple[bool, str]:
    """
    Returns (ok, message). If not ok, message contains diagnostic summary.
    """
    if t is None:
        return True, ""
    finite = torch.isfinite(t)
    if finite.all():
        return True, ""
    bad = (~finite).sum().item()
    msg = f"{name} has {bad} non-finite values | shape={tuple(t.shape)}"
    # provide a little more context when possible
    try:
        msg += f" | min={torch.nanmin(t).item():.6g} max={torch.nanmax(t).item():.6g}"
    except Exception:
        pass
    return False, msg

def _maybe_launch_tensorboard_inline(logdir: str, port: int = 6006) -> None:
    """
    In Colab/Jupyter, render TensorBoard inline.
    Note: plain `tensorboard --logdir ...` in a script will NOT auto-render in Colab.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return
        # Load TB magic + render
        ip.run_line_magic("load_ext", "tensorboard")
        ip.run_line_magic("tensorboard", f"--logdir {logdir} --port {port}")
    except Exception as e:
        print(f"   ⚠️ TensorBoard inline not available: {e}")

# --- LAUNCH TENSORBOARD (Colab/Jupyter Only) ---
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip:
        print("📊 Launching TensorBoard inline...")
        # Kill any existing TensorBoard process on port 6006
        try:
            # Use fuser to kill process on TCP port 6006 (Linux/Colab)
            ip.getoutput("fuser -k 6006/tcp")
        except:
            pass
            
        # Reload extension to be safe
        ip.run_line_magic('reload_ext', 'tensorboard') 
        ip.run_line_magic('tensorboard', '--logdir=runs/condor_brain --port=6006')
except Exception as e:
    # Not running in IPython or other error
    pass

# --- 1. DATA LOADING & PREP ---
print(f"\n[1/4] Loading & Processing {ROWS_TO_LOAD:,} Rows...")

# Auto-detect environment: Kaggle vs Colab vs Local
KAGGLE_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
COLAB_PATH = "/content/spy-iron-condor-trading/data/processed/mamba_institutional_1m.csv"
COLAB_ROOT_PATH = "data/processed/mamba_institutional_1m.csv"  # If already in repo root
LOCAL_PATH = "data/processed/mamba_institutional_1m.csv"

for p in [KAGGLE_PATH, COLAB_PATH, COLAB_ROOT_PATH, LOCAL_PATH]:
    if os.path.exists(p):
        DATA_PATH = p
        break
else:
    raise FileNotFoundError(f"Data file not found!")

print(f"   Data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH).iloc[-ROWS_TO_LOAD:]
print(f"   Shape: {df.shape}")

# V2.1 FEATURE SCHEMA (schema-driven feature count; dynamic indicators included)
FEATURE_COLS = FEATURE_COLS_V21
INPUT_DIM = INPUT_DIM_V21
print(f"   Using V2.1 Schema: {len(FEATURE_COLS)} features")
assert len(FEATURE_COLS) == INPUT_DIM, (
    f"Schema mismatch: len(FEATURE_COLS)={len(FEATURE_COLS)} != INPUT_DIM={INPUT_DIM}"
)

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
# Sanitize targets (prevent NaNs and Ints)
Y_feat_np = np.nan_to_num(Y_feat_np, nan=0.0, posinf=5.0, neginf=-5.0)
Y_feat_np = np.clip(Y_feat_np, -10.0, 10.0)

# =========================================================================
# CRITICAL: Compute dynamic features on UNIQUE SPOT bars, then merge back.
# (Options data has ~100 rows per timestamp with same OHLCV)
# =========================================================================
print("   Extracting unique spot bars...")
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
spot_key_cols = ['symbol', 'dt'] if 'dt' in df.columns else ['symbol']
if 'dt' not in df.columns and 'timestamp' in df.columns:
    spot_key_cols = ['symbol', 'timestamp']

# Find actual datetime column
dt_col = None
for c in ['dt', 'timestamp', 'datetime', 'date']:
    if c in df.columns:
        dt_col = c
        spot_key_cols = ['symbol', c] if 'symbol' in df.columns else [c]
        break

if dt_col is None:
    # No datetime column - use row order (rare case)
    print("   ⚠️ No datetime column found, computing on all rows...")
    df = compute_all_dynamic_features(df, close_col="close", high_col="high", low_col="low")
else:
    # Extract unique spots
    spot_df = df.drop_duplicates(subset=spot_key_cols)[spot_key_cols + ohlcv_cols].copy()
    spot_df = spot_df.sort_values(spot_key_cols).reset_index(drop=True)
    n_unique = len(spot_df)
    print(f"   Unique spot bars: {n_unique:,} (from {len(df):,} options rows)")
    if n_unique > 0:
        rows_per_bar = len(df) / float(n_unique)
        print(f"   Rows per spot bar (avg): {rows_per_bar:.2f}")
    
    # Compute dynamic features on spots only
    print("   Computing dynamic features on spot bars...")
    spot_df = compute_all_dynamic_features(spot_df, close_col="close", high_col="high", low_col="low")
    
    # Get dynamic columns and merge back
    dynamic_cols = [c for c in spot_df.columns if c not in spot_key_cols + ohlcv_cols]
    print(f"   Dynamic columns: {len(dynamic_cols)}")
    merge_cols = spot_key_cols + dynamic_cols
    df = df.merge(spot_df[merge_cols], on=spot_key_cols, how='left')

# X features (schema-driven columns)
X_np = df[FEATURE_COLS].values.astype(np.float32)
# IMPORTANT: Convert inf/-inf to NaN BEFORE semantic fill.
# Otherwise median/MAD can become inf and produce inf-inf -> NaN in robust_norm.
X_np = _sanitize_np_to_nan(X_np)

# Apply per-feature semantic NaN filling (not global 0.0)
X_np = apply_semantic_nan_fill(X_np, FEATURE_COLS)

# Final hard safety: after fill, ensure no NaN/inf remain in X_np
if not _isfinite_np(X_np):
    n_bad = np.sum(~np.isfinite(X_np))
    print(f"   ⚠️ X_np still has {n_bad:,} non-finite cells after semantic fill; coercing to 0.0")
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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
X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_val_np   = np.nan_to_num(X_val_np,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

median = np.median(X_train_np, axis=0, keepdims=True).astype(np.float32)
mad = np.median(np.abs(X_train_np - median), axis=0, keepdims=True).astype(np.float32)

# Clamp MAD to avoid division blowups and avoid NaNs from 0/0 on sparse features
mad = np.maximum(mad, 1e-6).astype(np.float32)

# If any stats are non-finite, something upstream is still broken; repair safely
if not np.isfinite(median).all() or not np.isfinite(mad).all():
    print("   ⚠️ Non-finite median/MAD detected; coercing stats to safe defaults")
    median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mad    = np.nan_to_num(mad,    nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)

def robust_norm(x):
    # Ensure no division by zero (mad has epsilon, but safety first)
    # Also handle if x has NaNs (though it shouldn't by now)
    x = np.nan_to_num(x, nan=0.0)
    out = (x - median) / (1.4826 * mad)
    out = np.clip(out, -10.0, 10.0)
    return out.astype(np.float32)

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
    input_dim=INPUT_DIM,  # V2.1: schema-driven feature count
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1,
    use_diffusion=True,
    diffusion_steps=DIFFUSION_STEPS_TRAIN
).to(device)

# --- INITIALIZATION FIX ---
# ⚠️ CRITICAL: Do NOT apply Xavier to Mamba internals (they have specialized init).
# Only re-init the output heads (experts, moe, policy head, etc.)
SAFE_INIT_PREFIXES = ('expert_', 'moe_head', 'policy_head', 'regime_detector', 'feature_head', 'horizon_forecaster')

for name, param in model.named_parameters():
    # Only init weights (not biases) in safe modules
    if any(name.startswith(prefix) for prefix in SAFE_INIT_PREFIXES):
        if 'weight' in name and param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
            print(f"   Xavier init: {name}")
        elif 'bias' in name:
            param.data.fill_(0.01)

print("   ✅ Safe Xavier Initialization applied to output heads only.")

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

# --- ASSERT DATA SANITY ---
# ⚠️ If training starts with NaNs in input, it will explode immediately
if hasattr(model, 'encoder'):
    if torch.isnan(model.encoder.layers[0].mixer.A_log).any():
        print("❌ ERROR: Model initialized with NaNs!")

x_sample, _, _, _ = next(iter(train_loader))
if torch.isnan(x_sample).any():
    print("❌ ERROR: Input data contains NaNs! robust_norm failed.")
    # Force fix (in memory hack, wont fix loader but wil warn)
    print("   🚑 Warning: Data loader producing NaNs. Check robust_norm.")

# --- 3. TRAINING LOOP ---
print("\n[3/4] Retraining Loop (Balanced Loss)...")
scaler = torch.amp.GradScaler('cuda')

# Initialize TensorBoard Writer
tb_logdir = "runs/condor_brain"
writer = SummaryWriter(log_dir=tb_logdir)

print("📊 Launching TensorBoard inline...")
if not "_TB_STARTED" in globals():
    _maybe_launch_tensorboard_inline(tb_logdir, port=6006)
    _TB_STARTED = True

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_pol_loss = 0
    total_feat_loss = 0
    total_diff_loss = 0
    
    # 🌟 Staged Training Logic 🌟
    use_diffusion = (epoch >= DIFFUSION_WARMUP_EPOCHS)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Diff={'ON' if use_diffusion else 'OFF'}]")
    skipped_batches = 0
    
    for batch_idx, (x_seq, y_pol, y_next, y_traj) in enumerate(pbar):
        x_seq = x_seq.to(device, non_blocking=True)
        y_pol = y_pol.to(device, non_blocking=True)
        y_next = y_next.to(device, non_blocking=True)
        y_traj = y_traj.to(device, non_blocking=True)
        
        # 🛡️ SANITIZE TARGETS (Just in case)
        y_pol = torch.nan_to_num(y_pol, nan=0.0)
        y_next = torch.nan_to_num(y_next, nan=0.0)
        y_traj = torch.nan_to_num(y_traj, nan=0.0)

        # HARD BATCH SANITY CHECK (inputs/targets)
        ok, msg = _check_finite_t("x_seq", x_seq)
        if not ok:
            pbar.write(f"⚠️  Skipping batch {batch_idx}: {msg}")
            skipped_batches += 1
            continue
        for nm, tens in [("y_pol", y_pol), ("y_next", y_next), ("y_traj", y_traj)]:
            ok, msg = _check_finite_t(nm, tens)
            if not ok:
                pbar.write(f"⚠️  Skipping batch {batch_idx}: {msg}")
                skipped_batches += 1
                continue
        
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

            # HARD OUTPUT SANITY CHECK (model outputs)
            ok, msg = _check_finite_t("outputs", outputs)
            if not ok:
                pbar.write(f"⚠️  Skipping batch {batch_idx}: {msg}")
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            ok, msg = _check_finite_t("feat_pred", feat_pred)
            if not ok:
                pbar.write(f"⚠️  Skipping batch {batch_idx}: {msg}")
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # 1. Policy Loss
            loss_pol = criterion_policy(outputs, y_pol)
            
            # 2. Feature Loss (Next Step)
            # 2. Feature Loss (Next Step)
            # Reduced scale from 1000.0 to 100.0 for stability
            loss_feat = criterion_forecast(feat_pred, y_next) * 100.0

            # 3. Diffusion Loss
            loss_diff = torch.tensor(0.0, device=device)
            if diff_loss_scalar is not None:
                # Handle DataParallel: If gathered loss is a vector (one per GPU), take mean
                if diff_loss_scalar.dim() > 0:
                    diff_loss_scalar = diff_loss_scalar.mean()
                loss_diff = diff_loss_scalar * 10.0 # Scale diffusion loss

                # ⚠️ SHOCK THERAPY: Clamp Diffusion Impact for first 50 batches of active diffusion
                if use_diffusion and batch_idx < 50:
                    loss_diff = torch.clamp(loss_diff, 0.0, 10.0)
            
            # Total Loss
            loss = (loss_pol * 2.0) + loss_feat + loss_diff 
        
        # Stability check
        if not torch.isfinite(loss):
             pbar.write(f"⚠️  Non-finite loss at batch {batch_idx}: P={loss_pol.item():.4f}, F={loss_feat.item():.4f}, D={loss_diff.item():.4f}")
             optimizer.zero_grad(set_to_none=True)
             continue

        scaler.scale(loss).backward()
        # 5. Optimize with Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # ⚡ CRITICAL FOR MAMBA ⚡
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
            
        # TensorBoard Logging (every batch)
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Loss/Total', loss.item(), global_step)
        writer.add_scalar('Loss/Policy', loss_pol.item(), global_step)
        writer.add_scalar('Loss/Feature', loss_feat.item(), global_step)
        writer.add_scalar('Loss/Diffusion', loss_diff.item(), global_step)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
    
    if skipped_batches:
        print(f"   ⚠️ Epoch {epoch+1}: skipped {skipped_batches} batches due to non-finite data/outputs.")
    
    avg_pol = total_pol_loss / len(train_loader)
    avg_feat = total_feat_loss / len(train_loader)
    avg_diff = total_diff_loss / len(train_loader)
    
    print(f"   Epoch {epoch+1} Train: Policy={avg_pol:.4f} | Feat={avg_feat:.4f} | Diff={avg_diff:.4f}")
    
    # TensorBoard Epoch Averages
    writer.add_scalar('Epoch/Policy_Loss', avg_pol, epoch+1)
    writer.add_scalar('Epoch/Feature_Loss', avg_feat, epoch+1)
    writer.add_scalar('Epoch/Diffusion_Loss', avg_diff, epoch+1)
    
    # === MONITOR & TENSORBOARD VISUALIZATION ===
    # This block replicates the rich logging from intelligence/train_condor_brain.py
    
    # 1. Compute per-head val losses (fast GPU accum)
    # create iterator once per epoch
    val_iter = iter(val_loader)
    
    def val_batch_wrapper(bi):
        try:
            batch = next(val_iter)
        except StopIteration:
            # Fallback if length calculation slightly off
            batch = next(iter(val_loader))
            
        x_seq, y_pol, y_next, y_traj = batch
        x_seq = x_seq.to(device, non_blocking=True)
        y_pol = y_pol.to(device, non_blocking=True)
        # Dummy regime (not used in this training phase)
        r = torch.zeros(x_seq.size(0), dtype=torch.long, device=device)
        return x_seq, y_pol, r

    # ⚠️ WORKAROUND: Colab bytecode cache is sticky. Wrap this in try-except 
    # so checkpointing can still happen even if validation fails.
    try:
        head_losses = compute_val_head_losses(
            model=model,
            get_batch_fn=val_batch_wrapper,
            n_batches=len(val_loader),
            device=device,
            amp_dtype=torch.float16 # Standard AMP
        )
        
        # 2. Log per-head scalars
        if writer is not None:
            for head_name, h_loss in head_losses.items():
                writer.add_scalar(f'HeadLoss/{head_name}', h_loss, epoch+1)
    except Exception as val_exc:
        print(f"   ⚠️ Validation skipped due to error: {val_exc}")
        head_losses = {} # Fallback empty dict
            
    # 3. Rich Image Logging (Scatter plots & Trajectories)
    # Run every epoch or every few epochs
    if writer is not None:
        try:
            # Get sample predictions
            samples = sample_predictions(
                model=model,
                get_batch_fn=lambda bi: next(iter(val_loader)),
                device=device,
                amp_dtype=torch.float16,
                n_samples=32
            )
            
            # A. Predicted vs Actual Scatter Plots
            for i, head_name in enumerate(MAIN_HEADS):
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                p = samples['preds'][:, i]
                t = samples['targets'][:, i]
                
                ax.scatter(t, p, alpha=0.6, s=50, c='blue', edgecolors='black')
                vmin, vmax = min(p.min(), t.min()), max(p.max(), t.max())
                margin = (vmax - vmin) * 0.1 + 0.01
                ax.plot([vmin-margin, vmax+margin], [vmin-margin, vmax+margin], 'k--', alpha=0.5)
                
                mae = np.mean(np.abs(p - t))
                corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 1e-6 else 0
                ax.set_title(f'{head_name}\nMAE={mae:.4f} | r={corr:.3f}')
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                
                # Convert to image
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=80)
                buf.seek(0)
                img = Image.open(buf)
                img_array = np.array(img)
                plt.close(fig)
                
                writer.add_image(f'Predictions/{head_name}', img_array, epoch+1, dataformats='HWC')
            
            # B. Horizon Trajectory (45-Day Forecast)
            if samples.get('forecast_data') is not None:
                forecast = samples['forecast_data'][0]  # Sample 0
                num_days = forecast.shape[0]
                days = np.arange(num_days)
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                # forecast components: [close, high, low, vol]
                ax.plot(days, forecast[:, 0], 'b-', label='Expected Close', linewidth=2)
                ax.fill_between(days, forecast[:, 2], forecast[:, 1], color='blue', alpha=0.2, label='High/Low Envelope')
                
                ax.set_title(f'HorizonForecaster: 45-Day Trajectory (Epoch {epoch+1})')
                ax.set_xlabel('Days from Now')
                ax.set_ylabel('Normalized Price Change')
                ax.legend()
                ax.grid(True, alpha=0.3)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img = Image.open(buf)
                img_array = np.array(img)
                plt.close(fig)
                writer.add_image('Horizon/Trajectory', img_array, epoch+1, dataformats='HWC')

            # Force flush to disk so Colab sees it immediately
            writer.flush()
            print(f"   [TensorBoard] 📸 Logged validation images (Scatter + Horizon) to {log_dir}")

        except Exception as e:
            print(f"[TensorBoard] Image logging error: {e}")

    # Save Checkpoint with Metadata
    save_path = f"condor_brain_retrain_e{epoch+1}.pth"
    
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        "state_dict": state_dict,
        "version": VERSION_V21,
        "feature_cols": FEATURE_COLS,
        "input_dim": INPUT_DIM,
        "median": median.astype(np.float32),
        "mad": mad.astype(np.float32),
        "seq_len": SEQ_LEN,
        "use_diffusion": True,
        "diffusion_steps": 50
    }
    
    torch.save(checkpoint, save_path)
    print(f"      💾 Saved: {save_path}")
    
    # ⬇️ Auto-download for Colab User (Safety Net)
    try:
        from google.colab import files
        files.download(save_path)
        print(f"      ⬇️ Auto-downloading {save_path}...")
    except:
        pass

print("✅ Retraining Complete.")
