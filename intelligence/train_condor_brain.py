"""
CondorBrain Training Script (Fast In-Memory)

Uses proven pattern from train_mamba.py:
- Load all data into memory (works for datasets that fit in RAM)
- Create sequences with numpy
- Use TensorDataset + DataLoader for training
"""
import sys
import os

# CRITICAL: Set CUDA memory allocator BEFORE importing torch
# Note: Only use expandable_segments - other options may conflict with PyTorch versions
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain import CondorBrain, CondorLoss, HAS_MAMBA

# ============================================================================
# ROBUST NORMALIZATION HELPERS (Fixes NaN training issue)
# ============================================================================

EPS = 1e-6

def safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    """Never replace inf with huge constants. Replace NaN/Inf with 0."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def robust_zscore_fit(X: np.ndarray):
    """Robust scaler using median + MAD (much safer than mean/std on heavy tails)."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad  # Convert MAD to std-like scale
    scale = np.where(scale < EPS, 1.0, scale)
    return med.astype(np.float32), scale.astype(np.float32)

def robust_zscore_transform(X: np.ndarray, med: np.ndarray, scale: np.ndarray, clip_val: float = 10.0) -> np.ndarray:
    """Apply robust z-score and clip to prevent extreme values."""
    X = (X - med) / (scale + EPS)
    X = np.clip(X, -clip_val, clip_val)
    return X.astype(np.float32)

def clamp_targets(y: np.ndarray) -> np.ndarray:
    """Clamp targets to reasonable ranges to prevent BF16 overflow."""
    y = safe_nan_to_num(y)
    # Target ordering: call_offset, put_offset, wing_width, dte, pop, roi, max_loss, confidence
    y[:, 0] = np.clip(y[:, 0], -100.0, 100.0)  # call offset
    y[:, 1] = np.clip(y[:, 1], -100.0, 100.0)  # put offset
    y[:, 2] = np.clip(y[:, 2], 0.5, 50.0)       # wing width
    y[:, 3] = np.clip(y[:, 3], 0.0, 120.0)      # dte
    y[:, 4] = np.clip(y[:, 4], 0.0, 1.0)        # pop (probability)
    y[:, 5] = np.clip(y[:, 5], -5.0, 5.0)       # expected roi
    y[:, 6] = np.clip(y[:, 6], 0.0, 5.0)        # max loss
    y[:, 7] = np.clip(y[:, 7], 0.0, 1.0)        # confidence
    return y.astype(np.float32)

def debug_tensor_stats(name: str, X: np.ndarray):
    """Print tensor statistics for debugging NaN issues."""
    finite = np.isfinite(X)
    bad = np.size(X) - finite.sum()
    mx = np.max(np.abs(X[finite])) if finite.any() else np.nan
    print(f"[DEBUG] {name}: shape={X.shape} bad={bad} max|x|={mx:.4f}")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'd_model': 1024,
    'n_layers': 32,
    'lookback': 240,
    'batch_size': 64,
    'epochs': 100,
    'lr': 1e-4,
    'model_path': 'models/condor_brain.pth'
}

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 
    'strike', 'cp_num', 'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te',
    'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark'
]

# ============================================================================
# DATA PREPARATION (Fast In-Memory)
# ============================================================================

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and targets for CondorBrain training."""
    print("[CondorBrain] Preparing features...")
    
    # Handle call_put encoding
    if 'call_put' in df.columns and 'cp_num' not in df.columns:
        df['cp_num'] = df['call_put'].map({'C': 1.0, 'P': -1.0}).fillna(0)
    
    # Generate synthetic targets (default offsets)
    df['target_call_offset'] = 2.0
    df['target_put_offset'] = 2.0
    df['target_wing_width'] = 5.0
    df['target_dte'] = 14.0
    df['was_profitable'] = 0.5
    df['realized_roi'] = 0.0
    df['realized_max_loss'] = 0.2
    df['confidence_target'] = 0.5
    
    # Regime labeling (with NaN safety)
    if 'regime_label' not in df.columns:
        if 'ivr' in df.columns:
            df['regime_label'] = pd.cut(
                df['ivr'].fillna(50),  # Default to normal regime
                bins=[-0.1, 30, 70, 101], 
                labels=[0, 1, 2]
            ).fillna(1).astype(int)  # Default to normal (1)
        else:
            df['regime_label'] = 1
    
    # Fill ALL NaNs
    df = df.ffill().bfill().fillna(0)
    
    # Build arrays
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target'
    ]
    
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    regime = df['regime_label'].values.astype(np.int64)
    
    # ROBUST SANITIZATION (never use 1e6 - it explodes in BF16!)
    print("[CondorBrain] Sanitizing data (safe)...")
    X = safe_nan_to_num(X)
    y = clamp_targets(y)
    
    # Scale volume (index 4) to avoid huge gradients
    if X.shape[1] > 4:
        X[:, 4] = np.log1p(np.clip(X[:, 4], 0.0, 1e9)).astype(np.float32)
    
    # ROBUST NORMALIZATION (median/MAD + clip to ±10)
    print("[CondorBrain] Robust-normalizing features (median/MAD)...")
    med, scale = robust_zscore_fit(X)
    X = robust_zscore_transform(X, med, scale, clip_val=10.0)
    
    # Debug output
    debug_tensor_stats("X_scaled", X)
    debug_tensor_stats("y_clamped", y)
    
    # Sanity check: confirm dtypes and ranges
    print(f"[SANITY] X dtype: {X.dtype}, min/max: {X.min():.4f}/{X.max():.4f}")
    print(f"[SANITY] y dtype: {y.dtype}, min/max: {y.min():.4f}/{y.max():.4f}")
    
    # Verify clean
    if np.isnan(X).any() or np.isinf(X).any():
        raise RuntimeError("[CRITICAL ERROR] X still contains NaN/Inf after sanitization!")
    if np.isnan(y).any() or np.isinf(y).any():
        raise RuntimeError("[CRITICAL ERROR] y still contains NaN/Inf after sanitization!")
    
    print(f"[CondorBrain] Features: {X.shape}, Targets: {y.shape}")
    return X, y, regime


def create_sequences_fast(X, y, regime, lookback):
    """Create sequences using numpy stride tricks for speed."""
    n_samples = len(X) - lookback
    n_features = X.shape[1]
    
    print(f"[CondorBrain] Creating {n_samples:,} sequences (lookback={lookback})...")
    
    # Use stride tricks for efficient view (no memory copy!)
    from numpy.lib.stride_tricks import as_strided
    
    X_seq = as_strided(
        X,
        shape=(n_samples, lookback, n_features),
        strides=(X.strides[0], X.strides[0], X.strides[1])
    ).copy()  # Copy to ensure contiguous memory
    
    y_seq = y[lookback:].copy()
    r_seq = regime[lookback:].copy()
    
    return X_seq, y_seq, r_seq


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CondorBrain")
    parser.add_argument("--local-data", type=str, required=True, help="Path to institutional CSV")
    parser.add_argument("--output", type=str, default="auto", help="Output model path ('auto' = generate from params)")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for debugging")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps (effective_batch = batch_size * accum_steps)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing to save memory")
    
    args = parser.parse_args()
    
    # Auto-generate output filename from params
    if args.output == "auto":
        lr_str = f"{args.lr:.0e}".replace("-", "")
        args.output = f"models/condor_brain_e{args.epochs}_d{args.d_model}_L{args.layers}_lr{lr_str}.pth"
    
    return args


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_condor_brain(args):
    """Training with proven in-memory pattern."""
    
    if not HAS_MAMBA:
        print("[Error] mamba-ssm not available.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[CondorBrain] Device: {device}")
    
    # Detect GPU capabilities
    use_bf16 = False
    is_h100 = False
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        
        print(f"[CondorBrain] GPU: {gpu_name}")
        print(f"[CondorBrain] GPU Memory: {gpu_mem:.1f} GB")
        print(f"[CondorBrain] Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        
        # H100 detection (compute capability 9.0+)
        is_h100 = compute_cap[0] >= 9 or 'H100' in gpu_name
        # BF16 supported on Ampere+ (8.0+)
        use_bf16 = compute_cap[0] >= 8
        
        # === MAXIMUM CUDA OPTIMIZATIONS ===
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels
        torch.backends.cudnn.deterministic = False  # Speed over reproducibility
        torch.set_float32_matmul_precision('high')  # Use Tensor Cores for FP32
        
        if is_h100:
            print("[CondorBrain] 🚀 H100 DETECTED - Maximum optimizations enabled!")
            # H100 has 80GB HBM3, can handle larger batches
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        opt_str = f"TF32, cuDNN benchmark, {'BF16' if use_bf16 else 'FP16'}"
        print(f"[CondorBrain] CUDA optimizations enabled: {opt_str}")
    
    # Load data
    print(f"\n[CondorBrain] Loading data from {args.local_data}...")
    if args.max_rows > 0:
        df = pd.read_csv(args.local_data, nrows=args.max_rows)
    else:
        df = pd.read_csv(args.local_data)
    print(f"[CondorBrain] Loaded {len(df):,} rows")
    
    # Prepare features
    X, y, regime = prepare_features(df)
    
    # Free dataframe memory
    del df
    import gc
    gc.collect()
    
    # Create sequences (FAST with stride tricks)
    X_seq, y_seq, r_seq = create_sequences_fast(X, y, regime, args.lookback)
    
    # Free original arrays
    del X, y, regime
    gc.collect()
    
    # Train/Val split (80/20)
    split = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]
    r_train, r_val = r_seq[:split], r_seq[split:]
    
    print(f"[CondorBrain] Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # DataLoaders with maximum parallel loading
    # H100 can handle more workers due to faster PCIe 5.0
    n_workers = 8 if is_h100 else 4
    
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(r_train)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=n_workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4  # Prefetch 4 batches per worker
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(r_val)
        ),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=n_workers // 2,
        prefetch_factor=2
    )
    
    # Model
    model = CondorBrain(
        d_model=args.d_model,
        n_layers=args.layers,
        input_dim=len(FEATURE_COLS)
    ).to(device)
    
    # Enable gradient checkpointing if requested (saves ~40% GPU memory)
    if args.grad_checkpoint:
        model.gradient_checkpointing = True
        print("[CondorBrain] Gradient checkpointing ENABLED (memory saver)")
    
    # NOTE: torch.compile() disabled - incompatible with Mamba's custom selective_scan_cuda kernels
    # Mamba already uses optimized Triton kernels internally
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CondorBrain] Model parameters: {n_params:,}")

    
    criterion = CondorLoss()
    
    # Fused AdamW is faster on modern GPUs (avoids kernel launch overhead)
    use_fused = device.type == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        fused=use_fused if use_fused else False
    )
    if use_fused:
        print("[CondorBrain] Using fused AdamW optimizer")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Use BF16 on H100/Ampere, FP16 on older GPUs
    scaler = GradScaler() if not use_bf16 else None
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Training
    effective_batch = args.batch_size * args.accum_steps
    print(f"\n{'='*60}")
    print(f"CONDORBRAIN TRAINING")
    print(f"{'='*60}")
    print(f"Model: {args.d_model}d x {args.layers} layers ({n_params/1e6:.1f}M params)")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size} x {args.accum_steps} accum = {effective_batch} effective")
    print(f"LR: {args.lr}, Grad Checkpoint: {args.grad_checkpoint}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)  # Zero at start of epoch
        
        for batch_idx, (batch_x, batch_y, batch_r) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_r = batch_r.to(device, non_blocking=True)
            
            with autocast('cuda', dtype=amp_dtype):
                outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                
                # Log dtype on first batch (sanity check)
                if batch_idx == 0 and epoch == 0:
                    print(f"[SANITY] outputs dtype: {outputs.dtype}, loss will compute in: float32")
                
                # HARD CHECK: Fail immediately on NaN outputs (don't silently skip!)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"[NaN ERROR] outputs contained NaN/Inf at batch {batch_idx}")
                    print(f"  batch_x max: {torch.max(torch.abs(batch_x)).item():.4f}")
                    print(f"  outputs max: {torch.max(torch.abs(outputs[~torch.isnan(outputs)])).item() if (~torch.isnan(outputs)).any() else 'ALL NaN'}")
                    raise RuntimeError("Model produced NaN/Inf outputs - check data normalization!")
                
                loss = criterion(outputs.float(), batch_y.float(), regime_probs.float(), batch_r)
                # Scale loss for gradient accumulation
                loss = loss / args.accum_steps
            
            # Hard check on loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[NaN ERROR] Loss is NaN/Inf at batch {batch_idx}")
                print(f"  outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"  batch_y range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
                raise RuntimeError("Loss is NaN/Inf - check loss function!")
            
            # Backward pass (accumulate gradients)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step every accum_steps batches
            if (batch_idx + 1) % args.accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * args.accum_steps  # Un-scale for logging
            n_batches += 1
        
        # Flush leftover gradients if epoch ended mid-accumulation
        if (batch_idx + 1) % args.accum_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Validation (with progress bar)
        print(f"[Epoch {epoch+1}] Running validation...")
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_r in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_r = batch_r.to(device, non_blocking=True)
                
                with autocast('cuda', dtype=amp_dtype):
                    outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                    loss = criterion(outputs.float(), batch_y.float(), regime_probs.float(), batch_r)
                val_loss += loss.item()
                n_val_batches += 1
        
        train_loss /= max(n_batches, 1)
        val_loss /= max(n_val_batches, 1)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        if device.type == 'cuda':
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {lr:.2e} | Time: {epoch_time:.1f}s | GPU alloc: {alloc:.1f}GB | reserved: {reserved:.1f}GB")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(args.output) or 'models', exist_ok=True)
            print(f"  [Saving] Writing model to {args.output}...")
            save_start = time.time()
            torch.save(model.state_dict(), args.output)
            save_time = time.time() - save_start
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f}) in {save_time:.1f}s")
        
        # Clear cache periodically
        if device.type == 'cuda' and epoch % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"{'='*60}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    train_condor_brain(args)
