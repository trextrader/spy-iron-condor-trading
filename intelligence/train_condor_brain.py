"""
CondorBrain Training Script (Fast In-Memory)

Uses proven pattern from train_mamba.py:
- Load all data into memory (works for datasets that fit in RAM)
- Create sequences with numpy
- Use TensorDataset + DataLoader for training
"""
import sys
import os
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
    
    # Regime labeling
    if 'regime_label' not in df.columns:
        if 'ivr' in df.columns:
            df['regime_label'] = pd.cut(
                df['ivr'], 
                bins=[-0.1, 30, 70, 101], 
                labels=[0, 1, 2]
            ).astype(int)
        else:
            df['regime_label'] = 1
    
    # Fill NaNs
    df = df.ffill().fillna(0)
    
    # Build arrays
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target'
    ]
    
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    regime = df['regime_label'].values.astype(np.int64)
    
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
    
    if device.type == 'cuda':
        print(f"[CondorBrain] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[CondorBrain] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
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
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(r_train)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(r_val)
        ),
        batch_size=args.batch_size,
        pin_memory=True
    )
    
    # Model
    model = CondorBrain(
        d_model=args.d_model,
        n_layers=args.layers,
        input_dim=len(FEATURE_COLS)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CondorBrain] Model parameters: {n_params:,}")
    
    criterion = CondorLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Training
    print(f"\n{'='*60}")
    print(f"CONDORBRAIN TRAINING")
    print(f"{'='*60}")
    print(f"Model: {args.d_model}d x {args.layers} layers ({n_params/1e6:.1f}M params)")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        epoch_start = time.time()
        
        for batch_x, batch_y, batch_r in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_r = batch_r.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                loss = criterion(outputs, batch_y, regime_probs, batch_r)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_r in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_r = batch_r.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                    loss = criterion(outputs, batch_y, regime_probs, batch_r)
                val_loss += loss.item()
                n_val_batches += 1
        
        train_loss /= max(n_batches, 1)
        val_loss /= max(n_val_batches, 1)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        if device.type == 'cuda':
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {lr:.2e} | Time: {epoch_time:.1f}s | GPU: {gpu_mem:.1f}GB")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(args.output) or 'models', exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        
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
