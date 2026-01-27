"""
CondorBrain Sweep Runner (Memory-Optimized)

Runs multiple training configurations sequentially using a LazySequenceDataset.
- Loads raw CSV (~1-2GB)
- Creates sequences on-the-fly (Zero memory/disk explosion)
- Fast training with parallel DataLoaders
"""
import sys
import sys
import os
# Fix fragmentation (Chunk Memory)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from intelligence.indicators.manifold_volatility import (
    curvature_proxy_from_returns,
    volatility_energy_from_curvature,
    dynamic_rsi,
)

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain import CondorBrain, CondorLoss, HAS_MAMBA

# ============================================================================
# CUDA OPTIMIZATIONS (Release the H100 Kraken)
# ============================================================================
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+
torch.backends.cudnn.allow_tf32 = True        # Allow TF32 on cuDNN
torch.backends.cudnn.benchmark = True         # Auto-tuner for fastest kernels
torch.backends.cudnn.deterministic = False    # Allow non-deterministic speedups
print(f"[System] CUDA Optimizations Enabled: TF32=True, Benchmark=True")

# ============================================================================
# LAZY DATASET (Critical for Memory Efficiency)
# ============================================================================

class LazySequenceDataset(Dataset):
    """
    Zero-copy dataset that slices sequences on-the-fly.
    Drastically reduces RAM usage compared to materializing all sequences.
    """
    def __init__(self, features, targets, regimes, lookback):
        self.features = features  # (N, F) float32 array
        self.targets = targets    # (N, T) float32 array
        self.regimes = regimes    # (N,) int64 array
        self.lookback = lookback
        # number of valid sequences is Total - Lookback
        self.length = len(features) - lookback 
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # x: input sequence [idx : idx+lookback]
        # y: target [idx+lookback] (prediction for next step or aligned step)
        # r: regime [idx+lookback]
        
        # NOTE: features is numpy array, slicing is fast view
        x = self.features[idx : idx + self.lookback]
        y = self.targets[idx + self.lookback]
        r = self.regimes[idx + self.lookback]
        
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(r, dtype=torch.long)

# ============================================================================
# DATA PREP (Standard Sanitization)
# ============================================================================

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 
    'strike', 'cp_num', 'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te',
    'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark'
]

def load_and_prep_data(csv_path, max_rows=0):
    print(f"[Data] Loading {csv_path}...")
    if max_rows > 0:
        df = pd.read_csv(csv_path, nrows=max_rows)
    else:
        df = pd.read_csv(csv_path)
        
    # Standard CondorBrain prep
    print("[Data] Processing features...")
    if 'call_put' in df.columns and 'cp_num' not in df.columns:
        df['cp_num'] = df['call_put'].map({'C': 1.0, 'P': -1.0}).fillna(0)

    if 'close' in df.columns:
        log_ret = np.log(df['close']).diff()
        curvature = curvature_proxy_from_returns(log_ret, span=64)
        vol_energy = volatility_energy_from_curvature(curvature)
        df['rsi_dyn'] = dynamic_rsi(df['close'], window=14, vol_energy=vol_energy)
        df['rsi'] = df['rsi_dyn']
        
    # Synthetic Targets (if missing)
    defaults = {
        'target_call_offset': 2.0, 'target_put_offset': 2.0, 'target_wing_width': 5.0,
        'target_dte': 14.0, 'was_profitable': 0.5, 'realized_roi': 0.0,
        'realized_max_loss': 0.2, 'confidence_target': 0.5
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
            
    # Regime
    if 'regime_label' not in df.columns:
        if 'ivr' in df.columns:
            df['regime_label'] = pd.cut(df['ivr'].fillna(50), bins=[-0.1, 30, 70, 101], labels=[0, 1, 2]).fillna(1).astype(int)
        else:
            df['regime_label'] = 1
            
    df = df.ffill().bfill().fillna(0)
    
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target'
    ]
    
    # Numpy conversion
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    regime = df['regime_label'].values.astype(np.int64)
    
    # Sanitization
    print("[Data] Sanitizing...")
    X = np.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5)
    y = np.nan_to_num(y, nan=0.0, posinf=1e5, neginf=-1e5)
    if X.shape[1] > 4:
        X[:, 4] = np.log1p(X[:, 4]) # Volume log scale
        
    # SCALING (Critical for Neural Net Stability)
    print("Stats before scaling:")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

    # Robust Feature Scaling (Mean=0, Std=1)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std
    
    # Clip after scaling to ensure no outliers break FP16
    X = np.clip(X, -10.0, 10.0) 
    
    # Scale huge targets? 
    # ROI, MaxLoss, Prob are small (0-1). 
    # Offsets (2.0) are small.
    # Wing Width (5.0) is small.
    # DTE (14.0) is medium.
    # Only need to be careful if targets are huge.
    
    print("Stats after scaling:")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    
    print(f"[Success] Loaded {len(X):,} rows into RAM.")
    return X, y, regime

# ============================================================================
# TRAIN LOOP
# ============================================================================

def train_one_config(args, config, datasets, device):
    d_model = config['d_model']
    layers = config['layers']
    lr = config['lr']
    batch = config['batch_size']
    
    train_ds, val_ds = datasets
    
    print(f"\n{'='*60}")
    print(f"RUNNING CONFIG: {d_model}d x {layers}L | LR: {lr} | Batch: {batch}")
    print(f"{'='*60}")
    
    # Fast Loaders with Persistent Workers
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, 
                              num_workers=4, persistent_workers=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, 
                            num_workers=2, persistent_workers=True, pin_memory=True)
    
    model = CondorBrain(d_model=d_model, n_layers=layers, input_dim=len(FEATURE_COLS)).to(device)
    
    # RESUME LOGIC
    if hasattr(args, 'resume_from') and args.resume_from:
        print(f"[Resume] Loading weights from {args.resume_from}...")
        try:
            state_dict = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(state_dict)
            print("[Resume] Success! Weights loaded.")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint: {e}")
            
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {n_params:,}")
    
    criterion = CondorLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    best_loss = float('inf')
    output_path = f"models/sweep/condor_d{d_model}_L{layers}_lr{lr:.0e}.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            batches = 0
            
            for bx, by, br in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
                bx, by, br = bx.to(device, non_blocking=True), by.to(device, non_blocking=True), br.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    out, r_logits, _ = model(bx, return_regime=True)
                    loss = criterion(out, by, r_logits, br) # Pass logits to criterion
                    
                if torch.isnan(loss) or torch.isinf(loss): continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                batches += 1
                
            # Val
            model.eval()
            val_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for bx, by, br in val_loader:
                    bx, by, br = bx.to(device, non_blocking=True), by.to(device, non_blocking=True), br.to(device, non_blocking=True)
                    with autocast('cuda'):
                        out, r_logits, _ = model(bx, return_regime=True)
                        loss = criterion(out, by, r_logits, br)
                    val_loss += loss.item()
                    v_batches += 1
                    
            train_loss /= max(batches, 1)
            val_loss /= max(v_batches, 1)
            scheduler.step()
            
            print(f"Epoch {epoch+1:2d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            if val_loss < best_loss and not np.isnan(val_loss):
                best_loss = val_loss
                torch.save(model.state_dict(), output_path)
                
    except KeyboardInterrupt:
        print("\n\n[STOP] Training interrupted by user!")
        emerg_path = output_path.replace(".pth", "_EMERGENCY.pth")
        print(f"[STOP] Saving emergency checkpoint to: {emerg_path}")
        torch.save(model.state_dict(), emerg_path)
        print("[STOP] Saved. safe to exit.")
        return best_loss

    print(f"✓ FINISHED. Best: {best_loss:.4f} -> {output_path}")
    return best_loss

# --- EVOLUTIONARY INTELLIGENCE ---
class EvolutionaryOptimizer:
    def __init__(self, iterations=20):
        self.iterations = iterations
        self.history = []  # List of {config, loss}
        self.best_loss = float('inf')
        self.best_config = None
        
        # 1. BASELINE: "The Beast" (Max Horsepower)
        self.queue = [{
            'd_model': 512,  # Sweet Spot for stability
            'layers': 32,    # Deep Reasoning
            'lr': 1e-4, 
            'batch_size': 1024,
            'lookback': 120,
            'id': 'Gen0_Baseline'
        }]
        
    def get_next_config(self, iteration):
        if self.queue:
            return self.queue.pop(0)

        # 2. INTELLIGENCE: Adapt based on history
        # Simple Mutational Strategy
        parent = self.best_config if self.best_config else self.history[-1]['config']
        new_config = parent.copy()
        new_config['id'] = f'Gen{iteration}_Mutant'
        
        # Mutation Logic (Bayesian Exploration)
        import random
        mutation_type = random.choice(['lr', 'depth', 'width', 'lookback'])
        
        if mutation_type == 'lr':
            # Explore Gradient Landscape
            factor = random.choice([0.5, 2.0])
            new_config['lr'] = np.clip(new_config['lr'] * factor, 1e-5, 1e-3)
            
        elif mutation_type == 'depth':
             # Vary reasoning depth
            delta = random.choice([-4, 4])
            new_config['layers'] = int(np.clip(new_config['layers'] + delta, 12, 48))
            
        elif mutation_type == 'width':
            # Vary capacity
            modes = [256, 512, 1024]
            current_idx = modes.index(new_config['d_model']) if new_config['d_model'] in modes else 1
            move = random.choice([-1, 1])
            new_idx = np.clip(current_idx + move, 0, 2)
            new_config['d_model'] = modes[new_idx]
            
        elif mutation_type == 'lookback':
            # Vary Context Window
            delta = random.choice([-30, 30])
            new_config['lookback'] = int(np.clip(new_config['lookback'] + delta, 60, 240))

        print(f"\n[Evolution] Mutating {mutation_type.upper()} -> {new_config}")
        return new_config

    def update(self, config, loss):
        entry = {'config': config, 'loss': loss}
        self.history.append(entry)
        
        if loss < self.best_loss:
            print(f"*** NEW BEST MODEL FOUND! Loss: {loss:.4f} ***")
            self.best_loss = loss
            self.best_config = config
        else:
            print(f"[Evolution] Degraded (Best: {self.best_loss:.4f})")

# --- EVOLUTIONARY INTELLIGENCE ---
class EvolutionaryOptimizer:
    def __init__(self, iterations=20):
        self.iterations = iterations
        self.history = []  # List of {config, loss}
        self.best_loss = float('inf')
        self.best_config = None
        
        # 1. BASELINE: "The Beast" (Max Horsepower)
        self.queue = [{
            'd_model': 512,  # Sweet Spot for stability
            'layers': 32,    # Deep Reasoning
            'lr': 1e-4, 
            'batch_size': 1024,
            'lookback': 120,
            'id': 'Gen0_Baseline'
        }]
        
    def get_next_config(self, iteration):
        if self.queue:
            return self.queue.pop(0)

        # 2. INTELLIGENCE: Adapt based on history
        # Simple Mutational Strategy
        parent = self.best_config if self.best_config else self.history[-1]['config']
        new_config = parent.copy()
        new_config['id'] = f'Gen{iteration}_Mutant'
        
        # Mutation Logic (Bayesian Exploration)
        import random
        mutation_type = random.choice(['lr', 'depth', 'width', 'lookback'])
        
        if mutation_type == 'lr':
            # Explore Gradient Landscape
            factor = random.choice([0.5, 2.0])
            new_config['lr'] = np.clip(new_config['lr'] * factor, 1e-5, 1e-3)
            
        elif mutation_type == 'depth':
             # Vary reasoning depth
            delta = random.choice([-4, 4])
            new_config['layers'] = int(np.clip(new_config['layers'] + delta, 12, 48))
            
        elif mutation_type == 'width':
            # Vary capacity
            modes = [256, 512, 1024]
            current_idx = modes.index(new_config['d_model']) if new_config['d_model'] in modes else 1
            move = random.choice([-1, 1])
            new_idx = np.clip(current_idx + move, 0, 2)
            new_config['d_model'] = modes[new_idx]
            
        elif mutation_type == 'lookback':
            # Vary Context Window
            delta = random.choice([-30, 30])
            new_config['lookback'] = int(np.clip(new_config['lookback'] + delta, 60, 240))

        print(f"\n[Evolution] Mutating {mutation_type.upper()} -> {new_config}")
        return new_config

    def update(self, config, loss):
        entry = {'config': config, 'loss': loss}
        self.history.append(entry)
        
        if loss < self.best_loss:
            print(f"*** NEW BEST MODEL FOUND! Loss: {loss:.4f} ***")
            self.best_loss = loss
            self.best_config = config
        else:
            print(f"[Evolution] Degraded (Best: {self.best_loss:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for faster sweep")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Evolution
    optimizer = EvolutionaryOptimizer(iterations=20)
    
    # Load Data ONCE (Raw Arrays)
    # We load with a max lookback to be safe, or just load everything
    print("Loading Data into Shared Memory...")
    # Use helper load function to get arrays
    X, y, r = load_and_prep_data(args.csv, max_rows=args.max_rows)
    
    print("\n--- STARTING 20-ITERATION EVOLUTIONARY SWEEP ---")
    
    for i in range(20): 
        cfg = optimizer.get_next_config(i)
        
        # Override lookback if config says so, else use args
        current_lookback = cfg.get('lookback', 120)
        args.lookback = current_lookback
        
        # Dynamic Dataset Creation (O(1) View)
        # Recalculate split based on new lookback
        n_total_seq = len(X) - current_lookback
        n_train = int(0.8 * n_total_seq)
        
        X_train = X[:n_train + current_lookback]
        y_train = y[:n_train + current_lookback]
        r_train = r[:n_train + current_lookback]
        
        X_val = X[n_train:]
        y_val = y[n_train:]
        r_val = r[n_train:]
        
        train_ds = LazySequenceDataset(X_train, y_train, r_train, current_lookback)
        val_ds = LazySequenceDataset(X_val, y_val, r_val, current_lookback)
        
        print(f"\n{'='*60}")
        print(f"ITERATION {i+1}/20: {cfg['id']}")
        print(f"CONFIG: {cfg}")
        print(f"Seq Len: {len(train_ds):,} Train | {len(val_ds):,} Val")
        print(f"{'='*60}")
        
        # Force epochs=3 for sweep (or use args.epochs if user wants control)
        # User said "run all 20 models...". 
        # We'll use args.epochs passed by user (default 5).
        
        try:
            loss = train_one_config(args, cfg, (train_ds, val_ds), device)
        except Exception as e:
            print(f"[ERROR] Config failed: {e}")
            loss = float('inf')
            
        optimizer.update(cfg, loss)
        
        # Log Progress
        pd.DataFrame([x['config'] | {'loss': x['loss']} for x in optimizer.history]).to_csv("models/sweep/evolution_log.csv")

    print("\n--- EVOLUTION COMPLETE ---")
    if optimizer.best_config:
        print(f"Best Configuration: {optimizer.best_config}")
        print(f"Best Loss: {optimizer.best_loss:.4f}")

if __name__ == "__main__":
    main()
