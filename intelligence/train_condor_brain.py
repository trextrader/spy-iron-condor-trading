"""
CondorBrain Training Script (GPU-Optimized)

A100-optimized training with:
- Streaming data loading (no full dataset in CPU RAM)
- Pinned memory for fast CPU→GPU transfer
- On-GPU sequence creation
- CUDA prefetching
"""
import sys
import os
import time
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.amp import autocast, GradScaler

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

# ============================================================================
# GPU-OPTIMIZED STREAMING DATASET
# ============================================================================

class StreamingCondorDataset(IterableDataset):
    """
    Memory-efficient streaming dataset that:
    - Reads CSV in chunks (not all at once)
    - Creates sequences on-the-fly
    - Uses pinned memory for fast GPU transfer
    """
    
    def __init__(
        self, 
        csv_path: str, 
        lookback: int = 240, 
        chunk_size: int = 100000,
        max_rows: int = 0,
        feature_cols: list = None,
        is_train: bool = True,
        train_ratio: float = 0.8
    ):
        self.csv_path = csv_path
        self.lookback = lookback
        self.chunk_size = chunk_size
        self.max_rows = max_rows
        self.is_train = is_train
        self.train_ratio = train_ratio
        
        self.feature_cols = feature_cols or [
            'open', 'high', 'low', 'close', 'volume', 
            'strike', 'cp_num', 'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te',
            'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark'
        ]
        
        self.target_cols = [
            'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
            'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target'
        ]
        
        # Count total rows (quick scan)
        self._count_rows()
        
    def _count_rows(self):
        """Quick row count without loading data."""
        with open(self.csv_path, 'r') as f:
            self.total_rows = sum(1 for _ in f) - 1  # Minus header
        if self.max_rows > 0:
            self.total_rows = min(self.total_rows, self.max_rows)
        
        # Train/val split point
        self.split_row = int(self.total_rows * self.train_ratio)
        
        if self.is_train:
            self.start_row = 0
            self.end_row = self.split_row
        else:
            self.start_row = self.split_row
            self.end_row = self.total_rows
            
        self.n_samples = self.end_row - self.start_row - self.lookback
        print(f"[StreamingDataset] {'Train' if self.is_train else 'Val'}: {self.n_samples:,} samples")
    
    def __len__(self):
        return self.n_samples
    
    def __iter__(self):
        """Stream chunks and yield sequences."""
        buffer = None
        buffer_start_idx = 0
        
        # Read CSV in chunks
        reader = pd.read_csv(
            self.csv_path, 
            chunksize=self.chunk_size,
            nrows=self.max_rows if self.max_rows > 0 else None
        )
        
        current_idx = 0
        
        for chunk in reader:
            # Prepare chunk
            chunk = self._prepare_chunk(chunk)
            
            # Determine if chunk is in our split range
            chunk_end = current_idx + len(chunk)
            
            if chunk_end < self.start_row:
                current_idx = chunk_end
                continue
            
            if current_idx > self.end_row:
                break
            
            # Maintain rolling buffer for lookback
            if buffer is None:
                buffer = chunk
                buffer_start_idx = current_idx
            else:
                buffer = pd.concat([buffer, chunk], ignore_index=True)
                # Trim buffer to save memory (keep only what we need)
                if len(buffer) > self.chunk_size + self.lookback:
                    trim_amount = len(buffer) - self.chunk_size - self.lookback
                    buffer = buffer.iloc[trim_amount:].reset_index(drop=True)
                    buffer_start_idx += trim_amount
            
            # Generate sequences from buffer
            for i in range(max(0, self.start_row - buffer_start_idx), min(len(buffer) - self.lookback, self.end_row - buffer_start_idx)):
                seq_start = i
                seq_end = i + self.lookback
                
                X_seq = buffer[self.feature_cols].iloc[seq_start:seq_end].values.astype(np.float32)
                y = buffer[self.target_cols].iloc[seq_end].values.astype(np.float32)
                regime = int(buffer['regime_label'].iloc[seq_end])
                
                yield torch.from_numpy(X_seq), torch.from_numpy(y), torch.tensor(regime, dtype=torch.long)
            
            current_idx = chunk_end
        
    def _prepare_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare a single chunk."""
        # Handle call_put encoding
        if 'call_put' in df.columns and 'cp_num' not in df.columns:
            df['cp_num'] = df['call_put'].map({'C': 1.0, 'P': -1.0}).fillna(0)
        
        # Generate synthetic targets if not present
        for col in self.target_cols:
            if col not in df.columns:
                if col == 'target_call_offset':
                    df[col] = 2.0
                elif col == 'target_put_offset':
                    df[col] = 2.0
                elif col == 'target_wing_width':
                    df[col] = 5.0
                elif col == 'target_dte':
                    df[col] = 14.0
                elif col == 'was_profitable':
                    df[col] = 0.5
                elif col == 'realized_roi':
                    df[col] = 0.0
                elif col == 'realized_max_loss':
                    df[col] = 0.2
                elif col == 'confidence_target':
                    df[col] = 0.5
        
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
        
        return df


# ============================================================================
# CUDA-OPTIMIZED DATA LOADER
# ============================================================================

def create_gpu_dataloaders(args, device):
    """Create streaming dataloaders with CUDA prefetching."""
    
    train_dataset = StreamingCondorDataset(
        csv_path=args.local_data,
        lookback=args.lookback,
        chunk_size=50000,  # 50k rows per chunk
        max_rows=args.max_rows,
        is_train=True
    )
    
    val_dataset = StreamingCondorDataset(
        csv_path=args.local_data,
        lookback=args.lookback,
        chunk_size=50000,
        max_rows=args.max_rows,
        is_train=False
    )
    
    # Pinned memory for fast CPU→GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.n_samples


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CondorBrain (GPU-Optimized)")
    parser.add_argument("--local-data", type=str, required=True, help="Path to institutional CSV")
    parser.add_argument("--output", type=str, default="models/condor_brain.pth", help="Output model path")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)  # Larger batch for A100
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for debugging")
    return parser.parse_args()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_condor_brain(args):
    """GPU-optimized training."""
    
    if not HAS_MAMBA:
        print("[Error] mamba-ssm not available.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[CondorBrain] Device: {device}")
    
    if device.type == 'cuda':
        print(f"[CondorBrain] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[CondorBrain] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable TF32 for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create streaming dataloaders (minimal CPU RAM usage)
    print(f"\n[CondorBrain] Creating streaming dataloaders...")
    train_loader, val_loader, n_train_samples = create_gpu_dataloaders(args, device)
    
    # Model (created directly on GPU)
    print(f"[CondorBrain] Initializing model on GPU...")
    model = CondorBrain(
        d_model=args.d_model,
        n_layers=args.layers,
        input_dim=24
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CondorBrain] Model parameters: {n_params:,}")
    
    criterion = CondorLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Training
    print(f"\n{'='*60}")
    print(f"CONDORBRAIN A100 TRAINING")
    print(f"{'='*60}")
    print(f"Model: {args.d_model}d x {args.layers} layers ({n_params/1e6:.1f}M params)")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Samples: {n_train_samples:,}")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        epoch_start = time.time()
        
        for batch_x, batch_y, batch_r in train_loader:
            # Move to GPU (non-blocking with pinned memory)
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_r = batch_r.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
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
        
        # GPU memory stats
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
