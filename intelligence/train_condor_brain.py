"""
CondorBrain Training Script

Trains the advanced multi-output Mamba 2 architecture for Iron Condor optimization.
Requires backtest-enhanced targets (realized P&L, optimal strikes, etc.)
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
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CondorBrain Multi-Output Model")
    parser.add_argument("--local-data", type=str, required=True, help="Path to institutional CSV")
    parser.add_argument("--targets-data", type=str, help="Path to backtest-enhanced targets CSV")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for debugging")
    return parser.parse_args()

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_condor_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and targets for CondorBrain training.
    
    Expected columns in df:
    - Standard 24 institutional features
    - Target columns (if backtest-enhanced):
        - target_call_offset: Optimal short call offset that was profitable
        - target_put_offset: Optimal short put offset that was profitable
        - target_wing_width: Optimal wing width
        - target_dte: Optimal DTE
        - realized_roi: Actual P&L if IC was entered
        - was_profitable: Binary (1 if profitable)
        - realized_max_loss: Actual max loss experienced
        - regime_label: 0=Low, 1=Normal, 2=High
    """
    print("[CondorBrain] Preparing features...")
    
    # Feature columns (must match condor_brain.py)
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'strike', 'cp_num', 'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te',
        'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark'
    ]
    
    # Handle call_put encoding
    if 'call_put' in df.columns and 'cp_num' not in df.columns:
        df['cp_num'] = df['call_put'].map({'C': 1.0, 'P': -1.0}).fillna(0)
    
    # Check for target columns (backtest-enhanced mode)
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target'
    ]
    
    has_targets = all(col in df.columns for col in target_cols[:4])
    
    if has_targets:
        print("[CondorBrain] Using backtest-enhanced targets")
    else:
        print("[CondorBrain] Generating synthetic targets from spot data")
        # Generate synthetic targets based on price movement
        df['target_call_offset'] = 2.0  # Default 2% offset
        df['target_put_offset'] = 2.0
        df['target_wing_width'] = 5.0
        df['target_dte'] = 14.0
        
        # Use future returns to estimate profitability
        if 'target_spot' in df.columns:
            df['was_profitable'] = (df['target_spot'].abs() < 1.0).astype(float)  # Profitable if low movement
            df['realized_roi'] = df['target_spot'].clip(-20, 20) / 100.0 * -1  # Inverse: low vol = profit
        else:
            df['was_profitable'] = 0.5
            df['realized_roi'] = 0.0
            
        df['realized_max_loss'] = 0.2
        df['confidence_target'] = 0.5
    
    # Regime labeling (if not present)
    if 'regime_label' not in df.columns:
        # Use IVR to determine regime
        if 'ivr' in df.columns:
            df['regime_label'] = pd.cut(
                df['ivr'], 
                bins=[-0.1, 30, 70, 101], 
                labels=[0, 1, 2]
            ).astype(int)
        else:
            df['regime_label'] = 1  # Default to normal
    
    # Fill NaNs
    df = df.ffill().fillna(0)
    
    # Build target array (8 values matching CondorBrain output)
    target_array = df[[
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target'
    ]].values.astype(np.float32)
    
    regime_labels = df['regime_label'].values.astype(np.int64)
    
    # Feature array
    X = df[feature_cols].values.astype(np.float32)
    
    print(f"[CondorBrain] Features: {X.shape}, Targets: {target_array.shape}")
    return X, target_array, regime_labels

def create_sequences(X, y, regime, lookback):
    """Create sequences for temporal training."""
    X_seq, y_seq, r_seq = [], [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
        r_seq.append(regime[i+lookback])
    return np.array(X_seq), np.array(y_seq), np.array(r_seq)

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_condor_brain(args):
    """Main training function."""
    
    if not HAS_MAMBA:
        print("[Error] mamba-ssm not available. Training requires Mamba.")
        print("Install with: pip install mamba-ssm causal-conv1d")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[CondorBrain] Using device: {device}")
    
    # Load data
    print(f"[CondorBrain] Loading data from {args.local_data}...")
    if args.max_rows > 0:
        df = pd.read_csv(args.local_data, nrows=args.max_rows)
    else:
        df = pd.read_csv(args.local_data)
    print(f"[CondorBrain] Loaded {len(df):,} rows")
    
    # Prepare features
    X, y, regime = prepare_condor_features(df)
    
    # Create sequences
    print(f"[CondorBrain] Creating sequences (lookback={args.lookback})...")
    X_seq, y_seq, r_seq = create_sequences(X, y, regime, args.lookback)
    print(f"[CondorBrain] Sequences: {X_seq.shape}")
    
    # Train/Val split (80/20)
    split = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]
    r_train, r_val = r_seq[:split], r_seq[split:]
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(r_train)
        ),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(r_val)
        ),
        batch_size=args.batch_size
    )
    
    # Model
    model = CondorBrain(
        d_model=args.d_model,
        n_layers=args.layers,
        input_dim=X.shape[1]
    ).to(device)
    
    criterion = CondorLoss(
        strike_weight=1.0,
        pnl_weight=2.0,
        risk_weight=1.5,
        prob_weight=1.0,
        regime_weight=0.5
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Training
    print(f"\n{'='*60}")
    print(f"CONDORBRAIN TRAINING")
    print(f"{'='*60}")
    print(f"Model: {args.d_model}d x {args.layers} layers")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y, batch_r in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_r = batch_r.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                loss = criterion(outputs, batch_y, regime_probs, batch_r)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, batch_r in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_r = batch_r.to(device)
                
                with autocast('cuda'):
                    outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                    loss = criterion(outputs, batch_y, regime_probs, batch_r)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), DEFAULT_CONFIG['model_path'])
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Model saved to: {DEFAULT_CONFIG['model_path']}")
    print(f"{'='*60}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    train_condor_brain(args)
