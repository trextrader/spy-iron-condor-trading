"""
Prepare Tensors for CondorBrain Sweep

Loads CSV data, sanitizes it (matching train_condor_brain.py), creates sequences,
and saves everything to a single .pt file for instant GPU loading.
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch

# Configuration matching train_condor_brain.py
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 
    'strike', 'cp_num', 'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te',
    'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark'
]

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and targets with strict sanitization."""
    print("[Preprocess] Preparing features...")
    
    # Handle call_put encoding
    if 'call_put' in df.columns and 'cp_num' not in df.columns:
        df['cp_num'] = df['call_put'].map({'C': 1.0, 'P': -1.0}).fillna(0)
    
    # Generate synthetic targets (default offsets) if missing
    defaults = {
        'target_call_offset': 2.0,
        'target_put_offset': 2.0,
        'target_wing_width': 5.0,
        'target_dte': 14.0,
        'was_profitable': 0.5,
        'realized_roi': 0.0,
        'realized_max_loss': 0.2,
        'confidence_target': 0.5
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    
    # Regime labeling (with NaN safety)
    if 'regime_label' not in df.columns:
        if 'ivr' in df.columns:
            df['regime_label'] = pd.cut(
                df['ivr'].fillna(50), 
                bins=[-0.1, 30, 70, 101], 
                labels=[0, 1, 2]
            ).fillna(1).astype(int)
        else:
            df['regime_label'] = 1
    
    # Fill ALL NaNs
    df = df.ffill().bfill().fillna(0)
    
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target'
    ]
    
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    regime = df['regime_label'].values.astype(np.int64)
    
    # SANITIZATION (Strict)
    print("[Preprocess] Sanitizing data...")
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Scale volume (index 4)
    if X.shape[1] > 4:
        X[:, 4] = np.log1p(X[:, 4])
        
    # Clamp extremes
    X = np.clip(X, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)
    
    return X, y, regime

def create_sequences_fast(X, y, regime, lookback):
    """Create sequences using numpy stride tricks."""
    n_samples = len(X) - lookback
    n_features = X.shape[1]
    
    print(f"[Preprocess] Creating {n_samples:,} sequences (lookback={lookback})...")
    
    from numpy.lib.stride_tricks import as_strided
    X_seq = as_strided(
        X,
        shape=(n_samples, lookback, n_features),
        strides=(X.strides[0], X.strides[0], X.strides[1])
    ).copy()
    
    y_seq = y[lookback:].copy()
    r_seq = regime[lookback:].copy()
    
    return X_seq, y_seq, r_seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV")
    parser.add_argument("--output", default="data/processed/condor_tensors.pt", help="Output .pt file")
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    
    print(f"[Preprocess] Loading {args.csv}...")
    if args.max_rows > 0:
        df = pd.read_csv(args.csv, nrows=args.max_rows)
    else:
        df = pd.read_csv(args.csv)
        
    X, y, r = prepare_features(df)
    
    # Garbage collect
    del df
    import gc
    gc.collect()
    
    X_seq, y_seq, r_seq = create_sequences_fast(X, y, r, args.lookback)
    
    # Split 80/20
    split = int(0.8 * len(X_seq))
    
    print("[Preprocess] Saving tensors...")
    data = {
        'X_train': torch.from_numpy(X_seq[:split]),
        'y_train': torch.from_numpy(y_seq[:split]),
        'r_train': torch.from_numpy(r_seq[:split]),
        'X_val': torch.from_numpy(X_seq[split:]),
        'y_val': torch.from_numpy(y_seq[split:]),
        'r_val': torch.from_numpy(r_seq[split:])
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(data, args.output)
    print(f"[Success] Saved to {args.output}")
    print(f"Train: {data['X_train'].shape}")
    print(f"Val:   {data['X_val'].shape}")

if __name__ == "__main__":
    main()
