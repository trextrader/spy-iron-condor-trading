
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas_ta as ta

# Try to import yfinance, else fail gracefully
try:
    import yfinance as yf
except ImportError:
    print("Please install yfinance: pip install yfinance")
    yf = None

from intelligence.mamba_engine import DeepMamba, HAS_MAMBA

# Configuration
CONFIG = {
    'symbol': 'SPY',
    'start_date': '2015-01-01',
    'end_date': '2024-12-31', # Train up to end of 2024
    'interval': '1h',         # 1-hour bars for stability (or 15m if data allows)
    'lookback': 60,
    'd_model': 256,           # Match Inference defaults (or 1024 if Pro)
    'layers': 6,
    'epochs': 20,
    'batch_size': 64,
    'lr': 1e-4,
    'model_path': 'models/mamba_active.pth'
}

def download_data():
    if not yf:
        raise ImportError("yfinance not installed")
    
    print(f"Downloading {CONFIG['symbol']} data from {CONFIG['start_date']}...")
    df = yf.download(CONFIG['symbol'], start=CONFIG['start_date'], end=CONFIG['end_date'], interval=CONFIG['interval'])
    if df.empty:
        raise ValueError("No data downloaded")
    
    # Flatten MultiIndex if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize columns
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    })
    return df

def prepare_features(df):
    print("Feature Engineering...")
    
    # Matches logic in mamba_engine.py precompute_all
    closes = df['close']
    
    # 1. Log Returns
    df['log_ret'] = np.log(closes / closes.shift(1)) * 100.0
    
    # 2. RSI
    df['rsi_14'] = ta.rsi(closes, length=14)
    # Normalize: (RSI - 50) / 10
    df['norm_rsi'] = (df['rsi_14'].fillna(50.0) - 50.0) / 10.0
    
    # 3. ATR
    df['atr'] = ta.atr(df['high'], df['low'], closes, length=14)
    df['atr_pct'] = df['atr'] / closes
    # Normalize: * 50
    df['norm_atr'] = df['atr_pct'].fillna(0.01) * 50.0
    
    # 4. Volume Ratio
    vol_sma = ta.sma(df['volume'], length=20)
    df['vol_ratio'] = df['volume'] / (vol_sma + 1.0)
    # Normalize: (Ratio - 1) * 2
    df['norm_vol'] = (df['vol_ratio'].fillna(1.0) - 1.0) * 2.0
    
    # Target: Next bar log return
    df['target'] = df['log_ret'].shift(-1)
    
    # Drop NaN
    df.dropna(inplace=True)
    
    # Select features matching 'precompute_all' order: [log_ret, rsi, atr, vol]
    feature_cols = ['log_ret', 'norm_rsi', 'norm_atr', 'norm_vol']
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    
    # Clamp extreme outliers
    X = np.clip(X, -5.0, 5.0)
    y = np.clip(y, -5.0, 5.0)
    
    return X, y

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback - 1]) # Predict NEXT step after sequence? 
        # Actually standard forecasting predicts y[i+lookback] (the step AFTER the window)
        # Our 'y' is already shifted (-1).
        # So X[0..59] (Bar 0 to 59) predicts y[59] (which is ret[60]).
    
    return np.array(Xs), np.array(ys)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    # 1. Data
    df = download_data()
    X, y = prepare_features(df)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, CONFIG['lookback'])
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, CONFIG['lookback'])
    
    print(f"Train samples: {len(X_train_seq)} | Val samples: {len(X_val_seq)}")
    
    # Loaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq)), 
                              batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq)), 
                            batch_size=CONFIG['batch_size'])
    
    # 2. Model
    if not HAS_MAMBA:
        print("Mamba not found. Cannot train.")
        return

    # Use Pro config if available
    model = DeepMamba(
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['layers'],
        d_state=32,
        d_conv=4,
        expand=2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    
    # 3. Loop
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            # Padding to d_model happens here if needed? 
            # DeepMamba expects input (B, L, d_model).
            # Our X is (B, L, 4).
            # We need a projection layer! 
            # Wait, DeepMamba in mamba_engine doesn't have an input projection?
            # Let's check mamba_engine source.
            
            # Temporary Fix: Pad with zeros to d_model inside loop
            B, L, F = X_batch.shape
            X_pad = torch.zeros((B, L, CONFIG['d_model']), device=device)
            X_pad[:, :, :F] = X_batch
            
            # Forward
            out = model(X_pad) # (B, 1) or (B, L, D)? DeepMamba returns (B, 1) usually via head
            
            loss = criterion(out.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                B, L, F = X_batch.shape
                X_pad = torch.zeros((B, L, CONFIG['d_model']), device=device)
                X_pad[:, :, :F] = X_batch
                
                out = model(X_pad)
                val_loss += criterion(out.squeeze(), y_batch)
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            # Save
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), CONFIG['model_path'])
            print(f"  [Saved] New best model: {avg_val:.4f}")

    print("Training Complete.")

if __name__ == "__main__":
    train()
