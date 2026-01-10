
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas_ta as ta
from datetime import datetime, timedelta

# Alpaca Imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from intelligence.mamba_engine import DeepMamba, HAS_MAMBA

# Configuration defaults
DEFAULT_CONFIG = {
    'symbol': 'SPY',
    'lookback': 60,
    'd_model': 1024,   # Default to Large
    'layers': 32,
    'epochs': 50,
    'batch_size': 128,
    'lr': 5e-5,
    'model_path': 'models/mamba_active.pth'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba on Alpaca Intraday Data")
    parser.add_argument("--key", type=str, required=True, help="Alpaca API Key")
    parser.add_argument("--secret", type=str, required=True, help="Alpaca Secret Key")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--years", type=int, default=2, help="Years of history to fetch")
    parser.add_argument("--timeframe", type=str, default="15Min", choices=["1Min", "5Min", "15Min", "1Hour"])
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=32)
    return parser.parse_args()

def download_alpaca_data(key, secret, symbol, years=2, tf_str="15Min"):
    print(f"[Alpaca] Connecting to fetch {years} years of {tf_str} data for {symbol}...")
    client = StockHistoricalDataClient(key, secret)
    
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365 * years)
    
    # Map timeframe string to Alpaca Object
    if tf_str == "15Min":
        tf = TimeFrame(15, TimeFrame.Minute)
    elif tf_str == "5Min":
        tf = TimeFrame(5, TimeFrame.Minute)
    elif tf_str == "1Min":
        tf = TimeFrame.Minute
    else:
        tf = TimeFrame.Hour

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_dt,
        end=end_dt
    )
    
    try:
        bars = client.get_stock_bars(req)
        df = bars.df
        
        # MultiIndex cleanup (symbol, timestamp) -> just timestamp
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True) # Drop symbol level
        
        print(f"[Alpaca] Downloaded {len(df)} bars.")
        return df
    except Exception as e:
        print(f"[Error] Alpaca download failed: {e}")
        return pd.DataFrame()

def prepare_features(df):
    if df.empty:
        raise ValueError("Cannot prepare features for empty DataFrame")
        
    print("Feature Engineering...")
    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]
    
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
        ys.append(y[i + lookback - 1]) 
    
    return np.array(Xs), np.array(ys)

def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    # 1. Data
    df = download_alpaca_data(args.key, args.secret, args.symbol, args.years, args.timeframe)
    if df.empty:
        return

    X, y = prepare_features(df)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, DEFAULT_CONFIG['lookback'])
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, DEFAULT_CONFIG['lookback'])
    
    print(f"Train samples: {len(X_train_seq)} | Val samples: {len(X_val_seq)}")

    # Loaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq)), 
                              batch_size=DEFAULT_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq)), 
                            batch_size=DEFAULT_CONFIG['batch_size'])
    
    # 2. Model
    if not HAS_MAMBA:
        print("Mamba not found. Cannot train.")
        return

    # Use Pro config if available
    d_model = args.d_model
    layers = args.layers
    
    print(f"Using DeepMamba Config: d_model={d_model}, layers={layers}")

    model = DeepMamba(
        d_model=d_model,
        n_layers=layers,
        d_state=32,
        d_conv=4,
        expand=2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=DEFAULT_CONFIG['lr'], weight_decay=1e-5)
    
    # 3. Loop
    best_loss = float('inf')
    
    for epoch in range(DEFAULT_CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Pad with zeros to d_model
            B, L, F = X_batch.shape
            X_pad = torch.zeros((B, L, d_model), device=device)
            X_pad[:, :, :F] = X_batch
            
            # Forward
            out = model(X_pad) # (B, 1)
            
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
                X_pad = torch.zeros((B, L, d_model), device=device)
                X_pad[:, :, :F] = X_batch
                
                out = model(X_pad)
                val_loss += criterion(out.squeeze(), y_batch)
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{DEFAULT_CONFIG['epochs']} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            # Save
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), DEFAULT_CONFIG['model_path'])
            print(f"  [Saved] New best model: {avg_val:.4f}")

    print("Training Complete.")

if __name__ == "__main__":
    train()
