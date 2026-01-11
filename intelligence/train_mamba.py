import sys
import os

# Memory Protection: Prevent fragmentation on smaller GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.config import RunConfig
except ImportError:
    print("\n[ERROR] 'core/config.py' not found.")
    print("Action: Copy 'core/config.template.py' to 'core/config.py' and fill in your API keys.")
    print("Command: cp core/config.template.py core/config.py\n")
    sys.exit(1)

# Alpaca Imports (Optional, only for download)
HAS_ALPACA = False
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    HAS_ALPACA = True
except ImportError:
    pass

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
    parser.add_argument("--key", type=str, help="Alpaca API Key (optional if in config.py)")
    parser.add_argument("--secret", type=str, help="Alpaca Secret Key (optional if in config.py)")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--years", type=int, default=2, help="Years of history to fetch")
    parser.add_argument("--timeframe", type=str, default="15Min", choices=["1Min", "5Min", "15Min", "1Hour"])
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=32)
    
    # Workflow flags
    parser.add_argument("--save-only", action="store_true", help="Download data and save to CSV, then exit (No Torch required)")
    parser.add_argument("--local-data", type=str, help="Path to local CSV to use for training instead of downloading")
    parser.add_argument("--output-csv", type=str, default="data/spy_training_data.csv")
    parser.add_argument("--model-name", type=str, help="Filename for the model (e.g., mamba_m5_v1.pth)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lookback", type=int, default=60, help="Number of bars for model context")
    
    return parser.parse_args()

def download_alpaca_data(key, secret, symbol, years=2, tf_str="15Min"):
    if not HAS_ALPACA:
        print("[Error] Alpaca SDK (alpaca-py) not found.")
        print("Install it with: pip install alpaca-py")
        return pd.DataFrame()
        
    if not key or not secret:
        print("[Error] Alpaca Keys required for download.")
        return pd.DataFrame()

    print(f"[Alpaca] Connecting to fetch {years} years of {tf_str} data for {symbol}...")
    client = StockHistoricalDataClient(key, secret)
    
    # Non-Pro Alpaca accounts cannot query the last 15 minutes of SIP data.
    # We use a 20-minute UTC buffer to be absolutely safe and avoid subscription errors.
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_dt = end_dt - timedelta(days=365 * years)
    
    # Map timeframe string to Alpaca Object
    if tf_str == "15Min":
        tf = TimeFrame(15, TimeFrameUnit.Minute)
    elif tf_str == "5Min":
        tf = TimeFrame(5, TimeFrameUnit.Minute)
    elif tf_str == "1Min":
        tf = TimeFrame.Minute # Or TimeFrame(1, TimeFrameUnit.Minute)
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
    
    # 5. Session Timing (Minutes from market open 9:30 AM EST)
    if 'timestamp' in df.columns:
        df['dt'] = pd.to_datetime(df['timestamp'])
    else:
        # Assume index is datetime if column missing
        df['dt'] = pd.to_datetime(df.index)
        
    df['hour'] = df['dt'].dt.hour
    df['min'] = df['dt'].dt.minute
    df['min_from_open'] = (df['hour'] - 9) * 60 + (df['min'] - 30)
    df['norm_time'] = np.clip(df['min_from_open'] / 390.0, 0, 1) # 390 mins in session
    
    # Target: Next bar log return
    df['target'] = df['log_ret'].shift(-1)
    
    # Drop NaN
    df.dropna(inplace=True)
    
    # Select features: [log_ret, rsi, atr, vol, time]
    feature_cols = ['log_ret', 'norm_rsi', 'norm_atr', 'norm_vol', 'norm_time']
    
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

def run_download(args):
    # Resolve Keys
    api_key = args.key
    api_secret = args.secret
    
    if not api_key:
        try:
            cfg = RunConfig()
            api_key = cfg.alpaca_key # Corrected attribute name
            api_secret = cfg.alpaca_secret # Corrected attribute name
            if api_key and "YOUR_" not in api_key:
                print("[Config] Using Alpaca Keys from config.py")
        except Exception as e:
            print(f"[Warning] Could not load config: {e}")
            
    if api_key and api_secret:
            df = download_alpaca_data(api_key, api_secret, args.symbol, args.years, args.timeframe)
            if not df.empty:
                os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
                df.to_csv(args.output_csv, index=False)
                print(f"[Success] Data saved to {args.output_csv}.")
    else:
            print("[Error] Must provide Alpaca Keys (via CLI or config.py)")

def run_training(args):
    # Lazy Imports
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    try:
        from intelligence.mamba_engine import DeepMamba, HAS_MAMBA
    except ImportError:
        HAS_MAMBA = False
        print("[Error] Could not import Mamba/Torch.")
        return

    # Load Data
    df = pd.DataFrame()
    if args.local_data:
        print(f"[Core] Loading local data from {args.local_data}...")
        try:
            df = pd.read_csv(args.local_data)
        except Exception as e:
            print(f"[Error] Failed to load local CSV: {e}")
            return
    else:
        # Try download via run_download helpers? 
        # Simpler: just tell user to use save-only first if locally.
        print("[Error] In Training Mode, --local-data is recommended. Or use download flow first.")
        return

    # Check Empty
    if df.empty:
        print("No data available.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f" NEURAL TRAINING DEVICE: {device.type.upper()}")
    print("="*60)
    
    if device.type != 'cuda':
        print("[CRITICAL WARNING] No GPU detected. Mamba training on CPU will be extremely slow (100x slower).")
        print("[Action] Please enable GPU in Colab (Runtime > Change runtime type > T4 / A100 GPU).")
        # Optional: sys.exit(1) if you want to be strict
    
    X, y = prepare_features(df)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Sequences
    lookback = args.lookback
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, lookback)
    
    print(f"Train samples: {len(X_train_seq)} | Val samples: {len(X_val_seq)} | Lookback: {lookback}")

    # Loaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq)), 
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq)), 
                            batch_size=args.batch_size)
    
    # Model
    if not HAS_MAMBA:
        print("Mamba not found. Cannot train.")
        return

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
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # Higher decay for Large model
    
    # Model Path Setup
    if args.model_name:
        model_filename = args.model_name
        if not model_filename.endswith('.pth'):
            model_filename += '.pth'
    elif args.timeframe == "5Min":
        model_filename = "mamba_m5_active.pth"
    else:
        model_filename = "mamba_active.pth"
        
    model_path = os.path.join("models", model_filename)

    # Best loss tracking
    best_loss = float('inf')
    
    # Initialize Gradient Scaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"[Memory] Initial VRAM Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Pad/Cast with Mixed Precision
            with torch.amp.autocast('cuda'):
                B, L, F = X_batch.shape
                X_pad = torch.zeros((B, L, d_model), device=device)
                X_pad[:, :, :F] = X_batch
                
                # Forward
                out = model(X_pad) 
                loss = criterion(out.squeeze(), y_batch)
            
            # Backward with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with torch.amp.autocast('cuda'):
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
            torch.save(model.state_dict(), model_path)
            print(f"  [Saved] New best model ({model_path}): {avg_val:.4f}")

    print("Training Complete.")

def main():
    args = parse_args()
    if args.save_only:
        run_download(args)
    else:
        run_training(args)

if __name__ == "__main__":
    main()
