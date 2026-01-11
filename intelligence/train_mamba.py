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
import torch
import torch.nn as nn
import torch.optim as optim

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
# research-backed institutional defaults
DEFAULT_CONFIG = {
    'symbol': 'SPY',
    'lookback': 240,   # Higher for 1m resolution (4 hours)
    'd_model': 1024,
    'layers': 32,      # 32-layer Selective-Scan SSM
    'epochs': 100,
    'batch_size': 128,
    'lr': 5e-5,
    'model_path': 'models/mamba_strategy_selector.pth'
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba on Alpaca Intraday Data")
    parser.add_argument("--key", type=str, help="Alpaca API Key (optional if in config.py)")
    parser.add_argument("--secret", type=str, help="Alpaca Secret Key (optional if in config.py)")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--years", type=float, default=2.0, help="Years of history to fetch")
    parser.add_argument("--timeframe", type=str, default="15Min", choices=["1Min", "5Min", "15Min", "1Hour"])
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=32)
    
    # Workflow flags
    parser.add_argument("--save-only", action="store_true", help="Download data and save to CSV, then exit (No Torch required)")
    parser.add_argument("--local-data", type=str, help="Path to local Spot CSV to use for training")
    parser.add_argument("--options-data", type=str, help="Path to local Options CSV to use for training")
    parser.add_argument("--output-csv", type=str, default="data/spy_training_data.csv")
    parser.add_argument("--model-name", type=str, help="Filename for the model (e.g., mamba_m5_v1.pth)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lookback", type=int, default=120, help="Number of bars for model context")
    parser.add_argument("--use-qqq", type=str2bool, default=True, help="Include QQQ correlation data (true/false)")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit number of rows loaded (0 for all)")
    
    return parser.parse_args()

def download_alpaca_data(key, secret, symbol, years=2, tf_str="15Min", use_qqq=True):
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

    try:
        # Request symbols
        symbols = [symbol]
        if use_qqq and symbol == 'SPY': symbols.append('QQQ')
        
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start_dt,
            end=end_dt
        )
        
        bars = client.get_stock_bars(req)
        df = bars.df
        
        # MultiIndex cleanup: pivot symbols to columns if multiple
        if isinstance(df.index, pd.MultiIndex):
            # We want columns like: SPY_close, QQQ_close...
            df = df.reset_index()
            df = df.pivot(index='timestamp', columns='symbol')
            # Flatten columns: (close, SPY) -> SPY_close
            df.columns = [f"{s}_{c.lower()}" for c, s in df.columns]
        
        print(f"[Alpaca] Downloaded data for {symbols}. Total bars: {len(df)}")
        return df
    except Exception as e:
        print(f"[Error] Alpaca download failed: {e}")
        return pd.DataFrame()

class CompositeLoss(torch.nn.Module):
    """
    Research-Backed Composite Loss for Options Trading:
    Huber (Stability) + Sharpe (ROI) + Turnover (Cost) + MADL (Sign)
    """
    def __init__(self, delta=1.0, alpha=0.1, lambd=0.01, sign_weight=0.5):
        super().__init__()
        self.huber = torch.nn.HuberLoss(delta=delta)
        self.alpha = alpha   # Sharpe weight
        self.lambd = lambd   # Turnover weight
        self.sign_weight = sign_weight # Directional weight
        
    def forward(self, pred, target):
        # Huber Baseline
        l_huber = self.huber(pred, target)
        
        # Sharpe-Ratio Lite (Maximize Return / Std of error)
        # We want to maximize ROI, so minimize inverse Sharpe
        ret = pred.mean()
        vol = pred.std() + 1e-6
        l_sharpe = -ret / vol
        
        # Directional Loss (MADL)
        l_sign = torch.mean(torch.abs(torch.sign(pred) - torch.sign(target)))
        
        # Combined Loss
        return l_huber + (self.alpha * l_sharpe) + (self.sign_weight * l_sign)

def prepare_features(df, options_df=None, use_qqq=True):
    if df.empty:
        raise ValueError("Cannot prepare features for empty DataFrame")
        
    print(f"Feature Engineering (Institutional 1m Foundation)...")
    
    # 0. Temporal Normalization
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'], utc=True)
    elif 'timestamp' in df.columns:
        df['dt'] = pd.to_datetime(df['timestamp'], utc=True)
        
    # Check if this is the new 1m foundation
    if 'ivr' in df.columns and 'spread_ratio' in df.columns:
        print("[Core] Mapping Institutional 1m features...")
        # Numeric Mapping for Strategy Selector
        df['cp_num'] = df['call_put'].map({'C': 1.0, 'P': -1.0}).fillna(0)
        
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 
            'strike', 'cp_num', 'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te',
            'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark'
        ]
        
        # Target is target_spot (log returns) combined with max_dd penalty
        # The model will predict 'target_spot'
        df['target'] = df['target_spot'].fillna(0)
    else:
        # Legacy/Automatic Logic
        main_close = 'SPY_close' if 'SPY_close' in df.columns else 'close'
        main_open = 'SPY_open' if 'SPY_open' in df.columns else 'open'
        main_high = 'SPY_high' if 'SPY_high' in df.columns else 'high'
        main_low = 'SPY_low' if 'SPY_low' in df.columns else 'low'
        main_vol = 'SPY_volume' if 'SPY_volume' in df.columns else 'volume'

        # Indicators ...
        df['rsi'] = ta.rsi(df[main_close], length=12)
        df['atr'] = ta.atr(df[main_high], df[main_low], df[main_close], length=12)
        df['vol_raw'] = df[main_vol]
        adx_df = ta.adx(df[main_high], df[main_low], df[main_close], length=12)
        df['adx'] = adx_df.iloc[:, 0] if adx_df is not None else np.nan
        bbands = ta.bbands(df[main_close], length=12)
        if bbands is not None:
            df['bb_lower'] = bbands.iloc[:, 0]
            df['bb_upper'] = bbands.iloc[:, 2]
        df['stoch_k'] = ta.stoch(df[main_high], df[main_low], df[main_close], k=12, d=3).iloc[:, 0]
        df['sma'] = ta.sma(df[main_close], length=12)
        psar = ta.psar(df[main_high], df[main_low], df[main_close])
        if psar is not None:
            df['psar'] = psar.iloc[:, 0].fillna(psar.iloc[:, 1])
            df['psar_mark'] = np.where(psar.iloc[:, 0].isna(), 1.0, -1.0)
            
        df['log_ret_hidden'] = np.log(df[main_close] / df[main_close].shift(1)) * 100.0
        df['target'] = df['log_ret_hidden'].shift(-1)
        
        feature_cols = [
            'open', 'high', 'low', 'close', 'vol_raw',
            'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark'
        ]

    df.dropna(subset=['target'], inplace=True)
    df[feature_cols] = df[feature_cols].ffill().fillna(0)
        
    print(f"[Prepare] Final Feature Matrix: {df[feature_cols].shape}")
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    
    return X, y

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback - 1]) 
    
    return np.array(Xs), np.array(ys)

def get_alpaca_keys(args):
    """Resolve Alpaca keys from CLI or config.py"""
    api_key = args.key
    api_secret = args.secret
    
    if not api_key:
        try:
            cfg = RunConfig()
            api_key = cfg.alpaca_key
            api_secret = cfg.alpaca_secret
            if api_key and "YOUR_" not in api_key:
                print("[Config] Using Alpaca Keys from config.py")
        except Exception as e:
            print(f"[Warning] Could not load config: {e}")
            
    return api_key, api_secret

def run_download(args):
    api_key, api_secret = get_alpaca_keys(args)
            
    if api_key and api_secret:
            df = download_alpaca_data(api_key, api_secret, args.symbol, args.years, args.timeframe, use_qqq=args.use_qqq)
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
            if args.max_rows > 0:
                df = pd.read_csv(args.local_data, nrows=args.max_rows)
            else:
                df = pd.read_csv(args.local_data)
        except Exception as e:
            print(f"[Error] Failed to load local CSV: {e}")
            return
    else:
        # Fallback to automatic download if key/secret available
        print("[Core] No local data provided. Attempting automatic download...")
        api_key, api_secret = get_alpaca_keys(args)
        if api_key and api_secret:
            df = download_alpaca_data(api_key, api_secret, args.symbol, args.years, args.timeframe, use_qqq=args.use_qqq)
            if not df.empty and args.output_csv:
                os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
                df.to_csv(args.output_csv, index=False)
                print(f"[Success] Data saved to {args.output_csv}.")
        else:
            print("[Error] No local data and no Alpaca keys found. Cannot proceed.")
            return

    # Load Options Data if provided
    options_df = None
    if args.options_data:
        print(f"[Core] Loading options metadata from {args.options_data}...")
        try:
            options_df = pd.read_csv(args.options_data)
        except Exception as e:
            print(f"[Error] Failed to load options CSV: {e}")
            # Non-critical, continue without options if possible? 
            # User wants apples-to-apples, so maybe fail here if they provided it.
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
    
    X, y = prepare_features(df, options_df=options_df, use_qqq=args.use_qqq)
    
    # Split (50/50 Training vs Validation per Trading Reality)
    split = int(len(X) * 0.5)
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
        expand=2,
        input_dim=X.shape[1]
    ).to(device)
    
    criterion = CompositeLoss(alpha=0.1, sign_weight=0.5)
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
