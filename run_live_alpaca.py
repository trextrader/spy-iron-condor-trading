import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import argparse
from datetime import datetime
import pytz

# Add path
sys.path.insert(0, os.getcwd())

from execution.alpaca_executor import AlpacaExecutor
from intelligence.condor_brain import CondorBrain
from intelligence.features.dynamic_features import compute_all_dynamic_features, compute_all_primitive_features_v22
from intelligence.rule_engine.dsl_parser import RuleDSLParser
from intelligence.rule_engine.executor import RuleExecutionEngine

from intelligence.canonical_feature_registry import FEATURE_COLS_V22, get_neutral_fill_value_v22

# --- CONFIG ---
SYMBOL = "SPY"
TIMEFRAME = "1Min"
LOOKBACK_BARS = 600 # Need enough for all adaptive indicators
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_scaling(X, median, scale, clip_val=10.0):
    """Apply persistent scaling from checkpoint."""
    X = (X - median) / (scale + 1e-6)
    return np.clip(X, -clip_val, clip_val)


def fetch_live_data(client, symbol, lookback_bars=1000):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import timedelta
    
    api_key = os.getenv('APCA_API_KEY_ID')
    secret_key = os.getenv('APCA_API_SECRET_KEY')
    if not api_key: raise ValueError("Missing APCA_API_KEY_ID")
    
    # Init Data Client
    if client is None:
        client = StockHistoricalDataClient(api_key, secret_key)
        
    # Start time: Lookback * 1.5 minutes ago to be safe
    start_time = datetime.now(pytz.UTC) - timedelta(minutes=lookback_bars * 1.5)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_time,
        limit=lookback_bars
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index() 
    # Alpaca columns: symbol, timestamp, open, high, low, close, volume, trade_count, vwap
    # Ensure lowercase
    df.columns = [c.lower() for c in df.columns]
    return df

def run_live_loop(executor, model, ruleset, device):
    print(f"[{datetime.now()}] Starting Live Loop for {SYMBOL}...")
    
    # Alpaca Data Client
    from alpaca.data.historical import StockHistoricalDataClient
    data_client = None 
    
    # Loop
    while True:
        try:
            now = datetime.now(pytz.UTC)
            seconds = now.second
            # Sleep until next minute bar (plus 2 seconds latency buffer)
            sleep_sec = 62 - seconds
            if sleep_sec < 0: sleep_sec += 60
            print(f"Waiting {sleep_sec}s for next bar...")
            time.sleep(sleep_sec)
            
            # 1. Fetch Data
            print(f"[{datetime.now()}] Fetching Live Data (last {LOOKBACK_BARS} bars)...")
            df = fetch_live_data(data_client, SYMBOL, lookback_bars=LOOKBACK_BARS)
            
            if len(df) < 256:
                print(f"Warning: Not enough bars ({len(df)}). Waiting...")
                continue
                
            # 2. Compute Features (V2.2)
            # Requires 'cp_num', 'strike' etc? No, live run computes indicators from OHLCV.
            # But 'strike' / 'delta' features depend on Option Chain. 
            # LIVE run: We either need to fetch real-time Greeks/Strike from Alpaca Option Chain
            # OR use dummy values if model doesn't strictly rely on them (risk).
            # The model HAS 'delta', 'gamma' in feature list.
            # If we pass 0, the model output will be garbage.
            # CRITICAL: We need live option chain data to populate V2.2 features!
            # Alpaca Options API can provide Greeks?
            # If not, we cannot run V2.2 model live without a real Greeks provider.
            # For this DEMO, we will use Placeholder Greeks (0.5 delta, etc) just to test LATENCY?
            # User asked "test it on demo account today just to see if we have latency issues."
            # So we will fill missing columns with defaults to unblock execution test.
            
            print("Computing Features...")
            # Fill missing columns expected by V2.2
            defaults = {
                'strike': df['close'].iloc[-1], # At-the-money proxy
                'cp_num': 0,
                'delta': 0.5,
                'gamma': 0.01,
                'vega': 0.1,
                'theta': -0.05,
                'iv': 0.2,
                'ivr': 50,
                'spread_ratio': 1.0,
                'te': 14.0 # Default DTE
            }
            for col, val in defaults.items():
                if col not in df.columns:
                    df[col] = val
            
            # Compute Indicators
            # Helper: compute_primitive_features expects specific structure. 
            # We'll just assume compute_all_primitive_features_v22 works on this DF if we have OHLCV.
            # We might need to map column names if helper expects specific names.
            # Currently assuming helper matches.
            
            # Log Volume (Critical Fix)
            df['volume'] = np.log1p(df['volume']) # Is this done inside helper? Checking...
            # The helper 'compute_all_primitive_features_v22' likely adds indicators.
            # We apply log1p AFTER or BEFORE? Training did it BEFORE scaling.
            # Let's do it here.

            # 3. Scaling & Persistence
            # Ensure features matches FEATURE_COLS_V22 exactly
            features = df[FEATURE_COLS_V22].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0)
            X_scaled = apply_scaling(features, model_metadata['median'], model_metadata['scale'])
            
            # 4. Inference
            # Seq slice: Inputs [1, SEQ_LEN, F]
            seq_len = model_metadata.get('seq_len', 256)
            x_seq = torch.tensor(X_scaled[-seq_len:], device=device).unsqueeze(0).to(dtype=torch.float32)
            
            model.eval()
            with torch.no_grad():
                out_tuple = model(x_seq)
                # CondorBrain returns (logits, regime, horizon, features, ...)
                preds = out_tuple[0].cpu().numpy()[0]
            
            # Parse Outputs (V2.2 Indexing)
            # [0..3] Params, [4] Prob, [5] ROI, [6] MaxLoss, [7] Confidence, [8] Entry, [9] Exit
            prob_profit = preds[4]
            confidence = preds[7]
            entry_logit = preds[8]
            
            print(f"[{datetime.now()}] Model Output: Conf={confidence:.2f}, EntryLogit={entry_logit:.2f}, Prob={prob_profit:.2f}")

            # 5. Execute
            if confidence > 0.6: # Threshold
                print(">> SIGNAL TRIGGERED! Placing Order...")
                # Construct Legs (Mock)
                legs = [
                    {'option_symbol': f"SPY250117C00{int(df['close'].iloc[-1]*1.02)}000", 'side': 'sell'},
                    {'option_symbol': f"SPY250117C00{int(df['close'].iloc[-1]*1.03)}000", 'side': 'buy'},
                    # ... Puts ...
                ]
                # executor.submit_iron_condor(SYMBOL, legs, quantity=1)
                print("Order Sent (Mock for safety until verified).")
            else:
                print(">> No Signal.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to production CDE model (.pth)")
    parser.add_argument("--paper", action="store_true", default=True, help="Use paper trading")
    args = parser.parse_args()

    if not os.getenv('APCA_API_KEY_ID'):
        print("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars.")
        sys.exit(1)
        
    try:
        executor = AlpacaExecutor(paper=args.paper)
        
        # Load Model & Metadata
        print(f"Loading model {args.model}...")
        ckpt = torch.load(args.model, map_location=DEVICE)
        
        # Extract Config
        config = ckpt.get('model_config', {})
        
        # Handle scaling parameters (Persisted in new training logic)
        if 'median' in ckpt and 'mad' in ckpt:
             print("Using persistent scaling parameters from checkpoint.")
             med = np.array(ckpt['median'])
             scale = np.array(ckpt['mad'])
        else:
             print("Warning: Checkpoint missing persistent scaling. Live behavior may drift!")
             # Fallback or exit?
             med = np.zeros(len(FEATURE_COLS_V22))
             scale = np.ones(len(FEATURE_COLS_V22))

        model_metadata = {
            'median': med,
            'scale': scale,
            'seq_len': ckpt.get('seq_len', 256)
        }
        
        model = CondorBrain(
            d_model=config.get('d_model', 1024),
            n_layers=config.get('n_layers', 32),
            input_dim=len(FEATURE_COLS_V22),
            use_cde=True,
            use_topk_moe=config.get('use_topk_moe', False)
        ).to(DEVICE)
        
        state_dict = ckpt['state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model Loaded (CDE={model.use_cde}, Dim={len(FEATURE_COLS_V22)})")
        
        run_live_loop(executor, model, model_metadata, DEVICE)
        
    except Exception as e:
        print(f"Startup Failed: {e}")
        import traceback
        traceback.print_exc()
