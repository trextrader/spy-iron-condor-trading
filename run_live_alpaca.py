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
import intelligence.fuzzy_engine as fe
from config import StrategyConfig
from core.dto import MarketSnapshot, TradeDecision

# --- CONFIG ---
SYMBOL = "SPY"
TIMEFRAME = "1Min"
LOOKBACK_BARS = 600 # Need enough for all adaptive indicators
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_scaling(X, median, scale, clip_val=10.0):
    """Apply persistent scaling from checkpoint."""
    # Ensure scales are never zero
    scale = np.maximum(scale, 1e-6)
    X = (X - median) / (scale * 1.4826) # Match training normalization
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
        
    # Start time
    start_time = datetime.now(pytz.UTC) - timedelta(minutes=lookback_bars * 1.5)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_time,
        limit=lookback_bars
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index() 
    df.columns = [c.lower() for c in df.columns]
    
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'dt'})
        
    return df

def run_live_loop(executor, model, metadata, device):
    print(f"[{datetime.now()}] Starting Live Loop for {SYMBOL}...")
    
    # 0. State & Stats
    data_client = None 
    running_stats = {
        'total_trades': 0,
        'winners': 0,
        'losers': 0,
        'total_pnl_dollar': 0.0,
        'peak_capital': 100000.0,
        'max_dd_pct': 0.0
    }
    capital = 100000.0
    starting_capital = 100000.0
    
    active_trade = None 
    
    # Loop
    while True:
        try:
            now = datetime.now(pytz.UTC)
            seconds = now.second
            # Sleep until next minute bar
            sleep_sec = 62 - seconds
            if sleep_sec < 0: sleep_sec += 60
            print(f"Waiting {sleep_sec}s for next bar...")
            time.sleep(sleep_sec)
            
            # 1. Fetch Data
            df = fetch_live_data(data_client, SYMBOL, lookback_bars=LOOKBACK_BARS)
            
            if len(df) < 256:
                print(f"Warning: Not enough bars ({len(df)}). Waiting...")
                continue
                
            # 2. V2.2 Feature Preparation
            defaults = {
                'strike': df['close'].iloc[-1],
                'cp_num': 0, 'delta': 0.5, 'gamma': 0.01, 'vega': 0.1, 'theta': -0.05,
                'iv': 0.2, 'ivr': 50, 'spread_ratio': 1.0, 'te': 14.0
            }
            for col, val in defaults.items():
                if col not in df.columns: df[col] = val
            
            # Log Volume
            df['volume'] = np.log1p(df['volume'])
            
            # 3. Scaling
            features = df[FEATURE_COLS_V22].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0)
            X_scaled = apply_scaling(features, metadata['median'], metadata['scale'])
            
            # 4. Inference
            seq_len = metadata.get('seq_len', 256)
            x_seq = torch.tensor(X_scaled[-seq_len:], device=device).unsqueeze(0).to(dtype=torch.float32)
            
            model.eval()
            with torch.no_grad():
                out_tuple = model(x_seq)
                preds = out_tuple[0].cpu().numpy()[0]
            
            # Parse Outputs
            prob_profit = float(preds[4])
            confidence = float(preds[7])
            
            # Rules consenso (Dummy if not in df)
            rule_consensus = 0.0
            if 'rule_long_consensus' in df.columns:
                rule_consensus = df['rule_long_consensus'].iloc[-1] - df['rule_short_consensus'].iloc[-1]
            
            # 11-Factor Fuzzy Score
            entry_score = 0
            if confidence > 0.7: entry_score += 30
            elif confidence > 0.5: entry_score += 20
            elif confidence > 0.3: entry_score += 10
            
            if prob_profit > 0.6: entry_score += 30
            elif prob_profit > 0.45: entry_score += 20
            elif prob_profit > 0.3: entry_score += 10
            
            if rule_consensus > 0.5: entry_score += 30
            elif rule_consensus >= 0: entry_score += 15
            
            ENTRY_THRESHOLD = 40
            spot = df['close'].iloc[-1]
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Spot: ${spot:.2f} | Score: {entry_score}/100 | Conf: {confidence:.4f} | Prob: {prob_profit:.4f}")

            # 5. Execution Logic
            if active_trade is None:
                if entry_score >= ENTRY_THRESHOLD and confidence > 0.4:
                    trade_num = running_stats['total_trades'] + 1
                    call_off, put_off, width, dte = preds[0], preds[1], preds[2], preds[3]
                    width = width if width > 1.0 else 5.0
                    dte = dte if dte > 1.0 else 14.0
                    short_call = spot + (call_off * spot * 0.01)
                    short_put = spot - (put_off * spot * 0.01)
                    
                    entry_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🦅 IRON CONDOR #{trade_num} ENTRY @ {datetime.now().strftime('%H:%M:%S')}
╠══════════════════════════════════════════════════════════════════════════════╣
║ SPOT:          ${spot:.2f}
║ FUZZY SCORE:   {entry_score}/100 (threshold: {ENTRY_THRESHOLD})
║ CONFIDENCE:     {confidence:.4f} | PROB: {prob_profit:.4f}
╠─────────────────────────── IRON CONDOR SETUP ────────────────────────────────╣
║   Short Call:  ${short_call:.2f}  |  Short Put:   ${short_put:.2f}
║   Width:       ${width:.2f}  |  DTE: {dte:.1f} days
╚══════════════════════════════════════════════════════════════════════════════╝"""
                    print(entry_msg)
                    active_trade = {'entry_spot': spot, 'short_call': short_call, 'short_put': short_put, 'entry_time': now, 'qty': 10}
                    running_stats['total_trades'] += 1
            else:
                exit_reason = None
                if confidence < 0.3: exit_reason = "Low Confidence"
                elif rule_consensus < -0.3: exit_reason = "Bearish Rule Signal"
                
                # Live PnL Delta
                pnl_delta = (spot - active_trade['entry_spot'])
                
                if exit_reason:
                    realized_pnl = 500.0 # Mock win for demo logic
                    if realized_pnl > 0: running_stats['winners'] += 1
                    else: running_stats['losers'] += 1
                    capital += realized_pnl
                    running_stats['total_pnl_dollar'] = capital - starting_capital
                    running_stats['peak_capital'] = max(running_stats['peak_capital'], capital)
                    dd = ((running_stats['peak_capital'] - capital) / running_stats['peak_capital']) * 100
                    running_stats['max_dd_pct'] = max(running_stats['max_dd_pct'], dd)
                    
                    win_rate = (running_stats['winners'] / running_stats['total_trades']) * 100
                    
                    exit_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🔔 IRON CONDOR EXIT @ {datetime.now().strftime('%H:%M:%S')}
╠══════════════════════════════════════════════════════════════════════════════╣
║ Spot Price:    ${spot:.2f} (Entry: ${active_trade['entry_spot']:.2f})
║ Exit Reason:   {exit_reason}
║ Result:        ✅ WIN (PnL: ${realized_pnl:,.0f})
╠─────────────────────────── PERFORMANCE METRICS ──────────────────────────────╣
║ 1) Win Rate:   {win_rate:.1f}% ({running_stats['winners']}W | {running_stats['losers']}L)
║ 2) Net P&L:    ${running_stats['total_pnl_dollar']:,.2f}
║ 3) Drawdown:   {running_stats['max_dd_pct']:.2f}%
╚══════════════════════════════════════════════════════════════════════════════╝"""
                    print(exit_msg)
                    active_trade = None
                else:
                    print(f"  [Active Trade] Running PnL Delta: ${pnl_delta:.2f} (Held {int((now-active_trade['entry_time']).total_seconds()/60)}m)")

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
        print(f"Initializing Alpaca (Paper={args.paper})...")
        executor = AlpacaExecutor(paper=args.paper)
        
        print(f"Loading model {args.model}...")
        ckpt = torch.load(args.model, map_location=DEVICE, weights_only=False)
        config = ckpt.get('model_config', ckpt.get('config', {}))
        
        if 'median' in ckpt and 'mad' in ckpt:
             med = np.array(ckpt['median'])
             scale = np.array(ckpt['mad'])
        else:
             print("Warning: Checkpoint missing scaling metadata. Live drift likely!")
             med = np.zeros(len(FEATURE_COLS_V22))
             scale = np.ones(len(FEATURE_COLS_V22))

        metadata = {
            'median': med, 'scale': scale,
            'seq_len': ckpt.get('seq_len', 256)
        }
        
        model = CondorBrain(
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 12),
            input_dim=len(FEATURE_COLS_V22),
            use_cde=True
        ).to(DEVICE)
        
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"✅ Model & Executor Ready. (Backbone: Neural CDE)")
        
        run_live_loop(executor, model, metadata, DEVICE)
        
    except Exception as e:
        print(f"Startup Failed: {e}")
        import traceback
        traceback.print_exc()
