import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import argparse
from datetime import datetime
import pytz
import json

# Add path
sys.path.insert(0, os.getcwd())

from execution.paper_executor import PaperExecutor
from intelligence.condor_brain import CondorBrain
from intelligence.features.dynamic_features import compute_all_dynamic_features, compute_all_primitive_features_v22
from intelligence.rule_engine.dsl_parser import RuleDSLParser
from intelligence.rule_engine.executor import RuleExecutionEngine

from intelligence.canonical_feature_registry import FEATURE_COLS_V22, get_neutral_fill_value_v22
import intelligence.fuzzy_engine as fe
try:
    from config import StrategyConfig
except ImportError:
    # Safe Fallback: Check if config is in core/ (Users often paste it there)
    core_config = os.path.join(os.getcwd(), 'core', 'config.py')
    if os.path.exists(core_config):
        print(f"⚠️ Warning: 'config.py' not found in root. Loading from '{core_config}'.")
        print(f"👉 Fix: Please move 'core/config.py' to './config.py' for standard behavior.")
        sys.path.append(os.path.join(os.getcwd(), 'core'))
        try:
             from config import StrategyConfig
        except ImportError:
             print("❌ Error: Failed to import StrategyConfig from core/config.py")
             sys.exit(1)
    else:
        print("❌ Error: 'config.py' not found in root or core/.")
        sys.exit(1)

from core.dto import MarketSnapshot, TradeDecision

# --- LOGGING HELPER ---
class TradeLogger:
    def __init__(self, log_dir="data/live"):
        self.log_path = os.path.join(log_dir, "trade_log.csv")
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("timestamp,spot,score,confidence,prob_profit,action,short_call,short_put,long_call,long_put,width,dte,trade_ids\n")
    
    def log_entry(self, spot, score, conf, prob, legs, trade_ids):
        # legs is list of dicts: [{'option_symbol': '...', 'side': '...'}, ...]
        # Extract symbols
        sc = next((l['option_symbol'] for l in legs if l['side'] == 'sell' and 'C' in l['option_symbol']), "N/A")
        sp = next((l['option_symbol'] for l in legs if l['side'] == 'sell' and 'P' in l['option_symbol']), "N/A")
        lc = next((l['option_symbol'] for l in legs if l['side'] == 'buy' and 'C' in l['option_symbol']), "N/A")
        lp = next((l['option_symbol'] for l in legs if l['side'] == 'buy' and 'P' in l['option_symbol']), "N/A")
        
        t_ids_str = "|".join(str(t) for t in trade_ids) if trade_ids else "DRY_RUN"
        
        row = f"{datetime.now()},{spot:.2f},{score},{conf:.4f},{prob:.4f},ENTRY,{sc},{sp},{lc},{lp},,N/A,{t_ids_str}\n"
        with open(self.log_path, "a") as f:
            f.write(row)
        print(f"📝 Logged Trade to {self.log_path}")

# Initialize Logger Global
trade_logger = TradeLogger()

class FeatureLogger:
    def __init__(self, feature_cols, log_dir="data/live"):
        self.log_path = os.path.join(log_dir, "feature_log.csv")
        self.cols = feature_cols
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                header = "timestamp," + ",".join(self.cols) + "\n"
                f.write(header)

    def log_features(self, timestamp, df_row):
        # df_row is a Series or dict
        vals = []
        for c in self.cols:
            val = df_row.get(c, 0.0)
            if isinstance(val, (int, float, np.float32, np.float64)):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
                
        line = f"{timestamp}," + ",".join(vals) + "\n"
        with open(self.log_path, "a") as f:
            f.write(line)

feature_logger = FeatureLogger(FEATURE_COLS_V22)

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

    return df

# --- HELPER: Greek Aggregation ---
def calculate_live_greeks(contracts, current_spot):
    """
    Aggregates Greeks from a list of Alpaca option contracts.
    Returns a dict of {delta, gamma, theta, vega, iv}.
    """
    if not contracts:
        return None
        
    deltas, gammas, thetas, vegas, ivs = [], [], [], [], []
    
    for c in contracts:
        if hasattr(c, 'greeks') and c.greeks:
            g = c.greeks
            if g.delta is not None: deltas.append(abs(g.delta)) 
            if g.gamma is not None: gammas.append(g.gamma)
            if g.theta is not None: thetas.append(g.theta)
            if g.vega is not None: vegas.append(g.vega)
        
        if c.implied_volatility is not None:
            ivs.append(c.implied_volatility)

    if not ivs: 
        return None
        
    return {
        'delta': float(np.mean(deltas)) if deltas else 0.5,
        'gamma': float(np.mean(gammas)) if gammas else 0.0,
        'theta': float(np.mean(thetas)) if thetas else 0.0,
        'vega': float(np.mean(vegas)) if vegas else 0.0,
        'iv': float(np.mean(ivs)) if ivs else 0.0,
        'svr': float(np.std(ivs)) if len(ivs) > 1 else 0.0 
    }

def find_closest_contract(client, symbol, contract_type, target_strike, ideal_date):
    """
    Finds the contract closest to target_strike and ideal_date.
    """
    try:
        from alpaca.data.requests import OptionContractsRequest
        req = OptionContractsRequest(
            underlying_symbols=[symbol],
            status='active',
            expiration_date_gte=ideal_date.date(),
            expiration_date_lte=(ideal_date + pd.Timedelta(days=5)).date(),
            type=contract_type,
            strike_price_gte=target_strike - 2,
            strike_price_lte=target_strike + 2,
            limit=10
        )
        res = client.get_option_contracts(req)
        contracts = res.option_contracts
    except ImportError:
        try:
             # Try generic strategy using OptionChainRequest
             from alpaca.data.requests import OptionChainRequest
             req = OptionChainRequest(
                 underlying_symbol=symbol,
                 type=contract_type,
                 expiration_date_gte=ideal_date.date(),
                 expiration_date_lte=(ideal_date + pd.Timedelta(days=5)).date(),
                 strike_price_gte=target_strike - 2,
                 strike_price_lte=target_strike + 2
             )
             chain_map = client.get_option_chain(req)
             contracts = []
             for sym, snap in chain_map.items():
                 try:
                     # Parse OSI Strike
                     strike = float(sym[-8:]) / 1000.0
                     contracts.append(type('obj', (object,), {'symbol': sym, 'strike_price': strike}))
                 except: 
                     continue
        except Exception as e_fallback:
             print(f"Fallback Search Error: {e_fallback}")
             return None
    except Exception as e:
        print(f"Contract Search Error: {e}")
        return None

    try:
        if not contracts:
            return None
        best = min(contracts, key=lambda x: abs(float(x.strike_price) - target_strike))
        return best.symbol
    except Exception as e:
        print(f"Contract Search Error: {e}")
        return None

def save_state(active_trades, filename="data/live/bot_state.json"):
    """Persists active trades to JSON for restart safety."""
    state = []
    for trade in active_trades:
        t = trade.copy()
        # Convert datetime to ISO string for JSON
        if 'entry_time' in t and isinstance(t['entry_time'], datetime):
            t['entry_time'] = t['entry_time'].isoformat()
        
        # Convert UUID objects to strings (Alpaca returns UUIDs)
        if 'trade_ids' in t and t['trade_ids']:
            t['trade_ids'] = [str(tid) for tid in t['trade_ids']]
            
        state.append(t)
    
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(state, f, indent=4)
        print(f"💾 Bot State Saved: {len(state)} active trades.")
    except Exception as e:
        print(f"❌ Failed to save bot state: {e}")

def load_state(filename="data/live/bot_state.json"):
    """Restores active trades from JSON."""
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r") as f:
            state = json.load(f)
        
        # Convert ISO strings back to datetime
        for t in state:
            if 'entry_time' in t:
                # Ensure UTC aware if it was saved as such
                dt = datetime.fromisoformat(t['entry_time'])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.UTC)
                t['entry_time'] = dt
        
        print(f"📢 Restored {len(state)} active trades from {filename}")
        return state
    except Exception as e:
        print(f"⚠️ Failed to load bot state: {e}")
        return []

# --- FUZZY GOVERNANCE CONFIG ---
FUZZY_WEIGHTS = {
    "neural_conf": 0.35,
    "prob_profit": 0.35,
    "ivr": 0.10,
    "trend": 0.10,
    "rsi": 0.10
}

def calculate_memberships(df, confidence, prob_profit):
    """Maps latest row and model outputs to fuzzy domain."""
    row = df.iloc[-1]
    
    # 1. Neural & Prob
    m_neural = min(1.0, confidence / 0.8)
    m_prob = max(0.0, (prob_profit - 0.45) / 0.2) if prob_profit >= 0.45 else 0.0
    
    # 2. IVR (if available)
    ivr = row.get('ivr', 0.5)
    m_ivr = min(1.0, ivr / 60.0) if ivr > 0 else 0.5
    
    # 3. Trend (ADX/PSAR)
    adx = row.get('adx_adaptive', 20.0)
    m_trend = max(0.0, 1.0 - (adx / 40.0)) # Lower ADX = Better for Condor
    
    # 4. RSI (Neutral check)
    rsi = row.get('rsi_dyn', 50.0)
    # Neutral 40-60 is ideal
    m_rsi = 1.0 - (abs(rsi - 50.0) / 50.0)
    
    return {
        "neural_conf": m_neural,
        "prob_profit": m_prob,
        "ivr": m_ivr,
        "trend": m_trend,
        "rsi": m_rsi
    }

def is_market_session():
    """
    Checks if we are within the SPY option trading window:
    9:35 AM - 4:10 PM ET (Monday-Friday).
    """
    tz_et = pytz.timezone("US/Eastern")
    now_et = datetime.now(tz_et)
    
    # Check Weekend
    if now_et.weekday() >= 5:
        return False, "Weekend"
    
    # Define bounds
    open_time = now_et.replace(hour=9, minute=35, second=0, microsecond=0)
    close_time = now_et.replace(hour=16, minute=10, second=0, microsecond=0)
    
    if now_et < open_time:
        return False, "Pre-Market"
    if now_et > close_time:
        return False, "Post-Market"
        
    return True, "Open"

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
    
    active_trades = load_state()
    MAX_CONCURRENT_TRADES = 4
    
    # Loop
    while True:
        try:
            # --- MARKET HOURS GATE ---
            session_open, reason = is_market_session()
            if not session_open:
                sys.stdout.write(f"\r💤 Market Closed ({reason}). Sleeping 60s...       ")
                sys.stdout.flush()
                time.sleep(60)
                continue

            now = datetime.now(pytz.UTC)
            seconds = now.second
            # Live Countdown
            sleep_sec = 62 - seconds
            if sleep_sec < 0: sleep_sec += 60
            
            while sleep_sec > 0:
                sys.stdout.write(f"\r⏳ Waiting {sleep_sec}s for next bar (Recording Live Chains)...   ")
                sys.stdout.flush()
                time.sleep(1)
                sleep_sec -= 1
            print() # Newline
            
            # --- LIVE DATA RECORDER START ---
            try:
                if not hasattr(run_live_loop, 'opt_client'):
                     try:
                         from alpaca.data.historical import OptionHistoricalDataClient
                         ak = os.getenv('APCA_API_KEY_ID')
                         sk = os.getenv('APCA_API_SECRET_KEY')
                         run_live_loop.opt_client = OptionHistoricalDataClient(ak, sk)
                     except:
                         run_live_loop.opt_client = None

                if run_live_loop.opt_client:
                    center_strike = 695 
                    if active_trades:
                        center_strike = active_trades[-1]['entry_spot']
                    
                    chain_path = None
                    all_contracts = []
                    try:
                        from alpaca.data.requests import OptionContractsRequest
                        
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        chain_dir = os.path.join("data", "live", "chains")
                        os.makedirs(chain_dir, exist_ok=True)
                        chain_path = os.path.join(chain_dir, f"spy_chain_{timestamp_str}.csv")
                        
                        start_date = datetime.now().date()
                        req_c = OptionContractsRequest(
                            underlying_symbols=[SYMBOL],
                            status='active',
                            expiration_date_gte=start_date,
                            expiration_date_lte=start_date + pd.Timedelta(days=7),
                            type='call',
                            limit=1000
                        )
                        req_p = OptionContractsRequest(
                            underlying_symbols=[SYMBOL],
                            status='active',
                            expiration_date_gte=start_date,
                            expiration_date_lte=start_date + pd.Timedelta(days=7),
                            type='put',
                            limit=1000
                        )
                        
                        res_c = run_live_loop.opt_client.get_option_contracts(req_c)
                        res_p = run_live_loop.opt_client.get_option_contracts(req_p)
                        
                        all_contracts = res_c.option_contracts + res_p.option_contracts
                        
                    except ImportError:
                        try:
                           from alpaca.data.requests import OptionChainRequest
                           req_chain = OptionChainRequest(
                               underlying_symbol=SYMBOL,
                               expiration_date_gte=datetime.now().date(),
                               expiration_date_lte=datetime.now().date() + pd.Timedelta(days=7),
                           )
                           res_chain = run_live_loop.opt_client.get_option_chain(req_chain)
                           all_contracts = list(res_chain.values())
                           for sym, snap in res_chain.items():
                               if not hasattr(snap, 'symbol'):
                                   snap.symbol = sym
                        except Exception as e_fb:
                            print(f" [Recorder Warning] Fallback chain fetch failed: {e_fb}")

                    except Exception as e_chain:
                        print(f"  [Recorder Warning] fetching chain failed: {e_chain}")

                    if all_contracts:
                        chain_df = pd.DataFrame([c.model_dump() for c in all_contracts])
                        cols = ['symbol', 'strike_price', 'expiration_date', 'type', 'open_interest', 'style', 'implied_volatility', 'greeks']
                        chain_df = chain_df[[c for c in cols if c in chain_df.columns]]
                        chain_df.to_csv(chain_path, index=False)
                        
                        syms = [c.symbol for c in all_contracts]
                        print(f"\n📚 Option Chain Recorded ({len(syms)} Contracts):")
                        print(" ".join(syms)) 
                        print("-" * 50)

            except Exception as e_rec:
                print(f" [Data Recorder Error] {e_rec}")
            # --- LIVE DATA RECORDER END ---

            
            # 1. Fetch Data
            df = fetch_live_data(data_client, SYMBOL, lookback_bars=LOOKBACK_BARS)

            # --- INJECT LIVE GREEKS ---
            current_greeks = None
            if 'all_contracts' in locals() and all_contracts:
                 current_greeks = calculate_live_greeks(all_contracts, df['close'].iloc[-1])
                 if current_greeks:
                     print(f"  [Greeks] IV: {current_greeks['iv']:.2f} | Gamma: {current_greeks['gamma']:.4f}")
            
            if current_greeks:
                for k, v in current_greeks.items():
                    if k not in df.columns:
                        df[k] = v 
                    else:
                        df.iloc[-1, df.columns.get_loc(k)] = v
            # ---------------------------

            
            if len(df) < 256:
                print(f"Warning: Not enough bars ({len(df)}). Waiting...")
                continue
                
            # 2. V2.2 Feature Preparation
            df = compute_all_dynamic_features(df, inplace=True)
            
            if 'lag_minutes' not in df.columns:
                 df['lag_minutes'] = 0.0 
            
            df = compute_all_primitive_features_v22(df, inplace=True)
            
            # C) Handle Targets & Missing Columns
            for col in FEATURE_COLS_V22:
                if col not in df.columns:
                    if col in ['target_spot', 'max_dd_60m', 'target_profit']:
                        df[col] = 0.0
                    elif col in ['delta', 'gamma', 'vega', 'theta', 'iv']:
                         df[col] = get_neutral_fill_value_v22(col)
                    else:
                        # Fallback for others (e.g. from V2.2 defaults)
                        fill_val = get_neutral_fill_value_v22(col)
                        df[col] = fill_val

            # Log Volume
            df_model_input = df.copy()
            df_model_input['volume'] = np.log1p(df_model_input['volume'])
            
            # --- LOG FEATURES FOR ACCOUNTABILITY ---
            try:
                latest_row = df_model_input.iloc[-1]
                feature_logger.log_features(now, latest_row)
            except Exception as e_flog:
                print(f"Feature Log Error: {e_flog}")

            # 3. Scaling
            features = df_model_input[FEATURE_COLS_V22].values.astype(np.float32)
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
            
            # Rules consenso
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

            # ---------------------------------------------------------
            # 5. Execution Logic (Multi-Trade Concurrency)
            # ---------------------------------------------------------
            
            # A) MANAGEMENT: Check ALL active trades for Exit Signals
            for i, trade in enumerate(active_trades[:]): # Iterate copy
                exit_reason = None
                
                # Update Trade Duration
                trade_duration = (now - trade['entry_time']).total_seconds() / 60.0
                
                # Rule 1: Neural Confidence Drop
                if confidence < 0.35:
                    exit_reason = f"Low Confidence ({confidence:.2f})"
                
                # Rule 2: Spot Breach (Gamma Risk) - Using stored strikes
                elif spot > trade['short_call'] or spot < trade['short_put']:
                    exit_reason = f"Strike Breach (Spot {spot:.2f})"
                
                # Rule 3: Time Exit
                elif trade_duration > 1440: # 24 Hours (increased from 2h)
                    exit_reason = "Time Limit (24h)"

                if exit_reason:
                    print(f"📉 CLOSING Trade #{i+1} ({trade['legs'][0]['option_symbol'] if trade['legs'] else 'N/A'}): {exit_reason}")
                    
                    success = True
                    if executor:
                        try:
                            # Actually call the close method
                            success = executor.close_iron_condor(trade['legs'], trade['qty'])
                        except Exception as e:
                            print(f"   ❌ Close Failed: {e}")
                            success = False

                    if success:
                        realized_pnl = 50.0 if "Breach" not in exit_reason else -100.0 # Mock PnL
                        if realized_pnl > 0: running_stats['winners'] += 1
                        else: running_stats['losers'] += 1
                        
                        capital += realized_pnl
                        active_trades.remove(trade)
                        save_state(active_trades) # Update Persistence
                        print(f"   ✅ Trade Closed. PnL: ${realized_pnl:.2f}")
                    else:
                        print("   ⚠️ Exit Command Failed. Keeping trade in state for retry.")
                else:
                    # Logging Pulse
                    pnl_delta = (spot - trade['entry_spot'])
                    print(f"  [Trade #{i+1}] PnL Delta: ${pnl_delta:.2f} (Held {int(trade_duration)}m)")

            # B) ENTRY: Look for NEW Trades (If slots available)
            if len(active_trades) < MAX_CONCURRENT_TRADES:
                
                if entry_score >= ENTRY_THRESHOLD and confidence > 0.4:
                    trade_num = running_stats['total_trades'] + 1
                    call_off, put_off, width, dte = preds[0], preds[1], preds[2], preds[3]
                    width = width if width > 1.0 else 5.0
                    dte = dte if dte > 1.0 else 14.0 
                    
                    target_short_call = spot + (call_off * spot * 0.01)
                    target_short_put = spot - (put_off * spot * 0.01)
                    target_long_call = target_short_call + width
                    target_long_put = target_short_put - width
                    
                    target_date = datetime.now() + pd.Timedelta(days=dte)
                    
                    # Check Duplicates
                    is_duplicate = False
                    for trade in active_trades:
                        if abs(trade['short_call'] - target_short_call) < 1.0:
                            is_duplicate = True
                            print(f"⚠️ Skipping Duplicate Strike Setup (${target_short_call:.2f})")
                            break
                    
                    if not is_duplicate:
                        # --- FUZZY POSITION SIZING (Phase 14) ---
                        memberships = calculate_memberships(df, confidence, prob_profit)
                        
                        # Max loss per contract (assuming 5-wide condor = $500)
                        max_loss = width * 100.0 if width > 0 else 500.0
                        
                        # Use fuzzy sizer
                        qty = fe.compute_position_size(
                            equity=capital,
                            max_loss_per_contract=max_loss,
                            memberships=memberships,
                            weights=FUZZY_WEIGHTS,
                            realized_vol=df['vol_ewma'].iloc[-1],
                            low_vol=0.05,
                            high_vol=0.30,
                            risk_fraction=0.02
                        )
                        
                        # Institutional Floor
                        if qty < 1: qty = 1
                        
                        print(f"📊 [Fuzzy Sizer] Score: {entry_score} -> Suggested Qty: {qty} (Risked: ${qty*max_loss:.2f})")

                        if not hasattr(run_live_loop, 'opt_client'):
                             try:
                                 from alpaca.data.historical import OptionHistoricalDataClient
                                 ak = os.getenv('APCA_API_KEY_ID')
                                 sk = os.getenv('APCA_API_SECRET_KEY')
                                 run_live_loop.opt_client = OptionHistoricalDataClient(ak, sk)
                             except:
                                 print("❌ Failed to init OptionClient for execution.")
                                 run_live_loop.opt_client = None

                        osi_legs = []
                        if run_live_loop.opt_client:
                            print(f"🔎 Solving Contracts (DTE={dte:.1f}d)...")
                            sc_sym = find_closest_contract(run_live_loop.opt_client, SYMBOL, 'call', target_short_call, target_date)
                            lc_sym = find_closest_contract(run_live_loop.opt_client, SYMBOL, 'call', target_long_call, target_date)
                            sp_sym = find_closest_contract(run_live_loop.opt_client, SYMBOL, 'put', target_short_put, target_date)
                            lp_sym = find_closest_contract(run_live_loop.opt_client, SYMBOL, 'put', target_long_put, target_date)
                            
                            if all([sc_sym, lc_sym, sp_sym, lp_sym]):
                                osi_legs = [
                                    {'option_symbol': sc_sym, 'side': 'sell'},
                                    {'option_symbol': lc_sym, 'side': 'buy'},
                                    {'option_symbol': sp_sym, 'side': 'sell'},
                                    {'option_symbol': lp_sym, 'side': 'buy'}
                                ]
                            else:
                                print(f"⚠️ Could not resolve all legs: SC={sc_sym}, LC={lc_sym}, SP={sp_sym}, LP={lp_sym}")

                        entry_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🦅 IRON CONDOR #{len(active_trades)+1} ENTRY @ {datetime.now().strftime('%H:%M:%S')}
╠══════════════════════════════════════════════════════════════════════════════╣
║ SPOT:          ${spot:.2f}
║ FUZZY SCORE:   {entry_score}/100 (threshold: {ENTRY_THRESHOLD})
║ CONFIDENCE:     {confidence:.4f} | PROB: {prob_profit:.4f}
╠─────────────────────────── IRON CONDOR SETUP ────────────────────────────────╣
║   Short Call:  ${target_short_call:.2f}  |  Short Put:   ${target_short_put:.2f}
║   Width:       ${width:.2f}  |  DTE: {dte:.1f} days
╠─────────────────────────── EXECUTION ────────────────────────────────────────╣
║   Resolved:    {len(osi_legs) == 4}
║   Quantity:    {qty}
║   Short Legs:  {osi_legs[0]['option_symbol'] if osi_legs else 'N/A'} / {osi_legs[2]['option_symbol'] if osi_legs else 'N/A'}
╚══════════════════════════════════════════════════════════════════════════════╝"""
                        print(entry_msg)
                        
                        # --- FRICTION GATE (Phase 14) ---
                        # In a real environment, we'd fetch Bid/Ask for the spread.
                        # For now, we use a proxy check on the 'spread_ratio' feature.
                        friction_ratio = df['spread_ratio'].iloc[-1]
                        MAX_ALLOWED_FRICTION = 0.05 # Revert if spread > 5% of mid
                        
                        if friction_ratio > MAX_ALLOWED_FRICTION:
                             print(f"🛑 [Friction Gate] Rejected! Spread Ratio {friction_ratio:.4f} > {MAX_ALLOWED_FRICTION} (Low Liquidity)")
                             osi_legs = [] # Invalidate legs to prevent entry
                        
                        # EXECUTE
                        t_ids = None 
                        if osi_legs and executor:
                            try:
                                t_ids = executor.submit_iron_condor(SYMBOL, osi_legs, quantity=qty)
                                print(f"🚀 SUBMITTED TO ALPACA! Quantity: {qty} | Trade IDs: {t_ids}")
                            except Exception as e:
                                print(f"❌ EXECUTION FAILED: {e}")
                        else:
                            print("⚠️ DRY RUN (No executor or unresolved legs)")
                        
                        # LOGGING
                        trade_logger.log_entry(spot, entry_score, confidence, prob_profit, osi_legs, t_ids)

                        new_trade = {
                            'entry_spot': spot, 
                            'short_call': target_short_call, 
                            'short_put': target_short_put, 
                            'entry_time': now, 
                            'qty': qty,
                            'legs': osi_legs,
                            'trade_ids': t_ids
                        }
                        active_trades.append(new_trade)
                        save_state(active_trades) # Update Persistence
                        running_stats['total_trades'] += 1
            
            # Update Running Stats
            running_stats['total_pnl_dollar'] = capital - starting_capital
            running_stats['peak_capital'] = max(running_stats['peak_capital'], capital)
            dd = ((running_stats['peak_capital'] - capital) / running_stats['peak_capital']) * 100
            running_stats['max_dd_pct'] = max(running_stats['max_dd_pct'], dd)

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

    # --- ENV VAR AUTO-POPULATION FROM CONFIG ---
    if not os.getenv('APCA_API_KEY_ID'):
        try:
            # Try to load keys from RunConfig (where user pasted them)
            try:
                from config import RunConfig
            except ImportError:
                 from core.config import RunConfig
            
            cfg = RunConfig()
            if cfg.alpaca_key and "YOUR" not in cfg.alpaca_key:
                print(f"🔑 Loaded Alpaca Keys from Config.")
                os.environ['APCA_API_KEY_ID'] = cfg.alpaca_key
                os.environ['APCA_API_SECRET_KEY'] = cfg.alpaca_secret
            else:
                 print("⚠️ Config has placeholder keys. Please set APCA_API_KEY_ID env var.")
        except Exception as e:
            print(f"⚠️ Failed to load keys from config: {e}")

    if not os.getenv('APCA_API_KEY_ID'):
        print("❌ Error: Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars.")
        sys.exit(1)
        
    try:
        # Initialize Trading Client explicitly
        from alpaca.trading.client import TradingClient
        ak = os.getenv('APCA_API_KEY_ID')
        sk = os.getenv('APCA_API_SECRET_KEY')
        trading_client = TradingClient(ak, sk, paper=args.paper)
        print(f"Initializing Alpaca (Paper={args.paper})...")

        # Use PaperExecutor with the real client for validation
        executor = PaperExecutor(run_config=None, trading_client=trading_client)
        
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
        print(f"✅ Model & Executor Ready. (Device: {DEVICE}) (Backbone: Neural CDE)")
        
        run_live_loop(executor, model, metadata, DEVICE)
        
    except Exception as e:
        print(f"Startup Failed: {e}")
        import traceback
        traceback.print_exc()
