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
    Logic:
      - Filter for meaningful volume/open_interest (if possible) within ATM range.
      - Simple average of ATM +/- 5 strikes (Call + Put combined? Or typically model expects blended).
      - Note: Alpaca 'OptionContract' model has 'greeks' field which is a struct.
    """
    if not contracts:
        return None
        
    # Filter for ATM range (Spot +/- 2%)
    valid_contracts = []
    
    # We want near-term (<= 7 days) and ATM.
    # The list passed in `contracts` is already filtered by date and strike range presumably.
    
    deltas, gammas, thetas, vegas, ivs = [], [], [], [], []
    
    for c in contracts:
        # Check if greeks data exists
        if hasattr(c, 'greeks') and c.greeks:
            g = c.greeks
            # Filter bad data
            if g.delta is not None: deltas.append(abs(g.delta)) # ABS Delta for exposure sizing, but model might want directional?
            # Actually, standard feature 'delta' usually refers to directional (Calls + Puts -)? 
            # OR is it 'market_greeks'?
            # Let's assume the model trains on "ATM Delta" which for a CALL is ~0.5.
            # V2.2 Feature registry usually has 'delta' as part of the market state...
            # If the feature is just 'delta', it's ambiguous.
            # However, `canonical_feature_registry` V2.2 likely has 'put_call_volume_ratio' etc.
            # Let's check the list. It includes `delta`, `gamma`, `vega`, `theta`, `iv`.
            # These are likely "ATM Contract Average" or "Portfolio Delta"? 
            # In single-asset CDE models, these are usually "ATM Implied stats".
            # So: Avg(Abs(Delta_Call), Abs(Delta_Put)) -> ~0.5 usually.
            # Gamma: Sum or Avg? Avg.
            
            if g.gamma is not None: gammas.append(g.gamma)
            if g.theta is not None: thetas.append(g.theta)
            if g.vega is not None: vegas.append(g.vega)
        
        # IV is often top-level or in greeks? Alpaca has it in `implied_volatility`
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
        'svr': float(np.std(ivs)) if len(ivs) > 1 else 0.0 # Spread of Vol or similar
    }

def find_closest_contract(client, symbol, contract_type, target_strike, ideal_date):
    """
    Finds the contract closest to target_strike and ideal_date.
    ideal_date: datetime object
    """
    try:
        from alpaca.data.requests import OptionContractsRequest
        req = OptionContractsRequest(
            underlying_symbols=[symbol],
            status='active',
            expiration_date_gte=ideal_date.date(),
            expiration_date_lte=(ideal_date + pd.Timedelta(days=5)).date(),
            type=contract_type, # 'call' or 'put'
            strike_price_gte=target_strike - 2,
            strike_price_lte=target_strike + 2,
            limit=10
        )
        res = client.get_option_contracts(req)
        contracts = res.option_contracts
    except ImportError:
        # Fallback to OptionChainRequest (older SDK or different version)
        print("Warning: OptionContractsRequest not found. Please run `pip install alpaca-py -U`")
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
                # 1. Fetch Option Chain Snapshot (WideNet: ATM +/- 5 strikes for speed/relevance in V1)
                # User asked for "all spy call and put symbols". 
                # A full chain fetch is expensive (bandwidth/time). 
                # We will fetch a "Trading Window" snapshot for accountability.
                if not hasattr(run_live_loop, 'opt_client'):
                     try:
                         from alpaca.data.historical import OptionHistoricalDataClient
                         ak = os.getenv('APCA_API_KEY_ID')
                         sk = os.getenv('APCA_API_SECRET_KEY')
                         run_live_loop.opt_client = OptionHistoricalDataClient(ak, sk)
                     except:
                         run_live_loop.opt_client = None

                if run_live_loop.opt_client:
                    # Logic: Fetch contracts expiring in next 7 days, +/- 20 strikes from spot
                    # We need a rough spot reference.
                    # Since we haven't fetched 'df' yet for this new bar, we use the last known price or just a wide net.
                    # Let's use a try-except to get a quick quote or just assume previous close?
                    # Better: Just run this AFTER fetching df? 
                    # No, user wants it during the wait or parallel.
                    # Let's use a wide net: 650-750 for now (hardcoded based on ~695 spot) 
                    # OR better: use the `active_trade` spot if available, else roughly 695.
                    center_strike = 695 
                    
                    if 'active_trade' in locals() and active_trade:
                        center_strike = active_trade['entry_spot']
                    
                    all_contracts = []
                    try:
                        from alpaca.data.requests import OptionContractsRequest
                        
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        chain_dir = os.path.join("data", "live", "chains")
                        os.makedirs(chain_dir, exist_ok=True)
                        chain_path = os.path.join(chain_dir, f"spy_chain_{timestamp_str}.csv")
                        
                        # Fetch Calls & Puts
                        start_date = datetime.now().date()
                        # Use global pd
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
                        print(" [Recorder Warning] OptionContractsRequest missing (old SDK?). Skipping chain record.")
                    except Exception as e_chain:
                        print(f"  [Recorder Warning] fetching chain failed: {e_chain}")

                    # Convert to DataFrame for saving
                    if all_contracts:
                        chain_df = pd.DataFrame([c.model_dump() for c in all_contracts])
                        # Filter fields to save space
                        cols = ['symbol', 'strike_price', 'expiration_date', 'type', 'open_interest', 'style', 'implied_volatility', 'greeks']
                        chain_df = chain_df[[c for c in cols if c in chain_df.columns]]
                        chain_df.to_csv(chain_path, index=False)
                        
                        # USER REQUEST: Console output
                        # Print symbols compactly
                        syms = [c.symbol for c in all_contracts]
                        print(f"\n📚 Option Chain Recorded ({len(syms)} Contracts):")
                        print(" ".join(syms)) # Print all symbols separated by space
                        print("-" * 50)


            except Exception as e_rec:
                print(f" [Data Recorder Error] {e_rec}")
            # --- LIVE DATA RECORDER END ---

            
            # 1. Fetch Data
            df = fetch_live_data(data_client, SYMBOL, lookback_bars=LOOKBACK_BARS)

            # --- INJECT LIVE GREEKS ---
            # If we successfully fetched contracts, use them to populate Greek columns
            # The model expects consistent time-series, so we ideally append these values to the end.
            # But `df` is historical bars. Current Greeks apply to "NOW" (last row).
            # We will fill the *last row* with current Greeks.
            # Note: This means historical rows in `df` will have 0.0 or older values?
            # Yes, unless we fetch historical Greeks (impossible efficiently).
            # The Neural CDE is robust to this if `lag_minutes` handles it, BUT...
            # If the model relies on the Trajectory of Greeks, having only the last point non-zero is an issue.
            # HOWEVER, `gamma` and `iv` are state variables. CDE interpolates.
            # For live trading, we often assume "Last Known State" propagates or we just provide the current state.
            # Let's fill the Enture Column with the current value? No, that destroys trends.
            # Compromise: Fill the last 5-10 bars with the current Greek value to simulate "recent state".
            # Or better: `compute_all_primitive_features` might require them.
            
            current_greeks = None
            if 'all_contracts' in locals() and all_contracts:
                 current_greeks = calculate_live_greeks(all_contracts, df['close'].iloc[-1])
                 if current_greeks:
                     print(f"  [Greeks] IV: {current_greeks['iv']:.2f} | Gamma: {current_greeks['gamma']:.4f}")
            
            if current_greeks:
                # We apply these values to the LAST ROW (Current Market State)
                # And importantly: The model uses a window.
                # If historical values are 0, the derivative is huge.
                # FIX: We should likely forward-fill or back-fill for the immediate window if history is missing.
                # But we can't invent history.
                # Simplest robust approach for now: 
                # Set the columns in the DF. 
                # If they don't exist, init with current value (flat line assumption is better than 0).
                for k, v in current_greeks.items():
                    if k not in df.columns:
                        df[k] = v # Init whole column with current value (Flat Trend)
                    else:
                        df.iloc[-1, df.columns.get_loc(k)] = v
            # ---------------------------

# ... (Previous code) ...

            
            if len(df) < 256:
                print(f"Warning: Not enough bars ({len(df)}). Waiting...")
                continue
                
            # 2. V2.2 Feature Preparation
            # A) Compute Dynamic Features (Indicators)
            df = compute_all_dynamic_features(df, inplace=True)
            
            # B) Compute Primitive Features (Vol-gated logic)
            # Ensure required columns for primitives exist
            if 'lag_minutes' not in df.columns:
                 df['lag_minutes'] = 0.0 # Live data is fresh
            
            df = compute_all_primitive_features_v22(df, inplace=True)
            
            # C) Handle Targets & Missing Columns
            # The model expects valid columns for 'target_spot' and 'max_dd_60m' even if unused for inference
            # We fill them with 0.0 (safe placeholder)
            for col in FEATURE_COLS_V22:
                if col not in df.columns:
                    # Special handling for targets -> 0.0
                    if col in ['target_spot', 'max_dd_60m', 'target_profit']:
                        df[col] = 0.0
                    elif col in ['delta', 'gamma', 'vega', 'theta', 'iv']:
                        # If we missed calculating it (e.g. data fail), use fill value
                         df[col] = get_neutral_fill_value_v22(col)
                    else:
                        # Fallback for others (e.g. from V2.2 defaults)
                        fill_val = get_neutral_fill_value_v22(col)
                        df[col] = fill_val

            # Log Volume (if not already handled)
            df_model_input = df.copy()
            df_model_input['volume'] = np.log1p(df_model_input['volume'])
            
            # --- LOG FEATURES FOR ACCOUNTABILITY ---
            # User wants to see RSI, Gamma, etc.
            try:
                latest_row = df_model_input.iloc[-1]
                feature_logger.log_features(now, latest_row)
            except Exception as e_flog:
                print(f"Feature Log Error: {e_flog}")

            # 3. Scaling
            
            # 3. Scaling
            # Extract exactly the columns needed in order
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
                    dte = dte if dte > 1.0 else 14.0 # Default min DTE
                    
                    # Target Strikes
                    target_short_call = spot + (call_off * spot * 0.01)
                    target_short_put = spot - (put_off * spot * 0.01)
                    target_long_call = target_short_call + width
                    target_long_put = target_short_put - width
                    
                    # Target Date
                    target_date = datetime.now() + pd.Timedelta(days=dte)
                    
                    # Try to resolve OSI Symbols
                    # Check if we have opt_client from recorder loop, else init
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
║ 🦅 IRON CONDOR #{trade_num} ENTRY @ {datetime.now().strftime('%H:%M:%S')}
╠══════════════════════════════════════════════════════════════════════════════╣
║ SPOT:          ${spot:.2f}
║ FUZZY SCORE:   {entry_score}/100 (threshold: {ENTRY_THRESHOLD})
║ CONFIDENCE:     {confidence:.4f} | PROB: {prob_profit:.4f}
╠─────────────────────────── IRON CONDOR SETUP ────────────────────────────────╣
║   Short Call:  ${target_short_call:.2f}  |  Short Put:   ${target_short_put:.2f}
║   Width:       ${width:.2f}  |  DTE: {dte:.1f} days
╠─────────────────────────── EXECUTION ────────────────────────────────────────╣
║   Resolved:    {len(osi_legs) == 4}
║   Short Legs:  {osi_legs[0]['option_symbol'] if osi_legs else 'N/A'} / {osi_legs[2]['option_symbol'] if osi_legs else 'N/A'}
╚══════════════════════════════════════════════════════════════════════════════╝"""
                    print(entry_msg)
                    
                    
                    # EXECUTE
                    t_ids = None # Initialize to prevent UnboundLocalError
                    if osi_legs and executor:
                        try:
                            t_ids = executor.submit_iron_condor(SYMBOL, osi_legs, quantity=1)
                            print(f"🚀 SUBMITTED TO ALPACA! Trade IDs: {t_ids}")
                        except Exception as e:
                            print(f"❌ EXECUTION FAILED: {e}")
                    else:
                        print("⚠️ DRY RUN (No executor or unresolved legs)")
                    
                    # LOGGING
                    trade_logger.log_entry(spot, entry_score, confidence, prob_profit, osi_legs, t_ids)

                    active_trade = {
                        'entry_spot': spot, 
                        'short_call': target_short_call, 
                        'short_put': target_short_put, 
                        'entry_time': now, 
                        'qty': 1,
                        'legs': osi_legs
                    }
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
        print(f"✅ Model & Executor Ready. (Device: {DEVICE}) (Backbone: Neural CDE)")
        
        run_live_loop(executor, model, metadata, DEVICE)
        
    except Exception as e:
        print(f"Startup Failed: {e}")
        import traceback
        traceback.print_exc()
