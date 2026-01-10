import itertools
import copy
import time
import os
import numpy as np
import pandas as pd
import datetime
from core.backtest_engine import run_backtest_headless
from core.config import StrategyConfig, RunConfig
from tabulate import tabulate

# ==========================================================
# OPTIMIZATION SEGMENTS
# Each segment refines a specific aspect of the strategy.
# The best params from Segment N become the baseline for Segment N+1.
# ==========================================================
OPTIMIZATION_SEGMENTS = [
    {
        "name": "Phase 1: Exits & Risk (The Pacing)",
        "description": "Optimizing profit taking, stop-loss behavior, and trade duration.",
        "params": {
            "profit_take_pct": [0.5, 0.7, 0.9],                  # Trimmed: Focus on key levels
            "loss_close_multiple": [1.0, 2.0, 3.0],              # Trimmed: 1x, 2x, 3x
            "max_hold_days": [10, 21, 45],                       # Trimmed: Short, Med, Long
            "max_account_risk_per_trade": [0.02, 0.05]           # Trimmed: Conservative vs Aggressive
        }
    },
    {
        "name": "Phase 2: Structure & Entries (The Vehicle)",
        "description": "Optimizing delta targets, spread widths, and credit requirements.",
        "params": {
            "target_short_delta_low": [0.12, 0.16],
            "target_short_delta_high": [0.20, 0.30],
            "wing_width_min": [5.0, 10.0],
            "min_credit_to_width": [0.10, 0.20],
            "use_skew_penalty": [True] # Locked to True to save time
        }
    },
    {
        "name": "Phase 3: Filters & Regime (The Safety)",
        "description": "Refining volatility gates and regime thresholds.",
        "params": {
            "iv_rank_min": [0.0, 20.0, 40.0],
            "vix_threshold": [25.0, 35.0, 45.0],
            "vix_threshold_low": [15.0, 20.0],
            "max_volatility_pct": [0.03] # Locked
        }
    },
    {
        "name": "Phase 4: Momentum Logic (RSI, Stoch, ADX)",
        "description": "Optimizing momentum and trend strength indicators.",
        "params": {
            "rsi_neutral_min": [30, 40],
            "rsi_neutral_max": [70], # Trend following check
            "stoch_neutral_min": [20],
            "stoch_neutral_max": [80],
            "adx_threshold_low": [20.0, 30.0],
            "use_adx_filter": [True]
        }
    },
    {
        "name": "Phase 5: Trend & Volatility Indicators (BBands, SMA, PSAR)",
        "description": "Optimizing Bollinger Bands, SMA distance, and Parabolic SAR.",
        "params": {
            "bbands_squeeze_threshold": [0.02], # Default
            "sma_max_distance": [0.02, 0.04],
            "psar_acceleration": [0.02, 0.025],
            "psar_max_acceleration": [0.20],
            "use_psar_filter": [True, False]
        }
    },
    {
        "name": "Phase 6: Fuzzy Weights (The Blend)",
        "description": "Optimizing the 11-factor fuzzy weight blend including Neural Network influence.",
        "params": {
            # === ACTIVE WEIGHTS (6 total = 729 combinations) ===
            "fuzzy_weight_neural": [0.10, 0.20, 0.30],    # Neural Net influence
            "fuzzy_weight_mtf": [0.10, 0.18, 0.25],       # Multi-Timeframe consensus
            "fuzzy_weight_iv": [0.10, 0.14, 0.20],        # IV Rank
            "fuzzy_weight_regime": [0.08, 0.11, 0.15],    # VIX Regime
            "fuzzy_weight_rsi": [0.05, 0.10, 0.15],       # RSI neutrality
            "fuzzy_weight_adx": [0.05, 0.10, 0.15],       # Trend strength (ADX)
            
            # === COMMENTED OUT (Uncomment to expand search - WARNING: exponential growth!) ===
            # "fuzzy_weight_bbands": [0.05, 0.09, 0.12],  # Bollinger position/squeeze
            # "fuzzy_weight_stoch": [0.04, 0.08, 0.12],   # Stochastic momentum
            # "fuzzy_weight_volume": [0.04, 0.07, 0.10],  # Volume confirmation
            # "fuzzy_weight_sma": [0.03, 0.06, 0.09],     # SMA distance
            # "fuzzy_weight_psar": [0.04, 0.07, 0.10],    # Parabolic SAR
        }
    }
]

def run_optimization(base_s_cfg: StrategyConfig, run_cfg: RunConfig, auto_confirm: bool = True):
    """
    Run phased serial optimization with time estimation.
    """
    print("\n" + "="*60)
    print(" HIGH-FIDELITY SEGMENTED OPTIMIZER")
    print(" Target: Maximize Net Profit / Max Drawdown")
    print("="*60 + "\n")

    # 1. Automatic Benchmarking & Data Loading
    print("[1/4] Loading Data & Running Hardware Benchmark...")
    
    # Pre-load data once
    from data_factory.sync_engine import MTFSyncEngine
    
    csv_path = os.path.join("data", "spot", f"{base_s_cfg.underlying}_5.csv")
    if not os.path.exists(csv_path):
         # Fallback to daily or other available source
         csv_path = os.path.join("data", "spot", f"{base_s_cfg.underlying}_1.csv")

    if not os.path.exists(csv_path):
         print(f"[ERROR] No data found at {csv_path}. Cannot optimize.")
         return

    full_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp']).dt.tz_localize(None)
    full_df.set_index("timestamp", inplace=True)
    full_df.sort_index(inplace=True)
    
    # === PERFORMANCE OPTIMIZATION: Resample Spot Data to 15-Min ===
    print("[Speed] Resampling Spot Data to 15-Minute Bars (15T)...")
    full_df = full_df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    # ==============================================================

    # Filter by date range if specified (Crucial for performance)
    if hasattr(run_cfg, 'backtest_start') and run_cfg.backtest_start:
        start_dt = pd.Timestamp(run_cfg.backtest_start)
        full_df = full_df[full_df.index >= start_dt]
    if hasattr(run_cfg, 'backtest_end') and run_cfg.backtest_end:
        end_dt = pd.Timestamp(run_cfg.backtest_end) + pd.Timedelta(days=1)
        full_df = full_df[full_df.index < end_dt]
    
    # Select Options Data Source (determine BEFORE applying sample limit)
    using_custom_options = getattr(run_cfg, 'options_data_path', None) and os.path.exists(run_cfg.options_data_path)
    
    # Apply backtest_samples limit ONLY if NOT using custom high-fidelity options data
    # This ensures full date range coverage for validation runs
    if not using_custom_options:
        if run_cfg.backtest_samples and run_cfg.backtest_samples > 0:
            if len(full_df) > run_cfg.backtest_samples:
                full_df = full_df.iloc[-run_cfg.backtest_samples:]
    else:
        print(f"      [High-Fidelity Mode] Using FULL Spot Data ({len(full_df)} bars) for validation.")
            
    
    # Select Options Data Source
    if getattr(run_cfg, 'options_data_path', None) and os.path.exists(run_cfg.options_data_path):
        options_path = run_cfg.options_data_path
        print(f"      [Data] Using Configured Options File: {options_path}")
    else:
        options_path = os.path.join("data", "synthetic_options", f"{base_s_cfg.underlying.lower()}_options_marks.csv")
    
    if not os.path.exists(options_path):
        print(f"[ERROR] Options data not found at {options_path}.")
        return

    # Optimize memory usage with explicit types
    opt_dtypes = {
        'strike': 'float32', 
        'bid_1545': 'float32', 
        'ask_1545': 'float32', 
        'underlying_last': 'float32',
        'delta_1545': 'float32',
        'gamma_1545': 'float32',
        'vega_1545': 'float32',
        'theta_1545': 'float32',
        'iv_1545': 'float32'
    }

    # Optimized Chunked Loader to prevent OOM
    print("      ...Reading CSV in chunks to optimize memory...")
    
    # 1. Define dtypes for memory efficiency
    # Note: 'date' and 'expiration' are parsed as dates
    dtypes_map = {
        'strike': 'float32', 
        'bid': 'float32', 'ask': 'float32', 'last_price': 'float32',
        'delta': 'float32', 'gamma': 'float32', 'vega': 'float32', 'theta': 'float32',
        'implied_volatility': 'float32',
        'underlying_last': 'float32',
        'contract_type': 'category',
        'option_symbol': 'str' # 'string' or object
    }
    
    # Fallback mappings for old files
    dtypes_input = dtypes_map.copy()
    dtypes_input.update({
        'bid_1545': 'float32', 'ask_1545': 'float32',
        'delta_1545': 'float32', 'gamma_1545': 'float32',
        'vega_1545': 'float32', 'theta_1545': 'float32',
        'iv_1545': 'float32'
    })

    chunk_size = 500000
    chunks = []
    
    try:
        reader = pd.read_csv(
            options_path, 
            # parse_dates=["date", "expiration"], # Removed to prevent errors on files missing these cols
            chunksize=chunk_size,
            dtype=dtypes_input
        )
        
        for i, chunk in enumerate(reader):
            # Explicit Parsing to handle string dates
            if 'date' in chunk.columns and chunk['date'].dtype == 'object':
                chunk['date'] = pd.to_datetime(chunk['date'])
            if 'expiration' in chunk.columns and chunk['expiration'].dtype == 'object':
                chunk['expiration'] = pd.to_datetime(chunk['expiration'])
            if 'timestamp' in chunk.columns and chunk['timestamp'].dtype == 'object':
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp']).dt.tz_localize(None)

            # Map columns explicitly per chunk
            cols_map = {
                'bid_1545': 'bid', 'ask_1545': 'ask', 'cp_flag': 'contract_type',
                'delta_1545': 'delta', 'gamma_1545': 'gamma',
                'vega_1545': 'vega', 'theta_1545': 'theta', 
                'iv_1545': 'implied_volatility',
                'last_price': 'last_price', # identity
                # Expanded Intraday Mappings
                'delta_intraday': 'delta', 'gamma_intraday': 'gamma',
                'theta_intraday': 'theta', 'vega_intraday': 'vega',
                'rho_intraday': 'rho', 'iv_intraday': 'implied_volatility',
                'close': 'last_price', 'symbol': 'option_symbol'
            }
            chunk.rename(columns=cols_map, inplace=True)
            
            # Robust defaults for missing columns
            for col in ['delta', 'gamma', 'vega', 'theta', 'implied_volatility']:
                if col not in chunk.columns:
                    chunk[col] = 0.0
            
            # Ensure float32 (in case default was 0.0 float64)
            float_cols = ['strike', 'bid', 'ask', 'last_price', 'delta', 'gamma', 'vega', 'theta', 'implied_volatility']
            for c in float_cols:
                if c in chunk.columns:
                    chunk[c] = chunk[c].astype('float32')

            if 'contract_type' not in chunk.columns:
                if 'type' in chunk.columns: chunk['contract_type'] = chunk['type']
                elif 'cp_flag' in chunk.columns: chunk['contract_type'] = chunk['cp_flag']
                else: chunk['contract_type'] = 'C' # Fallback
            
            # Convert category
            if chunk['contract_type'].dtype == 'object':
                 chunk['contract_type'] = chunk['contract_type'].astype('category')
                
            if 'bid' not in chunk.columns: 
                if 'last_price' in chunk.columns: chunk['bid'] = chunk['last_price']
                else: chunk['bid'] = 0.0
            if 'ask' not in chunk.columns: 
                if 'last_price' in chunk.columns: chunk['ask'] = chunk['last_price']
                else: chunk['ask'] = 0.0
            
            if 'last_price' not in chunk.columns:
                 chunk['last_price'] = (chunk['bid'] + chunk['ask']) / 2.0
                 
            # Create option_symbol if missing (iVolatility style)
            if 'option_symbol' not in chunk.columns and 'expiration' in chunk.columns and 'strike' in chunk.columns:
                 chunk['option_symbol'] = (
                    "SPY_" + 
                    chunk['expiration'].astype(str) + "_" + 
                    chunk['contract_type'].astype(str) + "_" + 
                    chunk['strike'].astype(str)
                 )
            
            # Parse components from option_symbol if missing (Expanded style)
            if 'option_symbol' in chunk.columns and ('expiration' not in chunk.columns or 'strike' not in chunk.columns):
                # Regex for standard OCC: Root(6) + YYMMDD + T(1) + Strike(8) = 21 chars typically
                # Example: SPY250725C00613000
                # We assume standard format
                extracted = chunk['option_symbol'].str.extract(r'([A-Z]+)(\d{6})([CP])(\d{8})')
                if not extracted.empty and extracted.isnull().sum().sum() == 0:
                    chunk['expiration'] = pd.to_datetime(extracted[1], format='%y%m%d', errors='coerce')
                    chunk['contract_type'] = extracted[2]
                    chunk['strike'] = extracted[3].astype(float) / 1000.0
                    if 'date' not in chunk.columns and 'timestamp' in chunk.columns:
                         chunk['date'] = pd.to_datetime(chunk['timestamp']).dt.date
                         # Ensure date is date object
                         chunk['date'] = pd.to_datetime(chunk['date'])

            # Prune to essentials (Exclude volume/oi to save RAM)
            essential_cols = [
                'date', 'timestamp', 'expiration', 'strike', 'contract_type', 
                'bid', 'ask', 'last_price',
                'delta', 'gamma', 'vega', 'theta', 'implied_volatility',
                'option_symbol'
            ]
            
            # Ensure only existing are selected
            final_cols = [c for c in essential_cols if c in chunk.columns]
            chunk = chunk[final_cols]
            
            # Filter by Date
            if hasattr(run_cfg, 'backtest_start') and run_cfg.backtest_start:
                s_dt = pd.Timestamp(run_cfg.backtest_start)
                # Debug Dates
                if i == 0:
                    print(f"      [DEBUG] Filter Start: {s_dt}")
                    if 'date' in chunk.columns:
                        print(f"      [DEBUG] Chunk Dates (Head): {chunk['date'].head().values}")
                        print(f"      [DEBUG] Chunk Dates (Tail): {chunk['date'].tail().values}")
                    else:
                        print(f"      [DEBUG] 'date' column MISSING in chunk!")
                        
                chunk = chunk[chunk['date'] >= s_dt]
            
            if hasattr(run_cfg, 'backtest_end') and run_cfg.backtest_end:
                e_dt = pd.Timestamp(run_cfg.backtest_end) + pd.Timedelta(days=1)
                chunk = chunk[chunk['date'] < e_dt]
                
            if chunk.empty:
                if i == 0: print("      [DEBUG] Chunk 0 empty after filtering!")
                continue

            # Deduplicate within chunk to save RAM
            # Deduplicate within chunk to save RAM (Conditional)
            if 'timestamp' in chunk.columns and 'option_symbol' in chunk.columns:
                 chunk.drop_duplicates(subset=['timestamp', 'option_symbol'], keep='last', inplace=True)
            elif 'date' in chunk.columns and 'option_symbol' in chunk.columns:
                 chunk.drop_duplicates(subset=['date', 'option_symbol'], keep='last', inplace=True)

            chunks.append(chunk)
            
            if i % 5 == 0:
                print(f"      ...Processed chunk {i+1}...", end="\r")
                
        print(f"      ...Consolidating {len(chunks)} chunks...")
        options_df = pd.concat(chunks, ignore_index=True, copy=False)
        
        # Deduplicate and Sort globally
        print(f"      ...Sorting and Deduplicating {len(options_df)} records...")
        
        if 'timestamp' in options_df.columns:
             options_df.sort_values(by=['timestamp', 'option_symbol'], inplace=True)
             options_df.drop_duplicates(subset=['timestamp', 'option_symbol'], keep='last', inplace=True)
        elif 'date' in options_df.columns and 'option_symbol' in options_df.columns:
             options_df.sort_values(by=['date', 'option_symbol'], inplace=True)
             options_df.drop_duplicates(subset=['date', 'option_symbol'], keep='last', inplace=True)
             
        print(f"      ...Unique records: {len(options_df)}")
        del chunks
        import gc; gc.collect()
        
    except Exception as e:
        print(f"[ERROR] Failed to load options data: {e}")
        return

    # Optimized: Indexing for fast retrieval
    print("      ...Indexing data for Backtest Engine...")
    preloaded_options = {}
    
    if 'timestamp' in options_df.columns:
        # Intraday Path: Convert to {timestamp: {symbol: record_dict}}
        # This matches backtest_engine's Intraday expectation (Dict of Dicts)
        # Rename columns to match what `core/backtest_engine.py` Intraday logic expects
        # Expects: 'type' (call/put), 'price', 'iv', 'delta', 'expiration', 'strike'
        rename_map = {
            'contract_type': 'type',
            'last_price': 'price',
            'implied_volatility': 'iv'
        }
        
        # Ensure timestamp sorting before grouping
        options_df.sort_values('timestamp', inplace=True)
        
        grouped = options_df.groupby('timestamp')
        for ts, group in grouped:
            # Create formatted records
            g_renamed = group.rename(columns=rename_map)
            
            # Create {symbol: {data}} structure
            # We exclude 'timestamp' from the inner dict as it's the key
            cols_to_dict = [c for c in g_renamed.columns if c not in ['timestamp', 'option_symbol']]
            
            # Using set_index to create the dict structure efficiently
            # Force conversion to native Python types for safety (float, str) where possible
            # But mostly handled by to_dict
            records = g_renamed.set_index('option_symbol')[cols_to_dict].to_dict('index')
            
            # Key Normalization: Ensure Naive Datetime for lookup
            # BacktestEngine strips tzinfo: dt_now.replace(tzinfo=None)
            ts_key = pd.Timestamp(ts).tz_localize(None).to_pydatetime()
            preloaded_options[ts_key] = records
            
    else:
        # Daily Path: Dict of DataFrames (Reference, no copy)
        # Matches backtest_engine's Daily expectation (DataFrame itertuples)
        for date, group in options_df.groupby('date'):
            preloaded_options[date.date()] = group
    
    # Final cleanup
    del options_df
    import gc
    gc.collect()

    # === DEBUG: Check Keys Alignment ===
    print(f"      [DEBUG] Spot Data Type (Index): {type(full_df.index)}")
    if len(full_df) > 0:
        print(f"      [DEBUG] Spot Sample (First 3): {full_df.index[:3]}")
    else:
        print(f"      [DEBUG] Spot Data is EMPTY!")

    if len(preloaded_options) > 0:
        sample_keys = list(preloaded_options.keys())[:3]
        print(f"      [DEBUG] Options Keys Type (Element): {[type(k) for k in sample_keys]}")
        print(f"      [DEBUG] Options Keys Sample (First 3): {sample_keys}")
    else:
        print(f"      [DEBUG] Options Data (preloaded_options) is EMPTY!")
    # ===================================

    # Pre-load MTF Sync Engine if requested
    preloaded_sync = None
    if run_cfg.use_mtf:
        preloaded_sync = MTFSyncEngine(base_s_cfg.underlying, run_cfg.mtf_timeframes)

    # === OPTIMIZATION: Pre-compute Neural Forecasts ONCE ===
    preloaded_neural = None
    if getattr(base_s_cfg, 'use_mamba_model', False):
        try:
            import pandas_ta as ta
            # Pre-calculate indicators on full_df (once)
            if 'rsi_14' not in full_df.columns:
                full_df['rsi_14'] = ta.rsi(full_df['close'], length=14)
            if 'atr_pct' not in full_df.columns:
                atr = ta.atr(full_df['high'], full_df['low'], full_df['close'], length=14)
                full_df['atr_pct'] = atr / full_df['close']
            if 'volume_ratio' not in full_df.columns:
                vol_sma = ta.sma(full_df['volume'], length=20)
                full_df['volume_ratio'] = full_df['volume'] / (vol_sma + 1.0)
            
            # Pre-compute Mamba signals (GPU batch)
            from intelligence.mamba_engine import MambaForecastEngine
            d_model = getattr(base_s_cfg, 'mamba_d_model', 256)
            layers = getattr(base_s_cfg, 'mamba_layers', 12)
            mamba_engine = MambaForecastEngine(d_model=d_model, layers=layers)
            preloaded_neural = mamba_engine.precompute_all(full_df, batch_size=4096)
            print(f"[Optimizer] Neural Signals Pre-Computed ONCE: {len(preloaded_neural)} bars")
        except Exception as e:
            print(f"[Optimizer] Neural pre-compute failed: {e}")

    bench_start = time.time()
    baseline_strat = run_backtest_headless(
        base_s_cfg,
        run_cfg,
        preloaded_df=full_df,
        preloaded_options=preloaded_options,
        preloaded_sync=preloaded_sync,
        preloaded_neural_forecasts=preloaded_neural,
        verbose=False, # Baseline run can be quiet
    )
    bench_end = time.time()
    baseline_duration = bench_end - bench_start
    print(f"  -> Baseline Backtest (Cached): {baseline_duration:.2f} seconds\n")

    # 2. Estimate Total Time
    total_combos_all_phases = 0
    for seg in OPTIMIZATION_SEGMENTS:
        keys = list(seg['params'].keys())
        values = list(seg['params'].values())
        total_combos_all_phases += len(list(itertools.product(*values)))

    estimated_total_sec = baseline_duration * total_combos_all_phases
    estimated_min = estimated_total_sec / 60
    
    print(f"[2/4] Optimization Plan (Segmented):")
    for i, seg in enumerate(OPTIMIZATION_SEGMENTS):
        vals = list(seg['params'].values())
        combos = len(list(itertools.product(*vals)))
        print(f"  Phase {i+1}: {seg['name']} | {combos} combos")
    print(f"  -> Total Combinations: {total_combos_all_phases}")
    print(f"  -> Estimated Time: {estimated_min:.1f} minutes")
    print("-" * 30)
    
    if not auto_confirm:
        confirm = input("Proceed with optimization? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    else:
        print("Auto-confirming proceed...")

    # 3. Execution Loop
    current_best_cfg = copy.deepcopy(base_s_cfg)
    cumulative_results = []
    
    start_time_global = time.time()

    for phase_idx, segment in enumerate(OPTIMIZATION_SEGMENTS):
        print(f"\n[{phase_idx+1}/{len(OPTIMIZATION_SEGMENTS)}] Executing: {segment['name']}")
        print(f"  Desc: {segment['description']}")
        
        keys = list(segment['params'].keys())
        values = list(segment['params'].values())
        combinations = list(itertools.product(*values))
        phase_results = []
        
        phase_total = len(combinations)
        print(f"  Testing {phase_total} combinations...")
        
        for i, combo in enumerate(combinations):
            # Start with the best config from previous phase
            s_cfg = copy.deepcopy(current_best_cfg)
            
            # Apply current phase params
            params_dict = dict(zip(keys, combo))
            for k, v in params_dict.items():
                setattr(s_cfg, k, v)
            
            # Display iteration info with settings
            settings_str = " | ".join([f"{k}={v}" for k, v in params_dict.items()])
            print(f"  [{i+1}/{phase_total}] {settings_str}")
                
            # Run Backtest (with pre-computed neural signals)
            strat = run_backtest_headless(s_cfg, run_cfg, 
                                          preloaded_df=full_df, 
                                          preloaded_options=preloaded_options,
                                          preloaded_sync=preloaded_sync,
                                          preloaded_neural_forecasts=preloaded_neural,
                                          verbose=False)
            
            if strat is not None:
                # Capture Metrics
                # Capture Metrics
                # FIX: Strategy uses Manual Accounting (Simulated Broker).
                # Current Equity = Realized PnL (strat.pnl) + Unrealized PnL (strat.current_unrealized_pnl)
                # Backtrader broker.get_value() is invalid because self.buy() is not called in this mode.
                realized_pnl = getattr(strat, 'pnl', 0.0)
                unrealized_pnl = getattr(strat, 'current_unrealized_pnl', 0.0)
                
                net_profit = realized_pnl + unrealized_pnl
                
                # Debug output to verify fix (temporary, can be removed later)
                # print(f"       [DEBUG] Realized=${realized_pnl:.0f} Unrealized=${unrealized_pnl:.0f} Net=${net_profit:.0f}")
                
                # net_profit = strat.pnl # OLD: Only counted closed trades
                max_dd = max(strat.drawdowns) if strat.drawdowns else 0.0
                np_dd_ratio = net_profit / max_dd if max_dd > 0 else 0.0
                
                # Basic stats for reporting
                wins = [t for t in strat.trade_log if t["result"] == "win"]
                losses = [t for t in strat.trade_log if t["result"] == "loss"]
                gross_profit = sum(t["amount"] for t in wins)
                gross_loss = abs(sum(t["amount"] for t in losses))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
                win_rate = (len(wins) / len(strat.trade_log) * 100) if strat.trade_log else 0.0
                
                # Show quick result
                print(f"       -> NP=${net_profit:.0f} | DD=${max_dd:.0f} | Ratio={np_dd_ratio:.2f} | Trades={len(strat.trade_log)}")
                
                result_entry = {
                    "phase": phase_idx + 1,
                    "params": params_dict, # Only the params changed in this phase
                    "full_config": copy.deepcopy(s_cfg), # Snapshot of full state
                    "net_profit": net_profit,
                    "max_dd": max_dd,
                    "np_dd_ratio": np_dd_ratio,
                    "profit_factor": profit_factor,
                    "trades": len(strat.trade_log),
                    "win_rate": win_rate
                }
                phase_results.append(result_entry)
            else:
                print(f"       -> [FAILED] No results")

        print(f"  Phase Complete: {phase_total}/{phase_total} [DONE]")
        
        # Analyze Phase Results
        if not phase_results:
            print("  [WARNING] No valid results in this phase. Keeping previous config.")
            continue
            
        # Sort by NP/DD Ratio
        sorted_phase = sorted(phase_results, key=lambda x: x['np_dd_ratio'], reverse=True)
        best_in_phase = sorted_phase[0]
        
        print(f"  -> Best Result in Phase: NP=${best_in_phase['net_profit']:.0f} | DD=${best_in_phase['max_dd']:.0f} | Ratio={best_in_phase['np_dd_ratio']:.2f}")
        print(f"  -> Params: {best_in_phase['params']}")
        
        # Upgrade the global best config for the next phase
        current_best_cfg = best_in_phase['full_config']
        cumulative_results.extend(sorted_phase) # Keep history

    total_time = (time.time() - start_time_global) / 60
    print(f"\n[4/4] Optimization Complete in {total_time:.1f} minutes.")
    
    # 4. Reporting (Top Global)
    # We want to see the best configuration encountered at the very end (or best of final phase)
    # But usually the last phase contains the most refined version.
    # Let's show the final config parameters.
    
    print("\n" + "="*80)
    print(" FINAL OPTIMIZED CONFIGURATION")
    print("="*80)
    
    final_params = {}
    # Extract keys we care about from the final config object
    relevant_keys = set()
    for seg in OPTIMIZATION_SEGMENTS:
        relevant_keys.update(seg['params'].keys())
        
    for k in relevant_keys:
        final_params[k] = getattr(current_best_cfg, k)
        print(f"  {k} = {final_params[k]}")

    # Save to file
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join("reports", f"optimization_final_{now_str}.csv")
    os.makedirs("reports", exist_ok=True)
    
    # Save the Cumulative top 50 across all phases (just for analysis)
    top_50 = sorted(cumulative_results, key=lambda x: x['np_dd_ratio'], reverse=True)[:50]
    csv_data = []
    for rank, res in enumerate(top_50, 1):
        row = {
            "Rank": rank,
            "Phase": res['phase'],
            "NP_DD_Ratio": round(res['np_dd_ratio'], 2),
            "NetProfit": round(res['net_profit'], 2),
            "MaxDD": round(res['max_dd'], 2),
            "ProfitFactor": round(res['profit_factor'], 2),
            "Trades": res['trades'],
            "WinRate": round(res['win_rate'], 2)
        }
        # Flatten params
        for k, v in res['params'].items():
            row[f"param_{k}"] = v
        csv_data.append(row)
        
    pd.DataFrame(csv_data).to_csv(report_path, index=False)
    print(f"\n[Report Saved] {report_path}")

    # Apply Best Params to Config Files
    print(f"\n[Auto-Apply] Updating configuration files with best parameters...")
    apply_best_params(final_params)


def apply_params_to_file(params, config_path):
    """Update a specific file with parameters."""
    if not os.path.exists(config_path):
        return False
        
    with open(config_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        updated = False
        stripped = line.strip()
        if "=" in stripped:
            first_part = stripped.split("=")[0].strip()
            key_candidate = first_part.split(":")[0].strip()
            
            if key_candidate in params:
                val = params[key_candidate]
                # Format float/int correctly
                if isinstance(val, float):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)

                indent = line[:line.find(line.strip())] if line.strip() else ""
                parts = line.split("#")
                comment = "#" + parts[1] if len(parts) > 1 else ""
                
                # Reconstruct
                if ":" in first_part:
                    type_part = first_part.split(":")[1]
                    new_line = f"{indent}{key_candidate}: {type_part} = {val_str}  {comment}\n"
                else:
                    new_line = f"{indent}{key_candidate} = {val_str}  {comment}\n"
                
                new_lines.append(new_line)
                updated = True
        
        if not updated:
            new_lines.append(line)
            
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    return True

def apply_best_params(params):
    """
    Update both core/config.template.py and core/config.py with the chosen parameters.
    """
    targets = ["core/config.template.py", "core/config.py"]
    for t in targets:
        success = apply_params_to_file(params, t)
        if success:
            print(f"  -> Updated: {t}")
