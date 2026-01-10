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
            "profit_take_pct": np.arange(0.40, 0.95, 0.1),       # 0.4, 0.5, ... 0.9
            "loss_close_multiple": np.arange(1.0, 3.1, 0.5),     # 1.0, 1.5, 2.0, 2.5, 3.0
            "max_hold_days": [10, 14, 21, 30],                   # Hold duration cap
            "max_account_risk_per_trade": [0.01, 0.02, 0.03]
        }
    },
    {
        "name": "Phase 2: Structure & Entries (The Vehicle)",
        "description": "Optimizing delta targets, spread widths, and credit requirements.",
        "params": {
            "target_short_delta_low": [0.10, 0.12, 0.15],
            "target_short_delta_high": [0.20, 0.25, 0.30],
            "wing_width_min": [5.0, 10.0],
            "min_credit_to_width": [0.10, 0.15, 0.20],
            "use_skew_penalty": [True, False]
        }
    },
    {
        "name": "Phase 3: Filters & Regime (The Safety)",
        "description": "Refining volatility gates and regime thresholds.",
        "params": {
            "iv_rank_min": [0.0, 10.0, 20.0, 30.0],
            "vix_threshold": [25.0, 30.0, 40.0],
            "vix_threshold_low": [15.0, 18.0, 20.0],
            "max_volatility_pct": [0.02, 0.03, 0.04]
        }
    },
    {
        "name": "Phase 4: Momentum Logic (RSI, Stoch, ADX)",
        "description": "Optimizing momentum and trend strength indicators.",
        "params": {
            "rsi_neutral_min": [30, 40],
            "rsi_neutral_max": [60, 70],
            "stoch_neutral_min": [20, 30],
            "stoch_neutral_max": [70, 80],
            "adx_threshold_low": [20.0, 25.0, 30.0], # Trend strength limit
            "use_adx_filter": [True, False]
        }
    },
    {
        "name": "Phase 5: Trend & Volatility Indicators (BBands, SMA, PSAR)",
        "description": "Optimizing Bollinger Bands, SMA distance, and Parabolic SAR.",
        "params": {
            "bbands_squeeze_threshold": [0.01, 0.02, 0.03],
            "sma_max_distance": [0.01, 0.02, 0.03, 0.04],
            "psar_acceleration": [0.02, 0.025],
            "psar_max_acceleration": [0.20, 0.25],
            "use_psar_filter": [True, False]
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

    # Filter by date range if specified (Crucial for performance)
    if hasattr(run_cfg, 'backtest_start') and run_cfg.backtest_start:
        start_dt = pd.Timestamp(run_cfg.backtest_start)
        full_df = full_df[full_df.index >= start_dt]
    if hasattr(run_cfg, 'backtest_end') and run_cfg.backtest_end:
        end_dt = pd.Timestamp(run_cfg.backtest_end) + pd.Timedelta(days=1)
        full_df = full_df[full_df.index < end_dt]
    
    if run_cfg.backtest_samples and run_cfg.backtest_samples > 0:
        if len(full_df) > run_cfg.backtest_samples:
            full_df = full_df.iloc[-run_cfg.backtest_samples:]
            
    options_path = os.path.join("data", "synthetic_options", f"{base_s_cfg.underlying.lower()}_options_marks.csv")
    if not os.path.exists(options_path):
        print(f"[ERROR] Synthetic options not found at {options_path}.")
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

    # Only load columns we actually need if possible, but for safety we load what we know
    # Not using usecols to avoid missing col errors, but float32 saves 50% RAM
    print("      ...Reading CSV with float32 precision to save RAM...")
    options_df = pd.read_csv(
        options_path, 
        parse_dates=["date", "expiration"], 
        dtype=opt_dtypes
    )
    
    # Prune immediately to essential columns to free more RAM
    essential_cols = [
        'date', 'expiration', 'strike', 'cp_flag', 
        'bid_1545', 'ask_1545', 'underlying_last', 
        'delta_1545', 'gamma_1545', 'vega_1545', 'theta_1545',
        'option_symbol', 'contract_type', 'type'
    ]
    # Filter only if they exist
    existing_essential = [c for c in essential_cols if c in options_df.columns]
    options_df = options_df[existing_essential]

    preloaded_options = {}
    for date, group in options_df.groupby('date'):
        preloaded_options[date.date()] = group.to_dict('records')
    
    # Manual garbage collection hint
    del options_df
    import gc
    gc.collect()

    # Pre-load MTF Sync Engine if requested
    preloaded_sync = None
    if run_cfg.use_mtf:
        preloaded_sync = MTFSyncEngine(base_s_cfg.underlying, run_cfg.mtf_timeframes)

    bench_start = time.time()
    baseline_strat = run_backtest_headless(
        base_s_cfg,
        run_cfg,
        preloaded_df=full_df,
        preloaded_options=preloaded_options,
        preloaded_sync=preloaded_sync,
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
        
        print(f"  Progress: 0/{len(combinations)}...", end="\r")
        
        for i, combo in enumerate(combinations):
            # Start with the best config from previous phase
            s_cfg = copy.deepcopy(current_best_cfg)
            
            # Apply current phase params
            params_dict = dict(zip(keys, combo))
            for k, v in params_dict.items():
                setattr(s_cfg, k, v)
                
            # Run Backtest
            strat = run_backtest_headless(s_cfg, run_cfg, 
                                          preloaded_df=full_df, 
                                          preloaded_options=preloaded_options,
                                          preloaded_sync=preloaded_sync,
                                          verbose=False)
            
            if strat is not None:
                # Capture Metrics
                net_profit = strat.pnl
                max_dd = max(strat.drawdowns) if strat.drawdowns else 0.0
                np_dd_ratio = net_profit / max_dd if max_dd > 0 else 0.0
                
                # Basic stats for reporting
                wins = [t for t in strat.trade_log if t["result"] == "win"]
                losses = [t for t in strat.trade_log if t["result"] == "loss"]
                gross_profit = sum(t["amount"] for t in wins)
                gross_loss = abs(sum(t["amount"] for t in losses))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
                win_rate = (len(wins) / len(strat.trade_log) * 100) if strat.trade_log else 0.0
                
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
            
            if (i+1) % 5 == 0:
                print(f"  Progress: {i+1}/{len(combinations)}...", end="\r")

        print(f"  Progress: {len(combinations)}/{len(combinations)} [DONE]")
        
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
