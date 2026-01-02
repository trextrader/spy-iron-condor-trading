import itertools
import copy
import time
import numpy as np
import pandas as pd
from core.backtest_engine import run_backtest_headless
from core.config import StrategyConfig, RunConfig
from tabulate import tabulate

# ==========================================================
# OPTIMIZATION MATRIX (Edit START, STOP, STEP here)
# ==========================================================
# Format: "param_name": np.arange(start, stop + step, step)
# Use np.arange for floats, range for ints if needed
# ==========================================================
OPTIMIZATION_MATRIX = {
    # Phase A: Exits (Optimized to Ratio: Net Profit / Max DD)
    "profit_take_pct": np.arange(0.10, 1.0, 0.2),
    "loss_close_multiple": np.arange(1.0, 5.0, 0.2),
    
    # Phase B: Entry (Uncomment params below to broaden the search)
    # "dte_min": range(7, 46, 1),
    # "dte_max": range(14, 91, 1),
    # "target_short_delta_low": np.arange(0.05, 0.26, 0.01),
    # "target_short_delta_high": np.arange(0.10, 0.36, 0.01),
    
    # Phase C: Filters
    #"iv_rank_min": np.arange(15, 55, 5),
    #"vix_threshold": np.arange(15, 55, 5),
}

def run_optimization(base_s_cfg: StrategyConfig, run_cfg: RunConfig):
    """
    Run phased serial optimization with time estimation.
    """
    print("\n" + "="*50)
    print(" HIGH-FIDELITY SERIAL OPTIMIZER")
    print(" Target: Maximize Net Profit / Max Drawdown")
    print("="*50 + "\n")

    # 1. Automatic Benchmarking & Data Loading
    print("[1/3] Loading Data & Running Hardware Benchmark...")
    
    # Pre-load data once
    import os
    csv_path = os.path.join("reports", base_s_cfg.underlying, f"{base_s_cfg.underlying}_5.csv")
    full_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp']).dt.tz_localize(None)
    full_df.set_index("timestamp", inplace=True)
    full_df.sort_index(inplace=True)
    
    if run_cfg.backtest_samples and run_cfg.backtest_samples > 0:
        if len(full_df) > run_cfg.backtest_samples:
            full_df = full_df.iloc[-run_cfg.backtest_samples:]
            
    options_path = os.path.join("data", "synthetic_options", f"{base_s_cfg.underlying}_5min.csv")
    options_df = pd.read_csv(options_path, parse_dates=["date", "expiration"])
    preloaded_options = {}
    for date, group in options_df.groupby('date'):
        preloaded_options[date.date()] = group.to_dict('records')

    bench_start = time.time()
    _ = run_backtest_headless(base_s_cfg, run_cfg, preloaded_df=full_df, preloaded_options=preloaded_options)
    bench_end = time.time()
    baseline_duration = bench_end - bench_start
    print(f"  -> Baseline Backtest (Cached): {baseline_duration:.2f} seconds\n")

    # 2. Build Grid
    keys = list(OPTIMIZATION_MATRIX.keys())
    values = list(OPTIMIZATION_MATRIX.values())
    combinations = list(itertools.product(*values))
    total_combos = len(combinations)
    
    estimated_total_sec = baseline_duration * total_combos
    estimated_min = estimated_total_sec / 60
    
    print(f"[2/3] Optimization Plan:")
    print(f"  -> Parameters: {', '.join(keys)}")
    print(f"  -> Total Combinations: {total_combos}")
    print(f"  -> Estimated Time: {estimated_min:.1f} minutes")
    print("-" * 30)
    
    confirm = input("Proceed with optimization? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # 3. Execution
    print(f"\n[3/3] Executing Grid Search...")
    results = []
    
    for i, combo in enumerate(combinations):
        s_cfg = copy.deepcopy(base_s_cfg)
        params_dict = dict(zip(keys, combo))
        
        # Apply params
        for k, v in params_dict.items():
            setattr(s_cfg, k, v)
            
        print(f"  [{i+1}/{total_combos}] Running...", end="\r")
        
        # Run Backtest
        strat = run_backtest_headless(s_cfg, run_cfg, preloaded_df=full_df, preloaded_options=preloaded_options)
        
        if strat is not None:
            net_profit = strat.pnl
            max_dd = max(strat.drawdowns) if strat.drawdowns else 0.0
            ratio = net_profit / max_dd if max_dd > 0 else (net_profit if net_profit > 0 else -999)
            
            # Calculate Sharpe (Annualized for 5-min bars: 78/day * 252 days)
            sharpe = 0.0
            if len(strat.equity_series) > 1:
                equity_curve = np.array(strat.equity_series)
                returns = np.diff(equity_curve) / equity_curve[:-1]
                if np.std(returns) > 0:
                    bars_per_year = 252 * 78
                    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)

            results.append({
                "params": params_dict,
                "net_profit": net_profit,
                "final_balance": run_cfg.backtest_cash + net_profit,
                "max_dd": max_dd,
                "ratio": ratio,
                "sharpe": sharpe,
                "trades": strat.trades,
                "win_rate": (strat.wins / strat.trades * 100) if strat.trades > 0 else 0
            })

    print(f"\n\nDone. Processing {len(results)} valid results...")

    # 4. Reporting (Top 100)
    sorted_results = sorted(results, key=lambda x: x['ratio'], reverse=True)
    top_100 = sorted_results[:100]
    
    table_headers = ["Rank", "Params", "Balance", "Profit", "Ratio", "Sharpe", "Trades", "Win %"]
    table_rows = []
    
    for rank, res in enumerate(top_100, 1):
        p = res['params']
        param_summary = ", ".join([f"{k}={v}" for k, v in p.items()])
        table_rows.append([
            rank,
            param_summary,
            f"${res['final_balance']:,.0f}",
            f"${res['net_profit']:,.0f}",
            f"{res['ratio']:.2f}",
            f"{res['sharpe']:.2f}",
            res['trades'],
            f"{res['win_rate']:.1f}%"
        ])

    print("\n" + "="*120)
    print(f" TOP {len(top_100)} OPTIMIZATION RESULTS (Ranked by Profit/DD)")
    print("="*120)
    print(tabulate(table_rows, headers=table_headers, tablefmt="simple"))
    print("="*120 + "\n")

    # 5. Apply Choice
    if sorted_results:
        print("Selection Interface:")
        print("  - Enter '1' to '100' to apply a specific configuration.")
        print("  - Enter 'n' to quit without applying.")
        
        choice = input("\nSelect Rank to apply: ")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_results):
                chosen = sorted_results[idx]
                print(f"APPLYING CONFIG #{choice}: {chosen['params']}")
                apply_best_params(chosen['params'])
                print("Configuration updated in core/config.template.py.")
            else:
                print("Invalid rank.")
        else:
            print("Selection ignored.")

def apply_params_to_file(params, config_path):
    """Update a specific file with parameters."""
    if not os.path.exists(config_path):
        return False
        
    with open(config_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        updated = False
        # We only want to replace lines that look like: key = val or key: type = val
        stripped = line.strip()
        if "=" in stripped:
            first_part = stripped.split("=")[0].strip()
            # Handle type hints: "dte_min: int" -> "dte_min"
            key_candidate = first_part.split(":")[0].strip()
            
            if key_candidate in params:
                val = params[key_candidate]
                indent = line[:line.find(line.strip())]
                parts = line.split("#")
                comment = "#" + parts[1] if len(parts) > 1 else ""
                
                # Reconstruct with original formatting / type hints
                if ":" in first_part:
                    type_part = first_part.split(":")[1]
                    new_line = f"{indent}{key_candidate}: {type_part} = {val}  {comment}\n"
                else:
                    new_line = f"{indent}{key_candidate} = {val}  {comment}\n"
                
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
