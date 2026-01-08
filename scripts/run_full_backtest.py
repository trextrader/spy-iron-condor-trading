
import sys
import os
import datetime as dt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import StrategyConfig, RunConfig
from core.backtest_engine import run_backtest_and_report

def main():
    print("=" * 70)
    print("RUNNING FULL PRODUCTION BACKTEST (LOCAL)")
    print("=" * 70)
    
    # Strategy Configuration
    s_cfg = StrategyConfig(
        max_positions=3,
        profit_take_pct=0.50,         # 50% profit target
        loss_close_multiple=1.00,     # Stop loss at 100% of debit (loss_close_multiple=1.0)
    )
    
    # Path to LARGE dataset
    # We use the M1 file with interpolated Greeks
    options_data_path = "data/alpaca_options/spy_options_intraday_large_with_greeks_m1.csv"
    
    # Check if file exists
    if not os.path.exists(options_data_path):
        print(f"Warning: Large data file not found at {options_data_path}")
        print("Falling back to standard file or ensuring you ran the pipeline.")
    
    # Run Configuration
    r_cfg = RunConfig(
        backtest_start=dt.date(2025, 6, 30), # Start of data
        backtest_end=dt.date(2026, 1, 6),    # End of data
        options_data_path=options_data_path,
        prefer_intraday=True, # Force intraday engine
        headless=True,
        # Simulation Settings
        starting_cash=100_000.0,
        backtest_cash=100_000.0,
        dynamic_sizing=True
    )
    
    print(f"Data Source: {options_data_path}")
    print(f"Period: {r_cfg.backtest_start} to {r_cfg.backtest_end}")
    
    results = run_backtest_and_report(s_cfg, r_cfg)
    
    print("\nBacktest execution complete.")

if __name__ == "__main__":
    main()
