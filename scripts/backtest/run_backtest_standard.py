"""
Non-MTF Backtest Script (Standard)
Uses: Synthetic options data (2.2GB full strike chain)
"""
import sys
import os
import datetime as dt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.config import StrategyConfig, RunConfig
from core.backtest_engine import run_backtest_and_report

def main():
    print("=" * 70)
    print("STANDARD BACKTEST - No MTF (Synthetic Data)")
    print("=" * 70)
    
    # Strategy WITHOUT MTF filtering and WITHOUT Fuzzy Sizing
    s_cfg = StrategyConfig(
        underlying="SPY",
        use_mtf_filter=False,          # DISABLE MTF
        use_fuzzy_sizing=False,        # DISABLE Fuzzy Logic (Static Sizing)
        iv_rank_min=0.0                # DISABLE IVR Gate for testing
    )
    
    # Use M5 Turbo Data (Generated from IVolatility)
    options_data_path = "data/alpaca_options/spy_options_intraday_large_with_greeks_m5.csv"
    
    if not os.path.exists(options_data_path):
        print(f"ERROR: Data file not found: {options_data_path}")
        return
    
    r_cfg = RunConfig(
        backtest_start=dt.date(2025, 1, 2),
        backtest_end=dt.date(2025, 3, 30), # Backtest Q1 2025 (Different valid range)
        options_data_path=options_data_path,
        prefer_intraday=True,
        use_mtf=False,                  # DISABLE MTF
        starting_cash=100_000.0,
        backtest_cash=100_000.0,
        backtest_samples=0,             # Load ALL data
        dynamic_sizing=False,           # Use static sizing if fuzzy is off
        backtest_plot=False             # Speed up large run
    )
    
    print(f"Data: {options_data_path}")
    print(f"MTF: Disabled")
    print(f"Period: {r_cfg.backtest_start} to {r_cfg.backtest_end}")
    
    run_backtest_and_report(s_cfg, r_cfg)
    print("\nStandard Backtest complete.")

if __name__ == "__main__":
    main()
