"""
Non-MTF Backtest Script (Standard/Synthetic)
Uses: Large synthetic options marks data (2.2GB)
"""
import sys
import os
import datetime as dt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import StrategyConfig, RunConfig
from core.backtest_engine import run_backtest_and_report

def main():
    print("=" * 70)
    print("STANDARD BACKTEST - No MTF (Synthetic Data)")
    print("=" * 70)
    
    # Strategy WITHOUT MTF filtering
    s_cfg = StrategyConfig(
        underlying="SPY",
        use_mtf_filter=False,          # DISABLE MTF
        use_liquidity_gate=False,      # Synthetic data has no liquidity info
        max_positions=3,
        profit_take_pct=0.50,
        loss_close_multiple=1.00,
    )
    
    # Use synthetic marks data (2.2GB)
    options_data_path = "data/synthetic_options/spy_options_marks.csv"
    
    if not os.path.exists(options_data_path):
        print(f"ERROR: Synthetic data file not found: {options_data_path}")
        print("Generate using: py data_factory/SyntheticOptionsEngine.py")
        return
    
    r_cfg = RunConfig(
        backtest_start=dt.date(2024, 1, 1),
        backtest_end=dt.date(2024, 12, 31),
        options_data_path=options_data_path,
        prefer_intraday=False,          # Use EOD-style processing
        use_mtf=False,                  # DISABLE MTF
        use_synthetic_options=True,     # Flag for synthetic data handling
        starting_cash=100_000.0,
        backtest_cash=100_000.0,
        dynamic_sizing=True,
        backtest_plot=True
    )
    
    print(f"Data: {options_data_path}")
    print(f"MTF: Disabled")
    print(f"Period: {r_cfg.backtest_start} to {r_cfg.backtest_end}")
    
    run_backtest_and_report(s_cfg, r_cfg)
    print("\nStandard Backtest complete.")

if __name__ == "__main__":
    main()
