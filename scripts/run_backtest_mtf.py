"""
MTF (Multi-Timeframe) Backtest Script
Uses: M1/M5/M15 intraday data with interpolated Greeks
"""
import sys
import os
import datetime as dt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import StrategyConfig, RunConfig
from core.backtest_engine import run_backtest_and_report

def main():
    print("=" * 70)
    print("MTF BACKTEST - Multi-Timeframe Analysis")
    print("=" * 70)
    
    # Strategy with MTF enabled
    s_cfg = StrategyConfig(
        underlying="SPY",
        use_mtf_filter=True,           # ENABLE MTF consensus
        mtf_consensus_min=0.20,
        mtf_consensus_max=0.80,
        use_liquidity_gate=True,
        max_positions=3,
        profit_take_pct=0.50,
        loss_close_multiple=1.00,
    )
    
    # MTF requires M1 data (M5/M15 derived internally)
    # Using synthetic data with full strike chain
    options_data_path = "data/synthetic_options/spy_options_marks.csv"
    
    if not os.path.exists(options_data_path):
        print(f"ERROR: MTF data file not found: {options_data_path}")
        print("Generate using: py -3.12 data_factory/SyntheticOptionsEngine.py")
        return
    
    r_cfg = RunConfig(
        backtest_start=dt.date(2025, 1, 1),
        backtest_end=dt.date(2025, 12, 31),
        options_data_path=options_data_path,
        prefer_intraday=True,
        use_mtf=True,                   # ENABLE MTF processing
        mtf_timeframes=['1', '5', '15'],
        starting_cash=100_000.0,
        backtest_cash=100_000.0,
        backtest_samples=0,             # Load ALL data
        dynamic_sizing=True,
        backtest_plot=True
    )
    
    print(f"Data: {options_data_path}")
    print(f"MTF Timeframes: {r_cfg.mtf_timeframes}")
    print(f"Period: {r_cfg.backtest_start} to {r_cfg.backtest_end}")
    
    run_backtest_and_report(s_cfg, r_cfg)
    print("\nMTF Backtest complete.")

if __name__ == "__main__":
    main()
