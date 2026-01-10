"""
Non-MTF Backtest Script (Standard)
Uses: New IVolatility/Alpaca M1 data WITHOUT MTF filtering
"""
import sys
import os
import datetime as dt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import StrategyConfig, RunConfig
from core.backtest_engine import run_backtest_and_report

def main():
    print("=" * 70)
    print("STANDARD BACKTEST - No MTF (New Data)")
    print("=" * 70)
    
    # Strategy WITHOUT MTF filtering
    s_cfg = StrategyConfig(
        underlying="SPY",
        use_mtf_filter=False,          # DISABLE MTF
        use_liquidity_gate=False,
        max_positions=3,
        profit_take_pct=0.50,
        loss_close_multiple=1.00,
    )
    
    # Use new IVolatility/Alpaca data
    options_data_path = "data/alpaca_options/spy_options_intraday_large_with_greeks_m1.csv"
    
    if not os.path.exists(options_data_path):
        print(f"ERROR: Data file not found: {options_data_path}")
        print("Run: py -3.12 scripts/run_production_pipeline.py")
        return
    
    r_cfg = RunConfig(
        backtest_start=dt.date(2025, 6, 30),
        backtest_end=dt.date(2026, 1, 6),
        options_data_path=options_data_path,
        prefer_intraday=True,
        use_mtf=False,                  # DISABLE MTF
        starting_cash=100_000.0,
        backtest_cash=100_000.0,
        backtest_samples=0,             # Load ALL data, not just last 500 bars
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
