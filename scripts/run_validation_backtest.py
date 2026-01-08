import sys
import os
import datetime as dt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import StrategyConfig, RunConfig
from core.backtest_engine import run_backtest_and_report

def main():
    print("="*60)
    print("VALIDATION BACKTEST: Intraday Data Integration")
    print("="*60)
    
    # Configure for validation
    s_cfg = StrategyConfig(
        underlying="SPY",
        # Use simple filter to ensure some trades happen
        use_mtf_filter=False,  
        use_liquidity_gate=False
    )
    
    r_cfg = RunConfig(
        backtest_start=dt.date(2025, 7, 1),
        backtest_end=dt.date(2025, 8, 30),
        backtest_cash=100_000.0,
        backtest_plot=False, # Disable plot for speed
        prefer_intraday=True # CRITICAL FLAG
    )
    
    # Run
    print(f"Targeting range: {r_cfg.backtest_start} to {r_cfg.backtest_end}")
    run_backtest_and_report(s_cfg, r_cfg)

if __name__ == "__main__":
    main()
