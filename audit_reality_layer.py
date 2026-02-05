import sys
import os
import datetime as dt
import pandas as pd
import torch

# Add workdir to path
sys.path.insert(0, os.getcwd())

from core.config import StrategyConfig, RunConfig
from core.backtest_engine import run_backtest_headless
from core.dto import OptionQuote, IronCondorLegs

def audit_day():
    print("=" * 80)
    print("SURGICAL REALITY AUDIT: 2025-01-02")
    print("=" * 80)
    
    s_cfg = StrategyConfig(
        underlying="SPY",
        use_mtf_filter=False,
        use_fuzzy_sizing=False,
        profit_take_pct=0.5,
        loss_close_multiple=2.0,
        iv_rank_min=0.0,
        vix_threshold=100.0,
        dte_min=0
    )
    
    r_cfg = RunConfig(
        backtest_start=dt.date(2025, 1, 2),
        backtest_end=dt.date(2025, 1, 10),
        options_data_path="data/alpaca_options/spy_options_intraday_large_with_greeks_m5.csv",
        prefer_intraday=True,
        starting_cash=100_000.0,
        backtest_cash=100_000.0,
        backtest_samples=0
    )
    
    # Run a headless mini-backtest
    print("[1/3] Running headless audit...")
    strat = run_backtest_headless(s_cfg, r_cfg)
    
    if strat is None or not strat.trade_log:
        print("FAIL: No trades executed during audit period. Try expanding date range or loosening filters.")
        return

    first_trade = strat.trade_log[0]
    print(f"\n[2/3] Tracing First Trade: {first_trade['start']} -> {first_trade['end']}")
    print(f"      Result: {first_trade['result']} | Amount: ${first_trade['amount']:.2f}")
    
    # Check for Greeks drift in strategy state
    print("\n[3/3] Auditing Greek Consistency...")
    # We'll need to look at the internal state during 'next' - difficult in post-mortem.
    # But we can check if the RiskManager ever updated.
    
    rm = strat.risk_manager
    print(f"      Final Portfolio Greeks: Delta={rm.current_greeks.delta:.4f}, Gamma={rm.current_greeks.gamma:.4f}")
    
    # If the trade is closed, Greeks should be 0. Let's see if we can find any snapshot.
    # Actually, let's just inspect the code again.
    # Verification of Invariant 4: Greeks used for Decision vs Risk.
    
    if strat.active_position is None:
        print("      Note: Market Greeks at end of trade were 0 (position closed).")
    
    print("\nAudit Script Finished.")

if __name__ == "__main__":
    audit_day()
