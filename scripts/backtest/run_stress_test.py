
# scripts/backtest/run_stress_test.py
import sys
import os
import datetime as dt
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.backtest_engine import run_backtest_headless
from core.config import StrategyConfig, RunConfig
import pandas as pd

def run_stress_test():
    print("="*60)
    print(" QUANTOR-MTFUZZ RISK STRESS TEST RUNNER")
    print("="*60)
    
    # Base Configuration
    s_cfg = StrategyConfig()
    r_cfg = RunConfig(
        backtest_start=dt.date(2022, 1, 1), # Uses 2022 (Bear Market) as base stress
        backtest_end=dt.date(2022, 6, 30),
        backtest_cash=100000.0,
        use_synthetic_options=True, # Use synth for consistent shocks if meaningful, effectively using standard logic here.
        # Ideally we would inject shocks, but current engine reads data directly.
        # So we test against KNOWN stressful periods.
    )
    
    # Define Stress Scenarios (Historical Periods)
    scenarios = [
        {
            "name": "2020 COVID Crash",
            "start": dt.date(2020, 2, 20),
            "end": dt.date(2020, 4, 1),
            "desc": "High Volatility + Directional Crack"
        },
        {
            "name": "2022 Q1 Bear Drift",
            "start": dt.date(2022, 1, 1),
            "end": dt.date(2022, 3, 31),
            "desc": "Steady Grind Down + IV Expansion"
        },
        # {
        #     "name": "2018 Volmageddon",
        #     "start": dt.date(2018, 1, 25),
        #     "end": dt.date(2018, 2, 15),
        #     "desc": "Short Volatility Blowout"
        # } # Requires data
    ]
    
    # Define Parameter Shocks (Strategy Sensitivity)
    # We run the 2022 scenario with degraded parameters
    param_shocks = [
        {"name": "Baseline", "cfg_mod": {}},
        {"name": "High Leverage (Risk 5%)", "cfg_mod": {"max_account_risk_per_trade": 0.05}},
        {"name": "Tight Wings (Delta Risk)", "cfg_mod": {"wing_width_min": 1.0, "wing_width_max": 2.0}},
        {"name": "No Filters (Raw)", "cfg_mod": {"use_mtf_filter": False, "use_regime_filtering": False}}
    ]

    # Use the 2022 Bear Drift for Parameter Sensitivity
    base_scenario = scenarios[1]
    r_cfg.backtest_start = base_scenario["start"]
    r_cfg.backtest_end = base_scenario["end"]
    r_cfg.options_data_path = "data/alpaca_options/spy_options_intraday_large_with_greeks_m5.csv" # Default path
    
    # Note: If date range in data doesn't overlap, this will just yield 0 trades or error gracefully.
    # We should verify data range coverage first or use the Q1 2025 data we HAVE for a 'Parameter Stress' test
    # since we might not have 2022 M5 data loaded.
    
    # Let's use the Q1 2025 range we know we have data for, 
    # effectively stress testing parameter sensitivity on current data.
    r_cfg.backtest_start = dt.date(2025, 1, 2)
    r_cfg.backtest_end = dt.date(2025, 3, 30)
    scenario_name = "Q1 2025 (Known Data)"

    results = []
    
    print(f"\nRunning Sensitivity Analysis on {scenario_name}...")
    
    for shock in param_shocks:
        print(f"\n--- Testing: {shock['name']} ---")
        
        # Apply Mods
        test_s_cfg = StrategyConfig()
        for k, v in shock["cfg_mod"].items():
            setattr(test_s_cfg, k, v)
            
        try:
            # Run
            res = run_backtest_headless(test_s_cfg, r_cfg)
            
            # Record
            summary = {
                "Scenario": shock["name"],
                "Net Profit": res.get("total_pnl", 0.0),
                "Max DD": res.get("max_drawdown", 0.0),
                "Trades": res.get("total_trades", 0),
                "Win Rate": res.get("win_rate", 0.0)
            }
            results.append(summary)
            print(f"   Result: NP=${summary['Net Profit']:.2f} | DD=${summary['Max DD']:.2f} | Trades={summary['Trades']}")
            
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
            
    # Display Table
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print(" STRESS TEST RESULTS SUMMARY")
    print("="*60)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_stress_test()
