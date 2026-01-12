
import os
import sys
import pandas as pd
import datetime as dt
from strategies.risk_manager import RiskManager, PortfolioGreeks
from core.config import StrategyConfig
from core.dto import IronCondorLegs, OptionQuote
from strategies.options_strategy import PositionState

def create_mock_quote(delta, gamma, vega, theta):
    return OptionQuote(
        strike=100, bid=1.0, ask=1.1, mid=1.05, 
        iv=0.2, delta=delta, gamma=gamma, vega=vega, theta=theta,
        symbol="MOCK", expiration=dt.date(2025,12,31), is_call=True
    )

def test_risk_manager_caps():
    print("Testing RiskManager Greeks Caps...")
    
    # 1. Setup Config with strict limits
    cfg = StrategyConfig()
    cfg.max_delta_per_trade = 10.0
    cfg.max_vega_per_trade = 100.0
    cfg.max_portfolio_delta = 20.0
    cfg.max_portfolio_vega = 200.0
    
    rm = RiskManager(cfg)
    
    # 2. Mock a generic Iron Condor Legs object
    # Flat position for legs, but we set Greeks to sum to specific values
    # Let's say we want a Net Trade Delta of 15.0 (which fails limit of 10.0)
    # IC Delta = (LongC - ShortC + LongP - ShortP)
    # We'll set LongC=16, ShortC=1, others=0 -> Net=15
    legs_fail_trade = IronCondorLegs(
        short_call=create_mock_quote(0.1, 0, 0, 0),
        long_call=create_mock_quote(0.25, 0, 0, 0), # Net +0.15 per unit
        short_put=create_mock_quote(0, 0, 0, 0),
        long_put=create_mock_quote(0, 0, 0, 0),
        net_credit=0.0,
        max_loss=0.0
    )
    
    # Check Fail Trade Delta
    # Qty 1 -> Delta 15.0 (exceeds 10.0 limit)
    current_port = PortfolioGreeks()
    ok, msg = rm.check_new_trade_risk(legs_fail_trade, 1, current_port)
    if not ok and "Trade Delta" in msg:
        print(" [PASS] Trade Delta Limit Enforced")
    else:
        print(f" [FAIL] Trade Delta Limit NOT Enforced: {msg}")

    # 3. Check Portfolio Limit
    # Current portfolio has Delta 15.0
    current_port.delta = 15.0
    
    # New trade adds Delta 5.1 (Total 20.1 > Limit 20.0)
    legs_add = IronCondorLegs(
        short_call=create_mock_quote(0.1, 0, 0, 0),
        long_call=create_mock_quote(0.151, 0, 0, 0), # Net +0.051 * 100 = 5.1
        short_put=create_mock_quote(0, 0, 0, 0),
        long_put=create_mock_quote(0, 0, 0, 0),
        net_credit=0.0,
        max_loss=0.0
    )
    
    ok, msg = rm.check_new_trade_risk(legs_add, 1, current_port)
    if not ok and "Portfolio Delta" in msg:
        print(" [PASS] Portfolio Delta Limit Enforced")
    else:
        print(f" [FAIL] Portfolio Delta Limit NOT Enforced: {msg}")

def test_drawdown_stop():
    print("\nTesting Drawdown Stop...")
    cfg = StrategyConfig()
    cfg.max_daily_drawdown_pct = 0.05 # 5%
    rm = RiskManager(cfg)
    
    today = dt.date(2025, 1, 1)
    
    # Day Start: $100k
    halt, msg = rm.check_drawdown_stop(100000, today)
    
    # Mid Day: Drop to $96k (4% loss) -> OK
    halt, msg = rm.check_drawdown_stop(96000, today)
    if not halt:
        print(" [PASS] Witin drawdown limit")
    else:
        print(f" [FAIL] Premature halt: {msg}")
        
    # Crash: Drop to $94k (6% loss) -> HALT
    halt, msg = rm.check_drawdown_stop(94000, today)
    if halt and "DAILY STOP" in msg:
        print(" [PASS] Daily Stop Triggered")
    else:
        print(f" [FAIL] Failed to trigger stop: {msg}")

if __name__ == "__main__":
    test_risk_manager_caps()
    test_drawdown_stop()
