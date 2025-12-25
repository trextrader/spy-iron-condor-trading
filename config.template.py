# config.py
from dataclasses import dataclass
import datetime as dt

@dataclass
class StrategyConfig:
    underlying: str = "SPY"
    dte_min: int = 30
    dte_max: int = 45
    target_short_delta_low: float = 0.15
    target_short_delta_high: float = 0.20
    wing_width_min: float = 5.0
    wing_width_max: float = 10.0
    min_credit_to_width: float = 0.25
    max_account_risk_per_trade: float = 0.02
    max_positions: int = 3
    max_portfolio_alloc: float = 0.15
    iv_rank_min: float = 30.0
    vix_threshold: float = 25.0
    profit_take_pct: float = 0.50
    loss_close_multiple: float = 1.5
    max_hold_days: int = 14
    delta_roll_threshold: float = 0.30
    allow_one_adjustment_per_side: bool = True
    dynamic_delta_hedge_threshold: float = 0.10
    dynamic_delta_hedge_unit: int = 10
    regime_iv_rank_widen: float = 40.0
    regime_vix_widen: float = 22.0
    width_widen_increment: float = 5.0
    
    # Dynamic Wing Logic
    base_wing_width: float = 5.0
    vix_threshold_low: float = 20.0
    vix_threshold_high: float = 30.0
    
    # Increments to add to base_wing_width based on VIX
    wing_increment_med: float = 2.5  # If VIX > 20
    wing_increment_high: float = 5.0 # If VIX > 30

@dataclass
class RunConfig:
    alpaca_key: str = "YOUR_ALPACA_API_KEY_HERE"
    alpaca_secret: str = "YOUR_ALPACA_SECRET_KEY_HERE"
    polygon_key: str = "YOUR_POLYGON_API_KEY_HERE"
    quantity: int = 1
    use_optimizer: bool = False
    backtest_start: dt.date = dt.date(2024, 1, 1)
    backtest_end: dt.date = dt.date(2024, 12, 31)
    backtest_cash: float = 25000.0   # default $25,000
    backtest_plot: bool = True
    backtest_samples: int = 500
    position_size_pct: float = 0.02   # default 2% of equity per trade
    dynamic_sizing: bool = False      # toggle fixed vs. equity-based sizing
