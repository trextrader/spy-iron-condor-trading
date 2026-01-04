# core/config.py - Integrated Configuration
from dataclasses import dataclass
import datetime as dt

@dataclass
class StrategyConfig:
    # === Core Strategy ===
    underlying: str = "SPY"
    dte_min: int = 30
    dte_max: int = 45
    target_short_delta_low: float = 0.15
    target_short_delta_high: float = 0.20
    
    # === Wing Width & Pricing ===
    wing_width_min: float = 5.0
    wing_width_max: float = 10.0
    min_credit_to_width: float = 0.25
    
    # === Risk Management ===
    max_account_risk_per_trade: float = 0.02
    max_positions: int = 3
    max_portfolio_alloc: float = 0.15
    
    # === Entry Filters ===
    iv_rank_min: float = 20.0  

    
    # === Exit Rules ===
    profit_take_pct:   float = 0.8999999999999999  
    loss_close_multiple:  float = 1.2  
    max_hold_days: int = 14
    
    # === Adjustment Rules ===
    delta_roll_threshold: float = 0.30
    allow_one_adjustment_per_side: bool = True
    
    # === Dynamic Delta Hedging ===
    dynamic_delta_hedge_threshold: float = 0.10
    dynamic_delta_hedge_unit: int = 10
    
    # === Regime-Based Wing Adjustment ===
    regime_iv_rank_widen: float = 40.0
    regime_vix_widen: float = 22.0
    width_widen_increment: float = 5.0
    
    # === Dynamic Wing Logic ===
    base_wing_width: float = 5.0
    vix_threshold: float = 25.0  
    wing_increment_med: float = 2.5   # If VIX > 20
    wing_increment_high: float = 5.0  # If VIX > 30
    
    # === MTF Enhancements ===
    use_mtf_filter: bool = True
    mtf_consensus_min: float = 0.40  # Min consensus (-1 to 1) to allow entry
    mtf_consensus_max: float = 0.60  # Max consensus (avoid strong trends)
    use_liquidity_gate: bool = True
    min_volume_threshold: int = 100
    max_volatility_pct: float = 0.02  # Max 2% range in recent bars
    
    # === Advanced Indicator Parameters ===
    # RSI
    rsi_neutral_min: float = 40.0
    rsi_neutral_max: float = 60.0
    
    # ADX (trend strength)
    adx_threshold_low: float = 25.0   # Below = weak trend (good for IC)
    adx_threshold_high: float = 40.0  # Above = strong trend (avoid IC)
    
    # Bollinger Bands
    bb_squeeze_threshold: float = 0.02  # 2% width = squeeze
    
    # Stochastic
    stoch_neutral_min: float = 30.0
    stoch_neutral_max: float = 70.0
    
    # SMA Distance
    sma_max_distance: float = 0.02  # 2% max deviation
    
    # Volume
    volume_min_ratio: float = 0.8  # At least 80% of average volume
    
    # === Fuzzy Weights ===
    # Default weights sum to 1.0
    fuzzy_weight_mtf: float = 0.25
    fuzzy_weight_iv: float = 0.20
    fuzzy_weight_regime: float = 0.15
    fuzzy_weight_rsi: float = 0.10
    fuzzy_weight_adx: float = 0.10
    fuzzy_weight_bbands: float = 0.10
    fuzzy_weight_volume: float = 0.05
    fuzzy_weight_sma: float = 0.05
    
    # === ATR Stop Settings ===
    use_atr_stops: bool = True
    atr_stop_base_multiplier: float = 1.5
    
    # === Trailing Stop ===
    use_trailing_stop: bool = False
    trailing_stop_activation_pct: float = 0.50  # Activate at 50% profit
    trailing_stop_distance_pct: float = 0.25    # Trail by 25% of credit
    
    # === Mamba Neural Forecasting ===
    use_mamba_model: bool = True               # Enable Mamba 2 neural engine
    mamba_d_model: int = 64                    # Model dimension
    fuzzy_weight_neural: float = 0.20          # Weight of neural signal in sizing


@dataclass
class RunConfig:
    # === API Keys ===
    alpaca_key: str = "YOUR_ALPACA_KEY_HERE"
    alpaca_secret: str = "YOUR_ALPACA_SECRET_HERE"
    polygon_key: str = "YOUR_POLYGON_KEY_HERE"
    
    # === Backtest Settings ===
    backtest_start: dt.date = dt.date(2024, 1, 1)
    backtest_end: dt.date = dt.date(2024, 12, 31)
    backtest_cash: float = 25000.0
    backtest_plot: bool = True
    backtest_samples: int = 500
    
    # === Position Sizing ===
    quantity: int = 1
    dynamic_sizing: bool = False
    position_size_pct: float = 0.05  # Default 5% risk per trade (Prevents 0 trades on small accts)
    
    # === MTF Settings ===
    use_mtf: bool = True
    mtf_timeframes: list = None  # ['5', '15', '60'] - set in __post_init__
    
    # === Optimizer ===
    use_optimizer: bool = False
    
    def __post_init__(self):
        if self.mtf_timeframes is None:
            self.mtf_timeframes = ['1', '5', '15']