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
    wing_width_min: float = 50.0  # ~7-8% of SPY at $650-700
    wing_width_max: float = 80.0  # ~12% of SPY at $650-700
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

    # === ADVANCED TECHNICAL INDICATORS ===

    # RSI (Relative Strength Index) Settings
    use_rsi_filter: bool = True
    rsi_period: int = 14
    rsi_neutral_min: float = 40.0  # Lower bound of neutral zone
    rsi_neutral_max: float = 60.0  # Upper bound of neutral zone

    # ADX (Average Directional Index) Settings
    use_adx_filter: bool = True
    adx_period: int = 14
    adx_threshold_low: float = 25.0  # Below this = weak trend (good)
    adx_threshold_high: float = 40.0  # Above this = strong trend (avoid)

    # Stochastic Oscillator Settings
    use_stoch_filter: bool = True
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth_k: int = 3
    stoch_neutral_min: float = 30.0
    stoch_neutral_max: float = 70.0

    # Bollinger Bands Settings
    use_bbands_filter: bool = True
    bbands_period: int = 20
    bbands_std: float = 2.0
    bbands_squeeze_threshold: float = 0.02  # 2% width = "squeeze"

    # Moving Average Settings
    use_sma_filter: bool = True
    sma_short_period: int = 20
    sma_long_period: int = 50
    sma_max_distance: float = 0.02  # Max 2% from SMA

    # Volume Confirmation Settings
    use_volume_filter: bool = True
    volume_ma_period: int = 20
    volume_min_ratio: float = 0.8  # Min 80% of average volume

    # === EXIT LOGIC ENHANCEMENTS ===

    # ATR-Based Dynamic Stops
    use_atr_stops: bool = True
    atr_period: int = 14
    atr_stop_multiplier_base: float = 1.5  # Base stop loss multiplier
    atr_stop_multiplier_min: float = 1.0   # Minimum (tight stops)
    atr_stop_multiplier_max: float = 2.5   # Maximum (wide stops)

    # Trailing Stop
    use_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.5  # Activate at 50% of max profit
    trailing_stop_trail_pct: float = 0.7        # Trail at 70% of max profit

    # Bollinger Band Breakout Exit
    use_bbands_exit: bool = True
    bbands_exit_touch_threshold: float = 0.95  # Exit if position > 95% or < 5%

    # === REBALANCED FUZZY WEIGHTS (9 Indicators) ===
    fuzzy_weight_mtf: float = 0.25      # Multi-timeframe consensus
    fuzzy_weight_iv: float = 0.18       # IV Rank
    fuzzy_weight_regime: float = 0.15   # VIX regime
    fuzzy_weight_rsi: float = 0.10      # RSI neutrality
    fuzzy_weight_adx: float = 0.10      # Trend strength
    fuzzy_weight_stoch: float = 0.07    # Stochastic momentum
    fuzzy_weight_bbands: float = 0.08   # Bollinger position/squeeze
    fuzzy_weight_volume: float = 0.04   # Volume confirmation
    fuzzy_weight_sma: float = 0.03      # SMA distance
    # Total: 1.00
    
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
    mtf_timeframes: list = None  # ['1', '5', '15'] - set in __post_init__
    
    # === Optimizer ===
    use_optimizer: bool = False
    
    def __post_init__(self):
        if self.mtf_timeframes is None:
            self.mtf_timeframes = ['1', '5', '15']