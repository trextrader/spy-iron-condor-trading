# core/config.template.py - TEMPLATE Configuration (Git-Safe)
# Copy to config.py and fill in your API keys
from dataclasses import dataclass
import datetime as dt

@dataclass
class StrategyConfig:
    # === Core Strategy ===
    underlying: str = "SPY"
    dte_min: int = 30
    dte_max: int = 45
    bt_end: str = "2026-01-31"
    bt_data_dir: str = "data"
    sample_size: int = 10
    enable_reporting: bool = True
    
    # -------------------------------------------------------------------
    # Options feed alignment + lag robustness
    # -------------------------------------------------------------------
    max_option_lag_sec: int = 600
    iv_decay_half_life_sec: int = 300
    lag_policy_default: str = "decay_then_cutoff"
    max_option_lag_sec_by_symbol: dict = None
    fail_fast_stale_rate: float = 0.20
    fail_fast_min_bars: int = 50
    vrp_lag_weighting: bool = True
    vrp_lag_weight_mode: str = "multiply"
    vrp_lag_penalty_scale: float = 1.045
    target_short_delta_low:  float = 0.08  
    target_short_delta_high: float = 0.25
    
    # === Wing Width & Pricing ===
    wing_width_min:  float = 5.00
    wing_width_max: float = 80.0
    min_credit_to_width:  float = 0.10
    
    # === Risk Management ===
    max_account_risk_per_trade: float = 0.02
    max_positions: int = 3
    max_portfolio_alloc: float = 0.15
    
    # === Entry Filters ===
    iv_rank_min:  float = 0.00  
    vix_threshold:  float = 25.00

    # === Exit Rules ===
    profit_take_pct:    float = 0.40  
    loss_close_multiple:   float = 1.00  
    max_hold_days:  int = 10  
    
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
    vix_threshold_low: float = 20.0
    vix_threshold_high: float = 30.0
    wing_increment_med: float = 2.5
    wing_increment_high: float = 5.0
    
    # === MTF Enhancements ===
    use_mtf_filter: bool = True
    use_fuzzy_sizing: bool = True
    mtf_consensus_min: float = 0.20
    mtf_consensus_max: float = 0.80
    use_liquidity_gate: bool = True
    min_volume_threshold: int = 100
    max_volatility_pct: float = 0.02

    # === ADVANCED TECHNICAL INDICATORS ===

    # RSI (Relative Strength Index) Settings
    use_rsi_filter: bool = True
    rsi_period: int = 14
    
    # === Fuzzy / Gaussian Logic ===
    min_gaussian_confidence: float = 0.20
    rsi_neutral_min:  float = 30
    rsi_neutral_max:  float = 60

    # ADX (Average Directional Index) Settings
    use_adx_filter: bool = True
    adx_period: int = 14
    adx_threshold_low: float = 25.0
    adx_threshold_high: float = 55.0

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
    bbands_squeeze_threshold: float = 0.02

    # Moving Average Settings
    use_sma_filter: bool = True
    sma_short_period: int = 20
    sma_long_period: int = 50
    sma_max_distance: float = 0.02

    # Volume Confirmation Settings
    use_volume_filter: bool = True
    volume_ma_period: int = 20
    volume_min_ratio: float = 0.8

    # Parabolic SAR Settings (10th Indicator)
    use_psar_filter: bool = True
    psar_acceleration: float = 0.02
    psar_max_acceleration: float = 0.20

    # === EXIT LOGIC ENHANCEMENTS ===

    # ATR-Based Dynamic Stops
    use_atr_stops: bool = True
    atr_period: int = 14
    atr_stop_multiplier_base: float = 1.5
    atr_stop_multiplier_min: float = 1.0
    atr_stop_multiplier_max: float = 2.5

    # Trailing Stop
    use_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.5
    trailing_stop_trail_pct: float = 0.7

    # Bollinger Band Breakout Exit
    use_bbands_exit: bool = True
    bbands_exit_touch_threshold: float = 0.95

    # === 10-FACTOR FUZZY WEIGHTS (Sum = 1.00) ===
    # Ft = Σ(w_j × μ_j) for j=1 to 10
    fuzzy_weight_mtf: float = 0.18       # Multi-timeframe consensus
    fuzzy_weight_iv: float = 0.14        # IV Rank
    fuzzy_weight_regime: float = 0.11    # VIX regime
    fuzzy_weight_rsi: float = 0.10       # RSI neutrality
    fuzzy_weight_adx: float = 0.10       # Trend strength
    fuzzy_weight_bbands: float = 0.09    # Bollinger position/squeeze
    fuzzy_weight_stoch: float = 0.08     # Stochastic momentum
    fuzzy_weight_volume: float = 0.07    # Volume confirmation
    fuzzy_weight_sma: float = 0.06       # SMA distance
    fuzzy_weight_psar: float = 0.07      # Parabolic SAR (10th)
    # Total: 0.18+0.14+0.11+0.10+0.10+0.09+0.08+0.07+0.06+0.07 = 1.00
    
    # === Mamba Neural Forecasting ===
    use_mamba_model: bool = True
    mamba_d_model: int = 64
    fuzzy_weight_neural: float = 0.20

    # === Position Sizing Guardrails ===
    min_total_qty_for_iron_condor: int = 2

    # === LIVE Liquidity Gate (ignored in backtests) ===
    enable_liquidity_gate: bool = True

    # === Spread limits expressed as % of mid price ===
    max_short_spread_pct: float = 0.25
    max_long_spread_pct: float  = 0.40
    
    # === Stage 3: Risk Controls ===
    max_portfolio_delta: float = 200.0   # Max net delta (SPY share equivalent)
    max_portfolio_gamma: float = 50.0    # Max net gamma
    max_portfolio_vega: float = 1000.0   # Max net vega
    max_delta_per_trade: float = 30.0    # Max net delta per new position
    max_vega_per_trade: float = 300.0    # Max vega per new position
    max_daily_drawdown_pct: float = 0.02 # Halt trading if daily loss > 2%


@dataclass
class RunConfig:
    # === API Keys (REPLACE THESE!) ===
    alpaca_key: str = "YOUR_ALPACA_KEY_HERE"
    alpaca_secret: str = "YOUR_ALPACA_SECRET_HERE"
    polygon_key: str = "YOUR_POLYGON_KEY_HERE"
    
    # === Backtest Settings ===
    backtest_start: dt.date = dt.date(2024, 1, 1)
    backtest_end: dt.date = dt.date(2024, 12, 31)
    backtest_cash: float = 25000.0
    backtest_plot: bool = True
    backtest_samples: int = 500
    options_data_path: str = None
    prefer_intraday: bool = True
    
    # === Position Sizing ===
    quantity: int = 1
    dynamic_sizing: bool = False
    position_size_pct: float = 0.05
    
    # === MTF Settings ===
    use_mtf: bool = True
    mtf_timeframes: list = None
    
    # === Optimizer ===
    use_optimizer: bool = False
    
    # === Synthetic Options Data Settings ===
    use_synthetic_options: bool = False
    iv_annual_volatility: float = 0.20
    risk_free_rate: float = 0.05
    
    # === Execution Mode ===
    paper_trading: bool = True
    allow_live_execution: bool = False

    # === Simulation Settings ===
    starting_cash: float = 25000.0
    slippage_pct: float = 0.01
    commission_per_contract: float = 0.65
    
    # === Stage 2: Strategy Depth ===
    use_skew_penalty: bool = True
    skew_penalty_weight: float = 0.5
    max_breach_prob: float = 0.35
    
    use_regime_filtering: bool = True
    regime_adx_trend: float = 25.0
    regime_sma_trend_dist: float = 0.02
    regime_vix_widen: float = 30.0
    
    wing_increment_high: float = 5.0
    wing_increment_med: float = 5.0
    
    # === Stage 3: Risk Controls ===
    max_portfolio_delta: float = 200.0
    max_portfolio_gamma: float = 50.0
    max_portfolio_vega: float = 2000.0
    
    max_delta_per_trade: float = 30.0
    max_vega_per_trade: float = 500.0
    
    max_daily_drawdown_pct: float = 0.02

    # === Analytics ===
    enable_trade_logging: bool = True
    trade_log_path: str = "data/trades/trade_log.csv"
    
    def __post_init__(self):
        if self.mtf_timeframes is None:
            self.mtf_timeframes = ['1', '5', '15']