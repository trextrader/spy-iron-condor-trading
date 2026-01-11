# core/config.py - Integrated Configuration
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
    # Hard cutoff for option snapshot staleness (seconds). If exceeded under cutoff
    # policies, the option chain is treated as empty.
    max_option_lag_sec: int = 600
    
    # IV confidence decay half-life (seconds). Confidence weight:
    #   iv_conf = 0.5 ** (lag_sec / iv_decay_half_life_sec)
    iv_decay_half_life_sec: int = 300
    
    # Default policy if strategy does not override:
    #   "hard_cutoff" | "decay_only" | "decay_then_cutoff"
    lag_policy_default: str = "decay_then_cutoff"
    
    # Per-symbol lag cutoffs (overrides max_option_lag_sec when present).
    # Example: {"SPY": 600, "QQQ": 600, "SPX": 900}
    max_option_lag_sec_by_symbol: dict = None
    
    # Fail-fast: abort run if stale fraction exceeds threshold after min bars.
    fail_fast_stale_rate: float = 0.20
    fail_fast_min_bars: int = 50
    
    # Lag-weighted VRP behavior (edge reduction under lag)
    vrp_lag_weighting: bool = True
    vrp_lag_weight_mode: str = "multiply"  # {"multiply","subtract"}
    vrp_lag_penalty_scale: float = 1.045
    target_short_delta_low:  float = 0.12  
    target_short_delta_high: float = 0.30
    
    # === Wing Width & Pricing ===
    wing_width_min:  float = 10.00  # ~7-8% of SPY at $650-700

    wing_width_max: float = 80.0  # ~12% of SPY at $650-700
    min_credit_to_width:  float = 0.10  # 15% credit minimum

    
    # === Risk Management ===
    max_account_risk_per_trade: float = 0.02
    max_positions: int = 3
    max_portfolio_alloc: float = 0.15
    
    # === Entry Filters ===
    iv_rank_min:  float = 40.0  
    vix_threshold:  float = 25.00  # Max VIX for new entries

    
    # === Exit Rules ===
    profit_take_pct:    float = 0.50  
    loss_close_multiple:   float = 1.00  
    max_hold_days:  int = 21  
    
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
    vix_threshold_low: float = 15.0
    vix_threshold_high: float = 30.0
    wing_increment_med: float = 2.5   # If VIX > 20
    wing_increment_high: float = 5.0  # If VIX > 30
    
    # === MTF Enhancements ===
    use_mtf_filter: bool = True
    use_fuzzy_sizing: bool = True    # Enable fuzzy logic position sizing
    mtf_consensus_min: float = 0.20  # Min consensus (-1 to 1) to allow entry
    mtf_consensus_max: float = 0.80  # Max consensus (avoid strong trends)
    use_liquidity_gate: bool = True
    min_volume_threshold: int = 100
    max_volatility_pct: float = 0.03  # Max 2% range in recent bars

    # === ADVANCED TECHNICAL INDICATORS ===

    # RSI (Relative Strength Index) Settings
    use_rsi_filter: bool = True
    rsi_period: int = 14
    
    # === Fuzzy / Gaussian Logic ===
    min_gaussian_confidence: float = 0.20
    rsi_neutral_min:  float = 30.0  # Lower bound of neutral zone

    rsi_neutral_max:  float = 70.0  # Upper bound of neutral zone


    # ADX (Average Directional Index) Settings
    use_adx_filter: bool = True
    adx_period: int = 14
    adx_threshold_low: float = 20.0  # Below this = weak trend (good)
    adx_threshold_high: float = 55.0  # Above this = strong trend (avoid)

    # Stochastic Oscillator Settings
    use_stoch_filter: bool = True
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth_k: int = 3
    stoch_neutral_min: float = 20.0
    stoch_neutral_max: float = 80.0

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

    # Parabolic SAR Settings (10th Indicator)
    use_psar_filter: bool = True
    psar_acceleration: float = 0.02  # AF starting value
    psar_max_acceleration: float = 0.20  # AF max value

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

    # === 10-FACTOR FUZZY WEIGHTS (Sum = 1.00) ===
    # Ft = Σ(w_j × μ_j) for j=1 to 10
    fuzzy_weight_mtf: float = 0.10       # Multi-timeframe consensus
    fuzzy_weight_iv: float = 0.10        # IV Rank
    fuzzy_weight_regime: float = 0.08    # VIX regime
    fuzzy_weight_rsi: float = 0.05       # RSI neutrality
    fuzzy_weight_adx: float = 0.05       # Trend strength
    fuzzy_weight_bbands: float = 0.15    # Bollinger position/squeeze
    fuzzy_weight_stoch: float = 0.15     # Stochastic momentum
    fuzzy_weight_volume: float = 0.12    # Volume confirmation
    fuzzy_weight_sma: float = 0.10       # SMA distance
    fuzzy_weight_psar: float = 0.10      # Parabolic SAR (10th)
    # Total: 0.18+0.14+0.11+0.10+0.10+0.09+0.08+0.07+0.06+0.07 = 1.00

    # === Mamba Neural Forecasting ===
    use_mamba_model: bool = True               # Enable Mamba 2 neural engine
    mamba_d_model: int = 64                    # Model dimension
    mamba_layers: int = 16                     # Model depth
    fuzzy_weight_neural: float = 0.10          # Weight of neural signal in sizing

    # === Position Sizing Guardrails ===
    min_total_qty_for_iron_condor: int = 2     # Minimum contracts for Iron Condor (must be >= 2 for two wings)

    # === LIVE Liquidity Gate (ignored in backtests) ===
    enable_liquidity_gate: bool = True

    # === Spread limits expressed as % of mid price ===
    max_short_spread_pct: float = 0.25   # 25%
    max_long_spread_pct: float  = 0.40   # wings are wider


@dataclass
class RunConfig:
    alpaca_key: str = "PKWCQL536DJKE7EJCP5OEETWE2"
    alpaca_secret: str = "8PK6xfXx13Hqna2ryjHyQCMAf6D1zj6kNGE96CjnzmKM"
    polygon_key: str = "DaTpJeDxfFdJiRN94x65wteJB3V6vPDp"
    
    # === Backtest Settings ===
    backtest_start: dt.date = dt.date(2024, 1, 1)
    backtest_end: dt.date = dt.date(2024, 12, 31)
    backtest_cash: float = 25000.0
    backtest_plot: bool = True
    backtest_samples: int = 500
    options_data_path: str = None  # Path to options data CSV (with Greeks)
    prefer_intraday: bool = True   # Prefer high-density intraday data if available
    
    # === Position Sizing ===
    quantity: int = 1
    dynamic_sizing: bool = False
    position_size_pct: float = 0.05  # 2% of equity per trade
    
    # === MTF Settings ===
    use_mtf: bool = True
    mtf_timeframes: list = None  # ['5', '60', 'D'] - set in __post_init__
    
    # === Optimizer ===
    use_optimizer: bool = False
    
    # === Synthetic Options Data Settings ===
    use_synthetic_options: bool = False  # Use Black-Scholes generated data
    iv_annual_volatility: float = 0.20  # 20% annualized volatility assumption
    risk_free_rate: float = 0.05  # 5% risk-free rate
    
    # === Execution Mode ===
    paper_trading: bool = True
    allow_live_execution: bool = False  # hard safety gate

    # === Simulation Settings ===
    starting_cash: float = 25000.0
    slippage_pct: float = 0.01
    commission_per_contract: float = 0.65
    
    # === Stage 2: Strategy Depth ===
    # Skew & Probability
    use_skew_penalty: bool = True     # Penalize strikes with poor skew benefit
    skew_penalty_weight: float = 0.5  # Weight of IV difference in selection
    max_breach_prob: float = 0.35     # Max probability of strike breach
    
    # Regime Logic
    use_regime_filtering: bool = True
    regime_adx_trend: float = 25.0    # ADX > 25 = Trending
    regime_sma_trend_dist: float = 0.02 # 2% from SMA = Trending
    regime_vix_widen: float = 30.0    # VIX > 30 = High Volatility
    
    # Conditional Wing Widths
    wing_increment_high: float = 5.0  # Widen by $5 in High Vol
    wing_increment_med: float = 5.0   # Widen by $5 in Trending
    
    # === Stage 3: Risk Controls ===
    # Hard Limits
    max_portfolio_delta: float = 200.0  # Max net delta (SPY equivalent)
    max_portfolio_gamma: float = 50.0   # Max net gamma
    max_portfolio_vega: float = 2000.0   # Max net vega
    
    # Per-Trade Limits
    max_delta_per_trade: float = 30.0
    max_vega_per_trade: float = 500.0
    
    # Drawdown Control
    max_daily_drawdown_pct: float = 0.02 # Halt trading if daily loss > 2%

    # === Analytics ===
    enable_trade_logging: bool = True
    trade_log_path: str = "data/trades/trade_log.csv"
    
    def __post_init__(self):
        if self.mtf_timeframes is None:
            self.mtf_timeframes = ['1', '5', '15']