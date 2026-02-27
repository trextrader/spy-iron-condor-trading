"""
schema_v43.py — CondorNet v4.3 Schema Constants & Validators
=============================================================
Single source of truth for all feature names, dtypes, bounds, contracts,
and configuration constants used across the v4.3 data pipeline, model,
strategy engine, and training loop.

All feature access MUST use names from this module — never by CSV column index.

Author: CondorNet v4.3 Implementation
Schema Version: v4.3.0
Date: 2026-02-23
"""

import math
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

# =============================================================================
# SCHEMA VERSION
# =============================================================================

SCHEMA_VERSION: str = "v4.3.0"
STRATEGY_JSON_VERSION: str = "1.0"

# =============================================================================
# TEMPORAL ALIGNMENT CONTRACT
# =============================================================================

CANONICAL_TF: str = "m5"
"""M5 close timestamps are the master clock for the entire v4.3 dataset."""

SPOT_PRICE_RULE: str = "M5 close at canonical timestamp T"
"""
Global rule: spot price used for moneyness = log(S/K) always comes from
M5 close at the canonical timestamp T. Chain mid-price implied spot is
never used (avoids circularity with the data being encoded).
"""

CHAIN_FORWARD_FILL_MAX_MINUTES: int = 5
"""
Options chain snapshot forward-fill limit. If the chain has no snapshot
at T, use the last snapshot at T' < T where T - T' <= 5 minutes.
Any T with no chain snapshot within this window is dropped.
"""

CANONICAL_TIMESTAMP_CONTRACT = {
    "master_clock": CANONICAL_TF,
    "m1_required_bars": 5,          # M1 must have >= 5 complete bars ending at T
    "m15_requires_complete_window": True,
    "h1_requires_complete_window": True,
    "chain_forward_fill_max_min": CHAIN_FORWARD_FILL_MAX_MINUTES,
    "on_failure": "drop",           # Drop T if any condition fails — never partial rows
}

# =============================================================================
# FEATURE NAMES — V2.2 SCHEMA (64 input features, by name)
# =============================================================================

# Group 1 — Price / Trend (12 features)
TF_GROUP1_PRICE: List[str] = [
    "open", "high", "low", "close",
    "MA", "FRAMA", "sma",
    "TurtleChn-Up", "TurtleChn-Low",
    "TrendSeekerUp", "TrendSeekerDown",
    "AnchoredVWAP",
]

# Group 2 — Momentum / Oscillators (10 features)
TF_GROUP2_MOMENTUM: List[str] = [
    "rsi_dyn", "stoch_k_dyn", "adx_adaptive",
    "ret_z", "log_return",
    "AccDistWill", "AccDistWillMovAvg",
    "WilderAccSwingIndex",
    "breakout_score",   # Ternary: -1/0/1
    "WeightedAlpha",
]

# Group 3 — Volatility (8 features)
TF_GROUP3_VOLATILITY: List[str] = [
    "ATR_recon", "atr_pct",
    "bb_upper_dyn", "bb_lower_dyn", "bb_mu_dyn", "bb_sigma_dyn",
    "bandwidth", "bw_expansion_rate",
]

# Group 4 — Volume / Market Microstructure (6 features)
TF_GROUP4_VOLUME: List[str] = [
    "volume", "trade_count",
    "OSC-Volume", "vol_energy", "vol_ewma",
    "spread_ratio",
]

# Group 5 — Regime / Probability Signals (10 features)
TF_GROUP5_REGIME: List[str] = [
    "psar_adaptive", "psar_mark", "psar_trend",  # psar_trend: Ternary -1/1
    "psar_reversion_mu",
    "pressure_up", "pressure_down",
    "max_dd_60m",
    "bb_percentile",
    "fuzzy_reversion_11",
    "consolidation_score",
]

# Group 6 — Risk / Composite Scores (5 features)
TF_GROUP6_RISK: List[str] = [
    "gap_risk_score",
    "chaos_membership",
    "kappa_proxy",
    "McClellanOsc",
    "Slope", "SlopeATR",
]

# Group 7 — Pivot / Structural Features (13 features — sparse, never imputed)
TF_GROUP7_PIVOTS: List[str] = [
    "PivotHigh", "PivotLow",
    "PivotResidual", "PivotResidualATR", "PivotResidualZ",
    "PivotCurvatureATR", "PivotCurvatureProxy",
    "PivotSegmentLengthBars", "PivotSegmentLengthMinutes",
    "PivotSegmentResidualStd", "PivotSegmentVolatility",
    "Slope",        # Also in Group 6 — dual-use
    "SlopeATR",     # Also in Group 6 — dual-use
]

# Correct pivot-only list (deduplicated)
TF_PIVOT_FEATURES: List[str] = [
    "PivotHigh", "PivotLow",
    "PivotResidual", "PivotResidualATR", "PivotResidualZ",
    "PivotCurvatureATR", "PivotCurvatureProxy",
    "PivotSegmentLengthBars", "PivotSegmentLengthMinutes",
    "PivotSegmentResidualStd", "PivotSegmentVolatility",
    "Slope", "SlopeATR",
]
"""13 sparse pivot features. NaN = no pivot event at that bar. NEVER imputed."""

# Full 64 input feature list (excludes labels target_spot + position_size_mult)
# Note: friction_ok_5/10/20/40/60 + tod_sin + tod_cos are computed features added by ETL
TF_FEATURE_NAMES: List[str] = (
    TF_GROUP1_PRICE +
    TF_GROUP2_MOMENTUM +
    TF_GROUP3_VOLATILITY +
    TF_GROUP4_VOLUME +
    TF_GROUP5_REGIME +
    TF_GROUP6_RISK
    # Group 7 pivots passed separately as pivot_mask, not as regular input features
)
# Deduplicate while preserving order
_seen: Set[str] = set()
TF_FEATURE_NAMES = [f for f in TF_FEATURE_NAMES if not (f in _seen or _seen.add(f))]

# Friction gate computed features (replace dead friction_ratio placeholder)
FRICTION_FEATURE_NAMES: List[str] = [
    "friction_ok_5", "friction_ok_10", "friction_ok_20", "friction_ok_40", "friction_ok_60",
]
"""
5 binary columns computed by ETL. Replace the constant friction_ratio column.
friction_ok_N = int(bid_ask_spread < (high.rolling(N).max() - low.rolling(N).min()))
"""

# Time-of-day features (computed by ETL)
TOD_FEATURE_NAMES: List[str] = ["tod_sin", "tod_cos"]
"""Circular encoding of minute_of_day / TOD_TRADING_MINUTES."""

# Regime persistence feature
REGIME_PERSISTENCE_FEATURE: str = "regime_persistence"
"""Integer: count of consecutive bars in same (vol_bucket, trend_bucket)."""

# =============================================================================
# IVR REVERSAL ENGINE FEATURES
# Derived from IVR Reversal Engine PineScript logic.
# These 9 features encode the IVR×stretch×pivot composite reversal signals.
# Computed during ETL from raw OHLCV + IV data. Model learns OPTIMAL WEIGHTS
# and COMBINATIONS — it is NOT hard-coding the PineScript logic.
#
# ETL computation formulas (MA_len=20, ATR_len=14, IVR_len=252):
#   ivr                   = (iv_src - iv_252d_low) / (iv_252d_high - iv_252d_low)
#   price_stretch         = (close - SMA(close, 20)) / (ATR(14) + ε)      clipped ±10
#   ivr_zone              = +1 if ivr>0.70, -1 if ivr<0.30, else 0        ternary
#   stretch_zone          = +1 if price_stretch>2.0, -1 if <-2.0, else 0  ternary
#   reversal_score        = 1.0×ivr + 1.0×price_stretch + (pivotStrength × pivotDir)
#   strong_pivot_slope_h  = slope of line connecting last two strong PIVOT HIGHs
#                           = (ph2 - ph1) / max(bar_dist_h, 1); NaN if < 2 strong highs
#   strong_pivot_slope_l  = slope of line connecting last two strong PIVOT LOWs
#   strong_pivot_bar_dist_h = bar count between last two strong PIVOT HIGHs; NaN initially
#   strong_pivot_bar_dist_l = bar count between last two strong PIVOT LOWs
#
# Pivot strength definitions (must match pivot ETL):
#   medium: ta.pivothigh(high, L=30, R=32) or ta.pivotlow(low, L=30, R=32)
#   strong: ta.pivothigh(high, L=69, R=69) or ta.pivotlow(low, L=69, R=69)
#   pivotStrength = 3 if strong, 2 if medium, 1 if weak (not used here), 0 if none
#   pivotDir      = +1 if pivot low, -1 if pivot high, 0 if no pivot
#
# Note: `ivr` is already in TF_FEATURE_NAMES via fuzzy_reversion_11 proxy.
#       These 9 features are the NEW ADDITIONS from the IVR Reversal Engine.
#       Enabling them changes d_tf_in by +9. Update --d-tf-in accordingly.
# =============================================================================

IVR_REVERSAL_FEATURE_NAMES: List[str] = [
    "price_stretch",            # (close - SMA20) / ATR14 — deviation from MA in ATR units
    "ivr_zone",                 # Ternary: +1=high IV (>0.70), -1=low IV (<0.30), 0=normal
    "stretch_zone",             # Ternary: +1=above K*ATR, -1=below -K*ATR, 0=normal
    "reversal_score",           # α×IVR + β×stretch + γ×(pivotStrength×pivotDir)
    "strong_pivot_slope_h",     # Slope between last two strong pivot HIGHS (descending → falling resistance)
    "strong_pivot_slope_l",     # Slope between last two strong pivot LOWS (ascending → rising support)
    "strong_pivot_bar_dist_h",  # Bar count between last two strong pivot HIGHS (pivot frequency)
    "strong_pivot_bar_dist_l",  # Bar count between last two strong pivot LOWS
    "bars_since_strong_pivot",  # Bars since last confirmed strong pivot (high or low)
]
"""
9 IVR Reversal Engine features.

Reversal signal logic the model learns from (NOT hard-coded):
  Bullish reversal = ivr_zone == -1 AND stretch_zone == -1 AND last pivot was LOW
                     AND bars_since_strong_pivot >= confirm_bars
  Bearish reversal = ivr_zone == +1 AND stretch_zone == +1 AND last pivot was HIGH
                     AND bars_since_strong_pivot >= confirm_bars

The model discovers which weights (α, β, γ, confirm_bars) are optimal.
NaN rules: strong_pivot_slope_* and strong_pivot_bar_dist_* are NaN until
two confirmed strong pivots of that type are available. Treat as SPARSE
(same masking as TF_PIVOT_FEATURES — NaN = no data, not zero).
"""

# IVR reversal features that are sparse (NaN = no strong pivot available yet)
IVR_REVERSAL_SPARSE: Set[str] = {
    "strong_pivot_slope_h", "strong_pivot_slope_l",
    "strong_pivot_bar_dist_h", "strong_pivot_bar_dist_l",
    "bars_since_strong_pivot",
}

# IVR reversal ternary features
IVR_REVERSAL_TERNARY: List[str] = ["ivr_zone", "stretch_zone"]

# Bounded ranges for IVR reversal features
IVR_REVERSAL_BOUNDED: Dict[str, Tuple[float, float]] = {
    "price_stretch":  (-10.0, 10.0),
    "ivr_zone":       (-1.0, 1.0),
    "stretch_zone":   (-1.0, 1.0),
    "reversal_score": (-10.0, 10.0),
}

# Configuration constants for IVR Reversal Engine (mirror PineScript defaults)
IVR_REVERSAL_CONFIG: Dict = {
    "ivr_lookback":       252,   # IVR lookback bars (1 year)
    "ivr_high_threshold": 0.70,  # IVR > 0.70 → high vol regime
    "ivr_low_threshold":  0.30,  # IVR < 0.30 → low vol regime
    "ma_length":          20,    # SMA length for price stretch
    "atr_length":         14,    # ATR length for price stretch
    "stretch_k":          2.0,   # Stretch threshold in ATR units
    "alpha_ivr":          1.0,   # Weight for IVR in reversal_score
    "beta_stretch":       1.0,   # Weight for stretch in reversal_score
    "gamma_pivot":        1.0,   # Weight for pivot strength×dir in reversal_score
    "score_threshold":    2.0,   # Minimum |reversal_score| to trigger reversal signal
    "confirm_bars":       5,     # Bars after pivot point required for confirmation
    "strong_left":        69,    # Strong pivot left lookback
    "strong_right":       69,    # Strong pivot right lookback
    "medium_left":        30,    # Medium pivot left lookback
    "medium_right":       32,    # Medium pivot right lookback
}

# All computed features added by ETL (not in raw CSV)
ETL_COMPUTED_FEATURES: List[str] = (
    FRICTION_FEATURE_NAMES + TOD_FEATURE_NAMES + [REGIME_PERSISTENCE_FEATURE]
    + IVR_REVERSAL_FEATURE_NAMES
)

# Labels (excluded from model input features)
TF_LABEL_NAMES: List[str] = [
    "target_spot",        # col 0: 5-bar fwd log return × 100 (set by data_pipeline_v43)
    "position_size_mult", # col 1: position size multiplier (0.5 / 0.75 / 1.0)
    "strategy_label",     # col 2: strategy class 0-9 (set by data_pipeline_v43)
    "pop",                # col 3: BSM Probability of Profit [0, 1]
    "ev",                 # col 4: Expected Value / close
    "max_loss",           # col 5: max loss / close
    "var_95",             # col 6: 95% VaR / close
    "cvar_95",            # col 7: 95% CVaR / close
]
"""Training targets. Excluded from model input.
Columns 0-1 present in raw CSVs. Columns 2-7 added by data_pipeline_v43.py.
_load_tf_csv fills missing columns with 0 — safe for incremental updates."""

# Sparse features — NaN is semantically meaningful, NEVER impute
TF_SPARSE_FEATURES: Set[str] = set(TF_PIVOT_FEATURES) | IVR_REVERSAL_SPARSE
"""NaN in these columns means 'no event occurred'. Do not fill with 0 or mean."""

# Ternary features — valid values are {-1, 0, 1} or {-1, 1}
TF_TERNARY_FEATURES: List[str] = ["breakout_score", "psar_trend"] + IVR_REVERSAL_TERNARY

# =============================================================================
# DTYPE MAP
# =============================================================================

TF_DTYPE_MAP: Dict[str, str] = {
    # Ternary → cast to int before use
    "breakout_score": "int64",
    "psar_trend":     "int64",
    # Low-cardinality legacy columns (kept for compatibility but replaced in v43)
    "friction_ratio":      "float64",   # Legacy placeholder — will be replaced by friction_ok_*
    "gap_risk_score":      "float64",
    "position_size_mult":  "float64",
    # Integer counts
    "volume":      "int64",
    "trade_count": "int64",
    # Everything else → float64
    "__default__": "float64",
}

# =============================================================================
# BOUNDED FEATURES — known valid ranges per the M1 column profile
# =============================================================================

TF_BOUNDED_FEATURES: Dict[str, Tuple[float, float]] = {
    "atr_pct":            (0.0, 1.0),
    "rsi_dyn":            (0.0, 100.0),
    "stoch_k_dyn":        (0.0, 100.0),
    "adx_adaptive":       (0.0, 100.0),
    "bb_percentile":      (0.0, 100.0),
    "psar_reversion_mu":  (0.0, 1.0),
    "max_dd_60m":         (0.0, 1.0),
    "consolidation_score":(0.0, 1.0),
    "chaos_membership":   (0.0, 1.0),
    "fuzzy_reversion_11": (0.0, 1.0),
    "pressure_up":        (0.0, 1.0),
    "pressure_down":      (0.0, 1.0),
    "vol_ewma":           (0.0, 1.0),
    "spread_ratio":       (0.0, 1.0),
    "gap_risk_score":     (0.0, 1.0),
}

# =============================================================================
# CROSS-TF ANOMALY FLAGS (from CROSS_TF_COLUMN_CONTRACT_v43.csv)
# =============================================================================

KNOWN_CROSS_TF_FLAGS: Dict[str, List[str]] = {
    "TYPE_MISMATCH": [
        "friction_ratio",   # float64 on M5/M15/H1 but constant on M1 — replaced by friction_ok_*
        "vol_energy",       # Minor dtype difference across TFs
    ],
    "RANGE_MISMATCH": [
        "WeightedAlpha",            # Different range on H1 vs M1
        "Slope", "SlopeATR",        # Wider range on lower TFs
        "PivotSegmentLengthMinutes",# Expected — units differ by TF
    ],
    "SPARSITY_MISMATCH": [
        # All pivot features are sparser on M5 than M1 (fewer bars = fewer pivot events)
        "PivotHigh", "PivotLow", "PivotResidual", "PivotResidualATR", "PivotResidualZ",
        "PivotCurvatureATR", "PivotCurvatureProxy",
        "PivotSegmentLengthBars", "PivotSegmentLengthMinutes",
        "PivotSegmentResidualStd", "PivotSegmentVolatility",
    ],
    "CONST_MISMATCH": [
        "friction_ratio",   # Constant 0.5 on M1 — dead placeholder, replaced in v43
    ],
}

# =============================================================================
# OPTIONS CHAIN SCHEMA
# =============================================================================

CHAIN_FEATURE_NAMES: List[str] = [
    "moneyness",       # log(spot / strike) — computed, not in raw CSV
    "days_to_exp",     # Calendar days to expiration
    "iv",              # Implied volatility (annualized)
    "delta",           # Option delta ∈ [-1, 1]
    "gamma",           # Option gamma ≥ 0
    "theta",           # Option theta (daily decay)
    "vega",            # Option vega
    "bid",             # Bid price
    "ask",             # Ask price
    "oi_norm",         # Open interest normalized within snapshot (log-scaled)
]
"""10 features per contract in the chain grid [B, N_contracts, 10]."""

CHAIN_LABEL_NAMES: List[str] = [
    "pop",        # Probability of Profit ∈ [0, 1]
    "ev",         # Expected Value ($)
    "max_profit", # Maximum possible profit ($)
    "max_loss",   # Maximum possible loss ($, positive number)
    "var_95",     # 95% Value at Risk ($, positive)
    "cvar_95",    # 95% Conditional VaR ($, positive, ≥ var_95)
]

# Chain grid configuration
CHAIN_GRID_CONFIG: Dict = {
    "n_strikes":         20,    # Target strikes per expiry bucket
    "n_expiries":        3,     # Target expiry buckets
    "max_contracts":     120,   # Hard cap — 60 nearest calls + 60 nearest puts by |moneyness|
    "liquidity_min_oi":  100,   # Mask strikes with open_interest < 100 (mask=True, not dropped)
    "pad_value":         0.0,   # Zero-pad for missing strikes/expiries
    "pad_with_mask":     True,  # Padded positions get mask=True → ignored by transformer attention
    "in_features":       10,    # = len(CHAIN_FEATURE_NAMES)
    "d_model":           64,    # Transformer d_model
    "n_heads":           4,     # Transformer attention heads
    "n_layers":          2,     # Transformer encoder layers
    "d_ff":              256,   # Transformer feedforward dim
    "d_chain":           128,   # Final chain embedding dim
}

# =============================================================================
# FRICTION GATE (SET IN STONE — HARD RULE)
# =============================================================================

FRICTION_CANDIDATE_WINDOWS: List[int] = [5, 10, 20, 40, 60]
"""
Candidate lookback windows (in bars) for the adaptive friction gate.

Per bar, per window N:
    HL_N = high.rolling(N).max() - low.rolling(N).min()
    friction_ok_N = int(bid_ask_spread < HL_N)

The gate opens if ANY window N passes (friction_ok_N = 1).
The model's attention over the 5 friction columns effectively selects which N matters.

CRITICAL: Applied PER LEG independently. All legs in a strategy must pass.
A single illiquid leg blocks the entire trade.

Prevents: entries/exits when bid-ask spread is too wide relative to recent
price range, which causes fills far above mid on longs and far below on shorts.
"""

FRICTION_GATE_LOGIC: str = "any_window"
"""
"any_window": gate opens if at least one friction_ok_N = 1
"all_windows": gate opens only if all windows pass (more conservative)
v4.3 uses "any_window" — the model learns which N is most predictive.
"""

# =============================================================================
# STRATEGY UNIVERSE
# =============================================================================

STRATEGY_TYPES: List[str] = [
    "single_call",
    "single_put",
    "bull_call_spread",
    "bear_put_spread",
    "straddle",
    "strangle",
    "butterfly_call",
    "iron_condor",
    "custom_multi_leg",
    "abstain",          # Class 10 — no trade signal
]
N_STRATEGY_TYPES: int = len(STRATEGY_TYPES)  # 10

STRATEGY_TYPE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(STRATEGY_TYPES)}
IDX_TO_STRATEGY_TYPE: Dict[int, str] = {i: s for i, s in enumerate(STRATEGY_TYPES)}

ABSTAIN_IDX: int = STRATEGY_TYPE_TO_IDX["abstain"]  # 9
ABSTAIN_CONFIDENCE_THRESHOLD: float = 0.60
"""
If softmax(strategy_logits).max() < this threshold, output abstain.
The model is not confident enough in any strategy — stand aside.
"""

ABSTAIN_LABEL_SCORE_CUTOFF: float = 0.40
"""
During label generation: if best candidate composite score < this cutoff,
the training label is set to abstain (is_ideal = abstain).
"""

# DTE-Strategy Affinity Matrix
# Rows: strategy type. Cols: DTE bucket.
# Values ∈ [0, 1]: prior probability that this strategy is appropriate at this DTE.
DTE_AFFINITY: Dict[str, Dict[str, float]] = {
    # Calibrated for 0-DTE SPY strategies (the primary use-case).
    # "0-2" values reflect how natural each strategy is at very short DTE.
    # iron_condor is SPY's most-traded 0-DTE income structure → 0.80 (was 0.20).
    # bull/bear spreads are solid defined-risk directional plays at 0-DTE → 0.65 (was 0.30).
    # butterfly_call is tighter than IC but viable at 0-DTE → 0.55 (was 0.20).
    "single_call":      {"0-2": 0.9, "3-7": 0.7, "8-21": 0.3, "22+": 0.1},
    "single_put":       {"0-2": 0.9, "3-7": 0.7, "8-21": 0.3, "22+": 0.1},
    "bull_call_spread": {"0-2": 0.6, "3-7": 0.7, "8-21": 0.9, "22+": 0.5},
    "bear_put_spread":  {"0-2": 0.6, "3-7": 0.7, "8-21": 0.9, "22+": 0.5},
    "straddle":         {"0-2": 0.8, "3-7": 0.5, "8-21": 0.3, "22+": 0.1},
    "strangle":         {"0-2": 0.7, "3-7": 0.5, "8-21": 0.4, "22+": 0.2},
    "butterfly_call":   {"0-2": 0.5, "3-7": 0.6, "8-21": 0.8, "22+": 0.7},
    "iron_condor":      {"0-2": 0.8, "3-7": 0.8, "8-21": 0.9, "22+": 1.0},
    "custom_multi_leg": {"0-2": 0.5, "3-7": 0.7, "8-21": 0.7, "22+": 0.5},
    "abstain":          {"0-2": 0.0, "3-7": 0.0, "8-21": 0.0, "22+": 0.0},
}

DTE_BUCKETS: List[str] = ["0-2", "3-7", "8-21", "22+"]

def get_dte_bucket(dte: int) -> str:
    """Map integer DTE to the affinity matrix bucket key."""
    if dte <= 2:
        return "0-2"
    elif dte <= 7:
        return "3-7"
    elif dte <= 21:
        return "8-21"
    else:
        return "22+"

def get_dte_affinity(strategy_type: str, dte: int) -> float:
    """Return affinity prior score for a (strategy_type, DTE) pair."""
    if strategy_type not in DTE_AFFINITY:
        return 0.5   # Unknown type — neutral prior
    return DTE_AFFINITY[strategy_type][get_dte_bucket(dte)]

# Regime → preferred strategies mapping (used as soft prior, not hard constraint)
REGIME_STRATEGY_AFFINITY: Dict[str, List[str]] = {
    "high_vol":        ["iron_condor", "strangle"],
    "low_vol":         ["bull_call_spread", "bear_put_spread", "butterfly_call"],
    "trending_bull":   ["single_call", "bull_call_spread"],
    "trending_bear":   ["single_put", "bear_put_spread"],
    "consolidation":   ["bull_call_spread", "bear_put_spread", "butterfly_call"],
    "choppy":          ["straddle", "strangle", "iron_condor"],
    "post_spike_mean": ["iron_condor", "butterfly_call"],
}

# =============================================================================
# STRATEGY JSON SCHEMA v1.0
# =============================================================================

STRATEGY_JSON_REQUIRED_FIELDS: List[str] = [
    "schema_version", "strategy_id", "strategy_type", "timestamp",
    "legs", "friction_gate_ok",
]

LEG_JSON_REQUIRED_FIELDS: List[str] = [
    "strike", "expiry_days", "option_type", "direction", "qty",
    "iv", "bid", "ask", "mid", "open_interest", "friction_ok",
]

LEG_JSON_OPTIONAL_FIELDS: List[str] = [
    "delta", "gamma", "theta", "vega", "volume",
]

STRATEGY_JSON_OPTIONAL_FIELDS: List[str] = [
    "pop", "ev", "max_profit", "max_loss", "var_95", "cvar_95",
    "net_delta", "net_gamma", "net_theta", "net_vega",
    "pin_risk_flag", "dte_affinity_score", "abstain_confidence",
    "liquidity_gate_ok", "skew_signal",
]

# =============================================================================
# OPTIONS PHYSICS CONSTANTS
# =============================================================================

MIN_STRIKE_INCREMENT: float = 0.50
"""
Minimum strike price increment for SPY options ($0.50).
Used in pin risk detection: if |spot - short_strike| < 0.5 * MIN_STRIKE_INCREMENT
and DTE <= 2, flag pin_risk = True.
"""

DEFAULT_RISK_FREE_RATE: float = 0.05
"""Annualized risk-free rate. Constant for v4.3 (per-expiry term structure deferred to v4.4)."""

# =============================================================================
# TIME-OF-DAY ENCODING
# =============================================================================

TOD_TRADING_MINUTES: int = 390
"""
US equity market trading minutes: 9:30am – 4:00pm = 6.5 hours = 390 minutes.
tod_sin = sin(2 * pi * minute_of_day / 390)
tod_cos = cos(2 * pi * minute_of_day / 390)
where minute_of_day = (hour - 9) * 60 + (minute - 30)
"""

MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 30
MARKET_CLOSE_HOUR: int = 16
MARKET_CLOSE_MINUTE: int = 0

def compute_tod_features(hour: int, minute: int) -> Tuple[float, float]:
    """
    Compute time-of-day sin/cos features for a given hour:minute.

    Args:
        hour: Hour in 24h format (9-16)
        minute: Minute (0-59)

    Returns:
        (tod_sin, tod_cos) circular encoding ∈ [-1, 1]
    """
    minute_of_day = (hour - MARKET_OPEN_HOUR) * 60 + (minute - MARKET_OPEN_MINUTE)
    minute_of_day = max(0, min(minute_of_day, TOD_TRADING_MINUTES))
    angle = 2.0 * math.pi * minute_of_day / TOD_TRADING_MINUTES
    return math.sin(angle), math.cos(angle)

# =============================================================================
# NORMALIZATION CONTRACT
# =============================================================================

NORMALIZATION_PATH: str = "data/Datasetv4/v43/normalization_v43.pkl"
"""
Path to saved RobustScaler dict. Keys = feature names, values = fitted scalers.
- Fit on train split ONLY (first 70% by date)
- Ternary, bounded, sparse features: passthrough (no scaling)
- Computed binary features (friction_ok_*): passthrough
- Double-fit raises RuntimeError
"""

# Features excluded from normalization (passthrough)
NORMALIZATION_PASSTHROUGH: Set[str] = (
    set(TF_TERNARY_FEATURES) |
    set(TF_SPARSE_FEATURES) |
    set(TF_BOUNDED_FEATURES.keys()) |
    set(FRICTION_FEATURE_NAMES) |
    {"tod_sin", "tod_cos", REGIME_PERSISTENCE_FEATURE}
)

# =============================================================================
# PIVOT MASK CONTRACT
# =============================================================================

PIVOT_MASK_CONTRACT: str = "four_separate"
"""
Four separate pivot masks are passed to CondorNet.forward():
  pivot_mask_m1:  [B, T, 13] bool
  pivot_mask_m5:  [B, T, 13] bool
  pivot_mask_m15: [B, T, 13] bool
  pivot_mask_h1:  [B, T, 13] bool

True = NaN at that position (no pivot event). Used by:
- PivotProjector: masked mean pooling before projection
- B3/B4 interpretability hooks (stacked [B, T, 52] view)

NOT pre-stacked before passing to forward(). Stacking done inside hooks only.
"""

N_PIVOT_FEATURES: int = len(TF_PIVOT_FEATURES)  # 13

# =============================================================================
# VALIDATION REPORT
# =============================================================================

@dataclass
class ValidationReport:
    """Result of validate_tf_dataframe() or validate_options_chain()."""
    source: str
    n_rows: int
    n_cols: int
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    type_mismatches: List[str] = field(default_factory=list)
    out_of_range: List[str] = field(default_factory=list)
    nan_pct: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.source}: {self.n_rows} rows, {self.n_cols} cols | "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        )

# =============================================================================
# VALIDATORS
# =============================================================================

def validate_tf_dataframe(df, tf_name: str) -> ValidationReport:
    """
    Validate a timeframe DataFrame against V2.2 schema.

    Checks:
    - All TF_FEATURE_NAMES present
    - Dtypes match TF_DTYPE_MAP
    - Bounded features within valid ranges
    - Ternary features have only valid values
    - Sparse features NOT imputed (NaN count consistent with known sparsity)
    - Timestamp column present and monotonically increasing

    Args:
        df: pandas DataFrame loaded from a TF CSV
        tf_name: One of "m1", "m5", "m15", "h1" (for logging)

    Returns:
        ValidationReport
    """
    import pandas as pd
    import numpy as np

    errors: List[str] = []
    warnings: List[str] = []
    missing: List[str] = []
    type_mis: List[str] = []
    out_of_range: List[str] = []
    nan_pct: Dict[str, float] = {}

    # 1. Timestamp check
    if "timestamp" not in df.columns:
        errors.append("Missing 'timestamp' column")
    else:
        if not df["timestamp"].is_monotonic_increasing:
            errors.append("Timestamps are not monotonically increasing")

    # 2. Required features present
    all_required = TF_FEATURE_NAMES + TF_LABEL_NAMES
    for col in all_required:
        if col not in df.columns:
            missing.append(col)
            errors.append(f"Missing required column: {col}")

    # 3. Dtype checks (only for present columns)
    for col, expected_dtype in TF_DTYPE_MAP.items():
        if col in df.columns and expected_dtype != "__default__":
            actual = str(df[col].dtype)
            if "int" in expected_dtype and "int" not in actual:
                type_mis.append(col)
                warnings.append(f"Column {col}: expected {expected_dtype}, got {actual}")

    # 4. Bounded feature range checks
    for col, (lo, hi) in TF_BOUNDED_FEATURES.items():
        if col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                actual_min = float(col_data.min())
                actual_max = float(col_data.max())
                if actual_min < lo - 1e-6 or actual_max > hi + 1e-6:
                    out_of_range.append(col)
                    warnings.append(
                        f"Column {col}: values [{actual_min:.4f}, {actual_max:.4f}] "
                        f"outside bounds [{lo}, {hi}]"
                    )

    # 5. Ternary feature checks
    for col in TF_TERNARY_FEATURES:
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            valid = {-1, 0, 1, -1.0, 0.0, 1.0}
            invalid = unique_vals - valid
            if invalid:
                errors.append(f"Ternary column {col} has invalid values: {invalid}")

    # 6. NaN pct tracking
    for col in TF_FEATURE_NAMES:
        if col in df.columns:
            pct = float(df[col].isna().mean() * 100)
            nan_pct[col] = round(pct, 3)

    # 7. Sparse features: warn if NaN rate is 0 (implies imputation happened upstream)
    for col in TF_PIVOT_FEATURES:
        if col in df.columns:
            pct = nan_pct.get(col, 0.0)
            if pct < 1.0 and tf_name == "m5":
                warnings.append(
                    f"Sparse pivot column {col} has only {pct:.1f}% NaN on {tf_name} — "
                    f"may have been incorrectly imputed upstream"
                )

    # 8. Cross-TF flag warnings
    for flag_type, cols in KNOWN_CROSS_TF_FLAGS.items():
        for col in cols:
            if col in df.columns:
                warnings.append(
                    f"Cross-TF flag {flag_type} on column {col} — handle per addendum spec"
                )

    passed = len(errors) == 0

    return ValidationReport(
        source=f"{tf_name}_dataset",
        n_rows=len(df),
        n_cols=len(df.columns),
        passed=passed,
        errors=errors,
        warnings=warnings,
        missing_features=missing,
        type_mismatches=type_mis,
        out_of_range=out_of_range,
        nan_pct=nan_pct,
    )


def validate_options_chain(df) -> ValidationReport:
    """
    Validate options chain DataFrame.

    Checks:
    - Required columns: timestamp, contract_id, strike, expiry_date, option_type,
      bid, ask, iv, delta, gamma, theta, vega, open_interest
    - bid <= ask (no inverted markets)
    - delta ∈ [-1, 1]
    - gamma >= 0
    - iv > 0
    - No duplicate (timestamp, contract_id) pairs
    - Timestamps parseable

    Args:
        df: pandas DataFrame loaded from options_2025_v43.csv

    Returns:
        ValidationReport
    """
    required_cols = [
        "timestamp", "contract_id", "strike", "expiry_date", "option_type",
        "bid", "ask", "iv", "delta", "gamma", "theta", "vega", "open_interest",
    ]
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Required columns
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required chain column: {col}")

    if errors:
        return ValidationReport(
            source="options_chain", n_rows=len(df), n_cols=len(df.columns),
            passed=False, errors=errors, warnings=warnings,
        )

    # 2. No inverted markets
    inverted = (df["bid"] > df["ask"]).sum()
    if inverted > 0:
        errors.append(f"{inverted} rows have bid > ask (inverted market)")

    # 3. Delta range
    bad_delta = ((df["delta"] < -1.0) | (df["delta"] > 1.0)).sum()
    if bad_delta > 0:
        errors.append(f"{bad_delta} rows have delta outside [-1, 1]")

    # 4. Gamma non-negative
    bad_gamma = (df["gamma"] < 0).sum()
    if bad_gamma > 0:
        errors.append(f"{bad_gamma} rows have gamma < 0")

    # 5. IV positive
    bad_iv = (df["iv"] <= 0).sum()
    if bad_iv > 0:
        errors.append(f"{bad_iv} rows have iv <= 0")

    # 6. Duplicate check
    dup_count = df.duplicated(subset=["timestamp", "contract_id"]).sum()
    if dup_count > 0:
        errors.append(f"{dup_count} duplicate (timestamp, contract_id) pairs")

    # 7. Liquidity warning
    low_oi = (df["open_interest"] < CHAIN_GRID_CONFIG["liquidity_min_oi"]).mean() * 100
    if low_oi > 20:
        warnings.append(f"{low_oi:.1f}% of contracts have OI < {CHAIN_GRID_CONFIG['liquidity_min_oi']}")

    passed = len(errors) == 0
    return ValidationReport(
        source="options_chain",
        n_rows=len(df),
        n_cols=len(df.columns),
        passed=passed,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# QUICK SELF-CHECK
# =============================================================================

if __name__ == "__main__":
    print(f"CondorNet Schema {SCHEMA_VERSION}")
    print(f"  TF input features:    {len(TF_FEATURE_NAMES)}")
    print(f"  Pivot features:       {len(TF_PIVOT_FEATURES)}")
    print(f"  Friction windows:     {FRICTION_CANDIDATE_WINDOWS}")
    print(f"  ETL computed:         {ETL_COMPUTED_FEATURES}")
    print(f"  Strategy types:       {N_STRATEGY_TYPES} ({STRATEGY_TYPES})")
    print(f"  Chain grid in_feat:   {CHAIN_GRID_CONFIG['in_features']}")
    print(f"  Chain d_chain:        {CHAIN_GRID_CONFIG['d_chain']}")
    print(f"  Normalization skip:   {len(NORMALIZATION_PASSTHROUGH)} features")

    # Verify no overlap between input features and labels
    overlap = set(TF_FEATURE_NAMES) & set(TF_LABEL_NAMES)
    assert not overlap, f"Labels in feature list: {overlap}"

    # Verify all ternary features are in the full list
    for f in TF_TERNARY_FEATURES:
        assert f in TF_FEATURE_NAMES, f"Ternary {f} not in TF_FEATURE_NAMES"

    # Verify DTE_AFFINITY has all strategy types
    for st in STRATEGY_TYPES:
        assert st in DTE_AFFINITY, f"Strategy {st} missing from DTE_AFFINITY"

    # Verify tod feature formula
    tod_sin, tod_cos = compute_tod_features(9, 30)   # Market open
    assert abs(tod_sin) < 1e-9, f"tod_sin at open should be 0, got {tod_sin}"
    tod_sin2, tod_cos2 = compute_tod_features(16, 0)  # Market close
    assert abs(tod_sin2) < 1e-9, f"tod_sin at close should be ~0, got {tod_sin2}"

    print("\nAll schema invariant checks PASSED")
