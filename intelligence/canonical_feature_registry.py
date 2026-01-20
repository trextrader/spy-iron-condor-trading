"""
Canonical Feature Registry V2.1 for CondorBrain Pipeline.

This module is the SINGLE SOURCE OF TRUTH for feature schema.
All training, inference, and backtest scripts MUST import from here.

Version History:
- v2.0: 24 static features (deprecated)
- v2.1: 32 dynamic features (current)
"""

from typing import List, Dict, Any

# =============================================================================
# V2.1 FEATURE COLUMNS (ORDER MATTERS - DO NOT REORDER)
# =============================================================================

FEATURE_COLS_V21: List[str] = [
    # === Market Primitives (5) ===
    "open",
    "high",
    "low",
    "close",
    "volume",
    
    # === Options / Chain State (9) ===
    "delta",
    "gamma",
    "vega",
    "theta",
    "iv",
    "ivr",
    "spread_ratio",
    "te",
    "strike",
    
    # === Strategy / Targets / Risk (2) ===
    "target_spot",
    "max_dd_60m",
    
    # === Dynamic / Regime-Aware Features (17) ===
    "log_return",           # 1m log return
    "vol_ewma",             # EWMA realized vol
    "ret_z",                # log_return / (vol_ewma + eps) - vol-normalized return
    "atr_pct",              # ATR / close
    "kappa_proxy",          # Curvature proxy (2nd diff / scale)
    "vol_energy",           # log(1 + α|κ|) - dynamic modulator
    
    "rsi_dyn",              # Curvature-weighted RSI
    "adx_adaptive",         # Cycle-adaptive ADX
    "psar_adaptive",        # ATR-scaled PSAR
    
    "bb_mu_dyn",            # Dynamic BB mean
    "bb_sigma_dyn",         # Dynamic BB sigma
    "bb_lower_dyn",         # Dynamic lower band
    "bb_upper_dyn",         # Dynamic upper band
    
    "stoch_k_dyn",          # Vol-normalized Stochastic %K
    
    "consolidation_score",  # 0..1 (higher = consolidation)
    "breakout_score",       # {-1, 0, +1} (band break event)
]

# Derived constants
INPUT_DIM_V21: int = len(FEATURE_COLS_V21)  # Should be 32
VERSION_V21: str = "condorbrain_v2.1_dynamic"

# =============================================================================
# NORMALIZATION POLICY
# =============================================================================

NORMALIZATION_POLICY_V21: Dict[str, Any] = {
    "method": "robust_z",
    "fit_split": "train_only",
    "formula": "(x - median) / (1.4826 * mad)",
    "mad_epsilon": 1e-8,
    "clip_min": -10.0,
    "clip_max": 10.0,
}

NAN_POLICY_V21: Dict[str, float] = {
    "nan": 0.0,      # Global fallback (used for most features)
    "posinf": 10.0,
    "neginf": -10.0,
}

# =============================================================================
# SEMANTIC NEUTRAL FILL VALUES (per-feature)
# =============================================================================
# For bounded oscillators, 0 means "extreme" not "neutral".
# Use these values instead of global nan=0.0 during warmup.

NEUTRAL_FILL_VALUES: Dict[str, float] = {
    # --- These are fine with 0.0 (0 = neutral/no signal) ---
    "log_return": 0.0,       # 0 = no change
    "ret_z": 0.0,            # 0 = neutral z-score
    "kappa_proxy": 0.0,      # 0 = no curvature
    "vol_energy": 0.0,       # 0 = no energy
    "psar_adaptive": 0.0,    # 0 = price at SAR level
    "breakout_score": 0.0,   # 0 = no breakout
    
    # --- These need non-zero neutral values (0 = extreme!) ---
    "rsi_dyn": 50.0,         # 50 = neutral RSI (not overbought/oversold)
    "adx_adaptive": 25.0,    # 25 = weak/no trend (ADX neutral zone)
    "stoch_k_dyn": 50.0,     # 50 = mid-range (not extreme)
    "consolidation_score": 0.5,  # 0.5 = neither consolidation nor expansion
    "ivr": 0.5,              # 0.5 = mid-rank IV
    
    # --- Volatility: use small positive (0 can cause div-by-zero) ---
    "vol_ewma": 0.001,       # Small positive vol
    "atr_pct": 0.005,        # ~0.5% typical ATR
    "bb_sigma_dyn": 1.0,     # 1 std dev (arbitrary but non-zero)
    
    # --- Price-based: forward-fill is better, but if forced use 0 ---
    # (These get normalized anyway, so 0 becomes ~median after scaling)
    "bb_mu_dyn": 0.0,        # Will be overwritten by ffill typically
    "bb_lower_dyn": 0.0,
    "bb_upper_dyn": 0.0,
}


def get_neutral_fill_value(feature_name: str) -> float:
    """Get the semantically neutral fill value for a feature."""
    return NEUTRAL_FILL_VALUES.get(feature_name, 0.0)


def apply_semantic_nan_fill(X: "np.ndarray", feature_cols: List[str]) -> "np.ndarray":
    """
    Apply per-feature semantic NaN filling instead of global 0.0.
    
    This prevents injecting fake 'extreme' signals during warmup periods.
    
    Args:
        X: Feature array of shape (N, F) with NaN values
        feature_cols: List of feature names matching X columns
        
    Returns:
        X with NaN replaced by semantically neutral values
    """
    import numpy as np
    
    X = X.copy()
    for i, col in enumerate(feature_cols):
        fill_val = get_neutral_fill_value(col)
        mask = np.isnan(X[:, i])
        X[mask, i] = fill_val
    
    # Also handle inf values
    X = np.clip(X, -1e10, 1e10)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    
    return X

# =============================================================================
# CHECKPOINT KEYS (what must be saved/loaded)
# =============================================================================

CHECKPOINT_REQUIRED_KEYS_V21: List[str] = [
    "state_dict",
    "feature_cols",
    "median",
    "mad",
    "input_dim",
    "version",
]

# =============================================================================
# DETAILED FEATURE REGISTRY (for documentation/validation)
# =============================================================================

FEATURE_REGISTRY_V21: Dict[str, Any] = {
    "version": VERSION_V21,
    "n_features": INPUT_DIM_V21,
    "dtype": "float32",
    "nan_policy": NAN_POLICY_V21,
    "scaling": NORMALIZATION_POLICY_V21,
    "feature_cols": FEATURE_COLS_V21,
    "features": [
        # Market Primitives
        {"name": "open", "i": 0, "kind": "price", "range": "SPY $"},
        {"name": "high", "i": 1, "kind": "price", "range": "SPY $"},
        {"name": "low", "i": 2, "kind": "price", "range": "SPY $"},
        {"name": "close", "i": 3, "kind": "price", "range": "SPY $"},
        {"name": "volume", "i": 4, "kind": "volume", "range": "shares"},
        
        # Options
        {"name": "delta", "i": 5, "kind": "greek", "range": "[-1, 1]"},
        {"name": "gamma", "i": 6, "kind": "greek", "range": "small +"},
        {"name": "vega", "i": 7, "kind": "greek", "range": "+"},
        {"name": "theta", "i": 8, "kind": "greek", "range": "often -"},
        {"name": "iv", "i": 9, "kind": "vol", "range": "[0, ~2]"},
        {"name": "ivr", "i": 10, "kind": "rank", "range": "[0, 1]"},
        {"name": "spread_ratio", "i": 11, "kind": "liquidity", "range": ">=0"},
        {"name": "te", "i": 12, "kind": "time", "range": "days"},
        {"name": "strike", "i": 13, "kind": "strike", "range": "SPY $"},
        
        # Strategy
        {"name": "target_spot", "i": 14, "kind": "target", "range": "SPY $"},
        {"name": "max_dd_60m", "i": 15, "kind": "risk", "range": "pct/USD"},
        
        # Dynamic Features
        {"name": "log_return", "i": 16, "kind": "kinematics", "range": "centered"},
        {"name": "vol_ewma", "i": 17, "kind": "volatility", "range": "+"},
        {"name": "ret_z", "i": 18, "kind": "normalized_return", "range": "centered"},
        {"name": "atr_pct", "i": 19, "kind": "volatility", "range": "[0, 0.05]"},
        {"name": "kappa_proxy", "i": 20, "kind": "geometry", "range": "centered"},
        {"name": "vol_energy", "i": 21, "kind": "energy", "range": "[0, inf)"},
        
        {"name": "rsi_dyn", "i": 22, "kind": "oscillator", "range": "[0, 100]"},
        {"name": "adx_adaptive", "i": 23, "kind": "trend", "range": "[0, 100]"},
        {"name": "psar_adaptive", "i": 24, "kind": "trend", "range": "centered"},
        
        {"name": "bb_mu_dyn", "i": 25, "kind": "band", "range": "SPY $"},
        {"name": "bb_sigma_dyn", "i": 26, "kind": "band", "range": "+"},
        {"name": "bb_lower_dyn", "i": 27, "kind": "band", "range": "SPY $"},
        {"name": "bb_upper_dyn", "i": 28, "kind": "band", "range": "SPY $"},
        
        {"name": "stoch_k_dyn", "i": 29, "kind": "oscillator", "range": "[0, 100]"},
        
        {"name": "consolidation_score", "i": 30, "kind": "regime", "range": "[0, 1]"},
        {"name": "breakout_score", "i": 31, "kind": "event", "range": "{-1,0,1}"},
    ],
}

# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_feature_cols(cols: List[str]) -> bool:
    """Check if provided columns match the canonical schema."""
    return cols == FEATURE_COLS_V21


def get_missing_features(df_cols: List[str]) -> List[str]:
    """Return list of features missing from DataFrame columns."""
    return [f for f in FEATURE_COLS_V21 if f not in df_cols]


def assert_schema_match(df_cols: List[str], strict: bool = True) -> None:
    """Raise error if DataFrame doesn't have required columns."""
    missing = get_missing_features(df_cols)
    if missing:
        msg = f"Missing {len(missing)} features: {missing[:5]}..."
        if strict:
            raise ValueError(msg)
        else:
            print(f"⚠️ WARNING: {msg}")


# =============================================================================
# LEGACY SUPPORT (V2.0 - 24 features)
# =============================================================================

FEATURE_COLS_V20: List[str] = [
    "open", "high", "low", "close", "volume",
    "delta", "gamma", "vega", "theta", "iv", "ivr",
    "spread_ratio", "te",
    "rsi", "atr", "adx",
    "bb_lower", "bb_upper",
    "stoch_k", "sma", "psar",
    "strike", "target_spot", "max_dd_60m",
]

INPUT_DIM_V20: int = 24
VERSION_V20: str = "condorbrain_v2.0_static"

# =============================================================================
# V2.2 FEATURE COLUMNS (14 CANONICAL PRIMITIVES INTEGRATED)
# =============================================================================
# Expands V2.1 (32 features) with 20 primitive outputs = 52 total

FEATURE_COLS_V22: List[str] = [
    # === V2.1 Base Features (32) ===
    # Market Primitives (5)
    "open", "high", "low", "close", "volume",
    # Options / Chain State (9)
    "delta", "gamma", "vega", "theta", "iv", "ivr", "spread_ratio", "te", "strike",
    # Strategy / Targets / Risk (2)
    "target_spot", "max_dd_60m",
    # Dynamic / Regime-Aware (16)
    "log_return", "vol_ewma", "ret_z", "atr_pct", "kappa_proxy", "vol_energy",
    "rsi_dyn", "adx_adaptive", "psar_adaptive",
    "bb_mu_dyn", "bb_sigma_dyn", "bb_lower_dyn", "bb_upper_dyn",
    "stoch_k_dyn", "consolidation_score", "breakout_score",
    
    # === V2.2 Primitive Outputs (20) ===
    # P001-P007: Bands, Microstructure, Regime
    "bb_percentile",        # P002: Bandwidth percentile rank
    "bw_expansion_rate",    # P002: Bandwidth expansion rate
    "volume_ratio",         # P003: Volume / SMA_20(Volume)
    "friction_ratio",       # P004: Spread / AvgRange
    "exec_allow",           # P004: Friction gate output
    "gap_risk_score",       # P005: Composite gap risk
    "iv_confidence",        # P006: exp(-λ * lag_minutes)
    "mtf_consensus",        # P007: Multi-timeframe consensus
    
    # M001-M004: Momentum
    "macd_norm",            # M001: MACD / vol_ewma
    "macd_signal_norm",     # M001: Signal / vol_ewma
    "macd_histogram",       # M001: macd_norm - signal_norm
    "plus_di",              # M002: +DI from ADX
    "minus_di",             # M002: -DI from ADX
    "psar_trend",           # M004: PSAR trend direction
    "psar_reversion_mu",    # M004: PSAR reversion membership
    
    # T001-T002: Topology (stubbed for V2.2, full in V2.3)
    "beta1_norm_stub",      # T001: Placeholder (=0.0)
    "chaos_membership",     # T002: Chaos membership
    "position_size_mult",   # T002: Position size multiplier
    
    # F001: Fuzzy
    "fuzzy_reversion_11",   # F001: 11-factor fuzzy score
]

INPUT_DIM_V22: int = len(FEATURE_COLS_V22)  # Should be 52
VERSION_V22: str = "condorbrain_v2.2_primitives"

# V2.2 Neutral Fill Values (additions)
NEUTRAL_FILL_VALUES_V22: Dict[str, float] = {
    **NEUTRAL_FILL_VALUES,  # Inherit V2.1 values
    # Primitives
    "bb_percentile": 50.0,      # Mid-percentile = neutral
    "bw_expansion_rate": 0.0,   # No expansion
    "volume_ratio": 1.0,        # Average volume
    "friction_ratio": 0.5,      # Mid-friction
    "exec_allow": 1.0,          # Allow by default
    "gap_risk_score": 0.0,      # No gap risk
    "iv_confidence": 1.0,       # Full confidence
    "mtf_consensus": 0.0,       # Neutral consensus
    "macd_norm": 0.0,           # No momentum
    "macd_signal_norm": 0.0,
    "macd_histogram": 0.0,
    "plus_di": 25.0,            # Neutral DI
    "minus_di": 25.0,
    "psar_trend": 0.0,          # No trend
    "psar_reversion_mu": 0.0,   # No reversion
    "beta1_norm_stub": 0.0,     # TDA stub
    "chaos_membership": 0.0,    # No chaos
    "position_size_mult": 1.0,  # Full size
    "fuzzy_reversion_11": 0.5,  # Neutral fuzzy
}

# V2.2 Checkpoint Keys
CHECKPOINT_REQUIRED_KEYS_V22: List[str] = [
    "state_dict",
    "feature_cols",
    "median",
    "mad",
    "input_dim",
    "version",
    "primitive_params",  # New: stores primitive hyperparameters
]

# V2.2 Feature Registry
FEATURE_REGISTRY_V22: Dict[str, Any] = {
    "version": VERSION_V22,
    "n_features": INPUT_DIM_V22,
    "dtype": "float32",
    "nan_policy": NAN_POLICY_V21,
    "scaling": NORMALIZATION_POLICY_V21,
    "feature_cols": FEATURE_COLS_V22,
    "primitive_integration": True,
    "tda_enabled": False,  # Deferred to V2.3
}


def get_neutral_fill_value_v22(feature_name: str) -> float:
    """Get the semantically neutral fill value for a V2.2 feature."""
    return NEUTRAL_FILL_VALUES_V22.get(feature_name, 0.0)

