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
    "nan": 0.0,
    "posinf": 10.0,
    "neginf": -10.0,
}

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
