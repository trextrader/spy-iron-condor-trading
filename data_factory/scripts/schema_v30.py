#!/usr/bin/env python3
"""
Dataset v3.0 Schema Definition

87-column schema for CondorNet HAL-S³N v4.0

This module defines the canonical column ordering and provides
schema validation utilities.
"""

# ============================================================================
# V3.0 SCHEMA DEFINITION (87 columns)
# ============================================================================

V30_SCHEMA = [
    # Core Identifiers (6)
    'timestamp', 'symbol', 'option_symbol', 'expiration', 'strike', 'call_put',

    # Price Layer (5)
    'underlying_price', 'open', 'high', 'low', 'close',

    # Options Surface (11)
    'delta', 'gamma', 'vega', 'theta', 'rho', 'iv', 'volume', 'open_interest',
    'te', 'ivr', 'spread_ratio',

    # Returns & Volatility (6)
    'log_return', 'vol_ewma', 'ret_z', 'atr_pct', 'kappa_proxy', 'vol_energy',

    # Trend & Momentum (12)
    'sma', 'rsi_dyn', 'adx_adaptive', 'psar_adaptive', 'psar_mark', 'psar_trend',
    'psar_reversion_mu', 'macd_norm', 'macd_signal_norm', 'macd_histogram',
    'plus_di', 'minus_di',

    # Bands & Consolidation (10)
    'bb_mu_dyn', 'bb_sigma_dyn', 'bb_lower_dyn', 'bb_upper_dyn', 'stoch_k_dyn',
    'bandwidth', 'bb_percentile', 'bw_expansion_rate', 'consolidation_score', 'breakout_score',

    # Flow & Breadth (4)
    'cmf', 'pressure_up', 'pressure_down', 'friction_ratio',

    # Fuzzy/Control Layer (8)
    'mtf_consensus', 'chaos_membership', 'position_size_mult', 'fuzzy_reversion_11',
    'exec_allow', 'gap_risk_score', 'risk_override', 'iv_confidence',

    # Targets (4)
    'target_spot', 'max_dd_60m', 'beta1_norm_stub', 'target_roi_calendar',

    # NEW v3.0 - Trend Structure (3)
    'FRAMA', 'Anchored_VWAP', 'McClellanOsc',

    # NEW v3.0 - IV Bands (3)
    'IV_High', 'IV_Mid', 'IV_Low',

    # NEW v3.0 - Options Chain Aggregates (4)
    'Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume', 'OSC_Volume',

    # NEW v3.0 - Flow & Breadth Extensions (4)
    'WeightedAlpha', 'WilderAccSwingIndex', 'AccDistWill', 'AccDistWillMovAvg',

    # NEW v3.0 - Microstructure (5)
    'aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread',
]

# Feature families for ablation experiments
FEATURE_FAMILIES = {
    'price': ['underlying_price', 'open', 'high', 'low', 'close'],
    'options_surface': ['delta', 'gamma', 'vega', 'theta', 'rho', 'iv', 'volume', 'open_interest', 'te', 'ivr', 'spread_ratio'],
    'volatility': ['log_return', 'vol_ewma', 'ret_z', 'atr_pct', 'kappa_proxy', 'vol_energy'],
    'trend_momentum': ['sma', 'rsi_dyn', 'adx_adaptive', 'psar_adaptive', 'macd_norm', 'macd_signal_norm', 'macd_histogram', 'plus_di', 'minus_di'],
    'bands_consolidation': ['bb_mu_dyn', 'bb_sigma_dyn', 'bb_lower_dyn', 'bb_upper_dyn', 'stoch_k_dyn', 'bandwidth', 'bb_percentile', 'bw_expansion_rate', 'consolidation_score', 'breakout_score'],
    'flow_breadth': ['cmf', 'pressure_up', 'pressure_down', 'friction_ratio'],
    'fuzzy_control': ['mtf_consensus', 'chaos_membership', 'position_size_mult', 'fuzzy_reversion_11', 'exec_allow', 'gap_risk_score', 'risk_override', 'iv_confidence'],
    'v30_trend_structure': ['FRAMA', 'Anchored_VWAP', 'McClellanOsc'],
    'v30_iv_bands': ['IV_High', 'IV_Mid', 'IV_Low'],
    'v30_chain_aggregates': ['Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume', 'OSC_Volume'],
    'v30_flow_extensions': ['WeightedAlpha', 'WilderAccSwingIndex', 'AccDistWill', 'AccDistWillMovAvg'],
    'v30_microstructure': ['aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread'],
}

# Expected value ranges for validation
EXPECTED_RANGES = {
    'delta': (-1.0, 1.0),
    'gamma': (0, 1.0),
    'iv': (0, 5.0),
    'ivr': (0, 1.0),
    'rsi_dyn': (0, 100),
    'bb_percentile': (-0.5, 1.5),  # Can slightly exceed 0-1
    'spread_ratio': (0, 1.0),
    'stoch_k_dyn': (0, 100),
}


def validate_schema(df, verbose=True):
    """Validate DataFrame has all 87 v3.0 columns."""
    missing = set(V30_SCHEMA) - set(df.columns)
    extra = set(df.columns) - set(V30_SCHEMA)

    passed = True

    if missing:
        if verbose:
            print(f"MISSING COLUMNS ({len(missing)}): {sorted(missing)}")
        passed = False

    if extra:
        if verbose:
            print(f"EXTRA COLUMNS ({len(extra)}): {sorted(extra)}")

    if passed and verbose:
        print(f"✅ Schema validation PASSED: {len(df.columns)} columns")

    return passed, missing, extra


def enforce_schema(df):
    """Enforce v3.0 schema on DataFrame, adding missing columns with 0."""
    for col in V30_SCHEMA:
        if col not in df.columns:
            df[col] = 0

    # Reorder to canonical order
    df = df[V30_SCHEMA]
    return df


if __name__ == "__main__":
    print(f"V3.0 Schema: {len(V30_SCHEMA)} columns")
    print(f"\nFeature Families: {len(FEATURE_FAMILIES)}")
    for family, cols in FEATURE_FAMILIES.items():
        print(f"  {family}: {len(cols)} features")
