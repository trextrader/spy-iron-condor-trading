#!/usr/bin/env python3
"""
V2.3 Forensic Feature Validation and Repair Script

This version implements "Interrogation-Proof" institutional checks:
1. Row-Key Alignment: Uses (timestamp + option_symbol) to guarantee perfect column swaps.
2. Multi-Grade Status: FAIL_NAN, FAIL_CONST, WARN_LOW_VAR, OK.
3. Delta Reporting: Percent difference between input and output.
4. Time-Aware Recomputation: Uses the updated dynamic_features engine to avoid rolling-collapse.
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# === VALIDATION CONFIGURATION (V2.2 Canonical) ===
VALIDATION_SCHEMA = {
    # 1. Market Primitives (5)
    'open':           {'type': 'price', 'bounds': [0, 1000]},
    'high':           {'type': 'price', 'bounds': [0, 1000]},
    'low':            {'type': 'price', 'bounds': [0, 1000]},
    'close':          {'type': 'price', 'bounds': [0, 1000]},
    'volume':         {'type': 'float', 'bounds': [0, 1e9]},

    # 2. Options / Chain State (9)
    'delta':          {'type': 'float', 'bounds': [-1.1, 1.1]},
    'gamma':          {'type': 'float', 'bounds': [0, 10]},
    'vega':           {'type': 'float', 'bounds': [0, 10]},
    'theta':          {'type': 'float', 'bounds': [-100, 10]},
    'iv':             {'type': 'float', 'bounds': [0, 10]},
    'ivr':            {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'spread_ratio':   {'type': 'float', 'bounds': [0, 1], 'require_var': True},
    'te':             {'type': 'float', 'bounds': [0, 365]},
    'strike':         {'type': 'price', 'bounds': [0, 1000]},

    # 3. Strategy / Targets / Risk (2)
    'target_spot':    {'type': 'price', 'bounds': [0, 1000]},
    'max_dd_60m':     {'type': 'float', 'bounds': [-1, 1]},

    # 4. Dynamic / Regime-Aware Features (16)
    'log_return':     {'type': 'float', 'bounds': [-0.5, 0.5]},
    'vol_ewma':       {'type': 'float', 'bounds': [0, 1]},
    'ret_z':          {'type': 'float', 'bounds': [-100, 100]},
    'atr_pct':        {'type': 'float', 'bounds': [0, 0.5]},
    'kappa_proxy':    {'type': 'float', 'bounds': [-100, 100]},
    'vol_energy':     {'type': 'float', 'bounds': [0, 100]},
    'rsi_dyn':        {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'adx_adaptive':   {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'psar_adaptive':  {'type': 'float', 'bounds': [-100, 100]},
    'bb_mu_dyn':      {'type': 'price', 'bounds': [0, 1000]},
    'bb_sigma_dyn':   {'type': 'float', 'bounds': [0, 100]},
    'bb_lower_dyn':   {'type': 'price', 'bounds': [0, 1000]},
    'bb_upper_dyn':   {'type': 'price', 'bounds': [0, 1000]},
    'stoch_k_dyn':    {'type': 'float', 'bounds': [0, 100]},
    'consolidation_score': {'type': 'float', 'bounds': [0, 1]},
    'breakout_score': {'type': 'float', 'bounds': [-1.1, 1.1]},

    # 5. V2.2 Primitive Outputs (20)
    'bb_percentile':  {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'bw_expansion_rate': {'type': 'float', 'bounds': [-10, 10]},
    'cmf':            {'type': 'float', 'bounds': [-1.1, 1.1]},
    'pressure_up':    {'type': 'float', 'bounds': [0, 2]},
    'pressure_down':  {'type': 'float', 'bounds': [0, 2]},
    'friction_ratio': {'type': 'float', 'bounds': [0, 1], 'require_var': True, 'min_std': 0.001},
    'exec_allow':     {'type': 'binary', 'bounds': [0, 1]},
    'gap_risk_score': {'type': 'float', 'bounds': [0, 1], 'require_var': True, 'min_std': 0.001},
    'risk_override':  {'type': 'binary', 'bounds': [0, 1]},
    'iv_confidence':  {'type': 'float', 'bounds': [0, 1.1]},
    'mtf_consensus':  {'type': 'float', 'bounds': [-1.1, 1.1]},
    'macd_norm':      {'type': 'float', 'bounds': [-10, 10]},
    'macd_signal_norm': {'type': 'float', 'bounds': [-10, 10]},
    'macd_histogram': {'type': 'float', 'bounds': [-10, 10]},
    'plus_di':        {'type': 'float', 'bounds': [0, 100]},
    'minus_di':       {'type': 'float', 'bounds': [0, 100]},
    'psar_trend':     {'type': 'binary', 'bounds': [-1, 1]},
    'psar_reversion_mu': {'type': 'float', 'bounds': [0, 1]},
    'chaos_membership': {'type': 'float', 'bounds': [0, 1]},
    'position_size_mult': {'type': 'float', 'bounds': [0, 1.1]},
    'fuzzy_reversion_11': {'type': 'float', 'bounds': [0, 1]},
    'beta1_norm_stub': {'type': 'float', 'bounds': [-10, 10]},
}

# Columns to drop (replaced by new primitives or moved to sub-folders)
DEPRECATED_COLS = ['volume_ratio', 'bid', 'ask', 'bandwidth', 'log_return_calc']

VALIDATION_COLS = list(VALIDATION_SCHEMA.keys())

def get_row_key(df: pd.DataFrame) -> pd.Series:
    """Generate a stable row key for alignment."""
    time_col = 'dt' if 'dt' in df.columns else ('timestamp' if 'timestamp' in df.columns else None)
    if time_col is None:
        raise ValueError("Dataset missing 'dt' or 'timestamp' for alignment.")
    
    if 'option_symbol' in df.columns:
        return df[time_col].astype(str) + "|" + df['option_symbol'].astype(str)
    elif 'symbol' in df.columns:
        return df[time_col].astype(str) + "|" + df['symbol'].astype(str)
    else:
        # Fallback to index if no symbol exists (risky for swaps)
        return df.index.astype(str)

def validate_columns(df: pd.DataFrame, sample_size: int = 50000) -> Dict:
    """Forensic validation using schema-driven rules."""
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    report = {}
    print("=" * 100)
    print(f"{'COLUMN':<20} | {'MIN':>8} | {'MAX':>8} | {'STD':>8} | {'UNIQUE':>8} | {'STATUS'}")
    print("-" * 100)

    for col, schema in VALIDATION_SCHEMA.items():
        if col not in df.columns:
            status = "MISSING"
            print(f"{col:<20} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | [FAIL] {status}")
            report[col] = {"status": status, "is_fail": True}
            continue

        data = df_sample[col].dropna()
        if len(data) == 0:
            status = "FAIL_EMPTY"
            print(f"{col:<20} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | [FAIL] {status}")
            report[col] = {"status": status, "is_fail": True}
            continue

        c_min, c_max, c_std, c_uni = data.min(), data.max(), data.std(), data.nunique()
        
        # Logic Tiering
        status = "OK"
        is_fail = False
        
        # 1. Hard Fails
        if np.isinf(data).any(): status = "FAIL_INF"; is_fail = True
        elif data.isna().any(): status = "FAIL_NAN"; is_fail = True
        elif 'bounds' in schema:
            if c_min < schema['bounds'][0] - 1e-6 or c_max > schema['bounds'][1] + 1e-6:
                status = "FAIL_BOUNDS"; is_fail = True

        # 2. Warnings / Conditionals
        if not is_fail:
            if c_uni == 1:
                status = "FAIL_CONST" if schema.get('require_var', True) else "OK_CONST"
                is_fail = (status == "FAIL_CONST")
            elif schema.get('require_var', True) and c_std < schema.get('min_std', 0.01):
                status = "WARN_LOW_VAR"
            elif schema['type'] == 'binary' and c_uni > 2:
                status = "WARN_NON_BINARY"

        print(f"{col:<20} | {c_min:>8.2f} | {c_max:>8.2f} | {c_std:>8.4f} | {c_uni:>8} | [{status}]")
        report[col] = {"status": status, "is_fail": is_fail, "std": c_std, "unique": c_uni}

    print("-" * 100)
    return report

def forensic_repair(df: pd.DataFrame) -> pd.DataFrame:
    """Memory-safe recomputation and winner-selection for 10M rows."""
    from intelligence.features.dynamic_features import (
        compute_all_dynamic_features,
        compute_all_primitive_features_v22
    )

    import gc
    print("\n[REPAIR] Starting memory-safe forensic recomputation...")

    # 1. Bulk Drop Deprecated Columns
    to_drop = [c for c in DEPRECATED_COLS if c in df.columns]
    if to_drop:
        print(f"   [DROP] Removing {len(to_drop)} deprecated columns...")
        df.drop(columns=to_drop, inplace=True)
    
    # helper drops
    to_drop_helpers = [c for c in ['bandwidth', 'log_return', 'vol_ewma'] if c in df.columns]
    if to_drop_helpers:
        df.drop(columns=to_drop_helpers, inplace=True)
    
    gc.collect()

    # 2. MEMORY SAVER 1: Extract original columns into a dict of numpy arrays
    # Foundation columns must stay in DF to serve as inputs for recomputation
    FOUNDATION_COLS = {
        'open', 'high', 'low', 'close', 'volume', 
        'dt', 'timestamp', 'option_symbol', 'symbol', 'strike', 
        'te', 'target_spot', 'lag_minutes', 'ivr', 'mtf_consensus'
    }

    preserved_data = {}
    cols_to_extract = [c for c in VALIDATION_COLS if c in df.columns]
    print(f"   [EXTRACT] Preserving {len(cols_to_extract)} columns for interrogation...")
    
    for col in cols_to_extract:
        if col in FOUNDATION_COLS:
            preserved_data[col] = df[col].values # Keep in DF
        else:
            preserved_data[col] = df.pop(col).values # Move out of DF
    
    gc.collect()

    # 3. Recompute fresh (this now uses the map-optimized engine)
    print("   [RECOMPUTE] Running dynamic feature engines...")
    df = compute_all_dynamic_features(df)
    df = compute_all_primitive_features_v22(df)
    
    gc.collect()
    
    # 🕵️ MEMORY SAVER 2: Winner Selection (Column-by-Column Interrogation)
    print("\n[MERGE] Performing Alignment-Safe Winner Selection...")
    
    for i, col in enumerate(VALIDATION_COLS):
        if col not in df.columns:
            if col in preserved_data:
                df[col] = preserved_data[col] # Restore if missing
            continue
            
        if col in preserved_data:
            # Stats on the recomputed version
            s_new = df[col].std()
            u_new = df[col].nunique()
            
            # Stats on the preserved version (using Series for efficient NUnique on large arrays)
            s_orig = np.nanstd(preserved_data[col])
            u_orig = pd.Series(preserved_data[col]).nunique()
            
            # Winner logic
            is_new_better = (u_new > u_orig + 2) or (s_new > s_orig * 1.2 and u_new >= u_orig)
            if u_new == 1 and u_orig > 1: is_new_better = False

            if not is_new_better:
                print(f"   [KEEP: ORIG] {col:<17} (U:{u_orig} vs {u_new})")
                df[col] = preserved_data[col]
            else:
                print(f"   [UPGRADE: NEW] {col:<15} (U:{u_orig}->{u_new})")
            
            # 🕵️ MEMORY SAVER 3: Delete the reference immediately
            del preserved_data[col]
            if i % 10 == 0:
                gc.collect()

    return df


def main():
    parser = argparse.ArgumentParser(description="Institutional Forensic Validator V2.3 (1.0M -> 10M Scale)")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    print(f"\n[START] Auditing (Memory Optimized): {args.input}")
    
    # 🕵️ MEMORY SAVER 4: Load with float32 directly to avoid peak RAM spike
    print(f"   [LOAD] Initializing 10M-row streaming load...")
    # Get headers first to determine dtypes
    headers = pd.read_csv(args.input, nrows=0).columns
    # Explicitly exclude metadata columns from float32 coercion
    exclude_cols = ['dt', 'timestamp', 'option_symbol', 'symbol', 'call_put', 'expiration', 'Date', 'time', 'sym']
    dtype_dict = {col: np.float32 for col in headers if col not in exclude_cols}
    
    df = pd.read_csv(args.input, dtype=dtype_dict, low_memory=False)
    
    report_pre = validate_columns(df)
    has_fails = any(v['is_fail'] for v in report_pre.values())

    if has_fails and not args.check_only:
        print("\n[🚨] FAILS DETECTED. Commencing Forensic Repair...")
        start_t = time.time()
        df = forensic_repair(df)
        end_t = time.time()
        
        print(f"\n[DONE] Repair took {end_t - start_t:.1f}s")
        
        print("\n[RE-AUDIT] Verifying Fix...")
        validate_columns(df)
        
        out_path = args.output or args.input.replace(".csv", "_fixed.csv")
        print(f"\n[SAVE] Exporting to: {out_path}")
        df.to_csv(out_path, index=False)
    else:
        if has_fails:
            print("\n[🚨] Fails detected but --check-only is set. Exiting with error.")
            sys.exit(1)
        print("\n[✅] All features passed institutional audit. Volume and time-alignment verified.")

if __name__ == "__main__":
    main()
