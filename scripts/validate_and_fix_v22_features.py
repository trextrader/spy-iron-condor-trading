#!/usr/bin/env python3
"""
V2.2 Feature Validation and Repair Script

This script:
1. Checks if V2.2 features have meaningful variance (not constant defaults)
2. If suspicious, ONLY recomputes the bad columns, preserving good ones
3. Merges good original columns with recomputed columns
4. Validates the fix
5. Saves corrected file

Usage:
    python scripts/validate_and_fix_v22_features.py --input data/processed/your_file.csv
    python scripts/validate_and_fix_v22_features.py --input data/processed/your_file.csv --output data/processed/fixed_file.csv
    python scripts/validate_and_fix_v22_features.py --input data/processed/your_file.csv --check-only
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Key V2.2 columns that must have variance for meaningful training/inference
VALIDATION_COLS = [
    'exec_allow',
    'friction_ratio',
    'gap_risk_score',
    'rsi_dyn',
    'adx_adaptive',
    'ivr',
    'volume_ratio',
    'bb_percentile',
    'macd_norm',
]

# Columns that CANNOT be recomputed from OHLCV alone (need external data)
CANNOT_RECOMPUTE = ['ivr']  # IVR requires options chain / VIX data

# Thresholds for suspicious detection
MIN_STD = 0.01          # Minimum standard deviation
MIN_UNIQUE = 3          # Minimum unique values


def validate_features(df: pd.DataFrame, sample_size: int = 50000) -> Tuple[bool, List[str], List[str], Dict]:
    """
    Validate V2.2 features for meaningful variance.

    Returns:
        (is_valid, good_cols, bad_cols, details_dict)
    """
    # Sample for speed on large files
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    results = {}
    good_cols = []
    bad_cols = []
    missing = []

    print("=" * 80)
    print("V2.2 FEATURE VALIDATION")
    print(f"Total rows: {len(df):,} | Sampled: {len(df_sample):,}")
    print("=" * 80)
    print(f"{'Column':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Unique':>8} {'Status':<12}")
    print("-" * 80)

    for col in VALIDATION_COLS:
        if col not in df_sample.columns:
            print(f"{col:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'MISSING':<12}")
            missing.append(col)
            bad_cols.append(col)
            continue

        col_data = df_sample[col].dropna()
        if len(col_data) == 0:
            print(f"{col:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'ALL NaN':<12}")
            bad_cols.append(col)
            continue

        col_min = col_data.min()
        col_max = col_data.max()
        col_mean = col_data.mean()
        col_std = col_data.std()
        col_unique = col_data.nunique()

        # Determine status
        if col_std < MIN_STD or col_unique < MIN_UNIQUE:
            status = "BAD"
            bad_cols.append(col)
        else:
            status = "GOOD"
            good_cols.append(col)

        results[col] = {
            'min': col_min, 'max': col_max, 'mean': col_mean,
            'std': col_std, 'unique': col_unique, 'status': status
        }

        print(f"{col:<20} {col_min:>10.4f} {col_max:>10.4f} {col_mean:>10.4f} {col_std:>10.4f} {col_unique:>8} {status:<12}")

    print("=" * 80)

    is_valid = len(bad_cols) == 0

    if good_cols:
        print(f"\n[OK] GOOD COLUMNS (will preserve): {good_cols}")
    if bad_cols:
        print(f"\n[WARNING] BAD COLUMNS (will recompute): {bad_cols}")
    if is_valid:
        print("\n[OK] All V2.2 features have meaningful variance!")

    return is_valid, good_cols, bad_cols, {'missing': missing, 'results': results}


def smart_recompute_v22_features(df: pd.DataFrame, good_cols: List[str], bad_cols: List[str]) -> pd.DataFrame:
    """
    Smart recomputation: Only recompute BAD columns, preserve GOOD ones.

    Strategy:
    1. Save GOOD columns from original df
    2. Drop ALL validation columns
    3. Recompute everything fresh
    4. Swap in the GOOD original columns (overwrite recomputed)
    """
    from intelligence.features.dynamic_features import (
        compute_all_dynamic_features,
        compute_all_primitive_features_v22
    )

    print("\n" + "=" * 80)
    print("SMART RECOMPUTATION (Preserve Good, Fix Bad)")
    print("=" * 80)

    # Step 1: Save good columns from original
    preserved_data = {}
    for col in good_cols:
        if col in df.columns:
            preserved_data[col] = df[col].copy()
            print(f"  [PRESERVE] Saving good column: {col}")

    # Also preserve columns that cannot be recomputed
    for col in CANNOT_RECOMPUTE:
        if col in df.columns and col not in preserved_data:
            col_std = df[col].std()
            if col_std > MIN_STD:  # Only preserve if it has some variance
                preserved_data[col] = df[col].copy()
                print(f"  [PRESERVE] Saving non-recomputable column: {col} (std={col_std:.4f})")

    # Step 2: Identify columns to drop (all V2.2 cols that exist)
    # We drop everything and recompute, then swap back the good ones
    all_v22_cols = VALIDATION_COLS + [
        'bw_expansion_rate', 'risk_override', 'iv_confidence', 'mtf_consensus',
        'macd_signal_norm', 'macd_histogram', 'plus_di', 'minus_di',
        'psar_trend', 'psar_reversion_mu', 'beta1_norm_stub', 'chaos_membership',
        'position_size_mult', 'fuzzy_reversion_11'
    ]
    cols_to_drop = [c for c in all_v22_cols if c in df.columns]

    if cols_to_drop:
        print(f"\n  [DROP] Removing {len(cols_to_drop)} V2.2 columns for fresh recomputation")
        df = df.drop(columns=cols_to_drop)

    # Ensure required OHLCV columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    missing_ohlcv = [c for c in required if c not in df.columns]
    if missing_ohlcv:
        raise ValueError(f"Missing required OHLCV columns: {missing_ohlcv}")

    # Step 3: Recompute everything fresh
    print("\n  [COMPUTE 1/2] Computing dynamic features (RSI, ADX, BB, etc.)...")
    df = compute_all_dynamic_features(
        df,
        close_col="close",
        high_col="high",
        low_col="low"
    )

    print("\n  [COMPUTE 2/2] Computing V2.2 primitive features...")
    df = compute_all_primitive_features_v22(
        df,
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
        spread_col="spread_ratio" if "spread_ratio" in df.columns else "close",
        inplace=True
    )

    # Step 4: Swap back the preserved GOOD columns
    print(f"\n  [MERGE] Restoring {len(preserved_data)} preserved columns...")
    for col, data in preserved_data.items():
        if len(data) == len(df):
            df[col] = data
            print(f"    [OK] Restored: {col}")
        else:
            print(f"    [SKIP] Length mismatch for {col}: {len(data)} vs {len(df)}")

    print("\n[OK] Smart recomputation complete!")
    return df


def main():
    parser = argparse.ArgumentParser(description="Validate and fix V2.2 features")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", default=None, help="Output CSV file path (default: overwrite input with _fixed suffix)")
    parser.add_argument("--check-only", action="store_true", help="Only check, don't recompute")
    parser.add_argument("--force-recompute", action="store_true", help="Force recompute even if valid")
    parser.add_argument("--sample", type=int, default=50000, help="Sample size for validation (default: 50000)")
    parser.add_argument("--rows", type=int, default=None, help="Only process last N rows (for testing)")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    # Load data
    print(f"\n[LOAD] Loading: {args.input}")
    if args.rows:
        # For large files, read only last N rows efficiently
        total_rows = sum(1 for _ in open(args.input)) - 1  # -1 for header
        skip_rows = max(0, total_rows - args.rows)
        df = pd.read_csv(args.input, skiprows=range(1, skip_rows + 1))
        print(f"   Loaded last {len(df):,} rows (of {total_rows:,} total)")
    else:
        df = pd.read_csv(args.input)
        print(f"   Loaded {len(df):,} rows")

    # Initial validation
    is_valid, good_cols, bad_cols, details = validate_features(df, sample_size=args.sample)

    if args.check_only:
        sys.exit(0 if is_valid else 1)

    # Recompute if needed
    if not is_valid or args.force_recompute:
        if is_valid and args.force_recompute:
            print("\n[WARNING] Force recompute requested despite valid features")
            good_cols = []  # Treat all as bad for force recompute
            bad_cols = VALIDATION_COLS

        df = smart_recompute_v22_features(df, good_cols, bad_cols)

        # Re-validate
        print("\n" + "=" * 80)
        print("POST-RECOMPUTE VALIDATION")
        print("=" * 80)
        is_valid_after, good_after, bad_after, details_after = validate_features(df, sample_size=args.sample)

        if bad_after:
            unfixable = [c for c in bad_after if c in CANNOT_RECOMPUTE]
            fixable_still_bad = [c for c in bad_after if c not in CANNOT_RECOMPUTE]

            if fixable_still_bad:
                print(f"\n[WARNING] Some columns still bad after recompute: {fixable_still_bad}")
                print("   This may indicate missing source data (spread_ratio, volume history, etc.)")

            if unfixable:
                print(f"\n[INFO] These columns cannot be recomputed from OHLCV: {unfixable}")
                print("   They require external data (options chain, VIX, etc.)")

        # Save output
        if args.output:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.input)
            output_path = f"{base}_v22fixed{ext}"

        print(f"\n[SAVE] Saving to: {output_path}")
        df.to_csv(output_path, index=False)
        print(f"   Saved {len(df):,} rows")

        print("\n" + "=" * 80)
        print("[DONE] Features validated and fixed.")
        print(f"   Output: {output_path}")
        print(f"   Good columns preserved: {len(good_cols)}")
        print(f"   Bad columns recomputed: {len(bad_cols)}")
        print("=" * 80)
    else:
        print("\n[OK] No recomputation needed - features are valid!")


if __name__ == "__main__":
    main()
