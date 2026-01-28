#!/usr/bin/env python3
"""
V2.2 Feature Validation and Repair Script

This script:
1. Checks if V2.2 features have meaningful variance (not constant defaults)
2. If suspicious, recomputes primitives from OHLCV data
3. Validates the fix
4. Saves corrected file

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

# Thresholds for suspicious detection
MIN_STD = 0.01          # Minimum standard deviation
MIN_UNIQUE = 3          # Minimum unique values


def validate_features(df: pd.DataFrame, sample_size: int = 50000) -> Tuple[bool, Dict]:
    """
    Validate V2.2 features for meaningful variance.

    Returns:
        (is_valid, details_dict)
    """
    # Sample for speed on large files
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    results = {}
    suspicious = []
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
            continue

        col_data = df_sample[col].dropna()
        if len(col_data) == 0:
            print(f"{col:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'ALL NaN':<12}")
            suspicious.append(col)
            continue

        col_min = col_data.min()
        col_max = col_data.max()
        col_mean = col_data.mean()
        col_std = col_data.std()
        col_unique = col_data.nunique()

        # Determine status
        if col_std < MIN_STD or col_unique < MIN_UNIQUE:
            status = "SUSPICIOUS"
            suspicious.append(col)
        else:
            status = "OK"

        results[col] = {
            'min': col_min, 'max': col_max, 'mean': col_mean,
            'std': col_std, 'unique': col_unique, 'status': status
        }

        print(f"{col:<20} {col_min:>10.4f} {col_max:>10.4f} {col_mean:>10.4f} {col_std:>10.4f} {col_unique:>8} {status:<12}")

    print("=" * 80)

    is_valid = len(suspicious) == 0 and len(missing) == 0

    if suspicious:
        print(f"\nWARNING:  SUSPICIOUS COLUMNS (constant/low variance): {suspicious}")
    if missing:
        print(f"\nWARNING:  MISSING COLUMNS: {missing}")
    if is_valid:
        print("\nOK: All V2.2 features have meaningful variance!")

    return is_valid, {'suspicious': suspicious, 'missing': missing, 'results': results}


def recompute_v22_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute V2.2 primitive features from OHLCV data.
    """
    from intelligence.features.dynamic_features import (
        compute_all_dynamic_features,
        compute_all_primitive_features_v22
    )

    print("\n" + "=" * 80)
    print("RECOMPUTING V2.2 FEATURES")
    print("=" * 80)

    # Drop existing suspicious columns to force recomputation
    cols_to_drop = [c for c in VALIDATION_COLS if c in df.columns]
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} existing columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Ensure required OHLCV columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    missing_ohlcv = [c for c in required if c not in df.columns]
    if missing_ohlcv:
        raise ValueError(f"Missing required OHLCV columns: {missing_ohlcv}")

    # Step 1: Compute dynamic features (RSI, ADX, etc.)
    print("\n[1/2] Computing dynamic features (RSI, ADX, BB, etc.)...")
    df = compute_all_dynamic_features(
        df,
        close_col="close",
        high_col="high",
        low_col="low"
    )

    # Step 2: Compute V2.2 primitives (friction, exec_allow, gap_risk, etc.)
    print("\n[2/2] Computing V2.2 primitive features...")
    df = compute_all_primitive_features_v22(
        df,
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
        spread_col="spread_ratio" if "spread_ratio" in df.columns else "close",
        inplace=True
    )

    print("\nOK: Feature recomputation complete!")
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
        print(f"ERROR: Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load data
    print(f"\n[LOAD] Loading: {args.input}")
    if args.rows:
        # For large files, read only last N rows efficiently
        # First count total rows
        total_rows = sum(1 for _ in open(args.input)) - 1  # -1 for header
        skip_rows = max(0, total_rows - args.rows)
        df = pd.read_csv(args.input, skiprows=range(1, skip_rows + 1))
        print(f"   Loaded last {len(df):,} rows (of {total_rows:,} total)")
    else:
        df = pd.read_csv(args.input)
        print(f"   Loaded {len(df):,} rows")

    # Initial validation
    is_valid, details = validate_features(df, sample_size=args.sample)

    if args.check_only:
        sys.exit(0 if is_valid else 1)

    # Recompute if needed
    if not is_valid or args.force_recompute:
        if is_valid and args.force_recompute:
            print("\nWARNING:  Force recompute requested despite valid features")

        df = recompute_v22_features(df)

        # Re-validate
        print("\n" + "=" * 80)
        print("POST-RECOMPUTE VALIDATION")
        print("=" * 80)
        is_valid_after, details_after = validate_features(df, sample_size=args.sample)

        if not is_valid_after:
            print("\nERROR: ERROR: Features still invalid after recomputation!")
            print("   This may indicate missing source data (spread_ratio, etc.)")
            sys.exit(1)

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
        print("OK: DONE! Features validated and fixed.")
        print(f"   Output: {output_path}")
        print("=" * 80)
    else:
        print("\nOK: No recomputation needed - features are valid!")


if __name__ == "__main__":
    main()
