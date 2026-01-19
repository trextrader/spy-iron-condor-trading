#!/usr/bin/env python3
"""
Precompute all dynamic features into a new CSV so training can do a simple:
    df = pd.read_csv(...)

Default:
  - input : data/processed/mamba_institutional_1m.csv
  - output: data/processed/mamba_institutional_1m_v21.csv

This script:
  1) Loads the huge dataset
  2) Runs compute_all_dynamic_features(df)
  3) Writes the enriched dataset (expected 32 columns in schema v2.1)

Usage examples:
  python scripts/precompute_features.py
  python scripts/precompute_features.py --input data/processed/mamba_institutional_1m.csv --output data/processed/mamba_institutional_1m_v21.csv
  python scripts/precompute_features.py --nrows 200000   # quick test subset
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# LOCKED IMPORT: exact module path for compute_all_dynamic_features
from intelligence.features.dynamic_features import compute_all_dynamic_features
from intelligence.canonical_feature_registry import FEATURE_COLS_V21, INPUT_DIM_V21


def _human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:,.2f}{unit}"
        n /= 1000.0
    return f"{n:,.2f}P"


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute dynamic features into a new CSV.")
    parser.add_argument(
        "--input",
        default=os.path.join("data", "processed", "mamba_institutional_1m.csv"),
        help="Input CSV path (huge dataset).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "processed", "mamba_institutional_1m_v21.csv"),
        help="Output CSV path (enriched dataset).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional: load only first N rows for a quick test.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Skip sorting by (symbol, dt). Sorting is usually recommended for rolling features.",
    )
    parser.add_argument(
        "--dt-col",
        default="dt",
        help="Datetime column name (default: dt).",
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=[None, "c", "pyarrow"],
        help="CSV engine. 'pyarrow' can be faster if installed.",
    )
    parser.add_argument(
        "--verify-schema",
        action="store_true",
        help="Verify output has exactly 32 columns matching V2.1 schema.",
    )
    args = parser.parse_args()

    inp = args.input
    out = args.output

    if not os.path.exists(inp):
        print(f"[ERROR] Input file not found: {inp}")
        return 2

    print(f"[Precompute] Input : {inp}")
    print(f"[Precompute] Output: {out}")
    print(f"[Precompute] Schema: V2.1 ({INPUT_DIM_V21} features expected)")
    if args.nrows:
        print(f"[Precompute] nrows: {args.nrows:,}")

    t0 = time.time()

    # Dtypes to reduce RAM
    dtype = {
        "symbol": "category",
        "call_put": "category",
    }

    # Read CSV
    read_kwargs = dict(
        low_memory=False,
        dtype=dtype,
        nrows=args.nrows,
    )
    if args.engine is not None:
        read_kwargs["engine"] = args.engine

    print("[Precompute] Loading CSV...")
    df = pd.read_csv(inp, **read_kwargs)

    if args.dt_col not in df.columns:
        print(f"[ERROR] dt column '{args.dt_col}' not found. Columns: {list(df.columns)[:20]} ...")
        return 2

    # Parse dt
    print("[Precompute] Parsing datetime...")
    df[args.dt_col] = pd.to_datetime(df[args.dt_col], utc=True, errors="coerce")

    bad_dt = df[args.dt_col].isna().sum()
    if bad_dt:
        print(f"[WARN] {bad_dt:,} rows have unparseable dt and will remain NaT.")

    if not args.no_sort:
        sort_cols = [c for c in ["symbol", args.dt_col] if c in df.columns]
        if sort_cols:
            print(f"[Precompute] Sorting by {sort_cols} ...")
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    before_cols = list(df.columns)
    before_n = len(before_cols)
    before_rows = len(df)
    print(f"[Precompute] Loaded rows: {before_rows:,} | cols: {before_n}")

    # =========================================================================
    # CRITICAL FIX: Options data has ~100 rows per 1-minute bar (one per strike)
    # Dynamic features must be computed on UNIQUE SPOT bars, then merged back.
    # Otherwise rolling indicators compute across strikes instead of time.
    # =========================================================================
    
    # Identify OHLCV columns (these are same for all strikes at same timestamp)
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    spot_key_cols = ['symbol', args.dt_col]
    
    # Step 1: Extract unique spot bars
    print("[Precompute] Extracting unique spot bars...")
    spot_df = df.drop_duplicates(subset=spot_key_cols)[spot_key_cols + ohlcv_cols].copy()
    spot_df = spot_df.sort_values(spot_key_cols).reset_index(drop=True)
    n_unique_bars = len(spot_df)
    print(f"[Precompute] Unique spot bars: {n_unique_bars:,} (from {before_rows:,} options rows)")
    
    # Step 2: Compute dynamic features on spot data only
    print("[Precompute] Computing dynamic features on spot bars...")
    t1 = time.time()
    
    spot_df = compute_all_dynamic_features(spot_df)
    
    t2 = time.time()
    
    # Get list of newly added columns
    dynamic_cols = [c for c in spot_df.columns if c not in spot_key_cols + ohlcv_cols]
    print(f"[Precompute] Dynamic columns computed: {dynamic_cols}")
    
    # Step 3: Merge dynamic features back to full options dataframe
    print("[Precompute] Merging dynamic features back to options data...")
    merge_cols = spot_key_cols + dynamic_cols
    out_df = df.merge(spot_df[merge_cols], on=spot_key_cols, how='left')
    
    # Verify merge didn't change row count
    if len(out_df) != before_rows:
        print(f"[WARN] Row count changed after merge: {before_rows:,} -> {len(out_df):,}")

    after_cols = list(out_df.columns)
    after_n = len(after_cols)

    added = [c for c in after_cols if c not in before_cols]
    missing = [c for c in before_cols if c not in after_cols]

    print(f"[Precompute] Feature compute time: {t2 - t1:.1f}s")
    print(f"[Precompute] Columns before: {before_n} | after: {after_n}")
    if added:
        print(f"[Precompute] Added columns ({len(added)}): {added}")
    if missing:
        print(f"[WARN] Missing original columns after feature compute ({len(missing)}): {missing}")

    # Report NaN rates in added cols
    if added:
        nan_report = {}
        for c in added:
            s = out_df[c]
            try:
                nan_report[c] = float(s.isna().mean())
            except Exception:
                continue
        if nan_report:
            top = sorted(nan_report.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print("[Precompute] Top NaN rates in added cols (up to 10):")
            for c, r in top:
                print(f"  - {c}: {r*100:.2f}%")

    # Verify schema if requested
    if args.verify_schema:
        missing_features = [f for f in FEATURE_COLS_V21 if f not in out_df.columns]
        if missing_features:
            print(f"[ERROR] Schema verification failed! Missing {len(missing_features)} features:")
            for f in missing_features[:10]:
                print(f"  - {f}")
            return 3
        print(f"[Precompute] ✅ Schema verified: all {INPUT_DIM_V21} V2.1 features present")

    # Write CSV
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print("[Precompute] Writing output CSV...")
    
    # Keep dt as ISO string
    out_df[args.dt_col] = out_df[args.dt_col].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    out_df.to_csv(out, index=False)

    t3 = time.time()
    elapsed = t3 - t0
    try:
        size_mb = os.path.getsize(out) / (1024 * 1024)
        print(f"[Precompute] ✅ Done in {elapsed:.1f}s | output size: {size_mb:,.1f} MB")
    except Exception:
        print(f"[Precompute] ✅ Done in {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
