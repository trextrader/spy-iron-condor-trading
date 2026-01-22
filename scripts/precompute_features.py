#!/usr/bin/env python3
"""
Precompute all dynamic features into a new CSV so training can do a simple:
    df = pd.read_csv(...)

Default:
  - input : data/processed/mamba_institutional_1m.csv
  - output: data/processed/mamba_institutional_1m_v21.csv

This script:
  1) Loads the huge dataset (with progress bar)
  2) Runs compute_all_dynamic_features(df) AND compute_all_primitive_features_v22(df)
  3) Writes the enriched dataset (expected 52+ columns in schema v2.2)

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
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# LOCKED IMPORT: exact module path for compute_all_dynamic_features
from intelligence.features.dynamic_features import (
    compute_all_dynamic_features, 
    compute_all_primitive_features_v22
)
from intelligence.canonical_feature_registry import FEATURE_COLS_V21, INPUT_DIM_V21, NEUTRAL_FILL_VALUES


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
    print(f"[Precompute] Schema: V2.1/V2.2 (Dynamic + Primitives)")
    if args.nrows:
        print(f"[Precompute] nrows: {args.nrows:,}")

    t0 = time.time()

    # Dtypes to reduce RAM
    dtype = {
        "symbol": "category",
        "call_put": "category",
    }

    # Read CSV with Chunking and Progress Bar
    read_kwargs = dict(
        low_memory=False,
        dtype=dtype,
        nrows=args.nrows,
        chunksize=500000, # Load in chunks to show progress
    )
    if args.engine is not None:
        read_kwargs["engine"] = args.engine

    print("[Precompute] Loading CSV...")
    chunks = []
    total_rows = 0
    
    with pd.read_csv(inp, **read_kwargs) as reader:
        for chunk in tqdm(reader, desc="Loading chunks", unit="chunk"):
            chunks.append(chunk)
            total_rows += len(chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks # free memory

    if args.dt_col not in df.columns:
        # Fallback check
        if 'date' in df.columns:
            args.dt_col = 'date'
        elif 'timestamp' in df.columns:
            args.dt_col = 'timestamp'
        else:
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

    # Identify OHLCV columns (these are same for all strikes at same timestamp)
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Use ONLY dt as the key for spot extraction. 
    spot_key_cols = [args.dt_col]
    
    # Step 1: Extract unique spot bars
    print("[Precompute] Extracting unique spot bars...")
    # Aux cols: spread_ratio if it exists (needed for Primitives)
    aux_cols = []
    if "spread_ratio" in df.columns:
        aux_cols.append("spread_ratio")
        
    spot_df = df.drop_duplicates(subset=spot_key_cols)[spot_key_cols + ohlcv_cols + aux_cols].copy()
    spot_df = spot_df.sort_values(spot_key_cols).reset_index(drop=True)
    n_unique_bars = len(spot_df)
    print(f"[Precompute] Unique spot bars: {n_unique_bars:,} (from {before_rows:,} options rows)")
    
    # Step 2: Compute dynamic features on spot data only
    print("[Precompute] Computing dynamic features on spot bars...")
    t1 = time.time()
    
    # V2.1 Dynamic features
    spot_df = compute_all_dynamic_features(spot_df)
    
    # -------------------------------------------------------------------------
    # CALCULATE MISSING FEATURES (Targets, SMA, IVR stub)
    # -------------------------------------------------------------------------
    print("   Computing targets (target_spot, max_dd_60m) and aux...")
    
    # 1. target_spot: Close price 60 mins into future
    spot_df['target_spot'] = spot_df['close'].shift(-60)
    
    # 2. max_dd_60m: Max Drawdown in next 60m
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=60)
    future_low_min = spot_df['low'].rolling(window=indexer).min()
    spot_df['max_dd_60m'] = (future_low_min - spot_df['close']) / spot_df['close']
    
    # 3. SMA (20-period)
    spot_df['sma'] = spot_df['close'].rolling(20).mean()
    
    # 4. PSAR Mark (raw) - usually computed by ta-lib. 
    # Or just use psar_adaptive as proxy if ta-lib missing.
    try:
        import talib
        spot_df['psar_mark'] = talib.SAR(spot_df['high'], spot_df['low'], acceleration=0.02, maximum=0.2)
    except:
        # fallback
        spot_df['psar_mark'] = 0.0

    print("   Targets computed.")
    
    # V2.2 Primitive features
    print("   Computing V2.2 primitive features...")
    # Ensure standard column names if they differ
    spot_df = compute_all_primitive_features_v22(
        spot_df,
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
        spread_col="spread_ratio" if "spread_ratio" in spot_df.columns else "close", 
    )
    
    t2 = time.time()
    
    # Get list of newly added columns
    dynamic_cols = [c for c in spot_df.columns if c not in spot_key_cols + ohlcv_cols + aux_cols]
    print(f"[Precompute] Dynamic columns computed: {len(dynamic_cols)}")
    print(f"[Precompute] Sample cols: {dynamic_cols[:5]} ...")
    
    # Step 3: Merge dynamic features back to full options dataframe
    print("[Precompute] Merging dynamic features back to options data...")
    merge_cols = spot_key_cols + dynamic_cols
    
    # Optimize dtypes
    for c in dynamic_cols:
         if spot_df[c].dtype == 'float64':
             spot_df[c] = spot_df[c].astype('float32')
             
    # Clean up df before merge
    legacy_replacements = [
        "rsi", "adx", "psar", 
        "bb_lower", "bb_upper", "bb_mu", "bb_sigma",
        "stoch_k", "atr", "sma", "psar_mark", "target_spot", "max_dd_60m"
    ]
    cols_to_drop = [c for c in dynamic_cols if c in df.columns]
    cols_to_drop.extend([c for c in legacy_replacements if c in df.columns])
    
    if cols_to_drop:
        cols_to_drop = list(set(cols_to_drop))
        print(f"   Dropping {len(cols_to_drop)} existing columns from original df...")
        df.drop(columns=cols_to_drop, inplace=True)
        
    out_df = df.merge(spot_df[merge_cols], on=spot_key_cols, how='left')
    
    print(f"[Precompute] Merge complete. Rows: {len(out_df):,}")
    
    # Post-Merge Features (TE, IVR)
    print("[Precompute] Computing Post-Merge Features (TE, IVR)...")
    
    # TE
    if 'expiration' in out_df.columns:
        # Fast convert if needed
        if out_df['expiration'].dtype == 'object':
             out_df['expiration'] = pd.to_datetime(out_df['expiration'], utc=True)
        elif out_df['expiration'].dt.tz is None:
             out_df['expiration'] = out_df['expiration'].dt.tz_localize('UTC')
        out_df['te'] = (out_df['expiration'] - out_df[args.dt_col]).dt.total_seconds() / 86400.0
    else:
        out_df['te'] = 0.0

    # IVR Stub
    if 'iv' in out_df.columns:
        out_df['ivr'] = 0.5 
    else:
        out_df['ivr'] = 0.0

    added = [c for c in out_df.columns if c not in before_cols]
    missing = [c for c in before_cols if c not in out_df.columns]
    
    # SEMANTIC NaN FILL
    print("[Precompute] Applying semantic NaN fill...")
    fill_items = [(col, val) for col, val in NEUTRAL_FILL_VALUES.items() if col in out_df.columns]
    for col, val in tqdm(fill_items, desc="Semantic fill", unit="col"):
        if out_df[col].hasnans:
            out_df[col] = out_df[col].fillna(val)
    
    fallback_cols = [col for col in added if col in out_df.columns]
    for col in tqdm(fallback_cols, desc="Fallback fill", unit="col"):
        if out_df[col].hasnans:
            out_df[col] = out_df[col].fillna(0.0)
    
    # Write CSV
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print("[Precompute] Writing output CSV...")
    
    out_df[args.dt_col] = out_df[args.dt_col].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out_df.to_csv(out, index=False)

    t3 = time.time()
    elapsed = t3 - t0
    print(f"[Precompute] [OK] Done in {elapsed:.1f}s")
    if os.path.exists(out):
        size_mb = os.path.getsize(out) / (1024 * 1024)
        print(f"             Output size: {size_mb:,.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
