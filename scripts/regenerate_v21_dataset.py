"""
regenerate_v21_dataset.py

Regenerate the V21 precomputed dataset from the original mamba_institutional_1m.csv.

This script:
1. Reads the original dataset (with static indicators)
2. Extracts unique OHLCV bars (1 per minute)
3. Computes all dynamic V2.1 and V2.2 features on the unique bars
4. Merges the dynamic features back (same value for all 100 options per minute)
5. Validates that exactly 100 rows exist per dt timestamp
6. Saves the regenerated dataset

Usage:
    python scripts/regenerate_v21_dataset.py --input data/processed/mamba_institutional_1m.csv --output data/processed/mamba_institutional_1m_v21_new.csv
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add repo root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)

from intelligence.features.dynamic_features import (
    compute_all_dynamic_features,
    compute_all_primitive_features_v22,
)

def main():
    parser = argparse.ArgumentParser(description="Regenerate V21 precomputed dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--output", type=str, required=True, help="Path for output dataset")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't regenerate")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="Chunk size for processing")
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"V21 Dataset Regeneration Script")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # --- Step 1: Load and analyze original dataset ---
    print("[1/5] Loading original dataset...")
    
    # Read in chunks to handle large files
    total_rows = 0
    unique_dts = set()
    first_dt = None
    last_dt = None
    chunk_num = 0
    
    print("   Reading chunks...")
    for chunk in pd.read_csv(args.input, chunksize=args.chunk_size, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        chunk_dts = chunk['dt'].unique()
        unique_dts.update(chunk_dts)
        if first_dt is None:
            first_dt = chunk['dt'].iloc[0]
        last_dt = chunk['dt'].iloc[-1]
        print(f"\r   Rows read: {total_rows:,} (chunk {chunk_num})", end="", flush=True)
    
    print()  # New line after progress
    
    n_unique_dt = len(unique_dts)
    expected_rows_per_dt = 100
    expected_total = n_unique_dt * expected_rows_per_dt
    
    print(f"   Total rows:     {total_rows:,}")
    print(f"   Unique dt vals: {n_unique_dt:,}")
    print(f"   Rows per dt:    {total_rows / n_unique_dt:.2f} (expected: 100)")
    print(f"   Date range:     {first_dt} → {last_dt}")
    
    if abs(total_rows / n_unique_dt - 100) > 0.1:
        print(f"   ⚠️ WARNING: Rows per dt is not exactly 100!")
    else:
        print(f"   ✅ Rows per dt is ~100 as expected")
    
    if args.validate_only:
        print("\n[VALIDATE-ONLY] Exiting without regeneration.")
        return
    
    # --- Step 2: Load full dataset ---
    print("\n[2/5] Loading full dataset into memory...")
    print(f"   Expected ~{total_rows:,} rows based on validation scan")
    
    # Load in chunks with progress
    chunks = []
    rows_loaded = 0
    for chunk in pd.read_csv(args.input, chunksize=args.chunk_size, low_memory=False):
        chunks.append(chunk)
        rows_loaded += len(chunk)
        print(f"\r   Rows loaded: {rows_loaded:,} / ~{total_rows:,}", end="", flush=True)
    
    print()  # New line
    df = pd.concat(chunks, ignore_index=True)
    del chunks  # Free memory
    print(f"   ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Identify datetime column
    dt_col = 'dt' if 'dt' in df.columns else 'timestamp'
    df[dt_col] = pd.to_datetime(df[dt_col])
    
    # --- Step 3: Extract unique OHLCV bars ---
    print("\n[3/5] Extracting unique OHLCV bars for dynamic feature computation...")
    
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    # Include spread_ratio if it exists (some primitives may use it)
    aux_cols = [c for c in ['spread_ratio'] if c in df.columns]  # Removed 'symbol' from aux_cols
    
    # For SPY options data, OHLCV is same for all 100 rows per dt
    # Use only dt as key (not symbol, since symbol varies per option but OHLCV doesn't)
    key_cols = [dt_col]
    
    # Get unique OHLCV rows (first occurrence per dt)
    # Dedupe by dt only (OHLCV is constant per minute)
    spot_df = df.drop_duplicates(subset=[dt_col])[[dt_col] + ohlcv_cols + aux_cols].copy()
    spot_df = spot_df.sort_values(dt_col).reset_index(drop=True)
    
    n_unique = len(spot_df)
    print(f"   Unique spot bars: {n_unique:,}")
    
    # --- Step 4: Compute dynamic features on spot bars ---
    print("\n[4/5] Computing dynamic features on spot bars...")
    
    # V2.1 Dynamic Features
    print("   Computing V2.1 dynamic features...")
    spot_df = compute_all_dynamic_features(
        spot_df, 
        close_col="close", 
        high_col="high", 
        low_col="low"
    )
    
    # V2.2 Primitive Features
    print("   Computing V2.2 primitive features...")
    spot_df = compute_all_primitive_features_v22(
        spot_df,
        close_col="close",
        high_col="high", 
        low_col="low",
        volume_col="volume",
        spread_col="spread_ratio" if "spread_ratio" in spot_df.columns else "close",
    )
    
    # Get list of new dynamic columns (exclude key cols and original OHLCV)
    original_cols = set(key_cols + ohlcv_cols + aux_cols)
    dynamic_cols = [c for c in spot_df.columns if c not in original_cols]
    print(f"   Dynamic columns computed: {len(dynamic_cols)}")
    print(f"   Columns: {dynamic_cols[:10]}..." if len(dynamic_cols) > 10 else f"   Columns: {dynamic_cols}")
    
    # --- Step 5: Merge dynamic features back to full dataset ---
    print("\n[5/5] Merging dynamic features back to full dataset...")
    
    # Only keep key cols + dynamic cols for merge
    merge_cols = key_cols + dynamic_cols
    merge_df = spot_df[merge_cols]
    
    # Drop any existing dynamic columns from original df to avoid conflicts
    for col in dynamic_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Merge
    df = df.merge(merge_df, on=key_cols, how='left')
    
    print(f"   Merged dataset shape: {df.shape}")
    
    # Validate: check a few dt values have exactly 100 rows
    sample_dts = df[dt_col].drop_duplicates().sample(min(10, n_unique))
    for sample_dt in sample_dts:
        count = len(df[df[dt_col] == sample_dt])
        if count != 100:
            print(f"   ⚠️ dt={sample_dt} has {count} rows (expected 100)")
    
    # Check that dynamic values are same for all 100 rows per dt
    print("\n   Validating dynamic feature consistency per dt...")
    sample_dt = df[dt_col].iloc[0]
    sample_group = df[df[dt_col] == sample_dt]
    for col in dynamic_cols[:5]:  # Check first 5 dynamic cols
        unique_vals = sample_group[col].nunique()
        if unique_vals > 1:
            print(f"   ⚠️ Column {col} has {unique_vals} unique values within same dt (expected 1)")
        else:
            print(f"   ✅ Column {col}: consistent within dt")
    
    # --- Save output with row-by-row progress ---
    print(f"\n   Saving to {args.output}...")
    print(f"   Writing {len(df):,} rows to file...")
    
    # Write with progress tracking
    total_to_write = len(df)
    write_chunk_size = 10000  # Write in chunks for efficiency but report per-row
    
    # Write header first
    df.iloc[:0].to_csv(args.output, index=False)
    
    rows_written = 0
    for start_idx in range(0, total_to_write, write_chunk_size):
        end_idx = min(start_idx + write_chunk_size, total_to_write)
        chunk = df.iloc[start_idx:end_idx]
        
        # Append chunk to file
        chunk.to_csv(args.output, mode='a', header=False, index=False)
        
        rows_written = end_idx
        pct = (rows_written / total_to_write) * 100
        print(f"\r   Rows written: {rows_written:,} / {total_to_write:,} ({pct:.1f}%)", end="", flush=True)
    
    print()  # New line after progress
    
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"   ✅ Saved! File size: {file_size_mb:.1f} MB")
    
    print(f"\n{'='*60}")
    print(f"Regeneration Complete!")
    print(f"{'='*60}")
    print(f"Output:      {args.output}")
    print(f"Total rows:  {len(df):,}")
    print(f"Columns:     {len(df.columns)}")
    print(f"Date range:  {df[dt_col].min()} → {df[dt_col].max()}")

if __name__ == "__main__":
    main()
