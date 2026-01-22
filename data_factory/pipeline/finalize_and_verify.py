import pandas as pd
import numpy as np
import os
import sys

# Add project root for imports
sys.path.append(os.getcwd())
from intelligence.canonical_feature_registry import FEATURE_COLS_V22

def verify_dataset(filepath):
    print(f"\n{'='*60}")
    print(f"VERIFYING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # 1. Header Check
    df_peek = pd.read_csv(filepath, nrows=100)
    cols = df_peek.columns.tolist()
    
    missing_features = [f for f in FEATURE_COLS_V22 if f not in cols]
    if missing_features:
        print(f"[X] Missing {len(missing_features)} features: {missing_features[:10]}")
    else:
        print(f"[OK] All 52 canonical features present.")
        
    # 2. Structural Check (100 rows per bar)
    chunk_size = 1000000
    total_rows = 0
    unique_ts = set()
    nan_counts = {f: 0 for f in FEATURE_COLS_V22}
    
    print("Processing chunks for structural integrity and NaNs...")
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        total_rows += len(chunk)
        unique_ts.update(chunk.get('timestamp', chunk.get('dt')).unique())
        
        # Check NaNs for features
        for f in FEATURE_COLS_V22:
            if f in chunk.columns:
                nan_counts[f] += chunk[f].isna().sum()
                
    n_unique = len(unique_ts)
    expected_rows = n_unique * 100
    
    print(f"Total Rows:      {total_rows:,}")
    print(f"Expected Rows:   {expected_rows:,} ({n_unique:,} bars * 100)")
    
    if total_rows == expected_rows:
        print("[OK] Structural integrity maintained (Exactly 100 rows per bar).")
    else:
        print(f"[X] Structural mismatch! Difference: {total_rows - expected_rows} rows.")
        
    # 3. NaN Report
    total_nans = sum(nan_counts.values())
    if total_nans == 0:
        print("[OK] Zero NaNs detected in feature columns.")
    else:
        print(f"[!] {total_nans:,} NaNs detected across feature columns.")
        for f, count in nan_counts.items():
            if count > 0:
                print(f"    - {f}: {count:,} NaNs")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    files = [
        'data/processed/mamba_institutional_2024_1m.csv',
        'data/processed/mamba_institutional_2025_1m.csv'
    ]
    for f in files:
        if os.path.exists(f):
            verify_dataset(f)
        else:
            print(f"File not found: {f}")
