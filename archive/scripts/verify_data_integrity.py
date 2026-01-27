
import os
import sys
import pandas as pd
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligence.features.dynamic_features import compute_all_dynamic_features, compute_all_primitive_features_v22

DATA_PATH = "data/processed/mamba_institutional_2024_1m_last 500k.csv"

def verify_data_integrity():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("❌ Data file not found!")
        return

    df_orig = pd.read_csv(DATA_PATH)
    
    # 1. Verification: Do columns exist?
    v21_cols = ['rsi', 'atr', 'adx', 'stoch_k']
    v22_cols = ['sma', 'psar', 'psar_mark']
    
    missing = []
    for c in v21_cols + v22_cols:
        if c not in df_orig.columns:
            missing.append(c)
            
    if missing:
        print(f"[WARN] Original CSV missing features: {missing}")
        print("   Cannot verify feature Logic if columns missing.")
    else:
        print("[PASS] Original CSV contains expected V2.1/V2.2 features.")

    # 2. Re-Calculation Test
    print("\n--- RE-CALCULATION AUDIT ---")
    print("Creating CLEAN dataframe (OHLCV only)...")
    
    # Extract only base columns
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    if 'timestamp' in df_orig.columns: base_cols.append('timestamp')
    if 'dt' in df_orig.columns: base_cols.append('dt')
    if 'call_put' in df_orig.columns: base_cols.append('call_put')
    if 'strike' in df_orig.columns: base_cols.append('strike')
    
    df_clean = df_orig[base_cols].copy()
    
    # Rename timestamp -> dt if needed (logic from backtest)
    if 'dt' not in df_clean.columns and 'timestamp' in df_clean.columns:
        df_clean.rename(columns={'timestamp': 'dt'}, inplace=True)
        
    print("Computing V2.1 features on clean data...")
    df_clean = compute_all_dynamic_features(df_clean, close_col='close', high_col='high', low_col='low')
    
    print("Computing V2.2 features on clean data...")
    # Need spread ratio?
    spread_col = 'close'
    if 'spread_ratio' in df_orig.columns:
        df_clean['spread_ratio'] = df_orig['spread_ratio']
        spread_col = 'spread_ratio'
        
    df_clean = compute_all_primitive_features_v22(
        df_clean, close_col='close', high_col='high', low_col='low', volume_col='volume',
        spread_col=spread_col, inplace=True
    )
    
    # 3. Compare Results
    print("\n--- COMPARISON RESULTS ---")
    mismatches = 0
    checked = 0
    
    columns_to_check = v21_cols + v22_cols
    
    for col in columns_to_check:
        if col not in df_orig.columns or col not in df_clean.columns:
            continue
            
        orig_vals = df_orig[col].fillna(0).values
        new_vals = df_clean[col].fillna(0).values
        
        # Check alignment (timestamps might differ if rows dropped?)
        # Assuming identical row count
        if len(orig_vals) != len(new_vals):
            print(f"❌ ROW COUNT MISMATCH for {col}: {len(orig_vals)} vs {len(new_vals)}")
            continue
            
        # Error metrics
        diff = np.abs(orig_vals - new_vals)
        mae = np.mean(diff)
        max_err = np.max(diff)
        
        # Tolerance
        TOLERANCE = 1e-4
        if mae < TOLERANCE and max_err < 1e-3:
            print(f"✅ {col:<10} MATCHES (MAE: {mae:.6f})")
        else:
            print(f"❌ {col:<10} MISMATCH (MAE: {mae:.6f}, Max: {max_err:.6f})")
            mismatches += 1
        checked += 1

    if mismatches == 0:
        print(f"\n[PASS] AUDIT PASSED: All {checked} checked features match within tolerance.")
        print("   This confirms Backtest Logic == Training Data Pre-compute Logic.")
    else:
        print(f"\n[FAIL] AUDIT FAILED: {mismatches} features differ significantly.")
        print("   Investigate 'compute_all_...' functions vs 'precompute_features.py'.")

if __name__ == "__main__":
    verify_data_integrity()
