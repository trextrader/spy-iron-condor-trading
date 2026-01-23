
import pandas as pd
import argparse
import sys

def audit_ohlcv_groups(file_path, output_csv="ohlcv_audit_results.csv"):
    print(f"Reading {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Standardize Column Names
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    
    # Identify OHLCV columns
    # We expect: open, high, low, close, volume. And timestamp/date/dt.
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Validate columns exist
    missing = [c for c in ohlcv_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing}")
        # Try fallbacks?
        return

    # Determine Timestamp Column
    ts_col = 'timestamp' if 'timestamp' in df.columns else ('dt' if 'dt' in df.columns else 'date')
    if ts_col not in df.columns:
        print("Warning: No timestamp column found. Grouping by OHLCV only.")
        group_cols = ohlcv_cols
    else:
        group_cols = [ts_col] + ohlcv_cols

    print(f"Grouping by: {group_cols}")
    
    # Logic: Sequential check
    # We want to identify consecutive blocks of identical OHLCV.
    # We can use the 'shift' trick to find boundaries.
    
    # Create a signature for each row (to detect changes)
    # We compare current row values vs prev row values.
    # If ANY of the group_cols change, it's a new group.
    
    # Efficient vectorization:
    # 1. Create a boolean mask of "row != prev_row"
    # 2. Cumsum to get group IDs
    
    # Fill NaNs to avoid comparison issues
    signature_df = df[group_cols].fillna(0)
    
    # Compare with shift
    # any(axis=1) checks if ANY column in the row differs from previous
    change_mask = (signature_df != signature_df.shift(1)).any(axis=1)
    
    # Group ID
    df['group_id'] = change_mask.cumsum()
    
    # Now group by this ID and count
    print("Calculating group sizes...")
    results = []
    
    # Iterate over groups to preserve order and get start rows
    # groupby(sort=False) respects order IF the groups were created sequentially (which they are by cumsum)
    grouped = df.groupby('group_id', sort=False)
    
    print(f"{'Start Row':<10} | {'Timestamp':<25} | {'Count':<5} | {'Status'}")
    print("-" * 60)
    
    for gid, group in grouped:
        start_idx = group.index[0] + 2 # +2 Because: +1 for 0-index, +1 for Header row (User asked "Row 2")
        count = len(group)
        
        # Get Timestamp for context
        ts_val = group[ts_col].iloc[0] if ts_col in df.columns else "N/A"
        
        status = "OK" if count == 100 else f"DIFF ({count})"
        
        # Print to console
        # Limit console output if too huge? Maybe just deviations or first few?
        # User said "until the end and print it to console". We'll print all.
        print(f"{start_idx:<10} | {str(ts_val):<25} | {count:<5} | {status}")
        
        results.append({
            'start_row_excel': start_idx,
            'timestamp': ts_val,
            'group_count': count,
            'status': status
        })

    # Save to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"\nAudit saved to: {output_csv}")
    
    # Summary
    print("\n--- SUMMARY ---")
    print(f"Total Groups: {len(res_df)}")
    perfect_groups = res_df[res_df['status'] == 'OK']
    print(f"Perfect 100-count groups: {len(perfect_groups)}")
    mismatched_groups = res_df[res_df['status'] != 'OK']
    if not mismatched_groups.empty:
        print("[FAIL] Found {len(mismatched_groups)} groups with non-100 count!")
        print(mismatched_groups.head(5))
    else:
        print("[PASS] All groups have exactly 100 rows.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = sys.argv[1]
    else:
        # Default fallback
        f = "data/processed/mamba_institutional_2024_1m_last 1mil.csv"
    
    audit_ohlcv_groups(f)
