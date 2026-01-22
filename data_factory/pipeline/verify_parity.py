import pandas as pd
import numpy as np
import os
import sys

def verify_file(filepath):
    print(f"\nVerifying: {filepath}")
    
    # Check headers
    h = pd.read_csv(filepath, nrows=0).columns.tolist()
    expected_h = [
        'timestamp', 'symbol', 'underlying_price', 'open', 'high', 'low', 'close',
        'option_symbol', 'expiration', 'strike', 'call_put', 'bid', 'ask', 'iv',
        'delta', 'gamma', 'theta', 'vega', 'rho', 'volume', 'open_interest'
    ]
    if h != expected_h:
        print(f"  [X] Header mismatch!")
    else:
        print(f"  [OK] Headers match expected.")

    chunk_size = 1000000
    total_rows = 0
    unique_ts = set()
    row_count_errors = []
    put_call_errors = []
    ohlcv_errors = []
    
    # Process in chunks
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        total_rows += len(chunk)
        
        # 1. Count rows per timestamp (Vectorized)
        ts_counts = chunk['timestamp'].value_counts()
        bad_counts = ts_counts[ts_counts != 100]
        # Ignore boundary timestamps (might be split between chunks)
        if not bad_counts.empty:
            # We filter out the first and last timestamp in the chunk as they might be split
            mid_bad = bad_counts.index[1:-1] if len(bad_counts) > 2 else []
            if len(mid_bad) > 0:
                row_count_errors.extend(mid_bad.tolist())
        
        unique_ts.update(ts_counts.index.tolist())
        
        # 2. Check put/call balance (Vectorized)
        # Only check timestamps that have exactly 100 rows in this chunk
        full_blocks = ts_counts[ts_counts == 100].index
        if not full_blocks.empty:
            block_data = chunk[chunk['timestamp'].isin(full_blocks)]
            pc_counts = block_data.groupby(['timestamp', 'call_put']).size().unstack(fill_value=0)
            bad_pc = pc_counts[(pc_counts.get('P', 0) != 50) | (pc_counts.get('C', 0) != 50)]
            if not bad_pc.empty:
                put_call_errors.extend(bad_pc.index.tolist())
        
        # 3. Check OHLCV consistency (Vectorized)
        ohlcv_cols = ['underlying_price', 'open', 'high', 'low', 'close']
        for col in ohlcv_cols:
            # Check if any timestamp has more than 1 unique value in this col
            # Only for full blocks
            if not full_blocks.empty:
                nunique = block_data.groupby('timestamp')[col].nunique()
                bad_ohlcv = nunique[nunique > 1]
                if not bad_ohlcv.empty:
                    ohlcv_errors.extend(bad_ohlcv.index.tolist())
                    break

    print(f"  Total Rows: {total_rows:,}")
    print(f"  Total Unique Timestamps: {len(unique_ts):,}")
    print(f"  Row Count Errors (mid-chunk): {len(row_count_errors)}")
    print(f"  Put/Call Balance Errors (full blocks): {len(put_call_errors)}")
    print(f"  OHLCV Inconsistency Errors: {len(ohlcv_errors)}")
    
    return len(unique_ts)

if __name__ == "__main__":
    f_2024 = 'data/processed/Spy_Options_2024_1m.csv'
    f_2025 = 'data/processed/Spy_Options_2025_1m_complete.csv'
    
    ts_2024 = verify_file(f_2024)
    ts_2025 = verify_file(f_2025)
    
    print(f"\nFinal Parity Check Summary:")
    print(f"  2024 Timestamps: {ts_2024:,}")
    print(f"  2025 Timestamps: {ts_2025:,}")
    
    if ts_2024 == ts_2025:
        print("  [OK] Timestamp counts match exactly!")
    else:
        diff = abs(ts_2024 - ts_2025)
        print(f"  [!] Timestamp count difference: {diff:,}")
        print(f"  This is expected if holidays or early closes differ between 2024 and 2025.")
