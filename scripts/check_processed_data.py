
import os
import pandas as pd
import glob

DATA_DIR = "data/processed"

def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def check_file(filepath):
    print(f"\n--------------------------------------------------------")
    print(f"[FILE] Checking: {os.path.basename(filepath)}")
    
    file_size = os.path.getsize(filepath)
    print(f"   Size: {format_bytes(file_size)}")
    
    try:
        # 1. Read Header
        df_head = pd.read_csv(filepath, nrows=5)
        print(f"   Columns: {len(df_head.columns)}")
        print(f"   Sample Cols: {list(df_head.columns[:5])} ...")
        
        # 2. Check Key Columns
        key_cols = ['date', 'close', 'open', 'high', 'low', 'volume']
        missing = [c for c in key_cols if c not in df_head.columns]
        if missing:
             # Try fallback 'dt' or 'timestamp'
             if 'dt' in df_head.columns or 'timestamp' in df_head.columns:
                 print(f"   [INFO] 'date' missing but found timestamp col.")
             else:
                 print(f"   [WARN] Missing Key Columns: {missing}")
        else:
             print(f"   [PASS] Key schema columns present.")
             
        # 3. Row Count (Chunked to save RAM)
        row_count = 0
        chunk_size = 1_000_000
        print("   Counting rows...", end="", flush=True)
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, usecols=[df_head.columns[0]]):
            row_count += len(chunk)
            print(".", end="", flush=True)
        print(f" Done.")
        print(f"   Total Rows: {row_count:,}")
        
        # 4. Logical Check (Size vs Rows)
        # 1M rows ~ 500MB with 60 cols.
        # 2025 file (6GB) -> Expect ~12M rows?
        # 2024 file (5.5GB) -> Expect ~11M rows?
        # One year of 1-min SPY options ~??? 
        # Wait, institutional file is OPTIONS data (multiple strikes per minute).
        # Avg ~100 rows per minute?
        # 1 year = 252 * 390 = 98,280 mins.
        # 98k * 100 = 9.8M rows. 
        # So 5.5GB for 2024 seems correct (~10M rows).
        # 2025 (Jan only) -> 20 days * 390 = 7,800 mins.
        # 7.8k * 100 = 780,000 rows. -> Should be ~400MB.
        # If 2025 file is 6GB, it has ~12M rows. That's 15x larger than expected.
        
        normalized_size = file_size / (row_count if row_count > 0 else 1)
        print(f"   Bytes/Row: {normalized_size:.1f}")

    except Exception as e:
        print(f"   [FAIL] Error reading file: {e}")

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    files.sort()
    
    print(f"[SCAN] Found {len(files)} files in {DATA_DIR}")
    
    for f in files:
        check_file(f)

if __name__ == "__main__":
    main()
