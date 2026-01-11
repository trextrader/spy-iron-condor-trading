
import pandas as pd
import os
import argparse

def validate_and_sort(file_path, date_col='date', timestamp_col='timestamp'):
    print(f"Checking {file_path}...")
    if not os.path.exists(file_path):
        print(f"  [Error] File not found.")
        return

    # Check file size to decide strategy
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")

    try:
        # Read headers first to check columns
        df_head = pd.read_csv(file_path, nrows=5)
        cols = df_head.columns.tolist()
        
        has_ts = timestamp_col in cols
        has_date = date_col in cols
        
        sort_cols = []
        if has_ts:
            sort_cols.append(timestamp_col)
        elif has_date:
            sort_cols.append(date_col)
            if 'time' in cols:
                sort_cols.append('time')
        
        if not sort_cols:
            print("  [Skip] No suitable date/timestamp column found for sorting.")
            return

        print(f"  Sorting by {sort_cols}...")
        
        # Load full file
        df = pd.read_csv(file_path, low_memory=False)
        
        # Convert sort columns to datetime logic
        if has_ts:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        if has_date:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
        print("  [Action] Sorting file (forcing chronological order)...")
        df = df.sort_values(by=sort_cols)
        
        # Write back
        print("  [Saving] Overwriting file...")
        df.to_csv(file_path, index=False)
        print("  [Done] File sorted and saved.")
            
    except Exception as e:
        print(f"  [Error] {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="Files to sort")
    args = parser.parse_args()
    
    for f in args.files:
        validate_and_sort(f)
