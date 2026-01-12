import pandas as pd
import numpy as np

def check_options_data(file_path):
    print(f"Checking: {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Total Rows
    print(f"Total Rows: {len(df):,}")
    
    # 2. Chronological Check
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    is_sorted = df['timestamp'].is_monotonic_increasing
    print(f"Chronologically Ordered: {is_sorted}")
    
    # If not sorted, show some info
    if not is_sorted:
        print("Warning: Data is not sorted! Sorting now for analysis...")
        df = df.sort_values('timestamp')
        
    # 3. Start/Stop
    start = df['timestamp'].min()
    end = df['timestamp'].max()
    print(f"Start Date: {start}")
    print(f"End Date: {end}")
    
    # 4. Unique Timestamps & Interval Integrity
    unique_ts = df['timestamp'].unique()
    print(f"Unique 15m Snapshots: {len(unique_ts):,}")
    
    # Expected intervals (rough check)
    # Filter to trading hours (13:30 to 20:00 UTC)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # Check for gaps in trading days
    days = df['timestamp'].dt.date.unique()
    print(f"Total Trading Days: {len(days)}")
    
    # Check first and last 10 unique timestamps
    print("\nFirst 5 unique timestamps:")
    print(sorted(unique_ts)[:5])
    print("\nLast 5 unique timestamps:")
    print(sorted(unique_ts)[-5:])

if __name__ == "__main__":
    import sys
    check_options_data(sys.argv[1])
