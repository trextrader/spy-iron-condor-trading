import pandas as pd
import numpy as np

def verify_structure(csv_path):
    print(f"Checking {csv_path}...")
    # Load first 1000 rows to see the pattern
    df = pd.read_csv(csv_path, nrows=1000)
    
    # Check for 'dt' or 'timestamp'
    time_col = 'dt' if 'dt' in df.columns else 'timestamp'
    if time_col not in df.columns:
        print(f"Error: Time column not found. Available: {df.columns.tolist()}")
        return

    # Group by time and see counts
    counts = df.groupby(time_col).size()
    print("\nCounts per timestamp (first few):")
    print(counts.head(10))
    
    # Check if they are consistent
    unique_counts = counts.unique()
    print(f"\nUnique grouping counts: {unique_counts}")
    
    # Check underlying OHLCV repetition
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    if all(c in df.columns for c in ohlcv_cols):
        # Pick the first timestamp group
        first_ts = counts.index[0]
        group = df[df[time_col] == first_ts]
        
        # Verify all OHLCV are same
        for col in ohlcv_cols:
            is_same = (group[col] == group[col].iloc[0]).all()
            print(f"OHLCV Column '{col}' is repeated: {is_same}")
    else:
        print("OHLCV columns missing.")

if __name__ == "__main__":
    verify_structure("data/processed/mamba_institutional_2025_1m_v22.csv")
