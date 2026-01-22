
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def analyze_dataset(filepath):
    print(f"Analyzing {filepath}...")
    
    # Peek at header
    df_head = pd.read_csv(filepath, nrows=5)
    columns = list(df_head.columns)
    print(f"\nTotal Columns: {len(columns)}")
    
    # Initialize counters
    total_rows = 0
    call_count = 0
    put_count = 0
    unique_dts = set()
    rows_per_min = []
    
    chunksize = 500000
    
    # Process in chunks
    for chunk in tqdm(pd.read_csv(filepath, chunksize=chunksize, usecols=['dt', 'call_put']), desc="Scanning rows"):
        total_rows += len(chunk)
        
        # Count Calls/Puts
        cp_counts = chunk['call_put'].value_counts()
        call_count += cp_counts.get('C', 0)
        put_count += cp_counts.get('P', 0)
        
        # Unique timestamps
        # Filter purely for strings or convert to string to avoid float/nan issues
        valid_dts = [str(x) for x in chunk['dt'].unique() if pd.notna(x)]
        unique_dts.update(valid_dts)
        
        # Rows per minute (approximate distrib)
        # Group by dt just for this chunk isn't perfect if minutes span chunks, 
        # but good enough for avg stats or we can accumulate counts per dt
        chunk_dt_counts = chunk['dt'].value_counts()
        rows_per_min.extend(chunk_dt_counts.values)

    print("\n" + "="*40)
    print("DATASET STATISTICS")
    print("="*40)
    print(f"File: {filepath}")
    print(f"Total Rows: {total_rows:,}")
    print(f"Total Columns: {len(columns)}")
    print(f"Date Range: {min(unique_dts)} to {max(unique_dts)}")
    print(f"Total Minutes: {len(unique_dts):,}")
    print("-" * 20)
    print("Option Type Distribution:")
    print(f"  Calls: {call_count:,} ({call_count/total_rows:.1%})")
    print(f"  Puts : {put_count:,} ({put_count/total_rows:.1%})")
    print("-" * 20)
    
    avg_rows = np.mean(rows_per_min)
    min_rows = np.min(rows_per_min)
    max_rows = np.max(rows_per_min)
    
    print("Density Stats (Rows per Minute):")
    print(f"  Average: {avg_rows:.2f}")
    print(f"  Min    : {min_rows}")
    print(f"  Max    : {max_rows}")
    print("-" * 20)
    
    print("\nColumn List:")
    # Format columns in a nice table with types
    dtypes = df_head.dtypes
    for i, col in enumerate(columns):
        print(f"{i+1:02d}. {col:<25} {dtypes[col]}")

if __name__ == "__main__":
    analyze_dataset(r"c:\SPYOptionTrader_test\data\processed\mamba_institutional_1m_v21.csv")
