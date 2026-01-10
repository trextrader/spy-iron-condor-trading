import pandas as pd
import os
import sys

def process_ivol_data():
    """
    Convert raw IVolatility data (spy_options_ivol_large.csv) 
    into the format expected by the Optimizer (spy_options_marks.csv).
    
    1. Reads raw file.
    2. Sorts by date.
    3. Normalizes column names and values (e.g. 'C' -> 'call').
    4. Saves to data/synthetic_options/spy_options_marks.csv
    """
    
    # Input/Output Config
    input_file = "data/ivolatility/spy_options_ivol_large.csv"
    output_dir = "data/synthetic_options"
    output_file = os.path.join(output_dir, "spy_options_marks.csv") # Outputting to 'spy' not 'spy_options_marks.csv' directly?
    # Optimizer looks for f"{base_s_cfg.underlying.lower()}_options_marks.csv" -> spx_options_marks.csv or spy_options_marks.csv
    # We should detect input symbol or assume SPY.
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records.")

    # Clean repeated headers if they exist (due to append mode)
    if 'date' in df.columns:
        df = df[df['date'] != 'date']
    
    # 1. Sort by Date
    print("Sorting by date and strike...")
    if 'date' in df.columns:
        df['sort_date'] = pd.to_datetime(df['date'], errors='coerce') # Handle any other bad data
        df = df.dropna(subset=['sort_date']) # Drop bad dates
        df.sort_values(by=['sort_date', 'expiration', 'strike'], inplace=True)
        df.drop(columns=['sort_date'], inplace=True)
    
    # 2. Rename Columns
    print("Mapping columns...")
    # Target Schema from SyntheticOptionsEngine:
    # timestamp,date,symbol,option_symbol,strike,expiration,contract_type,
    # bid,ask,last_price,bid_size,ask_size,volume,open_interest,
    # delta,gamma,theta,vega,implied_volatility
    
    rename_map = {
        'call_put': 'contract_type',
        'mean_price': 'last_price',
        'iv': 'implied_volatility',
        'underlying_price': 'underlying_last'
    }
    df.rename(columns=rename_map, inplace=True)

    # 3. Data Transformations
    
    # Contract Type: C/P -> call/put (if needed, though optimizer handles C/P)
    # But let's standardize to 'call'/'put' to match Synthetic engine output
    df['contract_type'] = df['contract_type'].map({'C': 'call', 'P': 'put', 'c': 'call', 'p': 'put'}).fillna(df['contract_type'])

    # Timestamp: Create from Date + 15:45:00 (EOD proxy)
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['date']) + pd.Timedelta(hours=15, minutes=45)
    
    # Option Symbol: Remove spaces from IVol format "SPY   25..." -> "SPY25..."
    if 'option_symbol' in df.columns:
        df['option_symbol'] = df['option_symbol'].str.replace(' ', '')

    # Fill Missing Columns
    if 'bid_size' not in df.columns:
        df['bid_size'] = 100
    if 'ask_size' not in df.columns:
        df['ask_size'] = 100
        
    # Ensure all required columns exist (fill with 0 if missing)
    required_cols = [
        'timestamp', 'date', 'symbol', 'option_symbol', 'strike', 'expiration', 'contract_type',
        'bid', 'ask', 'last_price', 'bid_size', 'ask_size', 'volume', 'open_interest',
        'delta', 'gamma', 'theta', 'vega', 'implied_volatility', 'underlying_last'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Missing column '{col}', filling with 0.")
            df[col] = 0

    # Select and Reorder
    final_df = df[required_cols]

    # 4. Save
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    process_ivol_data()
