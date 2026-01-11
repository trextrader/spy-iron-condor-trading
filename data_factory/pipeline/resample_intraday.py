import pandas as pd
import os

def resample_m1_df(df: pd.DataFrame, timeframe='5min') -> pd.DataFrame:
    """
    Resample M1 options DataFrame to target timeframe.
    Expects 'timestamp' column and proper numeric types.
    """
    if df.empty:
        return pd.DataFrame()
        
    if 'timestamp' not in df.columns:
        # Check index
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Error: 'timestamp' col missing and index not datetime.")
            return pd.DataFrame()
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Define aggregation rules
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Include other columns if present
    optional_cols = ['open_interest', 'vwap']
    for col in optional_cols:
        if col in df.columns:
            agg_rules[col] = 'last'
            
    # Include Greeks (take last known value for the interval)
    greek_cols = ['delta_intraday', 'gamma_intraday', 'theta_intraday', 'vega_intraday', 'rho_intraday', 'iv_intraday']
    for col in greek_cols:
        if col in df.columns:
            agg_rules[col] = 'last' # Snapshot at close of bar
            
    # Static columns (take first)
    static_cols = ['symbol', 'expiration', 'strike', 'option_type']
    # Careful: 'symbol' is the grouper usually.
    
    # Process per symbol to ensure correct boundaries
    # (Resampling a huge mixed-symbol DF directly is tricky with gaps)
    # But grouping by [symbol, pd.Grouper(key='timestamp', freq=timeframe)] works well.
    
    grouped = df.groupby(['symbol', pd.Grouper(key='timestamp', freq=timeframe)])
    
    resampled = grouped.agg(agg_rules)
    
    # Drop empty bins (where no M1 bars existed)
    resampled = resampled.dropna(subset=['close'])
    
    resampled = resampled.reset_index()
    
    # Restore static columns if lost (groupby 'symbol' puts keys in index)
    # The 'symbol' is in the index (level 0).
    # agg_rules didn't include it.
    # We verify static data
    
    return resampled

def main():
    base_dir = "data/alpaca_options"
    m1_file = os.path.join(base_dir, "spy_options_intraday_large_m1.csv") # Or merged file?
    # User said "alpaca will be the full m1... and also for m5 and m15"
    # Usually we resample the MERGED file (with Greeks) or just the bars?
    # If we resample matches, we want Greeks.
    # If we simply resample bars, we assume merge happens later.
    # User wants "data file" -> implies final product.
    
    # Let's resample the MERGED file if available, otherwise the raw file.
    # Currently pipeline produces: spy_options_intraday_large_m1.csv (Raw) -> Merged
    
    # Better to resample the MERGED file? 
    # But wait, Greeks are interpolated for M1.
    # If I resample M1 Greeks (which are interpolated M1), I get 'Last'.
    # This roughly works. M5 Close Greek = Greek at minute 5.
    
    merged_file = os.path.join(base_dir, "spy_options_intraday_with_greeks.csv")
    # But for "Phase 5", we create NEW files.
    # "spy_options_intraday_large_with_greeks.csv"
    
    # I'll enable this script to run on the 'Merged' output.
    pass

if __name__ == "__main__":
    main()
