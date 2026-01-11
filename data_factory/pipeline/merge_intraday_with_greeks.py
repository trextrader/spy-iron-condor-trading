#!/usr/bin/env python3
"""
Merge Alpaca intraday options OHLCV with IVolatility daily Greeks.
Interpolates Greeks throughout the trading day for M1/M5/M15 backtesting.
"""
import datetime as dt
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_ivolatility_greeks(file_path: str) -> pd.DataFrame:
    """Load IVolatility EOD data with Greeks."""
    df = pd.read_csv(file_path)
    
    # Standardize column names
    rename_map = {
        'date': 'greeks_date',
        'option symbol': 'ivol_symbol',
        'strike': 'greeks_strike',
        'Call/Put': 'greeks_type',
        'expiration': 'greeks_expiration',
        'iv': 'iv',
        'delta': 'delta',
        'gamma': 'gamma', 
        'theta': 'theta',
        'vega': 'vega',
        'rho': 'rho'
    }
    
    # Only rename columns that exist
    for old, new in rename_map.items():
        if old in df.columns and old != new:
            df = df.rename(columns={old: new})
    
    # Parse dates
    if 'greeks_date' in df.columns:
        # Filter out repeated headers
        df = df[df['greeks_date'] != 'date'].copy()
        df['greeks_date'] = pd.to_datetime(df['greeks_date']).dt.date
    
    # Force numeric conversion for Greeks
    greek_cols = ['iv', 'delta', 'gamma', 'theta', 'vega', 'rho']
    for col in greek_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    print(f"Loaded Greeks: {len(df)} records")
    if 'ivol_symbol' in df.columns:
        print(f"Sample Greek symbols: {df['ivol_symbol'].head().tolist()}")
    elif 'option_symbol' in df.columns:
        print(f"Sample Greek symbols: {df['option_symbol'].head().tolist()}")
    else:
        print(f"Sample Greek symbols: {df['option symbol'].head().tolist()}")
    
    return df


def load_alpaca_intraday(file_path: str) -> pd.DataFrame:
    """Load Alpaca M1 intraday data."""
    df = pd.read_csv(file_path)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    df['minute_of_day'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    
    return df


def create_matching_key(row) -> str:
    """Create a key to match Alpaca options with IVolatility Greeks."""
    # Alpaca uses OCC format: SPY251017C00580000
    # IVolatility uses: SPY   251017C00580000
    # Normalize both
    symbol = str(row.get('symbol', '')).replace(' ', '').strip()
    return symbol


def interpolate_greeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate daily Greeks for intraday bars using vectorized operations.
    
    Adjustments:
    - Theta: Decays linearly through the day (more time decay by EOD)
    - Delta/Gamma: Adjusted by underlying price change from open
    - IV: Held constant (daily snapshot)
    - Vega/Rho: Held constant
    """
    # Trading day parameters
    market_open_minute = 9 * 60 + 30  # 9:30 AM
    market_close_minute = 16 * 60      # 4:00 PM
    trading_minutes = market_close_minute - market_open_minute  # 390 minutes
    
    # === Theta Interpolation (Vectorized) ===
    # Fraction of day elapsed (clipped 0-1)
    minute_vals = df['minute_of_day'].fillna(market_open_minute)
    day_fraction = ((minute_vals - market_open_minute) / trading_minutes).clip(0, 1)
    
    # Apply fraction to daily theta
    df['theta_intraday'] = df['theta'] * day_fraction
    
    # === Delta Interpolation (Vectorized) ===
    # Price change from open (using bar close - bar open for simplicity, or ideally underlying price change)
    # Using bar moves is a proxy for underlying moves
    price_change = df['close'].fillna(0) - df['open'].fillna(0)
    
    # Calculate adjusted delta: δnew = δold + γ * ΔS
    # Use fillna(0) for missing gamma/delta to avoid NaN propagation where data exists
    params_mask = df['delta'].notna() & df['gamma'].notna()
    
    # Initialize with original delta
    df['delta_intraday'] = df['delta']
    
    # Apply adjustment where possible
    adj_delta = df.loc[params_mask, 'delta'] + df.loc[params_mask, 'gamma'] * price_change.loc[params_mask]
    
    # Clamp results based on option type
    is_call = df['option_type'] == 'call'
    
    # Vectorized clamping
    # Calls: 0 to 1
    # Puts: -1 to 0
    
    # We need to handle the clamping carefully.
    # Create temporary series for clamping
    final_delta = pd.Series(index=df.index, dtype='float64')
    final_delta.loc[params_mask] = adj_delta
    
    # Fill missing with original
    final_delta = final_delta.fillna(df['delta'])
    
    # Clamp calls
    call_mask = params_mask & is_call
    final_delta.loc[call_mask] = final_delta.loc[call_mask].clip(0, 1)
    
    # Clamp puts
    put_mask = params_mask & (~is_call)
    final_delta.loc[put_mask] = final_delta.loc[put_mask].clip(-1, 0)
    
    df['delta_intraday'] = final_delta
    
    # === Constant Greeks ===
    df['iv_intraday'] = df.get('iv', df.get('iv raw', None))
    df['gamma_intraday'] = df.get('gamma', None)
    df['vega_intraday'] = df.get('vega', None)
    df['rho_intraday'] = df.get('rho', None)
    
    return df



def merge_intraday_with_greeks(intraday_df: pd.DataFrame, 
                                greeks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Alpaca intraday bars with IVolatility daily Greeks.
    Match by: symbol (normalized), date, strike, option type.
    """
    print(f"Intraday bars: {len(intraday_df):,}")
    print(f"Greeks records: {len(greeks_df):,}")
    
    # Normalize Alpaca symbol for matching
    intraday_df['symbol_norm'] = intraday_df['symbol'].str.replace(' ', '').str.strip()
    
    # Normalize IVolatility symbol
    if 'ivol_symbol' in greeks_df.columns:
        greeks_df['symbol_norm'] = greeks_df['ivol_symbol'].str.replace(' ', '').str.strip()
    elif 'option symbol' in greeks_df.columns:
        greeks_df['symbol_norm'] = greeks_df['option symbol'].str.replace(' ', '').str.strip()
    elif 'option_symbol' in greeks_df.columns:
        greeks_df['symbol_norm'] = greeks_df['option_symbol'].str.replace(' ', '').str.strip()
    
    # Prepare merge keys
    intraday_df['merge_date'] = intraday_df['date']
    greeks_df['merge_date'] = greeks_df.get('greeks_date', greeks_df.get('date', None))
    
    # Merge on symbol and date
    merged = pd.merge(
        intraday_df,
        greeks_df[['symbol_norm', 'merge_date', 'iv', 'delta', 'gamma', 'theta', 'vega', 'rho']].drop_duplicates(),
        on=['symbol_norm', 'merge_date'],
        how='left'
    )
    
    matched = merged['delta'].notna().sum()
    print(f"Matched with Greeks: {matched:,} / {len(merged):,} ({100*matched/len(merged):.1f}%)")
    
    # Interpolate Greeks intraday
    merged = interpolate_greeks(merged)
    
    return merged


from data_factory.resample_intraday import resample_m1_df

def main():
    """Main merge routine."""
    print("=" * 70)
    print("Merge Intraday Options with Daily Greeks (Multi-Timeframe)")
    print("=" * 70)
    
    # Paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    alpaca_dir = os.path.join(data_dir, 'alpaca_options')
    ivol_dir = os.path.join(data_dir, 'ivolatility')
    
    # Point to LARGE production files
    intraday_file = os.path.join(alpaca_dir, 'spy_options_intraday_large_m1.csv')
    greeks_file = os.path.join(ivol_dir, 'spy_options_ivol_large.csv')
    
    output_file_m1 = os.path.join(alpaca_dir, 'spy_options_intraday_large_with_greeks_m1.csv')
    output_file_m5 = os.path.join(alpaca_dir, 'spy_options_intraday_large_with_greeks_m5.csv')
    output_file_m15 = os.path.join(alpaca_dir, 'spy_options_intraday_large_with_greeks_m15.csv')
    
    # Check if inputs exist
    # Note: For verification/testing, we might use the small file if large not exists?
    # User requested "large" pipeline.
    if not os.path.exists(intraday_file):
        print(f"Error: Large Intraday file not found: {intraday_file}")
        print("Run download_alpaca_matched.py (configured for large) first.")
        # Fallback to small for testing? No, stay strict to Phase 5.
        return
        
    if not os.path.exists(greeks_file):
        print(f"Error: Large Greeks file not found: {greeks_file}")
        print("Run download_ivolatility_options.py (configured for large) first.")
        return
    
    print(f"\nIntraday file: {intraday_file}")
    print(f"Greeks file: {greeks_file}")
    
    # [Step 1] Load data
    print("\n[Step 1] Loading data...")
    greeks_df = load_ivolatility_greeks(greeks_file)
    
    # Load M1 bars 
    print("Loading M1 bars...")
    # Chunking TODO for >2GB. For "Top 100", 100*390*130 = ~5M lines. Fits in RAM.
    intraday_df = load_alpaca_intraday(intraday_file)
    
    # [Step 2] Merge
    print("\n[Step 2] Merging intraday with Greeks...")
    merged_df = merge_intraday_with_greeks(intraday_df, greeks_df)
    
    # [Step 3] Save M1
    print("\n[Step 3] Saving M1 merged data...")
    merged_df.to_csv(output_file_m1, index=False)
    print(f"Saved M1 to {output_file_m1} ({len(merged_df):,} bars)")
    
    # [Step 4] Resample M5
    print("\n[Step 4] Resampling to M5...")
    df_m5 = resample_m1_df(merged_df, timeframe='5min')
    if not df_m5.empty:
        df_m5.to_csv(output_file_m5, index=False)
        print(f"Saved M5 to {output_file_m5} ({len(df_m5):,} bars)")
    
    # [Step 5] Resample M15
    print("\n[Step 5] Resampling to M15...")
    df_m15 = resample_m1_df(merged_df, timeframe='15min')
    if not df_m15.empty:
        df_m15.to_csv(output_file_m15, index=False)
        print(f"Saved M15 to {output_file_m15} ({len(df_m15):,} bars)")
    
    print("\n" + "=" * 70)
    print("Merge & Resample Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
