#!/usr/bin/env python3
"""
Complete 2025 Options Dataset - Add Missing Timestamps (OPTIMIZED BATCH VERSION)

This script:
1. Downloads SPY 1-minute OHLCV data for 2025 from Alpaca
2. Identifies timestamps in 2024 that are missing from 2025
3. Generates synthetic options for missing timestamps using FULL BATCH VECTORIZATION
4. Produces a complete 2025 dataset that matches 2024's structure

Usage:
    python complete_2025_dataset.py [--dry-run]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
from scipy.stats import norm
import time as time_module

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Constants
TARGET_PUTS = 50
TARGET_CALLS = 50
OPTIONS_PER_BAR = TARGET_PUTS + TARGET_CALLS
RISK_FREE_RATE = 0.05
NY = pytz.timezone("America/New_York")

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
SPOT_DIR = os.path.join(PROJECT_ROOT, 'data', 'spot')


# ============================================================================
# Black-Scholes Functions (vectorized)
# ============================================================================

def bs_d1_vec(S, K, T, r, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d1 = np.where((T <= 0) | (sigma <= 0) | (S <= 0) | (K <= 0), 0.0, d1)
    return d1

def bs_d2_vec(S, K, T, r, sigma):
    return bs_d1_vec(S, K, T, r, sigma) - sigma * np.sqrt(np.maximum(T, 0.001))

def bs_call_price_vec(S, K, T, r, sigma):
    d1 = bs_d1_vec(S, K, T, r, sigma)
    d2 = bs_d2_vec(S, K, T, r, sigma)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(T <= 0, np.maximum(0, S - K), price)

def bs_put_price_vec(S, K, T, r, sigma):
    d1 = bs_d1_vec(S, K, T, r, sigma)
    d2 = bs_d2_vec(S, K, T, r, sigma)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(T <= 0, np.maximum(0, K - S), price)

def bs_delta_vec(S, K, T, r, sigma, is_call):
    d1 = bs_d1_vec(S, K, T, r, sigma)
    return np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)

def bs_gamma_vec(S, K, T, r, sigma):
    d1 = bs_d1_vec(S, K, T, r, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        gamma = np.where((T <= 0) | (sigma <= 0) | (S <= 0), 0.0, gamma)
    return gamma

def bs_vega_vec(S, K, T, r, sigma):
    d1 = bs_d1_vec(S, K, T, r, sigma)
    vega = S * norm.pdf(d1) * np.sqrt(np.maximum(T, 0)) / 100
    return np.where((T <= 0) | (sigma <= 0) | (S <= 0), 0.0, vega)

def bs_theta_vec(S, K, T, r, sigma, is_call):
    d1 = bs_d1_vec(S, K, T, r, sigma)
    d2 = bs_d2_vec(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(np.maximum(T, 0.001)))
    call_term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    put_term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = (term1 + np.where(is_call, call_term2, put_term2)) / 365
    return np.where((T <= 0) | (sigma <= 0) | (S <= 0), 0.0, theta)

def bs_rho_vec(S, K, T, r, sigma, is_call):
    d2 = bs_d2_vec(S, K, T, r, sigma)
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return np.where(is_call, call_rho, put_rho)


# ============================================================================
# Data Download
# ============================================================================

def download_spy_ohlcv_2025():
    """Download SPY 1-minute bars for 2025 from Alpaca."""
    print("Downloading SPY 1-minute OHLCV data for 2025...", flush=True)
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from core.config import RunConfig
        
        config = RunConfig()
        client = StockHistoricalDataClient(config.alpaca_key, config.alpaca_secret)
        
        # Alpaca requires 15-min delay from current time
        end_time = datetime.now(pytz.utc) - timedelta(minutes=16)
        start_time = datetime(2025, 1, 2, tzinfo=pytz.utc)
        
        # Limit end time to not exceed current date if still in 2025
        if end_time > datetime(2025, 12, 31, 23, 59, tzinfo=pytz.utc):
            end_time = datetime(2025, 12, 31, 16, 0, tzinfo=pytz.utc)
        
        print(f"  Date range: {start_time.date()} to {end_time.date()}", flush=True)
        
        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols='SPY',
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time,
            adjustment='split'
        ))
        
        df = bars.df.reset_index()
        print(f"  Downloaded {len(df):,} raw bars", flush=True)
        
        # Filter to RTH only (9:30 AM - 4:00 PM ET)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['ts_ny'] = df['timestamp'].dt.tz_convert(NY)
        df = df[df['ts_ny'].dt.dayofweek < 5]  # Weekdays only
        
        t = df['ts_ny'].dt.time
        df = df[(t >= time(9, 30)) & (t <= time(16, 0))]
        
        df = df.drop(columns=['ts_ny'])
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        print(f"  After RTH filter: {len(df):,} bars", flush=True)
        
        # Save to spot directory
        os.makedirs(SPOT_DIR, exist_ok=True)
        out_path = os.path.join(SPOT_DIR, 'SPY_1_2025.csv')
        df.to_csv(out_path, index=False)
        print(f"  Saved to: {out_path}", flush=True)
        
        return df
        
    except Exception as e:
        print(f"  ERROR downloading from Alpaca: {e}", flush=True)
        # Check if file exists as fallback
        out_path = os.path.join(SPOT_DIR, 'SPY_1_2025.csv')
        if os.path.exists(out_path):
            print(f"  Falling back to existing file: {out_path}", flush=True)
            return pd.read_csv(out_path, parse_dates=['timestamp'])
        return None


def load_or_download_spy_ohlcv():
    """Load existing SPY OHLCV or download if not present."""
    ohlcv_path = os.path.join(SPOT_DIR, 'SPY_1_2025.csv')
    
    if os.path.exists(ohlcv_path):
        print(f"Loading existing SPY OHLCV from {ohlcv_path}...", flush=True)
        df = pd.read_csv(ohlcv_path, parse_dates=['timestamp'])
        print(f"  Loaded {len(df):,} bars", flush=True)
        return df
    else:
        return download_spy_ohlcv_2025()


# ============================================================================
# Generate Synthetic Options (BATCH VECTORIZED)
# ============================================================================

def generate_options_batch(timestamps, spots, ohlcv_data_dict, avg_iv=0.20):
    """
    Generate options for thousands of timestamps at once using full vectorization.
    """
    n_ts = len(timestamps)
    if n_ts == 0:
        return pd.DataFrame()
    
    # 1. Expand timestamps to 100 rows each
    ts_expanded = np.repeat(timestamps, OPTIONS_PER_BAR)
    spot_expanded = np.repeat(spots, OPTIONS_PER_BAR)
    
    # 2. Generate strike grid for each timestamp
    # Base pattern for one timestamp: 50 puts (ATM-49 to ATM), 50 calls (ATM to ATM+49)
    atm_rounded = np.round(spots).astype(int)
    
    # Create strike offsets: [-49, -48, ..., 0, 0, 1, ..., 49]
    # Actually simpler: puts are ATM-49..ATM, calls are ATM..ATM+49
    strike_offsets = np.concatenate([np.arange(-49, 1), np.arange(0, 50)])
    is_call_pattern = np.concatenate([np.zeros(50, dtype=bool), np.ones(50, dtype=bool)])
    
    # Broadcast to all timestamps
    all_strikes = np.repeat(atm_rounded, OPTIONS_PER_BAR) + np.tile(strike_offsets, n_ts)
    is_calls = np.tile(is_call_pattern, n_ts)
    
    # 3. Handle Expirations (30 DTE fixed for simplicity/consistency)
    ts_dt = pd.to_datetime(ts_expanded)
    exp_dates = ts_dt + pd.Timedelta(days=30)
    exp_dates_str = exp_dates.strftime('%Y-%m-%d')
    exp_sym_str = exp_dates.strftime('%y%m%d')
    
    # 4. Black-Scholes Inputs
    n_total = len(ts_expanded)
    T = np.full(n_total, 30 / 365.0)
    S = spot_expanded
    K = all_strikes
    sigma = np.full(n_total, avg_iv)
    r = RISK_FREE_RATE
    
    # 5. Vectorized Calculations
    call_prices = bs_call_price_vec(S, K, T, r, sigma)
    put_prices = bs_put_price_vec(S, K, T, r, sigma)
    prices = np.where(is_calls, call_prices, put_prices)
    
    deltas = bs_delta_vec(S, K, T, r, sigma, is_calls)
    gammas = bs_gamma_vec(S, K, T, r, sigma)
    thetas = bs_theta_vec(S, K, T, r, sigma, is_calls)
    vegas = bs_vega_vec(S, K, T, r, sigma)
    rhos = bs_rho_vec(S, K, T, r, sigma, is_calls)
    
    # 6. Option Symbols
    # Vectorized symbol construction
    cp_arr = np.where(is_calls, 'C', 'P')
    # Using list comprehension for strings as numpy string ops are messy
    strike_str_arr = [f"{int(s * 1000):08d}" for s in K]
    option_symbols = [f"SPY   {exp}{cp}{s_str}" for exp, cp, s_str in zip(exp_sym_str, cp_arr, strike_str_arr)]
    
    # 7. Collect OHLCV data
    # Create arrays for OHLCV columns
    opens = np.repeat([ohlcv_data_dict[ts]['open'] for ts in timestamps], OPTIONS_PER_BAR)
    highs = np.repeat([ohlcv_data_dict[ts]['high'] for ts in timestamps], OPTIONS_PER_BAR)
    lows = np.repeat([ohlcv_data_dict[ts]['low'] for ts in timestamps], OPTIONS_PER_BAR)
    closes = np.repeat([ohlcv_data_dict[ts]['close'] for ts in timestamps], OPTIONS_PER_BAR)
    
    # 8. Final DataFrame
    df = pd.DataFrame({
        'timestamp': [str(t) for t in ts_expanded],
        'symbol': 'SPY',
        'underlying_price': spot_expanded,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'option_symbol': option_symbols,
        'expiration': exp_dates_str,
        'strike': K,
        'call_put': cp_arr,
        'bid': np.round(prices * 0.98, 2),
        'ask': np.round(prices * 1.02, 2),
        'iv': np.round(sigma, 4),
        'delta': np.round(deltas, 4),
        'gamma': np.round(gammas, 4),
        'theta': np.round(thetas, 4),
        'vega': np.round(vegas, 4),
        'rho': np.round(rhos, 4),
        'volume': 0,
        'open_interest': 0
    })
    
    return df


# ============================================================================
# Main Completion Logic
# ============================================================================

def complete_2025_dataset(dry_run=False):
    """
    Complete the 2025 options dataset by adding missing timestamps.
    """
    print("=" * 60, flush=True)
    print("COMPLETE 2025 OPTIONS DATASET (OPTIMIZED)", flush=True)
    print("=" * 60, flush=True)
    
    # Step 1: Reference Timestamps from 2024
    print("\nStep 1: Extracting reference timestamps from 2024...", flush=True)
    path_2024 = os.path.join(DATA_DIR, 'Spy_Options_2024_1m.csv')
    
    timestamps_2024 = set()
    for chunk in pd.read_csv(path_2024, chunksize=1000000, usecols=['timestamp']):
        chunk['ts_norm'] = pd.to_datetime(chunk['timestamp']).dt.tz_localize(None).dt.floor('min')
        timestamps_2024.update(chunk['ts_norm'].unique())
    
    print(f"  2024 has {len(timestamps_2024):,} unique timestamps", flush=True)
    
    # Step 2: Existing Timestamps in 2025
    print("\nStep 2: Loading existing 2025 cleaned timestamps...", flush=True)
    path_2025_cleaned = os.path.join(DATA_DIR, 'Spy_Options_2025_1m_cleaned.csv')
    
    timestamps_2025 = set()
    for chunk in pd.read_csv(path_2025_cleaned, chunksize=1000000, usecols=['timestamp']):
        chunk['ts_norm'] = pd.to_datetime(chunk['timestamp']).dt.tz_localize(None).dt.floor('min')
        timestamps_2025.update(chunk['ts_norm'].unique())
    
    print(f"  2025 (current) has {len(timestamps_2025):,} unique timestamps", flush=True)
    
    # Shift 2024 to 2025 for comparison
    timestamps_ref = set()
    for ts in timestamps_2024:
        try:
            ts_ref = ts.replace(year=2025)
            timestamps_ref.add(ts_ref)
        except ValueError:
            pass # Feb 29
            
    missing_timestamps_set = timestamps_ref - timestamps_2025
    print(f"  Missing timestamps to add: {len(missing_timestamps_set):,}", flush=True)
    
    if len(missing_timestamps_set) == 0:
        print("\n✅ Dataset already complete!", flush=True)
        return

    # Step 3: Load/Get OHLCV
    print("\nStep 3: Loading SPY OHLCV for 2025...", flush=True)
    ohlcv_df = load_or_download_spy_ohlcv()
    if ohlcv_df is None:
        print("ERROR: No OHLCV data available.", flush=True)
        return
        
    ohlcv_df['ts_norm'] = pd.to_datetime(ohlcv_df['timestamp']).dt.tz_localize(None).dt.floor('min')
    ohlcv_lookup = ohlcv_df.set_index('ts_norm').to_dict('index')
    
    # Step 4: Prepare Batch Data
    if dry_run:
        print("\n[Dry Run] Analysis complete. Would add {len(missing_timestamps_set):,} timestamps.", flush=True)
        return

    missing_sorted = sorted(list(missing_timestamps_set))
    
    # Pre-calculate spots for ALL missing timestamps using fast indexing
    print("  Pre-calculating spots for missing timestamps...", flush=True)
    full_ohlcv = ohlcv_df.set_index('ts_norm').sort_index()
    # Reindex missing timestamps onto the OHLCV data to get nearest spots
    missing_ohlcv = full_ohlcv.reindex(missing_sorted, method='nearest', tolerance=pd.Timedelta(hours=1))
    
    output_path = os.path.join(DATA_DIR, 'Spy_Options_2025_1m_missing.csv')
    
    print(f"\nStep 5: Generating missing options in batches...", flush=True)
    
    batch_size_ts = 1000 # More frequent updates
    start_time = time_module.time()
    total_added = 0
    
    first_write = True
    
    for i in range(0, len(missing_sorted), batch_size_ts):
        batch_ts = missing_sorted[i:i+batch_size_ts]
        batch_ohlcv_rows = missing_ohlcv.loc[batch_ts]
        
        batch_spots = batch_ohlcv_rows['close'].fillna(590).values
        
        # Prepare ohlcv_data_dict for generate_options_batch
        # We can optimize generate_options_batch to take the dataframe directly
        batch_ohlcv_dict = batch_ohlcv_rows.to_dict('index')
        
        # Generate!
        gen_df = generate_options_batch(batch_ts, batch_spots, batch_ohlcv_dict)
        
        if not gen_df.empty:
            if first_write:
                gen_df.to_csv(output_path, index=False, mode='w')
                first_write = False
            else:
                gen_df.to_csv(output_path, index=False, mode='a', header=False)
            
            total_added += len(batch_ts)
            
        elapsed = time_module.time() - start_time
        print(f"  ✓ {total_added:,}/{len(missing_sorted):,} timestamps generated. Time: {elapsed:.1f}s", flush=True)

    # Step 5: Merge
    print("\nStep 5: Final merge into 'Spy_Options_2025_1m_complete.csv'...", flush=True)
    final_output = os.path.join(DATA_DIR, 'Spy_Options_2025_1m_complete.csv')
    
    import shutil
    print(f"  Copying initial 2025 cleaned data...", flush=True)
    shutil.copy(path_2025_cleaned, final_output)
    
    print(f"  Appending missing data chunks...", flush=True)
    # Using read_csv/to_csv in chunks for memory safety
    for chunk in pd.read_csv(output_path, chunksize=1000000):
        chunk.to_csv(final_output, index=False, mode='a', header=False)
        
    print(f"\n{'='*60}", flush=True)
    print("COMPLETION SUCCESSFUL", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Added timestamps: {total_added:,}", flush=True)
    print(f"Final file:       {final_output}", flush=True)
    print(f"Total rows approx: {total_added*100 + 68826*100:,}", flush=True)
    print(f"Total time:       {(time_module.time()-start_time)/60:.1f} minutes", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Complete 2025 Options Dataset (FAST)')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only')
    args = parser.parse_args()
    complete_2025_dataset(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
