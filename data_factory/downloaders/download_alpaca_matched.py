#!/usr/bin/env python3
"""
Download Alpaca M1 bars for SPECIFIC options found in the IVolatility dataset.
Ensures 100% overlap for backtesting.
"""
import datetime as dt
import pandas as pd
import os
import sys
import time
from typing import List

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.config import RunConfig


def load_ivol_targets(ivol_file: str, min_date: dt.date) -> dict:
    """
    Load IVolatility data and return a dict of {date: [symbols]} 
    for dates >= min_date.
    """
    print(f"Loading targets from {ivol_file}...")
    df = pd.read_csv(ivol_file)
    
    # Handle column names
    # Detect date column
    if 'date' in df.columns:
        date_col = 'date'
    elif 'greeks_date' in df.columns:
        date_col = 'greeks_date'
    elif 'trade_date' in df.columns:
        date_col = 'trade_date'
    else:
        print(f"Available columns: {df.columns.tolist()}")
        raise KeyError("No date column found in IVolatility file")
    
    if 'option symbol' in df.columns:
        sym_col = 'option symbol'
    elif 'option_symbol' in df.columns:
        sym_col = 'option_symbol'
    else:
        sym_col = 'ivol_symbol'
    
    # Filter repeated headers (if date column contains string 'date')
    df = df[df[date_col].astype(str) != date_col].copy()
    
    # Parse date
    df['dt'] = pd.to_datetime(df[date_col]).dt.date
    
    # Filter by date
    df = df[df['dt'] >= min_date]
    
    # helper to normalize symbol
    def norm(s):
        return str(s).replace(' ', '').strip()
    
    df['norm_sym'] = df[sym_col].apply(norm)
    
    targets = {}
    unique_dates = sorted(df['dt'].unique())
    print(f"Found {len(unique_dates)} dates matching criteria: {unique_dates}")
    
    for d in unique_dates:
        syms = df[df['dt'] == d]['norm_sym'].unique().tolist()
        targets[d] = syms
        print(f"  {d}: {len(syms)} symbols")
        
    return targets


def parse_occ_symbol(sym: str) -> dict:
    """Parse OCC symbol like SPY251017C00580000"""
    try:
        date_str = sym[3:9]
        opt_type = sym[9]
        strike_str = sym[10:]
        
        expiration = dt.datetime.strptime(date_str, "%y%m%d").date()
        strike = float(strike_str) / 1000
        is_call = (opt_type == 'C')
        
        return {
            'underlying': 'SPY',
            'expiration': expiration,
            'strike': strike,
            'option_type': 'call' if is_call else 'put'
        }
    except:
        return {
            'underlying': None,
            'expiration': None,
            'strike': None,
            'option_type': None
        }


def download_bars_for_day(client: OptionHistoricalDataClient, 
                          date: dt.date, 
                          symbols: List[str],
                          output_file: str) -> int:
    """Download M1 bars for a specific list of symbols on a specific day."""
    
    # Alpaca request requires start/end as datetime
    start_dt = dt.datetime.combine(date, dt.time(9, 30))
    end_dt = dt.datetime.combine(date, dt.time(16, 0))
    
    total_bars = 0
    batch_size = 50  # Reasonable batch size
    
    # Check if file exists to determine header
    file_exists = os.path.exists(output_file)
    write_header = not file_exists
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            req = OptionBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,
                start=start_dt,
                end=end_dt
            )
            bars = client.get_option_bars(req)
            df = bars.df
            
            if len(df) > 0:
                df = df.reset_index()
                
                # Parse symbol components
                parsed = df['symbol'].apply(lambda s: pd.Series(parse_occ_symbol(s)))
                df = pd.concat([df, parsed], axis=1)
                
                # Add date/time columns
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                df['time'] = pd.to_datetime(df['timestamp']).dt.time
                
                # Append
                df.to_csv(output_file, mode='a', header=write_header, index=False)
                write_header = False  # Only first write needs header
                
                total_bars += len(df)
                print(f"    Fetched {len(df)} bars for batch {i//batch_size + 1}", flush=True)
            
        except Exception as e:
            print(f"    Error batch {i}: {str(e)[:50]}", flush=True)
            
        time.sleep(0.1)
        
    return total_bars


def main():
    print("=" * 70)
    print("Alpaca Matched Options Downloader")
    print("Syncing regular M1 bars for IVolatility Greeks targets")
    print("=" * 70)
    
    # Config
    r = RunConfig()
    client = OptionHistoricalDataClient(r.alpaca_key, r.alpaca_secret)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    ivol_file = os.path.join(data_dir, 'ivolatility', 'spy_options_ivol_large.csv') # New Input
    alpaca_dir = os.path.join(data_dir, 'alpaca_options')
    output_file = os.path.join(alpaca_dir, 'spy_options_intraday_large_m1.csv') # New Output
    
    # 1. Load targets (All dates in IVolatility file)
    start_date = dt.date(2025, 6, 1) 
    targets = load_ivol_targets(ivol_file, start_date)
    
    if not targets:
        print("No targets found! Check date range.")
        return

    # Check for existing progress
    completed_dates = set()
    if os.path.exists(output_file):
        print(f"Checking existing file for progress: {output_file}")
        try:
            # Read just the date column to find completed days
            # Using usecols for speed
            existing_df = pd.read_csv(output_file, usecols=['date'])
            # Convert to date objects
            completed_dates = set(pd.to_datetime(existing_df['date']).dt.date.unique())
            print(f"Found {len(completed_dates)} already completed dates. Skipping them.")
        except Exception as e:
            print(f"Warning: Could not read existing file ({e}). Starting fresh.")
    
    # 2. Download
    total_all = 0
    sorted_days = sorted(targets.keys())
    
    for day in sorted_days:
        symbols = targets[day]
        
        if day in completed_dates:
            print(f"Skipping {day} (already done)")
            continue
            
        print(f"\nProcessing {day} ({len(symbols)} symbols)...")
        bars_count = download_bars_for_day(client, day, symbols, output_file)
        total_all += bars_count
        print(f"  -> Total bars for {day}: {bars_count}")
        
    print("\n" + "=" * 70)
    print(f"Done! Total M1 bars: {total_all:,}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
