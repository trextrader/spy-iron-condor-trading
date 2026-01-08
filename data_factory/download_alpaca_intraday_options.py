#!/usr/bin/env python3
"""
Download intraday (M1) SPY options data from Alpaca for backtesting.
Fetches 1-minute bars to support M1/M5/M15 multi-timeframe analysis.
"""
import datetime as dt
import pandas as pd
import os
import sys
import time
from typing import List

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import RunConfig


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


def download_intraday_bars(client: OptionHistoricalDataClient,
                            symbols: List[str],
                            start_date: dt.datetime,
                            end_date: dt.datetime,
                            output_file: str) -> int:
    """
    Download 1-minute bars for option symbols, writing incrementally to CSV.
    Returns total number of bars downloaded.
    """
    total_bars = 0
    first_write = True
    
    # Process in batches
    batch_size = 20  # Smaller batches for intraday data (more records per symbol)
    total_batches = (len(symbols) // batch_size) + 1
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} symbols)...", end=" ", flush=True)
        
        try:
            req = OptionBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,  # 1-minute bars
                start=start_date,
                end=end_date
            )
            bars = client.get_option_bars(req)
            
            df = bars.df
            if len(df) > 0:
                df = df.reset_index()
                
                # Parse symbol components
                parsed = df['symbol'].apply(lambda s: pd.Series(parse_occ_symbol(s)))
                df = pd.concat([df, parsed], axis=1)
                
                # Add date column
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                df['time'] = pd.to_datetime(df['timestamp']).dt.time
                
                # Append to CSV
                df.to_csv(output_file, mode='a', header=first_write, index=False)
                first_write = False
                
                total_bars += len(df)
                print(f"{len(df):,} bars (total: {total_bars:,})", flush=True)
            else:
                print("no data", flush=True)
                
        except Exception as e:
            print(f"error: {str(e)[:60]}", flush=True)
            continue
        
        time.sleep(0.2)  # Rate limiting
    
    return total_bars


def filter_options_by_dte(symbols: List[str], min_dte: int = 30, max_dte: int = 60) -> List[str]:
    """Filter option symbols by days to expiration."""
    today = dt.date.today()
    filtered = []
    
    for sym in symbols:
        parsed = parse_occ_symbol(sym)
        if parsed['expiration']:
            dte = (parsed['expiration'] - today).days
            if min_dte <= dte <= max_dte:
                filtered.append(sym)
    
    return filtered


def main():
    """Main download routine for intraday options data"""
    print("=" * 70)
    print("Alpaca Intraday (M1) Options Data Downloader")
    print("=" * 70)
    
    # Initialize client
    r = RunConfig()
    client = OptionHistoricalDataClient(r.alpaca_key, r.alpaca_secret)
    
    # Date range: 1 week of intraday data (to start - very data intensive)
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=7)
    
    print(f"\nDate range: {start_date.date()} to {end_date.date()}")
    print("Timeframe: 1-minute bars (M1)")
    
    # Get current option symbols
    print("\n[Step 1] Getting available option symbols...")
    try:
        req = OptionChainRequest(underlying_symbol='SPY')
        chain = client.get_option_chain(req)
        all_symbols = list(chain.keys())
        print(f"Found {len(all_symbols)} total option contracts")
    except Exception as e:
        print(f"Error getting chain: {e}")
        return
    
    # Filter to 30-60 DTE (iron condor range)
    print("\n[Step 2] Filtering to 30-60 DTE options...")
    symbols = filter_options_by_dte(all_symbols, min_dte=30, max_dte=60)
    print(f"Filtered to {len(symbols)} contracts with 30-60 DTE")
    
    if len(symbols) == 0:
        print("No options found in DTE range!")
        return
    
    # Limit to prevent API overload (intraday = many records)
    max_symbols = 200
    if len(symbols) > max_symbols:
        print(f"Limiting to {max_symbols} symbols to avoid API limits")
        symbols = symbols[:max_symbols]
    
    # Prepare output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'data', 'alpaca_options')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'spy_options_intraday_m1.csv')
    
    # Remove existing file to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Download
    print(f"\n[Step 3] Downloading M1 bars...")
    print(f"Output: {output_file}")
    print(f"Processing {len(symbols)} symbols over {(end_date - start_date).days} days...")
    print("This may take several minutes...")
    
    total_bars = download_intraday_bars(client, symbols, start_date, end_date, output_file)
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"Total M1 bars: {total_bars:,}")
    print(f"Output file: {output_file}")
    
    if total_bars > 0:
        file_size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"File size: {file_size_mb:.1f} MB")
        
        # Quick stats
        df = pd.read_csv(output_file, nrows=1000)
        print(f"\nSample columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
