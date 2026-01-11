#!/usr/bin/env python3
"""
Download 3 months of real historical SPY options data from Alpaca.
This replaces synthetic data with actual market data for backtesting.
"""
import datetime as dt
import pandas as pd
import os
import sys
from typing import List

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.config import RunConfig


def download_option_bars(client: OptionHistoricalDataClient,
                          symbols: List[str],
                          start_date: dt.datetime,
                          end_date: dt.datetime) -> pd.DataFrame:
    """
    Download historical bars for a list of option symbols.
    Uses BarSet.df to get DataFrame directly.
    """
    all_dfs = []
    
    # Process in batches to avoid API limits
    batch_size = 100
    total_batches = (len(symbols) // batch_size) + 1
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} symbols)...", end=" ")
        
        try:
            req = OptionBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            bars = client.get_option_bars(req)
            
            # Use .df to get DataFrame directly
            df = bars.df
            if len(df) > 0:
                # Reset index to get symbol and timestamp as columns
                df = df.reset_index()
                all_dfs.append(df)
                print(f"{len(df)} bars")
            else:
                print("no data")
                
        except Exception as e:
            print(f"error: {str(e)[:50]}")
            continue
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Parse symbol components
        def parse_occ_symbol(sym):
            """Parse OCC symbol like SPY251017C00580000"""
            try:
                # Find where date starts (after underlying)
                # SPY = 3 chars
                date_str = sym[3:9]
                opt_type = sym[9]
                strike_str = sym[10:]
                
                expiration = dt.datetime.strptime(date_str, "%y%m%d").date()
                strike = float(strike_str) / 1000
                is_call = (opt_type == 'C')
                
                return pd.Series({
                    'underlying': 'SPY',
                    'expiration': expiration,
                    'strike': strike,
                    'option_type': 'call' if is_call else 'put'
                })
            except:
                return pd.Series({
                    'underlying': None,
                    'expiration': None,
                    'strike': None,
                    'option_type': None
                })
        
        # Apply parsing
        parsed = combined['symbol'].apply(parse_occ_symbol)
        combined = pd.concat([combined, parsed], axis=1)
        
        # Convert timestamp to date
        combined['date'] = pd.to_datetime(combined['timestamp']).dt.date
        
        return combined
    
    return pd.DataFrame()


def main():
    """Main download routine"""
    print("=" * 60)
    print("Alpaca Historical Options Data Downloader")
    print("=" * 60)
    
    # Initialize client
    r = RunConfig()
    client = OptionHistoricalDataClient(r.alpaca_key, r.alpaca_secret)
    
    # Date range: 3 months back from today
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=90)
    
    print(f"\nDate range: {start_date.date()} to {end_date.date()}")
    
    # Get current option symbols
    print("\n[Step 1] Getting available option symbols...")
    try:
        req = OptionChainRequest(underlying_symbol='SPY')
        chain = client.get_option_chain(req)
        symbols = list(chain.keys())
        print(f"Found {len(symbols)} option contracts")
    except Exception as e:
        print(f"Error getting chain: {e}")
        return
    
    # Download historical bars
    print(f"\n[Step 2] Downloading historical bars...")
    print(f"This may take several minutes for {len(symbols)} symbols...")
    
    bars_df = download_option_bars(client, symbols, start_date, end_date)
    print(f"\nTotal: {len(bars_df)} bar records downloaded")
    
    if len(bars_df) == 0:
        print("No data retrieved. Check API subscription or date range.")
        return
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'data', 'alpaca_options')
    os.makedirs(output_dir, exist_ok=True)
    
    bars_file = os.path.join(output_dir, 'spy_options_bars.csv')
    bars_df.to_csv(bars_file, index=False)
    print(f"\nSaved to: {bars_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Total bars:     {len(bars_df):,}")
    print(f"Unique dates:   {bars_df['date'].nunique()}")
    print(f"Unique symbols: {bars_df['symbol'].nunique()}")
    print(f"Date range:     {bars_df['date'].min()} to {bars_df['date'].max()}")
    print(f"\nFile saved to: {bars_file}")
    print(f"File size: {os.path.getsize(bars_file) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
