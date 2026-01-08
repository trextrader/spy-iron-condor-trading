#!/usr/bin/env python3
"""
Download historical SPY options data with Greeks from IVolatility API.
Uses the official ivolatility Python library.
"""
import os
import sys
import datetime as dt
import time
import pandas as pd
import ivolatility as ivol

# Configuration
import concurrent.futures
import random

# Configuration
IVOL_API_KEY = "MFGkqVygN5NSgF2I"
OUTPUT_DIR = "data/ivolatility"
OUTPUT_FILE = "spy_options_ivol_large.csv" # Separate large dataset
MAX_WORKERS = 3
TARGET_OPTIONS_PER_DAY = 100

def setup_ivol():
    """Initialize IVolatility API connection."""
    ivol.setLoginParams(apiKey=IVOL_API_KEY)


def get_trading_dates(start_date: dt.date, end_date: dt.date) -> list:
    """Generate list of weekdays between dates."""
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current += dt.timedelta(days=1)
    return dates


def fetch_options_chain_for_date(symbol: str, trade_date: str) -> pd.DataFrame:
    """
    Fetch option chain IDs for a given date.
    Get options with 25-65 DTE (wider range for production).
    """
    try:
        # Calculate expiration range (25-65 DTE)
        date_obj = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        exp_from = (date_obj + dt.timedelta(days=25)).strftime("%Y-%m-%d")
        exp_to = (date_obj + dt.timedelta(days=65)).strftime("%Y-%m-%d")
        
        getOptsChain = ivol.setMethod('/equities/eod/option-series-on-date')
        
        # Get both calls and puts
        calls = getOptsChain(
            symbol=symbol,
            date=trade_date,
            expFrom=exp_from,
            expTo=exp_to,
            callPut='C'
        )
        
        puts = getOptsChain(
            symbol=symbol,
            date=trade_date,
            expFrom=exp_from,
            expTo=exp_to,
            callPut='P'
        )
        
        # Combine
        if calls is not None and puts is not None:
            return pd.concat([calls, puts], ignore_index=True)
        elif calls is not None:
            return calls
        elif puts is not None:
            return puts
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error fetching chain for {trade_date}: {e}")
        return pd.DataFrame()


def fetch_option_with_retry(option_id: int, trade_date: str, max_retries=3) -> pd.DataFrame:
    """Fetch single option with IV and Greeks, with retry logic."""
    for attempt in range(max_retries):
        try:
            getOpts = ivol.setMethod('/equities/eod/single-stock-option-raw-iv')
            data = getOpts(optionId=option_id, from_=trade_date, to=trade_date)
            if data is not None:
                return data
        except Exception as e:
            # Check for 429 or other transient errors
            if "429" in str(e) or "Too Many Requests" in str(e):
                waittime = (attempt + 1) * 2 + random.uniform(0, 1)
                time.sleep(waittime) # Backoff
            else:
                # Other error, maybe skip?
                pass
        
        time.sleep(0.5) # Basic rate limit between retries
        
    return pd.DataFrame()


def download_spy_options_year():
    """
    Download SPY options data using Threading.
    Scales to Top 500 options per day.
    """
    setup_ivol()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Date range: RESUME from 2025-06-28 to 2026-01-06
    end_date = dt.date(2026, 1, 6)
    start_date = dt.date(2025, 6, 28) 
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print("=" * 70)
    print(f"IVolatility MASSIVE Download (Target: Top {TARGET_OPTIONS_PER_DAY}/day)")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Output: {output_path}")
    print("=" * 70, flush=True)
    
    all_dates = get_trading_dates(start_date, end_date)
    # No sampling! We want the full dataset if possible, or maybe sample every 2 days?
    # User said "full eod dataset".
    sample_dates = all_dates # Full resolution
    
    # Check existing data to resume
    existing_dates = set()
    first_write = True
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            df_exist = pd.read_csv(output_path, usecols=['date'])
            # Assuming 'date' column exists
            dates = pd.to_datetime(df_exist['date']).dt.date
            existing_dates = set(dates)
            print(f"Found {len(existing_dates)} existing dates. Resuming...", flush=True)
            first_write = False
        except Exception as e:
            print(f"Warning/Error reading existing: {e}. Starting fresh/append.")
            first_write = False # Safe default to append if file exists
    
    total_records = 0
    
    for i, trade_date in enumerate(sample_dates):
        if trade_date in existing_dates:
            print(f"Skipping {trade_date} (already processed)")
            continue
            
        date_str = trade_date.strftime("%Y-%m-%d")
        print(f"\n[{i+1}/{len(sample_dates)}] Processing {date_str}...", flush=True)
        
        # Step 1: Get option chain
        chain = fetch_options_chain_for_date("SPY", date_str)
        
        if chain.empty:
            print(f"  No options found", flush=True)
            continue
        
        # Step 2: Intelligent Filter (Top N by Liquid)
        # Note: chain returned by 'option-series-on-date' does NOT have volume/OI info directly :(
        # We checked this in Step Id: 2287.
        # It only has: ['OptionSymbol', 'callPut', 'strike', 'expirationDate', 'optionId']
        
        # CRITICAL: We cannot sort by volume if we don't have it!
        # Strategy pivot: We must approximate liquidity by "Near the Data".
        # ATM options are usually most liquid.
        # We will take a wider ATM band.
        
        if 'strike' in chain.columns:
            chain['strike'] = pd.to_numeric(chain['strike'])
            unique_strikes = sorted(chain['strike'].unique())
            mid_idx = len(unique_strikes) // 2
            
            # Select 60 strikes around ATM (30 up, 30 down)
            # Typically strikes are $0.5 or $1 apart. 60 strikes = $30-$60 range.
            # This covers the high volume area reasonably well.
            atm_strikes = unique_strikes[max(0, mid_idx-60):min(len(unique_strikes), mid_idx+60)]
            
            mask = chain['strike'].isin(atm_strikes)
            liquid_candidates = chain[mask]
            
            # If we still have too many, subsample?
            # Or just take them all.
            # We want up to 500.
            if len(liquid_candidates) > TARGET_OPTIONS_PER_DAY:
                 # Prioritize near-term? No, standard sort.
                 # Just take the first N (closest to ATM usually if sorted by strike?)
                 # Actually unique_strikes is sorted.
                 # We selected a band around ATM.
                 # We can take random? Or just take the center?
                 pass
            option_ids = liquid_candidates['optionId'].unique()[:TARGET_OPTIONS_PER_DAY]
        else:
             option_ids = chain['optionId'].unique()[:TARGET_OPTIONS_PER_DAY]

        print(f"  Fetching Greeks for {len(option_ids)} options (Threaded)...", flush=True)
        
        day_rows = []
        
        # Threaded Fetch
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map option_ids to futures
            future_to_id = {executor.submit(fetch_option_with_retry, int(oid), date_str): oid for oid in option_ids}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_id):
                completed += 1
                if completed % 50 == 0:
                    print(f"    Progress: {completed}/{len(option_ids)}", end='\r', flush=True)
                    
                try:
                    data = future.result()
                    if not data.empty:
                        data['trade_date_fetch'] = date_str
                        day_rows.append(data)
                except Exception as exc:
                    pass
        
        print(f"    Fetched {len(day_rows)} valid records.")
        
        # Write Result
        if day_rows:
            day_df = pd.concat(day_rows, ignore_index=True)
            day_df.to_csv(output_path, mode='a', header=first_write, index=False)
            first_write = False
            total_records += len(day_df)
            
    print(f"\nDone. Total records: {total_records}")
    return total_records

if __name__ == "__main__":
    download_spy_options_year()
