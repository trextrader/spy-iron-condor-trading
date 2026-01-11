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
MAX_WORKERS = 10 # Increase threads for speed
TARGET_OPTIONS_PER_DAY = 1000 # Enough for 10% OTM wings

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


def fetch_options_chain_for_date(symbol: str, trade_date: str, max_retries=5) -> pd.DataFrame:
    """
    Fetch option chain IDs for a given date with Retry Logic.
    Get options with 25-65 DTE.
    """
    for attempt in range(max_retries):
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
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                wait = (attempt + 1) * 2 + random.uniform(1, 3)
                print(f"  429 Limit hit (chain). Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            else:
                print(f"Error fetching chain for {trade_date}: {e}")
                return pd.DataFrame()
    
    print(f"  Failed to fetch chain after {max_retries} retries.")
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
    Scales to Top 1000 options per day for Iron Condor breadth.
    """
    setup_ivol()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Date range: 3 Months for Validation (Jun 2025 - Aug 2025)
    end_date = dt.date(2025, 8, 30)
    start_date = dt.date(2025, 6, 1) 
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print("=" * 70)
    print(f"IVolatility MASSIVE Download (Target: Top {TARGET_OPTIONS_PER_DAY}/day)")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Range: {start_date} to {end_date}")
    print(f"Output: {output_path}")
    print("=" * 70, flush=True)

    all_dates = get_trading_dates(start_date, end_date)
    # Full resolution
    sample_dates = all_dates 
    
    # Load spot prices from existing EOD file for smart filtering
    spot_map = {}
    eod_file = os.path.join(OUTPUT_DIR, "spy_options_ivol_1year.csv")
    if os.path.exists(eod_file):
        try:
            spot_df = pd.read_csv(eod_file, usecols=['date', 'underlying_price'])
            spot_df['date'] = pd.to_datetime(spot_df['date']).dt.date
            spot_map = spot_df.drop_duplicates('date').set_index('date')['underlying_price'].to_dict()
            print(f"Loaded {len(spot_map)} spot prices from local file.")
        except Exception:
            pass

    # FORCE FRESH START for Large File (Overwrite bad data)
    if os.path.exists(output_path) and os.path.getsize(output_path) < 100 * 1024:
        print("Existing file is small (likely bad run). Overwriting...")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except: pass
            
    # Re-check existence
    existing_dates = set()
    first_write = True
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            df_exist = pd.read_csv(output_path, usecols=['date'])
            dates = pd.to_datetime(df_exist['date']).dt.date
            existing_dates = set(dates)
            print(f"Found {len(existing_dates)} existing dates. Resuming...", flush=True)
            first_write = False
        except Exception:
            first_write = False
    
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
        
        # Step 2: Intelligent Filter (ATM via Spot Price)
        # ------------------------------------------------
        spot_price = spot_map.get(trade_date)
        
        if spot_price is None:
            # Fallback: Fetch from API
            try:
                getPrices = ivol.setMethod('/equities/eod/stock-prices-on-date')
                spot_data = getPrices(symbol='SPY', date=date_str)
                if spot_data is not None and not spot_data.empty:
                    spot_price = float(spot_data.iloc[0]['close'])
                    print(f"  Fetched Spot: {spot_price}")
                else:
                    pass
            except Exception as e:
                print(f"Error fetching spot: {e}")

        if spot_price is None:
            print("  Warning: Spot price unavailable. Using chain middle.")
        else:
             pass # Already printed

        if 'strike' in chain.columns:
            chain['strike'] = pd.to_numeric(chain['strike'])
            
            if spot_price:
                 # WIDE Filtering: ±15% OTM to Capture Iron Condor Wings (Delta ~0.05)
                 lower_bound = spot_price * 0.85
                 upper_bound = spot_price * 1.15
                 
                 # Filter by Strike Range first
                 mask = (chain['strike'] >= lower_bound) & (chain['strike'] <= upper_bound)
                 filtered_chain = chain[mask]
                 
                 if filtered_chain.empty:
                     # Fallback to nearest if filter is empty (e.g. erratic spot data)
                     chain['distance'] = abs(chain['strike'] - spot_price)
                     liquid_candidates = chain.sort_values('distance')
                 else:
                     # Sort inside the valid range by distance to center
                     filtered_chain['distance'] = abs(filtered_chain['strike'] - spot_price)
                     liquid_candidates = filtered_chain.sort_values('distance')
                     
            else:
                unique_strikes = sorted(chain['strike'].unique())
                mid_idx = len(unique_strikes) // 2
                atm_strikes = unique_strikes[max(0, mid_idx-100):min(len(unique_strikes), mid_idx+100)]
                mask = chain['strike'].isin(atm_strikes)
                liquid_candidates = chain[mask]

            # Take top N candidates
            option_ids = liquid_candidates['optionId'].unique()[:TARGET_OPTIONS_PER_DAY]
        else:
             option_ids = chain['optionId'].unique()[:TARGET_OPTIONS_PER_DAY]

        print(f"  Fetching Greeks for {len(option_ids)} options (Threaded, ATM)...", flush=True)
        
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
