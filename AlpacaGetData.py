import os
import datetime as dt
import pandas as pd
import pytz
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config import RunConfig 

# --- Alpaca client ---
client = StockHistoricalDataClient(RunConfig.alpaca_key, RunConfig.alpaca_secret) 

def main():
    # 1. SYMBOL SELECTION
    symbol = input("Enter symbol (or ? to list available): ").strip().upper() 
    
    if symbol == "?":
        choice = input("Select categories: {F=Forex, C=Crypto, E=Equities, T=ETFs, All=All}: ").strip().upper() 
        url = "https://paper-api.alpaca.markets/v2/assets"
        headers = {
            "APCA-API-KEY-ID": RunConfig.alpaca_key,
            "APCA-API-SECRET-KEY": RunConfig.alpaca_secret
        }
        response = requests.get(url, headers=headers)
        assets = response.json() 
        
        filtered = []
        for a in assets:
            if a['status'] != 'active': continue
            
            cls = a.get('class', '').upper()
            name = a.get('name', '').upper()
            is_etf = 'ETF' in name or 'EXCHANGE TRADED FUND' in name
            
            # FIXED: "ALL" logic now displays every active symbol 
            if choice == 'ALL':
                filtered.append(a['symbol'])
            elif choice == 'F' and cls == 'FX': # Note: Forex symbols are often like 'EUR/USD'
                filtered.append(a['symbol'])
            elif choice == 'C' and cls == 'CRYPTO':
                filtered.append(a['symbol'])
            elif choice == 'E' and cls == 'US_EQUITY' and not is_etf:
                filtered.append(a['symbol'])
            elif choice == 'T' and is_etf:
                filtered.append(a['symbol'])

        if filtered:
            print(f"\n--- Found {len(filtered)} Symbols ---")
            for i in range(0, len(filtered), 8):
                print(", ".join(filtered[i:i+8]))
        else:
            print(f"No active symbols found for category: {choice}")
            
        symbol = input("\nEnter symbol from list: ").strip().upper() 

    # 2. FULL TIMEFRAME MAPPING
    print("\nAvailable: 1, 5, 15, 30, 60, 2H, 4H, D, W, M")
    tf_input = input("Enter timeframes (comma separated): ")
    selected_labels = [x.strip() for x in tf_input.split(",")] 

    tf_map = {
        "1":   TimeFrame(1, TimeFrameUnit.Minute),
        "5":   TimeFrame(5, TimeFrameUnit.Minute),
        "15":  TimeFrame(15, TimeFrameUnit.Minute),
        "30":  TimeFrame(30, TimeFrameUnit.Minute),
        "60":  TimeFrame(1, TimeFrameUnit.Hour),
        "2H":  TimeFrame(2, TimeFrameUnit.Hour),
        "4H":  TimeFrame(4, TimeFrameUnit.Hour),
        "D":   TimeFrame(1, TimeFrameUnit.Day),
        "W":   TimeFrame(1, TimeFrameUnit.Week),
        "M":   TimeFrame(1, TimeFrameUnit.Month)
    } 

    base_dir = os.path.join("reports", symbol)
    os.makedirs(base_dir, exist_ok=True)
    summary_stats = []

    # 3. INCREMENTAL UPDATE & SIP DATA FIX
    # Free tier requires a 15-minute delay on SIP data
    current_now = dt.datetime.now(pytz.utc) - dt.timedelta(minutes=16) 

    for label in selected_labels:
        if label not in tf_map: continue
        
        out_path = os.path.join(base_dir, f"{symbol}_{label}.csv")
        
        # SCAN FILE & APPEND LOGIC 
        if os.path.exists(out_path):
            existing_df = pd.read_csv(out_path, parse_dates=["timestamp"])
            if not existing_df.empty:
                # Start 1 second after last date to prevent partial/duplicate rows 
                start_dt = existing_df["timestamp"].max() + dt.timedelta(seconds=1)
                print(f"\nUpdating {label}: Resuming from {start_dt}")
                mode = 'update'
            else:
                start_dt = dt.datetime(2024, 1, 1, tzinfo=pytz.utc)
                mode = 'fresh'
        else:
            start_dt = dt.datetime(2024, 1, 1, tzinfo=pytz.utc)
            mode = 'fresh'

        try:
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf_map[label],
                start=start_dt,
                end=current_now,
                adjustment='split'
            ))
            
            new_df = bars.df.reset_index()
            
            if mode == 'update' and not new_df.empty:
                df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['timestamp'])
            else:
                df = new_df

            if not df.empty:
                df.to_csv(out_path, index=False)
                # 4. CLEANUP & INTERPOLATE
                final_count = sweep_and_interpolate(out_path, symbol)
                summary_stats.append(f"{label}: {final_count} total rows")
            else:
                print(f"No new data found for {label} (API delay may apply).")
                
        except Exception as e:
            print(f"Error processing {label}: {e}")

    print("\n" + "="*40 + f"\nCOMPLETED: {symbol}\n" + "="*40)
    for s in summary_stats: print(f" - {s}")

def sweep_and_interpolate(file_path, symbol_name):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    
    diffs = df["timestamp"].diff().dt.total_seconds()
    if diffs.isna().all(): return len(df)
    
    expected = int(diffs.mode()[0])
    # FIXED: Using lowercase 's' for modern pandas compliance
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=f"{expected}s") 
    df = df.set_index("timestamp").reindex(full_range)
    
    numeric_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    existing = [c for c in numeric_cols if c in df.columns]
    df[existing] = df[existing].interpolate(method="linear")
    
    price_cols = ["open", "high", "low", "close", "vwap"]
    for col in price_cols:
        if col in df.columns: df[col] = df[col].round(2)
        
    df["symbol"] = symbol_name
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df.to_csv(file_path, index=False)
    return len(df)

if __name__ == "__main__":
    main()