import os
import datetime as dt
import pandas as pd
import pytz
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import sys
import time

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.config import RunConfig

_run_config = RunConfig()
client = StockHistoricalDataClient(_run_config.alpaca_key, _run_config.alpaca_secret)

NY = pytz.timezone("America/New_York")


def main():
    # 1. SYMBOL SELECTION
    symbol = input("Enter symbol (or ? to list available): ").strip().upper()

    if symbol == "?":
        choice = input("Select categories: {F=Forex, C=Crypto, E=Equities, T=ETFs, All=All}: ").strip().upper()
        url = "https://paper-api.alpaca.markets/v2/assets"
        headers = {
            "APCA-API-KEY-ID": _run_config.alpaca_key,
            "APCA-API-SECRET-KEY": _run_config.alpaca_secret
        }
        response = requests.get(url, headers=headers)
        assets = response.json()

        filtered = []
        for a in assets:
            if a.get('status') != 'active':
                continue

            cls = a.get('class', '').upper()
            name = a.get('name', '').upper()
            is_etf = 'ETF' in name or 'EXCHANGE TRADED FUND' in name

            if choice == 'ALL':
                filtered.append(a['symbol'])
            elif choice == 'F' and cls == 'FX':
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

    # 2. TIMEFRAME SELECTION
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

    prefix_map = {
        "1": "m1", "5": "m5", "15": "m15", "30": "m30", "60": "h1",
        "2H": "h2", "4H": "h4", "D": "d1", "W": "w1", "M": "mo1"
    }

    # ✅ Always save under project root/data/spot
    # The file should be put in yearly folders
    base_dir = os.path.join(PROJECT_ROOT, "data", "spot")
    os.makedirs(base_dir, exist_ok=True)

    summary_stats = []

    # Free tier requires 15-minute delay on SIP data
    current_now = dt.datetime.now(pytz.utc) - dt.timedelta(minutes=16)

    years = [2020, 2021, 2022, 2023, 2024]

    print("\n[Note] Bars data API doesn't provide aggregate bid/ask.")
    print("[Note] We will download o, h, l, c, v, vwap as requested.")
    
    for year in years:
        year_dir = os.path.join(base_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        start_dt = dt.datetime(year, 1, 2 if year == 2020 else 1, tzinfo=pytz.utc)
        end_dt = dt.datetime(year, 12, 31, 23, 59, 59, tzinfo=pytz.utc)

        if start_dt > current_now:
            continue
        if end_dt > current_now:
            end_dt = current_now

        print(f"\n=== Processing Year {year} ===")

        for label in selected_labels:
            if label not in tf_map:
                print(f"[Skip] Unknown timeframe label: {label}")
                continue

            file_prefix = prefix_map.get(label, f"m{label}")
            out_name = f"{file_prefix}_{year}.csv"
            out_path = os.path.join(year_dir, out_name)

            # INCREMENTAL APPEND LOGIC
            if os.path.exists(out_path):
                # Using the expected new columns since it's already processed
                existing_df = pd.read_csv(out_path, parse_dates=["datetime"])
                if not existing_df.empty:
                    last_dt = existing_df["datetime"].max()
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=pytz.utc)
                    query_start_dt = last_dt + dt.timedelta(seconds=1)
                    print(f"\n[Update] {label} ({year}): Resuming from {query_start_dt}")
                    mode = 'update'
                else:
                    query_start_dt = start_dt
                    mode = 'fresh'
            else:
                query_start_dt = start_dt
                mode = 'fresh'
                print(f"\n[New] {label} ({year}): Starting fresh download from {start_dt}")

            if query_start_dt > end_dt:
                print(f"[Skip] Already retrieved all data up to {end_dt} for {year} / {label}")
                continue

            try:
                print(f"[Fetching] {symbol} {label} bars from {query_start_dt} to {end_dt}")
                bars = client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf_map[label],
                    start=query_start_dt,
                    end=end_dt,
                    adjustment='split'
                ))

                new_df = bars.df.reset_index()

                if mode == 'update' and not new_df.empty:
                    # new_df has default columns from Alpaca: timestamp, symbol, open, high, low, close, volume, trade_count, vwap
                    # existing_df has our mapped columns (datetime, o, h, etc.)
                    # Let's write the raw newly retrieved dataframe to a temp file and let the cleaning process handle it?
                    # No, it's easier to fetch everything as fresh or handle update if it's already there.
                    # Wait, if we are doing incremental, we should append raw, then pass to cleaning.
                    # For simplicity of the year-by-year download, let's just do fresh or handle properly.
                    
                    # Map new_df first to match our format before appending
                    temp_df = clean_equity_bars_session_only_df(new_df, label)
                    
                    if not temp_df.empty:
                        df = pd.concat([existing_df, temp_df], ignore_index=True).drop_duplicates(subset=['datetime'])
                    else:
                        df = existing_df
                else:
                    if new_df.empty:
                        df = pd.DataFrame()
                    else:
                        df = clean_equity_bars_session_only_df(new_df, label)

                if not df.empty:
                    df = df.sort_values(by="datetime")
                    df.to_csv(out_path, index=False)
                    print(f"[Saved] {out_path} - {len(df)} rows")
                    summary_stats.append(f"{year} - {label} ({out_name}): {len(df)} total rows")
                else:
                    print(f"[Warning] No new data found for {label} in {year}")

            except Exception as e:
                print(f"[Error] Processing {year} {label}: {e}")

    print("\n" + "="*60)
    print(f"DATA DOWNLOAD COMPLETE: {symbol}")
    print("="*60)
    for s in summary_stats:
        print(f"  {s}")
    print("="*60)


def _expected_freq_seconds(label: str) -> int:
    """Map timeframe label to expected seconds for market-hours grids."""
    if label.isdigit():
        return int(label) * 60
    label = label.upper()
    if label.endswith("H"):
        return int(label[:-1]) * 3600
    if label == "D":
        return 86400
    if label == "W":
        return 7 * 86400
    if label == "M":
        return 30 * 86400
    return 60


def clean_equity_bars_session_only_df(df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    """
    Clean & normalize Alpaca US equity bars without creating fake 24/7 candles.
    Takes a raw DataFrame from Alpaca API and returns a cleaned DataFrame with requested columns.
    """
    if df.empty:
        return df

    # Normalize timestamp -> timezone-aware UTC
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = ts
        df = df.dropna(subset=["timestamp"])

        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Convert to NY time for session filtering
        df["ts_ny"] = df["timestamp"].dt.tz_convert(NY)

        # Weekdays only
        df = df[df["ts_ny"].dt.dayofweek < 5]

        # Regular Trading Hours: 09:30–16:00 (inclusive end is tricky; keep <= 16:00)
        t = df["ts_ny"].dt.time
        rth_start = dt.time(9, 30)
        rth_end = dt.time(16, 0)
        df = df[(t >= rth_start) & (t <= rth_end)]

        if df.empty:
            return pd.DataFrame()

        # Determine expected frequency from timeframe label
        expected_sec = _expected_freq_seconds(tf_label)
        freq = f"{expected_sec}s"

        # Build per-day session grid (prevents 24/7 bars)
        session_days = df["ts_ny"].dt.normalize().unique()

        grids = []
        for day in session_days:
            start = (pd.Timestamp(day).tz_convert(NY).replace(hour=9, minute=30, second=0))
            end = (pd.Timestamp(day).tz_convert(NY).replace(hour=16, minute=0, second=0))
            grids.append(pd.date_range(start=start, end=end, freq=freq, tz=NY))

        if len(grids) > 1:
            full_ny_index = grids[0].append(grids[1:])
        else:
            full_ny_index = grids[0]
            
        full_utc_index = full_ny_index.tz_convert(pytz.utc)

        # Reindex onto the session-only grid
        df = df.set_index("timestamp").reindex(full_utc_index)

    # Interpolate only price-like columns; volume/trade_count should NOT be fractional
    price_cols = [c for c in ["open", "high", "low", "close", "vwap"] if c in df.columns]
    if price_cols:
        df[price_cols] = df[price_cols].interpolate(method="time", limit_direction="both")

    # Volume: fill missing with 0 then cast to int
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).round(0).clip(lower=0).astype("int64")

    # Round prices to match typical equity bar precision
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    # Reset index back to timestamp column
    df = df.reset_index().rename(columns={"index": "datetime", "timestamp": "datetime"})
    
    # Rename columns to requested format
    rename_map = {
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v"
    }
    df = df.rename(columns=rename_map)

    # Filter to requested columns only
    cols_to_keep = ["datetime", "o", "h", "l", "c", "v"]
    if "vwap" in df.columns:
        cols_to_keep.append("vwap")

    # Ensure columns exist before filtering to avoid errors
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[existing_cols]

    return df


if __name__ == "__main__":
    main()
