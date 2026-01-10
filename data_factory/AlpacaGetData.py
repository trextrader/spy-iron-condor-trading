# data_factory/AlpacaGetData.py
import os
import datetime as dt
import pandas as pd
import pytz
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import sys

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    # ✅ Always save under project root/data/spot
    base_dir = os.path.join(PROJECT_ROOT, "data", "spot")
    os.makedirs(base_dir, exist_ok=True)

    summary_stats = []

    # Free tier requires 15-minute delay on SIP data
    current_now = dt.datetime.now(pytz.utc) - dt.timedelta(minutes=16)

    for label in selected_labels:
        if label not in tf_map:
            print(f"[Skip] Unknown timeframe label: {label}")
            continue

        out_path = os.path.join(base_dir, f"{symbol}_{label}.csv")

        # INCREMENTAL APPEND LOGIC
        if os.path.exists(out_path):
            existing_df = pd.read_csv(out_path, parse_dates=["timestamp"])
            if not existing_df.empty:
                start_dt = existing_df["timestamp"].max()
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=pytz.utc)
                start_dt = start_dt + dt.timedelta(seconds=1)
                print(f"\n[Update] {label}: Resuming from {start_dt}")
                mode = 'update'
            else:
                start_dt = dt.datetime(2024, 1, 1, tzinfo=pytz.utc)
                mode = 'fresh'
        else:
            start_dt = dt.datetime(2024, 1, 1, tzinfo=pytz.utc)
            mode = 'fresh'
            print(f"\n[New] {label}: Starting fresh download from 2024-01-01")

        try:
            print(f"[Fetching] {symbol} {label} bars from {start_dt} to {current_now}")
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf_map[label],
                start=start_dt,
                end=current_now,
                adjustment='split'
            ))

            new_df = bars.df.reset_index()

            if mode == 'update' and not new_df.empty:
                df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['timestamp'])
            else:
                df = new_df

            if not df.empty:
                df.to_csv(out_path, index=False)
                print(f"[Saved] Raw data: {len(df)} rows")

                # ✅ CLEANUP: session-only grid + rounding + int casting (no 24/7 bars)
                final_count = clean_equity_bars_session_only(out_path, symbol, label)
                summary_stats.append(f"{label}: {final_count} total rows")
            else:
                print(f"[Warning] No new data found for {label} (API delay may apply)")

        except Exception as e:
            print(f"[Error] Processing {label}: {e}")

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


def clean_equity_bars_session_only(file_path: str, symbol_name: str, tf_label: str) -> int:
    """
    Clean & normalize Alpaca US equity bars without creating fake 24/7 candles.

    - Drops duplicates, sorts timestamps
    - Filters to Mon–Fri Regular Trading Hours (09:30–16:00 NY)
    - Builds an expected timestamp grid ONLY inside RTH (prevents 24/7 interpolation)
    - Interpolates price-like columns ONLY within-session gaps (rare)
    - Rounds prices to 2 decimals; casts volume/trade_count to integers
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    if df.empty:
        return 0

    # Normalize timestamp -> timezone-aware UTC
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
        # still write an empty but valid file
        out = df.drop(columns=["ts_ny"], errors="ignore")
        out.to_csv(file_path, index=False)
        print(f"[Cleaned] 0 rows after session filter")
        return 0

    # Determine expected frequency from timeframe label
    expected_sec = _expected_freq_seconds(tf_label)
    freq = f"{expected_sec}s"

    # Build per-day session grid (prevents 24/7 bars)
    # We generate timestamps in NY time, then convert to UTC for storage.
    session_days = df["ts_ny"].dt.normalize().unique()

    grids = []
    for day in session_days:
        # day is tz-aware Timestamp at midnight NY
        start = (pd.Timestamp(day).tz_convert(NY).replace(hour=9, minute=30, second=0))
        end = (pd.Timestamp(day).tz_convert(NY).replace(hour=16, minute=0, second=0))
        grids.append(pd.date_range(start=start, end=end, freq=freq, tz=NY))

    full_ny_index = grids[0].append(grids[1:]) if len(grids) > 1 else grids[0]
    full_utc_index = full_ny_index.tz_convert(pytz.utc)

    # Reindex onto the session-only grid
    df = df.set_index("timestamp").reindex(full_utc_index)

    # Interpolate only price-like columns; volume/trade_count should NOT be fractional
    price_cols = [c for c in ["open", "high", "low", "close", "vwap"] if c in df.columns]
    if price_cols:
        df[price_cols] = df[price_cols].interpolate(method="time", limit_direction="both")

    # Volume & trade_count: fill missing with 0 then cast to int
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).round(0).clip(lower=0).astype("int64")
    if "trade_count" in df.columns:
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0).round(0).clip(lower=0).astype("int64")

    # Round prices to match typical equity bar precision
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    # Symbol column
    df["symbol"] = symbol_name

    # Reset index back to timestamp column
    df = df.reset_index().rename(columns={"index": "timestamp"})

    # Drop helper cols if present
    df = df.drop(columns=["ts_ny"], errors="ignore")

    df.to_csv(file_path, index=False)
    print(f"[Cleaned] {len(df)} rows after RTH-only grid + rounding")
    return len(df)


if __name__ == "__main__":
    main()
