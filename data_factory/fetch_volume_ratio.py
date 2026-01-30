# data_factory/fetch_volume_ratio.py
"""
Fetch tick-level trades + quotes from Alpaca, classify as buy/sell using
Lee-Ready algorithm, aggregate to 1-minute bars, output volume_ratio.

Usage:
    python data_factory/fetch_volume_ratio.py --symbol SPY --start 2024-01-02 --end 2024-12-31 --output data/processed/spy_volume_ratio_2024.csv
"""
import sys
import argparse
import datetime as dt
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# Add project root for imports
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Alpaca API base
BASE_URL = "https://data.alpaca.markets/v2/stocks"


def get_headers():
    """Get Alpaca API headers from core/config.py"""
    try:
        from core.config import RunConfig
        rc = RunConfig()
        return {
            "accept": "application/json",
            "APCA-API-KEY-ID": rc.alpaca_key,
            "APCA-API-SECRET-KEY": rc.alpaca_secret
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load credentials from core/config.py: {e}")


def fetch_trades(symbol: str, start: str, end: str, headers: dict) -> pd.DataFrame:
    """
    Fetch all trades for symbol between start and end.
    Handles pagination via next_page_token.
    """
    url = f"{BASE_URL}/{symbol}/trades"
    all_trades = []
    page_token = None

    while True:
        params = {
            "start": start,
            "end": end,
            "limit": 10000,
            "sort": "asc"
        }
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"[ERROR] Trades fetch failed: {resp.status_code} - {resp.text[:200]}")
            break

        data = resp.json()
        trades = data.get("trades", [])
        if trades:
            all_trades.extend(trades)

        page_token = data.get("next_page_token")
        if not page_token:
            break

    if not all_trades:
        return pd.DataFrame(columns=['t', 'p', 's'])

    df = pd.DataFrame(all_trades)
    # Rename columns: t=timestamp, p=price, s=size
    df = df.rename(columns={'t': 'ts', 'p': 'price', 's': 'size'})
    df['ts'] = pd.to_datetime(df['ts'], format='ISO8601', utc=True)
    return df[['ts', 'price', 'size']]


def fetch_quotes(symbol: str, start: str, end: str, headers: dict) -> pd.DataFrame:
    """
    Fetch all quotes for symbol between start and end.
    Handles pagination via next_page_token.
    """
    url = f"{BASE_URL}/{symbol}/quotes"
    all_quotes = []
    page_token = None

    while True:
        params = {
            "start": start,
            "end": end,
            "limit": 10000,
            "sort": "asc"
        }
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"[ERROR] Quotes fetch failed: {resp.status_code} - {resp.text[:200]}")
            break

        data = resp.json()
        quotes = data.get("quotes", [])
        if quotes:
            all_quotes.extend(quotes)

        page_token = data.get("next_page_token")
        if not page_token:
            break

    if not all_quotes:
        return pd.DataFrame(columns=['t', 'bp', 'ap'])

    df = pd.DataFrame(all_quotes)
    # Rename: t=timestamp, bp=bid_price, ap=ask_price
    df = df.rename(columns={'t': 'ts', 'bp': 'bid_price', 'ap': 'ask_price'})
    df['ts'] = pd.to_datetime(df['ts'], format='ISO8601', utc=True)
    return df[['ts', 'bid_price', 'ask_price']]


def classify_trades_lee_ready(trades_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each trade as buyer-initiated (+1) or seller-initiated (-1)
    using Lee-Ready algorithm:
      - Trade >= ask -> buyer initiated
      - Trade <= bid -> seller initiated
      - Trade at midpoint -> use tick test
    """
    if trades_df.empty:
        trades_df['side'] = []
        return trades_df

    trades_df = trades_df.copy().sort_values('ts').reset_index(drop=True)
    quotes_df = quotes_df.copy().sort_values('ts').reset_index(drop=True)

    # Merge: for each trade, get the most recent quote
    trades_df = pd.merge_asof(
        trades_df,
        quotes_df[['ts', 'bid_price', 'ask_price']],
        on='ts',
        direction='backward'
    )

    # Calculate midpoint
    trades_df['mid'] = (trades_df['bid_price'] + trades_df['ask_price']) / 2

    # Vectorized Lee-Ready (fast path for most trades)
    sides = np.zeros(len(trades_df), dtype=np.int8)

    price = trades_df['price'].values
    bid = trades_df['bid_price'].values
    ask = trades_df['ask_price'].values
    mid = trades_df['mid'].values

    # Buy if price >= ask
    sides[price >= ask] = 1
    # Sell if price <= bid
    sides[price <= bid] = -1

    # For trades at midpoint, use tick test
    at_mid = (sides == 0)
    if at_mid.any():
        # Tick test: compare to previous price
        prev_price = np.roll(price, 1)
        prev_price[0] = price[0]

        uptick = price > prev_price
        downtick = price < prev_price

        sides[at_mid & uptick] = 1
        sides[at_mid & downtick] = -1

        # For zero-tick at midpoint, use previous side
        still_zero = (sides == 0)
        if still_zero.any():
            # Forward fill previous non-zero side
            for i in range(len(sides)):
                if sides[i] == 0 and i > 0:
                    sides[i] = sides[i-1] if sides[i-1] != 0 else 1

    trades_df['side'] = sides
    return trades_df


def aggregate_to_minute_bars(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tick trades to 1-minute bars with buy_volume, sell_volume, volume_ratio.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=['timestamp', 'buy_volume', 'sell_volume', 'volume_ratio'])

    trades_df = trades_df.copy()
    trades_df['minute'] = trades_df['ts'].dt.floor('1min')

    # Vectorized buy/sell volume
    trades_df['buy_vol'] = np.where(trades_df['side'] == 1, trades_df['size'], 0)
    trades_df['sell_vol'] = np.where(trades_df['side'] == -1, trades_df['size'], 0)

    # Aggregate by minute
    agg = trades_df.groupby('minute').agg({
        'buy_vol': 'sum',
        'sell_vol': 'sum',
    }).reset_index()

    agg.columns = ['timestamp', 'buy_volume', 'sell_volume']

    # Volume ratio: buy / sell (with epsilon)
    agg['volume_ratio'] = agg['buy_volume'] / (agg['sell_volume'] + 1)

    # Clip extreme ratios
    agg['volume_ratio'] = agg['volume_ratio'].clip(0.1, 10.0)

    return agg


def fetch_day_volume_ratio(symbol: str, day: dt.date, headers: dict) -> pd.DataFrame:
    """
    Fetch and compute volume ratio for a single day (RTH: 09:30-16:00 ET).
    """
    # RTH window in ISO format (UTC)
    # 09:30 ET = 14:30 UTC (EST) or 13:30 UTC (EDT)
    # Using full day and filtering later is simpler
    start = f"{day}T09:30:00-05:00"  # ET
    end = f"{day}T16:00:00-05:00"

    trades_df = fetch_trades(symbol, start, end, headers)
    if trades_df.empty:
        return pd.DataFrame(columns=['timestamp', 'buy_volume', 'sell_volume', 'volume_ratio'])

    quotes_df = fetch_quotes(symbol, start, end, headers)

    # Classify trades
    trades_classified = classify_trades_lee_ready(trades_df, quotes_df)

    # Aggregate to minute bars
    minute_bars = aggregate_to_minute_bars(trades_classified)

    return minute_bars


def main():
    parser = argparse.ArgumentParser(description="Fetch volume ratio from Alpaca tick data")
    parser.add_argument("--symbol", default="SPY", help="Symbol to fetch")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--progress", type=int, default=5, help="Progress interval (days)")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    print(f"[START] Fetching volume ratio for {args.symbol}")
    print(f"        Range: {start_date} to {end_date}")
    print(f"        Output: {args.output}")

    headers = get_headers()

    # Generate trading days (weekdays only)
    all_bars = []
    current = start_date
    days_to_process = []
    while current <= end_date:
        if current.weekday() < 5:  # Mon-Fri
            days_to_process.append(current)
        current += dt.timedelta(days=1)

    total_days = len(days_to_process)
    print(f"[INFO] Processing {total_days} trading days\n")

    ok_count = 0
    empty_count = 0
    fail_count = 0

    for i, day in enumerate(days_to_process, 1):
        try:
            bars = fetch_day_volume_ratio(args.symbol, day, headers)

            if bars.empty:
                empty_count += 1
            else:
                all_bars.append(bars)
                ok_count += 1

            if i % args.progress == 0 or i == total_days:
                print(f"  [{i}/{total_days}] {day} - {len(bars)} bars | ok={ok_count} empty={empty_count} fail={fail_count}")

        except Exception as e:
            fail_count += 1
            print(f"  [{i}/{total_days}] {day} - ERROR: {e}")

    print(f"\n[DONE] ok={ok_count} empty={empty_count} fail={fail_count}")

    if not all_bars:
        print("[WARN] No data collected!")
        return

    # Combine all days
    result = pd.concat(all_bars, ignore_index=True)
    result = result.sort_values('timestamp').reset_index(drop=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\n[SAVED] {len(result)} rows to {output_path}")
    print(f"\n[STATS]")
    print(f"  volume_ratio min:  {result['volume_ratio'].min():.3f}")
    print(f"  volume_ratio max:  {result['volume_ratio'].max():.3f}")
    print(f"  volume_ratio mean: {result['volume_ratio'].mean():.3f}")
    print(f"  volume_ratio std:  {result['volume_ratio'].std():.3f}")


if __name__ == "__main__":
    main()
