#!/usr/bin/env python3
"""
Download Dec 13-31 2025 SPY Options Bars from Alpaca (Linux/Python version)
Generates full OPRA symbol matrix, batches 10 per request.

Usage:
    python data_factory/downloaders/download_dec2025_options.py
"""

import requests
import time
import csv
import os

API_KEY    = "PKWCQL536DJKE7EJCP5OEETWE2"
API_SECRET = "8PK6xfXx13Hqna2ryjHyQCMAf6D1zj6kNGE96CjnzmKM"
BASE_URL   = "https://data.alpaca.markets/v1beta1/options/bars"
OUT_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dec_2025_options_bars.csv")

HEADERS = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}

# Confirmed expiration dates (YYMMDD)
EXPIRATIONS = [
    "251215", "251216", "251217", "251218", "251219",
    "251222", "251223", "251224", "251226",
    "251229", "251230", "251231",
]

LEAP_EXPIRATIONS = ["260116", "260320", "271219"]

# Build symbol matrix
near_term = []
for exp in EXPIRATIONS:
    for strike in range(670, 711):  # 670-710 inclusive
        s = f"{strike * 1000:08d}"
        near_term.append(f"SPY{exp}C{s}")
        near_term.append(f"SPY{exp}P{s}")

leaps = []
for exp in LEAP_EXPIRATIONS:
    for strike in range(650, 731, 5):  # 650-730 step 5
        s = f"{strike * 1000:08d}"
        leaps.append(f"SPY{exp}C{s}")
        leaps.append(f"SPY{exp}P{s}")

all_symbols = near_term + leaps

print("=" * 60)
print(f"  SPY Dec 2025 Options Bar Downloader (Python)")
print(f"  Near-term symbols:  {len(near_term)}")
print(f"  LEAP symbols:       {len(leaps)}")
print(f"  Total symbols:      {len(all_symbols)}")
print(f"  Output: {OUT_FILE}")
print("=" * 60)

# Write CSV header
with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["symbol", "timestamp", "open", "high", "low", "close",
                      "volume", "trade_count", "vwap"])

batch_size = 10
total_batches = (len(all_symbols) + batch_size - 1) // batch_size
total_bars = 0
symbols_with_data = 0
symbols_empty = 0
errors = 0
rate_limit_hits = 0
base_delay = 0.75  # seconds

t0 = time.time()

for i in range(0, len(all_symbols), batch_size):
    batch_num = i // batch_size + 1
    batch = all_symbols[i:i + batch_size]
    symbol_list = ",".join(batch)
    pct = batch_num / total_batches * 100

    print(f"Batch {batch_num}/{total_batches} ({pct:.1f}%)  symbols {i+1}-{i+len(batch)}...")

    params = {
        "symbols": symbol_list,
        "timeframe": "1Min",
        "start": "2025-12-13T00:00:00Z",
        "end": "2025-12-31T23:59:59Z",
        "limit": "10000",
    }

    # Retry with exponential backoff
    parsed = None
    for attempt in range(1, 5):
        try:
            resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                rate_limit_hits += 1
                backoff = base_delay * (2 ** attempt)
                print(f"  429 Rate limited - backing off {backoff:.1f}s (attempt {attempt}/4)")
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            parsed = resp.json()
            break
        except Exception as e:
            print(f"  Error: {e}")
            errors += 1
            break

    if parsed is None:
        continue

    bars_dict = parsed.get("bars", {})

    rows_to_write = []
    for sym in batch:
        bars = bars_dict.get(sym)
        if not bars:
            symbols_empty += 1
            continue

        symbols_with_data += 1
        total_bars += len(bars)
        b0 = bars[0]
        print(f"  {sym}  {len(bars)} bars  first={b0['t']} o={b0['o']} h={b0['h']} l={b0['l']} c={b0['c']} v={b0['v']}")

        for bar in bars:
            rows_to_write.append([
                sym, bar["t"], bar["o"], bar["h"], bar["l"], bar["c"],
                bar["v"], bar["n"], bar["vw"]
            ])

    # Batch write to CSV
    if rows_to_write:
        with open(OUT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)

    time.sleep(base_delay)

elapsed = time.time() - t0

print()
print("=" * 60)
print("  DOWNLOAD COMPLETE")
print(f"  Symbols with data:  {symbols_with_data}")
print(f"  Symbols empty:      {symbols_empty}")
print(f"  Rate limit hits:    {rate_limit_hits}")
print(f"  Errors:             {errors}")
print(f"  Total bars saved:   {total_bars:,}")
print(f"  Elapsed:            {elapsed:.1f}s")
print(f"  Output file:        {OUT_FILE}")
print("=" * 60)
