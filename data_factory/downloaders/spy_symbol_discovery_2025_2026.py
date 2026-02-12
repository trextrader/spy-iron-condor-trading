import time
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

# ----------------------------------------------------------
# Alpaca credentials
# ----------------------------------------------------------
API_KEY = "PKWCQL536DJKE7EJCP5OEETWE2"
API_SECRET = "8PK6xfXx13Hqna2ryjHyQCMAf6D1zj6kNGE96CjnzmKM"

client = OptionHistoricalDataClient(API_KEY, API_SECRET)

# ----------------------------------------------------------
# Load Top-100 symbols (SPY25 + SPY26)
# ----------------------------------------------------------
with open("spy_top100.txt") as f:
    top100 = [s.strip() for s in f.read().split() if s.strip()]

# Filter to SPY25 and SPY26 only
top100 = [s for s in top100 if s.startswith("SPY25") or s.startswith("SPY26")]

print(f"Loaded {len(top100)} SPY25/26 symbols")

# ----------------------------------------------------------
# Historical window: Feb 2025 → 15 minutes ago
# ----------------------------------------------------------
START = "2025-02-01"
END = (datetime.utcnow() - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"Downloading from {START} → {END}")

# ----------------------------------------------------------
# Free-plan safe parameters
# ----------------------------------------------------------
BATCH_SIZE = 5
WINDOW_DAYS = 3
SLEEP_SECONDS = 1

# ----------------------------------------------------------
# Helper: fetch bars for a batch of symbols
# ----------------------------------------------------------
def fetch_bars(symbols, start, end):
    req = OptionBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    return client.get_option_bars(req)

# ----------------------------------------------------------
# Main loop
# ----------------------------------------------------------
frames = []
date_range = pd.date_range(START, END, freq=f"{WINDOW_DAYS}D")

for i in range(0, len(top100), BATCH_SIZE):
    batch = top100[i:i+BATCH_SIZE]
    print(f"\nBatch {i//BATCH_SIZE+1}: {batch}")

    for j in range(len(date_range)-1):
        s = date_range[j]
        e = date_range[j+1]

        print(f"  Fetching {s} → {e}")

        try:
            bars = fetch_bars(batch, s, e)
            df = bars.df
            if not df.empty:
                frames.append(df)
        except Exception as ex:
            print("  Error:", ex)

        time.sleep(SLEEP_SECONDS)

# ----------------------------------------------------------
# Save results
# ----------------------------------------------------------
if frames:
    final_df = pd.concat(frames, ignore_index=True)
    final_df.to_csv("spy25_26_top100_m1_history.csv", index=False)
    print("\nSaved spy25_26_top100_m1_history.csv")
else:
    print("\nNo data returned.")
