import os
import time
import pandas as pd
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

print("Looking for spy_symbols.txt in:", os.getcwd())

symbol_file = "spy_symbols.txt"

if not os.path.exists(symbol_file):
    raise RuntimeError(f"ERROR: {symbol_file} not found in {os.getcwd()}")

with open(symbol_file, "r") as f:
    spy_symbols = f.read().split()

if len(spy_symbols) == 0:
    raise RuntimeError("ERROR: spy_symbols.txt is empty or unreadable.")

print(f"Loaded {len(spy_symbols)} SPY option symbols")

API_KEY = "PKWCQL536DJKE7EJCP5OEETWE2"
API_SECRET = "8PK6xfXx13Hqna2ryjHyQCMAf6D1zj6kNGE96CjnzmKM"

client = OptionHistoricalDataClient(API_KEY, API_SECRET)

# ----------------------------------------------------------
# Load your 1616 SPY symbols
# ----------------------------------------------------------
with open("spy_symbols.txt") as f:
    spy_symbols = f.read().split()

print(f"Loaded {len(spy_symbols)} SPY option symbols")

# ----------------------------------------------------------
# Pull 1-minute bars for last 2 days for liquidity ranking
# ----------------------------------------------------------
START = "2026-02-07"
END   = "2026-02-09"

BATCH_SIZE = 10
SLEEP_SECONDS = 1

frames = []

def fetch_bars(symbols, start, end):
    req = OptionBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    return client.get_option_bars(req)

for i in range(0, len(spy_symbols), BATCH_SIZE):
    batch = spy_symbols[i:i+BATCH_SIZE]
    print("Batch:", batch)

    try:
        bars = fetch_bars(batch, START, END)
        df = bars.df
        if not df.empty:
            frames.append(df)
    except Exception as ex:
        print("Error:", ex)

    time.sleep(SLEEP_SECONDS)

# ----------------------------------------------------------
# Combine and compute liquidity
# ----------------------------------------------------------
df = pd.concat(frames, ignore_index=True)

# Liquidity score: volume + trade_count
df["liq"] = df["volume"].fillna(0) + df["trade_count"].fillna(0)

# Rank by liquidity
top100 = (
    df.groupby("symbol")["liq"]
    .sum()
    .sort_values(ascending=False)
    .head(100)
    .index
    .tolist()
)

print("Top 100 most liquid SPY options:")
print(top100)

# Save for future use
with open("spy_top100.txt", "w") as f:
    f.write("\n".join(top100))

print("Saved spy_top100.txt")
