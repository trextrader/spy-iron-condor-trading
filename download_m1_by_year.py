import os
import pandas as pd
import datetime as dt
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

NY = pytz.timezone("America/New_York")

# Load API keys
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")

if not ALPACA_KEY or not ALPACA_SECRET:
    raise ValueError("Missing ALPACA_KEY or ALPACA_SECRET environment variables.")

client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

# Load earliest-year map
EARLIEST_FILE = "data/AllOptionsData/earliest_years.csv"
df_earliest = pd.read_csv(EARLIEST_FILE)

# Load tickers
symbols_env = os.getenv("SYMBOLS")
if not symbols_env:
    raise ValueError("SYMBOLS environment variable is required.")

symbols = [s.strip().upper() for s in symbols_env.split(",")]

# Hard stop at last full year
LAST_FULL_YEAR = 2025


def clean_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to Regular Trading Hours (NY, 09:30–16:00, weekdays)."""
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_ny"] = df["timestamp"].dt.tz_convert(NY)

    # Weekdays only
    df = df[df["ts_ny"].dt.dayofweek < 5]

    # RTH 09:30–16:00
    t = df["ts_ny"].dt.time
    df = df[(t >= dt.time(9, 30)) & (t <= dt.time(16, 0))]

    df = df.drop(columns=["ts_ny"], errors="ignore")
    return df


def download_year(symbol: str, year: int, out_path: str) -> None:
    """Download one year of M1 bars for a symbol, save to CSV."""

    # Never go beyond last full year
    if year > LAST_FULL_YEAR:
        print(f"    [Skip] {symbol} {year} (beyond last full year {LAST_FULL_YEAR})")
        return

    # Skip if file already exists
    if os.path.exists(out_path):
        print(f"    [Skip] {symbol} {year} already exists")
        return

    start = dt.datetime(year, 1, 1, tzinfo=pytz.utc)
    end = dt.datetime(year + 1, 1, 1, tzinfo=pytz.utc)

    print(f"    [Fetch] {symbol} {year}...")

    try:
        bars = client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
                end=end,
                adjustment="split",
            )
        )
    except Exception as e:
        msg = str(e)
        if "subscription does not permit querying recent SIP data" in msg:
            print(f"    [Skip] {symbol} {year} blocked by SIP delay")
            return
        print(f"    [Error] {symbol} {year}: {e}")
        return

    df = bars.df.reset_index()

    if df.empty:
        print(f"    [Skip] No data for {symbol} {year}")
        return

    df = clean_rth(df)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df.to_csv(out_path, index=False)

    print(f"    [Saved] {symbol} {year}: {len(df)} rows")


def main() -> None:
    print("\n========================================")
    print(" Multi‑Ticker M1 Downloader (Year‑Split)")
    print("========================================\n")

    for symbol in symbols:
        print(f"\n=== {symbol} ===")

        # Lookup earliest year
        row = df_earliest[df_earliest["symbol"] == symbol]
        if row.empty or pd.isna(row["earliest_year"].iloc[0]):
            print(f"  [Skip] No earliest-year entry for {symbol}")
            continue

        earliest_year = int(row["earliest_year"].iloc[0])
        print(f"  Earliest year: {earliest_year}")

        # Clamp to LAST_FULL_YEAR
        if earliest_year > LAST_FULL_YEAR:
            print(f"  [Skip] {symbol} earliest year {earliest_year} > {LAST_FULL_YEAR}")
            continue

        # Create folder
        folder = f"data/AllOptionsData/{symbol.lower()}"
        os.makedirs(folder, exist_ok=True)

        # Loop from earliest_year → LAST_FULL_YEAR
        for year in range(earliest_year, LAST_FULL_YEAR + 1):
            out_path = f"{folder}/{year}_m1.csv"
            download_year(symbol, year, out_path)

    print("\n========================================")
    print(" M1 Download Complete")
    print("========================================\n")


if __name__ == "__main__":
    main()
