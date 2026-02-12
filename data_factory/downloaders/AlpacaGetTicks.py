from datetime import datetime, timezone
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

client = StockHistoricalDataClient(
    "PKWCQL536DJKE7EJCP5OEETWE2",
    "8PK6xfXx13Hqna2ryjHyQCMAf6D1zj6kNGE96CjnzmKM",
)

start = datetime(2025, 1, 2, tzinfo=timezone.utc)
end = datetime.now(timezone.utc)

# ── 1-minute bars (SIP feed required for ETFs like SPY) ──────────────
print(f"Fetching 1-min bars  {start.date()} → {end.date()} ...")
bars_req = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
    start=start,
    end=end,
    feed=DataFeed.SIP,
)
bars = client.get_stock_bars(bars_req).df
print(f"  bars: {len(bars):,} rows")

# ── quotes ────────────────────────────────────────────────────────────
print("Fetching quotes ...")
quotes_req = StockQuotesRequest(
    symbol_or_symbols="SPY",
    start=start,
    end=end,
    feed=DataFeed.SIP,
)
quotes = client.get_stock_quotes(quotes_req).df
print(f"  quotes: {len(quotes):,} rows")

# Drop the symbol level from the MultiIndex so both frames align on timestamp
if isinstance(bars.index, pd.MultiIndex):
    bars = bars.droplevel("symbol")
if isinstance(quotes.index, pd.MultiIndex):
    quotes = quotes.droplevel("symbol")

# Resample quotes to 1-minute averages
quotes_1m = quotes.resample("1min").agg({
    "bid_price": "mean",
    "ask_price": "mean",
})

# Align with bars
combined = bars.join(quotes_1m, how="left")

# Add mid_price derived column
combined["mid_price"] = (combined["bid_price"] + combined["ask_price"]) / 2

# ── Save ──────────────────────────────────────────────────────────────
out_path = "SPY_1min_ticks.csv"
combined.to_csv(out_path)
print(f"Saved {len(combined):,} rows → {out_path}")
print(combined.head())
