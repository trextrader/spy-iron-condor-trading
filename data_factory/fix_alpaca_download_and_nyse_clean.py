# C:\SPYOptionTrader_test\data_factory\fix_alpaca_download_and_nyse_clean.py
import os
import sys
import re
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import pytz

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


NY = pytz.timezone("America/New_York")
UTC = pytz.utc


def _parse_iso_dt(s: str) -> dt.datetime:
    x = pd.to_datetime(s, utc=True, errors="raise")
    return x.to_pydatetime()


def parse_gaps_report(path: str):
    """
    Parses your gaps_2025.noweekends.txt produced by check_trading_calendar_gaps.py.

    Returns:
      missing_dates: list[date]
      extra_dates:   list[date]
    """
    txt = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    missing = []
    extra = []

    mode = None
    for line in txt:
        line = line.strip()
        if not line:
            continue

        if line.startswith("=== MISSING NYSE SESSIONS"):
            mode = "missing"
            continue
        if line.startswith("=== EXTRA NON-NYSE DATES"):
            mode = "extra"
            continue
        if line.startswith("count:"):
            continue

        m = re.match(r"^\d{4}-\d{2}-\d{2}$", line)
        if not m:
            continue

        d = dt.date.fromisoformat(line)
        if mode == "missing":
            missing.append(d)
        elif mode == "extra":
            extra.append(d)

    return missing, extra


def get_client():
    """
    Uses either environment vars:
      APCA_API_KEY_ID / APCA_API_SECRET_KEY
    OR (if present) your project's core.config.RunConfig().
    """
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY")
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET")

    if not key or not sec:
        # Try your project config if available
        try:
            # Make project root importable: ...\SPYOptionTrader_test\
            here = Path(__file__).resolve()
            project_root = here.parents[1]
            sys.path.insert(0, str(project_root))
            from core.config import RunConfig  # type: ignore

            rc = RunConfig()
            key = key or getattr(rc, "alpaca_key", None)
            sec = sec or getattr(rc, "alpaca_secret", None)
        except Exception:
            pass

    if not key or not sec:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "in your environment, or ensure core.config.RunConfig provides alpaca_key/alpaca_secret."
        )

    return StockHistoricalDataClient(key, sec)


def fetch_day_bars(client: StockHistoricalDataClient, symbol: str, day: dt.date, timeframe: TimeFrame) -> pd.DataFrame:
    """
    Fetch one NYSE session day as RTH-only bars:
      09:30:00 to 16:00:00 America/New_York (inclusive end)

    Returns a DataFrame with timestamp (UTC) + OHLCV (+ vwap/trade_count if provided).
    """
    start_ny = NY.localize(dt.datetime(day.year, day.month, day.day, 9, 30, 0))
    end_ny = NY.localize(dt.datetime(day.year, day.month, day.day, 16, 0, 0))

    start_utc = start_ny.astimezone(UTC)
    end_utc = end_ny.astimezone(UTC)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_utc,
        end=end_utc,
        adjustment="split",
    )

    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()

    if df.empty:
        return df

    # Normalize expected columns
    # Alpaca returns columns: symbol, timestamp, open, high, low, close, volume, trade_count, vwap
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df


def session_grid_fill(df: pd.DataFrame, symbol: str, tf_seconds: int) -> pd.DataFrame:
    """
    Enforce:
      - weekdays only
      - RTH 09:30–16:00 NY
      - full per-day timestamp grid (NY time) converted to UTC
      - interpolate price-like columns only
      - volume/trade_count -> int, fill missing as 0
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    df["ts_ny"] = df["timestamp"].dt.tz_convert(NY)
    df = df[df["ts_ny"].dt.dayofweek < 5]

    t = df["ts_ny"].dt.time
    rth_start = dt.time(9, 30)
    rth_end = dt.time(16, 0)
    df = df[(t >= rth_start) & (t <= rth_end)]
    if df.empty:
        return df.drop(columns=["ts_ny"], errors="ignore")

    freq = f"{tf_seconds}s"
    days = sorted(pd.Series(df["ts_ny"].dt.normalize().unique()).tolist())

    grids = []
    for day in days:
        day_ny = pd.Timestamp(day).tz_convert(NY)
        start = day_ny.replace(hour=9, minute=30, second=0)
        end = day_ny.replace(hour=16, minute=0, second=0)
        grids.append(pd.date_range(start=start, end=end, freq=freq, tz=NY))

    full_ny = grids[0].append(grids[1:]) if len(grids) > 1 else grids[0]
    full_utc = full_ny.tz_convert(UTC)

    df = df.set_index("timestamp").reindex(full_utc)

    price_cols = [c for c in ["open", "high", "low", "close", "vwap"] if c in df.columns]
    if price_cols:
        df[price_cols] = df[price_cols].interpolate(method="time", limit_direction="both")
        for c in price_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    for c in ["volume", "trade_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(0).clip(lower=0).astype("int64")

    df["symbol"] = symbol

    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df.drop(columns=["ts_ny"], errors="ignore")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframes", default="1", help='Comma list. For this backfill script, use "1" for 1-minute.')
    ap.add_argument("--start", required=True, help="ISO, e.g. 2025-01-02T00:00:00Z")
    ap.add_argument("--end", required=True, help="ISO, e.g. 2025-12-31T23:59:59Z")
    ap.add_argument("--backfill-gaps", required=True, help="Path to gaps_*.txt from check_trading_calendar_gaps.py")
    ap.add_argument("--out-dir", required=True, help="Directory to write outputs")
    ap.add_argument("--report", required=True, help="Write a human-readable report here")
    ap.add_argument("--chunksize", type=int, default=200000, help="Not used for Alpaca fetch; kept for future")
    args = ap.parse_args()

    if sys.version_info < (3, 10):
        raise RuntimeError("Use Python 3.10+ (recommended: py -3.12 ...)")

    symbol = args.symbol.strip().upper()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_utc = _parse_iso_dt(args.start)
    end_utc = _parse_iso_dt(args.end)

    # Only implement 1-minute in this pipeline (matches your 391-min session logic)
    tf = TimeFrame(1, TimeFrameUnit.Minute)
    tf_seconds = 60

    missing_dates, extra_dates = parse_gaps_report(args.backfill_gaps)

    client = get_client()

    report_lines = []
    report_lines.append(f"symbol={symbol}")
    report_lines.append(f"range_utc={start_utc.isoformat()} .. {end_utc.isoformat()}")
    report_lines.append(f"gaps_file={args.backfill_gaps}")
    report_lines.append(f"missing_dates={len(missing_dates)} extra_dates={len(extra_dates)}")
    if extra_dates:
        report_lines.append("extra_dates_in_gaps_file (FYI): " + ", ".join(d.isoformat() for d in extra_dates))

    all_days = []
    ok = 0
    empty = 0
    failed = 0

    for i, day in enumerate(missing_dates, 1):
        try:
            df = fetch_day_bars(client, symbol, day, tf)
            if df.empty:
                empty += 1
                report_lines.append(f"[EMPTY] {day.isoformat()} (no bars returned)")
                continue

            df = session_grid_fill(df, symbol=symbol, tf_seconds=tf_seconds)

            # Expected RTH minutes inclusive: 09:30..16:00 => 391 rows
            report_lines.append(f"[OK] {day.isoformat()} rows={len(df)} first={df['timestamp'].iloc[0]} last={df['timestamp'].iloc[-1]}")
            ok += 1

            # Write per-day file
            day_path = out_dir / f"{symbol}_1m_{day.isoformat()}.csv"
            df.to_csv(day_path, index=False)

            all_days.append(df)

            if i % 5 == 0:
                print(f"backfill progress: {i}/{len(missing_dates)} ok={ok} empty={empty} failed={failed}")

        except Exception as e:
            failed += 1
            report_lines.append(f"[FAIL] {day.isoformat()} err={repr(e)}")

    # Write combined backfill file
    combined_path = out_dir / f"{symbol}_1m_backfill_missing_days.csv"
    if all_days:
        combined = pd.concat(all_days, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        combined.to_csv(combined_path, index=False)
        report_lines.append(f"combined_backfill_csv={combined_path}")
        report_lines.append(f"combined_rows={len(combined)}")

    report_lines.append(f"SUMMARY ok={ok} empty={empty} failed={failed}")
    Path(args.report).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"WROTE report: {args.report}")
    if all_days:
        print(f"WROTE combined backfill: {combined_path}")
    else:
        print("No backfill data written (all missing days returned empty or failed).")


if __name__ == "__main__":
    main()
