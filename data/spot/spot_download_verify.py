#!/usr/bin/env python3
"""
check_missing_intraday_rows.py

Scans SPY intraday CSV files and reports missing regular-session bars,
excluding weekends and NYSE market holidays.

Expected filenames:
    m1_2020.csv
    m5_2020.csv
    m15_2020.csv
    h1_2020.csv
    ...

Assumptions:
- CSV has a timestamp column.
- Timestamps are parseable and represent market-time bars.
- Regular market session only: 09:30 to 16:00 America/New_York.
- For 60-minute bars, expected anchors are:
      10:00, 11:00, 12:00, 13:00, 14:00, 15:00
  The 09:30-10:00 partial bar is intentionally excluded.
- For 15-minute bars: 26 bars/day.
- For 5-minute bars: 78 bars/day.
- For 1-minute bars: 390 bars/day.

Outputs:
- missing_rows_report.csv
- file_summary_report.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pandas_market_calendars as mcal


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(".")
OUTPUT_MISSING_CSV = "missing_rows_report.csv"
OUTPUT_SUMMARY_CSV = "file_summary_report.csv"

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "datetime",
    "date",
    "time",
    "Timestamp",
    "Datetime",
    "Date",
    "Time",
]

FILE_SPECS: Dict[str, Dict[str, object]] = {
    "m1": {
        "freq": "1min",
        "bars_per_day": 390,
        "session_minutes": 1,
    },
    "m5": {
        "freq": "5min",
        "bars_per_day": 78,
        "session_minutes": 5,
    },
    "m15": {
        "freq": "15min",
        "bars_per_day": 26,
        "session_minutes": 15,
    },
    "h1": {
        "freq": "60min",
        "bars_per_day": 6,
        "session_minutes": 60,
    },
}

NYSE_CAL = mcal.get_calendar("NYSE")
NY_TZ = "America/New_York"


# ============================================================
# HELPERS
# ============================================================

def find_timestamp_column(df: pd.DataFrame) -> str:
    for col in TIMESTAMP_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find timestamp column. Tried: {TIMESTAMP_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def parse_timestamps(series: pd.Series) -> pd.Series:
    """
    Parse timestamps robustly and convert/localize to America/New_York.
    Also normalizes to minute precision.
    """
    raw = series.astype(str)

    # Path 1: parse as UTC-aware
    ts_utc = pd.to_datetime(raw, errors="coerce", utc=True)

    # Path 2: parse as naive local NY time
    ts_naive = pd.to_datetime(raw, errors="coerce")
    if getattr(ts_naive.dt, "tz", None) is None:
        ts_naive = ts_naive.dt.tz_localize(
            NY_TZ,
            nonexistent="shift_forward",
            ambiguous="NaT",
        )
    else:
        ts_naive = ts_naive.dt.tz_convert(NY_TZ)

    ts_utc_ny = ts_utc.dt.tz_convert(NY_TZ)

    if ts_utc_ny.isna().all() and ts_naive.isna().all():
        raise ValueError("Failed to parse all timestamps.")

    def score_session_fit(ts: pd.Series) -> int:
        valid = ts.dropna()
        if valid.empty:
            return -1
        h = valid.dt.hour
        m = valid.dt.minute
        mask = ((h > 9) | ((h == 9) & (m >= 30))) & (h <= 16)
        return int(mask.sum())

    score_utc = score_session_fit(ts_utc_ny)
    score_naive = score_session_fit(ts_naive)

    chosen = ts_utc_ny if score_utc > score_naive else ts_naive
    chosen = chosen.dt.floor("min")

    bad = int(chosen.isna().sum())
    if bad > 0:
        raise ValueError(f"Failed to parse/localize {bad} timestamps.")

    return chosen


def detect_stamp_mode(
    actual_ts_unique: pd.DatetimeIndex,
    year: int,
    tf_key: str,
) -> str:
    """
    Auto-detect whether timestamps are bar-start or bar-end stamped.
    Chooses the convention with the larger overlap.
    """
    exp_start = build_expected_index_for_year(year, tf_key, "start")
    exp_end = build_expected_index_for_year(year, tf_key, "end")

    overlap_start = len(actual_ts_unique.intersection(exp_start))
    overlap_end = len(actual_ts_unique.intersection(exp_end))

    return "end" if overlap_end > overlap_start else "start"


def expected_intraday_index_for_day(
    day: pd.Timestamp,
    tf_key: str,
    stamp_mode: str = "start",
) -> pd.DatetimeIndex:
    """
    Return expected timestamps for one trading day in America/New_York.

    stamp_mode:
      - 'start': bars are stamped by bar start
      - 'end'  : bars are stamped by bar end
    """
    if tf_key not in FILE_SPECS:
        raise ValueError(f"Unsupported timeframe: {tf_key}")

    if tf_key == "m1":
        base = pd.date_range(
            start=day.replace(hour=9, minute=30, second=0),
            end=day.replace(hour=15, minute=59, second=0),
            freq="1min",
            tz=NY_TZ,
        )
        return base + pd.Timedelta(minutes=1) if stamp_mode == "end" else base

    if tf_key == "m5":
        base = pd.date_range(
            start=day.replace(hour=9, minute=30, second=0),
            end=day.replace(hour=15, minute=55, second=0),
            freq="5min",
            tz=NY_TZ,
        )
        return base + pd.Timedelta(minutes=5) if stamp_mode == "end" else base

    if tf_key == "m15":
        base = pd.date_range(
            start=day.replace(hour=9, minute=30, second=0),
            end=day.replace(hour=15, minute=45, second=0),
            freq="15min",
            tz=NY_TZ,
        )
        return base + pd.Timedelta(minutes=15) if stamp_mode == "end" else base

    if tf_key == "h1":
        # Full-hour bars only; omit 09:30 partial bar.
        base = pd.DatetimeIndex([
            day.replace(hour=10, minute=0, second=0),
            day.replace(hour=11, minute=0, second=0),
            day.replace(hour=12, minute=0, second=0),
            day.replace(hour=13, minute=0, second=0),
            day.replace(hour=14, minute=0, second=0),
            day.replace(hour=15, minute=0, second=0),
        ])
        return base + pd.Timedelta(minutes=60) if stamp_mode == "end" else base

    raise ValueError(f"Unhandled timeframe: {tf_key}")


def get_trading_days_for_year(year: int) -> pd.DatetimeIndex:
    """
    Get NYSE valid trading days for a full calendar year in America/New_York.

    Important:
    schedule.index is already date-like trading days. Do NOT localize to UTC and
    convert to NY, or you can shift dates backward.
    """
    schedule = NYSE_CAL.schedule(
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
    )

    days = pd.DatetimeIndex(schedule.index)
    days = pd.DatetimeIndex(
        [pd.Timestamp(d).tz_localize(NY_TZ).normalize() for d in days]
    )
    return days

def build_expected_index_for_year(
    year: int,
    tf_key: str,
    stamp_mode: str = "start",
) -> pd.DatetimeIndex:
    trading_days = get_trading_days_for_year(year)
    pieces: List[pd.DatetimeIndex] = []

    for day in trading_days:
        pieces.append(expected_intraday_index_for_day(day, tf_key, stamp_mode))

    if not pieces:
        return pd.DatetimeIndex([], tz=NY_TZ)

    full_index = pieces[0]
    for idx in pieces[1:]:
        full_index = full_index.append(idx)
    return full_index


def infer_tf_and_year_from_filename(path: Path) -> Tuple[str, int]:
    """
    Expected forms:
      m1_2020.csv
      m5_2021.csv
      m15_2022.csv
      h1_2023.csv
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {path.name}")

    tf_key = parts[0].lower()
    year = int(parts[1])

    if tf_key not in FILE_SPECS:
        raise ValueError(f"Unknown timeframe key '{tf_key}' in {path.name}")

    return tf_key, year


def load_and_normalize_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = find_timestamp_column(df)
    df[ts_col] = parse_timestamps(df[ts_col])
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = df["timestamp"].dt.floor("min")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================
# MAIN CHECKER
# ============================================================

def analyze_file(path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    tf_key, year = infer_tf_and_year_from_filename(path)
    spec = FILE_SPECS[tf_key]

    df = load_and_normalize_csv(path)

    actual_ts = pd.DatetimeIndex(df["timestamp"])
    actual_ts_unique = pd.DatetimeIndex(sorted(pd.unique(actual_ts)))

    stamp_mode = detect_stamp_mode(actual_ts_unique, year, tf_key)
    expected_ts = build_expected_index_for_year(year, tf_key, stamp_mode)

    missing_ts = expected_ts.difference(actual_ts_unique)
    extra_ts = actual_ts_unique.difference(expected_ts)

    dup_mask = df["timestamp"].duplicated(keep=False)
    dup_df = df.loc[dup_mask, ["timestamp"]].copy()

    missing_records: List[Dict[str, object]] = []
    for ts in missing_ts:
        missing_records.append(
            {
                "file": path.name,
                "timeframe": tf_key,
                "year": year,
                "stamp_mode": stamp_mode,
                "issue_type": "missing_bar",
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
                "trade_date": ts.strftime("%Y-%m-%d"),
            }
        )

    extra_records: List[Dict[str, object]] = []
    for ts in extra_ts:
        extra_records.append(
            {
                "file": path.name,
                "timeframe": tf_key,
                "year": year,
                "stamp_mode": stamp_mode,
                "issue_type": "unexpected_timestamp",
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
                "trade_date": ts.strftime("%Y-%m-%d"),
            }
        )

    dup_records: List[Dict[str, object]] = []
    if not dup_df.empty:
        for ts in dup_df["timestamp"]:
            dup_records.append(
                {
                    "file": path.name,
                    "timeframe": tf_key,
                    "year": year,
                    "stamp_mode": stamp_mode,
                    "issue_type": "duplicate_timestamp",
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "trade_date": ts.strftime("%Y-%m-%d"),
                }
            )

    issue_df = pd.DataFrame(missing_records + extra_records + dup_records)

    trading_days = get_trading_days_for_year(year)
    expected_bars = len(expected_ts)

    summary = {
        "file": path.name,
        "timeframe": tf_key,
        "year": year,
        "stamp_mode": stamp_mode,
        "rows_in_file": int(len(df)),
        "unique_timestamps": int(len(actual_ts_unique)),
        "expected_bars": int(expected_bars),
        "expected_bars_per_day": int(spec["bars_per_day"]),
        "trading_days": int(len(trading_days)),
        "missing_bar_count": int(len(missing_ts)),
        "unexpected_timestamp_count": int(len(extra_ts)),
        "duplicate_timestamp_count": int(len(dup_df)),
        "status": "OK" if len(missing_ts) == 0 and len(extra_ts) == 0 and len(dup_df) == 0 else "ISSUES_FOUND",
    }

    return issue_df, summary


def main() -> int:
    # Scan recursively to support yearly subfolders
    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    target_files = []

    for p in csv_files:
        try:
            tf_key, _year = infer_tf_and_year_from_filename(p)
            if tf_key in FILE_SPECS:
                target_files.append(p)
        except Exception:
            continue

    if not target_files:
        print("No matching files found like m1_2020.csv, m5_2021.csv, m15_2022.csv, h1_2023.csv")
        return 1

    all_issue_dfs: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    print("=" * 72)
    print("CHECKING INTRADAY CSV FILES FOR MISSING TRADING-SESSION ROWS")
    print("=" * 72)

    for path in target_files:
        print(f"Processing: {path.name}")
        try:
            issue_df, summary = analyze_file(path)
            all_issue_dfs.append(issue_df)
            summary_rows.append(summary)

            print(
                f"  status={summary['status']} | "
                f"rows={summary['rows_in_file']} | "
                f"expected={summary['expected_bars']} | "
                f"missing={summary['missing_bar_count']} | "
                f"extra={summary['unexpected_timestamp_count']} | "
                f"dupes={summary['duplicate_timestamp_count']}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            summary_rows.append(
                {
                    "file": path.name,
                    "timeframe": "",
                    "year": "",
                    "rows_in_file": "",
                    "unique_timestamps": "",
                    "expected_bars": "",
                    "expected_bars_per_day": "",
                    "trading_days": "",
                    "missing_bar_count": "",
                    "unexpected_timestamp_count": "",
                    "duplicate_timestamp_count": "",
                    "status": f"ERROR: {e}",
                }
            )

    summary_df = pd.DataFrame(summary_rows)

        if all_issue_dfs:
        non_empty = [x for x in all_issue_dfs if not x.empty]
        issues_df = (
            pd.concat(non_empty, ignore_index=True)
            if non_empty
            else pd.DataFrame(
                columns=[
                    "file",
                    "timeframe",
                    "year",
                    "stamp_mode",
                    "issue_type",
                    "timestamp",
                    "trade_date",
                ]
            )
        )
    else:
        issues_df = pd.DataFrame(
            columns=[
                "file",
                "timeframe",
                "year",
                "stamp_mode",
                "issue_type",
                "timestamp",
                "trade_date",
            ]
        )

    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    issues_df.to_csv(OUTPUT_MISSING_CSV, index=False)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Summary report : {OUTPUT_SUMMARY_CSV}")
    print(f"Issue report   : {OUTPUT_MISSING_CSV}")

    if not issues_df.empty:
        print()
        print("Issue counts by type:")
        print(issues_df["issue_type"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())