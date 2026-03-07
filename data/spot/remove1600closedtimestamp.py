#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

NY_TZ = "America/New_York"


def find_timestamp_column(df: pd.DataFrame) -> str:
    for col in ["datetime", "timestamp", "date", "time", "Datetime", "Timestamp"]:
        if col in df.columns:
            return col
    raise ValueError(f"No timestamp column found. Columns: {list(df.columns)}")


def clean_intraday_file(path: Path) -> None:
    df = pd.read_csv(path)
    ts_col = find_timestamp_column(df)

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(NY_TZ)
    df[ts_col] = df[ts_col].dt.floor("min")

    stem = path.stem.lower()

    if stem.startswith("m1_") or stem.startswith("m5_") or stem.startswith("m15_"):
        before = len(df)
        df = df[df[ts_col].dt.strftime("%H:%M") != "16:00"].copy()
        after = len(df)

        # write back in UTC to preserve original file convention
        df[ts_col] = df[ts_col].dt.tz_convert("UTC")
        df.to_csv(path, index=False)

        print(f"{path}: removed {before - after} rows")
    else:
        print(f"{path}: skipped")


def main() -> int:
    base_dir = Path(".")
    year_dirs = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 4
    )

    if not year_dirs:
        print("No year folders found.")
        return 1

    print("=" * 72)
    print("REMOVING 16:00 EXTRA ROWS FROM M1 / M5 / M15")
    print("=" * 72)

    for year_dir in year_dirs:
        year = year_dir.name
        for tf in ["m1", "m5", "m15"]:
            path = year_dir / f"{tf}_{year}.csv"
            if path.exists():
                clean_intraday_file(path)
            else:
                print(f"{path}: missing")

    print("=" * 72)
    print("DONE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())