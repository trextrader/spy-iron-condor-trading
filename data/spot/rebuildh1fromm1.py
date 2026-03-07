#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


NY_TZ = "America/New_York"


def load_m1_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    ts_col = "datetime" if "datetime" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(NY_TZ)
    df = df.rename(columns={ts_col: "datetime"})

    # normalize common column names
    rename_map = {}
    if "o" in df.columns:
        rename_map["o"] = "open"
    if "h" in df.columns:
        rename_map["h"] = "high"
    if "l" in df.columns:
        rename_map["l"] = "low"
    if "c" in df.columns:
        rename_map["c"] = "close"
    if "v" in df.columns:
        rename_map["v"] = "volume"

    df = df.rename(columns=rename_map)

    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def rebuild_h1_from_m1(
    m1_path: str | Path,
    out_path: str | Path,
    mode: str = "full_hour_only",
) -> pd.DataFrame:
    """
    mode:
      - 'session_aligned' : 09:30,10:30,11:30,12:30,13:30,14:30,15:30
      - 'full_hour_only'  : 10:00,11:00,12:00,13:00,14:00,15:00
    """
    df = load_m1_csv(m1_path).copy()
    df = df.set_index("datetime")

    # Keep regular session only
    df = df.between_time("09:30", "15:59")

    grouped_rows = []

    for session_date, day_df in df.groupby(df.index.date):
        session_date = pd.Timestamp(session_date).tz_localize(NY_TZ)

        if mode == "session_aligned":
            anchors = [
                session_date.replace(hour=9, minute=30),
                session_date.replace(hour=10, minute=30),
                session_date.replace(hour=11, minute=30),
                session_date.replace(hour=12, minute=30),
                session_date.replace(hour=13, minute=30),
                session_date.replace(hour=14, minute=30),
                session_date.replace(hour=15, minute=30),
            ]

            intervals = [
                (anchors[0], anchors[0] + pd.Timedelta(minutes=60)),
                (anchors[1], anchors[1] + pd.Timedelta(minutes=60)),
                (anchors[2], anchors[2] + pd.Timedelta(minutes=60)),
                (anchors[3], anchors[3] + pd.Timedelta(minutes=60)),
                (anchors[4], anchors[4] + pd.Timedelta(minutes=60)),
                (anchors[5], anchors[5] + pd.Timedelta(minutes=60)),
                (anchors[6], session_date.replace(hour=16, minute=0)),
            ]

        elif mode == "full_hour_only":
            anchors = [
                session_date.replace(hour=10, minute=0),
                session_date.replace(hour=11, minute=0),
                session_date.replace(hour=12, minute=0),
                session_date.replace(hour=13, minute=0),
                session_date.replace(hour=14, minute=0),
                session_date.replace(hour=15, minute=0),
            ]

            intervals = [
                (anchors[0], anchors[0] + pd.Timedelta(minutes=60)),
                (anchors[1], anchors[1] + pd.Timedelta(minutes=60)),
                (anchors[2], anchors[2] + pd.Timedelta(minutes=60)),
                (anchors[3], anchors[3] + pd.Timedelta(minutes=60)),
                (anchors[4], anchors[4] + pd.Timedelta(minutes=60)),
                (anchors[5], session_date.replace(hour=16, minute=0)),
            ]

        else:
            raise ValueError("mode must be 'session_aligned' or 'full_hour_only'")

        for anchor, end_ts in intervals:
            bar = day_df[(day_df.index >= anchor) & (day_df.index < end_ts)]

            if bar.empty:
                grouped_rows.append(
                    {
                        "datetime": anchor,
                        "open": pd.NA,
                        "high": pd.NA,
                        "low": pd.NA,
                        "close": pd.NA,
                        "volume": 0,
                    }
                )
            else:
                grouped_rows.append(
                    {
                        "datetime": anchor,
                        "open": bar["open"].iloc[0],
                        "high": bar["high"].max(),
                        "low": bar["low"].min(),
                        "close": bar["close"].iloc[-1],
                        "volume": bar["volume"].sum(),
                    }
                )

    out = pd.DataFrame(grouped_rows)

    # Write back out in UTC to match your existing files
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.tz_convert("UTC")
    out = out.rename(
        columns={
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "volume": "v",
        }
    )
    out["vwap"] = pd.NA

    out.to_csv(out_path, index=False)
    return out


def discover_year_dirs(base_dir: Path) -> list[Path]:
    return sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 4
    )


def main() -> int:
    base_dir = Path(".")
    mode = "full_hour_only"   # change to "session_aligned" if needed

    year_dirs = discover_year_dirs(base_dir)
    if not year_dirs:
        print("[ERROR] No year folders found (e.g. 2020, 2021, ...)")
        return 1

    print("=" * 72)
    print("REBUILDING H1 FILES FROM M1 FILES")
    print("=" * 72)

    for year_dir in year_dirs:
        year = year_dir.name
        m1_path = year_dir / f"m1_{year}.csv"
        h1_path = year_dir / f"h1_{year}.csv"

        if not m1_path.exists():
            print(f"[SKIP] Missing source file: {m1_path}")
            continue

        print(f"[REBUILD] {h1_path} <- {m1_path} | mode={mode}")
        try:
            out = rebuild_h1_from_m1(
                m1_path=m1_path,
                out_path=h1_path,
                mode=mode,
            )
            non_null_rows = int(out["o"].notna().sum()) if "o" in out.columns else 0
            print(f"         wrote {len(out)} rows | populated rows={non_null_rows}")
        except Exception as e:
            print(f"[ERROR] {h1_path}: {e}")

    print("=" * 72)
    print("DONE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())