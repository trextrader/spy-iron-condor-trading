"""
unpack_parquet.py — Convert parquet files to per-year CSVs.

Usage:
    python scripts/unpack_parquet.py                    # uses ./data
    python scripts/unpack_parquet.py /path/to/data      # custom data root
    python scripts/unpack_parquet.py --out /some/dir    # write CSVs to a different tree
    python scripts/unpack_parquet.py --dry-run          # print what would happen, no writes

Output structure:
    <ticker>/<year>/options.csv
    <ticker>/<year>/underlying.csv
"""
import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas pyarrow")

# Column used to split each file by year
DATE_COLS = {
    "options.parquet":    "date",
    "underlying.parquet": "date",
}


def unpack_by_year(src: Path, out_ticker: Path, filename: str,
                   date_col: str, dry_run: bool) -> tuple[int, int, int]:
    ok = skipped = errors = 0
    try:
        df = pd.read_parquet(src)
    except Exception as exc:
        print(f"  ERROR reading {src.relative_to(src.parent.parent)}: {exc}")
        return 0, 0, 1

    if date_col not in df.columns:
        print(f"  WARN  {src.name}: no '{date_col}' column — writing as single file")
        dst = out_ticker / filename.replace(".parquet", ".csv")
        if dst.exists():
            print(f"  skip  {dst}  (exists)")
            return 0, 1, 0
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dst, index=False)
        print(f"  {'would write' if dry_run else 'ok'}  {dst}  ({len(df):,} rows)")
        return 1, 0, 0

    df[date_col] = pd.to_datetime(df[date_col])
    stem = filename.replace(".parquet", ".csv")

    for year, group in df.groupby(df[date_col].dt.year):
        dst = out_ticker / str(year) / stem
        if dst.exists():
            print(f"  skip  {dst.parent.name}/{dst.parent.parent.name}/{dst.name}  (exists)")
            skipped += 1
            continue
        if dry_run:
            print(f"  would write  {out_ticker.name}/{year}/{stem}  ({len(group):,} rows)")
            ok += 1
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            group.to_csv(dst, index=False)
            print(f"  ok    {out_ticker.name}/{year}/{stem}  ({len(group):,} rows)")
            ok += 1
        except Exception as exc:
            print(f"  ERROR {out_ticker.name}/{year}/{stem}: {exc}")
            errors += 1

    return ok, skipped, errors


def unpack(data_root: Path, out_root: Path, dry_run: bool) -> None:
    parquet_files = sorted(data_root.rglob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found under {data_root}")
        return

    total_ok = total_skip = total_err = 0
    for src in parquet_files:
        ticker = src.parent.name
        out_ticker = out_root / ticker
        date_col = DATE_COLS.get(src.name, "date")
        ok, skip, err = unpack_by_year(src, out_ticker, src.name, date_col, dry_run)
        total_ok   += ok
        total_skip += skip
        total_err  += err

    action = "would write" if dry_run else "written"
    print(f"\nDone — {action}: {total_ok}  skipped: {total_skip}  errors: {total_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert parquet tree to per-year CSVs.")
    ap.add_argument("data_root", nargs="?", default="data",
                    help="Root folder containing ticker sub-dirs (default: ./data)")
    ap.add_argument("--out", default=None,
                    help="Output root (default: same as data_root)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written without writing anything")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_root  = Path(args.out).expanduser().resolve() if args.out else data_root

    if not data_root.is_dir():
        sys.exit(f"data_root not found: {data_root}")

    print(f"data : {data_root}")
    print(f"out  : {out_root}")
    if args.dry_run:
        print("mode : dry-run\n")
    print()

    unpack(data_root, out_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
