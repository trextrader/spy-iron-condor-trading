"""
unpack_parquet.py — Convert all options/underlying parquet files to CSV.

Usage:
    python scripts/unpack_parquet.py                    # uses ./data
    python scripts/unpack_parquet.py /path/to/data      # custom data root
    python scripts/unpack_parquet.py --out /some/dir    # write CSVs to a different tree
    python scripts/unpack_parquet.py --dry-run          # print what would happen, no writes

Output: <ticker>/options.csv  and  <ticker>/underlying.csv  (mirroring parquet layout)
"""
import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas pyarrow")


def unpack(data_root: Path, out_root: Path, dry_run: bool) -> None:
    parquet_files = sorted(data_root.rglob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found under {data_root}")
        return

    ok = skipped = errors = 0
    for src in parquet_files:
        # Mirror the relative path but swap suffix
        rel = src.relative_to(data_root)
        dst = (out_root / rel).with_suffix(".csv")

        if dst.exists():
            print(f"  skip  {rel}  (CSV exists)")
            skipped += 1
            continue

        if dry_run:
            print(f"  would write  {rel.with_suffix('.csv')}")
            ok += 1
            continue

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_parquet(src)
            df.to_csv(dst, index=False)
            print(f"  ok    {rel}  →  {dst.relative_to(out_root)}  ({len(df):,} rows)")
            ok += 1
        except Exception as exc:
            print(f"  ERROR {rel}: {exc}")
            errors += 1

    action = "would write" if dry_run else "written"
    print(f"\nDone — {action}: {ok}  skipped: {skipped}  errors: {errors}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert parquet tree to CSV.")
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
