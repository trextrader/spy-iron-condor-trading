import argparse
from pathlib import Path
import pandas as pd

from intelligence.canonical_feature_registry import FEATURE_COLS_V22


def parse_args():
    p = argparse.ArgumentParser(description="Prune dataset to V2.2 schema with optional metadata columns")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--keep-timestamp", action="store_true", help="Keep timestamp column")
    p.add_argument("--keep-symbol", action="store_true", help="Keep symbol column")
    p.add_argument("--extra-keep", type=str, default="", help="Comma-separated extra columns to keep")
    p.add_argument("--chunksize", type=int, default=500_000)
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)

    base_keep = list(FEATURE_COLS_V22)
    if args.keep_timestamp:
        base_keep.insert(0, "timestamp")
    if args.keep_symbol:
        base_keep.insert(1 if args.keep_timestamp else 0, "symbol")

    if args.extra_keep:
        for col in [c.strip() for c in args.extra_keep.split(",") if c.strip()]:
            if col not in base_keep:
                base_keep.append(col)

    # Read header to validate
    header = pd.read_csv(inp, nrows=0).columns.tolist()
    missing = [c for c in base_keep if c not in header]
    if missing:
        raise SystemExit(f"Missing required columns: {missing[:10]}")

    # Stream write to avoid high memory use
    first = True
    for chunk in pd.read_csv(inp, chunksize=args.chunksize, usecols=base_keep, low_memory=False):
        chunk.to_csv(out, index=False, mode="w" if first else "a", header=first)
        first = False

    print(f"Wrote pruned dataset: {out} (cols={len(base_keep)})")


if __name__ == "__main__":
    main()
