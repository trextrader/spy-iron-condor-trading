#!/usr/bin/env python3
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Postcompute chain volume aggregates for v3.0 dataset.")
    parser.add_argument("--input", "-i", required=True, help="Input precomputed CSV")
    parser.add_argument("--output", "-o", required=True, help="Output CSV with chain volume aggregates")
    args = parser.parse_args()

    print(f"[LOAD] {args.input}")
    df = pd.read_csv(args.input, low_memory=False)

    # Ensure _date exists
    df['_date'] = pd.to_datetime(df['timestamp']).dt.date

    print("[AGG] Computing chain volume aggregates...")

    vol_agg = df.groupby('_date').apply(
        lambda g: pd.Series({
            'Options_Total_Volume': g['volume'].sum(),
            'Options_Put_Volume':   g.loc[g['call_put'] == 'P', 'volume'].sum(),
            'Options_Call_Volume':  g.loc[g['call_put'] == 'C', 'volume'].sum(),
        })
    ).reset_index()

    # Normalize types for merge
    df['_date'] = pd.to_datetime(df['_date']).dt.date
    vol_agg['_date'] = pd.to_datetime(vol_agg['_date']).dt.date

    print("[MERGE] Merging aggregates into dataset...")

    # Merge with suffixes to avoid accidental overwrites
    df = df.merge(vol_agg, on="_date", how="left", suffixes=("", "_new"))

    # Overwrite canonical columns with computed values
    for col in ["Options_Total_Volume", "Options_Put_Volume", "Options_Call_Volume"]:
        new_col = col + "_new"
        if new_col in df.columns:
            df[col] = df[new_col]
            df = df.drop(columns=[new_col])

    # Drop helper
    df = df.drop(columns=["_date"], errors="ignore")

    print(f"[SAVE] Writing output to {args.output}")
    df.to_csv(args.output, index=False)

    print("[DONE] Chain volume aggregates added successfully.")

if __name__ == "__main__":
    main()
