import argparse
import pandas as pd
import numpy as np
import sys
import os

def pct_true(s):
    s = s.dropna()
    if len(s) == 0:
        return 0.0
    return (s.astype(float) > 0.5).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to processed CSV")
    ap.add_argument("--rows", type=int, default=None, help="Optional: only read first N rows")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[Error] File not found: {args.data}")
        sys.exit(1)

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data, nrows=args.rows)

    cols = ["exec_allow", "risk_override", "ivr", "spread_ratio",
            "friction_ratio", "chaos_membership", "position_size_mult"]
    present = [c for c in cols if c in df.columns]

    print(f"Total rows: {len(df)}")
    print(f"Columns present: {present}")

    if "exec_allow" in df:
        print("\n[exec_allow] Value Counts:")
        print(df["exec_allow"].value_counts(dropna=False).head(10))
        print(f"[exec_allow] % True (>0.5): {pct_true(df['exec_allow']):.4f}")

    if "risk_override" in df:
        print(f"\n[risk_override] % True (>0.5): {pct_true(df['risk_override']):.4f}")

    if "ivr" in df:
        print("\n[ivr] Statistics:")
        print(df["ivr"].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]))

    if {"exec_allow", "risk_override"}.issubset(df.columns):
        pass_rate = ((df["exec_allow"] > 0.5) & (df["risk_override"] < 0.5)).mean()
        print(f"\n[GATE PASS RATE] (exec_allow=1 AND risk_override=0): {pass_rate:.4f}")
    else:
        print("\n[GATE PASS RATE] Skipping (missing columns)")

if __name__ == "__main__":
    main()
