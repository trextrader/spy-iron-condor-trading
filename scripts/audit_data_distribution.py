import pandas as pd
import numpy as np

import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=r"data/processed/mamba_institutional_2024_1m_last 1mil.csv", help="Path to dataset")
args = parser.parse_args()

path = args.data
if not os.path.exists(path):
    print(f"[ERROR] File not found: {path}")
    sys.exit(1)

print(f"Loading {path}...")
df = pd.read_csv(path)

def pct_true(s):
    s = s.dropna()
    return (s.astype(float) > 0.5).mean()

cols = ["exec_allow", "risk_override", "ivr", "spread_ratio", "friction_ratio", "chaos_membership", "position_size_mult"]
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

# Sanity check: How often do we pass the basic gates?
gate_cols = [c for c in ["exec_allow", "risk_override"] if c in df.columns]
if set(gate_cols) == {"exec_allow", "risk_override"}:
    pass_rate = ((df["exec_allow"] > 0.5) & (df["risk_override"] < 0.5)).mean()
    print(f"\n[GATE PASS RATE] (exec_allow=1 AND risk_override=0): {pass_rate:.4f}")
else:
    print("\n[GATE PASS RATE] Skipping (missing columns)")
