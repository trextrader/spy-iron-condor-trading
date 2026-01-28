#!/usr/bin/env python
"""
Check IVR Column Quality

Quick diagnostic to determine if a dataset needs IVR repair.

Usage:
    python scripts/check_ivr_quality.py --data path/to/data.csv
    python scripts/check_ivr_quality.py --data path/to/data.csv --rows 500
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Check IVR column quality")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows to check (default: 200)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] File not found: {args.data}")
        return 1

    import pandas as pd
    import numpy as np

    print("="*50)
    print("IVR COLUMN QUALITY CHECK")
    print("="*50)
    print(f"File: {args.data}")
    print(f"Rows: {args.rows}")
    print()

    df = pd.read_csv(args.data, nrows=args.rows)

    needs_repair = False

    # Check IVR
    if 'ivr' in df.columns:
        ivr = df['ivr']
        print("IVR Column:")
        print(f"  Exists:      Yes")
        print(f"  Min:         {ivr.min():.4f}")
        print(f"  Max:         {ivr.max():.4f}")
        print(f"  Mean:        {ivr.mean():.4f}")
        print(f"  Std:         {ivr.std():.4f}")
        print(f"  Unique vals: {ivr.nunique()}")
        print(f"  Sample:      {ivr.head(5).tolist()}")
        print()

        if ivr.std() < 1.0:
            print("  >>> STATUS: NEEDS REPAIR (IVR is constant)")
            needs_repair = True
        elif ivr.std() < 10.0:
            print("  >>> STATUS: LOW VARIATION (may need repair)")
            needs_repair = True
        else:
            print("  >>> STATUS: OK (IVR has good variation)")
    else:
        print("IVR Column: MISSING")
        print("  >>> STATUS: NEEDS REPAIR (IVR must be computed)")
        needs_repair = True

    print()

    # Check IV
    if 'iv' in df.columns:
        iv = df['iv']
        print("IV Column:")
        print(f"  Exists:      Yes")
        print(f"  Min:         {iv.min():.6f}")
        print(f"  Max:         {iv.max():.6f}")
        print(f"  Std:         {iv.std():.6f}")
        print(f"  Unique vals: {iv.nunique()}")

        if iv.std() < 0.001:
            print("  >>> IV is CONSTANT (will use vol_ewma for IVR)")
        else:
            print("  >>> IV has variation (can compute IVR from IV)")
    else:
        print("IV Column: MISSING")

    print()

    # Check vol_ewma (backup for IVR computation)
    if 'vol_ewma' in df.columns:
        vol = df['vol_ewma']
        print("vol_ewma Column:")
        print(f"  Exists:      Yes")
        print(f"  Std:         {vol.std():.6f}")
        print(f"  Unique vals: {vol.nunique()}")

        if vol.std() > 0.0001:
            print("  >>> vol_ewma has variation (good for IVR computation)")
        else:
            print("  >>> vol_ewma is constant (problem!)")
    else:
        print("vol_ewma Column: MISSING")

    print()
    print("="*50)

    if needs_repair:
        print("RESULT: Dataset NEEDS IVR repair")
        print()
        print("Run this command to fix:")
        print(f"  python scripts/precompute_features.py \\")
        print(f"      --input {args.data} \\")
        print(f"      --output {args.data.replace('.csv', '_v21_fixed.csv')}")
        return 1
    else:
        print("RESULT: Dataset IVR is OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
