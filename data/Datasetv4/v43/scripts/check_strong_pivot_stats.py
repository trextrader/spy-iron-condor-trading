#!/usr/bin/env python3
"""Show 2025 reference stats for strong_pivot cols to inform backfill implementation."""
import pandas as pd
from pathlib import Path

cols = [
    "strong_pivot_slope_h", "strong_pivot_slope_l",
    "strong_pivot_bar_dist_h", "strong_pivot_bar_dist_l",
    "bars_since_strong_pivot",
]

base = Path("data/Datasetv4/v43/2025")

for tf in ["m1", "m15", "h1"]:
    p = base / f"{tf}_dataset_v43_2025.csv"
    df = pd.read_csv(p, usecols=cols, nrows=2000)
    print(f"\n{tf}")
    print(df[cols].describe().round(4))
    # Show first non-NaN rows to understand value structure
    first_valid = df[cols].dropna(how="all").head(3)
    print("  first non-NaN rows:")
    print(first_valid.to_string())
