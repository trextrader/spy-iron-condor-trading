#!/usr/bin/env python3
"""Find columns that are still all-NaN in 2020-2024 normalized shells."""
import pandas as pd
from pathlib import Path

shells = Path("data/Datasetv4/v43/normalized_shells")
tfs = ["m1", "m5", "m15", "h1"]
years = [2020, 2021, 2022, 2023, 2024]

for tf in tfs:
    empty_union = set()
    for year in years:
        p = shells / str(year) / f"{tf}_dataset_v43_{year}.csv"
        df = pd.read_csv(p, nrows=500, low_memory=False)
        empty = [c for c in df.columns if df[c].isna().all()]
        empty_union.update(empty)
    print(f"\n{tf}  still-empty cols (nrows=500 sample):")
    for c in sorted(empty_union):
        print(f"  {c}")
