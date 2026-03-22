#!/usr/bin/env python3
"""
Compare empty columns between 2020-2024 normalized shells and 2025 reference.

Shows for each TF:
  - Cols empty in 2020-2024 but populated in 2025  → need backfill
  - Cols empty in both 2025 and 2020-2024           → intentionally empty
  - Cols in 2025 header but missing from 2020-2024  → schema gap
"""
import pandas as pd
from pathlib import Path

base   = Path("data/Datasetv4/v43")
shells = base / "normalized_shells"
tfs    = ["m1", "m5", "m15", "h1"]
years  = [2020, 2021, 2022, 2023, 2024]

for tf in tfs:
    # 2025 reference — full file for accurate empty detection
    ref_path = base / "2025" / f"{tf}_dataset_v43_2025.csv"
    ref_df   = pd.read_csv(ref_path, low_memory=False)
    ref_empty = set(c for c in ref_df.columns if ref_df[c].isna().all())
    ref_cols  = set(ref_df.columns)

    # 2020-2024 union of empty cols — full file
    empty_union = set()
    for year in years:
        p  = shells / str(year) / f"{tf}_dataset_v43_{year}.csv"
        df = pd.read_csv(p, low_memory=False)
        empty_union.update(c for c in df.columns if df[c].isna().all())

    need_backfill      = sorted(empty_union - ref_empty - (empty_union - ref_cols))
    intentionally_empty = sorted(empty_union & ref_empty)
    missing_from_schema = sorted(empty_union - ref_cols)

    print(f"\n{'='*60}")
    print(f"TF: {tf}  |  2025 empty: {len(ref_empty)}  |  2020-2024 empty: {len(empty_union)}")
    print(f"  Need backfill (populated in 2025, empty in 2020-2024): {need_backfill}")
    print(f"  Intentionally empty in both:                           {intentionally_empty}")
    print(f"  Empty in 2020-2024 but not in 2025 schema:             {missing_from_schema}")
