#!/usr/bin/env python3
"""Debug why _compute_strong_pivot_analytics returns all-NaN."""
import numpy as np
import pandas as pd
from pathlib import Path

p = Path("data/Datasetv4/v43/normalized_shells/2020/m1_dataset_v43_2020.csv")
df = pd.read_csv(p, low_memory=False)

print(f"Rows: {len(df)}")
print(f"PivotHigh dtype: {df['PivotHigh'].dtype}")
print(f"PivotLow  dtype: {df['PivotLow'].dtype}")
print(f"SlopeATR  dtype: {df['SlopeATR'].dtype}")

ph = df["PivotHigh"].values.astype(np.float32)
pl = df["PivotLow"].values.astype(np.float32)
slope_atr = df["SlopeATR"].values.astype(np.float64)

n_ph = (ph == 1.0).sum()
n_pl = (pl == 1.0).sum()
print(f"\nPivotHigh==1 count: {n_ph}")
print(f"PivotLow==1  count: {n_pl}")

pivot_mask = (ph == 1.0) | (pl == 1.0)
print(f"Total pivot bars:   {pivot_mask.sum()}")

pivot_atrs = np.abs(slope_atr[pivot_mask])
print(f"\nSlopeATR at pivot bars:")
print(f"  Count:    {len(pivot_atrs)}")
print(f"  NaN:      {np.isnan(pivot_atrs).sum()}")
print(f"  Finite:   {np.isfinite(pivot_atrs).sum()}")

valid_atrs = pivot_atrs[np.isfinite(pivot_atrs)]
if len(valid_atrs) > 0:
    print(f"  Min:      {valid_atrs.min():.6f}")
    print(f"  Max:      {valid_atrs.max():.6f}")
    print(f"  Mean:     {valid_atrs.mean():.6f}")
    print(f"  p80:      {np.percentile(valid_atrs, 80):.6f}")
    threshold = float(np.percentile(valid_atrs, 80)) if len(valid_atrs) >= 10 else 0.3
    print(f"  Threshold used: {threshold:.6f}")
    strong = (valid_atrs >= threshold).sum()
    print(f"  Pivots >= threshold: {strong}")
else:
    print("  No finite SlopeATR values at pivot bars!")

# Also check the SlopeATR column overall
sa = df["SlopeATR"].dropna()
print(f"\nSlopeATR overall: count={len(sa)}  min={sa.min():.6f}  max={sa.max():.6f}  mean={sa.mean():.6f}")

# Check first few pivot bars
pivot_idx = np.where(pivot_mask)[0][:10]
print(f"\nFirst 10 pivot bar indices: {pivot_idx}")
for i in pivot_idx:
    print(f"  bar {i:5d}: PH={ph[i]:.0f} PL={pl[i]:.0f} SlopeATR={slope_atr[i]:.6f}")
