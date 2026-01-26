import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from intelligence.features.dynamic_features import (
    compute_all_dynamic_features,
    compute_all_primitive_features_v22,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    ap.add_argument("--nrows", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[FAIL] data not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, nrows=args.nrows)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], utc=True)

    # Compute on spot bars only
    dt_col = None
    for c in ["dt", "timestamp", "datetime", "date"]:
        if c in df.columns:
            dt_col = c
            break
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    aux_cols = [c for c in ["spread_ratio", "lag_minutes"] if c in df.columns]

    if dt_col is None:
        print("[FAIL] no datetime column for spot-bar parity")
        sys.exit(1)

    spot_key_cols = ["symbol", dt_col] if "symbol" in df.columns else [dt_col]
    spot_df = df.drop_duplicates(subset=spot_key_cols)[spot_key_cols + ohlcv_cols + aux_cols].copy()
    spot_df = spot_df.sort_values(spot_key_cols).reset_index(drop=True)

    spot_calc = compute_all_dynamic_features(spot_df.copy(), close_col="close", high_col="high", low_col="low")
    spot_calc = compute_all_primitive_features_v22(
        spot_calc,
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
        spread_col="spread_ratio" if "spread_ratio" in spot_calc.columns else "close",
        inplace=True,
    )

    # Compare any columns that already exist in original spot_df
    common = [c for c in spot_calc.columns if c in spot_df.columns]
    diffs = []
    for c in common:
        a = spot_calc[c].to_numpy()
        b = spot_df[c].to_numpy()
        if a.shape == b.shape:
            max_diff = np.nanmax(np.abs(a - b))
            if np.isfinite(max_diff) and max_diff > args.tol:
                diffs.append((c, float(max_diff)))

    if diffs:
        diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
        print("[WARN] feature parity diffs (top 10):")
        for c, d in diffs[:10]:
            print(f"  {c}: max_diff={d:.6g}")
    else:
        print("[OK] feature parity within tolerance for existing columns")


if __name__ == "__main__":
    main()
