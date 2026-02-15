#!/usr/bin/env python3
"""
postcompute_regimeskew.py

Add regime + skew-based reversal features to CondorNet v3.0 dataset
to produce v4.0:

  - regime_trend_stretch
  - regime_iv_skew
  - regime_rv_iv_gap
  - regime_trend_stretch_flag
  - regime_iv_skew_flag
  - regime_reversal_condor_score

Default input:
  data/Datasetv3/condornet_v30_20260212_precomputed_fixed.csv

Default output:
  data/Datasetv3/condornet_v40_20260214.csv

Run (Lightning AI / local):
  python postcompute_regimeskew.py
  python postcompute_regimeskew.py --input path/to/v30.csv --output path/to/v40.csv \
      --k3 0.02 --k4 0.03 --k5 0.00
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_regime_skew_features(
    df: pd.DataFrame,
    k3: float,
    k4: float,
    k5: float,
) -> pd.DataFrame:
    """
    Add regime + skew-based reversal features to the dataframe.

    Uses existing columns:
      - underlying_price
      - sma
      - vol_ewma
      - iv (if present) or IV_Mid
      - IV_High, IV_Low

    New columns:
      - regime_trend_stretch
      - regime_iv_skew
      - regime_rv_iv_gap
      - regime_trend_stretch_flag
      - regime_iv_skew_flag
      - regime_reversal_condor_score
    """

    # --- Safety checks / fallbacks -----------------------------------------
    required_cols = ["underlying_price", "sma", "vol_ewma"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")

    # IV ATM proxy: prefer 'iv', else 'IV_Mid', else NaN
    if "iv" in df.columns:
        iv_atm = df["iv"].astype(float)
    elif "IV_Mid" in df.columns:
        iv_atm = df["IV_Mid"].astype(float)
    else:
        iv_atm = pd.Series(np.nan, index=df.index)

    # IV skew proxy: IV_Low - IV_High (downside puts richer than upside calls)
    if "IV_Low" in df.columns and "IV_High" in df.columns:
        iv_low = df["IV_Low"].astype(float)
        iv_high = df["IV_High"].astype(float)
        regime_iv_skew = iv_low - iv_high
    else:
        # Fallback: no skew info, fill with NaN
        regime_iv_skew = pd.Series(np.nan, index=df.index)

    # --- Trend stretch: (P_t - MA_n(t)) / MA_n(t) ---------------------------
    price = df["underlying_price"].astype(float)
    sma = df["sma"].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        regime_trend_stretch = (price - sma) / sma
    regime_trend_stretch = regime_trend_stretch.replace([np.inf, -np.inf], np.nan)

    # --- Realized vs implied vol gap: iv_atm - vol_ewma ---------------------
    vol_ewma = df["vol_ewma"].astype(float)
    regime_rv_iv_gap = iv_atm - vol_ewma

    # --- Flags based on thresholds -----------------------------------------
    # Trend stretched: (P - MA)/MA > k3
    regime_trend_stretch_flag = (regime_trend_stretch > k3).astype(int)

    # Skew elevated: IV_Low - IV_High > k4
    regime_iv_skew_flag = (regime_iv_skew > k4).astype(int)

    # RV < IV: iv_atm - vol_ewma > k5  (room for vol expansion)
    rv_lt_iv_flag = (regime_rv_iv_gap > k5).astype(int)

    # Combined reversal score (0–3)
    regime_reversal_condor_score = (
        regime_trend_stretch_flag
        + regime_iv_skew_flag
        + rv_lt_iv_flag
    )

    # --- Attach to dataframe -----------------------------------------------
    df = df.copy()
    df["regime_trend_stretch"] = regime_trend_stretch
    df["regime_iv_skew"] = regime_iv_skew
    df["regime_rv_iv_gap"] = regime_rv_iv_gap
    df["regime_trend_stretch_flag"] = regime_trend_stretch_flag
    df["regime_iv_skew_flag"] = regime_iv_skew_flag
    df["regime_reversal_condor_score"] = regime_reversal_condor_score

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Postcompute regime + skew-based reversal features for CondorNet dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/Datasetv3/condornet_v30_20260212_precomputed_fixed.csv",
        help="Input CSV (v3.0 precomputed dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/Datasetv3/condornet_v40_20260214.csv",
        help="Output CSV (v4.0 with regime+skew features)",
    )
    parser.add_argument(
        "--k3",
        type=float,
        default=0.02,
        help="Threshold for trend stretch (e.g., 0.02 = 2%% above MA)",
    )
    parser.add_argument(
        "--k4",
        type=float,
        default=0.03,
        help="Threshold for IV skew (IV_Low - IV_High > k4)",
    )
    parser.add_argument(
        "--k5",
        type=float,
        default=0.00,
        help="Threshold for RV<IV gap (iv_atm - vol_ewma > k5)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading input dataset: {input_path}")
    df = pd.read_csv(input_path)

    print("Adding regime + skew-based reversal features...")
    df_out = add_regime_skew_features(
        df,
        k3=args.k3,
        k4=args.k4,
        k5=args.k5,
    )

    print(f"Saving output dataset: {output_path}")
    df_out.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
