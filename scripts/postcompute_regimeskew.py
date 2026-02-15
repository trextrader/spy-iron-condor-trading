import argparse
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Reversal score computation
# ------------------------------------------------------------

def compute_reversal(close, sma_len, rv_len, z_len):
    """
    Computes the reversal score using:
    - Stretch (price vs SMA)
    - Realized volatility gap
    - Skew proxy
    """

    sma = close.rolling(sma_len).mean()
    stretch = (close - sma) / sma

    logret = np.log(close / close.shift(1))
    rv = logret.rolling(rv_len).std() * np.sqrt(252)

    iv = close
    gap = rv - iv
    skew = close

    def zscore(x, window):
        mu = x.rolling(window).mean()
        sd = x.rolling(window).std()
        return (x - mu) / sd.replace(0, np.nan)

    z_stretch = zscore(stretch, z_len)
    z_skew = zscore(skew, z_len)
    z_gap = zscore(gap, z_len)

    rev = 0.4 * z_stretch + 0.4 * z_skew + 0.2 * z_gap
    return rev


# ------------------------------------------------------------
# Alignment logic
# ------------------------------------------------------------

def compute_alignment(df):
    """
    Adds:
    - align_2of3
    - align_3of3
    - pairwise alignments
    """

    df["align_m5_m15"] = (
        (df["rev_m5_top"] & df["rev_m15_top"]) |
        (df["rev_m5_bot"] & df["rev_m15_bot"])
    ).astype(int)

    df["align_m15_h1"] = (
        (df["rev_m15_top"] & df["rev_h1_top"]) |
        (df["rev_m15_bot"] & df["rev_h1_bot"])
    ).astype(int)

    df["align_m5_h1"] = (
        (df["rev_m5_top"] & df["rev_h1_top"]) |
        (df["rev_m5_bot"] & df["rev_h1_bot"])
    ).astype(int)

    df["align_3of3"] = (
        (df["rev_m5_top"] & df["rev_m15_top"] & df["rev_h1_top"]) |
        (df["rev_m5_bot"] & df["rev_m15_bot"] & df["rev_h1_bot"])
    ).astype(int)

    df["align_2of3"] = (
        df["align_m5_m15"] |
        df["align_m15_h1"] |
        df["align_m5_h1"]
    ).astype(int)

    return df


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Correct dataset paths for your workflow
    # --------------------------------------------------------
    parser.add_argument(
        "--input",
        type=str,
        default="data/Datasetv3/condornet_v30_20260212_precomputed_fixed.csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/Datasetv3/condornet_v40_20260215.csv"
    )

    # --------------------------------------------------------
    # Tuned baseline parameters (CLI-exposed)
    # --------------------------------------------------------
    parser.add_argument("--m5_sma", type=float, default=32)
    parser.add_argument("--m5_rv", type=float, default=17)
    parser.add_argument("--m5_z", type=float, default=148)
    parser.add_argument("--m5_thresh", type=float, default=1)

    parser.add_argument("--m15_sma", type=float, default=99)
    parser.add_argument("--m15_rv", type=float, default=17)
    parser.add_argument("--m15_z", type=float, default=250)
    parser.add_argument("--m15_thresh", type=float, default=1)

    parser.add_argument("--h1_sma", type=float, default=39)
    parser.add_argument("--h1_rv", type=float, default=17)
    parser.add_argument("--h1_z", type=float, default=123)
    parser.add_argument("--h1_thresh", type=float, default=1)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Status: Start
    # --------------------------------------------------------
    print("🔧 Starting postcompute_regimeskew.py (v4.0 – Institutional REDO)…")
    print(f"📥 Loading dataset: {args.input}")

    # Load dataset
    df = pd.read_csv(args.input)
    print(f"✔ Loaded {len(df):,} rows x {len(df.columns)} cols")

    # --------------------------------------------------------
    # STEP 1: Extract unique spot bars
    # --------------------------------------------------------
    print("📊 Extracting unique spot bars for robust signal computation…")
    
    # Ensure timestamp is datetime for sorting
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Extract one row per unique timestamp to get the underlying underlying_price series
    # We use 'underlying_price' as the authoritative source for the reversal signals
    spot_bars = df.groupby("timestamp")[["underlying_price"]].first().sort_index().reset_index()
    spot_bars = spot_bars.rename(columns={"underlying_price": "close_underlying"})
    
    print(f"✔ Extracted {len(spot_bars):,} unique underlying time-bars")

    # --------------------------------------------------------
    # STEP 2: Compute Reversal Features on Underlying
    # --------------------------------------------------------
    print("🔄 Computing reversal features on UNDERLYING series (M5, M15, H1)…")
    
    close_under = spot_bars["close_underlying"]

    spot_bars["rev_m5"] = compute_reversal(
        close_under, int(args.m5_sma), int(args.m5_rv), int(args.m5_z)
    ).clip(-6, 6)

    spot_bars["rev_m15"] = compute_reversal(
        close_under, int(args.m15_sma), int(args.m15_rv), int(args.m15_z)
    ).clip(-6, 6)

    spot_bars["rev_h1"] = compute_reversal(
        close_under, int(args.h1_sma), int(args.h1_rv), int(args.h1_z)
    ).clip(-6, 6)

    print("✔ Underlying reversal scores computed and clamped to [-6, +6]")

    # --------------------------------------------------------
    # STEP 3: Compute Fuzzy z-scores on Underlying
    # --------------------------------------------------------
    print("🌫 Computing fuzzy z-scores on UNDERLYING signals…")

    def calc_z(s, window=200):
        return (s - s.rolling(window).mean()) / s.rolling(window).std().replace(0, np.nan)

    spot_bars["rev_m5_z"] = calc_z(spot_bars["rev_m5"])
    spot_bars["rev_m15_z"] = calc_z(spot_bars["rev_m15"])
    spot_bars["rev_h1_z"] = calc_z(spot_bars["rev_h1"])

    print("✔ Underlying fuzzy z-scores computed (no row-key fragmentation)")

    # --------------------------------------------------------
    # STEP 4: Merge back to options dataset
    # --------------------------------------------------------
    print("📡 Merging underlying features back to options rows…")
    
    # Drop existing v4 columns if they exist to avoid duplicates
    cols_to_drop = [
        "rev_m5", "rev_m15", "rev_h1", 
        "rev_m5_z", "rev_m15_z", "rev_h1_z",
        "rev_m5_top", "rev_m5_bot", "rev_m15_top", "rev_m15_bot", "rev_h1_top", "rev_h1_bot",
        "align_m5_m15", "align_m15_h1", "align_m5_h1", "align_3of3", "align_2of3"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = df.merge(
        spot_bars[["timestamp", "rev_m5", "rev_m15", "rev_h1", "rev_m5_z", "rev_m15_z", "rev_h1_z"]],
        on="timestamp",
        how="left"
    )
    df = df.drop(columns=["close_underlying"], errors='ignore')

    print("✔ Broadcast merge complete")

    # --------------------------------------------------------
    # STEP 5: Generate Signals and Alignment
    # --------------------------------------------------------
    print("🔍 Generating top/bottom signals and alignments…")

    df["rev_m5_top"] = (df["rev_m5"] > args.m5_thresh).astype(int)
    df["rev_m5_bot"] = (df["rev_m5"] < -args.m5_thresh).astype(int)

    df["rev_m15_top"] = (df["rev_m15"] > args.m15_thresh).astype(int)
    df["rev_m15_bot"] = (df["rev_m15"] < -args.m15_thresh).astype(int)

    df["rev_h1_top"] = (df["rev_h1"] > args.h1_thresh).astype(int)
    df["rev_h1_bot"] = (df["rev_h1"] < -args.h1_thresh).astype(int)

    # Add alignment
    df = compute_alignment(df)

    print("✔ Signals and alignment re-derived from clamped scores")

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------
    print(f"💾 Saving output to: {args.output}")
    df.to_csv(args.output, index=False)

    print("🎉 postcompute_regimeskew.py completed successfully.")


if __name__ == "__main__":
    main()
