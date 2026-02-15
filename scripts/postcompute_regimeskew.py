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

    parser.add_argument("--input", type=str, default="condornet_v40_fixed.csv")
    parser.add_argument("--output", type=str, default="condornet_v40_fixed.csv")

    # -----------------------------
    # Tuned baseline parameters
    # -----------------------------
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

    # Load dataset
    df = pd.read_csv(args.input)
    close = df["close"]

    # -----------------------------
    # Compute reversal features
    # -----------------------------
    df["rev_m5"] = compute_reversal(close, args.m5_sma, args.m5_rv, args.m5_z)
    df["rev_m15"] = compute_reversal(close, args.m15_sma, args.m15_rv, args.m15_z)
    df["rev_h1"] = compute_reversal(close, args.h1_sma, args.h1_rv, args.h1_z)

    # -----------------------------
    # Top/bottom signals
    # -----------------------------
    df["rev_m5_top"] = (df["rev_m5"] > args.m5_thresh).astype(int)
    df["rev_m5_bot"] = (df["rev_m5"] < -args.m5_thresh).astype(int)

    df["rev_m15_top"] = (df["rev_m15"] > args.m15_thresh).astype(int)
    df["rev_m15_bot"] = (df["rev_m15"] < -args.m15_thresh).astype(int)

    df["rev_h1_top"] = (df["rev_h1"] > args.h1_thresh).astype(int)
    df["rev_h1_bot"] = (df["rev_h1"] < -args.h1_thresh).astype(int)

    # -----------------------------
    # Fuzzy z‑scores
    # -----------------------------
    df["rev_m5_z"] = (df["rev_m5"] - df["rev_m5"].rolling(200).mean()) / df["rev_m5"].rolling(200).std()
    df["rev_m15_z"] = (df["rev_m15"] - df["rev_m15"].rolling(200).mean()) / df["rev_m15"].rolling(200).std()
    df["rev_h1_z"] = (df["rev_h1"] - df["rev_h1"].rolling(200).mean()) / df["rev_h1"].rolling(200).std()

    # -----------------------------
    # Alignment features
    # -----------------------------
    df = compute_alignment(df)

    # -----------------------------
    # Baseline parameter export
    # (for CondorNet multiplicative offsets)
    # -----------------------------
    df["m5_sma_base"] = args.m5_sma
    df["m5_rv_base"] = args.m5_rv
    df["m5_z_base"] = args.m5_z
    df["m5_thresh_base"] = args.m5_thresh

    df["m15_sma_base"] = args.m15_sma
    df["m15_rv_base"] = args.m15_rv
    df["m15_z_base"] = args.m15_z
    df["m15_thresh_base"] = args.m15_thresh

    df["h1_sma_base"] = args.h1_sma
    df["h1_rv_base"] = args.h1_rv
    df["h1_z_base"] = args.h1_z
    df["h1_thresh_base"] = args.h1_thresh

    # Save output (overwrite)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
