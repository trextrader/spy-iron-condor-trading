import pandas as pd
import numpy as np
import argparse
import glob
import os

def main():
    parser = argparse.ArgumentParser(description="Repair NaNs in Dataset V4 reversal features.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file or glob pattern")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    args = parser.parse_args()

    files = glob.glob(args.input)
    if not files:
        print(f"[ERROR] No files found for pattern: {args.input}")
        return
    
    input_file = sorted(files)[-1]
    print(f"📥 Loading dataset for repair: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")

    # Define columns to repair
    rev_cols = [
        "rev_m5", "rev_m15", "rev_h1",
        "rev_m5_z", "rev_m15_z", "rev_h1_z"
    ]
    
    # Check which ones exist
    valid_cols = [c for c in rev_cols if c in df.columns]
    
    if not valid_cols:
        print("[WARN] No reversal columns found to repair.")
    else:
        print(f"🔧 Repairing NaNs in: {valid_cols}")
        
        # Sort by timestamp to ensure temporal continuity for fill
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["timestamp", "option_symbol"])
        
        # Perform group-aware backfill/forwardfill
        # Since these are underlying signals, they should be the same for all options at a timestamp
        # We can just fill globally within the sorted frame
        for col in valid_cols:
            pre_nan = df[col].isna().sum()
            if pre_nan > 0:
                # Backfill then Forward fill then Zero fill
                df[col] = df[col].fillna(method="bfill").fillna(method="ffill").fillna(0.0)
                post_nan = df[col].isna().sum()
                print(f"  {col}: Fixed {pre_nan - post_nan} NaNs (Remaining: {post_nan})")
            else:
                print(f"  {col}: Already fully populated.")

    # Re-derive binary signals if any repairs were made
    if valid_cols:
        print("🔍 Syncing top/bottom signals...")
        if "rev_m5" in df.columns and "rev_m5_top" in df.columns:
            df["rev_m5_top"] = (df["rev_m5"] > 1.0).astype(int)
            df["rev_m5_bot"] = (df["rev_m5"] < -1.0).astype(int)
        if "rev_m15" in df.columns and "rev_m15_top" in df.columns:
            df["rev_m15_top"] = (df["rev_m15"] > 1.0).astype(int)
            df["rev_m15_bot"] = (df["rev_m15"] < -1.0).astype(int)
        if "rev_h1" in df.columns and "rev_h1_top" in df.columns:
            df["rev_h1_top"] = (df["rev_h1"] > 1.0).astype(int)
            df["rev_h1_bot"] = (df["rev_h1"] < -1.0).astype(int)

    print(f"💾 Saving repaired dataset to: {args.output}")
    df.to_csv(args.output, index=False)
    print("🎉 Repair completed successfully.")

if __name__ == "__main__":
    main()
