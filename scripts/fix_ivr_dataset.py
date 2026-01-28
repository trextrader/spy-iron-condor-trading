#!/usr/bin/env python
"""
Fix IVR (IV Rank) Column in Dataset

Problem: Some datasets have constant IVR=50 or IVR=0.15 which causes:
- Regime detection collapse (all "normal" regime)
- Model learns nothing about volatility regimes
- Poor generalization

Solution: Compute realistic IVR from available data:
1. If IV column varies: compute rolling IVR as percentile rank over 252-day window
2. If IV is constant: synthesize IVR from ATR%, vol_ewma, and price dynamics

Usage:
    python scripts/fix_ivr_dataset.py --input data.csv --output data_fixed.csv

Lightning AI:
    python scripts/fix_ivr_dataset.py \
        --input data/processed/mamba_institutional_2024_1m.csv \
        --output data/processed/mamba_institutional_2024_1m_v21_fixed.csv \
        --chunk-size 500000
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_ivr_from_iv(iv_series: pd.Series, window: int = 252 * 78) -> pd.Series:
    """
    Compute IV Rank as rolling percentile over lookback window.

    IVR = (Current IV - Min IV over window) / (Max IV - Min IV over window) * 100

    Args:
        iv_series: Series of IV values
        window: Lookback window (default: 252 days * 78 bars/day for 5min data)

    Returns:
        Series of IVR values (0-100 scale)
    """
    rolling_min = iv_series.rolling(window=window, min_periods=100).min()
    rolling_max = iv_series.rolling(window=window, min_periods=100).max()

    ivr = (iv_series - rolling_min) / (rolling_max - rolling_min + 1e-8) * 100
    ivr = ivr.clip(0, 100).fillna(50)  # Default to 50 (neutral) for NaN

    return ivr


def synthesize_ivr_from_dynamics(df: pd.DataFrame) -> pd.Series:
    """
    Synthesize IVR from price dynamics when IV is constant/unavailable.

    Uses:
    - ATR% (volatility proxy)
    - Log return volatility
    - Bollinger Band width
    - Volume ratio

    Returns:
        Series of synthetic IVR values (0-100 scale)
    """
    n = len(df)

    # Initialize base IVR
    ivr = np.full(n, 50.0, dtype=np.float32)

    # 1. ATR-based component (if atr_pct exists)
    if 'atr_pct' in df.columns:
        atr = df['atr_pct'].values
        atr_pct = pd.Series(atr).rank(pct=True).values * 100
        ivr = 0.4 * ivr + 0.6 * atr_pct

    # 2. Realized volatility component (if vol_ewma exists)
    if 'vol_ewma' in df.columns:
        vol = df['vol_ewma'].values
        vol_pct = pd.Series(vol).rank(pct=True).values * 100
        ivr = 0.5 * ivr + 0.5 * vol_pct

    # 3. Return magnitude component
    if 'log_return' in df.columns:
        ret_abs = np.abs(df['log_return'].values)
        ret_pct = pd.Series(ret_abs).rolling(window=1000, min_periods=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ).fillna(0.5).values * 100
        ivr = 0.7 * ivr + 0.3 * ret_pct

    # 4. Bollinger bandwidth component
    if 'bandwidth' in df.columns:
        bw = df['bandwidth'].values
        bw_pct = pd.Series(bw).rank(pct=True).values * 100
        ivr = 0.6 * ivr + 0.4 * bw_pct

    # Add noise for diversity
    np.random.seed(42)
    noise = np.random.uniform(-5, 5, n)
    ivr = np.clip(ivr + noise, 0, 100)

    return pd.Series(ivr, index=df.index)


def check_iv_quality(df: pd.DataFrame) -> dict:
    """Check if IV column has meaningful variation."""
    result = {
        "has_iv": "iv" in df.columns,
        "has_ivr": "ivr" in df.columns,
        "iv_is_constant": True,
        "ivr_is_constant": True,
        "iv_std": 0,
        "ivr_std": 0,
    }

    if "iv" in df.columns:
        iv = df["iv"].dropna()
        result["iv_std"] = iv.std()
        result["iv_is_constant"] = iv.std() < 0.001

    if "ivr" in df.columns:
        ivr = df["ivr"].dropna()
        result["ivr_std"] = ivr.std()
        result["ivr_is_constant"] = ivr.std() < 1.0

    return result


def process_dataset(input_path: str, output_path: str, chunk_size: int = 500000):
    """Process dataset to fix IVR values."""

    print("="*60)
    print("IVR FIX SCRIPT")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Step 1: Analyze first chunk to understand data quality
    print("[1/4] Analyzing IV/IVR quality...")
    sample_df = pd.read_csv(input_path, nrows=100000)
    quality = check_iv_quality(sample_df)

    print(f"  Has IV column:  {quality['has_iv']}")
    print(f"  Has IVR column: {quality['has_ivr']}")
    print(f"  IV std:         {quality['iv_std']:.6f}")
    print(f"  IVR std:        {quality['ivr_std']:.6f}")
    print(f"  IV is constant: {quality['iv_is_constant']}")
    print(f"  IVR is constant: {quality['ivr_is_constant']}")

    if not quality['ivr_is_constant'] and quality['ivr_std'] > 10:
        print("\n[INFO] IVR already has good variation! No fix needed.")
        print("       Copying file as-is...")
        # Just copy the file
        import shutil
        shutil.copy(input_path, output_path)
        print(f"[DONE] Copied to {output_path}")
        return

    # Step 2: Determine fix strategy
    if not quality['iv_is_constant']:
        print("\n[STRATEGY] Computing IVR from IV column (rolling percentile)")
        use_iv = True
    else:
        print("\n[STRATEGY] Synthesizing IVR from price dynamics (IV is constant)")
        use_iv = False

    # Step 3: Process in chunks
    print("\n[2/4] Counting total rows...")
    total_rows = sum(1 for _ in open(input_path, 'r')) - 1  # Subtract header
    print(f"  Total rows: {total_rows:,}")

    print("\n[3/4] Processing chunks...")

    # Read header
    header_df = pd.read_csv(input_path, nrows=0)
    columns = list(header_df.columns)

    # Ensure ivr column exists
    if 'ivr' not in columns:
        columns.append('ivr')

    # Process and write
    first_chunk = True
    rows_processed = 0

    # For IV-based IVR, we need to track rolling stats
    iv_history = [] if use_iv else None
    iv_window = 252 * 78  # ~252 trading days at 78 5-min bars/day

    for chunk in tqdm(
        pd.read_csv(input_path, chunksize=chunk_size, low_memory=False),
        total=(total_rows // chunk_size) + 1,
        desc="Processing"
    ):
        # Compute IVR
        if use_iv:
            # Use rolling IV percentile
            if iv_history:
                # Prepend history for rolling calculation
                combined_iv = pd.concat([pd.Series(iv_history), chunk['iv']])
                ivr_full = compute_ivr_from_iv(combined_iv, window=iv_window)
                chunk['ivr'] = ivr_full.iloc[len(iv_history):].values
            else:
                chunk['ivr'] = compute_ivr_from_iv(chunk['iv'], window=min(len(chunk), iv_window))

            # Update history (keep last window samples)
            iv_history = chunk['iv'].tolist()[-iv_window:]
        else:
            # Synthesize from dynamics
            chunk['ivr'] = synthesize_ivr_from_dynamics(chunk).values

        # Ensure IVR is in 0-100 range
        chunk['ivr'] = chunk['ivr'].clip(0, 100).astype(np.float32)

        # Write chunk
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(output_path, mode=mode, header=header, index=False)

        first_chunk = False
        rows_processed += len(chunk)

    # Step 4: Validate output
    print("\n[4/4] Validating output...")
    val_df = pd.read_csv(output_path, nrows=50000)
    new_quality = check_iv_quality(val_df)

    print(f"  New IVR std:    {new_quality['ivr_std']:.4f}")
    print(f"  New IVR min:    {val_df['ivr'].min():.2f}")
    print(f"  New IVR max:    {val_df['ivr'].max():.2f}")
    print(f"  New IVR mean:   {val_df['ivr'].mean():.2f}")

    # Check regime distribution
    low_vol = (val_df['ivr'] < 30).mean() * 100
    normal_vol = ((val_df['ivr'] >= 30) & (val_df['ivr'] <= 70)).mean() * 100
    high_vol = (val_df['ivr'] > 70).mean() * 100

    print(f"\n  Regime Distribution:")
    print(f"    Low Vol (<30):    {low_vol:.1f}%")
    print(f"    Normal (30-70):   {normal_vol:.1f}%")
    print(f"    High Vol (>70):   {high_vol:.1f}%")

    if new_quality['ivr_std'] > 10:
        print("\n[SUCCESS] IVR now has good variation!")
    else:
        print("\n[WARNING] IVR still has low variation - check input data quality")

    file_size = os.path.getsize(output_path) / (1024 * 1024 * 1024)
    print(f"\n[DONE] Output: {output_path}")
    print(f"       Size: {file_size:.2f} GB")
    print(f"       Rows: {rows_processed:,}")


def main():
    parser = argparse.ArgumentParser(description="Fix IVR column in dataset")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--chunk-size", type=int, default=500000, help="Chunk size for processing")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return 1

    process_dataset(args.input, args.output, args.chunk_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
