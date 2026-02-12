#!/usr/bin/env python3
"""
Precompute Features for CondorNet v3.0 Dataset
================================================
Fills all BLANK columns in the 87-column v3.0 dataset.

Input:  data/Datasetv3/condornet_v30_*.csv  (from build_dataset_v30.py)
Output: data/Datasetv3/condornet_v30_YYYYMMDD_precomputed.csv

Already populated from sources (~37 cols):
  - Options: timestamp, symbol, option_symbol, expiration, strike, call_put,
             rho, volume, open_interest, te, delta, gamma, vega, theta, iv,
             bid, ask, mark
  - Barchart: underlying_price, open, high, low, close, sma,
              FRAMA, Anchored_VWAP, McClellanOsc, IV_High, IV_Mid, IV_Low,
              Options_Total_Volume, Options_Put_Volume, Options_Call_Volume,
              OSC_Volume, WeightedAlpha, WilderAccSwingIndex,
              AccDistWill, AccDistWillMovAvg

Precomputed here (~50 cols):
  - Dynamic features: log_return, vol_ewma, ret_z, atr_pct, kappa_proxy, vol_energy,
                      rsi_dyn, adx_adaptive, psar_adaptive,
                      bb_mu_dyn, bb_sigma_dyn, bb_lower_dyn, bb_upper_dyn, stoch_k_dyn,
                      consolidation_score, breakout_score
  - Primitives: spread_ratio, bandwidth, bb_percentile, bw_expansion_rate,
                cmf, pressure_up, pressure_down, friction_ratio,
                exec_allow, gap_risk_score, risk_override, iv_confidence,
                macd_norm, macd_signal_norm, macd_histogram,
                plus_di, minus_di, psar_trend, psar_reversion_mu,
                beta1_norm_stub, chaos_membership, position_size_mult, fuzzy_reversion_11
  - Targets: target_spot, max_dd_60m, psar_mark, mtf_consensus
  - Derived: ivr, spread_ratio
  - Greeks (for Alpaca rows): delta, gamma, theta, vega, rho, iv (via BS + NR)
  - Microstructure: aggregate_bid_count, aggregate_ask_count, mean_bid, mean_ask, quote_spread

Usage:
  python scripts/precompute_features_v3.py
  python scripts/precompute_features_v3.py --input data/Datasetv3/condornet_v30_20260211.csv
  python scripts/precompute_features_v3.py --nrows 200000   # quick test subset
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import gc
import glob

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from intelligence.features.dynamic_features import (
    compute_all_dynamic_features,
    compute_all_primitive_features_v22
)
from intelligence.canonical_feature_registry import NEUTRAL_FILL_VALUES_V22

# Import BS Greeks calculator
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data_factory', 'scripts'))
from options_math import compute_greeks_df

# ============================================================================
# V3.0 SCHEMA (87 columns)
# ============================================================================
V30_SCHEMA = [
    'timestamp', 'symbol', 'underlying_price', 'open', 'high', 'low', 'close',
    'option_symbol', 'expiration', 'strike', 'call_put',
    'rho', 'volume', 'open_interest',
    'te', 'ivr', 'target_spot', 'sma', 'psar_mark', 'mtf_consensus',
    'log_return', 'vol_ewma', 'ret_z', 'atr_pct', 'kappa_proxy', 'vol_energy',
    'rsi_dyn', 'adx_adaptive', 'psar_adaptive',
    'bb_mu_dyn', 'bb_sigma_dyn', 'bb_lower_dyn', 'bb_upper_dyn', 'stoch_k_dyn',
    'consolidation_score', 'breakout_score',
    'spread_ratio', 'bandwidth', 'bb_percentile', 'bw_expansion_rate',
    'cmf', 'pressure_up', 'pressure_down', 'friction_ratio',
    'exec_allow', 'gap_risk_score', 'risk_override', 'iv_confidence',
    'macd_norm', 'macd_signal_norm', 'macd_histogram',
    'plus_di', 'minus_di', 'psar_trend', 'psar_reversion_mu',
    'beta1_norm_stub', 'chaos_membership', 'position_size_mult', 'fuzzy_reversion_11',
    'delta', 'gamma', 'vega', 'theta', 'iv',
    'max_dd_60m',
    'bid', 'ask', 'mark',
    'FRAMA', 'Anchored_VWAP', 'McClellanOsc',
    'IV_High', 'IV_Mid', 'IV_Low',
    'Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume', 'OSC_Volume',
    'WeightedAlpha', 'WilderAccSwingIndex', 'AccDistWill', 'AccDistWillMovAvg',
    'aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread',
]


def find_latest_v30(data_dir: str) -> str:
    """Find the most recent condornet_v30_*.csv file."""
    pattern = os.path.join(data_dir, "condornet_v30_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No v3.0 dataset found matching {pattern}")
    # Prefer non-precomputed files
    base_files = [f for f in files if 'precomputed' not in f]
    return base_files[-1] if base_files else files[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute v3.0 features into enriched CSV.")
    parser.add_argument("--input", default=None, help="Input v3.0 CSV. Auto-detects if not specified.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--nrows", type=int, default=None, help="Load only first N rows for testing.")
    parser.add_argument("--skip-greeks", action="store_true", help="Skip BS Greeks computation.")
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "data", "Datasetv3")

    # Find input
    if args.input:
        inp = args.input
    else:
        inp = find_latest_v30(data_dir)

    if not os.path.exists(inp):
        print(f"[ERROR] Input not found: {inp}")
        return 2

    # Output path
    if args.output:
        out = args.output
    else:
        from datetime import datetime
        out = os.path.join(data_dir, f"condornet_v30_{datetime.now().strftime('%Y%m%d')}_precomputed.csv")

    print("=" * 70)
    print("  CONDORNET v3.0 FEATURE PRECOMPUTATION")
    print(f"  Input:  {os.path.basename(inp)}")
    print(f"  Output: {os.path.basename(out)}")
    print("=" * 70)

    t0 = time.time()

    # ================================================================
    # STEP 1: Load dataset
    # ================================================================
    print("\n[STEP 1] Loading v3.0 dataset ...")
    read_kwargs = dict(low_memory=False, nrows=args.nrows)
    df = pd.read_csv(inp, **read_kwargs)
    print(f"  Loaded: {len(df):,} rows x {len(df.columns)} cols")

    # Parse dates
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if 'expiration' in df.columns:
        df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')

    # Coerce numerics
    numeric_skip = {'timestamp', 'symbol', 'option_symbol', 'expiration', 'call_put'}
    for col in df.columns:
        if col not in numeric_skip and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by timestamp for rolling features
    df = df.sort_values(['timestamp', 'option_symbol']).reset_index(drop=True)

    # Report initial coverage
    populated = [c for c in V30_SCHEMA if c in df.columns and df[c].notna().any()]
    blank = [c for c in V30_SCHEMA if c not in df.columns or not df[c].notna().any()]
    print(f"  Populated: {len(populated)} cols  |  Blank: {len(blank)} cols")

    # ================================================================
    # STEP 2: Extract unique spot bars and compute dynamic features
    # ================================================================
    print("\n[STEP 2] Extracting unique spot bars for dynamic feature computation ...")

    # Create date key for spot extraction
    df['_date'] = df['timestamp'].dt.date

    # Spot bars: one row per unique date with underlying OHLCV
    spot_cols = ['_date', 'open', 'high', 'low', 'close', 'underlying_price']
    available_spot = [c for c in spot_cols if c in df.columns]
    spot_df = df.drop_duplicates(subset=['_date'])[available_spot].copy()
    spot_df = spot_df.sort_values('_date').reset_index(drop=True)

    # Use underlying_price as close if close is from Barchart
    if 'volume' not in spot_df.columns:
        spot_df['volume'] = 0.0  # Placeholder for dynamic features that need volume

    n_bars = len(spot_df)
    print(f"  Unique spot bars: {n_bars:,}")

    # Compute V2.1 dynamic features on spot bars
    print("  Computing dynamic features (V2.1) ...")
    t1 = time.time()
    spot_df = compute_all_dynamic_features(spot_df)

    # Targets
    print("  Computing targets (target_spot, max_dd_60m) ...")
    spot_df['target_spot'] = spot_df['close'].shift(-1)  # Next day close (daily data)

    # max_dd_60m: For daily data, use next-bar min low as proxy
    if n_bars > 1:
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=min(5, n_bars))
        future_low_min = spot_df['low'].rolling(window=indexer, min_periods=1).min()
        spot_df['max_dd_60m'] = (future_low_min - spot_df['close']) / spot_df['close']
    else:
        spot_df['max_dd_60m'] = 0.0

    # SMA (already from Barchart, but recompute if missing)
    if spot_df['close'].notna().sum() >= 20:
        spot_df['sma_computed'] = spot_df['close'].rolling(20, min_periods=1).mean()

    # PSAR mark
    try:
        import talib
        spot_df['psar_mark'] = talib.SAR(spot_df['high'].values, spot_df['low'].values,
                                          acceleration=0.02, maximum=0.2)
    except ImportError:
        spot_df['psar_mark'] = spot_df['close']  # Fallback: use close as proxy

    # V2.2 Primitive features
    print("  Computing V2.2 primitive features ...")
    spot_df = compute_all_primitive_features_v22(
        spot_df,
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
        spread_col="close",  # No spread_ratio in spot bars, use close as proxy
    )

    t2 = time.time()
    dynamic_cols = [c for c in spot_df.columns if c not in available_spot + ['volume']]
    print(f"  Computed {len(dynamic_cols)} dynamic/primitive columns in {t2-t1:.1f}s")

    # ================================================================
    # STEP 3: Merge dynamic features back to full dataset
    # ================================================================
    print("\n[STEP 3] Merging dynamic features back to options rows ...")

    # Prepare merge: only new columns + join key
    merge_cols = ['_date'] + dynamic_cols
    spot_merge = spot_df[[c for c in merge_cols if c in spot_df.columns]].copy()

    # Drop existing columns that will be replaced
    cols_to_replace = [c for c in dynamic_cols if c in df.columns]
    if cols_to_replace:
        # Only replace if currently blank
        actually_replace = []
        for col in cols_to_replace:
            if not df[col].notna().any():
                actually_replace.append(col)
        if actually_replace:
            print(f"  Replacing {len(actually_replace)} blank columns with computed values")
            df = df.drop(columns=actually_replace)

    df = df.merge(spot_merge, on='_date', how='left', suffixes=('', '_new'))

    # Handle suffix conflicts: prefer new values for blank columns
    new_cols = [c for c in df.columns if c.endswith('_new')]
    for nc in new_cols:
        base = nc.replace('_new', '')
        if base in df.columns:
            # Fill blanks in base with new values
            mask = df[base].isna()
            df.loc[mask, base] = df.loc[mask, nc]
        else:
            df[base] = df[nc]
        df = df.drop(columns=[nc])

    print(f"  Merged: {len(df):,} rows")

    # ================================================================
    # STEP 4: Post-merge features (IVR, spread_ratio, microstructure)
    # ================================================================
    print("\n[STEP 4] Computing post-merge features ...")

    # IVR (Realized Volatility Rank proxy)
    if 'vol_ewma' in df.columns and df['vol_ewma'].notna().any():
        print("  Computing IVR (realized vol rank proxy) ...")
        # Use a shorter window for daily data
        rvr_window = max(60, min(252, len(spot_df)))
        vol_by_date = df.groupby('_date')['vol_ewma'].first()
        vol_min = vol_by_date.rolling(window=rvr_window, min_periods=5).min()
        vol_max = vol_by_date.rolling(window=rvr_window, min_periods=5).max()
        vol_range = (vol_max - vol_min).replace(0, 1.0)
        ivr_by_date = (((vol_by_date - vol_min) / vol_range) * 100.0).clip(0, 100)
        ivr_map = ivr_by_date.to_dict()
        df['ivr'] = df['_date'].map(ivr_map).fillna(50.0).astype(np.float32)

    # spread_ratio: bid-ask spread / mid price (from options data)
    if 'bid' in df.columns and 'ask' in df.columns:
        print("  Computing spread_ratio from bid/ask ...")
        bid = df['bid'].astype(float)
        ask = df['ask'].astype(float)
        mid = (bid + ask) / 2.0
        df['spread_ratio'] = np.where(mid > 0.01, (ask - bid) / mid, np.nan)

    # bandwidth: bb_upper - bb_lower (if available)
    if 'bb_upper_dyn' in df.columns and 'bb_lower_dyn' in df.columns:
        print("  Computing bandwidth ...")
        df['bandwidth'] = df['bb_upper_dyn'] - df['bb_lower_dyn']

    # ================================================================
    # STEP 5: Microstructure aggregates from options chain
    # ================================================================
    print("\n[STEP 5] Computing microstructure aggregates ...")

    if 'bid' in df.columns and 'ask' in df.columns:
        # Aggregate per date: count, mean bid/ask, spread
        agg = df.groupby('_date').agg(
            aggregate_bid_count=('bid', lambda x: x.notna().sum()),
            aggregate_ask_count=('ask', lambda x: x.notna().sum()),
            mean_bid=('bid', 'mean'),
            mean_ask=('ask', 'mean'),
        ).reset_index()
        agg['quote_spread'] = agg['mean_ask'] - agg['mean_bid']

        # Merge back
        for col in ['aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread']:
            if col in df.columns:
                df = df.drop(columns=[col])
        df = df.merge(agg, on='_date', how='left')
        print(f"  Microstructure: {len(agg)} date-level aggregates computed")
    else:
        print("  Skipping microstructure (no bid/ask data)")

    # ================================================================
    # STEP 6: Greeks computation (for rows missing Greeks)
    # ================================================================
    if not args.skip_greeks:
        print("\n[STEP 6] Computing Greeks via Black-Scholes ...")
        needs_greeks = df['delta'].isna() & df['underlying_price'].notna() & df['strike'].notna()
        n_needs = needs_greeks.sum()

        if n_needs > 0:
            print(f"  {n_needs:,} rows need Greeks computation")
            df = compute_greeks_df(df, spot_col='underlying_price', strike_col='strike',
                                   te_col='te', cp_col='call_put', mark_col='mark')
        else:
            print("  All rows already have Greeks")
    else:
        print("\n[STEP 6] Skipping Greeks (--skip-greeks)")

    # ================================================================
    # STEP 7: Semantic NaN fill
    # ================================================================
    print("\n[STEP 7] Applying semantic NaN fill ...")

    for col, val in NEUTRAL_FILL_VALUES_V22.items():
        if col in df.columns and df[col].isna().any():
            n_fill = df[col].isna().sum()
            df[col] = df[col].fillna(val)
            if n_fill > 0:
                print(f"  {col}: filled {n_fill:,} NaN with {val}")

    # Drop internal columns
    df = df.drop(columns=['_date'], errors='ignore')

    # ================================================================
    # STEP 8: Enforce schema and save
    # ================================================================
    print("\n[STEP 8] Enforcing 87-column schema and saving ...")

    # Ensure all schema columns exist
    for col in V30_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    out_df = df[V30_SCHEMA].copy()

    # Report final coverage
    print(f"\n  FINAL COVERAGE:")
    populated_final = []
    blank_final = []
    for col in V30_SCHEMA:
        pct = out_df[col].notna().mean() * 100
        if pct > 0:
            populated_final.append((col, pct))
        else:
            blank_final.append(col)

    print(f"  Populated: {len(populated_final)} / {len(V30_SCHEMA)} columns")
    for col, pct in populated_final:
        print(f"    {col:30s}  {pct:6.1f}%")

    if blank_final:
        print(f"\n  Still blank: {len(blank_final)} columns")
        for col in blank_final:
            print(f"    {col}")

    # Save
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out_df.to_csv(out, index=False)

    t_end = time.time()
    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"\n  Saved: {os.path.basename(out)}")
    print(f"  Rows: {len(out_df):,} x {len(out_df.columns)} cols")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Total elapsed: {t_end - t0:.1f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
