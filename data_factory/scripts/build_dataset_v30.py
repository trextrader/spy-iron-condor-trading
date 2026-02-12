#!/usr/bin/env python3
"""
Dataset v3.0 Construction Pipeline
===================================
Builds the 87-column CondorNet HAL-S3N v3.0 dataset from two source files:
  1. options_2025.csv          -- Real SPY options chains (daily per-contract)
  2. SPY_Barchart_*_1m_*.csv   -- Barchart M1 OHLCV + 14 technical studies

Output: data/Datasetv3/condornet_v30_YYYYMMDD.csv

Only direct source mappings and trivial derivations (te) are populated.
All feature-engineered columns are left BLANK for the precompute step.

Author: Claude Code (Implementation)
Based on: Dr. T. Jerry Mahabub, "CondorNet HAL-SNN New Upgraded Dataset" (Feb 10, 2026)
Schema: 87 columns = 65 legacy + 3 raw pricing + 19 new v3.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Datasetv3"

OPTIONS_FILE = DATA_DIR / "options_2025.csv"
BARCHART_FILE = DATA_DIR / "SPY_Barchart_Interactive_Chart_Range_1m_02_09_2026.csv"
OUTPUT_DIR = DATA_DIR  # Save output in same folder

# ============================================================================
# V3.0 SCHEMA -- 87 columns
# ============================================================================
V30_SCHEMA = [
    # --- Legacy v2.x core (65 columns) ---
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

    # --- Raw options pricing (3 columns -- preserved for spread/microstructure) ---
    'bid', 'ask', 'mark',

    # --- NEW v3.0 Trend Structure (3) ---
    'FRAMA', 'Anchored_VWAP', 'McClellanOsc',

    # --- NEW v3.0 IV Bands (3) ---
    'IV_High', 'IV_Mid', 'IV_Low',

    # --- NEW v3.0 Options Chain Aggregates (4) ---
    'Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume', 'OSC_Volume',

    # --- NEW v3.0 Flow & Breadth Extensions (4) ---
    'WeightedAlpha', 'WilderAccSwingIndex', 'AccDistWill', 'AccDistWillMovAvg',

    # --- NEW v3.0 Microstructure (5) ---
    'aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread',
]

assert len(V30_SCHEMA) == 87, f"Schema has {len(V30_SCHEMA)} columns, expected 87"


# ============================================================================
# STEP 1: Load Options Data
# ============================================================================
def load_options(filepath: Path) -> pd.DataFrame:
    """
    Load options_2025.csv and map columns to v3.0 schema.

    Source columns (21):
        contract_id, symbol, expiration, strike, type, last, mark, bid,
        bid_size, ask, ask_size, volume, open_interest, date,
        implied_volatility, delta, gamma, theta, vega, rho, in_the_money

    Directly mapped to v3.0 (17 columns):
        option_symbol, symbol, expiration, strike, call_put,
        volume, open_interest, timestamp, iv, delta, gamma, theta, vega, rho,
        bid, ask, mark
    Plus computed:
        te = (expiration - timestamp).days / 365
    """
    print(f"\n[STEP 1] Loading options data: {filepath.name}")
    t0 = time.time()

    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Raw rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # The CSV contains duplicate header rows embedded in the data
    # (artifact of multi-file concatenation). Drop any row where 'date' == 'date'.
    bad_rows = df['date'].astype(str).str.strip() == 'date'
    if bad_rows.any():
        n_bad = bad_rows.sum()
        df = df[~bad_rows].copy()
        print(f"  Dropped {n_bad} embedded header row(s)")

    # Rename source -> v3.0 schema
    df = df.rename(columns={
        'contract_id':        'option_symbol',
        'type':               'call_put',
        'date':               'timestamp',
        'implied_volatility': 'iv',
    })

    # Keep only the columns we need (direct mappings)
    options_keep = [
        'timestamp', 'symbol', 'option_symbol', 'expiration', 'strike', 'call_put',
        'rho', 'volume', 'open_interest',
        'delta', 'gamma', 'vega', 'theta', 'iv',
        'bid', 'ask', 'mark',
    ]
    df = df[options_keep].copy()

    # Coerce numeric columns (mixed types from embedded headers)
    numeric_cols = ['strike', 'rho', 'volume', 'open_interest',
                    'delta', 'gamma', 'vega', 'theta', 'iv',
                    'bid', 'ask', 'mark']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse dates
    df['timestamp']  = pd.to_datetime(df['timestamp'], errors='coerce')
    df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')

    # Drop any rows where date parsing failed
    before = len(df)
    df = df.dropna(subset=['timestamp', 'expiration']).copy()
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with unparseable dates")

    # Compute te (time-to-expiry) -- deterministic, from source columns only
    df['te'] = (df['expiration'] - df['timestamp']).dt.days / 365.0

    # Join key
    df['_join_date'] = df['timestamp'].dt.date

    elapsed = time.time() - t0
    print(f"  Mapped columns: {len(df.columns)}")
    print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"  Unique dates: {df['_join_date'].nunique()}")
    print(f"  Unique contracts: {df['option_symbol'].nunique():,}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return df


# ============================================================================
# STEP 2: Load Barchart M1 Studies and Aggregate to Daily
# ============================================================================
def load_barchart_daily(filepath: Path) -> pd.DataFrame:
    """
    Load Barchart M1 studies CSV:
      - Skip the study-definition header (row 0)
      - Handle latin-1 encoding (TM symbols)
      - Drop the footer row ("Downloaded from Barchart.com ...")
      - Aggregate from 1-minute to daily

    Daily aggregation:
        OHLCV   -> first open, max high, min low, last close
        Studies -> last non-NaN value of each trading day
    """
    print(f"\n[STEP 2] Loading Barchart M1 studies: {filepath.name}")
    t0 = time.time()

    df = pd.read_csv(filepath, skiprows=1, encoding='latin-1')
    print(f"  Raw rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # Drop the footer row ("Downloaded from Barchart.com ...")
    mask = df['Date Time'].astype(str).str.match(r'^\d{4}', na=False)
    dropped = (~mask).sum()
    df = df[mask].copy()
    if dropped > 0:
        print(f"  Dropped {dropped} non-data row(s) (footer)")

    # Parse timestamps
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df['_join_date'] = df['Date Time'].dt.date

    # Column name mapping: Barchart -> v3.0 schema
    BARCHART_MAP = {
        'Close':                    'close',
        'Open':                     'open',
        'High':                     'high',
        'Low':                      'low',
        'MA-Simple':                'sma',
        'FRAMA':                    'FRAMA',
        'Anchored VWAP':            'Anchored_VWAP',
        'McClellanOsc':             'McClellanOsc',
        'Implied Volatility High':  'IV_High',
        'Implied Volatility':       'IV_Mid',
        'Implied Volatility Low':   'IV_Low',
        'Options Total Volume':     'Options_Total_Volume',
        'Options Put Volume':       'Options_Put_Volume',
        'Options Call Volume':      'Options_Call_Volume',
        'OSC-Volume':               'OSC_Volume',
        'WeightedAlpha':            'WeightedAlpha',
        'WilderAccSwingIndex':      'WilderAccSwingIndex',
        'AccDistWill':              'AccDistWill',
        'AccDistWillMovAvg':        'AccDistWillMovAvg',
    }

    df = df.rename(columns=BARCHART_MAP)

    # Coerce all mapped columns to numeric
    for col in BARCHART_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Daily aggregation ---
    # OHLCV: proper bar aggregation
    ohlcv_agg = df.groupby('_join_date').agg(
        open=('open',  'first'),
        high=('high',  'max'),
        low=('low',    'min'),
        close=('close', 'last'),
    ).reset_index()

    # Studies: last non-NaN value per day (stateful indicators)
    study_cols = [
        'sma', 'FRAMA', 'Anchored_VWAP', 'McClellanOsc',
        'IV_High', 'IV_Mid', 'IV_Low',
        'Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume',
        'OSC_Volume', 'WeightedAlpha', 'WilderAccSwingIndex',
        'AccDistWill', 'AccDistWillMovAvg',
    ]
    available_studies = [c for c in study_cols if c in df.columns]
    study_agg = df.groupby('_join_date')[available_studies].last().reset_index()

    # Merge OHLCV + studies into one daily frame
    daily = ohlcv_agg.merge(study_agg, on='_join_date', how='outer')

    # underlying_price = daily close
    daily['underlying_price'] = daily['close']

    elapsed = time.time() - t0
    unique_dates = len(daily)
    print(f"  Aggregated to {unique_dates:,} trading days")
    print(f"  Date range: {daily['_join_date'].min()} to {daily['_join_date'].max()}")

    # Show which studies have data
    for col in available_studies:
        pct = daily[col].notna().mean() * 100
        print(f"    {col:30s}  {pct:5.1f}% populated")

    print(f"  Elapsed: {elapsed:.1f}s")
    return daily


# ============================================================================
# STEP 3: Merge and Build v3.0 Dataset
# ============================================================================
def build_v30(options_df: pd.DataFrame, barchart_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join options (base) with Barchart daily on date.
    Enforce 87-column v3.0 schema. Unpopulated columns stay blank (NaN).
    """
    print(f"\n[STEP 3] Merging options with Barchart daily ...")
    t0 = time.time()

    merged = options_df.merge(
        barchart_daily,
        on='_join_date',
        how='left',
        suffixes=('', '_bc'),
    )
    print(f"  Merged rows: {len(merged):,}")

    # If there are duplicate column names from the merge, drop the _bc copies
    bc_dupes = [c for c in merged.columns if c.endswith('_bc')]
    if bc_dupes:
        print(f"  Dropping duplicate-suffix columns: {bc_dupes}")
        merged = merged.drop(columns=bc_dupes)

    # Count Barchart coverage
    bc_hit = merged['underlying_price'].notna().sum()
    bc_pct = bc_hit / len(merged) * 100
    print(f"  Rows with Barchart data: {bc_hit:,} / {len(merged):,}  ({bc_pct:.1f}%)")

    # Drop join key
    merged = merged.drop(columns=['_join_date'], errors='ignore')

    # --- Enforce 87-column schema ---
    for col in V30_SCHEMA:
        if col not in merged.columns:
            merged[col] = np.nan

    result = merged[V30_SCHEMA].copy()

    # --- Report populated vs blank ---
    populated = []
    blank = []
    for col in V30_SCHEMA:
        if result[col].notna().any():
            populated.append(col)
        else:
            blank.append(col)

    elapsed = time.time() - t0
    print(f"\n  POPULATED ({len(populated)} columns):")
    for i, col in enumerate(populated):
        pct = result[col].notna().mean() * 100
        print(f"    {i+1:3d}. {col:30s}  {pct:6.1f}% filled")

    print(f"\n  BLANK ({len(blank)} columns) -- to be precomputed:")
    for i, col in enumerate(blank):
        print(f"    {i+1:3d}. {col}")

    print(f"\n  Total: {len(result.columns)} columns  |  Elapsed: {elapsed:.1f}s")
    return result


# ============================================================================
# STEP 4: Save
# ============================================================================
def save_output(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save the v3.0 dataset to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"condornet_v30_{datetime.now().strftime('%Y%m%d')}.csv"
    out_path = output_dir / out_name

    print(f"\n[STEP 4] Saving to {out_path.name} ...")
    t0 = time.time()
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {len(df):,} rows x {len(df.columns)} cols")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Elapsed: {elapsed:.1f}s")

    return out_path


# ============================================================================
# MAIN
# ============================================================================
def main():
    wall_start = time.time()

    print("=" * 70)
    print("  CONDORNET v3.0 DATASET CONSTRUCTION PIPELINE")
    print("  Source: options_2025.csv + Barchart M1 studies")
    print("  Schema: 87 columns (65 legacy + 3 raw pricing + 19 new v3.0)")
    print("=" * 70)

    # Validate source files exist
    for f in [OPTIONS_FILE, BARCHART_FILE]:
        if not f.exists():
            raise FileNotFoundError(f"Required source file not found: {f}")
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  Found: {f.name}  ({size_mb:.1f} MB)")

    # Step 1: Load options
    options_df = load_options(OPTIONS_FILE)

    # Step 2: Load & aggregate Barchart to daily
    barchart_daily = load_barchart_daily(BARCHART_FILE)

    # Step 3: Merge and build v3.0
    v30_df = build_v30(options_df, barchart_daily)

    # Step 4: Save
    out_path = save_output(v30_df, OUTPUT_DIR)

    # Final summary
    wall_elapsed = time.time() - wall_start
    print("\n" + "=" * 70)
    print("  CONSTRUCTION COMPLETE")
    print(f"  Output:  {out_path}")
    print(f"  Rows:    {len(v30_df):,}")
    print(f"  Columns: {len(v30_df.columns)}")
    print(f"  Total elapsed: {wall_elapsed:.1f}s")
    print("=" * 70)

    return v30_df


if __name__ == "__main__":
    main()
