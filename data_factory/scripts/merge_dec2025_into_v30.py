#!/usr/bin/env python3
"""
Merge December 2025 Alpaca Options Bars into CondorNet v3.0 Dataset
====================================================================
Reads the Alpaca 1-minute options bars CSV (dec_2025_options_bars.csv),
aggregates to daily per-contract, maps to the 87-column v3.0 schema,
joins with Barchart daily studies, and appends to the existing v3.0 dataset.

Alpaca bars columns: symbol, timestamp, open, high, low, close, volume, trade_count, vwap
  - These are OPTION price bars (not underlying)
  - Greeks/IV/bid/ask/open_interest are NOT available from bars endpoint

OCC Symbol Format: SPY251215C00682000
  - SPY       = underlying (chars 0-2)
  - 251215    = expiration YYMMDD (chars 3-8)
  - C         = call/put (char 9)
  - 00682000  = strike * 1000 (chars 10-17) -> 682.000

Usage:
    python data_factory/scripts/merge_dec2025_into_v30.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import re
import glob as globmod

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Datasetv3"
DOWNLOADERS_DIR = PROJECT_ROOT / "data_factory" / "downloaders"

# Alpaca bars output (from download_dec2025_options.ps1)
ALPACA_BARS_FILE = DOWNLOADERS_DIR / "dec_2025_options_bars.csv"

# Barchart M1 studies (for underlying price + studies join)
BARCHART_FILE = DATA_DIR / "SPY_Barchart_Interactive_Chart_Range_1m_02_09_2026.csv"

# Existing v3.0 dataset (find most recent)
V30_PATTERN = str(DATA_DIR / "condornet_v30_*.csv")

# ============================================================================
# V3.0 SCHEMA -- 87 columns (must match build_dataset_v30.py)
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

assert len(V30_SCHEMA) == 87, f"Schema has {len(V30_SCHEMA)} columns, expected 87"


# ============================================================================
# STEP 1: Parse OCC Option Symbols
# ============================================================================
def parse_occ_symbol(occ: str) -> dict:
    """
    Parse an OCC option symbol into components.

    Format: SPY251215C00682000
      - Underlying: SPY (first 3+ chars before the date)
      - Expiration: 251215 -> 2025-12-15
      - Type: C or P
      - Strike: 00682000 -> 682.000
    """
    # OCC format: underlying(variable) + YYMMDD + C/P + strike*1000 (8 digits)
    # For SPY: always 3-char underlying
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', occ)
    if not match:
        return None

    underlying = match.group(1)
    date_str = match.group(2)
    cp = match.group(3)
    strike_raw = match.group(4)

    # Parse expiration: YYMMDD -> 20YY-MM-DD
    year = 2000 + int(date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])

    return {
        'symbol': underlying,
        'expiration': pd.Timestamp(year=year, month=month, day=day),
        'call_put': 'call' if cp == 'C' else 'put',
        'strike': int(strike_raw) / 1000.0,
    }


# ============================================================================
# STEP 2: Load and Aggregate Alpaca Bars to Daily
# ============================================================================
def load_alpaca_bars(filepath: Path) -> pd.DataFrame:
    """
    Load Alpaca 1-min options bars and aggregate to daily per-contract.

    Input columns: symbol, timestamp, open, high, low, close, volume, trade_count, vwap
    Output: one row per contract per day with OHLCV aggregation + parsed OCC fields.
    """
    print(f"\n[STEP 1] Loading Alpaca options bars: {filepath.name}")
    t0 = time.time()

    df = pd.read_csv(filepath)
    print(f"  Raw rows: {len(df):,}  |  Columns: {list(df.columns)}")
    print(f"  Unique contracts: {df['symbol'].nunique()}")

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).copy()
    df['_join_date'] = df['timestamp'].dt.date

    # --- Daily aggregation per contract ---
    daily = df.groupby(['symbol', '_join_date']).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        trade_count=('trade_count', 'sum'),
        vwap=('vwap', 'last'),
    ).reset_index()

    print(f"  Aggregated to {len(daily):,} contract-days")
    print(f"  Date range: {daily['_join_date'].min()} to {daily['_join_date'].max()}")

    # --- Parse OCC symbols ---
    parsed_records = []
    failed = 0
    for _, row in daily.iterrows():
        parsed = parse_occ_symbol(row['symbol'])
        if parsed is None:
            failed += 1
            continue
        parsed_records.append({
            '_join_date': row['_join_date'],
            'option_symbol': row['symbol'],
            'symbol': parsed['symbol'],
            'expiration': parsed['expiration'],
            'call_put': parsed['call_put'],
            'strike': parsed['strike'],
            'volume': row['volume'],
            # Use daily close as "mark" (best estimate of settlement value)
            'mark': row['close'],
            # Store OHLCV as option-level pricing (not underlying)
            '_opt_open': row['open'],
            '_opt_high': row['high'],
            '_opt_low': row['low'],
            '_opt_close': row['close'],
            '_opt_vwap': row['vwap'],
        })

    if failed > 0:
        print(f"  WARNING: {failed} rows failed OCC symbol parsing")

    result = pd.DataFrame(parsed_records)
    result['timestamp'] = pd.to_datetime(result['_join_date'])

    # Compute te (time-to-expiry in years)
    result['te'] = (result['expiration'] - result['timestamp']).dt.days / 365.0

    elapsed = time.time() - t0
    print(f"  Parsed {len(result):,} contract-days")
    print(f"  Unique dates: {result['_join_date'].nunique()}")
    print(f"  Unique contracts: {result['option_symbol'].nunique()}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return result


# ============================================================================
# STEP 3: Load Barchart Daily (reuse logic from build script)
# ============================================================================
def load_barchart_daily(filepath: Path) -> pd.DataFrame:
    """Load Barchart M1 studies and aggregate to daily (same as build_dataset_v30.py)."""
    print(f"\n[STEP 2] Loading Barchart M1 studies: {filepath.name}")
    t0 = time.time()

    df = pd.read_csv(filepath, skiprows=1, encoding='latin-1')
    print(f"  Raw rows: {len(df):,}")

    # Drop footer row
    mask = df['Date Time'].astype(str).str.match(r'^\d{4}', na=False)
    df = df[mask].copy()

    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df['_join_date'] = df['Date Time'].dt.date

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

    for col in BARCHART_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Daily aggregation
    ohlcv_agg = df.groupby('_join_date').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
    ).reset_index()

    study_cols = [
        'sma', 'FRAMA', 'Anchored_VWAP', 'McClellanOsc',
        'IV_High', 'IV_Mid', 'IV_Low',
        'Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume',
        'OSC_Volume', 'WeightedAlpha', 'WilderAccSwingIndex',
        'AccDistWill', 'AccDistWillMovAvg',
    ]
    available_studies = [c for c in study_cols if c in df.columns]
    study_agg = df.groupby('_join_date')[available_studies].last().reset_index()

    daily = ohlcv_agg.merge(study_agg, on='_join_date', how='outer')
    daily['underlying_price'] = daily['close']

    # Filter to Dec 2025 only (the dates we need)
    from datetime import date
    dec_mask = daily['_join_date'].apply(lambda d: d >= date(2025, 12, 13))
    dec_daily = daily[dec_mask].copy()

    elapsed = time.time() - t0
    print(f"  Barchart Dec 2025+ days: {len(dec_daily)}")
    print(f"  Date range: {dec_daily['_join_date'].min()} to {dec_daily['_join_date'].max()}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return dec_daily


# ============================================================================
# STEP 4: Merge and Build v3.0 Rows
# ============================================================================
def build_dec_v30(alpaca_daily: pd.DataFrame, barchart_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join Alpaca contract-days with Barchart daily on date.
    Map to 87-column v3.0 schema. Unpopulated columns stay NaN.
    """
    print(f"\n[STEP 3] Merging Alpaca options with Barchart daily ...")
    t0 = time.time()

    merged = alpaca_daily.merge(
        barchart_daily,
        on='_join_date',
        how='left',
        suffixes=('', '_bc'),
    )
    print(f"  Merged rows: {len(merged):,}")

    # Drop _bc suffix duplicates
    bc_dupes = [c for c in merged.columns if c.endswith('_bc')]
    if bc_dupes:
        merged = merged.drop(columns=bc_dupes)

    # Barchart coverage
    bc_hit = merged['underlying_price'].notna().sum()
    bc_pct = bc_hit / len(merged) * 100 if len(merged) > 0 else 0
    print(f"  Rows with Barchart data: {bc_hit:,} / {len(merged):,}  ({bc_pct:.1f}%)")

    # Drop internal columns
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith('_')], errors='ignore')

    # Enforce 87-column schema
    for col in V30_SCHEMA:
        if col not in merged.columns:
            merged[col] = np.nan

    result = merged[V30_SCHEMA].copy()

    # Report
    populated = [c for c in V30_SCHEMA if result[c].notna().any()]
    blank = [c for c in V30_SCHEMA if not result[c].notna().any()]

    elapsed = time.time() - t0
    print(f"\n  POPULATED ({len(populated)} columns):")
    for i, col in enumerate(populated):
        pct = result[col].notna().mean() * 100
        print(f"    {i+1:3d}. {col:30s}  {pct:6.1f}% filled")

    print(f"\n  BLANK ({len(blank)} columns):")
    for i, col in enumerate(blank):
        print(f"    {i+1:3d}. {col}")

    print(f"\n  Total: {len(result.columns)} columns  |  Elapsed: {elapsed:.1f}s")
    return result


# ============================================================================
# STEP 5: Append to Existing v3.0 Dataset
# ============================================================================
def append_to_v30(dec_df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Find the existing v3.0 dataset, append Dec rows, save combined output.
    """
    print(f"\n[STEP 4] Appending to existing v3.0 dataset ...")
    t0 = time.time()

    # Find existing v3.0 file
    existing_files = sorted(globmod.glob(V30_PATTERN))
    if existing_files:
        existing_path = Path(existing_files[-1])  # most recent
        print(f"  Loading existing: {existing_path.name}")
        existing_df = pd.read_csv(existing_path, low_memory=False)
        print(f"  Existing rows: {len(existing_df):,}")

        # Check for date overlap
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], errors='coerce')
        existing_max = existing_df['timestamp'].max()
        print(f"  Existing date range ends: {existing_max}")

        # Concatenate
        combined = pd.concat([existing_df, dec_df], ignore_index=True)
        print(f"  Combined rows: {len(combined):,}")
    else:
        print(f"  No existing v3.0 file found, saving Dec data only")
        combined = dec_df

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"condornet_v30_{datetime.now().strftime('%Y%m%d')}_full.csv"
    out_path = output_dir / out_name

    combined.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {out_path.name}")
    print(f"  Total rows: {len(combined):,} x {len(combined.columns)} cols")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Elapsed: {elapsed:.1f}s")

    return out_path


# ============================================================================
# MAIN
# ============================================================================
def main():
    wall_start = time.time()

    print("=" * 70)
    print("  MERGE DEC 2025 ALPACA OPTIONS INTO CONDORNET v3.0")
    print("  Source: dec_2025_options_bars.csv (Alpaca 1-min bars)")
    print("  Join:   Barchart M1 studies (for underlying + studies)")
    print("  Target: Append to existing condornet_v30_*.csv")
    print("=" * 70)

    # Validate source files
    if not ALPACA_BARS_FILE.exists():
        raise FileNotFoundError(
            f"Alpaca bars file not found: {ALPACA_BARS_FILE}\n"
            f"Run download_dec2025_options.ps1 first."
        )
    size_mb = ALPACA_BARS_FILE.stat().st_size / (1024 * 1024)
    print(f"  Found: {ALPACA_BARS_FILE.name}  ({size_mb:.2f} MB)")

    if not BARCHART_FILE.exists():
        raise FileNotFoundError(f"Barchart file not found: {BARCHART_FILE}")
    print(f"  Found: {BARCHART_FILE.name}")

    # Step 1: Load & aggregate Alpaca bars
    alpaca_daily = load_alpaca_bars(ALPACA_BARS_FILE)

    # Step 2: Load Barchart daily (Dec 2025 portion)
    barchart_daily = load_barchart_daily(BARCHART_FILE)

    # Step 3: Merge into v3.0 schema
    dec_v30 = build_dec_v30(alpaca_daily, barchart_daily)

    # Step 4: Append to existing dataset
    out_path = append_to_v30(dec_v30, DATA_DIR)

    # Summary
    wall_elapsed = time.time() - wall_start
    print("\n" + "=" * 70)
    print("  MERGE COMPLETE")
    print(f"  Output:  {out_path}")
    print(f"  Dec rows added: {len(dec_v30):,}")
    print(f"  Total elapsed: {wall_elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
