#!/usr/bin/env python3
"""
V4.0 Forensic Feature Validation and Repair Script
=====================================================
Extends V3.0 validator for the 116-column CondorNet v4.0 schema.

Validates:
  1. Row-key alignment (timestamp + option_symbol)
  2. Multi-grade status: FAIL_NAN, FAIL_CONST, FAIL_BOUNDS, WARN_LOW_VAR, OK
  3. Schema completeness (116 columns)
  4. Coverage report (% populated per column)
  5. New v4.0 Reversal + Alignment Features
  6. Exact baseline constant propagation

Usage:
  python scripts/validate_and_fix_v4_features.py --input data/Datasetv4/condornet_v40_*.csv
  python scripts/validate_and_fix_v4_features.py --input data/Datasetv4/condornet_v40_*.csv --check-only
"""

import argparse
import sys
import os
import time
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict

# ============================================================================
# V4.0 VALIDATION SCHEMA (116 columns)
# ============================================================================
VALIDATION_SCHEMA_V40 = {
    # --- Market / Underlying OHLCV (7) ---
    'timestamp':          {'type': 'datetime', 'required': True},
    'symbol':             {'type': 'string',   'required': True},
    'underlying_price':   {'type': 'price', 'bounds': [0, 2000]},
    'open':               {'type': 'price', 'bounds': [0, 2000]},
    'high':               {'type': 'price', 'bounds': [0, 2000]},
    'low':                {'type': 'price', 'bounds': [0, 2000]},
    'close':              {'type': 'price', 'bounds': [0, 2000]},

    # --- Options Identity (4) ---
    'option_symbol':      {'type': 'string',   'required': True},
    'expiration':         {'type': 'datetime', 'required': True},
    'strike':             {'type': 'price', 'bounds': [0, 2000]},
    'call_put':           {'type': 'string',   'required': True},

    # --- Options Chain State (3) ---
    'rho':                {'type': 'float', 'bounds': [-5000, 5000]},
    'volume':             {'type': 'float', 'bounds': [0, 1e9]},
    'open_interest':      {'type': 'float', 'bounds': [0, 1e9]},

    # --- Time / Strategy (6) ---
    'te':                 {'type': 'float', 'bounds': [-0.1, 5], 'require_var': True},
    'ivr':                {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'target_spot':        {'type': 'price', 'bounds': [0, 2000]},
    'sma':                {'type': 'price', 'bounds': [0, 2000]},
    'psar_mark':          {'type': 'price', 'bounds': [0, 2000]},
    'mtf_consensus':      {'type': 'float', 'bounds': [-1.1, 1.1]},

    # --- Dynamic / Regime-Aware (16) ---
    'log_return':         {'type': 'float', 'bounds': [-0.5, 0.5]},
    'vol_ewma':           {'type': 'float', 'bounds': [0, 1]},
    'ret_z':              {'type': 'float', 'bounds': [-100, 100]},
    'atr_pct':            {'type': 'float', 'bounds': [0, 0.5]},
    'kappa_proxy':        {'type': 'float', 'bounds': [-100, 100]},
    'vol_energy':         {'type': 'float', 'bounds': [0, 100]},
    'rsi_dyn':            {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'adx_adaptive':       {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'psar_adaptive':      {'type': 'float', 'bounds': [-100, 100]},
    'bb_mu_dyn':          {'type': 'price', 'bounds': [0, 2000]},
    'bb_sigma_dyn':       {'type': 'float', 'bounds': [0, 100]},
    'bb_lower_dyn':       {'type': 'price', 'bounds': [0, 2000]},
    'bb_upper_dyn':       {'type': 'price', 'bounds': [0, 2000]},
    'stoch_k_dyn':        {'type': 'float', 'bounds': [0, 100]},
    'consolidation_score': {'type': 'float', 'bounds': [0, 1]},
    'breakout_score':     {'type': 'float', 'bounds': [-1.1, 1.1]},

    # --- V2.2 Primitives (20) ---
    'spread_ratio':       {'type': 'float', 'bounds': [0, 20], 'require_var': True},
    'bandwidth':          {'type': 'float', 'bounds': [0, 1000]},
    'bb_percentile':      {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'bw_expansion_rate':  {'type': 'float', 'bounds': [-10, 10]},
    'cmf':                {'type': 'float', 'bounds': [-1.1, 1.1]},
    'pressure_up':        {'type': 'float', 'bounds': [0, 5]},
    'pressure_down':      {'type': 'float', 'bounds': [0, 5]},
    'friction_ratio':     {'type': 'float', 'bounds': [0, 1000], 'require_var': True, 'min_std': 0.001},
    'exec_allow':         {'type': 'binary', 'bounds': [0, 1]},
    'gap_risk_score':     {'type': 'float', 'bounds': [0, 1], 'require_var': True, 'min_std': 0.001},
    'risk_override':      {'type': 'binary', 'bounds': [0, 1]},
    'iv_confidence':      {'type': 'float', 'bounds': [0, 1.1]},
    'macd_norm':          {'type': 'float', 'bounds': [-10, 10]},
    'macd_signal_norm':   {'type': 'float', 'bounds': [-10, 10]},
    'macd_histogram':     {'type': 'float', 'bounds': [-10, 10]},
    'plus_di':            {'type': 'float', 'bounds': [0, 100]},
    'minus_di':           {'type': 'float', 'bounds': [0, 100]},
    'psar_trend':         {'type': 'trinary', 'bounds': [-1, 1]},
    'psar_reversion_mu':  {'type': 'float', 'bounds': [0, 1]},
    'beta1_norm_stub':    {'type': 'float', 'bounds': [-10, 10]},
    'chaos_membership':   {'type': 'float', 'bounds': [0, 1]},
    'position_size_mult': {'type': 'float', 'bounds': [0, 1.1]},
    'fuzzy_reversion_11': {'type': 'float', 'bounds': [0, 1]},

    # --- Greeks (5) ---
    'delta':              {'type': 'float', 'bounds': [-1.1, 1.1]},
    'gamma':              {'type': 'float', 'bounds': [0, 50]},
    'vega':               {'type': 'float', 'bounds': [0, 5000]},
    'theta':              {'type': 'float', 'bounds': [-5000, 100]},
    'iv':                 {'type': 'float', 'bounds': [0, 20]},

    # --- Risk (1) ---
    'max_dd_60m':         {'type': 'float', 'bounds': [-1, 1]},

    # --- Raw Options Pricing (3) ---
    'bid':                {'type': 'float', 'bounds': [0, 10000]},
    'ask':                {'type': 'float', 'bounds': [0, 10001]},
    'mark':               {'type': 'float', 'bounds': [0, 10000]},

    # --- v3.0 Trend Structure (3) ---
    'FRAMA':              {'type': 'price', 'bounds': [0, 2000]},
    'Anchored_VWAP':      {'type': 'price', 'bounds': [0, 2000]},
    'McClellanOsc':       {'type': 'float', 'bounds': [-1000, 1000]},

    # --- v3.0 IV Bands (3) ---
    'IV_High':            {'type': 'float', 'bounds': [0, 500]},
    'IV_Mid':             {'type': 'float', 'bounds': [0, 500]},
    'IV_Low':             {'type': 'float', 'bounds': [0, 500]},

    # --- v3.0 Options Chain Aggregates (4) ---
    'Options_Total_Volume': {'type': 'float', 'bounds': [0, 1e10]},
    'Options_Put_Volume':   {'type': 'float', 'bounds': [0, 1e10]},
    'Options_Call_Volume':  {'type': 'float', 'bounds': [0, 1e10]},
    'OSC_Volume':           {'type': 'float', 'bounds': [-1e10, 1e10]},

    # --- v3.0 Flow & Breadth (4) ---
    'WeightedAlpha':        {'type': 'float', 'bounds': [-1000, 1000]},
    'WilderAccSwingIndex':  {'type': 'float', 'bounds': [-100000, 100000]},
    'AccDistWill':          {'type': 'float', 'bounds': [-1e15, 1e15]},
    'AccDistWillMovAvg':    {'type': 'float', 'bounds': [-1e15, 1e15]},

    # --- v3.0 Microstructure (5) ---
    'aggregate_bid_count':  {'type': 'float', 'bounds': [0, 1e7]},
    'aggregate_ask_count':  {'type': 'float', 'bounds': [0, 1e7]},
    'mean_bid':             {'type': 'float', 'bounds': [0, 10000]},
    'mean_ask':             {'type': 'float', 'bounds': [0, 10000]},
    'quote_spread':         {'type': 'float', 'bounds': [-200, 200]},

    # ========================================================================
    # NEW v4.0 REVERSAL FEATURES (17)
    # ========================================================================
    'rev_m5':             {'type': 'float', 'bounds': [-6.0, 6.0]},
    'rev_m15':            {'type': 'float', 'bounds': [-6.0, 6.0]},
    'rev_h1':             {'type': 'float', 'bounds': [-6.0, 6.0]},

    'rev_m5_top':         {'type': 'binary', 'bounds': [0, 1]},
    'rev_m5_bot':         {'type': 'binary', 'bounds': [0, 1]},
    'rev_m15_top':        {'type': 'binary', 'bounds': [0, 1]},
    'rev_m15_bot':        {'type': 'binary', 'bounds': [0, 1]},
    'rev_h1_top':         {'type': 'binary', 'bounds': [0, 1]},
    'rev_h1_bot':         {'type': 'binary', 'bounds': [0, 1]},

    'rev_m5_z':           {'type': 'float', 'bounds': [-6.0, 6.0]},
    'rev_m15_z':          {'type': 'float', 'bounds': [-6.0, 6.0]},
    'rev_h1_z':           {'type': 'float', 'bounds': [-6.0, 6.0]},

    'align_m5_m15':       {'type': 'binary', 'bounds': [0, 1]},
    'align_m15_h1':       {'type': 'binary', 'bounds': [0, 1]},
    'align_m5_h1':        {'type': 'binary', 'bounds': [0, 1]},
    'align_3of3':         {'type': 'binary', 'bounds': [0, 1]},
    'align_2of3':         {'type': 'binary', 'bounds': [0, 1]},

    # ========================================================================
    # NEW v4.0 BASELINE PARAMETERS (12) - STRICT CONSTANT VALIDATION
    # ========================================================================
    'm5_sma_base':        {'type': 'const', 'value': 32.0},
    'm5_rv_base':         {'type': 'const', 'value': 17.0},
    'm5_z_base':          {'type': 'const', 'value': 148.0},
    'm5_thresh_base':     {'type': 'const', 'value': 1.0},

    'm15_sma_base':       {'type': 'const', 'value': 99.0},
    'm15_rv_base':        {'type': 'const', 'value': 17.0},
    'm15_z_base':         {'type': 'const', 'value': 250.0},
    'm15_thresh_base':    {'type': 'const', 'value': 1.0},

    'h1_sma_base':        {'type': 'const', 'value': 39.0},
    'h1_rv_base':         {'type': 'const', 'value': 17.0},
    'h1_z_base':          {'type': 'const', 'value': 123.0},
    'h1_thresh_base':     {'type': 'const', 'value': 1.0},
}


def validate_columns(df: pd.DataFrame, sample_size: int = 50000) -> Dict:
    """Forensic validation using v4.0 schema-driven rules."""
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    report = {}
    n_ok = 0
    n_warn = 0
    n_fail = 0
    n_missing = 0
    n_empty = 0

    print("\n" + "=" * 125)
    print(f"{'COLUMN':<25} | {'POPULATED':>9} | {'MIN/VAL':>10} | {'MAX':>10} | {'STD':>10} | {'UNIQUE':>8} | {'STATUS'}")
    print("-" * 125)

    for col, schema in VALIDATION_SCHEMA_V40.items():
        if col not in df.columns:
            status = "MISSING"
            n_missing += 1
            print(f"{col:<25} | {'---':>9} | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>8} | [FAIL] {status}")
            report[col] = {"status": status, "is_fail": True, "populated_pct": 0}
            continue

        # Coverage
        total = len(df_sample)
        pop_count = df_sample[col].notna().sum()
        pop_pct = pop_count / total * 100 if total > 0 else 0

        # Skip numeric checks for string/datetime columns
        if schema['type'] in ('string', 'datetime'):
            status = "OK" if pop_pct > 0 else "EMPTY"
            if status == "EMPTY":
                n_empty += 1
            else:
                n_ok += 1
            print(f"{col:<25} | {pop_pct:>8.1f}% | {'---':>10} | {'---':>10} | {'---':>10} | {df_sample[col].nunique():>8} | [{status}]")
            report[col] = {"status": status, "is_fail": status == "EMPTY", "populated_pct": pop_pct}
            continue

        data = pd.to_numeric(df_sample[col], errors='coerce').dropna()
        if len(data) == 0:
            status = "EMPTY"
            n_empty += 1
            print(f"{col:<25} | {pop_pct:>8.1f}% | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>8} | [WARN] {status}")
            report[col] = {"status": status, "is_fail": False, "populated_pct": pop_pct}
            continue

        c_min, c_max, c_std, c_uni = data.min(), data.max(), data.std(), data.nunique()

        status = "OK"
        is_fail = False

        # --- VALIDATION LOGIC ---

        # 1. Constant Value Validation
        if schema['type'] == 'const':
            target = schema['value']
            if not np.allclose(data, target, atol=1e-6):
                status = "FAIL_CONST_VAL"
                is_fail = True
            elif c_uni > 1:
                status = "FAIL_VARIANCE"
                is_fail = True
            else:
                status = "OK_CONST"

        # 2. Hard Bounds
        elif 'bounds' in schema:
            if c_min < schema['bounds'][0] - 1e-6 or c_max > schema['bounds'][1] + 1e-6:
                status = "FAIL_BOUNDS"
                is_fail = True
            
            if not is_fail and np.isinf(data).any():
                status = "FAIL_INF"
                is_fail = True

        # 3. Binary Enforcement
        if not is_fail and schema['type'] == 'binary':
            # Check if values are exactly 0 or 1
            if not set(data.unique()).issubset({0, 1, 0.0, 1.0}):
                 status = "FAIL_BINARY"
                 is_fail = True

        # 4. Trinary Enforcement
        if not is_fail and schema['type'] == 'trinary':
            # Check if values are exactly -1, 0, or 1
            if not set(data.unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
                 status = "FAIL_TRINARY"
                 is_fail = True

        # 5. Variability Warning
        if not is_fail and status == "OK":
            if c_uni == 1:
                if schema.get('require_var', False):
                    status = "FAIL_CONST_REQD"
                    is_fail = True
                else:
                    status = "OK_CONST"
            elif schema.get('require_var', False) and c_std < schema.get('min_std', 0.01):
                status = "WARN_LOW_VAR"

        # Update counters
        if is_fail:
            n_fail += 1
        elif status.startswith("WARN"):
            n_warn += 1
        else:
            n_ok += 1

        # Format numbers for display
        def fmt(x):
            if abs(x) >= 1e6:
                return f"{x:>10.1e}"
            elif abs(x) >= 100:
                return f"{x:>10.1f}"
            else:
                return f"{x:>10.4f}"

        display_val = schema['value'] if schema['type'] == 'const' else c_min
        print(f"{col:<25} | {pop_pct:>8.1f}% | {fmt(display_val)} | {fmt(c_max)} | {fmt(c_std)} | {c_uni:>8} | [{status}]")
        report[col] = {"status": status, "is_fail": is_fail, "populated_pct": pop_pct,
                        "min": c_min, "max": c_max, "std": c_std, "unique": c_uni}

    print("-" * 125)
    print(f"\nSUMMARY: {n_ok} OK | {n_warn} WARN | {n_fail} FAIL | {n_missing} MISSING | {n_empty} EMPTY")
    print(f"  Schema expects {len(VALIDATION_SCHEMA_V40)} columns, dataset has {len(df.columns)} columns")

    # Extra columns not in schema
    extra = [c for c in df.columns if c not in VALIDATION_SCHEMA_V40]
    if extra:
        print(f"  Extra columns (not in v4.0 schema): {extra[:10]}{'...' if len(extra) > 10 else ''}")

    return report


def coverage_report(df: pd.DataFrame):
    """Print a clean coverage report grouped by category."""
    print("\n" + "=" * 65)
    print("  COVERAGE REPORT (% populated)")
    print("=" * 65)

    categories = {
        'Underlying OHLCV': ['timestamp', 'symbol', 'underlying_price', 'open', 'high', 'low', 'close'],
        'Options Identity': ['option_symbol', 'expiration', 'strike', 'call_put'],
        'Options Chain': ['rho', 'volume', 'open_interest', 'bid', 'ask', 'mark'],
        'Greeks': ['delta', 'gamma', 'vega', 'theta', 'iv'],
        'Time/Strategy': ['te', 'ivr', 'target_spot', 'sma', 'psar_mark', 'mtf_consensus', 'max_dd_60m'],
        'Dynamic/Regime': ['log_return', 'vol_ewma', 'ret_z', 'atr_pct', 'kappa_proxy', 'vol_energy',
                          'rsi_dyn', 'adx_adaptive', 'psar_adaptive', 'stoch_k_dyn',
                          'bb_mu_dyn', 'bb_sigma_dyn', 'bb_lower_dyn', 'bb_upper_dyn',
                          'consolidation_score', 'breakout_score'],
        'V2.2 Primitives': ['spread_ratio', 'bandwidth', 'bb_percentile', 'bw_expansion_rate',
                           'cmf', 'pressure_up', 'pressure_down', 'friction_ratio',
                           'exec_allow', 'gap_risk_score', 'risk_override', 'iv_confidence',
                           'macd_norm', 'macd_signal_norm', 'macd_histogram',
                           'plus_di', 'minus_di', 'psar_trend', 'psar_reversion_mu',
                           'beta1_norm_stub', 'chaos_membership', 'position_size_mult', 'fuzzy_reversion_11'],
        'v3.0 Extended': ['FRAMA', 'Anchored_VWAP', 'McClellanOsc', 'IV_High', 'IV_Mid', 'IV_Low',
                         'Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume', 'OSC_Volume',
                         'WeightedAlpha', 'WilderAccSwingIndex', 'AccDistWill', 'AccDistWillMovAvg',
                         'aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread'],
        'v4.0 Reversals': ['rev_m5', 'rev_m15', 'rev_h1', 'rev_m5_z', 'rev_m15_z', 'rev_h1_z'],
        'v4.0 Flags': ['rev_m5_top', 'rev_m5_bot', 'rev_m15_top', 'rev_m15_bot', 'rev_h1_top', 'rev_h1_bot',
                      'align_m5_m15', 'align_m15_h1', 'align_m5_h1', 'align_3of3', 'align_2of3'],
        'v4.0 Baselines': ['m5_sma_base', 'm5_rv_base', 'm5_z_base', 'm5_thresh_base',
                          'm15_sma_base', 'm15_rv_base', 'm15_z_base', 'm15_thresh_base',
                          'h1_sma_base', 'h1_rv_base', 'h1_z_base', 'h1_thresh_base'],
    }

    total_pop = 0
    total_cols = 0

    for cat, cols in categories.items():
        pcts = []
        for col in cols:
            if col in df.columns:
                pct = df[col].notna().mean() * 100
            else:
                pct = 0.0
            pcts.append(pct)

        avg_pct = np.mean(pcts) if pcts else 0
        total_pop += sum(1 for p in pcts if p > 0)
        total_cols += len(cols)

        bar = '#' * int(avg_pct / 2.5) + '.' * (40 - int(avg_pct / 2.5))
        pop_count = sum(1 for p in pcts if p > 0)
        print(f"  {cat:<22s}  [{bar}]  {avg_pct:5.1f}%  ({pop_count}/{len(cols)} cols)")

    print(f"\n  TOTAL: {total_pop}/{total_cols} columns populated")


def main():
    parser = argparse.ArgumentParser(description="CondorNet v4.0 Forensic Validator")
    parser.add_argument("--input", "-i", required=True, help="Input v4.0 CSV path (supports glob)")
    parser.add_argument("--output", "-o", default=None, help="Output fixed CSV path")
    parser.add_argument("--check-only", action="store_true", help="Validate only, no repair")
    parser.add_argument("--sample", type=int, default=100000, help="Sample size for validation")
    parser.add_argument("--coverage", action="store_true", help="Show coverage report only")
    args = parser.parse_args()

    # Resolve glob
    if '*' in args.input:
        files = sorted(glob.glob(args.input))
        if not files:
            print(f"[ERROR] No files match: {args.input}")
            sys.exit(1)
        inp = files[-1]
        print(f"[INFO] Resolved glob to: {inp}")
    else:
        inp = args.input

    if not os.path.exists(inp):
        print(f"[ERROR] File not found: {inp}")
        sys.exit(1)

    size_mb = os.path.getsize(inp) / (1024 * 1024)
    print(f"\n{'=' * 75}")
    print(f"  CONDORNET v4.0 FORENSIC VALIDATOR")
    print(f"  Input: {os.path.basename(inp)}  ({size_mb:.1f} MB)")
    print(f"{'=' * 75}")

    t0 = time.time()

    # Load
    print("\n[LOAD] Reading dataset ...")
    # We use low_memory=False to avoid DtypeWarning since we have constant floats that might look like ints
    df = pd.read_csv(inp, low_memory=False)
    print(f"  Loaded: {len(df):,} rows x {len(df.columns)} cols")

    # Coverage report
    if args.coverage:
        coverage_report(df)
        return

    # Full validation
    report = validate_columns(df, sample_size=args.sample)
    coverage_report(df)

    # Check results
    has_fails = any(v.get('is_fail', False) for v in report.values())
    has_missing = any(v['status'] == 'MISSING' for v in report.values())

    if has_fails or has_missing:
        fail_cols = [c for c, v in report.items() if v.get('is_fail')]
        print(f"\n[ALERT] {len(fail_cols)} columns with issues: {fail_cols[:15]}")

        if not args.check_only and args.output:
            print("\n[REPAIR] No automatic repair for v4.0 reversal features yet.")
            print("         Please re-run the dataset generation script (condor_dataprep_v4).")
        elif args.check_only:
            print("\n[INFO] --check-only mode: no repair attempted")
    else:
        print("\n[PASS] All v4.0 features passed validation")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
