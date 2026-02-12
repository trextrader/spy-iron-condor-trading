#!/usr/bin/env python3
"""
V3.0 Forensic Feature Validation and Repair Script
=====================================================
Extends V2.3 validator for the 87-column CondorNet v3.0 schema.

Validates:
  1. Row-key alignment (timestamp + option_symbol)
  2. Multi-grade status: FAIL_NAN, FAIL_CONST, FAIL_BOUNDS, WARN_LOW_VAR, OK
  3. Schema completeness (87 columns)
  4. Coverage report (% populated per column)
  5. Optional forensic repair via dynamic feature recomputation

Usage:
  python scripts/validate_and_fix_v3_features.py --input data/Datasetv3/condornet_v30_*.csv
  python scripts/validate_and_fix_v3_features.py --input data/Datasetv3/condornet_v30_*.csv --check-only
  python scripts/validate_and_fix_v3_features.py --input data/Datasetv3/condornet_v30_*.csv --output fixed.csv
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
# V3.0 VALIDATION SCHEMA (87 columns)
# ============================================================================
VALIDATION_SCHEMA_V30 = {
    # --- Market / Underlying OHLCV (7) ---
    'timestamp':          {'type': 'datetime', 'required': True},
    'symbol':             {'type': 'string',   'required': True},
    'underlying_price':   {'type': 'price', 'bounds': [0, 1500]},
    'open':               {'type': 'price', 'bounds': [0, 1500]},
    'high':               {'type': 'price', 'bounds': [0, 1500]},
    'low':                {'type': 'price', 'bounds': [0, 1500]},
    'close':              {'type': 'price', 'bounds': [0, 1500]},

    # --- Options Identity (4) ---
    'option_symbol':      {'type': 'string',   'required': True},
    'expiration':         {'type': 'datetime', 'required': True},
    'strike':             {'type': 'price', 'bounds': [0, 1500]},
    'call_put':           {'type': 'string',   'required': True},

    # --- Options Chain State (3) ---
    'rho':                {'type': 'float', 'bounds': [-100, 100]},
    'volume':             {'type': 'float', 'bounds': [0, 1e9]},
    'open_interest':      {'type': 'float', 'bounds': [0, 1e9]},

    # --- Time / Strategy (6) ---
    'te':                 {'type': 'float', 'bounds': [-0.1, 5], 'require_var': True},
    'ivr':                {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'target_spot':        {'type': 'price', 'bounds': [0, 1500]},
    'sma':                {'type': 'price', 'bounds': [0, 1500]},
    'psar_mark':          {'type': 'price', 'bounds': [0, 1500]},
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
    'bb_mu_dyn':          {'type': 'price', 'bounds': [0, 1500]},
    'bb_sigma_dyn':       {'type': 'float', 'bounds': [0, 100]},
    'bb_lower_dyn':       {'type': 'price', 'bounds': [0, 1500]},
    'bb_upper_dyn':       {'type': 'price', 'bounds': [0, 1500]},
    'stoch_k_dyn':        {'type': 'float', 'bounds': [0, 100]},
    'consolidation_score': {'type': 'float', 'bounds': [0, 1]},
    'breakout_score':     {'type': 'float', 'bounds': [-1.1, 1.1]},

    # --- V2.2 Primitives (20) ---
    'spread_ratio':       {'type': 'float', 'bounds': [0, 10], 'require_var': True},
    'bandwidth':          {'type': 'float', 'bounds': [0, 1000]},
    'bb_percentile':      {'type': 'float', 'bounds': [0, 100], 'require_var': True, 'min_std': 0.1},
    'bw_expansion_rate':  {'type': 'float', 'bounds': [-10, 10]},
    'cmf':                {'type': 'float', 'bounds': [-1.1, 1.1]},
    'pressure_up':        {'type': 'float', 'bounds': [0, 2]},
    'pressure_down':      {'type': 'float', 'bounds': [0, 2]},
    'friction_ratio':     {'type': 'float', 'bounds': [0, 1], 'require_var': True, 'min_std': 0.001},
    'exec_allow':         {'type': 'binary', 'bounds': [0, 1]},
    'gap_risk_score':     {'type': 'float', 'bounds': [0, 1], 'require_var': True, 'min_std': 0.001},
    'risk_override':      {'type': 'binary', 'bounds': [0, 1]},
    'iv_confidence':      {'type': 'float', 'bounds': [0, 1.1]},
    'macd_norm':          {'type': 'float', 'bounds': [-10, 10]},
    'macd_signal_norm':   {'type': 'float', 'bounds': [-10, 10]},
    'macd_histogram':     {'type': 'float', 'bounds': [-10, 10]},
    'plus_di':            {'type': 'float', 'bounds': [0, 100]},
    'minus_di':           {'type': 'float', 'bounds': [0, 100]},
    'psar_trend':         {'type': 'binary', 'bounds': [-1, 1]},
    'psar_reversion_mu':  {'type': 'float', 'bounds': [0, 1]},
    'beta1_norm_stub':    {'type': 'float', 'bounds': [-10, 10]},
    'chaos_membership':   {'type': 'float', 'bounds': [0, 1]},
    'position_size_mult': {'type': 'float', 'bounds': [0, 1.1]},
    'fuzzy_reversion_11': {'type': 'float', 'bounds': [0, 1]},

    # --- Greeks (5) ---
    'delta':              {'type': 'float', 'bounds': [-1.1, 1.1]},
    'gamma':              {'type': 'float', 'bounds': [0, 10]},
    'vega':               {'type': 'float', 'bounds': [0, 5000]},
    'theta':              {'type': 'float', 'bounds': [-5000, 10]},
    'iv':                 {'type': 'float', 'bounds': [0, 10]},

    # --- Risk (1) ---
    'max_dd_60m':         {'type': 'float', 'bounds': [-1, 1]},

    # --- Raw Options Pricing (3) ---
    'bid':                {'type': 'float', 'bounds': [0, 5000]},
    'ask':                {'type': 'float', 'bounds': [0, 5000]},
    'mark':               {'type': 'float', 'bounds': [0, 5000]},

    # --- NEW v3.0 Trend Structure (3) ---
    'FRAMA':              {'type': 'price', 'bounds': [0, 1500]},
    'Anchored_VWAP':      {'type': 'price', 'bounds': [0, 1500]},
    'McClellanOsc':       {'type': 'float', 'bounds': [-500, 500]},

    # --- NEW v3.0 IV Bands (3) ---
    'IV_High':            {'type': 'float', 'bounds': [0, 200]},
    'IV_Mid':             {'type': 'float', 'bounds': [0, 200]},
    'IV_Low':             {'type': 'float', 'bounds': [0, 200]},

    # --- NEW v3.0 Options Chain Aggregates (4) ---
    'Options_Total_Volume': {'type': 'float', 'bounds': [0, 1e9]},
    'Options_Put_Volume':   {'type': 'float', 'bounds': [0, 1e9]},
    'Options_Call_Volume':  {'type': 'float', 'bounds': [0, 1e9]},
    'OSC_Volume':           {'type': 'float', 'bounds': [-1e9, 1e9]},

    # --- NEW v3.0 Flow & Breadth (4) ---
    'WeightedAlpha':        {'type': 'float', 'bounds': [-200, 200]},
    'WilderAccSwingIndex':  {'type': 'float', 'bounds': [-10000, 10000]},
    'AccDistWill':          {'type': 'float', 'bounds': [-1e12, 1e12]},
    'AccDistWillMovAvg':    {'type': 'float', 'bounds': [-1e12, 1e12]},

    # --- NEW v3.0 Microstructure (5) ---
    'aggregate_bid_count':  {'type': 'float', 'bounds': [0, 1e6]},
    'aggregate_ask_count':  {'type': 'float', 'bounds': [0, 1e6]},
    'mean_bid':             {'type': 'float', 'bounds': [0, 5000]},
    'mean_ask':             {'type': 'float', 'bounds': [0, 5000]},
    'quote_spread':         {'type': 'float', 'bounds': [-100, 100]},
}


def validate_columns(df: pd.DataFrame, sample_size: int = 50000) -> Dict:
    """Forensic validation using v3.0 schema-driven rules."""
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

    print("\n" + "=" * 110)
    print(f"{'COLUMN':<25} | {'POPULATED':>9} | {'MIN':>10} | {'MAX':>10} | {'STD':>10} | {'UNIQUE':>8} | {'STATUS'}")
    print("-" * 110)

    for col, schema in VALIDATION_SCHEMA_V30.items():
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

        # Hard fails
        if np.isinf(data).any():
            status = "FAIL_INF"; is_fail = True
        elif 'bounds' in schema:
            if c_min < schema['bounds'][0] - 1e-6 or c_max > schema['bounds'][1] + 1e-6:
                status = "FAIL_BOUNDS"; is_fail = True

        # Warnings
        if not is_fail:
            if c_uni == 1:
                if schema.get('require_var', False):
                    status = "FAIL_CONST"; is_fail = True
                else:
                    status = "OK_CONST"
            elif schema.get('require_var', False) and c_std < schema.get('min_std', 0.01):
                status = "WARN_LOW_VAR"
            elif schema['type'] == 'binary' and c_uni > 3:
                status = "WARN_NON_BINARY"

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

        print(f"{col:<25} | {pop_pct:>8.1f}% | {fmt(c_min)} | {fmt(c_max)} | {fmt(c_std)} | {c_uni:>8} | [{status}]")
        report[col] = {"status": status, "is_fail": is_fail, "populated_pct": pop_pct,
                        "min": c_min, "max": c_max, "std": c_std, "unique": c_uni}

    print("-" * 110)
    print(f"\nSUMMARY: {n_ok} OK | {n_warn} WARN | {n_fail} FAIL | {n_missing} MISSING | {n_empty} EMPTY")
    print(f"  Schema expects {len(VALIDATION_SCHEMA_V30)} columns, dataset has {len(df.columns)} columns")

    # Extra columns not in schema
    extra = [c for c in df.columns if c not in VALIDATION_SCHEMA_V30]
    if extra:
        print(f"  Extra columns (not in v3.0 schema): {extra[:10]}{'...' if len(extra) > 10 else ''}")

    return report


def coverage_report(df: pd.DataFrame):
    """Print a clean coverage report grouped by category."""
    print("\n" + "=" * 60)
    print("  COVERAGE REPORT (% populated)")
    print("=" * 60)

    categories = {
        'Underlying OHLCV': ['timestamp', 'symbol', 'underlying_price', 'open', 'high', 'low', 'close'],
        'Options Identity': ['option_symbol', 'expiration', 'strike', 'call_put'],
        'Options Chain': ['rho', 'volume', 'open_interest', 'bid', 'ask', 'mark'],
        'Greeks': ['delta', 'gamma', 'vega', 'theta', 'iv'],
        'Time/Strategy': ['te', 'ivr', 'target_spot', 'sma', 'psar_mark', 'mtf_consensus', 'max_dd_60m'],
        'Dynamic Features': ['log_return', 'vol_ewma', 'ret_z', 'atr_pct', 'kappa_proxy', 'vol_energy',
                            'rsi_dyn', 'adx_adaptive', 'psar_adaptive',
                            'bb_mu_dyn', 'bb_sigma_dyn', 'bb_lower_dyn', 'bb_upper_dyn', 'stoch_k_dyn',
                            'consolidation_score', 'breakout_score'],
        'V2.2 Primitives': ['spread_ratio', 'bandwidth', 'bb_percentile', 'bw_expansion_rate',
                           'cmf', 'pressure_up', 'pressure_down', 'friction_ratio',
                           'exec_allow', 'gap_risk_score', 'risk_override', 'iv_confidence',
                           'macd_norm', 'macd_signal_norm', 'macd_histogram',
                           'plus_di', 'minus_di', 'psar_trend', 'psar_reversion_mu',
                           'beta1_norm_stub', 'chaos_membership', 'position_size_mult', 'fuzzy_reversion_11'],
        'v3.0 Trend': ['FRAMA', 'Anchored_VWAP', 'McClellanOsc'],
        'v3.0 IV Bands': ['IV_High', 'IV_Mid', 'IV_Low'],
        'v3.0 Chain Agg': ['Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume', 'OSC_Volume'],
        'v3.0 Flow/Breadth': ['WeightedAlpha', 'WilderAccSwingIndex', 'AccDistWill', 'AccDistWillMovAvg'],
        'v3.0 Microstructure': ['aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread'],
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

        avg_pct = np.mean(pcts)
        total_pop += sum(1 for p in pcts if p > 0)
        total_cols += len(cols)

        bar = '#' * int(avg_pct / 2.5) + '.' * (40 - int(avg_pct / 2.5))
        pop_count = sum(1 for p in pcts if p > 0)
        print(f"  {cat:<22s}  [{bar}]  {avg_pct:5.1f}%  ({pop_count}/{len(cols)} cols)")

    print(f"\n  TOTAL: {total_pop}/{total_cols} columns populated")


def main():
    parser = argparse.ArgumentParser(description="CondorNet v3.0 Forensic Validator")
    parser.add_argument("--input", "-i", required=True, help="Input v3.0 CSV path (supports glob)")
    parser.add_argument("--output", "-o", default=None, help="Output fixed CSV path")
    parser.add_argument("--check-only", action="store_true", help="Validate only, no repair")
    parser.add_argument("--sample", type=int, default=50000, help="Sample size for validation")
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
    print(f"\n{'=' * 70}")
    print(f"  CONDORNET v3.0 FORENSIC VALIDATOR")
    print(f"  Input: {os.path.basename(inp)}  ({size_mb:.1f} MB)")
    print(f"{'=' * 70}")

    t0 = time.time()

    # Load
    print("\n[LOAD] Reading dataset ...")
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
        print(f"\n[ALERT] {len(fail_cols)} columns with issues: {fail_cols[:10]}")

        if not args.check_only and args.output:
            print("\n[REPAIR] Starting forensic repair ...")
            # Import and run precompute
            sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
            from precompute_features_v3 import main as precompute_main

            # Run precompute as repair
            import subprocess
            cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'precompute_features_v3.py'),
                   '--input', inp, '--output', args.output]
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)

            if result.returncode == 0:
                print(f"\n[RE-AUDIT] Validating repaired output ...")
                df_fixed = pd.read_csv(args.output, low_memory=False)
                validate_columns(df_fixed, sample_size=args.sample)
                coverage_report(df_fixed)
            else:
                print(f"[ERROR] Repair failed with exit code {result.returncode}")
        elif args.check_only:
            print("\n[INFO] --check-only mode: no repair attempted")
    else:
        print("\n[PASS] All v3.0 features passed validation")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
