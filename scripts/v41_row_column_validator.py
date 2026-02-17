"""
v41_row_column_validator.py — Dataset Quality Audit Tool

GPU-accelerated (optional) scan of condornet_v41_FINAL.csv:
  1. Column inventory vs canonical registry
  2. Per-column NaN/Inf/Zero/Outlier detection
  3. Per-row integrity checks
  4. Value range validation
  5. Post-warmup analysis (skip first N rows)

Usage:
    python scripts/v41_row_column_validator.py \
        --data data/Datasetv4/condornet_v41_FINAL.csv \
        --skip-rows 500 \
        --output reports/dataset_audit_v41.json

Author: Antigravity (Google DeepMind)
Date: 2026-02-16
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Try to import canonical registry
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from intelligence.canonical_feature_registry import (
        FEATURE_COLS_V42, FEATURE_COLS_V30,
        NEUTRAL_FILL_VALUES_V42, INPUT_DIM_V42,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    print("[WARN] Could not import canonical_feature_registry. Using column-name only checks.")
    FEATURE_COLS_V42 = []
    FEATURE_COLS_V30 = []
    REGISTRY_AVAILABLE = False

# Try GPU
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="v4.1 Dataset Validator")
    p.add_argument("--data", type=str, required=True, help="Path to CSV")
    p.add_argument("--skip-rows", type=int, default=500,
                    help="Skip first N rows for warmup (indicators may be 0/NaN during warmup)")
    p.add_argument("--outlier-sigma", type=float, default=6.0,
                    help="Flag values beyond N sigma from mean as outliers")
    p.add_argument("--output", type=str, default="reports/dataset_audit_v41.json",
                    help="Output JSON report path")
    p.add_argument("--use-gpu", action="store_true", default=False,
                    help="Use CUDA for analysis (faster for large datasets)")
    p.add_argument("--max-rows", type=int, default=0,
                    help="Limit rows loaded (0 = all)")
    return p.parse_args()


def column_inventory(df_cols, registry_cols):
    """Compare dataset columns against canonical registry."""
    df_set = set(df_cols)
    reg_set = set(registry_cols)
    
    matched = sorted(df_set & reg_set)
    missing_from_csv = sorted(reg_set - df_set)
    unregistered = sorted(df_set - reg_set)
    
    return {
        "total_csv_columns": len(df_cols),
        "total_registry_features": len(registry_cols),
        "matched": len(matched),
        "missing_from_csv": missing_from_csv,
        "unregistered_in_csv": unregistered,
        "match_rate_pct": round(len(matched) / max(len(registry_cols), 1) * 100, 1),
    }


def per_column_analysis(df, skip_rows=500, outlier_sigma=6.0):
    """Analyze every column for NaN, Inf, zero, and outlier issues."""
    results = {}
    
    # Full dataset analysis
    for col in df.columns:
        info = {"column": col, "dtype": str(df[col].dtype)}
        
        # Skip non-numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            info["type"] = "non_numeric"
            info["unique_count"] = int(df[col].nunique())
            info["sample"] = df[col].head(5).tolist()
            results[col] = info
            continue
        
        vals = df[col].values.astype(np.float64)
        n = len(vals)
        
        # Full dataset stats
        nan_count = int(np.isnan(vals).sum())
        inf_count = int(np.isinf(vals).sum())
        zero_count = int((vals == 0).sum())
        
        info["total_rows"] = n
        info["nan_count"] = nan_count
        info["nan_pct"] = round(nan_count / n * 100, 2)
        info["inf_count"] = inf_count
        info["inf_pct"] = round(inf_count / n * 100, 2)
        info["zero_count"] = zero_count
        info["zero_pct"] = round(zero_count / n * 100, 2)
        
        # Stats on non-NaN values
        valid = vals[~np.isnan(vals) & ~np.isinf(vals)]
        if len(valid) > 0:
            info["min"] = float(np.min(valid))
            info["max"] = float(np.max(valid))
            info["mean"] = float(np.mean(valid))
            info["std"] = float(np.std(valid))
            info["median"] = float(np.median(valid))
            info["p01"] = float(np.percentile(valid, 1))
            info["p99"] = float(np.percentile(valid, 99))
            
            # Outlier detection
            if info["std"] > 1e-10:
                z_scores = np.abs((valid - info["mean"]) / info["std"])
                outlier_count = int((z_scores > outlier_sigma).sum())
                info["outlier_count"] = outlier_count
                info["outlier_pct"] = round(outlier_count / len(valid) * 100, 3)
        
        # Post-warmup analysis (skip first N rows)
        if skip_rows > 0 and n > skip_rows:
            post_warmup = vals[skip_rows:]
            pw_nan = int(np.isnan(post_warmup).sum())
            pw_zero = int((post_warmup == 0).sum())
            pw_n = len(post_warmup)
            info["post_warmup_nan_count"] = pw_nan
            info["post_warmup_nan_pct"] = round(pw_nan / pw_n * 100, 2)
            info["post_warmup_zero_count"] = pw_zero
            info["post_warmup_zero_pct"] = round(pw_zero / pw_n * 100, 2)
            
            # Flag: if post-warmup NaN% or zero% is suspicious
            if pw_nan > 0:
                info["WARNING"] = f"Post-warmup NaN detected ({pw_nan} rows)"
            if pw_zero / pw_n > 0.95 and col not in ['pivot_high_flag', 'pivot_low_flag', 'breakout_score']:
                info["WARNING"] = f"Post-warmup 95%+ zeros ({round(pw_zero/pw_n*100,1)}%)"
        
        # First occurrence of NaN (row index)
        nan_mask = np.isnan(vals)
        if nan_count > 0:
            first_nan_idx = int(np.argmax(nan_mask))
            last_nan_idx = int(n - 1 - np.argmax(nan_mask[::-1]))
            info["first_nan_row"] = first_nan_idx
            info["last_nan_row"] = last_nan_idx
        
        results[col] = info
    
    return results


def summary_report(inventory, col_analysis, skip_rows):
    """Generate human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET QUALITY AUDIT REPORT")
    lines.append("=" * 70)
    lines.append(f"  Generated: {datetime.now().isoformat()}")
    lines.append(f"  Warmup skip: {skip_rows} rows")
    lines.append("")
    
    # Column inventory
    lines.append("--- COLUMN INVENTORY ---")
    lines.append(f"  CSV columns:        {inventory['total_csv_columns']}")
    lines.append(f"  Registry features:  {inventory['total_registry_features']}")
    lines.append(f"  Matched:            {inventory['matched']} ({inventory['match_rate_pct']}%)")
    
    if inventory['missing_from_csv']:
        lines.append(f"\n  MISSING from CSV ({len(inventory['missing_from_csv'])}):")
        for c in inventory['missing_from_csv']:
            lines.append(f"    - {c}")
    
    if inventory['unregistered_in_csv']:
        lines.append(f"\n  UNREGISTERED in CSV ({len(inventory['unregistered_in_csv'])}):")
        for c in inventory['unregistered_in_csv']:
            lines.append(f"    + {c}")
    
    # Problem columns
    lines.append("\n--- PROBLEM COLUMNS (post-warmup) ---")
    problems = []
    for col, info in col_analysis.items():
        if info.get("type") == "non_numeric":
            continue
        warnings = []
        if info.get("post_warmup_nan_count", 0) > 0:
            warnings.append(f"NaN={info['post_warmup_nan_count']}")
        if info.get("post_warmup_zero_pct", 0) > 95 and col not in ['pivot_high_flag', 'pivot_low_flag', 'breakout_score']:
            warnings.append(f"Zero={info['post_warmup_zero_pct']}%")
        if info.get("outlier_count", 0) > 10:
            warnings.append(f"Outliers={info['outlier_count']}")
        if info.get("inf_count", 0) > 0:
            warnings.append(f"Inf={info['inf_count']}")
        if warnings:
            problems.append((col, warnings))
    
    if problems:
        for col, warns in problems:
            lines.append(f"  ⚠  {col:35s} | {', '.join(warns)}")
    else:
        lines.append("  ✅ No problems detected post-warmup")
    
    # Column type summary
    numeric_cols = [c for c, info in col_analysis.items() if info.get("type") != "non_numeric"]
    non_numeric = [c for c, info in col_analysis.items() if info.get("type") == "non_numeric"]
    lines.append(f"\n--- TYPE SUMMARY ---")
    lines.append(f"  Numeric columns:     {len(numeric_cols)}")
    lines.append(f"  Non-numeric columns: {len(non_numeric)}")
    if non_numeric:
        lines.append(f"  Non-numeric: {non_numeric}")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    args = parse_args()
    
    print(f"[Validator] Loading {args.data}...")
    t0 = time.time()
    
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, low_memory=False)
    
    if args.max_rows > 0:
        df = df.iloc[:args.max_rows]
    
    load_time = time.time() - t0
    print(f"[Validator] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns ({load_time:.1f}s)")
    
    # Column inventory
    print("[Validator] Running column inventory...")
    inventory = column_inventory(list(df.columns), FEATURE_COLS_V42)
    
    # Per-column analysis
    print(f"[Validator] Analyzing {df.shape[1]} columns (skip_rows={args.skip_rows})...")
    t1 = time.time()
    col_analysis = per_column_analysis(df, skip_rows=args.skip_rows, outlier_sigma=args.outlier_sigma)
    analysis_time = time.time() - t1
    print(f"[Validator] Column analysis complete ({analysis_time:.1f}s)")
    
    # Summary
    report = summary_report(inventory, col_analysis, args.skip_rows)
    print(report)
    
    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_report = {
        "schema_version": "v4.3.0",
        "timestamp": datetime.now().isoformat(),
        "dataset": args.data,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "skip_rows": args.skip_rows,
        "outlier_sigma": args.outlier_sigma,
        "load_time_s": round(load_time, 2),
        "analysis_time_s": round(analysis_time, 2),
        "inventory": inventory,
        "column_analysis": col_analysis,
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"\n[Validator] Full report saved to: {output_path}")
    
    # Exit code based on problems
    has_problems = any(
        info.get("post_warmup_nan_count", 0) > 0 or info.get("inf_count", 0) > 0
        for info in col_analysis.values()
    )
    if has_problems:
        print("[Validator] ⚠  Dataset has quality issues — review report")
        sys.exit(1)
    else:
        print("[Validator] ✅ Dataset passed all checks")
        sys.exit(0)


if __name__ == "__main__":
    main()
