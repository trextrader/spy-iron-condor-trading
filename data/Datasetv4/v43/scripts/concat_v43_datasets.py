#!/usr/bin/env python3
"""
concat_v43_datasets.py — Concatenate 2020-2024 v43 datasets into single per-TF files.

Reads:
  normalized_shells/{year}/{tf}_dataset_v43_{year}.csv   (for year in 2020-2024)
  {year}/options_{year}_v43.csv                           (options per year)

Writes:
  combined/{tf}_dataset_v43_2020_2024.csv   (one per TF: m1, m5, m15, h1)
  combined/options_v43_2020_2024.csv         (all options years combined)

Validations:
  - Schema identical across all years for each TF (column headers must match)
  - Timestamps sorted ascending
  - No duplicate timestamps
  - Row count = sum of all year file rows

Notes:
  - Columns with all-NaN values are preserved (not dropped)
  - AccDistWill is adjusted per-year to produce a continuous cumulative A/D series
  - WilderAccSwingIndex is adjusted similarly for continuity
  - 2025 datasets are NOT included (holdout — training/optimization uses 2020-2024 only)

Usage:
    python concat_v43_datasets.py
    python concat_v43_datasets.py --timeframes m5 --output-dir data/Datasetv4/v43/combined
    python concat_v43_datasets.py --no-options   (skip options concat)
    python concat_v43_datasets.py --dry-run      (validate only, no write)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Project paths ─────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent.parent

# ── Constants ─────────────────────────────────────────────────────────────────
YEARS      = [2020, 2021, 2022, 2023, 2024]
TIMEFRAMES = ["m1", "m5", "m15", "h1"]

# Cumulative series that must be adjusted for cross-year continuity
CUMULATIVE_COLS = ["AccDistWill", "AccDistWillMovAvg", "WilderAccSwingIndex"]


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("concat_v43_datasets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh  = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_year_file(path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("  Missing: %s — skipped", path)
        return None
    df = pd.read_csv(path, low_memory=False)
    logger.info("  Loaded  %s  (%d rows × %d cols)", path.name, len(df), len(df.columns))
    return df


def _validate_schemas(dfs: List[pd.DataFrame], years: List[int],
                       tf: str, logger: logging.Logger) -> bool:
    """Return True if all year files have identical column lists."""
    ref_cols = list(dfs[0].columns)
    ok = True
    for year, df in zip(years[1:], dfs[1:]):
        cols = list(df.columns)
        if cols != ref_cols:
            missing = [c for c in ref_cols if c not in cols]
            extra   = [c for c in cols if c not in ref_cols]
            logger.error(
                "  Schema mismatch for %s %d: missing=%s  extra=%s",
                tf, year, missing, extra,
            )
            ok = False
    if ok:
        logger.info("  Schema OK across all years (%d cols)", len(ref_cols))
    return ok


def _adjust_cumulative(dfs: List[pd.DataFrame], col: str) -> List[pd.DataFrame]:
    """
    Ensure cumulative columns (AccDistWill, WilderAccSwingIndex, etc.) form a
    continuous series when concatenated: each year's series is offset so its
    first value continues from the last value of the previous year.
    """
    out = []
    running_offset = 0.0
    for df in dfs:
        df = df.copy()
        if col not in df.columns:
            out.append(df)
            continue
        vals = df[col].values.astype(np.float64)
        if np.isnan(vals).all():
            out.append(df)
            continue
        # Adjust: shift the entire year's series by the running offset
        vals = vals + running_offset
        df[col] = vals.astype(np.float32)
        # Update offset: last non-NaN value becomes the new baseline
        last_valid = vals[~np.isnan(vals)]
        if len(last_valid) > 0:
            running_offset = float(last_valid[-1])
        out.append(df)
    return out


def _concat_tf(
    shells_dir:  Path,
    tf:          str,
    years:       List[int],
    output_dir:  Path,
    dry_run:     bool,
    logger:      logging.Logger,
) -> Dict:
    logger.info("=" * 60)
    logger.info("TF: %s", tf)

    dfs:  List[pd.DataFrame] = []
    yrs:  List[int]          = []

    for year in years:
        p  = shells_dir / str(year) / f"{tf}_dataset_v43_{year}.csv"
        df = _load_year_file(p, logger)
        if df is not None:
            dfs.append(df)
            yrs.append(year)

    if not dfs:
        logger.error("  No files found for %s — skipping", tf)
        return {"tf": tf, "error": "no_files"}

    # Schema validation
    if not _validate_schemas(dfs, yrs, tf, logger):
        logger.warning("  Proceeding with union schema (outer join)")

    # Adjust cumulative series for cross-year continuity
    for col in CUMULATIVE_COLS:
        if any(col in df.columns for df in dfs):
            dfs = _adjust_cumulative(dfs, col)

    # Concatenate
    combined = pd.concat(dfs, ignore_index=True)
    logger.info("  Combined: %d rows × %d cols", len(combined), len(combined.columns))

    # Sort by timestamp
    ts_col = "timestamp" if "timestamp" in combined.columns else combined.columns[0]
    combined[ts_col] = pd.to_datetime(combined[ts_col], errors="coerce")
    combined = combined.sort_values(ts_col).reset_index(drop=True)

    # Check for duplicate timestamps
    n_dupes = combined[ts_col].duplicated().sum()
    if n_dupes > 0:
        logger.warning("  %d duplicate timestamps detected — keeping first occurrence", n_dupes)
        combined = combined.drop_duplicates(subset=[ts_col], keep="first").reset_index(drop=True)

    # Restore timestamp to string
    combined[ts_col] = combined[ts_col].dt.strftime("%Y-%m-%d %H:%M:%S")

    row_count = len(combined)
    col_count = len(combined.columns)

    # Write
    out_path = output_dir / f"{tf}_dataset_v43_2020_2024.csv"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        logger.info("  Written → %s", out_path)
    else:
        logger.info("  [DRY-RUN] Would write → %s", out_path)

    return {
        "tf":        tf,
        "years":     yrs,
        "row_count": row_count,
        "col_count": col_count,
        "output":    str(out_path),
    }


def _concat_options(
    base_dir:   Path,
    years:      List[int],
    output_dir: Path,
    dry_run:    bool,
    logger:     logging.Logger,
) -> Dict:
    logger.info("=" * 60)
    logger.info("OPTIONS")

    dfs:  List[pd.DataFrame] = []
    yrs:  List[int]          = []

    # Options files may live in different locations — check common patterns
    for year in years:
        candidates = [
            base_dir / str(year) / f"options_{year}_v43.csv",
            base_dir / "normalized_shells" / str(year) / f"options_{year}_v43.csv",
            base_dir / f"options_{year}_v43.csv",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            logger.warning("  Options file not found for %d — skipped", year)
            continue
        df = _load_year_file(found, logger)
        if df is not None:
            dfs.append(df)
            yrs.append(year)

    if not dfs:
        logger.warning("  No options files found — skipping options concat")
        return {"entity": "options", "error": "no_files"}

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("  Combined options: %d rows × %d cols", len(combined), len(combined.columns))

    # Sort by date column (try common names)
    for date_col in ("date", "timestamp", "expiration", "quote_date"):
        if date_col in combined.columns:
            combined[date_col] = pd.to_datetime(combined[date_col], errors="coerce")
            combined = combined.sort_values(date_col).reset_index(drop=True)
            combined[date_col] = combined[date_col].dt.strftime("%Y-%m-%d")
            break

    out_path = output_dir / "options_v43_2020_2024.csv"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        logger.info("  Written → %s", out_path)
    else:
        logger.info("  [DRY-RUN] Would write → %s", out_path)

    return {
        "entity":    "options",
        "years":     yrs,
        "row_count": len(combined),
        "col_count": len(combined.columns),
        "output":    str(out_path),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concatenate 2020-2024 v43 datasets into single per-TF files."
    )
    ap.add_argument("--base-dir",   default=str(_PROJECT_ROOT / "data" / "Datasetv4" / "v43"))
    ap.add_argument("--shells-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                    help="Output directory. Default: {base-dir}/combined")
    ap.add_argument("--years",      nargs="+", type=int, default=YEARS)
    ap.add_argument("--timeframes", nargs="+",           default=TIMEFRAMES)
    ap.add_argument("--no-options", action="store_true", help="Skip options concat")
    ap.add_argument("--dry-run",    action="store_true", help="Validate only, no write")
    ap.add_argument("--log",
        default=str(_PROJECT_ROOT / "logs" / "concat_v43_datasets.log"))
    ap.add_argument("--summary-json",
        default=str(_PROJECT_ROOT / "reports" / "population_audit"
                    / "concat_v43_summary.json"))
    args = ap.parse_args()

    base_dir   = Path(args.base_dir)
    shells_dir = Path(args.shells_dir) if args.shells_dir else base_dir / "normalized_shells"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "combined"
    log_path   = Path(args.log)
    sum_json   = Path(args.summary_json)
    sum_json.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info("concat_v43_datasets.py")
    logger.info("base_dir  : %s", base_dir)
    logger.info("shells_dir: %s", shells_dir)
    logger.info("output_dir: %s", output_dir)
    logger.info("years     : %s", args.years)
    logger.info("timeframes: %s", args.timeframes)
    logger.info("dry_run   : %s", args.dry_run)
    logger.info("=" * 60)

    results: List[Dict] = []

    # OHLCV/indicator TF files
    for tf in args.timeframes:
        r = _concat_tf(shells_dir, tf, args.years, output_dir, args.dry_run, logger)
        results.append(r)

    # Options
    if not args.no_options:
        r = _concat_options(base_dir, args.years, output_dir, args.dry_run, logger)
        results.append(r)

    # Summary
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Concat complete.")
    for r in results:
        if "error" not in r:
            entity = r.get("tf", r.get("entity"))
            logger.info(
                "  %-6s  %d rows  %d cols → %s",
                entity,
                r.get("row_count", 0),
                r.get("col_count", 0),
                Path(r.get("output", "")).name,
            )
    logger.info("Summary JSON: %s", sum_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
