#!/usr/bin/env python3
"""
Fix 2020 m5 schema artifacts so it matches the 2025 m5 reference.

Removes:
    PivotHigh_x, PivotLow_x, PivotHigh_y, PivotLow_y

Requires canonical columns:
    PivotHigh, PivotLow

Behavior:
- preserves row count
- preserves timestamp order
- optionally writes in-place or to a new file
- validates schema parity against 2025 reference after cleanup

Usage:
    python fix_2020_m5_schema_artifacts.py
    python fix_2020_m5_schema_artifacts.py --input /path/to/m5_dataset_v43_2020.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ARTIFACT_COLS = ["PivotHigh_x", "PivotLow_x", "PivotHigh_y", "PivotLow_y"]
CANONICAL_COLS = ["PivotHigh", "PivotLow"]

# Resolve project root regardless of CWD
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent.parent  # scripts/v43/Datasetv4/data/root


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fix_2020_m5_schema_artifacts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def load_header(path: Path) -> list[str]:
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def clean_2020_m5_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CANONICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required canonical column: {col}")

    drop_cols = [c for c in ARTIFACT_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns remain after cleanup: {dupes}")

    return df


def assert_schema_match(df: pd.DataFrame, ref_cols: list[str]) -> None:
    got = list(df.columns)
    if got != ref_cols:
        missing = [c for c in ref_cols if c not in got]
        extra = [c for c in got if c not in ref_cols]
        raise AssertionError(
            "Schema mismatch after cleanup.\n"
            f"Missing vs ref: {missing}\n"
            f"Extra vs ref:   {extra}\n"
            f"Ref count={len(ref_cols)}  Got count={len(got)}"
        )


def summarize_changes(before_cols: list[str], after_cols: list[str]) -> dict:
    return {
        "before_col_count": len(before_cols),
        "after_col_count": len(after_cols),
        "removed_cols": [c for c in before_cols if c not in after_cols],
        "added_cols":   [c for c in after_cols  if c not in before_cols],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Remove PivotHigh/Low _x/_y merge artifacts from 2020 m5 dataset."
    )
    ap.add_argument(
        "--input",
        default=str(
            _PROJECT_ROOT / "data" / "Datasetv4" / "v43" / "normalized_shells"
            / "2020" / "m5_dataset_v43_2020.csv"
        ),
        help="Path to 2020 m5 dataset (default: normalized_shells/2020/m5_dataset_v43_2020.csv)",
    )
    ap.add_argument(
        "--reference",
        default=str(
            _PROJECT_ROOT / "data" / "Datasetv4" / "v43" / "2025"
            / "m5_dataset_v43_2025.csv"
        ),
        help="Path to 2025 m5 reference dataset.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Optional output path. If omitted, overwrites input in-place.",
    )
    ap.add_argument(
        "--log",
        default=str(_PROJECT_ROOT / "logs" / "fix_2020_m5_schema_artifacts.log"),
        help="Log file path.",
    )
    ap.add_argument(
        "--summary-json",
        default=str(
            _PROJECT_ROOT / "reports" / "schema_fix"
            / "fix_2020_m5_schema_artifacts_summary.json"
        ),
        help="Summary JSON output path.",
    )
    args = ap.parse_args()

    input_path   = Path(args.input)
    ref_path     = Path(args.reference)
    output_path  = Path(args.output) if args.output else input_path
    log_path     = Path(args.log)
    summary_json = Path(args.summary_json)

    logger = setup_logger(log_path)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting 2020 m5 schema artifact cleanup")
    logger.info("Input    : %s", input_path)
    logger.info("Reference: %s", ref_path)
    logger.info("Output   : %s", output_path)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1
    if not ref_path.exists():
        logger.error("Reference file not found: %s", ref_path)
        return 1

    # ── Load ─────────────────────────────────────────────────────────────────
    logger.info("Loading input file …")
    df = pd.read_csv(input_path, low_memory=False)
    ref_cols   = load_header(ref_path)
    before_cols = list(df.columns)
    before_rows = len(df)

    logger.info("Input columns: %d  rows: %d", len(before_cols), before_rows)
    logger.info("Reference columns: %d", len(ref_cols))

    artifacts_found = [c for c in ARTIFACT_COLS if c in df.columns]
    if not artifacts_found:
        logger.info("No artifact columns found. File may already be clean.")

    ts_before = df["timestamp"].copy() if "timestamp" in df.columns else None

    # ── Clean ─────────────────────────────────────────────────────────────────
    cleaned = clean_2020_m5_artifacts(df)

    # ── Validate ──────────────────────────────────────────────────────────────
    if len(cleaned) != before_rows:
        raise AssertionError(
            f"Row count changed: before={before_rows}, after={len(cleaned)}"
        )
    if ts_before is not None and not cleaned["timestamp"].equals(ts_before):
        raise AssertionError("Timestamp order changed during cleanup")

    assert_schema_match(cleaned, ref_cols)

    # ── Write ─────────────────────────────────────────────────────────────────
    logger.info("Writing cleaned file to: %s", output_path)
    cleaned.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    after_cols = list(cleaned.columns)
    summary    = summarize_changes(before_cols, after_cols)
    summary.update(
        {
            "input":                  str(input_path),
            "reference":              str(ref_path),
            "output":                 str(output_path),
            "row_count":              before_rows,
            "schema_match_reference": True,
        }
    )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Cleanup complete")
    logger.info("  Rows preserved  : %d", before_rows)
    logger.info("  Removed columns : %s", summary["removed_cols"])
    logger.info("  Schema matches 2025 reference exactly ✓")
    logger.info("  Summary JSON    : %s", summary_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
