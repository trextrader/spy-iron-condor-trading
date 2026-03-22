#!/usr/bin/env python3
"""
verify_schema_parity_v43.py — Verify 2020-2024 column schema matches 2025 reference.

Checks for every year × TF combination:
  - Exact column match (same columns in the same order)
  - Set match (same columns, different order)
  - Columns present in 2025 but missing from the year file
  - Columns present in the year file but absent from 2025
  - Position diffs (columns that exist in both but are at different indices)

Outputs:
  reports/population_audit/schema_parity_summary.csv
  reports/population_audit/schema_parity_summary.json
  logs/verify_schema_parity_v43.log

Usage:
    python verify_schema_parity_v43.py
    python verify_schema_parity_v43.py --years 2022 2023 --timeframes m1 m5
    python verify_schema_parity_v43.py --fix-order   (rewrite files to match 2025 column order)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ── Project paths ─────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent.parent

# ── Constants ─────────────────────────────────────────────────────────────────
YEARS      = [2020, 2021, 2022, 2023, 2024]
TIMEFRAMES = ["m1", "m5", "m15", "h1"]


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("verify_schema_parity_v43")
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


# ── Path helpers ──────────────────────────────────────────────────────────────

def ref_path(base_dir: Path, tf: str) -> Path:
    return base_dir / "2025" / f"{tf}_dataset_v43_2025.csv"

def year_path(shells_dir: Path, year: int, tf: str) -> Path:
    return shells_dir / str(year) / f"{tf}_dataset_v43_{year}.csv"

def load_cols(path: Path) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_schema(ref_cols: List[str], year_cols: List[str]) -> Dict:
    ref_set  = set(ref_cols)
    year_set = set(year_cols)

    missing_from_year = [c for c in ref_cols  if c not in year_set]
    extra_in_year     = [c for c in year_cols if c not in ref_set]

    exact_match = (ref_cols == year_cols)
    set_match   = (sorted(ref_cols) == sorted(year_cols))

    # Position differences: columns present in both but at different positions
    pos_diffs = []
    if not exact_match and set_match:
        for i, (rc, yc) in enumerate(zip(ref_cols, year_cols)):
            if rc != yc:
                pos_diffs.append({
                    "index":    i,
                    "ref_col":  rc,
                    "year_col": yc,
                })

    # If not even a set match, find where in the year file each ref col appears
    col_positions: Dict[str, int] = {c: i for i, c in enumerate(year_cols)}
    out_of_order = []
    if not exact_match and not pos_diffs:
        # Full non-set-match: list how ref cols map to year positions
        for ref_i, rc in enumerate(ref_cols):
            if rc in col_positions:
                year_i = col_positions[rc]
                if year_i != ref_i:
                    out_of_order.append({
                        "col":          rc,
                        "ref_index":    ref_i,
                        "year_index":   year_i,
                    })

    return {
        "exact_match":        exact_match,
        "set_match":          set_match,
        "ref_col_count":      len(ref_cols),
        "year_col_count":     len(year_cols),
        "missing_from_year":  missing_from_year,
        "extra_in_year":      extra_in_year,
        "pos_diffs":          pos_diffs,     # only populated for exact-set but wrong-order
        "out_of_order":       out_of_order,  # only populated for partial-set matches
    }


# ── Fix column order ──────────────────────────────────────────────────────────

def fix_column_order(path: Path, ref_cols: List[str], logger: logging.Logger) -> bool:
    """Rewrite CSV to match ref_cols column order. Returns True on success."""
    df = pd.read_csv(path, low_memory=False)
    year_set = set(df.columns)

    missing = [c for c in ref_cols if c not in year_set]
    if missing:
        logger.error("  Cannot fix order for %s — missing cols: %s", path.name, missing)
        return False

    extra = [c for c in df.columns if c not in set(ref_cols)]
    if extra:
        logger.warning("  %s has extra cols not in ref (will append at end): %s", path.name, extra)
        ordered = ref_cols + extra
    else:
        ordered = ref_cols

    df = df[ordered]
    df.to_csv(path, index=False)
    logger.info("  Fixed column order → %s", path.name)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify 2020-2024 column schema vs 2025 reference."
    )
    ap.add_argument("--base-dir",   default=str(_PROJECT_ROOT / "data" / "Datasetv4" / "v43"))
    ap.add_argument("--shells-dir", default=None)
    ap.add_argument("--years",      nargs="+", type=int, default=YEARS)
    ap.add_argument("--timeframes", nargs="+",           default=TIMEFRAMES)
    ap.add_argument(
        "--fix-order",
        action="store_true",
        help="Rewrite year files to match 2025 column order (in-place).",
    )
    ap.add_argument(
        "--log",
        default=str(_PROJECT_ROOT / "logs" / "verify_schema_parity_v43.log"),
    )
    ap.add_argument(
        "--summary-csv",
        default=str(_PROJECT_ROOT / "reports" / "population_audit"
                    / "schema_parity_summary.csv"),
    )
    ap.add_argument(
        "--summary-json",
        default=str(_PROJECT_ROOT / "reports" / "population_audit"
                    / "schema_parity_summary.json"),
    )
    args = ap.parse_args()

    base_dir   = Path(args.base_dir)
    shells_dir = Path(args.shells_dir) if args.shells_dir else base_dir / "normalized_shells"
    log_path   = Path(args.log)
    sum_csv    = Path(args.summary_csv)
    sum_json   = Path(args.summary_json)
    sum_csv.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_path)
    logger.info("=" * 70)
    logger.info("verify_schema_parity_v43.py")
    logger.info("base_dir  : %s", base_dir)
    logger.info("shells_dir: %s", shells_dir)
    logger.info("years     : %s", args.years)
    logger.info("timeframes: %s", args.timeframes)
    logger.info("fix_order : %s", args.fix_order)
    logger.info("=" * 70)

    rows:    List[Dict] = []
    details: Dict       = {}
    any_fail = False

    for tf in args.timeframes:
        rp = ref_path(base_dir, tf)
        if not rp.exists():
            logger.warning("Reference file missing: %s — skipping TF %s", rp, tf)
            continue
        ref_cols = load_cols(rp)
        details[tf] = {}

        logger.info("── TF: %s  (ref cols: %d) ──", tf, len(ref_cols))

        for year in args.years:
            yp = year_path(shells_dir, year, tf)
            if not yp.exists():
                logger.warning("  Year file missing: %s", yp)
                rows.append({
                    "year": year, "timeframe": tf,
                    "exact_match": False, "set_match": False,
                    "ref_cols": len(ref_cols), "year_cols": 0,
                    "missing_count": len(ref_cols), "extra_count": 0,
                    "status": "FILE_MISSING",
                })
                continue

            year_cols = load_cols(yp)
            cmp       = compare_schema(ref_cols, year_cols)

            if cmp["exact_match"]:
                status = "OK_EXACT"
                marker = "✓"
            elif cmp["set_match"]:
                status = "OK_SET_ONLY"
                marker = "⚠ order mismatch"
                any_fail = True
            else:
                status = "SCHEMA_MISMATCH"
                marker = "✗"
                any_fail = True

            logger.info(
                "  %d  %s  ref=%d  year=%d  missing=%d  extra=%d  [%s]",
                year, tf,
                cmp["ref_col_count"], cmp["year_col_count"],
                len(cmp["missing_from_year"]), len(cmp["extra_in_year"]),
                marker,
            )

            if cmp["missing_from_year"]:
                logger.warning("    Missing from year: %s", cmp["missing_from_year"])
            if cmp["extra_in_year"]:
                logger.warning("    Extra in year:     %s", cmp["extra_in_year"])
            if cmp["pos_diffs"]:
                logger.warning("    Order diffs (first 5): %s", cmp["pos_diffs"][:5])
            if cmp["out_of_order"]:
                logger.warning("    Out-of-order cols (first 5): %s", cmp["out_of_order"][:5])

            # Fix order if requested and only set-match fails
            if args.fix_order and not cmp["exact_match"] and cmp["set_match"]:
                fix_column_order(yp, ref_cols, logger)
                status = "FIXED_ORDER"

            rows.append({
                "year":           year,
                "timeframe":      tf,
                "exact_match":    cmp["exact_match"],
                "set_match":      cmp["set_match"],
                "ref_cols":       cmp["ref_col_count"],
                "year_cols":      cmp["year_col_count"],
                "missing_count":  len(cmp["missing_from_year"]),
                "extra_count":    len(cmp["extra_in_year"]),
                "missing_cols":   "|".join(cmp["missing_from_year"]),
                "extra_cols":     "|".join(cmp["extra_in_year"]),
                "status":         status,
            })

            details[tf][str(year)] = cmp

    # ── Write outputs ──────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    df_out.to_csv(sum_csv, index=False)
    logger.info("Summary CSV  : %s", sum_csv)

    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    logger.info("Summary JSON : %s", sum_json)

    # ── Console table ──────────────────────────────────────────────────────
    logger.info("\n")
    logger.info("=" * 100)
    logger.info("SCHEMA PARITY SUMMARY")
    logger.info("=" * 100)
    display_cols = [
        "year", "timeframe", "ref_cols", "year_cols",
        "exact_match", "set_match", "missing_count", "extra_count", "status",
    ]
    if not df_out.empty:
        logger.info("\n%s", df_out[display_cols].to_string(index=False))
    logger.info("=" * 100)

    if any_fail:
        logger.warning("RESULT: Schema mismatches found — inspect details above.")
        logger.info("Tip: run with --fix-order to rewrite files to 2025 column order.")
    else:
        logger.info("RESULT: All files match 2025 reference exactly ✓")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
