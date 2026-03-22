#!/usr/bin/env python3
"""
Audit 2020-2024 dataset population progress against 2025 reference files.

Reports per year/timeframe:
  - total columns
  - exact/set schema parity vs 2025 reference
  - populated count, empty count
  - all-null columns, all-zero columns
  - NaN % per target column
  - first valid row per target column

Outputs:
  reports/population_audit/population_progress_summary.csv
  reports/population_audit/population_progress_summary.json
  reports/population_audit/population_progress_detailed.json
  logs/audit_population_progress_v43.log

Usage:
    python audit_population_progress_v43.py
    python audit_population_progress_v43.py --years 2020 2021
    python audit_population_progress_v43.py --timeframes m1 m5
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

# Columns expected to be empty in some timeframes (by design, matching 2025)
INTENTIONALLY_EMPTY_BY_TF: Dict[str, List[str]] = {
    "m15": ["TrendSeekerUp", "TrendSeekerDown", "TurtleChn-Up", "TurtleChn-Low"],
    "h1":  [],  # 2025 h1 has Pivot cols empty but 2020-2024 have them populated — leave as-is
}

# Target columns tracked across all population scripts
ALL_TARGET_COLS = [
    # Script 2 — OHLCV block
    "psar_trend", "psar_mark", "psar_adaptive", "psar_reversion_mu",
    "pressure_up", "pressure_down",
    "gap_risk_score", "max_dd_60m",
    "adx_adaptive", "bb_percentile", "rsi_dyn", "stoch_k_dyn",
    # Script 4 — Fuzzy block
    "chaos_membership", "fuzzy_reversion_11",
    "consolidation_score", "breakout_score",
    "position_size_mult", "spread_ratio", "kappa_proxy", "friction_ratio",
    "McClellanOsc",
    # Script 5 — Platform export
    "TrendSeekerUp", "TrendSeekerDown", "TurtleChn-Up", "TurtleChn-Low",
    "FRAMA", "WeightedAlpha", "OSC-Volume",
    "AccDistWill", "AccDistWillMovAvg", "WilderAccSwingIndex",
    # M5-only — Script 6
    "pop", "max_loss", "ev", "var_95", "cvar_95",
    "strategy_label", "exit_signal",
    # M5-only — Script 7
    "ps_pnl_pct", "ps_credit_norm", "ps_bars_held", "ps_dte_frac",
    "ps_delta_exp", "ps_gamma_exp", "ps_theta_pos", "ps_iv_change",
    "ps_high_water", "ps_mae", "ps_unrealized_norm",
    # Conditional / other
    "trade_count", "target_spot",
]


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("audit_population_progress_v43")
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

def target_file(shells_dir: Path, year: int, tf: str) -> Path:
    return shells_dir / str(year) / f"{tf}_dataset_v43_{year}.csv"


def reference_file(base_dir: Path, tf: str) -> Path:
    return base_dir / "2025" / f"{tf}_dataset_v43_2025.csv"


# ── Column classification ─────────────────────────────────────────────────────

def classify_column_state(s: pd.Series) -> Dict:
    """Return a dict describing the population state of a single column."""
    info: Dict = {
        "is_all_nan":       bool(s.isna().all()),
        "nan_pct":          round(float(s.isna().mean()), 6),
        "first_valid_row":  None,
    }

    fvi = s.first_valid_index()
    if fvi is not None:
        try:
            info["first_valid_row"] = int(s.index.get_loc(fvi))
        except Exception:
            info["first_valid_row"] = None

    if pd.api.types.is_numeric_dtype(s):
        filled = s.fillna(0.0)
        info["is_all_zero"] = bool(np.isclose(filled.abs().sum(), 0.0))
        info["zero_pct"]    = round(float((filled == 0.0).mean()), 6)
        info["min"]         = float(s.min()) if not s.isna().all() else None
        info["max"]         = float(s.max()) if not s.isna().all() else None
    else:
        info["is_all_zero"] = False
        info["zero_pct"]    = None
        info["min"]         = None
        info["max"]         = None

    return info


def _is_empty(state: Dict) -> bool:
    return state["is_all_nan"] or state.get("is_all_zero", False)


# ── File audit ────────────────────────────────────────────────────────────────

def audit_file(path: Path, ref_cols: List[str], timeframe: str) -> Dict:
    df   = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)

    missing_vs_ref = [c for c in ref_cols if c not in cols]
    extra_vs_ref   = [c for c in cols     if c not in ref_cols]

    per_col: Dict[str, Dict] = {}
    populated_count = 0
    empty_count     = 0
    all_null_cols:  List[str] = []
    all_zero_cols:  List[str] = []

    intentional_empty = INTENTIONALLY_EMPTY_BY_TF.get(timeframe, [])

    for col in cols:
        state = classify_column_state(df[col])
        per_col[col] = state

        if _is_empty(state):
            empty_count += 1
            if state["is_all_nan"]:
                all_null_cols.append(col)
            elif state.get("is_all_zero"):
                all_zero_cols.append(col)
        else:
            populated_count += 1

    # Target-column detail
    target_detail: Dict[str, Dict] = {}
    for col in ALL_TARGET_COLS:
        if col not in df.columns:
            target_detail[col] = {"status": "MISSING_FROM_HEADER"}
            continue
        state = classify_column_state(df[col])
        if col in intentional_empty:
            status = "INTENTIONALLY_EMPTY"
        elif _is_empty(state):
            status = "EMPTY"
        else:
            status = "POPULATED"
        target_detail[col] = {
            "status":          status,
            "nan_pct":         state["nan_pct"],
            "first_valid_row": state["first_valid_row"],
        }

    return {
        "file":              str(path),
        "row_count":         len(df),
        "col_count":         len(cols),
        "schema_ok_exact":   cols == ref_cols,
        "schema_ok_set":     sorted(cols) == sorted(ref_cols),
        "missing_vs_ref":    missing_vs_ref,
        "extra_vs_ref":      extra_vs_ref,
        "populated_count":   populated_count,
        "empty_count":       empty_count,
        "all_null_cols":     all_null_cols,
        "all_zero_cols":     all_zero_cols,
        "target_detail":     target_detail,
        "per_col":           per_col,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def build_population_summary(
    base_dir:   Path,
    shells_dir: Path,
    years:      List[int],
    timeframes: List[str],
    logger:     logging.Logger,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for tf in timeframes:
        ref_path = reference_file(base_dir, tf)
        if not ref_path.exists():
            logger.warning("Missing reference file: %s", ref_path)
            continue
        ref_cols = list(pd.read_csv(ref_path, nrows=0).columns)

        for year in years:
            path = target_file(shells_dir, year, tf)
            if not path.exists():
                logger.warning("Missing dataset file: %s", path)
                continue

            audit = audit_file(path, ref_cols, tf)

            # Count target columns by status
            td = audit["target_detail"]
            n_populated = sum(1 for v in td.values() if v.get("status") == "POPULATED")
            n_empty     = sum(1 for v in td.values() if v.get("status") == "EMPTY")
            n_missing   = sum(1 for v in td.values() if v.get("status") == "MISSING_FROM_HEADER")

            rows.append({
                "year":                  year,
                "timeframe":             tf,
                "row_count":             audit["row_count"],
                "col_count":             audit["col_count"],
                "schema_ok_exact":       audit["schema_ok_exact"],
                "schema_ok_set":         audit["schema_ok_set"],
                "missing_vs_ref_count":  len(audit["missing_vs_ref"]),
                "extra_vs_ref_count":    len(audit["extra_vs_ref"]),
                "populated_count":       audit["populated_count"],
                "empty_count":           audit["empty_count"],
                "target_populated":      n_populated,
                "target_empty":          n_empty,
                "target_missing_header": n_missing,
                "missing_vs_ref":        "|".join(audit["missing_vs_ref"]),
                "extra_vs_ref":          "|".join(audit["extra_vs_ref"]),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["timeframe", "year"])
        .reset_index(drop=True)
    )


# ── Detailed JSON ─────────────────────────────────────────────────────────────

def write_detailed_json(
    base_dir:    Path,
    shells_dir:  Path,
    years:       List[int],
    timeframes:  List[str],
    output_json: Path,
    logger:      logging.Logger,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    detailed: Dict = {}

    for tf in timeframes:
        ref_path = reference_file(base_dir, tf)
        if not ref_path.exists():
            continue
        ref_cols = list(pd.read_csv(ref_path, nrows=0).columns)
        detailed[tf] = {}

        for year in years:
            path = target_file(shells_dir, year, tf)
            if not path.exists():
                continue
            result = audit_file(path, ref_cols, tf)
            # Drop per_col to keep detailed JSON manageable
            result.pop("per_col", None)
            detailed[tf][str(year)] = result

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    logger.info("Detailed JSON: %s", output_json)


# ── Report writers ────────────────────────────────────────────────────────────

def write_reports(
    df:          pd.DataFrame,
    summary_csv: Path,
    summary_json: Path,
    logger:      logging.Logger,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    logger.info("Summary CSV: %s", summary_csv)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    logger.info("Summary JSON: %s", summary_json)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit 2020-2024 dataset population progress vs 2025 reference."
    )
    ap.add_argument(
        "--base-dir",
        default=str(_PROJECT_ROOT / "data" / "Datasetv4" / "v43"),
        help="Root v43 directory (contains 2025/ reference subfolder).",
    )
    ap.add_argument(
        "--shells-dir",
        default=None,
        help="Directory with normalized_shells/{year}/ subdirs. Default: {base-dir}/normalized_shells",
    )
    ap.add_argument("--years",      nargs="+", type=int, default=YEARS)
    ap.add_argument("--timeframes", nargs="+",           default=TIMEFRAMES)
    ap.add_argument(
        "--log",
        default=str(_PROJECT_ROOT / "logs" / "audit_population_progress_v43.log"),
    )
    ap.add_argument(
        "--summary-csv",
        default=str(_PROJECT_ROOT / "reports" / "population_audit" / "population_progress_summary.csv"),
    )
    ap.add_argument(
        "--summary-json",
        default=str(_PROJECT_ROOT / "reports" / "population_audit" / "population_progress_summary.json"),
    )
    ap.add_argument(
        "--detailed-json",
        default=str(_PROJECT_ROOT / "reports" / "population_audit" / "population_progress_detailed.json"),
    )
    args = ap.parse_args()

    base_dir   = Path(args.base_dir)
    shells_dir = Path(args.shells_dir) if args.shells_dir else base_dir / "normalized_shells"
    log_path   = Path(args.log)

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info("audit_population_progress_v43.py")
    logger.info("base_dir  : %s", base_dir)
    logger.info("shells_dir: %s", shells_dir)
    logger.info("years     : %s", args.years)
    logger.info("timeframes: %s", args.timeframes)
    logger.info("=" * 60)

    df_summary = build_population_summary(
        base_dir, shells_dir, args.years, args.timeframes, logger
    )

    if df_summary.empty:
        logger.error("No files were audited. Check base_dir and shells_dir paths.")
        return 1

    write_reports(df_summary, Path(args.summary_csv), Path(args.summary_json), logger)
    write_detailed_json(
        base_dir, shells_dir, args.years, args.timeframes, Path(args.detailed_json), logger
    )

    # ── Console summary ────────────────────────────────────────────────────
    logger.info("\n")
    logger.info("=" * 100)
    logger.info("POPULATION PROGRESS SUMMARY")
    logger.info("=" * 100)
    display_cols = [
        "year", "timeframe", "row_count", "col_count",
        "schema_ok_exact", "populated_count", "empty_count",
        "target_populated", "target_empty", "target_missing_header",
    ]
    logger.info("\n%s", df_summary[display_cols].to_string(index=False))
    logger.info("=" * 100)
    logger.info("Audit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
