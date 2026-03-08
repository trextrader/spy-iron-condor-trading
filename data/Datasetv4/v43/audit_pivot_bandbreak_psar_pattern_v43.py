#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_pivot_bandbreak_psar_pattern_v43.py

Purpose
-------
Audit how well the user's band-break + PSAR shift + Bollinger midline pass-through
sequence aligns with sparse pivot highs / pivot lows across all years and timeframes.

Pattern audited
---------------
Bearish reversal candidate:
    - an upper-band break occurred 1..4 bars before current bar
    - PSAR is bearish on current bar OR shifted bearish recently
    - current bar passes through / closes below BB midline

Bullish reversal candidate:
    - a lower-band break occurred 1..4 bars before current bar
    - PSAR is bullish on current bar OR shifted bullish recently
    - current bar passes through / closes above BB midline

Pivot audit logic
-----------------
For each PivotHigh:
    - find best nearby bearish candidate inside a search window
    - measure:
        break_idx - pivot_idx
        confirm_idx - pivot_idx
        absolute timing error
    - classify strict / loose / no match

For each PivotLow:
    - same using bullish candidates

Outputs
-------
1) Per-pivot audit CSV
2) Per-file summary CSV
3) Global summary CSV

Notes
-----
- Missing pivots remain sparse / NaN.
- This is an audit script, not a training feature generator.
- It is allowed to look both before and after the pivot for evaluation purposes.
- The script computes PSAR if PSAR columns are absent.

Example
-------
python audit_pivot_bandbreak_psar_pattern_v43.py \
    --input-root normalized_shells \
    --output-dir reports/pivot_pattern_audit \
    --years 2020 2021 2022 2023 2024 \
    --timeframes m1 m5 m15 h1

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12

REQUIRED_CORE = [
    "open",
    "high",
    "low",
    "close",
    "bb_upper_dyn",
    "bb_mu_dyn",
    "bb_lower_dyn",
    "PivotHigh",
    "PivotLow",
]

DEFAULT_TIMEFRAMES = ["m1", "m5", "m15", "h1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit pivot alignment of band-break / PSAR / BB-midline pattern.")
    p.add_argument("--input-root", required=True, type=str, help="Root folder, e.g. normalized_shells")
    p.add_argument("--output-dir", required=True, type=str, help="Output report folder")
    p.add_argument("--years", nargs="*", default=None, help="Years to include")
    p.add_argument("--timeframes", nargs="*", default=DEFAULT_TIMEFRAMES, help="Timeframe prefixes")
    p.add_argument("--glob", type=str, default="*.csv", help="CSV glob inside each year folder")
    p.add_argument("--search-window", type=int, default=200, help="Pivot-centered audit search window")
    p.add_argument("--break-lookback-min", type=int, default=1, help="Min bars since band break")
    p.add_argument("--break-lookback-max", type=int, default=4, help="Max bars since band break")
    p.add_argument("--strict-break-near", type=int, default=2, help="Strict allowed abs(break_idx - pivot_idx)")
    p.add_argument("--strict-confirm-after", type=int, default=8, help="Strict allowed confirm bars after pivot")
    p.add_argument("--loose-break-near", type=int, default=8, help="Loose allowed abs(break_idx - pivot_idx)")
    p.add_argument("--loose-confirm-after", type=int, default=20, help="Loose allowed confirm bars after pivot")
    p.add_argument("--psar-step", type=float, default=0.02, help="PSAR acceleration step")
    p.add_argument("--psar-max-step", type=float, default=0.2, help="PSAR maximum acceleration")
    p.add_argument("--float-format", type=str, default="%.10g")
    return p.parse_args()


def collect_files(input_root: Path, years: Optional[List[str]], timeframes: List[str], pattern: str) -> List[Path]:
    files: List[Path] = []
    if years:
        year_dirs = [input_root / str(y) for y in years]
    else:
        year_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

    for yd in year_dirs:
        if not yd.exists():
            continue
        for f in sorted(yd.glob(pattern)):
            stem = f.stem.lower()
            if any(stem.startswith(tf.lower()) for tf in timeframes):
                files.append(f)

    return files


def require_columns(df: pd.DataFrame, cols: List[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def get_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["timestamp", "datetime", "DateTime", "date", "Date"]:
        if c in df.columns:
            return c
    return None


def num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def compute_psar(high: np.ndarray, low: np.ndarray, step: float = 0.02, max_step: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        psar
        bull_state (1 bullish, 0 bearish)
    """
    n = len(high)
    psar = np.full(n, np.nan, dtype=float)
    bull = np.full(n, 0, dtype=np.int8)

    if n < 2:
        return psar, bull

    # Initialize trend from first two closes using highs/lows proxy
    init_bull = high[1] + low[1] >= high[0] + low[0]
    bull[0] = 1 if init_bull else 0
    bull[1] = bull[0]

    if init_bull:
        psar[1] = low[0]
        ep = max(high[0], high[1])
    else:
        psar[1] = high[0]
        ep = min(low[0], low[1])

    af = step

    for i in range(2, n):
        prev_psar = psar[i - 1]
        if bull[i - 1] == 1:
            cand = prev_psar + af * (ep - prev_psar)
            cand = min(cand, low[i - 1], low[i - 2])

            if low[i] < cand:
                bull[i] = 0
                psar[i] = ep
                ep = low[i]
                af = step
            else:
                bull[i] = 1
                psar[i] = cand
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            cand = prev_psar + af * (ep - prev_psar)
            cand = max(cand, high[i - 1], high[i - 2])

            if high[i] > cand:
                bull[i] = 1
                psar[i] = ep
                ep = high[i]
                af = step
            else:
                bull[i] = 0
                psar[i] = cand
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)

    if np.isnan(psar[0]):
        psar[0] = low[0] if bull[1] == 1 else high[0]

    return psar, bull


def build_psar_columns(df: pd.DataFrame, step: float, max_step: float) -> pd.DataFrame:
    out = df.copy()

    if "PSAR" in out.columns:
        psar = num(out["PSAR"])
        if "PSARBullState" in out.columns:
            bull_state = pd.to_numeric(out["PSARBullState"], errors="coerce").fillna(0).astype(int).to_numpy()
        else:
            close = num(out["close"])
            bull_state = (psar < close).astype(int)
    else:
        high = num(out["high"])
        low = num(out["low"])
        psar, bull_state = compute_psar(high, low, step=step, max_step=max_step)
        out["PSAR"] = psar

    out["PSARBullState"] = bull_state
    out["PSARBearState"] = 1 - out["PSARBullState"]
    out["PSARBullShiftFlag"] = ((out["PSARBullState"].shift(1).fillna(out["PSARBullState"]) == 0) & (out["PSARBullState"] == 1)).astype(int)
    out["PSARBearShiftFlag"] = ((out["PSARBearState"].shift(1).fillna(out["PSARBearState"]) == 0) & (out["PSARBearState"] == 1)).astype(int)

    return out


def recent_break_age(flag: np.ndarray, min_age: int, max_age: int) -> np.ndarray:
    n = len(flag)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        best = np.nan
        for k in range(min_age, max_age + 1):
            j = i - k
            if j >= 0 and flag[j] == 1:
                best = float(k)
                break
        out[i] = best
    return out


def bars_since_event(flag: np.ndarray) -> np.ndarray:
    n = len(flag)
    out = np.full(n, np.nan, dtype=float)
    last_idx = None
    for i in range(n):
        if flag[i] == 1:
            last_idx = i
            out[i] = 0.0
        elif last_idx is not None:
            out[i] = float(i - last_idx)
    return out


def detect_pattern_candidates(
    df: pd.DataFrame,
    break_lookback_min: int,
    break_lookback_max: int,
) -> pd.DataFrame:
    out = df.copy()

    open_ = num(out["open"])
    high = num(out["high"])
    low = num(out["low"])
    close = num(out["close"])
    bb_upper = num(out["bb_upper_dyn"])
    bb_mu = num(out["bb_mu_dyn"])
    bb_lower = num(out["bb_lower_dyn"])

    upper_break = ((high > bb_upper) & np.isfinite(high) & np.isfinite(bb_upper)).astype(int)
    lower_break = ((low < bb_lower) & np.isfinite(low) & np.isfinite(bb_lower)).astype(int)

    out["UpperBandBreakFlag"] = upper_break
    out["LowerBandBreakFlag"] = lower_break

    out["UpperBandBreakAge_4"] = recent_break_age(upper_break, break_lookback_min, break_lookback_max)
    out["LowerBandBreakAge_4"] = recent_break_age(lower_break, break_lookback_min, break_lookback_max)

    out["UpperBandBreakInLast4"] = np.isfinite(out["UpperBandBreakAge_4"]).astype(int)
    out["LowerBandBreakInLast4"] = np.isfinite(out["LowerBandBreakAge_4"]).astype(int)

    out["BBCrossDownMu"] = ((open_ > bb_mu) & (close < bb_mu)).astype(int)
    out["BBCrossUpMu"] = ((open_ < bb_mu) & (close > bb_mu)).astype(int)

    out["BBPassThroughMu"] = ((high >= bb_mu) & (low <= bb_mu)).astype(int)
    out["CloseBelowBBMuFlag"] = (close < bb_mu).astype(int)
    out["CloseAboveBBMuFlag"] = (close > bb_mu).astype(int)

    out["BearMidlineConfirm"] = (((low <= bb_mu) & (close < bb_mu)) | ((open_ > bb_mu) & (close < bb_mu))).astype(int)
    out["BullMidlineConfirm"] = (((high >= bb_mu) & (close > bb_mu)) | ((open_ < bb_mu) & (close > bb_mu))).astype(int)

    out["BarsSincePSARBearShift"] = bars_since_event(out["PSARBearShiftFlag"].to_numpy(dtype=int))
    out["BarsSincePSARBullShift"] = bars_since_event(out["PSARBullShiftFlag"].to_numpy(dtype=int))

    # Strict current-state version
    out["BearCandidateStrict"] = (
        (out["UpperBandBreakInLast4"] == 1)
        & (out["PSARBearState"] == 1)
        & (out["BearMidlineConfirm"] == 1)
    ).astype(int)

    out["BullCandidateStrict"] = (
        (out["LowerBandBreakInLast4"] == 1)
        & (out["PSARBullState"] == 1)
        & (out["BullMidlineConfirm"] == 1)
    ).astype(int)

    # Slightly looser version: allow recent PSAR shift or current state
    out["BearCandidateLoose"] = (
        (out["UpperBandBreakInLast4"] == 1)
        & (
            (out["PSARBearState"] == 1)
            | ((out["BarsSincePSARBearShift"] >= 0) & (out["BarsSincePSARBearShift"] <= 4))
        )
        & (out["BearMidlineConfirm"] == 1)
    ).astype(int)

    out["BullCandidateLoose"] = (
        (out["LowerBandBreakInLast4"] == 1)
        & (
            (out["PSARBullState"] == 1)
            | ((out["BarsSincePSARBullShift"] >= 0) & (out["BarsSincePSARBullShift"] <= 4))
        )
        & (out["BullMidlineConfirm"] == 1)
    ).astype(int)

    # Recover break index from age
    upper_age = pd.to_numeric(out["UpperBandBreakAge_4"], errors="coerce").to_numpy(dtype=float)
    lower_age = pd.to_numeric(out["LowerBandBreakAge_4"], errors="coerce").to_numpy(dtype=float)

    upper_break_idx = np.full(len(out), np.nan, dtype=float)
    lower_break_idx = np.full(len(out), np.nan, dtype=float)

    for i in range(len(out)):
        if np.isfinite(upper_age[i]):
            upper_break_idx[i] = i - int(upper_age[i])
        if np.isfinite(lower_age[i]):
            lower_break_idx[i] = i - int(lower_age[i])

    out["UpperBreakIdx"] = upper_break_idx
    out["LowerBreakIdx"] = lower_break_idx

    return out


def score_candidate_for_pivot(
    pivot_idx: int,
    break_idx: int,
    confirm_idx: int,
    direction: str,
) -> float:
    """
    Lower is better.
    For highs, we like break near pivot and confirm shortly after pivot.
    For lows, same mirrored in time since confirm usually occurs shortly after pivot too.
    """
    break_err = abs(break_idx - pivot_idx)

    # Confirmation is usually at or after pivot; allow before but penalize.
    if confirm_idx >= pivot_idx:
        confirm_err = confirm_idx - pivot_idx
    else:
        confirm_err = 2.0 * (pivot_idx - confirm_idx)

    return float(break_err + confirm_err)


def classify_match(
    pivot_idx: int,
    break_idx: int,
    confirm_idx: int,
    strict_break_near: int,
    strict_confirm_after: int,
    loose_break_near: int,
    loose_confirm_after: int,
) -> str:
    break_err = abs(break_idx - pivot_idx)
    confirm_after = confirm_idx - pivot_idx

    if break_err <= strict_break_near and 0 <= confirm_after <= strict_confirm_after:
        return "strict"
    if break_err <= loose_break_near and -2 <= confirm_after <= loose_confirm_after:
        return "loose"
    return "off_window"


def analyze_pivots_for_file(
    df: pd.DataFrame,
    src: Path,
    year: str,
    timeframe: str,
    time_col: Optional[str],
    search_window: int,
    strict_break_near: int,
    strict_confirm_after: int,
    loose_break_near: int,
    loose_confirm_after: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []

    pivot_high = num(df["PivotHigh"])
    pivot_low = num(df["PivotLow"])

    high_candidates = df.index[df["BearCandidateLoose"] == 1].to_numpy()
    low_candidates = df.index[df["BullCandidateLoose"] == 1].to_numpy()

    def ts_at(i: int):
        return df.iloc[i][time_col] if time_col is not None else np.nan

    def price_at(i: int, col: str):
        try:
            return pd.to_numeric(pd.Series([df.iloc[i][col]]), errors="coerce").iloc[0]
        except Exception:
            return np.nan

    # Audit PivotHigh against bearish pattern
    for pidx in np.where(np.isfinite(pivot_high))[0]:
        lo = max(0, pidx - search_window)
        hi = min(len(df) - 1, pidx + search_window)

        cands = high_candidates[(high_candidates >= lo) & (high_candidates <= hi)]

        record = {
            "file": str(src),
            "year": year,
            "timeframe": timeframe,
            "pivot_type": "PivotHigh",
            "pivot_idx": int(pidx),
            "pivot_timestamp": ts_at(pidx),
            "pivot_price": pivot_high[pidx],
            "matched": 0,
            "match_class": "none",
            "best_score": np.nan,
            "best_confirm_idx": np.nan,
            "best_confirm_timestamp": np.nan,
            "best_break_idx": np.nan,
            "best_break_timestamp": np.nan,
            "break_to_pivot_bars": np.nan,
            "confirm_to_pivot_bars": np.nan,
            "search_window": int(search_window),
            "notes": "",
        }

        best_score = None
        best_data = None

        for cidx in cands:
            br = df.iloc[cidx]["UpperBreakIdx"]
            if not np.isfinite(br):
                continue
            br = int(br)

            score = score_candidate_for_pivot(pidx, br, int(cidx), direction="bear")
            if best_score is None or score < best_score:
                best_score = score
                best_data = (br, int(cidx), score)

        if best_data is not None:
            br, cf, score = best_data
            cls = classify_match(
                pivot_idx=pidx,
                break_idx=br,
                confirm_idx=cf,
                strict_break_near=strict_break_near,
                strict_confirm_after=strict_confirm_after,
                loose_break_near=loose_break_near,
                loose_confirm_after=loose_confirm_after,
            )

            record.update({
                "matched": 1 if cls in {"strict", "loose"} else 0,
                "match_class": cls,
                "best_score": score,
                "best_confirm_idx": cf,
                "best_confirm_timestamp": ts_at(cf),
                "best_break_idx": br,
                "best_break_timestamp": ts_at(br),
                "break_to_pivot_bars": br - pidx,
                "confirm_to_pivot_bars": cf - pidx,
            })
            if cls == "off_window":
                record["notes"] = "best bearish candidate found in audit window, but outside match tolerances"
        else:
            record["notes"] = "no bearish candidate in audit window"

        rows.append(record)

    # Audit PivotLow against bullish pattern
    for pidx in np.where(np.isfinite(pivot_low))[0]:
        lo = max(0, pidx - search_window)
        hi = min(len(df) - 1, pidx + search_window)

        cands = low_candidates[(low_candidates >= lo) & (low_candidates <= hi)]

        record = {
            "file": str(src),
            "year": year,
            "timeframe": timeframe,
            "pivot_type": "PivotLow",
            "pivot_idx": int(pidx),
            "pivot_timestamp": ts_at(pidx),
            "pivot_price": pivot_low[pidx],
            "matched": 0,
            "match_class": "none",
            "best_score": np.nan,
            "best_confirm_idx": np.nan,
            "best_confirm_timestamp": np.nan,
            "best_break_idx": np.nan,
            "best_break_timestamp": np.nan,
            "break_to_pivot_bars": np.nan,
            "confirm_to_pivot_bars": np.nan,
            "search_window": int(search_window),
            "notes": "",
        }

        best_score = None
        best_data = None

        for cidx in cands:
            br = df.iloc[cidx]["LowerBreakIdx"]
            if not np.isfinite(br):
                continue
            br = int(br)

            score = score_candidate_for_pivot(pidx, br, int(cidx), direction="bull")
            if best_score is None or score < best_score:
                best_score = score
                best_data = (br, int(cidx), score)

        if best_data is not None:
            br, cf, score = best_data
            cls = classify_match(
                pivot_idx=pidx,
                break_idx=br,
                confirm_idx=cf,
                strict_break_near=strict_break_near,
                strict_confirm_after=strict_confirm_after,
                loose_break_near=loose_break_near,
                loose_confirm_after=loose_confirm_after,
            )

            record.update({
                "matched": 1 if cls in {"strict", "loose"} else 0,
                "match_class": cls,
                "best_score": score,
                "best_confirm_idx": cf,
                "best_confirm_timestamp": ts_at(cf),
                "best_break_idx": br,
                "best_break_timestamp": ts_at(br),
                "break_to_pivot_bars": br - pidx,
                "confirm_to_pivot_bars": cf - pidx,
            })
            if cls == "off_window":
                record["notes"] = "best bullish candidate found in audit window, but outside match tolerances"
        else:
            record["notes"] = "no bullish candidate in audit window"

        rows.append(record)

    pivot_audit = pd.DataFrame(rows)

    if pivot_audit.empty:
        summary = pd.DataFrame([{
            "file": str(src),
            "year": year,
            "timeframe": timeframe,
            "pivot_count": 0,
            "matched_count": 0,
            "strict_count": 0,
            "loose_count": 0,
            "match_rate": np.nan,
            "median_abs_break_error": np.nan,
            "median_abs_confirm_error": np.nan,
        }])
        return pivot_audit, summary

    break_err = pivot_audit["break_to_pivot_bars"].abs()
    confirm_err = pivot_audit["confirm_to_pivot_bars"].abs()

    summary = pd.DataFrame([{
        "file": str(src),
        "year": year,
        "timeframe": timeframe,
        "pivot_count": int(len(pivot_audit)),
        "matched_count": int((pivot_audit["matched"] == 1).sum()),
        "strict_count": int((pivot_audit["match_class"] == "strict").sum()),
        "loose_count": int((pivot_audit["match_class"] == "loose").sum()),
        "match_rate": float((pivot_audit["matched"] == 1).mean()),
        "median_abs_break_error": float(np.nanmedian(break_err)) if break_err.notna().any() else np.nan,
        "median_abs_confirm_error": float(np.nanmedian(confirm_err)) if confirm_err.notna().any() else np.nan,
    }])

    return pivot_audit, summary


def process_file(
    src: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(src)
    require_columns(df, REQUIRED_CORE, src)

    year = src.parent.name
    stem = src.stem.lower()
    timeframe = next((tf for tf in args.timeframes if stem.startswith(tf.lower())), "unknown")

    time_col = get_time_col(df)

    df = build_psar_columns(df, step=args.psar_step, max_step=args.psar_max_step)
    df = detect_pattern_candidates(
        df,
        break_lookback_min=args.break_lookback_min,
        break_lookback_max=args.break_lookback_max,
    )

    pivot_audit, file_summary = analyze_pivots_for_file(
        df=df,
        src=src,
        year=year,
        timeframe=timeframe,
        time_col=time_col,
        search_window=args.search_window,
        strict_break_near=args.strict_break_near,
        strict_confirm_after=args.strict_confirm_after,
        loose_break_near=args.loose_break_near,
        loose_confirm_after=args.loose_confirm_after,
    )

    return pivot_audit, file_summary


def build_global_summary(file_summaries: pd.DataFrame) -> pd.DataFrame:
    if file_summaries.empty:
        return pd.DataFrame()

    grp = (
        file_summaries
        .groupby(["year", "timeframe"], dropna=False, as_index=False)
        .agg(
            files=("file", "count"),
            pivot_count=("pivot_count", "sum"),
            matched_count=("matched_count", "sum"),
            strict_count=("strict_count", "sum"),
            loose_count=("loose_count", "sum"),
            median_file_match_rate=("match_rate", "median"),
            median_abs_break_error=("median_abs_break_error", "median"),
            median_abs_confirm_error=("median_abs_confirm_error", "median"),
        )
    )

    grp["overall_match_rate"] = grp["matched_count"] / grp["pivot_count"].replace(0, np.nan)
    return grp


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(
        input_root=input_root,
        years=args.years,
        timeframes=args.timeframes,
        pattern=args.glob,
    )

    if not files:
        print("[DONE] No files found.")
        return

    all_pivots: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []

    print(f"[INFO] Found {len(files)} files")

    for src in files:
        try:
            print(f"[LOAD] {src}")
            pivot_audit, file_summary = process_file(src, args)
            all_pivots.append(pivot_audit)
            all_summaries.append(file_summary)
            print(f"[OK] {src} pivots={len(pivot_audit)}")
        except Exception as e:
            print(f"[ERROR] {src}: {e}")

    pivots_df = pd.concat(all_pivots, ignore_index=True) if all_pivots else pd.DataFrame()
    summaries_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    global_summary_df = build_global_summary(summaries_df)

    pivots_path = output_dir / "pivot_pattern_audit_rows.csv"
    files_path = output_dir / "pivot_pattern_audit_file_summary.csv"
    global_path = output_dir / "pivot_pattern_audit_global_summary.csv"

    pivots_df.to_csv(pivots_path, index=False, float_format=args.float_format)
    summaries_df.to_csv(files_path, index=False, float_format=args.float_format)
    global_summary_df.to_csv(global_path, index=False, float_format=args.float_format)

    print(f"[WRITE] {pivots_path}")
    print(f"[WRITE] {files_path}")
    print(f"[WRITE] {global_path}")

    if not global_summary_df.empty:
        print("\n[GLOBAL SUMMARY]")
        print(global_summary_df.to_string(index=False))
    else:
        print("\n[GLOBAL SUMMARY] empty")


if __name__ == "__main__":
    main()