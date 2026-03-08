#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimize_pivot_pattern_params_h1_2022.py

Grid-search BB + PSAR parameters for h1 / 2022 against sparse pivots.

Inputs required in CSV:
    open, high, low, close, PivotHigh, PivotLow

Outputs:
    - ranked parameter table CSV
    - optional top-N printed to terminal

This script recomputes:
    - dynamic Bollinger bands
    - PSAR
    - band-break / PSAR / BB-midline pattern
    - pivot alignment audit

Designed for first-pass optimization on h1 / 2022.
CPU is sufficient for this pass.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


# ---------------------------------------------------------------------
# Dynamic BB
# ---------------------------------------------------------------------

def curvature_energy(close: pd.Series, span: int = 64) -> pd.Series:
    r = np.log(close / close.shift(1))
    dr = r.diff()
    d2r = dr.diff()
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    kappa = (d2r / scale).ewm(span=max(8, span // 4), adjust=False).mean()
    vol_energy = np.log1p(kappa.abs())
    return vol_energy.astype(np.float32)


def compute_bb_dynamic(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int,
    base_k: float,
    k_period: int = 14,
) -> pd.DataFrame:
    vol_energy = curvature_energy(close)

    mu = close.rolling(window, min_periods=1).mean()
    sigma = close.rolling(window, min_periods=1).std(ddof=0)

    g = (1.0 + vol_energy).clip(lower=1.0)
    sigma_dyn = sigma * g
    upper = mu + base_k * sigma_dyn
    lower = mu - base_k * sigma_dyn

    bandwidth = sigma_dyn
    bw_roll_min = bandwidth.rolling(100, min_periods=1).min()
    bw_roll_max = bandwidth.rolling(100, min_periods=1).max()
    bb_pct = 100 * (bandwidth - bw_roll_min) / (bw_roll_max - bw_roll_min + 1e-12)
    bw_expansion = bandwidth.pct_change(5).fillna(0)

    lowest = low.rolling(k_period, min_periods=1).min()
    highest = high.rolling(k_period, min_periods=1).max()
    raw_k = 100 * (close - lowest) / (highest - lowest + 1e-12)
    stoch_k = raw_k.rolling(3, min_periods=1).mean()
    compression = 1 + 0.5 * vol_energy
    stoch_k_dyn = 50 + (stoch_k - 50) / compression

    out = pd.DataFrame(index=close.index)
    out["bb_mu_dyn"] = mu.astype(np.float32)
    out["bb_sigma_dyn"] = sigma_dyn.astype(np.float32)
    out["bb_lower_dyn"] = lower.astype(np.float32)
    out["bb_upper_dyn"] = upper.astype(np.float32)
    out["bb_percentile"] = bb_pct.clip(0, 100).astype(np.float32)
    out["bandwidth"] = bandwidth.astype(np.float32)
    out["bw_expansion_rate"] = bw_expansion.clip(-1, 1).astype(np.float32)
    out["stoch_k_dyn"] = stoch_k_dyn.clip(0, 100).astype(np.float32)
    return out


# ---------------------------------------------------------------------
# PSAR
# ---------------------------------------------------------------------

def compute_psar(
    high: np.ndarray,
    low: np.ndarray,
    start_af: float,
    step_af: float,
    max_af: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(high)
    psar = np.full(n, np.nan, dtype=float)
    bull = np.full(n, 0, dtype=np.int8)

    if n < 2:
        return psar, bull

    init_bull = high[1] + low[1] >= high[0] + low[0]
    bull[0] = 1 if init_bull else 0
    bull[1] = bull[0]

    if init_bull:
        psar[1] = low[0]
        ep = max(high[0], high[1])
    else:
        psar[1] = high[0]
        ep = min(low[0], low[1])

    af = start_af

    for i in range(2, n):
        prev_psar = psar[i - 1]

        if bull[i - 1] == 1:
            cand = prev_psar + af * (ep - prev_psar)
            cand = min(cand, low[i - 1], low[i - 2])

            if low[i] < cand:
                bull[i] = 0
                psar[i] = ep
                ep = low[i]
                af = start_af
            else:
                bull[i] = 1
                psar[i] = cand
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step_af, max_af)

        else:
            cand = prev_psar + af * (ep - prev_psar)
            cand = max(cand, high[i - 1], high[i - 2])

            if high[i] > cand:
                bull[i] = 1
                psar[i] = ep
                ep = high[i]
                af = start_af
            else:
                bull[i] = 0
                psar[i] = cand
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step_af, max_af)

    if np.isnan(psar[0]):
        psar[0] = low[0] if bull[1] == 1 else high[0]

    return psar, bull


# ---------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------

def recent_break_age(flag: np.ndarray, min_age: int = 1, max_age: int = 4) -> np.ndarray:
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


def detect_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    o = pd.to_numeric(out["open"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    bb_u = pd.to_numeric(out["bb_upper_dyn"], errors="coerce").to_numpy(dtype=float)
    bb_m = pd.to_numeric(out["bb_mu_dyn"], errors="coerce").to_numpy(dtype=float)
    bb_l = pd.to_numeric(out["bb_lower_dyn"], errors="coerce").to_numpy(dtype=float)

    upper_break = (h > bb_u).astype(int)
    lower_break = (l < bb_l).astype(int)

    out["UpperBandBreakFlag"] = upper_break
    out["LowerBandBreakFlag"] = lower_break

    out["UpperBandBreakAge_4"] = recent_break_age(upper_break, 1, 4)
    out["LowerBandBreakAge_4"] = recent_break_age(lower_break, 1, 4)

    out["UpperBandBreakInLast4"] = np.isfinite(out["UpperBandBreakAge_4"]).astype(int)
    out["LowerBandBreakInLast4"] = np.isfinite(out["LowerBandBreakAge_4"]).astype(int)

    out["PSARBearState"] = (out["PSARBullState"] == 0).astype(int)
    out["PSARBullShiftFlag"] = ((out["PSARBullState"].shift(1).fillna(out["PSARBullState"]) == 0) & (out["PSARBullState"] == 1)).astype(int)
    out["PSARBearShiftFlag"] = ((out["PSARBearState"].shift(1).fillna(out["PSARBearState"]) == 0) & (out["PSARBearState"] == 1)).astype(int)

    out["BarsSincePSARBearShift"] = bars_since_event(out["PSARBearShiftFlag"].to_numpy(dtype=int))
    out["BarsSincePSARBullShift"] = bars_since_event(out["PSARBullShiftFlag"].to_numpy(dtype=int))

    out["BearMidlineConfirm"] = (((l <= bb_m) & (c < bb_m)) | ((o > bb_m) & (c < bb_m))).astype(int)
    out["BullMidlineConfirm"] = (((h >= bb_m) & (c > bb_m)) | ((o < bb_m) & (c > bb_m))).astype(int)

    out["BearCandidateLoose"] = (
        (out["UpperBandBreakInLast4"] == 1)
        & (out["PSARBearState"] == 1)
        & (out["BearMidlineConfirm"] == 1)
    ).astype(int)

    out["BullCandidateLoose"] = (
        (out["LowerBandBreakInLast4"] == 1)
        & (out["PSARBullState"] == 1)
        & (out["BullMidlineConfirm"] == 1)
    ).astype(int)

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


# ---------------------------------------------------------------------
# Pivot audit
# ---------------------------------------------------------------------

def score_candidate_for_pivot(pivot_idx: int, break_idx: int, confirm_idx: int) -> float:
    break_err = abs(break_idx - pivot_idx)
    confirm_err = (confirm_idx - pivot_idx) if confirm_idx >= pivot_idx else 2.0 * (pivot_idx - confirm_idx)
    return float(break_err + confirm_err)


def classify_match(
    pivot_idx: int,
    break_idx: int,
    confirm_idx: int,
    strict_break_near: int = 2,
    strict_confirm_after: int = 8,
    loose_break_near: int = 8,
    loose_confirm_after: int = 20,
) -> str:
    break_err = abs(break_idx - pivot_idx)
    confirm_after = confirm_idx - pivot_idx

    if break_err <= strict_break_near and 0 <= confirm_after <= strict_confirm_after:
        return "strict"
    if break_err <= loose_break_near and -2 <= confirm_after <= loose_confirm_after:
        return "loose"
    return "off_window"


def audit_one_df(df: pd.DataFrame, search_window: int = 200) -> Dict[str, float]:
    pivot_high = pd.to_numeric(df["PivotHigh"], errors="coerce").to_numpy(dtype=float)
    pivot_low = pd.to_numeric(df["PivotLow"], errors="coerce").to_numpy(dtype=float)

    high_candidates = df.index[df["BearCandidateLoose"] == 1].to_numpy()
    low_candidates = df.index[df["BullCandidateLoose"] == 1].to_numpy()

    rows = []

    for pidx in np.where(np.isfinite(pivot_high))[0]:
        lo = max(0, pidx - search_window)
        hi = min(len(df) - 1, pidx + search_window)
        cands = high_candidates[(high_candidates >= lo) & (high_candidates <= hi)]

        best = None
        best_score = None
        for cidx in cands:
            br = df.iloc[cidx]["UpperBreakIdx"]
            if not np.isfinite(br):
                continue
            br = int(br)
            score = score_candidate_for_pivot(pidx, br, int(cidx))
            if best_score is None or score < best_score:
                best_score = score
                best = (br, int(cidx), score)

        if best is None:
            rows.append(("PivotHigh", "none", np.nan, np.nan))
        else:
            br, cf, _ = best
            cls = classify_match(pidx, br, cf)
            rows.append(("PivotHigh", cls, abs(br - pidx), abs(cf - pidx)))

    for pidx in np.where(np.isfinite(pivot_low))[0]:
        lo = max(0, pidx - search_window)
        hi = min(len(df) - 1, pidx + search_window)
        cands = low_candidates[(low_candidates >= lo) & (low_candidates <= hi)]

        best = None
        best_score = None
        for cidx in cands:
            br = df.iloc[cidx]["LowerBreakIdx"]
            if not np.isfinite(br):
                continue
            br = int(br)
            score = score_candidate_for_pivot(pidx, br, int(cidx))
            if best_score is None or score < best_score:
                best_score = score
                best = (br, int(cidx), score)

        if best is None:
            rows.append(("PivotLow", "none", np.nan, np.nan))
        else:
            br, cf, _ = best
            cls = classify_match(pidx, br, cf)
            rows.append(("PivotLow", cls, abs(br - pidx), abs(cf - pidx)))

    if not rows:
        return {
            "pivot_count": 0,
            "strict_count": 0,
            "loose_count": 0,
            "matched_count": 0,
            "match_rate": np.nan,
            "median_abs_break_error": np.nan,
            "median_abs_confirm_error": np.nan,
            "score": -np.inf,
        }

    audit = pd.DataFrame(rows, columns=["pivot_type", "match_class", "abs_break_error", "abs_confirm_error"])

    pivot_count = len(audit)
    strict_count = int((audit["match_class"] == "strict").sum())
    loose_count = int((audit["match_class"] == "loose").sum())
    matched_count = strict_count + loose_count
    match_rate = matched_count / pivot_count if pivot_count else np.nan

    med_break = float(np.nanmedian(audit["abs_break_error"])) if audit["abs_break_error"].notna().any() else np.nan
    med_confirm = float(np.nanmedian(audit["abs_confirm_error"])) if audit["abs_confirm_error"].notna().any() else np.nan

    score = (
        4.0 * strict_count
        + 1.5 * loose_count
        - 0.10 * (0.0 if np.isnan(med_break) else med_break)
        - 0.10 * (0.0 if np.isnan(med_confirm) else med_confirm)
    )

    return {
        "pivot_count": pivot_count,
        "strict_count": strict_count,
        "loose_count": loose_count,
        "matched_count": matched_count,
        "match_rate": match_rate,
        "median_abs_break_error": med_break,
        "median_abs_confirm_error": med_confirm,
        "score": score,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to 2022 h1 CSV")
    ap.add_argument("--output", required=True, help="Path to ranked results CSV")
    ap.add_argument("--search-window", type=int, default=200)

    ap.add_argument("--bb-windows", nargs="*", type=int, default=[16, 20, 24, 30, 36])
    ap.add_argument("--bb-ks", nargs="*", type=float, default=[1.6, 1.8, 2.0, 2.2, 2.4])

    ap.add_argument("--psar-starts", nargs="*", type=float, default=[0.01, 0.02, 0.03])
    ap.add_argument("--psar-steps", nargs="*", type=float, default=[0.01, 0.02, 0.03])
    ap.add_argument("--psar-maxes", nargs="*", type=float, default=[0.10, 0.14, 0.20, 0.30])

    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    required = ["open", "high", "low", "close", "PivotHigh", "PivotLow"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    results = []

    bb_cache: Dict[Tuple[int, float], pd.DataFrame] = {}

    total = (
        len(args.bb_windows)
        * len(args.bb_ks)
        * len(args.psar_starts)
        * len(args.psar_steps)
        * len(args.psar_maxes)
    )
    done = 0

    for bb_window, bb_k in itertools.product(args.bb_windows, args.bb_ks):
        bb_key = (bb_window, bb_k)
        if bb_key not in bb_cache:
            bb_cache[bb_key] = compute_bb_dynamic(
                close=close,
                high=high,
                low=low,
                window=bb_window,
                base_k=bb_k,
                k_period=14,
            )

        bb_df = bb_cache[bb_key]

        for psar_start, psar_step, psar_max in itertools.product(
            args.psar_starts, args.psar_steps, args.psar_maxes
        ):
            done += 1

            work = df.copy()
            for col in bb_df.columns:
                work[col] = bb_df[col]

            psar, bull = compute_psar(
                high=pd.to_numeric(work["high"], errors="coerce").to_numpy(dtype=float),
                low=pd.to_numeric(work["low"], errors="coerce").to_numpy(dtype=float),
                start_af=psar_start,
                step_af=psar_step,
                max_af=psar_max,
            )
            work["PSAR"] = psar
            work["PSARBullState"] = bull

            work = detect_candidates(work)
            metrics = audit_one_df(work, search_window=args.search_window)

            results.append({
                "bb_window": bb_window,
                "bb_k": bb_k,
                "psar_start": psar_start,
                "psar_step": psar_step,
                "psar_max": psar_max,
                **metrics,
            })

            if done % 50 == 0 or done == total:
                print(f"[{done}/{total}] complete")

    res = pd.DataFrame(results)
    res = res.sort_values(
        by=["score", "strict_count", "matched_count", "match_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)

    print("\nTOP RESULTS")
    print(res.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()