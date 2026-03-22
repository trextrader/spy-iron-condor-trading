#!/usr/bin/env python3
"""
Populate deterministic OHLCV-derived indicator columns for 2020-2024.

Target columns (causal, OHLCV-only):
    psar_trend, psar_mark, psar_adaptive, psar_reversion_mu
    pressure_up, pressure_down
    gap_risk_score, max_dd_60m
    adx_adaptive, bb_percentile, rsi_dyn, stoch_k_dyn

Notes:
  - Causal only: uses current and past bars, never future bars
  - Idempotent: skips already-populated columns unless --force-all-targets
  - trade_count is NOT fabricated here (SOURCE_REQUIRED policy)
  - target_spot is NOT populated here (LABEL policy — use data_pipeline_v43.py)
  - Historical files live in normalized_shells/{year}/; use --shells-dir to override

Usage:
    python populate_ohlcv_indicators_v43.py
    python populate_ohlcv_indicators_v43.py --years 2022 2023 --timeframes m1 m5
    python populate_ohlcv_indicators_v43.py --force-all-targets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Project paths ─────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent.parent

# ── Constants ─────────────────────────────────────────────────────────────────
YEARS      = [2020, 2021, 2022, 2023, 2024]
TIMEFRAMES = ["m1", "m5", "m15", "h1"]

TARGET_COLS = [
    "psar_trend",
    "psar_mark",
    "psar_adaptive",
    "psar_reversion_mu",
    "pressure_up",
    "pressure_down",
    "gap_risk_score",
    "max_dd_60m",
    "adx_adaptive",
    "bb_percentile",
    "rsi_dyn",
    "stoch_k_dyn",
]

# Bounds used for validation
COL_BOUNDS = {
    "bb_percentile":  (0.0,   1.0),
    "rsi_dyn":        (0.0, 100.0),
    "stoch_k_dyn":    (0.0, 100.0),
    "adx_adaptive":   (0.0, 100.0),
    "psar_trend":     (-1.0,  1.0),
}

# Number of bars used for the max_dd_60m rolling window per timeframe.
# m1/m5/m15: exact 60-minute equivalent.
# h1: 4 bars (4-hour lookback) — window=1 is always 0 (single bar max==min).
TF_TO_BARS_60M = {
    "m1":  60,
    "m5":  12,
    "m15":  4,
    "h1":   4,
}

COLUMN_FILL_POLICY: Dict[str, str] = {
    "trade_count":    "SOURCE_REQUIRED",   # not in OHLCV shells; skip
    "target_spot":    "LABEL_PIPELINE",    # 5-bar fwd log return; use data_pipeline_v43.py
    "TrendSeekerUp":  "PLATFORM_OR_PROXY", # script 5
    "TrendSeekerDown":"PLATFORM_OR_PROXY",
    "TurtleChn-Up":   "PLATFORM_OR_PROXY",
    "TurtleChn-Low":  "PLATFORM_OR_PROXY",
    "FRAMA":          "PLATFORM_OR_PROXY",
    "McClellanOsc":   "DEFER_IF_BREADTH_MISSING",
}


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("populate_ohlcv_indicators_v43")
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


# ── Path helpers ──────────────────────────────────────────────────────────────

def target_file(shells_dir: Path, year: int, tf: str) -> Path:
    return shells_dir / str(year) / f"{tf}_dataset_v43_{year}.csv"


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _rolling_percentile(series: pd.Series, window: int = 100) -> pd.Series:
    """Causal rolling percentile rank of series[-1] within window."""
    def _pct_rank(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        last = arr[-1]
        if np.isnan(last):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= last).mean())

    return series.rolling(window=window, min_periods=5).apply(
        lambda x: _pct_rank(np.asarray(x, dtype=float)), raw=True
    )


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI, clipped to [0, 100]."""
    delta    = close.diff()
    up       = delta.clip(lower=0.0)
    down     = (-delta).clip(lower=0.0)
    avg_gain = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).clip(0.0, 100.0).astype(np.float32)


def compute_stoch_k(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Stochastic %K, clipped to [0, 100]."""
    rolling_low  = low.rolling(period, min_periods=period).min()
    rolling_high = high.rolling(period, min_periods=period).max()
    denom = (rolling_high - rolling_low).replace(0.0, np.nan)
    stoch = 100.0 * (close - rolling_low) / denom
    return stoch.fillna(50.0).clip(0.0, 100.0).astype(np.float32)


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Wilder-smoothed ADX, clipped to [0, 100]."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move   > down_move) & (up_move   > 0), up_move,   0.0)
    minus_dm = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    tr_s  = pd.Series(tr).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    pDM_s = pd.Series(plus_dm,  index=high.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    mDM_s = pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di  = 100.0 * pDM_s / tr_s.replace(0, np.nan)
    minus_di = 100.0 * mDM_s / tr_s.replace(0, np.nan)

    dx  = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return adx.fillna(25.0).clip(0.0, 100.0).astype(np.float32)


def compute_psar_block(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_step:  float = 0.02,
    af_max:   float = 0.20,
) -> pd.DataFrame:
    """
    Classic Parabolic SAR.

    Adds columns:
        psar_mark         - raw SAR price level
        psar_trend        - +1.0 (uptrend) / -1.0 (downtrend)
        psar_adaptive     - normalised distance: (close - SAR) / close
        psar_reversion_mu - 10-bar EMA of psar_adaptive (mean-reversion signal)
    """
    high  = df["high"].astype(float).values
    low   = df["low"].astype(float).values
    close = df["close"].astype(float).values
    n     = len(df)

    psar_arr  = np.empty(n, dtype=np.float64)
    trend_arr = np.empty(n, dtype=np.float64)

    bull = True
    ep   = high[0]
    af   = af_start
    psar_arr[0]  = low[0]
    trend_arr[0] = 1.0

    for i in range(1, n):
        prev_psar = psar_arr[i - 1]

        cur_psar = prev_psar + af * (ep - prev_psar)

        if bull:
            cur_psar = min(cur_psar, low[i - 1])
            if i > 1:
                cur_psar = min(cur_psar, low[i - 2])

            if low[i] < cur_psar:
                bull     = False
                cur_psar = ep
                ep       = low[i]
                af       = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            cur_psar = max(cur_psar, high[i - 1])
            if i > 1:
                cur_psar = max(cur_psar, high[i - 2])

            if high[i] > cur_psar:
                bull     = True
                cur_psar = ep
                ep       = high[i]
                af       = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

        psar_arr[i]  = cur_psar
        trend_arr[i] = 1.0 if bull else -1.0

    out = df.copy()
    out["psar_mark"]    = psar_arr
    out["psar_trend"]   = trend_arr.astype(np.float32)
    out["psar_adaptive"] = (
        pd.Series(close - psar_arr, index=df.index) / pd.Series(close, index=df.index).replace(0, np.nan)
    ).fillna(0.0).astype(np.float32)
    out["psar_reversion_mu"] = (
        pd.Series(close - psar_arr, index=df.index)
        .ewm(span=10, adjust=False)
        .mean()
        .fillna(0.0)
        .astype(np.float32)
    )
    return out


def compute_pressure_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Directional pressure from bar body position.

    pressure_up   - upward body fraction of bar range   in [0, 1]
    pressure_down - downward body fraction of bar range in [0, 1]
    """
    out  = df.copy()
    rng  = (out["high"] - out["low"]).replace(0, np.nan)
    body = out["close"] - out["open"]

    out["pressure_up"]   = ((body.clip(lower=0)  / rng).fillna(0.0)).clip(0.0, 1.0).astype(np.float32)
    out["pressure_down"] = (((-body).clip(lower=0) / rng).fillna(0.0)).clip(0.0, 1.0).astype(np.float32)
    return out


def compute_gap_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Normalised overnight/intraday gap size.

    gap_risk_score = |open - prev_close| / (0.01 * prev_close)
    (number of 1%-moves; 0 = no gap; ~1 = 1% gap)
    """
    prev_close = df["close"].shift(1)
    gap_abs    = (df["open"] - prev_close).abs()
    denom      = (prev_close.abs() * 0.01).replace(0, np.nan)
    return (gap_abs / denom).fillna(0.0).astype(np.float32)


def compute_max_dd_60m(df: pd.DataFrame, timeframe: str) -> pd.Series:
    """
    Rolling maximum drawdown over the 60-minute equivalent window.

    Returned as a non-positive fraction (e.g. -0.012 = -1.2% drawdown).
    Timeframe-aware window sizes:
        m1:  60 bars
        m5:  12 bars
        m15:  4 bars
        h1:   1 bar  (current-bar high vs close)
    """
    bars         = TF_TO_BARS_60M[timeframe]
    rolling_peak = df["close"].rolling(window=bars, min_periods=1).max()
    dd           = (df["close"] - rolling_peak) / rolling_peak.replace(0, np.nan)
    return dd.fillna(0.0).astype(np.float32)


def compute_bb_percentile(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """
    Causal rolling percentile rank of bb_sigma_dyn (or bandwidth) within window.
    Falls back to 0.5 if neither is present.
    """
    if "bb_sigma_dyn" in df.columns:
        src = df["bb_sigma_dyn"]
    elif "bandwidth" in df.columns:
        src = df["bandwidth"]
    else:
        return pd.Series(0.5, index=df.index, dtype=np.float32)

    return _rolling_percentile(src, window=window).fillna(0.5).clip(0.0, 1.0).astype(np.float32)


# ── Core computation ──────────────────────────────────────────────────────────

def populate_ohlcv_block(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Compute all target indicator columns from OHLCV data."""
    out = compute_psar_block(df)
    out = compute_pressure_block(out)

    out["gap_risk_score"] = compute_gap_risk_score(out)
    out["max_dd_60m"]     = compute_max_dd_60m(out, timeframe=timeframe)
    out["adx_adaptive"]   = compute_adx(out["high"], out["low"], out["close"])
    out["rsi_dyn"]        = compute_rsi(out["close"])
    out["stoch_k_dyn"]    = compute_stoch_k(out["high"], out["low"], out["close"])
    out["bb_percentile"]  = compute_bb_percentile(out)

    return out


# ── Idempotency helper ────────────────────────────────────────────────────────

def _is_unpopulated(s: pd.Series) -> bool:
    """True if a column is all-NaN (not yet computed).

    We intentionally do NOT treat all-zero as unpopulated: a legitimate
    computation (e.g. max_dd_60m with a short lookback) can produce zeros,
    and re-triggering on zeros causes non-idempotent second runs.
    Use --force-all-targets to unconditionally recompute.
    """
    return bool(s.isna().all())


# ── File processor ────────────────────────────────────────────────────────────

def process_file(
    path: Path,
    timeframe: str,
    force_all_targets: bool,
    logger: logging.Logger,
) -> Dict:
    logger.info("  Processing: %s", path)
    df = pd.read_csv(path, low_memory=False)
    computed = populate_ohlcv_block(df, timeframe=timeframe)

    touched : List[str] = []
    skipped : List[str] = []

    for col in TARGET_COLS:
        if col not in computed.columns:
            skipped.append(f"{col}:not_computed")
            continue

        if col not in df.columns:
            df[col] = computed[col]
            touched.append(col)
        else:
            if force_all_targets or _is_unpopulated(df[col]):
                df[col] = computed[col]
                touched.append(col)
            else:
                skipped.append(f"{col}:already_populated")

    df.to_csv(path, index=False)

    # ── Per-column stats ───────────────────────────────────────────────────
    stats: Dict = {}
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        entry: Dict = {"nan_pct": round(float(s.isna().mean()), 6)}
        if pd.api.types.is_numeric_dtype(s):
            entry["zero_pct"] = round(float((s.fillna(0.0) == 0.0).mean()), 6)
            entry["min"]      = float(s.min()) if not s.isna().all() else None
            entry["max"]      = float(s.max()) if not s.isna().all() else None

            # Bounds check
            if col in COL_BOUNDS:
                lo, hi = COL_BOUNDS[col]
                oob = ((s < lo) | (s > hi)).sum()
                if oob:
                    entry["out_of_bounds"] = int(oob)
        stats[col] = entry

    logger.info(
        "    touched=%d skipped=%d rows=%d",
        len(touched), len(skipped), len(df),
    )

    return {
        "file":      str(path),
        "timeframe": timeframe,
        "rows":      len(df),
        "touched":   touched,
        "skipped":   skipped,
        "stats":     stats,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Populate OHLCV-derived indicator columns for 2020-2024 v43 datasets."
    )
    ap.add_argument(
        "--base-dir",
        default=str(_PROJECT_ROOT / "data" / "Datasetv4" / "v43"),
        help="Root v43 directory (contains 2025/ subfolder for reference).",
    )
    ap.add_argument(
        "--shells-dir",
        default=None,
        help=(
            "Directory containing normalized_shells/{year}/ subdirs. "
            "Defaults to {base_dir}/normalized_shells"
        ),
    )
    ap.add_argument("--years",      nargs="+", type=int, default=YEARS)
    ap.add_argument("--timeframes", nargs="+",           default=TIMEFRAMES)
    ap.add_argument(
        "--log",
        default=str(_PROJECT_ROOT / "logs" / "populate_ohlcv_indicators_v43.log"),
    )
    ap.add_argument(
        "--summary-json",
        default=str(
            _PROJECT_ROOT / "reports" / "population_audit"
            / "populate_ohlcv_indicators_v43_summary.json"
        ),
    )
    ap.add_argument(
        "--force-all-targets",
        action="store_true",
        help="Recompute and overwrite target columns even if already populated.",
    )
    args = ap.parse_args()

    base_dir   = Path(args.base_dir)
    shells_dir = Path(args.shells_dir) if args.shells_dir else base_dir / "normalized_shells"
    log_path   = Path(args.log)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info("populate_ohlcv_indicators_v43.py")
    logger.info("shells_dir : %s", shells_dir)
    logger.info("years      : %s", args.years)
    logger.info("timeframes : %s", args.timeframes)
    logger.info("force      : %s", args.force_all_targets)
    logger.info("=" * 60)

    # Print fill policy for deferred columns
    for col, policy in COLUMN_FILL_POLICY.items():
        logger.info("  POLICY  %-28s  %s", col, policy)

    all_results: List[Dict] = []

    for year in args.years:
        for tf in args.timeframes:
            path = target_file(shells_dir, year, tf)
            if not path.exists():
                logger.warning("Missing: %s", path)
                continue
            try:
                result = process_file(path, tf, args.force_all_targets, logger)
                result["year"] = year
                all_results.append(result)
            except Exception as exc:
                logger.exception("FAILED %s: %s", path, exc)
                all_results.append(
                    {"file": str(path), "year": year, "timeframe": tf, "error": str(exc)}
                )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Done. Files processed: %d", len(all_results))
    logger.info("Summary JSON: %s", summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
