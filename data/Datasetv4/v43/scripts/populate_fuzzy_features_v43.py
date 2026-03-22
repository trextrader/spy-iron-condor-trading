#!/usr/bin/env python3
"""
populate_fuzzy_features_v43.py  —  Script 4: Fuzzy / Regime Feature Block

Populates 9 fuzzy/regime columns for 2020-2024 datasets across all timeframes.

Target columns:
    kappa_proxy           — normalized 2nd derivative of log returns (ℝ)
    chaos_membership      — linear ramp on vol_energy proxy [0, 1]
    fuzzy_reversion_11    — 11-input fuzzy reversion score [0, 1]
    consolidation_score   — inverse range-to-width ratio [0, 1]
    breakout_score        — Bollinger Band touch {-1.0, 0.0, +1.0}
    position_size_mult    — step function on chaos level {0.5, 0.75, 1.0}
    spread_ratio          — (high-low) / mid  [0, ~0.05]
    friction_ratio        — sigmoid(spread / (avg_range * atr_norm)) [0, 1]
    McClellanOsc          — single-instrument A/D proxy (EMA19 - EMA39 of sign(Δclose))

Canonical sources (scripts/indicators/):
    kappa_proxy, vol_energy  ← returns_volatility.py
    chaos_membership         ← chaos_regime.py
    position_size_mult       ← execution_risk.py
    fuzzy_reversion_11       ← execution_risk.py
    consolidation_score,
    breakout_score,
    spread_ratio             ← reversal_stack.py
    friction_ratio           ← volume_flow.py
    McClellanOsc             ← mcclellan_osc.py

Note on McClellanOsc:
    Uses single-instrument A/D proxy (sign of close diff) rather than true
    market breadth (ADVA/DECL). Matches mcclellan_osc.py implementation.
    Pass --skip-mcclellan to leave the column untouched.

Note on fuzzy_reversion_11:
    mu_mtf, mu_ivr, mu_vix, mu_sma, mu_bb are stubbed at 0.5 (matching
    execution_risk.py implementation). The remaining 6 factors are computed.

Usage:
    python populate_fuzzy_features_v43.py
    python populate_fuzzy_features_v43.py --years 2020 2021 --timeframes m5
    python populate_fuzzy_features_v43.py --force-all-targets
    python populate_fuzzy_features_v43.py --skip-mcclellan
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Project paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent.parent

# ── Constants ──────────────────────────────────────────────────────────────────
YEARS      = [2020, 2021, 2022, 2023, 2024]
TIMEFRAMES = ["m1", "m5", "m15", "h1"]

TARGET_COLS = [
    "kappa_proxy",
    "chaos_membership",
    "fuzzy_reversion_11",
    "consolidation_score",
    "breakout_score",
    "position_size_mult",
    "spread_ratio",
    "friction_ratio",
    "McClellanOsc",
]


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("populate_fuzzy_features_v43")
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


# ── Path helpers ───────────────────────────────────────────────────────────────

def target_file(shells_dir: Path, year: int, tf: str) -> Path:
    return shells_dir / str(year) / f"{tf}_dataset_v43_{year}.csv"


# ── Idempotency ────────────────────────────────────────────────────────────────

def _is_unpopulated(s: pd.Series) -> bool:
    """True if all-NaN (not yet computed). All-zero is treated as populated."""
    return bool(s.isna().all())


# ── Canonical implementations (mirrors scripts/indicators/) ───────────────────

def _curvature_energy(close: pd.Series, span: int = 64) -> pd.Series:
    """vol_energy = log1p(|kappa|)  — shared helper used by multiple indicators."""
    r     = np.log(close / close.shift(1))
    dr    = r.diff()
    d2r   = dr.diff()
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    kappa = (d2r / scale).ewm(span=max(8, span // 4), adjust=False).mean()
    return np.log1p(kappa.abs()).astype(np.float32)


def compute_kappa_proxy(close: pd.Series, span: int = 64) -> pd.Series:
    """
    Curvature proxy κ — normalized 2nd derivative of log returns.
    Source: returns_volatility.py:compute_curvature_features
    """
    r     = np.log(close / close.shift(1))
    dr    = r.diff()
    d2r   = dr.diff()
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    kappa = (d2r / scale).ewm(span=max(8, span // 4), adjust=False).mean()
    return kappa.astype(np.float32)


def compute_chaos_membership(close: pd.Series,
                              chaos_thresh: float = 2.0,
                              relax_thresh: float = 1.0) -> pd.Series:
    """
    Chaos membership ∈ [0, 1].
    Linear ramp: relax_thresh → 0.0, chaos_thresh → 1.0.
    Source: chaos_regime.py:compute_chaos_membership
    """
    vol_energy   = _curvature_energy(close)
    beta1_proxy  = vol_energy * 2.0  # scale to β₁ range
    membership   = ((beta1_proxy - relax_thresh) /
                    (chaos_thresh - relax_thresh + 1e-12)).clip(0, 1)
    return membership.astype(np.float32)


def compute_position_size_mult(close: pd.Series) -> pd.Series:
    """
    Position size multiplier — step function on chaos gate.
    chaos_gate ≤ 1.0 → 1.0  (full size)
    1.0 < chaos_gate ≤ 2.0  → 0.75
    chaos_gate > 2.0         → 0.5
    Source: execution_risk.py:compute_position_size_mult
    """
    vol_energy  = _curvature_energy(close)
    chaos_gate  = (vol_energy * 2.0).clip(0, 5)
    mult        = np.where(chaos_gate > 2.0, 0.5,
                  np.where(chaos_gate > 1.0, 0.75, 1.0))
    return pd.Series(mult, index=close.index).astype(np.float32)


def compute_reversal_stack(high: pd.Series, low: pd.Series, close: pd.Series,
                            lookback: int = 64) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Consolidation score, breakout score, spread ratio.
    Source: reversal_stack.py:compute_reversal_stack

    Returns: (consolidation_score, breakout_score, spread_ratio)
    """
    vol_energy = _curvature_energy(close)

    # Dynamic Bollinger bands
    mu        = close.rolling(20, min_periods=1).mean()
    sigma     = close.rolling(20, min_periods=1).std(ddof=0)
    g         = (1.0 + vol_energy).clip(lower=1.0)
    sigma_dyn = sigma * g
    upper     = mu + 2.0 * sigma_dyn
    lower     = mu - 2.0 * sigma_dyn

    # Rolling price range
    hi          = close.rolling(lookback, min_periods=1).max()
    lo          = close.rolling(lookback, min_periods=1).min()
    price_range = (hi - lo).abs()

    # Mean band width
    mean_bw = sigma_dyn.rolling(lookback, min_periods=1).mean() + 1e-12

    # Consolidation: inverse of range-to-width ratio
    ratio              = price_range / mean_bw
    consolidation_score = (1.0 / (1.0 + ratio)).clip(0, 1)

    # Breakout: +1 above upper, -1 below lower, 0 inside
    breakout_score = ((close > upper).astype(float) -
                      (close < lower).astype(float))

    # Spread ratio: (high - low) / mid
    mid          = (high + low) / 2.0 + 1e-12
    spread_ratio = (high - low) / mid

    return (consolidation_score.astype(np.float32),
            breakout_score.astype(np.float32),
            spread_ratio.astype(np.float32))


def _compute_atr_pct(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
    """ATR as fraction of close — used internally by friction_ratio."""
    prev_c = close.shift(1)
    tr     = pd.concat([high - low,
                        (high - prev_c).abs(),
                        (low  - prev_c).abs()], axis=1).max(axis=1)
    atr    = tr.ewm(alpha=1 / period, adjust=False).mean()
    return (atr / close.replace(0, np.nan)).fillna(0).astype(np.float32)


def compute_friction_ratio(high: pd.Series, low: pd.Series, close: pd.Series,
                            window: int = 20) -> pd.Series:
    """
    Friction ratio = sigmoid(spread / (avg_range × avg_atr_pct)).
    No bid/ask in OHLCV → spread hardcoded to 0.01 (matches volume_flow.py).
    Source: volume_flow.py:compute_friction_ratio
    """
    spread   = pd.Series(0.01, index=close.index, dtype=np.float32)
    hl_range = (high - low).rolling(window, min_periods=1).mean()
    atr_pct  = _compute_atr_pct(high, low, close)
    atr_norm = atr_pct.rolling(window, min_periods=1).mean() + 1e-12
    friction = spread / (hl_range * atr_norm + 1e-12)
    return (1.0 / (1.0 + np.exp(-friction))).astype(np.float32)


def compute_fuzzy_reversion_11(close: pd.Series, high: pd.Series,
                                low: pd.Series, volume: pd.Series) -> pd.Series:
    """
    11-input fuzzy reversion score ∈ [0, 1].
    Inputs 1-6 are computed; inputs 7-11 (mu_mtf, mu_ivr, mu_vix, mu_sma,
    mu_bb) are stubbed at 0.5 — matches execution_risk.py implementation.
    Source: execution_risk.py:compute_fuzzy_reversion
    """
    vol_energy = _curvature_energy(close)

    # mu_rsi — vol-energy-weighted RSI divergence from 50
    delta   = close.diff()
    gains   = delta.clip(lower=0)
    losses  = (-delta).clip(lower=0)
    weights = (1.0 + vol_energy).astype(float)
    w_g     = (gains   * weights).rolling(14, min_periods=1).sum()
    w_l     = (losses  * weights).rolling(14, min_periods=1).sum()
    w_s     = weights.rolling(14, min_periods=1).sum() + 1e-12
    rs      = (w_g / w_s) / (w_l / w_s + 1e-12)
    rsi     = 100 - 100 / (1 + rs)
    mu_rsi  = ((rsi - 50).abs() / 50).clip(0, 1)

    # mu_stoch — stochastic divergence from 50
    ll      = low.rolling(14,  min_periods=1).min()
    hh      = high.rolling(14, min_periods=1).max()
    stoch   = 100 * (close - ll) / (hh - ll + 1e-12)
    mu_stoch = ((stoch - 50).abs() / 50).clip(0, 1)

    # mu_adx — low ADX = weak trend = reversion likely
    prev_c  = close.shift(1)
    tr      = pd.concat([high - low,
                         (high - prev_c).abs(),
                         (low  - prev_c).abs()], axis=1).max(axis=1)
    up      = high.diff()
    down    = -low.diff()
    plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0),   index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)
    alpha   = 1 / 14
    tr_s    = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean()  / (tr_s + 1e-12)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / (tr_s + 1e-12)
    dx      = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx     = dx.ewm(alpha=alpha, adjust=False).mean()
    mu_adx  = (1.0 - adx / 50.0).clip(0, 1)

    # mu_bbsqueeze — low bandwidth percentile = squeeze = reversion likely
    sigma       = close.rolling(20, min_periods=1).std(ddof=0)
    g           = (1.0 + vol_energy).clip(lower=1.0)
    bw          = sigma * g
    bw_min      = bw.rolling(100, min_periods=1).min()
    bw_max      = bw.rolling(100, min_periods=1).max()
    bb_pct      = (bw - bw_min) / (bw_max - bw_min + 1e-12)
    mu_bbsqueeze = (1.0 - bb_pct).clip(0, 1)

    # mu_psar — price distance from SMA (ATR-normalised)
    atr        = tr.ewm(alpha=1 / 14, adjust=False).mean()
    sma        = close.rolling(20, min_periods=1).mean()
    dist_sma   = ((close - sma) / (atr + 1e-12)).abs()
    mu_psar    = np.exp(-dist_sma * 0.5).clip(0, 1)

    # mu_vol — CMF (volume-based buying/selling pressure)
    hl         = high - low + 1e-12
    clv        = (2 * close - low - high) / hl
    cmf        = ((clv * volume).rolling(20, min_periods=1).sum() /
                  (volume.rolling(20, min_periods=1).sum() + 1e-12))
    mu_vol     = ((cmf + 1.0) / 2.0).clip(0, 1)

    # Weighted aggregation (stubbed inputs fixed at 0.5)
    score = (0.15 * mu_rsi        +
             0.05 * mu_stoch      +
             0.05 * mu_adx        +
             0.10 * mu_psar       +
             0.03 * mu_bbsqueeze  +
             0.02 * mu_vol        +
             0.25 * 0.5           +   # mu_mtf  stub
             0.15 * 0.5           +   # mu_ivr  stub
             0.10 * 0.5           +   # mu_vix  stub
             0.05 * 0.5           +   # mu_sma  stub
             0.05 * 0.5)              # mu_bb   stub

    return score.clip(0, 1).astype(np.float32)


def compute_mcclellan(close: pd.Series,
                      fast: int = 19, slow: int = 39) -> pd.Series:
    """
    McClellan Oscillator — single-instrument A/D proxy.
    ad = sign(close.diff())  (+1 up bar, -1 down bar, 0 flat)
    McClellanOsc = EMA(ad, fast) - EMA(ad, slow)
    Source: mcclellan_osc.py:compute_mcclellan
    """
    delta    = close.diff()
    ad       = np.sign(delta).fillna(0)
    ema_fast = ad.ewm(span=fast, adjust=False).mean()
    ema_slow = ad.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).astype(np.float32)


# ── Block populator ────────────────────────────────────────────────────────────

def populate_fuzzy_block(df: pd.DataFrame,
                         skip_mcclellan: bool = False) -> pd.DataFrame:
    """Compute all fuzzy-block columns and return updated df."""
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    df["kappa_proxy"]        = compute_kappa_proxy(close)
    df["chaos_membership"]   = compute_chaos_membership(close)
    df["position_size_mult"] = compute_position_size_mult(close)

    cs, bs, sr              = compute_reversal_stack(high, low, close)
    df["consolidation_score"] = cs
    df["breakout_score"]      = bs
    df["spread_ratio"]        = sr

    df["friction_ratio"]     = compute_friction_ratio(high, low, close)
    df["fuzzy_reversion_11"] = compute_fuzzy_reversion_11(close, high, low, volume)

    if not skip_mcclellan:
        df["McClellanOsc"] = compute_mcclellan(close)

    return df


# ── File processor ─────────────────────────────────────────────────────────────

def process_file(path: Path,
                 force_all_targets: bool = False,
                 skip_mcclellan: bool = False,
                 logger: logging.Logger = None) -> Dict:
    log = logger.info if logger else print

    df = pd.read_csv(path, low_memory=False)

    # Determine which target cols to compute
    active_targets = [c for c in TARGET_COLS
                      if not (skip_mcclellan and c == "McClellanOsc")]

    # Check which need computing
    needs_compute = []
    for col in active_targets:
        if col not in df.columns:
            needs_compute.append(col)
        elif force_all_targets or _is_unpopulated(df[col]):
            needs_compute.append(col)

    if not needs_compute:
        log("    all %d target cols already populated — skipping", len(active_targets))
        return {"file": str(path), "rows": len(df), "touched": [], "skipped": active_targets}

    # Compute the full block (always compute together for consistency)
    computed = populate_fuzzy_block(df.copy(), skip_mcclellan=skip_mcclellan)

    touched: List[str] = []
    skipped: List[str] = []

    for col in active_targets:
        if col not in computed.columns:
            skipped.append(f"{col}:not_computed")
            continue
        if col in needs_compute:
            df[col] = computed[col]
            touched.append(col)
        else:
            skipped.append(f"{col}:already_populated")

    df.to_csv(path, index=False)

    log("    touched=%d skipped=%d rows=%d", len(touched), len(skipped), len(df))

    # Per-column stats for summary
    stats: Dict = {}
    for col in touched:
        s = df[col]
        entry: Dict = {"nan_pct": round(float(s.isna().mean()), 6)}
        if pd.api.types.is_numeric_dtype(s):
            entry["min"] = float(s.min()) if not s.isna().all() else None
            entry["max"] = float(s.max()) if not s.isna().all() else None
        stats[col] = entry

    return {"file": str(path), "rows": len(df), "touched": touched, "skipped": skipped, "stats": stats}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Populate fuzzy/regime feature block (Script 4) for 2020-2024 v43 datasets."
    )
    ap.add_argument(
        "--shells-dir",
        default=None,
        help="Directory containing {year}/ subdirs. Default: {base-dir}/normalized_shells",
    )
    ap.add_argument(
        "--base-dir",
        default=str(_PROJECT_ROOT / "data" / "Datasetv4" / "v43"),
    )
    ap.add_argument("--years",      nargs="+", type=int, default=YEARS)
    ap.add_argument("--timeframes", nargs="+",           default=TIMEFRAMES)
    ap.add_argument(
        "--force-all-targets",
        action="store_true",
        help="Recompute all target columns even if already populated.",
    )
    ap.add_argument(
        "--skip-mcclellan",
        action="store_true",
        help="Leave McClellanOsc untouched (single-instrument proxy not desired).",
    )
    ap.add_argument(
        "--log",
        default=str(_PROJECT_ROOT / "logs" / "populate_fuzzy_features_v43.log"),
    )
    ap.add_argument(
        "--summary-json",
        default=str(_PROJECT_ROOT / "reports" / "population_audit"
                    / "populate_fuzzy_features_v43_summary.json"),
    )
    args = ap.parse_args()

    base_dir   = Path(args.base_dir)
    shells_dir = Path(args.shells_dir) if args.shells_dir else base_dir / "normalized_shells"
    log_path   = Path(args.log)

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info("populate_fuzzy_features_v43.py")
    logger.info("shells_dir      : %s", shells_dir)
    logger.info("years           : %s", args.years)
    logger.info("timeframes      : %s", args.timeframes)
    logger.info("force           : %s", args.force_all_targets)
    logger.info("skip_mcclellan  : %s", args.skip_mcclellan)
    logger.info("=" * 60)

    results: List[Dict] = []

    for year in args.years:
        for tf in args.timeframes:
            path = shells_dir / str(year) / f"{tf}_dataset_v43_{year}.csv"
            if not path.exists():
                logger.warning("Missing: %s", path)
                continue
            logger.info("  Processing: %s", path)
            result = process_file(
                path,
                force_all_targets=args.force_all_targets,
                skip_mcclellan=args.skip_mcclellan,
                logger=logger,
            )
            results.append(result)

    logger.info("=" * 60)
    logger.info("Done. Files processed: %d", len(results))

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Summary JSON: %s", summary_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
