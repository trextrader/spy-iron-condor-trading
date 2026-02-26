#!/usr/bin/env python3
"""data_pipeline_v43.py -- CondorNet v4.3 Offline ETL Pipeline
=============================================================

Enriches the 4 TF dataset CSVs with:

  A) ETL Input Features (added to all 4 TFs):
       - tod_sin, tod_cos           -- circular time-of-day encoding
       - friction_ok_5/10/20/40/60  -- binary bid-ask vs rolling range gate
       - regime_persistence         -- consecutive bars in same volxtrend bucket
       - 9 IVR reversal features    -- price_stretch, ivr_zone, stretch_zone,
                                      reversal_score, strong_pivot_slope_h/l,
                                      strong_pivot_bar_dist_h/l,
                                      bars_since_strong_pivot

  B) Multi-task Training Labels (M5 only, all normalized):
       - strategy_label  -- rule-based IC/abstain classification (int 0-9)
       - pop             -- BSM Probability of Profit (0.15-delta IC, 0-DTE)
       - ev              -- Expected Value / close
       - max_loss        -- (spread_width - credit) / close
       - var_95          -- 95% Value at Risk / close
       - cvar_95         -- 95% CVaR / close

  C) Normalized target_spot:
       Replaces raw SPY close (~540) with 5-bar fwd log return x 100 (~ +/-3).
       This prevents spot_mse from dominating training loss.

Usage (run BEFORE condor_train_net_v43.py):
  python intelligence/data_pipeline_v43.py \\
      --data-dir data/Datasetv4/v43/      ? default
      --output-suffix _etl               ? saves as *_etl.csv (omit for in-place)

  In-place overwrite creates .csv.bak backups automatically.

Flags:
  --force          Re-compute ETL features even if already present
  --dry-run        Print diagnostics without writing files
  --fwd-bars N     Forward bars for target_spot (default 5 -> 25 min on M5)
  --target-delta F Short strike delta for IC PoP (default 0.15)
  --dte F          Days to expiry for IC labels (default 1.0 -> 0-DTE)
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# ?? Schema constants ?????????????????????????????????????????????????????????
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from intelligence.schema_v43 import (
        FRICTION_CANDIDATE_WINDOWS,
        IVR_REVERSAL_CONFIG,
        IVR_REVERSAL_FEATURE_NAMES,
        FRICTION_FEATURE_NAMES,
        TOD_FEATURE_NAMES,
        REGIME_PERSISTENCE_FEATURE,
        ABSTAIN_IDX,
        STRATEGY_TYPE_TO_IDX,
    )
    IRON_CONDOR_IDX = STRATEGY_TYPE_TO_IDX.get("iron_condor", 7)
except ImportError:
    FRICTION_CANDIDATE_WINDOWS = [5, 10, 20, 40, 60]
    IVR_REVERSAL_FEATURE_NAMES = [
        "price_stretch", "ivr_zone", "stretch_zone", "reversal_score",
        "strong_pivot_slope_h", "strong_pivot_slope_l",
        "strong_pivot_bar_dist_h", "strong_pivot_bar_dist_l",
        "bars_since_strong_pivot",
    ]
    FRICTION_FEATURE_NAMES = [f"friction_ok_{n}" for n in FRICTION_CANDIDATE_WINDOWS]
    TOD_FEATURE_NAMES          = ["tod_sin", "tod_cos"]
    REGIME_PERSISTENCE_FEATURE = "regime_persistence"
    ABSTAIN_IDX     = 9
    IRON_CONDOR_IDX = 7
    IVR_REVERSAL_CONFIG = {
        "ivr_lookback": 252, "ivr_high_threshold": 0.70, "ivr_low_threshold": 0.30,
        "ma_length": 20, "atr_length": 14, "stretch_k": 2.0,
        "alpha_ivr": 1.0, "beta_stretch": 1.0, "gamma_pivot": 1.0,
        "score_threshold": 2.0, "confirm_bars": 5,
        "strong_left": 69, "strong_right": 69,
        "medium_left": 30, "medium_right": 32,
    }

# Label columns produced by this pipeline
PIPELINE_LABEL_NAMES: List[str] = [
    "strategy_label",
    "pop",
    "ev",
    "max_loss",
    "var_95",
    "cvar_95",
]

# Trading session constants
TOD_OPEN_MINUTE   = 9 * 60 + 30   # 570 minutes since midnight
TOD_TRADING_MINUTES = 390          # 9:30 -> 16:00
EPS = 1e-8


# =============================================================================
# A) ETL FEATURE COMPUTATION
# =============================================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True-range ATR. Falls back to atr_pct x close if OHLCV absent."""
    if all(c in df.columns for c in ("high", "low", "close")):
        prev_c = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
    if "atr_pct" in df.columns and "close" in df.columns:
        return df["atr_pct"] * df["close"]
    return pd.Series(np.ones(len(df), dtype=np.float64), index=df.index)


def compute_tod_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tod_sin, tod_cos from timestamp column."""
    ts          = pd.to_datetime(df["timestamp"])
    minute_od   = (ts.dt.hour * 60 + ts.dt.minute - TOD_OPEN_MINUTE).clip(0, TOD_TRADING_MINUTES)
    fraction    = minute_od / TOD_TRADING_MINUTES
    df["tod_sin"] = np.sin(2 * np.pi * fraction).astype(np.float32)
    df["tod_cos"] = np.cos(2 * np.pi * fraction).astype(np.float32)
    return df


def compute_friction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add friction_ok_N columns (binary: bid-ask spread < rolling N-bar range).

    bid_ask_spread ~ spread_ratio x close (if available), else bar range x 0.1.
    """
    if "spread_ratio" in df.columns and "close" in df.columns:
        bid_ask = df["spread_ratio"] * df["close"]
    elif all(c in df.columns for c in ("high", "low")):
        bid_ask = (df["high"] - df["low"]) * 0.10
    else:
        bid_ask = pd.Series(np.zeros(len(df)), index=df.index)

    for N in FRICTION_CANDIDATE_WINDOWS:
        roll_range = (
            df["high"].rolling(N, min_periods=1).max()
            - df["low"].rolling(N, min_periods=1).min()
        ).clip(lower=EPS)
        df[f"friction_ok_{N}"] = (bid_ask < roll_range).astype(np.int8)
    return df


def compute_regime_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime_persistence: count of consecutive bars in the same
    (vol_bucket, trend_bucket) pair.

    vol_bucket : 0=low (vol_ewma <= 0.003), 1=mid, 2=high (vol_ewma > 0.007)
    trend_bucket: psar_trend value (-1, 0, 1); falls back to sign(breakout_score).
    """
    if "vol_ewma" in df.columns:
        v = df["vol_ewma"].fillna(0)
        vol_bucket = pd.cut(v, bins=[-np.inf, 0.003, 0.007, np.inf],
                            labels=[0, 1, 2]).astype(float)
    else:
        vol_bucket = pd.Series(np.zeros(len(df)), index=df.index)

    if "psar_trend" in df.columns:
        trend_bucket = df["psar_trend"].fillna(0).astype(int)
    elif "breakout_score" in df.columns:
        trend_bucket = np.sign(df["breakout_score"].fillna(0)).astype(int)
    else:
        trend_bucket = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    regime      = list(zip(vol_bucket.values, trend_bucket.values))
    persistence = np.ones(len(df), dtype=np.int32)
    for i in range(1, len(df)):
        persistence[i] = persistence[i - 1] + 1 if regime[i] == regime[i - 1] else 1
    df["regime_persistence"] = persistence
    return df


def compute_ivr_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 9 IVR Reversal Engine features.

    Uses fuzzy_reversion_11 (bounded [0,1]) as the IVR proxy since actual
    per-bar implied-vol-rank is not stored separately in the TF CSVs.
    """
    cfg = IVR_REVERSAL_CONFIG
    N   = len(df)

    sma20 = df["close"].rolling(cfg["ma_length"], min_periods=1).mean()
    atr14 = _compute_atr(df, cfg["atr_length"])

    # ?? price_stretch ?????????????????????????????????????????????????????
    price_stretch = ((df["close"] - sma20) / (atr14 + EPS)).clip(-10.0, 10.0).fillna(0.0)
    df["price_stretch"] = price_stretch.astype(np.float32)

    # ?? IVR proxy ?????????????????????????????????????????????????????????
    if "fuzzy_reversion_11" in df.columns:
        ivr = df["fuzzy_reversion_11"].fillna(0.5).clip(0.0, 1.0)
    elif "vol_ewma" in df.columns:
        v   = df["vol_ewma"].fillna(0)
        lo  = v.rolling(cfg["ivr_lookback"], min_periods=20).min()
        hi  = v.rolling(cfg["ivr_lookback"], min_periods=20).max()
        ivr = ((v - lo) / (hi - lo + EPS)).clip(0.0, 1.0)
    else:
        ivr = pd.Series(np.full(N, 0.5), index=df.index)

    # ?? ivr_zone (ternary) ????????????????????????????????????????????????
    ivr_zone = np.zeros(N, dtype=np.int8)
    ivr_zone[ivr.values > cfg["ivr_high_threshold"]] =  1
    ivr_zone[ivr.values < cfg["ivr_low_threshold"]]  = -1
    df["ivr_zone"] = ivr_zone

    # ?? stretch_zone (ternary) ????????????????????????????????????????????
    k = cfg["stretch_k"]
    stretch_zone = np.zeros(N, dtype=np.int8)
    stretch_zone[price_stretch.values >  k] =  1
    stretch_zone[price_stretch.values < -k] = -1
    df["stretch_zone"] = stretch_zone

    # ?? Pivot events ?????????????????????????????????????????????????????
    has_ph = ~df["PivotHigh"].isna() if "PivotHigh" in df.columns else pd.Series(False, index=df.index)
    has_pl = ~df["PivotLow"].isna()  if "PivotLow"  in df.columns else pd.Series(False, index=df.index)

    # Strong pivots: pivot price deviates > 2 x ATR14 from SMA20
    if "PivotHigh" in df.columns:
        strong_h = has_ph & ((df["PivotHigh"] - sma20).abs() > 2 * atr14)
    else:
        strong_h = pd.Series(False, index=df.index)

    if "PivotLow" in df.columns:
        strong_l = has_pl & ((df["PivotLow"] - sma20).abs() > 2 * atr14)
    else:
        strong_l = pd.Series(False, index=df.index)

    # ?? pivot_strength x direction (running carry-forward) ???????????????
    pivot_sd   = np.zeros(N, dtype=np.float32)
    last_dir   = 0.0
    sh_arr     = strong_h.values
    sl_arr     = strong_l.values
    ph_arr     = has_ph.values
    pl_arr     = has_pl.values
    for i in range(N):
        if sh_arr[i]:
            last_dir = -3.0              # strong high -> bearish
        elif sl_arr[i]:
            last_dir = 3.0               # strong low  -> bullish
        elif ph_arr[i]:
            last_dir = max(-2.0, last_dir - 1.0)
        elif pl_arr[i]:
            last_dir = min( 2.0, last_dir + 1.0)
        pivot_sd[i] = last_dir

    # ?? reversal_score ????????????????????????????????????????????????????
    rev_score = (
        cfg["alpha_ivr"]    * ivr.values
        + cfg["beta_stretch"] * price_stretch.values
        + cfg["gamma_pivot"]  * pivot_sd
    ).clip(-10.0, 10.0)
    df["reversal_score"] = rev_score.astype(np.float32)

    # ?? Strong pivot slope / bar-distance (filled forward post 2nd event) ?
    sph = np.full(N, np.nan, dtype=np.float64)   # strong_pivot_slope_h
    spl = np.full(N, np.nan, dtype=np.float64)   # strong_pivot_slope_l
    bdh = np.full(N, np.nan, dtype=np.float64)   # strong_pivot_bar_dist_h
    bdl = np.full(N, np.nan, dtype=np.float64)   # strong_pivot_bar_dist_l
    bsp = np.full(N, np.nan, dtype=np.float64)   # bars_since_strong_pivot

    ph_price = df["PivotHigh"].values if "PivotHigh" in df.columns else np.full(N, np.nan)
    pl_price = df["PivotLow"].values  if "PivotLow"  in df.columns else np.full(N, np.nan)
    close_arr = df["close"].values

    sh_hist: List[Tuple[int, float]] = []
    sl_hist: List[Tuple[int, float]] = []
    last_sp_bar: Optional[int] = None

    for i in range(N):
        if sh_arr[i]:
            price = ph_price[i] if not np.isnan(ph_price[i]) else close_arr[i]
            sh_hist.append((i, price))
            last_sp_bar = i
            if len(sh_hist) >= 2:
                b1, p1 = sh_hist[-2]; b2, p2 = sh_hist[-1]
                bd = max(b2 - b1, 1)
                sph[i] = (p2 - p1) / bd
                bdh[i] = float(bd)

        if sl_arr[i]:
            price = pl_price[i] if not np.isnan(pl_price[i]) else close_arr[i]
            sl_hist.append((i, price))
            last_sp_bar = i
            if len(sl_hist) >= 2:
                b1, p1 = sl_hist[-2]; b2, p2 = sl_hist[-1]
                bd = max(b2 - b1, 1)
                spl[i] = (p2 - p1) / bd
                bdl[i] = float(bd)

        # Forward-fill slope / distance after the 2nd pivot is confirmed
        if i > 0:
            if np.isnan(sph[i]) and not np.isnan(sph[i - 1]): sph[i] = sph[i - 1]
            if np.isnan(spl[i]) and not np.isnan(spl[i - 1]): spl[i] = spl[i - 1]
            if np.isnan(bdh[i]) and not np.isnan(bdh[i - 1]): bdh[i] = bdh[i - 1]
            if np.isnan(bdl[i]) and not np.isnan(bdl[i - 1]): bdl[i] = bdl[i - 1]

        if last_sp_bar is not None:
            bsp[i] = float(i - last_sp_bar)

    df["strong_pivot_slope_h"]    = sph
    df["strong_pivot_slope_l"]    = spl
    df["strong_pivot_bar_dist_h"] = bdh
    df["strong_pivot_bar_dist_l"] = bdl
    df["bars_since_strong_pivot"] = bsp
    return df


def compute_etl_features(df: pd.DataFrame, tf_name: str = "m5") -> pd.DataFrame:
    """Run all ETL feature computations on a TF dataframe."""
    print(f"  [ETL:{tf_name}] TOD features...")
    df = compute_tod_features(df)
    print(f"  [ETL:{tf_name}] Friction gates...")
    df = compute_friction_features(df)
    print(f"  [ETL:{tf_name}] Regime persistence...")
    df = compute_regime_persistence(df)
    print(f"  [ETL:{tf_name}] IVR reversal features...")
    df = compute_ivr_reversal_features(df)
    return df


# =============================================================================
# B) MULTI-TASK LABEL COMPUTATION  (M5 only)
# =============================================================================

# ---------------------------------------------------------------------------
# B0) Multi-strategy label helper — called from compute_multitask_labels()
# ---------------------------------------------------------------------------

def _compute_strategy_labels(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign the best-fit strategy label per bar using the full strategy catalog.

    Reads pop / ev / max_loss from df (already normalized by close).
    For each bar:
      1. Evaluate eligibility mask for every template in STRATEGY_CATALOG.
      2. Compute a composite score per eligible template.
      3. Select the template with the highest score; abstain if score < cutoff
         or no template is eligible.
      4. Adjust pop / ev / max_loss using the winning template's family multipliers.

    Composite score (quality gate = PIPELINE_SCORE_CUTOFF = 0.15):
        score = 0.40*pop_adj + 0.35*tanh(3*ev_adj) - 0.15*tanh(2*ml_adj)
                + 0.10*dte_affinity

    Returns:
        strategy_label : np.ndarray[int8,  N]
        pop_adj        : np.ndarray[float32, N]
        ev_adj         : np.ndarray[float32, N]
        ml_adj         : np.ndarray[float32, N]
    """
    # Pipeline-local score floor calibrated to normalized ev/close ≈ 1e-4.
    # The schema ABSTAIN_LABEL_SCORE_CUTOFF (0.40) targets a richer scorer;
    # here the quality gate is provided by the PREDICATE CONDITIONS themselves —
    # if a bar satisfies no template's predicates it becomes abstain organically.
    # This floor (0.15) only eliminates degenerate/negative scores.
    PIPELINE_SCORE_CUTOFF = 0.15

    try:
        from intelligence.strategy_templates_v43 import (
            compute_predicate_atoms, STRATEGY_CATALOG,
        )
        from intelligence.schema_v43 import (
            STRATEGY_TYPE_TO_IDX, ABSTAIN_IDX, get_dte_affinity,
        )
    except ImportError:
        # Fallback: IC-only (preserves v4.3 behaviour if templates missing)
        import warnings
        warnings.warn(
            "[LBL] strategy_templates_v43 not found — falling back to IC-only labels",
            RuntimeWarning,
        )
        N = len(df)
        ic_mask = np.ones(N, dtype=bool)
        if "adx_adaptive"  in df.columns:
            ic_mask &= df["adx_adaptive"].fillna(50).values < 25.0
        if "breakout_score" in df.columns:
            ic_mask &= df["breakout_score"].fillna(0).values == 0
        if "rsi_dyn" in df.columns:
            rsi = df["rsi_dyn"].fillna(50).values
            ic_mask &= (rsi >= 35.0) & (rsi <= 65.0)
        lbl = np.full(N, 9, dtype=np.int8)   # 9 = abstain
        lbl[ic_mask] = 7                      # 7 = iron_condor
        pop = df["pop"].values.astype(np.float32)
        ev  = df["ev"].values.astype(np.float32)
        ml  = df["max_loss"].values.astype(np.float32)
        return lbl, pop, ev, ml

    N      = len(df)
    catalog = STRATEGY_CATALOG
    K       = len(catalog)

    # Pre-read base economics (already normalised by close)
    base_pop = df["pop"].values.astype(np.float64)
    base_ev  = df["ev"].values.astype(np.float64)
    base_ml  = df["max_loss"].values.astype(np.float64)

    # Map template → integer class index (for output array)
    class_ids = np.array(
        [STRATEGY_TYPE_TO_IDX.get(t.v43_class, ABSTAIN_IDX) for t in catalog],
        dtype=np.int8,
    )

    # DTE affinity scalar per template (constant for this pipeline call)
    # Pipeline default: dte_days=1.0 → DTE bucket "0-2"
    dte_aff = np.array(
        [get_dte_affinity(t.v43_class, 1) for t in catalog],
        dtype=np.float64,
    )

    # Predicate atoms (one vectorized pass over entire df)
    atoms = compute_predicate_atoms(df)

    # Build [N, K] score matrix
    NEG_INF = -1e9
    scores  = np.full((N, K), NEG_INF, dtype=np.float64)

    for k, template in enumerate(catalog):
        mask = template.get_eligibility_mask(atoms)   # bool[N]
        if not mask.any():
            continue

        # Adjusted economics for this template
        pop_k = np.clip(
            base_pop * template.pop_scale + template.pop_offset, 0.01, 0.99
        )
        ev_k  = base_ev * template.ev_scale
        ml_k  = base_ml * template.max_loss_scale

        score = (
            0.40 * pop_k
            + 0.35 * np.tanh(3.0 * ev_k)
            - 0.15 * np.tanh(2.0 * ml_k)
            + 0.10 * dte_aff[k]
        )
        scores[mask, k] = score[mask]

    # Best template per bar
    best_k     = np.argmax(scores, axis=1)        # [N]
    best_score = scores[np.arange(N), best_k]     # [N]

    # Abstain where no template eligible or composite score too low
    no_eligible  = (scores == NEG_INF).all(axis=1)
    low_score    = best_score < PIPELINE_SCORE_CUTOFF
    abstain_mask = no_eligible | low_score

    strategy_label = class_ids[best_k].copy()
    strategy_label[abstain_mask] = np.int8(ABSTAIN_IDX)

    # Build adjusted pop/ev/max_loss arrays using the winning template
    pop_out = base_pop.copy()
    ev_out  = base_ev.copy()
    ml_out  = base_ml.copy()

    for k, template in enumerate(catalog):
        win = (best_k == k) & ~abstain_mask
        if not win.any():
            continue
        pop_out[win] = np.clip(
            base_pop[win] * template.pop_scale + template.pop_offset, 0.01, 0.99
        )
        ev_out[win]  = base_ev[win]  * template.ev_scale
        ml_out[win]  = base_ml[win]  * template.max_loss_scale

    return (
        strategy_label.astype(np.int8),
        pop_out.astype(np.float32),
        ev_out.astype(np.float32),
        ml_out.astype(np.float32),
    )

def _bsm_pop_ic(sigma_annual: np.ndarray, T: float, target_delta: float) -> np.ndarray:
    """
    Black-Scholes Probability of Profit for a symmetric zero-DTE iron condor.

    Short strikes chosen so that N(d2) = target_delta on each side.

    PoP = P(K_put < S_T < K_call)
        = 2 x N(z_? - 0.5 x sigma x sqrtT)  -  1
    where z_? = N??(1 - target_delta) ~ 1.036 for ?=0.15.

    Returns probabilities clipped to [0.01, 0.99].
    """
    z_delta   = norm.ppf(1.0 - target_delta)     # ~ 1.036 for delta=0.15
    sigma_T   = sigma_annual * np.sqrt(T) + EPS  # avoid log(0)
    pop       = 2.0 * norm.cdf(z_delta - 0.5 * sigma_T) - 1.0
    return np.clip(pop, 0.01, 0.99)


def compute_multitask_labels(
    df: pd.DataFrame,
    fwd_bars:        int            = 5,
    target_delta:    float          = 0.15,
    dte_days:        float          = 1.0,
    spread_atr_mult: float          = 2.0,
    credit_fraction: float          = 0.35,
    options_path:    Optional[str]  = None,
) -> pd.DataFrame:
    """
    Compute all multi-task training labels for M5 bars.

    Parameters
    ----------
    options_path : path to the options CSV (options_YYYY_v43.csv).
        When provided, steps 3+4 are overridden with real market IC economics
        (delta-based PoP, actual bid/ask credit, real max-loss) for every
        bar whose date appears in the options file.  Bars with no matching
        date fall back to the BSM / ATR-based values.

    Labels added / updated:
      target_spot    -> 5-bar fwd log return x 100 (replaces raw SPY price)
      strategy_label -> rule-based IC/abstain class (int 0-9)
      pop            -> BSM PoP for 0.15-delta IC at dte_days DTE
      ev             -> Expected Value / close
      max_loss       -> (spread_width - credit) / close
      var_95         -> 95% VaR / close (IC payoff at 5th-pctile return)
      cvar_95        -> 95% CVaR / close  ~ var_95 x 1.3

    All risk metrics are normalized by close so they are scale-invariant
    across different SPY price levels.
    """
    N     = len(df)
    close = df["close"].values.astype(np.float64)

    # ?? 1. Normalize target_spot ???????????????????????????????????????????
    fwd_close   = df["close"].shift(-fwd_bars).values.astype(np.float64)
    valid       = np.isfinite(fwd_close) & (fwd_close > 0) & (close > 0)
    log_ret_fwd = np.where(valid, np.log(fwd_close / close) * 100.0, 0.0)
    df["target_spot"] = log_ret_fwd.astype(np.float32)

    # ?? 2. Annualized daily vol proxy ??????????????????????????????????????
    if "vol_ewma" in df.columns:
        sigma_daily = df["vol_ewma"].fillna(0.005).values.astype(np.float64).clip(1e-5)
    else:
        sigma_daily = df["atr_pct"].fillna(0.005).values.astype(np.float64).clip(1e-5)
    sigma_annual = sigma_daily * np.sqrt(252)

    # ?? 3. Black-Scholes PoP ??????????????????????????????????????????????
    T   = dte_days / 252.0
    pop = _bsm_pop_ic(sigma_annual, T, target_delta)
    df["pop"] = pop.astype(np.float32)

    # ?? 4. Spread economics (ATR-based) ???????????????????????????????????
    atr14        = _compute_atr(df, 14).values.astype(np.float64)
    spread_width = atr14 * spread_atr_mult                              # e.g., 2 x ATR
    credit       = spread_width * credit_fraction                        # e.g., 35% of width
    max_loss     = np.maximum(spread_width - credit, EPS)               # width - credit

    df["max_loss"] = (max_loss / (close + EPS)).astype(np.float32)
    df["ev"]       = ((pop * credit - (1.0 - pop) * max_loss) / (close + EPS)).astype(np.float32)

    # ?? 4b. Override with real market IC economics if options_path provided ???
    if options_path is not None:
        try:
            from intelligence.options_features_v43 import build_options_daily_summary  # noqa
            opt_sum = build_options_daily_summary(str(options_path), verbose=True)

            # Build date-string series aligned to df rows
            ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
            bar_dates = pd.to_datetime(df[ts_col], errors="coerce").dt.strftime("%Y-%m-%d")

            # Map per-day real values onto every bar of that day
            pop_real    = bar_dates.map(opt_sum["ic_pop_real"]).values.astype(np.float64)
            credit_real = bar_dates.map(opt_sum["ic_credit_raw"]).values.astype(np.float64)
            ml_real     = bar_dates.map(opt_sum["ic_max_loss_raw"]).values.astype(np.float64)

            real_mask = np.isfinite(pop_real) & np.isfinite(credit_real) & np.isfinite(ml_real)
            n_real    = int(real_mask.sum())

            if n_real > 0:
                pop_arr   = df["pop"].values.astype(np.float64)
                ml_arr    = df["max_loss"].values.astype(np.float64)
                ev_arr    = df["ev"].values.astype(np.float64)

                pop_arr[real_mask] = pop_real[real_mask]
                ml_arr[real_mask]  = ml_real[real_mask] / (close[real_mask] + EPS)
                ev_arr[real_mask]  = (
                    pop_real[real_mask] * credit_real[real_mask]
                    - (1.0 - pop_real[real_mask]) * ml_real[real_mask]
                ) / (close[real_mask] + EPS)

                df["pop"]      = pop_arr.astype(np.float32)
                df["max_loss"] = ml_arr.astype(np.float32)
                df["ev"]       = ev_arr.astype(np.float32)

                print(f"  [OPT] Real options economics applied to "
                      f"{n_real:,}/{len(df):,} bars ({n_real/len(df)*100:.1f}%)")
                print(f"  [OPT] pop   mean={pop_arr[real_mask].mean():.4f}  "
                      f"std={pop_arr[real_mask].std():.4f}")
                print(f"  [OPT] ev    mean={ev_arr[real_mask].mean():.6f}  "
                      f"std={ev_arr[real_mask].std():.6f}")
            else:
                print("  [OPT] WARN: No bar dates matched options data -- using BSM fallback")

        except Exception as _opt_exc:
            print(f"  [OPT] WARN: options load failed ({_opt_exc}) -- using BSM fallback")

    # ?? 5. VaR & CVaR (lognormal IC payoff at 5th percentile) ????????????
    sigma_T      = sigma_annual * np.sqrt(T)
    z_delta      = norm.ppf(1.0 - target_delta)
    strike_dist  = z_delta * sigma_T           # distance to short strike in sigma_T units
    ret_05       = norm.ppf(0.05) * sigma_T    # 5th-pctile log return (negative)

    abs_ret = np.abs(ret_05)
    loss_frac = np.where(
        abs_ret <= strike_dist,
        0.0,
        np.minimum((abs_ret - strike_dist) / (strike_dist + EPS), 1.0),
    )
    var_95 = loss_frac * max_loss / (close + EPS)
    df["var_95"]  = var_95.astype(np.float32)
    df["cvar_95"] = (var_95 * 1.3).astype(np.float32)

    # ?? 6. Multi-strategy label (replaces IC-only heuristic) ??????????????
    strategy_label, pop_adj, ev_adj, ml_adj = _compute_strategy_labels(df)
    df["strategy_label"] = strategy_label.astype(np.int8)
    df["pop"]      = pop_adj.astype(np.float32)
    df["ev"]       = ev_adj.astype(np.float32)
    df["max_loss"] = ml_adj.astype(np.float32)
    # var_95 / cvar_95 are risk-environment measures; kept at IC-baseline values.

    # ?? Diagnostics ???????????????????????????????????????????????????????
    from collections import Counter
    label_counts = Counter(strategy_label.tolist())
    try:
        from intelligence.schema_v43 import IDX_TO_STRATEGY_TYPE
    except ImportError:
        IDX_TO_STRATEGY_TYPE = {}
    print(f"  [LBL] strategy_label distribution ({N} bars):")
    for idx in sorted(label_counts):
        name = IDX_TO_STRATEGY_TYPE.get(idx, f"class_{idx}")
        pct  = label_counts[idx] / N * 100.0
        print(f"         {idx}: {name:20s}  {label_counts[idx]:5d} ({pct:5.1f}%)")
    print(f"  [LBL] pop   -- mean={df['pop'].mean():.4f}  std={df['pop'].std():.4f}")
    print(f"  [LBL] ev    -- mean={df['ev'].mean():.6f}  std={df['ev'].std():.6f}")
    print(f"  [LBL] target_spot (log ret x100) -- "
          f"mean={log_ret_fwd.mean():.4f}  std={log_ret_fwd.std():.4f}")
    return df


# =============================================================================
# C) PIPELINE ORCHESTRATION
# =============================================================================

def _etl_present(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ("tod_sin", "friction_ok_5",
                                         "regime_persistence", "price_stretch"))


def _labels_present(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in PIPELINE_LABEL_NAMES)


def process_tf(
    input_path:     Path,
    output_path:    Path,
    tf_name:        str,
    compute_labels: bool            = False,
    label_kwargs:   Optional[Dict]  = None,
    force:          bool            = False,
    dry_run:        bool            = False,
) -> pd.DataFrame:
    """Load, enrich, and save one TF CSV."""
    print(f"\n{'='*60}")
    print(f"[PIPELINE] {tf_name.upper()}: {input_path.name}")

    df = pd.read_csv(str(input_path), low_memory=False)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")

    # ?? ETL features ??????????????????????????????????????????????????????
    if _etl_present(df) and not force:
        print(f"  OK ETL features already present -- skipping (--force to recompute)")
    else:
        df = compute_etl_features(df, tf_name=tf_name)
        new_etl = [c for c in (
            TOD_FEATURE_NAMES
            + [f"friction_ok_{n}" for n in FRICTION_CANDIDATE_WINDOWS]
            + [REGIME_PERSISTENCE_FEATURE]
            + IVR_REVERSAL_FEATURE_NAMES
        ) if c in df.columns]
        print(f"  + Added {len(new_etl)} ETL features")

    # ?? Multi-task labels (M5 only) ???????????????????????????????????????
    if compute_labels:
        if _labels_present(df) and not force:
            print(f"  OK Labels already present -- skipping (--force to recompute)")
        else:
            df = compute_multitask_labels(df, **(label_kwargs or {}))
            print(f"  + Added {len(PIPELINE_LABEL_NAMES)} label columns")

    if dry_run:
        print(f"  [DRY RUN] Would write {len(df.columns)} cols -> {output_path.name}")
        return df

    # ?? Write output ??????????????????????????????????????????????????????
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == input_path.resolve() and input_path.exists():
        bak = input_path.with_suffix(".csv.bak")
        shutil.copy2(str(input_path), str(bak))
        print(f"  Backup -> {bak.name}")

    df.to_csv(str(output_path), index=False)
    print(f"  Saved  {len(df):,} rows x {len(df.columns)} cols -> {output_path.name}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="CondorNet v4.3 ETL Pipeline")
    p.add_argument("--data-dir",      type=str, default="data/Datasetv4/v43/",
                   help="Dataset directory (default: data/Datasetv4/v43/)")
    p.add_argument("--output-suffix", type=str, default="",
                   help="Suffix appended before .csv ('' = in-place overwrite with .bak backup)")
    p.add_argument("--force",    action="store_true",
                   help="Recompute ETL/labels even if already present")
    p.add_argument("--dry-run",  action="store_true",
                   help="Diagnostics only -- do not write any files")
    p.add_argument("--fwd-bars",     type=int,   default=5,
                   help="Forward bars for target_spot normalization (default 5 = 25 min on M5)")
    p.add_argument("--target-delta", type=float, default=0.15,
                   help="Short strike delta for IC PoP (default 0.15)")
    p.add_argument("--dte",          type=float, default=1.0,
                   help="Days to expiry assumption for IC labels (default 1.0 = 0-DTE)")
    p.add_argument("--options-path", type=str, default=None,
                   help="Path to options CSV (e.g. data/Datasetv4/v43/options_2025_v43.csv). "
                        "When provided, pop/ev/max_loss are derived from real market "
                        "bid/ask/delta data instead of BSM/ATR proxies.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    sfx      = args.output_suffix

    # (filename, compute_labels)
    tf_files: Dict[str, Tuple[str, bool]] = {
        "m1":  ("m1_dataset_v43_final.csv",  False),
        "m5":  ("m5_dataset_v43_final.csv",  True),    # M5 receives labels
        "m15": ("m15_dataset_v43_final.csv", False),
        "h1":  ("h1_dataset_v43_final.csv",  False),
    }

    label_kwargs = {
        "fwd_bars":      args.fwd_bars,
        "target_delta":  args.target_delta,
        "dte_days":      args.dte,
        "options_path":  args.options_path,   # None → BSM fallback
    }

    for tf_name, (filename, compute_labels) in tf_files.items():
        input_path  = data_dir / filename
        output_name = filename.replace(".csv", f"{sfx}.csv") if sfx else filename
        output_path = data_dir / output_name

        if not input_path.exists():
            print(f"\nWARN  {input_path} not found -- skipping {tf_name.upper()}")
            continue

        process_tf(
            input_path     = input_path,
            output_path    = output_path,
            tf_name        = tf_name,
            compute_labels = compute_labels,
            label_kwargs   = label_kwargs if compute_labels else None,
            force          = args.force,
            dry_run        = args.dry_run,
        )

    print("\n" + "="*60)
    print("[PIPELINE] Complete.")
    if sfx:
        print(f"\nRe-run training with enriched files:")
        for tf_name, (filename, _) in tf_files.items():
            enriched = filename.replace(".csv", f"{sfx}.csv")
            print(f"  --{tf_name}-file {enriched}")
    else:
        print("Files updated in-place. Run condor_train_net_v43.py normally.")


if __name__ == "__main__":
    main()
