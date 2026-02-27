"""
intelligence/options_features_v43.py
─────────────────────────────────────
Reads the raw SPY options CSV (daily snapshots, 21 standard columns) and
produces a per-trading-date summary of Iron Condor economics for the v4.3
data pipeline.

Input CSV expected columns (subset used):
    timestamp, expiration, strike, type,
    bid, ask, mark, implied_volatility, delta

Output DataFrame (index = date string "YYYY-MM-DD"):
    atm_iv          – IV of nearest ATM call (raw, annualised decimal)
    ic_pop_real     – delta-based IC PoP  (float 0-1)
    ic_credit_raw   – net IC credit in dollars per 1-contract spread
    ic_max_loss_raw – IC max loss  in dollars per 1-contract spread
    ic_ev_raw       – IC EV in dollars = pop*credit - (1-pop)*max_loss
    sc_strike       – short call strike used
    sp_strike       – short put  strike used
    lc_strike       – long  call strike used
    lp_strike       – long  put  strike used
    expiry_used     – expiration date actually used (string)
    dte_used        – DTE of expiry_used

Usage
-----
    from intelligence.options_features_v43 import build_options_daily_summary
    summary = build_options_daily_summary("data/Datasetv4/v43/options_2025_v43.csv")
    # summary.loc["2025-01-02"] -> Series with IC economics for that date
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── tuneable defaults (can be overridden via function args) ──────────────────
_DEFAULT_SHORT_DELTA = 0.175   # target delta magnitude for short legs
_DEFAULT_LONG_DELTA  = 0.10    # target delta magnitude for long (wing) legs
_DEFAULT_DTE_MAX     = 7       # prefer 0-DTE; fall back up to this DTE


# ── internal helpers ─────────────────────────────────────────────────────────

def _best_leg(chain: pd.DataFrame, target_delta: float) -> Optional[pd.Series]:
    """Row in *chain* whose |delta| is closest to *target_delta*. None if empty."""
    if chain.empty:
        return None
    idx = (chain["delta"].abs() - abs(target_delta)).abs().idxmin()
    return chain.loc[idx]


def _leg_price(row: pd.Series) -> float:
    """
    Best available price for a single leg.
    Priority: mark > mid(bid,ask) > penny floor.
    """
    mark = float(row.get("mark", 0) or 0)
    if mark > 0:
        return mark
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    mid = (bid + ask) / 2
    return mid if mid > 0 else 0.01   # penny floor for illiquid contracts


def _ic_economics(
    sc: pd.Series,
    sp: pd.Series,
    lc: pd.Series,
    lp: pd.Series,
) -> dict:
    """
    Compute IC economics from four leg rows.

    Convention (per standard retail IC):
        Short Call  – sell at sc price (credit)
        Short Put   – sell at sp price (credit)
        Long  Call  – buy  at lc price (debit)
        Long  Put   – buy  at lp price (debit)

    Returns dollar amounts per 1-contract spread (multiply by 100 for per-lot).
    """
    sc_px = _leg_price(sc)
    sp_px = _leg_price(sp)
    lc_px = _leg_price(lc)
    lp_px = _leg_price(lp)

    credit  = sc_px + sp_px - lc_px - lp_px    # net credit received

    c_width = float(lc["strike"]) - float(sc["strike"])  # call spread width
    p_width = float(sp["strike"]) - float(lp["strike"])  # put  spread width
    spread_w = max(c_width, p_width, 1.0)                # widest side (floor $1)

    # Max loss: worst-case loss on the widest leg minus credit received
    max_loss = spread_w - credit
    if max_loss <= 0:                   # degenerate: credit >= width → use width
        max_loss = spread_w

    # Delta-based probability of profit:
    #   P(IC profitable) ≈ P(stays below short call) + P(stays above short put) - 1
    #                     = (1 - sc_delta) + (1 - |sp_delta|) - 1
    #                     = 1 - sc_delta - |sp_delta|
    sc_d = abs(float(sc["delta"]))
    sp_d = abs(float(sp["delta"]))
    pop  = float(np.clip(1.0 - sc_d - sp_d, 0.01, 0.99))

    ev = pop * max(credit, 0.0) - (1.0 - pop) * max_loss

    return {
        "ic_credit_raw":   round(credit,   4),
        "ic_max_loss_raw": round(max_loss, 4),
        "ic_ev_raw":       round(ev,       4),
        "ic_pop_real":     round(pop,      6),
        "sc_strike":       float(sc["strike"]),
        "sp_strike":       float(sp["strike"]),
        "lc_strike":       float(lc["strike"]),
        "lp_strike":       float(lp["strike"]),
    }


# ── main API ─────────────────────────────────────────────────────────────────

def build_options_daily_summary(
    options_path:       str | Path,
    target_short_delta: float = _DEFAULT_SHORT_DELTA,
    target_long_delta:  float = _DEFAULT_LONG_DELTA,
    dte_max:            int   = _DEFAULT_DTE_MAX,
    verbose:            bool  = True,
) -> pd.DataFrame:
    """
    Load the full options CSV and return a DataFrame indexed by date string
    ("YYYY-MM-DD") with one row per trading day containing IC economics.

    Parameters
    ----------
    options_path        : path to the options CSV
    target_short_delta  : delta magnitude for short legs  (default 0.175)
    target_long_delta   : delta magnitude for long  legs  (default 0.10)
    dte_max             : max DTE considered (prefers 0; fallback ≤ dte_max)
    verbose             : print progress and summary stats

    Returns
    -------
    pd.DataFrame indexed by "YYYY-MM-DD" date string
    """
    options_path = Path(options_path)
    if not options_path.exists():
        raise FileNotFoundError(f"Options file not found: {options_path}")

    if verbose:
        print(f"\n[OPT] Loading {options_path.name} ...")

    opts = pd.read_csv(str(options_path), low_memory=False)

    # Explicit datetime conversion (pandas 2.x dropped implicit parse_dates)
    opts["timestamp"]  = pd.to_datetime(opts["timestamp"],  errors="coerce")
    opts["expiration"] = pd.to_datetime(opts["expiration"], errors="coerce")

    if verbose:
        n_dates = opts["timestamp"].dt.date.nunique()
        print(f"[OPT] Loaded {len(opts):,} rows across {n_dates} unique dates")

    # Compute DTE for each row
    opts["_date"] = opts["timestamp"].dt.date
    opts["_exp"]  = opts["expiration"].dt.date
    opts["_dte"]  = (opts["_exp"] - opts["_date"]).apply(lambda x: x.days)

    # Keep only near-term expirations (0-DTE up to dte_max)
    near = opts[(opts["_dte"] >= 0) & (opts["_dte"] <= dte_max)].copy()
    if verbose:
        print(f"[OPT] {len(near):,} rows with DTE 0-{dte_max} "
              f"across {near['_date'].nunique()} dates")

    rows   = []
    n_ok   = 0
    n_skip = 0

    for date, day in near.groupby("_date"):
        # Sort candidate expirations by DTE ascending (prefer 0-DTE)
        exp_dtes = (
            day[["_exp", "_dte"]]
            .drop_duplicates()
            .sort_values("_dte")
        )

        found = False
        for _, exp_row in exp_dtes.iterrows():
            exp  = exp_row["_exp"]
            dte  = int(exp_row["_dte"])
            snap = day[day["_exp"] == exp]

            calls = snap[snap["type"] == "call"].copy()
            puts  = snap[snap["type"] == "put"].copy()
            if calls.empty or puts.empty:
                continue

            # ATM IV: call with delta closest to 0.50
            atm_call = _best_leg(calls[calls["delta"] > 0], target_delta=0.50)
            atm_iv   = float(atm_call["implied_volatility"]) if atm_call is not None else np.nan

            # Short legs (nearest to target_short_delta)
            sc = _best_leg(calls[calls["delta"] > 0],   target_delta=target_short_delta)
            sp = _best_leg(puts[puts["delta"] < 0],     target_delta=target_short_delta)
            if sc is None or sp is None:
                continue

            # Long legs: must be further OTM than the short legs
            lc_pool = calls[(calls["delta"] > 0) & (calls["strike"] > sc["strike"])]
            lp_pool = puts[(puts["delta"] < 0)  & (puts["strike"] < sp["strike"])]
            lc = _best_leg(lc_pool, target_delta=target_long_delta)
            lp = _best_leg(lp_pool, target_delta=target_long_delta)
            if lc is None or lp is None:
                continue

            # All four legs found — compute economics
            econ = _ic_economics(sc, sp, lc, lp)
            rows.append({
                "date":        str(date),
                "atm_iv":      round(atm_iv, 6) if np.isfinite(atm_iv) else np.nan,
                "expiry_used": str(exp),
                "dte_used":    dte,
                **econ,
            })
            n_ok  += 1
            found  = True
            break   # found a good expiry for this date

        if not found:
            n_skip += 1

    if not rows:
        raise RuntimeError(
            "[OPT] No valid IC legs found. Check delta ranges and dte_max. "
            f"File: {options_path}"
        )

    summary = pd.DataFrame(rows).set_index("date")

    if verbose:
        print(f"\n[OPT] Summary: {n_ok} dates OK, {n_skip} skipped (no valid 4-leg IC)")
        print(f"[OPT] DTE distribution:\n"
              + summary["dte_used"].value_counts().sort_index().to_string(header=False))
        print(f"\n[OPT] atm_iv      mean={summary['atm_iv'].mean():.4f}  "
              f"std={summary['atm_iv'].std():.4f}")
        print(f"[OPT] ic_pop_real mean={summary['ic_pop_real'].mean():.4f}  "
              f"std={summary['ic_pop_real'].std():.4f}")
        print(f"[OPT] credit      mean=${summary['ic_credit_raw'].mean():.4f}  "
              f"std=${summary['ic_credit_raw'].std():.4f}")
        print(f"[OPT] max_loss    mean=${summary['ic_max_loss_raw'].mean():.4f}  "
              f"std=${summary['ic_max_loss_raw'].std():.4f}")
        print(f"[OPT] ic_ev_raw   mean=${summary['ic_ev_raw'].mean():.4f}  "
              f"std=${summary['ic_ev_raw'].std():.4f}")

    return summary
