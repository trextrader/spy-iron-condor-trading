"""
optimizer_engine.py — K-Candidate Vectorised Backtest Engine
============================================================
Runs K parameter candidates in a single bar-major loop.
State arrays [K] are updated in parallel via numpy broadcasting.

Iron Butterfly leg structure per candidate k:
    Short call + short put at short_strike (≈ ATM, closest to short_delta)
    Long  call at short_strike + spread_width
    Long  put  at short_strike - spread_width
    Net credit = (SC_bid + SP_bid) − (LC_ask + LP_ask)

Exit rules (per candidate, configurable):
    1. stop_loss_dollar  — close when loss >= N dollars
    2. profit_target     — close when gain >= N dollars  (None → 50% of credit × qty × 100)
    3. max_dte_exit      — close when DTE_remaining <= N (0 = let expire)
    4. hold_days         — close when calendar days held >= N

Simplification vs full backtester:
    • max 1 concurrent position per candidate (no capital-gated multi-pos)
    • mark-to-market uses chain lookup for short legs, bid for long legs
    • margin per contract = max(0, spread_width − credit) × 100
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from optimizer_prep import OptimizerContext
from candidate_codec import CandidateBatch, candidates_to_configs

# ── Constants ─────────────────────────────────────────────────────────────────
STARTING_EQUITY     = 100_000.0
IC_MULTIPLIER       = 100          # 1 contract = 100 shares
MIN_BARS_COOLDOWN   = 5            # bars between entries per candidate
ENTRY_THRESHOLD     = 0.55         # v43 entry_signal threshold
POP_THRESHOLD       = 0.50         # v43 pop threshold
BARS_PER_DAY        = 78.0         # M5 bars per trading day  (~6.5 h × 12)


# ── Return types ─────────────────────────────────────────────────────────────

@dataclass
class BatchEvalResult:
    """Results for K evaluated candidates."""
    K: int
    net_pnl:    np.ndarray   # [K] dollars
    net_pct:    np.ndarray   # [K] %
    max_dd:     np.ndarray   # [K] % (positive number)
    objective:  np.ndarray   # [K] scalar to maximise
    wins:       np.ndarray   # [K] int
    losses:     np.ndarray   # [K] int
    total:      np.ndarray   # [K] int
    win_rate:   np.ndarray   # [K] fraction
    profit_factor: np.ndarray  # [K]
    wall_sec:   float
    fidelity:   str


@dataclass
class ObjectiveSpec:
    """Specifies how to compute the scalar objective from trade results."""
    min_trades:  int   = 12
    max_dd_cap:  float = 10.0   # max drawdown cap (hard constraint)
    min_pf:      float = 1.0    # min profit factor (soft penalty)

    def compute(self, net_pct: np.ndarray, max_dd: np.ndarray,
                total: np.ndarray, pf: np.ndarray) -> np.ndarray:
        """Return objective[K] — higher is better.

        Objective = net_pct / (effective_dd + 2.0)
          - effective_dd = max(max_dd, 1.0)  — floor prevents denominator gaming
            when a candidate has 0% drawdown (all-win streaks get no free lunch)
          - Additional penalties for insufficient trades or extreme drawdown
        """
        # Clamp max_dd to minimum 1.0% so 0-drawdown strategies aren't
        # artificially rewarded with a near-zero denominator
        eff_dd = np.maximum(max_dd, 1.0)
        obj = net_pct / (eff_dd + 2.0)

        # Hard constraint: insufficient trades → big negative penalty
        obj = np.where(total < self.min_trades, -100.0 + net_pct * 0.01, obj)
        # Soft constraint: extreme drawdown cap
        obj = np.where(max_dd > self.max_dd_cap, obj * 0.5, obj)
        # Soft constraint: profit factor below threshold (was declared but never applied)
        obj = np.where(pf < self.min_pf, obj * 0.7, obj)
        return obj


# ── Chain helper ──────────────────────────────────────────────────────────────

def _lookup_price(
    right:   np.ndarray,    # [M] int8  0=call 1=put
    strike:  np.ndarray,    # [M] float32
    delta:   np.ndarray,    # [M] float32  (abs delta, may be NaN)
    bid:     np.ndarray,    # [M] float32
    ask:     np.ndarray,    # [M] float32
    target_right:  int,     # 0 or 1
    target_strike: float,
    side:          str,     # "short" → use bid; "long" → use ask
) -> float:
    """Lookup bid/ask for the option closest to target_strike."""
    mask = right == target_right
    if not mask.any():
        return np.nan
    stk = strike[mask]
    b   = bid[mask]
    a   = ask[mask]
    idx = int(np.argmin(np.abs(stk - target_strike)))
    return float(b[idx]) if side == "short" else float(a[idx])


def _find_best_structure_iron_bfly(
    spot:    float,
    right:   np.ndarray,   # chain slice
    strike:  np.ndarray,
    dte:     np.ndarray,
    delta:   np.ndarray,
    bid:     np.ndarray,
    ask:     np.ndarray,
    # K-length parameter arrays
    target_dte:   np.ndarray,  # [K]
    short_delta:  np.ndarray,  # [K]
    spread_width: np.ndarray,  # [K]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find best iron butterfly structure for K candidates simultaneously.

    Returns
    -------
    credit       [K] float32 — net credit per share  (NaN if invalid)
    short_strike [K] float32 — short call/put strike chosen
    actual_width [K] float32 — actual spread width in the chain
    valid        [K] bool    — True if structure is tradeable
    actual_dte   [K] float32 — DTE of the selected expiry (from chain, NOT target_dte)
    """
    M = len(strike)
    K = len(target_dte)

    credit_out  = np.full(K, np.nan, np.float32)
    ss_out      = np.zeros(K, np.float32)
    aw_out      = np.zeros(K, np.float32)
    valid_out   = np.zeros(K, bool)
    dte_out     = np.full(K, np.nan, np.float32)   # actual selected DTE

    if M == 0:
        return credit_out, ss_out, aw_out, valid_out, dte_out

    calls_mask = right == 0
    puts_mask  = right == 1

    # Calls and puts available?
    if not calls_mask.any() or not puts_mask.any():
        return credit_out, ss_out, aw_out, valid_out, dte_out

    for k in range(K):
        td    = float(target_dte[k])
        sd    = float(short_delta[k])
        sw    = float(spread_width[k])

        # ── 1. Select expiry closest to target_dte ─────────────────────
        dte_diff   = np.abs(dte - td)
        best_dte   = dte[np.argmin(dte_diff)]
        exp_mask   = np.abs(dte - best_dte) < 0.6

        cm = calls_mask & exp_mask
        pm = puts_mask  & exp_mask

        if not cm.any() or not pm.any():
            continue

        # Use mid prices (bid+ask)/2 for all leg pricing.
        # Using bid/ask creates artificial round-trip slippage when OTM wings
        # have bid=0 (common in real data) — buying at ask but selling at $0.
        # Mid pricing is the standard backtesting convention.
        call_mid = (bid[cm] + ask[cm]) / 2.0
        put_mid  = (bid[pm] + ask[pm]) / 2.0

        # ── 2. Short strike: closest abs(delta) to sd (or ATM) ─────────
        call_strikes = strike[cm]
        call_deltas  = np.abs(delta[cm])

        if np.isnan(call_deltas).all():
            atm_idx = int(np.argmin(np.abs(call_strikes - spot)))
        else:
            valid_d = np.isfinite(call_deltas)
            delta_err = np.where(valid_d, np.abs(call_deltas - sd), np.inf)
            atm_idx   = int(np.argmin(delta_err))

        short_stk      = float(call_strikes[atm_idx])
        short_call_mid = float(call_mid[atm_idx])

        # ── 3. Short put: same strike ───────────────────────────────────
        put_strikes = strike[pm]

        sp_dist = np.abs(put_strikes - short_stk)
        sp_idx  = int(np.argmin(sp_dist))
        if float(sp_dist[sp_idx]) > max(2.0, sw * 0.25):
            continue                                     # no put near short strike
        short_put_mid = float(put_mid[sp_idx])

        # ── 4. Long call at short_stk + sw ─────────────────────────────
        lc_target = short_stk + sw
        lc_dist   = np.abs(call_strikes - lc_target)
        lc_idx    = int(np.argmin(lc_dist))
        if float(lc_dist[lc_idx]) > max(2.0, sw * 0.40):
            continue
        long_call_mid = float(call_mid[lc_idx])
        actual_lc_stk = float(call_strikes[lc_idx])

        # ── 5. Long put at short_stk − sw ──────────────────────────────
        lp_target = short_stk - sw
        lp_dist   = np.abs(put_strikes - lp_target)
        lp_idx    = int(np.argmin(lp_dist))
        if float(lp_dist[lp_idx]) > max(2.0, sw * 0.40):
            continue
        long_put_mid  = float(put_mid[lp_idx])
        actual_lp_stk = float(put_strikes[lp_idx])

        # ── 6. Net credit (all at mid) ──────────────────────────────────
        credit = (short_call_mid + short_put_mid) - (long_call_mid + long_put_mid)

        actual_width = (actual_lc_stk - short_stk + short_stk - actual_lp_stk) / 2.0

        min_credit = max(0.10, 0.05 * sw)
        if credit < min_credit:
            continue

        credit_out[k]  = credit
        ss_out[k]      = short_stk
        aw_out[k]      = max(actual_width, 0.0)
        valid_out[k]   = True
        dte_out[k]     = best_dte   # actual DTE of selected expiry from chain

    return credit_out, ss_out, aw_out, valid_out, dte_out


def _find_best_structure_iron_condor(
    spot:    float,
    right:   np.ndarray,
    strike:  np.ndarray,
    dte:     np.ndarray,
    delta:   np.ndarray,
    bid:     np.ndarray,
    ask:     np.ndarray,
    target_dte:   np.ndarray,  # [K]
    short_delta:  np.ndarray,  # [K] — OTM delta target (0.15-0.30 typical)
    spread_width: np.ndarray,  # [K]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find iron condor structure for K candidates: separate OTM short call and put strikes.

    Returns
    -------
    credit       [K] float32 — net credit per share
    ss_call      [K] float32 — short call strike
    ss_put       [K] float32 — short put strike
    actual_width [K] float32 — average wing width
    valid        [K] bool
    actual_dte   [K] float32
    """
    M = len(strike)
    K = len(target_dte)

    credit_out = np.full(K, np.nan, np.float32)
    ssc_out    = np.zeros(K, np.float32)
    ssp_out    = np.zeros(K, np.float32)
    aw_out     = np.zeros(K, np.float32)
    valid_out  = np.zeros(K, bool)
    dte_out    = np.full(K, np.nan, np.float32)

    if M == 0:
        return credit_out, ssc_out, ssp_out, aw_out, valid_out, dte_out

    calls_mask = right == 0
    puts_mask  = right == 1
    if not calls_mask.any() or not puts_mask.any():
        return credit_out, ssc_out, ssp_out, aw_out, valid_out, dte_out

    for k in range(K):
        td = float(target_dte[k])
        sd = float(short_delta[k])   # OTM delta target
        sw = float(spread_width[k])

        dte_diff = np.abs(dte - td)
        best_dte = dte[np.argmin(dte_diff)]
        exp_mask = np.abs(dte - best_dte) < 0.6

        cm = calls_mask & exp_mask
        pm = puts_mask  & exp_mask
        if not cm.any() or not pm.any():
            continue

        call_mid     = (bid[cm] + ask[cm]) / 2.0
        put_mid      = (bid[pm] + ask[pm]) / 2.0
        call_strikes = strike[cm]
        put_strikes  = strike[pm]
        call_deltas  = np.abs(delta[cm])
        put_deltas   = np.abs(delta[pm])

        # Short call: OTM call closest to sd (call delta ~ sd, e.g. 0.20)
        if np.isnan(call_deltas).all():
            sc_idx = int(np.argmin(np.abs(call_strikes - spot * 1.015)))
        else:
            valid_d = np.isfinite(call_deltas)
            delta_err = np.where(valid_d, np.abs(call_deltas - sd), np.inf)
            sc_idx = int(np.argmin(delta_err))
        sc_stk = float(call_strikes[sc_idx])
        sc_mid = float(call_mid[sc_idx])

        # Short put: OTM put closest to sd (put delta ~ sd, e.g. 0.20)
        if np.isnan(put_deltas).all():
            sp_idx = int(np.argmin(np.abs(put_strikes - spot * 0.985)))
        else:
            valid_d = np.isfinite(put_deltas)
            delta_err = np.where(valid_d, np.abs(put_deltas - sd), np.inf)
            sp_idx = int(np.argmin(delta_err))
        sp_stk = float(put_strikes[sp_idx])
        sp_mid = float(put_mid[sp_idx])

        # IC validity: short call must be above short put
        if sc_stk <= sp_stk:
            continue

        # Long call at sc_stk + sw
        lc_target = sc_stk + sw
        lc_dist   = np.abs(call_strikes - lc_target)
        lc_idx    = int(np.argmin(lc_dist))
        if float(lc_dist[lc_idx]) > max(2.0, sw * 0.40):
            continue
        lc_mid = float(call_mid[lc_idx])
        actual_lc_stk = float(call_strikes[lc_idx])

        # Long put at sp_stk - sw
        lp_target = sp_stk - sw
        lp_dist   = np.abs(put_strikes - lp_target)
        lp_idx    = int(np.argmin(lp_dist))
        if float(lp_dist[lp_idx]) > max(2.0, sw * 0.40):
            continue
        lp_mid = float(put_mid[lp_idx])
        actual_lp_stk = float(put_strikes[lp_idx])

        credit = (sc_mid + sp_mid) - (lc_mid + lp_mid)
        actual_width = (actual_lc_stk - sc_stk + sp_stk - actual_lp_stk) / 2.0

        min_credit = max(0.10, 0.05 * sw)
        if credit < min_credit:
            continue

        credit_out[k] = credit
        ssc_out[k]    = sc_stk
        ssp_out[k]    = sp_stk
        aw_out[k]     = max(actual_width, 0.0)
        valid_out[k]  = True
        dte_out[k]    = best_dte

    return credit_out, ssc_out, ssp_out, aw_out, valid_out, dte_out


def _find_best_structure_short_call(
    spot:    float,
    right:   np.ndarray,
    strike:  np.ndarray,
    dte:     np.ndarray,
    delta:   np.ndarray,
    bid:     np.ndarray,
    ask:     np.ndarray,
    target_dte:  np.ndarray,  # [K]
    short_delta: np.ndarray,  # [K]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find short call (single-leg) structure for K candidates.

    Returns
    -------
    credit   [K] float32 — premium received per share
    ss_call  [K] float32 — short call strike
    valid    [K] bool
    dte      [K] float32
    """
    K = len(target_dte)
    credit_out = np.full(K, np.nan, np.float32)
    ss_out     = np.zeros(K, np.float32)
    valid_out  = np.zeros(K, bool)
    dte_out    = np.full(K, np.nan, np.float32)

    if len(strike) == 0:
        return credit_out, ss_out, valid_out, dte_out

    calls_mask = right == 0
    if not calls_mask.any():
        return credit_out, ss_out, valid_out, dte_out

    for k in range(K):
        td = float(target_dte[k])
        sd = float(short_delta[k])

        dte_diff = np.abs(dte - td)
        best_dte = dte[np.argmin(dte_diff)]
        exp_mask = np.abs(dte - best_dte) < 0.6

        cm = calls_mask & exp_mask
        if not cm.any():
            continue

        call_mid     = (bid[cm] + ask[cm]) / 2.0
        call_strikes = strike[cm]
        call_deltas  = np.abs(delta[cm])

        if np.isnan(call_deltas).all():
            sc_idx = int(np.argmin(np.abs(call_strikes - spot * 1.015)))
        else:
            valid_d = np.isfinite(call_deltas)
            delta_err = np.where(valid_d, np.abs(call_deltas - sd), np.inf)
            sc_idx = int(np.argmin(delta_err))

        sc_stk = float(call_strikes[sc_idx])
        sc_mid = float(call_mid[sc_idx])

        if sc_mid < 0.05:   # min premium filter
            continue

        credit_out[k] = sc_mid
        ss_out[k]     = sc_stk
        valid_out[k]  = True
        dte_out[k]    = best_dte

    return credit_out, ss_out, valid_out, dte_out


def _find_best_structure_short_put(
    spot:    float,
    right:   np.ndarray,
    strike:  np.ndarray,
    dte:     np.ndarray,
    delta:   np.ndarray,
    bid:     np.ndarray,
    ask:     np.ndarray,
    target_dte:  np.ndarray,  # [K]
    short_delta: np.ndarray,  # [K]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find short put (single-leg) structure. Returns credit, ss_put, valid, dte."""
    K = len(target_dte)
    credit_out = np.full(K, np.nan, np.float32)
    ss_out     = np.zeros(K, np.float32)
    valid_out  = np.zeros(K, bool)
    dte_out    = np.full(K, np.nan, np.float32)

    if len(strike) == 0:
        return credit_out, ss_out, valid_out, dte_out

    puts_mask = right == 1
    if not puts_mask.any():
        return credit_out, ss_out, valid_out, dte_out

    for k in range(K):
        td = float(target_dte[k])
        sd = float(short_delta[k])

        dte_diff = np.abs(dte - td)
        best_dte = dte[np.argmin(dte_diff)]
        exp_mask = np.abs(dte - best_dte) < 0.6

        pm = puts_mask & exp_mask
        if not pm.any():
            continue

        put_mid     = (bid[pm] + ask[pm]) / 2.0
        put_strikes = strike[pm]
        put_deltas  = np.abs(delta[pm])

        if np.isnan(put_deltas).all():
            sp_idx = int(np.argmin(np.abs(put_strikes - spot * 0.985)))
        else:
            valid_d = np.isfinite(put_deltas)
            delta_err = np.where(valid_d, np.abs(put_deltas - sd), np.inf)
            sp_idx = int(np.argmin(delta_err))

        sp_stk = float(put_strikes[sp_idx])
        sp_mid = float(put_mid[sp_idx])

        if sp_mid < 0.05:
            continue

        credit_out[k] = sp_mid
        ss_out[k]     = sp_stk
        valid_out[k]  = True
        dte_out[k]    = best_dte

    return credit_out, ss_out, valid_out, dte_out


_MtM_DIAG_CALLS = 0   # module-level counter; reset on each new engine call
_MtM_DIAG_LIMIT = 5  # print first N MtM evaluations


def _mark_to_market(
    spot:          float,
    right:         np.ndarray,
    strike:        np.ndarray,
    dte_chain:     np.ndarray,
    bid:           np.ndarray,
    ask:           np.ndarray,
    # Per-candidate open position info (only for open_mask candidates)
    entry_ss_call: np.ndarray,   # [K] short call strike (= short put for iron butterfly)
    entry_ss_put:  np.ndarray,   # [K] short put strike  (separate for iron condor)
    entry_width:   np.ndarray,   # [K]
    open_mask:     np.ndarray,   # [K] bool
    entry_dte:     np.ndarray,   # [K]  actual DTE at entry (entry_dte_at_entry)
    *,
    diag: bool = False,
) -> np.ndarray:
    """
    Estimate current debit-to-close for K open iron butterfly/condor positions.
    Returns debit_per_share[K]  (NaN → use fallback).

    Uses entry_ss_call and entry_ss_put for separate short strikes (iron condor),
    or both equal to the same ATM strike (iron butterfly).
    """
    global _MtM_DIAG_CALLS

    K = len(entry_ss_call)
    debit = np.full(K, np.nan, np.float32)

    if len(strike) == 0:
        return debit

    calls_mask = right == 0
    puts_mask  = right == 1

    for k in range(K):
        if not open_mask[k]:
            continue
        ssc = float(entry_ss_call[k])
        ssp = float(entry_ss_put[k])
        sw  = float(entry_width[k])
        td  = float(entry_dte[k])

        dte_diff  = np.abs(dte_chain - td)
        best_dte  = dte_chain[np.argmin(dte_diff)]
        exp_mask  = np.abs(dte_chain - best_dte) < 0.6

        cm = calls_mask & exp_mask
        pm = puts_mask  & exp_mask

        if not cm.any() or not pm.any():
            if diag and _MtM_DIAG_CALLS < _MtM_DIAG_LIMIT:
                print(f"  [MtM diag k={k}] td={td:.1f} best_dte={best_dte:.1f} "
                      f"exp_opts={exp_mask.sum()} cm={cm.sum()} pm={pm.sum()} → SKIP (no calls/puts)")
            continue

        mid_here = (bid[exp_mask] + ask[exp_mask]) / 2.0
        r_here   = right[exp_mask]
        s_here   = strike[exp_mask]

        sc_mid = _lookup_price(r_here, s_here, np.array([]), mid_here, mid_here, 0, ssc,      "short")
        sp_mid = _lookup_price(r_here, s_here, np.array([]), mid_here, mid_here, 1, ssp,      "short")
        lc_mid = _lookup_price(r_here, s_here, np.array([]), mid_here, mid_here, 0, ssc + sw, "short")
        lp_mid = _lookup_price(r_here, s_here, np.array([]), mid_here, mid_here, 1, ssp - sw, "short")

        if diag and _MtM_DIAG_CALLS < _MtM_DIAG_LIMIT:
            call_strikes_here = strike[cm]
            sc_ask_raw = _lookup_price(r_here, s_here, np.array([]), bid[exp_mask], ask[exp_mask], 0, ssc, "long")
            sc_bid_raw = _lookup_price(r_here, s_here, np.array([]), bid[exp_mask], ask[exp_mask], 0, ssc, "short")
            lc_bid_raw = _lookup_price(r_here, s_here, np.array([]), bid[exp_mask], ask[exp_mask], 0, ssc + sw, "short")
            print(f"  [MtM diag k={k}] spot={spot:.2f} ssc={ssc:.2f} ssp={ssp:.2f} sw={sw:.2f} td={td:.1f} best_dte={best_dte:.1f}")
            print(f"    exp_opts={exp_mask.sum()} (calls={cm.sum()} puts={pm.sum()})  "
                  f"DTE range: [{dte_chain[exp_mask].min():.1f}, {dte_chain[exp_mask].max():.1f}]")
            print(f"    call strikes (sample): {call_strikes_here[:6].tolist()}")
            print(f"    SC bid/ask/mid: {sc_bid_raw:.3f}/{sc_ask_raw:.3f}/{sc_mid:.3f}  LC bid={lc_bid_raw:.3f} lc_mid={lc_mid:.3f}")
            print(f"    SC_mid={sc_mid:.4f}  SP_mid={sp_mid:.4f}  LC_mid={lc_mid:.4f}  LP_mid={lp_mid:.4f}")
            print(f"    → debit(mid)={sc_mid+sp_mid-lc_mid-lp_mid:.4f}")
            _MtM_DIAG_CALLS += 1

        if any(np.isnan(v) for v in [sc_mid, sp_mid, lc_mid, lp_mid]):
            continue

        debit[k] = (sc_mid + sp_mid) - (lc_mid + lp_mid)

    return debit


def _mark_to_market_single_leg(
    right:         np.ndarray,
    strike:        np.ndarray,
    dte_chain:     np.ndarray,
    bid:           np.ndarray,
    ask:           np.ndarray,
    entry_ss:      np.ndarray,   # [K] short strike (call or put)
    open_mask:     np.ndarray,   # [K] bool
    entry_dte:     np.ndarray,   # [K]
    option_right:  int,          # 0=call, 1=put
) -> np.ndarray:
    """MtM for single-leg short positions (short_call or short_put)."""
    K = len(entry_ss)
    debit = np.full(K, np.nan, np.float32)

    if len(strike) == 0:
        return debit

    leg_mask = right == option_right

    for k in range(K):
        if not open_mask[k]:
            continue
        ss = float(entry_ss[k])
        td = float(entry_dte[k])

        dte_diff = np.abs(dte_chain - td)
        best_dte = dte_chain[np.argmin(dte_diff)]
        exp_mask = np.abs(dte_chain - best_dte) < 0.6

        lm = leg_mask & exp_mask
        if not lm.any():
            continue

        mid_here = (bid[lm] + ask[lm]) / 2.0
        stk_here = strike[lm]

        dist = np.abs(stk_here - ss)
        idx  = int(np.argmin(dist))
        if float(dist[idx]) > 5.0:   # max 5-point miss
            continue

        debit[k] = float(mid_here[idx])   # current mid price = debit to close

    return debit


# ── Main simulation ───────────────────────────────────────────────────────────

def run_backtest_optimizer_batch(
    ctx:        OptimizerContext,
    candidates: CandidateBatch,
    *,
    base_config:    Dict,
    objective_spec: ObjectiveSpec,
    fidelity:       str   = "full",   # "fast" | "medium" | "full"
    strategy_idx_filter: int = 8,     # iron_butterfly index
    strategy_family:     str = "iron_butterfly",  # "iron_butterfly" | "iron_condor" | "short_call" | "short_put"
    entry_threshold: float = ENTRY_THRESHOLD,
    pop_threshold:   float = POP_THRESHOLD,
    cooldown_bars:   int   = MIN_BARS_COOLDOWN,
    verbose: bool = False,
) -> BatchEvalResult:
    """
    Run K parameter candidates over T bars in a single vectorised loop.

    One concurrent position per candidate (max_positions=1 simplification).
    """
    t0 = time.time()
    K  = candidates.K

    # ── Fidelity gate ─────────────────────────────────────────────────────
    if fidelity == "fast":
        T_end = ctx.fast_end
    elif fidelity == "medium":
        T_end = ctx.medium_end
    else:
        T_end = ctx.T
    T_end = min(T_end, ctx.T)

    # ── Unpack candidate parameters ───────────────────────────────────────
    p = candidates.params
    target_dte_arr   = p.get('target_dte',    np.full(K, 21.0))
    short_delta_arr  = p.get('short_delta',   np.full(K, 0.45))
    spread_width_arr = p.get('spread_width',  np.full(K, 5.0))
    stop_loss_arr    = p.get('stop_loss_dollar', np.full(K, 600.0))
    profit_tgt_arr   = p.get('profit_target',    np.full(K, 1500.0))
    max_dte_exit_arr = p.get('max_dte_exit',  np.zeros(K))
    hold_days_arr    = p.get('hold_days',     np.full(K, 7.0))

    # ── State arrays ──────────────────────────────────────────────────────
    equity       = np.full(K, STARTING_EQUITY, np.float64)
    peak         = np.full(K, STARTING_EQUITY, np.float64)
    max_dd       = np.zeros(K, np.float64)
    open_mask    = np.zeros(K, bool)
    entry_credit = np.zeros(K, np.float64)   # per share
    entry_qty    = np.ones(K,  np.int32)
    entry_ss_call = np.zeros(K, np.float64)  # short call strike
    entry_ss_put  = np.zeros(K, np.float64)  # short put strike (=call for IB, separate for IC)
    entry_width  = np.zeros(K, np.float64)   # actual width (0 for single-leg)
    entry_bar    = np.full(K, -999, np.int32)
    entry_dte_at_entry = np.zeros(K, np.float64)
    last_entry   = np.full(K, -999, np.int32)
    wins         = np.zeros(K, np.int32)
    losses       = np.zeros(K, np.int32)
    gross_win    = np.zeros(K, np.float64)
    gross_loss   = np.zeros(K, np.float64)

    # ── Verbose: print candidate param summary ────────────────────────────
    if verbose:
        print(f"\n[engine] run_backtest_optimizer_batch  K={K}  fidelity={fidelity}  "
              f"T_end={T_end}/{ctx.T}  strategy_family={strategy_family}")
        print(f"[engine] Candidate param ranges across K={K}:")
        for name, arr in candidates.params.items():
            print(f"  {name:>20s}:  min={arr.min():.3g}  max={arr.max():.3g}  "
                  f"unique={len(np.unique(arr))}")
        print(f"[engine] Pulling {ctx.T} bars from {ctx.device} → CPU numpy…")

    # Pull ctx arrays to CPU numpy for speed (avoid .item() in tight loop)
    spot_np         = ctx.spot.cpu().numpy()
    gate_e_np       = ctx.gate_entry.cpu().numpy()
    gate_p_np       = ctx.gate_pop.cpu().numpy()
    sidx_np         = ctx.strategy_idx.cpu().numpy()
    abstain_np      = ctx.abstain.cpu().numpy().astype(bool)
    bar_offsets_np  = ctx.bar_offsets.cpu().numpy()
    right_np        = ctx.option_right.cpu().numpy()
    strike_np       = ctx.option_strike.cpu().numpy()
    dte_np          = ctx.option_dte.cpu().numpy()
    delta_np        = ctx.option_delta.cpu().numpy()
    bid_np          = ctx.opt_bid.cpu().numpy()
    ask_np          = ctx.opt_ask.cpu().numpy()
    ts_np           = ctx.timestamps.cpu().numpy()

    # Reset MtM diagnostic counter for this batch run
    global _MtM_DIAG_CALLS
    _MtM_DIAG_CALLS = 0

    # ── Gate-eligible bar count (always printed — critical diagnostic) ─────
    gate_eligible = int((sidx_np[:T_end] == strategy_idx_filter).sum())
    non_abstain   = int((~abstain_np[:T_end]).sum())
    print(f"[engine] strategy_family={strategy_family}  "
          f"strategy_idx_filter={strategy_idx_filter}  "
          f"gate_eligible_bars={gate_eligible}/{T_end}  "
          f"non_abstain={non_abstain}/{T_end}")
    if gate_eligible == 0:
        print(f"[engine] WARNING: 0 bars match sidx=={strategy_idx_filter}. "
              f"No trades will be entered. v43 model may not predict this class frequently. "
              f"Check --strategies argument vs v43 model training distribution.")

    # ── Verbose: progress tracking ────────────────────────────────────────
    _progress_interval = max(1, T_end // 10)   # print every 10%
    _gate_fires   = 0   # bars that passed ALL gate checks
    _total_entries= 0   # total entry events across all K
    _chain_debit_hits = 0   # MtM resolved via chain (not intrinsic fallback)
    _chain_debit_miss = 0   # MtM fell back to intrinsic value
    # Exit-type counters (summed across all K candidates)
    _exit_sl  = 0   # stop-loss hits
    _exit_pt  = 0   # profit-target hits
    _exit_dte = 0   # DTE-threshold exits
    _exit_hd  = 0   # hold-days exits
    _exit_exp = 0   # expiry exits
    # Hold-time samples (calendar days held at exit) — first 20 exits logged
    _hold_samples: list = []

    for i in range(T_end):
        spot   = float(spot_np[i])
        es     = float(gate_e_np[i])
        pop    = float(gate_p_np[i])
        sidx   = int(sidx_np[i])
        abt    = bool(abstain_np[i])

        # Chain slice for this bar
        s_off = int(bar_offsets_np[i])
        e_off = int(bar_offsets_np[i + 1])
        if s_off < e_off:
            r_sl  = right_np[s_off:e_off]
            s_sl  = strike_np[s_off:e_off]
            d_sl  = dte_np[s_off:e_off]
            da_sl = delta_np[s_off:e_off]
            b_sl  = bid_np[s_off:e_off]
            a_sl  = ask_np[s_off:e_off]
            has_chain = True
        else:
            has_chain = False

        # ── A. Process exits for open positions ───────────────────────────
        any_open = open_mask.any()
        if any_open:
            days_held = np.where(
                open_mask,
                (float(ts_np[i]) - ts_np[entry_bar].clip(0)) / 86400.0,
                0.0,
            )
            dte_remaining = np.where(open_mask, entry_dte_at_entry - days_held, 0.0)

            # Estimate current debit via chain lookup
            # Pass entry_dte_at_entry (actual DTE when entered, decremented by
            # days_held) so MtM finds the SAME expiry we entered, not target param.
            current_dte = np.where(open_mask,
                                   entry_dte_at_entry - days_held,
                                   entry_dte_at_entry)
            if verbose and _MtM_DIAG_CALLS < _MtM_DIAG_LIMIT and open_mask.any():
                for _dk in np.where(open_mask)[0][:3]:
                    print(f"  [MtM ctx k={_dk}] entry_credit={entry_credit[_dk]:.4f}  "
                          f"entry_ss_call={entry_ss_call[_dk]:.2f}  "
                          f"entry_ss_put={entry_ss_put[_dk]:.2f}  "
                          f"entry_width={entry_width[_dk]:.2f}  "
                          f"entry_dte={entry_dte_at_entry[_dk]:.1f}  "
                          f"current_dte={current_dte[_dk]:.4f}  "
                          f"days_held={days_held[_dk]:.4f}")
            if has_chain:
                if strategy_family in ("short_call",):
                    debit_per_share = _mark_to_market_single_leg(
                        r_sl, s_sl, d_sl, b_sl, a_sl,
                        entry_ss_call, open_mask, current_dte, option_right=0,
                    )
                elif strategy_family in ("short_put",):
                    debit_per_share = _mark_to_market_single_leg(
                        r_sl, s_sl, d_sl, b_sl, a_sl,
                        entry_ss_put, open_mask, current_dte, option_right=1,
                    )
                else:  # iron_butterfly, iron_condor, generic 4-leg
                    debit_per_share = _mark_to_market(
                        spot, r_sl, s_sl, d_sl, b_sl, a_sl,
                        entry_ss_call, entry_ss_put, entry_width,
                        open_mask, current_dte, diag=verbose,
                    )
            else:
                debit_per_share = np.full(K, np.nan, np.float32)

            # Track chain vs intrinsic coverage for diagnostic
            if verbose:
                _chain_ok = np.isfinite(debit_per_share) & open_mask
                _chain_debit_hits += int(_chain_ok.sum())
                _chain_debit_miss += int((~_chain_ok & open_mask).sum())

            # Fallback: intrinsic value when chain not available.
            # Strategy-aware: single-leg vs 4-leg structures have different payoffs.
            if strategy_family in ("short_call",):
                intrinsic_debit = np.maximum(spot - entry_ss_call, 0.0)
            elif strategy_family in ("short_put",):
                intrinsic_debit = np.maximum(entry_ss_put - spot, 0.0)
            else:  # 4-leg (iron_butterfly or iron_condor)
                intrinsic_call = np.maximum(spot - entry_ss_call, 0.0)
                intrinsic_put  = np.maximum(entry_ss_put - spot, 0.0)
                intrinsic_wing_call = np.maximum(spot - (entry_ss_call + entry_width), 0.0)
                intrinsic_wing_put  = np.maximum((entry_ss_put - entry_width) - spot, 0.0)
                intrinsic_debit = (intrinsic_call + intrinsic_put
                                    - intrinsic_wing_call - intrinsic_wing_put)
            debit_per_share = np.where(
                np.isfinite(debit_per_share),
                debit_per_share,
                intrinsic_debit.astype(np.float32),
            )

            # Unrealised P&L per position
            unrealized = np.where(
                open_mask,
                (entry_credit - debit_per_share) * entry_qty * IC_MULTIPLIER,
                0.0,
            )

            # Profit target (None → 50 % of max credit)
            dyn_profit_tgt = np.where(
                profit_tgt_arr > 0,
                profit_tgt_arr,
                entry_credit * entry_qty * IC_MULTIPLIER * 0.50,
            )

            # Exit conditions
            sl_hit    = open_mask & (unrealized <= -np.abs(stop_loss_arr))
            pt_hit    = open_mask & (unrealized >= dyn_profit_tgt)
            dte_exit  = open_mask & (max_dte_exit_arr > 0) & (dte_remaining <= max_dte_exit_arr)
            hd_exit   = open_mask & (days_held >= hold_days_arr)
            exp_exit  = open_mask & (dte_remaining <= 0)
            exit_mask = sl_hit | pt_hit | dte_exit | hd_exit | exp_exit

            # ── Exit diagnostic: daily samples + any exit trigger ────────────
            _exit_diag_entry = int(entry_bar[open_mask][0]) if open_mask.any() else -1
            _print_exit_row = (
                verbose and open_mask.any()
                and (exit_mask.any()                          # any exit fires
                     or (i - _exit_diag_entry) % 78 == 0     # daily sample (78 bars/day)
                     or i <= _exit_diag_entry + 5)            # first 5 bars always
            )
            if _print_exit_row:
                for _ek in np.where(open_mask)[0][:2]:
                    print(f"  [exit diag bar={i} k={_ek}] "
                          f"days_held={days_held[_ek]:.3f}  "
                          f"dte_rem={dte_remaining[_ek]:.3f}  "
                          f"unrealized={unrealized[_ek]:.2f}  "
                          f"hold_days={hold_days_arr[_ek]:.1f}  "
                          f"stop_loss={stop_loss_arr[_ek]:.0f}  "
                          f"max_dte_exit={max_dte_exit_arr[_ek]:.0f}  "
                          f"sl={sl_hit[_ek]}  pt={pt_hit[_ek]}  "
                          f"dte_ex={dte_exit[_ek]}  hd={hd_exit[_ek]}  "
                          f"exp={exp_exit[_ek]}  EXIT={exit_mask[_ek]}")

            # Realise P&L for exiting positions
            if exit_mask.any():
                pnl = np.where(exit_mask, unrealized, 0.0)
                equity = np.where(exit_mask, equity + pnl, equity)
                peak   = np.maximum(peak, equity)
                dd_now = (peak - equity) / peak * 100.0
                max_dd = np.maximum(max_dd, dd_now)

                # Win/loss accounting
                new_wins   = exit_mask & (pnl > 0)
                new_losses = exit_mask & (pnl <= 0)
                wins    = np.where(new_wins,   wins   + 1, wins)
                losses  = np.where(new_losses, losses + 1, losses)
                gross_win  = np.where(new_wins,   gross_win  + pnl, gross_win)
                gross_loss = np.where(new_losses, gross_loss + np.abs(pnl), gross_loss)

                # Exit-type accounting (verbose)
                if verbose:
                    _exit_sl  += int(sl_hit.sum())
                    _exit_pt  += int(pt_hit.sum())
                    _exit_dte += int((dte_exit & ~sl_hit & ~pt_hit).sum())
                    _exit_hd  += int((hd_exit  & ~sl_hit & ~pt_hit & ~dte_exit).sum())
                    _exit_exp += int((exp_exit  & ~sl_hit & ~pt_hit & ~dte_exit & ~hd_exit).sum())
                    # Sample hold times from first 20 exits
                    if len(_hold_samples) < 20:
                        for _k in range(K):
                            if exit_mask[_k] and len(_hold_samples) < 20:
                                _hold_samples.append({
                                    "k": _k, "bar": i,
                                    "days_held": round(float(days_held[_k]), 2),
                                    "dte_rem": round(float(dte_remaining[_k]), 2),
                                    "unrealized": round(float(unrealized[_k]), 2),
                                    "reason": ("SL" if sl_hit[_k] else
                                               "PT" if pt_hit[_k] else
                                               "DTE" if dte_exit[_k] else
                                               "HD" if hd_exit[_k] else "EXP"),
                                })

                # Clear exited positions
                open_mask = np.where(exit_mask, False, open_mask)

        # ── B. Check entries for candidates without open positions ─────────
        if not has_chain:
            continue

        # Verbose progress print every 10%
        if verbose and (i % _progress_interval == 0):
            n_open = int(open_mask.sum())
            pct    = 100.0 * i / T_end
            print(f"  [engine] {pct:>5.1f}%  bar={i:>5}/{T_end}  "
                  f"open={n_open}/{K}  gate_fires={_gate_fires}  "
                  f"total_entries={_total_entries}")

        # Gate checks (vectorised over K but all use same bar signal)
        gate_ok = (es > entry_threshold) and (pop > pop_threshold) and \
                  (not abt) and (sidx == strategy_idx_filter)
        if not gate_ok:
            continue

        _gate_fires += 1

        # Cooldown check per candidate
        cooldown_ok = (i - last_entry) >= cooldown_bars

        # Candidates eligible for entry
        eligible = cooldown_ok & (~open_mask)
        if not eligible.any():
            continue

        # Find structure for all eligible candidates (strategy-family dispatch)
        td_elig  = np.where(eligible, target_dte_arr,   21.0).astype(np.float32)
        sd_elig  = np.where(eligible, short_delta_arr,   0.30).astype(np.float32)
        sw_elig  = np.where(eligible, spread_width_arr,  5.0).astype(np.float32)

        # Initialise per-bar outputs (NaN/zero → invalid until structure found)
        credit     = np.full(K, np.nan, np.float32)
        ss_call    = np.zeros(K, np.float32)
        ss_put     = np.zeros(K, np.float32)
        aw         = np.zeros(K, np.float32)
        valid      = np.zeros(K, bool)
        struct_dte = np.full(K, np.nan, np.float32)

        if strategy_family == "iron_butterfly":
            credit, ss_call, aw, valid, struct_dte = _find_best_structure_iron_bfly(
                spot, r_sl, s_sl, d_sl, da_sl, b_sl, a_sl,
                td_elig, sd_elig, sw_elig,
            )
            ss_put = ss_call.copy()   # IB: same strike for both shorts

        elif strategy_family == "iron_condor":
            credit, ss_call, ss_put, aw, valid, struct_dte = _find_best_structure_iron_condor(
                spot, r_sl, s_sl, d_sl, da_sl, b_sl, a_sl,
                td_elig, sd_elig, sw_elig,
            )

        elif strategy_family == "short_call":
            credit, ss_call, valid, struct_dte = _find_best_structure_short_call(
                spot, r_sl, s_sl, d_sl, da_sl, b_sl, a_sl,
                td_elig, sd_elig,
            )
            # No wings: width = 0, put strike unused (set to call strike as sentinel)
            aw     = np.zeros(K, np.float32)
            ss_put = ss_call.copy()

        elif strategy_family == "short_put":
            credit, ss_put, valid, struct_dte = _find_best_structure_short_put(
                spot, r_sl, s_sl, d_sl, da_sl, b_sl, a_sl,
                td_elig, sd_elig,
            )
            aw      = np.zeros(K, np.float32)
            ss_call = ss_put.copy()

        else:
            # Unknown family → fall back to iron butterfly (logs a warning once)
            if not hasattr(run_backtest_optimizer_batch, '_warned_family'):
                run_backtest_optimizer_batch._warned_family = set()
            if strategy_family not in run_backtest_optimizer_batch._warned_family:
                print(f"[engine] WARNING: unknown strategy_family={strategy_family!r}. "
                      f"Falling back to iron_butterfly simulation.")
                run_backtest_optimizer_batch._warned_family.add(strategy_family)
            credit, ss_call, aw, valid, struct_dte = _find_best_structure_iron_bfly(
                spot, r_sl, s_sl, d_sl, da_sl, b_sl, a_sl,
                td_elig, sd_elig, sw_elig,
            )
            ss_put = ss_call.copy()

        can_enter = eligible & valid

        if can_enter.any():
            # Quantity: 1 contract (simplified, matches minimum)
            qty = np.ones(K, np.int32)

            # Margin estimate: 4-leg → width − credit; single-leg → premium × 20% spot equivalent
            if strategy_family in ("short_call", "short_put"):
                margin_per_ct = credit * IC_MULTIPLIER * 2.0   # simplified naked margin
            else:
                margin_per_ct = np.maximum(aw - credit, 0) * IC_MULTIPLIER
            margin_per_ct = np.where(can_enter, margin_per_ct, 0.0)

            # Simple capital check: equity × 40% ceiling
            max_deploy    = equity * 0.40
            capital_ok    = can_enter & (margin_per_ct <= max_deploy)

            if capital_ok.any():
                entry_credit  = np.where(capital_ok, credit.astype(np.float64),   entry_credit)
                entry_qty     = np.where(capital_ok, qty,                           entry_qty)
                entry_ss_call = np.where(capital_ok, ss_call.astype(np.float64),   entry_ss_call)
                entry_ss_put  = np.where(capital_ok, ss_put.astype(np.float64),    entry_ss_put)
                entry_width   = np.where(capital_ok, aw.astype(np.float64),        entry_width)
                entry_bar     = np.where(capital_ok, i,                             entry_bar)
                entry_dte_at_entry = np.where(capital_ok,
                                              struct_dte.astype(np.float64),
                                              entry_dte_at_entry)
                open_mask  = np.where(capital_ok, True,  open_mask)
                last_entry = np.where(capital_ok, i,     last_entry)

                if verbose:
                    n_entered = int(capital_ok.sum())
                    _total_entries += n_entered
                    dte_entered = struct_dte[capital_ok]
                    _ssc = ss_call[capital_ok]
                    _ssp = ss_put[capital_ok]
                    print(f"  [engine]   bar={i:>5}  ENTRY: {n_entered} candidate(s)  "
                          f"spot={spot:.2f}  family={strategy_family}  "
                          f"credits=[{credit[capital_ok].min():.2f}–{credit[capital_ok].max():.2f}]  "
                          f"ss_call=[{_ssc.min():.0f}–{_ssc.max():.0f}]  "
                          f"ss_put=[{_ssp.min():.0f}–{_ssp.max():.0f}]  "
                          f"actual_dte=[{dte_entered.min():.2f}–{dte_entered.max():.2f}]")

    # ── Force-close any remaining open positions at end ───────────────────
    if open_mask.any():
        pnl = np.where(open_mask, -np.abs(stop_loss_arr) * 0.5, 0.0)  # conservative end-of-sim close
        equity   = np.where(open_mask, equity + pnl, equity)
        peak     = np.maximum(peak, equity)
        dd_now   = (peak - equity) / peak * 100.0
        max_dd   = np.maximum(max_dd, dd_now)
        new_wins  = open_mask & (pnl > 0)
        new_losses= open_mask & (pnl <= 0)
        wins   = np.where(new_wins,   wins   + 1, wins)
        losses = np.where(new_losses, losses + 1, losses)
        gross_win  = np.where(new_wins,   gross_win  + pnl,       gross_win)
        gross_loss = np.where(new_losses, gross_loss + np.abs(pnl), gross_loss)

    # ── Verbose: final diagnostics ────────────────────────────────────────
    if verbose:
        print(f"\n[engine] Simulation complete  fidelity={fidelity}  bars={T_end}")
        print(f"[engine] Gate stats:  fires={_gate_fires}  total_entries={_total_entries}")
        _total_exits = _exit_sl + _exit_pt + _exit_dte + _exit_hd + _exit_exp
        if _total_exits > 0:
            print(f"[engine] Exit breakdown (all K candidates combined):")
            print(f"  stop_loss={_exit_sl}  profit_target={_exit_pt}  dte_exit={_exit_dte}  "
                  f"hold_days={_exit_hd}  expiry={_exit_exp}  total={_total_exits}")
        if _hold_samples:
            avg_days = sum(s['days_held'] for s in _hold_samples) / len(_hold_samples)
            avg_dte  = sum(s['dte_rem']   for s in _hold_samples) / len(_hold_samples)
            reasons  = [s['reason'] for s in _hold_samples]
            print(f"[engine] Hold-time samples (first {len(_hold_samples)} exits):")
            print(f"  avg_days_held={avg_days:.2f}  avg_dte_remaining={avg_dte:.2f}")
            print(f"  exit_reasons: {', '.join(reasons)}")
            print(f"  [detail] k  days_held  dte_rem  unrealized  reason")
            for s in _hold_samples[:10]:
                print(f"           {s['k']:>2}  {s['days_held']:>9.2f}  {s['dte_rem']:>7.2f}  "
                      f"{s['unrealized']:>10.2f}  {s['reason']}")
        if (_chain_debit_hits + _chain_debit_miss) > 0:
            chain_cov = 100.0 * _chain_debit_hits / (_chain_debit_hits + _chain_debit_miss)
            print(f"[engine] MtM coverage:  chain={_chain_debit_hits} ({chain_cov:.1f}%)  "
                  f"intrinsic_fallback={_chain_debit_miss} ({100-chain_cov:.1f}%)")
            if chain_cov < 50.0:
                print(f"  [WARN] >50% MtM used intrinsic fallback — P&L accuracy may be low")
        print(f"[engine] Per-candidate summary (K={K}):")
        print(f"  {'k':>4}  {'trades':>6}  {'wins':>5}  {'losses':>6}  {'net_pct':>8}  "
              f"{'max_dd':>7}  {'pf':>6}  {'obj':>8}")
        total_c = wins + losses
        wr_c    = wins / np.maximum(total_c, 1)
        # Capped profit factor: avoid quadrillion display when gross_loss=0
        pf_c    = np.where(gross_loss > 0, gross_win / gross_loss,
                           np.where(gross_win > 0, 999.0, 1.0))
        obj_c   = objective_spec.compute(
            (equity - STARTING_EQUITY) / STARTING_EQUITY * 100.0, max_dd, total_c, pf_c)
        for k in range(K):
            flag = " *** BEST" if k == int(np.nanargmax(obj_c)) else ""
            print(f"  {k:>4}  {total_c[k]:>6}  {wins[k]:>5}  {losses[k]:>6}  "
                  f"{(equity[k]-STARTING_EQUITY)/STARTING_EQUITY*100:>8.2f}%  "
                  f"{max_dd[k]:>6.2f}%  {pf_c[k]:>6.2f}  {obj_c[k]:>8.3f}{flag}")

    # ── Metrics ───────────────────────────────────────────────────────────
    net_pnl  = equity - STARTING_EQUITY
    net_pct  = net_pnl / STARTING_EQUITY * 100.0
    total    = wins + losses
    wr       = wins / np.maximum(total, 1).astype(np.float64)
    # Profit factor: cap at 999.0 when gross_loss=0 (avoid quadrillion display)
    pf       = np.where(gross_loss > 0,
                        gross_win / gross_loss,
                        np.where(gross_win > 0, 999.0, 1.0))
    objective = objective_spec.compute(net_pct, max_dd, total, pf)

    wall_sec = time.time() - t0
    return BatchEvalResult(
        K=K,
        net_pnl   = net_pnl,
        net_pct   = net_pct,
        max_dd    = max_dd,
        objective = objective,
        wins      = wins,
        losses    = losses,
        total     = total,
        win_rate  = wr,
        profit_factor = pf,
        wall_sec  = wall_sec,
        fidelity  = fidelity,
    )
