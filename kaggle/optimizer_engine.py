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
    min_trades:  int   = 5
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

        # ── 2. Short strike: closest abs(delta) to sd (or ATM) ─────────
        call_strikes = strike[cm]
        call_deltas  = np.abs(delta[cm])
        call_bids    = bid[cm]

        if np.isnan(call_deltas).all():
            atm_idx = int(np.argmin(np.abs(call_strikes - spot)))
        else:
            valid_d = np.isfinite(call_deltas)
            delta_err = np.where(valid_d, np.abs(call_deltas - sd), np.inf)
            atm_idx   = int(np.argmin(delta_err))

        short_stk     = float(call_strikes[atm_idx])
        short_call_bid = float(call_bids[atm_idx])

        # ── 3. Short put: same strike ───────────────────────────────────
        put_strikes = strike[pm]
        put_bids    = bid[pm]
        put_asks    = ask[pm]

        sp_dist = np.abs(put_strikes - short_stk)
        sp_idx  = int(np.argmin(sp_dist))
        if float(sp_dist[sp_idx]) > max(2.0, sw * 0.25):
            continue                                     # no put near short strike
        short_put_bid = float(put_bids[sp_idx])

        # ── 4. Long call at short_stk + sw ─────────────────────────────
        call_asks = ask[cm]
        lc_target = short_stk + sw
        lc_dist   = np.abs(call_strikes - lc_target)
        lc_idx    = int(np.argmin(lc_dist))
        if float(lc_dist[lc_idx]) > max(2.0, sw * 0.40):
            continue
        long_call_ask = float(call_asks[lc_idx])
        actual_lc_stk = float(call_strikes[lc_idx])

        # ── 5. Long put at short_stk − sw ──────────────────────────────
        lp_target = short_stk - sw
        lp_dist   = np.abs(put_strikes - lp_target)
        lp_idx    = int(np.argmin(lp_dist))
        if float(lp_dist[lp_idx]) > max(2.0, sw * 0.40):
            continue
        long_put_ask  = float(put_asks[lp_idx])
        actual_lp_stk = float(put_strikes[lp_idx])

        # ── 6. Net credit ───────────────────────────────────────────────
        credit = (short_call_bid + short_put_bid) - (long_call_ask + long_put_ask)

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


def _mark_to_market(
    spot:          float,
    right:         np.ndarray,
    strike:        np.ndarray,
    dte_chain:     np.ndarray,
    bid:           np.ndarray,
    ask:           np.ndarray,
    # Per-candidate open position info (only for open_mask candidates)
    short_strike:  np.ndarray,   # [K]
    entry_width:   np.ndarray,   # [K]
    open_mask:     np.ndarray,   # [K] bool
    target_dte:    np.ndarray,   # [K]  (to find right expiry in chain)
) -> np.ndarray:
    """
    Estimate current debit-to-close for K open iron butterfly positions.
    Returns debit_per_share[K]  (NaN → use fallback).
    """
    K = len(short_strike)
    debit = np.full(K, np.nan, np.float32)

    if len(strike) == 0:
        return debit

    calls_mask = right == 0
    puts_mask  = right == 1

    for k in range(K):
        if not open_mask[k]:
            continue
        ss  = float(short_strike[k])
        sw  = float(entry_width[k])
        td  = float(target_dte[k])

        dte_diff  = np.abs(dte_chain - td)
        best_dte  = dte_chain[np.argmin(dte_diff)]
        exp_mask  = np.abs(dte_chain - best_dte) < 0.6

        cm = calls_mask & exp_mask
        pm = puts_mask  & exp_mask

        if not cm.any() or not pm.any():
            continue

        # Short call ask (cost to buy back)
        sc_ask = _lookup_price(right[exp_mask], strike[exp_mask], np.array([]), bid[exp_mask], ask[exp_mask], 0, ss, "long")
        # Short put ask
        sp_ask = _lookup_price(right[exp_mask], strike[exp_mask], np.array([]), bid[exp_mask], ask[exp_mask], 1, ss, "long")
        # Long call bid (sell back)
        lc_bid = _lookup_price(right[exp_mask], strike[exp_mask], np.array([]), bid[exp_mask], ask[exp_mask], 0, ss + sw, "short")
        # Long put bid
        lp_bid = _lookup_price(right[exp_mask], strike[exp_mask], np.array([]), bid[exp_mask], ask[exp_mask], 1, ss - sw, "short")

        if any(np.isnan(v) for v in [sc_ask, sp_ask, lc_bid, lp_bid]):
            continue

        debit[k] = (sc_ask + sp_ask) - (lc_bid + lp_bid)

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
    entry_ss     = np.zeros(K, np.float64)   # short strike
    entry_width  = np.zeros(K, np.float64)   # actual width
    entry_bar    = np.full(K, -999, np.int32)
    entry_dte_at_entry = np.zeros(K, np.float64)
    last_entry   = np.full(K, -999, np.int32)
    wins         = np.zeros(K, np.int32)
    losses       = np.zeros(K, np.int32)
    gross_win    = np.zeros(K, np.float64)
    gross_loss   = np.zeros(K, np.float64)

    # ── Verbose: print candidate param summary ────────────────────────────
    if verbose:
        print(f"\n[engine] run_backtest_optimizer_batch  K={K}  fidelity={fidelity}  T_end={T_end}/{ctx.T}")
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
            if has_chain:
                debit_per_share = _mark_to_market(
                    spot, r_sl, s_sl, d_sl, b_sl, a_sl,
                    entry_ss, entry_width, open_mask, target_dte_arr,
                )
            else:
                debit_per_share = np.full(K, np.nan, np.float32)

            # Track chain vs intrinsic coverage for diagnostic
            if verbose:
                _chain_ok = np.isfinite(debit_per_share) & open_mask
                _chain_debit_hits += int(_chain_ok.sum())
                _chain_debit_miss += int((~_chain_ok & open_mask).sum())

            # Fallback: intrinsic value when chain not available
            intrinsic_call = np.maximum(spot - entry_ss, 0.0)
            intrinsic_put  = np.maximum(entry_ss - spot, 0.0)
            intrinsic_wing_call = np.maximum(spot - (entry_ss + entry_width), 0.0)
            intrinsic_wing_put  = np.maximum((entry_ss - entry_width) - spot, 0.0)
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

        # Find structure for all eligible candidates simultaneously
        td_elig  = np.where(eligible, target_dte_arr,   21.0).astype(np.float32)
        sd_elig  = np.where(eligible, short_delta_arr,   0.45).astype(np.float32)
        sw_elig  = np.where(eligible, spread_width_arr,  5.0).astype(np.float32)

        credit, ss, aw, valid, struct_dte = _find_best_structure_iron_bfly(
            spot, r_sl, s_sl, d_sl, da_sl, b_sl, a_sl,
            td_elig, sd_elig, sw_elig,
        )

        can_enter = eligible & valid

        if can_enter.any():
            # Quantity: 1 contract (simplified, matches minimum)
            qty = np.ones(K, np.int32)

            # Margin estimate
            margin_per_ct = np.maximum(aw - credit, 0) * IC_MULTIPLIER
            margin_per_ct = np.where(can_enter, margin_per_ct, 0.0)

            # Simple capital check: equity × 40% ceiling
            max_deploy    = equity * 0.40
            capital_ok    = can_enter & (margin_per_ct <= max_deploy)

            if capital_ok.any():
                entry_credit = np.where(capital_ok, credit.astype(np.float64), entry_credit)
                entry_qty    = np.where(capital_ok, qty,                         entry_qty)
                entry_ss     = np.where(capital_ok, ss.astype(np.float64),       entry_ss)
                entry_width  = np.where(capital_ok, aw.astype(np.float64),       entry_width)
                entry_bar    = np.where(capital_ok, i,                            entry_bar)

                # Use DTE from the SELECTED EXPIRY returned by _find_best_structure_iron_bfly
                # (not a separate argmin that may pick DTE=1 when te column is all-NaN)
                entry_dte_at_entry = np.where(capital_ok,
                                              struct_dte.astype(np.float64),
                                              entry_dte_at_entry)
                open_mask          = np.where(capital_ok, True,  open_mask)
                last_entry         = np.where(capital_ok, i,     last_entry)

                if verbose:
                    n_entered = int(capital_ok.sum())
                    _total_entries += n_entered
                    dte_entered = struct_dte[capital_ok]
                    print(f"  [engine]   bar={i:>5}  ENTRY: {n_entered} candidate(s)  spot={spot:.2f}  "
                          f"credits=[{credit[capital_ok].min():.2f}–{credit[capital_ok].max():.2f}]  "
                          f"ss=[{ss[capital_ok].min():.0f}–{ss[capital_ok].max():.0f}]  "
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
