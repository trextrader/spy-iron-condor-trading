"""exit_stack.py — CondorNet v4.3 Exit Scaffold Phases 4–6

Framework: "Entries with Exits and Cognitive Holds, Predictive Analytic Folds"

Phase 4: CapitalConstraintEngine — deterministic portfolio capital gating (Section II)
Phase 5: HardExitRules           — non-learnable hard-stop taxonomy (Sections IV.1, IV.3, IV.5, V.2)
Phase 6: ExitDecisionStack       — full multi-stage production exit stack (Section X)

Exit logic hierarchy (Section X):
    Exit if:
        HardExit
        OR PortfolioFlattenTriggered
        OR (p_exit > threshold AND NOT in ProtectedHoldZone)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Capital Constraint Engine
# ──────────────────────────────────────────────────────────────────────────────

class CapitalConstraintEngine:
    """
    Deterministic portfolio-level capital gating (framework Section II).
    No gradients — pure arithmetic guard.

    C_max   = alpha × L × B_t
    C_avail = C_max − Σ C_i   (sum over open position margins)
    Entry allowed iff C_strategy ≤ C_avail

    Also gates on:
      - Portfolio flatten:  Σ UnrealizedPnL_i > 0  →  Close All
      - Capital emergency:  (B_peak − B_t) / B_peak > d_max
    """

    def __init__(
        self,
        alpha: float = 0.15,   # Max portfolio allocation as fraction of equity
        L: float = 1.0,        # Leverage factor (1.0 = cash-account, no leverage)
        d_max: float = 0.20,   # Peak-to-trough drawdown threshold for emergency exit
    ):
        self.alpha = alpha
        self.L = L
        self.d_max = d_max

    # ── Capital limits ──────────────────────────────────────────────────────

    def max_capital(self, B_t: float) -> float:
        """Maximum capital allocatable across all positions."""
        return self.alpha * self.L * B_t

    def available_capital(self, B_t: float, open_margins: List[float]) -> float:
        """Capital available for a new position."""
        return max(0.0, self.max_capital(B_t) - sum(open_margins))

    def entry_allowed(
        self,
        B_t: float,
        open_margins: List[float],
        C_strategy: float,
    ) -> bool:
        """Return True if sufficient capital exists for the proposed trade."""
        return C_strategy <= self.available_capital(B_t, open_margins)

    # ── Portfolio-level exit triggers ───────────────────────────────────────

    def portfolio_flatten_triggered(self, unrealized_pnls: List[float]) -> bool:
        """
        Flatten rule (framework Section II): if aggregate unrealized P&L across
        all open positions is positive, lock in gains by closing everything.
        """
        return bool(unrealized_pnls) and sum(unrealized_pnls) > 0.0

    def capital_emergency_triggered(self, B_t: float, B_peak: float) -> bool:
        """
        Emergency exit if peak-to-trough drawdown exceeds d_max.
        (Framework Section II — 'Capital Emergency Exit')
        """
        if B_peak <= 0:
            return False
        return (B_peak - B_t) / B_peak > self.d_max


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 — Hard Exit Rule Taxonomy
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HardExitResult:
    triggered: bool
    reason: str   # "" when not triggered


class HardExitRules:
    """
    Deterministic hard-stop guardrails that cannot be overridden by model output.
    (Framework Sections IV.1, IV.3, IV.5, V.2)

    Rules evaluated in priority order:
      1. Max Loss Exit        — cost ≥ max_loss_credit_mult × entry credit
      2. Delta Violation      — |net_IC_delta| > max_net_delta
      3. Capital Emergency    — (B_peak − B_t) / B_peak > max_drawdown_pct
      4. Pivot Containment    — spot crosses predicted pivot against short-strike side
    """

    def __init__(
        self,
        max_loss_credit_mult: float = 2.0,   # 200% of credit received → exit
        max_net_delta: float = 0.30,          # Net IC |delta| threshold
        max_drawdown_pct: float = 0.20,       # Peak-to-trough emergency threshold
    ):
        self.max_loss_credit_mult = max_loss_credit_mult
        self.max_net_delta = max_net_delta
        self.max_drawdown_pct = max_drawdown_pct

    def check(
        self,
        credit_received: float,
        current_cost: float,
        net_delta: float,
        B_t: float,
        B_peak: float,
        spot: Optional[float] = None,
        pivot_high: Optional[float] = None,
        pivot_low: Optional[float] = None,
        short_call_strike: Optional[float] = None,
        short_put_strike: Optional[float] = None,
    ) -> HardExitResult:

        # 1. Max Loss Exit  (framework Section IV.1)
        #    Trigger when current debit exceeds N × entry credit
        if credit_received > 0 and current_cost >= self.max_loss_credit_mult * credit_received:
            return HardExitResult(True, "hard_max_loss")

        # 2. Delta Violation  (framework Section IV.3)
        #    Trigger when net IC delta exceeds the balanced threshold
        if abs(net_delta) > self.max_net_delta:
            return HardExitResult(True, "hard_delta_violation")

        # 3. Capital Emergency Exit  (framework Section IV.5)
        #    Trigger when total equity drawdown from peak exceeds threshold
        if B_peak > 0 and (B_peak - B_t) / B_peak > self.max_drawdown_pct:
            return HardExitResult(True, "hard_capital_emergency")

        # 4. Pivot Containment Hard Stop  (framework Section V.2)
        #    "Not a soft gate but a hard stop" — if spot crosses predicted pivot
        #    against the short-strike direction, exit immediately.
        #    pivot_high / pivot_low populated from CondorNet pivot_pred_head output.
        if spot is not None and pivot_high is not None and pivot_low is not None:
            if short_call_strike is not None and spot >= pivot_high:
                return HardExitResult(True, "hard_pivot_call_breach")
            if short_put_strike is not None and spot <= pivot_low:
                return HardExitResult(True, "hard_pivot_put_breach")

        return HardExitResult(False, "")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6 — Production Exit Stack
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitDecision:
    should_exit: bool
    reason: str
    source: str      # "hard" | "neural" | "portfolio" | "none"
    p_exit: float = 0.0


class ExitDecisionStack:
    """
    Full multi-stage production exit stack (framework Section X).

    Exit hierarchy:
        1. HardExit                          — always wins, no overrides
        2. PortfolioFlattenTriggered         — lock in aggregate gains
        3. Neural exit signal + hold zones   — p_exit > threshold AND NOT protected

    Protected Hold Zones suppress premature neural exits:
      • Pre-decay zone    : dte_remaining > dte_protected_min
      • Theta-favorable   : ps_theta_pos > theta_hold_floor
      • High-water locked : ps_high_water > high_water_hold_floor

    Neural integration:
      • p_exit sourced from CondorNet exit_signal sigmoid output [0, 1]
      • Falls back to 0.5 (neutral) when model output not yet wired
      • pivot_high / pivot_low sourced from CondorNet pivot_pred_head output
      • Both are None-safe — rules are simply skipped when unavailable
    """

    def __init__(
        self,
        p_exit_threshold: float = 0.70,        # Neural exit signal threshold
        dte_protected_min: int = 14,            # Bars remaining for pre-decay protection
        theta_hold_floor: float = 0.005,        # Min theta to qualify for theta-hold zone
        high_water_hold_floor: float = 0.80,    # High-water pnl_pct fraction for protection
        hard_rules: Optional[HardExitRules] = None,
        capital_engine: Optional[CapitalConstraintEngine] = None,
    ):
        self.p_exit_threshold = p_exit_threshold
        self.dte_protected_min = dte_protected_min
        self.theta_hold_floor = theta_hold_floor
        self.high_water_hold_floor = high_water_hold_floor
        self.hard_rules = hard_rules or HardExitRules()
        self.capital_engine = capital_engine or CapitalConstraintEngine()

    # ── Hold-zone logic ─────────────────────────────────────────────────────

    def in_protected_hold_zone(
        self,
        dte_remaining: int,
        ps_theta_pos: float = 0.0,
        ps_high_water: float = 0.0,
    ) -> bool:
        """
        Return True if position is in a protected hold zone where neural exits
        are suppressed.  Hard exits always override this.
        """
        if dte_remaining > self.dte_protected_min:
            return True   # Pre-decay zone: let theta work
        if ps_theta_pos > self.theta_hold_floor:
            return True   # Theta still accruing meaningfully
        if ps_high_water > self.high_water_hold_floor:
            return True   # Near peak value — don't rush out
        return False

    # ── Main evaluate ────────────────────────────────────────────────────────

    def evaluate(
        self,
        # Position financials
        credit_received: float,
        current_cost: float,
        net_delta: float,
        B_t: float,
        B_peak: float,
        # Neural signal (from CondorNet exit_signal head)
        p_exit: float = 0.5,
        # Hold-zone inputs (from pos_state vector or simple heuristic)
        dte_remaining: int = 0,
        ps_theta_pos: float = 0.0,
        ps_high_water: float = 0.0,
        # Pivot containment inputs (from CondorNet pivot_pred_head)
        spot: Optional[float] = None,
        pivot_high: Optional[float] = None,
        pivot_low: Optional[float] = None,
        short_call_strike: Optional[float] = None,
        short_put_strike: Optional[float] = None,
        # Portfolio flatten inputs
        all_unrealized_pnls: Optional[List[float]] = None,
    ) -> ExitDecision:

        # ── Stage 1: Hard exits (always override everything) ────────────────
        hard = self.hard_rules.check(
            credit_received=credit_received,
            current_cost=current_cost,
            net_delta=net_delta,
            B_t=B_t,
            B_peak=B_peak,
            spot=spot,
            pivot_high=pivot_high,
            pivot_low=pivot_low,
            short_call_strike=short_call_strike,
            short_put_strike=short_put_strike,
        )
        if hard.triggered:
            return ExitDecision(True, hard.reason, "hard", p_exit)

        # ── Stage 2: Portfolio flatten (positive aggregate unrealized P&L) ──
        pnls = all_unrealized_pnls if all_unrealized_pnls is not None else []
        if self.capital_engine.portfolio_flatten_triggered(pnls):
            return ExitDecision(True, "portfolio_flatten", "portfolio", p_exit)

        # ── Stage 3: Neural exit signal gated by hold zones ─────────────────
        if p_exit > self.p_exit_threshold:
            if not self.in_protected_hold_zone(dte_remaining, ps_theta_pos, ps_high_water):
                return ExitDecision(True, "neural_exit", "neural", p_exit)

        return ExitDecision(False, "", "none", p_exit)
