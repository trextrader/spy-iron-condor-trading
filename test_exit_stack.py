"""test_exit_stack.py — Validates Phases 4–6 exit scaffold.

Run on Lightning AI after git pull:
    python test_exit_stack.py

All tests should print PASS. Any FAIL means a regression was introduced.
"""

import sys
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []

def check(label, condition):
    if condition:
        print(f"  [{PASS}] {label}")
    else:
        print(f"  [{FAIL}] {label}")
        _failures.append(label)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ──────────────────────────────────────────────────────────────────────────────
# Import guard
# ──────────────────────────────────────────────────────────────────────────────
section("Import Check")
try:
    from intelligence.exit_stack import (
        CapitalConstraintEngine,
        HardExitRules, HardExitResult,
        ExitDecisionStack, ExitDecision,
    )
    check("intelligence.exit_stack imports cleanly", True)
except Exception as exc:
    check(f"intelligence.exit_stack imports cleanly  [{exc}]", False)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Capital Constraint Engine
# ──────────────────────────────────────────────────────────────────────────────
section("Phase 4 — CapitalConstraintEngine")

cce = CapitalConstraintEngine(alpha=0.15, L=1.0, d_max=0.20)

# max_capital
check("max_capital(100_000) = 15_000", abs(cce.max_capital(100_000) - 15_000) < 0.01)

# available_capital with no open positions
check("available_capital(100_000, []) = 15_000", abs(cce.available_capital(100_000, []) - 15_000) < 0.01)

# available_capital with some margin used
check("available_capital(100_000, [5_000]) = 10_000", abs(cce.available_capital(100_000, [5_000]) - 10_000) < 0.01)

# entry_allowed — passes
check("entry_allowed(100_000, [], 10_000) = True", cce.entry_allowed(100_000, [], 10_000))

# entry_allowed — rejected (C_strategy > C_avail)
check("entry_allowed(100_000, [], 20_000) = False", not cce.entry_allowed(100_000, [], 20_000))

# portfolio_flatten_triggered
check("flatten_triggered([100]) = True",  cce.portfolio_flatten_triggered([100.0]))
check("flatten_triggered([-50]) = False", not cce.portfolio_flatten_triggered([-50.0]))
check("flatten_triggered([])    = False", not cce.portfolio_flatten_triggered([]))

# capital_emergency_triggered
check("emergency(80_000, 100_000) = True",  cce.capital_emergency_triggered(80_000, 100_000))   # 20% drawdown
check("emergency(85_000, 100_000) = False", not cce.capital_emergency_triggered(85_000, 100_000)) # 15% < 20%
check("emergency(100_000, 0)     = False", not cce.capital_emergency_triggered(100_000, 0))      # peak=0 guard

# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 — Hard Exit Rule Taxonomy
# ──────────────────────────────────────────────────────────────────────────────
section("Phase 5 — HardExitRules")

hard = HardExitRules(max_loss_credit_mult=2.0, max_net_delta=0.30, max_drawdown_pct=0.20)

# Baseline — no triggers
result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.05,
                    B_t=100_000, B_peak=100_000)
check("Baseline: no hard exit triggered", not result.triggered)

# 1. Max Loss Exit (cost >= 2× credit)
result = hard.check(credit_received=1.0, current_cost=2.0, net_delta=0.05,
                    B_t=100_000, B_peak=100_000)
check("hard_max_loss fires at cost=2×credit", result.triggered and result.reason == "hard_max_loss")

result = hard.check(credit_received=1.0, current_cost=1.99, net_delta=0.05,
                    B_t=100_000, B_peak=100_000)
check("hard_max_loss does NOT fire at cost=1.99×credit", not result.triggered)

# 2. Delta Violation
result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.35,
                    B_t=100_000, B_peak=100_000)
check("hard_delta_violation fires at |delta|=0.35", result.triggered and result.reason == "hard_delta_violation")

result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=-0.35,
                    B_t=100_000, B_peak=100_000)
check("hard_delta_violation fires at delta=-0.35", result.triggered and result.reason == "hard_delta_violation")

result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.29,
                    B_t=100_000, B_peak=100_000)
check("hard_delta_violation does NOT fire at |delta|=0.29", not result.triggered)

# 3. Capital Emergency
result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.05,
                    B_t=79_000, B_peak=100_000)
check("hard_capital_emergency fires at 21% drawdown", result.triggered and result.reason == "hard_capital_emergency")

result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.05,
                    B_t=81_000, B_peak=100_000)
check("hard_capital_emergency does NOT fire at 19% drawdown", not result.triggered)

# 4. Pivot Containment
result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.05,
                    B_t=100_000, B_peak=100_000,
                    spot=600.0, pivot_high=598.0, pivot_low=580.0,
                    short_call_strike=595.0, short_put_strike=575.0)
check("hard_pivot_call_breach fires when spot >= pivot_high", result.triggered and result.reason == "hard_pivot_call_breach")

result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.05,
                    B_t=100_000, B_peak=100_000,
                    spot=578.0, pivot_high=610.0, pivot_low=580.0,
                    short_call_strike=605.0, short_put_strike=575.0)
check("hard_pivot_put_breach fires when spot <= pivot_low", result.triggered and result.reason == "hard_pivot_put_breach")

result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.05,
                    B_t=100_000, B_peak=100_000,
                    spot=590.0, pivot_high=600.0, pivot_low=580.0,
                    short_call_strike=595.0, short_put_strike=575.0)
check("Pivot containment does NOT fire when inside range", not result.triggered)

# Pivot containment skipped when pivots are None
result = hard.check(credit_received=1.0, current_cost=1.5, net_delta=0.05,
                    B_t=100_000, B_peak=100_000,
                    spot=620.0, pivot_high=None, pivot_low=None)
check("Pivot containment safely skipped when pivot=None", not result.triggered)

# ──────────────────────────────────────────────────────────────────────────────
# Phase 6 — ExitDecisionStack
# ──────────────────────────────────────────────────────────────────────────────
section("Phase 6 — ExitDecisionStack")

stack = ExitDecisionStack(
    p_exit_threshold=0.70,
    dte_protected_min=14,
    theta_hold_floor=0.005,
    high_water_hold_floor=0.80,
    hard_rules=HardExitRules(max_loss_credit_mult=2.0, max_net_delta=0.30, max_drawdown_pct=0.20),
    capital_engine=CapitalConstraintEngine(alpha=0.15, d_max=0.20),
)

# Baseline — all neutral, no exit
dec = stack.evaluate(credit_received=1.0, current_cost=1.2, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.40,
                     dte_remaining=20, ps_theta_pos=0.01, ps_high_water=0.50)
check("Baseline: no exit (p_exit=0.40, dte=20)", not dec.should_exit)
check("Baseline source is 'none'", dec.source == "none")

# Hard exit wins regardless of p_exit or hold zones
dec = stack.evaluate(credit_received=1.0, current_cost=2.1, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.10,
                     dte_remaining=20, ps_theta_pos=0.05, ps_high_water=0.90)
check("Hard max_loss exit fires even with p_exit=0.10 and hold zones", dec.should_exit)
check("Hard exit source is 'hard'", dec.source == "hard")
check("Hard exit reason is hard_max_loss", dec.reason == "hard_max_loss")

# Neural exit fires when p_exit > threshold AND not in hold zone
dec = stack.evaluate(credit_received=1.0, current_cost=1.2, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.85,
                     dte_remaining=5, ps_theta_pos=0.001, ps_high_water=0.50)
check("Neural exit fires (p_exit=0.85, dte=5, not protected)", dec.should_exit)
check("Neural exit source is 'neural'", dec.source == "neural")

# Neural exit suppressed by pre-decay hold zone (dte > 14)
dec = stack.evaluate(credit_received=1.0, current_cost=1.2, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.85,
                     dte_remaining=20, ps_theta_pos=0.0, ps_high_water=0.0)
check("Neural exit suppressed by pre-decay hold zone (dte=20 > 14)", not dec.should_exit)

# Neural exit suppressed by theta hold zone
dec = stack.evaluate(credit_received=1.0, current_cost=1.2, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.85,
                     dte_remaining=5, ps_theta_pos=0.01, ps_high_water=0.0)
check("Neural exit suppressed by theta hold zone (theta=0.01 > 0.005)", not dec.should_exit)

# Neural exit suppressed by high-water hold zone
dec = stack.evaluate(credit_received=1.0, current_cost=1.2, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.85,
                     dte_remaining=5, ps_theta_pos=0.001, ps_high_water=0.85)
check("Neural exit suppressed by high-water hold zone (hw=0.85 > 0.80)", not dec.should_exit)

# Portfolio flatten overrides hold zone but not hard exits
dec = stack.evaluate(credit_received=1.0, current_cost=1.2, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.10,
                     dte_remaining=20, ps_theta_pos=0.05, ps_high_water=0.90,
                     all_unrealized_pnls=[500.0])
check("Portfolio flatten fires (positive pnl, p_exit low, hold zone active)", dec.should_exit)
check("Portfolio flatten source is 'portfolio'", dec.source == "portfolio")

# p_exit exactly at threshold — should NOT trigger (strictly greater-than)
dec = stack.evaluate(credit_received=1.0, current_cost=1.2, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.70,
                     dte_remaining=5, ps_theta_pos=0.001, ps_high_water=0.0)
check("Neural exit does NOT fire at p_exit == threshold (strictly >)", not dec.should_exit)

# p_exit just above threshold — triggers
dec = stack.evaluate(credit_received=1.0, current_cost=1.2, net_delta=0.05,
                     B_t=100_000, B_peak=100_000, p_exit=0.701,
                     dte_remaining=5, ps_theta_pos=0.001, ps_high_water=0.0)
check("Neural exit fires at p_exit = 0.701 > 0.700", dec.should_exit)

# ──────────────────────────────────────────────────────────────────────────────
# Config / backtest_engine import check
# ──────────────────────────────────────────────────────────────────────────────
section("Integration — Config & Backtest Engine")

try:
    from core.config import StrategyConfig
    cfg = StrategyConfig()
    check("StrategyConfig has use_exit_stack", hasattr(cfg, 'use_exit_stack'))
    check("StrategyConfig has capital_constraint_alpha", hasattr(cfg, 'capital_constraint_alpha'))
    check("StrategyConfig has exit_hard_max_loss_mult", hasattr(cfg, 'exit_hard_max_loss_mult'))
    check("StrategyConfig has exit_hard_max_delta", hasattr(cfg, 'exit_hard_max_delta'))
    check("StrategyConfig has exit_hard_max_dd_pct", hasattr(cfg, 'exit_hard_max_dd_pct'))
    check("StrategyConfig has exit_stack_p_threshold", hasattr(cfg, 'exit_stack_p_threshold'))
    check("StrategyConfig has exit_stack_dte_protected", hasattr(cfg, 'exit_stack_dte_protected'))
    check("use_exit_stack default is True", cfg.use_exit_stack is True)
    check("exit_stack_p_threshold default is 0.70", abs(cfg.exit_stack_p_threshold - 0.70) < 1e-9)
    check("exit_hard_max_loss_mult default is 2.0", abs(cfg.exit_hard_max_loss_mult - 2.0) < 1e-9)
    check("exit_hard_max_delta default is 0.30", abs(cfg.exit_hard_max_delta - 0.30) < 1e-9)
except Exception as exc:
    check(f"Config check failed: {exc}", False)

try:
    import core.backtest_engine as be
    check("core.backtest_engine imports with exit_stack wired", True)
except Exception as exc:
    check(f"core.backtest_engine import failed: {exc}", False)

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
section("Results")
total = 0
# count all check calls: just use _failures list
print(f"\n  Failures: {len(_failures)}")
if _failures:
    for f in _failures:
        print(f"    ✗ {f}")
    print(f"\n  RESULT: {len(_failures)} test(s) FAILED")
    sys.exit(1)
else:
    print(f"\n  RESULT: ALL TESTS PASSED ✓")
