# Session Report: Iron Condor Position Sizing Fixes
**Date:** January 4, 2026  
**Repo:** [spy-iron-condor-trading](https://github.com/trextrader/spy-iron-condor-trading)

---

## Executive Summary

Fixed critical position sizing bug where Iron Condors were being reduced to 1 contract (invalid for 2-wing strategies). Added safety guards, config knobs, regression tests, and log improvements.

**Result:** Backtest now runs successfully with 11 trades, 100% win rate, $1,940 profit on $25k.

---

## Commits Made This Session

| Commit | Description |
|--------|-------------|
| `d625448` | Multi-leg safety improvements (assertion guard, config knob, regression tests) |
| `0361746` | Throttle credit rejection logging |
| `18a02c0` | Position sizing and config refinements |
| `a77ad41` | Enforce 2-contract floor for Iron Condor fuzzy scaling |

---

## Files Changed

### Core Changes

#### [backtest_engine.py](file:///c:/SPYOptionTrader_test/core/backtest_engine.py)
- Added `fallback_total_qty` and `min_total_qty_for_two_wings` to facade extras
- Wired up `min_total_qty_for_iron_condor` config parameter
- Added log throttling for credit rejection (once per day or Δ ≥ $0.10)
- Fixed Profit Factor to show `INF` when wins exist but no losses
- Fixed Data Window to use actual data dates (not trade dates)
- Added Open Trades count to metrics

#### [facade.py](file:///c:/SPYOptionTrader_test/qtmf/facade.py)
- Made scaling floor dynamic (`min_floor = 2` for Iron Condors, `1` otherwise)
- Moved `require_two_wings` check earlier in flow
- Changed post-scaling floor from hardcoded `1` to `max(min_floor, scaled_qty)`
- Added assertion guard to catch sizing logic bugs at runtime

#### [config.template.py](file:///c:/SPYOptionTrader_test/core/config.template.py)
- Added `min_total_qty_for_iron_condor: int = 2` config parameter

### New Files

#### [tests/test_iron_condor_sizing.py](file:///c:/SPYOptionTrader_test/tests/test_iron_condor_sizing.py) [NEW]
Regression test suite with 6 tests:
- `test_fallback_respects_two_wing_minimum` — Original 1-lot failure
- `test_scaling_never_reduces_below_floor` — Aggressive scaling can't violate floor
- `test_both_wings_get_at_least_one_contract` — Each wing gets at least 1
- `test_rejection_when_confidence_too_low` — Low confidence rejected properly
- `test_configurable_minimum_is_respected` — Custom minimums work
- `test_single_leg_allows_one_contract` — Single-leg strategies unaffected

---

## Bug Fixed: 1-Lot Failure

### Root Cause
```python
# BEFORE (broken):
scaled_qty = int(total_qty * g)
if total_qty >= 1:
    scaled_qty = max(1, scaled_qty)  # ❌ Always returned 1
```

### Solution
```python
# AFTER (fixed):
min_floor = min_total_for_two if require_two_wings else 1
scaled_qty = int(total_qty * g)
scaled_qty = max(min_floor, scaled_qty)  # ✅ Respects 2-contract minimum
```

---

## Backtest Results (Post-Fix)

```
Data Window:      2025-07-03 to 2025-12-03 (153 days)
Final Equity:     $26,940.00
Net Profit:       $1,940.00 (7.76%)
Max Drawdown:     $244.00 (0.90%)
Sharpe Ratio:     1.47
Profit Factor:    INF
Total Trades:     11
Win Rate:         100.00%
```

---

## Test Results

```
$ py -3.12 -m pytest -q test_iron_condor_sizing.py
......                                               [100%]
6 passed in 0.18s
```

---

## Summary of Improvements

| Category | Before | After |
|----------|--------|-------|
| Iron Condor minimum | Could drop to 1 | Always ≥ 2 (configurable) |
| Credit rejection logs | Spam (100s of lines) | Throttled (1/day) |
| Profit Factor display | "0.00" on all wins | "INF" |
| Data Window reporting | Trade dates only | Full data range |
| Open Trades tracking | Missing | Added |
| Minimum qty config | Hardcoded | `min_total_qty_for_iron_condor` |
| Regression tests | None | 6 tests |
| Assertion guards | None | Runtime safety net |


---

## Repository Sync Addendum (2026-01-24)

This document is part of the synchronized documentation set. The authoritative engineering spec and audit references are:

- `docs/INTEGRATION_PLAN_MASTER.md`
- `docs/INTERFACE_CATALOG.md`

Key alignment requirements:
1. Feature schema selection by **name** (V2.2) only; no CSV order dependence.
2. Dataset column order differs across years; schema validation must be strict.
3. Model config metadata (layers/heads/input_dim) must match deployed checkpoints.

If this document conflicts with the master spec, the master spec governs implementation.
