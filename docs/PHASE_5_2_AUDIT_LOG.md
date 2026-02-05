# Phase 5.2: Truth Alignment Audit Log

**Started:** 2026-02-05
**Status:** COMPLETE - FIXES IMPLEMENTED
**Auditor:** Claude Code (Opus 4.5)

## Overview
Ensure the backtester's world is identical to the live broker's world in how it prices, fills, and values positions.

**Primary File Audited:** `kaggle/condor_brain_backtest_v2.py`

---

## Task 1: Price Source Consistency Audit
**Status:** COMPLETE - CRITICAL ISSUES FOUND

### Objective
Verify that the backtester uses the same price source for:
- [x] Entry fills - **USES `close`**
- [x] Exit fills - **USES `close`**
- [x] Mark-to-market valuation - **USES `close`**
- [ ] Greeks evaluation - **NOT USED IN BACKTEST LOOP**
- [ ] Risk sizing inputs - **USES MODEL OUTPUT, NOT MARKET DATA**

### Price Sources Found

| Location | File:Line | Price Type | Notes |
|----------|-----------|------------|-------|
| Leg Selection | `condor_brain_backtest_v2.py:445-448` | `close` | `short_call_row['close'].values[0]` |
| Marks Cache | `condor_brain_backtest_v2.py:657` | `close` | `marks = dict(zip(chain_slice['option_symbol'], chain_slice['close']))` |
| Entry Credit | `condor_brain_backtest_v2.py:739` | `close` | `cre = legs['short_call_close'] + legs['short_put_close'] - ...` |
| MTM Update | `condor_brain_backtest_v2.py:489` | `close` | `debit = (sc + sp) - (lc + lp)` via marks dict |

### CRITICAL ISSUE #1: Using `close` Instead of `bid`/`ask`/`mid`

**Problem:** The backtester uses the `close` column for all pricing:
```python
# Line 657
marks = dict(zip(chain_slice['option_symbol'], chain_slice['close']))
```

**Reality:** In live trading:
- **Selling options** (short call/put): Filled at **BID**
- **Buying options** (long call/put): Filled at **ASK**
- **MTM valuation**: Should use **MID** (midpoint of bid/ask)

**Impact:** Using `close` creates phantom edge because:
1. `close` is often the last trade, not where you'd actually fill
2. Ignores bid/ask spread (typically $0.05-$0.20 per option)
3. For 4-leg Iron Condor, this can be $0.20-$0.80 per spread in hidden slippage

### Recommendation
Replace `close` with proper bid/ask logic:
```python
# Entry: Sell shorts at BID, buy longs at ASK
entry_credit = (short_call_bid + short_put_bid) - (long_call_ask + long_put_ask)

# MTM: Use MID for fair value
mtm_value = (short_call_mid + short_put_mid) - (long_call_mid + long_put_mid)
```

---

## Task 2: Greeks Truth Alignment Audit
**Status:** COMPLETE - GREEKS NOT USED IN SIMULATION

### Objective
Verify Greeks are identical across:
- [x] Training dataset - Has `delta`, `gamma`, `theta`, `vega`, `iv` columns
- [ ] Backtester - **GREEKS LOADED BUT NOT USED IN SIMULATION LOOP**
- [ ] Live broker / BS analytics - N/A

### Findings

**Greeks Loading (Line 154-257 `load_data_and_features`):**
- Loads Greeks from CSV if available
- Computes dynamic features including `iv`, `ivr`

**Greeks Usage in Simulation:**
- **NOT USED** - The simulation loop (lines 639-761) does not reference Greeks
- Model outputs `(call_off, put_off, width, dte, prob, roi, max_loss, conf, entry_logit, exit_logit)`
- Leg selection uses strike proximity, not delta targeting

### ISSUE #2: No Delta-Based Strike Selection

**Problem:** `find_best_legs` (line 407-450) selects strikes by dollar offset:
```python
# Line 416-417
s_call_target = spot + (call_off * spot * 0.01)  # % of spot
s_put_target = spot - (put_off * spot * 0.01)
```

**Reality:** Professional IC traders select by delta (e.g., 0.16 delta shorts).

**Impact:** Training on model-suggested offsets but not validating against delta creates:
1. Inconsistent probability of profit across different IV environments
2. No hedge ratio awareness

### Sample Timestamp Comparisons
| Timestamp | Source | Delta | Gamma | Theta | Vega | IV |
|-----------|--------|-------|-------|-------|------|-----|
| N/A | Dataset | Present | Present | Present | Present | Present |
| N/A | Backtester | **NOT USED** | **NOT USED** | **NOT USED** | **NOT USED** | **NOT USED** |

---

## Task 3: Fill Semantics & Atomicity Audit
**Status:** COMPLETE - MULTIPLE ISSUES FOUND

### Objective
Verify:
- [x] Multi-leg orders are treated atomically - **ASSUMED ATOMIC, NOT VERIFIED**
- [ ] Partial fills are handled correctly - **NO PARTIAL FILL LOGIC**
- [ ] Slippage is modeled realistically - **NO SLIPPAGE MODEL**
- [x] No leg is filled at a price that never existed - **USES ACTUAL CHAIN DATA**

### Fill Logic Trace

| Operation | File:Line | Atomic? | Price Source | Slippage Model |
|-----------|-----------|---------|--------------|----------------|
| IC Entry | `condor_brain_backtest_v2.py:734-750` | ASSUMED | `close` | NONE |
| IC Exit (Risk Stop) | `condor_brain_backtest_v2.py:665-683` | ASSUMED | `close` via marks | NONE |
| IC Exit (Expiration) | `condor_brain_backtest_v2.py:693-710` | ASSUMED | `close` via marks | NONE |
| MTM Update | `condor_brain_backtest_v2.py:476-499` | N/A | `close` | N/A |

### ISSUE #3: No Slippage Model

**Problem:** Entry and exit assume perfect fills at `close` price.

**Code (Lines 738-743):**
```python
cre = legs['short_call_close'] + legs['short_put_close'] - legs['long_call_close'] - legs['long_put_close']
if cre > 0.10:  # Min credit filter
    max_loss = (legs['width'] - cre) * IC_MULTIPLIER * IC_CONTRACTS
```

**Reality:**
- Bid/ask spread costs $0.05-$0.20 per leg × 4 legs = $0.20-$0.80 per trade
- Market impact on larger orders
- Latency slippage during fast moves

### ISSUE #4: No Partial Fill Handling

**Problem:** All fills are assumed complete.

**Code (Line 750):**
```python
open_trades.append(new_trade)  # Assumes full fill
```

**Reality:** Large orders may partially fill, leaving delta exposure.

### ISSUE #5: Atomicity Assumed, Not Verified

**Problem:** No check that all 4 legs have valid quotes at the same timestamp.

**Code (Line 649):**
```python
s, e = ts_ranges.get(ts, (None, None))
if s is None:
    continue  # Skip if no chain
```

This checks for chain existence but not that all 4 specific strikes are present.

### Comparison with `core/backtest_engine.py`

The older backtester has better execution modeling:

| Feature | `kaggle/condor_brain_backtest_v2.py` | `core/backtest_engine.py` |
|---------|--------------------------------------|---------------------------|
| Price Source | `close` | `mid` (line 94, 417) |
| Slippage | None | `slippage_rate` param (line 76) |
| Atomicity Check | None | `is_atomic` return (line 92) |
| Execution Simulation | None | `simulate_broker_execution()` |

---

## Critical Issues Found

1. **CRITICAL: Price Source Mismatch** - Uses `close` instead of bid/ask/mid. Creates phantom edge of $0.20-$0.80 per trade.
   - **STATUS: FIXED** - Added `synthesize_chain_prices()` and separate `marks_bid/marks_ask/marks_mid` dictionaries

2. **HIGH: No Slippage Model** - Perfect fills assumed. Real-world slippage not modeled.
   - **STATUS: FIXED** - Added `EXEC_SLIPPAGE_PER_LEG = 0.02` and applied in `calculate_entry_fill()` / `calculate_exit_fill()`

3. **HIGH: Greeks Not Used** - Model trained with Greeks but backtest ignores them for strike selection.
   - **STATUS: FIXED** - Added delta-based strike selection in `find_best_legs()` with `EXEC_TARGET_SHORT_DELTA = 0.16`

4. **MEDIUM: No Partial Fill Logic** - Assumes 100% fill rate.
   - **STATUS: PARTIALLY FIXED** - Added `validate_leg_liquidity()` with volume/OI checks; full partial fill simulation deferred

5. **MEDIUM: Atomicity Not Verified** - Doesn't check all 4 legs have valid quotes.
   - **STATUS: FIXED** - Added `EXEC_ATOMICITY_STRICT` flag and validation in `calculate_entry_fill()` / `calculate_exit_fill()`

---

## Recommendations

### Priority 1: Fix Price Source (BLOCKING)
```python
# Replace line 657:
# OLD: marks = dict(zip(chain_slice['option_symbol'], chain_slice['close']))
# NEW: Use bid/ask/mid columns
marks_bid = dict(zip(chain_slice['option_symbol'], chain_slice['bid']))
marks_ask = dict(zip(chain_slice['option_symbol'], chain_slice['ask']))
marks_mid = dict(zip(chain_slice['option_symbol'],
                     (chain_slice['bid'] + chain_slice['ask']) / 2))
```

### Priority 2: Add Slippage Model
```python
# Add after line 739:
SLIPPAGE_PER_LEG = 0.02  # $0.02 per leg
cre_after_slippage = cre - (SLIPPAGE_PER_LEG * 4)
```

### Priority 3: Add Atomicity Check
```python
# Replace leg selection with validation:
def find_best_legs_with_validation(chain_df, spot, ...):
    legs = find_best_legs(chain_df, spot, ...)
    if legs is None:
        return None
    # Verify all legs have valid quotes
    required_symbols = [legs['short_call_symbol'], legs['long_call_symbol'],
                        legs['short_put_symbol'], legs['long_put_symbol']]
    if not all(sym in marks_bid and sym in marks_ask for sym in required_symbols):
        return None  # Atomicity violation
    return legs
```

---

## Next Phases (After 5.2)
- Phase 5.3: Single-vs-Batched Backtest Equivalence
- Phase 5.4: Replay Determinism Theorem
- Phase 5.5: Execution Reality Modeling

---

## Audit Trail

| Date | Action | Finding |
|------|--------|---------|
| 2026-02-05 | Initial audit | 5 critical/high issues found |
| | | Primary: Price source uses `close` not bid/ask |
| 2026-02-05 | Phase 5.2 Fixes Implemented | All 5 issues addressed in `condor_brain_backtest_v2.py` |
| | Helper functions added | `synthesize_bid_ask()`, `synthesize_chain_prices()`, `validate_leg_liquidity()`, `validate_leg_delta()`, `calculate_entry_fill()`, `calculate_exit_fill()` |
| | Config constants added | `EXEC_SLIPPAGE_PER_LEG`, `EXEC_TARGET_SHORT_DELTA`, `EXEC_ATOMICITY_STRICT`, etc. |
| | Entry logic updated | Uses `calculate_entry_fill()` with bid/ask + slippage |
| | Risk stop exit updated | Uses `calculate_exit_fill()` with bid/ask + slippage |
| | Expiration exit updated | Uses `calculate_exit_fill()` with bid/ask + slippage |
| | MTM valuation updated | Uses MID for fair value, calculates realistic exit in `update_mark()` |
