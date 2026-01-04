# Quantor-MTFuzz™ — What Changed / What's New Report

**Date:** January 4, 2026  
**Version:** Post-Safety Hardening Release

---

## 1. Executive Overview

This change-set is primarily a **safety + correctness hardening pass** across the trade sizing pipeline and reporting layer:

- **Sizing safety is now deterministic** for Iron Condors: once the system falls back to 2 contracts, nothing later in the pipeline can reduce it below the 2-wing minimum.
- **Credit rejection logging is throttled** to avoid noisy spam when the market is persistently below min-credit.
- **Backtest reporting is more truthful and robust:**
  - Data Window shows the actual data date range
  - Profit Factor prints `INF` when there are wins and no losses
  - "Open Trades" is now part of the metrics summary

**Net result:** No more "1-lot condor" regressions, cleaner logs, and more accurate end-of-run stats.

---

## 2. backtest_engine.py — Changes and New Outputs

### 2.1 Credit rejection log throttling (1 msg/day)
When an entry is rejected due to insufficient credit, the engine now logs that rejection **at most once per day** (instead of every bar).

**Impact:**
- Cleaner logs during low-premium regimes
- Still records the fact that "min_credit is blocking entries"

### 2.2 Iron Condor fallback sizing: `fallback_total_qty = 2`
Backtest entry now has a safety fallback so that if sizing resolves to 0/None/invalid, Iron Condors default to **2 total contracts**.

**Impact:**
- Avoids "silent no-trade" or malformed 1-contract condors
- Aligns with the two-wing construction constraint enforced downstream

### 2.3 Profit Factor now prints `INF` with wins and no losses
Profit Factor calculation now handles the mathematically correct edge case:
- If `gross_losses == 0` and `gross_wins > 0`, Profit Factor is infinite
- The report prints this explicitly as `INF`

**Impact:**
- Removes misleading PF values (like divide-by-zero behavior or forced caps)
- Matches standard trading analytics conventions

### 2.4 Data Window uses actual loaded data dates
The backtest header now reports the **actual first/last bar dates** from the dataset (not "trade dates", and not placeholders).

**Impact:**
- Correct audit trail when no trades occur, or trades happen late in the sample
- Report matches what the engine actually processed

### 2.5 "Open Trades" count added to metrics
Metrics now include the number of trades still open at end-of-run.

**Impact:**
- Instantly shows whether the backtest ended flat or still holding risk
- Prevents confusion in partial windows or early-stop runs

### 2.6 Indicators used for entry context (snapshot features)
The strategy entry context includes a compact "indicator snapshot":
- RSI
- ADX
- Bollinger Bands position / distance
- Stochastic oscillator
- Volume ratio / relative volume
- Distance to SMA / trend deviation
- MTF context hooks (where enabled)

**Impact:**
- Creates a consistent feature vector for fuzzy + NN gating layers
- Makes it easier to add regime classification and probabilistic entry filters later (Stage 2–3 roadmap)

### 2.7 Neural engine integration (Mamba) with CPU-safe fallback
- If CUDA/Mamba-SSM is unavailable, uses `MockMambaKernel [CPU-Compatible]`
- Engine still initializes and emits neural diagnostics:
  ```
  Conf=0.50 | P(Bull)=0.10 | P(Bear)=0.10
  ```

**Impact:**
- Architecture supports real neural gate without hard-failing on CPU dev machines
- Backtests remain deterministic and runnable everywhere

---

## 3. qtmf/facade.py — Sizing Pipeline Safety Hardening

### 3.1 Dynamic `min_floor` scaling rule
A dynamic minimum floor is now applied depending on strategy type:
- **Iron Condor:** `min_floor = 2`
- **Otherwise:** `min_floor = 1`

### 3.2 `require_two_wings` check moved earlier
The "two wings required" rule is now enforced **early in the flow**, before later scaling/adjustments can interfere.

**Why this matters:**
Previously, it was possible for:
1. fallback to set qty = 2
2. later scaling/math to reduce it back to 1

That is now **structurally prevented**.

### 3.3 Scaling floor respects 2-contract minimum unconditionally
Even if fuzzy scaling would normally push quantity down, the condor minimum is enforced as an invariant.

**Net result:** "2-wing minimum" is now **guaranteed end-to-end**

---

## 4. strategies/options_strategy.py — Improved Execution Handling

File was tightened for better real-world execution state tolerance:
- More defensive handling around open/close transitions
- Better behavior when:
  - Pricing is missing for one leg
  - Position is already closed but loop encounters it again
  - Exit reason triggers while marks are stale

**Impact:**
- Fewer edge-case crashes
- More reliable trade lifecycle completion (entry → MTM updates → exit)

---

## 5. core/config.template.py — Updated Defaults

Template defaults updated to match the new safety model:
- System defaults into configuration that won't accidentally produce invalid 1-lot condors
- Default thresholds align with newer gating + reporting expectations

---

## 6. New Test Coverage

### tests/test_iron_condor_sizing.py (6 passing)

Regression suite defending the historical bug: *"fuzzy scaling reduced quantity back to 1 even after fallback to 2"*

Tests validate:
- Fallback activates when sizing would compute to 0/None
- Two-wing minimum is enforced
- Scaling cannot reduce below 2 for condors
- Facade returns approved plan with invariant-safe quantity

```
6 passed in 0.18s
```

---

## 7. Observable Changes in Runtime Output

Backtest log demonstrates improvements working together:
- Trades open with `Qty: 2`
- Clean reporting:
  - `Data Window: 2025-07-03 to 2025-08-01 (29 days)`
  - `Profit Factor: INF`
  - `Open Trades: 0`
- Neural engine initializes with CPU fallback and produces diagnostics

---

## 8. What Did NOT Change

- No slippage/commission model added (still a roadmap item)
- No new regime classifier logic "active" beyond indicator snapshot + hooks
- Neural line operates as safe stub when Mamba-SSM isn't present

---

## 9. Recommended Next Upgrades (Aligned to Roadmap)

### 1. Market Realism
- Slippage + commission model (entry/exit and per-leg)
- Spread-aware fills using synthetic bid/ask envelopes

### 2. Strike Selection Sophistication
- Skew/term-structure aware wing selection (delta + IV skew penalty)
- Probability-of-touch / probability-of-breach gating using DTE + sigma

### 3. Regime Gating
- Trend vs mean reversion classifier (Kalman / HMM / OU-score style)
- Apply as hard gate or sizing modifier

### 4. Portfolio-Level Risk Caps
- Aggregate delta/gamma/vega exposure limits across open condors
- Risk-budget sizing instead of per-trade sizing only
