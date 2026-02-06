# Phase 5.5: Execution Reality Modeling

**Started:** 2026-02-06
**Status:** IN PROGRESS
**Auditor:** Claude Code (Opus 4.5)

## Overview

**Objective:** Model the messiness of real market execution on top of the proven deterministic foundation from Phases 5.1-5.4.

**Foundation:** The Replay Determinism Theorem (Phase 5.4) guarantees that every enhancement in this phase is measurable, reversible, and auditable.

**Primary Files:**
- `kaggle/condor_brain_backtest_v2.py` - Backtester execution layer
- `intelligence/execution_reality.py` - New execution reality module (to be created)

---

## Execution Reality Components

### 1. Fill Latency Model

**Problem:** Orders take 50-200ms to reach exchange; price moves during transit.

**Model:**
```python
class LatencyModel:
    def __init__(self, mean_latency_ms=100, std_latency_ms=30):
        self.mean = mean_latency_ms
        self.std = std_latency_ms

    def simulate_latency(self, seed=None) -> float:
        """Returns latency in milliseconds."""
        return max(10, np.random.normal(self.mean, self.std))

    def price_drift_during_latency(self, volatility, latency_ms) -> float:
        """Estimate price movement during order transit."""
        # sqrt(t) scaling for Brownian motion
        t_seconds = latency_ms / 1000
        return volatility * np.sqrt(t_seconds / 252 / 6.5 / 60)
```

**Impact:** Entry credit reduced by expected drift.

| Latency | Typical Drift (20% IV) | Impact on $1.50 Credit |
|---------|------------------------|------------------------|
| 50ms | ~$0.01 | -0.7% |
| 100ms | ~$0.015 | -1.0% |
| 200ms | ~$0.02 | -1.3% |

**Status:** [ ] PENDING

---

### 2. Queue Position Model

**Problem:** At a given price level, orders fill FIFO. We may not be first in queue.

**Model:**
```python
class QueuePositionModel:
    def __init__(self, aggression_factor=0.5):
        # 0 = passive (back of queue), 1 = aggressive (cross spread)
        self.aggression = aggression_factor

    def fill_probability(self, our_size, queue_depth, time_at_level_ms) -> float:
        """Probability of fill given queue position."""
        if self.aggression >= 1.0:
            return 1.0  # Crossing spread always fills

        # Simplified: linear decay based on queue position
        position_in_queue = queue_depth * (1 - self.aggression)
        fill_rate = 1.0 / (1 + position_in_queue / our_size)
        return min(1.0, fill_rate * (time_at_level_ms / 1000))
```

**Impact:** Some orders may not fill, requiring re-evaluation.

**Status:** [ ] PENDING

---

### 3. Spread Dynamics Model

**Problem:** Spreads widen during volatility, news events, and illiquid periods.

**Model:**
```python
class SpreadDynamicsModel:
    def __init__(self, base_spread_ratio=0.02):
        self.base_spread = base_spread_ratio

    def dynamic_spread(self, base_spread, vix, time_of_day, recent_volume) -> float:
        """Calculate spread based on market conditions."""
        # VIX multiplier
        vix_mult = 1.0 + max(0, (vix - 15) / 30)

        # Time-of-day multiplier (wider at open/close)
        if time_of_day < 9.5 or time_of_day > 15.5:  # First/last 30 min
            tod_mult = 1.5
        elif time_of_day < 10.0 or time_of_day > 15.0:
            tod_mult = 1.2
        else:
            tod_mult = 1.0

        # Volume multiplier (low volume = wider spread)
        vol_mult = 1.0 / max(0.5, min(2.0, recent_volume / 100))

        return base_spread * vix_mult * tod_mult * vol_mult
```

**Impact:** Entry credits lower during volatile/illiquid periods.

**Status:** [ ] PENDING

---

### 4. Volatility Shock Model

**Problem:** Sudden volatility spikes cause:
- Spread blowouts
- Quote withdrawal
- Price gaps

**Model:**
```python
class VolatilityShockModel:
    def __init__(self, shock_threshold_pct=2.0):
        self.threshold = shock_threshold_pct

    def detect_shock(self, price_change_pct, time_window_sec) -> bool:
        """Detect if a volatility shock is occurring."""
        annualized_move = abs(price_change_pct) * np.sqrt(252 * 6.5 * 60 * 60 / time_window_sec)
        return annualized_move > 50  # >50% annualized = shock

    def shock_impact(self, normal_spread) -> dict:
        """Return degraded execution conditions during shock."""
        return {
            'spread_multiplier': 3.0,
            'fill_probability': 0.3,
            'quote_staleness_ms': 500,
            'should_avoid_entry': True
        }
```

**Impact:** No entries during shocks; wider exits if forced.

**Status:** [ ] PENDING

---

### 5. Broken Spread Detection

**Problem:** Sometimes bid > ask (crossed market) or quotes are stale/invalid.

**Model:**
```python
class BrokenSpreadDetector:
    def __init__(self, max_spread_ratio=0.15, min_price=0.01):
        self.max_spread = max_spread_ratio
        self.min_price = min_price

    def is_valid_quote(self, bid, ask, last_trade=None) -> Tuple[bool, str]:
        """Check if a quote is valid for execution."""
        if bid <= 0 or ask <= 0:
            return False, "zero_price"
        if bid > ask:
            return False, "crossed_market"
        if (ask - bid) / ask > self.max_spread:
            return False, "spread_too_wide"
        if ask < self.min_price:
            return False, "price_too_low"
        if last_trade and abs(ask - last_trade) / last_trade > 0.20:
            return False, "quote_stale"
        return True, "ok"
```

**Impact:** Reject entries with broken quotes; wait for valid market.

**Status:** [ ] PENDING

---

### 6. Market Microstructure

**Problem:** Options have tick sizes, lot sizes, and minimum increments.

**Model:**
```python
class MicrostructureModel:
    def __init__(self):
        # SPY options tick sizes
        self.tick_size_below_3 = 0.01
        self.tick_size_above_3 = 0.05

    def round_to_tick(self, price) -> float:
        """Round price to valid tick increment."""
        if price < 3.0:
            return round(price / self.tick_size_below_3) * self.tick_size_below_3
        else:
            return round(price / self.tick_size_above_3) * self.tick_size_above_3

    def can_improve_price(self, current_bid, current_ask) -> bool:
        """Check if there's room to improve inside the spread."""
        tick = self.tick_size_above_3 if current_bid >= 3.0 else self.tick_size_below_3
        return (current_ask - current_bid) > tick
```

**Impact:** Price rounding affects actual fill prices.

**Status:** [ ] PENDING

---

### 7. Quote Staleness Model

**Problem:** During fast markets, displayed quotes may be outdated.

**Model:**
```python
class QuoteStalenessModel:
    def __init__(self, max_age_ms=500):
        self.max_age = max_age_ms

    def quote_reliability(self, quote_age_ms, market_speed) -> float:
        """Return reliability score 0-1 based on quote age."""
        # Faster markets = quotes go stale faster
        effective_age = quote_age_ms * market_speed
        return max(0, 1 - effective_age / self.max_age)

    def adjust_for_staleness(self, bid, ask, reliability) -> Tuple[float, float]:
        """Widen spread to account for staleness uncertainty."""
        uncertainty = (1 - reliability) * (ask - bid)
        return bid - uncertainty, ask + uncertainty
```

**Impact:** Stale quotes require wider safety margin.

**Status:** [ ] PENDING

---

### 8. Time-of-Day Liquidity Regimes

**Problem:** Liquidity varies dramatically through the trading day.

**Model:**
```python
class TimeOfDayLiquidityModel:
    def __init__(self):
        # Liquidity multipliers by hour (ET)
        self.liquidity_curve = {
            9: 0.6,   # Open: volatile, wide spreads
            10: 0.9,  # Settling
            11: 1.0,  # Normal
            12: 0.8,  # Lunch lull
            13: 0.9,  # Afternoon pickup
            14: 1.0,  # Normal
            15: 1.2,  # Pre-close liquidity
            16: 0.5,  # Close: volatile
        }

    def get_liquidity_multiplier(self, hour) -> float:
        """Get liquidity multiplier for given hour."""
        return self.liquidity_curve.get(hour, 1.0)

    def adjust_fill_expectations(self, base_fill_prob, hour) -> float:
        """Adjust fill probability based on time of day."""
        return base_fill_prob * self.get_liquidity_multiplier(hour)
```

**Impact:** Entry thresholds should be time-aware.

**Status:** [ ] PENDING

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ExecutionRealityEngine                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Latency    │  │  Queue      │  │  Spread Dynamics    │  │
│  │  Model      │  │  Position   │  │  Model              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Volatility │  │  Broken     │  │  Microstructure     │  │
│  │  Shock      │  │  Spread     │  │  Model              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────────────────────────────┐  │
│  │  Quote      │  │  Time-of-Day Liquidity              │  │
│  │  Staleness  │  │  Model                              │  │
│  └─────────────┘  └─────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    simulate_realistic_fill()                 │
│                                                             │
│  Input: (legs, chain, market_state, timestamp)              │
│  Output: (fill_price, fill_qty, fill_time, diagnostics)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Test Strategy

### Unit Tests
Each model component tested in isolation:
```python
def test_latency_model_bounds():
    model = LatencyModel(mean=100, std=30)
    latencies = [model.simulate_latency() for _ in range(1000)]
    assert min(latencies) >= 10  # Floor enforced
    assert 50 < np.mean(latencies) < 150  # Reasonable range

def test_spread_widens_with_vix():
    model = SpreadDynamicsModel()
    spread_low_vix = model.dynamic_spread(0.02, vix=12, time_of_day=12, recent_volume=100)
    spread_high_vix = model.dynamic_spread(0.02, vix=35, time_of_day=12, recent_volume=100)
    assert spread_high_vix > spread_low_vix
```

### Integration Tests
Full execution path with all models active:
```python
def test_realistic_fill_determinism():
    """Verify realistic fills are still deterministic with seed."""
    engine = ExecutionRealityEngine(seed=42)

    fill_1 = engine.simulate_realistic_fill(legs, chain, state, ts)
    engine.reset(seed=42)
    fill_2 = engine.simulate_realistic_fill(legs, chain, state, ts)

    assert fill_1 == fill_2  # Determinism preserved!
```

### Regression Tests
Compare Phase 5.2 (simple slippage) vs Phase 5.5 (full reality):
```python
def test_reality_impact_measurable():
    """Phase 5.5 should show lower P&L than Phase 5.2."""
    result_simple = run_backtest(execution_mode='simple')
    result_reality = run_backtest(execution_mode='realistic')

    # Realistic should be worse (more conservative)
    assert result_reality.total_pnl < result_simple.total_pnl
    # But not catastrophically so
    assert result_reality.total_pnl > result_simple.total_pnl * 0.5
```

---

## Success Criteria

1. **All 8 components implemented** with configurable parameters
2. **Determinism preserved** - same seed = same fills (RDT still holds)
3. **Measurable impact** - P&L reduction quantified vs Phase 5.2
4. **Reversible** - can toggle components on/off for A/B testing
5. **Auditable** - every fill decision logged with full reasoning

---

## Audit Trail

| Date | Action | Finding |
|------|--------|---------|
| 2026-02-06 | Phase 5.5 started | Framework created |
