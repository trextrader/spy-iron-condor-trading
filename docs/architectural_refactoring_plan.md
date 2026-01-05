# Quantor-MTFuzz Architectural Refactoring Plan
## Migration from Current to Target Architecture

> [!IMPORTANT]
> This plan refactors the existing codebase to match the **full Quantor-MTFuzz specification** with proper module separation, DTOs, and testability.

---

## Current State Analysis

### Existing Structure
```
c:\SPYOptionTrader_test\
├── core/
│   ├── main.py              ✅ (Entry point exists)
│   ├── backtest_engine.py   ⚠️  (Monolithic - needs splitting)
│   ├── config.py            ⚠️  (Needs DTO refactor)
│   └── risk_manager.py      ✅ (New, Stage 3)
├── strategies/
│   └── options_strategy.py  ⚠️  (Mixed concerns)
├── intelligence/
│   ├── fuzzy_engine.py      ⚠️  (Needs FIS refactor)
│   ├── regime_filter.py     ✅
│   └── mamba_engine.py      ✅
├── data_factory/
│   ├── sync_engine.py       ⚠️  (MTF only)
│   └── polygon_client.py    ⚠️  (Needs abstraction)
└── analytics/               ❌ (Missing - needs creation)
```

### Key Issues
1. **Monolithic `backtest_engine.py`**: 1100+ lines mixing data, strategy, sizing, risk
2. **No DTOs**: Direct coupling between modules
3. **Missing `analytics/`**: VRP, divergence, skew logic scattered
4. **Fuzzy logic incomplete**: No proper FIS pipeline (fuzzify → infer → defuzz)
5. **No test coverage**: Zero unit tests for critical logic

---

## Target Architecture (Quantor-MTFuzz Spec)

### Module Hierarchy
```
core/
├── types.py          [NEW] → DTOs (MarketSnapshot, TradeDecision, etc.)
├── engine.py         [NEW] → TradingEngine orchestrator
├── trade_router.py   [NEW] → Order execution abstraction
└── mode.py           [NEW] → RunMode enum

data_factory/
├── data_engine.py    [NEW] → Unified snapshot provider
├── spot_bars.py      [NEW] → Bar data abstraction
├── option_chain.py   [NEW] → Chain data abstraction
└── aux_feeds.py      [NEW] → VIX/ES/gap data

strategies/
└── iron_condor.py    [REFACTOR] → Signal gating only (no sizing)

intelligence/
├── fis_sizer.py      [NEW] → FIS pipeline orchestrator
├── fuzzifier.py      [NEW] → Feature extraction + membership
├── inference_engine.py [NEW] → Mamdani rule engine
└── defuzzifier.py    [NEW] → Centroid defuzzification

analytics/            [NEW FOLDER]
├── realized_vol.py   → RV calculator
├── divergence.py     → SPY-ES Z-score
├── carry_model.py    → Cost-of-carry
├── skew.py           → IV skew metric
└── gaps.py           → Gap analyzer

risk/
├── risk_manager.py   [REFACTOR] → Approval gate
├── expected_shortfall.py [NEW] → CVaR
├── beta_weighting.py [NEW] → Portfolio delta
├── pot_monitor.py    [NEW] → Probability of touch
└── structure_validator.py [NEW] → Invariants

tests/                [NEW FOLDER]
├── test_realized_vol.py
├── test_divergence.py
├── test_fis_sizer.py
└── test_engine_lifecycle.py
```

---

## Phase 1: Create Foundation (DTOs + Analytics)

### 1.1 Create `core/types.py`
```python
@dataclass(frozen=True)
class MarketSnapshot:
    ts: datetime
    symbol: str
    spot: float
    bars: pd.DataFrame
    option_chain: pd.DataFrame
    vix: Optional[float] = None
    es_price: Optional[float] = None
```

**Migration**: Extract from `backtest_engine.py` implicit state

### 1.2 Create `analytics/` Module
| File | Migrates From | Purpose |
|------|---------------|---------|
| `realized_vol.py` | `backtest_engine.compute_realized_vol()` | RV calculator |
| `divergence.py` | [NEW] | SPY-ES Z-score |
| `skew.py` | `options_strategy.nearest_by_delta()` skew logic | IV skew metric |
| `gaps.py` | [NEW] | Gap classification |

**Action**: Move existing functions, add new ones per spec

---

## Phase 2: Refactor Strategy Layer

### 2.1 Slim Down `strategies/iron_condor.py`
**Current**: 427 lines mixing strike selection, sizing hints, regime logic  
**Target**: ~150 lines - **signal gating only**

```python
class IronCondorStrategy:
    def evaluate(self, snapshot: MarketSnapshot) -> TradeDecision:
        """
        Signal gating: VRP check, divergence filter, gap logic.
        Returns TradeDecision (should_trade, bias, rationale).
        Does NOT size the trade.
        """
```

**Migration**:
- ✅ Keep: `build_condor()`, `nearest_by_delta()`, strike selection
- ❌ Remove: Sizing logic → move to `intelligence/fis_sizer.py`
- ❌ Remove: Regime width adjustment → move to strategy config

### 2.2 Extract `strategies/options_strategy.py` → Multiple Files
- `OptionQuote`, `IronCondorLegs` → `core/types.py`
- `build_condor()` → Keep in `iron_condor.py`
- `regime_wing_width()` → Delete (replaced by FIS)

---

## Phase 3: Build Intelligence Layer (FIS)

### 3.1 Create FIS Pipeline
```
intelligence/
├── fis_sizer.py         → Orchestrator
├── fuzzifier.py         → ADX/RSI/IVR → memberships
├── inference_engine.py  → Rule base (IF-THEN)
├── defuzzifier.py       → Centroid → confidence [0,1]
```

### 3.2 Migrate Existing Fuzzy Logic
**Current**: `fuzzy_engine.py` has membership functions but no inference  
**Target**: Full Mamdani pipeline

| Current Function | New Location | Notes |
|------------------|--------------|-------|
| `calculate_rsi_membership()` | `fuzzifier.py` | Keep as-is |
| `calculate_adx_membership()` | `fuzzifier.py` | Keep as-is |
| `compute_position_size()` | `fis_sizer.py` | Refactor to use FIS |

**New Logic Needed**:
- `MamdaniEngine.infer()`: Rule base (e.g., "IF ADX low AND RSI neutral → Aggressive")
- `Defuzzifier.centroid()`: Convert aggregated membership → scalar

---

## Phase 4: Refactor Data Layer

### 4.1 Create `data_factory/data_engine.py`
**Purpose**: Unified `MarketSnapshot` provider

```python
class DataEngine:
    def stream(self) -> Iterable[MarketSnapshot]:
        """Yield time-ordered snapshots (backtest or live)."""
```

**Migration**:
- Extract from `backtest_engine.next()` loop
- Integrate `sync_engine.py` (MTF) as a sub-provider
- Add `aux_feeds.py` for VIX/ES

### 4.2 Abstract Option Chain Loading
**Current**: Hardcoded CSV path in `backtest_engine.py`  
**Target**: `OptionChainProvider` interface

```python
class OptionChainProvider:
    def get_chain(self, ts, symbol) -> pd.DataFrame:
        # Backtest: read from synthetic CSV
        # Live: call broker API
```

---

## Phase 5: Enhance Risk Layer

### 5.1 Extend `risk_manager.py`
**Current**: Basic Greek checks  
**Target**: Full approval gate per spec

```python
class RiskManager:
    def approve(self, sized: SizedDecision, snapshot: MarketSnapshot) -> Approval:
        # 1. Expected Shortfall (CVaR)
        # 2. Beta-weighted delta
        # 3. POT monitor
        # 4. Structure validator
```

### 5.2 Create New Risk Modules
| Module | Purpose | Key Function |
|--------|---------|--------------|
| `expected_shortfall.py` | CVaR calculator | `es(pnl, alpha)` |
| `beta_weighting.py` | Portfolio delta normalization | `spy_equiv_delta()` |
| `pot_monitor.py` | Probability of touch | `pot(delta)` |
| `structure_validator.py` | Invariant enforcement | `clamp_quantity()` |

---

## Phase 6: Create Orchestration Layer

### 6.1 Create `core/engine.py`
**Purpose**: Replace monolithic `backtest_engine.py` loop

```python
class TradingEngine:
    def run(self):
        for snapshot in self.data.stream():
            decision = self.strategy.evaluate(snapshot)
            if not decision.should_trade:
                continue
            
            sized = self.sizer.size(decision, snapshot)
            approved = self.risk.approve(sized, snapshot)
            
            if approved.approved:
                self.router.execute(approved.order_plan, snapshot)
```

**Migration**:
- Extract from `IronCondorStrategy.next()` loop
- Keep `backtest_engine.py` for Backtrader compatibility (legacy)
- New `engine.py` for clean architecture

---

## Phase 7: Testing Infrastructure

### 7.1 Create `tests/` Folder
Per traceability matrix:

```
tests/
├── test_realized_vol.py          → RV window calculation
├── test_divergence.py            → Z-score bounds
├── test_skew.py                  → Skew penalty
├── test_gaps.py                  → Gap threshold
├── test_fis_sizer.py             → FIS monotonicity
├── test_expected_shortfall.py    → ES tail mean
├── test_beta_weighting.py        → Beta delta
├── test_structure_validator.py   → Min floor clamping
└── test_engine_lifecycle.py      → End-to-end smoke test
```

### 7.2 Test Template (Pytest)
```python
# tests/test_realized_vol.py
from analytics.realized_vol import RealizedVolCalculator

def test_realized_vol_window():
    calc = RealizedVolCalculator()
    prices = [100, 101, 100.5, 102, 101.5]
    rv = calc.compute_realized_vol(prices, window=4)
    assert 0.0 < rv < 1.0  # Sanity check
```

---

## Migration Strategy

### Option 1: Parallel Development (Recommended)
1. **Keep `backtest_engine.py` working** (for Stage 3 verification)
2. **Build new architecture in parallel** (`core/engine.py`, `analytics/`, etc.)
3. **Gradual cutover**: Test new modules, then swap in `main.py`

### Option 2: In-Place Refactor (Risky)
1. Refactor `backtest_engine.py` directly
2. Risk: Breaking existing functionality during migration

**Recommendation**: **Option 1** - Build new, test thoroughly, then cutover

---

## Immediate Next Steps

### Step 1: Create DTOs (1 hour)
- [ ] Create `core/types.py` with all dataclasses
- [ ] Update imports in existing files

### Step 2: Create Analytics Module (2 hours)
- [ ] Create `analytics/` folder
- [ ] Migrate `compute_realized_vol()` → `realized_vol.py`
- [ ] Create `divergence.py`, `skew.py`, `gaps.py` per spec
- [ ] Write unit tests for each

### Step 3: Create FIS Pipeline (3 hours)
- [ ] Create `intelligence/fis_sizer.py` orchestrator
- [ ] Create `fuzzifier.py` (migrate existing membership functions)
- [ ] Create `inference_engine.py` (new rule base)
- [ ] Create `defuzzifier.py` (centroid logic)
- [ ] Write FIS integration test

### Step 4: Create Risk Modules (2 hours)
- [ ] Create `risk/expected_shortfall.py`
- [ ] Create `risk/beta_weighting.py`
- [ ] Create `risk/pot_monitor.py`
- [ ] Create `risk/structure_validator.py`
- [ ] Integrate into `risk_manager.py`

### Step 5: Build Orchestration (2 hours)
- [ ] Create `core/engine.py`
- [ ] Create `data_factory/data_engine.py`
- [ ] Wire up full pipeline
- [ ] Run 1-day backtest smoke test

---

## Success Criteria

✅ **Architecture**:
- Clean module separation (no circular imports)
- All logic testable in isolation
- DTOs decouple modules

✅ **Functionality**:
- Backtest produces same results as before
- All unit tests pass
- Traceability matrix complete

✅ **Maintainability**:
- Each module < 200 lines
- Clear docstrings (NumPy style)
- Type hints on all public APIs

---

## Estimated Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 (DTOs + Analytics) | 3 hours | `core/types.py`, `analytics/*` |
| Phase 2 (Strategy Refactor) | 2 hours | Slim `iron_condor.py` |
| Phase 3 (FIS Pipeline) | 4 hours | `intelligence/fis_*` |
| Phase 4 (Data Layer) | 2 hours | `data_factory/data_engine.py` |
| Phase 5 (Risk Layer) | 3 hours | `risk/*` modules |
| Phase 6 (Orchestration) | 2 hours | `core/engine.py` |
| Phase 7 (Testing) | 4 hours | Full test suite |
| **Total** | **20 hours** | Production-ready system |

---

## Decision Point

**Proceed with Phase 1 now?**
- Create `core/types.py`
- Create `analytics/` folder with initial modules
- Write first unit tests

This establishes the foundation without breaking existing code.
