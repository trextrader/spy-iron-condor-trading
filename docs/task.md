# Implementation Tasks - Quantor-MTFuzz Architecture

## Stage 0 — Baseline & Safety ✅
- [x] Benchmark runner (save runtime per 10k bars)
- [x] Baseline metrics to JSON after backtest
- [x] Config sanity checks in main.py
- [x] Document benchmark in README

## Stage 1 — Data & Market Realism ✅
- [x] Add `risk_free_rate`/`iv_annual_volatility` to RunConfig
- [x] Realized volatility from 5m bars
- [x] IV Rank proxy (rolling percentile)
- [x] Bid/ask slippage model
- [x] Commission model (per contract)

## Stage 2 — Strategy Depth ✅
- [x] Skew/term-structure strike selection
- [x] IV skew penalty in nearest_by_delta
- [x] Probabilistic entry filter
- [x] Regime classifier
- [x] Conditional wing width

## Stage 3 — Risk & Portfolio Controls (In Progress)
- [x] Add Greeks to `OptionQuote` (gamma, vega, theta)
- [x] Create `RiskManager` class
- [x] Integrated RiskManager into backtest loop
- [ ] Portfolio Greeks tracking (aggregation)
- [ ] Risk caps (delta/gamma/vega limits)
- [ ] Risk budget sizing
- [ ] Stress test runner

---

# Quantor-MTFuzz Architectural Refactoring

## Phase 1 — DTOs & Analytics Foundation ✅
- [x] Create `core/types.py` (MarketSnapshot, TradeDecision, SizedDecision, Approval, OrderPlan)
- [x] Create `analytics/realized_vol.py` (RV calculator)
- [x] Create `analytics/divergence.py` (SPY-ES Z-score)
- [x] Create `analytics/skew.py` (IV skew metric)
- [x] Create `analytics/gaps.py` (Gap analyzer)
- [x] Create `analytics/carry_model.py` (Cost-of-carry)
- [x] Create `analytics/indicators.py` (ADX/RSI/IV Rank)
- [x] Create `tests/` folder with pytest infrastructure
- [x] Write unit tests (18 tests passing)

## Phase 2 — Pipeline Trace & Data Factory ✅
- [x] Data Pipeline Components
  - [x] `data_factory/spot_bars.py` - Multi-timeframe OHLCV provider (1/5/15m)
  - [x] `data_factory/option_chain.py` - Options chain with Greeks
  - [x] `data_factory/aux_feeds.py` - Gap analysis helpers
  - [x] `data_factory/data_engine.py` - MarketSnapshot streaming
- [x] Intelligence Integration
  - [x] `intelligence/fuzzifier.py` - Real ADX/RSI/IV Rank extraction
- [x] Stub Trading Engine
  - [x] `core/engine.py` - TradingEngine with stub components
- [x] Trace Harness
  - [x] `run_engine_trace_one_day.py` - Pipeline trace with diagnostics
- [x] Testing & Validation
  - [x] 79 bars processed successfully (2025-07-03)
  - [x] 624 options per snapshot
  - [x] Complete pipeline: DataEngine → Strategy → Sizer → Risk → Router

## Phase 2.5 — Lag-Aware Alignment System ✅
- [x] Configuration Infrastructure
  - [x] Added lag alignment config to `RunConfig`/`StrategyConfig`
  - [x] Per-symbol lag limits (SPY: 600s, QQQ: 600s, SPX: 900s)
  - [x] IV decay half-life configuration (300s default)
  - [x] Fail-fast thresholds
- [x] ChainAlignment Engine
  - [x] `ChainAlignment` dataclass (chain, used_ts, mode, lag_sec, iv_conf)
  - [x] IV confidence decay: `0.5^(lag_sec / half_life)`
  - [x] Policy modes: hard_cutoff / decay_only / decay_then_cutoff
  - [x] Stale detection with empty chain handling
- [x] DataEngine Enhancements
  - [x] Auto-overlap day selection (spot ∩ options dates)
  - [x] Strategy-level alignment policy integration
  - [x] Fail-fast mode (abort if stale rate > 20%)
  - [x] Comprehensive diagnostics tracking
- [x] MarketSnapshot Metadata
  - [x] `option_used_ts`, `option_lag_sec`, `option_iv_conf`, `option_align_mode`
- [x] Strategy Integration
  - [x] `alignment_policy()` method for strategy overrides
  - [x] `lag_weighted_edge()` function for VRP adjustment
  - [x] IV edge *= iv_conf (multiply mode)
- [x] Architecture Documentation
  - [x] `system_overview.dot` - Full system architecture
  - [x] `lag_alignment_flow.dot` - Alignment flowchart
  - [x] `data_pipeline_detailed.dot` - CSV → Decision pipeline
  - [x] PNG visualizations generated
- [x] Validation
  - [x] Trace harness updated with alignment metadata display
  - [x] All code committed and pushed to GitHub

## Phase 3 — Strategy Refactor (Planned)
- [ ] Slim down `strategies/iron_condor.py` to signal gating only
- [ ] Move `OptionQuote`, `IronCondorLegs` to `core/types.py`
- [ ] Remove sizing logic (move to FIS in Phase 4)
- [ ] Integrate `analytics.*` modules into strategy

## Phase 4 — FIS Pipeline (Planned)
- [ ] Create `intelligence/fis_sizer.py` (orchestrator)
- [ ] Enhance `intelligence/fuzzifier.py` (full feature extraction)
- [ ] Create `intelligence/inference_engine.py` (Mamdani rule base)
- [ ] Create `intelligence/defuzzifier.py` (centroid)
- [ ] Migrate existing fuzzy logic from `fuzzy_engine.py`
- [ ] Write FIS integration tests

## Phase 5 — Risk Layer Enhancement (Planned)
- [ ] Create `risk/expected_shortfall.py` (CVaR)
- [ ] Create `risk/beta_weighting.py` (portfolio delta)
- [ ] Create `risk/pot_monitor.py` (probability of touch)
- [ ] Create `risk/structure_validator.py` (invariants)
- [ ] Refactor `risk_manager.py` to use new modules

## Phase 6 — Full Orchestration (Planned)
- [ ] Enhance `core/engine.py` (complete TradingEngine)
- [ ] Create `core/trade_router.py` (order execution)
- [ ] Create `core/mode.py` (RunMode enum)
- [ ] Wire up full pipeline (data → strategy → FIS → risk → router)
- [ ] Run multi-day backtest validation

## Phase 7 — Testing & Validation (Planned)
- [ ] Complete all unit tests per traceability matrix
- [ ] End-to-end integration tests
- [ ] Regression tests with golden datasets
- [ ] Performance benchmarking

---

## Stage 4 — Advanced Strategy Expansion (Future)
- [ ] Create `CalendarStrategy` (Low Vol)
- [ ] Create `BrokenWingButterflyStrategy` (Skew/Bias)
- [ ] Implement `StrategySelector` logic
- [ ] Divergence Monitor (SPY vs ES)

## Stage 5 — Neuro-Fuzzy Integration (Future)
- [ ] Connect Fuzzy Sizing to Trade Execution
- [ ] Optimize Membership Functions
- [ ] Mamba Model Training Pipeline
- [ ] Full System Integration Tests
