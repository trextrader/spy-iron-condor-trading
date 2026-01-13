# Implementation Tasks - Quantor-MTFuzz Architecture

## Stage 0 — Baseline & Safety ✅
- ✅ Benchmark runner (save runtime per 10k bars)
- ✅ Baseline metrics to JSON after backtest
- ✅ Config sanity checks in main.py
- ✅ Document benchmark in README

## Stage 1 — Data & Market Realism ✅
- ✅ Add risk_free_rate/iv_annual_volatility to RunConfig
- ✅ Realized volatility from 5m bars
- ✅ IV Rank proxy (rolling percentile)
- ✅ Bid/ask slippage model
- ✅ Commission model (per contract)

## Stage 2 — Strategy Depth ✅
- ✅ Skew/term-structure strike selection
- ✅ IV skew penalty in nearest_by_delta
- ✅ Probabilistic entry filter
- ✅ Regime classifier
- ✅ Conditional wing width

## Stage 3 — Risk & Portfolio Controls (In Progress)
- ✅ Add Greeks to OptionQuote (gamma, vega, theta)
- ✅ Create RiskManager class
- ✅ Integrated RiskManager into backtest loop
- ✅ Portfolio Greeks tracking (aggregation)
- ✅ Risk caps (delta/gamma/vega limits)
- ✅ Risk budget sizing (Handled by FIS Sizer)
- ✅ Stress test runner (scripts/backtest/run_stress_test.py)

# Quantor-MTFuzz Architectural Refactoring

## Phase 1 — DTOs & Analytics Foundation ✅
- ✅ Create core/types.py (MarketSnapshot, TradeDecision, SizedDecision, Approval, OrderPlan)
- ✅ Create analytics/realized_vol.py (RV calculator)
- ✅ Create analytics/divergence.py (SPY-ES Z-score)
- ✅ Create analytics/skew.py (IV skew metric)
- ✅ Create analytics/gaps.py (Gap analyzer)
- ✅ Create analytics/carry_model.py (Cost-of-carry)
- ✅ Create analytics/indicators.py (ADX/RSI/IV Rank)
- ✅ Create tests/ folder with pytest infrastructure
- ✅ Write unit tests (18 tests passing)

## Phase 2 — Pipeline Trace & Data Factory ✅
**Data Pipeline Components**
- ✅ data_factory/spot_bars.py - Multi-timeframe OHLCV provider (1/5/15m)
- ✅ data_factory/option_chain.py - Options chain with Greeks
- ✅ data_factory/aux_feeds.py - Gap analysis helpers
- ✅ data_factory/data_engine.py - MarketSnapshot streaming

**Intelligence Integration**
- ✅ intelligence/fuzzifier.py - Real ADX/RSI/IV Rank extraction

**Stub Trading Engine**
- ✅ core/engine.py - TradingEngine with stub components

**Trace Harness**
- ✅ run_engine_trace_one_day.py - Pipeline trace with diagnostics

**Testing & Validation**
- ✅ 79 bars processed successfully (2025-07-03)
- ✅ 624 options per snapshot
- ✅ Complete pipeline: DataEngine → Strategy → Sizer → Risk → Router

## Phase 2.5 — Lag-Aware Alignment System ✅
**Configuration Infrastructure**
- ✅ Added lag alignment config to RunConfig/StrategyConfig
- ✅ Per-symbol lag limits (SPY: 600s, QQQ: 600s, SPX: 900s)
- ✅ IV decay half-life configuration (300s default)
- ✅ Fail-fast thresholds

**ChainAlignment Engine**
- ✅ ChainAlignment dataclass (chain, used_ts, mode, lag_sec, iv_conf)
- ✅ IV confidence decay: 0.5^(lag_sec / half_life)
- ✅ Policy modes: hard_cutoff / decay_only / decay_then_cutoff
- ✅ Stale detection with empty chain handling

**DataEngine Enhancements**
- ✅ Auto-overlap day selection (spot ∩ options dates)
- ✅ Strategy-level alignment policy integration
- ✅ Fail-fast mode (abort if stale rate > 20%)
- ✅ Comprehensive diagnostics tracking

**MarketSnapshot Metadata**
- ✅ option_used_ts, option_lag_sec, option_iv_conf, option_align_mode

**Strategy Integration**
- ✅ alignment_policy() method for strategy overrides
- ✅ lag_weighted_edge() function for VRP adjustment
- ✅ IV edge *= iv_conf (multiply mode)

**Architecture Documentation**
- ✅ system_overview.dot - Full system architecture
- ✅ lag_alignment_flow.dot - Alignment flowchart
- ✅ data_pipeline_detailed.dot - CSV → Decision pipeline
- ✅ PNG visualizations generated

**Validation**
- ✅ Trace harness updated with alignment metadata display
- ✅ All code committed and pushed to GitHub

## Phase 3 — Strategy Refactor (Planned)
- ✅ Slim down strategies/iron_condor.py to signal gating only (Done: Refactored to generate_trade_signal)
- ✅ Move OptionQuote, IronCondorLegs to core/types.py (Done: core/dto.py)
- ✅ Remove sizing logic (move to FIS in Phase 4) (Done: Logic migrated to intelligence/fis_sizer.py)
- [ ] Integrate analytics.* modules into strategy

## Phase 4 — FIS Pipeline (Planned)
- ✅ Create intelligence/fis_sizer.py (orchestrator)
- ✅ Enhance intelligence/fuzzifier.py (Verified: Existing implementation robust)
- ✅ Create intelligence/inference_engine.py (Mamdani/Additive rule base)
- ✅ Create intelligence/defuzzifier.py (centroid/volatility scaling)
- ✅ Migrate existing fuzzy logic from fuzzy_engine.py (Completed)
- [ ] Write FIS integration tests

## Phase 5 — Risk Layer Enhancement (Completed)
- ✅ Create risk/expected_shortfall.py (CVaR)
- ✅ Create risk/beta_weighting.py (portfolio delta)
- ✅ Create risk/pot_monitor.py (probability of touch)
- ✅ Create risk/structure_validator.py (invariants)
- ✅ Refactor risk_manager.py to use new modules

## Phase 6 — Full Orchestration (Planned)
- [ ] Enhance core/engine.py (complete TradingEngine)
- [ ] Create core/trade_router.py (order execution)
- [ ] Create core/mode.py (RunMode enum)
- [ ] Wire up full pipeline (data → strategy → FIS → risk → router)
- [ ] Run multi-day backtest validation

## Phase 7 — Testing & Validation (Planned)
- [ ] Complete all unit tests per traceability matrix
- [ ] End-to-end integration tests
- [ ] Regression tests with golden datasets
- [ ] Performance benchmarking

## Stage 4 — Advanced Strategy Expansion (Future)
- [ ] Create CalendarStrategy (Low Vol)
- [ ] Create BrokenWingButterflyStrategy (Skew/Bias)
- [ ] Implement StrategySelector logic
- [ ] Divergence Monitor (SPY vs ES)

## Stage 5 — Neuro-Fuzzy Integration (Future)
- [ ] Connect Fuzzy Sizing to Trade Execution
- [ ] Optimize Membership Functions
- [ ] Mamba Model Training Pipeline
- [ ] Full System Integration Tests
