# Quantor-MTFuzz™ Development Roadmap

## Executive Summary

Quantor-MTFuzz is a specialized SPY Iron Condor system designed to unify high-fidelity backtesting, parameter optimization, and live/paper execution under a single, disciplined workflow. The core design emphasizes realistic mark-to-market accounting, multi-timeframe signal filtering, and dynamic position sizing to control risk while maintaining operational readiness for live deployment.

### Architecture Overview

The current architecture cleanly separates responsibilities:
- `main.py` — Orchestrates runs
- `backtest_engine.py` — High-resolution simulation
- `optimizer.py` — Phased grid search
- `options_strategy.py` — Strategy logic and trade management

Intelligence modules (fuzzy sizing and regime filters) provide higher-order context, while data modules support both synthetic and live workflows. This structure is consistent with production-ready trading research platforms.

### Current State: "Research Ready" + "Paper Ready"

**Key Strengths:**
- Leg-accurate backtest engine with MTM P&L
- Coherent logging and visual reporting
- Serial optimizer with caching and CSV top-100 outputs
- Live/paper execution via Alpaca integration
- Documented math foundations and operational workflow

---

## Staged Improvement Plan

### Stage 0 — Baseline & Safety
| Task | File(s) | Status |
|------|---------|--------|
| Benchmark runner (10k bars/sec) | `benchmark_cpu.py` | ✅ |
| Baseline metrics to JSON | `backtest_engine.py` | ✅ |
| Config sanity checks | `main.py` | ✅ |
| Document benchmark in README | `README.md` | ✅ |

### Stage 1 — Data & Market Realism
| Task | File(s) | Status |
|------|---------|--------|
| Add `risk_free_rate`/`iv_annual_volatility` to RunConfig | `config.template.py` | ✅ |
| Realized volatility from 5m bars | `backtest_engine.py` | ✅ |
| IV Rank proxy (rolling percentile) | `backtest_engine.py` | ✅ |
| Bid/ask slippage model | `backtest_engine.py` | ✅ |
| Commission model (per contract) | `backtest_engine.py` | ✅ |

### Stage 2 — Strategy Depth
| Task | File(s) | Status |
|------|---------|--------|
| Skew/term-structure strike selection | `options_strategy.py` | ✅ |
| IV skew penalty in `nearest_by_delta` | `options_strategy.py` | ✅ |
| Probabilistic entry filter (breach probability) | `options_strategy.py` | ✅ |
| Regime classifier (trend vs mean-reversion) | `fuzzy_engine.py` | ✅ |
| Conditional wing width based on regime | `options_strategy.py` | ✅ |

### Stage 3 — Risk & Portfolio Controls
| Task | File(s) | Status |
|------|---------|--------|
| Portfolio Greeks tracking | `core/risk_manager.py` | ✅ |
| Risk caps (delta/gamma/vega limits) | `core/config.py`, `core/risk_manager.py` | ✅ |
| Risk budget sizing | `strategies/options_strategy.py` | ✅ |
| Stress test runner (Historical Shocks) | `scripts/backtest/run_stress_test.py` | ✅ |

### Stage 4 — Performance & Compute
| Task | File(s) | Status |
|------|---------|--------|
| Parallelize optimizer (multiprocessing) | `optimizer.py` | ⬜ |
| Nested dict chain indexing | `backtest_engine.py` | ⬜ |
| Pre-indexed lookups in `build_condor` | `options_strategy.py` | ⬜ |
| Optional JIT/Numba for PnL loop | `backtest_engine.py` | ⬜ |

### Stage 5 — Reporting & Validation
| Task | File(s) | Status |
|------|---------|--------|
| PDF report: DTE/credit/regime distributions | `backtest_engine.py` | ⬜ |
| `test_pnl.py` — PnL/DD sanity tests | `tests/` | ⬜ |
| `test_strategy_filters.py` — Filter tests | `tests/` | ⬜ |
| Regression dataset (fixed seed) | `data/`, `tests/` | ⬜ |

---

## Performance & Sophistication Improvements

### Data/Market Realism
- Replace stub IV/VIX with realized measures from price/vol data
- Add slippage/commission and bid-ask modeling
- Use American option pricing (binomial/LSM) for early exercise effects

### Advanced Trading Techniques
- Skew/term-structure aware wing selection
- Portfolio-level Greeks and risk caps (delta/gamma/vega limits)
- Dynamic hedging rules beyond delta (vanna/volga)

### Mathematics/Physics-Inspired Modeling
- Stochastic volatility (Heston/SABR) or local volatility calibration
- Jump diffusion or regime-switching models (OU regime for volatility)
- Kalman filter or particle filter for latent regime detection

### Compute Optimization
- Parallelize grid search (multiprocessing/joblib)
- Pre-index option chains by date/strike/expiry
- Vectorize or JIT (Numba) hot loops

---

## Roadmap Flowchart

![Quantor-MTFuzz Roadmap: Completed vs Pending](diagrams/roadmap_flowchart.png)

---

## Completed Milestones ✅

| Category | Task | Status |
|----------|------|--------|
| Backtesting | Leg-based PositionState structure | ✅ |
| Backtesting | Synthetic options integration | ✅ |
| P&L | Mark-to-market calculation | ✅ |
| Trade Mgmt | Individual leg exit logic | ✅ |
| Diagnostics | Entry/exit logging | ✅ |
| Reporting | PDF visualizations | ✅ |
| Optimization | Grid search (NP/DD ratio) | ✅ |
| Performance | CPU benchmark | ✅ |
| Stability | Backtrader crash fix | ✅ |
| Metrics | Sharpe, Profit Factor, Expectancy | ✅ |
| Broker | Alpaca paper trading | ✅ |
| Docs | README, CONTRIBUTING, CLAUDE.md | ✅ |
| Testing | 6 regression tests for position sizing | ✅ |
