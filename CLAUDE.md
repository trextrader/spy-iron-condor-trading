# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantor-MTFuzz is a high-fidelity algorithmic trading system for SPY Iron Condor options strategies. It uses a Dual-Data Engine architecture that separates strategy signals (from SPY 5-minute bars) from option pricing (from synthetic Black-Scholes options data) to ensure realistic backtesting.

## Essential Commands

### Running Backtests
```powershell
# Full-year high-fidelity backtest with MTF and dynamic sizing
python core/main.py --mode backtest --use-mtf --dynamic-sizing --bt-samples 0

# Quick backtest with 500 bars (default)
python core/main.py --mode backtest --use-mtf --dynamic-sizing

# Debug: Test MTF impact (No Fuzzy)
python core/main.py --mode backtest --use-mtf --no-fuzzy

# Debug: Test Fuzzy impact (No MTF)
python core/main.py --mode backtest --use-fuzzy --dynamic-sizing --no-mtf-filter

# Custom date range backtest
python core/main.py --mode backtest --bt-start 2024-01-01 --bt-end 2024-06-30
```

### Running Optimization
```powershell
# Run phased serial optimization with hardware benchmarking
python core/main.py --mode backtest --use-mtf --dynamic-sizing --bt-samples 0 --use-optimizer
```
- Outputs timestamped CSV reports to `reports/top100_YYYYMMDD_HHMMSS.csv`
- Sorted by Net Profit / Max Drawdown ratio (higher is better)
- **5-Phase Matrix**: Edit `OPTIMIZATION_SEGMENTS` in `core/optimizer.py:17` to modify search space:
  1. Exits & Risk
  2. Structure & Entries
  3. Filters & Regime
  4. Momentum Logic
  5. Trend & Volatility (PSAR)

### Live Paper Trading
```powershell
# Alpaca paper trading
python core/main.py --mode live --alpaca --alpaca-key YOUR_KEY --alpaca-secret YOUR_SECRET --polygon-key YOUR_KEY

# Local paper simulation
python core/main.py --mode live --polygon-key YOUR_KEY
```

### Data Generation
```powershell
# Generate synthetic options chains (Black-Scholes)
python data_factory/SyntheticOptionsEngine.py

# Fetch historical SPY data (Alpaca)
python data_factory/AlpacaGetData.py
```

## Architecture & Design Patterns

### Dual-Data Engine (Critical Concept)
The system uses two separate data sources that must be understood together:

1. **Strategy Clock** (`data/spot/SPY_1.csv`, `SPY_5.csv`, `SPY_15.csv`)
   - Drives entry/exit timing via Multi-Timeframe technical indicators
   - Provides spot price for delta-based strike selection
   - Never modify timeframes without regenerating MTF sync data

2. **Quote & Execution Layer** (`data/synthetic_options/spy_options_marks.csv`)
   - Provides theoretical option prices (Black-Scholes model)
   - Used for mark-to-market P&L calculations
   - Contains bid/ask/mid/Greeks for each strike/expiration

## Technology Stack
- **Backtest**: `backtrader` (Custom customized), `matplotlib` (Reporting)
- **Data**: `pandas`, `numpy`, `polygon-api-client`
- **Technical Analysis**: `pandas-ta` (Indicators: RSI, ADX, BBands, Stoch, etc.)
- **Intelligence**: 
  - **Fuzzy Logic**: 10-Factor Inference Engine (`qtmf.facade`)
  - **Neural**: Mamba 2 State-Space Model (`intelligence.mamba_engine`)

## Architecture
- **Core**: `main.py` -> `RunConfig` -> `BacktestEngine`
- **Data Layer**: `MTFSyncEngine` (1m, 5m, 15m) + `SyntheticOptionsEngine` (Pricing)
- **Intelligence Layer**: 
  - `qtmf/`: Central Neuro-Fuzzy Facade (Adaptive Credit Logic + 10-Factor Filters)
  - `intelligence/mamba_engine.py`: Neural Market State Forecasting (Truthful Mamba Backend: Mock/Real)
  - `intelligence/fuzzy_engine.py`: Membership Functions
- **Strategy**: `ZeroDTE_IC` (Iron Condor) located in `core/backtest_engine.py`
   - Provides theoretical option prices (Black-Scholes model)
   - Used for mark-to-market P&L calculations
   - Contains bid/ask/mid/Greeks for each strike/expiration

**Key Insight**: The backtest engine loads both datasets into memory at the start of optimization runs to avoid 10x performance penalty from repeated disk I/O.

### 4-Leg Iron Condor Standard
All trades MUST have exactly 4 legs (Short Call, Long Call, Short Put, Long Put). The entry logic in `strategies/options_strategy.py:101` (`build_condor`) enforces a fail-fast rule: if any leg cannot be constructed with valid pricing at the target delta, the entire trade is rejected. Never modify this to allow partial positions.

### Configuration Split Architecture
- `StrategyConfig` (`core/config.py:6`): Trading logic parameters (DTE, deltas, exits, filters)
- `RunConfig` (`core/config.py:61`): Environment settings (API keys, backtest dates, plotting flags)

This split allows the optimizer to modify strategy parameters while preserving runtime environment settings.

### Options Data Lookup Pattern
When working with options data in `core/backtest_engine.py`, the pattern is:
```python
# Pre-grouped options by date for O(1) lookup
options_by_date = {}
for date, group in options_df.groupby('date'):
    options_by_date[date.date()] = group.to_dict('records')

# Later in strategy loop:
chain = options_by_date.get(current_date, [])
```
Never use linear scans through the full options dataframe during backtest loops.

### MTF Sync Engine Caching
The `MTFSyncEngine` (`data_factory/sync_engine.py:5`) loads all timeframes (1m, 5m, 15m) into memory once and provides `get_snapshot(current_time)` for temporal alignment. When modifying backtest logic, always check if `preloaded_sync` is passed to avoid redundant disk reads during optimization.

## Critical Files & Interdependencies

### Entry Point Chain
1. `core/main.py:113` - Parses CLI args and dispatches to backtest/live/optimizer
2. If `--use-optimizer`: calls `core/optimizer.py:35` (`run_optimization`)
3. Otherwise: calls `core/backtest_engine.py:23` (`run_backtest_headless` or `run_backtest_and_report`)

### Strategy Execution Flow
1. `core/backtest_engine.py:81` - Defines `IronCondorStrategy(bt.Strategy)`
2. Strategy delegates to `strategies/options_strategy.py:101` (`build_condor`) for leg construction
3. Strike selection uses `strategies/options_strategy.py:63` (`nearest_by_delta`)
4. Exit logic in backtest engine checks profit/loss thresholds from `StrategyConfig.profit_take_pct` and `loss_close_multiple`

### Data Factory Pipeline
- `data_factory/AlpacaGetData.py` → Downloads raw 5-minute bars → Saves to `data/spot/`
- `data_factory/SyntheticOptionsEngine.py` → Reads spot prices → Generates option chains using Black-Scholes → Saves to `data/synthetic_options/`
- `data_factory/sync_engine.py` → Loads multi-timeframe data for aligned lookups

### Intelligence Modules
- `intelligence/fuzzy_engine.py` - Implements 10-factor Fuzzy Logic Position Sizing:
  - **Core**: MTF Consensus, IV Rank, VIX Regime
  - **Momentum**: RSI (40-60 neutral), Stochastic (30-70 neutral)
  - **Trend**: ADX (<25 weak trend), SMA Distance (<2%), **Parabolic SAR** (crossover detection)
  - **Volatility**: Bollinger Bands (Squeeze & Range)
  - **Volume**: Relative Volume Ratio (>0.8)
- `data_factory/sync_engine.py` - Calcluates all indicators during data load (autodetects `pandas-ta`, falls back to robust manual calculation if missing).
- `intelligence/regime_filter.py` - Prevents entry during high VIX or low IV Rank conditions.

## Code Modification Guidelines

### When Editing Optimizer
The optimization matrix in `core/optimizer.py:18` uses **inclusive ranges**:
```python
"profit_take_pct": np.arange(0.50, 1.4, 0.2)  # 0.5, 0.7, 0.9, 1.1, 1.3
"loss_close_multiple": np.arange(1.0, 5.0, 0.2)  # 1.0, 1.2, ..., 4.8
```
The `stop + step` pattern ensures the endpoint is included. Never use `np.arange(start, stop, step)` without adding step to stop.

### Backtrader State Evaluation Bug
**Never** use `if strat:` to check if a Backtrader strategy object exists. This triggers AttributeError due to Backtrader's line-logic internals. Always use:
```python
if strat is not None:  # Correct
```
See `DEVELOPMENT_STATUS.md:9` for historical context.

### Price Cache for "N/A" Handling
The backtest engine implements a "Last Known Price" cache to handle intermittent gaps in synthetic options data. When modifying option pricing logic, ensure you don't break this fallback mechanism in `core/backtest_engine.py`.

### Timezone Normalization
All timestamps from CSVs must be converted to timezone-naive datetimes:
```python
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
```
This prevents look-ahead bias in MTF alignment. See `DEVELOPMENT_STATUS.md:13` for bug history.

## Performance Optimization Notes

### In-Memory Data Caching
The optimizer pre-loads dataframes once and passes them via `preloaded_df`, `preloaded_options`, `preloaded_sync` to avoid 10x slowdown. When adding new backtest parameters, ensure they don't trigger additional disk reads inside the optimization loop.

### Hardware Benchmarking
The optimizer runs a baseline backtest before grid search to estimate total runtime. This benchmark duration is stored in `baseline_duration` and multiplied by the number of combinations. Don't modify this without updating the time estimation logic in `core/optimizer.py:71`.

## Key Mathematical Invariants

### Strike Selection
Strikes are chosen by minimizing `|market_delta - target_delta|` for probability alignment. The target delta band is `[target_short_delta_low, target_short_delta_high]` (default 0.15-0.20).

### P&L Calculation
```
PnL(t) = (Credit_received - Current_cost) × Quantity × 100
Current_cost = (Short_call_mid - Long_call_mid) + (Short_put_mid - Long_put_mid)
```

### Optimizer Objective
Maximizes `Net Profit / Max Drawdown` ratio. Results in `top100_*.csv` are sorted descending by this metric.

### Sharpe Ratio
Annualized using 5-minute bars: `Sharpe = sqrt(19,656) × mean(returns) / std(returns)` where 19,656 = 252 trading days × 78 bars/day.

## Configuration Management

### Applying Optimized Parameters
After an optimization run:
1. Open `reports/top100_YYYYMMDD_HHMMSS.csv`
2. Identify Rank 1 (best risk-adjusted performance)
3. Update `core/config.py` StrategyConfig defaults with Rank 1 values
4. Also update `core/config.template.py` for version control

Current applied parameters (as of last optimization):
- `profit_take_pct = 0.9`
- `loss_close_multiple = 1.2`
- `iv_rank_min = 20.0`
- `vix_threshold = 25.0`

### API Key Setup
1. Copy `core/config.template.py` to `core/config.py`
2. Replace placeholder strings in RunConfig:
   - `alpaca_key`
   - `alpaca_secret`
   - `polygon_key`

## Windows-Specific Notes

All command examples use PowerShell syntax. The codebase uses `os.path.join()` for cross-platform path handling, so Unix-style paths in code are automatically converted.

## Development Status

The system is "Live-Ready" for paper trading. See `DEVELOPMENT_STATUS.md` for:
- Resolved bugs (optimizer bottlenecks, Backtrader crashes, timezone issues)
- Remaining backlog (slippage modeling, delta hedging v2, CI/CD pipeline)

## Testing Philosophy

This codebase uses a hybrid validation approach:

### Regression Tests
```bash
py -3.12 -m pytest tests/test_iron_condor_sizing.py -v
```
- **6 tests** validate Iron Condor position sizing invariants
- Prevents the "1-lot failure" bug from returning (minimum 2 contracts enforced)

### Integration Validation
1. Full-year backtests with known data (`--bt-samples 0`)
2. Optimizer regression checks (comparing Rank 1 metrics across runs)
3. Alpaca smoke tests (unauthorized response = connectivity verified)

When making changes, always run a full backtest and compare metrics to baseline before committing.


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
