# Project Tasks & Roadmap

## ✅ Completed Milestones

### Core Architecture
- [x] **Backtest Engine**: 5-minute bar simulation, slippage, trade logging.
- [x] **Event Loop**: `precompute_all` for fast batch processing.
- [x] **Fuzzy Logic**: 10-factor position sizing system.

### Neural Intelligence (Mamba 2)
- [x] **Model Architecture**: Integrated `DeepMamba` (SSM) with Mamba 2 blocks.
- [x] **GPU Acceleration**: Implemented batch inference (`precompute_all`) to saturate T4/A100 GPUs.
- [x] **Training Pipeline**: Created `intelligence/train_mamba.py` for end-to-end training.
- [x] **Data Feeds**: Added Alpaca 15-Minute Intraday fetcher (`--save-only` workflow).
- [x] **Inference**: Configurable `d_model`/`layers` for model sizing (256/12 vs 1024/32).

### Optimizer
- [x] **Segmentation**: 6-Phase serial optimization strategy.
- [x] **Phase 6**: Fuzzy Weight Optimization (Neural, MTF, IV, Regime, RSI, ADX).
- [x] **Metrics**: Net Profit / Max Drawdown targeting.
- [x] **Integration**: Seamless loading of `models/mamba_active.pth`.
- [x] **Performance**: Optimized backtest loop (300x speedup via deduplication).
- [x] **Critical Fixes**: Resolved "Zero Trades" bug, implemented explicit Expiration Exit logic, and fixed Python/Pandas timestamp synchronization.
- [x] **Intraday Support**: Full support for Alpaca-style intraday options data in the optimizer.

### Data Factory
- [x] **Intraday Expansion**: `expand_options_intraday.py` generates 1-min and 5-min synthetic options from daily ivolatility.
- [x] **Data Validator**: Auto-detects date column in various CSV formats.

## 🚧 In Progress

### Alpha Generation
- [ ] **Feature Engineering**: Add VIX Term Structure (Contango/Backwardation) to Mamba inputs.
- [ ] **Classification**: Switch Mamba target from "Log Return regression" to "Classify Regime (Bull/Bear)".
- [ ] **Hyperparameter Tuning**: Run a dedicated sweep for Mamba (Learning Rate, Dropout).

### Live Trading
- [ ] **Alpaca Execution**: Verify order routing in live paper account.
- [ ] **Latency**: Benchmark end-to-end tick-to-trade latency.

## 🔮 Future Roadmap
- **Portfolio Mode**: Trade SPY, QQQ, and IWM simultaneously.
- **Reinforcement Learning**: Use PPO to train the exit logic policy.
- **Hedge Fund Reporting**: Generate tear sheets with Sharpe/Sortino ratios.


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

---

## Side Burner Tasks (Deferred)

- [ ] **Dataset Pruning to V2.2 Schema**: Use `scripts/utils/prune_dataset_to_schema.py` to strip non-feature columns. Deferred for now to avoid disrupting pipelines.
