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
