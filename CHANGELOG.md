# Changelog

## [v4.3.0] - 2026-02-24
### CondorNet™ v4.3 — Multi-Strategy Options Intelligence

Major architecture upgrade from v4.0 single-TF model to multi-TF + options chain intelligence.

#### Architecture (10.9M Parameters)
- **MultiTFProjector**: 4 independent TF inputs (M1/M5/M15/H1) `[B,T,64]` → `[B,T,256]` joint
- **PivotProjector**: 13 sparse pivot features → `[B,T,16]` dense embedding (NaN-masked)
- **TFFusionBlock**: Fuses TF joint + pivot projection with residual connection
- **OptionsChainEncoder**: Transformer over chain grid `[B,N,10]` → `[B,128]` (2L/4H, moneyness-ranked positional encoding, skew signal extraction)
- **JointFusionLayer**: `[B,T,256]` + `[B,128]` → `[B,T,384]` with LayerNorm
- **StrategyHead**: 10 strategy types (incl. abstain) + 4-leg params + entry signal
- **RiskMetricHead**: PoP, EV, MaxLoss, VaR₉₅, CVaR₉₅ (CVaR ≥ VaR enforced)
- **PivotPredictionHead**: Anticipates reversals at [5, 10, 20, 35, 70] bar horizons
- **PositionSizeHead**: PoP-blended fuzzy sizing
- **8 Predicate Gates** (up from 5): iv_rank, spread_ratio, rsi, delta_rsi, momentum_reversal, gap_risk, greeks_pressure, iv_regime_frac

#### Dataset v4.3
- 64 input features across 7 functional groups (V2.2 schema) + 9 IVR Reversal Engine features
- 13 sparse pivot features with NaN masking (medium + strong only, no weak)
- Full options chain grid (120 contracts max, 10 features each)
- Schema version `v4.3.0` with comprehensive validators (`schema_v43.py`)

#### Training
- Training Run 12 (latest): 31 epochs on T4, batch=256, lr=1e-4, lookback=200
- Loss convergence: train 2.24→0.82, val 1.38→0.97
- Best model at epoch 6 (val_loss=0.9574)
- Deep observation reports with 7-section JSON schema

#### New Files
- `intelligence/condor_brain_net_v43.py` — Full v4.3 model architecture
- `intelligence/condor_train_net_v43.py` — v4.3 training loop with deep observation
- `intelligence/schema_v43.py` — Feature schema, strategy types, validators
- `intelligence/predicate_discovery_v43.py` — Predicate learning and pruning

---

## [v2.3.0] - 2026-02-07
### Phase 6: CondorBrain GUI - Real-time Training Visualization

This release introduces the **CondorBrain GUI** - a full-stack dashboard for visualizing, configuring, and controlling the CondorNet training system.

#### GUI Features
- **Dashboard**: Real-time system overview with equity curves, metrics, and activity feed
- **Training Page**: Dedicated real-time training visualization with:
  - Live metric cards for all 12 loss components with sparklines
  - Streaming loss trajectory chart (Recharts)
  - Real-time fuzzy gate activation heatmap (Canvas)
  - Diagnostics panel (LR, gradient norm, scaler)
  - Training controls (Start/Stop simulation)
- **Model Introspection**: Post-training diagnostics with loss trajectories, fuzzy heatmaps, gate distributions, epoch summaries
- **WebSocket Real-time Updates**: Auto-reconnect with channel-based subscriptions
- **Lightning AI Support**: Auto-detection of cloudspace URLs for remote deployment

#### Training Enhancements
- **Epoch Checkpointing**: `--checkpoint-every N` saves checkpoint every N epochs
- **Environment-based Telemetry**: `--gui-telemetry local|lightai|kaggle|colab`
- **Diagnostics Export**: `--save-diagnostics` saves JSON for Model Introspection

#### New CLI Flags
```bash
# Training with GUI telemetry and checkpoints
python intelligence/condor_train_net.py \
  --config configs/condor_net_config_v46.yaml \
  --gui-telemetry lightai \
  --save-diagnostics \
  --checkpoint-every 1 \
  --checkpoint-dir models/checkpoints
```

#### Tech Stack
- **Backend**: FastAPI + Pydantic v2 + WebSockets
- **Frontend**: React 18 + TypeScript + TailwindCSS + Zustand + Recharts

---

## [v2.2.0] - 2026-01-16
### Major Enhancements: Advanced Model Architecture
This release introduces six advanced modules that significantly improve model expressivity, regime detection, and risk-adjusted optimization.

#### New Modules
- **CompositeCondorLoss** (`intelligence/condor_loss.py`): Multi-objective loss combining Huber prediction, Sharpe proxy, drawdown penalty, and turnover penalty. Expected +15-25% Sharpe improvement.
- **VolGatedAttn** (`intelligence/vol_gated_attn.py`): Dynamic volatility-gated attention inserted after CondorNet layers 7, 15, 23. Adapts receptive field based on market regime.
- **TopKMoE** (`intelligence/topk_moe.py`): Sparse mixture-of-experts with top-k routing. Activates only 1 of 3 experts per sample for 3x inference efficiency.
- **Manifold Volatility** (`intelligence/indicators/manifold_volatility.py`): Menger curvature proxy, volatility energy, and dynamic RSI features.
- **TDA Signature** (`intelligence/indicators/tda_signature.py`): Persistent homology regime detection using Takens embedding and H1 cycles.
- **Policy Outputs** (`intelligence/indicators/policy_outputs.py`): Measure-theoretic state discretization for interpretable Q-table policies.

#### Training CLI Enhancements
- `--composite-loss` flag with `--loss-lambdas` for custom weighting
- `--vol-gated-attn` (default: enabled) and `--no-vol-gated-attn`
- `--topk-moe` with `--moe-experts` and `--moe-k` parameters

#### Bug Fixes
- Fixed cuDNN GRU BF16 incompatibility in HorizonForecaster
- Fixed `return_experts` handling when TopKMoE is active
- Fixed `sample_predictions` null handling for MoE mode

#### Documentation
- Added Section 12 to `scientific_spec.md` with rigorous mathematical formulations
- Created `enhanced_architecture.dot` Graphviz diagram

---

## [v2.0.1] - 2026-01-11
### Features
- **Multi-timeframe support:** Added 1m/5m/15m support with automatic file selection.
- **Auto-overlap day selection:** Improved alignment diagnostics.
- **Lag-aware IV decay system:** Enhancing option pricing precision.

### Documentation
- **Architecture Diagrams:** Added comprehensive Graphviz diagrams in `docs/architecture/`.


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
