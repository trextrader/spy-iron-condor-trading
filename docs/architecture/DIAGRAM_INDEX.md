# Complete Diagram Index

This document provides a comprehensive index of all architecture diagrams in the CondorBrain system, organized by category.

---

## Quick Reference

| Category | Count | Key Diagrams |
|----------|-------|--------------|
| Master Maps | 4 | `complete_system_v23.png`, `system_overview.png` |
| GUI & Training | 5 | `gui_architecture.png`, `training_telemetry_flow.png` |
| Intelligence Core | 7 | `condornet_v43_architecture.png`, `CondorNet_NN_Architecture.png`, `enhanced_architecture.png` |
| Data Pipeline | 3 | `data_pipeline_detailed.png`, `lag_alignment_flow.png` |
| Decision Logic | 6 | `fuzzy_sizing_pipeline.png`, `membership_curves.png` |
| Execution | 2 | `institutional_execution_flow.png`, `optimization_pipeline.png` |
| Audit & Comparison | 2 | `condornet_learned_logic.pdf`, `audit_cde_comparison_flow.png` |

---

## Master Maps

### complete_system_v23.png (NEW)
**File:** [complete_system_v23.png](complete_system_v23.png)
**Source:** [complete_system_v23.dot](complete_system_v23.dot)
**Description:** Complete CondorBrain system architecture v2.3 showing all layers: Data, Intelligence (CondorNet v4.0 era; see `condornet_v43_architecture.png` for current v4.3), Training, Execution, and the GUI Layer (Phase 6).

### system_overview.png
**File:** [system_overview.png](system_overview.png)
**Source:** [system_overview.dot](system_overview.dot)
**Description:** High-level conceptual view of the Quantor-MTFuzz trading system with lag-aware alignment. Shows Data Ingestion, Intelligence, Analytics, and Execution layers.

### diagram_map.png
**File:** [diagram_map.png](diagram_map.png)
**Source:** [diagram_map.dot](diagram_map.dot)
**Description:** Master navigational map linking all diagrams organized by functional cluster (Data, Intelligence, Logic, Execution, Optimization).

### full_system_architecture.png
**File:** [full_system_architecture.png](full_system_architecture.png)
**Source:** [full_system_architecture.dot](full_system_architecture.dot)
**Description:** Detailed V4.0 specification diagram with CondorNet and Diffusion components. See `condornet_v43_architecture.png` for the current v4.3 architecture.

### condornet_v43_architecture.png (v4.3 — CURRENT)
**File:** [condornet_v43_architecture.png](condornet_v43_architecture.png)
**Description:** Complete CondorNet™ v4.3 architecture diagram showing the multi-source data fusion pipeline: 4× MultiTFProjector → PivotProjector + TFFusionBlock → OptionsChainEncoder → JointFusionLayer → v4.2 ETD-1/CDE Core → StrategyHead (10 types) + RiskMetricHead (PoP/EV/VaR/CVaR) + PivotPredictionHead + PositionSizeHead. 10.9M parameters.

---

## GUI & Training (Phase 6 - NEW)

### gui_architecture.png
**File:** [gui_architecture.png](gui_architecture.png)
**Source:** [gui_architecture.dot](gui_architecture.dot)
**Description:** Complete CondorBrain GUI architecture showing:
- Frontend: React 18 pages (Dashboard, Training, Introspection, Backtest)
- Training components: TrainingHeader, MetricGrid, StreamingLossChart, StreamingHeatmap
- State management: useWebSocket, useTrainingTelemetry, Zustand stores
- Backend: FastAPI routers (/api/training/*, /api/config/*, /api/backtest/*)
- WebSocket Manager for real-time broadcasts

### training_telemetry_flow.png
**File:** [training_telemetry_flow.png](training_telemetry_flow.png)
**Source:** [training_telemetry_flow.dot](training_telemetry_flow.dot)
**Description:** Real-time training telemetry data flow showing:
- Training script metrics collection (12 loss components, diagnostics, fuzzy activations)
- HTTP POST transport to backend endpoints
- WebSocket channel broadcasting (training.step, training.epoch, training.fuzzy)
- Frontend hooks and component data distribution
- Environment URL detection for local/lightai/kaggle/colab

### training_components.png
**File:** [training_components.png](training_components.png)
**Source:** [training_components.dot](training_components.dot)
**Description:** Detailed breakdown of Training page React components:
- TrainingHeader: Epoch/step counters, circular progress, ETA
- MetricGrid: 12 LiveMetricCards with sparklines
- StreamingLossChart: Recharts LineChart with component toggles
- StreamingHeatmap: Canvas-based fuzzy gate heatmap
- DiagnosticsPanel: LR, gradient norm, scaler mini-charts
- TrainingControls: Start/Stop simulation buttons

### epoch_checkpointing.png
**File:** [epoch_checkpointing.png](epoch_checkpointing.png)
**Source:** [epoch_checkpointing.dot](epoch_checkpointing.dot)
**Description:** Epoch checkpointing system showing:
- Training loop with checkpoint decision logic
- State saved: model weights, optimizer state, scheduler state, epoch number
- Output files: models/checkpoints/condornet_epoch_*.pt
- Training resumption flow with full state restoration

---

## Intelligence Core (CondorNet v4.0 → v4.3)

### CondorNet_NN_Architecture.png
**File:** [CondorNet_NN_Architecture.png](CondorNet_NN_Architecture.png)
**Source:** [CondorNet_NN_Architecture.dot](CondorNet_NN_Architecture.dot)
**Description:** CondorNet neural network architecture with ETD-1 exponential integrator, TFT control, and Neural CDE path response.

### CondorNet_EquationGraph.png
**File:** [CondorNet_EquationGraph.png](CondorNet_EquationGraph.png)
**Source:** [CondorNet_EquationGraph.dot](CondorNet_EquationGraph.dot)
**Description:** CondorNet equation data flow showing ETD-1 × Neural CDE × Predicate Gates integration.

### CondorNet_Operator_Block.png
**File:** [CondorNet_Operator_Block.png](CondorNet_Operator_Block.png)
**Source:** [CondorNet_Operator_Block.dot](CondorNet_Operator_Block.dot)
**Description:** CondorNet master update cycle with ETD-1 exponential integrator + Neural CDE + TFT Control.

### enhanced_architecture.png
**File:** [enhanced_architecture.png](enhanced_architecture.png)
**Source:** [enhanced_architecture.dot](enhanced_architecture.dot)
**Description:** CondorNet backbone with specialized output heads (Diffusion, TopKMoE).

### condor_intelligence_flow.png
**File:** [condor_intelligence_flow.png](condor_intelligence_flow.png)
**Source:** [condor_intelligence_flow.dot](condor_intelligence_flow.dot)
**Description:** Standard logical flow through CondorNet Intelligence Core from input tensors to trading signals.

### condor_intelligence_flow_premium.png
**File:** [condor_intelligence_flow_premium.png](condor_intelligence_flow_premium.png)
**Source:** [condor_intelligence_flow_premium.dot](condor_intelligence_flow_premium.dot)
**Description:** Enhanced intelligence flow with TDA and Generative Diffusion components.

---

## Data Pipeline

### data_pipeline_detailed.png
**File:** [data_pipeline_detailed.png](data_pipeline_detailed.png)
**Source:** [data_pipeline_detailed.dot](data_pipeline_detailed.dot)
**Description:** Detailed data ingestion showing MarketSnapshot object creation and feature alignment.

### lag_alignment_flow.png
**File:** [lag_alignment_flow.png](lag_alignment_flow.png)
**Source:** [lag_alignment_flow.dot](lag_alignment_flow.dot)
**Description:** Critical timestamp synchronization logic for 15-minute delayed data alignment.

### pipeline_diagram.png
**File:** [pipeline_diagram.png](pipeline_diagram.png)
**Source:** [pipeline_diagram.dot](pipeline_diagram.dot)
**Description:** ETL scripts and processes for raw data fetching from Alpaca/IVolatility.

---

## Decision Logic & Sizing

### fuzzy_sizing_pipeline.png
**File:** [fuzzy_sizing_pipeline.png](fuzzy_sizing_pipeline.png)
**Source:** [fuzzy_sizing_pipeline.dot](fuzzy_sizing_pipeline.dot)
**Description:** Fuzzy logic inference system calculating Fuzzy Confidence Score for position sizing.

### membership_curves.png
**File:** [membership_curves.png](membership_curves.png)
**Source:** [membership_curves.dot](membership_curves.dot)
**Description:** Fuzzy membership functions (RSI, VIX) normalizing inputs to 0-1 confidence scores.

### dataflow.png
**File:** [dataflow.png](dataflow.png)
**Source:** [dataflow.dot](dataflow.dot)
**Description:** Comprehensive mapping of technical indicators, hard gates, and fuzzy logic flow.

### position_sizing.png
**File:** [position_sizing.png](position_sizing.png)
**Source:** [position_sizing.dot](position_sizing.dot)
**Description:** Kelly Criterion and risk management logic for contract quantity determination.

### entryexitdecision.png
**File:** [entryexitdecision.png](entryexitdecision.png)
**Source:** [entryexitdecision.dot](entryexitdecision.dot)
**Description:** Decision-making process for Iron Condor trade entry/exit based on signal strength.

### exit_priority.png
**File:** [exit_priority.png](exit_priority.png)
**Source:** [exit_priority.dot](exit_priority.dot)
**Description:** Hierarchy of exit conditions: profit taking, stop losses, technical invalidations.

---

## Execution & Optimization

### institutional_execution_flow.png
**File:** [institutional_execution_flow.png](institutional_execution_flow.png)
**Source:** [institutional_execution_flow.dot](institutional_execution_flow.dot)
**Description:** Operational workflow for broker trade execution with order lifecycle management.

### optimization_pipeline.png
**File:** [optimization_pipeline.png](optimization_pipeline.png)
**Source:** [optimization_pipeline.dot](optimization_pipeline.dot)
**Description:** Feedback loop for model retraining and hyperparameter tuning.

---

## Audit & Comparison

### condornet_learned_logic.pdf (NEW)
**File:** [condornet_learned_logic.pdf](condornet_learned_logic.pdf) | [PNG](condornet_learned_logic.png)
**Source:** [condornet_learned_logic.dot](condornet_learned_logic.dot)
**Description:** Complete visualization of CondorNet learned decision logic from Epoch 3 checkpoint. Shows:
- **Input Features**: 54-dimensional feature space (price, greeks, technical, momentum, regime, risk)
- **5 Predicate Gates**: Learned thresholds for Vol Spike, Liquidity Lock, Trend Reversal, Gap Guard, Gamma Hedge
- **8 SuperSets × 32 Sets**: Relational logic with operator weights (<, >, =) and top feature comparisons
- **4 State Blocks**: Market Physics (h), Portfolio (v), Momentum (m), Regime (r)
- **10 Output Targets**: Strike offsets, DTE, confidence, entry/exit signals with neural sensitivity scores
- **Operator Distribution**: ~33% each for <, >, = with steepness=20 soft sigmoid

### audit_cde_comparison_flow.png
**File:** [audit_cde_comparison_flow.png](audit_cde_comparison_flow.png)
**Source:** [audit_cde_comparison_flow.dot](audit_cde_comparison_flow.dot)
**Description:** Multi-Model Comparison Audit tool architecture showing 9-stage pipeline from CLI through analysis to report generation.

---

## CondorNet Mathematical Derivations

### CondorNet_Equation_Derivation.png
**File:** [CondorNet_Equation_Derivation.png](CondorNet_Equation_Derivation.png)
**Source:** [CondorNet_Equation_Derivation.dot](CondorNet_Equation_Derivation.dot)
**Description:** Mathematical derivation of CondorNet equations.

### CondorNet_Mathematical_Integrator_Flow.png
**File:** [CondorNet_Mathematical_Integrator_Flow.png](CondorNet_Mathematical_Integrator_Flow.png)
**Source:** [CondorNet_Mathematical_Integrator_Flow.dot](CondorNet_Mathematical_Integrator_Flow.dot)
**Description:** Mathematical integrator flow for CondorNet.

### CondorNet_DecisionTree.png
**File:** [CondorNet_DecisionTree.png](CondorNet_DecisionTree.png)
**Source:** [CondorNet_DecisionTree.dot](CondorNet_DecisionTree.dot)
**Description:** Decision tree representation of CondorNet logic.

---

## Rule Engine & Gaussian DAG

### CondorBrain_V22_RuleEngine_Architecture.png
**File:** [CondorBrain_V22_RuleEngine_Architecture.png](CondorBrain_V22_RuleEngine_Architecture.png)
**Source:** [CondorBrain_V22_RuleEngine_Architecture.dot](CondorBrain_V22_RuleEngine_Architecture.dot)
**Description:** V2.2 Rule Engine architecture.

### condor_brain_gaussian_dag.png
**File:** [condor_brain_gaussian_dag.png](condor_brain_gaussian_dag.png)
**Source:** [condor_brain_gaussian_dag.dot](condor_brain_gaussian_dag.dot)
**Description:** Gaussian DAG representation of CondorBrain.

---

## Logos & Branding

- **CondorBrain Logo transparent.png** - Logo with transparent background
- **CondorBrain logo mark.png** - Logo mark

---

## Regenerating Diagrams

All `.dot` files can be regenerated to PNG using Graphviz:

```bash
# Single diagram
dot -Tpng diagram_name.dot -o diagram_name.png

# All diagrams in directory
for f in *.dot; do dot -Tpng "$f" -o "${f%.dot}.png"; done
```

---

**© 2026 by Dr. T. Jerry Mahabub, Ph.D — All rights reserved.**
