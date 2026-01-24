# Architecture Diagrams Glossary

This directory contains the visual specifications for the CondorBrain system. Below is a description of each diagram's purpose.

## 🗺️ Master Maps
- **[diagram_map.png](diagram_map.png)**: The master navigational map that links all other diagrams together, organized by functional cluster (Data, Intelligence, Logic, Execution, Optimization).
- **[system_overview.png](system_overview.png)**: A high-level conceptual view of the entire system, showing the interaction between Data, Intelligence, and Execution layers.
- **[full_system_architecture.png](full_system_architecture.png)**: The detailed V2.2 specification diagram, acting as the primary reference for the complete system architecture including Mamba 2 and Diffusion components.
- **[phase2_phase3_architecture.png](phase2_phase3_architecture.png)**: Roadmap diagram outlining the transition from basic indicator logic (Phase 2) to advanced risk controls and execution (Phase 3).

## 🧠 Intelligence Core
- **[enhanced_architecture.png](enhanced_architecture.png)**: Focuses specifically on the Mamba 2 backbone and its connection to the new specialized output heads (Diffusion, TopKMoE).
- **[mamba_engine_logic.png](mamba_engine_logic.png)**: Internal schematic of a single Mamba 2 block, showing the Selective Scan mechanism (SSM) and hardware-accelerated pathways.
- **[condor_intelligence_flow_premium.png](condor_intelligence_flow_premium.png)**: Enhanced version of the intelligence flow including Topological Data Analysis (TDA) and Generative Diffusion components.
- **[condor_intelligence_flow.png](condor_intelligence_flow.png)**: Standard logical flow through the Neural Intelligence Core, detailing the transformation from input tensors to trading signals.

## 💧 Data Pipeline
- **[pipeline_diagram.png](pipeline_diagram.png)**: Illustrates the ETL (Extract, Transform, Load) scripts and processes that fetch raw data from Alpaca/IVolatility and prepare it for the model.
- **[data_pipeline_detailed.png](data_pipeline_detailed.png)**: Detailed view of the data ingestion interactions, specifically focusing on the `MarketSnapshot` object creation and feature alignment.
- **[lag_alignment_flow.png](lag_alignment_flow.png)**: Visualizes the critical timestamp synchronization logic to ensuring 15-minute delayed data is correctly aligned with real-time signals.

## 🟢 Decision Logic & Sizing
- **[dataflow.png](dataflow.png)**: Comprehensive mapping of all technical indicators, hard gates, and their flow into the fuzzy logic system.
- **[fuzzy_sizing_pipeline.png](fuzzy_sizing_pipeline.png)**: Details the fuzzy logic inference system that calculates the `Fuzzy Confidence Score` used for position sizing.
- **[membership_curves.png](membership_curves.png)**: Visual representation of the fuzzy membership functions (e.g., RSI, VIX) that normalize inputs into 0-1 confidence scores.
- **[position_sizing.png](position_sizing.png)**: Illustrates the Kelly Criterion and risk management logic that determines the exact number of contracts to trade.
- **[entryexitdecision.png](entryexitdecision.png)**: Logic tree diagram showing the decision-making process for entering and exiting Iron Condor trades based on signal strength.
- **[exit_priority.png](exit_priority.png)**: Hierarchy of exit conditions, ranking profit taking, stop losses, and technical invalidations (e.g., BB Breakout) by priority.

## 🟠 Execution & Optimization
- **[institutional_execution_flow.png](institutional_execution_flow.png)**: The operational workflow for executing trades with a broker, including order lifecycle management and risk checks.
- **[optimization_pipeline.png](optimization_pipeline.png)**: The feedback loop for model retraining and hyperparameter tuning, connecting training performance back to production parameters.

---
**© 2026 by Dr. T. Jerry Mahabub, Ph.D — All rights reserved.**

---

## Addendum (2026-01-24): Schema Drift Warning

Several diagrams in this folder reference a 24-feature input manifold and 24-layer model. The repo has V2.1 (32) and V2.2 (52) feature schemas, and production configs vary by profile. To avoid drift:

- Treat `intelligence/canonical_feature_registry.py` as SSOT for input schema.
- Enforce explicit feature selection by name (not CSV order).
- Update diagram captions when model configs change.


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
