# Technical Architecture Summary: CondorIntelligence Flow

The CondorIntelligence system is a 6-layer neural-symbolic pipeline designed for high-frequency SPY option trading. It integrates a state-of-the-art Mamba-2 Selective State Space Model (SSM) with a Neural-Fuzzy decision suite to transform raw market physics into risk-weighted trade intent.

## 1. Data Inception (24-Feature Manifold)
The pipeline begins at the **Input Manifold**, where 24 discrete market parameters are ingested:
- **Spot Dynamics (5):** OHLCV data providing the price action baseline.
- **Option Physics (5):** Black-Scholes-Merton Greeks (Delta, Gamma, Vega, Theta, Rho) capturing the surface curvature.
- **Volatility Context (3):** IV Rank, VIX, and bid/ask spreads for regime assessment.
- **Technicals (11):** High-level indicators (RSI, ADX, Bollinger Bands, Parabolic SAR) for trend confirmation.

## 2. Tactical Preprocessing
Data is passed through an **O(1) GPU-Resident Data Loader** using `unfold` views. This eliminates CPU-side batch materialization and host-to-device (H2D) overhead. Features are normalized using a robust **Med/MAD Z-Score** approach and clipped via **Tanh** to ensure numerical stability.

## 3. Mamba-2 Neural Backbone
The core intelligence resides in a **24-layer Mamba-2 Selective SSM**. Unlike traditional Transformers, Mamba-2 offers $O(N)$ linear scaling while maintaining long-range dependency tracking. It utilizes hardware-aware discretization (Zero-Order Hold) and selective scan kernels to extract high-fidelity latents from the temporal series.

## 4. Multi-Channel Intelligence (23 Synergy Heads)
The Mamba backbone feeds into **23 discrete output heads** that train simultaneously:
- **Policy Branch (8 heads):** Direct prediction of Iron Condor strikes (Offsets), Width, DTE, and Probability of Profit (POP).
- **Regime Experts (3 heads):** Part of a Mixture-of-Experts (MoE) system that gates predictions based on Low, Normal, or High volatility states.
- **Horizon Forecasters (12 heads):** Probabilistic price trajectory envelopes for 1, 3, 5, and 10-day look-aheads.

## 5. Multi-Objective Learning (CondorLoss)
The model is optimized via **CondorLoss**, a Lagrangian-based composite objective:
- **Huber/MSE:** For precise strike offset and trajectory prediction.
- **Cross-Entropy:** For regime classification and direction probabilities.
- **Risk Weighting:** Loss components are weighted to prioritize strike precision over auxiliary metrics.

## 6. Neural-Fuzzy Decision Suite
The final layer bridges the gap between neural forecasts and capital preservation:
- **Fuzzy Engine:** An 11-factor inference system (membership tiers) that validates Mamba predictions against deterministic rules.
- **Money Management:** A dynamic sizing algorithm that scales trade exposure based on the **Predictive Alignment** of all 23 heads. If the Horizon Forecaster aligns with the Policy Branch under a stable Regime, exposure scales to $100\%$ of risk; any divergence triggers defensive scaling.


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

## Model Profile Alignment (Addendum)

This summary references a 24-feature input manifold in places; production and training may use V2.2 (52) features. The authoritative schema is defined in `intelligence/canonical_feature_registry.py`, and checkpoints must embed their own `feature_cols` and `input_dim` metadata.
