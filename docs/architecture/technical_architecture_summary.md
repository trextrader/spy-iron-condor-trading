# Technical Architecture Summary: CondorIntelligence Flow

The CondorIntelligence system is a 6-layer neural-symbolic pipeline designed for high-frequency SPY option trading. It integrates the novel **CondorNet™** unified architecture with a Neural-Fuzzy decision suite to transform raw market physics into risk-weighted trade intent.

> **Architecture Evolution:** CondorNet™ replaces the previous Mamba-2 SSM (NaN explosions) and Neural CDE (overfitting) implementations with a mathematically principled fusion of ETD-1 exponential integration, TFT control synthesis, and Neural CDE path response.

## 0. CondorNet™ Master Equation

$$
x_k = e^{A_\theta(u_k)\Delta t_k} x_{k-1} + \Delta t_k \varphi_1(A_\theta(u_k)\Delta t_k) B_\theta(u_k) + G_\theta(x_{k-1}, u_k) \Delta X_k + D(\text{Greeks}_k, r_{k-1}, q_k)
$$

Where $\varphi_1(M) = M^{-1}(e^M - I)$ is the ETD-1 basis function.

## 1. Data Inception (54-Feature Manifold V2.2)
The pipeline begins at the **Input Manifold**, where 54 discrete market parameters are ingested:
- **Spot Dynamics (5):** OHLCV data providing the price action baseline.
- **Option Physics (5):** Black-Scholes-Merton Greeks (Delta, Gamma, Vega, Theta, Rho) capturing the surface curvature.
- **Volatility Context (3):** IV Rank, VIX, and spread ratio for regime assessment.
- **Dynamic Indicators (8):** Curvature proxy, volatility energy, adaptive RSI/ADX/PSAR.
- **Rule Primitives (22):** DSL engine outputs for interpretable feature engineering.
- **Technicals (11):** High-level indicators (RSI, ADX, Bollinger Bands, Parabolic SAR) for trend confirmation.

## 2. Tactical Preprocessing
Data is passed through an **O(1) GPU-Resident Data Loader** using `unfold` views. This eliminates CPU-side batch materialization and host-to-device (H2D) overhead. Features are normalized using a robust **Med/MAD Z-Score** approach and clipped via **Tanh** to ensure numerical stability.

## 3. CondorNet™ Neural Backbone
The core intelligence resides in the **CondorNet™ unified architecture**, which fuses three paradigms:

| Component | Role | Mathematical Term |
|-----------|------|-------------------|
| **TFT Control** | Long-range temporal context | $u_k = \text{TFT}(X_{1:k})$ |
| **ETD-1 Core** | Stable matrix exponential evolution | $e^{A\Delta t}x + \Delta t \varphi_1 B$ |
| **Neural CDE** | Path-dependent market response | $G_\theta(x, u) \cdot \Delta X_k$ |
| **Predicate Gates** | Regime-aware gating | 5 canonical inequality predicates |

The **4-block augmented state** $x_k = [h_k; v_k; m_k; r_k]$ captures:
- **h_k**: Latent risk manifold (256 dims)
- **v_k**: Portfolio state - Greeks, P&L (32 dims)
- **m_k**: Risk memory - drawdown, VaR (64 dims)
- **r_k**: Regime combinatorics from predicates (32 dims)

## 4. Multi-Channel Intelligence (10 Policy Heads)
The CondorNet backbone feeds into **10 discrete output heads**:
- **Policy Branch (8 heads):** Direct prediction of Iron Condor strikes (Offsets), Width, DTE, POP, ROI, Max Loss, Confidence.
- **Entry/Exit Signals (2 heads):** Entry and exit logits for trade timing.

## 5. Multi-Objective Learning (Composite CondorNet Loss)
The model is optimized via **10-component composite loss**:
- **NPDD Loss:** Net Profit / Drawdown prediction
- **Sharpe Loss:** Risk-adjusted returns (negative, to maximize)
- **Drawdown Penalty:** Max drawdown regularization
- **Turnover Loss:** Trade frequency penalty
- **Fuzzy Consistency:** Sizing consistency with fuzzy engine
- **Pattern Entropy:** Predicate diversity
- **Group Invariance:** Signature consistency under permutation
- **Correlation Loss:** Cross-head decorrelation
- **Energy Loss:** State magnitude regularization
- **Growth Loss:** Capital growth rate

## 6. Neural-Fuzzy Decision Suite
The final layer bridges the gap between neural forecasts and capital preservation:
- **Fuzzy Engine:** An 11-factor inference system (membership tiers) that validates CondorNet predictions against deterministic rules.
- **Money Management:** A dynamic sizing algorithm that scales trade exposure based on the **Predictive Alignment** of policy heads, predicate gate activations, and fuzzy membership scores.

---

## Repository Sync Addendum (2026-02-04)

This document is part of the synchronized documentation set. The authoritative engineering spec and audit references are:

- `docs/CONDORNET_IMPLEMENTATION_PLAN.md` (primary spec)
- `docs/INTEGRATION_PLAN_MASTER.md`
- `docs/INTERFACE_CATALOG.md`

Key alignment requirements:
1. Feature schema selection by **name** (V2.2) only; no CSV order dependence.
2. Dataset column order differs across years; schema validation must be strict.
3. Model config metadata (layers/heads/input_dim) must match deployed checkpoints.
4. **CondorNet backbone is now default** - implemented in `intelligence/condor_brain_net.py`.

If this document conflicts with the master spec, the master spec governs implementation.

---

*Document Version: 4.0 (CondorNet™) | Last Updated: 2026-02-04*
