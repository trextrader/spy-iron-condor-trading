# Scientific Specification: CondorNet™ v4.3 — Fused ETD-1 × TFT × Neural CDE Architecture

**Version:** 4.3.0
**Date:** 2026-02-24
**Architecture:** Multi-Timeframe ETD-1 + Temporal Fusion Transformer + Neural CDE + Symbolic Predicate Routing
**Parameters:** 10,955,687 (~11M)
**Status:** Active training — Run 14 best val loss 0.9553 (Epoch 7)

---

## Abstract

CondorNet™ v4.3 is a multi-modal, multi-objective parametric policy network for full-spectrum SPY options strategy selection across ten strategy classes: single calls, single puts, bull call spreads, bear put spreads, straddles, strangles, butterfly calls, iron condors, custom multi-leg structures, and a learned abstain class. The system synthesizes three mathematically complementary temporal integration paradigms — **Exponential Time Differencing order-1 (ETD-1)**, **Temporal Fusion Transformer (TFT)**, and **Neural Controlled Differential Equations (Neural CDE)** — into a unified backbone that simultaneously handles stiff multi-scale dynamics, cross-timeframe attention, and continuous path-driven latent state evolution. These are not sequential stages but fused components that interact through shared latent representations.

The input manifold consists of four independent timeframe projections (M1, M5, M15, H1) each encoding 52 base technical features plus 12 ETL-computed features totaling a 64-dimensional per-timeframe input vector, a separate 13-dimensional sparse pivot geometry pathway, and a Transformer-encoded 120-contract options chain embedding capturing the full implied volatility surface. The three streams are fused through a **JointFusionLayer** before entering the ETD-1 backbone, which integrates temporal dynamics as a stiff ordinary differential equation using the exact exponential integrator formula $\mathbf{z}_{t+1} = e^{h\mathbf{L}} \mathbf{z}_t + h\,\varphi_1(h\mathbf{L})\,\mathbf{g}(t)$, avoiding the Courant–Friedrichs–Lewy instability constraint that limits explicit Euler discretizations to short lookback windows.

Atop the ETD-1 temporal backbone, a three-tier **symbolic routing hierarchy** — 64 learned predicate comparisons over 2,016 unique pairwise feature relations, composing into 32 fuzzy sets and 16 supersets via continuous gate routing $\lambda = \sigma(z/\tau)$ with temperature $\tau = 3$ — provides a directly inspectable fuzzy rule base where every superset activation reads back as a human-readable conjunction of feature comparisons such as $\texttt{m5.rsi\_dyn} > \texttt{m15.adx\_adaptive}$. The Neural CDE component furnishes a path-response layer that treats the full input sequence as a controlled differential equation driven by the signal control path, capturing long-range dependency without the $O(T^2)$ cost of full attention.

Training is governed by a **9-component composite loss function** with independent annealing schedules, supervised simultaneously on strategy class selection (10-class cross-entropy), probability-of-profit calibration (binary cross-entropy), expected value, VaR/CVaR risk boundaries, fuzzy position sizing (five MSE objectives), and three regularizers targeting gate diversity, predicate pattern entropy, and Huber robustness. Gate logit stability is enforced through a four-floor regularization system — variance, interquartile range, median absolute deviation, and a novel **logit tail penalty** $\mathcal{L}_{\text{tail}} = \alpha_{\text{tail}}\,\mathbb{E}[\max(0, |z| - L)^2]$ that directly penalizes squared excess beyond a saturation band of $|z| > 8$, preventing rare extreme logit outliers that bulk-distribution controls miss.

> **Migration Note:** This document supersedes the CDE v3.0 specification (2026-01-27). The CDE component remains present as the path-response layer but is no longer the primary temporal integrator. The ETD-1 + TFT fusion constitutes the new backbone. See `docs/legacy/` for archived documentation.

---

## Table of Contents

1. [Input Data & Feature Engineering](#1-input-data--feature-engineering)
2. [Multi-Timeframe Projection Layer](#2-multi-timeframe-projection-layer)
3. [Pivot Projection Module](#3-pivot-projection-module)
4. [TFF Fusion Block — Temporal Fusion Transformer](#4-tff-fusion-block--temporal-fusion-transformer)
5. [Options Chain Encoder](#5-options-chain-encoder)
6. [Joint Fusion Layer](#6-joint-fusion-layer)
7. [ETD-1 Backbone: Exponential Time Differencing](#7-etd-1-backbone-exponential-time-differencing)
8. [Predicate Gate Architecture](#8-predicate-gate-architecture)
9. [Relational Logic Layer & Symbolic Routing Hierarchy](#9-relational-logic-layer--symbolic-routing-hierarchy)
10. [Gate Routing: Continuous Relaxation](#10-gate-routing-continuous-relaxation)
11. [Neural CDE Path-Response Layer](#11-neural-cde-path-response-layer)
12. [Output Heads & Strategy Universe](#12-output-heads--strategy-universe)
13. [9-Component Composite Loss Function](#13-9-component-composite-loss-function)
14. [Gate Regularization System](#14-gate-regularization-system)
15. [Normalization & Data Pipeline](#15-normalization--data-pipeline)
16. [Training Configuration](#16-training-configuration)
17. [Empirical Convergence Analysis — Runs 7–14](#17-empirical-convergence-analysis--runs-714)
18. [Spectral Stability & A-Matrix Analysis](#18-spectral-stability--a-matrix-analysis)
19. [Epoch Interpretability System](#19-epoch-interpretability-system)
20. [Iron Condor P&L Mathematics](#20-iron-condor-pl-mathematics)
21. [Audit & Deployment Gate](#21-audit--deployment-gate)

---

## 1. Input Data & Feature Engineering

### 1.1 Dataset Structure

The v4.3 training corpus is a four-timeframe multi-modal dataset aligned to the M5 canonical clock. All features are precomputed by `data_factory/sync_engine.py` using the V2.2 schema defined in `intelligence/schema_v43.py`.

| Timeframe | File | Rows | Role |
|-----------|------|------|------|
| M1 | `m1_dataset_v43_final.csv` | 92,462 | Micro-structure precision |
| **M5** | `m5_dataset_v43_final.csv` | **18,494** | **Canonical clock** |
| M15 | `m15_dataset_v43_final.csv` | 6,166 | Regime context |
| H1 | `h1_dataset_v43_final.csv` | 1,661 | Macro trend anchor |
| Options Chain | `options_2025_v43.csv` | 2,320,198 contracts | IV surface / 239 timestamps |

The M5 timestamp defines the master clock $\mathcal{T}$. All other timeframes are forward-filled to align at each $t \in \mathcal{T}$, with a maximum look-back fill of 5 minutes for the options chain. Any timestamp lacking a valid chain snapshot within this window is dropped.

### 1.2 Base Feature Schema (52 per timeframe)

The V2.2 schema defines 52 base input features organized into six groups:

| Group | Count | Features |
|-------|-------|----------|
| **Price / Trend** | 12 | `open, high, low, close, MA, FRAMA, sma, TurtleChn-Up, TurtleChn-Low, TrendSeekerUp, TrendSeekerDown, AnchoredVWAP` |
| **Momentum** | 10 | `rsi_dyn, stoch_k_dyn, adx_adaptive, ret_z, log_return, AccDistWill, AccDistWillMovAvg, WilderAccSwingIndex, breakout_score, WeightedAlpha` |
| **Volatility** | 8 | `ATR_recon, atr_pct, bb_upper_dyn, bb_lower_dyn, bb_mu_dyn, bb_sigma_dyn, bandwidth, bw_expansion_rate` |
| **Volume / Microstructure** | 6 | `volume, trade_count, OSC-Volume, vol_energy, vol_ewma, spread_ratio` |
| **Regime / Probability** | 10 | `psar_adaptive, psar_mark, psar_trend, psar_reversion_mu, pressure_up, pressure_down, max_dd_60m, bb_percentile, fuzzy_reversion_11, consolidation_score` |
| **Risk / Composite** | 6 | `gap_risk_score, chaos_membership, kappa_proxy, McClellanOsc, Slope, SlopeATR` |

### 1.3 ETL-Computed Features (12, extending to 64 total)

The following 12 features are appended to the base 52 during ETL, yielding the 64-dimensional input vector $x_t \in \mathbb{R}^{64}$ fed to the RelationalLogicLayer:

| Index | Feature | Formula | Physical Meaning |
|-------|---------|---------|-----------------|
| 52 | `friction_ok_5` | $\mathbb{1}[\text{spread} < \text{HL}_5]$ | Liquidity gate at 5-bar window |
| 53 | `friction_ok_10` | $\mathbb{1}[\text{spread} < \text{HL}_{10}]$ | Liquidity gate at 10-bar window |
| 54 | `friction_ok_20` | $\mathbb{1}[\text{spread} < \text{HL}_{20}]$ | Liquidity gate at 20-bar window |
| 55 | `friction_ok_40` | $\mathbb{1}[\text{spread} < \text{HL}_{40}]$ | Liquidity gate at 40-bar window |
| 56 | `friction_ok_60` | $\mathbb{1}[\text{spread} < \text{HL}_{60}]$ | Liquidity gate at 60-bar window |
| 57 | `tod_sin` | $\sin(2\pi \cdot t_{\min}/390)$ | Circular time-of-day encoding |
| 58 | `tod_cos` | $\cos(2\pi \cdot t_{\min}/390)$ | Circular time-of-day encoding |
| 59 | `regime_persistence` | Count of consecutive bars in same $(v_b, \tau_b)$ | Regime stability counter |
| 60 | `price_stretch` | $(C_t - \text{SMA}_{20}) / (\text{ATR}_{14} + \varepsilon)$, clipped $\pm 10$ | Mean deviation in ATR units |
| 61 | `ivr_zone` | $+1$ if $\text{IVR} > 0.70$; $-1$ if $\text{IVR} < 0.30$; else $0$ | IV rank regime ternary |
| 62 | `stretch_zone` | $+1$ if $\text{price\_stretch} > 2$; $-1$ if $< -2$; else $0$ | Stretch regime ternary |
| 63 | `reversal_score` | $\alpha \cdot \text{IVR} + \beta \cdot \text{price\_stretch} + \gamma \cdot (\text{pivotStr} \times \text{pivotDir})$ | Composite reversal signal |

**Friction gate formulation:** For each window $N \in \{5, 10, 20, 40, 60\}$:

$$\text{HL}_N = \max_{i \leq t}(\text{high}_{t-N:t}) - \min_{i \leq t}(\text{low}_{t-N:t})$$

$$\text{friction\_ok}_N = \mathbb{1}[\text{bid\_ask\_spread}_t < \text{HL}_N]$$

The gate opens if **any** window passes (OR logic), allowing the model's attention over 5 friction columns to learn which lookback is most relevant for current conditions.

**Curvature proxy and manifold-aware indicators** (within base 52):

$$\kappa_t = \frac{\text{EMA}_{64}(r_t - 2r_{t-1} + r_{t-2})}{\sigma_t + \varepsilon}, \qquad E_t = \ln(1 + \alpha|\kappa_t|), \quad \alpha = 1000$$

$$\text{rsi\_dyn}(t) = \text{RSI}_{14}(t) \cdot (1 + \beta E_t), \qquad \text{adx\_adaptive}(t) = \frac{\text{ADX}_{14}(t)}{1 + \gamma E_t}$$

$$\text{psar\_adaptive}(t) = \frac{C_t - \text{PSAR}_t}{\text{ATR}_{14}(t) + \varepsilon}$$

### 1.4 Pivot Feature Pathway (13 sparse, separate)

Thirteen pivot geometry features are passed separately as a masked pathway and never concatenated with base features:

$$\mathcal{P} = \{\texttt{PivotHigh}, \texttt{PivotLow}, \texttt{PivotResidual}, \texttt{PivotResidualATR}, \texttt{PivotResidualZ}, \texttt{PivotCurvatureATR}, \texttt{PivotCurvatureProxy},$$
$$\texttt{PivotSegmentLengthBars}, \texttt{PivotSegmentLengthMinutes}, \texttt{PivotSegmentResidualStd}, \texttt{PivotSegmentVolatility}, \texttt{Slope}, \texttt{SlopeATR}\}$$

NaN = no pivot event at bar $t$. These are **never imputed** — sparsity is semantically meaningful and handled via masked mean pooling.

### 1.5 Options Chain Features (10 per contract × 120 contracts)

The chain grid consists of the 60 nearest calls and 60 nearest puts sorted by $|\text{moneyness}|$:

| Feature | Symbol | Notes |
|---------|--------|-------|
| `moneyness` | $m = \ln(S/K)$ | Computed; not in raw CSV |
| `days_to_exp` | $\tau$ | Calendar days |
| `iv` | $\sigma_{\text{imp}}$ | Annualized implied volatility |
| `delta` | $\Delta \in [-1,1]$ | Black-Scholes delta |
| `gamma` | $\Gamma \geq 0$ | Second derivative of option price |
| `theta` | $\Theta$ | Daily time decay |
| `vega` | $\mathcal{V}$ | Sensitivity to IV change |
| `bid` | | Bid price |
| `ask` | | Ask price |
| `oi_norm` | | Log-scaled open interest, normalized within snapshot |

Contracts with OI $< 100$ are masked (not dropped) — padded positions receive `mask=True` and are ignored by transformer attention.

---

## 2. Multi-Timeframe Projection Layer

### 2.1 Architecture

Four independent linear projectors map each timeframe's feature tensor from $\mathbb{R}^{52}$ to a shared latent dimension $d_H = 128$:

$$\text{MultiTFProjector}: \quad \mathbf{h}^{(\ell)}_t = \text{LayerNorm}\left(W^{(\ell)}_{\text{proj}} \mathbf{x}^{(\ell)}_t + \mathbf{b}^{(\ell)}_{\text{proj}}\right), \quad \ell \in \{\text{M1}, \text{M5}, \text{M15}, \text{H1}\}$$

where $W^{(\ell)}_{\text{proj}} \in \mathbb{R}^{128 \times 52}$, producing per-timeframe tensors of shape $[\mathcal{B}, T, 128]$ for batch size $\mathcal{B}$ and lookback $T = 200$.

**Independence property:** Each projector maintains its own weight matrix with no parameter sharing. This allows the network to learn distinct "languages" for M1 micro-structure versus H1 macro trend without interference. Cross-timeframe information is exchanged only in the TFFusionBlock (Section 4), preserving clean inductive bias.

### 2.2 Dimensional Analysis

$$\underbrace{[\mathcal{B}, T, 52]}_{\text{per TF input}} \xrightarrow{W^{(\ell)}_{\text{proj}}} \underbrace{[\mathcal{B}, T, 128]}_{\text{per TF hidden}} \times 4 \xrightarrow{\text{stack}} \underbrace{[\mathcal{B}, T, 512]}_{\text{concat before TFFusion}}$$

The four projections are concatenated along the feature dimension, yielding $\mathbf{H}_t^{\text{TF}} \in \mathbb{R}^{512}$ per timestep before fusion.

---

## 3. Pivot Projection Module

### 3.1 Masked Mean Pooling

Pivot features are sparse — NaN at most timesteps. The `PivotProjector` applies masked mean pooling to collapse the 13-dimensional pivot vector into a dense $d_{\text{pivot}} = 16$-dimensional embedding:

$$\tilde{\mathbf{p}}_t = \frac{\sum_{j=1}^{13} m_{t,j} \cdot p_{t,j}}{\sum_{j=1}^{13} m_{t,j} + \varepsilon}, \qquad m_{t,j} = \mathbb{1}[\text{not NaN}(p_{t,j})]$$

$$\mathbf{h}^{(\text{piv})}_t = W_{\text{piv}} \tilde{\mathbf{p}}_t + \mathbf{b}_{\text{piv}} \in \mathbb{R}^{16}$$

When no pivot event occurs at $t$ (all NaN), $\tilde{\mathbf{p}}_t = \mathbf{0}$ and the pivot pathway contributes a zero vector — the model learns to distinguish "no pivot" from actual structural geometry.

### 3.2 Sparse Signal Semantics

The masking operation ensures the model never treats imputed values as genuine pivot signals. This is critical because pivot events (strong highs/lows, curvature inflections) are rare but high-information events — treating NaN as zero would introduce systematic bias toward incorrectly sparse entries.

---

## 4. TFF Fusion Block — Temporal Fusion Transformer

### 4.1 Architecture Overview

The `TFFusionBlock` fuses the stacked 4-timeframe representation $\mathbf{H}^{\text{TF}} \in \mathbb{R}^{512}$ with the pivot embedding $\mathbf{h}^{(\text{piv})} \in \mathbb{R}^{16}$ using a TFT-inspired architecture:

$$\mathbf{H}^{\text{fused}} = \text{TFFusionBlock}\left(\mathbf{H}^{\text{TF}},\, \mathbf{h}^{(\text{piv})}\right) \in [\mathcal{B}, T, 256]$$

### 4.2 Gated Residual Network (GRN)

Each sub-module uses a Gated Residual Network for controlled information flow:

$$\text{GRN}(\mathbf{a}, \mathbf{c}) = \text{LayerNorm}\left(\mathbf{a} + \text{GLU}(W_1 \mathbf{a} + W_2 \mathbf{c} + \mathbf{b}_1)\right)$$

$$\text{GLU}(\mathbf{x}) = \mathbf{x}_{[:d]} \odot \sigma(\mathbf{x}_{[d:]}), \quad \text{(Gated Linear Unit)}$$

where $\mathbf{c}$ is an optional context vector (the pivot embedding). The gating mechanism allows the network to suppress irrelevant features dynamically — a pivot absence $\mathbf{h}^{(\text{piv})} = \mathbf{0}$ leaves the TF stream unperturbed.

### 4.3 Multi-Head Self-Attention (Cross-Timeframe)

The concatenated TF tensor is processed through multi-head attention to model cross-timeframe dependencies:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

$$\mathbf{H}^{\text{attn}} = \text{MultiHead}(\mathbf{H}^{\text{TF}}) = \text{concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $h = 4$ heads, $d_k = 512/h = 128$. The attention mechanism discovers time-varying importance weighting across the four timeframe representations: in high-volatility regimes the M1 micro-structure heads may dominate; in trending regimes the H1 macro anchor may dominate.

### 4.4 Variable Selection Network

A soft variable selection weight vector calibrates the relative importance of each timeframe:

$$\mathbf{v} = \text{softmax}\!\left(W_{\text{sel}} \cdot \text{concat}(\mathbf{H}^{\text{TF}}_{t}, \mathbf{h}^{(\text{piv})}_{t})\right) \in \mathbb{R}^4$$

$$\mathbf{H}^{\text{sel}}_t = \sum_{\ell \in \{M1,M5,M15,H1\}} v^{(\ell)}_t \cdot \mathbf{h}^{(\ell)}_t$$

This provides a differentiable attention mechanism over the four timeframes, interpretable as a learned multi-resolution selector.

---

## 5. Options Chain Encoder

### 5.1 Transformer Architecture

The `OptionsChainEncoder` maps the 120-contract chain grid to a 128-dimensional embedding using a 2-layer Transformer encoder:

$$\text{Chain input}: \mathbf{C} \in [\mathcal{B}, 120, 10]$$

$$\mathbf{C}^{(0)} = \text{Linear}_{10 \to 64}(\mathbf{C}), \quad \mathbf{C}^{(0)} \in [\mathcal{B}, 120, 64]$$

$$\mathbf{C}^{(l)} = \text{TransformerEncoderLayer}^{(l)}(\mathbf{C}^{(l-1)}), \quad l \in \{1, 2\}$$

$$\mathbf{h}^{(\text{chain})} = \text{Linear}_{64 \to 128}\left(\text{MaskedMeanPool}(\mathbf{C}^{(2)})\right) \in \mathbb{R}^{128}$$

Transformer configuration: $d_{\text{model}} = 64$, $n_{\text{heads}} = 4$, $d_{\text{ff}} = 256$. Masked positions (low-OI contracts) are excluded from the pooling via the liquidity mask.

### 5.2 IV Surface Encoding

The chain encoder ingests the full implied volatility term structure across 120 strikes spanning multiple expiry buckets. The moneyness coordinate $m = \ln(S/K)$ together with $\tau$ (days to expiration) provides a discretization of the IV surface in $(m, \tau)$ space:

$$\sigma_{\text{imp}}(m, \tau) \approx \sigma_{\text{imp}}(0, \tau) + \frac{\partial \sigma}{\partial m} m + \frac{1}{2} \frac{\partial^2 \sigma}{\partial m^2} m^2 + \cdots$$

The transformer learns to encode skew ($\partial \sigma / \partial m$), term structure ($\partial \sigma / \partial \tau$), and convexity simultaneously, producing a latent surface embedding that informs strategy selection: steep put skew favors bear put spreads; flat term structure with high ATM IV favors iron condors; rising term structure with low ATM IV favors butterfly structures.

---

## 6. Joint Fusion Layer

The `JointFusionLayer` concatenates the three streams and projects to $d_{\text{joint}} = 128$:

$$\mathbf{z}^{(\text{joint})}_t = W_J \cdot \text{concat}\!\left(\mathbf{H}^{\text{fused}}_t \in \mathbb{R}^{256},\; \mathbf{h}^{(\text{chain})} \in \mathbb{R}^{128}\right) + \mathbf{b}_J$$

$$\mathbf{z}^{(\text{joint})}_t \in \mathbb{R}^{128}, \qquad [\mathcal{B}, T, 384] \xrightarrow{W_J \in \mathbb{R}^{128 \times 384}} [\mathcal{B}, T, 128]$$

This bottleneck fusion forces the model to compress the multi-modal information — price dynamics across four timescales, pivot geometry, and the IV surface — into a unified 128-dimensional state before temporal integration. The compression acts as an information bottleneck regularizer that prevents the backbone from overfitting to any single modality.

---

## 7. ETD-1 Backbone: Exponential Time Differencing

### 7.1 Mathematical Motivation: Stiff ODEs

Financial time series are inherently **stiff** — they exhibit dynamics operating simultaneously on vastly different timescales (fast intraday oscillations with decay time $\tau_{\text{fast}} \approx 5$ bars co-existing with slow trend structures with $\tau_{\text{slow}} \approx 200$ bars). The stiffness ratio $S = \tau_{\text{slow}}/\tau_{\text{fast}} \approx 40$ renders standard explicit Euler integration numerically unstable: stability requires step size $h < 2/|\lambda_{\max}|$ where $\lambda_{\max}$ is the largest eigenvalue of the system Jacobian.

**ETD-1** (Exponential Time Differencing, first-order) circumvents the CFL stability constraint by treating the linear part of the dynamics exactly via matrix exponentiation, achieving **A-stability**: stable for all step sizes regardless of stiffness.

### 7.2 ETD-1 Formulation

The latent state $\mathbf{z}_t \in \mathbb{R}^{d_H}$ evolves according to a partitioned ODE:

$$\frac{d\mathbf{z}}{dt} = \underbrace{\mathbf{L}\,\mathbf{z}}_{\text{stiff linear}} + \underbrace{\mathbf{g}(t)}_{\text{nonlinear forcing}}$$

where $\mathbf{L} \in \mathbb{R}^{d_H \times d_H}$ is a learnable linear decay operator and $\mathbf{g}(t) = W_g \mathbf{z}^{(\text{joint})}_t + \mathbf{b}_g$ is the input injection function derived from the joint fusion output.

The exact solution over one step $h = 1$:

$$\mathbf{z}_{t+1} = e^{h\mathbf{L}} \mathbf{z}_t + h\,\varphi_1(h\mathbf{L})\,\mathbf{g}(t)$$

where the **$\varphi_1$ function** is the first exponential integrator function:

$$\varphi_1(\mathbf{X}) = \mathbf{X}^{-1}(e^{\mathbf{X}} - \mathbf{I}) = \int_0^1 e^{(1-s)\mathbf{X}}\,ds$$

For diagonal $\mathbf{L} = \text{diag}(\lambda_1, \ldots, \lambda_{d_H})$ (efficient implementation):

$$z_{t+1,k} = e^{h\lambda_k} z_{t,k} + h \cdot \frac{e^{h\lambda_k} - 1}{h\lambda_k} \cdot g_k(t)$$

### 7.3 Stability Analysis

For $\text{Re}(\lambda_k) < 0$ (decaying modes), $|e^{h\lambda_k}| < 1$ for all $h > 0$ — the ETD-1 scheme is **unconditionally stable** for the linear part regardless of step size. For the forcing term, the $\varphi_1$ coefficient satisfies:

$$\left|\frac{e^{h\lambda_k} - 1}{h\lambda_k}\right| \leq 1 \quad \forall\, h\lambda_k \leq 0$$

This bounds the influence of any single input on the state update, preventing divergence even when $|\lambda_k| \gg 1$ (stiff modes).

### 7.4 Comparison with Euler and Runge-Kutta

| Method | Stability | Order | Stiff Suitability | Cost per Step |
|--------|-----------|-------|------------------|---------------|
| Explicit Euler | Conditional: $|h\lambda| < 2$ | 1st | Poor | $O(d_H)$ |
| RK4 | Conditional: $|h\lambda| < 2.79$ | 4th | Poor | $4 \times O(d_H)$ |
| **ETD-1** | **A-stable: all $h > 0$** | **1st** | **Excellent** | $O(d_H)$ |
| Crank-Nicolson | A-stable | 2nd | Good | Requires linear solve $O(d_H^2)$ |

### 7.5 Spectral Radius Diagnostic

The spectral radius $\rho(\mathbf{J})$ of the Jacobian of the complete system (ETD-1 + predicate routing) is tracked per epoch as a proxy for representational compression:

$$\rho(\mathbf{J}) = \max_i |\lambda_i(\mathbf{J})|$$

**Interpretation:** Rising spectral radius in later epochs ($\rho$ increasing from 0.013 to 0.163 across Runs 12–14) indicates growing signal amplification — the model is committing stronger representational energy to features it has learned are predictive. The `spectral_radius_delta` in the EPOCH_Comparison JSON records the deviation of each epoch's $\rho$ from the best-model checkpoint.

---

## 8. Predicate Gate Architecture

### 8.1 Predicate Gate Layer

Upstream of the relational logic hierarchy, 8 **hard predicate gates** act as learned input selectors that concentrate gradient flow toward the subset of predicates most activated by the training distribution:

$$\lambda^{(\text{gate})}_k = \sigma\!\left(\frac{z^{(\text{gate})}_k}{\tau}\right), \quad k = 1, \ldots, 8, \quad \tau = 3$$

$$\mathbf{x}^{(\text{gated})} = \lambda^{(\text{gate})} \odot \mathbf{x}^{(\text{in})}$$

Each gate $k$ selects a subset of the 64 predicate activations. Gates with $\lambda \approx 0$ suppress their predicate block entirely; gates with $\lambda \approx 1$ pass their block through at full strength.

### 8.2 Gate Logit Dynamics

The gate logit $z_k$ evolves under gradient descent from the composite loss. In healthy training:
- $z_k \in [-8, 8]$: gate is in its responsive regime, $\sigma'(z/\tau) > 0$
- $z_k < -8$: gate is nearly fully off ($\lambda < 0.0003$), gradient flow approaches zero
- $z_k > 8$: gate is nearly fully on ($\lambda > 0.9997$), gradient flow approaches zero

Saturated gates ($|z_k| \gg 8$) no longer update via gradient descent — only weight decay and the logit tail penalty (Section 14) can recover them. This is the primary motivation for the tail penalty regularizer.

### 8.3 Predicate Count

The full v4.3 backbone contains:
- 8 predicate gates (hierarchical selectors)
- 64 predicates (relational logic units, Section 9)
- 32 sets per superset
- 16 supersets
- Total routing paths: $64 \times 32 \times 16 = 32{,}768$ addressable paths

---

## 9. Relational Logic Layer & Symbolic Routing Hierarchy

### 9.1 Pairwise Feature Comparisons

The `RelationalLogicLayer` operates on $n_{\text{inputs}} = 64$ features (the full ETL feature vector) and computes comparisons over all pairs from the upper triangular index set:

$$\mathcal{I} = \{(i,j) \mid 0 \leq i < j < n_{\text{inputs}}\}, \quad |\mathcal{I}| = \binom{64}{2} = 2{,}016$$

For each pair $(i, j)$, three soft comparison operators are computed:

$$c^{(<)}_{ij} = \sigma\!\left(\frac{x_j - x_i}{\sigma_{\text{scale}} + \varepsilon}\right), \quad c^{(>)}_{ij} = 1 - c^{(<)}_{ij}, \quad c^{(=)}_{ij} = 1 - 2\left|c^{(<)}_{ij} - 0.5\right|$$

The dominant operator is:

$$\text{op}^*(i,j) = \arg\max_{o \in \{<,>,=\}} w_o \cdot c^{(o)}_{ij}$$

where $w_o$ are learned operator weights, giving the interpretable comparison `feature_i op* feature_j`.

### 9.2 Predicate Composition (64 Predicates)

Each of the 64 predicates selects and combines a subset of the 2,016 pairwise comparisons. The top-5 most active pair indices per predicate are logged per epoch via `top_pair_indices` and translated to readable names using the `_FULL_FEATURE_NAMES` list (64 entries, indices 0–63).

**Example from Epoch 7, Run 14:**
```
superset 0 → set 0:
  bb_upper_dyn > bb_sigma_dyn    (dominant: >)   w=[0.0498, 0.0587, 0.0332]
  rsi_dyn > bandwidth            (dominant: >)   w=[0.0437, 0.0656, 0.0094]
  atr_pct < ivr_zone             (dominant: <)   w=[0.0493, 0.0325, 0.0289]
  open > friction_ok_10          (dominant: >)   w=[0.0395, 0.0451, 0.0330]
```

### 9.3 Set Routing (32 Sets per Superset)

Sets implement fuzzy **conjunction** (AND) semantics — a set activates when all its constituent predicates are simultaneously active:

$$A_s = \prod_{k \in \mathcal{P}_s} \lambda_k^{(\text{pred})}, \qquad A_s \in [0, 1]$$

where $\mathcal{P}_s$ is the set of predicate indices assigned to set $s$ and $\lambda_k^{(\text{pred})} = \sigma(z_k^{(\text{pred})}/\tau)$.

The product formulation is equivalent to the probabilistic AND operation under the assumption of independent predicate activations, and provides a soft differentiable relaxation of logical conjunction.

### 9.4 Superset Routing (16 Supersets)

Supersets implement fuzzy **disjunction** (OR) semantics — a superset activates when at least one of its constituent sets is active:

$$A_{ss} = 1 - \prod_{s \in \mathcal{S}_{ss}} (1 - A_s), \qquad A_{ss} \in [0, 1]$$

where $\mathcal{S}_{ss}$ is the collection of sets assigned to superset $ss$. The product-of-complements formulation is the standard probabilistic OR relaxation.

### 9.5 Interpretability of the Hierarchy

The predicate–set–superset hierarchy constitutes an inspectable three-level **fuzzy rule base**:

- **Level 1 (Predicates):** Atomic comparisons — "Is $\texttt{rsi\_dyn}$ greater than $\texttt{bandwidth}$?"
- **Level 2 (Sets):** Conjunctions — "Is $\texttt{rsi\_dyn} > \texttt{bandwidth}$ AND $\texttt{atr\_pct} < \texttt{ivr\_zone}$ AND $\ldots$?"
- **Level 3 (Supersets):** Disjunctions — "Does any of the following market condition sets hold?"

This is not a metaphor for learned representations — it is a directly readable fuzzy rule base. The `top_comparisons` field in each epoch JSON provides the full human-readable rule text. Unlike attention weights in transformers, these comparisons have unambiguous semantic meaning: each one states a directional relationship between two named market features.

---

## 10. Gate Routing: Continuous Relaxation

### 10.1 Sigmoid Gate with Temperature

All gate activations — predicate gates, predicate activations, and set activations — use the sigmoid with learned temperature $\tau$:

$$\lambda = \sigma\!\left(\frac{z}{\tau}\right) = \frac{1}{1 + e^{-z/\tau}}$$

**Temperature role:** $\tau = 3$ (configured via `--gate-temp 3`) controls the sharpness of the gate:

| $|z|$ | $\tau=1$ | $\tau=3$ (v4.3) | Interpretation |
|--------|----------|-----------------|----------------|
| 1 | $\sigma(1)=0.731$ | $\sigma(0.33)=0.582$ | Soft preference |
| 3 | $\sigma(3)=0.953$ | $\sigma(1.0)=0.731$ | Moderate commitment |
| 8 | $\sigma(8)=0.9997$ | $\sigma(2.67)=0.935$ | Strong routing |
| 33 | $\sigma(33)\approx 1$ | $\sigma(11)=0.9999$ | Near-binary |

Higher temperature $\tau$ means the gate requires larger logit magnitude to commit to hard 0/1 decisions, preserving gradient flow during training while still achieving decisive routing at inference.

### 10.2 Gradient Flow Analysis

The gradient of the loss through a gate is:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \lambda} \cdot \lambda(1-\lambda) \cdot \frac{1}{\tau}$$

The sigmoid derivative $\lambda(1-\lambda)$ approaches zero as $|z| \to \infty$ — the **vanishing gradient problem** for saturated gates. At $z = -37.586$ (observed in Run 14, Epoch 7):

$$\lambda(1-\lambda)\big|_{z=-37.586/3} \approx \sigma(-12.53)(1-\sigma(-12.53)) \approx 3.6 \times 10^{-6}$$

This gate receives essentially zero gradient from the loss and can only be recovered by weight decay ($\nabla_z \mathcal{L}_{\text{WD}} = -\eta \cdot \lambda_{\text{WD}} \cdot z$) or the logit tail penalty (Section 14).

### 10.3 Lambda Distribution Diagnostics

Per-epoch gate health is logged with the following statistics:

| Statistic | Healthy Range | Physical Meaning |
|-----------|---------------|-----------------|
| $\lambda_{\mu}$ | 0.40–0.65 | Average gate activation level |
| $\lambda_{\sigma}$ | 0.15–0.30 | Routing diversity |
| $\lambda_{p05}$ | > 0.05 | Gates not fully off |
| $\lambda_{p95}$ | < 0.95 | Gates not fully on |
| $\texttt{clamp}(\lambda \leq \varepsilon)$ | < 5% | Fraction of fully-off gates |
| $z_{\text{std}}$ | 1.0–3.0 | Logit spread (bulk) |
| $z_{\text{IQR}}$ | $\geq 0.8$ | Logit interquartile range |
| $z_{\text{MAD}}$ | $\geq 0.7$ | Logit median absolute deviation |

---

## 11. Neural CDE Path-Response Layer

### 11.1 Controlled Differential Equation Formulation

The Neural CDE component provides a path-driven response mechanism operating on the ETD-1 output states $\mathbf{z}^{(\text{ETD})}_t$. The latent state $\mathbf{u} \in \mathbb{R}^{d_H}$ evolves according to:

$$d\mathbf{u}_t = f(\mathbf{u}_t;\,\theta)\,d\mathbf{X}_t$$

$$\mathbf{u}_T = \mathbf{u}_0 + \int_0^T f(\mathbf{u}_t;\,\theta)\,d\mathbf{X}_t$$

where $\mathbf{X}_t$ is the **control path** — the continuous interpolation of the joint fusion sequence, and $f: \mathbb{R}^{d_H} \to \mathbb{R}^{d_H \times d_{\text{joint}}}$ is the learned vector field.

### 11.2 Vector Field Network

The vector field $f$ is parameterized as a 2-layer MLP with tanh stabilization:

$$f(\mathbf{u}) = \tanh\!\left(W_2 \cdot \text{SiLU}(W_1 \mathbf{u} + \mathbf{b}_1) + \mathbf{b}_2\right) \in \mathbb{R}^{d_H \times d_{\text{joint}}}$$

The $\tanh$ activation ensures $\|f(\mathbf{u})\|_\infty \leq 1$, bounding the rate of state change per unit of control path increment and preventing state explosion over the $T = 200$ step lookback.

### 11.3 Explicit Euler Discretization

On the observation grid:

$$\mathbf{u}_{t+1} = \mathbf{u}_t + f(\mathbf{u}_t) \cdot (\mathbf{z}^{(\text{joint})}_{t+1} - \mathbf{z}^{(\text{joint})}_t)$$

The **control increment** $d\mathbf{X}_t = \mathbf{z}^{(\text{joint})}_{t+1} - \mathbf{z}^{(\text{joint})}_t$ measures how much the fused representation changes from bar to bar. Regime shifts (large $d\mathbf{X}_t$) drive proportionally larger state updates, while quiet consolidation periods (small $d\mathbf{X}_t$) produce minimal latent state evolution — a natural signal-to-noise filtering property.

### 11.4 Role in the Fused Architecture

The CDE operates **in series** after the ETD-1: the ETD-1 handles temporal integration of the fused multi-modal sequence while the CDE captures the differential path geometry of that sequence. The ETD-1 state $\mathbf{z}^{(\text{ETD})}_T$ provides stable long-range memory; the CDE state $\mathbf{u}_T$ provides a path-sensitive local response to recent control signal structure. Both states are concatenated before the predicate gate layer:

$$\mathbf{h}_T = \text{concat}(\mathbf{z}^{(\text{ETD})}_T,\, \mathbf{u}_T) \in \mathbb{R}^{2d_H}$$

---

## 12. Output Heads & Strategy Universe

### 12.1 Strategy Universe (10 Classes)

CondorNet v4.3 selects among all ten strategy classes simultaneously — not a hierarchical or post-hoc filter:

| Class | Index | Description | Preferred Regime |
|-------|-------|-------------|-----------------|
| `single_call` | 0 | Directional long call | Bullish, low IV |
| `single_put` | 1 | Directional long put | Bearish, low IV |
| `bull_call_spread` | 2 | Long call + short higher call | Moderately bullish |
| `bear_put_spread` | 3 | Long put + short lower put | Moderately bearish |
| `straddle` | 4 | ATM call + ATM put | High IV, large move expected |
| `strangle` | 5 | OTM call + OTM put | High IV, very large move expected |
| `butterfly_call` | 6 | 3-strike call butterfly | Low IV, precise target |
| `iron_condor` | 7 | Short strangle + wings | Neutral, high IV, mean reversion |
| `custom_multi_leg` | 8 | Flexible multi-leg | Context-dependent |
| `abstain` | 9 | No trade | Uncertainty / regime ambiguity |

The **abstain class** is a first-class citizen in the loss function — the model learns *when not to trade* as an intrinsic component of the policy, not as a threshold applied post-hoc.

### 12.2 Five Output Heads

$$\mathbf{h}_T \xrightarrow{\text{5 heads}} \{\hat{\mathbf{s}},\, \hat{p},\, \hat{v},\, \hat{\mathbf{r}},\, \hat{q}\}$$

| Head | Output | Shape | Activation | Loss Type |
|------|--------|-------|------------|-----------|
| **Strategy** | $\hat{\mathbf{s}} = \text{softmax}(W_s \mathbf{h}_T)$ | $[10]$ | Softmax | Cross-entropy |
| **PoP** | $\hat{p} = \sigma(W_p \mathbf{h}_T)$ | $[1]$ | Sigmoid | Binary cross-entropy |
| **EV** | $\hat{v} = W_v \mathbf{h}_T$ | $[1]$ | Linear | MSE |
| **Risk** | $\hat{\mathbf{r}} = W_r \mathbf{h}_T$ | $[3]$ (VaR, CVaR, max\_loss) | Softplus | MSE |
| **Size** | $\hat{q} = \sigma(W_q \mathbf{h}_T)$ | $[1]$ | Sigmoid $\in [0,1]$ | MSE |

### 12.3 Strategy Head Mathematics

$$\hat{\mathbf{s}} = \text{softmax}\!\left(\frac{W_s \mathbf{h}_T}{\sqrt{d_H}}\right), \quad \hat{s}_k = \frac{e^{(W_s \mathbf{h}_T)_k / \sqrt{d_H}}}{\sum_{j=0}^{9} e^{(W_s \mathbf{h}_T)_j / \sqrt{d_H}}}$$

Abstain is triggered at inference if:

$$\max_k \hat{s}_k < \theta_{\text{abstain}} = 0.60$$

The `output_class_norms` in epoch JSONs track the L2 norm of each class's output vector, providing a measure of which strategies the model is committing representational energy to:

```json
"iron_condor": 0.622,  "bear_put_spread": 0.622,  "abstain": 0.621
```

Uniform norms (~0.60) indicate the model is not yet specialized; spreading norms indicate emerging strategy discrimination.

---

## 13. 9-Component Composite Loss Function

### 13.1 Loss Formulation

The total training objective is a weighted sum of nine components:

$$\mathcal{L}_{\text{total}} = \sum_{c \in \mathcal{C}} w_c(e) \cdot \mathcal{L}_c$$

where $w_c(e)$ is the epoch-dependent annealed weight for component $c$.

### 13.2 Component Definitions

| Component | Symbol | Weight (initial) | Type | Formula |
|-----------|--------|-----------------|------|---------|
| `strategy_ce` | $\mathcal{L}_s$ | 1.0 (constant) | Cross-entropy | $-\sum_{k} y_k \log \hat{s}_k$ |
| `pop_bce` | $\mathcal{L}_p$ | 1.0 (constant) | Binary CE | $-y_p \log \hat{p} - (1-y_p)\log(1-\hat{p})$ |
| `ev_mse` | $\mathcal{L}_v$ | 0.5 → 0.8 | MSE | $\|\hat{v} - v^*\|^2$ |
| `risk_mse` | $\mathcal{L}_r$ | 0.5 (constant) | MSE | $\|\hat{\mathbf{r}} - \mathbf{r}^*\|^2$ |
| `size_mse` | $\mathcal{L}_q$ | 0.4 (constant) | MSE | $\|\hat{q} - q^*\|^2$ |
| `spot_mse` | $\mathcal{L}_{\text{spot}}$ | 0.0 (disabled) | MSE | Auxiliary target spot price |
| `fuzzy_var` | $\mathcal{L}_{\text{fv}}$ | 0.2 → 0.05 | Regularizer | Gate output variance penalty |
| `pattern_ent` | $\mathcal{L}_{\text{ent}}$ | 0.1 (constant) | Regularizer | $-\sum_k \lambda_k \log \lambda_k$ |
| `robust` | $\mathcal{L}_{\text{hub}}$ | 0.2 (constant) | Huber | $\text{Huber}_\delta(\hat{v} - v^*, \hat{\mathbf{r}} - \mathbf{r}^*)$ |

### 13.3 Annealing Schedules

Weight annealing is linear interpolation between start and end values:

$$w_c(e) = w_c^{(0)} + \frac{\min(e, e_{\text{end}}) - e_{\text{start}}}{e_{\text{end}} - e_{\text{start}}} \cdot \left(w_c^{(T)} - w_c^{(0)}\right)$$

| Component | $e_{\text{start}}$ | $e_{\text{end}}$ | $w^{(0)}$ | $w^{(T)}$ | Rationale |
|-----------|--------------------|-----------------|-----------|-----------|-----------|
| `pop_bce` | 0 | 1 | 1.0 | 1.0 | Constant — plateaus early (~0.613) |
| `ev_mse` | 5 | 20 | 0.5 | 0.8 | Delayed ramp: stabilize before EV supervision |
| `fuzzy_var` | 10 | 30 | 0.2 | 0.05 | Decay as model stabilizes routing |

**Critical engineering note — pop_bce:** The pop_bce loss plateaus at entropy $-\ln(0.5) \approx 0.613$ when the PoP head converges to the unconditional prior. Any weight ramp above 1.0 would monotonically inflate val loss by $\Delta w \times 0.613$ per epoch — a measurement artifact that would cause early-stopping to select a suboptimal epoch. The fix (commit `6186ee2`) eliminates this artifact by keeping pop_bce weight constant at 1.0.

### 13.4 Huber Loss Component

$$\text{Huber}_\delta(e) = \begin{cases} \frac{1}{2}e^2 & |e| \leq \delta \\ \delta|e| - \frac{1}{2}\delta^2 & |e| > \delta \end{cases}$$

Applied jointly to EV and risk predictions, the Huber component provides outlier robustness: large prediction errors (e.g., rare high-vol days) are penalized linearly rather than quadratically, preventing the loss from being dominated by a small number of extreme samples.

---

## 14. Gate Regularization System

### 14.1 Motivation

Without explicit regularization, gate logits drift toward saturation under gradient descent: the model discovers that committing strongly to binary routing reduces training loss, but saturated gates ($|z| \gg 8$) have near-zero gradients and lose the ability to adapt. Four complementary regularizers maintain gate logit diversity and prevent outlier saturation.

### 14.2 Floor 1: Variance Penalty

$$\mathcal{L}_{\text{var}} = \alpha_z \cdot \max\!\left(0,\, v_{\text{target}} - \text{Var}(z)\right)^2, \quad \alpha_z = 0.10, \quad v_{\text{target}} = 2.0$$

**Target:** $\text{Var}(z) \geq 2.0$. The squared hinge ensures the penalty is zero when variance is healthy and grows quadratically when it collapses.

### 14.3 Floor 2: IQR Penalty

$$\mathcal{L}_{\text{IQR}} = \alpha_{\text{IQR}} \cdot \max\!\left(0,\, I_{\text{target}} - \text{IQR}(z)\right)^2, \quad \alpha_{\text{IQR}} = 0.08, \quad I_{\text{target}} = 1.0$$

**Target:** $\text{IQR}(z) \geq 1.0$. The IQR is robust to outliers — it measures the spread of the middle 50% of logits and penalizes collapse of the bulk distribution independently of extreme values.

### 14.4 Floor 3: MAD Penalty

$$\mathcal{L}_{\text{MAD}} = \alpha_{\text{MAD}} \cdot \max\!\left(0,\, M_{\text{target}} - \text{MAD}(z)\right)^2, \quad \alpha_{\text{MAD}} = 0.05, \quad M_{\text{target}} = 0.9$$

**Target:** $\text{MAD}(z) \geq 0.9$. The Median Absolute Deviation $= \text{median}(|z - \text{median}(z)|)$ provides a third, independently robust measure of spread. The three-floor system (var + IQR + MAD) cross-validates gate diversity — a gate distribution can fail one check while passing others, so requiring all three provides stronger guarantees.

### 14.5 Floor 4: Logit Tail Penalty

The three floor penalties protect the **bulk distribution** but are insensitive to rare extreme outliers: the squared penalty on mean values is dominated by the bulk, so a single gate at $z = -37$ contributes only $\sim 0.003\%$ to a mean computed over thousands of gates.

The **logit tail penalty** directly targets absolute excess beyond a saturation band $L = 8$:

$$\mathcal{L}_{\text{tail}} = \alpha_{\text{tail}} \cdot \mathbb{E}\!\left[\max(0,\, |z| - L)^2\right]$$

where $L = 8$ corresponds to $\sigma(\pm 8) = 0.9997$ — full saturation. Any gate at $|z| > 8$ is effectively hard-clamped and contributes a squared penalty proportional to its excess.

**Gradient effect:** For a gate at $z = -37$ with $L = 8$:

$$\frac{\partial \mathcal{L}_{\text{tail}}}{\partial z} = \alpha_{\text{tail}} \cdot 2 \cdot (|z| - L) \cdot \text{sign}(z) = 0.03 \times 2 \times 29 \times (-1) = -1.74$$

This is a direct, strong pull toward $z = 0$ regardless of the sigmoid gradient vanishing. At $\alpha_{\text{tail}} = 0.01$ (Run 12) this gradient was $-0.58$ — present but insufficient to overcome weight accumulation. At $\alpha_{\text{tail}} = 0.1$ (Run 13) it was $-5.8$ — strong enough to disrupt routing structure. The empirically optimal value $\alpha_{\text{tail}} = 0.03$ (Run 14, best val = 0.9553) provides a gradient of $-1.74$, balancing suppression of outliers with preservation of the model's ability to commit to strong routing decisions.

### 14.6 Diversity Penalty

$$\mathcal{L}_{\text{div}} = \alpha_{\text{div}} \cdot \max\!\left(0,\, \sigma_{\text{target}} - \text{std}(\lambda)\right)^2, \quad \alpha_{\text{div}} = 0.005, \quad \sigma_{\text{target}} = 0.14$$

Penalizes collapse of the $\lambda$ distribution toward uniformity, ensuring the model maintains diverse routing strength across gates.

### 14.7 Warmup Behavior

Gate regularizers are applied from the first batch (`--logit-warmup-frac 0.0`). This ensures stable routing from epoch 1 — in earlier runs without warmup, the first 1–2 epochs showed logit collapse that set a poor routing trajectory for subsequent epochs.

### 14.8 Regularizer Interaction Summary

| Regularizer | Targets | Blind Spots |
|-------------|---------|-------------|
| Variance floor | Bulk spread collapse | Outlier saturation |
| IQR floor | Bulk spread (robust) | Outlier saturation |
| MAD floor | Bulk spread (robust) | Outlier saturation |
| **Tail penalty** | **Outlier saturation** | Bulk distribution |
| Diversity penalty | $\lambda$ uniformity collapse | — |

The system is designed so that each regularizer fills the blind spots of the others.

---

## 15. Normalization & Data Pipeline

### 15.1 RobustScaler

All continuous features are normalized using RobustScaler (fit on train split only):

$$x_{\text{scaled}} = \frac{x - \text{median}(x)}{1.4826 \times \text{MAD}(x)}$$

The constant 1.4826 makes MAD consistent with standard deviation for normal distributions. Clipping to $[-10, 10]$ prevents Black Swan market events from generating pathological gradients.

### 15.2 Passthrough Features

The following features are **excluded from normalization** (passthrough):

- **Ternary features:** `breakout_score`, `psar_trend`, `ivr_zone`, `stretch_zone` ∈ {-1, 0, 1}
- **Bounded [0,1]:** `atr_pct`, `rsi_dyn`, `stoch_k_dyn`, `adx_adaptive`, `bb_percentile`, `psar_reversion_mu`, `max_dd_60m`, `consolidation_score`, `chaos_membership`, `fuzzy_reversion_11`, `pressure_up`, `pressure_down`, `vol_ewma`, `spread_ratio`, `gap_risk_score`
- **Sparse pivot features:** All 13 pivot features (NaN must be preserved)
- **Binary friction gates:** `friction_ok_5/10/20/40/60` ∈ {0, 1}
- **Circular encodings:** `tod_sin`, `tod_cos` ∈ [-1, 1]
- **Regime persistence counter:** `regime_persistence`

### 15.3 V43Dataset Sequence Construction

The `V43Dataset` constructs fixed-length windows of shape $[\mathcal{B}, T, d_{\text{feat}}]$:

$$\mathcal{D} = \left\{(X_{\tau:\tau+T},\, \mathbf{y}_{\tau+T}) \mid \tau \in [0,\, N-T)\right\}$$

where $T = 200$ (lookback), $N$ = total M5 rows = 18,494, yielding approximately 18,294 sequences.

**Train/val split:** Chronological (no shuffle) — first 80% by timestamp for training, last 20% for validation. This prevents look-ahead bias that would result from random splitting of time series data.

---

## 16. Training Configuration

### 16.1 Current Optimal Configuration (Run 14)

```bash
py -3.12 intelligence/condor_train_net_v43.py \
  --data-dir data/Datasetv4/v43/ \
  --m1-file m1_dataset_v43_final.csv \
  --m5-file m5_dataset_v43_final.csv \
  --m15-file m15_dataset_v43_final.csv \
  --h1-file h1_dataset_v43_final.csv \
  --options-file options_2025_v43.csv \
  --d-joint 128 --epochs 40 --batch-size 256 --lr 1e-4 \
  --lookback 200 --accum-steps 2 --patience 25 \
  --d-h 128 --d-v 16 --d-m 32 --d-r 16 --d-control 64 \
  --n-predicates 64 --n-sets 32 --n-super-sets 16 --gate-temp 3 \
  --diversity-alpha 0.005 --diversity-std-target 0.14 \
  --logit-var-alpha 0.10 --logit-var-target 2.0 \
  --logit-iqr-alpha 0.08 --logit-iqr-target 1.0 \
  --logit-mad-alpha 0.05 --logit-mad-target 0.9 \
  --logit-tail-alpha 0.03 --logit-tail-band 8.0 \
  --weight-decay 1e-3 \
  --report-dir reports/v43TrainRun14 \
  --output models/condor_net_v43.pth
```

### 16.2 Hyperparameter Reference Table

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--d-joint` | 128 | JointFusion output dimension |
| `--d-h` | 128 | ETD-1 / CDE hidden state dimension |
| `--d-v` | 16 | Pivot projection dimension |
| `--d-m` | 32 | Momentum sub-space dimension |
| `--d-r` | 16 | Residual connection dimension |
| `--d-control` | 64 | CDE control dimension |
| `--n-predicates` | 64 | Number of relational predicates |
| `--n-sets` | 32 | Sets per superset |
| `--n-super-sets` | 16 | Total supersets |
| `--gate-temp` | 3 | Gate sigmoid temperature $\tau$ |
| `--lookback` | 200 | Sequence lookback $T$ |
| `--batch-size` | 256 | Per-GPU batch size |
| `--accum-steps` | 2 | Gradient accumulation steps (effective batch = 512) |
| `--lr` | $10^{-4}$ | Peak learning rate (cosine decay) |
| `--weight-decay` | $10^{-3}$ | AdamW weight decay |
| `--patience` | 25 | Early stopping patience (epochs) |
| Total Parameters | 10,955,687 | ~11M |

### 16.3 Cosine Learning Rate Schedule

$$\text{LR}(e) = \text{LR}_{\text{min}} + \frac{1}{2}(\text{LR}_{\text{max}} - \text{LR}_{\text{min}})\left(1 + \cos\!\left(\frac{\pi e}{E_{\text{total}}}\right)\right)$$

with $\text{LR}_{\text{max}} = 10^{-4}$, $\text{LR}_{\text{min}} \approx 0$, $E_{\text{total}} = 40$. The LR decays from $9.98 \times 10^{-5}$ at epoch 1 to $1.20 \times 10^{-5}$ at epoch 31 (typical early stop epoch).

### 16.4 Gradient Accumulation

Effective batch gradient:

$$\nabla_\theta^{\text{eff}} = \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta \mathcal{L}^{(k)}, \qquad K = 2$$

Effective batch size = $256 \times 2 = 512$ sequences. This provides stable gradient estimates across more diverse market contexts per update while maintaining GPU memory efficiency.

---

## 17. Empirical Convergence Analysis — Runs 7–14

### 17.1 Run Summary Table

| Run | Key Change | Best Epoch | Best Val | Notes |
|-----|-----------|-----------|---------|-------|
| 7 | pop_bce ramp 0.8→1.2 (bug) | E2 | 0.9586 | Annealing artifact: true best was E7 (0.8806 corrected) |
| 12 | pop_bce fixed at 1.0; α_tail=0.01 | E6 | 0.9574 | max\|z\|→59 (penalty too weak) |
| 13 | α_tail=0.1 | E5 | 0.9895 | Regression: penalty disrupted routing at E7 |
| **14** | **α_tail=0.03** | **E7** | **0.9553** | **New best; max\|z\| contained ~30** |

### 17.2 Pop_bce Annealing Artifact (Run 7 Diagnosis)

The pop_bce head converges to the unconditional binary entropy $H(0.5) = \ln 2 \approx 0.613$ by epoch 2 — the PoP signal is difficult to predict beyond the prior. With a weight ramp from 0.8 to 1.2 over 10 epochs, the measured val loss includes an artificial linear trend:

$$\mathcal{L}_{\text{val}}^{\text{measured}}(e) = \mathcal{L}_{\text{val}}^{\text{true}}(e) + \underbrace{0.613 \times \Delta w(e)}_{\text{artifact}}$$

At $\Delta w = 0.04$ per epoch, the artifact adds $\approx 0.025$ per epoch to the measured val loss, causing early stopping to select E2 (best measured) when the true minimum was E7 (val = 0.8806 corrected). Fix: constant pop_bce weight = 1.0.

### 17.3 Run 14 Convergence Detail

```
Epoch  Train   Val     diff_sets  max|z|   tail%  mean_excess  LR
  1   2.136   1.272      13      23.31    1.95%   0.164    9.98e-05
  2   1.145   1.043       8      25.76    2.53%   0.239    9.94e-05
  3   1.004   1.010       6      27.61    2.97%   0.313    9.86e-05
  4   0.951   1.002       5      28.99    2.93%   0.325    9.76e-05
  5   0.923   0.990       4      30.52    2.13%   0.268    9.62e-05
  6   0.907   1.005       4      30.88    1.53%   0.129    9.46e-05
  7★  0.882   0.9553      6      16.50    1.54%   0.007    9.26e-05  ← BEST
  8   0.868   0.993       6      18.89    2.54%   0.013    9.05e-05
  9   0.862   0.994       7      19.84    1.95%   0.008    8.80e-05
 10   0.841   0.992       6      21.58    1.70%   0.005    8.54e-05
```

**Phase transition at Epoch 7:** max|z| drops sharply from 30.88 (E6) to 16.50 (E7) — the tail penalty accumulated sufficient gradient pressure at α=0.03 to pull back the outlier gates between epochs. This represents successful gate logit normalization rather than routing disruption (unlike Run 13 where the same transition at α=0.1 degraded val loss).

### 17.4 Train-Val Gap Analysis

| Run | Gap at Best Epoch | Gap at Final Epoch | Interpretation |
|-----|-------------------|-------------------|----------------|
| 12 | 0.060 (E6: 0.897 vs 0.957) | 0.298 (E31: 0.719 vs 1.017) | Genuine overfitting after E6 |
| 13 | 0.063 (E5: 0.923 vs 0.990) | 0.315 (E30: 0.736 vs 1.051) | Similar gap, worse floor |
| 14 | 0.068 (E7: 0.882 vs 0.955) | — (still running at E21) | E7 still holds |

The train-val gap at the best epoch (~0.06–0.07) is consistent across all runs, suggesting the gap is determined by the intrinsic difficulty of the task (dataset size, label noise, distribution complexity) rather than regularization inadequacy.

---

## 18. Spectral Stability & A-Matrix Analysis

### 18.1 Spectral Radius as a Training Diagnostic

The backbone spectral radius $\rho(\mathbf{J})$ where $\mathbf{J}$ is the Jacobian of the full forward pass is computed per epoch. The `spectral_radius_delta` in the EPOCH_Comparison JSON records:

$$\Delta\rho(e) = \rho(\mathbf{J}^{(e)}) - \rho(\mathbf{J}^{(e^*)})$$

where $e^*$ is the best-model epoch. Positive $\Delta\rho$ indicates growing amplification relative to the best checkpoint.

**Empirical patterns across runs:**
- $\Delta\rho$ growing monotonically positive: overfitting (model amplifies training features)
- $\Delta\rho$ near zero: stable routing regime
- $\Delta\rho$ negative then positive: initial compression then expansion

In Runs 12–14, spectral radius grows from +0.013 at E1 to +0.163 at E31, consistent with the model building increasingly committed (and training-set-specific) routing paths.

### 18.2 A-Matrix: Correlation Structure

Each epoch produces an activation correlation matrix saved as `Epoch{N}_A_Matrix.csv`. The Pearson correlation $\rho_{ij}$ between predicate activations $\lambda_i$ and $\lambda_j$ across the validation set:

$$\rho_{ij} = \frac{\text{Cov}(\lambda_i, \lambda_j)}{\sqrt{\text{Var}(\lambda_i) \cdot \text{Var}(\lambda_j)}}$$

The overall Frobenius norm of the off-diagonal A-matrix:

$$\rho_{\text{epoch}} = \frac{1}{N(N-1)} \sum_{i \neq j} |\rho_{ij}|$$

is logged as `rho=X.XXXX` in training output. High $\rho_{\text{epoch}}$ indicates predicate co-activation (correlated routes); low $\rho_{\text{epoch}}$ indicates diverse, independent route usage. Target: $\rho_{\text{epoch}} < 0.15$ (Run 14 Epoch 7: $\rho = 0.1301$).

---

## 19. Epoch Interpretability System

### 19.1 Epoch JSON Structure

Each training epoch produces a JSON artifact at `reports/{run}/Epoch{N}_CNv43_DSv43_{timestamp}.json` with the following top-level structure:

```json
{
  "summary": {
    "epoch": 7,
    "version": "CNv43_DSv43",
    "schema_version": "v4.3.0",
    "train_loss": 0.8818,
    "val_loss": 0.9553,
    "val_train_gap": 0.0735,
    "is_best": true,
    "parameters": 10955687,
    "architecture": {"d_joint": 128, "d_chain": 128, "d_pivot": 16, "n_strategy_types": 10}
  },
  "learned_logic": {
    "predicates": { ... },
    "super_set": { "n_super_sets": 16, "super_sets": [ ... ] },
    "strategy_head": { "output_class_norms": { ... } },
    "risk_head": { ... },
    "pivot_head": { ... },
    "fuzzy_gates": { ... }
  },
  "a_matrix": { ... }
}
```

### 19.2 Superset Readout

Each superset entry contains its 32 sets, each with 5 top comparisons resolved to human-readable form:

```json
{
  "index": 0,
  "n_sets": 32,
  "sets": [{
    "index": 0,
    "operator_weights": {"<": 33.1, ">": 33.0, "=": 33.9},
    "top_pair_indices": [1238, 705, 570, 259, 209],
    "top_comparisons": [
      {"comparison": "bb_upper_dyn > bb_sigma_dyn", "dominant_op": ">",
       "weights": {"<": 0.0498, ">": 0.0587, "=": 0.0332}},
      {"comparison": "rsi_dyn > bandwidth",         "dominant_op": ">",
       "weights": {"<": 0.0437, ">": 0.0656, "=": 0.0094}},
      {"comparison": "atr_pct < ivr_zone",          "dominant_op": "<",
       "weights": {"<": 0.0493, ">": 0.0325, "=": 0.0289}}
    ]
  }]
}
```

The pair index resolves to feature names via the full 64-entry `_FULL_FEATURE_NAMES` list (indices 0–51: base TF features; 52–63: ETL-computed friction, ToD, regime, and IVR reversal features).

### 19.3 EPOCH_Comparison JSON

The comparison JSON tracks structural drift from the best-model checkpoint:

| Field | Meaning |
|-------|---------|
| `same_predicates` / `different_predicates` | Number of predicate gate activations unchanged |
| `same_sets` / `different_sets` | Number of set routing paths unchanged (out of 512) |
| `same_super_sets` / `different_super_sets` | Superset routing stability |
| `spectral_radius_delta` | $\Delta\rho$ from best checkpoint |
| `strategy_head_diff.changed_classes` | Strategy classes with different argmax outputs |
| `val_loss` / `train_loss` | Per-epoch losses for trend analysis |

**`different_sets` interpretation:** At the best epoch, typically 4–6 different sets — the model has found a stable routing structure. In later overfit epochs, different_sets grows to 15–20, indicating the model is still searching for optimal routing rather than having converged.

### 19.4 Interpretability Mathematics: Pair Index Decoding

Feature pair $(i,j)$ is stored as the index $k$ into the upper triangular matrix:

$$k = \text{triu\_index}(i, j, n) = ni - \frac{i(i+1)}{2} + j - i - 1, \qquad 0 \leq i < j < n$$

Inverse lookup:

$$i = \left\lfloor \frac{2n - 1 - \sqrt{(2n-1)^2 - 8k}}{2} \right\rfloor, \qquad j = k - \text{triu\_index}(i, i+1, n) + i + 1$$

Implemented via `numpy.triu_indices(n_inputs, k=1)` for $n_{\text{inputs}} = 64$, yielding $\binom{64}{2} = 2{,}016$ unique feature pairs.

---

## 20. Iron Condor P&L Mathematics

### 20.1 Four-Leg Structure

An Iron Condor consists of exactly four legs. The fail-fast rule in `build_condor()` rejects any trade where a valid leg price cannot be obtained:

| Leg | Strike | Direction | Delta Target |
|-----|--------|-----------|-------------|
| Short Call | $K_{sc} = S_t + \delta_C \cdot \sigma_{\text{ATM}}$ | Sell | $[0.15, 0.20]$ |
| Long Call | $K_{lc} = K_{sc} + W$ | Buy | — |
| Short Put | $K_{sp} = S_t - \delta_P \cdot \sigma_{\text{ATM}}$ | Sell | $[-0.20, -0.15]$ |
| Long Put | $K_{lp} = K_{sp} - W$ | Buy | — |

### 20.2 Credit and P&L Boundaries

**Net credit received:**

$$C_{\text{net}} = (C_{K_{sc}} - C_{K_{lc}}) + (P_{K_{sp}} - P_{K_{lp}})$$

where $C_K$ and $P_K$ denote call and put mid-prices at strike $K$.

**Maximum profit:** $\Pi_{\max} = C_{\text{net}} \times N \times 100$

**Maximum loss:** $L_{\max} = (W - C_{\text{net}}) \times N \times 100$

**Probability of Profit (theoretical BSM):**

$$\text{PoP} = N(d_1(K_{sc})) - N(d_1(K_{sp}))$$

where $N(\cdot)$ is the standard normal CDF and $d_1(K) = [\ln(S/K) + (r + \sigma^2/2)\tau] / (\sigma\sqrt{\tau})$.

### 20.3 Mark-to-Market P&L

$$\text{PnL}(t) = (C_{\text{net}} - \text{CurrentCost}(t)) \times N \times 100$$

$$\text{CurrentCost}(t) = (C_{K_{sc},t} - C_{K_{lc},t}) + (P_{K_{sp},t} - P_{K_{lp},t})$$

where all option prices are mid-prices from the synthetic Black-Scholes chain at time $t$.

### 20.4 VaR and CVaR (Target Labels)

**Value at Risk (95%):**

$$\text{VaR}_{95} = \inf\{l : P(\text{Loss} > l) \leq 0.05\}$$

**Conditional VaR (Expected Shortfall at 95%):**

$$\text{CVaR}_{95} = \frac{1}{0.05} \int_{0.95}^{1} \text{VaR}_u\, du = \mathbb{E}[\text{Loss} \mid \text{Loss} \geq \text{VaR}_{95}]$$

Both are estimated from Monte Carlo simulation of the options positions over the distribution of SPY log-returns. These serve as regression targets for the `risk_head`.

---

## 21. Audit & Deployment Gate

### 21.1 Checkpoint Stability Criteria

A model checkpoint is deployment-safe if all of the following hold:

| Criterion | Threshold | Diagnostic |
|-----------|-----------|------------|
| Val loss improvement | $\mathcal{L}_{\text{val}}^{(e)} < \mathcal{L}_{\text{val}}^{(e^*-1)}$ | Best epoch selection |
| Train-val gap at best epoch | $< 0.10$ | Generalization quality |
| Gate logit std | $1.0 \leq z_{\text{std}} \leq 3.0$ | Routing diversity |
| Gate logit IQR | $\geq 0.8$ | Bulk spread |
| Gate logit MAD | $\geq 0.7$ | Robust spread |
| Tail fraction | $< 10\%$ gates beyond $|z| = 8$ | Saturation control |
| Lambda clamp fraction | $< 5\%$ at $\lambda \leq \varepsilon$ | Hard-off gates |
| Spectral radius delta | $|\Delta\rho| < 0.10$ from best | Stability of routing |
| Different sets | $\leq 8$ from best checkpoint | Structural stability |
| Strategy head norms | No single class dominates by $> 0.15$ | Calibration balance |

### 21.2 Deployment Abstain Logic

At inference, the model abstains if:

$$\max_k \hat{s}_k < 0.60 \quad \lor \quad \hat{p} < 0.45 \quad \lor \quad \text{friction\_gate} = 0$$

The friction gate requires at least one `friction_ok_N = 1` for any window $N \in \{5, 10, 20, 40, 60\}$ — equivalent to requiring that the bid-ask spread is smaller than the recent price range at some lookback scale. This is enforced **per leg** in multi-leg strategies.

### 21.3 Perturbation Attribution

Feature importance is assessed via permutation:

$$I_j = \frac{1}{K} \sum_{k=1}^{K} \left[\mathcal{L}(f(X^{(j,k)}), y) - \mathcal{L}(f(X), y)\right]$$

where $X^{(j,k)}$ has feature $j$ randomly permuted across the validation batch. High $I_j$ identifies features critical to the current checkpoint's decisions.

### 21.4 Gradient Saliency

Local feature sensitivity at inference:

$$S_j = \frac{1}{N} \sum_{i=1}^{N} \left|\frac{\partial \hat{s}^{(i)}}{\partial x_j^{(i)}}\right|$$

Combined with the `top_comparisons` readout from the epoch JSON, this provides a two-level interpretability system: $S_j$ gives aggregate feature importance; `top_comparisons` gives the specific relational rules the model has learned to use.

### 21.5 Information Geometry: Hessian Analysis

The Hessian trace of the loss landscape:

$$\text{Tr}(H) = \text{Tr}(\nabla^2 \mathcal{L})$$

Lower trace correlates with flatter minima and improved out-of-distribution generalization (PAC-Bayes sharpness bounds). Monitored per checkpoint to ensure training converges to flat minima rather than sharp basins.

---

## Appendix A: Architecture Quick-Reference

```
Raw Input (4 TF × [B,T,52]) + Options Chain ([B,120,10]) + Pivot ([B,T,13])
         │                           │                        │
   MultiTFProjector             OptionsChainEncoder      PivotProjector
   4×[B,T,128]                  [B,128]                  [B,T,16]
         │                           │                        │
         └──────────── TFFusionBlock ─────────────────────────┘
                            [B,T,256]
                                │
                        JointFusionLayer
                    [B,T,384] → [B,T,128]
                                │
                         ETD-1 Backbone
                  z_{t+1} = e^{hL}z_t + hφ₁(hL)g(t)
                            [B,T,128]
                                │
                    Neural CDE Path-Response
                  u_{t+1} = u_t + f(u_t)·dX_t
                            [B,T,128]
                                │
              concat([z_T, u_T]) [B,256]
                                │
          8 Predicate Gates → 64 Predicates
          C(64,2)=2016 pairs → 32 Sets → 16 Supersets
          λ = σ(z/τ),  τ=3
                                │
                        5 Output Heads
          strategy(10) · pop(1) · ev(1) · risk(3) · size(1)
```

---

## Appendix B: Feature Index Map (Full 64-Entry)

| Idx | Feature | Group | Idx | Feature | Group |
|-----|---------|-------|-----|---------|-------|
| 0 | open | Price | 32 | OSC-Volume | Volume |
| 1 | high | Price | 33 | vol_energy | Volume |
| 2 | low | Price | 34 | vol_ewma | Volume |
| 3 | close | Price | 35 | spread_ratio | Volume |
| 4 | MA | Price | 36 | psar_adaptive | Regime |
| 5 | FRAMA | Price | 37 | psar_mark | Regime |
| 6 | sma | Price | 38 | psar_trend | Regime |
| 7 | TurtleChn-Up | Price | 39 | psar_reversion_mu | Regime |
| 8 | TurtleChn-Low | Price | 40 | pressure_up | Regime |
| 9 | TrendSeekerUp | Price | 41 | pressure_down | Regime |
| 10 | TrendSeekerDown | Price | 42 | max_dd_60m | Regime |
| 11 | AnchoredVWAP | Price | 43 | bb_percentile | Regime |
| 12 | rsi_dyn | Momentum | 44 | fuzzy_reversion_11 | Regime |
| 13 | stoch_k_dyn | Momentum | 45 | consolidation_score | Regime |
| 14 | adx_adaptive | Momentum | 46 | gap_risk_score | Risk |
| 15 | ret_z | Momentum | 47 | chaos_membership | Risk |
| 16 | log_return | Momentum | 48 | kappa_proxy | Risk |
| 17 | AccDistWill | Momentum | 49 | McClellanOsc | Risk |
| 18 | AccDistWillMovAvg | Momentum | 50 | Slope | Risk |
| 19 | WilderAccSwingIndex | Momentum | 51 | SlopeATR | Risk |
| 20 | breakout_score | Momentum | 52 | friction_ok_5 | ETL |
| 21 | WeightedAlpha | Momentum | 53 | friction_ok_10 | ETL |
| 22 | ATR_recon | Volatility | 54 | friction_ok_20 | ETL |
| 23 | atr_pct | Volatility | 55 | friction_ok_40 | ETL |
| 24 | bb_upper_dyn | Volatility | 56 | friction_ok_60 | ETL |
| 25 | bb_lower_dyn | Volatility | 57 | tod_sin | ETL |
| 26 | bb_mu_dyn | Volatility | 58 | tod_cos | ETL |
| 27 | bb_sigma_dyn | Volatility | 59 | regime_persistence | ETL |
| 28 | bandwidth | Volatility | 60 | price_stretch | IVR |
| 29 | bw_expansion_rate | Volatility | 61 | ivr_zone | IVR |
| 30 | volume | Volume | 62 | stretch_zone | IVR |
| 31 | trade_count | Volume | 63 | reversal_score | IVR |

---

*Document Version: 4.3.0 | Last Updated: 2026-02-24 | Active model: `models/condor_net_v43.pth`*
