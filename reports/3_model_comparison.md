# Multi-Model Neural CDE Interpretability Audit Report

**Generated:** 2026-01-31 10:08:53
**Models Compared:** 3
**Samples Used:** 3000
**Seed:** 42

---

## Appendix A: Mathematical Foundations

This appendix provides the mathematical basis for the interpretability metrics used throughout this audit.

### A.1 Neural CDE Architecture

The CondorBrain uses **Neural Controlled Differential Equations** for continuous-time market dynamics modeling.

**Core CDE Equation:**

$$
dZ_t = f(Z_t; \theta) \, dX_t
$$

Where $Z_t \in \mathbb{R}^H$ is the latent state and $f: \mathbb{R}^H \to \mathbb{R}^{H \times D}$ is the learned vector field.

**Integral Form:**

$$
Z_T = Z_0 + \int_0^T f(Z_t) \, dX_t
$$

**Explicit Euler Discretization:**

$$
Z_{t+1} = Z_t + f(Z_t) \cdot (X_{t+1} - X_t)
$$

**Vector Field Network (Stability-Bounded):**

$$
f(z) = \tanh(W_2 \cdot \text{SiLU}(W_1 \cdot z + b_1) + b_2)
$$

The tanh activation ensures $\|f(z)\|_\infty \leq 1$, preventing state explosion over long sequences.

### A.2 Permutation Importance

Feature importance is computed by measuring prediction degradation when features are randomly shuffled:

$$
I_j = \frac{1}{K} \sum_{k=1}^{K} \left[ \mathcal{L}(f(X^{(j,k)}), y) - \mathcal{L}(f(X), y) \right]
$$

Where $X^{(j,k)}$ denotes data with feature $j$ permuted in repetition $k$.

**Normalization:**

$$
\hat{I}_j = \frac{I_j}{\sum_{i=1}^{D} I_i} \times 100\%
$$

### A.3 Divergence Metrics

**Cosine Similarity:**

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

**Jensen-Shannon Divergence:**

$$
D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)
$$

Where $M = \frac{1}{2}(P + Q)$ and $D_{KL}$ is the Kullback-Leibler divergence:

$$
D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

**Wasserstein Distance (Earth Mover's):**

$$
W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]
$$

For 1D distributions, this simplifies to the area between CDFs:

$$
W_1(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| \, dx
$$

### A.4 Gradient Saliency

Saliency maps measure input sensitivity via backpropagated gradients:

$$
S_j = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial \hat{y}^{(i)}}{\partial x_j^{(i)}} \right|
$$

**Temporal Saliency (across sequence positions):**

$$
S_t = \sum_{j=1}^{D} \left| \frac{\partial \hat{y}}{\partial x_{t,j}} \right|
$$

**Gradient Alignment (between models A and B):**

$$
\rho_{grad} = \frac{\mathbf{g}_A \cdot \mathbf{g}_B}{\|\mathbf{g}_A\| \|\mathbf{g}_B\|}
$$

### A.5 Fisher Information Matrix

The Fisher Information Matrix captures parameter sensitivity:

$$
\mathcal{F}_{ij} = \mathbb{E}\left[ \frac{\partial \log p(y|x;\theta)}{\partial \theta_i} \frac{\partial \log p(y|x;\theta)}{\partial \theta_j} \right]
$$

**Empirical Estimation (diagonal approximation):**

$$
\hat{\mathcal{F}}_{jj} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{\partial \mathcal{L}^{(i)}}{\partial \theta_j} \right)^2
$$

**Fisher Overlap (between models):**

$$
\text{Overlap} = \frac{\mathbf{f}_A \cdot \mathbf{f}_B}{\|\mathbf{f}_A\| \|\mathbf{f}_B\|}
$$

### A.6 Hessian Eigenspectrum

The Hessian matrix of the loss landscape reveals optimization geometry:

$$
H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}
$$

**Eigenvalue Interpretation:**

- $\lambda_{max} > 0$: Maximum curvature (sharpness of minimum)
- $\text{Tr}(H) = \sum_i \lambda_i$: Average curvature
- $\lambda_{max} / \lambda_{min}$: Condition number (training stability)

**Flatness Correlation:**
Flatter minima (lower $\lambda_{max}$) correlate with better generalization.

### A.7 Stability Physics

**Mean Squared Deviation (MSD):**

$$
\text{MSD}(A, B) = \frac{1}{N} \sum_{i=1}^{N} (y_A^{(i)} - y_B^{(i)})^2
$$

**Variance Ratio:**

$$
\rho_{var} = \frac{\text{Var}(y_A)}{\text{Var}(y_B)}
$$

Values near 1.0 indicate similar output distributions; deviations suggest regime sensitivity differences.

**Kolmogorov-Smirnov Statistic:**

$$
D_{KS} = \sup_x |F_A(x) - F_B(x)|
$$

Maximum difference between empirical CDFs, sensitive to both location and shape differences.

---

## 1. Executive Summary

**Best Model:** `epoch_3_013026`
**Overall Health:** 12 strengths identified, 9 concerns flagged

**Key Divergence (epoch_3_013026 vs epoch_4_013026):**
- Cosine Similarity: 0.983
- Spearman Rank Correlation: 0.984
- Top-10 Feature Overlap: 100%

## 2. Configuration & Preprocessing Contract

### Feature Registry
- **Input Dimension:** 54
- **Feature Schema:** V2.2 (FEATURE_COLS_V22)
- **Total Features:** 54

### Sequence Configuration
- `epoch_3_013026`: seq_len = 240
- `epoch_4_013026`: seq_len = 240
- `epoch_5_013026`: seq_len = 240

### Leakage Masking
- `target_spot` -> 0
- `max_dd_60m` -> 0

### Scaling
- Method: Robust Scaling (median/MAD)
- Clip Range: [-10, +10]
- NaN/Inf Handling: Replace with 0

## 3. Per-Model Interpretability

### epoch_3_013026

**All Features (Permutation Importance):**
| Rank | Feature | Importance (%) |
|------|---------|----------------|
| 1 | ivr | 11.88 |
| 2 | theta | 9.55 |
| 3 | delta | 8.47 |
| 4 | pressure_down | 6.25 |
| 5 | gap_risk_score | 5.12 |
| 6 | pressure_up | 4.89 |
| 7 | bw_expansion_rate | 4.49 |
| 8 | macd_histogram | 4.35 |
| 9 | macd_norm | 3.73 |
| 10 | atr_pct | 3.13 |
| 11 | macd_signal_norm | 3.12 |
| 12 | plus_di | 2.35 |
| 13 | bb_sigma_dyn | 2.20 |
| 14 | vol_ewma | 2.00 |
| 15 | psar_trend | 1.97 |
| 16 | gamma | 1.96 |
| 17 | minus_di | 1.91 |
| 18 | mtf_consensus | 1.76 |
| 19 | vega | 1.69 |
| 20 | te | 1.46 |
| 21 | log_return | 1.43 |
| 22 | kappa_proxy | 1.33 |
| 23 | ret_z | 1.30 |
| 24 | strike | 1.27 |
| 25 | vol_energy | 1.08 |
| 26 | bb_percentile | 1.07 |
| 27 | friction_ratio | 1.03 |
| 28 | breakout_score | 1.01 |
| 29 | consolidation_score | 0.93 |
| 30 | fuzzy_reversion_11 | 0.89 |
| 31 | cmf | 0.81 |
| 32 | adx_adaptive | 0.69 |
| 33 | open | 0.65 |
| 34 | stoch_k_dyn | 0.57 |
| 35 | close | 0.56 |
| 36 | psar_adaptive | 0.56 |
| 37 | rsi_dyn | 0.49 |
| 38 | bb_mu_dyn | 0.49 |
| 39 | bb_upper_dyn | 0.44 |
| 40 | bb_lower_dyn | 0.36 |
| 41 | high | 0.22 |
| 42 | low | 0.18 |
| 43 | chaos_membership | 0.16 |
| 44 | position_size_mult | 0.16 |
| 45 | exec_allow | 0.02 |
| 46 | volume | 0.00 |
| 47 | iv | 0.00 |
| 48 | spread_ratio | 0.00 |
| 49 | target_spot | 0.00 |
| 50 | max_dd_60m | 0.00 |
| 51 | risk_override | 0.00 |
| 52 | iv_confidence | 0.00 |
| 53 | psar_reversion_mu | 0.00 |
| 54 | beta1_norm_stub | 0.00 |

**Surrogate Tree R2:** 0.6814 (High interpretability)

**Output Statistics (ROI Head):**
- Mean: -0.0591, Std: 0.0723
- Skewness: 0.597, Kurtosis: -1.213

### epoch_4_013026

**All Features (Permutation Importance):**
| Rank | Feature | Importance (%) |
|------|---------|----------------|
| 1 | ivr | 16.19 |
| 2 | theta | 10.31 |
| 3 | delta | 9.98 |
| 4 | pressure_down | 5.87 |
| 5 | gap_risk_score | 4.57 |
| 6 | pressure_up | 4.07 |
| 7 | macd_histogram | 4.03 |
| 8 | bw_expansion_rate | 3.95 |
| 9 | macd_norm | 3.29 |
| 10 | atr_pct | 2.85 |
| 11 | macd_signal_norm | 2.84 |
| 12 | plus_di | 2.05 |
| 13 | minus_di | 1.97 |
| 14 | gamma | 1.70 |
| 15 | vega | 1.64 |
| 16 | psar_trend | 1.59 |
| 17 | te | 1.46 |
| 18 | mtf_consensus | 1.46 |
| 19 | strike | 1.36 |
| 20 | bb_sigma_dyn | 1.32 |
| 21 | breakout_score | 1.26 |
| 22 | log_return | 1.22 |
| 23 | ret_z | 1.20 |
| 24 | vol_energy | 1.03 |
| 25 | kappa_proxy | 1.02 |
| 26 | bb_percentile | 0.98 |
| 27 | vol_ewma | 0.95 |
| 28 | fuzzy_reversion_11 | 0.94 |
| 29 | open | 0.91 |
| 30 | consolidation_score | 0.90 |
| 31 | adx_adaptive | 0.81 |
| 32 | friction_ratio | 0.80 |
| 33 | cmf | 0.77 |
| 34 | close | 0.72 |
| 35 | stoch_k_dyn | 0.64 |
| 36 | bb_mu_dyn | 0.61 |
| 37 | rsi_dyn | 0.55 |
| 38 | bb_upper_dyn | 0.53 |
| 39 | psar_adaptive | 0.52 |
| 40 | bb_lower_dyn | 0.42 |
| 41 | high | 0.31 |
| 42 | low | 0.21 |
| 43 | chaos_membership | 0.11 |
| 44 | position_size_mult | 0.09 |
| 45 | exec_allow | 0.01 |
| 46 | volume | 0.00 |
| 47 | iv | 0.00 |
| 48 | spread_ratio | 0.00 |
| 49 | target_spot | 0.00 |
| 50 | max_dd_60m | 0.00 |
| 51 | risk_override | 0.00 |
| 52 | iv_confidence | 0.00 |
| 53 | psar_reversion_mu | 0.00 |
| 54 | beta1_norm_stub | 0.00 |

**Surrogate Tree R2:** 0.7477 (High interpretability)

**Output Statistics (ROI Head):**
- Mean: -0.0603, Std: 0.0776
- Skewness: 0.740, Kurtosis: -0.871

### epoch_5_013026

**All Features (Permutation Importance):**
| Rank | Feature | Importance (%) |
|------|---------|----------------|
| 1 | ivr | 17.55 |
| 2 | theta | 11.73 |
| 3 | delta | 11.05 |
| 4 | pressure_down | 5.87 |
| 5 | pressure_up | 4.17 |
| 6 | gap_risk_score | 4.05 |
| 7 | macd_histogram | 3.97 |
| 8 | bw_expansion_rate | 3.89 |
| 9 | macd_norm | 3.14 |
| 10 | macd_signal_norm | 2.82 |
| 11 | atr_pct | 2.60 |
| 12 | minus_di | 2.01 |
| 13 | plus_di | 1.71 |
| 14 | gamma | 1.66 |
| 15 | psar_trend | 1.65 |
| 16 | vega | 1.59 |
| 17 | te | 1.33 |
| 18 | mtf_consensus | 1.28 |
| 19 | strike | 1.20 |
| 20 | ret_z | 1.19 |
| 21 | log_return | 1.14 |
| 22 | bb_sigma_dyn | 1.08 |
| 23 | vol_energy | 1.02 |
| 24 | breakout_score | 0.99 |
| 25 | kappa_proxy | 0.99 |
| 26 | bb_percentile | 0.88 |
| 27 | fuzzy_reversion_11 | 0.87 |
| 28 | friction_ratio | 0.84 |
| 29 | open | 0.77 |
| 30 | consolidation_score | 0.77 |
| 31 | cmf | 0.74 |
| 32 | adx_adaptive | 0.72 |
| 33 | vol_ewma | 0.67 |
| 34 | close | 0.66 |
| 35 | stoch_k_dyn | 0.58 |
| 36 | bb_mu_dyn | 0.52 |
| 37 | rsi_dyn | 0.51 |
| 38 | psar_adaptive | 0.51 |
| 39 | bb_upper_dyn | 0.42 |
| 40 | bb_lower_dyn | 0.30 |
| 41 | high | 0.24 |
| 42 | low | 0.18 |
| 43 | chaos_membership | 0.07 |
| 44 | position_size_mult | 0.04 |
| 45 | exec_allow | 0.01 |
| 46 | volume | 0.00 |
| 47 | iv | 0.00 |
| 48 | spread_ratio | 0.00 |
| 49 | target_spot | 0.00 |
| 50 | max_dd_60m | 0.00 |
| 51 | risk_override | 0.00 |
| 52 | iv_confidence | 0.00 |
| 53 | psar_reversion_mu | 0.00 |
| 54 | beta1_norm_stub | 0.00 |

**Surrogate Tree R2:** 0.8012 (High interpretability)

**Output Statistics (ROI Head):**
- Mean: -0.0591, Std: 0.0800
- Skewness: 0.767, Kurtosis: -0.759

## 4. Pairwise Divergence Analysis

### epoch_3_013026 vs epoch_4_013026

**Similarity Metrics:**
- Cosine Similarity: 0.9829
- Spearman Correlation: 0.9840 (p=1.25e-40)
- Kendall Tau: 0.9254

**Distance Metrics:**
- Jensen-Shannon Divergence: 0.0716
- Wasserstein Distance: 0.0028

**Top-K Feature Overlap:**
- Top-5 Overlap: 100%
- Top-10 Overlap: 100%

**Interpretation:**
- Near-identical importance direction (cosine=0.983): models learned same feature weighting
- Minimal information distance (JS=0.072): importance distributions nearly identical
- Rank order preserved (Spearman=0.984): feature priority ordering unchanged

**All Importance Shifts (sorted by magnitude):**
| Feature | Shift (%) |
|---------|-----------|
| ivr | -4.32 |
| delta | -1.51 |
| vol_ewma | +1.05 |
| bb_sigma_dyn | +0.88 |
| pressure_up | +0.82 |
| theta | -0.76 |
| gap_risk_score | +0.55 |
| bw_expansion_rate | +0.54 |
| macd_norm | +0.44 |
| pressure_down | +0.39 |

### epoch_3_013026 vs epoch_5_013026

**Similarity Metrics:**
- Cosine Similarity: 0.9729
- Spearman Correlation: 0.9772 (p=1.16e-36)
- Kendall Tau: 0.9211

**Distance Metrics:**
- Jensen-Shannon Divergence: 0.0955
- Wasserstein Distance: 0.0039

**Top-K Feature Overlap:**
- Top-5 Overlap: 80%
- Top-10 Overlap: 90%

**Interpretation:**
- Near-identical importance direction (cosine=0.973): models learned same feature weighting
- Minimal information distance (JS=0.096): importance distributions nearly identical
- Rank order preserved (Spearman=0.977): feature priority ordering unchanged

**All Importance Shifts (sorted by magnitude):**
| Feature | Shift (%) |
|---------|-----------|
| ivr | -5.67 |
| delta | -2.58 |
| theta | -2.18 |
| vol_ewma | +1.33 |
| bb_sigma_dyn | +1.12 |
| gap_risk_score | +1.06 |
| pressure_up | +0.72 |
| plus_di | +0.64 |
| bw_expansion_rate | +0.61 |
| macd_norm | +0.58 |

### epoch_4_013026 vs epoch_5_013026

**Similarity Metrics:**
- Cosine Similarity: 0.9977
- Spearman Correlation: 0.9965 (p=1.17e-57)
- Kendall Tau: 0.9699

**Distance Metrics:**
- Jensen-Shannon Divergence: 0.0368
- Wasserstein Distance: 0.0014

**Top-K Feature Overlap:**
- Top-5 Overlap: 80%
- Top-10 Overlap: 90%

**Interpretation:**
- Near-identical importance direction (cosine=0.998): models learned same feature weighting
- Minimal information distance (JS=0.037): importance distributions nearly identical
- Rank order preserved (Spearman=0.996): feature priority ordering unchanged

**All Importance Shifts (sorted by magnitude):**
| Feature | Shift (%) |
|---------|-----------|
| theta | -1.42 |
| ivr | -1.36 |
| delta | -1.07 |
| gap_risk_score | +0.51 |
| plus_di | +0.34 |
| vol_ewma | +0.28 |
| breakout_score | +0.27 |
| atr_pct | +0.24 |
| bb_sigma_dyn | +0.24 |
| mtf_consensus | +0.17 |

## 5. Stability Analysis (Physics-Inspired)

### epoch_3_013026 vs epoch_4_013026

| Head | Correlation | MSD | Var Ratio | KS Stat | Wasserstein |
|------|-------------|-----|-----------|---------|-------------|
| call_offset | 0.011 | 0.1749 | 0.677 | 0.111 | 0.0460 |
| put_offset | 0.012 | 0.1671 | 0.643 | 0.112 | 0.0469 |
| wing_width | 0.012 | 0.7912 | 0.825 | 0.105 | 0.0810 |
| dte | 0.012 | 36.0767 | 0.767 | 0.128 | 0.5683 |
| pop | 0.010 | 0.0004 | 6.894 | 0.262 | 0.0091 |
| roi | 0.012 | 0.0111 | 0.867 | 0.157 | 0.0077 |
| max_loss | 0.010 | 0.0322 | 1.288 | 0.107 | 0.0170 |
| confidence | 0.000 | 0.0046 | 0.796 | 0.194 | 0.0143 |
| entry | 0.002 | 0.4145 | 0.563 | 0.221 | 0.2102 |
| exit | 0.002 | 0.4546 | 1.332 | 0.096 | 0.0906 |

**Physics Interpretation:**
- Low coherence (avg correlation=0.008): models responding differently to same inputs
- Large energy gap (MSD=3.8127): significant output magnitude differences
- Minor distribution shift (KS=0.149): subtle statistical differences

### epoch_3_013026 vs epoch_5_013026

| Head | Correlation | MSD | Var Ratio | KS Stat | Wasserstein |
|------|-------------|-----|-----------|---------|-------------|
| call_offset | -0.036 | 0.1819 | 0.683 | 0.110 | 0.0498 |
| put_offset | -0.038 | 0.1759 | 0.637 | 0.148 | 0.0502 |
| wing_width | -0.038 | 0.8435 | 0.802 | 0.117 | 0.0957 |
| dte | -0.037 | 38.3062 | 0.750 | 0.121 | 0.6352 |
| pop | -0.034 | 0.0004 | 6.257 | 0.221 | 0.0079 |
| roi | -0.040 | 0.0121 | 0.816 | 0.158 | 0.0100 |
| max_loss | -0.041 | 0.0340 | 1.267 | 0.113 | 0.0166 |
| confidence | -0.035 | 0.0049 | 0.753 | 0.128 | 0.0148 |
| entry | 0.012 | 0.4367 | 0.519 | 0.237 | 0.2228 |
| exit | -0.044 | 0.4820 | 1.314 | 0.116 | 0.1113 |

**Physics Interpretation:**
- Low coherence (avg correlation=-0.033): models responding differently to same inputs
- Large energy gap (MSD=4.0478): significant output magnitude differences
- Minor distribution shift (KS=0.147): subtle statistical differences

### epoch_4_013026 vs epoch_5_013026

| Head | Correlation | MSD | Var Ratio | KS Stat | Wasserstein |
|------|-------------|-----|-----------|---------|-------------|
| call_offset | -0.023 | 0.2142 | 1.009 | 0.034 | 0.0090 |
| put_offset | -0.021 | 0.2102 | 0.991 | 0.203 | 0.0147 |
| wing_width | -0.022 | 0.9089 | 0.973 | 0.031 | 0.0203 |
| dte | -0.023 | 42.7409 | 0.979 | 0.130 | 0.1382 |
| pop | 0.009 | 0.0001 | 0.886 | 0.117 | 0.0016 |
| roi | -0.018 | 0.0126 | 0.941 | 0.047 | 0.0027 |
| max_loss | -0.020 | 0.0290 | 0.984 | 0.037 | 0.0027 |
| confidence | 0.011 | 0.0051 | 0.946 | 0.129 | 0.0021 |
| entry | -0.002 | 0.4961 | 0.921 | 0.031 | 0.0240 |
| exit | 0.011 | 0.3829 | 0.986 | 0.037 | 0.0244 |

**Physics Interpretation:**
- Low coherence (avg correlation=-0.010): models responding differently to same inputs
- Large energy gap (MSD=4.5000): significant output magnitude differences
- Stable amplitude (variance ratio=0.96): similar output volatility
- Minor distribution shift (KS=0.080): subtle statistical differences

## 6. Gradient Saliency Analysis

### epoch_3_013026
**All Features (Gradient Saliency):**
| Rank | Feature | Saliency |
|------|---------|----------|
| 1 | ivr | 0.2885 |
| 2 | gap_risk_score | 0.2620 |
| 3 | te | 0.2299 |
| 4 | atr_pct | 0.2008 |
| 5 | theta | 0.1735 |
| 6 | delta | 0.1734 |
| 7 | vol_ewma | 0.1687 |
| 8 | psar_adaptive | 0.1656 |
| 9 | high | 0.1603 |
| 10 | breakout_score | 0.1572 |
| 11 | exec_allow | 0.1326 |
| 12 | pressure_up | 0.1305 |
| 13 | low | 0.1292 |
| 14 | psar_trend | 0.1251 |
| 15 | open | 0.1159 |
| 16 | mtf_consensus | 0.1057 |
| 17 | strike | 0.1034 |
| 18 | bb_mu_dyn | 0.1031 |
| 19 | consolidation_score | 0.1022 |
| 20 | stoch_k_dyn | 0.1020 |
| 21 | cmf | 0.1019 |
| 22 | bw_expansion_rate | 0.1006 |
| 23 | plus_di | 0.0972 |
| 24 | rsi_dyn | 0.0968 |
| 25 | pressure_down | 0.0911 |
| 26 | adx_adaptive | 0.0906 |
| 27 | bb_lower_dyn | 0.0905 |
| 28 | ret_z | 0.0895 |
| 29 | bb_sigma_dyn | 0.0889 |
| 30 | close | 0.0873 |
| 31 | position_size_mult | 0.0871 |
| 32 | bb_upper_dyn | 0.0814 |
| 33 | kappa_proxy | 0.0799 |
| 34 | fuzzy_reversion_11 | 0.0793 |
| 35 | iv_confidence | 0.0779 |
| 36 | spread_ratio | 0.0761 |
| 37 | log_return | 0.0754 |
| 38 | macd_norm | 0.0750 |
| 39 | macd_histogram | 0.0748 |
| 40 | friction_ratio | 0.0736 |
| 41 | gamma | 0.0730 |
| 42 | psar_reversion_mu | 0.0714 |
| 43 | iv | 0.0698 |
| 44 | target_spot | 0.0674 |
| 45 | bb_percentile | 0.0671 |
| 46 | chaos_membership | 0.0667 |
| 47 | max_dd_60m | 0.0660 |
| 48 | minus_di | 0.0660 |
| 49 | beta1_norm_stub | 0.0647 |
| 50 | risk_override | 0.0644 |
| 51 | vega | 0.0625 |
| 52 | volume | 0.0600 |
| 53 | macd_signal_norm | 0.0531 |
| 54 | vol_energy | 0.0504 |

**Temporal Focus:** Recent 20% weight=0.0095, Distant 20% weight=0.4441

### epoch_4_013026
**All Features (Gradient Saliency):**
| Rank | Feature | Saliency |
|------|---------|----------|
| 1 | gap_risk_score | 0.1502 |
| 2 | te | 0.1403 |
| 3 | ivr | 0.1313 |
| 4 | vol_ewma | 0.1122 |
| 5 | theta | 0.1030 |
| 6 | delta | 0.1010 |
| 7 | breakout_score | 0.0948 |
| 8 | atr_pct | 0.0937 |
| 9 | strike | 0.0802 |
| 10 | pressure_up | 0.0753 |
| 11 | exec_allow | 0.0728 |
| 12 | high | 0.0712 |
| 13 | low | 0.0660 |
| 14 | consolidation_score | 0.0646 |
| 15 | pressure_down | 0.0634 |
| 16 | bb_upper_dyn | 0.0631 |
| 17 | bb_mu_dyn | 0.0629 |
| 18 | psar_adaptive | 0.0605 |
| 19 | chaos_membership | 0.0553 |
| 20 | stoch_k_dyn | 0.0553 |
| 21 | position_size_mult | 0.0547 |
| 22 | rsi_dyn | 0.0533 |
| 23 | gamma | 0.0518 |
| 24 | psar_trend | 0.0512 |
| 25 | mtf_consensus | 0.0510 |
| 26 | cmf | 0.0508 |
| 27 | vega | 0.0506 |
| 28 | open | 0.0484 |
| 29 | iv_confidence | 0.0475 |
| 30 | bb_sigma_dyn | 0.0474 |
| 31 | risk_override | 0.0468 |
| 32 | iv | 0.0453 |
| 33 | bw_expansion_rate | 0.0449 |
| 34 | adx_adaptive | 0.0448 |
| 35 | target_spot | 0.0444 |
| 36 | psar_reversion_mu | 0.0441 |
| 37 | max_dd_60m | 0.0439 |
| 38 | spread_ratio | 0.0438 |
| 39 | beta1_norm_stub | 0.0438 |
| 40 | fuzzy_reversion_11 | 0.0434 |
| 41 | bb_lower_dyn | 0.0433 |
| 42 | close | 0.0425 |
| 43 | ret_z | 0.0424 |
| 44 | volume | 0.0421 |
| 45 | plus_di | 0.0403 |
| 46 | macd_histogram | 0.0393 |
| 47 | minus_di | 0.0385 |
| 48 | macd_norm | 0.0381 |
| 49 | friction_ratio | 0.0371 |
| 50 | bb_percentile | 0.0366 |
| 51 | kappa_proxy | 0.0355 |
| 52 | vol_energy | 0.0340 |
| 53 | macd_signal_norm | 0.0336 |
| 54 | log_return | 0.0306 |

**Temporal Focus:** Recent 20% weight=0.0071, Distant 20% weight=0.2203

### epoch_5_013026
**All Features (Gradient Saliency):**
| Rank | Feature | Saliency |
|------|---------|----------|
| 1 | gap_risk_score | 0.3113 |
| 2 | ivr | 0.3084 |
| 3 | te | 0.2822 |
| 4 | vol_ewma | 0.2467 |
| 5 | atr_pct | 0.2350 |
| 6 | theta | 0.2272 |
| 7 | delta | 0.2077 |
| 8 | high | 0.1985 |
| 9 | exec_allow | 0.1871 |
| 10 | breakout_score | 0.1830 |
| 11 | pressure_up | 0.1634 |
| 12 | low | 0.1537 |
| 13 | strike | 0.1447 |
| 14 | consolidation_score | 0.1427 |
| 15 | psar_adaptive | 0.1415 |
| 16 | mtf_consensus | 0.1233 |
| 17 | stoch_k_dyn | 0.1198 |
| 18 | bb_lower_dyn | 0.1195 |
| 19 | open | 0.1194 |
| 20 | pressure_down | 0.1171 |
| 21 | bb_mu_dyn | 0.1165 |
| 22 | bb_sigma_dyn | 0.1162 |
| 23 | bb_upper_dyn | 0.1144 |
| 24 | close | 0.1142 |
| 25 | rsi_dyn | 0.1134 |
| 26 | position_size_mult | 0.1119 |
| 27 | cmf | 0.1056 |
| 28 | max_dd_60m | 0.1026 |
| 29 | bw_expansion_rate | 0.1016 |
| 30 | adx_adaptive | 0.0966 |
| 31 | spread_ratio | 0.0964 |
| 32 | target_spot | 0.0959 |
| 33 | gamma | 0.0957 |
| 34 | plus_di | 0.0940 |
| 35 | chaos_membership | 0.0926 |
| 36 | vega | 0.0923 |
| 37 | iv | 0.0911 |
| 38 | psar_reversion_mu | 0.0899 |
| 39 | iv_confidence | 0.0898 |
| 40 | fuzzy_reversion_11 | 0.0877 |
| 41 | macd_histogram | 0.0875 |
| 42 | beta1_norm_stub | 0.0866 |
| 43 | ret_z | 0.0827 |
| 44 | risk_override | 0.0814 |
| 45 | macd_signal_norm | 0.0813 |
| 46 | psar_trend | 0.0809 |
| 47 | macd_norm | 0.0798 |
| 48 | minus_di | 0.0774 |
| 49 | volume | 0.0773 |
| 50 | friction_ratio | 0.0725 |
| 51 | bb_percentile | 0.0693 |
| 52 | log_return | 0.0689 |
| 53 | kappa_proxy | 0.0668 |
| 54 | vol_energy | 0.0644 |

**Temporal Focus:** Recent 20% weight=0.0077, Distant 20% weight=0.5420

### Gradient Alignment: epoch_3_013026 vs epoch_4_013026
- High feature attention alignment (0.985): models focus on same features
- Temporal attention aligned (0.997): same recency bias

### Gradient Alignment: epoch_3_013026 vs epoch_5_013026
- High feature attention alignment (0.991): models focus on same features
- Temporal attention aligned (0.984): same recency bias

### Gradient Alignment: epoch_4_013026 vs epoch_5_013026
- High feature attention alignment (0.994): models focus on same features
- Temporal attention aligned (0.986): same recency bias

## 7. Hessian & Fisher Analysis

### Hessian Eigenspectrum

### Fisher Information
**epoch_3_013026 vs epoch_4_013026:**
- Uneven sensitivity distribution (spread=20.1x): parameter importance concentrated

**epoch_3_013026 vs epoch_5_013026:**
- Reduced sensitivity in expert_normal (ratio=0.14x): layer may be undertrained
- Uneven sensitivity distribution (spread=34.1x): parameter importance concentrated

**epoch_4_013026 vs epoch_5_013026:**
- Relatively uniform sensitivity (spread=1.8x)

## 8. Surrogate Decision Tree Rules

### epoch_3_013026
**R2 Score:** 0.6814
```
|--- ivr <= 0.674
|   |--- bb_sigma_dyn <= 0.832
|   |   |--- ivr <= 0.304
|   |   |   |--- delta <= 0.695
|   |   |   |   |--- value: [-0.112]
|   |   |   |--- delta >  0.695
|   |   |   |   |--- value: [-0.093]
|   |   |--- ivr >  0.304
|   |   |   |--- friction_ratio <= -0.550
|   |   |   |   |--- value: [-0.035]
|   |   |   |--- friction_ratio >  -0.550
|   |   |   |   |--- value: [-0.080]
|   |--- bb_sigma_dyn >  0.832
|   |   |--- ivr <= -0.058
|   |   |   |--- bb_sigma_dyn <= 2.089
|   |   |   |   |--- value: [-0.096]
|   |   |   |--- bb_sigma_dyn >  2.089
|   |   |   |   |--- value: [-0.049]
|   |   |--- ivr >  -0.058
|   |   |   |--- vol_ewma <= 1.062
|   |   |   |   |--- value: [-0.043]
|   |   |   |--- vol_ewma >  1.062
|   |   |   |   |--- value: [0.006]
|--- ivr >  0.674
|   |--- ivr <= 1.513
|   |   |--- bb_sigma_dyn <= 1.139
|   |   |   |--- atr_pct <= 0.945
|   |   |   |   |--- value: [-0.034]
|   |   |   |--- atr_pct >  0.945
|   |   |   |   |--- value: [0.001]
|   |   |--- bb_sigma_dyn >  1.139
|   |   |   |--- bb_percentile <= 0.502
|   |   |   |   |--- value: [0.035]
|   |   |   |--- bb_percentile >  0.502
|   |   |   |   |--- value: [0.006]
|   |--- ivr >  1.513
|   |   |--- ivr <= 2.332
|   |   |   |--- bb_sigma_dyn <= -0.373
|   |   |   |   |--- value: [-0.002]
|   |   |   |--- bb_sigma_dyn >  -0.373
|   |   |   |   |--- value: [0.035]
|   |   |--- ivr >  2.332
|   |   |   |--- bb_sigma_dyn <= 2.203
|   |   |   |   |--- value: [0.047]
|   |   |   |--- bb_sigma_dyn >  2.203
|   |   |   |   |--- value: [0.067]

```

### epoch_4_013026
**R2 Score:** 0.7477
```
|--- ivr <= 0.687
|   |--- ivr <= 0.152
|   |   |--- delta <= 0.654
|   |   |   |--- log_return <= -1.541
|   |   |   |   |--- value: [-0.087]
|   |   |   |--- log_return >  -1.541
|   |   |   |   |--- value: [-0.119]
|   |   |--- delta >  0.654
|   |   |   |--- delta <= 0.879
|   |   |   |   |--- value: [-0.083]
|   |   |   |--- delta >  0.879
|   |   |   |   |--- value: [-0.106]
|   |--- ivr >  0.152
|   |   |--- atr_pct <= 0.338
|   |   |   |--- theta <= 0.009
|   |   |   |   |--- value: [-0.062]
|   |   |   |--- theta >  0.009
|   |   |   |   |--- value: [-0.094]
|   |   |--- atr_pct >  0.338
|   |   |   |--- ivr <= 0.332
|   |   |   |   |--- value: [-0.062]
|   |   |   |--- ivr >  0.332
|   |   |   |   |--- value: [-0.036]
|--- ivr >  0.687
|   |--- ivr <= 1.626
|   |   |--- ivr <= 1.166
|   |   |   |--- atr_pct <= 0.949
|   |   |   |   |--- value: [-0.024]
|   |   |   |--- atr_pct >  0.949
|   |   |   |   |--- value: [0.001]
|   |   |--- ivr >  1.166
|   |   |   |--- atr_pct <= 1.337
|   |   |   |   |--- value: [0.004]
|   |   |   |--- atr_pct >  1.337
|   |   |   |   |--- value: [0.036]
|   |--- ivr >  1.626
|   |   |--- bb_sigma_dyn <= 1.909
|   |   |   |--- vol_ewma <= -0.319
|   |   |   |   |--- value: [0.020]
|   |   |   |--- vol_ewma >  -0.319
|   |   |   |   |--- value: [0.054]
|   |   |--- bb_sigma_dyn >  1.909
|   |   |   |--- ivr <= 1.906
|   |   |   |   |--- value: [0.063]
|   |   |   |--- ivr >  1.906
|   |   |   |   |--- value: [0.084]

```

### epoch_5_013026
**R2 Score:** 0.8012
```
|--- ivr <= 0.496
|   |--- ivr <= 0.157
|   |   |--- delta <= 0.650
|   |   |   |--- ivr <= -0.012
|   |   |   |   |--- value: [-0.120]
|   |   |   |--- ivr >  -0.012
|   |   |   |   |--- value: [-0.097]
|   |   |--- delta >  0.650
|   |   |   |--- delta <= 0.952
|   |   |   |   |--- value: [-0.092]
|   |   |   |--- delta >  0.952
|   |   |   |   |--- value: [-0.113]
|   |--- ivr >  0.157
|   |   |--- atr_pct <= 0.957
|   |   |   |--- ivr <= 0.277
|   |   |   |   |--- value: [-0.083]
|   |   |   |--- ivr >  0.277
|   |   |   |   |--- value: [-0.053]
|   |   |--- atr_pct >  0.957
|   |   |   |--- value: [-0.031]
|--- ivr >  0.496
|   |--- ivr <= 1.522
|   |   |--- ivr <= 1.270
|   |   |   |--- atr_pct <= 1.346
|   |   |   |   |--- value: [-0.015]
|   |   |   |--- atr_pct >  1.346
|   |   |   |   |--- value: [0.005]
|   |   |--- ivr >  1.270
|   |   |   |--- bb_sigma_dyn <= 1.586
|   |   |   |   |--- value: [0.008]
|   |   |   |--- bb_sigma_dyn >  1.586
|   |   |   |   |--- value: [0.045]
|   |--- ivr >  1.522
|   |   |--- ivr <= 1.732
|   |   |   |--- vol_ewma <= 0.260
|   |   |   |   |--- value: [0.021]
|   |   |   |--- vol_ewma >  0.260
|   |   |   |   |--- value: [0.059]
|   |   |--- ivr >  1.732
|   |   |   |--- atr_pct <= 2.934
|   |   |   |   |--- value: [0.078]
|   |   |   |--- atr_pct >  2.934
|   |   |   |   |--- value: [0.103]

```

## 9. Technical Assessment: Pros & Cons

### epoch_3_013026

**Strengths:**
- High surrogate fidelity (R2=0.681 >= 0.50): decision tree rules are reliable summary
- Signal features consistent with peer checkpoints (avg top-10 overlap=0.95)
- Attribution not overly concentrated (top feature=11.9%, entropy=3.30)
- Strong attribution consistency (SHAP vs Permutation top-5 overlap=80%)

**Concerns:**
- Possible collapse: near-constant outputs detected (variance near zero)
- Output drift: call_offset correlation=0.01 (below 0.70 vs peer)
- Output drift: call_offset correlation=-0.04 (below 0.70 vs peer)

### epoch_4_013026

**Strengths:**
- High surrogate fidelity (R2=0.748 >= 0.50): decision tree rules are reliable summary
- Signal features consistent with peer checkpoints (avg top-10 overlap=0.95)
- Attribution not overly concentrated (top feature=16.2%, entropy=3.21)
- Strong attribution consistency (SHAP vs Permutation top-5 overlap=100%)

**Concerns:**
- Possible collapse: near-constant outputs detected (variance near zero)
- Output drift: call_offset correlation=0.01 (below 0.70 vs peer)
- Output drift: call_offset correlation=-0.02 (below 0.70 vs peer)

### epoch_5_013026

**Strengths:**
- High surrogate fidelity (R2=0.801 >= 0.50): decision tree rules are reliable summary
- Signal features consistent with peer checkpoints (avg top-10 overlap=0.90)
- Attribution not overly concentrated (top feature=17.6%, entropy=3.12)
- Strong attribution consistency (SHAP vs Permutation top-5 overlap=80%)

**Concerns:**
- Possible collapse: near-constant outputs detected (variance near zero)
- Output drift: call_offset correlation=-0.04 (below 0.70 vs peer)
- Output drift: call_offset correlation=-0.02 (below 0.70 vs peer)

## 10. Final Recommendation & Ranking

### Overall Ranking
| Rank | Model | Score | Breakdown |
|------|-------|-------|-----------|
| 1 | epoch_3_013026 | 12 | Pros:8, Cons:-3, Interpretability:3, Feature Balance:2, Consistency:2 |
| 2 | epoch_4_013026 | 12 | Pros:8, Cons:-3, Interpretability:3, Feature Balance:2, Consistency:2 |
| 3 | epoch_5_013026 | 12 | Pros:8, Cons:-3, Interpretability:3, Feature Balance:2, Consistency:2 |

### Recommendation
**Recommended Model:** `epoch_3_013026`

**Rationale (All Strengths):**
- High surrogate fidelity (R2=0.681 >= 0.50): decision tree rules are reliable summary
- Signal features consistent with peer checkpoints (avg top-10 overlap=0.95)
- Attribution not overly concentrated (top feature=11.9%, entropy=3.30)
- Strong attribution consistency (SHAP vs Permutation top-5 overlap=80%)

**Runner-up:** `epoch_4_013026`

---

## References

1. **Neural CDE Theory:** Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series* (2020)
2. **System Architecture:** `docs/scientific_spec.md` - Complete mathematical specification
3. **Feature Schema:** V2.2 with 54 input dimensions (OHLCV, Greeks, dynamic indicators)
4. **Loss Function:** Composite Risk-Aligned Loss with Huber, Sharpe proxy, and drawdown penalties

**Composite Loss Function (from training):**

$$
\mathcal{L}_{\text{composite}} = \lambda_1 \mathcal{L}_{\text{pred}} - \lambda_2 \mathcal{L}_{\text{sharpe}} + \lambda_3 \mathcal{L}_{\text{dd}} + \lambda_4 \mathcal{L}_{\text{turn}}
$$

Where:
- $\mathcal{L}_{\text{pred}}$: Huber loss for strike predictions
- $\mathcal{L}_{\text{sharpe}}$: Negative Sharpe ratio (maximized via gradient descent)
- $\mathcal{L}_{\text{dd}}$: Soft drawdown penalty
- $\mathcal{L}_{\text{turn}}$: Turnover penalty for position stability

---
*Report generated by audit_cde_comparison.py | Mathematical foundations from docs/scientific_spec.md*