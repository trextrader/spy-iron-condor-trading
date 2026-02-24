# Technical Architecture Summary: CondorIntelligence Flow (v4.3)

The CondorIntelligence system is a multi-layer neural-symbolic pipeline designed for multi-strategy SPY option trading. It integrates the novel **CondorNet™ v4.3** unified architecture with a Neural-Fuzzy decision suite to transform raw market physics into risk-weighted trade intent across **10 strategy types**.

> **Architecture Evolution:** CondorNet™ v4.3 extends the v4.0 ETD-1/CDE/TFT core with: **4× multi-timeframe input fusion**, **options chain transformer encoding**, **pivot prediction**, **10 strategy types**, and **risk metric heads** (PoP/EV/MaxLoss/VaR₉₅/CVaR₉₅).

## 0. CondorNet™ v4.3 Architecture (10.9M Parameters)

```
CondorNetV43 [B,T,64] × 4 TFs + [B,N,10] chain + [B,T,13] pivots
├── MultiTFProjector       4 × [B,T,64] → [B,T,256]  (per-TF projection + concat)
├── PivotProjector         [B,T,13] → [B,T,16]       (sparse NaN-masked)
├── TFFusionBlock          [B,T,256] + [B,T,16] → [B,T,256]  (residual fusion)
├── OptionsChainEncoder    [B,N,10] → [B,128]         (Transformer 2L/4H + skew)
├── JointFusionLayer       [B,T,256] + [B,128] → [B,T,384]
├── CondorNet v42 Core     ETD-1 + TFT + CDE + 8 predicate gates
├── StrategyHead           [B,384] → 10 logits + [B,4,5] leg params + entry
├── RiskMetricHead         [B,384] → PoP, EV, MaxLoss, VaR₉₅, CVaR₉₅
├── PivotPredictionHead    [B,384] → P(pivot at [5,10,20,35,70] bars)
├── PositionSizeHead       PoP-blended fuzzy sizing [B,1]
└── SpotPredHead           [B,1] target spot regression
```

### ETD-1 Core State Evolution

$$x_k = e^{A_\theta(u_k)\Delta t_k} x_{k-1} + \Delta t_k \varphi_1(A_\theta(u_k)\Delta t_k) B_\theta(u_k) + G_\theta(x_{k-1}, u_k) \Delta X_k + D(\text{Greeks}_k, r_{k-1}, q_k)$$

Where $\varphi_1(M) = M^{-1}(e^M - I)$ is the ETD-1 basis function.

## 1. Data Inception (v4.3 — Multi-Source Input)

The pipeline ingests **4 timeframes × 64 features** + **options chain** + **pivot events**:

### Per-Timeframe Features (64 each × 4 TFs: M1, M5, M15, H1)
- **Price / Trend (12):** OHLCV, MA, FRAMA, SMA, TurtleChannel, TrendSeeker, AnchoredVWAP
- **Momentum / Oscillators (10):** RSI_dyn, Stoch_K_dyn, ADX_adaptive, log_return, AccDist, breakout_score, WeightedAlpha
- **Volatility (8):** ATR, BB bands (upper/lower/mu/sigma), bandwidth, expansion_rate
- **Volume / Microstructure (6):** volume, trade_count, OSC-Volume, vol_energy, vol_ewma, spread_ratio
- **Regime / Probability (10):** PSAR (adaptive/mark/trend/reversion_mu), pressure up/down, max_dd, bb_percentile, fuzzy_reversion_11, consolidation_score
- **Risk / Composite (5+):** gap_risk_score, chaos_membership, kappa_proxy, McClellanOsc, Slope/SlopeATR

### Options Chain Grid ([B, N, 10])
- 10 features per contract: moneyness, days_to_exp, iv, delta, gamma, theta, vega, bid, ask, oi_norm
- Up to 120 contracts (60 calls + 60 puts by |moneyness|)

### Sparse Pivot Features ([B, T, 13])
- PivotHigh, PivotLow, PivotResidual/ATR/Z, PivotCurvature, SegmentLength, SegmentVol, Slope/SlopeATR
- NaN = no pivot event (never imputed); medium + strong strength only

## 2. Tactical Preprocessing
Data is passed through an **O(1) GPU-Resident Data Loader** using `unfold` views. Features are normalized using robust **Med/MAD Z-Score** and clipped via **Tanh** for stability. Ternary, bounded, and sparse features bypass normalization.

## 3. CondorNet™ v4.3 Neural Backbone

| Component | Role | Mathematical Term |
|-----------|------|-------------------|
| **MultiTFProjector** | 4× TF projection | $z = [\text{proj}_{M1}; \text{proj}_{M5}; \text{proj}_{M15}; \text{proj}_{H1}]$ |
| **PivotProjector** | Sparse pivot embedding | $p_{\text{embed}} = \text{Proj}(p \odot \neg\text{mask})$ |
| **TFFusionBlock** | TF + Pivot fusion | $z_f = \text{LN}(\text{Lin}(z \| p_e) + z)$ |
| **OptionsChainEncoder** | Chain surface encoding | $c = \text{TransformerEnc}(\text{ChainGrid})$ |
| **JointFusion** | TF + Chain combination | $j = \text{LN}(z_f \| \text{broadcast}(c))$ |
| **ETD-1 Core** | Stable matrix exponential | $e^{A\Delta t}x + \Delta t \varphi_1 B$ |
| **Neural CDE** | Path-dependent response | $G_\theta(x, u) \cdot \Delta X_k$ |
| **8 Predicate Gates** | Regime-aware gating | 8 differentiable inequality predicates |

The **4-block augmented state** $x_k = [h_k; v_k; m_k; r_k]$ captures:
- **h_k**: Latent risk manifold (256 dims)
- **v_k**: Portfolio state - Greeks, P&L (32 dims)
- **m_k**: Risk memory - drawdown, VaR (64 dims)
- **r_k**: Regime combinatorics from predicates (32 dims)

## 4. Multi-Strategy Intelligence (v4.3 Output Heads)

| Head | Output | Purpose |
|------|--------|---------|
| **StrategyHead** | 10 strategy logits + [B,4,5] leg params + entry | Multi-strategy selection with leg parameterization |
| **RiskMetricHead** | PoP, EV, MaxLoss, VaR₉₅, CVaR₉₅ | Per-strategy risk quantification |
| **PivotPredictionHead** | P(pivot at [5,10,20,35,70]) | Anticipatory reversal detection |
| **PositionSizeHead** | [B,1] fuzzy-blended size | PoP-weighted position sizing |
| **SpotPredHead** | [B,1] target spot | Auxiliary regression target |

## 5. Multi-Objective Learning (Composite CondorNet Loss)
The model is optimized via **10-component composite loss**: NPDD, Sharpe, Drawdown, Turnover, Fuzzy Consistency, Pattern Entropy, Group Invariance, Correlation, Energy, and Growth.

## 6. Neural-Fuzzy Decision Suite
The final layer bridges neural forecasts and capital preservation:
- **Fuzzy Engine:** 11-factor inference system that validates CondorNet predictions against deterministic rules.
- **Money Management:** Dynamic sizing based on Predictive Alignment of policy heads, predicate gate activations, and fuzzy membership scores.

---

## Repository Sync Addendum (2026-02-24)

Authoritative engineering specs:
- `intelligence/condor_brain_net_v43.py` (primary architecture)
- `intelligence/schema_v43.py` (feature schema and strategy types)
- `docs/CONDORNET_IMPLEMENTATION_PLAN.md` (implementation spec)

If this document conflicts with the code, the code governs implementation.

---

*Document Version: 4.3 (CondorNet™) | Last Updated: 2026-02-24*
