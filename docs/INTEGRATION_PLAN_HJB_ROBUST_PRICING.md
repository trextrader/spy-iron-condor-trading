# Comprehensive Integration Plan: HJB Framework & Robust Pricing-Hedging Duality

**Version:** 1.0
**Date:** January 23, 2026
**Author:** Integration Engineering Team
**Status:** PLANNING PHASE

---

## Executive Summary

This document provides a detailed, incremental integration plan for incorporating two advanced mathematical frameworks into the existing CondorBrain-CondorIntelligence DeepMamba2 neural architecture:

1. **Robust Pricing-Hedging Duality** (Hou-Obłój Framework)
2. **Passerini-Vázquez HJB Framework** (Optimal Trading with Alpha Predictors)

The plan is designed for **incremental testing** with **easy rollback** at each stage.

---

## Table of Contents

1. [Current Architecture Overview](#1-current-architecture-overview)
2. [Integration Phase 1: Prediction Sets & Martingale Calibration](#2-phase-1-prediction-sets--martingale-calibration)
3. [Integration Phase 2: Enhanced Loss Function](#3-phase-2-enhanced-loss-function)
4. [Integration Phase 3: HJB Solver & No-Trade Zones](#4-phase-3-hjb-solver--no-trade-zones)
5. [Integration Phase 4: Alpha Predictors (OU Process)](#5-phase-4-alpha-predictors-ou-process)
6. [Integration Phase 5: Execution Cost & Fill Probability](#6-phase-5-execution-cost--fill-probability)
7. [Integration Phase 6: Fuzzy Logic as Measure](#7-phase-6-fuzzy-logic-as-measure)
8. [Integration Phase 7: Computational Physics Features](#8-phase-7-computational-physics-features)
9. [Integration Phase 8: Governance & Risk Override Layer](#9-phase-8-governance--risk-override-layer)
10. [Testing & Validation Strategy](#10-testing--validation-strategy)
11. [Rollback Procedures](#11-rollback-procedures)
12. [File Manifest](#12-file-manifest)

---

## 1. Current Architecture Overview

### 1.1 Existing Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT CONDORBRAIN PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA INGESTION LAYER                                                       │
│  ├─ data_factory/sync_engine.py (MTF 1m/5m/15m + Indicators)               │
│  ├─ data_factory/pipeline/SyntheticOptionsEngine.py (Black-Scholes)        │
│  └─ intelligence/canonical_feature_registry.py (32 Features V2.1)          │
│                                                                             │
│  FEATURE ENGINEERING LAYER                                                  │
│  ├─ intelligence/features/dynamic_features.py (V2.1 Dynamic)               │
│  └─ intelligence/primitives/*.py (14 Institutional Rules)                  │
│                                                                             │
│  MODEL TRAINING LAYER                                                       │
│  ├─ intelligence/train_condor_brain.py (GPU Training Loop)                 │
│  ├─ intelligence/condor_brain.py (CondorBrain Architecture)                │
│  │   ├─ 32-layer Mamba-2 SSM Backbone                                      │
│  │   ├─ VolGatedAttn (layers 8, 16, 24)                                    │
│  │   ├─ TopKMoE (3 regime experts)                                         │
│  │   ├─ HorizonForecaster (45-day trajectory)                              │
│  │   └─ ConditionalDiffusionHead (optional)                                │
│  └─ intelligence/condor_loss.py (5-Component Composite Loss)               │
│                                                                             │
│  INFERENCE & EXECUTION LAYER                                                │
│  ├─ qtmf/facade.py (benchmark_and_size - 10-Factor Fuzzy)                  │
│  ├─ intelligence/fuzzy_engine.py (Membership Functions)                    │
│  └─ intelligence/rule_engine/executor.py (14 Rules + Gates)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Integration Touchpoints

| Component | File | Current Function | Integration Target |
|-----------|------|------------------|-------------------|
| Loss Function | `condor_loss.py` | 5-component composite | Add robust pricing terms |
| Model Output | `condor_brain.py` | 8 IC parameters | Add prediction set membership |
| Sizing Logic | `qtmf/facade.py` | 10-factor fuzzy | Add martingale-calibrated bounds |
| Rule Engine | `rule_engine/executor.py` | 14 rules | Add HJB no-trade zone gates |
| Feature Registry | `canonical_feature_registry.py` | 32 features | Add α-predictor features |

### 1.3 Mathematical Invariants to Preserve

**CRITICAL - DO NOT MODIFY:**
1. `CondorExpertHead` output activations (indices 0-7 constrained)
2. `CompositeCondorLoss` clamping of rule_signals to [0, 1]
3. `benchmark_and_size` minimum floor assertion (≥2 contracts)
4. `MTFSyncEngine` indicator calculation order
5. Robust Z-score normalization in training pipeline

---

## 2. Phase 1: Prediction Sets & Martingale Calibration

### 2.1 Objective

Implement the core mathematical objects from Hou-Obłój:
- **Prediction Set I ⊂ Ω**: Encodes beliefs about feasible price paths
- **Martingale Measures M^I**: Probability measures calibrated to observed prices

### 2.2 New Files to Create

```
intelligence/
├─ robust/
│   ├─ __init__.py
│   ├─ prediction_set.py      # Prediction set construction
│   ├─ martingale_measure.py  # Martingale calibration
│   └─ path_filter.py         # Path filtering utilities
```

### 2.3 Mathematical Specification

#### 2.3.1 Prediction Set Construction

The prediction set I encodes beliefs about market paths:

```python
# intelligence/robust/prediction_set.py

class PredictionSet:
    """
    Encodes the set I ⊂ Ω of paths deemed possible.

    Mathematical Definition:
        I = {ω ∈ Ω : constraint_1(ω) ∧ constraint_2(ω) ∧ ... ∧ constraint_n(ω)}

    For SPY Iron Condor trading, constraints include:
        1. IVR regime bounds (e.g., IVR ∈ [20, 80])
        2. Liquidity filters (spread_ratio < threshold)
        3. Gap risk exclusions (overnight gap < max_gap)
        4. Chaos membership threshold (β₁-gated curvature < chaos_max)
    """

    def __init__(self, config: PredictionSetConfig):
        self.ivr_bounds = config.ivr_bounds          # (min, max)
        self.spread_threshold = config.spread_max    # Max spread ratio
        self.gap_threshold = config.gap_max          # Max overnight gap %
        self.chaos_threshold = config.chaos_max      # Max chaos membership

    def membership(self, state: MarketState) -> float:
        """
        Compute fuzzy membership μ_I(state) ∈ [0, 1].

        Returns 1.0 if state is fully within I, 0.0 if excluded,
        or a value in (0, 1) for boundary cases.

        Mathematical Formula:
            μ_I(x) = min(μ_ivr(x), μ_spread(x), μ_gap(x), μ_chaos(x))
        """
        mu_ivr = self._ivr_membership(state.ivr)
        mu_spread = self._spread_membership(state.spread_ratio)
        mu_gap = self._gap_membership(state.gap_risk_score)
        mu_chaos = self._chaos_membership(state.chaos_membership)

        return min(mu_ivr, mu_spread, mu_gap, mu_chaos)
```

#### 2.3.2 Martingale Measure Calibration

```python
# intelligence/robust/martingale_measure.py

class MartingaleMeasureSet:
    """
    The set M^I of calibrated martingale measures supported on I.

    Mathematical Definition:
        M^I = {P ∈ M : P(I) = 1, E_P[X_j] = π_j for all j ∈ X}

    Where:
        - M is the set of all martingale measures
        - X is the set of calibration instruments (options)
        - π_j is the observed price of instrument j

    For CondorBrain, calibration instruments include:
        - ATM call/put prices
        - 25-delta call/put prices (wings)
        - Variance swap rate (VIX)
    """

    def calibrate(self, options_chain: pd.DataFrame, spot: float) -> CalibrationResult:
        """
        Calibrate martingale measure to observed option prices.

        Uses the duality:
            V_{X,P,I}(G) = sup_{P ∈ M^I} E_P[G(S)]

        Returns calibration quality metrics and implied parameters.
        """
        # Extract calibration targets
        atm_iv = self._get_atm_iv(options_chain, spot)
        wing_ivs = self._get_wing_ivs(options_chain, spot)

        # Fit parametric martingale measure (e.g., mixture of lognormals)
        params = self._fit_mixture_model(atm_iv, wing_ivs)

        return CalibrationResult(
            params=params,
            calibration_error=self._compute_error(params, options_chain),
            prediction_set_support=self._check_support(params)
        )
```

### 2.4 Integration Points

| Existing File | Integration Hook | Change Description |
|---------------|-----------------|-------------------|
| `condor_brain.py:431` | After `last_hidden = x[:, -1, :]` | Add prediction set membership to output |
| `qtmf/facade.py:126` | After `min_gaussian_conf` check | Add prediction set gate |
| `train_condor_brain.py` | Feature preparation | Include martingale calibration features |

### 2.5 Testing Checkpoint

**Test 1.1: Prediction Set Membership**
```bash
python -m pytest tests/test_prediction_set.py -v
# Expected: All paths with IVR ∈ [20, 80] have μ_I > 0.5
```

**Test 1.2: Martingale Calibration**
```bash
python -m pytest tests/test_martingale_calibration.py -v
# Expected: Calibration error < 0.01 for ATM options
```

**Rollback Trigger:** If calibration error > 0.05 or training loss increases > 10%

---

## 3. Phase 2: Enhanced Loss Function

### 3.1 Objective

Extend `CompositeCondorLoss` with robust pricing-hedging terms:
- Superhedging cost penalty
- Martingale measure divergence
- Prediction set consistency

### 3.2 Mathematical Specification

The enhanced loss function:

```
L_total = λ₁·L_pred + λ₂·L_sharpe + λ₃·L_dd + λ₄·L_turn + λ₅·L_rule
        + λ₆·L_superhedge + λ₇·L_martingale + λ₈·L_prediction_set
```

Where:

**L_superhedge** (Robust Pricing Penalty):
```
L_superhedge = max(0, predicted_premium - V_{X,P,I}(G))
```
Penalizes if model predicts premium higher than superhedging bound.

**L_martingale** (Measure Divergence):
```
L_martingale = KL(P_model || P_calibrated)
```
Penalizes divergence from calibrated martingale measure.

**L_prediction_set** (Belief Consistency):
```
L_prediction_set = (1 - μ_I(state)) · confidence
```
Penalizes high confidence outside prediction set.

### 3.3 Code Changes

```python
# intelligence/condor_loss.py - ADDITIONS

class EnhancedCompositeCondorLoss(CompositeCondorLoss):
    """
    Extended loss with robust pricing-hedging terms.

    New components (λ₆, λ₇, λ₈):
        6. Superhedging bound violation
        7. Martingale measure divergence
        8. Prediction set consistency
    """

    def __init__(
        self,
        lambdas: Tuple[float, ...] = (1.0, 0.5, 0.1, 0.1, 1.0, 0.2, 0.1, 0.3),
        huber_delta: float = 1.0,
        dd_tau: float = 0.02
    ):
        # First 5 lambdas go to parent
        super().__init__(lambdas[:5], huber_delta, dd_tau)
        self.lambda_superhedge = lambdas[5] if len(lambdas) > 5 else 0.0
        self.lambda_martingale = lambdas[6] if len(lambdas) > 6 else 0.0
        self.lambda_prediction_set = lambdas[7] if len(lambdas) > 7 else 0.0

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        last_weights: Optional[torch.Tensor] = None,
        rule_signals: Optional[torch.Tensor] = None,
        # NEW: Robust pricing inputs
        superhedge_bounds: Optional[torch.Tensor] = None,  # (B,)
        martingale_logprobs: Optional[torch.Tensor] = None,  # (B,)
        prediction_set_membership: Optional[torch.Tensor] = None  # (B,)
    ) -> torch.Tensor:
        # Get base loss from parent
        base_loss = super().forward(y_pred, y_true, returns, weights, last_weights, rule_signals)

        device = y_pred.device
        l_super = torch.tensor(0.0, device=device)
        l_mart = torch.tensor(0.0, device=device)
        l_pred_set = torch.tensor(0.0, device=device)

        # 6. Superhedging Bound Violation
        if superhedge_bounds is not None and self.lambda_superhedge > 0:
            # y_pred[:, 4] is prob_profit, which relates to premium
            predicted_value = y_pred[:, 4]  # Or use expected_roi
            violation = torch.relu(predicted_value - superhedge_bounds)
            l_super = violation.mean()

        # 7. Martingale Divergence (negative log-prob)
        if martingale_logprobs is not None and self.lambda_martingale > 0:
            l_mart = -martingale_logprobs.mean()  # Maximize probability under calibrated measure

        # 8. Prediction Set Consistency
        if prediction_set_membership is not None and self.lambda_prediction_set > 0:
            confidence = y_pred[:, 7]  # Confidence output
            # Penalize high confidence outside prediction set
            l_pred_set = ((1.0 - prediction_set_membership) * confidence).mean()

        total = (
            base_loss +
            self.lambda_superhedge * l_super +
            self.lambda_martingale * l_mart +
            self.lambda_prediction_set * l_pred_set
        )

        return total
```

### 3.4 Testing Checkpoint

**Test 2.1: Loss Backward Compatibility**
```bash
# Run existing training with new loss (lambdas 6-8 = 0)
python intelligence/train_condor_brain.py --epochs 1 --loss-class EnhancedCompositeCondorLoss
# Expected: Identical loss values to baseline
```

**Test 2.2: Gradient Flow**
```bash
python -m pytest tests/test_enhanced_loss_gradients.py -v
# Expected: All new terms have valid gradients
```

**Rollback Trigger:** If NaN in any loss component or training diverges

---

## 4. Phase 3: HJB Solver & No-Trade Zones

### 4.1 Objective

Implement the Passerini-Vázquez HJB framework:
- Value function V(t, x, q)
- No-Trade (NT) zone boundaries b±(t, x)
- Five trading regions logic

### 4.2 New Files to Create

```
intelligence/
├─ hjb/
│   ├─ __init__.py
│   ├─ value_function.py    # V(t, x, q) computation
│   ├─ nt_zones.py          # No-trade zone boundaries
│   ├─ trading_regions.py   # 5-region classification
│   └─ ou_process.py        # Ornstein-Uhlenbeck alpha model
```

### 4.3 Mathematical Specification

#### 4.3.1 Value Function

The HJB equation for optimal trading:

```
D_{t,x}V + (λ/2)(q - q*)² + min_u [C|u| + Ku² + u·∂V/∂q - gu] = 0
```

Where:
- q* = μ/λ (Markowitz target position)
- g(t, x) = E[∫_t^{2T} x_s ds | x_t = x] (integrated gain from fast signal)
- C = half-spread (linear cost)
- K = temporary impact coefficient

```python
# intelligence/hjb/value_function.py

class HJBValueFunction:
    """
    Computes the value function V(t, x, q) for optimal trading.

    For Iron Condors, we adapt the single-asset framework:
        - q represents net delta exposure
        - μ represents daily alpha (IC premium capture rate)
        - x_t represents intraday alpha deviation (mean-reverting)
        - λ represents risk aversion (position variance penalty)
    """

    def __init__(
        self,
        mu: float,           # Daily alpha (expected premium capture)
        lambda_risk: float,  # Risk aversion parameter
        kappa: float,        # OU mean reversion speed
        eta: float,          # OU volatility
        half_spread: float,  # Linear cost C
        temp_impact: float,  # Quadratic impact K
        T: float = 1.0       # Trading horizon (1 = end of day)
    ):
        self.mu = mu
        self.lambda_risk = lambda_risk
        self.kappa = kappa
        self.eta = eta
        self.C = half_spread
        self.K = temp_impact
        self.T = T

        # Markowitz target
        self.q_star = mu / lambda_risk

    def integrated_gain(self, t: float, x: float) -> float:
        """
        Compute g(t, x) = E[∫_t^{2T} x_s ds | x_t = x]

        For OU process: g(t, x) = x · (2T - t) + (1 - exp(-κ(2T - t))) / κ
        """
        time_to_end = 2 * self.T - t
        return x * time_to_end + (1 - np.exp(-self.kappa * time_to_end)) / self.kappa

    def compute_value(self, t: float, x: float, q: float) -> float:
        """
        Approximate V(t, x, q) ≈ (λ/2)(2T - t)(q - q*)²

        Ignoring quadratic impact term (K → 0) for closed-form solution.
        """
        time_to_end = 2 * self.T - t
        return 0.5 * self.lambda_risk * time_to_end * (q - self.q_star) ** 2

    def partial_q(self, t: float, x: float, q: float) -> float:
        """
        ∂V/∂q = λ(2T - t)(q - q*)
        """
        time_to_end = 2 * self.T - t
        return self.lambda_risk * time_to_end * (q - self.q_star)
```

#### 4.3.2 No-Trade Zone Boundaries

```python
# intelligence/hjb/nt_zones.py

class NoTradeZoneComputer:
    """
    Computes NT zone boundaries b±(t, x).

    NT Zone Definition:
        b±(t, x) = q* ± (g(t, x) - C) / (2T - t)

    If q ∈ [b₋, b₊], optimal action is to hold (do not trade).
    """

    def __init__(self, value_function: HJBValueFunction):
        self.vf = value_function

    def compute_boundaries(self, t: float, x: float) -> Tuple[float, float]:
        """
        Returns (b_minus, b_plus) NT zone boundaries.
        """
        g = self.vf.integrated_gain(t, x)
        time_to_end = 2 * self.vf.T - t

        if time_to_end <= 0:
            # At terminal time, NT zone collapses to q*
            return self.vf.q_star, self.vf.q_star

        width = (g - self.vf.C) / time_to_end

        return self.vf.q_star - width, self.vf.q_star + width

    def classify_position(self, t: float, x: float, q: float) -> str:
        """
        Classify current position relative to NT zone.

        Returns: 'nt_zone', 'buy_zone', or 'sell_zone'
        """
        b_minus, b_plus = self.compute_boundaries(t, x)

        if b_minus <= q <= b_plus:
            return 'nt_zone'
        elif q < b_minus:
            return 'buy_zone'
        else:
            return 'sell_zone'
```

#### 4.3.3 Five Trading Regions (with Limit Orders)

```python
# intelligence/hjb/trading_regions.py

class TradingRegionClassifier:
    """
    Implements the 5-region classification for market + limit orders.

    Regions:
        1. Buy Market:  g > C(1 + P⁺)/P⁺ + ∂V/∂q
        2. Buy Limit:   C + ∂V/∂q < g < C(1 + P⁺)/P⁺ + ∂V/∂q
        3. Market-Making (NT): |g - ∂V/∂q| < C
        4. Sell Limit:  C(1 + P⁻)/P⁻ - ∂V/∂q < g < C + ∂V/∂q
        5. Sell Market: g < C(1 + P⁻)/P⁻ - ∂V/∂q

    Where P± are fill probabilities for limit orders.
    """

    def __init__(
        self,
        value_function: HJBValueFunction,
        fill_prob_buy: float = 0.5,   # P⁺
        fill_prob_sell: float = 0.5   # P⁻
    ):
        self.vf = value_function
        self.P_plus = fill_prob_buy
        self.P_minus = fill_prob_sell

    def classify(self, t: float, x: float, q: float) -> TradingDecision:
        """
        Classify market state into one of 5 trading regions.
        """
        g = self.vf.integrated_gain(t, x)
        dV_dq = self.vf.partial_q(t, x, q)
        C = self.vf.C

        # Thresholds
        buy_market_thresh = C * (1 + self.P_plus) / self.P_plus + dV_dq
        buy_limit_lower = C + dV_dq
        sell_limit_upper = C + dV_dq  # Same as buy_limit_lower
        sell_market_thresh = C * (1 + self.P_minus) / self.P_minus - dV_dq

        if g > buy_market_thresh:
            return TradingDecision(region='buy_market', order_type='market', direction='buy')
        elif buy_limit_lower < g <= buy_market_thresh:
            return TradingDecision(region='buy_limit', order_type='limit', direction='buy')
        elif abs(g - dV_dq) < C:
            return TradingDecision(region='market_making', order_type='none', direction='hold')
        elif sell_market_thresh < g <= sell_limit_upper:
            return TradingDecision(region='sell_limit', order_type='limit', direction='sell')
        else:
            return TradingDecision(region='sell_market', order_type='market', direction='sell')
```

### 4.4 Integration Points

| Existing File | Integration Hook | Change Description |
|---------------|-----------------|-------------------|
| `rule_engine/executor.py` | Gate stack (Phase 3) | Add HJB NT zone gate |
| `qtmf/facade.py` | After fuzzy scaling | Modulate by NT zone distance |
| `condor_brain.py` | New output head | Add NT zone probability output |

### 4.5 New Gate: HJB No-Trade Zone Gate

```python
# intelligence/primitives/hjb_gates.py

def compute_nt_zone_gate(
    t: float,           # Current time (0 = open, 1 = close)
    alpha_x: float,     # Current intraday alpha (OU state)
    position_q: float,  # Current net delta exposure
    hjb_params: dict    # HJB parameters
) -> float:
    """
    Gate G011: HJB No-Trade Zone Gate

    Returns:
        1.0 if position is within NT zone (block new trades)
        0.0 if position is outside NT zone (allow trades)
        Intermediate values for fuzzy boundary handling
    """
    nt_computer = NoTradeZoneComputer(HJBValueFunction(**hjb_params))
    b_minus, b_plus = nt_computer.compute_boundaries(t, alpha_x)

    # Fuzzy membership in NT zone
    if b_minus <= position_q <= b_plus:
        # Distance from boundary (normalized)
        dist_to_boundary = min(position_q - b_minus, b_plus - position_q)
        zone_width = b_plus - b_minus
        if zone_width > 0:
            # Higher value = deeper in NT zone = stronger block
            return min(1.0, dist_to_boundary / (0.5 * zone_width))
        return 1.0
    return 0.0
```

### 4.6 Testing Checkpoint

**Test 3.1: NT Zone Boundaries**
```bash
python -m pytest tests/test_hjb_nt_zones.py -v
# Expected: b₋ < q* < b₊ for all valid inputs
```

**Test 3.2: Trading Region Classification**
```bash
python -m pytest tests/test_trading_regions.py -v
# Expected: 5 regions cover all possible states without overlap
```

**Test 3.3: NT Zone Gate Integration**
```bash
python scripts/validate_nt_zone_gate.py --backtest
# Expected: Fewer trades when in NT zone, no impact on out-of-zone trades
```

**Rollback Trigger:** If backtest Sharpe drops > 15% or trade frequency drops > 50%

---

## 5. Phase 4: Alpha Predictors (OU Process)

### 5.1 Objective

Implement Ornstein-Uhlenbeck alpha modeling for intraday signal:
- Daily alpha μ (constant drift)
- Intraday alpha x_t (mean-reverting)

### 5.2 Mathematical Specification

The alpha decomposition:
```
μ_t = μ + x_t

dx_t = -κ·x_t·dt + η·dZ_t
```

Where:
- μ = daily expected alpha (e.g., average premium capture)
- κ = mean reversion speed (calibrated to SPY microstructure)
- η = volatility of intraday alpha shocks

### 5.3 Code Implementation

```python
# intelligence/hjb/ou_process.py

class OUAlphaPredictor:
    """
    Ornstein-Uhlenbeck process for intraday alpha prediction.

    Calibrates to high-frequency SPY data and provides:
        - Real-time alpha state estimation
        - Forward alpha prediction (integrated gain)
        - Parameter estimation from historical data
    """

    def __init__(self, kappa: float = 5.0, eta: float = 0.01, mu_daily: float = 0.001):
        self.kappa = kappa   # Mean reversion speed (per day)
        self.eta = eta       # Volatility
        self.mu_daily = mu_daily  # Daily drift

        # State tracking
        self.x_t = 0.0       # Current OU state
        self.last_update = None

    def update(self, observed_return: float, dt: float):
        """
        Update OU state given observed return.

        Uses Kalman-like update:
            x_{t+dt} = x_t · exp(-κ·dt) + innovation
        """
        # Mean reversion
        decay = np.exp(-self.kappa * dt)
        predicted_x = self.x_t * decay

        # Innovation (observed - predicted)
        innovation = observed_return - self.mu_daily * dt - predicted_x * dt

        # Kalman gain (simplified)
        K = 0.5  # Could be adaptive

        self.x_t = predicted_x + K * innovation

    def predict_integrated_gain(self, t: float, horizon: float) -> float:
        """
        Compute g(t, x_t) = E[∫_t^{t+horizon} x_s ds | x_t]

        For OU process:
            g = x_t · horizon + (1 - exp(-κ·horizon)) / κ
        """
        return self.x_t * horizon + (1 - np.exp(-self.kappa * horizon)) / self.kappa

    @classmethod
    def fit_from_data(cls, returns: np.ndarray, dt: float = 1/78) -> 'OUAlphaPredictor':
        """
        Fit OU parameters from historical return data.

        Uses AR(1) regression:
            r_{t+1} = α + β·r_t + ε_t
            κ = -ln(β) / dt
            η = std(ε) / sqrt(dt)
        """
        from statsmodels.tsa.ar_model import AutoReg

        model = AutoReg(returns, lags=1).fit()
        beta = model.params[1]

        kappa = -np.log(abs(beta) + 1e-10) / dt
        eta = np.std(model.resid) / np.sqrt(dt)
        mu_daily = model.params[0] / dt

        return cls(kappa=kappa, eta=eta, mu_daily=mu_daily)
```

### 5.4 Integration with CondorBrain

Add OU alpha state as input feature:

```python
# intelligence/canonical_feature_registry.py - ADDITIONS

# New features for alpha prediction (indices 32-34 in V2.2)
ALPHA_FEATURES = {
    'ou_state': {
        'index': 32,
        'description': 'Current OU alpha state x_t',
        'normalization': 'robust_zscore',
        'nan_fill': 0.0
    },
    'integrated_gain': {
        'index': 33,
        'description': 'Expected integrated gain g(t, x_t)',
        'normalization': 'robust_zscore',
        'nan_fill': 0.0
    },
    'time_to_close': {
        'index': 34,
        'description': 'Fraction of trading day remaining (0=close, 1=open)',
        'normalization': 'none',
        'nan_fill': 0.5
    }
}
```

### 5.5 Testing Checkpoint

**Test 4.1: OU Parameter Fitting**
```bash
python -m pytest tests/test_ou_fitting.py -v
# Expected: Fitted κ between 1 and 20 for typical SPY data
```

**Test 4.2: Integrated Gain Computation**
```bash
python -m pytest tests/test_integrated_gain.py -v
# Expected: g monotonically decreases as t → T
```

**Rollback Trigger:** If OU state variance explodes (> 10x historical)

---

*[Document continues in Part 2...]*

---

## Addendum (2026-01-24): Master Spec Linkage

This Part 1 plan is preserved as-is. The authoritative engineering spec is now in:

- `docs/INTEGRATION_PLAN_MASTER.md`

Use Part 1 for narrative context and phased breakdowns, but implement strictly according to the master spec (I/O contracts, stability analysis, interface tables).
