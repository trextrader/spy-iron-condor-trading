# Integration Plan Part 2: Phases 5-8 + Testing & Rollback

---

## 6. Phase 5: Execution Cost & Fill Probability

### 6.1 Objective

Implement execution cost modeling and ML-based fill probability estimation:
- Linear cost C (half-spread modeling)
- Temporary impact K (market impact)
- Fill probability P± (limit order fill prediction)

### 6.2 New Files to Create

```
intelligence/
├─ execution/
│   ├─ __init__.py
│   ├─ cost_model.py          # C and K estimation
│   ├─ fill_probability.py    # ML fill probability
│   └─ order_router.py        # Market vs limit order selection
```

### 6.3 Mathematical Specification

#### 6.3.1 Execution Cost Model

```python
# intelligence/execution/cost_model.py

class ExecutionCostModel:
    """
    Models execution costs for SPY options trading.

    Components:
        1. Linear Cost C (half-spread): Immediate cost of crossing spread
        2. Temporary Impact K: Price impact from order flow

    For Iron Condors, costs are aggregated across 4 legs.
    """

    def __init__(
        self,
        base_spread_pct: float = 0.001,   # Base spread as % of premium
        vol_spread_coef: float = 0.5,     # Spread widens with IV
        impact_coef: float = 0.0001       # Market impact coefficient
    ):
        self.base_spread_pct = base_spread_pct
        self.vol_spread_coef = vol_spread_coef
        self.impact_coef = impact_coef

    def estimate_half_spread(self, iv: float, dte: float, moneyness: float) -> float:
        """
        Estimate half-spread C for an option contract.

        Spread Model:
            C = base × (1 + vol_coef × IV) × f(DTE) × g(moneyness)

        Where:
            f(DTE) = exp(-DTE/30) captures time decay widening
            g(moneyness) = 1 + |log(K/S)| captures OTM widening
        """
        time_factor = np.exp(-dte / 30.0)
        moneyness_factor = 1.0 + abs(np.log(moneyness + 1e-6))

        return self.base_spread_pct * (1 + self.vol_spread_coef * iv) * time_factor * moneyness_factor

    def estimate_temporary_impact(self, order_size: int, avg_volume: float) -> float:
        """
        Estimate temporary market impact K.

        Impact Model (Kyle's Lambda):
            K = λ × (size / avg_volume)^0.5
        """
        if avg_volume <= 0:
            return self.impact_coef

        size_ratio = order_size / avg_volume
        return self.impact_coef * np.sqrt(size_ratio)

    def total_execution_cost(
        self,
        legs: List[OptionLeg],
        sizes: List[int],
        market_data: dict
    ) -> float:
        """
        Total execution cost for Iron Condor (4 legs).

        Returns: Total cost in dollars
        """
        total = 0.0
        for leg, size in zip(legs, sizes):
            C = self.estimate_half_spread(leg.iv, leg.dte, leg.moneyness)
            K = self.estimate_temporary_impact(size, leg.avg_volume)
            total += (C + K * size) * leg.premium * 100 * size

        return total
```

#### 6.3.2 Fill Probability Model

```python
# intelligence/execution/fill_probability.py

class FillProbabilityModel:
    """
    ML-based fill probability estimation for limit orders.

    Uses features from order book state and microstructure:
        - Queue position
        - Spread width
        - Recent fill rates
        - Time of day
        - Volatility regime

    Output: P(fill within horizon | limit price, features)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = self._load_or_create_model(model_path)
        self.feature_scaler = None

    def _load_or_create_model(self, path: str) -> Any:
        """Load trained model or create default."""
        if path and os.path.exists(path):
            import joblib
            return joblib.load(path)

        # Default: Logistic regression with reasonable priors
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0)

    def prepare_features(self, order_book_state: dict) -> np.ndarray:
        """
        Extract features for fill probability prediction.

        Features:
            1. Normalized spread (spread / mid)
            2. Queue depth ratio (our_side / total)
            3. Recent fill rate (fills_last_minute / volume)
            4. Time to close (0 = close, 1 = open)
            5. Volatility regime indicator
            6. Order size relative to avg
        """
        features = [
            order_book_state.get('normalized_spread', 0.001),
            order_book_state.get('queue_depth_ratio', 0.5),
            order_book_state.get('recent_fill_rate', 0.1),
            order_book_state.get('time_to_close', 0.5),
            order_book_state.get('vol_regime', 0),  # 0=low, 1=normal, 2=high
            order_book_state.get('relative_size', 1.0)
        ]
        return np.array(features).reshape(1, -1)

    def predict(self, order_book_state: dict) -> Tuple[float, float]:
        """
        Predict fill probability.

        Returns:
            (P_fill_buy, P_fill_sell): Fill probabilities for buy/sell limit orders
        """
        features = self.prepare_features(order_book_state)

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features)[0]
            # Assume model outputs [no_fill, fill] probabilities
            p_fill = probs[1] if len(probs) > 1 else probs[0]
        else:
            # Fallback: Use sigmoid of linear prediction
            p_fill = 1.0 / (1.0 + np.exp(-self.model.predict(features)[0]))

        # Slight asymmetry: Sells typically fill faster in down markets
        vol_regime = order_book_state.get('vol_regime', 0)
        if vol_regime == 2:  # High vol
            p_buy = p_fill * 0.9
            p_sell = p_fill * 1.1
        else:
            p_buy = p_fill
            p_sell = p_fill

        return min(1.0, p_buy), min(1.0, p_sell)
```

#### 6.3.3 Order Router

```python
# intelligence/execution/order_router.py

class HJBOrderRouter:
    """
    Combines HJB trading regions with execution cost analysis
    to select optimal order type.

    Decision Logic:
        1. Classify into 5 HJB regions
        2. If market order region: Execute immediately
        3. If limit order region: Estimate fill prob, compare expected costs
        4. If NT zone: Hold (no trade)
    """

    def __init__(
        self,
        trading_classifier: TradingRegionClassifier,
        cost_model: ExecutionCostModel,
        fill_model: FillProbabilityModel
    ):
        self.trading_classifier = trading_classifier
        self.cost_model = cost_model
        self.fill_model = fill_model

    def route_order(
        self,
        t: float,
        x: float,
        q: float,
        target_q: float,
        order_book_state: dict,
        option_leg: OptionLeg
    ) -> OrderDecision:
        """
        Determine optimal order type and parameters.

        Returns OrderDecision with:
            - order_type: 'market', 'limit', or 'none'
            - price: Limit price (if limit order)
            - urgency: 0-1 urgency score
            - expected_cost: Estimated execution cost
        """
        # Step 1: HJB region classification
        region_decision = self.trading_classifier.classify(t, x, q)

        if region_decision.region == 'market_making':
            return OrderDecision(
                order_type='none',
                price=None,
                urgency=0.0,
                expected_cost=0.0,
                reason='nt_zone'
            )

        # Step 2: Determine direction and size
        direction = 'buy' if target_q > q else 'sell'
        size = abs(target_q - q)

        # Step 3: Estimate costs
        p_buy, p_sell = self.fill_model.predict(order_book_state)
        p_fill = p_buy if direction == 'buy' else p_sell

        C = self.cost_model.estimate_half_spread(
            option_leg.iv, option_leg.dte, option_leg.moneyness
        )

        # Step 4: Compare market vs limit expected costs
        # Market: Immediate fill at cost C
        # Limit: Fill with prob P at cost 0, otherwise re-evaluate
        market_cost = C
        limit_expected_cost = (1 - p_fill) * C * 1.5  # Penalty for non-fill

        if region_decision.region in ['buy_market', 'sell_market']:
            # HJB says use market order
            return OrderDecision(
                order_type='market',
                price=None,
                urgency=1.0,
                expected_cost=market_cost,
                reason=region_decision.region
            )
        else:
            # HJB says limit is acceptable
            if limit_expected_cost < market_cost:
                mid_price = order_book_state.get('mid_price', option_leg.premium)
                limit_price = mid_price * (0.999 if direction == 'buy' else 1.001)

                return OrderDecision(
                    order_type='limit',
                    price=limit_price,
                    urgency=0.5,
                    expected_cost=limit_expected_cost,
                    reason=region_decision.region
                )
            else:
                return OrderDecision(
                    order_type='market',
                    price=None,
                    urgency=0.8,
                    expected_cost=market_cost,
                    reason='limit_too_risky'
                )
```

### 6.4 Integration Points

| Existing File | Integration Hook | Change Description |
|---------------|-----------------|-------------------|
| `core/backtest_engine.py` | Order execution | Use cost model for slippage |
| `strategies/options_strategy.py` | `build_condor` | Route through HJB order router |
| `qtmf/facade.py` | After sizing | Adjust size for expected costs |

### 6.5 Testing Checkpoint

**Test 5.1: Cost Model Sanity**
```bash
python -m pytest tests/test_cost_model.py -v
# Expected: Half-spread increases with IV and decreases with DTE
```

**Test 5.2: Fill Probability Calibration**
```bash
python scripts/calibrate_fill_model.py --data data/historical_fills.csv
# Expected: AUC > 0.65 on held-out data
```

**Test 5.3: Order Router Integration**
```bash
python -m pytest tests/test_order_router.py -v
# Expected: Market orders in urgent regions, limits in patient regions
```

**Rollback Trigger:** If backtest execution costs increase > 20%

---

## 7. Phase 6: Fuzzy Logic as Measure

### 7.1 Objective

Extend fuzzy logic to operate as a measure on prediction sets:
```
S_trade(x) = S_max · μ_fuzzy(x) · φ_mamba(x) · RiskFactor(x)
```

This creates a mathematically rigorous connection between:
- Prediction sets (belief constraints)
- Fuzzy memberships (confidence degrees)
- Position sizing (capital allocation)

### 7.2 Mathematical Specification

The fuzzy measure μ_fuzzy can be interpreted as a probability measure on the prediction set:

```
E_μ_fuzzy[S_trade] = ∫_I S_trade(x) dμ_fuzzy(x)
```

### 7.3 Code Enhancement

```python
# intelligence/fuzzy_engine.py - ENHANCEMENTS

class FuzzyMeasure:
    """
    Interprets fuzzy confidence as a measure on prediction sets.

    Mathematical Framework:
        Given prediction set I ⊂ Ω, the fuzzy measure μ assigns
        membership degrees μ(x) ∈ [0, 1] to each state x.

        The position size S_trade is computed as an integral:
            S_trade = S_max · ∫_I μ(x) · φ_mamba(x) · r(x) dP(x)

        Where:
            - μ(x) = fuzzy confidence (10-factor weighted)
            - φ_mamba(x) = neural confidence from CondorBrain
            - r(x) = risk factor (drawdown, max loss constraints)
            - P = reference probability (empirical measure)
    """

    def __init__(
        self,
        prediction_set: PredictionSet,
        weights: Dict[str, float] = None
    ):
        self.prediction_set = prediction_set
        self.weights = weights or DEFAULT_FUZZY_WEIGHTS

    def compute_measure(self, state: MarketState) -> float:
        """
        Compute fuzzy measure μ(state).

        Returns product of:
            1. Prediction set membership (hard constraint)
            2. Fuzzy confidence (soft scoring)
        """
        # Hard constraint: Must be in prediction set
        ps_membership = self.prediction_set.membership(state)
        if ps_membership < 0.1:
            return 0.0  # Outside prediction set

        # Soft scoring: 10-factor fuzzy confidence
        memberships = self._compute_memberships(state)
        fuzzy_conf = sum(self.weights[k] * memberships[k] for k in self.weights)

        return ps_membership * fuzzy_conf

    def integrate_position_size(
        self,
        S_max: float,
        neural_conf: float,
        risk_factor: float,
        state: MarketState
    ) -> float:
        """
        Compute position size as fuzzy-weighted integral.

        S_trade = S_max · μ_fuzzy(state) · φ_mamba · RiskFactor
        """
        mu = self.compute_measure(state)
        return S_max * mu * neural_conf * risk_factor

    def _compute_memberships(self, state: MarketState) -> Dict[str, float]:
        """Compute individual membership functions."""
        return {
            'mtf': calculate_mtf_membership(state.mtf_snapshot),
            'iv': calculate_iv_membership(state.ivr),
            'regime': calculate_regime_membership(state.vix),
            'rsi': calculate_rsi_membership(state.rsi),
            'adx': calculate_adx_membership(state.adx),
            'bbands': calculate_bbands_membership(state.bb_position, state.bb_width),
            'stoch': calculate_stoch_membership(state.stoch_k),
            'volume': calculate_volume_membership(state.volume_ratio),
            'sma': calculate_sma_distance_membership(state.sma_distance),
            'psar': calculate_psar_membership(state.psar_position)
        }
```

### 7.4 Integration with QTMF Facade

```python
# qtmf/facade.py - MODIFICATIONS

def benchmark_and_size_v2(
    trade_intent: TradeIntent,
    prediction_set: Optional[PredictionSet] = None,
    hjb_params: Optional[dict] = None
) -> SizingPlan:
    """
    Enhanced sizing with prediction set and HJB integration.

    New Parameters:
        prediction_set: Belief constraints (from robust pricing)
        hjb_params: HJB value function parameters (from optimal trading)

    The sizing formula becomes:
        S = S_max · μ_I(x) · μ_fuzzy(x) · φ_mamba(x) · RiskFactor · NT_gate
    """
    # ... existing code up to fuzzy computation ...

    # NEW: Prediction Set Gate
    ps_membership = 1.0
    if prediction_set is not None:
        market_state = _build_market_state(trade_intent)
        ps_membership = prediction_set.membership(market_state)

        if ps_membership < 0.1:
            return SizingPlan(
                approved=False,
                reason='outside_prediction_set',
                total_qty=0,
                put_qty=0,
                call_qty=0,
                diagnostics={'ps_membership': ps_membership}
            )

    # NEW: HJB No-Trade Zone Gate
    nt_gate = 1.0
    if hjb_params is not None:
        t = trade_intent.extras.get('time_of_day', 0.5)
        x = trade_intent.extras.get('ou_alpha_state', 0.0)
        q = trade_intent.extras.get('current_position', 0.0)

        nt_computer = NoTradeZoneComputer(HJBValueFunction(**hjb_params))
        position_class = nt_computer.classify_position(t, x, q)

        if position_class == 'nt_zone':
            nt_gate = 0.0  # Block new trades
            return SizingPlan(
                approved=False,
                reason='hjb_no_trade_zone',
                total_qty=0,
                put_qty=0,
                call_qty=0,
                diagnostics={'nt_zone': True, 't': t, 'x': x, 'q': q}
            )

    # Apply prediction set and HJB modulation to scaling
    g = compute_scaling_factor(fused_conf, sigma_star, min_scale=0.10)
    g_modulated = g * ps_membership * nt_gate

    # ... rest of existing logic with g_modulated instead of g ...
```

### 7.5 Testing Checkpoint

**Test 6.1: Fuzzy Measure Normalization**
```bash
python -m pytest tests/test_fuzzy_measure.py -v
# Expected: μ(x) ∈ [0, 1] for all states
```

**Test 6.2: Prediction Set Integration**
```bash
python -m pytest tests/test_ps_integration.py -v
# Expected: Zero sizing when outside prediction set
```

**Rollback Trigger:** If sizing logic produces negative values or assertion failures

---

## 8. Phase 7: Computational Physics Features

### 8.1 Objective

Add advanced features from computational physics:
- Curvature proxy κ_t
- Volatility energy E_t
- Persistent homology signature Π_t

### 8.2 Mathematical Specification

#### 8.2.1 Curvature Proxy

```
κ_t = EMA_64(r̈_t) / (σ_t + ε)

Where:
    r̈_t = r_t - 2r_{t-1} + r_{t-2}  (second difference of log returns)
    σ_t = EWMA volatility
```

#### 8.2.2 Volatility Energy

```
E_t = ln(1 + α|κ_t|)
```

High energy signals regime transitions and volatility spikes.

#### 8.2.3 Persistent Homology (Simplified)

```
Π_t = sum of top-J persistence lifetimes from Takens embedding
```

### 8.3 Code Implementation

```python
# intelligence/features/physics_features.py

def compute_curvature_proxy(returns: pd.Series, span: int = 64) -> pd.Series:
    """
    Compute curvature proxy κ_t.

    Mathematical Definition:
        r̈_t = r_t - 2r_{t-1} + r_{t-2}  (second difference)
        scale_t = EMA(|r'_t|, span)      (local scale)
        κ_t = EMA(r̈_t, span/4) / (scale_t + ε)

    Interpretation:
        High |κ| indicates rapid directional change (potential reversal)
        Low |κ| indicates smooth trend continuation
    """
    # First difference
    dr = returns.diff()

    # Second difference (curvature)
    d2r = dr.diff()

    # Local scale (EMA of absolute first differences)
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12

    # Normalized curvature
    kappa = d2r / scale

    # Smooth with shorter EMA
    return kappa.ewm(span=max(8, span // 4), adjust=False).mean()


def compute_volatility_energy(kappa: pd.Series, alpha: float = 1.0) -> pd.Series:
    """
    Compute volatility energy E_t = ln(1 + α|κ_t|).

    Interpretation:
        E_t > 1.0: High energy (regime transition likely)
        E_t < 0.5: Low energy (stable regime)
    """
    return np.log1p(alpha * kappa.abs())


def compute_persistent_homology_signature(
    prices: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 5,
    top_j: int = 5
) -> float:
    """
    Compute simplified persistent homology signature.

    Uses Takens delay embedding to create point cloud,
    then estimates 1D persistent homology via distance matrix.

    Note: Full TDA requires specialized libraries (giotto-tda, ripser).
    This provides an approximation using pairwise distances.

    Returns:
        Π_t: Sum of top-J "persistence" estimates (proxy for true lifetimes)
    """
    n = len(prices)
    if n < embedding_dim * delay:
        return 0.0

    # Takens embedding: x_i = [p_i, p_{i+d}, p_{i+2d}, ...]
    embedded = np.zeros((n - (embedding_dim - 1) * delay, embedding_dim))
    for i in range(embedding_dim):
        start = i * delay
        end = n - (embedding_dim - 1 - i) * delay
        embedded[:, i] = prices[start:end]

    # Normalize
    embedded = (embedded - embedded.mean(axis=0)) / (embedded.std(axis=0) + 1e-10)

    # Pairwise distance matrix
    from scipy.spatial.distance import pdist
    distances = pdist(embedded, metric='euclidean')

    # Proxy for persistence: Range of distance scales
    # (True homology would use filtrations and boundary matrices)
    sorted_dists = np.sort(distances)

    # Take differences between consecutive distance quantiles as "lifetimes"
    quantiles = np.percentile(sorted_dists, np.linspace(10, 90, 20))
    lifetimes = np.diff(quantiles)

    # Sum top-J lifetimes
    top_lifetimes = np.sort(lifetimes)[-top_j:]
    return float(np.sum(top_lifetimes))
```

### 8.4 Feature Registry Update

```python
# intelligence/canonical_feature_registry.py - V2.2 ADDITIONS

PHYSICS_FEATURES = {
    'kappa_proxy': {
        'index': 35,
        'description': 'Curvature proxy κ_t (normalized second derivative)',
        'normalization': 'robust_zscore',
        'nan_fill': 0.0,
        'compute_fn': 'compute_curvature_proxy'
    },
    'vol_energy': {
        'index': 36,
        'description': 'Volatility energy E_t = ln(1 + α|κ|)',
        'normalization': 'robust_zscore',
        'nan_fill': 0.5,
        'compute_fn': 'compute_volatility_energy'
    },
    'homology_sig': {
        'index': 37,
        'description': 'Persistent homology signature Π_t',
        'normalization': 'robust_zscore',
        'nan_fill': 0.0,
        'compute_fn': 'compute_persistent_homology_signature'
    }
}

# Updated total: 32 (V2.1) + 3 (Alpha) + 3 (Physics) = 38 features
FEATURE_DIM_V2_2 = 38
```

### 8.5 Testing Checkpoint

**Test 7.1: Curvature Computation**
```bash
python -m pytest tests/test_physics_features.py::test_curvature -v
# Expected: κ peaks at trend reversals
```

**Test 7.2: Energy Detection**
```bash
python -m pytest tests/test_physics_features.py::test_energy -v
# Expected: E_t > 1.0 during regime transitions
```

**Test 7.3: Feature Integration**
```bash
python scripts/validate_feature_registry.py --version 2.2
# Expected: All 38 features compute without NaN
```

**Rollback Trigger:** If training loss increases > 5% with new features

---

## 9. Phase 8: Governance & Risk Override Layer

### 9.1 Objective

Implement a unified governance layer that combines:
- HJB optimal control (execution timing)
- Robust pricing bounds (premium validation)
- Prediction set constraints (belief filtering)
- Fuzzy risk gating (confidence thresholds)

### 9.2 Governance Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GOVERNANCE LAYER (NEW)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Level 1: PREDICTION SET FILTER                                            │
│  ├─ IVR bounds check                                                        │
│  ├─ Liquidity gate (spread_ratio < threshold)                              │
│  ├─ Gap risk exclusion                                                      │
│  └─ Chaos membership filter                                                 │
│                                                                             │
│  Level 2: HJB OPTIMAL CONTROL                                              │
│  ├─ No-Trade Zone gate (position within NT bounds)                         │
│  ├─ Trading region classification (market/limit/hold)                      │
│  └─ Alpha predictor state check (OU stability)                             │
│                                                                             │
│  Level 3: ROBUST PRICING VALIDATION                                        │
│  ├─ Superhedging bound check (premium ≤ V_{X,P,I})                        │
│  ├─ Martingale calibration quality                                         │
│  └─ Greeks risk limits (delta/gamma/vega bounds)                           │
│                                                                             │
│  Level 4: FUZZY CONFIDENCE GATE                                            │
│  ├─ 10-factor fuzzy score threshold                                        │
│  ├─ Neural confidence minimum                                              │
│  └─ Composite confidence fusion                                            │
│                                                                             │
│  Level 5: INSTITUTIONAL RULES                                              │
│  ├─ 14 rule primitives (A1-F3)                                             │
│  ├─ Rule block flag enforcement                                            │
│  └─ Position size scaling                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Code Implementation

```python
# intelligence/governance/governance_layer.py

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class GovernanceLevel(Enum):
    PREDICTION_SET = 1
    HJB_CONTROL = 2
    ROBUST_PRICING = 3
    FUZZY_CONFIDENCE = 4
    INSTITUTIONAL_RULES = 5


@dataclass
class GovernanceResult:
    approved: bool
    blocked_at: Optional[GovernanceLevel]
    reason: str
    diagnostics: dict
    risk_score: float  # 0 = safe, 1 = max risk


class GovernanceLayer:
    """
    Unified governance for CondorBrain trading decisions.

    Implements a 5-level risk filter cascade:
        Level 1: Prediction Set (hard belief constraints)
        Level 2: HJB Optimal Control (timing and position)
        Level 3: Robust Pricing (premium validation)
        Level 4: Fuzzy Confidence (soft scoring)
        Level 5: Institutional Rules (safety gates)

    Each level can BLOCK, WARN, or PASS the trade.
    """

    def __init__(
        self,
        prediction_set: PredictionSet,
        hjb_params: dict,
        martingale_calibrator: MartingaleMeasureSet,
        fuzzy_weights: dict,
        rule_executor: RuleExecutor
    ):
        self.prediction_set = prediction_set
        self.hjb_params = hjb_params
        self.martingale = martingale_calibrator
        self.fuzzy_weights = fuzzy_weights
        self.rule_executor = rule_executor

    def evaluate(
        self,
        trade_intent: TradeIntent,
        market_state: MarketState,
        model_output: CondorSignal
    ) -> GovernanceResult:
        """
        Run full governance evaluation.

        Returns GovernanceResult with approval status and diagnostics.
        """
        diagnostics = {}

        # Level 1: Prediction Set
        ps_membership = self.prediction_set.membership(market_state)
        diagnostics['ps_membership'] = ps_membership

        if ps_membership < 0.1:
            return GovernanceResult(
                approved=False,
                blocked_at=GovernanceLevel.PREDICTION_SET,
                reason=f'outside_prediction_set(μ={ps_membership:.3f})',
                diagnostics=diagnostics,
                risk_score=1.0
            )

        # Level 2: HJB Optimal Control
        t = market_state.time_of_day
        x = market_state.ou_alpha_state
        q = market_state.current_position

        nt_computer = NoTradeZoneComputer(HJBValueFunction(**self.hjb_params))
        position_class = nt_computer.classify_position(t, x, q)
        diagnostics['hjb_position'] = position_class

        if position_class == 'nt_zone':
            return GovernanceResult(
                approved=False,
                blocked_at=GovernanceLevel.HJB_CONTROL,
                reason='hjb_no_trade_zone',
                diagnostics=diagnostics,
                risk_score=0.3  # Not high risk, just suboptimal timing
            )

        # Level 3: Robust Pricing Validation
        calibration = self.martingale.calibrate(market_state.options_chain, market_state.spot)
        diagnostics['calibration_error'] = calibration.calibration_error

        if calibration.calibration_error > 0.05:
            return GovernanceResult(
                approved=False,
                blocked_at=GovernanceLevel.ROBUST_PRICING,
                reason=f'poor_calibration(err={calibration.calibration_error:.3f})',
                diagnostics=diagnostics,
                risk_score=0.7
            )

        # Level 4: Fuzzy Confidence Gate
        fuzzy_conf = self._compute_fuzzy_confidence(market_state)
        neural_conf = model_output.confidence
        fused_conf = 0.6 * neural_conf + 0.4 * fuzzy_conf
        diagnostics['fuzzy_conf'] = fuzzy_conf
        diagnostics['neural_conf'] = neural_conf
        diagnostics['fused_conf'] = fused_conf

        if fused_conf < 0.55:
            return GovernanceResult(
                approved=False,
                blocked_at=GovernanceLevel.FUZZY_CONFIDENCE,
                reason=f'low_confidence(fused={fused_conf:.3f})',
                diagnostics=diagnostics,
                risk_score=0.5
            )

        # Level 5: Institutional Rules
        rule_result = self.rule_executor.evaluate(market_state)
        diagnostics['rule_block'] = rule_result.block_flag
        diagnostics['rule_reason'] = rule_result.reason

        if rule_result.block_flag:
            return GovernanceResult(
                approved=False,
                blocked_at=GovernanceLevel.INSTITUTIONAL_RULES,
                reason=f'rule_block({rule_result.reason})',
                diagnostics=diagnostics,
                risk_score=0.8
            )

        # All levels passed
        risk_score = self._compute_composite_risk(ps_membership, fused_conf, calibration)
        diagnostics['risk_score'] = risk_score

        return GovernanceResult(
            approved=True,
            blocked_at=None,
            reason='approved',
            diagnostics=diagnostics,
            risk_score=risk_score
        )

    def _compute_fuzzy_confidence(self, state: MarketState) -> float:
        """Compute 10-factor fuzzy confidence."""
        memberships = {
            'mtf': calculate_mtf_membership(state.mtf_snapshot),
            'iv': calculate_iv_membership(state.ivr),
            'regime': calculate_regime_membership(state.vix),
            'rsi': calculate_rsi_membership(state.rsi),
            'adx': calculate_adx_membership(state.adx),
            'bbands': calculate_bbands_membership(state.bb_position, state.bb_width),
            'stoch': calculate_stoch_membership(state.stoch_k),
            'volume': calculate_volume_membership(state.volume_ratio),
            'sma': calculate_sma_distance_membership(state.sma_distance),
            'psar': calculate_psar_membership(state.psar_position)
        }
        return sum(self.fuzzy_weights[k] * memberships[k] for k in self.fuzzy_weights)

    def _compute_composite_risk(
        self,
        ps_membership: float,
        fused_conf: float,
        calibration: CalibrationResult
    ) -> float:
        """
        Compute composite risk score.

        Lower is better: 0 = very safe, 1 = maximum risk
        """
        ps_risk = 1.0 - ps_membership
        conf_risk = 1.0 - fused_conf
        cal_risk = min(1.0, calibration.calibration_error * 10)

        return 0.4 * ps_risk + 0.4 * conf_risk + 0.2 * cal_risk
```

### 9.4 Integration with Main Pipeline

```python
# core/main.py - MODIFICATION

def execute_trade_with_governance(
    trade_intent: TradeIntent,
    market_state: MarketState,
    model_output: CondorSignal,
    governance: GovernanceLayer
) -> ExecutionResult:
    """
    Execute trade through governance layer.
    """
    # Step 1: Governance evaluation
    gov_result = governance.evaluate(trade_intent, market_state, model_output)

    if not gov_result.approved:
        logger.warning(
            f"Trade blocked at {gov_result.blocked_at.name}: {gov_result.reason}"
        )
        return ExecutionResult(
            executed=False,
            reason=gov_result.reason,
            governance_diagnostics=gov_result.diagnostics
        )

    # Step 2: Sizing (now with governance risk adjustment)
    sizing_plan = benchmark_and_size_v2(
        trade_intent,
        prediction_set=governance.prediction_set,
        hjb_params=governance.hjb_params
    )

    if not sizing_plan.approved:
        return ExecutionResult(
            executed=False,
            reason=sizing_plan.reason,
            sizing_diagnostics=sizing_plan.diagnostics
        )

    # Step 3: Execute (with risk-adjusted size)
    adjusted_qty = int(sizing_plan.total_qty * (1.0 - 0.5 * gov_result.risk_score))
    adjusted_qty = max(2, adjusted_qty)  # Minimum 2 for Iron Condor

    return execute_condor(
        model_output,
        put_qty=sizing_plan.put_qty,
        call_qty=sizing_plan.call_qty,
        total_qty=adjusted_qty
    )
```

### 9.5 Testing Checkpoint

**Test 8.1: Governance Cascade**
```bash
python -m pytest tests/test_governance_layer.py -v
# Expected: Each level blocks appropriately
```

**Test 8.2: Risk Score Calibration**
```bash
python scripts/calibrate_risk_scores.py --backtest
# Expected: Higher risk scores correlate with worse outcomes
```

**Test 8.3: Full Pipeline Integration**
```bash
python core/main.py --mode backtest --use-governance --bt-samples 0
# Expected: Similar or better Sharpe, lower drawdown
```

**Rollback Trigger:** If backtest net profit drops > 20%

---

## 10. Testing & Validation Strategy

### 10.1 Test Hierarchy

```
tests/
├─ unit/
│   ├─ test_prediction_set.py
│   ├─ test_martingale_calibration.py
│   ├─ test_hjb_value_function.py
│   ├─ test_nt_zones.py
│   ├─ test_ou_process.py
│   ├─ test_cost_model.py
│   ├─ test_fill_probability.py
│   ├─ test_physics_features.py
│   └─ test_governance_layer.py
├─ integration/
│   ├─ test_enhanced_loss.py
│   ├─ test_condor_brain_v22.py
│   ├─ test_qtmf_v2.py
│   └─ test_full_pipeline.py
└─ regression/
    ├─ test_baseline_parity.py      # Ensure V2.1 behavior preserved
    ├─ test_training_stability.py   # No NaN/explosion
    └─ test_backtest_metrics.py     # Sharpe, DD, NP/DD
```

### 10.2 Validation Commands

```bash
# Phase 1 Validation
python -m pytest tests/unit/test_prediction_set.py tests/unit/test_martingale_calibration.py -v

# Phase 2 Validation
python -m pytest tests/unit/test_enhanced_loss.py tests/regression/test_baseline_parity.py -v

# Phase 3 Validation
python -m pytest tests/unit/test_hjb_*.py tests/unit/test_nt_zones.py -v

# Phase 4 Validation
python -m pytest tests/unit/test_ou_process.py -v

# Phase 5 Validation
python -m pytest tests/unit/test_cost_model.py tests/unit/test_fill_probability.py -v

# Phase 6 Validation
python -m pytest tests/integration/test_qtmf_v2.py -v

# Phase 7 Validation
python -m pytest tests/unit/test_physics_features.py -v

# Phase 8 Validation
python -m pytest tests/unit/test_governance_layer.py tests/integration/test_full_pipeline.py -v

# Full Regression Suite
python -m pytest tests/regression/ -v --tb=short
```

### 10.3 Baseline Metrics (Must Preserve)

| Metric | Current Baseline | Acceptable Range |
|--------|-----------------|------------------|
| Sharpe Ratio | 1.85 | ≥ 1.5 |
| Max Drawdown | -12.3% | ≤ -18% |
| NP/DD Ratio | 2.4 | ≥ 1.8 |
| Win Rate | 68% | ≥ 60% |
| Avg Trade Duration | 4.2 days | 2-7 days |

---

## 11. Rollback Procedures

### 11.1 Per-Phase Rollback

Each phase has isolated modules. To rollback:

```bash
# Phase N rollback
git checkout HEAD~1 -- intelligence/{module_name}/
python -m pytest tests/regression/ -v  # Verify baseline restored
```

### 11.2 Full System Rollback

If multiple phases cause issues:

```bash
# Create backup branch before starting
git checkout -b pre-integration-backup

# After integration, if rollback needed:
git checkout pre-integration-backup
git checkout -b main-restored
```

### 11.3 Feature Flags

All new features are gated:

```python
# core/config.py

class FeatureFlags:
    USE_PREDICTION_SETS = False        # Phase 1
    USE_ENHANCED_LOSS = False          # Phase 2
    USE_HJB_CONTROL = False            # Phase 3
    USE_ALPHA_PREDICTORS = False       # Phase 4
    USE_EXECUTION_COSTS = False        # Phase 5
    USE_FUZZY_MEASURE = False          # Phase 6
    USE_PHYSICS_FEATURES = False       # Phase 7
    USE_GOVERNANCE_LAYER = False       # Phase 8
```

Toggle flags to enable/disable features without code changes.

---

## 12. File Manifest

### 12.1 New Files to Create

```
intelligence/
├─ robust/
│   ├─ __init__.py
│   ├─ prediction_set.py
│   ├─ martingale_measure.py
│   └─ path_filter.py
├─ hjb/
│   ├─ __init__.py
│   ├─ value_function.py
│   ├─ nt_zones.py
│   ├─ trading_regions.py
│   └─ ou_process.py
├─ execution/
│   ├─ __init__.py
│   ├─ cost_model.py
│   ├─ fill_probability.py
│   └─ order_router.py
├─ features/
│   └─ physics_features.py
├─ governance/
│   ├─ __init__.py
│   └─ governance_layer.py
└─ primitives/
    └─ hjb_gates.py

tests/
├─ unit/
│   ├─ test_prediction_set.py
│   ├─ test_martingale_calibration.py
│   ├─ test_hjb_value_function.py
│   ├─ test_nt_zones.py
│   ├─ test_ou_process.py
│   ├─ test_cost_model.py
│   ├─ test_fill_probability.py
│   ├─ test_physics_features.py
│   └─ test_governance_layer.py
├─ integration/
│   ├─ test_enhanced_loss.py
│   ├─ test_condor_brain_v22.py
│   ├─ test_qtmf_v2.py
│   └─ test_full_pipeline.py
└─ regression/
    ├─ test_baseline_parity.py
    ├─ test_training_stability.py
    └─ test_backtest_metrics.py
```

### 12.2 Files to Modify

| File | Changes |
|------|---------|
| `intelligence/condor_loss.py` | Add EnhancedCompositeCondorLoss |
| `intelligence/condor_brain.py` | Add prediction set output, V2.2 features |
| `intelligence/canonical_feature_registry.py` | Add V2.2 features (38 total) |
| `intelligence/fuzzy_engine.py` | Add FuzzyMeasure class |
| `qtmf/facade.py` | Add benchmark_and_size_v2 |
| `intelligence/rule_engine/executor.py` | Add HJB NT zone gate |
| `core/config.py` | Add FeatureFlags |
| `core/main.py` | Add governance integration |

---

## Summary

This integration plan provides a mathematically rigorous, incrementally testable path for incorporating:

1. **Robust Pricing-Hedging Duality**: Prediction sets, martingale measures, superhedging bounds
2. **Passerini-Vázquez HJB Framework**: Optimal trading control, no-trade zones, alpha predictors
3. **Execution Optimization**: Cost modeling, fill probability, order routing
4. **Governance Layer**: Unified 5-level risk filtering

Each phase:
- Has isolated modules (easy rollback)
- Includes specific test checkpoints
- Defines rollback triggers
- Preserves baseline behavior via feature flags

**Estimated Implementation Order:**
1. Phase 1-2 (Foundation): Prediction sets + Loss enhancement
2. Phase 3-4 (Control): HJB solver + Alpha predictors
3. Phase 5 (Execution): Cost and fill modeling
4. Phase 6-7 (Features): Fuzzy measure + Physics features
5. Phase 8 (Integration): Governance layer

**Total New Files:** 18
**Total Modified Files:** 8
**Total Test Files:** 13

---

## Addendum (2026-01-24): Audit Findings + Validation Protocol

This Part 2 plan is preserved as-is. The master spec now includes:
- finite-difference stability + boundary conditions
- martingale calibration algorithm (constrained optimization)
- DTO/I-O appendix and interface tables
- audit findings against repo code and datasets
- validation + stress protocol

Reference:
- `docs/INTEGRATION_PLAN_MASTER.md`
- `docs/INTERFACE_CATALOG.md`
