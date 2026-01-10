# Quantor-MTFuzz: Advanced Options Trading System
## Iron Condor Algorithmic Trading with Neuro-Fuzzy Intelligence

**Quantor-MTFuzz** is a production-grade algorithmic trading system for SPY Iron Condor strategies, combining Multi-Timeframe (MTF) technical intelligence, Fuzzy Logic position sizing, and institutional-grade risk management. Built on the Quantor-MTFuzz architectural specification with clean module separation, comprehensive testing, and mathematical rigor.

---

## 🚀 Key Features

### Core Capabilities
- **High-Fidelity Backtesting**: 5-minute bar simulation with accurate mark-to-market P&L, leg-by-leg exit logic, and realistic slippage/commissions
- **Phased Serial Optimization**: Grid-search engine optimizing for **Net Profit / Max Drawdown** ratio with hardware benchmarking
- **10-Factor Fuzzy Intelligence**: Dynamic position sizing based on MTF Consensus, IV Rank, VIX Regime, RSI, ADX, Bollinger Bands, Stochastic, Volume, SMA Distance, and **Parabolic SAR**
- **Mamba 2 Neural Forecasting**: State-space model predicting market regimes (Bear/Neutral/Bull) and volatility states
- **Enhanced Risk Controls**: Portfolio Greeks tracking, drawdown caps, and beta-weighted delta limits
- **Alpaca Integration**: Seamless live paper trading via Alpaca-Py SDK
- **Professional Reporting**: Automated PDF reports with equity curves, strike overlays, P&L distributions

### 🔬 5-Phase Serial Optimization (NEW)
The system employs a segmented optimization engine to tune 30+ parameters without overfitting:

1.  **Phase 1: Exits & Risk** (Timing)
    *   Targets: Profit targets, stop-losses, max hold days, risk limits
    *   Goal: Optimize trade pacing and risk/reward ratios.
2.  **Phase 2: Structure & Entries** (Mechanics)
    *   Targets: Delta selection (0.10-0.30), wing widths, credit minimums
    *   Goal: Tune the instrument structure for the current market.
3.  **Phase 3: Filters & Regime** (Safety)
    *   Targets: IV Rank gates, VIX thresholds, volatility caps
    *   Goal: Refine entry gates to avoid high-risk environments.
4.  **Phase 4: Momentum Logic** (Signals)
    *   Targets: RSI/Stoch neutral zones, ADX trend thresholds
    *   Goal: Optimize momentum-based entry logic.
5.  **Phase 5: Trend & Volatility** (Context)
    *   Targets: Bollinger Band squeeze, SMA distance, PSAR acceleration
    *   Goal: Fine-tune trend-following and mean-reversion triggers.

### 🔍 Optimizable Parameter Reference (5-Phase Matrix)

Below is the complete list of parameters tunable via the `core/optimizer.py` engine, matched with defaults from `core/config.py`.

#### 1. Phase 1: Exits & Risk (The Pacing)
| Parameter | Default | Optimization Range |
|:---|:---:|:---|
| `profit_take_pct` | 0.40 | 0.40 - 0.95 (Step 0.1) |
| `loss_close_multiple` | 1.00 | 1.0 - 3.0 (Step 0.5) |
| `max_hold_days` | 10 | 10, 14, 21, 30 |
| `max_account_risk_per_trade` | 0.02 | 1%, 2%, 3% |

#### 2. Phase 2: Structure & Entries (The Vehicle)
| Parameter | Default | Optimization Range |
|:---|:---:|:---|
| `target_short_delta_low` | 0.08 | 0.10, 0.12, 0.15 |
| `target_short_delta_high` | 0.25 | 0.20, 0.25, 0.30 |
| `wing_width_min` | 5.0 | 5.0, 10.0 |
| `min_credit_to_width` | 0.10 | 10%, 15%, 20% |
| `use_skew_penalty` | True | True, False |

#### 3. Phase 3: Filters & Regime (The Safety)
| Parameter | Default | Optimization Range |
|:---|:---:|:---|
| `iv_rank_min` | 0.0 | 0.0, 10.0, 20.0, 30.0 |
| `vix_threshold` | 25.0 | 25.0, 30.0, 40.0 |
| `vix_threshold_low` | 20.0 | 15.0, 18.0, 20.0 |
| `max_volatility_pct` | 0.02 | 2%, 3%, 4% |

#### 4. Phase 4: Momentum Logic (RSI, Stoch, ADX)
| Parameter | Default | Optimization Range |
|:---|:---:|:---|
| `rsi_neutral_min` | 30 | 30, 40 |
| `rsi_neutral_max` | 60 | 60, 70 |
| `stoch_neutral_min` | 30 | 20, 30 |
| `stoch_neutral_max` | 70 | 70, 80 |
| `adx_threshold_low` | 25.0 | 20.0, 25.0, 30.0 |
| `use_adx_filter` | True | True, False |

#### 5. Phase 5: Trend & Volatility (BBands, SMA, PSAR)
| Parameter | Default | Optimization Range |
|:---|:---:|:---|
| `bbands_squeeze_threshold` | 0.02 | 1%, 2%, 3% |
| `sma_max_distance` | 0.02 | 1%, 2%, 3%, 4% |
| `psar_acceleration` | 0.02 | 0.02, 0.025 |
| `psar_max_acceleration` | 0.20 | 0.20, 0.25 |
| `use_psar_filter` | True | True, False |

### Phase 1+ Analytics & Data Pipeline (NEW)
- **Volatility Risk Premium (VRP)**: Realized vs Implied volatility edge detection
- **SPY-ES Divergence**: Z-score based divergence trading signals
- **IV Skew Analysis**: Crash-risk detection via put/call skew metrics
- **Gap Classification**: Overnight gap-fill mean-reversion logic
- **Cost-of-Carry**: Futures fair value and basis calculations
- **Real Indicators**: ADX/RSI/IV Rank with Wilder smoothing for 5-minute bars
- **Lag-Aware Data Pipeline**: Institutional-grade timestamp alignment with IV confidence decay
  - Auto-overlap day selection (spot ∩ options)
  - ChainAlignment engine (exact/prior/stale/none modes)
  - IV confidence decay: `iv_conf = 0.5^(lag_sec / 300)`
  - Per-symbol lag limits (SPY/QQQ: 600s, SPX: 900s)
  - Fail-fast mode (abort if stale rate > 20%)
  - Comprehensive diagnostics (exact match%, lag distribution, IV conf stats)

---

## 🏗️ Architecture: Quantor-MTFuzz Specification

### Module Hierarchy
```
core/
├── dto.py            → DTOs (MarketSnapshot, TradeDecision, etc.)
├── engine.py         → TradingEngine orchestrator
├── backtest_engine.py → Backtrader integration (legacy)
├── config.py         → Configuration model
└── risk_manager.py   → Portfolio risk gate

data_factory/         [NEW - Phase 2]
├── spot_bars.py      → Multi-timeframe OHLCV provider (1/5/15m)
├── option_chain.py   → Lag-aware chain alignment with IV decay
├── aux_feeds.py      → Gap analysis helpers
├── data_engine.py    → MarketSnapshot streaming + auto-overlap
├── sync_engine.py    → MTF data synchronization (legacy)
└── polygon_client.py → Market data provider (legacy)

strategies/
└── options_strategy.py → Iron Condor signal gating & strike selection

intelligence/
├── fuzzifier.py      → Feature extraction (ADX/RSI/IV Rank)
├── fuzzy_engine.py   → Fuzzy position sizing
├── regime_filter.py  → Market regime classification
└── mamba_engine.py   → Neural forecasting (Mamba 2)

analytics/            [Phase 1]
├── realized_vol.py   → Realized volatility calculator
├── divergence.py     → SPY-ES Z-score
├── skew.py           → IV skew metrics
├── gaps.py           → Gap analyzer
├── carry_model.py    → Cost-of-carry model
└── indicators.py     → ADX/RSI/IV Rank

risk/
└── risk_manager.py   → Greeks tracking, drawdown caps

tests/                [Phase 1]
└── test_*.py         → 18 unit tests (pytest)
```

---

## 🧮 Mathematical Foundation

### 1. Volatility Risk Premium (VRP) - `analytics/realized_vol.py`

**Theory**: The VRP is the edge captured by selling options when implied volatility exceeds realized volatility.

#### Realized Volatility Calculation
$$
RV^2 = \frac{252}{N} \sum_{t=1}^{N} r_t^2
$$

Where:
- $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$ = log return at time $t$
- $N$ = rolling window length (e.g., 78 bars for 1 trading day of 5-min data)
- $252$ = annualization factor (trading days per year)

$$
RV = \sqrt{RV^2}
$$

#### VRP Signal
$$
VRP = IV_{ATM} - RV
$$

**Entry Gate**: Trade only if $VRP > \theta_{VRP}$ (e.g., $\theta_{VRP} = 0.02$ or 2%)

**Implementation**:
```python
from analytics.realized_vol import RealizedVolCalculator

calc = RealizedVolCalculator()
rv = calc.compute_realized_vol(close_prices, window=78)
vrp = implied_vol - rv

if vrp > 0.02:
    # Favorable environment for selling premium
    pass
```

---

### 2. SPY-ES Divergence Trading - `analytics/divergence.py`

**Theory**: ES futures and SPY ETF should track closely due to arbitrage. Divergences signal temporary mispricing.

#### Spread Calculation
$$
S_t = P_{SPY,t} - \frac{P_{ES,t}}{10}
$$

Where:
- $P_{SPY,t}$ = SPY spot price
- $P_{ES,t}$ = ES futures price
- Factor of 10 accounts for ES contract multiplier ($SPY \approx ES / 10$)

#### Z-Score Normalization
$$
z_t = \frac{S_t - \mu_S}{\sigma_S}
$$

Where:
- $\mu_S = \frac{1}{N}\sum_{i=t-N+1}^{t} S_i$ = rolling mean of spread
- $\sigma_S = \sqrt{\frac{1}{N-1}\sum_{i=t-N+1}^{t} (S_i - \mu_S)^2}$ = rolling std dev

**Trading Signals**:
- $z > +2$: SPY overvalued vs ES → **Bearish bias** (widen put wing or reduce size)
- $z < -2$: SPY undervalued vs ES → **Bullish bias** (widen call wing)
- $|z| \leq 2$: Neutral (standard Iron Condor)

**Implementation**:
```python
from analytics.divergence import DivergenceZScore

div = DivergenceZScore()
spread = div.compute_spread(spy_price=580.0, es_price=5800.0)
z = div.zscore(spread_series, lookback=50)

if z > 2.0:
    bias = "bearish"
elif z < -2.0:
    bias = "bullish"
```

---

### 3. IV Skew Analysis - `analytics/skew.py`

**Theory**: Put skew (higher IV for OTM puts) indicates crash risk. Steep skew suggests expensive downside protection.

#### Skew Metric
$$
\text{Skew} = \frac{IV_{put} - IV_{call}}{IV_{ATM}}
$$

Where:
- $IV_{put}$ = Implied volatility of OTM put (e.g., 15-delta)
- $IV_{call}$ = Implied volatility of OTM call (e.g., 15-delta)
- $IV_{ATM}$ = Implied volatility of ATM option

**Interpretation**:
- $\text{Skew} > 0.15$: **Steep put skew** → Crash risk elevated → Widen put wing or reduce size
- $\text{Skew} \approx 0$: Flat skew → Normal market
- $\text{Skew} < 0$: Call skew (rare) → Potential rally risk

**Risk Adjustment**:
$$
W_{put} = W_{base} + \Delta W \cdot \mathbb{1}_{\{\text{Skew} > \theta_{skew}\}}
$$

Where:
- $W_{put}$ = Put wing width
- $W_{base}$ = Base wing width (e.g., 5 strikes)
- $\Delta W$ = Increment (e.g., 5 strikes)
- $\theta_{skew}$ = Skew threshold (e.g., 0.15)

**Implementation**:
```python
from analytics.skew import SkewCalculator

skew_calc = SkewCalculator()
skew = skew_calc.skew_metric(iv_put=0.30, iv_call=0.20, iv_atm=0.25)

if skew_calc.is_steep_skew(skew, threshold=0.15):
    # Widen put wing or reduce position size
    put_wing_width += 5
```

---

### 4. Gap Analysis - `analytics/gaps.py`

**Theory**: Small overnight gaps (< 0.19%) tend to fill intraday due to mean-reversion. Large gaps indicate momentum continuation.

#### Gap Percentage
$$
G = \frac{|P_{open} - P_{prev\_close}|}{P_{prev\_close}}
$$

**Classification**:
- $G \leq 0.0019$ (0.19%): **Small gap** → High probability of fill → Enable legged entry
- $G > 0.0019$: **Large gap** → Momentum continuation → Standard entry or skip

**Gap Direction**:
$$
\text{Direction} = \begin{cases}
\text{Gap Up} & \text{if } P_{open} > P_{prev\_close} \cdot 1.0001 \\
\text{Gap Down} & \text{if } P_{open} < P_{prev\_close} \cdot 0.9999 \\
\text{No Gap} & \text{otherwise}
\end{cases}
$$

**Implementation**:
```python
from analytics.gaps import GapAnalyzer

gaps = GapAnalyzer()
gap_pct = gaps.gap_pct(open_price=580.5, prev_close=580.0)
is_small = gaps.is_small_gap(open_price=580.5, prev_close=580.0)

if is_small:
    # Enable legged entry (enter call/put spreads separately)
    allow_legged_entry = True
```

---

### 5. Cost-of-Carry Model - `analytics/carry_model.py`

**Theory**: Futures fair value is determined by the cost of carrying the underlying asset (interest minus dividends).

#### Fair Value Formula
$$
F = S \cdot e^{(r - q) \tau}
$$

Where:
- $F$ = Futures fair value
- $S$ = Spot price (SPY)
- $r$ = Risk-free rate (annualized, e.g., 0.05 = 5%)
- $q$ = Dividend yield (annualized, e.g., 0.015 = 1.5%)
- $\tau$ = Time to expiration (years)

#### Basis
$$
\text{Basis} = F_{market} - S
$$

**Interpretation**:
- $\text{Basis} > 0$: **Contango** (normal market)
- $\text{Basis} < 0$: **Backwardation** (stress/demand for spot)

**Implementation**:
```python
from analytics.carry_model import CostOfCarry

carry = CostOfCarry()
fair_value = carry.fair_value(spot=580.0, r=0.05, q=0.015, tau_years=0.25)
basis = carry.basis(futures_price=582.0, spot=580.0)

if basis < 0:
    # Backwardation - potential stress signal
    reduce_size = True
```

---

### 6. Technical Indicators (10-Factor Fuzzy System) - `intelligence/fuzzy_engine.py`

The fuzzy position sizing system uses 10 indicators, each with a membership function that maps to [0, 1]:

$$F_t = \sum_{j=1}^{10} w_j \cdot \mu_j$$

| # | Factor | Weight | Description |
|---|--------|--------|-------------|
| 1 | MTF Consensus | 0.18 | Multi-timeframe alignment |
| 2 | IV Rank | 0.14 | Implied volatility percentile |
| 3 | VIX Regime | 0.11 | Market fear index |
| 4 | RSI | 0.10 | Momentum oscillator |
| 5 | ADX | 0.10 | Trend strength |
| 6 | Bollinger Bands | 0.09 | Volatility regime |
| 7 | Stochastic | 0.08 | Overbought/Oversold |
| 8 | PSAR | 0.07 | Trend reversal |
| 9 | Volume | 0.07 | Liquidity confirmation |
| 10 | SMA Distance | 0.06 | Mean reversion |

---

#### 6.1 MTF Consensus (w = 0.18)

**Theory**: Multi-timeframe alignment measures agreement across 1m, 5m, and 15m timeframes. Neutral consensus favors Iron Condors.

$$
\mu_{MTF} = 1 - |C_{1/5/15} - 0.5| \times 2
$$

Where $C_{1/5/15} \in [0, 1]$ is the weighted consensus:
- $C = 0.5$: Perfect neutral → $\mu = 1.0$ (ideal for IC)
- $C = 0$ or $1$: Strong trend → $\mu = 0.0$ (avoid)

---

#### 6.2 IV Rank (w = 0.14)

**Theory**: High IV Rank means elevated implied volatility relative to history—favorable for selling premium.

$$
IVR = 100 \cdot \frac{IV_t - \min(IV_{window})}{\max(IV_{window}) - \min(IV_{window})}
$$

**Membership**:
$$
\mu_{IV} = \min\left(1.0, \frac{IVR}{60}\right)
$$

- $IVR \geq 60$: $\mu = 1.0$ (excellent for premium selling)
- $IVR = 30$: $\mu = 0.5$ (moderate)
- $IVR = 0$: $\mu = 0.0$ (avoid selling premium)

---

#### 6.3 VIX Regime (w = 0.11)

**Theory**: Low VIX indicates stable markets; high VIX signals fear and potential oversized moves.

**Membership**:
$$
\mu_{VIX} = \begin{cases}
1.0 & \text{if } VIX \leq 12 \\
1 - \frac{VIX - 12}{18} & \text{if } 12 < VIX < 30 \\
0.0 & \text{if } VIX \geq 30
\end{cases}
$$

- $VIX \leq 12$: $\mu = 1.0$ (calm market, ideal)
- $VIX = 20$: $\mu \approx 0.56$ (moderate caution)
- $VIX \geq 30$: $\mu = 0.0$ (high risk, skip)

---

#### 6.4 RSI (w = 0.10) - Wilder Smoothing

$$
RS = \frac{EMA_{\alpha}(\text{Gain})}{EMA_{\alpha}(\text{Loss})}
$$

$$
RSI = 100 - \frac{100}{1 + RS}
$$

Where $\alpha = \frac{1}{period}$ (default $period = 14$)

**Membership** (neutral zone 40-60 is optimal):
$$
\mu_{RSI} = \begin{cases}
1.0 & \text{if } 40 \leq RSI \leq 60 \\
\frac{RSI}{40} & \text{if } RSI < 40 \\
\frac{100 - RSI}{40} & \text{if } RSI > 60
\end{cases}
$$

- $RSI \in [40, 60]$: $\mu = 1.0$ (neutral momentum, ideal)
- $RSI < 30$ or $RSI > 70$: $\mu \to 0$ (extreme, avoid)

---

#### 6.5 ADX (w = 0.10) - Wilder Smoothing

$$
+DI = 100 \cdot \frac{EMA_{\alpha}(+DM)}{ATR}
$$

$$
-DI = 100 \cdot \frac{EMA_{\alpha}(-DM)}{ATR}
$$

$$
DX = 100 \cdot \frac{|+DI - (-DI)|}{+DI + (-DI)}
$$

$$
ADX = EMA_{\alpha}(DX)
$$

**Membership** (low ADX = weak trend = favorable):
$$
\mu_{ADX} = \begin{cases}
1.0 & \text{if } ADX \leq 25 \\
1 - \frac{ADX - 25}{15} & \text{if } 25 < ADX < 40 \\
0.0 & \text{if } ADX \geq 40
\end{cases}
$$

- $ADX < 25$: $\mu = 1.0$ (ranging market, ideal)
- $ADX > 40$: $\mu = 0.0$ (strong trend, avoid)

---

#### 6.6 Bollinger Bands (w = 0.09)

$$
\text{Upper} = SMA_{20} + 2 \cdot \sigma_{20}
$$

$$
\text{Lower} = SMA_{20} - 2 \cdot \sigma_{20}
$$

$$
BB_{position} = \frac{Price - Lower}{Upper - Lower}
$$

$$
BB_{width} = \frac{Upper - Lower}{SMA_{20}}
$$

**Membership** (middle of bands = ideal):
$$
\mu_{BB} = 0.7 \times (1 - |BB_{position} - 0.5| \times 2) + 0.3 \times \max(0, 1 - \frac{BB_{width}}{0.04})
$$

- $BB_{position} = 0.5$: $\mu \to 1.0$ (price at center)
- $BB_{position} < 0.05$ or $> 0.95$: $\mu \to 0$ (touching bands, avoid)

---

#### 6.7 Stochastic Oscillator (w = 0.08)

$$
\%K = 100 \cdot \frac{Close - Low_{14}}{High_{14} - Low_{14}}
$$

$$
\%D = SMA_3(\%K)
$$

**Membership** (neutral zone 30-70 is optimal):
$$
\mu_{Stoch} = \begin{cases}
1.0 & \text{if } 30 \leq \%K \leq 70 \\
\frac{\%K}{30} & \text{if } \%K < 30 \\
\frac{100 - \%K}{30} & \text{if } \%K > 70
\end{cases}
$$

- $\%K \in [30, 70]$: $\mu = 1.0$ (neutral, ideal)
- $\%K < 20$ or $> 80$: $\mu \to 0$ (extreme, avoid)

---

#### 6.8 Parabolic SAR (w = 0.07)

$$
SAR_{t+1} = SAR_t + AF \cdot (EP - SAR_t)
$$

Where:
- $AF$ = Acceleration Factor (0.02 → 0.20)
- $EP$ = Extreme Point (highest high / lowest low)

**Membership** (crossover = ideal):
$$
\mu_{PSAR} = 1 - |P_{position}|
$$

Where $P_{position} \in [-1, +1]$:
- $P_{position} = 0$: PSAR crossover → $\mu = 1.0$ (ideal for IC)
- $|P_{position}| = 1$: Strong trend → $\mu = 0.0$ (avoid)

---

#### 6.9 Volume Ratio (w = 0.07)

$$
V_{ratio} = \frac{Volume_t}{SMA_{20}(Volume)}
$$

**Membership** (adequate liquidity required):
$$
\mu_{Vol} = \min\left(1.0, \frac{V_{ratio}}{0.8}\right)
$$

- $V_{ratio} \geq 0.8$: $\mu = 1.0$ (adequate liquidity)
- $V_{ratio} < 0.4$: $\mu < 0.5$ (poor fills likely)

---

#### 6.10 SMA Distance (w = 0.06)

$$
D_{SMA} = \frac{Price - SMA_{20}}{SMA_{20}}
$$

**Membership** (near equilibrium = ideal):
$$
\mu_{SMA} = \begin{cases}
1 - \frac{|D_{SMA}|}{0.02} & \text{if } |D_{SMA}| \leq 0.02 \\
0.0 & \text{if } |D_{SMA}| > 0.02
\end{cases}
$$

- $|D_{SMA}| = 0$: $\mu = 1.0$ (at equilibrium)
- $|D_{SMA}| > 2\%$: $\mu = 0.0$ (extended, mean reversion risk)

---

**Implementation**:
```python
from intelligence.fuzzy_engine import (
    calculate_mtf_membership,
    calculate_iv_membership,
    calculate_regime_membership,
    calculate_rsi_membership,
    calculate_adx_membership,
    calculate_bbands_membership,
    calculate_stoch_membership,
    calculate_psar_membership,
    calculate_volume_membership,
    calculate_sma_distance_membership,
    compute_fuzzy_confidence
)

# All memberships in [0, 1]
memberships = {
    "mtf": 0.60, "iv": 0.80, "regime": 0.70,
    "rsi": 0.50, "adx": 0.85, "bbands": 0.65,
    "stoch": 0.55, "psar": 0.90, "volume": 0.75, "sma": 0.80
}

weights = {
    "mtf": 0.18, "iv": 0.14, "regime": 0.11, "rsi": 0.10,
    "adx": 0.10, "bbands": 0.09, "stoch": 0.08, "psar": 0.07,
    "volume": 0.07, "sma": 0.06
}

Ft = compute_fuzzy_confidence(memberships, weights)
print(f"Fuzzy Confidence: {Ft:.2f}")  # → 0.71
```

---

## 📊 Iron Condor Strategy Mathematics

### Structure
An Iron Condor consists of 4 legs:

1. **Short Call** ($C_s$): Sell OTM call (e.g., 15-delta)
2. **Long Call** ($C_l$): Buy further OTM call (protection)
3. **Short Put** ($P_s$): Sell OTM put (e.g., 15-delta)
4. **Long Put** ($P_l$): Buy further OTM put (protection)

### Credit Received
$$
Credit = (C_s - C_l) + (P_s - P_l)
$$

### Maximum Loss
$$
MaxLoss = W - Credit
$$

Where $W$ = wing width (e.g., $W = K_{C_l} - K_{C_s} = K_{P_s} - K_{P_l}$)

### Mark-to-Market P&L
$$
PnL_t = (Credit - Cost_t) \times Q \times 100
$$

Where:
- $Cost_t = (C_{s,t} - C_{l,t}) + (P_{s,t} - P_{l,t})$ = current replacement cost
- $Q$ = number of contracts
- $100$ = option multiplier

### Exit Conditions

#### Profit Take
$$
PnL_t \geq Credit \cdot \alpha_{PT}
$$

Example: $\alpha_{PT} = 0.50$ (50% of max profit)

#### Stop Loss
$$
PnL_t \leq -Credit \cdot \alpha_{SL}
$$

Example: $\alpha_{SL} = 2.00$ (200% of credit = max loss)

#### DTE Exit
$$
DTE_t \leq DTE_{exit}
$$

Example: $DTE_{exit} = 21$ days

---

## 🛡️ Risk Management

### Portfolio Greeks Tracking

#### Delta
$$
\Delta_{portfolio} = \sum_{i=1}^{N} \Delta_{trade_i} \cdot Q_i
$$

**Limit**: $|\Delta_{portfolio}| \leq \Delta_{max}$ (e.g., $\Delta_{max} = 200$)

#### Gamma
$$
\Gamma_{portfolio} = \sum_{i=1}^{N} \Gamma_{trade_i} \cdot Q_i
$$

**Limit**: $|\Gamma_{portfolio}| \leq \Gamma_{max}$ (e.g., $\Gamma_{max} = 50$)

#### Vega
$$
\mathcal{V}_{portfolio} = \sum_{i=1}^{N} \mathcal{V}_{trade_i} \cdot Q_i
$$

**Limit**: $|\mathcal{V}_{portfolio}| \leq \mathcal{V}_{max}$ (e.g., $\mathcal{V}_{max} = 500$)

### Drawdown Cap
$$
DD_t = \frac{Equity_t - Equity_{peak}}{Equity_{peak}}
$$

**Limit**: $DD_t \geq DD_{max}$ (e.g., $DD_{max} = -0.02$ or -2%)

### Risk Budget Sizing
$$
Q_{risk} = \left\lfloor \frac{RiskBudget}{MaxLoss_{contract}} \right\rfloor
$$

$$
Q_{final} = \min(Q_{fuzzy}, Q_{risk})
$$

---

## 🧪 Testing Infrastructure

### Unit Tests (Phase 1)
```bash
pytest -q tests/
# 18 passed in 0.78s
```

**Test Coverage**:
- `test_realized_vol.py`: RV calculation, VRP gate
- `test_divergence.py`: Z-score bounds
- `test_skew.py`: Skew penalty triggers
- `test_gap.py`: Small gap threshold
- `test_fis_sizer.py`: FIS monotonicity (stub)
- `test_risk_es.py`: Expected Shortfall (stub)
- `test_beta_weighting.py`: Beta-weighted delta (stub)
- `test_structure_validator.py`: Quantity clamping (stub)
- `test_engine_lifecycle.py`: Smoke test

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/SPYOptionTrader.git
cd SPYOptionTrader
pip install -r requirements.txt
```

### Run Backtest
```bash
py core/main.py --mode backtest --bt-start 2025-07-01 --bt-end 2025-12-31
```

### CLI Options Reference

| Category | Option | Default | Description |
|----------|--------|---------|-------------|
| **Mode** | `--mode` | `backtest` | `backtest` or `live` |
| **Backtest** | `--bt-start` | `2024-01-01` | Start date (YYYY-MM-DD) |
| | `--bt-end` | `2024-12-31` | End date (YYYY-MM-DD) |
| | `--bt-cash` | `25000.0` | Starting capital |
| | `--options-data` | None | Path to options CSV with Greeks |
| | `--no-plot` | False | Disable equity curve plot |
| **Strategy** | `--underlying` | `SPY` | Symbol to trade |
| | `--quantity` | `1` | Contracts per trade |
| | `--dte-min` | `30` | Minimum days to expiration |
| | `--dte-max` | `45` | Maximum days to expiration |
| **Delta** | `--delta-low` | `0.10` | Lower delta bound |
| | `--delta-high` | `0.25` | Upper delta bound |
| **Wings** | `--wing-min` | `5.0` | Minimum wing width ($) |
| | `--wing-max` | `10.0` | Maximum wing width ($) |
| | `--min-credit-ratio` | `0.15` | Minimum credit/width ratio |
| **Filters** | `--ivr-min` | `0.0` | Minimum IV Rank |
| | `--vix-max` | `25.0` | Maximum VIX for entry |
| **Exits** | `--profit-pct` | `0.50` | Profit take (% of credit) |
| | `--loss-multiple` | `1.5` | Stop loss (multiple of credit) |
| | `--max-hold-days` | `14` | Maximum holding period |
| **Positions** | `--max-positions` | `3` | Max concurrent positions |
| | `--max-alloc` | `0.15` | Max portfolio allocation |
| **MTF** | `--use-mtf` | False | Enable MTF consensus filter |
| | `--no-mtf-filter` | False | Disable MTF filter |
| | `--mtf-timeframes` | `1,5,15` | Timeframes (comma-separated) |
| | `--mtf-consensus-min` | `0.40` | Min consensus score |
| | `--mtf-consensus-max` | `0.60` | Max consensus score |
| **Sizing** | `--dynamic-sizing` | False | Enable dynamic position sizing |
| | `--position-size-pct` | `0.05` | Position size (% of equity) |
| **Hedging** | `--enable-hedge` | False | Enable delta hedging |
| | `--hedge-threshold` | `0.10` | Delta threshold for hedge |
| **Other** | `--use-optimizer` | False | Run parameter optimizer |
| | `--alpaca` | False | Use Alpaca broker |
| | `--no-liquidity-gate` | False | Disable liquidity checks |

### Run Optimizer
```bash
py -3.12 core/optimizer.py --samples 100
```

### Run Tests
```bash
pytest -q tests/
```

---

## 🔄 Production Data Pipeline

### Overview
The system includes a fully automated data pipeline that generates high-density options data for backtesting.

### Pipeline Scripts
| Script | Purpose |
|--------|---------|
| `scripts/run_production_pipeline.py` | **Master orchestrator** - Runs all steps sequentially |
| `data_factory/download_ivolatility_options.py` | Downloads Greeks from IVolatility (Top 100 ATM/day) |
| `data_factory/download_alpaca_matched.py` | Fetches M1 bars from Alpaca |
| `data_factory/merge_intraday_with_greeks.py` | Merges and resamples to M1/M5/M15 |

### Run Full Pipeline
```bash
py scripts/run_production_pipeline.py
# Runtime: ~3-4 hours (API rate limits)
```

### Data Directory Structure
```
data/
├── ivolatility/
│   └── spy_options_ivol_large.csv      # IVolatility Greeks (pipeline output)
├── alpaca_options/
│   ├── spy_options_bars.csv            # Historical bars
│   ├── spy_options_bars_with_greeks.csv # Bars + Greeks (legacy)
│   └── spy_options_intraday_large_with_greeks_m1.csv  # MTF M1 data (pipeline)
└── synthetic_options/
    └── spy_options_marks.csv           # 2.2GB synthetic data
```

---

## 🧪 Backtest Scripts

### MTF Backtest (Multi-Timeframe)
Uses M1/M5/M15 intraday data with multi-timeframe consensus filtering:
```bash
py scripts/run_backtest_mtf.py
```
- **Data**: `spy_options_intraday_large_with_greeks_m1.csv`
- **MTF Filter**: Enabled (`use_mtf=True`)
- **Timeframes**: 1m, 5m, 15m consensus

### Standard Backtest (Non-MTF)
Uses synthetic options data without MTF filtering:
```bash
py scripts/run_backtest_standard.py
```
- **Data**: `spy_options_marks.csv` (2.2GB)
- **MTF Filter**: Disabled (`use_mtf=False`)
- **Mode**: EOD-style processing

### Custom Backtest
```bash
py scripts/run_full_backtest.py
```

---

## 📈 Performance Metrics

### Key Metrics Tracked
- **Net Profit**: Total realized P&L
- **Max Drawdown**: Peak-to-trough equity decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Expectancy Ratio**: Average win / Average loss

### Optimizer Objective
$$
\text{Fitness} = \frac{NetProfit}{|MaxDrawdown|}
$$

---

## 📚 Documentation

- **[Task List](docs/task.md)**: Detailed task breakdown with Phase 2 & 2.5 completion
- **[Phase 2 & 2.5 Walkthrough](docs/walkthrough.md)**: Implementation summary with architecture diagrams
- **[Implementation Plan](docs/implementation_plan.md)**: Complete architectural roadmap
- **[Architectural Refactoring Plan](docs/architectural_refactoring_plan.md)**: Migration strategy

---

## 🤝 Contributing

This project follows the **Quantor-MTFuzz Architectural Specification** with strict module separation and comprehensive testing.

### Development Workflow
1. Create feature branch
2. Write unit tests first (TDD)
3. Implement feature
4. Run full test suite: `pytest tests/`
5. Update documentation
6. Submit PR

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Built on:
- **Backtrader**: Python backtesting framework
- **Alpaca-Py**: Live trading SDK
- **Mamba**: State-space neural architecture
- **scikit-fuzzy**: Fuzzy logic toolkit

---

**Commit**: `3dd6ff4` (Phase 1: DTOs, Analytics, and Test Infrastructure)  
**Tests**: 18 passing  
**Status**: Production-ready for backtesting, paper trading ready
