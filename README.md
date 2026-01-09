# Quantor-MTFuzz: Advanced Options Trading System
## Iron Condor Algorithmic Trading with Neuro-Fuzzy Intelligence

**Quantor-MTFuzz** is a production-grade algorithmic trading system for SPY Iron Condor strategies, combining Multi-Timeframe (MTF) technical intelligence, Fuzzy Logic position sizing, and institutional-grade risk management. Built on the Quantor-MTFuzz architectural specification with clean module separation, comprehensive testing, and mathematical rigor.

---

## 🚀 Key Features

### Core Capabilities
- **High-Fidelity Backtesting**: 5-minute bar simulation with accurate mark-to-market P&L, leg-by-leg exit logic, and realistic slippage/commissions
- **Phased Serial Optimization**: Grid-search engine optimizing for **Net Profit / Max Drawdown** ratio with hardware benchmarking
- **9-Factor Fuzzy Intelligence**: Dynamic position sizing based on MTF Consensus, IV Rank, VIX Regime, RSI, ADX, Bollinger Bands, Stochastic, Volume, and SMA Distance
- **Mamba 2 Neural Forecasting**: State-space model predicting market regimes (Bear/Neutral/Bull) and volatility states
- **Enhanced Risk Controls**: Portfolio Greeks tracking, drawdown caps, and beta-weighted delta limits
- **Alpaca Integration**: Seamless live paper trading via Alpaca-Py SDK
- **Professional Reporting**: Automated PDF reports with equity curves, strike overlays, P&L distributions

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

### 6. Technical Indicators - `analytics/indicators.py`

#### RSI (Wilder Smoothing)
$$
RS = \frac{EMA_{\alpha}(\text{Gain})}{EMA_{\alpha}(\text{Loss})}
$$

$$
RSI = 100 - \frac{100}{1 + RS}
$$

Where $\alpha = \frac{1}{period}$ (e.g., $period = 14$)

#### ADX (Wilder Smoothing)
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

**Interpretation**:
- $ADX < 20$: Ranging market (favorable for Iron Condors)
- $ADX > 25$: Trending market (reduce size or skip)

#### IV Rank
$$
IVR = 100 \cdot \frac{IV_t - \min(IV_{window})}{\max(IV_{window}) - \min(IV_{window})}
$$

**Implementation**:
```python
from analytics.indicators import IndicatorPack

indicators = IndicatorPack.from_inputs(
    bars=df_5m,
    iv_atm_series=iv_series,
    adx_period=14,
    rsi_period=14,
    iv_rank_window=78*60  # ~60 trading days
)

print(f"ADX: {indicators.adx:.2f}")
print(f"RSI: {indicators.rsi:.2f}")
print(f"IV Rank: {indicators.iv_rank:.1f}")
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
py -3.12 core/main.py --mode backtest --bt-start 2025-07-01 --bt-end 2025-12-31
```

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
