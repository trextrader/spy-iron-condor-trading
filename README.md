# Quantor-MTFuzz: Iron Condor Algorithmic Trading

**Quantor-MTFuzz** is a high-fidelity algorithmic trading system designed specifically for SPY Iron Condor strategies. It combines Multi-Timeframe (MTF) technical intelligence with Fuzzy Logic position sizing to execute risk-managed options trades in both backtesting and live paper-trading environments.

---

## 🚀 Key Features

- **High-Fidelity Backtesting**: Simulates every 5-minute bar with accurate mark-to-market P&L, leg-by-leg exit logic (Profit Take, Stop Loss, Expiration), and price caching to handle data gaps.
- **Phased Serial Optimization**: A custom grid-search engine that optimizes for the **Net Profit / Max Drawdown** ratio. Includes automatic hardware benchmarking and time-to-completion estimation.
- **9-Factor Fuzzy Intelligence**: Dynamic position sizing based on MTF Consensus, IV Rank, VIX Regime, RSI, ADX, Bollinger Bands, Stochastic, Volume, and SMA Distance.
- **Mamba 2 Neural Forecasting**: Integrated state-space model that predicts market state probabilities (Bear/Neutral/Bull) and volatility regimes to bias trade sizing.
- **Enhanced Risk Controls**: MTF trend filters, ATR-based dynamic stops, and trailing stop exits.
- **Alpaca Integration**: Seamless transition to live paper trading using the Alpaca-Py SDK.
- **Professional Reporting**: Generates automated PDF reports with equity curves, strike overlays, P&L distributions, and monthly performance grids.

---

## 🏗️ Engineering Architecture: The Dual-Data Engine

Quantor-MTFuzz operates as a high-fidelity **Dual-Data Engine**, separating the "Strategy Clock" from the "Option Pricing":

### 1. The Strategy Clock & Signal Layer (`reports/SPY/`)
The system uses the underlying SPY 5-minute price data as its primary heart. This data determines:
- **Consensus Timing**: Moving averages, RSI, and MTF (Multi-Timeframe) intelligence are calculated across 5m, 15m, and 60m timeframes to decide if a trade environment is "Favorable."
- **Spot Reference**: The current stock price is used as the anchor to look up deltas and strikes for the options legs.

### 2. The Quote & Execution Layer (`data/synthetic_options/`)
While the SPY price drives the decision, the **Synthetic Options** data provides the actual market reality:
- **Replacement Cost**: The bot uses these files to determine exactly what your Iron Condor is worth at any given moment.
- **Realistic PnL**: Mark-to-Market (MtM) calculations are based on the mid-prices in the options data, ensuring you are testing against dollar-accurate execution.

---

## 🦅 Strategy Execution: The 4-Leg Standard

This system is built exclusively for the **Iron Condor** structure. Every trade launched by the engine—whether in backtest or optimization—follows a strict **4-Leg Symmetry Rule**:

1.  **Call Wing**: One Short Call (e.g., 15 Delta) and one Long Call (as protection).
2.  **Put Wing**: One Short Put (e.g., 15 Delta) and one Long Put (as protection).
3.  **Strict Cohesion**: The strategy engine is programmed with a "Fail-Fast" entry gate. If even one of these 4 legs cannot be found with valid pricing or at the target delta in the options chain, **the entire trade is rejected**.
4.  **Credit-Based Management**: PnL is tracked as a single unit (Net Credit vs. Debit Cost), but individual leg exits (PT/SL) can be triggered if chosen.

By requiring all 4 legs to be present, the backtester ensures that you are only analyzing "True" Iron Condors that could actually be filled in the market.

### Position Sizing Guardrails

Iron Condors have a structural requirement: they need **at least 2 contracts** (one per wing). The system enforces this via multiple safety layers:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Config Minimum** | `min_total_qty_for_iron_condor: 2` | Configurable floor for Iron Condor sizing |
| **Fallback Floor** | `fallback_total_qty: 2` | Minimum when risk-based sizing produces 0 |
| **Scaling Floor** | `max(min_floor, scaled_qty)` | Prevents fuzzy scaling from reducing below minimum |
| **Assertion Guard** | `assert total_qty >= min_floor` | Runtime safety net to catch logic bugs |
| **Regression Tests** | `tests/test_iron_condor_sizing.py` | 6 tests to prevent the 1-lot bug from returning |

**Run Tests:**
```bash
py -3.12 -m pytest tests/test_iron_condor_sizing.py -v
```

---

## 🧮 Mathematical Foundation (Script-by-Script)

### 1. `core/backtest_engine.py`: high-Fidelity MtM PnL
The engine tracks the real-time replacement cost of the Iron Condor spread at every 5-minute bar. The floating PnL at time $t$ is calculated as:
$$PnL_{t} = (Credit_{0} - Cost_{t}) \times Q \times 100$$
Where $Cost_{t}$ is the net cost to "buy back" the spread:
$$Cost_{t} = (C_{short} - C_{long}) + (P_{short} - P_{long})$$

### 2. `strategies/options_strategy.py`: Exit & Entry Math
Optimizable parameters define the boundaries for trade management:
- **Profit Take (`profit_take_pct`)**: Exit triggered if $PnL_{t} \ge Credit_{0} \cdot \alpha_{pt}$
- **Stop Loss (`loss_close_multiple`)**: Exit triggered if $PnL_{t} \le -Credit_{0} \cdot \alpha_{loss}$
- **DTE Window**: Entry valid only if $T_{exp} - T_{now} \in [DTE_{min}, DTE_{max}]$
- **Wing Width**: The structural spread $W$ must satisfy $W \ge Wing_{min}$.

### 3. `core/trade_decision.py`: Probability-Based Strike Selection
Strikes are selected by minimizing the distance between target deltas ($\delta_{target}$) and market deltas ($\delta_{market}$). This is a proxy for matching the strategy's probability of success:
$$Strike = \text{argmin}(|\delta_{market} - \delta_{target}|)$$

### 4. `intelligence/regime_filter.py`: Volatility Governance
Filters ensure the strategy only operates in favorable risk-adjusted environments:
- **IV Rank (IVR)**: Normalizes current IV against the annual range:
  $$IVR = 100 \times \frac{IV_{current} - IV_{min\_year}}{IV_{max\_year} - IV_{min\_year}} \dots \text{Filter: } IVR \ge IVR_{min}$$
- **VIX Filter**: A hard threshold to avoid extreme tail-risk events:
  $$\text{Filter: } VIX_{current} \le VIX_{max}$$

### 5. `intelligence/fuzzy_engine.py`: Weighted Intelligence Aggregation
Position sizing scales dynamically using a Sugeno-style weighted average to aggregate membership functions ($\mu$):
$$Signal = \frac{\sum_{i=1}^{n} w_i \cdot \mu_i(x)}{\sum_{i=1}^{n} \mu_i(x)}$$

### 6. `intelligence/mamba_engine.py`: Neural State-Space Modeling (Mamba 2)

The system forecasts market regime using a **Selective State-Space Model (SSM)**, which is the mathematical foundation of the Mamba 2 architecture:

#### Core State-Space Equations

The continuous-time state-space model is defined as:

$$\frac{dh(t)}{dt} = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

For discrete-time implementation (our 5-minute bars), this is approximated via the **Zero-Order Hold (ZOH) discretization:**

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

Where:
- $h_t \in \mathbb{R}^{d_{model} \times d_{state}}$ is the hidden state (memory)
- $x_t \in \mathbb{R}^{d_{model}}$ is the input feature vector
- $y_t \in \mathbb{R}^3$ is the output probability distribution $[P_{bear}, P_{neutral}, P_{bull}]$
- $\bar{A}, \bar{B}$ are the discretized state transition matrices

#### The Selective Scan Mechanism

Unlike traditional RNNs, Mamba 2 uses a **selective scan** where the matrices $B$ and $C$ are input-dependent:

$$B_t = \text{Linear}_{B}(x_t)$$
$$C_t = \text{Linear}_{C}(x_t)$$

This allows the model to selectively "remember" or "forget" based on input content—critical for filtering noise in volatile markets.

#### Input Feature Vector (9 Factors)

The input $x_t$ is constructed from normalized technical indicators:

$$x_t = \begin{bmatrix} 
\frac{P_t - P_{t-1}}{P_{t-1}} \\
\frac{RSI_{14} - 50}{50} \\
ATR\% \times 10 \\
VolumeRatio - 1 \\
\vdots 
\end{bmatrix}$$

#### Output Interpretation

The raw output scalar $y_{raw} \in [-1, 1]$ (via $\tanh$) is converted to probabilities:

| $y_{raw}$ Range | Market State | Probability Distribution |
|-----------------|--------------|-------------------------|
| $y > 0.2$ | Bullish | $P_{bull} = 0.6 + 0.2y$, $P_{neutral} = 0.3$, $P_{bear} = 0.1$ |
| $y < -0.2$ | Bearish | $P_{bear} = 0.6 + 0.2\|y\|$, $P_{neutral} = 0.3$, $P_{bull} = 0.1$ |
| $-0.2 \le y \le 0.2$ | Neutral | $P_{neutral} = 0.8$, $P_{bull} = 0.1$, $P_{bear} = 0.1$ |

#### Confidence Calculation

Model confidence scales with signal strength:
$$Confidence = 0.5 + \frac{|y_{raw}|}{2}$$

This produces confidence values from 0.5 (minimum, when $y \approx 0$) to 1.0 (maximum, when $|y| = 1$).

#### Integration with Fuzzy System

The neural forecast is fused with the 9-factor fuzzy system via weighted aggregation:

$$G_{fused} = 0.60 \times G_{gaussian} + 0.40 \times F_t + w_{neural} \times Confidence_{mamba}$$

## Market Realism (Stage 1)

The backtest engine now includes professional-grade market mechanics to prevent "simulation bias":

### 1. Realized Volatility & IV Rank
Instead of relying on random placeholders, the system calculates realized volatility dynamically from 5-minute bars:
- **Realized Vol**: Annualized standard deviation of returns over a 20-period lookback.
- **IV Rank Proxy**: Rolling percentile rank of current volatility vs. trailing 252-bar history.

### 2. Transaction Cost Model
To align with live execution, the backtest enforces:
- **Slippage**: default `$0.02` per contract applied to *each leg* on both entry and exit.
- **Commission**: default `$0.65` per contract deducted from P&L.
- **Net Reporting**: Logs show both Gross P&L and Net P&L (after costs).

Where $G_{gaussian}$ is the weighted sum of fuzzy memberships and $F_t$ is the fuzzy confidence score.

### 7. `core/optimizer.py`: Risk-Adjusted Optimization Ratio
The primary objective is to maximize the **$\Phi$ Recovery Ratio**, prioritizing capital preservation:
$$\Phi = \frac{\text{Net Profit}}{\text{Maximum Drawdown}}$$

### 8. Performance Metrics Glossary
The following metrics are used to rank results in the `top100_[timestamp].csv` report:

- **Profit Factor ($PF$)**: Measures the relationship between gross winnings and gross losses.
  $$PF = \frac{\sum \text{Gross Profits}}{\sum |\text{Gross Losses}|}$$
- **Expectancy ($E$)**: The expected dollar value of each trade, including losses.
  $$E = (AvgWin \times WR) - (AvgLoss \times (1 - WR))$$
- **Win Rate ($WR$)**: The ratio of profitable trades to total trades.
  $$WR = \frac{\text{Winning Trades}}{\text{Total Trades}}$$
- **Sharpe Ratio ($S$)**: Annualized risk-adjusted returns calculated from 5-minute equity returns.
  $$S = \sqrt{N} \times \frac{\mu_{returns}}{\sigma_{returns}}$$
  *(Where $N = 19,656$ for annual 5-minute bars in a 252-day trading year).*

---

## 📊 Benchmarking

Run the hardware benchmark to measure system performance and project optimizer runtimes:

```bash
py -3.12 core/benchmark_cpu.py
```

**Outputs:**
- `reports/benchmark.json` — Structured metrics for CI/baseline tracking
- Console output with bars/second and projected optimizer time

**JSON Format:**
```json
{
  "timestamp": "2026-01-04T15:00:00",
  "test_bars": 5000,
  "runtime_seconds": 0.8,
  "bars_per_second": 6250,
  "projected_100_runs_minutes": 1.33,
  "system": {"cpu_physical_cores": 8, "ram_total_gb": 32, "python_version": "3.12.7"}
}
```

**Baseline Metrics:**
After each backtest, key metrics are saved to `reports/baseline_metrics.json` for regression tracking.

---

## 🛠️ Installation

1. **Setup Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

1. **API Keys**: Copy `core/config.template.py` to `core/config.py` and enter your keys.
2. **Parameters**: The winning parameters from our initial optimization (**IVR 20, VIX 25, TP 60%, SL 2.0x**) are pre-set as the defaults in the configuration files.
3. **Latest Optimizer Selection**: Rank-1 applied to defaults: `profit_take_pct=0.9`, `loss_close_multiple=1.2`.

---

## 📈 Execution Modes

### 1. High-Fidelity Backtest
```bash
python core/main.py --mode backtest --use-mtf --dynamic-sizing --bt-samples 0
```
- Results are saved to `reports/backtest_report.pdf`.

### 2. Strategy Optimization
Find the optimal parameters for your risk profile:
```powershell
python core/main.py --mode backtest --use-mtf --dynamic-sizing --bt-samples 0 --use-optimizer
```
- Performance is measured by the **Net Profit / Max Drawdown** ratio.
- The script will benchmark your hardware first and provide a time estimate.
- Once finished, you can select the best configuration from the Top 100 Leaderboard.

#### **Optimizer Reporting (New)**
Every optimization run generates a persistent, timestamped CSV report stored in the `reports/` directory:
- **Location**: `reports/top100_YYYYMMDD_HHMMSS.csv`
- **Sorting Logic**: The results are strictly ordered by the **Net Profit / Max Drawdown Ratio** (Rank 1 is the best risk-adjusted performance).
- **Report Fields**:
  - `Rank`: Leaderboard position (1-100).
  - `NetProfit`: Total dollar profit over the period.
  - `MaxDD`: The largest peak-to-valley account dip (in dollars).
  - `NP_DD_Ratio`: The primary recovery/risk metric.
  - `ProfitFactor`: Gross Wins divided by Gross Losses.
  - `Expectancy`: The average dollar value expected per trade (E-Ratio).
  - `Sharpe`: Annualized risk-adjusted return.
  - `Wins / Losses`: Individual counts of winning vs losing trades.
  - `WinRate`: Percentage of successful trades.
  - `[Strategy Params]`: All optimized inputs (e.g., `iv_rank_min`, `profit_take_pct`, etc.) are included as columns for direct analysis.

### 3. Live Paper Trading (Alpaca)
Transition your strategy to the real market:
```powershell
python core/main.py --mode live --alpaca --alpaca-key YOUR_KEY --alpaca-secret YOUR_SECRET --polygon-key YOUR_KEY
```

---

## ⌨️ Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | string | `backtest` | Execution mode: `live` or `backtest` |
| `--polygon-key` | string | `None` | Polygon.io API key for market data |
| `--alpaca-key` | string | `None` | Alpaca API Key ID |
| `--alpaca-secret` | string | `None` | Alpaca API Secret Key |
| `--underlying` | string | `SPY` | Underlying ticker symbol |
| `--quantity` | int | `1` | Base number of contracts per trade |
| `--dte-min` | int | `30` | Minimum Days to Expiration for entry |
| `--dte-max` | int | `45` | Maximum Days to Expiration for entry |
| `--delta-low` | float | `0.15` | Target short delta (lower bound) |
| `--delta-high` | float | `0.20` | Target short delta (upper bound) |
| `--wing-min` | float | `5.0` | Minimum wing width (Iron Condor spread) |
| `--wing-max` | float | `10.0` | Maximum wing width |
| `--min-credit-ratio`| float | `0.25` | Min credit/width ratio (e.g. 0.25 = 25% of width) |
| `--ivr-min` | float | `30.0` | Minimum IV Rank threshold for entry |
| `--vix-max` | float | `25.0` | Maximum VIX threshold for entry |
| `--profit-pct` | float | `0.50` | Profit take percentage (e.g., 0.50 = 50%) |
| `--loss-multiple` | float | `1.5` | Stop loss multiple (e.g., 1.5 = 150% of credit) |
| `--max-hold-days` | int | `14` | Max days to hold a position |
| `--max-positions` | int | `3` | maximum concurrent positions |
| `--max-alloc` | float | `0.15` | Maximum portfolio allocation per trade |
| `--regime-ivr-widen`| float | `40.0` | IVR threshold to trigger wing widening |
| `--regime-vix-widen`| float | `22.0` | VIX threshold to trigger wing widening |
| `--width-increment` | float | `5.0` | Points to add when widening wings |
| `--enable-hedge` | flag | `False` | Enable delta hedging (shares) |
| `--hedge-threshold`| float | `0.10` | Portfolio delta threshold to trigger hedge |
| `--use-mtf` | flag | `False` | Enable Multi-Timeframe consensus filters |
| `--mtf-consensus-min`| float | `0.40` | Min consensus score for long-side signal |
| `--mtf-consensus-max`| float | `0.60` | Max consensus score for short-side signal |
| `--mtf-timeframes` | string | `1,5,15` | Timeframes to sync (comma-separated) |
| `--no-liquidity-gate`| flag | `False` | Disable bid/ask spread checks |
| `--bt-cash` | float | `25000.0` | Starting cash for backtesting |
| `--bt-start` | string | `2024-01-01` | Backtest start date (YYYY-MM-DD) |
| `--bt-end` | string | `2024-12-31` | Backtest end date (YYYY-MM-DD) |
| `--bt-samples` | int | `500` | Bars to sample (Set `0` for full year) |
| `--no-plot` | flag | `False` | Disable chart/PDF generation |
| `--dynamic-sizing` | flag | `False` | Enable Fuzzy Logic dynamic position sizing |
| `--position-size-pct`| float | `0.05` | Target allocation % for fuzzy sizing |
| `--alpaca` | flag | `False` | Use Alpaca Broker for paper/live trading |
| `--use-optimizer` | flag | `False` | Run the phased serial grid search optimizer |

---

## 🧪 Optimizable Parameters (`core/optimizer.py`)

To modify the search space, edit the `OPTIMIZATION_MATRIX` in `core/optimizer.py`.

| Parameter | Type | Inclusive Range Format | Description |
| :--- | :--- | :--- | :--- |
| `profit_take_pct` | Float | `np.arange(start, stop + step, step)` | Target profit % (e.g., 0.1 to 1.0) |
| `loss_close_multiple` | Float | `np.arange(start, stop + step, step)` | Stop loss multiple (e.g., 1.0 to 5.0) |
| `dte_min` | Int | `range(start, stop + step, step)` | Minimum days to expiration |
| `dte_max` | Int | `range(start, stop + step, step)` | Maximum days to expiration |
| `target_short_delta_low` | Float | `np.arange(start, stop + step, step)` | Lower bound for target short delta |
| `target_short_delta_high` | Float | `np.arange(start, stop + step, step)` | Upper bound for target short delta |
| `wing_width_min` | Float | `np.arange(start, stop + step, step)` | Minimum allowed wing width |
| `iv_rank_min` | Float | `np.arange(start, stop + step, step)` | Minimum required IV Rank |
| `vix_threshold` | Float | `np.arange(start, stop + step, step)` | Maximum allowed VIX level |

---

## 📁 Repository Structure

```text
\spy-iron-condor-trading\
|   CONTRIBUTING.md         # Guidelines for team collaboration
|   DEVELOPMENT_STATUS.md   # Current roadmap and bug tracker
|   PolyOptionsData.py      # Legacy Polygon options fetching script
|   requirements.txt        # Project dependencies
+---analytics/
|       audit_logger.py     # Detailed trade and system logging
|       metrics.py          # Portfolio and performance math
+---core/
|       backtest_engine.py  # High-fidelity Backtrader engine
|       benchmark_cpu.py    # Hardware speed benchmarking
|       broker.py           # Abstract broker interface (Alpaca/Paper)
|       config.py           # Active user configuration
|       liquidity_gate.py   # Bid/Ask spread and volume checks
|       main.py             # CLI Entry point
|       optimizer.py        # Grid search and reporting engine
|       trade_decision.py   # Entry/Exit signal processing
+---data/
|   \---synthetic_options/  # Generated option chains (prices/Greeks)
+---data_factory/
|       AlpacaGetData.py    # Fetcher for underlying stock data
|       polygon_client.py   # Wrapper for Polygon.io API
|       sync_engine.py      # Multi-timeframe data alignment
|       SyntheticOptionsEngine.py # BS-model option chain generator
+---execution/
|       paper_executor.py   # Live paper-trading order management
+---intelligence/
|       fuzzy_engine.py     # Sugeno fuzzy inference for sizing
|       regime_filter.py    # Volatility and trend filters
|       mamba_engine.py     # Neural state-space forecasting
+---qtmf/
|       facade.py           # Central Neuro-Fuzzy Facade
|       models.py           # TradeIntent & SizingPlan Data Structures
+---reports/
|   \---SPY/                # 5m/15m/60m underlying price data
+---strategies/
|       options_strategy.py # Iron Condor logic and leg management
\---tests/
        test_iron_condor_sizing.py  # Regression tests for position sizing
```

### **File Glossary & Functions**

#### **Root Files**
- `main.py`: The central hub that coordinates backtesting, optimization, and live trading.
- `requirements.txt`: Defines the Python environment (Pandas, Backtrader, NumPy, etc.).

#### **Core Module (`/core`)**
- `backtest_engine.py`: Implements the high-fidelity simulator that handles mark-to-market P&L and multi-leg strategies.
- `optimizer.py`: Manages the phased serial grid search, hardware benchmarking, and generates the `top100_*.csv` reports.
- `config.py`: Stores all strategy parameters, API keys, and execution settings.
- `trade_decision.py`: Decides when to open or close positions based on strategy signals.

#### **Data Factory (`/data_factory`)**
- `SyntheticOptionsEngine.py`: The mathematical core that generates theoretical option prices and Greeks using the Black-Scholes model.
- `AlpacaGetData.py`: Automates the downloading of historical 5-minute bars for the underlying ticker.
- `sync_engine.py`: Ensures that data from different timeframes (5m, 15m, 60m) are perfectly aligned for technical analysis.

#### **Intelligence Module (`/intelligence`)**
- `fuzzy_engine.py`: uses fuzzy logic to scale trade quantities based on market confidence and volatility regimes.
- `regime_filter.py`: Prevents trading during extreme volatility (VIX spikes) or low IV Rank environments.

#### **Strategies Module (`/strategies`)**
- `options_strategy.py`: Contains the logic for selecting Condor legs by Delta, calculating credits, and managing individual leg exits.

---

## ⚖️ Disclaimer
*This software is for educational and research purposes only. Trading options involves significant risk. The developers are not responsible for any financial losses incurred through the use of this software.*
