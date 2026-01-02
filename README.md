# Quantor-MTFuzz: Iron Condor Algorithmic Trading

**Quantor-MTFuzz** is a high-fidelity algorithmic trading system designed specifically for SPY Iron Condor strategies. It combines Multi-Timeframe (MTF) technical intelligence with Fuzzy Logic position sizing to execute risk-managed options trades in both backtesting and live paper-trading environments.

---

## 🚀 Key Features

- **High-Fidelity Backtesting**: Simulates every 5-minute bar with accurate mark-to-market P&L, leg-by-leg exit logic (Profit Take, Stop Loss, Expiration), and price caching to handle data gaps.
- **Phased Serial Optimization**: A custom grid-search engine that optimizes for the **Net Profit / Max Drawdown** ratio. Includes automatic hardware benchmarking and time-to-completion estimation.
- **Fuzzy Intel Engine**: Dynamic position sizing based on VIX regime, IV Rank, and MTF consensus.
- **Alpaca Integration**: Seamless transition to live paper trading using the Alpaca-Py SDK.
- **Professional Reporting**: Generates automated PDF reports with equity curves, strike overlays, P&L distributions, and monthly performance grids.

---

## 🧮 Mathematical Foundation (Script-by-Script)

### 1. `core/backtest_engine.py`: high-Fidelity MtM PnL
The engine tracks the real-time replacement cost of the Iron Condor spread at every 5-minute bar. The floating PnL at time $t$ is calculated as:
$$PnL_{t} = (Credit_{0} - Cost_{t}) \times Q \times 100$$
Where $Cost_{t}$ is the net cost to "buy back" the spread:
$$Cost_{t} = (C_{short} - C_{long}) + (P_{short} - P_{long})$$
Max Drawdown ($MDD$) is tracked bar-by-bar against the peak equity reached ($E_{peak}$):
$$DD_{t} = 1 - \frac{E_{t}}{E_{peak}} \dots MDD = \max(DD_{t})$$

### 2. `intelligence/fuzzy_engine.py`: Sugeno Fuzzy Inference
Position sizing scales dynamically based on market regimes. We use a Sugeno-style weighted average to aggregate membership functions ($\mu$) for VIX and IV Rank:
$$Signal = \frac{\sum_{i=1}^{n} w_i \cdot \mu_i(x)}{\sum_{i=1}^{n} \mu_i(x)}$$
The resulting $Signal \in [0, 1]$ is used as a multiplier for the base contract quantity $Q$.

### 3. `core/trade_decision.py`: Delta-Based Strike Selection
Strikes are selected by minimizing the distance between the target delta ($\delta_{target}$) and the available market deltas ($\delta_{market}$):
$$Strike = \text{argmin}(|\delta_{market} - \delta_{target}|)$$
This ensures the spread maintains the desired probability of profit and risk-reward profile regardless of stock price movement.

### 4. `core/optimizer.py`: Risk-Adjusted Optimization
The grid search objective is to maximize the ratio of total earnings to the largest realized loss (Maximum Drawdown), ensuring the strategy is "robust" rather than just profitable:
$$\text{Maximize } \Phi = \frac{\sum \text{Net Profit}}{\text{Max Drawdown}}$$
Serial execution is benchmarked via a baseline duration $D_{base}$ to estimate total time $T_{total}$ for $N$ combinations:
$$T_{total} = N \cdot D_{base}$$

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
| `--mtf-timeframes` | string | `5,15,60` | Timeframes to sync (comma-separated) |
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

- `core/`: Primary logic including the `backtest_engine`, `optimizer`, and `broker` interfaces.
- `strategies/`: Iron Condor strategy definitions and option leg logic.
- `intelligence/`: Fuzzy logic engine and MTF membership functions.
- `data_factory/`: Data synchronization and Polygon API integration.
- `data/`: Local storage for synthetic and historical datasets.

---

## ⚖️ Disclaimer
*This software is for educational and research purposes only. Trading options involves significant risk. The developers are not responsible for any financial losses incurred through the use of this software.*
