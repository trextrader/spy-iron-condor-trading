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

## 🧮 Mathematical Foundation

### 1. Mark-to-Market (MtM) P&L
The strategy tracks the real-time replacement cost of the Iron Condor spread. The floating P&L at any time $t$ is defined as:
$$P\&L_{t} = (Credit_{0} - Cost_{t}) \times Q \times 100$$
Where the current cost to close the spread ($Cost_{t}$) is:
$$Cost_{t} = (C_{short} - C_{long}) + (P_{short} - P_{long})$$
*   $Credit_{0}$: Net premium received at entry.
*   $Q$: Number of contracts (Quantity).

### 2. Fuzzy Intelligence Aggregation
Position sizing and regime detection use a Sugeno-style fuzzy inference engine. The final intelligence signal is calculated using a weighted average of membership functions ($\mu$):
$$Signal = \frac{\sum_{i=1}^{n} w_i \cdot \mu_i(x)}{\sum_{i=1}^{n} \mu_i(x)}$$
This signal dynamically scales $Q$ based on VIX levels and IV Rank consensus.

### 3. Optimization Objective
The system is optimized to maximize the risk-adjusted return ratio, specifically targeting the recovery speed relative to peak-to-valley loss:
$$Ratio = \frac{\text{Net Profit}}{\text{Maximum Drawdown}}$$
Unlike the Sharpe ratio, this metric prioritizes capital preservation and the ability of the strategy to "earn its way out" of drawdowns.

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

## 📁 Repository Structure

- `core/`: Primary logic including the `backtest_engine`, `optimizer`, and `broker` interfaces.
- `strategies/`: Iron Condor strategy definitions and option leg logic.
- `intelligence/`: Fuzzy logic engine and MTF membership functions.
- `data_factory/`: Data synchronization and Polygon API integration.
- `data/`: Local storage for synthetic and historical datasets.

---

## ⚖️ Disclaimer
*This software is for educational and research purposes only. Trading options involves significant risk. The developers are not responsible for any financial losses incurred through the use of this software.*
