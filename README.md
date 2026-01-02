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

## 🛠️ Installation

1. **Python Version**: Ensure you are using **Python 3.12**.
2. **Setup Dependencies**:
   ```powershell
   pip install backtrader alpaca-py mplfinance tabulate matplotlib pandas numpy psutil
   ```

---

## ⚙️ Configuration

1. **API Keys**: Copy `core/config.template.py` to `core/config.py` and enter your keys:
   - `polygon_key`: Required for real-time data and backtesting.
   - `alpaca_key` / `alpaca_secret`: Required for paper/live trading.
2. **Strategy Parameters**: Tune your core strategy logic (DTE, Delta, IVR thresholds) directly in `core/config.py`.

---

## 📈 Execution Modes

### 1. High-Fidelity Backtest
Run a simulation over historical data with full reporting:
```powershell
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
