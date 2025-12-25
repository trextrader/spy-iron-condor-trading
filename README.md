# SPY Iron Condor Trading System

A professional-grade options trading system for SPY iron condors with dynamic wing adjustment, regime filtering, and comprehensive backtesting capabilities.

## 🚀 Features

- **Dynamic Wing Width**: Adjusts strike width based on VIX and IV Rank regimes
- **Smart Strike Selection**: Delta-based targeting for optimal premium collection
- **Credit-to-Width Validation**: Ensures adequate premium for risk taken
- **Position Management**: Automated profit taking, stop losses, and vertical rolls
- **Delta Hedging**: Portfolio-level delta neutralization
- **Backtrader Integration**: Professional backtesting with detailed metrics
- **Comprehensive Reporting**: PDF reports with equity curves, drawdowns, and trade markers

## 📋 Prerequisites

- Python 3.10+
- Alpaca API account (for historical data)
- Polygon.io API account (optional, for live market data)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/trextrader/spy-iron-condor-trading.git
cd spy-iron-condor-trading
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install backtrader matplotlib pandas numpy tabulate mplfinance requests pytz alpaca-py
```

### 4. Configure API Keys
```bash
copy config.template.py config.py
notepad config.py
```

Edit and replace placeholder values with your actual API keys.

## 📊 Usage

### Download Historical Data
```bash
python AlpacaGetData.py
```
Enter "SPY" for symbol, select timeframe "5"

### Run Backtest
```bash
python main.py --mode backtest --bt-cash 25000
```

### Advanced Options
```bash
python main.py --mode backtest --bt-cash 50000 --wing-min 5.0 --wing-max 10.0 --ivr-min 30.0 --vix-max 25.0 --profit-pct 0.50 --loss-multiple 1.5 --max-hold-days 14
```

## 📁 Project Structure
```
spy-iron-condor-trading/
├── main.py                    # CLI entry point
├── backtest_engine.py         # Backtrader integration
├── options_strategy.py        # Iron condor logic
├── broker.py                  # Broker abstraction
├── polygon_client.py          # Polygon.io client
├── AlpacaGetData.py          # Data downloader
├── config.py                  # API keys (GITIGNORED)
├── config.template.py         # Template
└── reports/                   # Generated reports
```

## 🎯 Strategy Logic

### Entry Criteria
1. IV Rank ≥ 30
2. VIX ≤ 25
3. Under max position limit (3)
4. Portfolio allocation below 15%

### Strike Selection
- DTE: 30-45 days
- Short Delta: 0.15-0.20
- Wing Width: $5-$10 (widens in high vol)
- Min credit-to-width: 0.25

### Exit Rules
1. Profit Target: 50% of max profit
2. Stop Loss: 1.5x credit received
3. Time Decay: Close at 14 DTE
4. Breach: Roll once per side

## 🤝 Contributing
```bash
git pull
git checkout -b feature/your-feature
# Make changes
git add .
git commit -m "Add: description"
git push origin feature/your-feature
```

## ⚠️ Security

- NEVER commit `config.py`
- Each developer maintains own API keys locally
- Always use `config.template.py` as template

## 📈 Output Reports

- **trades.csv**: Detailed trade log with OHLCV and spreads
- **backtest_report.pdf**: 3-page visual report with charts

---

**Last Updated**: December 2024
