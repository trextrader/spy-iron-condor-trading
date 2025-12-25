import os

def setup_project_structure():
    # Define the directory tree
    folders = [
        "core",
        "data_factory",
        "intelligence",
        "intelligence/optimizers",
        "intelligence/models",
        "strategies",
        "analytics",
        "reports/SPY"
    ]

    # Define new placeholder files with initial imports/templates
    files = {
        "intelligence/regime_filter.py": "# Regime & Liquidity Filters\nimport pandas as pd\n\ndef check_liquidity_gate(df: pd.DataFrame, current_spread: float, n_bars: int = 10) -> bool:\n    \"\"\"Rule: If Avg(H-L) > Current Spread, market is liquid enough to overcome cost.\"\"\"\n    if df.empty or len(df) < n_bars:\n        return False\n    df['hl_range'] = df['high'] - df['low']\n    avg_hl_range = df['hl_range'].tail(n_bars).mean()\n    return avg_hl_range > current_spread\n",
        
        "intelligence/fuzzy_engine.py": "# Fuzzy Multi-TF Logic\nimport numpy as np\n\ndef get_tf_membership(price_data: dict) -> float:\n    \"\"\"Stub: Returns conviction between -1.0 and 1.0\"\"\"\n    return 0.0\n\ndef calculate_mtf_consensus(tf_scores: dict) -> float:\n    \"\"\"Weights: D: 25%, 4H: 25%, 1H: 20%, 15M: 15%, 5M: 10%, 1M: 5%\"\"\"\n    weights = {'D': 0.25, '4H': 0.25, '60': 0.20, '15': 0.15, '5': 0.10, '1': 0.05}\n    consensus = sum(tf_scores.get(tf, 0) * w for tf, w in weights.items())\n    return consensus\n",
        
        "analytics/audit_logger.py": "# Honest Broker Audit\nimport datetime as dt\n\ndef log_execution_quality(symbol, order_id, bid, ask, fill_price):\n    mid = (bid + ask) / 2.0\n    slippage = fill_price - mid\n    print(f\"[AUDIT] {symbol} {order_id} | Slippage: {slippage:.4f} | Spread: {ask-bid:.4f}\")\n",
        
        "data_factory/multi_tf_store.py": "# Multi-Timeframe Data Aggregator\nimport pandas as pd\nimport os\n\nclass DataStore:\n    def __init__(self, symbol):\n        self.symbol = symbol\n        self.base_path = f'reports/{symbol}'\n\n    def load_tf(self, tf_label):\n        path = os.path.join(self.base_path, f'{self.symbol}_{tf_label}.csv')\n        return pd.read_csv(path, parse_dates=['timestamp'])\n"
    }

    # Create Folders
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

    # Create Files
    for file_path, content in files.items():
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Initialized file: {file_path}")

    print("\nProject Structure Ready.")

if __name__ == "__main__":
    setup_project_structure()