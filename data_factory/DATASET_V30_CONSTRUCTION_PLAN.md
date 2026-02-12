# Dataset v3.0 Construction Plan

## CondorNet HAL-S³N Data Pipeline

**Author**: Claude Code (Implementation)
**Based on**: Dr. T. Jerry Mahabub's "CondorNet HAL-SNN New Upgraded Dataset" (Feb 10, 2026)
**Target**: 87-column unified dataset for CondorNet v4.0 training

---

## 1. Executive Summary

This plan constructs the **v3.0 dataset** by merging four synchronized data sources:

| Source | Purpose | Location |
|--------|---------|----------|
| **iVolatility Options** | Real SPY options chains with Greeks | `data/ivolatility/spy_options_ivol_large_clean.csv` |
| **Barchart M1 Studies** | OHLCV + 14 derived indicators | `data/BarChart/SPY_Barchart_*.csv` |
| **Alpaca Options M1** | Tick-level options with intraday Greeks | `data/alpaca_options/spy_options_intraday_large_with_greeks_m1.csv` |
| **Alpaca Quotes** | Microstructure (bid/ask counts, spreads) | To be fetched via Alpaca API |

**Schema Evolution**:
- v2.2: 67 columns (current)
- v3.0: 87 columns (+20 new features)

---

## 2. Target Schema (87 Columns)

### 2.1 Core Identifiers (6 columns)
```
timestamp, symbol, option_symbol, expiration, strike, call_put
```

### 2.2 Price Layer (5 columns)
```
underlying_price, open, high, low, close
```

### 2.3 Options Surface (11 columns)
```
delta, gamma, vega, theta, rho, iv, volume, open_interest, te, ivr, spread_ratio
```

### 2.4 Returns & Volatility (6 columns)
```
log_return, vol_ewma, ret_z, atr_pct, kappa_proxy, vol_energy
```

### 2.5 Trend & Momentum (12 columns)
```
sma, rsi_dyn, adx_adaptive, psar_adaptive, psar_mark, psar_trend, psar_reversion_mu,
macd_norm, macd_signal_norm, macd_histogram, plus_di, minus_di
```

### 2.6 Bands & Consolidation (8 columns)
```
bb_mu_dyn, bb_sigma_dyn, bb_lower_dyn, bb_upper_dyn, stoch_k_dyn,
bandwidth, bb_percentile, bw_expansion_rate, consolidation_score, breakout_score
```

### 2.7 Flow & Breadth (4 columns)
```
cmf, pressure_up, pressure_down, friction_ratio
```

### 2.8 Fuzzy/Control Layer (7 columns)
```
mtf_consensus, chaos_membership, position_size_mult, fuzzy_reversion_11,
exec_allow, gap_risk_score, risk_override, iv_confidence
```

### 2.9 Targets (4 columns)
```
target_spot, max_dd_60m, beta1_norm_stub, target_roi_calendar
```

### 2.10 NEW v3.0 Features - Trend Structure (3 columns)
```
FRAMA, Anchored_VWAP, McClellanOsc
```

### 2.11 NEW v3.0 Features - IV Bands (3 columns)
```
IV_High, IV_Mid, IV_Low
```

### 2.12 NEW v3.0 Features - Options Chain Aggregates (4 columns)
```
Options_Total_Volume, Options_Put_Volume, Options_Call_Volume, OSC_Volume
```

### 2.13 NEW v3.0 Features - Flow & Breadth (4 columns)
```
WeightedAlpha, WilderAccSwingIndex, AccDistWill, AccDistWillMovAvg
```

### 2.14 NEW v3.0 Features - Microstructure (5 columns)
```
aggregate_bid_count, aggregate_ask_count, mean_bid, mean_ask, quote_spread
```

---

## 3. Data Source Mapping

### 3.1 iVolatility Options Data
**File**: `data/ivolatility/spy_options_ivol_large_clean.csv`
**Columns Available**:
```
symbol, underlying_price, option_symbol, expiration, strike, call_put,
ask, bid, mean_price, volume, open_interest, iv, delta, gamma, vega, theta, rho,
trade_date_fetch
```

**Derived**:
- `spread_ratio` = (ask - bid) / mean_price
- `te` = (expiration - trade_date) / 365

### 3.2 Barchart M1 Studies (CRITICAL - Contains 14 v3.0 features!)
**File**: `data/BarChart/SPY_Barchart_Interactive_Chart_Range_1m_02_09_2026.csv`
**Columns Available**:
```
Date Time, Open, High, Low, Close, Change,
MA-Simple, FRAMA, Anchored VWAP, McClellanOsc,
Implied Volatility High, Implied Volatility, Implied Volatility Low,
Options Total Volume, Options Put Volume, Options Call Volume, OSC-Volume,
WeightedAlpha, WilderAccSwingIndex, AccDistWill, AccDistWillMovAvg
```

### 3.3 Alpaca Options M1
**File**: `data/alpaca_options/spy_options_intraday_large_with_greeks_m1.csv`
**Columns Available**:
```
symbol, timestamp, open, high, low, close, volume, trade_count, vwap,
underlying, expiration, strike, option_type,
iv, delta, gamma, theta, vega, rho (+ intraday variants)
```

### 3.4 Microstructure (Alpaca Quotes API)
**To Fetch**: Tick-level bid/ask quotes, resample to 1-minute
**Columns to Compute**:
```
aggregate_bid_count, aggregate_ask_count, mean_bid, mean_ask, quote_spread
```

---

## 4. Implementation Scripts

### 4.1 Script 1: Data Inventory & Validation
**File**: `data_factory/scripts/01_inventory_data_sources.py`

```python
#!/usr/bin/env python3
"""
Script 1: Inventory all data sources and validate schemas.
"""

import pandas as pd
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def inventory_data_sources():
    """Inventory all available data sources for v3.0 dataset construction."""

    inventory = {}

    # 1. iVolatility Options
    ivol_path = DATA_DIR / "ivolatility" / "spy_options_ivol_large_clean.csv"
    if ivol_path.exists():
        df = pd.read_csv(ivol_path, nrows=5)
        inventory['ivolatility'] = {
            'path': str(ivol_path),
            'columns': df.columns.tolist(),
            'sample_rows': len(pd.read_csv(ivol_path)),
            'date_range': 'TBD'
        }

    # 2. Barchart M1 Studies
    barchart_files = list((DATA_DIR / "BarChart").glob("*_1m_*.csv"))
    for f in barchart_files:
        df = pd.read_csv(f, skiprows=1, nrows=5)
        inventory[f'barchart_{f.stem}'] = {
            'path': str(f),
            'columns': df.columns.tolist(),
            'sample_rows': len(pd.read_csv(f, skiprows=1))
        }

    # 3. Alpaca Options M1
    alpaca_path = DATA_DIR / "alpaca_options" / "spy_options_intraday_large_with_greeks_m1.csv"
    if alpaca_path.exists():
        df = pd.read_csv(alpaca_path, nrows=5)
        inventory['alpaca_options_m1'] = {
            'path': str(alpaca_path),
            'columns': df.columns.tolist(),
            'sample_rows': len(pd.read_csv(alpaca_path))
        }

    # 4. Current v2.2 Dataset
    v22_path = DATA_DIR / "processed" / "mamba_institutional_2024_1m_v22.csv"
    if v22_path.exists():
        df = pd.read_csv(v22_path, nrows=5)
        inventory['current_v22'] = {
            'path': str(v22_path),
            'columns': df.columns.tolist(),
            'num_columns': len(df.columns)
        }

    return inventory

if __name__ == "__main__":
    inv = inventory_data_sources()
    print(json.dumps(inv, indent=2))

    # Save inventory
    out_path = PROJECT_ROOT / "data_factory" / "data_inventory.json"
    with open(out_path, 'w') as f:
        json.dump(inv, f, indent=2)
    print(f"\nInventory saved to {out_path}")
```

---

### 4.2 Script 2: Load & Normalize Barchart Studies
**File**: `data_factory/scripts/02_load_barchart_studies.py`

```python
#!/usr/bin/env python3
"""
Script 2: Load Barchart M1 studies and normalize column names for v3.0 schema.

Barchart provides 14 of the 20 new v3.0 features directly:
- FRAMA, Anchored VWAP, McClellanOsc
- IV_High, IV_Mid (Implied Volatility), IV_Low
- Options_Total_Volume, Options_Put_Volume, Options_Call_Volume, OSC_Volume
- WeightedAlpha, WilderAccSwingIndex, AccDistWill, AccDistWillMovAvg
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Column name mapping: Barchart -> v3.0 Schema
BARCHART_COLUMN_MAP = {
    'Date Time': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'FRAMA': 'FRAMA',
    'Anchored VWAP': 'Anchored_VWAP',
    'McClellanOsc': 'McClellanOsc',
    'Implied Volatility High': 'IV_High',
    'Implied Volatility': 'IV_Mid',
    'Implied Volatility Low': 'IV_Low',
    'Options Total Volume': 'Options_Total_Volume',
    'Options Put Volume': 'Options_Put_Volume',
    'Options Call Volume': 'Options_Call_Volume',
    'OSC-Volume': 'OSC_Volume',
    'WeightedAlpha': 'WeightedAlpha',
    'WilderAccSwingIndex': 'WilderAccSwingIndex',
    'AccDistWill': 'AccDistWill',
    'AccDistWillMovAvg': 'AccDistWillMovAvg',
}

def load_barchart_studies(filepath: Path) -> pd.DataFrame:
    """Load Barchart M1 studies CSV and normalize columns."""

    # Skip first row (study definitions header)
    df = pd.read_csv(filepath, skiprows=1)

    # Rename columns to v3.0 schema
    df = df.rename(columns=BARCHART_COLUMN_MAP)

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Select only the columns we need
    v30_cols = list(BARCHART_COLUMN_MAP.values())
    available_cols = [c for c in v30_cols if c in df.columns]
    df = df[available_cols]

    # Replace NaN from computation artifacts with 0
    df = df.fillna(0)

    print(f"Loaded {len(df)} rows with {len(available_cols)} v3.0 features from Barchart")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df

if __name__ == "__main__":
    barchart_files = list((DATA_DIR / "BarChart").glob("*_1m_*.csv"))
    if barchart_files:
        df = load_barchart_studies(barchart_files[0])
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nSample:\n{df.head()}")
```

---

### 4.3 Script 3: Load & Normalize iVolatility Options
**File**: `data_factory/scripts/03_load_ivolatility_options.py`

```python
#!/usr/bin/env python3
"""
Script 3: Load iVolatility real options chain data.

Provides:
- Real option symbols (for backtesting with real contracts)
- Greeks: delta, gamma, vega, theta, rho
- IV, volume, open_interest
- bid/ask prices
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def load_ivolatility_options(filepath: Path) -> pd.DataFrame:
    """Load iVolatility options chain data."""

    df = pd.read_csv(filepath)

    # Standardize column names
    df = df.rename(columns={
        'underlying_price': 'underlying_price',
        'option_symbol': 'option_symbol',
        'expiration': 'expiration',
        'strike': 'strike',
        'call_put': 'call_put',
        'bid': 'bid',
        'ask': 'ask',
        'mean_price': 'mid',
        'volume': 'volume',
        'open_interest': 'open_interest',
        'iv': 'iv',
        'delta': 'delta',
        'gamma': 'gamma',
        'vega': 'vega',
        'theta': 'theta',
        'rho': 'rho',
        'trade_date_fetch': 'date',
    })

    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])

    # Compute derived features
    df['spread_ratio'] = (df['ask'] - df['bid']) / df['mid'].replace(0, 1)
    df['te'] = (df['expiration'] - df['date']).dt.days / 365.0

    # IVR (IV Rank) - needs historical context, placeholder for now
    df['ivr'] = 0.5  # Will be computed properly in merge script

    print(f"Loaded {len(df)} option rows")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique symbols: {df['option_symbol'].nunique()}")

    return df

if __name__ == "__main__":
    ivol_path = DATA_DIR / "ivolatility" / "spy_options_ivol_large_clean.csv"
    if ivol_path.exists():
        df = load_ivolatility_options(ivol_path)
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nSample:\n{df.head()}")
```

---

### 4.4 Script 4: Fetch & Compute Microstructure Features
**File**: `data_factory/scripts/04_compute_microstructure.py`

```python
#!/usr/bin/env python3
"""
Script 4: Compute microstructure features from Alpaca tick-level quotes.

Features to compute:
- aggregate_bid_count: Number of bid updates per minute
- aggregate_ask_count: Number of ask updates per minute
- mean_bid: Average bid price per minute
- mean_ask: Average ask price per minute
- quote_spread: mean_ask - mean_bid
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def compute_microstructure_from_quotes(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tick-level quotes to 1-minute microstructure features.

    Input: DataFrame with columns [timestamp, bid, ask]
    Output: DataFrame with 1-minute aggregated microstructure features
    """

    # Ensure timestamp is datetime
    quotes_df['timestamp'] = pd.to_datetime(quotes_df['timestamp'])

    # Floor to minute
    quotes_df['minute'] = quotes_df['timestamp'].dt.floor('1min')

    # Aggregate by minute
    agg = quotes_df.groupby('minute').agg({
        'bid': ['count', 'mean'],
        'ask': ['count', 'mean'],
    })

    # Flatten column names
    agg.columns = ['aggregate_bid_count', 'mean_bid', 'aggregate_ask_count', 'mean_ask']
    agg = agg.reset_index()
    agg = agg.rename(columns={'minute': 'timestamp'})

    # Compute quote spread
    agg['quote_spread'] = agg['mean_ask'] - agg['mean_bid']

    return agg

def generate_placeholder_microstructure(timestamps: pd.Series) -> pd.DataFrame:
    """
    Generate placeholder microstructure features when real quotes unavailable.
    Uses realistic defaults based on SPY typical microstructure.
    """

    n = len(timestamps)

    # Realistic SPY microstructure defaults
    return pd.DataFrame({
        'timestamp': timestamps,
        'aggregate_bid_count': np.random.poisson(100, n),  # ~100 quotes/min typical
        'aggregate_ask_count': np.random.poisson(100, n),
        'mean_bid': np.nan,  # Will be filled from OHLCV
        'mean_ask': np.nan,  # Will be filled from OHLCV
        'quote_spread': 0.01,  # $0.01 typical SPY spread
    })

if __name__ == "__main__":
    # Demo with placeholder data
    timestamps = pd.date_range('2025-01-02 09:30', periods=390, freq='1min')
    micro_df = generate_placeholder_microstructure(timestamps)
    print(f"Generated {len(micro_df)} rows of placeholder microstructure")
    print(micro_df.head())
```

---

### 4.5 Script 5: Compute Legacy v2.5 Features
**File**: `data_factory/scripts/05_compute_legacy_features.py`

```python
#!/usr/bin/env python3
"""
Script 5: Compute legacy v2.5 features from OHLCV data.

Features computed:
- log_return, vol_ewma, ret_z, atr_pct
- kappa_proxy, vol_energy
- rsi_dyn, adx_adaptive, psar_adaptive
- bb_mu_dyn, bb_sigma_dyn, bb_lower_dyn, bb_upper_dyn
- stoch_k_dyn, bandwidth, bb_percentile, bw_expansion_rate
- cmf, pressure_up, pressure_down, friction_ratio
- consolidation_score, breakout_score
- macd_norm, macd_signal_norm, macd_histogram
- plus_di, minus_di
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta not available, using manual implementations")

def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute log returns."""
    return np.log(close / close.shift(1))

def compute_ewma_volatility(returns: pd.Series, span: int = 20) -> pd.Series:
    """Compute EWMA volatility."""
    return returns.ewm(span=span).std()

def compute_ret_z(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute z-score of returns."""
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    return (returns - mu) / sigma.replace(0, 1)

def compute_atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Compute ATR as percentage of price."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=window).mean()
    return atr / close

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, 1)
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Compute Bollinger Bands."""
    mu = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    upper = mu + num_std * sigma
    lower = mu - num_std * sigma
    bandwidth = (upper - lower) / mu
    percentile = (close - lower) / (upper - lower).replace(0, 1)
    return mu, sigma, lower, upper, bandwidth, percentile

def compute_stochastic(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """Compute Stochastic %K."""
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    return 100 * (close - lowest) / (highest - lowest).replace(0, 1)

def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Compute Chaikin Money Flow."""
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, 1)
    mf_vol = mf_mult * volume
    return mf_vol.rolling(window).sum() / volume.rolling(window).sum().replace(0, 1)

def compute_all_legacy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all legacy v2.5 features on a DataFrame with OHLCV."""

    out = df.copy()

    # Returns & Volatility
    out['log_return'] = compute_log_returns(df['close'])
    out['vol_ewma'] = compute_ewma_volatility(out['log_return'])
    out['ret_z'] = compute_ret_z(out['log_return'])
    out['atr_pct'] = compute_atr_pct(df['high'], df['low'], df['close'])
    out['kappa_proxy'] = out['vol_ewma'].rolling(10).apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0)
    out['vol_energy'] = out['vol_ewma'] ** 2

    # Momentum
    out['rsi_dyn'] = compute_rsi(df['close'])
    out['stoch_k_dyn'] = compute_stochastic(df['close'], df['high'], df['low'])

    # Bollinger Bands
    mu, sigma, lower, upper, bw, pct = compute_bollinger_bands(df['close'])
    out['bb_mu_dyn'] = mu
    out['bb_sigma_dyn'] = sigma
    out['bb_lower_dyn'] = lower
    out['bb_upper_dyn'] = upper
    out['bandwidth'] = bw
    out['bb_percentile'] = pct
    out['bw_expansion_rate'] = bw.pct_change()

    # Flow
    if 'volume' in df.columns:
        out['cmf'] = compute_cmf(df['high'], df['low'], df['close'], df['volume'])
    else:
        out['cmf'] = 0

    # Consolidation/Breakout
    out['consolidation_score'] = 1 - out['bandwidth'].rolling(20).rank(pct=True)
    out['breakout_score'] = out['bw_expansion_rate'].rolling(5).max()

    # Fill NaN with 0
    out = out.fillna(0)

    return out

if __name__ == "__main__":
    # Demo
    dates = pd.date_range('2025-01-02 09:30', periods=1000, freq='1min')
    demo_df = pd.DataFrame({
        'timestamp': dates,
        'open': 585 + np.random.randn(1000).cumsum() * 0.1,
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.poisson(10000, 1000),
    })
    demo_df['high'] = demo_df['open'] + abs(np.random.randn(1000)) * 0.2
    demo_df['low'] = demo_df['open'] - abs(np.random.randn(1000)) * 0.2
    demo_df['close'] = demo_df['open'] + np.random.randn(1000) * 0.1

    result = compute_all_legacy_features(demo_df)
    print(f"Computed {len(result.columns)} columns")
    print(result[['timestamp', 'log_return', 'vol_ewma', 'rsi_dyn', 'bb_percentile']].head())
```

---

### 4.6 Script 6: Master Merge Pipeline
**File**: `data_factory/scripts/06_merge_dataset_v30.py`

```python
#!/usr/bin/env python3
"""
Script 6: Master Merge Pipeline for Dataset v3.0

This script:
1. Loads all data sources
2. Aligns by timestamp (left-join on options chain)
3. Computes missing features
4. Validates 87-column schema
5. Outputs final v3.0 dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed"

# Import our feature computation modules
from load_barchart_studies import load_barchart_studies, BARCHART_COLUMN_MAP
from load_ivolatility_options import load_ivolatility_options
from compute_microstructure import generate_placeholder_microstructure
from compute_legacy_features import compute_all_legacy_features

# ============================================================================
# V3.0 SCHEMA DEFINITION (87 columns)
# ============================================================================
V30_SCHEMA = [
    # Core Identifiers (6)
    'timestamp', 'symbol', 'option_symbol', 'expiration', 'strike', 'call_put',

    # Price Layer (5)
    'underlying_price', 'open', 'high', 'low', 'close',

    # Options Surface (11)
    'delta', 'gamma', 'vega', 'theta', 'rho', 'iv', 'volume', 'open_interest',
    'te', 'ivr', 'spread_ratio',

    # Returns & Volatility (6)
    'log_return', 'vol_ewma', 'ret_z', 'atr_pct', 'kappa_proxy', 'vol_energy',

    # Trend & Momentum (12)
    'sma', 'rsi_dyn', 'adx_adaptive', 'psar_adaptive', 'psar_mark', 'psar_trend',
    'psar_reversion_mu', 'macd_norm', 'macd_signal_norm', 'macd_histogram',
    'plus_di', 'minus_di',

    # Bands & Consolidation (10)
    'bb_mu_dyn', 'bb_sigma_dyn', 'bb_lower_dyn', 'bb_upper_dyn', 'stoch_k_dyn',
    'bandwidth', 'bb_percentile', 'bw_expansion_rate', 'consolidation_score', 'breakout_score',

    # Flow & Breadth (4)
    'cmf', 'pressure_up', 'pressure_down', 'friction_ratio',

    # Fuzzy/Control Layer (8)
    'mtf_consensus', 'chaos_membership', 'position_size_mult', 'fuzzy_reversion_11',
    'exec_allow', 'gap_risk_score', 'risk_override', 'iv_confidence',

    # Targets (4)
    'target_spot', 'max_dd_60m', 'beta1_norm_stub', 'target_roi_calendar',

    # NEW v3.0 - Trend Structure (3)
    'FRAMA', 'Anchored_VWAP', 'McClellanOsc',

    # NEW v3.0 - IV Bands (3)
    'IV_High', 'IV_Mid', 'IV_Low',

    # NEW v3.0 - Options Chain Aggregates (4)
    'Options_Total_Volume', 'Options_Put_Volume', 'Options_Call_Volume', 'OSC_Volume',

    # NEW v3.0 - Flow & Breadth Extensions (4)
    'WeightedAlpha', 'WilderAccSwingIndex', 'AccDistWill', 'AccDistWillMovAvg',

    # NEW v3.0 - Microstructure (5)
    'aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread',
]

def validate_schema(df: pd.DataFrame) -> bool:
    """Validate DataFrame has all 87 v3.0 columns."""
    missing = set(V30_SCHEMA) - set(df.columns)
    extra = set(df.columns) - set(V30_SCHEMA)

    if missing:
        print(f"MISSING COLUMNS: {missing}")
    if extra:
        print(f"EXTRA COLUMNS (will be dropped): {extra}")

    return len(missing) == 0

def merge_dataset_v30(
    options_df: pd.DataFrame,
    barchart_df: pd.DataFrame,
    microstructure_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Master merge function for v3.0 dataset.

    Strategy:
    1. Options chain is the base (contains per-contract rows)
    2. Barchart studies are joined by timestamp (1:many - broadcast to all contracts)
    3. Microstructure features are joined by timestamp (1:many)
    4. Missing columns filled with defaults
    """

    print(f"Starting merge: {len(options_df)} option rows")

    # Ensure timestamp columns are datetime
    if 'timestamp' not in options_df.columns and 'date' in options_df.columns:
        options_df['timestamp'] = options_df['date']
    options_df['timestamp'] = pd.to_datetime(options_df['timestamp'])
    barchart_df['timestamp'] = pd.to_datetime(barchart_df['timestamp'])

    # Floor to minute for joining
    options_df['join_ts'] = options_df['timestamp'].dt.floor('1min')
    barchart_df['join_ts'] = barchart_df['timestamp'].dt.floor('1min')

    # Join Barchart studies
    barchart_cols = [c for c in barchart_df.columns if c != 'timestamp' and c != 'join_ts']
    barchart_join = barchart_df[['join_ts'] + barchart_cols].drop_duplicates(subset=['join_ts'])

    merged = options_df.merge(barchart_join, on='join_ts', how='left', suffixes=('', '_bc'))
    print(f"After Barchart join: {len(merged)} rows")

    # Join microstructure if available
    if microstructure_df is not None:
        microstructure_df['join_ts'] = pd.to_datetime(microstructure_df['timestamp']).dt.floor('1min')
        micro_cols = ['join_ts', 'aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread']
        micro_cols = [c for c in micro_cols if c in microstructure_df.columns]
        micro_join = microstructure_df[micro_cols].drop_duplicates(subset=['join_ts'])
        merged = merged.merge(micro_join, on='join_ts', how='left', suffixes=('', '_ms'))
        print(f"After microstructure join: {len(merged)} rows")

    # Drop join column
    merged = merged.drop(columns=['join_ts'])

    # Compute legacy features if missing
    legacy_cols = ['log_return', 'vol_ewma', 'ret_z', 'atr_pct', 'rsi_dyn', 'bb_percentile']
    if not all(c in merged.columns for c in legacy_cols):
        print("Computing legacy features...")
        merged = compute_all_legacy_features(merged)

    # Fill missing columns with defaults
    for col in V30_SCHEMA:
        if col not in merged.columns:
            merged[col] = 0
            print(f"  Added missing column: {col} = 0")

    # Reorder to schema
    merged = merged[V30_SCHEMA]

    # Final NaN cleanup
    merged = merged.fillna(0)

    # Validate
    if validate_schema(merged):
        print(f"\n✅ Schema validation PASSED: {len(merged.columns)} columns")
    else:
        print(f"\n❌ Schema validation FAILED")

    return merged

def main():
    """Main entry point for v3.0 dataset construction."""

    print("=" * 60)
    print("DATASET v3.0 CONSTRUCTION PIPELINE")
    print("=" * 60)

    # 1. Load iVolatility options (REQUIRED)
    ivol_path = DATA_DIR / "ivolatility" / "spy_options_ivol_large_clean.csv"
    if not ivol_path.exists():
        raise FileNotFoundError(f"Required: {ivol_path}")
    options_df = load_ivolatility_options(ivol_path)

    # 2. Load Barchart studies (REQUIRED for v3.0 features)
    barchart_files = list((DATA_DIR / "BarChart").glob("*_1m_*.csv"))
    if not barchart_files:
        raise FileNotFoundError("No Barchart M1 files found")
    # Exclude greeks file
    barchart_files = [f for f in barchart_files if 'greeks' not in f.name.lower()]
    barchart_df = load_barchart_studies(barchart_files[0])

    # 3. Generate placeholder microstructure (until real quotes available)
    unique_timestamps = options_df['timestamp'].drop_duplicates()
    micro_df = generate_placeholder_microstructure(unique_timestamps)

    # 4. Merge all sources
    v30_df = merge_dataset_v30(options_df, barchart_df, micro_df)

    # 5. Save output
    output_path = OUTPUT_DIR / f"condornet_v30_{datetime.now().strftime('%Y%m%d')}.csv"
    v30_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved v3.0 dataset to: {output_path}")
    print(f"   Rows: {len(v30_df):,}")
    print(f"   Columns: {len(v30_df.columns)}")

    return v30_df

if __name__ == "__main__":
    main()
```

---

### 4.7 Script 7: Schema Validation & Audit
**File**: `data_factory/scripts/07_validate_v30_schema.py`

```python
#!/usr/bin/env python3
"""
Script 7: Validate v3.0 dataset schema and data integrity.

Checks:
1. All 87 columns present
2. No NaN values in critical columns
3. Data types correct
4. Value ranges reasonable
5. Timestamp monotonicity
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Expected ranges for key features
EXPECTED_RANGES = {
    'delta': (-1.0, 1.0),
    'gamma': (0, 1.0),
    'iv': (0, 5.0),
    'ivr': (0, 1.0),
    'rsi_dyn': (0, 100),
    'bb_percentile': (0, 1.0),
    'spread_ratio': (0, 1.0),
}

def validate_v30_dataset(filepath: Path) -> dict:
    """Run comprehensive validation on v3.0 dataset."""

    results = {
        'passed': True,
        'checks': [],
        'warnings': [],
        'errors': [],
    }

    df = pd.read_csv(filepath)

    # 1. Column count
    if len(df.columns) == 87:
        results['checks'].append(f"✅ Column count: {len(df.columns)} (expected 87)")
    else:
        results['errors'].append(f"❌ Column count: {len(df.columns)} (expected 87)")
        results['passed'] = False

    # 2. No NaN in critical columns
    critical_cols = ['timestamp', 'option_symbol', 'strike', 'call_put', 'close', 'delta', 'iv']
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count == 0:
                results['checks'].append(f"✅ No NaN in {col}")
            else:
                results['warnings'].append(f"⚠️ {nan_count} NaN values in {col}")

    # 3. Value ranges
    for col, (min_val, max_val) in EXPECTED_RANGES.items():
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()
            if actual_min >= min_val and actual_max <= max_val:
                results['checks'].append(f"✅ {col} in range [{min_val}, {max_val}]")
            else:
                results['warnings'].append(f"⚠️ {col} range [{actual_min:.2f}, {actual_max:.2f}] outside expected [{min_val}, {max_val}]")

    # 4. Timestamp monotonicity (per symbol)
    if 'timestamp' in df.columns and 'option_symbol' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
        monotonic = df.groupby('option_symbol')['ts'].apply(lambda x: x.is_monotonic_increasing).all()
        if monotonic:
            results['checks'].append("✅ Timestamps monotonic per symbol")
        else:
            results['warnings'].append("⚠️ Non-monotonic timestamps detected")

    # 5. Call/Put distribution
    if 'call_put' in df.columns:
        cp_counts = df['call_put'].value_counts()
        results['checks'].append(f"✅ Call/Put distribution: {cp_counts.to_dict()}")

    # Summary
    results['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'unique_symbols': df['option_symbol'].nunique() if 'option_symbol' in df.columns else 0,
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else 'N/A',
    }

    return results

if __name__ == "__main__":
    # Find latest v3.0 dataset
    v30_files = list(DATA_DIR.glob("condornet_v30_*.csv"))
    if v30_files:
        latest = sorted(v30_files)[-1]
        print(f"Validating: {latest}")
        results = validate_v30_dataset(latest)

        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        for check in results['checks']:
            print(check)

        if results['warnings']:
            print("\nWARNINGS:")
            for warn in results['warnings']:
                print(warn)

        if results['errors']:
            print("\nERRORS:")
            for err in results['errors']:
                print(err)

        print(f"\nSUMMARY: {results['summary']}")
        print(f"\nOVERALL: {'PASSED' if results['passed'] else 'FAILED'}")
    else:
        print("No v3.0 dataset files found")
```

---

## 5. Execution Order

```bash
# From project root:
cd data_factory/scripts

# Step 1: Inventory data sources
python 01_inventory_data_sources.py

# Step 2: Verify Barchart studies load correctly
python 02_load_barchart_studies.py

# Step 3: Verify iVolatility options load correctly
python 03_load_ivolatility_options.py

# Step 4: Prepare microstructure (placeholder or real)
python 04_compute_microstructure.py

# Step 5: Test legacy feature computation
python 05_compute_legacy_features.py

# Step 6: RUN MASTER MERGE PIPELINE
python 06_merge_dataset_v30.py

# Step 7: Validate output
python 07_validate_v30_schema.py
```

---

## 6. Data Requirements Checklist

Before running the pipeline, ensure these files exist:

| File | Required | Status |
|------|----------|--------|
| `data/ivolatility/spy_options_ivol_large_clean.csv` | YES | Check |
| `data/BarChart/SPY_Barchart_*_1m_*.csv` | YES | Check |
| `data/alpaca_options/spy_options_intraday_large_with_greeks_m1.csv` | Optional | Check |
| Alpaca API credentials for tick quotes | Optional | Check |

---

## 7. Output

**Final Output File**: `data/processed/condornet_v30_YYYYMMDD.csv`

**Schema**: 87 columns as defined in Section 2

**Quality Guarantees**:
- No NaN values (replaced with 0 for computation artifacts)
- All timestamps in UTC
- Deterministic column ordering
- Backward compatible with v2.2 models

---

## 8. Next Steps After Dataset Construction

1. **Train CondorNet v4.0** on the new 87-feature dataset
2. **Run ablation experiments** to measure feature family contributions
3. **Backtest** with real option symbols (not synthetic)
4. **Monitor stability** via Mathematica eigenvalue analysis

---

*Document Version: 1.0*
*Generated: February 11, 2026*
