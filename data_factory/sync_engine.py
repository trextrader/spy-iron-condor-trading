# data_factory/sync_engine.py
import pandas as pd
import numpy as np
import os

from intelligence.indicators.manifold_volatility import (
    curvature_proxy_from_returns,
    volatility_energy_from_curvature,
    dynamic_rsi as dynamic_rsi_raw,
)

# Try to import pandas-ta, fall back to manual calculations if not available
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("[Warning] pandas-ta not installed. Using fallback indicator calculations.")

class MTFSyncEngine:
    def __init__(self, symbol, timeframes=['1', '5', '15']):
        self.symbol = symbol
        self.timeframes = timeframes
        self.data_store = {}
        self._load_all()

    def _calculate_indicators(self, df, timeframe):
        """
        Calculate all technical indicators for a given timeframe.
        Uses pandas-ta library if available, else fallback to manual.
        """
        df = df.copy()
        
        log_ret = np.log(df['close']).diff()
        curvature = curvature_proxy_from_returns(log_ret, span=64)
        vol_energy = volatility_energy_from_curvature(curvature)

        if HAS_PANDAS_TA:
            # === MOMENTUM INDICATORS ===
            
            # Dynamic RSI (14-period)
            df['rsi_14'] = dynamic_rsi_raw(df['close'], window=14, vol_energy=vol_energy)
            
            # Stochastic: %K and %D lines
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            if stoch is not None and not stoch.empty:
                df['stoch_k'] = stoch.iloc[:, 0]  # First column is %K
                df['stoch_d'] = stoch.iloc[:, 1]  # Second column is %D
            else:
                df['stoch_k'] = 50.0
                df['stoch_d'] = 50.0
            
            # === TREND INDICATORS ===
            
            # ADX: Average Directional Index (14-period)
            adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_result is not None and not adx_result.empty:
                df['adx_14'] = adx_result.iloc[:, 0]  # ADX
                df['di_plus'] = adx_result.iloc[:, 1]  # DI+
                df['di_minus'] = adx_result.iloc[:, 2]  # DI-
            else:
                df['adx_14'] = 20.0
                df['di_plus'] = 20.0
                df['di_minus'] = 20.0
            
            # Simple Moving Averages
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            
            # === VOLATILITY INDICATORS ===
            
            # Bollinger Bands (20-period, 2 std dev)
            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None and not bbands.empty:
                df['bb_upper'] = bbands.iloc[:, 2]  # Upper band
                df['bb_middle'] = bbands.iloc[:, 1]  # Middle band
                df['bb_lower'] = bbands.iloc[:, 0]  # Lower band
            else:
                df['bb_upper'] = df['close']
                df['bb_middle'] = df['close']
                df['bb_lower'] = df['close']
            
            # ATR: Average True Range (14-period)
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # PSAR: Parabolic SAR (0.02, 0.2)
            psar = ta.psar(df['high'], df['low'], df['close'], af=0.02, max_af=0.2)
            if psar is not None and not psar.empty:
                # pandas-ta PSAR returns multiple columns; we want the active one
                # psarl (long) or psars (short)
                psar_val = psar.iloc[:, 0] 
                df['psar'] = psar_val
                # Derived: Percentage distance (positive = price above SAR = bullish)
                # We normalize by close price to get a trend strength metric
                df['psar_position'] = (df['close'] - df['psar']) / df['close']
            else:
                df['psar'] = df['close']
                df['psar_position'] = 0.0
            
            # === VOLUME INDICATORS ===
            df['volume_ma_20'] = ta.sma(df['volume'], length=20)
        else:
            # Fallback: Dynamic RSI calculation
            df['rsi_14'] = dynamic_rsi_raw(df['close'], window=14, vol_energy=vol_energy)
            
            # Fallback: Manual SMA
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Fallback: Manual Bollinger Bands
            df['bb_middle'] = df['sma_20']
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (std * 2)
            df['bb_lower'] = df['bb_middle'] - (std * 2)
            
            # Fallback: Manual ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(window=14).mean()
            
            # Fallback: Set defaults for complex indicators
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0
            df['adx_14'] = 20.0
            df['di_plus'] = 20.0
            df['di_minus'] = 20.0
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['psar_position'] = 0.0 # Default if no PSAR
        
        # === DERIVED INDICATORS ===
        
        # Distance from SMA (percentage)
        df['sma_distance'] = (df['close'] - df['sma_20']) / df['sma_20'].replace(0, np.nan)
        
        # Bollinger Band Position (0.0 = lower, 1.0 = upper)
        bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range).clip(0, 1)
        
        # Bollinger Band Width (percentage of middle band)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        
        # ATR as percentage of close
        df['atr_pct'] = df['atr_14'] / df['close'].replace(0, np.nan)

        # === CHAIKIN MONEY FLOW (replaces volume_ratio) ===
        eps = 1e-9
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + eps)
        mf_multiplier = mf_multiplier.clip(-1.0, 1.0)
        mf_volume = mf_multiplier * df['volume']
        df['cmf'] = (mf_volume.rolling(20).sum() / (df['volume'].rolling(20).sum() + eps)).clip(-1.0, 1.0)

        # === DIRECTIONAL PRESSURE (replaces bid/ask) ===
        range_ = df['high'] - df['low'] + eps
        df['pressure_up'] = (df['close'] - df['open']).clip(lower=0) / range_
        df['pressure_down'] = (df['open'] - df['close']).clip(lower=0) / range_

        # === HANDLE NaN VALUES ===
        df = df.ffill()

        # Fill remaining NaNs with neutral defaults
        defaults = {
            'rsi_14': 50.0, 'adx_14': 20.0, 'stoch_k': 50.0, 'stoch_d': 50.0,
            'bb_position': 0.5, 'bb_width': 0.02,
            'cmf': 0.0, 'pressure_up': 0.0, 'pressure_down': 0.0,
            'sma_distance': 0.0, 'atr_pct': 0.01, 'psar_position': 0.0
        }
        for col, default in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        
        return df

    def _load_all(self):
        print(f"\n[DataSync] Syncing Timeframes for {self.symbol}...")
        for tf in self.timeframes:
            path = os.path.join("data", "spot", f"{self.symbol}_{tf}.csv")
            
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['timestamp'])
                
                # Strip TZ awareness for consistent comparison
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate all indicators for this timeframe
                df = self._calculate_indicators(df, tf)
                
                self.data_store[tf] = df
                print(f"  -> {tf} Loaded & Normalized (Naive) + Indicators")
            else:
                print(f"  !! Missing {tf} at {path}")

    def get_snapshot(self, current_time):
        # Ensure the incoming clock time is also naive for comparison
        if current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)
            
        snapshot = {}
        for tf, df in self.data_store.items():
            # Standard "Look-back" mask
            mask = df.index <= current_time
            if mask.any():
                snapshot[tf] = df.loc[mask].iloc[-1].to_dict()

            else:
                snapshot[tf] = None
        return snapshot
