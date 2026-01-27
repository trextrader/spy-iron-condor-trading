import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pandas_ta as ta
import logging

from intelligence.indicators.manifold_volatility import (
    curvature_proxy_from_returns,
    volatility_energy_from_curvature,
    dynamic_rsi,
)
# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Feature Engineering Parity Constants (Must match train_mamba.py)
INSTITUTIONAL_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 
    'strike', 'cp_num', 'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te',
    'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark'
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class DeepMamba(nn.Module):
    def __init__(self, d_model=1024, n_layers=32, d_state=32, d_conv=4, expand=2, input_dim=24):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1) # Predicts target (e.g., Log Return or Max DD)

    def forward(self, x):
        # x: (Batch, SeqLen, InputDim)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output_head(x[:, -1, :]) # Return last state

class MockMambaKernel:
    """Fallback for CPU/Non-Mamba environments."""
    def __init__(self, d_model=1024):
        self.d_model = d_model
    def __call__(self, x):
        return x # Identity transform for mock

class MambaForecastEngine:
    def __init__(self, d_model=1024, n_layers=32, lookback=240, input_dim=24):
        self.d_model = d_model
        self.n_layers = n_layers
        self.lookback = lookback
        self.input_dim = input_dim
        self.is_cuda = torch.cuda.is_available()
        
        # Init Model
        if HAS_MAMBA and self.is_cuda:
            try:
                self.model = DeepMamba(
                    d_model=d_model, 
                    n_layers=n_layers, 
                    input_dim=input_dim
                ).cuda()
                
                # Load Priorities
                weights_candidates = [
                    os.path.join("models", "mamba_strategy_selector.pth"),
                    os.path.join("models", "mamba_m5_active.pth"),
                    os.path.join("models", "mamba_active.pth")
                ]
                
                weights_path = None
                for path in weights_candidates:
                    if os.path.exists(path):
                        weights_path = path
                        break

                if weights_path:
                    state = torch.load(weights_path)
                    self.model.load_state_dict(state)
                    print(f"[MambaEngine] Loaded Institutional weights: {weights_path}")
                
                self.model.eval()
            except Exception as e:
                logger.error(f"Failed to init Mamba: {e}. Falling back.")
                self.model = None
        else:
            self.model = None
            print("[MambaEngine] Running in MOCK/CPU mode.")

    def prepare_features(self, df: pd.DataFrame, options_df: pd.DataFrame = None) -> np.ndarray:
        """Extract sequence features with Institutional Synchronization."""
        if df.empty:
            return np.zeros((self.lookback, self.input_dim))
            
        window = df.iloc[-self.lookback:].copy()
        if len(window) < self.lookback:
            pad = pd.concat([window.iloc[0:1]] * (self.lookback - len(window)), ignore_index=True)
            window = pd.concat([pad, window], ignore_index=True)

        # Columns
        main_close = 'SPY_close' if 'SPY_close' in window.columns else 'close'
        main_open = 'SPY_open' if 'SPY_open' in window.columns else 'open'
        main_high = 'SPY_high' if 'SPY_high' in window.columns else 'high'
        main_low = 'SPY_low' if 'SPY_low' in window.columns else 'low'
        main_vol = 'SPY_volume' if 'SPY_volume' in window.columns else 'volume'

        # Indicators
        log_ret = np.log(window[main_close]).diff()
        curvature = curvature_proxy_from_returns(log_ret, span=64)
        vol_energy = volatility_energy_from_curvature(curvature)
        window['rsi'] = dynamic_rsi(window[main_close], window=12, vol_energy=vol_energy)
        window['atr'] = ta.atr(window[main_high], window[main_low], window[main_close], length=12)
        adx_df = ta.adx(window[main_high], window[main_low], window[main_close], length=12)
        window['adx'] = adx_df.iloc[:, 0] if adx_df is not None else np.nan
        bbands = ta.bbands(window[main_close], length=12)
        if bbands is not None:
            window['bb_lower'] = bbands.iloc[:, 0]
            window['bb_upper'] = bbands.iloc[:, 2]
        window['stoch_k'] = ta.stoch(window[main_high], window[main_low], window[main_close], k=12).iloc[:, 0]
        window['sma'] = ta.sma(window[main_close], length=12)
        psar = ta.psar(window[main_high], window[main_low], window[main_close])
        if psar is not None:
            window['psar'] = psar.iloc[:, 0].fillna(psar.iloc[:, 1])
            window['psar_mark'] = np.where(psar.iloc[:, 0].isna(), 1.0, -1.0)
        else:
            window['psar_mark'] = 0.0

        # Institutional Alpha features
        if 'ivr' not in window.columns:
            rets = np.log(window[main_close] / window[main_close].shift(1))
            rv = rets.rolling(window=12).std() * np.sqrt(252 * 390)
            window['ivr'] = (rv - rv.min()) / (rv.max() - rv.min() + 1e-6) * 100.0
            
        if 'spread_ratio' not in window.columns:
            window['spread_ratio'] = 0.002
                
        if 'cp_num' not in window.columns:
            if 'call_put' in window.columns:
                window['cp_num'] = window['call_put'].map({'C':1, 'P':-1, 'call':1, 'put':-1}).fillna(0)
            else:
                window['cp_num'] = 0.0

        # Fill Greeks if missing
        for col in ['strike', 'delta', 'gamma', 'vega', 'theta', 'iv', 'te']:
            if col not in window.columns: window[col] = 0.0

        window[INSTITUTIONAL_FEATURES] = window[INSTITUTIONAL_FEATURES].ffill().fillna(0)
        return window[INSTITUTIONAL_FEATURES].values.astype(np.float32)

    def predict(self, df: pd.DataFrame, options_df: pd.DataFrame = None) -> float:
        """Single-step inference."""
        if self.model is None:
            return 0.0
            
        X = self.prepare_features(df, options_df)
        X_tensor = torch.from_numpy(X).unsqueeze(0).cuda()
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred = self.model(X_tensor)
        return pred.item()
