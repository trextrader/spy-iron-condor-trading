from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import mamba_ssm (Linux/CUDA only)
# If missing, we fall back to the MockMamba kernel for CPU inference
try:
    import torch
    import torch.nn as nn
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    # Mock torch/nn for CPU fallback compatibility (typing mostly)
    if 'torch' not in globals():
        torch = Any
        nn = Any

logger = logging.getLogger(__name__)


@dataclass
class ForecastState:
    """Output of the neural forecasting engine."""
    
    # Probabilities for market direction
    prob_bear: float
    prob_neutral: float
    prob_bull: float
    
    # Predicted volatility regime
    pred_vol_regime: str  # 'LOW', 'MEDIUM', 'HIGH'
    
    # Confidence score of the model
    confidence: float
    
    # Raw value (e.g., predicted return)
    raw_output: float
    
    # Backend source (MOCK_CPU or CUDA_REAL)
    model_backend: str = "UNKNOWN"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction_probs": [self.prob_bear, self.prob_neutral, self.prob_bull],
            "vol_regime": self.pred_vol_regime,
            "neural_conf": self.confidence,
            "raw_pred": self.raw_output
        }


class MockMambaKernel:
    """CPU-compatible simulation of Mamba 2 inference behavior."""
    
    def __init__(self, d_model: int = 64, d_state: int = 16):
        self.d_model = d_model
        self.d_state = d_state
        self.hidden_state = np.zeros((d_model, d_state))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.hidden_state = 0.9 * self.hidden_state + 0.1 * np.expand_dims(x, axis=-1)
        out = np.mean(self.hidden_state, axis=-1)
        return out


class DeepMamba(nn.Module):
    """Deep Mamba Network for Financial Time-Series."""
    def __init__(self, d_model: int, n_layers: int = 4, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1) # Regressor head
        
        # Initialize head with small weights to avoid exploding gradients initially
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, L, D)
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Take last time step
        last_state = x[:, -1, :] # (B, D)
        out = self.head(last_state) # (B, 1)
        return torch.tanh(out) # Squash to -1..1


class MambaForecastEngine:
    """Next-Gen Neural Forecasting using Mamba 2 Architecture."""
    
    def __init__(self, d_model: int = 1024, lookback: int = 60, layers: int = 32):
        self.d_model = d_model
        self.lookback = lookback
        self.layers = layers
        self.is_cuda = HAS_MAMBA and torch.cuda.is_available()
        self.kernel = None
        self.model = None
        
        self._initialize_model()
        
    def _initialize_model(self):
        if self.is_cuda:
            logger.info(f"Initializing CUDA DeepMamba Kernel (d_model={self.d_model}, layers={self.layers})...")
            try:
                self.model = DeepMamba(
                    d_model=self.d_model, 
                    n_layers=self.layers,
                    d_state=32,    # Increased state size
                    d_conv=4,
                    expand=2
                ).cuda()
                self.model.eval() # Inference mode (no dropout etc)
                mem_alloc = torch.cuda.memory_allocated() / 1e6
                logger.info(f"Model loaded on GPU. VRAM used: {mem_alloc:.2f} MB")
            except Exception as e:
                logger.error(f"Failed to init CUDA model: {e}. Fallback to CPU.")
                self.is_cuda = False
                self.kernel = MockMambaKernel(d_model=self.d_model)
        else:
            logger.warning("Mamba-SSM not found or No CUDA. Using MockMambaKernel [CPU-Compatible].")
            self.kernel = MockMambaKernel(d_model=self.d_model)
            
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract sequence features for Mamba.
        Returns shape (Lookback, d_model)
        """
        if df.empty:
            return np.zeros((self.lookback, self.d_model))
            
        # Take last N rows
        window = df.iloc[-self.lookback:].copy()
        
        # If not enough data, pad with first row
        if len(window) < self.lookback:
            pad_len = self.lookback - len(window)
            first_row = window.iloc[0:1]
            padding = pd.concat([first_row] * pad_len, ignore_index=True)
            window = pd.concat([padding, window], ignore_index=True)

        features_seq = []
        
        # Pre-calc columns to avoid loop overhead
        closes = window['close'].values
        opens = window['open'].values if 'open' in window.columns else closes
        # safe access
        rsi = window['rsi_14'].fillna(50.0).infer_objects(copy=False).values if 'rsi_14' in window.columns else np.full(len(window), 50.0)
        atr_pct = window['atr_pct'].fillna(0.01).infer_objects(copy=False).values if 'atr_pct' in window.columns else np.full(len(window), 0.01)
        vol_ratio = window['volume_ratio'].fillna(1.0).infer_objects(copy=False).values if 'volume_ratio' in window.columns else np.full(len(window), 1.0)
        
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]
        
        log_ret = np.log(closes / prev_closes + 1e-9)
        
        for i in range(len(window)):
            # Feature Vector (Dim 4) -> Embedded to D_Model
            feat = [
                log_ret[i] * 100.0,    # Scale up
                (rsi[i] - 50.0) / 10.0,
                atr_pct[i] * 50.0,
                (vol_ratio[i] - 1.0) * 2.0
            ]
            
            # Simple manual embedding (padding)
            # ideally we'd use learnable embedding, but for now we pad
            f_vec = np.zeros(self.d_model)
            f_vec[:len(feat)] = feat
            
            # Add some "positional encoding" simulation (noise or sine) to d_model tail?
            # For now, just features.
            features_seq.append(f_vec)
            
        return np.array(features_seq, dtype=np.float32)

    def predict_state(self, market_data: pd.DataFrame) -> ForecastState:
        """Run inference to predict next-step market state."""
        
        # 1. Feature Engineering
        x_seq = self.prepare_features(market_data) # (L, D)
        
        raw_output = 0.0
        
        # 2. Inference
        if self.is_cuda and self.model is not None:
            with torch.no_grad():
                # Convert to Tensor (B=1, L, D)
                x_tensor = torch.from_numpy(x_seq).unsqueeze(0).cuda()
                
                # Forward
                try:
                    out = self.model(x_tensor) # (1, 1) due to tanh head
                    raw_output = float(out.item())
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    raw_output = 0.0
        else:
            # Run Mock Kernel (just use last step)
            last_vec = x_seq[-1]
            out_vec = self.kernel.forward(last_vec)
            raw_output = np.tanh(out_vec[0])
            
        # 3. Interpret Output
        # Signal > 0.2 => Bullish, < -0.2 => Bearish
        
        p_bull = 0.33
        p_bear = 0.33
        p_neutral = 0.34
        
        # Add some gain to raw_output for probability separation
        score = raw_output
        
        if score > 0.15:
            p_bull = 0.6 + min(0.3, score * 0.5)
            p_neutral = max(0.05, 1.0 - p_bull - 0.1)
            p_bear = 1.0 - p_bull - p_neutral
        elif score < -0.15:
            p_bear = 0.6 + min(0.3, abs(score) * 0.5)
            p_neutral = max(0.05, 1.0 - p_bear - 0.1)
            p_bull = 1.0 - p_bear - p_neutral
        else:
            p_neutral = 0.8
            p_bull = 0.1
            p_bear = 0.1
            
        # Regime detection
        vol_score = abs(score)
        if vol_score > 0.7:
            regime = 'HIGH'
        elif vol_score > 0.3:
            regime = 'MEDIUM'
        else:
            regime = 'LOW'
            
        return ForecastState(
            prob_bear=p_bear,
            prob_neutral=p_neutral,
            prob_bull=p_bull,
            pred_vol_regime=regime,
            confidence=0.5 + (abs(score) / 2),
            raw_output=score,
            model_backend="CUDA_DEEP_MAMBA" if self.is_cuda else "MOCK_CPU"
        )
