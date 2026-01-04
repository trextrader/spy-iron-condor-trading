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
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction_probs": [self.prob_bear, self.prob_neutral, self.prob_bull],
            "vol_regime": self.pred_vol_regime,
            "neural_conf": self.confidence,
            "raw_pred": self.raw_output
        }


class MockMambaKernel:
    """CPU-compatible simulation of Mamba 2 inference behavior.
    
    In a real deployment, this would be replaced by the compiled CUDA kernel
    or a pure-PyTorch CPU implementation of the SSM scan.
    
    This mock uses simple heuristics to generate plausible 'forecasts'
    based on the input features, suitable for backtesting pipeline verification.
    """
    
    def __init__(self, d_model: int = 64, d_state: int = 16):
        self.d_model = d_model
        self.d_state = d_state
        self.hidden_state = np.zeros((d_model, d_state))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Simulate a state update and projection.
        
        Args:
            x: Input feature vector of shape (batch, d_model)
            
        Returns:
            Output vector of shape (batch, d_model)
        """
        # Simple recurrent update simulation
        # h_t = 0.9 * h_{t-1} + 0.1 * x
        # This mocks the 'memory' aspect of an SSM
        self.hidden_state = 0.9 * self.hidden_state + 0.1 * np.expand_dims(x, axis=-1)
        
        # Project to output (mean of state)
        out = np.mean(self.hidden_state, axis=-1)
        return out


class MambaForecastEngine:
    """Next-Gen Neural Forecasting using Mamba 2 Architecture."""
    
    def __init__(self, d_model: int = 64, lookback: int = 60):
        self.d_model = d_model
        self.lookback = lookback
        self.is_cuda = HAS_MAMBA
        self.kernel = None
        
        self._initialize_model()
        
    def _initialize_model(self):
        if self.is_cuda:
            logger.info("Initializing CUDA Mamba 2 Kernel...")
            # hypothetical initialization of the real model
            # self.model = Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2).cuda()
            pass
        else:
            logger.warning("Mamba-SSM not found (CUDA missing?). Using MockMambaKernel [CPU-Compatible].")
            self.kernel = MockMambaKernel(d_model=self.d_model)
            
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and normalize features for Mamba.
        
        Features:
        - Log Returns (5m)
        - RSI-14
        - Volatility (ATR %)
        - Volume Ratio
        """
        if df.empty:
            return np.zeros(self.d_model)
            
        last_row = df.iloc[-1]
        
        # safely get features or default to 0
        feat = [
            last_row.get('close', 100.0) / df.iloc[-2]['close'] - 1.0 if len(df) > 1 else 0.0,
            (last_row.get('rsi_14', 50.0) - 50.0) / 50.0,  # Center around 0
            last_row.get('atr_pct', 0.01) * 10.0,          # Scale up
            (last_row.get('volume_ratio', 1.0) - 1.0)      # Center around 0
        ]
        
        # Pad to model dimension
        features = np.array(feat + [0.0] * (self.d_model - len(feat)))
        return features

    def predict_state(self, market_data: pd.DataFrame) -> ForecastState:
        """Run inference to predict next-step market state."""
        
        # 1. Feature Engineering
        x = self.prepare_features(market_data)
        
        # 2. Inference (Mock or Real)
        if self.is_cuda:
            # Code to run real Mamba inference
            raw_output = 0.0 
        else:
            # Run Mock Kernel
            out_vec = self.kernel.forward(x)
            raw_output = np.tanh(out_vec[0]) # Squeeze to scalar signal (-1 to 1)
            
        # 3. Interpret Output
        # Helper logic to convert raw signal to probabilities
        # Signal > 0.3 => Bullish, < -0.3 => Bearish, else Neutral
        
        p_bull = 0.33
        p_bear = 0.33
        p_neutral = 0.34
        
        if raw_output > 0.2:
            p_bull = 0.6 + (raw_output * 0.2)
            p_neutral = 0.3
            p_bear = 0.1
        elif raw_output < -0.2:
            p_bear = 0.6 + (abs(raw_output) * 0.2)
            p_neutral = 0.3
            p_bull = 0.1
        else:
            p_neutral = 0.8
            p_bull = 0.1
            p_bear = 0.1
            
        # Normalize
        total = p_bull + p_neutral + p_bear
        probs = [p_bear/total, p_neutral/total, p_bull/total]
        
        # Regime detection logic (simple heuristic for mock)
        vol_score = abs(raw_output)
        if vol_score > 0.6:
            regime = 'HIGH'
        elif vol_score > 0.3:
            regime = 'MEDIUM'
        else:
            regime = 'LOW'
            
        return ForecastState(
            prob_bear=probs[0],
            prob_neutral=probs[1],
            prob_bull=probs[2],
            pred_vol_regime=regime,
            confidence=0.5 + (abs(raw_output) / 2), # Higher signal = higher confidence
            raw_output=float(raw_output)
        )
