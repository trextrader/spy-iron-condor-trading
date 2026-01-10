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
                
                # Load Learned Brain
                weights_path = os.path.join("models", "mamba_active.pth")
                if os.path.exists(weights_path):
                    try:
                        state = torch.load(weights_path)
                        self.model.load_state_dict(state)
                        print(f"[MambaEngine] Loaded trained weights from {weights_path}")
                    except Exception as e:
                        print(f"[Warning] Failed to load weights: {e}")
                else:
                    print("[MambaEngine] Warning: No trained weights found. Using RANDOM initialization (Untrained).")

                self.model.eval() # Inference mode (no dropout etc)
                
                # Warmup pass to force allocation
                dummy_input = torch.zeros(1, self.lookback, self.d_model).cuda()
                _ = self.model(dummy_input)
                
                mem_alloc = torch.cuda.memory_allocated() / 1e6
                print(f"[MambaEngine] Model loaded on GPU. VRAM used: {mem_alloc:.2f} MB")
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

    def precompute_all(self, df: pd.DataFrame, batch_size: int = 4096) -> pd.DataFrame:
        """Run batch inference on the entire dataset to maximize GPU usage.
        
        Returns:
            DataFrame with columns ['mamba_bull', 'mamba_bear', 'mamba_neutral', 'mamba_conf', 'mamba_raw']
            indexed matching the input df.
        """
        if df.empty:
            return pd.DataFrame()

        print(f"[MambaEngine] Pre-computing signals for {len(df)} bars (Batch Size: {batch_size})...")
        
        # 1. Vectorized Feature Preparation
        # Create rolling window features efficiently
        
        # We need sequences of length 'lookback' for each time step
        # This is memory intensive if we naïvely duplicate. 
        # But Mamba is efficient. Let's create a rolling strided view or just iterate efficiently.
        # For GPU speed, we'll prep batches of sequences.
        
        closes = df['close'].values.astype(np.float32)
        
        # Pre-calc normalized features
        if 'rsi_14' in df.columns:
            rsi = (df['rsi_14'].fillna(50.0).values - 50.0) / 10.0
        else:
            rsi = np.zeros_like(closes)
            
        if 'atr_pct' in df.columns:
            atr = df['atr_pct'].fillna(0.01).values * 50.0
        else:
            atr = np.zeros_like(closes)
            
        if 'volume_ratio' in df.columns:
            vol = (df['volume_ratio'].fillna(1.0).values - 1.0) * 2.0
        else:
            vol = np.zeros_like(closes)

        # Log returns
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]
        log_rets = np.log(closes / (prev_closes + 1e-9)) * 100.0
        
        # Stack features: (N, 4)
        data_matrix = np.stack([log_rets, rsi, atr, vol], axis=1).astype(np.float32)
        
        # Pad to d_model directly? 
        # No, we embed (pad) during batch creation to save RAM 
        
        results = []
        
        # Iterate in batches
        total_rows = len(df)
        
        # We can't generate forecast for first 'lookback' bars easily (padding needed)
        # We'll valid-pad or similar.
        
        # Prepare Tensor Batches
        # Optimized: Create a sliding window view using torch.unfold if possible, 
        # or simple loop. Sliding window of (Batch, Time, Feat)
        
        # Convert to Tensor
        full_tensor = torch.from_numpy(data_matrix) # (N, 4)
        if self.is_cuda:
            full_tensor = full_tensor.cuda()
            
        # We need (B, L, D). 
        # Create indices for the batch
        indices = np.arange(total_rows)
        
        preds_list = []
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            current_batch_size = end_idx - start_idx
            
            # Construct batch of sequences (Slowest part if done in python loop)
            # We need sequences [t-L+1 : t+1] for each t in batch
            # Doing this efficiently:
            
            # Just grab a slice [start-lookback : end] and unfold it
            slice_start = max(0, start_idx - self.lookback + 1)
            slice_end = end_idx
            
            # Grab context + batch data
            # (Context + Batch, Feat)
            sub_data = full_tensor[slice_start:slice_end] 
            
            # Pad left if at start of file
            if slice_start == 0 and start_idx < self.lookback:
                 pad_amt = (self.lookback - 1) - start_idx
                 if pad_amt > 0:
                     # (Pad, Feat)
                     padding = torch.zeros((pad_amt, 4), device=sub_data.device)
                     sub_data = torch.cat([padding, sub_data], dim=0)
            
            # Now unfold: (Batch, Lookback, Feat)
            # sub_data is roughly Batch + Lookback length
            # unfold(dim, size, step)
            try:
                # Unfold creates (Batch, Feat, Lookback) or similar? 
                # unfold(dimension, size, step)
                # We want a window of size 'lookback' sliding with step 1
                windows = sub_data.unfold(0, self.lookback, 1) 
                
                # windows shape: (Batch, Feat, Lookback) -> need (Batch, Lookback, Feat)
                windows = windows.permute(0, 2, 1)
                
                # Pad features to d_model (Batch, Lookback, d_model)
                B, L, F = windows.shape
                if F < self.d_model:
                     x_input = torch.zeros((B, L, self.d_model), device=windows.device)
                     x_input[:, :, :F] = windows
                else:
                     x_input = windows
                     
                # Inference
                if self.is_cuda and self.model:
                    with torch.no_grad():
                         out = self.model(x_input) # (B, 1)
                         preds_list.extend(out.cpu().numpy().flatten())
                else:
                     # Mock fallback
                     preds_list.extend(np.zeros(B))
                     
            except Exception as e:
                # Fallback for shape errors
                print(f"Batch Error: {e}")
                preds_list.extend(np.zeros(current_batch_size))

        # Pad results to match df length if needed (unfold might reduce?)
        # Unfold returns N - size + 1. 
        # We carefully constructed input. 
        # If logic matches, len(preds) == len(df).
        # Let's truncate or pad just in case.
        preds = np.array(preds_list)
        if len(preds) < total_rows:
            preds = np.concatenate([np.zeros(total_rows - len(preds)), preds])
        elif len(preds) > total_rows:
            preds = preds[:total_rows]

        # Post-process to probabilities (Vectorized)
        scores = preds
        
        # P_bull
        p_bull = np.where(scores > 0.15, 0.6 + np.minimum(0.3, scores * 0.5), 
                          np.where(scores < -0.15, 1.0 - (0.6 + np.minimum(0.3, np.abs(scores) * 0.5)) - 0.05, 0.1))
        
        # P_bear
        p_bear = np.where(scores < -0.15, 0.6 + np.minimum(0.3, np.abs(scores) * 0.5), 
                          np.where(scores > 0.15, 1.0 - p_bull - 0.05, 0.1))
        
        # Normalize neutral
        p_neutral = 1.0 - p_bull - p_bear
        p_neutral = np.clip(p_neutral, 0.0, 1.0)
        
        # Re-normalize sum to 1
        sums = p_bull + p_bear + p_neutral
        p_bull /= sums
        p_bear /= sums
        p_neutral /= sums
        
        return pd.DataFrame({
            'mamba_bull': p_bull,
            'mamba_bear': p_bear,
            'mamba_neutral': p_neutral,
            'mamba_raw': scores,
            'mamba_conf': 0.5 + (np.abs(scores) / 2.0)
        }, index=df.index)

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
