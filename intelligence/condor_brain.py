"""
CondorBrain: Advanced Multi-Output Mamba 2 Architecture for Iron Condor Optimization

This module implements a specialized neural architecture that directly outputs
Iron Condor trading parameters instead of just price predictions.

Enhancements (2026-01-15):
- VolGatedAttn: Dynamic volatility-gated attention after layers 8, 16, 24
- TopKMoE: Sparse mixture-of-experts for regime-specialized output
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# Import enhancement modules
from intelligence.vol_gated_attn import VolGatedAttn
from intelligence.topk_moe import TopKMoE, BatchedTopKMoE


# ============================================================================
# BF16-FRIENDLY RMSNORM (Critical for throughput)
# ============================================================================

class RMSNorm(nn.Module):
    """
    BF16-friendly RMSNorm (no mean subtraction; stable without FP32 casts).
    y = x * rsqrt(mean(x^2) + eps) * weight
    
    Unlike nn.RMSNorm, this returns the same dtype as input (BF16 stays BF16).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep stats in FP32 for stability, but return in original dtype
        dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = (x_fp32 * rms).to(dtype)
        return y * self.weight.to(dtype)

# ============================================================================
# OUTPUT DATA STRUCTURES
# ============================================================================

@dataclass
class CondorSignal:
    """Output structure for Iron Condor predictions with price trajectory."""
    # Iron Condor Parameters
    short_call_offset: float   # % distance from spot for short call
    short_put_offset: float    # % distance from spot for short put
    wing_width: float          # $ width of wings
    dte_selection: float       # Optimal days to expiry
    prob_profit: float         # Probability of profit (0-1)
    expected_roi: float        # Expected ROI if trade executes
    max_loss_pct: float        # Predicted max loss scenario (0-1)
    confidence: float          # Model confidence (0-1)
    regime: str                # Detected regime: 'low', 'normal', 'high'
    
    # Multi-Horizon Price Trajectory (Optional)
    daily_forecast: Optional[np.ndarray] = None  # (num_days, 4) [close, high, low, vol]
    max_high_pct: float = 0.0   # Max % price could go above current over DTE
    max_low_pct: float = 0.0    # Max % price could go below current over DTE
    
    def is_valid_trade(self, min_confidence: float = 0.6, min_pop: float = 0.5) -> bool:
        """Check if this signal meets minimum trade criteria."""
        return self.confidence >= min_confidence and self.prob_profit >= min_pop
    
    def strikes_are_safe(self, spot_price: float) -> Tuple[bool, bool]:
        """
        Check if predicted price range stays within strike boundaries.
        
        Returns:
            (call_safe, put_safe): True if strikes are outside predicted range
        """
        call_strike = spot_price * (1 + self.short_call_offset / 100)
        put_strike = spot_price * (1 - self.short_put_offset / 100)
        
        predicted_max = spot_price * (1 + self.max_high_pct / 100)
        predicted_min = spot_price * (1 - self.max_low_pct / 100)
        
        call_safe = predicted_max < call_strike
        put_safe = predicted_min > put_strike
        
        return call_safe, put_safe
    
    def to_dict(self) -> dict:
        return {
            'short_call_offset': self.short_call_offset,
            'short_put_offset': self.short_put_offset,
            'wing_width': self.wing_width,
            'dte_selection': self.dte_selection,
            'prob_profit': self.prob_profit,
            'expected_roi': self.expected_roi,
            'max_loss_pct': self.max_loss_pct,
            'confidence': self.confidence,
            'regime': self.regime,
            'max_high_pct': self.max_high_pct,
            'max_low_pct': self.max_low_pct
        }


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class RegimeDetector(nn.Module):
    """Classifies current market regime based on IVR and volatility features."""
    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # Low, Normal, High (Logits)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model) - the last hidden state
        return self.classifier(x)  # (B, 3)

# ============================================================================
# MULTI-HORIZON FORECASTER (Price Trajectory Prediction)
# ============================================================================

class HorizonForecaster(nn.Module):
    """
    Predicts future price trajectory up to max_horizon days ahead.
    
    This allows the model to 'see' the entire life of an options contract
    and make informed strike selections based on predicted price range.
    """
    
    def __init__(self, d_model: int = 1024, max_horizon: int = 45, bars_per_day: int = 390):
        super().__init__()
        self.max_horizon = max_horizon
        self.bars_per_day = bars_per_day
        
        # Horizon encoding
        self.horizon_embed = nn.Embedding(max_horizon, 64)
        
        # Recurrent forecaster (predicts daily summaries)
        self.forecast_rnn = nn.GRU(
            input_size=d_model + 64,  # Hidden state + horizon embedding
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output heads per day
        self.price_head = nn.Linear(512, 4)  # [expected_close, high, low, volatility]
        self.range_head = nn.Linear(512, 2)  # [max_high_pct, max_low_pct] from current
        
    def forward(self, hidden: torch.Tensor, num_days: int = 45) -> dict:
        """
        Predict price trajectory for next num_days.
        
        Args:
            hidden: (B, d_model) - hidden state from backbone
            num_days: Number of days to forecast (typically matches DTE)
            
        Returns:
            dict with:
                - daily_forecast: (B, num_days, 4) - [close, high, low, vol] per day
                - max_range: (B, 2) - max high/low % over entire horizon
        """
        B = hidden.size(0)
        device = hidden.device
        
        # Generate horizon embeddings
        day_indices = torch.arange(num_days, device=device).unsqueeze(0).expand(B, -1)
        day_embed = self.horizon_embed(day_indices)  # (B, num_days, 64)
        
        # Expand hidden for each day
        hidden_expanded = hidden.unsqueeze(1).expand(-1, num_days, -1)  # (B, num_days, d_model)
        
        # Concat and forecast
        rnn_input = torch.cat([hidden_expanded, day_embed], dim=-1)  # (B, num_days, d_model+64)
        # cuDNN GRU requires FP32 contiguous input (BF16 not supported)
        rnn_input = rnn_input.float().contiguous()
        rnn_out, _ = self.forecast_rnn(rnn_input)  # (B, num_days, 512)
        
        # Daily predictions
        daily_forecast = self.price_head(rnn_out)  # (B, num_days, 4)
        
        # Max range over horizon (for strike selection)
        max_range = self.range_head(rnn_out[:, -1, :])  # Use final day's hidden
        
        return {
            'daily_forecast': daily_forecast,
            'max_range': max_range
        }

# ============================================================================
# EXPERT HEADS (Regime-Specific)
# ============================================================================

class CondorExpertHead(nn.Module):
    """Single expert head that outputs 8 IC parameters."""
    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # 8 outputs
        )
        
        # Output activations (applied post-forward)
        self.offset_activation = nn.Sigmoid()  # Constrain to 0-1, then scale
        self.prob_activation = nn.Sigmoid()    # 0-1 probabilities
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.fc(x)  # (B, 8)
        
        # Apply appropriate activations per output
        out = torch.zeros_like(raw)
        out[:, 0] = self.offset_activation(raw[:, 0]) * 5.0    # short_call_offset: 0-5%
        out[:, 1] = self.offset_activation(raw[:, 1]) * 5.0    # short_put_offset: 0-5%
        out[:, 2] = self.offset_activation(raw[:, 2]) * 10.0   # wing_width: 0-$10
        out[:, 3] = 2 + self.offset_activation(raw[:, 3]) * 43  # dte: 2-45 days
        out[:, 4] = self.prob_activation(raw[:, 4])            # prob_profit: 0-1
        out[:, 5] = torch.tanh(raw[:, 5]) * 0.5                # expected_roi: -50% to +50%
        out[:, 6] = self.prob_activation(raw[:, 6])            # max_loss_pct: 0-1
        out[:, 7] = self.prob_activation(raw[:, 7])            # confidence: 0-1
        
        return out

# ============================================================================
# CONDORBRAIN: MAIN ARCHITECTURE
# ============================================================================

class CondorBrain(nn.Module):
    """
    Advanced Mamba 2 architecture for direct Iron Condor optimization.
    
    Features:
    - 32-layer Selective State Space backbone
    - 3 regime-specific expert heads (Low/Normal/High volatility)
    - Gated mixture-of-experts output
    - 8-dimensional output for complete IC parameterization
    """
    
    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 32,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        input_dim: int = 24,
        use_vol_gated_attn: bool = True,
        vol_attn_layers: List[int] = None,
        use_topk_moe: bool = False,
        moe_n_experts: int = 3,
        moe_k: int = 1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.use_vol_gated_attn = use_vol_gated_attn
        self.use_topk_moe = use_topk_moe
        
        # Default attention insertion points: after layers 8, 16, 24 (0-indexed: 7, 15, 23)
        if vol_attn_layers is None:
            # For 32 layers: [7, 15, 23]; for 24 layers: [7, 15, 23]; etc.
            self.vol_attn_layers = [i for i in [7, 15, 23] if i < n_layers]
        else:
            self.vol_attn_layers = vol_attn_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Mamba backbone (32 layers)
        if HAS_MAMBA:
            self.layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                ) for _ in range(n_layers)
            ])
        else:
            # Mock layers for CPU testing
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(n_layers)
            ])
        
        # Volatility-gated attention modules (inserted after specified layers)
        if use_vol_gated_attn:
            self.vol_attn_modules = nn.ModuleDict({
                str(layer_idx): VolGatedAttn(d_model, n_heads=8)
                for layer_idx in self.vol_attn_layers
            })
            logger.info(f"[CondorBrain] VolGatedAttn enabled after layers: {self.vol_attn_layers}")
        else:
            self.vol_attn_modules = None
        
        # Use our custom BF16-friendly RMSNorm (returns same dtype as input)
        self.norm = RMSNorm(d_model)
        
        # Regime detector
        self.regime_detector = RegimeDetector(d_model)
        
        # Multi-horizon forecaster (price trajectory up to 45 days)
        self.horizon_forecaster = HorizonForecaster(d_model, max_horizon=45)
        
        # Output heads: TopKMoE or traditional 3 experts
        if use_topk_moe:
            self.moe_head = BatchedTopKMoE(
                d_model=d_model,
                output_dim=8,
                n_experts=moe_n_experts,
                k=moe_k
            )
            logger.info(f"[CondorBrain] TopKMoE enabled: {moe_n_experts} experts, k={moe_k}")
            self.expert_low = None
            self.expert_normal = None
            self.expert_high = None
        else:
            self.moe_head = None
            # 3 Expert heads (Low, Normal, High volatility)
            self.expert_low = CondorExpertHead(d_model)
            self.expert_normal = CondorExpertHead(d_model)
            self.expert_high = CondorExpertHead(d_model)
        
        # Legacy single-output head (for backward compatibility)
        self.legacy_head = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, return_regime: bool = True, return_experts: bool = False, forecast_days: int = 0) -> Any:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, SeqLen, InputDim)
            return_regime: If True, also return regime probabilities
            return_experts: If True, also return discrete expert outputs
            forecast_days: If > 0, generate price trajectory for this many days
            
        Returns:
            outputs: (B, 8) IC parameters
            regime_probs: (B, 3) regime probabilities [Low, Normal, High]
            horizon_forecast: dict with daily predictions (if forecast_days > 0)
            experts: dict with discrete expert outputs (if return_experts > 0)
        """
        # Input projection
        x = self.input_proj(x)
        
        # DTYPE PROBE: Track where FP32 upcast happens (one-time)
        if not hasattr(self, '_dtype_sanity_printed'):
            self._dtype_sanity_printed = False
        
        if (not self._dtype_sanity_printed) and x.is_cuda:
            print(f"[DTYPE PROBE] after input_proj: {x.dtype}")
        
        # Pass through Mamba layers (with optional gradient checkpointing)
        for i, layer in enumerate(self.layers):
            if self.training and getattr(self, 'gradient_checkpointing', False):
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            
            # Apply VolGatedAttn after specified layers
            if self.use_vol_gated_attn and self.vol_attn_modules is not None:
                if str(i) in self.vol_attn_modules:
                    x = self.vol_attn_modules[str(i)](x)
            
            # Print dtype after first layer only
            if i == 0 and (not self._dtype_sanity_printed) and x.is_cuda:
                print(f"[DTYPE PROBE] after layer[0] (Mamba): {x.dtype}")
        
        # RMSNorm is BF16-friendly, no cast needed
        x = self.norm(x)
        
        if (not self._dtype_sanity_printed) and x.is_cuda:
            print(f"[DTYPE PROBE] after RMSNorm: {x.dtype}")
            self._dtype_sanity_printed = True
        
        # Take last timestep
        last_hidden = x[:, -1, :]  # (B, d_model)
        
        # Detect regime (logits)
        regime_logits = self.regime_detector(last_hidden)  # (B, 3)
        
        # Probabilities for weighting experts
        # Softmax often promotes to FP32 internally; cast back to BF16 so expert mix stays BF16
        regime_probs = torch.softmax(regime_logits.float(), dim=-1).to(last_hidden.dtype)
        
        # Multi-horizon price forecast (if requested)
        horizon_forecast = None
        if forecast_days > 0:
            horizon_forecast = self.horizon_forecaster(last_hidden, num_days=forecast_days)
        
        # Get expert outputs (TopKMoE or traditional 3-expert MoE)
        if self.use_topk_moe and self.moe_head is not None:
            # Sparse TopKMoE output
            outputs = self.moe_head(last_hidden)  # (B, 8)
        else:
            # Traditional 3-expert weighted MoE
            out_low = self.expert_low(last_hidden)       # (B, 8)
            out_normal = self.expert_normal(last_hidden)  # (B, 8)
            out_high = self.expert_high(last_hidden)      # (B, 8)
            
            # Ensure experts stay in same dtype as hidden (BF16)
            if out_low.dtype != last_hidden.dtype:
                out_low = out_low.to(last_hidden.dtype)
                out_normal = out_normal.to(last_hidden.dtype)
                out_high = out_high.to(last_hidden.dtype)
            
            # Mixture-of-experts: weighted sum by regime probabilities (BF16 end-to-end)
            outputs = (
                regime_probs[:, 0:1] * out_low +
                regime_probs[:, 1:2] * out_normal +
                regime_probs[:, 2:3] * out_high
            )
        
        # Return package
        res = [outputs]
        if return_regime:
            res.append(regime_logits)
        else:
            res.append(None)
            
        res.append(horizon_forecast)
        
        if return_experts:
            # TopKMoE doesn't have discrete expert outputs
            if self.use_topk_moe:
                res.append(None)
            else:
                res.append({
                    'low': out_low,
                    'normal': out_normal,
                    'high': out_high
                })
            
        return tuple(res) if len(res) > 1 else outputs
    
    def predict_legacy(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy single-output prediction for backward compatibility."""
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        # Final normalization stays in model dtype (BF16) for throughput; loss is cast to FP32 externally
        x = self.norm(x)
        return self.legacy_head(x[:, -1, :])

# ============================================================================
# CONDOR LOSS FUNCTION
# ============================================================================

class CondorLoss(nn.Module):
    """
    Multi-objective loss function for CondorBrain training.
    
    Components:
    - Strike accuracy (offset predictions vs optimal)
    - P&L alignment (ROI prediction vs realized)
    - Risk calibration (max loss prediction)
    - Probability calibration (Brier score for POP)
    - Regime consistency (auxiliary loss)
    """
    
    def __init__(
        self,
        strike_weight: float = 1.0,
        pnl_weight: float = 2.0,
        risk_weight: float = 1.5,
        prob_weight: float = 1.0,
        regime_weight: float = 0.5
    ):
        super().__init__()
        self.strike_weight = strike_weight
        self.pnl_weight = pnl_weight
        self.risk_weight = risk_weight
        self.prob_weight = prob_weight
        self.regime_weight = regime_weight
        
        self.huber = nn.HuberLoss(delta=1.0)
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None,
        regime_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute composite loss.
        
        Args:
            pred: (B, 8) predicted IC parameters
            target: (B, 8) target IC parameters
            regime_probs: (B, 3) predicted regime probabilities
            regime_labels: (B,) ground truth regime indices
        """
        # Strike offset loss (indices 0, 1)
        l_strike = self.huber(pred[:, 0:2], target[:, 0:2])
        
        # Wing width loss (index 2)
        l_width = self.huber(pred[:, 2], target[:, 2])
        
        # DTE loss (index 3)
        l_dte = self.huber(pred[:, 3], target[:, 3])
        
        # P&L alignment (index 5 = expected_roi)
        l_pnl = self.huber(pred[:, 5], target[:, 5])
        
        # Risk calibration (index 6 = max_loss_pct)
        l_risk = self.huber(pred[:, 6], target[:, 6])
        
        # Probability calibration - Brier score for POP (index 4)
        l_prob = torch.mean((pred[:, 4] - target[:, 4]) ** 2)
        
        # Regime loss (if labels provided)
        l_regime = torch.tensor(0.0, device=pred.device)
        if regime_probs is not None and regime_labels is not None:
            l_regime = self.ce(regime_probs, regime_labels)
        
        # Combine
        total_loss = (
            self.strike_weight * (l_strike + l_width + l_dte) +
            self.pnl_weight * l_pnl +
            self.risk_weight * l_risk +
            self.prob_weight * l_prob +
            self.regime_weight * l_regime
        )
        
        return total_loss

# ============================================================================
# INFERENCE ENGINE (PRODUCTION OPTIMIZED)
# ============================================================================

class CondorBrainEngine:
    """
    High-performance inference engine for CondorBrain.
    
    Optimizations:
    - torch.compile() for JIT fusion (PyTorch 2.0+)
    - FP16/BF16 inference for 2x memory & speed
    - CUDA warmup to eliminate cold-start latency
    - Single-pass inference (no redundant forward calls)
    - Early-exit confidence filtering (like bloom filter pre-check)
    - Pre-allocated tensors to avoid allocation overhead
    """
    
    def __init__(
        self,
        model_path: str = None,  # Auto-discover if None
        d_model: int = 1024,
        n_layers: int = 24,  # Default to 24 for v2.2
        input_dim: int = 24,
        lookback: int = 240,
        use_compile: bool = True,
        use_fp16: bool = True,
        warmup_iterations: int = 3,
        # v2.2 Enhancement flags
        use_vol_gated_attn: bool = True,
        use_topk_moe: bool = True,
        moe_n_experts: int = 3,
        moe_k: int = 1
    ):
        self.lookback = lookback
        self.input_dim = input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        
        # Auto-discover model file if not specified
        if model_path is None:
            model_path = self._find_latest_model()
        
        # Initialize model with enhancement flags
        self.model = CondorBrain(
            d_model=d_model,
            n_layers=n_layers,
            input_dim=input_dim,
            use_vol_gated_attn=use_vol_gated_attn,
            use_topk_moe=use_topk_moe,
            moe_n_experts=moe_n_experts,
            moe_k=moe_k
        ).to(self.device)
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)  # strict=False for enhancement compat
            logger.info(f"[CondorBrain] Loaded weights from {model_path}")
        else:
            logger.warning(f"[CondorBrain] No weights found at {model_path}. Using random init.")
        
        self.model.eval()
        
        # === OPTIMIZATION 1: FP16 Inference ===
        if self.use_fp16:
            self.model = self.model.half()
            logger.info("[CondorBrain] FP16 inference enabled (2x memory savings)")
        
        # === OPTIMIZATION 2: torch.compile (PyTorch 2.0+) ===
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("[CondorBrain] torch.compile enabled (JIT fusion)")
            except Exception as e:
                logger.warning(f"[CondorBrain] torch.compile failed: {e}")
        
        # === OPTIMIZATION 3: Pre-allocate tensors ===
        self._dummy_input = torch.zeros(
            (1, lookback, input_dim), 
            dtype=torch.float16 if self.use_fp16 else torch.float32,
            device=self.device
        )
        
        # === OPTIMIZATION 4: CUDA Warmup ===
        if self.device.type == 'cuda' and warmup_iterations > 0:
            logger.info(f"[CondorBrain] Warming up CUDA ({warmup_iterations} iterations)...")
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = self.model(self._dummy_input, return_regime=True, forecast_days=0)
            torch.cuda.synchronize()
            logger.info("[CondorBrain] CUDA warmup complete - ready for low-latency inference")
        
        # Regime names cache
        self._regime_names = ['low', 'normal', 'high']
        
        # Early-exit threshold (like bloom filter k-factor)
        self.min_confidence_threshold = 0.3  # Skip expensive trajectory if confidence < 0.3
    
    @staticmethod
    def _find_latest_model() -> str:
        """Auto-discover the latest CondorBrain model file."""
        import glob
        
        models_dir = "models"
        if not os.path.exists(models_dir):
            return None
        
        # Priority order for model discovery
        candidates = [
            # Exact match
            os.path.join(models_dir, "condor_brain.pth"),
            # Active symlink
            os.path.join(models_dir, "condor_brain_active.pth"),
        ]
        
        # Check priority candidates first
        for path in candidates:
            if os.path.exists(path):
                return path
        
        # Glob for any condor_brain_*.pth files and pick the most recently modified
        pattern = os.path.join(models_dir, "condor_brain_*.pth")
        matches = glob.glob(pattern)
        
        if matches:
            # Sort by modification time, newest first
            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return matches[0]
        
        return None
        
    def predict(
        self, 
        features: np.ndarray, 
        forecast_days: int = 14,
        skip_trajectory_if_low_conf: bool = True
    ) -> CondorSignal:
        """
        Lightning-fast inference with optional early-exit.
        
        Args:
            features: (SeqLen, InputDim) numpy array
            forecast_days: Days to forecast (default 14, set 0 for auto)
            skip_trajectory_if_low_conf: Skip expensive horizon calc if confidence < threshold
            
        Returns:
            CondorSignal with all parameters
        """
        # Ensure correct shape
        if len(features.shape) == 2:
            features = features[np.newaxis, ...]
        
        # Convert to tensor with correct dtype
        dtype = torch.float16 if self.use_fp16 else torch.float32
        x = torch.from_numpy(features.astype(np.float32)).to(dtype).to(self.device)
        
        with torch.no_grad():
            # === SINGLE PASS (no redundant forward calls) ===
            # Always run with fixed forecast_days to avoid double-pass
            actual_forecast_days = forecast_days if forecast_days > 0 else 14
            
            outputs, regime_probs, horizon = self.model(
                x, return_regime=True, forecast_days=actual_forecast_days
            )
        
        # Convert to numpy (CPU)
        out = outputs[0].float().cpu().numpy()
        reg = regime_probs[0].float().cpu().numpy()
        
        # Determine regime
        regime_idx = int(np.argmax(reg))
        
        # === EARLY EXIT: Skip trajectory processing if low confidence ===
        confidence = float(out[7])
        daily_forecast = None
        max_high_pct = 0.0
        max_low_pct = 0.0
        
        if confidence >= self.min_confidence_threshold or not skip_trajectory_if_low_conf:
            # Only process trajectory if confidence passes threshold
            if horizon is not None:
                daily_forecast = horizon['daily_forecast'][0].float().cpu().numpy()
                max_range = horizon['max_range'][0].float().cpu().numpy()
                max_high_pct = float(max_range[0]) * 100.0
                max_low_pct = float(max_range[1]) * 100.0
        
        return CondorSignal(
            short_call_offset=float(out[0]),
            short_put_offset=float(out[1]),
            wing_width=float(out[2]),
            dte_selection=float(out[3]),
            prob_profit=float(out[4]),
            expected_roi=float(out[5]),
            max_loss_pct=float(out[6]),
            confidence=confidence,
            regime=self._regime_names[regime_idx],
            daily_forecast=daily_forecast,
            max_high_pct=max_high_pct,
            max_low_pct=max_low_pct
        )
    
    def predict_batch(self, features_batch: np.ndarray, forecast_days: int = 14) -> list:
        """
        Batch inference for multiple samples (even faster per-sample).
        
        Args:
            features_batch: (Batch, SeqLen, InputDim) numpy array
            
        Returns:
            List of CondorSignal objects
        """
        dtype = torch.float16 if self.use_fp16 else torch.float32
        x = torch.from_numpy(features_batch.astype(np.float32)).to(dtype).to(self.device)
        
        with torch.no_grad():
            outputs, regime_probs, horizon = self.model(
                x, return_regime=True, forecast_days=forecast_days
            )
        
        # Batch convert
        out_np = outputs.float().cpu().numpy()
        reg_np = regime_probs.float().cpu().numpy()
        
        signals = []
        for i in range(len(out_np)):
            regime_idx = int(np.argmax(reg_np[i]))
            
            daily_forecast = None
            max_high_pct = 0.0
            max_low_pct = 0.0
            if horizon is not None:
                daily_forecast = horizon['daily_forecast'][i].float().cpu().numpy()
                max_range = horizon['max_range'][i].float().cpu().numpy()
                max_high_pct = float(max_range[0]) * 100.0
                max_low_pct = float(max_range[1]) * 100.0
            
            signals.append(CondorSignal(
                short_call_offset=float(out_np[i, 0]),
                short_put_offset=float(out_np[i, 1]),
                wing_width=float(out_np[i, 2]),
                dte_selection=float(out_np[i, 3]),
                prob_profit=float(out_np[i, 4]),
                expected_roi=float(out_np[i, 5]),
                max_loss_pct=float(out_np[i, 6]),
                confidence=float(out_np[i, 7]),
                regime=self._regime_names[regime_idx],
                daily_forecast=daily_forecast,
                max_high_pct=max_high_pct,
                max_low_pct=max_low_pct
            ))
        
        return signals
    
    def benchmark(self, n_iterations: int = 100) -> dict:
        """Benchmark inference latency."""
        import time
        
        # Warmup
        for _ in range(10):
            _ = self.model(self._dummy_input, return_regime=True, forecast_days=14)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(self._dummy_input, return_regime=True, forecast_days=14)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p99_ms': np.percentile(times, 99)
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'CondorBrain',
    'CondorBrainEngine', 
    'CondorSignal',
    'CondorLoss',
    'HorizonForecaster',
    'HAS_MAMBA'
]
