"""
CondorNet™ Training Script

Training pipeline for the mathematically faithful CondorNet architecture.
Implements the 10-component composite loss from the specification.

Usage:
    python intelligence/condor_train_net.py --local-data data/institutional/2024.csv \
        --d-h 256 --d-v 32 --d-m 64 --d-r 32 --epochs 100 --batch-size 128

Author: Claude Code (Opus 4.5)
Version: 1.0.0
Date: 2026-02-03
"""

import sys
import os
import math
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence

# CUDA optimizations before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# GUI Telemetry (optional - only if GUI backend is available)
try:
    from gui.backend.services.training_emitter import get_emitter, init_emitter
    from gui.backend.routers.websocket import get_ws_manager
    GUI_TELEMETRY_AVAILABLE = True
except ImportError:
    GUI_TELEMETRY_AVAILABLE = False
    def get_emitter(backend_url=None):
        return None


def get_backend_url(env: str) -> str:
    """Get the backend URL based on the environment.

    Args:
        env: One of 'local', 'lightai', 'kaggle', 'colab'

    Returns:
        Backend URL for the training emitter
    """
    if env == 'local':
        return "http://localhost:8000"

    elif env == 'lightai':
        # Lightning AI cloudspaces use subdomain-based routing
        # e.g., https://8000-{cloudspace_id}.cloudspaces.litng.ai
        cloudspace_host = os.environ.get('LIGHTNING_CLOUDSPACE_HOST', '')
        if cloudspace_host:
            return f"https://8000-{cloudspace_host}"
        # Fallback: try to construct from teamspace
        teamspace = os.environ.get('LIGHTNING_TEAMSPACE', '')
        if teamspace:
            return f"https://8000-{teamspace}.cloudspaces.litng.ai"
        print("[Warning] LIGHTNING_CLOUDSPACE_HOST not found, using localhost")
        return "http://localhost:8000"

    elif env == 'kaggle':
        # Kaggle notebooks - typically need ngrok or similar tunneling
        # Check for KAGGLE_BACKEND_URL environment variable
        kaggle_url = os.environ.get('KAGGLE_BACKEND_URL', '')
        if kaggle_url:
            return kaggle_url
        print("[Warning] KAGGLE_BACKEND_URL not set, using localhost")
        return "http://localhost:8000"

    elif env == 'colab':
        # Google Colab - typically need ngrok or similar tunneling
        # Check for COLAB_BACKEND_URL environment variable
        colab_url = os.environ.get('COLAB_BACKEND_URL', '')
        if colab_url:
            return colab_url
        print("[Warning] COLAB_BACKEND_URL not set, using localhost")
        return "http://localhost:8000"

    else:
        return "http://localhost:8000"

# Training Diagnostics (for Model Introspection page)
try:
    from intelligence.training.diagnostics import TrainingDiagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    TrainingDiagnostics = None

from intelligence.condor_brain_net import (
    CondorNet,
    group_invariant_loss,
    spectral_radius_loss,
)
from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V22,
    NEUTRAL_FILL_VALUES_V22,
    select_feature_frame,
)

FEATURE_COLS = FEATURE_COLS_V22


# =============================================================================
# COMPOSITE LOSS (10 COMPONENTS)
# =============================================================================

class CompositeCondorNetLoss(nn.Module):
    """
    10-component composite loss for CondorNet training.
    """
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_npdd: float = 1.0,
        lambda_sharpe: float = 0.1,
        lambda_dd: float = 0.3,
        lambda_turnover: float = 0.1,
        lambda_fuzzy: float = 0.2,
        lambda_pattern_ent: float = 0.05,
        lambda_group_inv: float = 0.1,
        lambda_rho: float = 0.1,
        lambda_energy: float = 0.01,
        lambda_growth: float = 0.1,
        use_clamping: bool = True,
    ):
        super().__init__()
        self.use_clamping = use_clamping
        self.lambdas = {
            'mse': lambda_mse,
            'npdd': lambda_npdd,
            'sharpe': lambda_sharpe,
            'dd': lambda_dd,
            'turnover': lambda_turnover,
            'fuzzy': lambda_fuzzy,
            'pattern_ent': lambda_pattern_ent,
            'group_inv': lambda_group_inv,
            'rho': lambda_rho,
            'energy': lambda_energy,
            'growth': lambda_growth,
        }

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        gates: torch.Tensor = None,
        state: torch.Tensor = None,
        A_matrix: torch.Tensor = None,
        pred_signature: Optional[nn.Module] = None,
        returns: torch.Tensor = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # FORCE ALL LOSS MATH TO FP32
        device = predictions.device
        predictions = predictions.float()
        targets = targets.float()
        if gates is not None:
            gates = gates.float()
        if state is not None:
            state = state.float()
        if A_matrix is not None:
            A_matrix = A_matrix.float()
        if returns is not None:
            returns = returns.float()

        # Debug Prints for first batch
        if not hasattr(self, '_printed_loss_dtype_debug'):
            print(f"\n[CompositeCondorNetLoss Debug] Initial Dtypes & Status:")
            print(f"  Predictions: {predictions.dtype}, Targets: {targets.dtype}")
            print(f"  Autocast Active: {torch.is_autocast_enabled()}")
            self._printed_loss_dtype_debug = True

        def debug_val(name, val):
            if not hasattr(self, '_batch0_val_debug'):
                print(f"  [VAL DEBUG] {name}: {val.item() if val.numel()==1 else val.mean().item()} | Inf: {torch.isinf(val).any()} | NaN: {torch.isnan(val).any()}")

        components: Dict[str, torch.Tensor] = {}

        # === 0. Supervised Regression (Iron Condor Alignment) ===
        # Ensure model learns the intended strikes (offsets), widths, and timing
        components['mse'] = F.mse_loss(predictions, targets)
        debug_val('mse', components['mse'])

        # === 1. NPDD Loss (Weighted by Confidence) ===
        # PREPARE WEIGHTED RETURNS FOR GRADIENT FLOW ===
        # Scale realized returns by model confidence to allow gradients to flow
        # Confidence is index 7 in predictions (B, 10)
        z_confidence = torch.sigmoid(predictions[:, 7]).view(1, -1) # (1, B)
        # Weight returns by confidence (Broadcast returns: (1, B))
        w_returns = returns * z_confidence

        # Log-space stability for cumulative metrics
        # log(1+r) avoids exploding cumprod
        log_ret = torch.log1p(w_returns.clamp(-0.9, 5.0))
        cum_log_ret = torch.cumsum(log_ret, dim=-1)
        # V7: Tighten clamp to [-5, 5] (exp(5) is ~148x growth) to prevent trillions-scale loss
        cum_log_ret = cum_log_ret.clamp(-5.0, 5.0)
        cum_ret = torch.exp(cum_log_ret)

        # === 1. NPDD Loss (Weighted by Confidence) ===
        if returns is not None and returns.numel() > 0:
            mean_w_ret = w_returns.mean(dim=-1)
            running_max = torch.cummax(cum_ret, dim=-1)[0]
            dd = (running_max - cum_ret) / (running_max + 1e-6)
            # CLAMP max_dd to min 0.02 (2% floor) to prevent explosion
            max_dd = dd.max(dim=-1)[0].clamp(min=0.02)
            npdd = mean_w_ret / max_dd
            # V11: Soft-Clamping (tanh) preserves gradient flow even at extreme magnitudes
            unclamped_npdd = -npdd.mean()
            if self.use_clamping:
                components['npdd'] = torch.tanh(unclamped_npdd / 50.0) * 50.0
            else:
                components['npdd'] = unclamped_npdd
            debug_val('npdd', components['npdd'])
            debug_val('unclamped_npdd', unclamped_npdd)
        else:
            components['npdd'] = F.mse_loss(predictions, targets)

        # === 2. Sharpe Loss (Weighted) ===
        if returns is not None and returns.shape[-1] > 1:
            mean_w_ret = w_returns.mean(dim=-1)
            # CLAMP volatility to min 0.01 (1% floor) to prevent explosion
            std_w_ret = w_returns.std(dim=-1).clamp(min=0.01)
            sharpe = mean_w_ret / std_w_ret * math.sqrt(252 * 78)
            # V11: Soft-Clamping (tanh) preserves gradient flow even at extreme magnitudes
            unclamped_sharpe = -sharpe.mean()
            if self.use_clamping:
                components['sharpe'] = torch.tanh(unclamped_sharpe / 50.0) * 50.0
            else:
                components['sharpe'] = unclamped_sharpe
            debug_val('sharpe', components['sharpe'])
            debug_val('unclamped_sharpe', unclamped_sharpe)
        else:
            components['sharpe'] = torch.tensor(0.0, device=device)

        # === 3. Drawdown Loss (Weighted) ===
        if returns is not None and returns.numel() > 0:
            # Re-calculate DD on weighted series
            running_max = torch.cummax(cum_ret, dim=-1)[0]
            dd = (running_max - cum_ret) / (running_max + 1e-6)
            max_dd = dd.max(dim=-1)[0]
            components['dd'] = max_dd.mean()
            debug_val('dd', components['dd'])
        else:
            components['dd'] = torch.tensor(0.0, device=device)

        # === 4. Turnover Loss ===
        entry_logits = predictions[:, 8]
        exit_logits = predictions[:, 9]
        components['turnover'] = torch.abs(entry_logits).mean() + torch.abs(exit_logits).mean()
        debug_val('turnover', components['turnover'])

        # === 5. Fuzzy Loss ===
        # Calculate on probabilities (sigmoid) to keep bound [0, 0.25]
        prob_conf = torch.sigmoid(predictions[:, 7])
        components['fuzzy'] = torch.var(prob_conf)
        debug_val('fuzzy', components['fuzzy'])

        # === 6. Pattern Entropy ===
        if gates is not None:
            gate_probs = gates.mean(dim=0).clamp(1e-8, 1 - 1e-8)
            entropy = -(gate_probs * torch.log(gate_probs)).sum()
            components['pattern_ent'] = -entropy
            debug_val('pattern_ent', components['pattern_ent'])
        else:
            components['pattern_ent'] = torch.tensor(0.0, device=device)

        # === 7. Group Invariance ===
        if gates is not None and pred_signature is not None:
            # FORCE gates to FP32 to match pred_signature
            gates_fp32 = gates.float()
            components['group_inv'] = group_invariant_loss(
                pred_signature,
                gates_fp32,
                n_permutations=2,
            ).float()
            debug_val('group_inv', components['group_inv'])
        else:
            components['group_inv'] = torch.tensor(0.0, device=device)

        # === 8. Spectral Radius ===
        if A_matrix is not None:
            components['rho'] = spectral_radius_loss(A_matrix, dt=dt, target_rho=0.99)
            debug_val('rho', components['rho'])
        else:
            components['rho'] = torch.tensor(0.0, device=device)

        # === 9. Energy Loss ===
        if state is not None:
            # Normalize by state dimension
            d_x = state.shape[-1] if state.dim() > 1 else 128
            components['energy'] = (state ** 2).mean() / d_x
            # CLIP energy loss value to 100 max early in training
            components['energy'] = components['energy'].clamp(max=100.0)
            debug_val('energy', components['energy'])
        else:
            components['energy'] = torch.tensor(0.0, device=device)

        # === 10. Growth Loss (Weighted) ===
        if returns is not None and returns.numel() > 0:
            final_growth = cum_ret[..., -1].mean()
            components['growth'] = -final_growth
            debug_val('growth', components['growth'])
        else:
            components['growth'] = torch.tensor(0.0, device=device)

        # Mark debug batch complete
        if not hasattr(self, '_batch0_val_debug'):
            self._batch0_val_debug = True

        total_loss = sum(self.lambdas[k] * v for k, v in components.items())
        debug_val('TOTAL_LOSS', total_loss)
        return total_loss, components


# =============================================================================
# DATA PREPARATION
# =============================================================================

EPS = 1e-6

def safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def robust_zscore_fit(X: np.ndarray):
    """Robust scaler using median + MAD."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < EPS, 1.0, scale)
    return med.astype(np.float32), scale.astype(np.float32)


def robust_zscore_transform(X: np.ndarray, med: np.ndarray, scale: np.ndarray,
                            clip_val: float = 10.0) -> np.ndarray:
    """Apply robust z-score and clip."""
    X = (X - med) / (scale + EPS)
    X = np.clip(X, -clip_val, clip_val)
    return X.astype(np.float32)


def clamp_targets(y: np.ndarray) -> np.ndarray:
    """Clamp targets to reasonable ranges."""
    y = safe_nan_to_num(y)
    y[:, 0] = np.clip(y[:, 0], -100.0, 100.0)  # call offset
    y[:, 1] = np.clip(y[:, 1], -100.0, 100.0)  # put offset
    y[:, 2] = np.clip(y[:, 2], 0.5, 50.0)      # wing width
    y[:, 3] = np.clip(y[:, 3], 0.0, 1.0)       # dte (normalized)
    y[:, 4] = np.clip(y[:, 4], 0.0, 1.0)       # pop
    y[:, 5] = np.clip(y[:, 5], -5.0, 5.0)      # roi
    y[:, 6] = np.clip(y[:, 6], 0.0, 5.0)       # max loss
    y[:, 7] = np.clip(y[:, 7], 0.0, 1.0)       # confidence
    if y.shape[1] >= 10:
        y[:, 8] = np.clip(y[:, 8], -5.0, 5.0)  # entry_logit
        y[:, 9] = np.clip(y[:, 9], -5.0, 5.0)  # exit_logit
    return y.astype(np.float32)


def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> tuple:
    """
    Prepare features and targets for CondorNet training.

    Returns:
        (X, y, regime, bar_index, med, scale)
    """
    print("[CondorNet] Preparing features...")

    # Handle call_put encoding
    if 'call_put' in df.columns and 'cp_num' not in df.columns:
        df['cp_num'] = df['call_put'].map({'C': 1.0, 'P': -1.0}).fillna(0)

    n = len(df)
    rng = np.random.default_rng(42)

    # IVR normalization
    ivr_raw = df['ivr'].astype(float).values if 'ivr' in df.columns else np.full(n, np.nan)
    finite_mask = np.isfinite(ivr_raw)
    if finite_mask.any():
        ivr_max = np.nanmax(ivr_raw[finite_mask])
        if ivr_max <= 1.5:
            ivr_0_100 = np.where(finite_mask, ivr_raw * 100.0, 50.0)
        else:
            ivr_0_100 = np.where(finite_mask, ivr_raw, 50.0)
    else:
        ivr_0_100 = np.full(n, 50.0)
    ivr_0_100 = np.clip(ivr_0_100, 0.0, 100.0).astype(np.float32)
    df['ivr'] = ivr_0_100

    # Regime labels
    low_vol = ivr_0_100 < 30
    high_vol = ivr_0_100 > 70
    normal_vol = ~low_vol & ~high_vol

    # Generate targets
    df['target_call_offset'] = np.where(
        low_vol, 2.5 + rng.uniform(-0.5, 0.5, n),
        np.where(high_vol, 1.5 + rng.uniform(-0.3, 0.3, n),
                 2.0 + rng.uniform(-0.4, 0.4, n)))

    df['target_put_offset'] = np.where(
        low_vol, 2.5 + rng.uniform(-0.5, 0.5, n),
        np.where(high_vol, 1.5 + rng.uniform(-0.3, 0.3, n),
                 2.0 + rng.uniform(-0.4, 0.4, n)))

    df['target_wing_width'] = np.where(
        low_vol, 6.0 + rng.uniform(-1, 1, n),
        np.where(high_vol, 4.0 + rng.uniform(-0.5, 0.5, n),
                 5.0 + rng.uniform(-0.8, 0.8, n)))

    df['target_dte'] = np.where(
        low_vol, 21.0 + rng.uniform(-5, 5, n),
        np.where(high_vol, 7.0 + rng.uniform(-2, 2, n),
                 14.0 + rng.uniform(-3, 3, n)))
    df['target_dte'] = df['target_dte'] / 45.0  # Normalize

    rsi = df['rsi'].fillna(50).values if 'rsi' in df.columns else np.full(n, 50.0)
    rsi_neutral = (rsi > 40) & (rsi < 60)

    df['was_profitable'] = np.where(
        rsi_neutral, 0.6 + rng.uniform(0, 0.15, n),
        np.where((rsi < 30) | (rsi > 70), 0.3 + rng.uniform(0, 0.1, n),
                 0.45 + rng.uniform(0, 0.1, n)))

    df['realized_roi'] = np.clip(
        (ivr_0_100 / 100 - 0.5) * 0.3 + rng.uniform(-0.1, 0.1, n),
        -0.5, 0.5).astype(np.float32)

    df['realized_max_loss'] = np.where(
        high_vol, 0.4 + rng.uniform(0, 0.2, n),
        np.where(low_vol, 0.1 + rng.uniform(0, 0.1, n),
                 0.2 + rng.uniform(0, 0.15, n)))

    adx = df['adx'].fillna(25).values if 'adx' in df.columns else np.full(n, 25.0)
    conf_rsi = np.where(rsi_neutral, 0.3, 0.1)
    conf_ivr = np.where(normal_vol, 0.3, np.where(low_vol, 0.2, 0.1))
    conf_adx = np.where(adx < 25, 0.3, 0.1)
    df['confidence_target'] = np.clip(
        conf_rsi + conf_ivr + conf_adx + rng.uniform(-0.1, 0.1, n), 0.1, 0.95)

    # Entry/exit targets
    entry_score = np.zeros(n, dtype=np.float32)
    entry_score += np.where(rsi_neutral, 1.0, -0.5)
    entry_score += np.where(ivr_0_100 > 30, 0.5, -0.5)
    entry_score += np.where(adx < 30, 0.5, -0.3)
    entry_score += rng.uniform(-0.3, 0.3, n)
    df['entry_target'] = entry_score

    exit_score = np.zeros(n, dtype=np.float32)
    exit_score += np.where((rsi < 30) | (rsi > 70), 1.0, -0.5)
    exit_score += np.where(high_vol, 0.8, -0.3)
    exit_score += np.where(adx > 35, 0.5, -0.2)
    exit_score += rng.uniform(-0.3, 0.3, n)
    df['exit_target'] = exit_score

    # Regime label
    if 'regime_label' not in df.columns:
        df['regime_label'] = pd.cut(
            pd.Series(ivr_0_100),
            bins=[-0.1, 30, 70, 101],
            labels=[0, 1, 2]
        ).fillna(1).astype(int)

    # REPLACED: df.ffill().bfill().fillna(0) -> Avoid doubling memory
    # Only fill critical columns if they have NaNs
    for c in ['rsi', 'adx', 'ivr']:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(NEUTRAL_FILL_VALUES_V22.get(c, 0.0))

    # Build arrays
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target',
        'entry_target', 'exit_target'
    ]

    X = select_feature_frame(df, feature_cols, strict=True).values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    regime = df['regime_label'].values.astype(np.int64)

    # TIMESTAMPS -> bar_index
    if 'dt' in df.columns or 'timestamp' in df.columns:
        time_col = 'dt' if 'dt' in df.columns else 'timestamp'
        bar_index = df.groupby(time_col).ngroup().values.astype(np.int64)
    else:
        bar_index = (np.arange(n) // 100).astype(np.int64)

    # FREE DATAFRAME IMMEDIATELY
    del df

    # Sanitize
    X = safe_nan_to_num(X)
    y = clamp_targets(y)

    # Scale volume (inplace where possible)
    if X.shape[1] > 4:
        X[:, 4] = np.log1p(np.clip(X[:, 4], 0.0, 1e9)).astype(np.float32)

    # Robust normalization (inplace transform)
    med, scale = robust_zscore_fit(X)
    X = robust_zscore_transform(X, med, scale)

    print(f"[CondorNet] Features: {X.shape}, Targets: {y.shape}")
    return X, y, regime, bar_index, med, scale


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CondorNet")

    # Data
    parser.add_argument("--local-data", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=0)

    # Model architecture
    parser.add_argument("--d-h", type=int, default=256, help="Latent physics dim")
    parser.add_argument("--d-v", type=int, default=32, help="Portfolio state dim")
    parser.add_argument("--d-m", type=int, default=64, help="Risk memory dim")
    parser.add_argument("--d-r", type=int, default=32, help="Regime/combinatorics dim")
    parser.add_argument("--d-control", type=int, default=128, help="TFT control dim")
    parser.add_argument("--n-layers", type=int, default=2, help="TFT layers")

    # Training - Logic Complexity (Scale down if OOM on smaller GPUs)
    parser.add_argument("--n-predicates", type=int, default=512, help="Number of predicate gates")
    parser.add_argument("--n-sets", type=int, default=256, help="Number of logic sets")
    parser.add_argument("--n-super-sets", type=int, default=128, help="Number of super-sets")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--accum-steps", type=int, default=4)

    # Loss weights
    parser.add_argument("--lambda-mse", type=float, default=1.0, help="Weight for supervised regression")
    parser.add_argument("--lambda-npdd", type=float, default=1.0)
    parser.add_argument("--lambda-sharpe", type=float, default=0.1)
    parser.add_argument("--lambda-dd", type=float, default=0.3)
    parser.add_argument("--lambda-turnover", type=float, default=0.1)
    parser.add_argument("--lambda-fuzzy", type=float, default=0.2)
    parser.add_argument("--lambda-pattern-ent", type=float, default=0.05)
    parser.add_argument("--lambda-group-inv", type=float, default=0.1)
    parser.add_argument("--lambda-rho", type=float, default=0.1)
    parser.add_argument("--lambda-energy", type=float, default=0.01)
    parser.add_argument("--lambda-growth", type=float, default=0.1)
    parser.add_argument("--use-clamping", action="store_true", default=False, help="Use soft-clamping (tanh) for Sharpe/NPDD")
    parser.add_argument("--no-clamping", action="store_false", dest="use_clamping", help="Explicitly disable soft-clamping")

    # Output
    parser.add_argument("--output", type=str, default="auto")

    # Options
    parser.add_argument("--no-sparsity", action="store_true",
                        help="Disable sparsity constraints")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose per-batch logging")
    parser.add_argument("--gui-telemetry", type=str, nargs='?', const='local', default=None,
                        choices=['local', 'lightai', 'kaggle', 'colab'],
                        help="Send training metrics to GUI (local|lightai|kaggle|colab)")
    parser.add_argument("--save-diagnostics", action="store_true",
                        help="Save diagnostics to models/diagnostics/ for GUI Model Introspection")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save checkpoint every N epochs (0=disabled, 1=every epoch)")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints",
                        help="Directory for epoch checkpoints")

    args = parser.parse_args()

    if args.output == "auto":
        lr_str = f"{args.lr:.0e}".replace("-", "")
        args.output = f"models/condor_net_e{args.epochs}_dh{args.d_h}_lr{lr_str}.pth"

    return args


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_condor_net(args):
    """Main training function for CondorNet."""

    # GUI Telemetry setup
    emitter = None
    if args.gui_telemetry and GUI_TELEMETRY_AVAILABLE:
        backend_url = get_backend_url(args.gui_telemetry)
        emitter = get_emitter(backend_url=backend_url)
        print(f"[CondorNet] GUI telemetry enabled ({args.gui_telemetry} -> {backend_url})")
    elif args.gui_telemetry:
        print("[CondorNet] GUI telemetry requested but not available (GUI backend not running)")

    # Training Diagnostics setup (for Model Introspection)
    diagnostics = None
    if getattr(args, 'save_diagnostics', False) and DIAGNOSTICS_AVAILABLE:
        diagnostics = TrainingDiagnostics(
            num_fuzzy_gates=args.n_predicates,
            smoothing_window=200,
            log_interval=10,
        )
        diagnostics.metadata = {
            "model": "CondorNet",
            "d_h": args.d_h,
            "d_v": args.d_v,
            "d_m": args.d_m,
            "d_r": args.d_r,
            "n_predicates": args.n_predicates,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        }
        print("[CondorNet] Diagnostics logging enabled")
    elif getattr(args, 'save_diagnostics', False):
        print("[CondorNet] Diagnostics requested but module not available")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[CondorNet] Device: {device}")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"[CondorNet] GPU: {gpu_name} ({gpu_mem:.1f}GB, CC {compute_cap[0]}.{compute_cap[1]})")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        use_bf16 = compute_cap[0] >= 8
    else:
        use_bf16 = False

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Load data with usecols optimization
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target',
        'entry_target', 'exit_target'
    ]
    aux_cols = ['dt', 'timestamp', 'regime_label', 'ivr', 'vix', 'rsi', 'adx', 'call_put']
    
    # Merge all required cols
    load_cols = list(set(FEATURE_COLS + target_cols + aux_cols))
    
    # Check what's actually in the CSV to avoid FileNotFoundError on missing aux cols
    header = pd.read_csv(args.local_data, nrows=0).columns
    use_cols = [c for c in load_cols if c in header]
    
    print(f"\n[CondorNet] Loading data from {args.local_data} (using {len(use_cols)}/ {len(header)} columns)...")
    if args.max_rows > 0:
        df = pd.read_csv(args.local_data, nrows=args.max_rows, usecols=use_cols)
    else:
        df = pd.read_csv(args.local_data, usecols=use_cols)
    print(f"[CondorNet] Loaded {len(df):,} rows")

    X, y, regime, bar_index, med, scale = prepare_features(df, FEATURE_COLS)
    # del df # Done inside prepare_features now

    # Split
    split_row = int(len(X) * 0.8)
    X_train, X_val = X[:split_row], X[split_row:]
    y_train, y_val = y[:split_row], y[split_row:]
    bar_train, bar_val = bar_index[:split_row], bar_index[split_row:]

    # Move to GPU
    L = args.lookback
    B = args.batch_size

    # Standardize to FP32 for AMP (autocast handles the speedup)
    X_train_t = torch.from_numpy(X_train).to(device=device, dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    X_val_t = torch.from_numpy(X_val).to(device=device, dtype=torch.float32)
    y_val_t = torch.from_numpy(y_val).to(device=device, dtype=torch.float32)

    # Sequence views
    n_train_seq = len(X_train) - L
    n_val_seq = len(X_val) - L
    X_train_seq = X_train_t.unfold(0, L, 1).permute(0, 2, 1)
    X_val_seq = X_val_t.unfold(0, L, 1).permute(0, 2, 1)

    # bar_index views to create timestep_mask
    bar_train_t = torch.from_numpy(bar_train).to(device=device, dtype=torch.long)
    bar_val_t = torch.from_numpy(bar_val).to(device=device, dtype=torch.long)
    
    # Unfold bar_index to create views (B, L)
    bar_train_seq = bar_train_t.unfold(0, L, 1)
    bar_val_seq = bar_val_t.unfold(0, L, 1)
    
    # LAZY MASK: We compute specific batch masks in the loop to save RAM/VRAM
    # mask_train_seq = (bar_train_seq[:, 1:] != bar_train_seq[:, :-1]).float()
    # mask_val_seq = (bar_val_seq[:, 1:] != bar_val_seq[:, :-1]).float()

    n_train_batches = n_train_seq // B
    n_val_batches = max(1, n_val_seq // B)

    print(f"[CondorNet] Train: {n_train_seq:,} sequences, {n_train_batches} batches")
    print(f"[CondorNet] Val: {n_val_seq:,} sequences, {n_val_batches} batches")

    # Model
    model = CondorNet(
        d_input=len(FEATURE_COLS),
        d_h=args.d_h,
        d_v=args.d_v,
        d_m=args.d_m,
        d_r=args.d_r,
        d_control=args.d_control,
        n_layers=args.n_layers,
        n_predicates=args.n_predicates,
        n_sets=args.n_sets,
        n_super_sets=args.n_super_sets,
        enforce_sparsity=not args.no_sparsity,
    ).to(device)

    # REMOVED: Explicit conversion to bf16/fp16. 
    # Standard AMP keeps weights in FP32 and uses autocast for operations.
    # This is MUCH more stable for the backward pass on T4/A100.
    print(f"[CondorNet] Model weights kept in {next(model.parameters()).dtype} (AMP Stable Mode)")

    # IMPORTANT: keep entire model (including pred_signature) in one dtype
    # Do NOT override model.pred_signature dtype here.

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CondorNet] Parameters: {n_params:,}")

    # Loss function
    criterion = CompositeCondorNetLoss(
        lambda_mse=args.lambda_mse,
        lambda_npdd=args.lambda_npdd,
        lambda_sharpe=args.lambda_sharpe,
        lambda_dd=args.lambda_dd,
        lambda_turnover=args.lambda_turnover,
        lambda_fuzzy=args.lambda_fuzzy,
        lambda_pattern_ent=args.lambda_pattern_ent,
        lambda_group_inv=args.lambda_group_inv,
        lambda_rho=args.lambda_rho,
        lambda_energy=args.lambda_energy,
        lambda_growth=args.lambda_growth,
        use_clamping=args.use_clamping,
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler() if not use_bf16 else None

    # Training
    print(f"\n{'='*60}")
    print(f"CONDORNET TRAINING")
    print(f"{'='*60}")
    print(f"State: d_h={args.d_h}, d_v={args.d_v}, d_m={args.d_m}, d_r={args.d_r}")
    print(f"Total state dim: {model.spec.d_x}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # TensorBoard
    run_name = f"condor_dh{args.d_h}_lr{args.lr}_bs{args.batch_size}"
    writer = SummaryWriter(log_dir=f"runs/condor_net/{run_name}")
    global_step = 0
    training_start_time = time.time()

    best_val_loss = float('inf')
    patience_counter = 0
    total_steps = args.epochs * n_train_batches

    # Emit training start
    if emitter:
        emitter.emit_status(
            is_training=True,
            current_epoch=1,
            current_step=0,
            total_steps=total_steps,
            progress_pct=0.0,
            eta_seconds=None,
        )

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_components = {k: 0.0 for k in criterion.lambdas.keys()}

        pbar = tqdm(range(n_train_batches), desc=f"Epoch {epoch+1}", leave=False)

        optimizer.zero_grad(set_to_none=True)

        for batch_idx in pbar:
            s = batch_idx * B
            e = s + B
            x_b = X_train_seq[s:e].to(device)
            y_b = y_train_t[s + L - 1 : e + L - 1].to(device)
            
            # LAZY MASK CALCULATION (Per batch)
            bar_b = bar_train_seq[s:e].to(device)
            mask_b = (bar_b[:, 1:] != bar_b[:, :-1]).float()

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                # BCS: Pass timestep_mask and bar_index to forward
                outputs, diag = model(x_b, return_diagnostics=True, timestep_mask=mask_b, bar_index=bar_b)

            # --- LOSS CALCULATION (OUTSIDE AUTOCAST FOR PRECISION) ---
            # Synthetic return series for Sharpe/Drawdown (across the batch)
            roi_returns = y_b[:, 5].unsqueeze(0)  # (1, B)

            loss, components = criterion(
                outputs,
                y_b,
                gates=diag.get('predicates'),
                state=diag.get('z_final'),
                A_matrix=model.get_A_matrix(),
                pred_signature=model.pred_signature,
                returns=roi_returns,
            )

            # Debug Prints for Crash
            if batch_idx == 0:
                print(f"\n[DEBUG BATCH 0] System Inventory:")
                print(f"  Input x_b: {x_b.dtype}")
                print(f"  Model Weights: {next(model.parameters()).dtype}")
                print(f"  Outputs (Raw): {outputs.dtype}")
                print(f"  Final Loss: {loss.item():.6f} ({loss.dtype}), GradFn: {loss.grad_fn}")
                for k, v in components.items():
                    if torch.is_tensor(v):
                        val = v.item() if v.numel() == 1 else v.mean().item()
                        print(f"    - {k}: {val:.6f} ({v.dtype}), GradFn: {v.grad_fn}")

                if scaler is not None:
                    print(f"  Scaler: enabled={scaler.is_enabled()}, scale={scaler.get_scale()}")

            if scaler is not None:
                try:
                    scaler.scale(loss).backward()
                except RuntimeError as e:
                    print(f"\n!!! BACKWARD CRASHED in batch {batch_idx}:")
                    print(f"Error: {e}")
                    # If it's a dtype mismatch, we can't do much here but print what we have
                    raise e
                
                if (batch_idx + 1) % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (batch_idx + 1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            for k, v in components.items():
                epoch_components[k] += v.item() if torch.is_tensor(v) else v

            # TensorBoard: log every 100 steps
            if global_step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                for k, v in components.items():
                    val = v.item() if torch.is_tensor(v) else v
                    writer.add_scalar(f'train/{k}', val, global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if getattr(args, 'verbose', False) and (batch_idx + 1) % 10 == 0:
                comp_str = ' | '.join(
                    [f"{k}:{(v.item() if torch.is_tensor(v) else v):.3f}"
                     for k, v in list(components.items())[:5]]
                )
                print(f"  [B{batch_idx+1:04d}] loss={loss.item():.4f} | {comp_str}")

            # Diagnostics: log every 10 batches
            if diagnostics and (batch_idx + 1) % 10 == 0:
                diagnostics.log_loss(
                    step=global_step,
                    epoch=epoch + 1,
                    loss=loss,
                    mse=components['mse'],
                    npdd=components['npdd'],
                    sharpe=components['sharpe'],
                    dd=components['dd'],
                    turnover=components['turnover'],
                    fuzzy=components['fuzzy'],
                    pattern_ent=components['pattern_ent'],
                    group_inv=components['group_inv'],
                    rho=components['rho'],
                    energy=components['energy'],
                    growth=components['growth'],
                    lr=scheduler.get_last_lr()[0],
                    scaler_scale=scaler.get_scale() if scaler else None,
                )
                # Log fuzzy activations if available
                if 'predicates' in diag and diag['predicates'] is not None:
                    diagnostics.log_fuzzy(
                        step=global_step,
                        epoch=epoch + 1,
                        fuzzy_activations=diag['predicates'],
                    )

            # GUI Telemetry: emit every 10 batches
            if emitter and (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - training_start_time
                steps_done = epoch * n_train_batches + batch_idx + 1
                steps_remaining = total_steps - steps_done
                eta_seconds = int((elapsed / steps_done) * steps_remaining) if steps_done > 0 else None
                progress_pct = (steps_done / total_steps) * 100

                emitter.emit_step(
                    step=global_step,
                    epoch=epoch + 1,
                    loss=loss.item(),
                    mse=components['mse'].item() if torch.is_tensor(components['mse']) else components['mse'],
                    npdd=components['npdd'].item() if torch.is_tensor(components['npdd']) else components['npdd'],
                    sharpe=components['sharpe'].item() if torch.is_tensor(components['sharpe']) else components['sharpe'],
                    dd=components['dd'].item() if torch.is_tensor(components['dd']) else components['dd'],
                    turnover=components['turnover'].item() if torch.is_tensor(components['turnover']) else components['turnover'],
                    fuzzy=components['fuzzy'].item() if torch.is_tensor(components['fuzzy']) else components['fuzzy'],
                    pattern_ent=components['pattern_ent'].item() if torch.is_tensor(components['pattern_ent']) else components['pattern_ent'],
                    group_inv=components['group_inv'].item() if torch.is_tensor(components['group_inv']) else components['group_inv'],
                    rho=components['rho'].item() if torch.is_tensor(components['rho']) else components['rho'],
                    energy=components['energy'].item() if torch.is_tensor(components['energy']) else components['energy'],
                    growth=components['growth'].item() if torch.is_tensor(components['growth']) else components['growth'],
                    lr=scheduler.get_last_lr()[0],
                    scaler_scale=scaler.get_scale() if scaler else None,
                )

                emitter.emit_status(
                    is_training=True,
                    current_epoch=epoch + 1,
                    current_step=steps_done,
                    total_steps=total_steps,
                    progress_pct=progress_pct,
                    eta_seconds=eta_seconds,
                )

        scheduler.step()
        avg_train_loss = epoch_loss / n_train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        vbar = tqdm(range(n_val_batches), desc="Validation", leave=False)
        with torch.no_grad():
            for batch_idx in vbar:
                s = batch_idx * B
                e = s + B
                x_b = X_val_seq[s:e].to(device)
                y_b = y_val_t[s + L - 1 : e + L - 1].to(device)
                
                # LAZY MASK CALCULATION (Per batch)
                bar_b = bar_val_seq[s:e].to(device)
                mask_b = (bar_b[:, 1:] != bar_b[:, :-1]).float()

                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    outputs, diag = model(x_b, return_diagnostics=True, timestep_mask=mask_b, bar_index=bar_b)
                    
                    A_matrix = model.get_A_matrix()
                    gates = diag.get('predicates')
                    state = diag.get('z_final')

                    loss, _ = criterion(
                        outputs.float(), 
                        y_b, 
                        gates=gates,
                        state=state,
                        A_matrix=A_matrix,
                        pred_signature=model.pred_signature,
                        returns=y_b[:, 5].unsqueeze(0) # Calc per batch in val too
                    )

                val_loss += loss.item()
                vbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / n_val_batches

        # TensorBoard: epoch-level metrics
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
        for k, v in epoch_components.items():
            writer.add_scalar(f'epoch/{k}', v / n_train_batches, epoch)

        print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # GUI Telemetry: epoch summary
        if emitter:
            emitter.emit_epoch_summary(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                metrics={k: v / n_train_batches for k, v in epoch_components.items()},
                is_best=(avg_val_loss < best_val_loss),
            )

        # Save epoch checkpoint (every N epochs)
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"condornet_epoch_{epoch+1:03d}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'state_dims': {
                    'd_h': args.d_h, 'd_v': args.d_v,
                    'd_m': args.d_m, 'd_r': args.d_r,
                },
                'feature_cols': FEATURE_COLS,
                'normalization': {'median': med.tolist(), 'scale': scale.tolist()},
            }, checkpoint_path)
            print(f"  -> Checkpoint saved: {checkpoint_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'state_dims': {
                    'd_h': args.d_h, 'd_v': args.d_v,
                    'd_m': args.d_m, 'd_r': args.d_r,
                },
                'feature_cols': FEATURE_COLS,
                'normalization': {'median': med.tolist(), 'scale': scale.tolist()},
            }, args.output)
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*60}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"{'='*60}")

    # GUI Telemetry: training complete
    if emitter:
        total_duration = int(time.time() - training_start_time)
        emitter.emit_complete(
            epochs=epoch + 1,
            final_loss=avg_train_loss,
            best_val_loss=best_val_loss,
            duration_seconds=total_duration,
        )
        emitter.emit_status(
            is_training=False,
            current_epoch=epoch + 1,
            current_step=total_steps,
            total_steps=total_steps,
            progress_pct=100.0,
        )

    # Save diagnostics to file for Model Introspection
    if diagnostics:
        diagnostics_dir = Path("models/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        diagnostics_path = diagnostics_dir / f"condornet_{timestamp}.json"
        diagnostics.save(str(diagnostics_path))
        print(f"[CondorNet] Diagnostics saved to {diagnostics_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    args = parse_args()
    train_condor_net(args)
