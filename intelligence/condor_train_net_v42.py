"""
CondorNet™ v5 Training Laboratory

Scientific training pipeline with smart convergence detection, full interpretability
reports, A_matrix stability tracking, and epoch-vs-epoch comparison analytics.

Implements the 10-component composite loss, intra-epoch convergence monitoring,
generalization-gated best-model selection, and per-epoch forensic artifacts.

Usage:
    python intelligence/condor_train_net.py --local-data data/Datasetv3/condornet_v30_precomputed.csv \
        --d-h 128 --d-v 16 --d-m 32 --d-r 16 --n-predicates 64 --n-sets 32 --n-super-sets 8 \
        --epochs 5 --batch-size 256 --lr 1e-4 --lookback 240 -v --checkpoint-every 1 --save-diagnostics

Author: Claude Code (Opus 4.6)
Version: 5.0.0
Date: 2026-02-12
"""

import sys
import os
import shutil
import math
import time
import copy
import glob
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence

# =============================================================================
# VERSION CONSTANTS
# =============================================================================
CN_VERSION = "CNv5"
DS_VERSION = "DSv3"

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

from intelligence.condor_brain_net_v42 import (
    CondorNet,
    group_invariant_loss,
    spectral_radius_loss,
)
from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V22,
    FEATURE_COLS_V30,
    FEATURE_COLS_V42,
    NEUTRAL_FILL_VALUES_V22,
    NEUTRAL_FILL_VALUES_V30,
    select_feature_frame,
)


import itertools

def build_dataloaders(batch_size: int, seq_len: int, data_path: str = "data/Datasetv4/condornet_v41_FINAL.csv"):
    """
    Standardized v4.2 dataloader constructor (Robust Future-Proof).
    Returns: train_loader, val_loader, test_loader
    
    Adapts automatically to prepare_features return signature:
    - (X, y)
    - (X, y, meta...)
    - {X:..., y:..., ...}
    
    Handles static metadata (med, scale) vs time-series extras.
    """
    print(f"[CondorNet] build_dataloaders called with path: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Use V4.2 features by default
    active_feature_cols = FEATURE_COLS_V42
    
    # ---- 1. Call prepare_features and normalize return type ----
    outputs = prepare_features(df, active_feature_cols)

    extra_names = []
    if isinstance(outputs, tuple):
        X = outputs[0]
        y = outputs[1]
        extras = outputs[2:] if len(outputs) > 2 else []
        extra_names = [f"extra_{i}" for i in range(len(extras))]
    elif isinstance(outputs, dict):
        X = outputs["X"]
        y = outputs["y"]
        extras = [v for k, v in outputs.items() if k not in ("X", "y")]
        extra_names = [k for k in outputs.keys() if k not in ("X", "y")]
    else:
        raise ValueError("Unsupported return type from prepare_features")

    # ---- 2. Convert X, y to tensors ----
    # Ensure float32 for main data
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Convert extras to tensors (handle various types)
    extras_tensors = []
    for e in extras:
        if isinstance(e, (np.ndarray, list)):
            try:
                t = torch.tensor(e)
                # Cast flow/int types appropriately if needed, or default to float32 if float features
                if t.dtype == torch.float64: t = t.float()
                extras_tensors.append(t)
            except Exception as exc:
                print(f"Warning: Could not convert extra to tensor: {exc}")
                extras_tensors.append(e) # Keep as is if fails
        elif isinstance(e, torch.Tensor):
            extras_tensors.append(e)
        else:
            extras_tensors.append(e) # Scalar or other

    # ---- 3. Identify static vs dynamic extras ----
    # Static extras: shape (D,) or (D, something) or scalar
    # Dynamic extras: same length as X (N, ...)
    static_extras = []
    dynamic_extras = []

    N_full = X.shape[0]

    for e in extras_tensors:
        if hasattr(e, "shape") and e.shape[0] == N_full:
            dynamic_extras.append(e)
        else:
            static_extras.append(e)

    # ---- 4. Memory-Safe Sequence Dataset ----
    # Instead of pre-slicing (which explodes RAM), we use a custom Dataset
    # that slices on-the-fly in __getitem__.
    
    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, extras, seq_len):
            self.X = X
            self.y = y
            self.extras = extras
            self.seq_len = seq_len
            self.N = len(X)

        def __len__(self):
            return max(0, self.N - self.seq_len)

        def __getitem__(self, idx):
            # Input sequence: [idx, idx+seq_len)
            X_seq = self.X[idx : idx + self.seq_len]
            
            # Target: typically we predict step at t+seq_len
            # (Matches original behavior where y was aligned with end of sequence)
            y_target = self.y[idx + self.seq_len]

            # Extras handling
            extra_seq = []
            for e in self.extras:
                # Dynamic extra: slice it like X
                if hasattr(e, "shape") and e.shape[0] == self.N:
                    extra_seq.append(e[idx : idx + self.seq_len])
                # Static extra: pass as is
                else:
                    extra_seq.append(e)

            return (X_seq, y_target, *extra_seq)

    # Instantiate dataset with all extras (dynamic + static)
    dataset = SequenceDataset(X, y, dynamic_extras + static_extras, seq_len)
    
    if len(dataset) <= 0:
        raise ValueError(f"Dataset length ({len(X)}) is smaller than sequence length ({seq_len}).")

    # ---- 5. Split and Create Loaders ----
    # 80/10/10 split
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    # Use Subset for memory efficiency (views, not copies)
    train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
    val_dataset   = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    test_dataset  = torch.utils.data.Subset(dataset, range(n_train + n_val, n_total))

    # Determine num_workers: 0 for safety/debugging, >0 for speed if needed
    # Pin memory if CUDA available
    pin_memory = torch.cuda.is_available()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    
    # Check if empty (e.g. extreme filtering)
    if len(train_loader) == 0:
        print("[CondorNet] Warning: Train loader is empty!")
    
    return train_loader, val_loader, test_loader

# FEATURE_COLS will be selected dynamically based on args.data_version
DEFAULT_FEATURE_COLS = FEATURE_COLS_V30


# =============================================================================
# SEQUENCE DATASET (Module-Level for use by both build_dataloaders & train_condor_net)
# =============================================================================

class SequenceDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that slices sequences on-the-fly in __getitem__."""
    def __init__(self, X, y, regime, seq_len):
        self.X = X
        self.y = y
        self.regime = regime
        self.seq_len = seq_len
        self.N = len(X)

    def __len__(self):
        return max(0, self.N - self.seq_len)

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        r_target = self.regime[idx + self.seq_len]
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
            torch.tensor(r_target, dtype=torch.float32),
        )

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


# =============================================================================
# ATOMIC SAVE & CHECKPOINT ROTATION
# =============================================================================

def atomic_save(data: dict, target_path, validate: bool = True):
    """Save checkpoint atomically: write to .tmp, validate, then rename.
    
    Prevents corrupted saves from partial writes or crashes mid-save.
    """
    target_path = Path(target_path)
    tmp_path = target_path.with_suffix('.tmp')
    
    try:
        # Write to temp file
        torch.save(data, tmp_path)
        
        # Validate: attempt to load the checkpoint
        if validate:
            _ = torch.load(tmp_path, map_location='cpu', weights_only=False)
        
        # Atomic rename (same filesystem = atomic on POSIX, best-effort on Windows)
        shutil.move(str(tmp_path), str(target_path))
        return True
    except Exception as e:
        print(f"  [SAVE ERROR] Failed to save {target_path.name}: {e}")
        # Clean up temp file if it exists
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def rotate_checkpoints(checkpoint_dir, keep_n: int = 3):
    """Keep only the N most recent .pth checkpoints in a directory.
    
    Files are sorted by modification time (newest first).
    Set keep_n=0 to keep all checkpoints.
    """
    if keep_n <= 0:
        return  # Keep all
    
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = sorted(
        checkpoint_dir.glob("Epoch*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,  # Newest first
    )
    
    for old_ckpt in ckpts[keep_n:]:
        try:
            old_ckpt.unlink()
            print(f"  [ROTATE] Removed old checkpoint: {old_ckpt.name}")
        except Exception as e:
            print(f"  [ROTATE] Failed to remove {old_ckpt.name}: {e}")


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
    
    # Initialize IV Rank source
    if 'iv_rank' in df.columns:
        ivr_0_100 = df['iv_rank'].values
    elif 'ivr' in df.columns:
         # Check if IVR is 0-1 or 0-100. Heuristic: if max <= 1.0, scale up.
        vals = df['ivr'].values
        if vals.max() <= 1.0:
            ivr_0_100 = vals * 100.0
        else:
            ivr_0_100 = vals
    else:
        print("[CondorNet] Warning: No iv_rank or ivr column found. Defaulting to 50.0")
        ivr_0_100 = np.full(n, 50.0, dtype=np.float32)

    ivr_0_100 = np.clip(ivr_0_100, 0.0, 100.0).astype(np.float32)
    df['ivr'] = ivr_0_100

    # SEED-BASED RNG FOR CONSISTENCY
    # If a seed wasn't passed, default to 42 (backward compatibility), 
    # but ideally we pass it from args. Since signature doesn't have it,
    # we rely on global numpy seed or local rng.
    # We will use a local RNG for target generation to be safe.
    # Note: caller should have set np.random.seed(args.seed) globally too.
    # But let's check if we can pass it. 
    # For now, we'll use a fixed seed 42 here to match previous behavior 
    # OR we can assume np.random is already seeded.
    # To be safe and deterministic per run:
    rng = np.random.default_rng(42) 

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
            df[c] = df[c].ffill().fillna(NEUTRAL_FILL_VALUES_V30.get(c, 0.0))

    # Build arrays
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target',
        'entry_target', 'exit_target'
    ]

    # === V4.2 Feature Backfill / Generation ===
    # If the dataset is V4.1, we need to generate these structural features on the fly.
    
    # 1. Session Timing (minutes_from_open, minutes_to_close, norm_session_time)
    if 'minutes_from_open' not in df.columns and ('timestamp' in df.columns or 'date' in df.columns):
        # Infer from timestamp if available
        try:
            ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                df[ts_col] = pd.to_datetime(df[ts_col])
            
            # Simple assumption: Market Open 9:30 AM ET, Close 4:00 PM ET
            # We don't have timezone info easily, so standardizing on minute of day
            minutes_of_day = df[ts_col].dt.hour * 60 + df[ts_col].dt.minute
            market_open_min = 9 * 60 + 30
            market_close_min = 16 * 60
            
            df['minutes_from_open'] = minutes_of_day - market_open_min
            df['minutes_to_close'] = market_close_min - minutes_of_day
            total_session_min = market_close_min - market_open_min
            df['norm_session_time'] = (df['minutes_from_open'] / total_session_min).clip(0, 1)
            
            # Session Phase
            # 1: Open (first 30m), 2: Mid, 3: Close (last 30m), 4: Ext (outside)
            conditions = [
                (df['minutes_from_open'] < 30) & (df['minutes_from_open'] >= 0),
                (df['minutes_to_close'] < 30) & (df['minutes_to_close'] >= 0),
                (df['minutes_from_open'] >= 30) & (df['minutes_to_close'] >= 30)
            ]
            choices = [1.0, 3.0, 2.0] # Open, Close, Mid
            df['session_phase'] = np.select(conditions, choices, default=4.0).astype(np.float32)
            
        except Exception as e:
            print(f"[CondorNet] Warning: Failed to generate timing features: {e}")

    # 2. Structural Features Defaulting (if logical generation is too complex for here)
    # We use the registry's neutral values for anything still missing.
    from intelligence.canonical_feature_registry import NEUTRAL_FILL_VALUES_V42
    
    miss_cols = [c for c in feature_cols if c not in df.columns]
    if miss_cols:
        print(f"[CondorNet] Backfilling {len(miss_cols)} missing V4.2 features with neutral values.")
        for col in miss_cols:
            fill_val = NEUTRAL_FILL_VALUES_V42.get(col, 0.0)
            df[col] = np.full(n, fill_val, dtype=np.float32)

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
    # NEW: Data Version Control for v4.2/v4.1
    parser.add_argument("--data-version", type=str, default="v4.2", choices=["v3.0", "v4.1", "v4.2"],
                        help="Feature schema version (v3.0, v4.1, or v4.2). Default: v3.0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

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
    parser.add_argument("--checkpoint-dir", type=str, default="models/ckpts",
                        help="Directory for epoch checkpoints")

    # GPU Performance Optimization
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. T4: 2-4, A100: 4-8, H100: 8-12, L4: 2-4")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Batches pre-loaded per worker. T4: 2, A100: 2-4, H100: 4, L4: 2")
    parser.add_argument("--persistent-workers", action="store_true", default=True,
                        help="Keep workers alive between epochs (default: True)")
    parser.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers",
                        help="Disable persistent workers")

    # Deep Observability
    parser.add_argument("--deep-observe", action="store_true", default=False,
                        help="Enable per-batch deep component introspection reports")
    parser.add_argument("--observe-every", type=int, default=100,
                        help="Emit deep observation report every N batches (default: 100)")

    # Checkpoint management
    parser.add_argument("--keep-ckpts", type=int, default=3,
                        help="Keep only the N most recent epoch checkpoints (default: 3, 0=keep all)")

    # V5: Smart convergence
    parser.add_argument("--convergence-threshold", type=float, default=0.1,
                        help="Loss threshold to trigger convergence tracking (default: 0.1)")
    parser.add_argument("--convergence-patience", type=int, default=10,
                        help="Batches without improvement before stopping epoch (default: 10)")
    parser.add_argument("--convergence-active", action="store_true", default=False,
                        help="Actually cut epochs on convergence (default: observe-only)")

    args = parser.parse_args()

    if args.output == "auto":
        lr_str = f"{args.lr:.0e}".replace("-", "")
        args.output = f"models/{CN_VERSION}_{DS_VERSION}_Best_{datetime.now().strftime('%m%d%Y_%H%M%S')}.pth"

    return args


# =============================================================================
# V5: SMART EPOCH MANAGER
# =============================================================================

class SmartEpochManager:
    """
    Intra-epoch convergence detection.

    When batch loss drops below `threshold`, begins a patience window.
    If `patience` consecutive batches fail to improve, signals epoch stop.
    Captures a model state_dict snapshot at the best loss within the window.

    In observe-only mode (active=False), logs when it would stop but does not
    actually signal a break — allows safe burn-in before enabling.
    """

    def __init__(self, threshold: float = 0.1, patience: int = 10, active: bool = False):
        self.threshold = threshold
        self.patience = patience
        self.active = active
        self.reset()

    def reset(self):
        """Reset for a new epoch."""
        self.tracking = False
        self.patience_counter = 0
        self.best_loss_in_window = float('inf')
        self.best_snapshot = None
        self.has_snapshot = False

    def update(self, loss: float, model: nn.Module) -> bool:
        """
        Feed a batch loss. Returns True if the epoch should stop.

        Args:
            loss: Current batch loss
            model: Current model (for snapshotting state_dict)

        Returns:
            True if epoch should stop (only if active=True)
        """
        if loss < self.threshold:
            if not self.tracking:
                self.tracking = True
                self.best_loss_in_window = loss
                self.best_snapshot = copy.deepcopy(model.state_dict())
                self.has_snapshot = True
                self.patience_counter = 0
                return False

            # Already tracking
            if loss < self.best_loss_in_window:
                # Improvement — reset patience, update snapshot
                self.best_loss_in_window = loss
                self.best_snapshot = copy.deepcopy(model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                if self.active:
                    return True
                else:
                    # Observe-only: log but don't stop
                    print(f"\n  [CONVERGENCE OBSERVE] Would stop here "
                          f"(best={self.best_loss_in_window:.6f}, "
                          f"patience={self.patience_counter}/{self.patience})")
                    self.patience_counter = 0  # Reset to keep observing
        else:
            # Loss above threshold — reset tracking
            if self.tracking:
                self.tracking = False
                self.patience_counter = 0

        return False


# =============================================================================
# V5: INTERPRETABILITY EXTRACTION
# =============================================================================

# Import audit tool functions
try:
    from intelligence.audit_condornet_interpretability import (
        extract_learned_logic,
        generate_trading_rules,
    )
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


def extract_epoch_interpretability(
    model: nn.Module,
    epoch: int,
    train_loss: float,
    val_loss: float,
    feature_cols: list,
    args,
    is_best: bool = False,
) -> dict:
    """
    Extract full interpretability report from the live model.

    Reuses extract_learned_logic() from audit_condornet_interpretability.py
    and adds epoch-level metadata + A_matrix stability metrics.
    """
    timestamp = datetime.now().isoformat()
    n_params = sum(p.numel() for p in model.parameters())

    report = {
        "summary": {
            "epoch": epoch,
            "version": f"{CN_VERSION}_{DS_VERSION}",
            "timestamp": timestamp,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_train_gap": float(abs(val_loss - train_loss)),
            "is_best": is_best,
            "parameters": n_params,
            "architecture": {
                "d_h": args.d_h, "d_v": args.d_v,
                "d_m": args.d_m, "d_r": args.d_r,
                "n_predicates": args.n_predicates,
                "n_sets": args.n_sets,
                "n_super_sets": args.n_super_sets,
            },
        },
    }

    # Core logic extraction via audit tool
    if AUDIT_AVAILABLE:
        model.eval()
        with torch.no_grad():
            logic = extract_learned_logic(model, feature_cols)
        model.train()

        # Extend predicates with V3.0 thresholds
        if hasattr(model, 'pred_gates'):
            pg = model.pred_gates
            logic["predicates"]["iv_regime_frac_threshold"] = float(pg.iv_regime_frac_thresh.data)
            logic["predicates"]["put_flow_threshold"] = float(pg.put_flow_thresh.data)
            logic["predicates"]["spread_stress_mult_threshold"] = float(pg.spread_stress_mult_thresh.data)

        report["learned_logic"] = logic
        report["trading_rules"] = generate_trading_rules(logic)
    else:
        # Minimal extraction without audit tool
        report["learned_logic"] = _extract_minimal_logic(model)
        report["trading_rules"] = []

    # A_matrix stability metrics
    try:
        with torch.no_grad():
            A = model.get_A_matrix().cpu().float().numpy()
        eigenvalues = np.linalg.eigvals(A)
        magnitudes = np.abs(eigenvalues)
        spectral_radius = float(np.max(magnitudes))
        top_5 = sorted(magnitudes, reverse=True)[:5]

        # Block norms
        spec = model.spec
        block_norms = {}
        for key, param in model.A_theta.named_parameters():
            if 'weight' in key:
                block_name = key.replace('.weight', '')
                block_norms[block_name] = float(param.data.norm().item())

        report["a_matrix"] = {
            "spectral_radius": spectral_radius,
            "shape": list(A.shape),
            "top_5_eigenvalue_magnitudes": [float(x) for x in top_5],
            "block_norms": block_norms,
        }
    except Exception as e:
        report["a_matrix"] = {"error": str(e)}

    return report


def _extract_minimal_logic(model: nn.Module) -> dict:
    """Fallback extraction when audit tool is not available."""
    logic = {"predicates": {}, "super_set": {}, "output_head": {}}

    if hasattr(model, 'pred_gates'):
        pg = model.pred_gates
        logic["predicates"] = {
            "iv_rank_threshold": float(pg.iv_rank_thresh.data),
            "spread_ratio_threshold": float(pg.spread_frac_thresh.data),
            "rsi_threshold": float(pg.rsi_thresh.data),
            "gap_fraction_threshold": float(pg.gap_frac_thresh.data),
            "gamma_threshold": float(pg.gamma_thresh.data),
            "iv_regime_frac_threshold": float(pg.iv_regime_frac_thresh.data),
            "put_flow_threshold": float(pg.put_flow_thresh.data),
            "spread_stress_mult_threshold": float(pg.spread_stress_mult_thresh.data),
            "steepness": pg.steepness,
        }

    if hasattr(model, 'super_sets') and model.super_sets is not None:
        logic["super_set"]["n_super_sets"] = len(model.super_sets)

    return logic


# =============================================================================
# V5: A_MATRIX CSV EXPORT
# =============================================================================

def export_a_matrix_csv(model: nn.Module, output_path: Path) -> dict:
    """
    Export the full A_matrix to CSV and compute stability metrics.

    Returns dict with spectral_radius, top eigenvalues, etc.
    """
    with torch.no_grad():
        A = model.get_A_matrix().cpu().float().numpy()

    # Save CSV
    np.savetxt(str(output_path), A, delimiter=',', fmt='%.8f')

    # Stability metrics
    eigenvalues = np.linalg.eigvals(A)
    magnitudes = np.abs(eigenvalues)
    spectral_radius = float(np.max(magnitudes))
    top_5 = sorted(magnitudes, reverse=True)[:5]

    return {
        "spectral_radius": spectral_radius,
        "top_5_eigenvalue_magnitudes": [float(x) for x in top_5],
        "shape": list(A.shape),
    }


# =============================================================================
# V5: EPOCH COMPARISON
# =============================================================================

def _cosine_similarity_scalar(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flat arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))


def compute_epoch_comparison(epoch_snapshots: list, best_epoch: int) -> dict:
    """
    Compare each epoch's interpretability snapshot against the best model.

    Args:
        epoch_snapshots: List of dicts with 'epoch', 'train_loss', 'val_loss', 'interpretability'
        best_epoch: Epoch number of the current best model

    Returns:
        Comparison dict for JSON export
    """
    comparison = {
        "best_model": {
            "epoch": best_epoch,
        },
        "comparisons": {},
    }

    # Find best snapshot
    best_snap = None
    for snap in epoch_snapshots:
        if snap['epoch'] == best_epoch:
            best_snap = snap
            comparison["best_model"]["val_loss"] = snap['val_loss']
            comparison["best_model"]["train_loss"] = snap['train_loss']
            break

    if best_snap is None:
        # Best epoch not yet in snapshots (e.g., no epoch passed generalization yet)
        # Compare against first epoch as baseline
        best_snap = epoch_snapshots[0]
        comparison["best_model"]["note"] = "No epoch passed generalization gate yet; comparing against Epoch 1"

    best_logic = best_snap['interpretability'].get('learned_logic', {})

    for snap in epoch_snapshots:
        epoch_key = f"Epoch{snap['epoch']}"

        if snap['epoch'] == best_epoch:
            comparison["comparisons"][epoch_key] = "BEST MODEL - omitted from comparison"
            continue

        epoch_logic = snap['interpretability'].get('learned_logic', {})

        # Compare predicates (canonical thresholds)
        pred_same, pred_diff = _compare_predicates(
            best_logic.get('predicates', {}),
            epoch_logic.get('predicates', {}),
        )

        # Compare sets and super-sets (by weight norms)
        set_same, set_diff = _compare_structure(
            best_logic.get('super_set', {}),
            epoch_logic.get('super_set', {}),
            level='sets',
        )
        ss_same, ss_diff = _compare_structure(
            best_logic.get('super_set', {}),
            epoch_logic.get('super_set', {}),
            level='super_sets',
        )

        # Compare fuzzy gates (output head norms)
        fg_same, fg_diff = _compare_fuzzy_gates(
            best_logic.get('output_head', {}),
            epoch_logic.get('output_head', {}),
        )

        # Spectral radius delta
        best_rho = best_snap['interpretability'].get('a_matrix', {}).get('spectral_radius', 0)
        epoch_rho = snap['interpretability'].get('a_matrix', {}).get('spectral_radius', 0)

        comparison["comparisons"][epoch_key] = {
            "same_predicates": pred_same,
            "different_predicates": pred_diff,
            "same_sets": set_same,
            "different_sets": set_diff,
            "same_super_sets": ss_same,
            "different_super_sets": ss_diff,
            "same_fuzzy_gates": fg_same,
            "different_fuzzy_gates": fg_diff,
            "spectral_radius_delta": round(epoch_rho - best_rho, 6),
            "val_loss": snap['val_loss'],
            "train_loss": snap['train_loss'],
        }

    return comparison


def _compare_predicates(best_preds: dict, epoch_preds: dict, tol: float = 0.01) -> Tuple[int, int]:
    """Compare canonical threshold values with tolerance."""
    same = 0
    diff = 0
    all_keys = set(list(best_preds.keys()) + list(epoch_preds.keys()))
    # Only compare numeric threshold keys
    threshold_keys = [k for k in all_keys if 'threshold' in k.lower() or 'thresh' in k.lower()]

    for key in threshold_keys:
        best_val = best_preds.get(key)
        epoch_val = epoch_preds.get(key)
        if best_val is not None and epoch_val is not None:
            if isinstance(best_val, (int, float)) and isinstance(epoch_val, (int, float)):
                if abs(best_val - epoch_val) <= tol:
                    same += 1
                else:
                    diff += 1

    return same, diff


def _compare_structure(best_ss: dict, epoch_ss: dict, level: str = 'sets') -> Tuple[int, int]:
    """Compare set or super-set structure by counting matching/differing elements."""
    same = 0
    diff = 0

    best_list = best_ss.get('super_sets', [])
    epoch_list = epoch_ss.get('super_sets', [])

    if level == 'super_sets':
        # Compare at super-set level (number and general structure)
        n_best = len(best_list)
        n_epoch = len(epoch_list)
        same = min(n_best, n_epoch)
        diff = abs(n_best - n_epoch)
        return same, diff

    # Sets level: compare within each super-set
    for i, (b_ss, e_ss) in enumerate(zip(best_list, epoch_list)):
        b_sets = b_ss.get('sets', [])
        e_sets = e_ss.get('sets', [])

        for j, (b_set, e_set) in enumerate(zip(b_sets, e_sets)):
            # Compare by operator weights if available
            b_ops = b_set.get('operator_weights', {})
            e_ops = e_set.get('operator_weights', {})
            if b_ops and e_ops:
                # Check if dominant operator is the same
                b_dom = max(b_ops, key=b_ops.get) if b_ops else None
                e_dom = max(e_ops, key=e_ops.get) if e_ops else None
                if b_dom == e_dom:
                    same += 1
                else:
                    diff += 1
            else:
                # Compare by weight norm with 1% tolerance
                b_norm = b_set.get('projection_weight_norm', 0)
                e_norm = e_set.get('projection_weight_norm', 0)
                if b_norm > 0 and abs(b_norm - e_norm) / b_norm < 0.01:
                    same += 1
                else:
                    diff += 1

    return same, diff


def _compare_fuzzy_gates(best_oh: dict, epoch_oh: dict) -> Tuple[int, int]:
    """Compare output head sensitivities."""
    same = 0
    diff = 0

    best_sens = best_oh.get('sensitivities', {})
    epoch_sens = epoch_oh.get('sensitivities', {})

    for target in best_sens:
        if target not in epoch_sens:
            diff += 1
            continue
        # Compare which block drives this target
        b_primary = max(best_sens[target], key=best_sens[target].get) if best_sens[target] else None
        e_primary = max(epoch_sens[target], key=epoch_sens[target].get) if epoch_sens[target] else None
        if b_primary == e_primary:
            same += 1
        else:
            diff += 1

    return same, diff


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_condor_net(args):
    """Main training function for CondorNet v5."""

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
    # Select Feature Schema
    if args.data_version in ["v4.2", "v4.1"]:
        print(f"[CondorNet] Using {args.data_version} feature schema (91 features)")
        active_feature_cols = FEATURE_COLS_V42
    else:
        print(f"[CondorNet] Using v3.0 feature schema (79 features)")
        active_feature_cols = FEATURE_COLS_V30

    # Merge all required cols
    load_cols = list(set(active_feature_cols + target_cols + aux_cols))

    print(f"[CondorNet] Loading data from {args.local_data}...")
    if args.local_data.endswith('.parquet'):
        df = pd.read_parquet(args.local_data)
    else:
        try:
            df = pd.read_csv(args.local_data)
        except Exception as e:
            print(f"[Error] Failed to load data: {e}")
            return
            
    if args.max_rows > 0:
        df = df.iloc[:args.max_rows]

    # Use active_feature_cols instead of global FEATURE_COLS
    X, y, regime, bar_index, med, scale = prepare_features(df, active_feature_cols)

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    r_train, r_val = regime[:split_idx], regime[split_idx:]
    bar_train, bar_val = bar_index[:split_idx], bar_index[split_idx:]
    
    # Datasets
    train_ds = SequenceDataset(X_train, y_train, r_train, args.lookback)
    val_ds = SequenceDataset(X_val, y_val, r_val, args.lookback)

    # GPU-optimized DataLoader config (hardcoded for T4/A100 performance)
    print(f"[CondorNet] DataLoader: workers=4, pin_memory=True, "
          f"persistent=True, prefetch=2")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
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
    print(f"CONDORNET V5 TRAINING LABORATORY")
    print(f"Version: {CN_VERSION}_{DS_VERSION}")
    print(f"{'='*60}")
    print(f"State: d_h={args.d_h}, d_v={args.d_v}, d_m={args.d_m}, d_r={args.d_r}")
    print(f"Total state dim: {model.spec.d_x}")
    print(f"Logic: {args.n_predicates} predicates, {args.n_sets} sets, {args.n_super_sets} super-sets")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Convergence: threshold={args.convergence_threshold}, patience={args.convergence_patience}, active={args.convergence_active}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # TensorBoard
    run_name = f"condor_dh{args.d_h}_lr{args.lr}_bs{args.batch_size}"
    writer = SummaryWriter(log_dir=f"runs/condor_net/{run_name}")
    global_step = 0
    training_start_time = time.time()

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_train_gap = float('inf')
    best_epoch = 0
    patience_counter = 0
    total_steps = args.epochs * n_train_batches
    epoch_snapshots = []  # V5: For epoch comparison reports
    convergence_manager = SmartEpochManager(
        threshold=args.convergence_threshold,
        patience=args.convergence_patience,
        active=args.convergence_active,
    )

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
        convergence_manager.reset()  # V5: Reset convergence tracking per epoch
        epoch_batches_run = 0

        pbar = tqdm(range(n_train_batches), desc=f"Epoch {epoch+1}", leave=False)

        optimizer.zero_grad(set_to_none=True)

        # ==========================================================
        # A0: TRAINING LOOP HOOK ORDER (v4.3 Contract)
        # ==========================================================
        # Each batch executes in this EXACT order:
        #   1. FORWARD      → model(x_b, return_diagnostics=True)
        #   2. LOSS          → criterion(outputs, targets)
        #   3. BACKWARD      → loss.backward()
        #   4. OPTIMIZER     → optimizer.step()
        #   5. DEEP OBSERVE  → capture diagnostics, write JSON
        #   6. CHECKPOINT    → atomic_save (if triggered)
        #   7. TELEMETRY     → emit to GUI/dashboard
        #
        # Rules:
        #   - No diagnostic capture before BACKWARD (gradients unavailable)
        #   - No checkpoint before DEEP OBSERVE (report must cover save batch)
        #   - All hooks respect this ordering strictly
        # ==========================================================

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
            epoch_batches_run += 1
            for k, v in components.items():
                epoch_components[k] += v.item() if torch.is_tensor(v) else v

            # V5: Convergence tracking
            should_stop = convergence_manager.update(loss.item(), model)
            if should_stop:
                print(f"\n  [CONVERGENCE] Epoch {epoch+1} stopped at batch {batch_idx+1}/{n_train_batches} "
                      f"(best_loss={convergence_manager.best_loss_in_window:.6f})")
                break

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

                # Emit fuzzy gate activations for heatmap
                if 'predicates' in diag and diag['predicates'] is not None:
                    preds = diag['predicates']
                    # Average across batch, take first 10 gates
                    if preds.dim() >= 2:
                        activations = preds.mean(dim=0)[:10].detach().cpu().tolist()
                    else:
                        activations = preds[:10].detach().cpu().tolist()
                    emitter.emit_fuzzy(
                        step=global_step,
                        epoch=epoch + 1,
                        activations=activations,
                    )

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, epoch_batches_run)

        # V5: If convergence manager captured a snapshot, restore it for validation
        if convergence_manager.has_snapshot:
            print(f"  [V5] Restoring convergence snapshot for validation (loss={convergence_manager.best_loss_in_window:.6f})")
            model.load_state_dict(convergence_manager.best_snapshot)

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

        # ============================================================
        # V5: ALWAYS save epoch checkpoint
        # ============================================================
        epoch_ts = datetime.now().strftime("%m%d%Y_%H%M%S")
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        epoch_ckpt_name = f"Epoch{epoch+1}_{CN_VERSION}_{DS_VERSION}_{epoch_ts}.pth"
        checkpoint_path = checkpoint_dir / epoch_ckpt_name

        epoch_checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'version': f"{CN_VERSION}_{DS_VERSION}",
            'state_dims': {
                'd_h': args.d_h, 'd_v': args.d_v,
                'd_m': args.d_m, 'd_r': args.d_r,
            },
            'hyperparameters': {
                'n_predicates': args.n_predicates,
                'n_sets': args.n_sets,
                'n_super_sets': args.n_super_sets,
                'n_layers': args.n_layers,
                'd_control': args.d_control,
            },
            'feature_cols': active_feature_cols,
            'normalization': {'median': med.tolist(), 'scale': scale.tolist()},
        }
        saved_ok = atomic_save(epoch_checkpoint_data, checkpoint_path)
        if saved_ok:
            print(f"  -> Checkpoint saved: {epoch_ckpt_name}")
            rotate_checkpoints(checkpoint_dir, keep_n=args.keep_ckpts)
        else:
            print(f"  -> WARNING: Checkpoint save FAILED for {epoch_ckpt_name}")

        # ============================================================
        # V5: Interpretability report
        # ============================================================
        interp_report = extract_epoch_interpretability(
            model, epoch + 1, avg_train_loss, avg_val_loss,
            active_feature_cols, args, is_best=False,
        )
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        interp_path = reports_dir / f"Epoch{epoch+1}_{CN_VERSION}_{DS_VERSION}_{epoch_ts}.json"
        with open(interp_path, 'w') as f:
            json.dump(interp_report, f, indent=2)
        print(f"  -> Interpretability: {interp_path.name}")

        # Store snapshot for epoch comparison
        epoch_snapshots.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'interpretability': interp_report,
            'timestamp': epoch_ts,
        })

        # ============================================================
        # V5: A_matrix CSV export
        # ============================================================
        a_matrix_dir = Path("models/A_matrices")
        a_matrix_dir.mkdir(parents=True, exist_ok=True)
        a_csv_path = a_matrix_dir / f"Epoch{epoch+1}_A_Matrix.csv"
        a_stability = export_a_matrix_csv(model, a_csv_path)
        print(f"  -> A_matrix: {a_csv_path.name} (rho={a_stability['spectral_radius']:.4f})")

        # ============================================================
        # V5: Best-model logic with generalization gate
        # ============================================================
        val_train_gap = abs(avg_val_loss - avg_train_loss)

        if avg_val_loss < avg_train_loss:
            # Passes generalization check
            is_new_best = False
            if best_val_loss == float('inf'):
                is_new_best = True
                reason = "First epoch passing generalization"
            elif avg_val_loss < best_val_loss and val_train_gap < best_val_train_gap:
                is_new_best = True
                reason = (f"Better val ({avg_val_loss:.4f} < {best_val_loss:.4f}) "
                          f"AND tighter gap ({val_train_gap:.4f} < {best_val_train_gap:.4f})")
            elif avg_val_loss < best_val_loss:
                is_new_best = True
                reason = (f"Better val ({avg_val_loss:.4f} < {best_val_loss:.4f}), "
                          f"gap widened ({val_train_gap:.4f} vs {best_val_train_gap:.4f})")
            else:
                reason = (f"Val not better ({avg_val_loss:.4f} >= {best_val_loss:.4f}), "
                          f"best remains Epoch {best_epoch}")

            if is_new_best:
                best_val_loss = avg_val_loss
                best_train_loss = avg_train_loss
                best_val_train_gap = val_train_gap
                best_epoch = epoch + 1
                patience_counter = 0

                # Update interpretability report flag
                interp_report['summary']['is_best'] = True
                with open(interp_path, 'w') as f:
                    json.dump(interp_report, f, indent=2)

                # Save/overwrite best model
                os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
                best_checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': best_val_loss,
                    'val_train_gap': val_train_gap,
                    'version': f"{CN_VERSION}_{DS_VERSION}",
                    'state_dims': {
                        'd_h': args.d_h, 'd_v': args.d_v,
                        'd_m': args.d_m, 'd_r': args.d_r,
                    },
                    'hyperparameters': {
                        'n_predicates': args.n_predicates,
                        'n_sets': args.n_sets,
                        'n_super_sets': args.n_super_sets,
                        'n_layers': args.n_layers,
                        'd_control': args.d_control,
                    },
                    'feature_cols': active_feature_cols,
                    'normalization': {'median': med.tolist(), 'scale': scale.tolist()},
                }
                saved_ok = atomic_save(best_checkpoint_data, args.output)
                if saved_ok:
                    print(f"  -> NEW BEST MODEL (Epoch {epoch+1}): {reason}")
                    print(f"     Saved: {args.output}")
                else:
                    print(f"  -> WARNING: Best model save FAILED for Epoch {epoch+1}")
            else:
                patience_counter += 1
                print(f"  -> Epoch {epoch+1} generalizes but NOT new best: {reason}")
        else:
            patience_counter += 1
            print(f"  -> Epoch {epoch+1} OVERFITTING: val_loss ({avg_val_loss:.4f}) >= train_loss ({avg_train_loss:.4f})")

        # ============================================================
        # V5: Epoch comparison (after epoch >= 2)
        # ============================================================
        if len(epoch_snapshots) >= 2:
            comparison = compute_epoch_comparison(epoch_snapshots, best_epoch)
            # Rotate file: delete old, write new
            for old_file in glob.glob(str(reports_dir / "EPOCH_Comparison_*.json")):
                os.remove(old_file)
            comp_path = reports_dir / f"EPOCH_Comparison_{epoch_ts}.json"
            with open(comp_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"  -> Comparison: {comp_path.name}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    print(f"\n{'='*60}")
    print(f"CONDORNET V5 TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Best train loss: {best_train_loss:.4f}")
    print(f"Best val-train gap: {best_val_train_gap:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"Epoch checkpoints: {args.checkpoint_dir}/")
    print(f"Interpretability reports: reports/")
    print(f"A_matrices: models/A_matrices/")
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
