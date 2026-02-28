"""
condor_train_net_v43.py — CondorNet™ v4.3 Training Loop
========================================================
Multi-timeframe (M1/M5/M15/H1) + Options Chain training for CondorNetV43.

Key differences from v42:
  - V43Dataset loads 4 TF CSVs + options chain CSV
  - Custom collate_fn right-pads ragged chain grids
  - 9-component loss with linear weight annealing
  - Pivot prediction BCE loss
  - Crash sentinel (.training_active file)
  - --resume flag for checkpoint continuation

Implements F7.1–F7.12 from task_checklist_v43_full.md.

Author: CondorNet v4.3 Implementation
Date: 2026-02-23
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import shutil
import sys
import time
import warnings
import argparse

# Suppress spurious PyTorch warning: GradScaler's scaler.step(optimizer) registers
# the optimizer step but PyTorch's scheduler tracker may not see it, causing a
# false-positive warning about scheduler.step() order. The actual order is correct.
warnings.filterwarnings(
    "ignore",
    message=".*Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`.*",
)
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# VERSION CONSTANTS
# =============================================================================
CN_VERSION = "CNv43"
DS_VERSION = "DSv43"

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

from intelligence.condor_brain_net_v43 import (
    CondorNetV43,
    CondorNetOutput,
    build_condornet_v43,
)
from intelligence.schema_v43 import (
    SCHEMA_VERSION,
    TF_FEATURE_NAMES,
    TF_PIVOT_FEATURES,
    TF_LABEL_NAMES,
    TF_SPARSE_FEATURES,
    CHAIN_FEATURE_NAMES,
    CHAIN_GRID_CONFIG,
    STRATEGY_TYPES,
    N_STRATEGY_TYPES,
    N_PIVOT_FEATURES,
    ABSTAIN_IDX,
    NORMALIZATION_PASSTHROUGH,
    FRICTION_FEATURE_NAMES,
    TOD_FEATURE_NAMES,
    REGIME_PERSISTENCE_FEATURE,
    IVR_REVERSAL_FEATURE_NAMES,
    POS_STATE_NAMES,
    N_POS_STATE,
)

# Full 64-feature name list in the exact order they are concatenated into the
# model's input tensor: 52 base TF features + 5 friction gates + 2 ToD + 1
# regime persistence + first 4 IVR reversal features (price_stretch, ivr_zone,
# stretch_zone, reversal_score).  Indices 52-63 previously rendered as feat_N.
_FULL_FEATURE_NAMES: List[str] = (
    TF_FEATURE_NAMES
    + FRICTION_FEATURE_NAMES                # [52] friction_ok_5/10/20/40/60
    + TOD_FEATURE_NAMES                     # [57] tod_sin, tod_cos
    + [REGIME_PERSISTENCE_FEATURE]          # [59] regime_persistence
    + IVR_REVERSAL_FEATURE_NAMES[:4]        # [60] price_stretch, ivr_zone, stretch_zone, reversal_score
)

# GUI Telemetry (optional)
try:
    from gui.backend.services.training_emitter import get_emitter
    GUI_TELEMETRY_AVAILABLE = True
except ImportError:
    GUI_TELEMETRY_AVAILABLE = False
    def get_emitter(backend_url=None):
        return None

EPS = 1e-6


# =============================================================================
# GATE DIVERSITY REGULARIZER (Run 7)
# =============================================================================

def gate_diversity_loss(lambda_probs: torch.Tensor,
                        std_target: float = 0.14,
                        alpha: float = 0.01,
                        eps: float = 1e-3) -> torch.Tensor:
    """
    Enforce non-collapse of gating distribution.

    Penalizes runs where std(λ) < std_target — smooth hinge fires only when
    routing diversity is below target, gradient is zero above it.

        L = alpha * softplus((std_target - std(λ)) / eps) * eps

    Args:
        lambda_probs: on-graph super_gate tensor (outputs.final_gate), any shape
        std_target:   target minimum routing std (default 0.14)
        alpha:        loss coefficient (pre-multiplied, managed by compute_alpha)
        eps:          softplus smoothing scale

    Returns:
        scalar loss tensor (differentiable zero when diversity is sufficient)
    """
    if lambda_probs is None:
        return torch.tensor(0.0, device='cpu')

    gate_flat = lambda_probs.float().view(-1)
    if gate_flat.numel() < 2:
        return gate_flat.new_zeros(())

    # std across all gate values in the batch (our gate is scalar-per-sample)
    std_mean = gate_flat.std(unbiased=False)

    # smooth hinge: zero when std >= std_target
    hinge = F.softplus((std_target - std_mean) / eps) * eps
    return alpha * hinge


def compute_alpha(epoch: int, total_epochs: int,
                  alpha_max: float = 0.01,
                  warmup_frac: float = 0.3) -> float:
    """
    Linear α ramp: 0 → alpha_max over first warmup_frac fraction of epochs.
    Constant at alpha_max thereafter.

    Special case: warmup_frac=0.0 returns alpha_max immediately (no ramp,
    full strength from epoch 0 / batch 0).  Without this guard, max(1, 0)=1
    would cause epoch 0 to still ramp from zero, defeating --logit-warmup-frac 0.0.
    """
    if warmup_frac <= 0.0:
        return alpha_max
    warmup_epochs = max(1, int(total_epochs * warmup_frac))
    if epoch >= warmup_epochs:
        return alpha_max
    return alpha_max * (epoch / warmup_epochs)


def _parse_epoch_id(eid) -> int:
    """
    Normalize epoch identifier to int.
    Handles: int 30, '30', 'Epoch30', 'EpEpoch30'.
    Extracts the trailing contiguous digit sequence.
    """
    if isinstance(eid, int):
        return eid
    s = str(eid)
    digits = ''
    for c in reversed(s):
        if c.isdigit():
            digits = c + digits
        elif digits:
            break
    return int(digits) if digits else 0


# =============================================================================
# LOGIT-SPACE GATE REGULARIZERS (Run 8)
# =============================================================================

def _logit_safe(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Numerically stable logit: log(p) - log(1-p) with clamped input."""
    p = p.float().clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def logit_variance_floor(lambda_probs: torch.Tensor,
                         var_target: float = 2.0,
                         alpha: float = 0.02,
                         eps: float = 1e-3,
                         logit_eps: float = 1e-4) -> torch.Tensor:
    """
    Enforce Var(z_approx) >= var_target, where z_approx = logit(λ).

    Attacks σ_z contraction at the source: if the pre-sigmoid logit distribution
    collapses toward a spike at 0, this hinge pushes variance back up.
    Gradient flows through λ → hierarchical_logic → relational layers.

    var_target=2.0 corresponds to z_std≈1.4, matching healthy epoch-1 gate_logit
    std observed in Run 7.
    """
    if lambda_probs is None:
        return torch.tensor(0.0, device='cpu')
    z = _logit_safe(lambda_probs, eps=logit_eps).reshape(-1)
    if z.numel() < 2:
        return z.new_zeros(())
    var_z = z.var(unbiased=False)
    hinge = F.softplus((var_target - var_z) / eps) * eps
    return alpha * hinge


def logit_iqr_floor(lambda_probs: torch.Tensor,
                    iqr_target: float = 1.0,
                    alpha: float = 0.01,
                    eps: float = 1e-3,
                    logit_eps: float = 1e-4) -> torch.Tensor:
    """
    Enforce IQR(z_approx) >= iqr_target to prevent bulk-degeneracy.

    Variance can be inflated by a hard-negative outlier tail while the bulk
    (middle 50%) stays pinned near 0 (IQR≈0.024, as observed in Run 7 epoch 15).
    This hinge forces the middle 50% to spread, directly addressing the bimodal
    pathology where most mass sits at λ≈0.5 and a sparse tail saturates hard-off.
    """
    if lambda_probs is None:
        return torch.tensor(0.0, device='cpu')
    z = _logit_safe(lambda_probs, eps=logit_eps).reshape(-1)
    if z.numel() < 2:
        return z.new_zeros(())
    q25 = torch.quantile(z, 0.25)
    q75 = torch.quantile(z, 0.75)
    iqr = q75 - q25
    hinge = F.softplus((iqr_target - iqr) / eps) * eps
    return alpha * hinge


def logit_mad_floor(lambda_probs: torch.Tensor,
                    mad_target: float = 0.9,
                    alpha: float = 0.05,
                    eps: float = 1e-3,
                    logit_eps: float = 1e-4) -> torch.Tensor:
    """
    Enforce mean(|z_approx|) >= mad_target where z_approx = logit(λ).

    Variance and IQR floors permit small symmetric distributions that happen to
    be centered at 0 — e.g. z ~ N(0, 0.1) gives var=0.01 and IQR≈0.13, both
    low, but also mean|z|≈0.08.  MAD directly pushes the distribution *away
    from zero mass*, complementing the spread penalties.

    mad_target=0.9 corresponds to N(0, σ≈1.1), consistent with the healthy
    gate_logit magnitude observed at epoch 1 of Run 7/8.
    """
    if lambda_probs is None:
        return torch.tensor(0.0, device='cpu')
    z = _logit_safe(lambda_probs, eps=logit_eps).reshape(-1)
    if z.numel() < 2:
        return z.new_zeros(())
    mad = z.abs().mean()
    hinge = F.softplus((mad_target - mad) / eps) * eps
    return alpha * hinge


def logit_tail_penalty(lambda_probs: torch.Tensor,
                       clip_band: float = 8.0,
                       alpha: float = 0.01,
                       logit_eps: float = 1e-4) -> torch.Tensor:
    """
    Squared-excess penalty on gate logit magnitudes beyond a safe band.

    Targets the runaway hard-off / hard-on tail that variance, IQR, and MAD
    floors cannot stop — those controls act on the distribution bulk while a
    handful of extreme outliers (z → -33) sail past undetected.

    Loss fires only on the excess beyond |z| > clip_band:

        L_tail = alpha * E[max(0, |z| - clip_band)²]

    At clip_band=8, σ(±8) = 0.9997 — the gate is already fully saturated, so
    any further magnitude growth is pure waste and kills gradient flow for those
    gates.  The squared excess gives a smooth, proportional pull-back.

    Intentionally OUTSIDE autocast (fp32) — consistent with other logit
    regularizers in this file.

    Args:
        lambda_probs: final_gate tensor, any shape, values in (0,1)
        clip_band:    safe |z| radius; excess beyond this is penalized (default 8.0)
        alpha:        loss coefficient (default 0.01)
        logit_eps:    clamping eps fed into _logit_safe (default 1e-4)

    Returns:
        scalar loss tensor (zero when all |z| <= clip_band)
    """
    if lambda_probs is None:
        return torch.tensor(0.0, device='cpu')
    z = _logit_safe(lambda_probs, eps=logit_eps).reshape(-1)
    if z.numel() < 1:
        return z.new_zeros(())
    excess = (z.abs() - clip_band).clamp(min=0.0)   # max(0, |z| - L)
    return alpha * excess.pow(2).mean()


# =============================================================================
# CRASH SENTINEL
# =============================================================================
SENTINEL_PATH = Path(".training_active")

def write_sentinel():
    """Write crash sentinel file on training start."""
    SENTINEL_PATH.write_text(
        f"pid={os.getpid()}\nstarted={datetime.now().isoformat()}\nversion={CN_VERSION}_{DS_VERSION}\n"
    )

def remove_sentinel():
    """Remove crash sentinel on clean exit."""
    if SENTINEL_PATH.exists():
        SENTINEL_PATH.unlink()

def check_sentinel() -> bool:
    """Check if a previous training crashed (sentinel exists)."""
    return SENTINEL_PATH.exists()


# =============================================================================
# ROBUST NORMALIZATION
# =============================================================================

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


def safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =============================================================================
# ATOMIC SAVE & CHECKPOINT ROTATION
# =============================================================================

def atomic_save(data: dict, target_path, validate: bool = True):
    """Save checkpoint atomically: write to .tmp, validate, then rename."""
    target_path = Path(target_path)
    tmp_path = target_path.with_suffix('.tmp')
    try:
        torch.save(data, tmp_path)
        if validate:
            _ = torch.load(tmp_path, map_location='cpu', weights_only=False)
        shutil.move(str(tmp_path), str(target_path))
        return True
    except Exception as e:
        print(f"  [SAVE ERROR] Failed to save {target_path.name}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def rotate_checkpoints(checkpoint_dir, keep_n: int = 3):
    """Keep only the N most recent .pth checkpoints."""
    if keep_n <= 0:
        return
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = sorted(
        checkpoint_dir.glob("Epoch*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_ckpt in ckpts[keep_n:]:
        try:
            old_ckpt.unlink()
            print(f"  [ROTATE] Removed old checkpoint: {old_ckpt.name}")
        except Exception as e:
            print(f"  [ROTATE] Failed to remove {old_ckpt.name}: {e}")


# =============================================================================
# V43 DATASET
# =============================================================================

class V43Dataset(torch.utils.data.Dataset):
    """
    Multi-timeframe + options chain dataset for CondorNet v4.3.

    Each sample returns:
        x_m1:      [T, d_tf]    M1 features
        x_m5:      [T, d_tf]    M5 features (canonical TF)
        x_m15:     [T, d_tf]    M15 features
        x_h1:      [T, d_tf]    H1 features
        chain:     [N, 10]      Options chain grid (variable N per sample)
        chain_mask:[N]          Bool — True=padded
        pivot_m5:  [T, 13]      Pivot mask for M5 (True=NaN)
        pos_state: [T, 11]      Position state vector (Phase 3; zeros when absent)
        targets:   dict         Training targets (strategy_label, pop, ev, etc.)
    """

    def __init__(
        self,
        m1_features:   np.ndarray,           # [N_total, d_tf]
        m5_features:   np.ndarray,           # [N_total, d_tf]
        m15_features:  np.ndarray,           # [N_total, d_tf]
        h1_features:   np.ndarray,           # [N_total, d_tf]
        m5_pivots:     np.ndarray,           # [N_total, 13] — NaN-bearing pivot features
        labels:        dict,                  # {name: np.ndarray[N_total, ...]}
        chain_snapshots: dict,               # {date_str: np.ndarray[N_contracts, 10]}
        m5_dates:      np.ndarray,           # [N_total] date strings for matching options
        seq_len:       int = 64,
        m5_pos_state:  Optional[np.ndarray] = None,  # [N_total, 11] Phase 3 (None → zeros)
    ):
        self.m1 = m1_features
        self.m5 = m5_features
        self.m15 = m15_features
        self.h1 = h1_features
        self.pivots = m5_pivots
        self.labels = labels
        self.chain_snapshots = chain_snapshots
        self.m5_dates = m5_dates
        self.seq_len = seq_len
        self.pos_state = m5_pos_state   # None if Phase 3 not yet computed
        self.N = len(m5_features)

    def __len__(self):
        return max(0, self.N - self.seq_len)

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len

        # TF sequences [T, d_tf]
        x_m1 = torch.tensor(self.m1[s:e], dtype=torch.float32)
        x_m5 = torch.tensor(self.m5[s:e], dtype=torch.float32)
        x_m15 = torch.tensor(self.m15[s:e], dtype=torch.float32)
        x_h1 = torch.tensor(self.h1[s:e], dtype=torch.float32)

        # Pivot mask: True where NaN (no pivot event)
        pivot_raw = self.pivots[s:e]  # [T, 13]
        pivot_mask = torch.tensor(np.isnan(pivot_raw), dtype=torch.bool)
        pivot_values = torch.tensor(
            np.nan_to_num(pivot_raw, nan=0.0), dtype=torch.float32
        )

        # Chain snapshot for the last timestep in the window
        target_idx = e - 1
        target_date = self.m5_dates[target_idx]
        if target_date in self.chain_snapshots:
            chain = torch.tensor(self.chain_snapshots[target_date], dtype=torch.float32)
            chain_mask = torch.zeros(chain.shape[0], dtype=torch.bool)
        else:
            # No chain data — single dummy contract, fully masked
            chain = torch.zeros(1, len(CHAIN_FEATURE_NAMES), dtype=torch.float32)
            chain_mask = torch.ones(1, dtype=torch.bool)

        # Labels at target timestep
        target_labels = {}
        for key, arr in self.labels.items():
            target_labels[key] = torch.tensor(arr[target_idx], dtype=torch.float32)

        # Position state [T, 11] — zeros when Phase 3 not yet computed
        if self.pos_state is not None:
            pos_state = torch.tensor(self.pos_state[s:e], dtype=torch.float32)
        else:
            pos_state = torch.zeros(self.seq_len, N_POS_STATE, dtype=torch.float32)

        return {
            'x_m1': x_m1,
            'x_m5': x_m5,
            'x_m15': x_m15,
            'x_h1': x_h1,
            'chain': chain,
            'chain_mask': chain_mask,
            'pivot_mask': pivot_mask,
            'pivot_values': pivot_values,
            'pos_state': pos_state,
            'labels': target_labels,
        }


# =============================================================================
# COLLATE FUNCTION (right-pad ragged chain grids)
# =============================================================================

def v43_collate_fn(batch: list) -> dict:
    """
    Custom collate for V43Dataset.

    Right-pads chain grids to the max N_contracts in the batch.
    Generates chain_mask BoolTensor (True=padded).

    F7.11: Right-pad chain grids to batch max N_contracts.
    """
    B = len(batch)

    # Stack fixed-size tensors
    x_m1 = torch.stack([b['x_m1'] for b in batch])
    x_m5 = torch.stack([b['x_m5'] for b in batch])
    x_m15 = torch.stack([b['x_m15'] for b in batch])
    x_h1 = torch.stack([b['x_h1'] for b in batch])
    pivot_mask = torch.stack([b['pivot_mask'] for b in batch])
    pivot_values = torch.stack([b['pivot_values'] for b in batch])
    pos_state = torch.stack([b['pos_state'] for b in batch])

    # Pad chain grids to max N in batch
    chain_lengths = [b['chain'].shape[0] for b in batch]
    max_n = max(chain_lengths)
    n_feat = batch[0]['chain'].shape[1]  # 10

    chains_padded = torch.zeros(B, max_n, n_feat, dtype=torch.float32)
    chain_masks = torch.ones(B, max_n, dtype=torch.bool)  # True=padded

    for i, b in enumerate(batch):
        n = chain_lengths[i]
        chains_padded[i, :n, :] = b['chain']
        chain_masks[i, :n] = b['chain_mask']

    # Collate labels
    label_keys = batch[0]['labels'].keys()
    labels = {}
    for key in label_keys:
        labels[key] = torch.stack([b['labels'][key] for b in batch])

    return {
        'x_m1': x_m1,
        'x_m5': x_m5,
        'x_m15': x_m15,
        'x_h1': x_h1,
        'chain': chains_padded,
        'chain_mask': chain_masks,
        'pivot_mask': pivot_mask,
        'pivot_values': pivot_values,
        'pos_state': pos_state,
        'labels': labels,
    }


# =============================================================================
# LOSS WEIGHT ANNEALING
# =============================================================================

def load_loss_weights(config_path: str = "configs/loss_weights_v43.json") -> dict:
    """Load loss weight configuration from JSON."""
    with open(config_path) as f:
        return json.load(f)


def get_annealed_weight(
    initial: float, schedule: Optional[dict], epoch: int
) -> float:
    """Compute annealed weight at given epoch using linear interpolation."""
    if schedule is None:
        return initial
    start = schedule["start_epoch"]
    end = schedule["end_epoch"]
    end_weight = schedule["end_weight"]
    if epoch <= start:
        return initial
    if epoch >= end:
        return end_weight
    # Linear interpolation
    t = (epoch - start) / (end - start)
    return initial + t * (end_weight - initial)


# =============================================================================
# CONDOR LOSS V43 — 9-COMPONENT COMPOSITE LOSS
# =============================================================================

class CondorLossV43(nn.Module):
    """
    11-component composite loss for CondorNet v4.3.

    Components:
        1.  strategy_ce:           Cross-entropy over 10 strategy classes
        2.  pop_bce:               Binary cross-entropy for PoP calibration
        3.  ev_mse:                MSE for Expected Value prediction
        4.  risk_mse:              MSE for VaR + CVaR + max_loss combined
        5.  size_mse:              MSE for fuzzy position size output
        6. spot_mse:     MSE for target spot price (auxiliary)
        7. fuzzy_var:    Variance penalty on gate outputs
        8.  pattern_ent:           Entropy regularizer on predicate activations
        9.  robust:                Huber loss for outlier-robust EV/risk
        10. consensus_consistency: MTF gate shaping (entry_signal vs agreement)
        11. exit_bce:              Binary CE for SimExit head (0.0 placeholder → SimExit engine)

    Supports linear weight annealing per configs/loss_weights_v43.json.
    """

    def __init__(self, loss_config: dict):
        super().__init__()
        self.initial_weights = loss_config["initial"]
        schedules = loss_config.get("annealing", {}).get("schedules", {})
        self.schedules = schedules
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for weight annealing."""
        self.current_epoch = epoch

    def _get_weight(self, name: str) -> float:
        """Get current weight for a loss component (with annealing)."""
        initial = self.initial_weights.get(name, 0.0)
        schedule = self.schedules.get(name, None)
        return get_annealed_weight(initial, schedule, self.current_epoch)

    def forward(
        self,
        outputs: CondorNetOutput,
        labels: dict,
        gates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the 9-component loss.

        Args:
            outputs: CondorNetOutput from model forward pass
            labels:  dict with keys: strategy_label, pop, ev, max_loss,
                     var_95, cvar_95, position_size, target_spot
            gates:   Optional predicate gate activations for entropy/variance

        Returns:
            (total_loss, component_dict)
        """
        device = outputs.strategy_logits.device
        components: Dict[str, torch.Tensor] = {}

        # === 1. Strategy Cross-Entropy ===
        if 'strategy_label' in labels:
            strategy_targets = labels['strategy_label'].long()
            components['strategy_ce'] = F.cross_entropy(
                outputs.strategy_logits, strategy_targets
            )
        else:
            components['strategy_ce'] = torch.tensor(0.0, device=device)

        # === 2. PoP BCE ===
        # F.binary_cross_entropy is blocked inside ANY autocast context (not dtype-dependent).
        # Disable autocast for this specific call only; .float() ensures float32 inputs.
        if 'pop' in labels:
            pop_target = labels['pop'].unsqueeze(-1) if labels['pop'].dim() == 1 else labels['pop']
            with torch.amp.autocast(device_type=device.type, enabled=False):
                components['pop_bce'] = F.binary_cross_entropy(
                    outputs.pop.float().clamp(1e-7, 1 - 1e-7), pop_target.float()
                )
        else:
            components['pop_bce'] = torch.tensor(0.0, device=device)

        # === 3. EV MSE ===
        if 'ev' in labels:
            ev_target = labels['ev'].unsqueeze(-1) if labels['ev'].dim() == 1 else labels['ev']
            components['ev_mse'] = F.mse_loss(outputs.ev, ev_target)
        else:
            components['ev_mse'] = torch.tensor(0.0, device=device)

        # === 4. Risk MSE (VaR + CVaR + max_loss combined) ===
        risk_loss = torch.tensor(0.0, device=device)
        risk_count = 0
        if 'max_loss' in labels:
            ml_target = labels['max_loss'].unsqueeze(-1) if labels['max_loss'].dim() == 1 else labels['max_loss']
            risk_loss = risk_loss + F.mse_loss(outputs.max_loss, ml_target)
            risk_count += 1
        if 'var_95' in labels:
            var_target = labels['var_95'].unsqueeze(-1) if labels['var_95'].dim() == 1 else labels['var_95']
            risk_loss = risk_loss + F.mse_loss(outputs.var_95, var_target)
            risk_count += 1
        if 'cvar_95' in labels:
            cvar_target = labels['cvar_95'].unsqueeze(-1) if labels['cvar_95'].dim() == 1 else labels['cvar_95']
            risk_loss = risk_loss + F.mse_loss(outputs.cvar_95, cvar_target)
            risk_count += 1
        components['risk_mse'] = risk_loss / max(risk_count, 1)

        # === 5. Size MSE ===
        if 'position_size' in labels:
            size_target = labels['position_size'].unsqueeze(-1) if labels['position_size'].dim() == 1 else labels['position_size']
            components['size_mse'] = F.mse_loss(outputs.position_size, size_target)
        else:
            components['size_mse'] = torch.tensor(0.0, device=device)

        # === 6. Spot MSE (auxiliary task) ===
        if 'target_spot' in labels:
            spot_target = labels['target_spot'].unsqueeze(-1) if labels['target_spot'].dim() == 1 else labels['target_spot']
            components['spot_mse'] = F.mse_loss(outputs.spot_pred, spot_target)
        else:
            components['spot_mse'] = torch.tensor(0.0, device=device)

        # === 7. Fuzzy Variance ===
        if gates is not None and gates.numel() > 0:
            gate_probs = torch.sigmoid(gates) if gates.min() < 0 else gates
            components['fuzzy_var'] = torch.var(gate_probs.mean(dim=0))
        else:
            components['fuzzy_var'] = torch.tensor(0.0, device=device)

        # === 8. Pattern Entropy ===
        if gates is not None and gates.numel() > 0:
            gate_mean = gates.mean(dim=0).clamp(1e-8, 1 - 1e-8)
            entropy = -(gate_mean * torch.log(gate_mean)).sum()
            components['pattern_ent'] = -entropy  # Minimize negative entropy = maximize entropy
        else:
            components['pattern_ent'] = torch.tensor(0.0, device=device)

        # === 9. Robust (Huber) ===
        robust_loss = torch.tensor(0.0, device=device)
        if 'ev' in labels:
            ev_target = labels['ev'].unsqueeze(-1) if labels['ev'].dim() == 1 else labels['ev']
            robust_loss = robust_loss + F.smooth_l1_loss(outputs.ev, ev_target)
        if 'max_loss' in labels:
            ml_target = labels['max_loss'].unsqueeze(-1) if labels['max_loss'].dim() == 1 else labels['max_loss']
            robust_loss = robust_loss + F.smooth_l1_loss(outputs.max_loss, ml_target)
        components['robust'] = robust_loss

        # === 10. MTF Consensus Consistency ===
        # When all TFs strongly agree (high consensus_strength), entry_signal should
        # be confident. This shapes the MTFConsensusGate to learn that agreement
        # is a reliable entry signal.
        #
        # Loss: cs × (1 − entry_signal)  — penalizes low entry when consensus is high.
        # Weight 0.05 keeps it a soft hint rather than a hard constraint.
        if outputs.consensus_strength is not None:
            cs = outputs.consensus_strength.squeeze(-1)   # [B] ∈ [0,1]
            es = outputs.entry_signal.squeeze(-1).clamp(1e-7, 1 - 1e-7)   # [B]
            with torch.amp.autocast(device_type=device.type, enabled=False):
                consistency_loss = (cs.float() * (1.0 - es.float())).mean()
            components['consensus_consistency'] = consistency_loss
        else:
            components['consensus_consistency'] = torch.tensor(0.0, device=device)

        # === 11. Exit BCE (SimExit head) ===
        # Binary cross-entropy against exit_signal label (0=hold, 1=exit).
        # Placeholder labels are 0.0; loss is structurally 0 until the
        # SimExit labeling engine populates non-zero targets.
        # Mirror pop_bce pattern: autocast disabled, .float() + clamp.
        if 'exit_signal' in labels and outputs.exit_signal is not None:
            exit_target = labels['exit_signal'].unsqueeze(-1) if labels['exit_signal'].dim() == 1 else labels['exit_signal']
            with torch.amp.autocast(device_type=device.type, enabled=False):
                components['exit_bce'] = F.binary_cross_entropy(
                    outputs.exit_signal.float().clamp(1e-7, 1 - 1e-7), exit_target.float()
                )
        else:
            components['exit_bce'] = torch.tensor(0.0, device=device)

        # === Total Loss (weighted sum with annealing) ===
        total_loss = torch.tensor(0.0, device=device)
        for name, value in components.items():
            w = self._get_weight(name)
            total_loss = total_loss + w * value

        return total_loss, components


# =============================================================================
# DATA LOADING — 4 TFs + OPTIONS CHAIN
# =============================================================================

def _load_tf_csv(path: str, tf_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single TF dataset CSV.

    Returns:
        features:  [N, d_tf] — non-pivot input features
        pivots:    [N, 13]   — raw pivot columns (NaN-bearing)
        labels:    [N, 2]    — target_spot, position_size_mult
    """
    print(f"  [DATA] Loading {tf_name} from {path}...")
    df = pd.read_csv(path)
    print(f"    -> {len(df)} rows, {len(df.columns)} columns")

    # Extract feature columns (schema order; missing columns → 0)
    feature_cols = [c for c in TF_FEATURE_NAMES if c in df.columns]
    missing = [c for c in TF_FEATURE_NAMES if c not in df.columns]
    if missing:
        print(f"    ⚠  Missing {len(missing)} features: {missing[:5]}...")

    features = df[feature_cols].values.astype(np.float32)
    if len(feature_cols) < len(TF_FEATURE_NAMES):
        pad_width = len(TF_FEATURE_NAMES) - len(feature_cols)
        features = np.hstack([features, np.zeros((len(df), pad_width), dtype=np.float32)])

    # Extract pivot columns (keep NaN — semantically meaningful)
    pivot_cols = [c for c in TF_PIVOT_FEATURES if c in df.columns]
    if pivot_cols:
        pivots = df[pivot_cols].values.astype(np.float64)
        if len(pivot_cols) < N_PIVOT_FEATURES:
            pivots = np.hstack([
                pivots,
                np.full((len(df), N_PIVOT_FEATURES - len(pivot_cols)), np.nan)
            ])
    else:
        pivots = np.full((len(df), N_PIVOT_FEATURES), np.nan)

    # Extract labels
    label_arrays = np.zeros((len(df), len(TF_LABEL_NAMES)), dtype=np.float32)
    for i, lbl in enumerate(TF_LABEL_NAMES):
        if lbl in df.columns:
            label_arrays[:, i] = df[lbl].fillna(0).values

    return features, pivots, label_arrays


def _load_pos_state(path: str) -> Optional[np.ndarray]:
    """
    Load Position State Vector columns (Phase 3) from the M5 CSV.

    Returns:
        np.ndarray[float32, (N, 11)] if all 11 ps_* columns are present, else None.
    """
    try:
        ps_set = set(POS_STATE_NAMES)
        df = pd.read_csv(path, usecols=lambda c: c in ps_set, low_memory=False)
    except Exception as exc:
        print(f"  [DATA] PosState load failed ({exc}) — Phase 3 disabled")
        return None
    present = [c for c in POS_STATE_NAMES if c in df.columns]
    if len(present) < N_POS_STATE:
        return None
    return df[POS_STATE_NAMES].fillna(0.0).values.astype(np.float32)


def _load_options_chain(path: str, max_contracts: int = 120) -> dict:
    """
    Load options chain CSV and build per-timestamp snapshots.

    The raw CSV columns (21):
        timestamp, contract_id, symbol, expiration, strike, type,
        last, mark, bid, bid_size, ask, ask_size, volume,
        open_interest, implied_volatility, delta, gamma, theta,
        vega, rho, in_the_money

    Returns dict mapping timestamp_index → np.ndarray[N, 10] feature grid.
    """
    print(f"  [DATA] Loading options chain from {path}...")

    # Chunked read for large files — low_memory=False prevents mixed-dtype warnings
    df = pd.read_csv(path, parse_dates=['timestamp'], low_memory=False)
    print(f"    -> {len(df)} contracts across "
          f"{df['timestamp'].nunique()} timestamps")

    # Coerce numeric columns that may have loaded as object due to mixed types
    numeric_cols = [
        'strike', 'last', 'mark', 'bid', 'bid_size', 'ask', 'ask_size',
        'volume', 'open_interest', 'implied_volatility',
        'delta', 'gamma', 'theta', 'vega', 'rho',
    ]
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Build chain grid features per timestamp
    # Schema: moneyness, days_to_exp, iv, delta, gamma, theta, vega, bid, ask, oi_norm
    snapshots = {}
    grouped = df.groupby('timestamp')

    for ts, group in grouped:
        n = min(len(group), max_contracts)
        grid = np.zeros((n, len(CHAIN_FEATURE_NAMES)), dtype=np.float32)

        # Select top contracts by open_interest (now guaranteed numeric)
        if len(group) > max_contracts:
            group = group.nlargest(max_contracts, 'open_interest')

        g = group.head(n)

        # Moneyness = log(spot / strike) — spot ≈ mark for ITM proxy
        spot_approx = g['mark'].median() if 'mark' in g.columns else g['last'].median()
        if spot_approx > 0:
            grid[:, 0] = np.log(spot_approx / g['strike'].values.clip(1) + EPS)  # moneyness

        # Days to expiry
        if 'expiration' in g.columns:
            try:
                exp_dates = pd.to_datetime(g['expiration'], format='mixed')
                grid[:, 1] = (exp_dates - ts).dt.days.values.clip(0)
            except Exception:
                grid[:, 1] = 0

        # IV, delta, gamma, theta, vega
        grid[:, 2] = g['implied_volatility'].fillna(0).values
        grid[:, 3] = g['delta'].fillna(0).values
        grid[:, 4] = g['gamma'].fillna(0).values
        grid[:, 5] = g['theta'].fillna(0).values
        grid[:, 6] = g['vega'].fillna(0).values

        # Bid, ask
        grid[:, 7] = g['bid'].fillna(0).values
        grid[:, 8] = g['ask'].fillna(0).values

        # OI normalized (log-scaled within snapshot)
        oi = g['open_interest'].fillna(0).values.astype(np.float32)
        oi_log = np.log1p(oi)
        oi_max = oi_log.max() if oi_log.max() > 0 else 1.0
        grid[:, 9] = oi_log / oi_max

        date_str = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]
        snapshots[date_str] = grid

    print(f"    -> Built {len(snapshots)} chain snapshots")
    return snapshots


def build_dataloaders_v43(args) -> Tuple:
    """
    Load all 4 TF CSVs + options chain, normalize, split, and create DataLoaders.

    Returns:
        (train_loader, val_loader, normalization_stats, n_train_batches, n_val_batches)
    """
    data_dir = Path(args.data_dir)

    # Load 4 TF datasets
    m1_feat, m1_piv, m1_lbl = _load_tf_csv(str(data_dir / args.m1_file), "m1")
    m5_feat, m5_piv, m5_lbl = _load_tf_csv(str(data_dir / args.m5_file), "m5")
    m15_feat, m15_piv, _ = _load_tf_csv(str(data_dir / args.m15_file), "m15")
    h1_feat, h1_piv, _ = _load_tf_csv(str(data_dir / args.h1_file), "h1")

    # ── Canonical alignment: M5 is the master clock (schema_v43 contract) ──
    # Each TF CSV is at its native resolution (M1=1min, M5=5min, M15=15min, H1=1hr).
    # Align all to M5 length by forward-filling coarser TFs and subsampling M1.
    # Previous approach (min_len = shortest = H1 = 1,661) discarded 91% of data.
    canon_len = len(m5_feat)  # 18,494 — the authoritative length

    def _upsample_to_canon(feat: np.ndarray, target: int) -> np.ndarray:
        """Forward-fill a coarser-TF array to M5 canonical length."""
        ratio = max(round(target / len(feat)), 1)
        out = np.repeat(feat, ratio, axis=0)
        if len(out) < target:                       # pad last row if short
            out = np.vstack([out, np.tile(out[-1:], (target - len(out), 1))])
        return out[:target]

    def _downsample_to_canon(feat: np.ndarray, target: int) -> np.ndarray:
        """Subsample a finer-TF array to M5 canonical length (last bar of window)."""
        ratio = max(len(feat) // target, 1)
        idx = np.arange(ratio - 1, ratio * target, ratio)[:target]
        return feat[idx]

    print(f"  [DATA] Native lengths: m1={len(m1_feat)}, m5={len(m5_feat)}, "
          f"m15={len(m15_feat)}, h1={len(h1_feat)}")

    # M1 (finer) → subsample to M5 resolution (last M1 bar of each 5-bar window)
    m1_feat = _downsample_to_canon(m1_feat, canon_len)
    # M15, H1 (coarser) → forward-fill to M5 resolution
    m15_feat = _upsample_to_canon(m15_feat, canon_len)
    h1_feat  = _upsample_to_canon(h1_feat,  canon_len)

    print("  [DATA] Extracting alignment dates from M5...")
    m5_df = pd.read_csv(str(data_dir / args.m5_file), usecols=['timestamp'])
    m5_dates = m5_df['timestamp'].str.slice(0, 10).values

    # M5 and its pivots/labels are already canonical length
    m5_piv = m5_piv[:canon_len]
    m5_lbl = m5_lbl[:canon_len]
    m5_dates = m5_dates[:canon_len]

    min_len = canon_len  # keep variable name for downstream splits
    print(f"  [DATA] Aligned length: {min_len} (canonical M5 — all TFs resampled)")

    # Normalize features (fit on train split only = first 70%)
    split_train = int(min_len * 0.70)
    split_val = int(min_len * 0.85)

    # Fit normalization on train portion
    m1_med, m1_scl = robust_zscore_fit(safe_nan_to_num(m1_feat[:split_train]))
    m5_med, m5_scl = robust_zscore_fit(safe_nan_to_num(m5_feat[:split_train]))
    m15_med, m15_scl = robust_zscore_fit(safe_nan_to_num(m15_feat[:split_train]))
    h1_med, h1_scl = robust_zscore_fit(safe_nan_to_num(h1_feat[:split_train]))

    # Transform all
    m1_feat = robust_zscore_transform(safe_nan_to_num(m1_feat), m1_med, m1_scl)
    m5_feat = robust_zscore_transform(safe_nan_to_num(m5_feat), m5_med, m5_scl)
    m15_feat = robust_zscore_transform(safe_nan_to_num(m15_feat), m15_med, m15_scl)
    h1_feat = robust_zscore_transform(safe_nan_to_num(h1_feat), h1_med, h1_scl)

    # Build labels dict
    # TF_LABEL_NAMES col layout (set by data_pipeline_v43.py):
    #   0: target_spot   1: position_size_mult  2: strategy_label
    #   3: pop           4: ev                  5: max_loss
    #   6: var_95        7: cvar_95
    def _col(arr: np.ndarray, idx: int) -> Optional[np.ndarray]:
        return arr[:min_len, idx] if arr.shape[1] > idx else None

    labels = {
        'target_spot':    m5_lbl[:min_len, 0],
        'position_size':  m5_lbl[:min_len, 1],
        'strategy_label': (
            m5_lbl[:min_len, 2] if m5_lbl.shape[1] > 2
            else np.full(min_len, ABSTAIN_IDX, dtype=np.float32)
        ),
    }
    # Risk labels: only added when data_pipeline_v43.py has been run
    for col_idx, name in ((3, 'pop'), (4, 'ev'), (5, 'max_loss'),
                          (6, 'var_95'), (7, 'cvar_95'), (8, 'exit_signal')):
        v = _col(m5_lbl, col_idx)
        if v is not None:
            labels[name] = v

    _have = [k for k in ('pop', 'ev', 'max_loss', 'var_95', 'cvar_95') if k in labels]
    if _have:
        print(f"  [DATA] Risk labels loaded from CSV: {_have}")
    else:
        print("  [DATA] Risk labels absent — run data_pipeline_v43.py to generate them")

    # Load options chain
    chain_path = data_dir / args.options_file
    chain_snapshots = {}
    if chain_path.exists():
        chain_snapshots = _load_options_chain(str(chain_path), CHAIN_GRID_CONFIG['max_contracts'])

    # Load Position State Vector (Phase 3 — 11 ps_* columns from M5 CSV)
    m5_pos_state_full = _load_pos_state(str(data_dir / args.m5_file))
    if m5_pos_state_full is not None:
        m5_pos_state_full = m5_pos_state_full[:canon_len]
        print(f"  [DATA] Position state (Phase 3): {m5_pos_state_full.shape} loaded ✓")
    else:
        print("  [DATA] Position state absent — re-run data_pipeline_v43.py --force to generate Phase 3 columns")

    # Split: train=70%, val=15%, test=15% (temporal, not shuffled)
    ps_train = m5_pos_state_full[:split_train]          if m5_pos_state_full is not None else None
    ps_val   = m5_pos_state_full[split_train:split_val] if m5_pos_state_full is not None else None

    train_ds = V43Dataset(
        m1_feat[:split_train], m5_feat[:split_train],
        m15_feat[:split_train], h1_feat[:split_train],
        m5_piv[:split_train],
        {k: v[:split_train] for k, v in labels.items()},
        chain_snapshots,
        m5_dates[:split_train],
        seq_len=args.lookback,
        m5_pos_state=ps_train,
    )
    val_ds = V43Dataset(
        m1_feat[split_train:split_val], m5_feat[split_train:split_val],
        m15_feat[split_train:split_val], h1_feat[split_train:split_val],
        m5_piv[split_train:split_val],
        {k: v[split_train:split_val] for k, v in labels.items()},
        chain_snapshots,
        m5_dates[split_train:split_val],
        seq_len=args.lookback,
        m5_pos_state=ps_val,
    )

    print(f"  [DATA] Train: {len(train_ds)} sequences | Val: {len(val_ds)} sequences")

    # DataLoaders (GPU-optimized)
    workers = min(args.num_workers, 4)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=v43_collate_fn,
        num_workers=workers, pin_memory=True,
        persistent_workers=(workers > 0), prefetch_factor=2 if workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=v43_collate_fn,
        num_workers=workers, pin_memory=True,
        persistent_workers=(workers > 0), prefetch_factor=2 if workers > 0 else None,
    )

    norm_stats = {
        'm1': (m1_med, m1_scl), 'm5': (m5_med, m5_scl),
        'm15': (m15_med, m15_scl), 'h1': (h1_med, h1_scl),
    }

    n_train_batches = len(train_loader)
    n_val_batches = max(1, len(val_loader))

    return train_loader, val_loader, norm_stats, n_train_batches, n_val_batches


# =============================================================================
# SMART EPOCH MANAGER (convergence detection)
# =============================================================================

class SmartEpochManager:
    """Intra-epoch convergence detection — mirrors v42 implementation."""

    def __init__(self, threshold: float = 1e-5, patience: int = 50, active: bool = True):
        self.threshold = threshold
        self.patience = patience
        self.active = active
        self.best_loss_in_window = float('inf')
        self.best_snapshot = None
        self.has_snapshot = False
        self.counter = 0

    def reset(self):
        self.best_loss_in_window = float('inf')
        self.best_snapshot = None
        self.has_snapshot = False
        self.counter = 0

    def update(self, loss: float, model: nn.Module) -> bool:
        if not self.active:
            return False
        if loss < self.best_loss_in_window - self.threshold:
            self.best_loss_in_window = loss
            self.best_snapshot = copy.deepcopy(model.state_dict())
            self.has_snapshot = True
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# =============================================================================
# EPOCH ANALYTICS — A/B matrix, interpretability JSON, epoch comparison
# =============================================================================

def _build_pair_lookup(n_inputs: int, feature_names: List[str]) -> Dict[int, Tuple[str, str]]:
    """Map pair_idx (upper-triangle ordering) → (feat_a, feat_b) name strings.

    RelationalLogicLayer pairs features by torch.triu_indices(n_inputs, n_inputs, offset=1).
    Indices beyond len(feature_names) fall back to 'feat_N' generics.
    """
    iu = np.triu_indices(n_inputs, k=1)
    n = len(iu[0])
    return {
        k: (
            feature_names[int(iu[0][k])] if int(iu[0][k]) < len(feature_names) else f"feat_{iu[0][k]}",
            feature_names[int(iu[1][k])] if int(iu[1][k]) < len(feature_names) else f"feat_{iu[1][k]}",
        )
        for k in range(n)
    }


def _extract_logic_v43(model: nn.Module) -> dict:
    """
    Extract learned logic from CondorNetV43.
    Covers: v42 backbone predicates + supersets, v43 strategy/risk/pivot heads,
    TF projector Frobenius contributions.
    """
    logic: dict = {
        "predicates": {},
        "super_set": {},
        "strategy_head": {},
        "risk_head": {},
        "pivot_head": {},
        "fuzzy_gates": {},
    }

    # ── v42 backbone — CondorNetV43 stores it as model.condor_core ──
    backbone = getattr(model, 'condor_core', getattr(model, 'backbone', model))

    # Predicate gates
    if hasattr(backbone, 'pred_gates'):
        pg = backbone.pred_gates
        preds: dict = {}
        for attr in ['iv_rank_thresh', 'spread_frac_thresh', 'rsi_thresh',
                     'gap_frac_thresh', 'gamma_thresh', 'iv_regime_frac_thresh',
                     'put_flow_thresh', 'spread_stress_mult_thresh']:
            if hasattr(pg, attr):
                t = getattr(pg, attr)
                preds[attr.replace('_thresh', '_threshold')] = float(
                    t.data if isinstance(t, torch.Tensor) else t
                )
        if hasattr(pg, 'steepness'):
            s = pg.steepness
            preds['steepness'] = float(s.data if isinstance(s, torch.Tensor) else s)
        logic["predicates"] = preds

    # SuperSets (relational logic)
    if hasattr(backbone, 'super_sets') and len(backbone.super_sets) > 0:
        # Build pair lookup once from the first RelationalLogicLayer's n_inputs.
        # Uses _FULL_FEATURE_NAMES (64 entries: 52 base TF + 12 ETL-computed) so
        # that friction_ok_*, tod_sin/cos, regime_persistence, and the four IVR
        # reversal features resolve to readable names instead of feat_N.
        pair_lookup: Dict[int, Tuple[str, str]] = {}
        first_ss = backbone.super_sets[0]
        if hasattr(first_ss, 'sets') and len(first_ss.sets) > 0:
            first_rl = getattr(first_ss.sets[0], 'relational_logic', None)
            if first_rl is not None:
                n_rl_inputs = getattr(first_rl, 'n_inputs', len(_FULL_FEATURE_NAMES))
                pair_lookup = _build_pair_lookup(n_rl_inputs, _FULL_FEATURE_NAMES)

        ss_list = []
        for ss_idx, ss in enumerate(backbone.super_sets):
            ss_info = {
                "index": ss_idx,
                "n_sets": len(ss.sets) if hasattr(ss, 'sets') else 0,
                "sets": [],
            }
            if hasattr(ss, 'sets'):
                for i, pset in enumerate(ss.sets):
                    set_info: dict = {"index": i}
                    if hasattr(pset, 'relational_logic'):
                        rl = pset.relational_logic
                        if hasattr(rl, 'steepness'):
                            st = rl.steepness
                            set_info["steepness"] = float(
                                st.data if isinstance(st, torch.Tensor) else st
                            )
                        w = rl.projection.weight.detach().cpu().numpy()
                        n_pairs = getattr(rl, 'n_pairs', w.shape[1] // 3)
                        if w.shape[1] == n_pairs * 3:
                            w_r = w.reshape(w.shape[0], n_pairs, 3)
                            op_w = np.abs(w_r).sum(axis=0)
                            total = float(op_w.sum()) + 1e-8
                            set_info["operator_weights"] = {
                                "<": round(float(op_w[:, 0].sum()) / total * 100, 1),
                                ">": round(float(op_w[:, 1].sum()) / total * 100, 1),
                                "=": round(float(op_w[:, 2].sum()) / total * 100, 1),
                            }
                            imp = op_w.sum(axis=1)
                            top5 = np.argsort(imp)[-5:][::-1]
                            set_info["top_pair_indices"] = [int(p) for p in top5]
                            # Human-readable comparisons (restored from v4.2)
                            comparisons = []
                            for p_idx in top5:
                                ops_row = op_w[int(p_idx)]
                                dom_op = ["<", ">", "="][int(np.argmax(ops_row))]
                                feat_a, feat_b = pair_lookup.get(
                                    int(p_idx), (f"feat_{p_idx}a", f"feat_{p_idx}b")
                                )
                                comparisons.append({
                                    "comparison": f"{feat_a} {dom_op} {feat_b}",
                                    "dominant_op": dom_op,
                                    "weights": {
                                        "<": round(float(ops_row[0]), 4),
                                        ">": round(float(ops_row[1]), 4),
                                        "=": round(float(ops_row[2]), 4),
                                    },
                                })
                            set_info["top_comparisons"] = comparisons
                        else:
                            set_info["weight_norm"] = float(
                                rl.projection.weight.norm().item()
                            )
                    ss_info["sets"].append(set_info)
            ss_list.append(ss_info)
        logic["super_set"] = {"n_super_sets": len(ss_list), "super_sets": ss_list}

    # ── v43 StrategyHead ──
    if hasattr(model, 'strategy_head'):
        sh = model.strategy_head
        head_seq = getattr(sh, 'strategy_head', None)
        if head_seq is not None:
            last_linear = None
            for layer in reversed(list(head_seq.children())):
                if hasattr(layer, 'weight'):
                    last_linear = layer
                    break
            if last_linear is not None:
                w = last_linear.weight.detach().cpu().numpy()
                norms = np.linalg.norm(w, axis=1)
                logic["strategy_head"] = {
                    "output_class_norms": {
                        STRATEGY_TYPES[i]: float(norms[i])
                        for i in range(min(len(norms), N_STRATEGY_TYPES))
                    }
                }

    # ── v43 RiskMetricHead ──
    if hasattr(model, 'risk_head'):
        rh = model.risk_head
        shared = getattr(rh, 'shared', None)
        if shared is not None:
            first_linear = next(
                (m for m in shared.modules() if hasattr(m, 'weight')), None
            )
            if first_linear is not None:
                logic["risk_head"] = {
                    "shared_input_norm": float(
                        np.linalg.norm(first_linear.weight.detach().cpu().numpy())
                    )
                }

    # ── v43 PivotPredictionHead ──
    horizons = [5, 10, 20, 35, 70]
    if hasattr(model, 'pivot_pred_head'):
        ph = model.pivot_pred_head
        for attr_name, key in [('pivot_high_head', 'high_horizon_weights'),
                                ('pivot_low_head', 'low_horizon_weights')]:
            head = getattr(ph, attr_name, None)
            if head is not None:
                last = next(
                    (m for m in reversed(list(head.modules())) if hasattr(m, 'weight')),
                    None,
                )
                if last is not None:
                    w = last.weight.detach().cpu().numpy()
                    norms = np.linalg.norm(w, axis=1)
                    logic["pivot_head"][key] = {
                        f"h{h}": float(norms[i])
                        for i, h in enumerate(horizons[:len(norms)])
                    }

    # ── v43 TF Projector Frobenius norms ──
    if hasattr(model, 'tf_projector'):
        tfp = model.tf_projector
        contrib: dict = {}
        for tf_name in ['m1', 'm5', 'm15', 'h1']:
            # MultiTFProjector stores projectors as proj_m1, proj_m5, proj_m15, proj_h1
            proj = getattr(tfp, f'proj_{tf_name}', None)
            if proj is not None:
                first = next(
                    (m for m in proj.modules() if hasattr(m, 'weight')), None
                )
                if first is not None:
                    contrib[tf_name] = float(
                        np.linalg.norm(first.weight.detach().cpu().numpy(), 'fro')
                    )
        if contrib:
            logic["fuzzy_gates"]["tf_projector_frobenius"] = contrib
            logic["fuzzy_gates"]["tf_ranked"] = sorted(
                contrib, key=contrib.get, reverse=True
            )

    return logic


def export_matrices_csv(model: nn.Module, output_dir: Path, epoch: int) -> dict:
    """Export A_matrix (and B_matrix if available) per epoch. Returns stability metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    # A_matrix
    try:
        with torch.no_grad():
            A = model.get_A_matrix().cpu().float().numpy()
        a_path = output_dir / f"Epoch{epoch}_A_Matrix.csv"
        np.savetxt(str(a_path), A, delimiter=',', fmt='%.8f')
        eigvals = np.linalg.eigvals(A)
        mags = np.abs(eigvals)
        result["a_matrix"] = {
            "shape": list(A.shape),
            "spectral_radius": float(np.max(mags)),
            "top_5_eigenvalue_magnitudes": [float(x) for x in sorted(mags, reverse=True)[:5]],
            "frobenius_norm": float(np.linalg.norm(A, 'fro')),
        }
        print(f"  -> A_matrix: {a_path.name} "
              f"(rho={result['a_matrix']['spectral_radius']:.4f})")
    except Exception as e:
        result["a_matrix"] = {"error": str(e)}
        print(f"  -> A_matrix export failed: {e}")

    # B_matrix — BlockMatrixB has no .weight; use full_matrix() which
    # concatenates B_h, B_v, B_m, B_r weight rows into [d_x, d_control].
    backbone = getattr(model, 'condor_core', getattr(model, 'backbone', model))
    try:
        b_theta = getattr(backbone, 'B_theta', None)
        if b_theta is not None:
            with torch.no_grad():
                if hasattr(b_theta, 'full_matrix'):
                    B = b_theta.full_matrix().cpu().float().numpy()
                elif hasattr(b_theta, 'weight'):
                    B = b_theta.weight.cpu().float().numpy()
                else:
                    B = None
            if B is not None:
                b_path = output_dir / f"Epoch{epoch}_B_Matrix.csv"
                np.savetxt(str(b_path), B, delimiter=',', fmt='%.8f')
                result["b_matrix"] = {
                    "shape": list(B.shape),
                    "frobenius_norm": float(np.linalg.norm(B, 'fro')),
                    "column_norms_max": float(np.linalg.norm(B, axis=0).max()),
                    "column_norms_mean": float(np.linalg.norm(B, axis=0).mean()),
                }
                print(f"  -> B_matrix: {b_path.name} "
                      f"shape={B.shape} "
                      f"(frob={result['b_matrix']['frobenius_norm']:.4f})")
    except Exception as e:
        result["b_matrix"] = {"error": str(e)}
        print(f"  -> B_matrix export failed: {e}")

    return result


def extract_epoch_interpretability_v43(
    model: nn.Module,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args,
    is_best: bool = False,
) -> dict:
    """Per-epoch interpretability report: logic + A/B matrix stability."""
    n_params = sum(p.numel() for p in model.parameters())
    report: dict = {
        "summary": {
            "epoch": epoch,
            "version": f"{CN_VERSION}_{DS_VERSION}",
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_train_gap": float(val_loss - train_loss),
            "is_best": is_best,
            "parameters": n_params,
            "architecture": {
                "d_joint": args.d_joint,
                "d_chain": args.d_chain,
                "d_pivot": args.d_pivot,
                "n_strategy_types": args.n_strategy_types,
            },
        },
    }

    model.eval()
    with torch.no_grad():
        logic = _extract_logic_v43(model)
    model.train()
    report["learned_logic"] = logic

    # Matrix stability (exports CSV side-effect)
    mat_dir = Path("models/A_matrices")
    mat_metrics = export_matrices_csv(model, mat_dir, epoch)
    report.update(mat_metrics)

    return report


def _compare_predicates(best_preds: dict, epoch_preds: dict,
                        tol: float = 0.01) -> Tuple[int, int]:
    same, diff = 0, 0
    all_keys = set(list(best_preds.keys()) + list(epoch_preds.keys()))
    threshold_keys = [k for k in all_keys if 'threshold' in k or 'thresh' in k]
    for key in threshold_keys:
        b_v, e_v = best_preds.get(key), epoch_preds.get(key)
        if isinstance(b_v, (int, float)) and isinstance(e_v, (int, float)):
            if abs(b_v - e_v) <= tol:
                same += 1
            else:
                diff += 1
    return same, diff


def _compare_structure(best_ss: dict, epoch_ss: dict,
                       level: str = 'sets') -> Tuple[int, int]:
    best_list = best_ss.get('super_sets', [])
    epoch_list = epoch_ss.get('super_sets', [])
    if level == 'super_sets':
        # Compare internal sets instead of just length
        same, diff = 0, 0
        for b_ss, e_ss in zip(best_list, epoch_list):
            for b_set, e_set in zip(b_ss.get('sets', []), e_ss.get('sets', [])):
                b_ops = b_set.get('operator_weights', {})
                e_ops = e_set.get('operator_weights', {})
                if b_ops and e_ops:
                    if max(b_ops, key=b_ops.get) == max(e_ops, key=e_ops.get):
                        same += 1
                    else:
                        diff += 1
                else:
                    b_n = b_set.get('weight_norm', 0)
                    e_n = e_set.get('weight_norm', 0)
                    if b_n > 0 and abs(b_n - e_n) / (b_n + 1e-8) < 0.01:
                        same += 1
                    else:
                        diff += 1
        # Add remaining unmatched as different
        diff += abs(len(best_list) - len(epoch_list))
        return same, diff
    same, diff = 0, 0
    for b_ss, e_ss in zip(best_list, epoch_list):
        for b_set, e_set in zip(b_ss.get('sets', []), e_ss.get('sets', [])):
            b_ops = b_set.get('operator_weights', {})
            e_ops = e_set.get('operator_weights', {})
            if b_ops and e_ops:
                if max(b_ops, key=b_ops.get) == max(e_ops, key=e_ops.get):
                    same += 1
                else:
                    diff += 1
            else:
                b_n = b_set.get('weight_norm', 0)
                e_n = e_set.get('weight_norm', 0)
                if b_n > 0 and abs(b_n - e_n) / (b_n + 1e-8) < 0.01:
                    same += 1
                else:
                    diff += 1
    return same, diff


def _compare_strategy_heads(best_logic: dict, epoch_logic: dict) -> dict:
    b_sh = best_logic.get('strategy_head', {}).get('output_class_norms', {})
    e_sh = epoch_logic.get('strategy_head', {}).get('output_class_norms', {})
    changed = [
        {"class": cls, "delta": round(abs(b_sh[cls] - e_sh.get(cls, 0)), 4)}
        for cls in b_sh
        if abs(b_sh[cls] - e_sh.get(cls, 0)) > 0.05
    ]
    return {"changed_classes": changed, "n_changed": len(changed)}


def _compare_pivot_heads(best_logic: dict, epoch_logic: dict) -> dict:
    b_ph = best_logic.get('pivot_head', {})
    e_ph = epoch_logic.get('pivot_head', {})
    changes: dict = {}
    for key in ['high_horizon_weights', 'low_horizon_weights']:
        b_w = b_ph.get(key, {})
        e_w = e_ph.get(key, {})
        for h in b_w:
            delta = abs(b_w[h] - e_w.get(h, 0))
            if delta > 0.02:
                changes[f"{key}_{h}"] = round(delta, 4)
    return changes


def compute_epoch_comparison(epoch_snapshots: list, best_epoch: int) -> dict:
    """Diff each epoch's learned logic against the best model epoch."""
    comparison: dict = {"best_model": {"epoch": best_epoch}, "comparisons": {}}

    best_snap = next(
        (s for s in epoch_snapshots if _parse_epoch_id(s['epoch']) == best_epoch), None
    )
    if best_snap is None:
        best_snap = epoch_snapshots[0]
        comparison["best_model"]["note"] = "No best epoch yet — using first as baseline"

    comparison["best_model"]["val_loss"] = best_snap.get('val_loss', 0)
    comparison["best_model"]["train_loss"] = best_snap.get('train_loss', 0)
    best_logic = best_snap['interpretability'].get('learned_logic', {})

    for snap in epoch_snapshots:
        epoch_num = _parse_epoch_id(snap['epoch'])   # normalize "Epoch30" → 30
        ek = f"Epoch{epoch_num}"
        if epoch_num == best_epoch:
            comparison["comparisons"][ek] = "BEST MODEL — omitted from diff"
            continue

        epoch_logic = snap['interpretability'].get('learned_logic', {})
        pred_same, pred_diff = _compare_predicates(
            best_logic.get('predicates', {}), epoch_logic.get('predicates', {})
        )
        set_same, set_diff = _compare_structure(
            best_logic.get('super_set', {}), epoch_logic.get('super_set', {}), 'sets'
        )
        ss_same, ss_diff = _compare_structure(
            best_logic.get('super_set', {}), epoch_logic.get('super_set', {}), 'super_sets'
        )
        strat_diff = _compare_strategy_heads(best_logic, epoch_logic)
        pivot_diff = _compare_pivot_heads(best_logic, epoch_logic)

        b_rho = best_snap['interpretability'].get('a_matrix', {}).get('spectral_radius', 0)
        e_rho = snap['interpretability'].get('a_matrix', {}).get('spectral_radius', 0)

        comparison["comparisons"][ek] = {
            "same_predicates": pred_same, "different_predicates": pred_diff,
            "same_sets": set_same, "different_sets": set_diff,
            "same_super_sets": ss_same, "different_super_sets": ss_diff,
            "strategy_head_diff": strat_diff,
            "pivot_head_diff": pivot_diff,
            "spectral_radius_delta": round(float(e_rho) - float(b_rho), 6),
            "val_loss": snap.get('val_loss', 0),
            "train_loss": snap.get('train_loss', 0),
        }

    return comparison


# =============================================================================
# ARGUMENT PARSER (F7.1)
# =============================================================================

def parse_args_v43():
    """
    Parse training arguments — includes all v42 flags + new v43 flags.
    """
    p = argparse.ArgumentParser(description=f"CondorNet v4.3 Training ({CN_VERSION})")

    # === Data ===
    p.add_argument('--data-dir', type=str, default='data/Datasetv4/v43/',
                   help='Base directory for v43 dataset files')
    p.add_argument('--m1-file', type=str, default='m1_dataset_v43_final.csv')
    p.add_argument('--m5-file', type=str, default='m5_dataset_v43_final.csv')
    p.add_argument('--m15-file', type=str, default='m15_dataset_v43_final.csv')
    p.add_argument('--h1-file', type=str, default='h1_dataset_v43_final.csv')
    p.add_argument('--options-file', type=str, default='options_2025_v43.csv')
    p.add_argument('--labels-file', type=str, default='',
                   help='Optional precomputed labels CSV (strategy + risk)')
    p.add_argument('--loss-config', type=str, default='configs/loss_weights_v43.json')

    # === Architecture ===
    p.add_argument('--d-tf-in', type=int, default=len(TF_FEATURE_NAMES),
                   help='TF input dimension (auto from schema)')
    p.add_argument('--d-joint', type=int, default=256)
    p.add_argument('--d-chain', type=int, default=CHAIN_GRID_CONFIG['d_chain'])
    p.add_argument('--d-pivot', type=int, default=16)
    p.add_argument('--d-fused', type=int, default=256)
    p.add_argument('--n-strategy-types', type=int, default=N_STRATEGY_TYPES)
    p.add_argument('--d-h', type=int, default=256)
    p.add_argument('--d-v', type=int, default=32)
    p.add_argument('--d-m', type=int, default=64)
    p.add_argument('--d-r', type=int, default=32)
    p.add_argument('--d-control', type=int, default=128)
    p.add_argument('--n-predicates', type=int, default=8)
    p.add_argument('--n-sets', type=int, default=4)
    p.add_argument('--n-super-sets', type=int, default=1)
    p.add_argument('--gate-temp', type=float, default=3.0,
                   help='Initial gate temperature τ (Run 5+). Amplifies routing logits '
                        'so λ std targets 0.15–0.30.  Learnable; clamped to [1, 10].')
    p.add_argument('--diversity-alpha', type=float, default=0.005,
                   help='λ-std diversity hinge coefficient (demoted in Run 8 to sanity rail). '
                        'Ramps 0 → α over first 30%% of epochs.')
    p.add_argument('--diversity-std-target', type=float, default=0.14,
                   help='Target minimum std(λ) for diversity hinge (default 0.14).')
    # Run 8: logit-space regularizers
    p.add_argument('--logit-var-alpha', type=float, default=0.02,
                   help='Run 8: logit variance floor coefficient. '
                        'Ramps 0 → α over first 10%% of epochs.')
    p.add_argument('--logit-var-target', type=float, default=2.0,
                   help='Run 8: target minimum Var(z_approx). '
                        '2.0 ≈ z_std 1.4 (observed at healthy epoch 1, Run 7).')
    p.add_argument('--logit-iqr-alpha', type=float, default=0.02,
                   help='Run 8: IQR floor coefficient. '
                        'Ramps 0 → α over first 10%% of epochs.')
    p.add_argument('--logit-iqr-target', type=float, default=1.0,
                   help='Run 8: target minimum IQR(z_approx). '
                        '1.0 forces middle 50%% to spread ≥1 logit unit.')
    p.add_argument('--logit-eps', type=float, default=1e-4,
                   help='Run 8: ε for logit clamping in _logit_safe (default 1e-4).')
    p.add_argument('--logit-warmup-frac', type=float, default=0.0,
                   help='Run 9: fraction of epochs for logit-space α warmup ramp. '
                        '0.0 = full α from epoch 1 (no warmup). Default: 0.0.')
    p.add_argument('--logit-mad-alpha', type=float, default=0.05,
                   help='Run 9: MAD floor coefficient. Forces mean(|z|) ≥ mad_target. '
                        'Attacks symmetric shrinkage toward 0 that var+IQR can miss. '
                        'Default: 0.05.')
    p.add_argument('--logit-mad-target', type=float, default=0.9,
                   help='Run 9: target minimum mean(|z_approx|). '
                        '0.9 ≈ N(0, σ=1.1) — healthy gate magnitude. Default: 0.9.')
    p.add_argument('--logit-tail-alpha', type=float, default=0.0,
                   help='Tail penalty coefficient. Penalizes squared excess of |z| beyond '
                        '--logit-tail-band. Targets runaway hard-off/on outliers that '
                        'var/IQR/MAD floors miss. 0.0 = disabled. Recommended: 0.01.')
    p.add_argument('--logit-tail-band', type=float, default=8.0,
                   help='Safe |z| radius for tail penalty. Excess beyond this is penalized. '
                        'σ(±8)≈0.9997 — gates are fully saturated at this magnitude. '
                        'Default: 8.0.')

    # === Training ===
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lookback', type=int, default=64, help='Sequence length T')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--accum-steps', type=int, default=4,
                   help='Gradient accumulation steps (F7.12)')
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    p.add_argument('--min-delta', type=float, default=0.0,
                   help='Minimum improvement required to reset patience counter. '
                        '0.0 = any improvement (default, compatible with old runs). '
                        '0.001 = recommended to ignore micro-gains and fire patience at '
                        'true convergence (~epoch 40-46 on typical runs).')
    p.add_argument('--grad-clip', type=float, default=1.0)

    # === Convergence ===
    p.add_argument('--convergence-threshold', type=float, default=1e-5)
    p.add_argument('--convergence-patience', type=int, default=50)
    p.add_argument('--convergence-active', action='store_true', default=False,
                   help='Enable intra-epoch early stopping (default: OFF). '
                        'When disabled, patience operates at epoch boundaries only.')

    # === Checkpointing / Output ===
    p.add_argument('--output', type=str, default='models/condornet_v43_best.pth')
    p.add_argument('--checkpoint-dir', type=str, default='models/ckpts/')
    p.add_argument('--keep-ckpts', type=int, default=3)
    p.add_argument('--report-dir', type=str, default='reports',
                   help='Directory for epoch JSON reports and EPOCH_Comparison files. '
                        'Created if it does not exist. Default: reports/')

    # === Resume & crash recovery (F7.8, F7.9) ===
    p.add_argument('--resume', type=str, default='',
                   help='Path to checkpoint to resume from')

    # === DataLoader ===
    p.add_argument('--num-workers', type=int, default=4)

    # === Debugging ===
    p.add_argument('--verbose', action='store_true', default=False)
    p.add_argument('--dry-run', action='store_true', default=False,
                   help='Run 1 epoch with 5 batches for testing')

    return p.parse_args()


# =============================================================================
# MAIN TRAINING FUNCTION (F7.6)
# =============================================================================

def train_condor_net_v43(args):
    """
    Full v4.3 training loop implementing the 7-step hook order contract:
        1. FORWARD   → model(x_m1, x_m5, x_m15, x_h1, chain, chain_mask, pivot, pivot_mask)
        2. LOSS      → criterion(outputs, labels, gates)
        3. BACKWARD  → loss.backward() (with gradient accumulation)
        4. OPTIMIZER → optimizer.step()
        5. OBSERVE   → log diagnostics
        6. CHECKPOINT → atomic_save
        7. TELEMETRY  → emit to GUI/TensorBoard
    """
    # ── Device setup ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"\n{'='*60}")
    print(f"CONDORNET v4.3 TRAINING LABORATORY")
    print(f"Version: {CN_VERSION}_{DS_VERSION} | Schema: {SCHEMA_VERSION}")
    print(f"{'='*60}")
    print(f"Device: {device} | AMP dtype: {amp_dtype}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Crash sentinel (F7.9) ───────────────────────────────────────────
    if check_sentinel():
        print(f"  ⚠  CRASH SENTINEL detected from previous run — clearing and starting fresh.")
        remove_sentinel()   # auto-clear; use --resume only to restore checkpoint state
    write_sentinel()

    # ── GUI telemetry ───────────────────────────────────────────────────
    emitter = get_emitter() if GUI_TELEMETRY_AVAILABLE else None

    # ── Data loading ────────────────────────────────────────────────────
    print(f"\n[DATA] Loading v4.3 multi-TF datasets...")
    train_loader, val_loader, norm_stats, n_train_batches, n_val_batches = \
        build_dataloaders_v43(args)

    print(f"[DATA] Train batches: {n_train_batches} | Val batches: {n_val_batches}")

    # ── Model ───────────────────────────────────────────────────────────
    # Keys must exactly match CondorNetV43.__init__ parameter names.
    # d_joint / d_fused / chain_d_ff are not CondorNetV43 params — omitted.
    config = {
        'd_tf_in':           args.d_tf_in,
        'd_chain':           args.d_chain,
        'd_pivot_proj':      args.d_pivot,           # was 'd_pivot'
        'd_pivot_in':        N_PIVOT_FEATURES,        # was 'n_pivot_features'
        'n_strategy_types':  args.n_strategy_types,
        'chain_in_features': CHAIN_GRID_CONFIG['in_features'],
        'd_model_chain':     CHAIN_GRID_CONFIG['d_model'],   # was 'chain_d_model'
        'n_heads_chain':     CHAIN_GRID_CONFIG['n_heads'],   # was 'chain_n_heads'
        'n_layers_chain':    CHAIN_GRID_CONFIG['n_layers'],  # was 'chain_n_layers'
        'd_h':               args.d_h,
        'd_v':               args.d_v,
        'd_m':               args.d_m,
        'd_r':               args.d_r,
        'd_control':         args.d_control,
        'n_predicates':      args.n_predicates,
        'n_sets':            args.n_sets,
        'n_super_sets':      args.n_super_sets,
        'gate_temp_init':    args.gate_temp,
    }
    model = build_condornet_v43(**config)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] CondorNetV43 — {n_params:,} parameters")
    print(f"[MODEL] Weights dtype: {next(model.parameters()).dtype} (AMP Stable Mode)")

    # ── Loss function (F7.5, F7.10) ─────────────────────────────────────
    loss_config = load_loss_weights(args.loss_config)
    criterion = CondorLossV43(loss_config)

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if not use_bf16 else None

    # ── τ trainability verification (Run 7) ──────────────────────────────
    print("\n--- Tau Parameter Verification ---")
    _tau_found = False
    for _tname, _tp in model.named_parameters():
        if "tau" in _tname.lower() or "gate_temperature" in _tname.lower():
            print(f"  Found τ param : {_tname} | val={_tp.item():.4f} "
                  f"| requires_grad={_tp.requires_grad}")
            _tau_found = True
    if not _tau_found:
        print("  WARNING: τ/gate_temperature NOT found in model.named_parameters() — "
              "τ is not learnable")

    # Verify τ is inside the optimizer param groups
    _tau_ids = {
        id(p) for n, p in model.named_parameters()
        if "tau" in n.lower() or "gate_temperature" in n.lower()
    }
    _opt_ids = {id(p) for pg in optimizer.param_groups for p in pg['params']}
    if _tau_ids and not _tau_ids.intersection(_opt_ids):
        print("  WARNING: τ NOT present in optimizer param groups — gradient will NOT apply!")
    elif _tau_ids:
        print("  τ is in optimizer param groups — τ IS trainable")
    print("----------------------------------\n")

    # ── Resume from checkpoint (F7.8) ───────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_train_gap = float('inf')
    best_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"\n[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Use strict=False to allow loading checkpoints from older model versions
        # (e.g., before consensus_gate / chain_stats_proj were added).
        # New parameters initialize from their __init__ defaults (consensus_gate
        # zero-init → neutral gate; chain_stats_proj zero-init → no chain stat effect).
        missing, unexpected = [], []
        try:
            incompatible = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            missing    = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
        except RuntimeError as e:
            print(f"  [RESUME] WARNING: load_state_dict failed ({e}). Starting fresh.")
        if missing:
            print(f"  [RESUME] New parameters (zero/default init): {missing[:10]}"
                  f"{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"  [RESUME] Stale keys ignored: {unexpected[:10]}"
                  f"{'...' if len(unexpected) > 10 else ''}")
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"[RESUME] Continuing from epoch {start_epoch + 1}, "
              f"best_val_loss={best_val_loss:.4f}")

    # ── Training state ──────────────────────────────────────────────────
    total_steps = args.epochs * n_train_batches
    patience_counter = 0
    global_step = start_epoch * n_train_batches
    training_start_time = time.time()
    epoch_snapshots: list = []      # accumulate per-epoch interpretability snapshots

    # Global worst-logit tracker across ALL epochs (Run 7)
    worst_logit_value = 0.0         # tracks most negative gate logit seen
    worst_logit_epoch = None
    worst_logit_batch = None

    convergence_manager = SmartEpochManager(
        threshold=args.convergence_threshold,
        patience=args.convergence_patience,
        active=args.convergence_active,
    )

    # ── TensorBoard ─────────────────────────────────────────────────────
    run_name = f"condorv43_lr{args.lr}_bs{args.batch_size}_{datetime.now():%m%d%H%M}"
    writer = SummaryWriter(log_dir=f"runs/condor_v43/{run_name}")

    print(f"\n{'='*60}")
    print(f"Training Config:")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  Lookback: {args.lookback} | Accum Steps: {args.accum_steps}")
    print(f"  Grad Clip: {args.grad_clip} | Patience: {args.patience}")
    print(f"  Gate τ init: {args.gate_temp} | λ-hinge α: {args.diversity_alpha} (std_target: {args.diversity_std_target})")
    _lwf_e = max(1, int(args.epochs * args.logit_warmup_frac)) if args.logit_warmup_frac > 0 else 0
    print(f"  Z-var floor: α={args.logit_var_alpha} var_target={args.logit_var_target} | "
          f"IQR floor: α={args.logit_iqr_alpha} iqr_target={args.logit_iqr_target} | "
          f"MAD floor: α={args.logit_mad_alpha} mad_target={args.logit_mad_target} "
          f"(warmup: {_lwf_e} epochs{' [immediate full-α]' if args.logit_warmup_frac == 0.0 else ''})")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    # ── Emit training start ─────────────────────────────────────────────
    if emitter:
        emitter.emit_status(
            is_training=True, current_epoch=start_epoch + 1,
            current_step=0, total_steps=total_steps,
            progress_pct=0.0, eta_seconds=None,
        )

    # ================================================================
    # EPOCH LOOP
    # ================================================================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.set_epoch(epoch)

        # α ramp: 0 → diversity_alpha_max over first 30% of epochs (Run 7, demoted)
        alpha_sched = compute_alpha(
            epoch, args.epochs,
            alpha_max=args.diversity_alpha,
            warmup_frac=0.3,
        )
        # α ramp: logit-space regularizers (Run 8/9, warmup_frac from CLI)
        alpha_z_sched = compute_alpha(
            epoch, args.epochs,
            alpha_max=args.logit_var_alpha,
            warmup_frac=args.logit_warmup_frac,
        )
        alpha_iqr_sched = compute_alpha(
            epoch, args.epochs,
            alpha_max=args.logit_iqr_alpha,
            warmup_frac=args.logit_warmup_frac,
        )
        alpha_mad_sched = compute_alpha(
            epoch, args.epochs,
            alpha_max=args.logit_mad_alpha,
            warmup_frac=args.logit_warmup_frac,
        )

        epoch_loss = 0.0
        epoch_components = {}
        epoch_batches_run = 0
        convergence_manager.reset()

        # Buffers for hierarchical gate diagnostics (accumulated per-batch, logged per-epoch)
        _gate_logit_buf: list = []
        _super_gate_buf: list = []
        # Per-epoch worst-batch tracker (reported in outlier warning)
        _worst_batch_idx = -1
        _worst_batch_abs = 0.0
        # Tier 1: per-epoch worst z-space batch (min IQR / min var across all batches)
        _worst_batch_ziqr = float('inf')
        _worst_batch_ziqr_idx = -1
        _worst_batch_zvar = float('inf')
        _worst_batch_zvar_idx = -1

        max_batches = 5 if args.dry_run else n_train_batches
        pbar = tqdm(enumerate(train_loader), total=max_batches,
                    desc=f"Epoch {epoch+1}", leave=False)

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in pbar:
            if args.dry_run and batch_idx >= 5:
                break

            # Move to device
            x_m1 = batch['x_m1'].to(device)
            x_m5 = batch['x_m5'].to(device)
            x_m15 = batch['x_m15'].to(device)
            x_h1 = batch['x_h1'].to(device)
            chain = batch['chain'].to(device)
            chain_mask = batch['chain_mask'].to(device)
            piv_mask = batch['pivot_mask'].to(device)
            piv_vals = batch['pivot_values'].to(device)
            pos_state_batch = batch['pos_state'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}

            # ── 1. FORWARD + LOSS (inside autocast) ─────────────────
            # Criterion must share the same dtype context as the forward
            # pass — running it outside autocast creates a Float/Half
            # gradient mismatch on backward. GradScaler provides the
            # precision benefit; keeping both inside is the standard pattern.
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                     enabled=(device.type == 'cuda')):
                outputs = model(
                    x_m1=x_m1, x_m5=x_m5, x_m15=x_m15, x_h1=x_h1,
                    chain=chain, chain_mask=chain_mask,
                    pivot_features=piv_vals, pivot_mask=piv_mask,
                    pos_state=pos_state_batch,
                )
                # ── 2. LOSS ─────────────────────────────────────────
                loss, components = criterion(outputs, labels)

                # ── λ-std diversity hinge (Run 7, demoted to sanity rail) ─
                if outputs.final_gate is not None and alpha_sched > 0.0:
                    _div_loss = gate_diversity_loss(
                        outputs.final_gate,
                        std_target=args.diversity_std_target,
                        alpha=alpha_sched,
                    )
                    loss = loss + _div_loss
                    components['gate_diversity'] = _div_loss.detach()

            # ── Gate logit-space regularizers (Run 8/9) ──────────────────────
            # Intentionally OUTSIDE autocast: _logit_safe() forces .float() so
            # all hinge math is fp32.  This eliminates any risk of fp16 underflow
            # on small hinge gradients even when amp_dtype=torch.float16.
            _zvar_loss = _ziqr_loss = _zmad_loss = None
            if outputs.final_gate is not None:
                if alpha_z_sched > 0.0:
                    _zvar_loss = logit_variance_floor(
                        outputs.final_gate,
                        var_target=args.logit_var_target,
                        alpha=alpha_z_sched,
                        logit_eps=args.logit_eps,
                    )
                    loss = loss + _zvar_loss
                    components['logit_zvar'] = _zvar_loss.detach()

                if alpha_iqr_sched > 0.0:
                    _ziqr_loss = logit_iqr_floor(
                        outputs.final_gate,
                        iqr_target=args.logit_iqr_target,
                        alpha=alpha_iqr_sched,
                        logit_eps=args.logit_eps,
                    )
                    loss = loss + _ziqr_loss
                    components['logit_ziqr'] = _ziqr_loss.detach()

                if alpha_mad_sched > 0.0:
                    _zmad_loss = logit_mad_floor(
                        outputs.final_gate,
                        mad_target=args.logit_mad_target,
                        alpha=alpha_mad_sched,
                        logit_eps=args.logit_eps,
                    )
                    loss = loss + _zmad_loss
                    components['logit_zmad'] = _zmad_loss.detach()

                # ── Logit tail penalty (targets runaway |z| outliers) ─────────
                if args.logit_tail_alpha > 0.0:
                    _ztail_loss = logit_tail_penalty(
                        outputs.final_gate,
                        clip_band=args.logit_tail_band,
                        alpha=args.logit_tail_alpha,
                        logit_eps=args.logit_eps,
                    )
                    loss = loss + _ztail_loss
                    components['logit_ztail'] = _ztail_loss.detach()

            # Accumulate hierarchical gate stats.
            # _last_gate_logit is set on the v42 backbone (model.condor_core), not the
            # top-level CondorNetV43 wrapper — use the same lookup pattern as _extract_logic_v43.
            _backbone = getattr(model, 'condor_core', model)
            if hasattr(_backbone, '_last_gate_logit') and _backbone._last_gate_logit is not None:
                _bl = _backbone._last_gate_logit.float().cpu()
                _gate_logit_buf.append(_bl)
                _super_gate_buf.append(_backbone._last_super_gate.float().cpu())
                # Per-epoch worst-batch tracking
                _bl_abs = max(abs(_bl.min().item()), _bl.max().item())
                if _bl_abs > _worst_batch_abs:
                    _worst_batch_abs = _bl_abs
                    _worst_batch_idx = batch_idx
                # Global worst-logit tracker (Run 7)
                _bl_min = _bl.min().item()
                if _bl_min < worst_logit_value:
                    worst_logit_value = _bl_min
                    worst_logit_epoch = epoch + 1
                    worst_logit_batch = batch_idx
                # Tier 1: per-epoch z-space worst-batch tracking (gate_logit ≈ logit(λ))
                _bl_flat = _bl.view(-1)
                if _bl_flat.numel() >= 4:
                    _bz_iqr = (_bl_flat.quantile(0.75) - _bl_flat.quantile(0.25)).item()
                    _bz_var = _bl_flat.var(unbiased=False).item()
                    if _bz_iqr < _worst_batch_ziqr:
                        _worst_batch_ziqr = _bz_iqr
                        _worst_batch_ziqr_idx = batch_idx
                    if _bz_var < _worst_batch_zvar:
                        _worst_batch_zvar = _bz_var
                        _worst_batch_zvar_idx = batch_idx
                # Multi-batch gate trace: B00, B01, B05, B10 — α-verification
                if batch_idx in {0, 1, 5, 10}:
                    _sg_f   = _super_gate_buf[-1].view(-1)
                    _z_std_b = _bl_flat.std().item()
                    _z_iqr_b = (_bl_flat.quantile(0.75) - _bl_flat.quantile(0.25)).item() \
                               if _bl_flat.numel() >= 4 else 0.0
                    _z_mad_b = _bl_flat.abs().mean().item()
                    _lmb_s   = _sg_f.std().item()
                    _zvar_s  = f"  zvar={_zvar_loss.item():.5f}" if _zvar_loss is not None else ""
                    _ziqr_s  = f"  ziqr={_ziqr_loss.item():.5f}" if _ziqr_loss is not None else ""
                    _zmad_s  = f"  zmad={_zmad_loss.item():.5f}" if _zmad_loss is not None else ""
                    # tail penalty: pct of |z| > band and mean excess
                    if args.logit_tail_alpha > 0.0 and _bl_flat.numel() > 0:
                        _tail_mask = _bl_flat.abs() > args.logit_tail_band
                        _pct_tail = 100.0 * _tail_mask.float().mean().item()
                        _mean_excess = (_bl_flat.abs() - args.logit_tail_band).clamp(min=0).mean().item()
                        _ztail_s = f"  tail>{args.logit_tail_band:.0f}: {_pct_tail:.2f}% excess={_mean_excess:.3f}"
                    else:
                        _ztail_s = ""
                    print(f"  [Gate B{batch_idx:02d}] λ_std={_lmb_s:.4f}  "
                          f"z_std={_z_std_b:.4f}  z_iqr={_z_iqr_b:.4f}  z_mad={_z_mad_b:.4f}  "
                          f"α_z={alpha_z_sched:.4f}  α_iqr={alpha_iqr_sched:.4f}  α_mad={alpha_mad_sched:.4f}"
                          f"{_zvar_s}{_ziqr_s}{_zmad_s}{_ztail_s}")

            # Gradient accumulation scaling (F7.12)
            scaled_loss = loss / args.accum_steps

            # Debug — first batch of first epoch
            if batch_idx == 0 and epoch == start_epoch:
                print(f"\n[DEBUG BATCH 0] System Inventory:")
                print(f"  Input shapes: m1={x_m1.shape}, chain={chain.shape}")
                print(f"  Model weights: {next(model.parameters()).dtype}")
                print(f"  Strategy logits: {outputs.strategy_logits.shape}")
                print(f"  Loss: {loss.item():.6f} ({loss.dtype})")
                for k, v in components.items():
                    val = v.item() if v.numel() == 1 else v.mean().item()
                    print(f"    - {k}: {val:.6f}")

            # ── 3. BACKWARD ─────────────────────────────────────────
            if scaler is not None:
                try:
                    scaler.scale(scaled_loss).backward()
                except RuntimeError as e:
                    print(f"\n!!! BACKWARD CRASHED at batch {batch_idx}: {e}")
                    raise
            else:
                scaled_loss.backward()

            # ── 4. OPTIMIZER (every accum_steps) ────────────────────
            if (batch_idx + 1) % args.accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Track epoch stats
            epoch_loss += loss.item()
            epoch_batches_run += 1
            for k, v in components.items():
                prev = epoch_components.get(k, 0.0)
                epoch_components[k] = prev + (v.item() if torch.is_tensor(v) else v)

            # ── 5. OBSERVE ──────────────────────────────────────────
            should_stop = convergence_manager.update(loss.item(), model)
            if should_stop:
                print(f"\n  [CONVERGENCE] Epoch {epoch+1} stopped at batch "
                      f"{batch_idx+1}/{n_train_batches} "
                      f"(best={convergence_manager.best_loss_in_window:.6f})")
                break

            # ── 7. TELEMETRY ────────────────────────────────────────
            if global_step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                for k, v in components.items():
                    val = v.item() if torch.is_tensor(v) else v
                    writer.add_scalar(f'train/{k}', val, global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if args.verbose and (batch_idx + 1) % 10 == 0:
                comp_str = ' | '.join(
                    [f"{k}:{(v.item() if torch.is_tensor(v) else v):.3f}"
                     for k, v in list(components.items())[:5]]
                )
                print(f"  [B{batch_idx+1:04d}] loss={loss.item():.4f} | {comp_str}")

        # ── Scheduler step ──────────────────────────────────────────────
        scheduler.step()
        avg_train_loss = epoch_loss / max(1, epoch_batches_run)

        # Restore convergence snapshot for validation
        if convergence_manager.has_snapshot:
            print(f"  [V43] Restoring convergence snapshot "
                  f"(loss={convergence_manager.best_loss_in_window:.6f})")
            model.load_state_dict(convergence_manager.best_snapshot)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches_run = 0
        vbar = tqdm(enumerate(val_loader), total=n_val_batches,
                    desc="Validation", leave=False)
        with torch.no_grad():
            for vb_idx, batch in vbar:
                x_m1 = batch['x_m1'].to(device)
                x_m5 = batch['x_m5'].to(device)
                x_m15 = batch['x_m15'].to(device)
                x_h1 = batch['x_h1'].to(device)
                chain = batch['chain'].to(device)
                chain_mask = batch['chain_mask'].to(device)
                piv_mask = batch['pivot_mask'].to(device)
                piv_vals = batch['pivot_values'].to(device)
                pos_state_batch = batch['pos_state'].to(device)
                labels = {k: v.to(device) for k, v in batch['labels'].items()}

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                         enabled=(device.type == 'cuda')):
                    outputs = model(
                        x_m1=x_m1, x_m5=x_m5, x_m15=x_m15, x_h1=x_h1,
                        chain=chain, chain_mask=chain_mask,
                        pivot_features=piv_vals, pivot_mask=piv_mask,
                        pos_state=pos_state_batch,
                    )
                    loss, _ = criterion(outputs, labels)

                val_loss += loss.item()
                val_batches_run += 1
                vbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / max(1, val_batches_run)

        # TensorBoard: epoch-level
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
        for k, v in epoch_components.items():
            writer.add_scalar(f'epoch/{k}', v / max(1, epoch_batches_run), epoch)

        # Print epoch-level weights for annealed components
        annealed_info = []
        for name in criterion.schedules:
            w = criterion._get_weight(name)
            annealed_info.append(f"{name}={w:.3f}")
        anneal_str = f" | Annealed: {', '.join(annealed_info)}" if annealed_info else ""

        val_train_gap = avg_val_loss - avg_train_loss  # signed: +ve = val > train (normal)
        print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | gap: {val_train_gap:+.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}"
              f"{anneal_str}")
        _comp_avg_str = "  ".join(
            f"{k}={v / max(1, epoch_batches_run):.4f}"
            for k, v in epoch_components.items()
        )
        print(f"  [EPOCH COMPONENTS] {_comp_avg_str}")

        # ── Hierarchical Gate Diagnostics ────────────────────────────────────
        if not _gate_logit_buf:
            print(f"  [WARNING] Gate stats missing epoch {epoch+1} — "
                  f"check _last_gate_logit attribute path (expected on model.condor_core)")
        if _gate_logit_buf:
            all_logits = torch.cat(_gate_logit_buf)          # (N_batches, 1)
            all_lambda = torch.cat(_super_gate_buf)
            lm  = all_lambda.mean().item()
            ls  = all_lambda.std().item() if all_lambda.numel() > 1 else 0.0
            lp05 = all_lambda.quantile(0.05).item()
            lp50 = all_lambda.quantile(0.50).item()
            lp95 = all_lambda.quantile(0.95).item()
            _gl_std_ep = all_logits.std().item()
            # Read current τ from backbone (set in forward as _last_gate_temperature)
            _backbone = getattr(model, 'condor_core', model)
            tau_val = getattr(_backbone, '_last_gate_temperature', float('nan'))
            lambda_ok = 0.15 <= ls <= 0.30
            lambda_status = "OK" if lambda_ok else ("LOW" if ls < 0.15 else "HIGH")
            print(f"  Gate logit : mean={all_logits.mean():.3f}  std={all_logits.std():.3f}"
                  f"  min={all_logits.min():.3f}  max={all_logits.max():.3f}  tau={tau_val:.3f}")
            print(f"  Lambda     : p05={lp05:.3f}  p50={lp50:.3f}  p95={lp95:.3f}"
                  f"  mu={lm:.3f}  std={ls:.4f}  [target: 0.15-0.30 → {lambda_status}]")
            writer.add_scalar('epoch/gate_logit_std', _gl_std_ep, epoch)
            writer.add_scalar('epoch/lambda_mean',    lm,  epoch)
            writer.add_scalar('epoch/lambda_std',     ls,  epoch)
            writer.add_scalar('epoch/gate_temperature', tau_val, epoch)
            # ── Gate health checks (three independent failure modes) ──────────
            # 1. Always-on saturation: gate stuck at 1 → model ignores gate path
            if lm > 0.95 and ls < 0.01:
                _sat_n = getattr(model, '_sat_epoch_count', 0) + 1
                model._sat_epoch_count = _sat_n
                print(f"  [WARNING] GATE SATURATED (always-on) epoch {epoch+1}: "
                      f"lambda_mu={lm:.4f} > 0.95  lambda_std={ls:.4f} < 0.01 "
                      f"(consecutive: {_sat_n})")
            else:
                model._sat_epoch_count = 0
            # 2. Always-off collapse: gate stuck at 0 → z_gated ≈ 0 every step
            if lm < 0.05 and ls < 0.01:
                print(f"  [WARNING] GATE DEAD (always-off) epoch {epoch+1}: "
                      f"lambda_mu={lm:.4f} < 0.05  lambda_std={ls:.4f} < 0.01")
            # 3. Dead zone: logit range too wide → sigmoid saturated on both tails,
            #    gradients vanish even if mean λ looks healthy.
            if _gl_std_ep > 6.0:
                print(f"  [WARNING] GATE DEAD ZONE epoch {epoch+1}: "
                      f"gate_logit_std={_gl_std_ep:.3f} > 6.0 "
                      f"(sigmoid saturated on both tails — gradient dying)")
            # 4. Outlier spike: logit min or max is extreme relative to std,
            #    indicating a small number of samples dominating gate variance.
            #    Threshold: |min| or max > 5 × std (5-sigma outlier).
            _gl_min = all_logits.min().item()
            _gl_max = all_logits.max().item()
            _outlier_thresh = 5.0 * max(_gl_std_ep, 1e-6)
            if abs(_gl_min) > _outlier_thresh or _gl_max > _outlier_thresh:
                _gl_iqr = (all_logits.quantile(0.75) - all_logits.quantile(0.25)).item()
                _worst_b_str = (f"  worst_batch={_worst_batch_idx}"
                                if _worst_batch_idx >= 0 else "")
                print(f"  [WARNING] GATE LOGIT OUTLIER epoch {epoch+1}: "
                      f"min={_gl_min:.3f}  max={_gl_max:.3f}  std={_gl_std_ep:.3f}  "
                      f"IQR={_gl_iqr:.3f}{_worst_b_str}  (extreme logits inflating std — "
                      f"lambda_std={ls:.4f} may understate routing diversity)")
            writer.add_scalar('epoch/diversity_alpha', alpha_sched, epoch)
            writer.add_scalar('epoch/alpha_z',   alpha_z_sched,   epoch)
            writer.add_scalar('epoch/alpha_iqr',  alpha_iqr_sched, epoch)

            # ── Z-space epoch diagnostics (Run 8/9) ──────────────────
            with torch.no_grad():
                _z_all = _logit_safe(all_lambda, eps=args.logit_eps).view(-1)
                _z_std_ep  = _z_all.std().item()
                _z_iqr_ep  = (_z_all.quantile(0.75) - _z_all.quantile(0.25)).item()
                _z_var_ep  = _z_all.var(unbiased=False).item()
                _z_mad_ep  = _z_all.abs().mean().item()
            _z_std_ok  = 1.0 <= _z_std_ep <= 3.0  # upper bound 3.0: floor is at var=2.0, no ceiling concern
            _z_iqr_ok  = _z_iqr_ep >= 0.8
            _z_mad_ok  = _z_mad_ep >= 0.7
            _z_std_tag = "OK" if _z_std_ok else ("LOW" if _z_std_ep < 1.0 else "HIGH")
            _z_iqr_tag = "OK" if _z_iqr_ok else "LOW"
            _z_mad_tag = "OK" if _z_mad_ok else "LOW"
            print(f"  Z-space   : std={_z_std_ep:.4f} [{_z_std_tag}]  "
                  f"var={_z_var_ep:.4f}  IQR={_z_iqr_ep:.4f} [{_z_iqr_tag}]  "
                  f"MAD={_z_mad_ep:.4f} [{_z_mad_tag}]  "
                  f"[targets: std 1.0-3.0, IQR ≥ 0.8, MAD ≥ 0.7]")
            writer.add_scalar('epoch/z_std',  _z_std_ep,  epoch)
            writer.add_scalar('epoch/z_var',  _z_var_ep,  epoch)
            writer.add_scalar('epoch/z_iqr',  _z_iqr_ep,  epoch)
            writer.add_scalar('epoch/z_mad',  _z_mad_ep,  epoch)
            writer.add_scalar('epoch/alpha_mad', alpha_mad_sched, epoch)
            # Tier 1: worst-batch z-space report
            if _worst_batch_ziqr < float('inf'):
                print(f"  Z worst-batch: z_IQR_min={_worst_batch_ziqr:.4f} "
                      f"(batch {_worst_batch_ziqr_idx})  "
                      f"z_var_min={_worst_batch_zvar:.4f} "
                      f"(batch {_worst_batch_zvar_idx})")
            # Tier 1: clamp hit-rate (fraction of λ hitting the ε boundary)
            with torch.no_grad():
                _lf = all_lambda.view(-1)
                _n_lf = max(_lf.numel(), 1)
                _n_low  = (_lf <= args.logit_eps).sum().item()
                _n_high = (_lf >= 1.0 - args.logit_eps).sum().item()
            print(f"  Clamp hits  : λ≤ε {100.*_n_low/_n_lf:.2f}%  "
                  f"λ≥1-ε {100.*_n_high/_n_lf:.2f}%  (ε={args.logit_eps})")
            # Tail penalty epoch summary (only when penalty is active)
            if args.logit_tail_alpha > 0.0 and _gate_logit_buf:
                _all_z_ep = torch.cat([b.view(-1) for b in _gate_logit_buf])
                _tail_mask_ep = _all_z_ep.abs() > args.logit_tail_band
                _pct_tail_ep  = 100.0 * _tail_mask_ep.float().mean().item()
                _mean_exc_ep  = (_all_z_ep.abs() - args.logit_tail_band).clamp(min=0).mean().item()
                _max_abs_ep   = _all_z_ep.abs().max().item()
                print(f"  Tail penalty: |z|>{args.logit_tail_band:.0f} → {_pct_tail_ep:.3f}%  "
                      f"mean_excess={_mean_exc_ep:.4f}  max|z|={_max_abs_ep:.2f}  "
                      f"α_tail={args.logit_tail_alpha}")

        # ── 6. CHECKPOINT ───────────────────────────────────────────────
        epoch_ts = datetime.now().strftime("%m%d%Y_%H%M%S")
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        epoch_ckpt_name = f"Epoch{epoch+1}_{CN_VERSION}_{DS_VERSION}_{epoch_ts}.pth"

        ckpt_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'version': f"{CN_VERSION}_{DS_VERSION}",
            'schema_version': SCHEMA_VERSION,
            'config': config,
            'normalization': {
                k: (m.tolist(), s.tolist()) for k, (m, s) in norm_stats.items()
            },
        }
        if atomic_save(ckpt_data, checkpoint_dir / epoch_ckpt_name):
            print(f"  -> Checkpoint saved: {epoch_ckpt_name}")
            rotate_checkpoints(checkpoint_dir, keep_n=args.keep_ckpts)

        # ── Best model: lowest val loss always wins ──────────────────────
        # Run 5: removed the `val < train` gate — that heuristic blocked
        # saving genuinely better checkpoints (Run 4 epoch 7 val=1.8488
        # was rejected; only epoch 2 val=2.099 was saved).
        # Correct criterion: strictly lower validation loss.
        # Note: val > train is normal once dropout/augmentation are active;
        # it does NOT indicate overfitting. Patience counter is the real guard.
        # val_train_gap already computed above before epoch summary print.

        is_new_best = False
        if best_val_loss == float('inf'):
            is_new_best = True
            reason = "First epoch"
        elif avg_val_loss < best_val_loss - args.min_delta:
            is_new_best = True
            reason = f"Better val ({avg_val_loss:.4f} < {best_val_loss:.4f} - {args.min_delta})"
        else:
            reason = f"Val not better ({avg_val_loss:.4f} >= {best_val_loss:.4f} - {args.min_delta}), best remains Epoch {best_epoch}"

        if is_new_best:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            best_val_train_gap = val_train_gap
            best_epoch = epoch + 1
            patience_counter = 0

            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            best_data = dict(ckpt_data)
            best_data['best_val_loss'] = best_val_loss
            if atomic_save(best_data, args.output):
                print(f"  -> NEW BEST MODEL (Epoch {epoch+1}): {reason}")
                print(f"     Saved: {args.output}")
        else:
            patience_counter += 1
            print(f"  -> Epoch {epoch+1}: {reason}")

        # ── EPOCH ANALYTICS (interpretability + A/B matrices + comparison) ──
        interp_report = extract_epoch_interpretability_v43(
            model, epoch + 1, avg_train_loss, avg_val_loss, args,
            is_best=(epoch + 1 == best_epoch),
        )
        # Update is_best flag on the interp report now that best_epoch is set
        interp_report["summary"]["is_best"] = (epoch + 1 == best_epoch)

        reports_dir = Path(args.report_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        interp_path = reports_dir / f"Epoch{epoch+1}_{CN_VERSION}_{DS_VERSION}_{epoch_ts}.json"
        with open(interp_path, 'w') as f:
            json.dump(interp_report, f, indent=2)
        print(f"  -> Interpretability: {interp_path.name}")

        # Accumulate snapshot for epoch comparison
        epoch_snapshots.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "timestamp": epoch_ts,
            "interpretability": interp_report,
        })

        # Epoch-vs-epoch comparison (after >= 2 epochs)
        if len(epoch_snapshots) >= 2:
            comparison = compute_epoch_comparison(epoch_snapshots, best_epoch)
            # Rotate: delete old comparison file, write fresh one
            for old_f in glob.glob(str(reports_dir / "EPOCH_Comparison_*.json")):
                try:
                    os.remove(old_f)
                except OSError:
                    pass
            comp_path = reports_dir / f"EPOCH_Comparison_{epoch_ts}.json"
            with open(comp_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"  -> Comparison: {comp_path.name}")

        # TensorBoard: A_matrix spectral radius
        a_rho = interp_report.get("a_matrix", {}).get("spectral_radius", None)
        if a_rho is not None:
            writer.add_scalar('epoch/spectral_radius', a_rho, epoch)
        b_frob = interp_report.get("b_matrix", {}).get("frobenius_norm", None)
        if b_frob is not None:
            writer.add_scalar('epoch/b_matrix_frobenius', b_frob, epoch)

        # GUI Telemetry: epoch summary
        if emitter:
            emitter.emit_epoch_summary(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                metrics={k: v / max(1, epoch_batches_run)
                         for k, v in epoch_components.items()},
                is_best=(epoch + 1 == best_epoch),
            )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # ================================================================
    # TRAINING COMPLETE
    # ================================================================
    print(f"\n{'='*60}")
    print(f"CONDORNET v4.3 TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Best train loss: {best_train_loss:.4f}")
    print(f"Best gap: {best_val_train_gap:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Reports: reports/")
    print(f"A/B Matrices: models/A_matrices/")
    print(f"{'='*60}")

    # Global worst gate logit summary (Run 7)
    print(f"\n--- Worst Gate Logit (across full training) ---")
    print(f"  Min logit: {worst_logit_value:.6f}")
    if worst_logit_epoch is not None:
        print(f"  Occurred:  epoch {worst_logit_epoch}, batch {worst_logit_batch}")
    else:
        print("  (no gate logits recorded — gate_temperature may not be active)")
    print(f"------------------------------------------------")

    # Clean exit
    writer.close()
    remove_sentinel()

    if emitter:
        total_duration = int(time.time() - training_start_time)
        emitter.emit_complete(
            epochs=epoch + 1, final_loss=avg_train_loss,
            best_val_loss=best_val_loss, duration_seconds=total_duration,
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    args = parse_args_v43()
    try:
        train_condor_net_v43(args)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user")
        remove_sentinel()
    except Exception as e:
        print(f"\n[FATAL] Training crashed: {e}")
        # Sentinel intentionally NOT removed on crash → detected on next run
        raise
