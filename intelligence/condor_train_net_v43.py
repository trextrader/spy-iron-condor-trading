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
import argparse
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
        x_m1:    [T, d_tf]    M1 features
        x_m5:    [T, d_tf]    M5 features (canonical TF)
        x_m15:   [T, d_tf]    M15 features
        x_h1:    [T, d_tf]    H1 features
        chain:   [N, 10]      Options chain grid (variable N per sample)
        chain_mask: [N]       Bool — True=padded
        pivot_m5: [T, 13]     Pivot mask for M5 (True=NaN)
        targets:  dict        Training targets (strategy_label, pop, ev, etc.)
    """

    def __init__(
        self,
        m1_features: np.ndarray,    # [N_total, d_tf]
        m5_features: np.ndarray,    # [N_total, d_tf]
        m15_features: np.ndarray,   # [N_total, d_tf]
        h1_features: np.ndarray,    # [N_total, d_tf]
        m5_pivots: np.ndarray,      # [N_total, 13] — NaN-bearing pivot features
        labels: dict,               # {name: np.ndarray[N_total, ...]}
        chain_snapshots: dict,      # {idx: np.ndarray[N_contracts, 10]}
        seq_len: int = 64,
    ):
        self.m1 = m1_features
        self.m5 = m5_features
        self.m15 = m15_features
        self.h1 = h1_features
        self.pivots = m5_pivots
        self.labels = labels
        self.chain_snapshots = chain_snapshots
        self.seq_len = seq_len
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
        if target_idx in self.chain_snapshots:
            chain = torch.tensor(self.chain_snapshots[target_idx], dtype=torch.float32)
            chain_mask = torch.zeros(chain.shape[0], dtype=torch.bool)
        else:
            # No chain data — single dummy contract, fully masked
            chain = torch.zeros(1, len(CHAIN_FEATURE_NAMES), dtype=torch.float32)
            chain_mask = torch.ones(1, dtype=torch.bool)

        # Labels at target timestep
        target_labels = {}
        for key, arr in self.labels.items():
            target_labels[key] = torch.tensor(arr[target_idx], dtype=torch.float32)

        return {
            'x_m1': x_m1,
            'x_m5': x_m5,
            'x_m15': x_m15,
            'x_h1': x_h1,
            'chain': chain,
            'chain_mask': chain_mask,
            'pivot_mask': pivot_mask,
            'pivot_values': pivot_values,
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
    9-component composite loss for CondorNet v4.3.

    Components:
        1. strategy_ce:  Cross-entropy over 10 strategy classes
        2. pop_bce:      Binary cross-entropy for PoP calibration
        3. ev_mse:       MSE for Expected Value prediction
        4. risk_mse:     MSE for VaR + CVaR + max_loss combined
        5. size_mse:     MSE for fuzzy position size output
        6. spot_mse:     MSE for target spot price (auxiliary)
        7. fuzzy_var:    Variance penalty on gate outputs
        8. pattern_ent:  Entropy regularizer on predicate activations
        9. robust:       Huber loss for outlier-robust EV/risk

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
        if 'pop' in labels:
            pop_target = labels['pop'].unsqueeze(-1) if labels['pop'].dim() == 1 else labels['pop']
            components['pop_bce'] = F.binary_cross_entropy(
                outputs.pop.clamp(1e-7, 1 - 1e-7), pop_target
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

    # Chunked read for large files
    df = pd.read_csv(path, parse_dates=['timestamp'])
    print(f"    -> {len(df)} contracts across "
          f"{df['timestamp'].nunique()} timestamps")

    # Build chain grid features per timestamp
    # Schema: moneyness, days_to_exp, iv, delta, gamma, theta, vega, bid, ask, oi_norm
    snapshots = {}
    grouped = df.groupby('timestamp')
    ts_sorted = sorted(grouped.groups.keys())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_sorted)}

    for ts, group in grouped:
        n = min(len(group), max_contracts)
        grid = np.zeros((n, len(CHAIN_FEATURE_NAMES)), dtype=np.float32)

        # Select top contracts by open_interest
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
                exp_dates = pd.to_datetime(g['expiration'])
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

        idx = ts_to_idx.get(ts, None)
        if idx is not None:
            snapshots[idx] = grid

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

    # Use the shortest TF length as the alignment point
    min_len = min(len(m1_feat), len(m5_feat), len(m15_feat), len(h1_feat))
    print(f"  [DATA] Aligned length: {min_len} (m1={len(m1_feat)}, m5={len(m5_feat)}, "
          f"m15={len(m15_feat)}, h1={len(h1_feat)})")

    m1_feat = m1_feat[:min_len]
    m5_feat = m5_feat[:min_len]
    m15_feat = m15_feat[:min_len]
    h1_feat = h1_feat[:min_len]
    m5_piv = m5_piv[:min_len]
    m5_lbl = m5_lbl[:min_len]

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
    labels = {
        'target_spot': m5_lbl[:, 0],
        'position_size': m5_lbl[:, 1],
    }
    # Add strategy label (defaults to abstain if not available yet)
    labels['strategy_label'] = np.full(min_len, ABSTAIN_IDX, dtype=np.float32)

    # Load options chain
    chain_path = data_dir / args.options_file
    chain_snapshots = {}
    if chain_path.exists():
        chain_snapshots = _load_options_chain(str(chain_path), CHAIN_GRID_CONFIG['max_contracts'])

    # Split: train=70%, val=15%, test=15% (temporal, not shuffled)
    train_ds = V43Dataset(
        m1_feat[:split_train], m5_feat[:split_train],
        m15_feat[:split_train], h1_feat[:split_train],
        m5_piv[:split_train],
        {k: v[:split_train] for k, v in labels.items()},
        {k: v for k, v in chain_snapshots.items() if k < split_train},
        seq_len=args.lookback,
    )
    val_ds = V43Dataset(
        m1_feat[split_train:split_val], m5_feat[split_train:split_val],
        m15_feat[split_train:split_val], h1_feat[split_train:split_val],
        m5_piv[split_train:split_val],
        {k: v[split_train:split_val] for k, v in labels.items()},
        {k - split_train: v for k, v in chain_snapshots.items()
         if split_train <= k < split_val},
        seq_len=args.lookback,
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

    # === Training ===
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lookback', type=int, default=64, help='Sequence length T')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--accum-steps', type=int, default=4,
                   help='Gradient accumulation steps (F7.12)')
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    p.add_argument('--grad-clip', type=float, default=1.0)

    # === Convergence ===
    p.add_argument('--convergence-threshold', type=float, default=1e-5)
    p.add_argument('--convergence-patience', type=int, default=50)
    p.add_argument('--convergence-active', action='store_true', default=True)

    # === Checkpointing ===
    p.add_argument('--output', type=str, default='models/condornet_v43_best.pth')
    p.add_argument('--checkpoint-dir', type=str, default='models/ckpts/')
    p.add_argument('--keep-ckpts', type=int, default=3)

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
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Crash sentinel (F7.9) ───────────────────────────────────────────
    if check_sentinel():
        print(f"  ⚠  CRASH SENTINEL detected from previous run!")
        if not args.resume:
            print(f"     Use --resume to continue, or delete .training_active manually")
    write_sentinel()

    # ── GUI telemetry ───────────────────────────────────────────────────
    emitter = get_emitter() if GUI_TELEMETRY_AVAILABLE else None

    # ── Data loading ────────────────────────────────────────────────────
    print(f"\n[DATA] Loading v4.3 multi-TF datasets...")
    train_loader, val_loader, norm_stats, n_train_batches, n_val_batches = \
        build_dataloaders_v43(args)

    print(f"[DATA] Train batches: {n_train_batches} | Val batches: {n_val_batches}")

    # ── Model ───────────────────────────────────────────────────────────
    config = {
        'd_tf_in': args.d_tf_in,
        'd_joint': args.d_joint,
        'd_chain': args.d_chain,
        'd_pivot': args.d_pivot,
        'd_fused': args.d_fused,
        'n_pivot_features': N_PIVOT_FEATURES,
        'n_strategy_types': args.n_strategy_types,
        'chain_in_features': CHAIN_GRID_CONFIG['in_features'],
        'chain_d_model': CHAIN_GRID_CONFIG['d_model'],
        'chain_n_heads': CHAIN_GRID_CONFIG['n_heads'],
        'chain_n_layers': CHAIN_GRID_CONFIG['n_layers'],
        'chain_d_ff': CHAIN_GRID_CONFIG['d_ff'],
    }
    model = build_condornet_v43(config)
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

    # ── Resume from checkpoint (F7.8) ───────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_train_gap = float('inf')
    best_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"\n[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
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

        epoch_loss = 0.0
        epoch_components = {}
        epoch_batches_run = 0
        convergence_manager.reset()

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
            labels = {k: v.to(device) for k, v in batch['labels'].items()}

            # ── 1. FORWARD ──────────────────────────────────────────
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                     enabled=(device.type == 'cuda')):
                outputs = model(
                    x_m1=x_m1, x_m5=x_m5, x_m15=x_m15, x_h1=x_h1,
                    chain=chain, chain_mask=chain_mask,
                    pivot_features=piv_vals, pivot_mask=piv_mask,
                )

            # ── 2. LOSS (outside autocast for precision) ────────────
            loss, components = criterion(outputs, labels)

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
                labels = {k: v.to(device) for k, v in batch['labels'].items()}

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                         enabled=(device.type == 'cuda')):
                    outputs = model(
                        x_m1=x_m1, x_m5=x_m5, x_m15=x_m15, x_h1=x_h1,
                        chain=chain, chain_mask=chain_mask,
                        pivot_features=piv_vals, pivot_mask=piv_mask,
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

        print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
              f"{anneal_str}")

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

        # ── Best model (generalization gate) ────────────────────────────
        val_train_gap = abs(avg_val_loss - avg_train_loss)

        if avg_val_loss < avg_train_loss:
            is_new_best = False
            if best_val_loss == float('inf'):
                is_new_best = True
                reason = "First epoch passing generalization"
            elif avg_val_loss < best_val_loss and val_train_gap < best_val_train_gap:
                is_new_best = True
                reason = (f"Better val ({avg_val_loss:.4f} < {best_val_loss:.4f}) "
                          f"AND tighter gap ({val_train_gap:.4f} < "
                          f"{best_val_train_gap:.4f})")
            elif avg_val_loss < best_val_loss:
                is_new_best = True
                reason = f"Better val ({avg_val_loss:.4f} < {best_val_loss:.4f})"
            else:
                reason = f"Val not better, best remains Epoch {best_epoch}"

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
                print(f"  -> Epoch {epoch+1} generalizes but NOT new best: {reason}")
        else:
            patience_counter += 1
            print(f"  -> Epoch {epoch+1} OVERFITTING: val ({avg_val_loss:.4f}) "
                  f">= train ({avg_train_loss:.4f})")

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
    print(f"{'='*60}")

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
