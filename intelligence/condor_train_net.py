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
from typing import Dict, Tuple, Optional

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
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain_net import (
    CondorNet,
    group_invariant_loss,
    spectral_radius_loss,
)
from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V22,
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
        lambda_npdd: float = 1.0,
        lambda_sharpe: float = 0.2,
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

        # === PREPARE WEIGHTED RETURNS FOR GRADIENT FLOW ===
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

        # === 1. NPDD Loss (Weighted) ===
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


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and targets for CondorNet training.

    Returns:
        (X, y, regime, med, scale)
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

    df = df.ffill().bfill().fillna(0)

    # Build arrays
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target',
        'entry_target', 'exit_target'
    ]

    X = select_feature_frame(df, FEATURE_COLS, strict=True).values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    regime = df['regime_label'].values.astype(np.int64)

    # Sanitize
    X = safe_nan_to_num(X)
    y = clamp_targets(y)

    # Scale volume
    if X.shape[1] > 4:
        X[:, 4] = np.log1p(np.clip(X[:, 4], 0.0, 1e9)).astype(np.float32)

    # Robust normalization
    med, scale = robust_zscore_fit(X)
    X = robust_zscore_transform(X, med, scale)

    print(f"[CondorNet] Features: {X.shape}, Targets: {y.shape}")
    return X, y, regime, med, scale


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

    # Training
    parser.add_argument("--n-predicates", type=int, default=1024)
    parser.add_argument("--n-sets", type=int, default=512)
    parser.add_argument("--n-super-sets", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--accum-steps", type=int, default=4)

    # Loss weights
    parser.add_argument("--lambda-npdd", type=float, default=1.0)
    parser.add_argument("--lambda-sharpe", type=float, default=0.2)
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

    # Load data
    print(f"\n[CondorNet] Loading data from {args.local_data}...")
    if args.max_rows > 0:
        df = pd.read_csv(args.local_data, nrows=args.max_rows)
    else:
        df = pd.read_csv(args.local_data)
    print(f"[CondorNet] Loaded {len(df):,} rows")

    X, y, regime, med, scale = prepare_features(df)
    del df

    # Split
    split_row = int(len(X) * 0.8)
    X_train, X_val = X[:split_row], X[split_row:]
    y_train, y_val = y[:split_row], y[split_row:]

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

    # Loss
    criterion = CompositeCondorNetLoss(
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

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_components = {k: 0.0 for k in criterion.lambdas.keys()}

        pbar = tqdm(range(n_train_batches), desc=f"Epoch {epoch+1}", leave=False)

        optimizer.zero_grad(set_to_none=True)

        for batch_idx in pbar:
            s = batch_idx * B
            e = s + B

            batch_x = X_train_seq[s:e]  # (B, L, F)
            batch_y = y_train_t[s + L:e + L]  # (B, 10)

            # Move to device (KEEP AS FLOAT32 for stable backward pass)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass with standard AMP
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs, diag = model(batch_x, return_diagnostics=True)

                A_matrix = model.get_A_matrix()
                gates = diag.get('predicates')
                state = diag.get('z_final')

            # --- LOSS CALCULATION (OUTSIDE AUTOCAST FOR PRECISION) ---
            # Synthetic return series for Sharpe/Drawdown (across the batch)
            roi_returns = batch_y[:, 5].unsqueeze(0)  # (1, B)

            loss, components = criterion(
                outputs,
                batch_y,
                gates=gates,
                state=state,
                A_matrix=A_matrix,
                pred_signature=model.pred_signature,
                returns=roi_returns,
            )

            # Debug Prints for Crash
            if batch_idx == 0:
                print(f"\n[DEBUG BATCH 0] System Inventory:")
                print(f"  Input batch_x: {batch_x.dtype}")
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

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if getattr(args, 'verbose', False) and (batch_idx + 1) % 10 == 0:
                comp_str = ' | '.join(
                    [f"{k}:{(v.item() if torch.is_tensor(v) else v):.3f}"
                     for k, v in list(components.items())[:5]]
                )
                print(f"  [B{batch_idx+1:04d}] loss={loss.item():.4f} | {comp_str}")

        scheduler.step()
        avg_train_loss = epoch_loss / n_train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx in range(min(n_val_batches, 50)):
                s = batch_idx * B
                e = min(s + B, n_val_seq)
                if e <= s:
                    break

                batch_x = X_val_seq[s:e]
                batch_y = y_val_t[s + L:e + L]

                roi_returns = batch_y[:, 5].unsqueeze(0)

                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs, diag = model(batch_x, return_diagnostics=True)
                    
                    A_matrix = model.get_A_matrix()
                    gates = diag.get('predicates')
                    state = diag.get('z_final')

                    loss, _ = criterion(
                        outputs.float(), 
                        batch_y,
                        gates=gates,
                        state=state,
                        A_matrix=A_matrix,
                        pred_signature=model.pred_signature,
                        returns=roi_returns
                    )

                val_loss += loss.item()

        avg_val_loss = val_loss / min(n_val_batches, 50)

        print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    args = parse_args()
    train_condor_net(args)
