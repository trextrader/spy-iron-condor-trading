"""
CondorBrain Training with Differentiable Predicate Discovery

Jointly learns:
1. Which inequality patterns (predicates) are predictive
2. How to combine predicates with raw features
3. The neural backbone weights

The discovered predicates are:
- Human-readable: "H[0] > H[1] AND L[0] > L[1]"
- Scale-invariant: relations, not absolute values
- Exportable: save to JSON for decision tree augmentation

Usage:
  python intelligence/train_with_predicates.py \
      --local-data data/processed/mamba_institutional_2024_1m_v22.csv \
      --epochs 20 \
      --predicate-slots 2048 \
      --max-active-predicates 256 \
      --sparsity-weight 0.001 \
      --output models/condor_with_predicates.pth

Author: Claude Code
Version: 1.0.0
"""

import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import time
import argparse
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.canonical_feature_registry import FEATURE_COLS_V22, VERSION_V22

# Try to import existing CondorBrain components
try:
    from intelligence.condor_brain import CondorBrain, HAS_CDE
    HAS_CONDOR_BRAIN = True
except ImportError:
    HAS_CONDOR_BRAIN = False
    HAS_CDE = False

# =============================================================================
# PREDICATE GRAMMAR (Inline for self-contained training)
# =============================================================================

ALL_FIELDS = FEATURE_COLS_V22.copy()
FIELD_TO_IDX = {f: i for i, f in enumerate(ALL_FIELDS)}
IDX_TO_FIELD = {i: f for i, f in enumerate(ALL_FIELDS)}
N_FIELDS = len(ALL_FIELDS)


class ArithOp(IntEnum):
    NONE = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4


class CompareOp(IntEnum):
    GT = 0
    LT = 1
    GTE = 2
    LTE = 3
    EQ = 4


class LogicOp(IntEnum):
    AND = 0
    OR = 1


COMPARE_SYMBOLS = ['>', '<', '>=', '<=', '==']
ARITH_SYMBOLS = ['', '+', '-', '*', '/']
LOGIC_SYMBOLS = ['AND', 'OR']


@dataclass
class Atom:
    field1: int
    lookback1: int
    arith_op: ArithOp
    field2: int = 0
    lookback2: int = 0

    def is_simple(self) -> bool:
        return self.arith_op == ArithOp.NONE

    def __str__(self) -> str:
        f1 = IDX_TO_FIELD.get(self.field1, f"F{self.field1}")
        if self.is_simple():
            return f"{f1}[{self.lookback1}]"
        f2 = IDX_TO_FIELD.get(self.field2, f"F{self.field2}")
        op = ARITH_SYMBOLS[self.arith_op]
        return f"{{{f1}[{self.lookback1}]{op}{f2}[{self.lookback2}]}}"


@dataclass
class Inequality:
    left: Atom
    compare: CompareOp
    right: Atom

    def __str__(self) -> str:
        return f"{self.left} {COMPARE_SYMBOLS[self.compare]} {self.right}"


@dataclass
class Predicate:
    inequalities: List[Inequality]
    logic_ops: List[LogicOp]

    def __str__(self) -> str:
        if len(self.inequalities) == 1:
            return str(self.inequalities[0])
        parts = [str(self.inequalities[0])]
        for ineq, op in zip(self.inequalities[1:], self.logic_ops):
            parts.append(f" {LOGIC_SYMBOLS[op]} {ineq}")
        return ''.join(parts)


# =============================================================================
# DIFFERENTIABLE PREDICATE SELECTOR
# =============================================================================

class PredicateSelector(nn.Module):
    """
    Learns which inequality predicates are useful from a combinatorial search space.

    Each "slot" learns to decode into inequality parameters:
    - Left atom (field, lookback, optional arithmetic)
    - Compare operator
    - Right atom (field, lookback, optional arithmetic)

    Importance weights control which predicates are active.
    Sparsity loss encourages finding minimal predicate sets.
    """

    def __init__(
        self,
        n_slots: int = 2048,
        max_active: int = 256,
        d_embed: int = 128,
        n_fields: int = N_FIELDS,
        max_lookback: int = 128,
        temperature: float = 1.0
    ):
        super().__init__()

        self.n_slots = n_slots
        self.max_active = max_active
        self.n_fields = n_fields
        self.max_lookback = max_lookback
        self.temperature = temperature

        # Learnable predicate embeddings
        self.predicate_embeddings = nn.Parameter(torch.randn(n_slots, d_embed) * 0.02)

        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.GELU(),
            nn.LayerNorm(d_embed * 2),
            nn.Linear(d_embed * 2, d_embed * 2),
            nn.GELU(),
        )

        # Parameter heads
        self.left_field1 = nn.Linear(d_embed * 2, n_fields)
        self.left_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.left_arith = nn.Linear(d_embed * 2, 5)
        self.left_field2 = nn.Linear(d_embed * 2, n_fields)
        self.left_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        self.compare_op = nn.Linear(d_embed * 2, 5)

        self.right_field1 = nn.Linear(d_embed * 2, n_fields)
        self.right_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.right_arith = nn.Linear(d_embed * 2, 5)
        self.right_field2 = nn.Linear(d_embed * 2, n_fields)
        self.right_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        # Importance scores (learnable sparsity)
        self.importance_logits = nn.Parameter(torch.zeros(n_slots))

    def _gumbel_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Differentiable sampling using Gumbel-softmax."""
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            indices = (probs * torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype)).sum(-1)
        else:
            indices = logits.argmax(-1).float()
        return indices

    def forward(self, return_params: bool = False):
        """
        Returns:
            importance: (n_slots,) importance weights
            params: (n_slots, 11) predicate parameters if return_params=True
        """
        importance = torch.sigmoid(self.importance_logits)

        if not return_params:
            return importance, None

        hidden = self.decoder(self.predicate_embeddings)

        params = torch.stack([
            self._gumbel_sample(self.left_field1(hidden)),
            self._gumbel_sample(self.left_lookback1(hidden)),
            self._gumbel_sample(self.left_arith(hidden)),
            self._gumbel_sample(self.left_field2(hidden)),
            self._gumbel_sample(self.left_lookback2(hidden)),
            self._gumbel_sample(self.compare_op(hidden)),
            self._gumbel_sample(self.right_field1(hidden)),
            self._gumbel_sample(self.right_lookback1(hidden)),
            self._gumbel_sample(self.right_arith(hidden)),
            self._gumbel_sample(self.right_field2(hidden)),
            self._gumbel_sample(self.right_lookback2(hidden)),
        ], dim=-1)

        return importance, params

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty on active predicates."""
        return torch.sigmoid(self.importance_logits).sum()

    def get_active_predicates(
        self,
        threshold: float = 0.1,
        max_return: int = None
    ) -> Tuple[List[Predicate], np.ndarray, List[str]]:
        """Extract human-readable predicates above importance threshold."""
        self.eval()
        with torch.no_grad():
            importance, params = self.forward(return_params=True)

            mask = importance > threshold
            active_idx = torch.where(mask)[0]

            if len(active_idx) == 0:
                return [], np.array([]), []

            active_importance = importance[active_idx]
            sort_idx = torch.argsort(active_importance, descending=True)
            active_idx = active_idx[sort_idx]

            if max_return and len(active_idx) > max_return:
                active_idx = active_idx[:max_return]

            active_params = params[active_idx].cpu().numpy().astype(int)
            active_importance = importance[active_idx].cpu().numpy()

            predicates = []
            names = []

            for i, p in enumerate(active_params):
                left = Atom(p[0], p[1], ArithOp(p[2]), p[3], p[4])
                right = Atom(p[6], p[7], ArithOp(p[8]), p[9], p[10])
                ineq = Inequality(left, CompareOp(p[5]), right)
                pred = Predicate([ineq], [])
                predicates.append(pred)
                names.append(f"{pred} (imp={active_importance[i]:.3f})")

            return predicates, active_importance, names


# =============================================================================
# PREDICATE EVALUATOR (Vectorized for GPU)
# =============================================================================

def evaluate_predicates_gpu(
    data: torch.Tensor,  # (batch, seq, n_fields) or (N, n_fields)
    params: torch.Tensor,  # (K, 11) predicate parameters
    importance: torch.Tensor,  # (K,) importance weights
    max_active: int = 256,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Evaluate predicates on GPU using vectorized operations.

    This is a differentiable approximation that allows gradients to flow
    back to the predicate selector.

    Args:
        data: Input data tensor
        params: Predicate parameters (K predicates × 11 params)
        importance: Importance weights for each predicate
        max_active: Maximum active predicates

    Returns:
        (batch, seq, max_active) or (N, max_active) predicate activations
    """
    # Handle 2D vs 3D input
    if data.dim() == 2:
        data = data.unsqueeze(0)  # (1, N, F)
        squeeze_out = True
    else:
        squeeze_out = False

    batch, seq_len, n_fields = data.shape
    K = params.shape[0]
    device = data.device

    # Select top-K by importance
    top_k = min(max_active, K)
    _, top_idx = torch.topk(importance, top_k)
    active_params = params[top_idx]  # (top_k, 11)
    active_importance = importance[top_idx]  # (top_k,)

    # Extract parameters
    l_f1 = active_params[:, 0].long()  # (top_k,)
    l_n1 = active_params[:, 1].long()
    l_op = active_params[:, 2].long()
    l_f2 = active_params[:, 3].long()
    l_n2 = active_params[:, 4].long()
    cmp = active_params[:, 5].long()
    r_f1 = active_params[:, 6].long()
    r_n1 = active_params[:, 7].long()
    r_op = active_params[:, 8].long()
    r_f2 = active_params[:, 9].long()
    r_n2 = active_params[:, 10].long()

    # Max lookback for this batch
    max_lb = torch.max(torch.stack([l_n1, l_n2, r_n1, r_n2])).item()
    max_lb = min(max_lb, seq_len - 1)

    # Initialize output
    result = torch.zeros(batch, seq_len, max_active, device=device, dtype=data.dtype)

    # Evaluate predicates (vectorized over batch and predicates)
    for t in range(max_lb, seq_len):
        # Gather field values at lookback positions for all predicates
        # data: (batch, seq, n_fields)
        # We need data[b, t - lookback, field] for each predicate

        # Left atom value
        left_val = torch.zeros(batch, top_k, device=device, dtype=data.dtype)
        for k in range(top_k):
            t_l1 = max(0, t - l_n1[k].item())
            left_val[:, k] = data[:, t_l1, l_f1[k]]

            if l_op[k] > 0:  # Compound atom
                t_l2 = max(0, t - l_n2[k].item())
                val2 = data[:, t_l2, l_f2[k]]
                if l_op[k] == 1:  # ADD
                    left_val[:, k] = left_val[:, k] + val2
                elif l_op[k] == 2:  # SUB
                    left_val[:, k] = left_val[:, k] - val2
                elif l_op[k] == 3:  # MUL
                    left_val[:, k] = left_val[:, k] * val2
                elif l_op[k] == 4:  # DIV
                    left_val[:, k] = left_val[:, k] / (val2 + eps)

        # Right atom value
        right_val = torch.zeros(batch, top_k, device=device, dtype=data.dtype)
        for k in range(top_k):
            t_r1 = max(0, t - r_n1[k].item())
            right_val[:, k] = data[:, t_r1, r_f1[k]]

            if r_op[k] > 0:
                t_r2 = max(0, t - r_n2[k].item())
                val2 = data[:, t_r2, r_f2[k]]
                if r_op[k] == 1:
                    right_val[:, k] = right_val[:, k] + val2
                elif r_op[k] == 2:
                    right_val[:, k] = right_val[:, k] - val2
                elif r_op[k] == 3:
                    right_val[:, k] = right_val[:, k] * val2
                elif r_op[k] == 4:
                    right_val[:, k] = right_val[:, k] / (val2 + eps)

        # Compare (soft for differentiability during training)
        diff = left_val - right_val  # (batch, top_k)

        # Soft comparisons using sigmoid
        steepness = 10.0  # Controls sharpness of comparison
        for k in range(top_k):
            if cmp[k] == 0:  # GT: left > right
                result[:, t, k] = torch.sigmoid(steepness * diff[:, k])
            elif cmp[k] == 1:  # LT: left < right
                result[:, t, k] = torch.sigmoid(-steepness * diff[:, k])
            elif cmp[k] == 2:  # GTE
                result[:, t, k] = torch.sigmoid(steepness * diff[:, k])
            elif cmp[k] == 3:  # LTE
                result[:, t, k] = torch.sigmoid(-steepness * diff[:, k])
            elif cmp[k] == 4:  # EQ (approximate)
                result[:, t, k] = torch.exp(-steepness * diff[:, k].abs())

    # Weight by importance
    result[:, :, :top_k] = result[:, :, :top_k] * active_importance.unsqueeze(0).unsqueeze(0)

    if squeeze_out:
        result = result.squeeze(0)

    return result


# =============================================================================
# PREDICATE-AUGMENTED MODEL
# =============================================================================

class PredicateAugmentedCondorBrain(nn.Module):
    """
    CondorBrain augmented with differentiable predicate discovery.

    Architecture:
    1. PredicateSelector learns which inequalities are useful
    2. Predicates evaluated on raw data → boolean features
    3. Predicate features + raw features → input projection
    4. Transformer/CDE backbone processes combined features
    5. Output heads produce predictions

    During training:
    - Predicate parameters and importance are jointly optimized
    - Sparsity loss keeps predicate count manageable
    - Discovered predicates become permanent features
    """

    def __init__(
        self,
        input_dim: int = 54,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_predicate_slots: int = 2048,
        max_active_predicates: int = 256,
        n_output_heads: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_active = max_active_predicates

        # Predicate discovery module
        self.predicate_selector = PredicateSelector(
            n_slots=n_predicate_slots,
            max_active=max_active_predicates,
            n_fields=input_dim,
        )

        # Combined input dimension
        combined_dim = input_dim + max_active_predicates

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.output_heads = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_output_heads),
        )

        # Predicate feature aggregator (learns to combine predicate activations over time)
        self.pred_aggregator = nn.Sequential(
            nn.Linear(max_active_predicates, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq, input_dim) - raw features
        return_pred_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with predicate evaluation.

        Args:
            x: Raw input features (already normalized)
            return_pred_features: If True, also return predicate activations

        Returns:
            outputs: (batch, n_output_heads)
            pred_features: (batch, seq, max_active) if return_pred_features
        """
        batch, seq_len, _ = x.shape

        # Get predicate parameters and importance
        importance, params = self.predicate_selector(return_params=True)

        # Evaluate predicates on input data
        pred_features = evaluate_predicates_gpu(
            x, params, importance, self.max_active
        )

        # Concatenate raw features with predicate features
        combined = torch.cat([x, pred_features], dim=-1)  # (batch, seq, input_dim + max_active)

        # Project to model dimension
        h = self.input_proj(combined)  # (batch, seq, d_model)

        # Process through backbone
        h = self.backbone(h)  # (batch, seq, d_model)

        # Take last timestep for prediction
        h_last = h[:, -1, :]  # (batch, d_model)

        # Output
        outputs = self.output_heads(h_last)

        if return_pred_features:
            return outputs, pred_features
        return outputs

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sparsity_weight: float = 0.001,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with sparsity regularization.

        Returns:
            total_loss: Combined loss tensor
            metrics: Dictionary of loss components
        """
        # Prediction loss (MSE)
        pred_loss = F.mse_loss(pred, target)

        # Sparsity loss (L1 on importance)
        sparse_loss = self.predicate_selector.sparsity_loss()

        # Total loss
        total_loss = pred_loss + sparsity_weight * sparse_loss

        # Count active predicates
        with torch.no_grad():
            importance, _ = self.predicate_selector(return_params=False)
            n_active = (importance > 0.1).sum().item()

        metrics = {
            'pred_loss': pred_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'total_loss': total_loss.item(),
            'n_active_predicates': n_active,
        }

        return total_loss, metrics

    def get_discovered_predicates(self) -> List[str]:
        """Return human-readable list of discovered predicates."""
        _, _, names = self.predicate_selector.get_active_predicates(threshold=0.1)
        return names


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(df: pd.DataFrame, lookback: int = 128):
    """Prepare features and targets for training."""
    print(f"[Data] Preparing {len(df):,} rows...")

    # Select feature columns
    available_cols = [c for c in FEATURE_COLS_V22 if c in df.columns]
    print(f"[Data] Using {len(available_cols)}/{len(FEATURE_COLS_V22)} features")

    # Extract features
    X = df[available_cols].fillna(0).values.astype(np.float32)

    # Normalize (robust z-score)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = np.where(mad < 1e-8, 1.0, 1.4826 * mad)
    X = np.clip((X - med) / scale, -10, 10).astype(np.float32)

    # Generate targets (simplified - use available or compute)
    if 'target_spot' in df.columns:
        close = df['close'].fillna(method='ffill').values
        target = df['target_spot'].fillna(close).values
        ret_5m = np.zeros(len(df))
        ret_5m[:-5] = (close[5:] - close[:-5]) / (close[:-5] + 1e-8)

        y = np.column_stack([
            np.zeros(len(df)),  # call_offset
            np.zeros(len(df)),  # put_offset
            np.full(len(df), 5.0),  # wing_width
            np.full(len(df), 7.0),  # dte
            np.full(len(df), 0.7),  # pop
            ret_5m,  # roi
            np.full(len(df), 0.5),  # max_loss
            np.full(len(df), 0.5),  # confidence
            (ret_5m > 0).astype(float),  # entry_target
            (ret_5m < 0).astype(float),  # exit_target
        ]).astype(np.float32)
    else:
        # Fallback: use close returns
        close = df['close'].values
        ret = np.zeros(len(df))
        ret[:-1] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)

        y = np.column_stack([
            np.zeros(len(df)),
            np.zeros(len(df)),
            np.full(len(df), 5.0),
            np.full(len(df), 7.0),
            np.full(len(df), 0.7),
            ret,
            np.full(len(df), 0.5),
            np.full(len(df), 0.5),
            (ret > 0).astype(float),
            (ret < 0).astype(float),
        ]).astype(np.float32)

    # Clamp targets
    y = np.clip(y, -10, 10)
    y = np.nan_to_num(y, 0)

    print(f"[Data] X shape: {X.shape}, y shape: {y.shape}")

    return X, y, med, scale


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_with_predicates(args):
    """Main training loop with predicate discovery."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"PREDICATE-AUGMENTED TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Predicate slots: {args.predicate_slots}")
    print(f"Max active predicates: {args.max_active}")
    print(f"Sparsity weight: {args.sparsity_weight}")

    # CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        use_bf16 = torch.cuda.is_bf16_supported()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16: {use_bf16}")
    else:
        use_bf16 = False

    # Load data
    print(f"\nLoading data from {args.local_data}...")
    df = pd.read_csv(args.local_data, nrows=args.max_rows if args.max_rows > 0 else None)
    print(f"Loaded {len(df):,} rows")

    X, y, med, scale = prepare_data(df, args.lookback)

    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    del df, X, y
    import gc
    gc.collect()

    # Move to GPU
    L = args.lookback
    dtype_gpu = torch.bfloat16 if use_bf16 else torch.float16

    X_train_t = torch.from_numpy(X_train).to(device=device, dtype=dtype_gpu)
    y_train_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    X_val_t = torch.from_numpy(X_val).to(device=device, dtype=dtype_gpu)
    y_val_t = torch.from_numpy(y_val).to(device=device, dtype=torch.float32)

    # Create sequence views
    X_train_seq = X_train_t.unfold(0, L, 1).permute(0, 2, 1)
    X_val_seq = X_val_t.unfold(0, L, 1).permute(0, 2, 1)

    n_train = X_train_seq.shape[0]
    n_val = X_val_seq.shape[0]
    B = args.batch_size
    n_train_batches = n_train // B
    n_val_batches = max(1, n_val // B)

    print(f"Train sequences: {n_train:,}, batches: {n_train_batches}")
    print(f"Val sequences: {n_val:,}, batches: {n_val_batches}")

    # Model
    model = PredicateAugmentedCondorBrain(
        input_dim=X_train_t.shape[1],
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.n_heads,
        n_predicate_slots=args.predicate_slots,
        max_active_predicates=args.max_active,
        n_output_heads=10,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    # Training
    best_val_loss = float('inf')
    discovered_predicates = []
    history = {'train_loss': [], 'val_loss': [], 'n_predicates': []}

    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # === TRAIN ===
        model.train()
        train_loss = 0
        train_metrics = {'pred_loss': 0, 'sparse_loss': 0, 'n_active': 0}

        # Shuffle batch indices
        perm = torch.randperm(n_train_batches)

        for bi in range(n_train_batches):
            idx = perm[bi].item()
            s = idx * B
            e = s + B

            batch_x = X_train_seq[s:e]
            batch_y = y_train_t[s + L:e + L]

            optimizer.zero_grad()

            if scaler:
                with autocast('cuda', dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    pred = model(batch_x)
                    loss, metrics = model.compute_loss(pred, batch_y, args.sparsity_weight)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(batch_x)
                loss, metrics = model.compute_loss(pred, batch_y, args.sparsity_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += metrics['total_loss']
            train_metrics['pred_loss'] += metrics['pred_loss']
            train_metrics['sparse_loss'] += metrics['sparse_loss']
            train_metrics['n_active'] = metrics['n_active_predicates']

        train_loss /= n_train_batches
        train_metrics['pred_loss'] /= n_train_batches

        # === VALIDATE ===
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for bi in range(n_val_batches):
                s = bi * B
                e = min(s + B, n_val)

                batch_x = X_val_seq[s:e]
                batch_y = y_val_t[s + L:e + L]

                with autocast('cuda', dtype=torch.bfloat16 if use_bf16 and scaler else torch.float32):
                    pred = model(batch_x)
                    loss = F.mse_loss(pred, batch_y)

                val_loss += loss.item()

        val_loss /= n_val_batches

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['n_predicates'].append(train_metrics['n_active'])

        # Get current discovered predicates
        discovered_predicates = model.get_discovered_predicates()

        # Print progress
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} (pred:{train_metrics['pred_loss']:.4f}) | "
              f"Val: {val_loss:.4f} | "
              f"Predicates: {train_metrics['n_active']:3d} | "
              f"LR: {lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # Show sample predicates periodically
        if (epoch + 1) % 5 == 0 and discovered_predicates:
            print(f"  Sample predicates: {discovered_predicates[:3]}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'discovered_predicates': discovered_predicates,
                'config': {
                    'd_model': args.d_model,
                    'n_layers': args.layers,
                    'n_heads': args.n_heads,
                    'predicate_slots': args.predicate_slots,
                    'max_active': args.max_active,
                    'input_dim': X_train_t.shape[1],
                },
                'normalization': {
                    'median': med.tolist(),
                    'scale': scale.tolist(),
                },
            }, args.output)
            print(f"  --> Saved best model (val_loss={val_loss:.4f})")

    # === FINAL EXPORT ===
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final active predicates: {len(discovered_predicates)}")

    # Export discovered predicates
    predicates, importance, names = model.predicate_selector.get_active_predicates(threshold=0.1)

    pred_export = {
        'n_predicates': len(predicates),
        'training_epochs': args.epochs,
        'best_val_loss': best_val_loss,
        'predicates': [
            {
                'expression': str(pred),
                'importance': float(imp),
            }
            for pred, imp in zip(predicates, importance)
        ]
    }

    pred_path = args.output.replace('.pth', '_predicates.json')
    with open(pred_path, 'w') as f:
        json.dump(pred_export, f, indent=2)
    print(f"Exported {len(predicates)} predicates to {pred_path}")

    # Print top predicates
    if names:
        print(f"\nTop 20 Discovered Predicates:")
        print("-" * 60)
        for i, name in enumerate(names[:20]):
            print(f"  {i+1:2d}. {name}")

    return model, discovered_predicates


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CondorBrain with Predicate Discovery")

    # Data
    parser.add_argument("--local-data", type=str, required=True,
                        help="Path to training CSV")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Limit rows (0 = all)")

    # Model architecture
    parser.add_argument("--d-model", type=int, default=256,
                        help="Model dimension")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Number of attention heads")

    # Predicate discovery
    parser.add_argument("--predicate-slots", type=int, default=2048,
                        help="Number of predicate slots to learn")
    parser.add_argument("--max-active", type=int, default=256,
                        help="Maximum active predicates")
    parser.add_argument("--sparsity-weight", type=float, default=0.001,
                        help="Sparsity regularization weight")

    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lookback", type=int, default=128,
                        help="Sequence length")

    # Output
    parser.add_argument("--output", type=str, default="models/condor_with_predicates.pth",
                        help="Output model path")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model, predicates = train_with_predicates(args)
