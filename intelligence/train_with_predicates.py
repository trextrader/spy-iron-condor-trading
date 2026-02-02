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
from intelligence.predicate_discovery import (
    PredicateSelector, evaluate_predicates_gpu, 
    TemplateType, ArithOp, CompareOp, LogicOp, Atom, Predicate
)

# Try to import existing CondorBrain components
try:
    from intelligence.condor_brain import CondorBrain, HAS_CDE
    HAS_CONDOR_BRAIN = True
except ImportError:
    HAS_CONDOR_BRAIN = False
    HAS_CDE = False

# Using discovery components from intelligence.predicate_discovery


# =============================================================================
# PREDICATE GRAMMAR (Inline for self-contained training)
# =============================================================================

ALL_FIELDS = FEATURE_COLS_V22.copy()
FIELD_TO_IDX = {f: i for i, f in enumerate(ALL_FIELDS)}
IDX_TO_FIELD = {i: f for i, f in enumerate(ALL_FIELDS)}
N_FIELDS = len(ALL_FIELDS)


COMPARE_SYMBOLS = ['>', '<', '>=', '<=', '==']
ARITH_SYMBOLS = ['', '+', '-', '*', '/']
LOGIC_SYMBOLS = ['AND', 'OR']


@dataclass
class Inequality:
    left: Atom
    compare: CompareOp
    right: Atom

    def __str__(self) -> str:
        return f"{self.left} {COMPARE_SYMBOLS[self.compare]} {self.right}"


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
        max_active_predicates: int = 512,  # Paper: 800+
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
