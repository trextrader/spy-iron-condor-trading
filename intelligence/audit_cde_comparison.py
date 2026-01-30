#!/usr/bin/env python
"""
Multi-Model CDE Interpretability Comparison Audit (Comprehensive Edition)

Compares 2+ Neural CDE checkpoints with:
- Permutation importance rankings
- Surrogate decision tree analysis
- Mathematical divergence metrics (Cosine, JS, Spearman, Wasserstein)
- Physics-inspired stability analysis (Energy, Coherence, KS)
- Gradient alignment / saliency analysis
- Fisher Information estimation
- Hessian eigenspectrum (loss landscape curvature)
- SHAP-style attribution comparison
- Mutual Information analysis
- Comprehensive visualizations with interpretation

Usage:
    python intelligence/audit_cde_comparison.py \
        --models models/epoch_1_012926.pth models/epoch_3_013026.pth \
        --data data/processed/mamba_institutional_2024_1m_v22.csv \
        --samples 3000 \
        --output reports/model_comparison.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import sys
import os
import json
import warnings
from datetime import datetime
from collections import OrderedDict
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_squared_error, mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import wasserstein_distance, entropy
from scipy.linalg import eigvalsh
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.getcwd())

from intelligence.condor_brain import CondorBrain
from intelligence.canonical_feature_registry import FEATURE_COLS_V22, INPUT_DIM_V22

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-6

# Output head names for interpretation
HEAD_NAMES = [
    'call_offset', 'put_offset', 'wing_width', 'dte',
    'pop', 'roi', 'max_loss', 'confidence', 'entry', 'exit'
]

# Color schemes
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95C623']
CMAP_DIVERGE = 'RdBu_r'
CMAP_SEQ = 'viridis'


def safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_cde_model(ckpt_path, input_dim=None):
    """Load a CDE model from checkpoint."""
    if input_dim is None:
        input_dim = INPUT_DIM_V22

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    seq_len = ckpt.get('seq_len', 256)
    config = ckpt.get('model_config', ckpt.get('config', {}))
    d_model = config.get('d_model', 128)
    n_layers = config.get('n_layers', 2)
    use_topk = config.get('use_topk_moe', False)

    model = CondorBrain(
        d_model=d_model,
        n_layers=n_layers,
        input_dim=input_dim,
        use_cde=True,
        use_topk_moe=use_topk
    )

    state_dict = ckpt['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, ckpt, seq_len


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_permutation_importance(model, X, feature_names, seq_len, n_samples=1000):
    """Compute feature importance via permutation."""
    n_samples = min(n_samples, len(X) - seq_len)
    indices = np.random.choice(len(X) - seq_len, n_samples, replace=False)

    batch_X = []
    for idx in indices:
        x_seq = X[idx : idx + seq_len]
        if len(x_seq) == seq_len:
            batch_X.append(torch.tensor(x_seq, dtype=torch.float32))

    if not batch_X:
        return {}

    X_base = torch.stack(batch_X).to(DEVICE)

    with torch.no_grad():
        base_out = model(X_base)
        if isinstance(base_out, tuple):
            base_out = base_out[0]

    valid_mask = torch.isfinite(base_out).all(dim=1)
    n_valid = valid_mask.sum().item()

    if n_valid == 0:
        return {}

    X_base = X_base[valid_mask]
    base_out = base_out[valid_mask]

    importances = {}
    per_head_importances = {h: {} for h in HEAD_NAMES[:base_out.shape[1]]}

    for i, fname in enumerate(tqdm(feature_names, desc="Permutation importance", leave=False)):
        X_perm = X_base.clone()
        perm_indices = torch.randperm(n_valid)
        X_perm[:, :, i] = X_perm[perm_indices, :, i]

        with torch.no_grad():
            perm_out = model(X_perm)
            if isinstance(perm_out, tuple):
                perm_out = perm_out[0]

        # Global importance
        diff_tensor = torch.abs(base_out - perm_out)
        diff_per_sample = diff_tensor.mean(dim=1)
        idx_valid_perm = torch.isfinite(diff_per_sample)

        if idx_valid_perm.sum() > 0:
            diff = diff_per_sample[idx_valid_perm].mean().item()
        else:
            diff = 0.0

        importances[fname] = diff

        # Per-head importance
        for h_idx, h_name in enumerate(HEAD_NAMES[:base_out.shape[1]]):
            head_diff = torch.abs(base_out[:, h_idx] - perm_out[:, h_idx])
            valid_head = torch.isfinite(head_diff)
            if valid_head.sum() > 0:
                per_head_importances[h_name][fname] = head_diff[valid_head].mean().item()
            else:
                per_head_importances[h_name][fname] = 0.0

    # Normalize to 0-100
    total_impact = sum(importances.values())
    if total_impact > 0:
        for k in importances:
            importances[k] = (importances[k] / total_impact) * 100.0

    # Normalize per-head
    for h_name in per_head_importances:
        total = sum(per_head_importances[h_name].values())
        if total > 0:
            for k in per_head_importances[h_name]:
                per_head_importances[h_name][k] = (per_head_importances[h_name][k] / total) * 100.0

    return importances, per_head_importances


def compute_gradient_saliency(model, X, seq_len, n_samples=500):
    """Compute input gradient saliency maps."""
    n_samples = min(n_samples, len(X) - seq_len)
    indices = np.random.choice(len(X) - seq_len, n_samples, replace=False)

    all_grads = []

    model.eval()
    for idx in tqdm(indices, desc="Computing gradients", leave=False):
        seq = X[idx : idx + seq_len]
        x_tensor = torch.tensor(seq, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        x_tensor.requires_grad_(True)

        out = model(x_tensor)
        if isinstance(out, tuple):
            out = out[0]

        # Sum all outputs for gradient
        loss = out.sum()

        model.zero_grad()
        loss.backward()

        if x_tensor.grad is not None:
            grad = x_tensor.grad.detach().cpu().numpy()[0]  # (T, D)
            all_grads.append(grad)

        x_tensor.requires_grad_(False)

    if not all_grads:
        return None

    # Average absolute gradient across samples
    grads = np.stack(all_grads)  # (N, T, D)
    mean_abs_grad = np.mean(np.abs(grads), axis=(0, 1))  # (D,)

    # Temporal gradient pattern
    temporal_grad = np.mean(np.abs(grads), axis=(0, 2))  # (T,)

    return {
        'feature_saliency': mean_abs_grad,
        'temporal_saliency': temporal_grad,
        'raw_grads': grads,
    }


def estimate_fisher_information(model, X, seq_len, n_samples=300):
    """Estimate Fisher Information Matrix diagonal (parameter sensitivity)."""
    n_samples = min(n_samples, len(X) - seq_len)
    indices = np.random.choice(len(X) - seq_len, n_samples, replace=False)

    # Collect squared gradients for each parameter
    fisher_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

    model.eval()
    for idx in tqdm(indices, desc="Estimating Fisher", leave=False):
        seq = X[idx : idx + seq_len]
        x_tensor = torch.tensor(seq, device=DEVICE, dtype=torch.float32).unsqueeze(0)

        out = model(x_tensor)
        if isinstance(out, tuple):
            out = out[0]

        # Use log-likelihood proxy (negative MSE from mean)
        log_prob = -((out - out.mean()) ** 2).sum()

        model.zero_grad()
        log_prob.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher_diag:
                fisher_diag[name] += param.grad.detach() ** 2

    # Average
    for name in fisher_diag:
        fisher_diag[name] /= n_samples

    # Summarize by layer
    layer_fisher = {}
    for name, values in fisher_diag.items():
        layer_name = name.split('.')[0]
        if layer_name not in layer_fisher:
            layer_fisher[layer_name] = 0.0
        layer_fisher[layer_name] += values.sum().item()

    return fisher_diag, layer_fisher


def estimate_hessian_spectrum(model, X, seq_len, n_samples=100, n_eigenvalues=20):
    """Estimate top eigenvalues of the Hessian (loss landscape curvature)."""
    n_samples = min(n_samples, len(X) - seq_len)
    indices = np.random.choice(len(X) - seq_len, n_samples, replace=False)

    # Collect parameters
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    if n_params > 100000:
        print(f"  [Hessian] Too many params ({n_params}), using stochastic approximation")
        # Use Hutchinson's estimator for large models
        return estimate_hessian_hutchinson(model, X, seq_len, indices, n_eigenvalues)

    # For smaller models, compute full Hessian
    hessian_approx = torch.zeros(n_params, n_params, device=DEVICE)

    model.eval()
    for idx in tqdm(indices[:50], desc="Estimating Hessian", leave=False):  # Limit for speed
        seq = X[idx : idx + seq_len]
        x_tensor = torch.tensor(seq, device=DEVICE, dtype=torch.float32).unsqueeze(0)

        out = model(x_tensor)
        if isinstance(out, tuple):
            out = out[0]

        loss = ((out) ** 2).mean()

        # First-order gradients
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        grad_vec = torch.cat([g.view(-1) for g in grads])

        # Approximate Hessian via outer product (Gauss-Newton approximation)
        hessian_approx += torch.outer(grad_vec, grad_vec).detach()

    hessian_approx /= len(indices[:50])

    # Compute eigenvalues
    try:
        eigenvalues = torch.linalg.eigvalsh(hessian_approx.cpu())
        eigenvalues = eigenvalues.numpy()
        eigenvalues = np.sort(eigenvalues)[::-1][:n_eigenvalues]
    except Exception as e:
        print(f"  [Hessian] Eigenvalue computation failed: {e}")
        eigenvalues = np.zeros(n_eigenvalues)

    return eigenvalues


def estimate_hessian_hutchinson(model, X, seq_len, indices, n_eigenvalues=20, n_vectors=50):
    """Stochastic Hessian trace estimation using Hutchinson's method."""
    traces = []

    params = [p for p in model.parameters() if p.requires_grad]

    for _ in tqdm(range(n_vectors), desc="Hutchinson estimation", leave=False):
        # Random Rademacher vector
        v = [torch.randint_like(p, 0, 2).float() * 2 - 1 for p in params]

        total_hvp = 0.0

        for idx in indices[:20]:
            seq = X[idx : idx + seq_len]
            x_tensor = torch.tensor(seq, device=DEVICE, dtype=torch.float32).unsqueeze(0)

            out = model(x_tensor)
            if isinstance(out, tuple):
                out = out[0]

            loss = ((out) ** 2).mean()

            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

            # Hessian-vector product
            grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))
            hvp = torch.autograd.grad(grad_v, params, allow_unused=True)

            # v^T H v approximates trace
            for hi, vi in zip(hvp, v):
                if hi is not None:
                    total_hvp += (hi * vi).sum().item()

        traces.append(total_hvp / len(indices[:20]))

    # Return trace estimate and variance
    trace_mean = np.mean(traces)
    trace_std = np.std(traces)

    # Approximate top eigenvalue from trace (very rough)
    approx_eigenvalues = np.array([trace_mean / (i + 1) for i in range(n_eigenvalues)])

    return approx_eigenvalues


def compute_shap_approximation(model, X, feature_names, seq_len, n_samples=200, n_background=50):
    """Approximate SHAP values using sampling-based Shapley estimation."""
    n_samples = min(n_samples, len(X) - seq_len)
    indices = np.random.choice(len(X) - seq_len, n_samples, replace=False)

    # Background samples for expected value
    bg_indices = np.random.choice(len(X) - seq_len, n_background, replace=False)
    background = []
    for idx in bg_indices:
        seq = X[idx : idx + seq_len]
        background.append(torch.tensor(seq, dtype=torch.float32))
    background = torch.stack(background).to(DEVICE)

    with torch.no_grad():
        bg_out = model(background)
        if isinstance(bg_out, tuple):
            bg_out = bg_out[0]
        expected_value = bg_out.mean(dim=0).cpu().numpy()

    # SHAP approximation via feature ablation
    n_features = len(feature_names)
    shap_values = np.zeros((n_samples, n_features, len(expected_value)))

    model.eval()
    for s_idx, idx in enumerate(tqdm(indices, desc="SHAP approximation", leave=False)):
        seq = X[idx : idx + seq_len]
        x_tensor = torch.tensor(seq, device=DEVICE, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            base_out = model(x_tensor)
            if isinstance(base_out, tuple):
                base_out = base_out[0]
            base_pred = base_out[0].cpu().numpy()

        # Ablate each feature
        for f_idx in range(n_features):
            x_ablated = x_tensor.clone()
            # Replace with background mean
            x_ablated[:, :, f_idx] = background[:, :, f_idx].mean()

            with torch.no_grad():
                ablated_out = model(x_ablated)
                if isinstance(ablated_out, tuple):
                    ablated_out = ablated_out[0]
                ablated_pred = ablated_out[0].cpu().numpy()

            # SHAP ≈ (original - ablated) contribution
            shap_values[s_idx, f_idx, :] = base_pred - ablated_pred

    # Average absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))  # (n_features,)

    # Per-head SHAP
    per_head_shap = {}
    for h_idx, h_name in enumerate(HEAD_NAMES[:shap_values.shape[2]]):
        per_head_shap[h_name] = np.mean(np.abs(shap_values[:, :, h_idx]), axis=0)

    return {
        'mean_abs_shap': dict(zip(feature_names, mean_abs_shap)),
        'per_head_shap': per_head_shap,
        'raw_shap': shap_values,
        'expected_value': expected_value,
    }


def compute_mutual_information(X, feature_names, seq_len, n_samples=5000):
    """Compute pairwise mutual information between features."""
    n_samples = min(n_samples, len(X))

    # Use last timestep of sequences
    indices = np.random.choice(len(X) - seq_len, n_samples, replace=False)
    data = X[indices + seq_len - 1]  # (n_samples, n_features)

    n_features = len(feature_names)
    mi_matrix = np.zeros((n_features, n_features))

    # Discretize for MI calculation
    n_bins = 20
    data_discrete = np.zeros_like(data, dtype=int)
    for i in range(n_features):
        data_discrete[:, i] = np.digitize(data[:, i], np.linspace(data[:, i].min(), data[:, i].max(), n_bins))

    for i in tqdm(range(n_features), desc="Computing MI", leave=False):
        for j in range(i, n_features):
            mi = mutual_info_score(data_discrete[:, i], data_discrete[:, j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    # Normalize to [0, 1]
    mi_max = mi_matrix.max()
    if mi_max > 0:
        mi_matrix /= mi_max

    return mi_matrix


def train_surrogate_tree(model, X, feature_cols, seq_len, n_samples=5000, target_head=5):
    """Train surrogate decision tree for a specific output head."""
    n_samples = min(n_samples, len(X) - seq_len)
    idxs = np.random.randint(0, len(X) - seq_len, size=n_samples)

    input_states = []
    targets = []

    with torch.no_grad():
        for idx in idxs:
            seq = X[idx : idx + seq_len]
            last_step = seq[-1]

            x_tensor = torch.tensor(seq, device=DEVICE).unsqueeze(0).float()
            out = model(x_tensor)
            if isinstance(out, tuple):
                out = out[0]

            pred = out[0, target_head].item()

            if np.isfinite(pred):
                input_states.append(last_step)
                targets.append(pred)

    if len(input_states) < 50:
        return None, None, None, None

    input_states = np.array(input_states)
    targets = np.array(targets)

    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=30)
    tree.fit(input_states, targets)

    r2 = tree.score(input_states, targets)
    rules = export_text(tree, feature_names=feature_cols, decimals=3)

    tree_importances = dict(zip(feature_cols, tree.feature_importances_ * 100))

    return tree, r2, rules, tree_importances


def compute_output_statistics(model, X, seq_len, n_samples=2000):
    """Compute output distribution statistics for each head."""
    n_samples = min(n_samples, len(X) - seq_len)
    idxs = np.random.randint(0, len(X) - seq_len, size=n_samples)

    all_outputs = []

    with torch.no_grad():
        for idx in idxs:
            seq = X[idx : idx + seq_len]
            x_tensor = torch.tensor(seq, device=DEVICE).unsqueeze(0).float()
            out = model(x_tensor)
            if isinstance(out, tuple):
                out = out[0]
            all_outputs.append(out[0].cpu().numpy())

    outputs = np.array(all_outputs)
    valid_mask = np.isfinite(outputs).all(axis=1)
    outputs = outputs[valid_mask]

    stats_dict = {}
    for i, head_name in enumerate(HEAD_NAMES[:outputs.shape[1]]):
        col = outputs[:, i]
        stats_dict[head_name] = {
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'min': float(np.min(col)),
            'max': float(np.max(col)),
            'skew': float(stats.skew(col)),
            'kurtosis': float(stats.kurtosis(col)),
            'q25': float(np.percentile(col, 25)),
            'q75': float(np.percentile(col, 75)),
        }

    return stats_dict, outputs
