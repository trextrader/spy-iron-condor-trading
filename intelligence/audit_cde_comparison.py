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


# =============================================================================
# DIVERGENCE & COMPARISON METRICS
# =============================================================================

def compute_divergence_metrics(imp1, imp2, feature_names):
    """Compute mathematical divergence between two importance distributions."""
    v1 = np.array([imp1.get(f, 0) for f in feature_names])
    v2 = np.array([imp2.get(f, 0) for f in feature_names])

    # Normalize to probability distributions
    v1_norm = v1 / (v1.sum() + EPS)
    v2_norm = v2 / (v2.sum() + EPS)

    # Cosine similarity
    cos_sim = 1 - cosine(v1_norm + EPS, v2_norm + EPS)

    # Jensen-Shannon divergence
    js_div = jensenshannon(v1_norm + EPS, v2_norm + EPS)

    # Wasserstein distance (Earth Mover's Distance)
    wass_dist = wasserstein_distance(v1_norm, v2_norm)

    # Spearman rank correlation
    rank1 = stats.rankdata(-v1)
    rank2 = stats.rankdata(-v2)
    spearman_r, spearman_p = stats.spearmanr(rank1, rank2)

    # Kendall's Tau (rank correlation)
    kendall_tau, kendall_p = stats.kendalltau(rank1, rank2)

    # Top-K agreement
    top5_1 = set(sorted(imp1.keys(), key=lambda k: imp1[k], reverse=True)[:5])
    top5_2 = set(sorted(imp2.keys(), key=lambda k: imp2[k], reverse=True)[:5])
    top5_overlap = len(top5_1 & top5_2) / 5.0

    top10_1 = set(sorted(imp1.keys(), key=lambda k: imp1[k], reverse=True)[:10])
    top10_2 = set(sorted(imp2.keys(), key=lambda k: imp2[k], reverse=True)[:10])
    top10_overlap = len(top10_1 & top10_2) / 10.0

    # Largest shifts
    shifts = {f: (imp1.get(f, 0) - imp2.get(f, 0)) for f in feature_names}
    top_shifts = sorted(shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    # Rank changes
    rank_dict1 = {f: r for r, f in enumerate(sorted(imp1.keys(), key=lambda k: imp1[k], reverse=True), 1)}
    rank_dict2 = {f: r for r, f in enumerate(sorted(imp2.keys(), key=lambda k: imp2[k], reverse=True), 1)}
    rank_changes = {f: rank_dict1.get(f, 0) - rank_dict2.get(f, 0) for f in feature_names}
    top_rank_changes = sorted(rank_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    return {
        'cosine_similarity': cos_sim,
        'jensen_shannon_divergence': js_div,
        'wasserstein_distance': wass_dist,
        'spearman_correlation': spearman_r,
        'spearman_pvalue': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_pvalue': kendall_p,
        'top5_overlap': top5_overlap,
        'top10_overlap': top10_overlap,
        'largest_shifts': top_shifts,
        'largest_rank_changes': top_rank_changes,
    }


def compute_gradient_alignment(grads1, grads2):
    """Compute alignment between gradient saliency maps."""
    if grads1 is None or grads2 is None:
        return None

    s1 = grads1['feature_saliency']
    s2 = grads2['feature_saliency']

    # Normalize
    s1_norm = s1 / (np.linalg.norm(s1) + EPS)
    s2_norm = s2 / (np.linalg.norm(s2) + EPS)

    # Cosine alignment
    alignment = np.dot(s1_norm, s2_norm)

    # Per-feature absolute difference
    diff = np.abs(s1 - s2)

    # Temporal alignment
    t1 = grads1['temporal_saliency']
    t2 = grads2['temporal_saliency']
    t1_norm = t1 / (np.linalg.norm(t1) + EPS)
    t2_norm = t2 / (np.linalg.norm(t2) + EPS)
    temporal_alignment = np.dot(t1_norm, t2_norm)

    return {
        'feature_alignment': alignment,
        'temporal_alignment': temporal_alignment,
        'feature_diff': diff,
    }


def compute_stability_metrics(outputs1, outputs2):
    """Physics-inspired stability analysis between two models' outputs."""
    n = min(len(outputs1), len(outputs2))
    o1, o2 = outputs1[:n], outputs2[:n]

    metrics = {}

    for i, head_name in enumerate(HEAD_NAMES[:o1.shape[1]]):
        col1, col2 = o1[:, i], o2[:, i]

        # Mean squared difference (energy)
        msd = np.mean((col1 - col2) ** 2)

        # Correlation (coherence)
        corr, _ = stats.pearsonr(col1, col2)

        # Variance ratio (amplitude stability)
        var_ratio = np.var(col1) / (np.var(col2) + EPS)

        # Distribution shift (Kolmogorov-Smirnov test)
        ks_stat, ks_pval = stats.ks_2samp(col1, col2)

        # Wasserstein distance between output distributions
        wass = wasserstein_distance(col1, col2)

        # Energy distance
        energy_dist = 2 * np.mean(np.abs(col1[:, None] - col2[None, :])) - \
                      np.mean(np.abs(col1[:, None] - col1[None, :])) - \
                      np.mean(np.abs(col2[:, None] - col2[None, :]))

        metrics[head_name] = {
            'msd': float(msd),
            'correlation': float(corr),
            'variance_ratio': float(var_ratio),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'wasserstein': float(wass),
            'energy_distance': float(energy_dist),
        }

    return metrics


def compare_fisher_information(fisher1, fisher2):
    """Compare Fisher Information between models."""
    common_layers = set(fisher1.keys()) & set(fisher2.keys())

    ratios = {}
    for layer in common_layers:
        f1 = fisher1[layer]
        f2 = fisher2[layer]
        if f2 > EPS:
            ratios[layer] = f1 / f2
        else:
            ratios[layer] = float('inf') if f1 > EPS else 1.0

    return ratios


def compare_hessian_spectra(eigs1, eigs2):
    """Compare Hessian eigenvalue spectra."""
    n = min(len(eigs1), len(eigs2))
    e1, e2 = eigs1[:n], eigs2[:n]

    # Spectral norm ratio (largest eigenvalue)
    spectral_ratio = e1[0] / (e2[0] + EPS)

    # Trace ratio (sum of eigenvalues)
    trace_ratio = e1.sum() / (e2.sum() + EPS)

    # Condition number (ratio of largest to smallest)
    cond1 = e1[0] / (np.abs(e1[-1]) + EPS)
    cond2 = e2[0] / (np.abs(e2[-1]) + EPS)

    # Eigenvalue distribution correlation
    corr, _ = stats.pearsonr(e1, e2)

    return {
        'spectral_ratio': spectral_ratio,
        'trace_ratio': trace_ratio,
        'condition_number_1': cond1,
        'condition_number_2': cond2,
        'eigenvalue_correlation': corr,
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_importance_comparison(results, feature_names, output_dir):
    """Plot feature importance comparison across models."""
    model_names = list(results['models'].keys())
    n_models = len(model_names)

    # Get all importances
    all_imps = {name: results['models'][name].get('importances', {}) for name in model_names}

    # Sort by first model's importance
    sorted_features = sorted(feature_names, key=lambda f: all_imps[model_names[0]].get(f, 0), reverse=True)[:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Bar chart comparison
    ax = axes[0]
    x = np.arange(len(sorted_features))
    width = 0.8 / n_models

    for i, name in enumerate(model_names):
        imp = all_imps[name]
        values = [imp.get(f, 0) for f in sorted_features]
        ax.barh(x + i * width, values, width, label=name, color=COLORS[i % len(COLORS)], alpha=0.8)

    ax.set_yticks(x + width * (n_models - 1) / 2)
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.set_xlabel('Importance (%)', fontsize=11)
    ax.set_title('Feature Importance Comparison (Top 20)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Rank comparison scatter
    ax = axes[1]
    if n_models >= 2:
        imp1 = all_imps[model_names[0]]
        imp2 = all_imps[model_names[1]]

        rank1 = {f: r for r, f in enumerate(sorted(imp1.keys(), key=lambda k: imp1[k], reverse=True), 1)}
        rank2 = {f: r for r, f in enumerate(sorted(imp2.keys(), key=lambda k: imp2[k], reverse=True), 1)}

        x_ranks = [rank1.get(f, len(feature_names)) for f in feature_names]
        y_ranks = [rank2.get(f, len(feature_names)) for f in feature_names]

        ax.scatter(x_ranks, y_ranks, alpha=0.6, c=COLORS[0], s=50, edgecolors='white')

        # Add diagonal
        max_rank = max(max(x_ranks), max(y_ranks))
        ax.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.5, label='Perfect agreement')

        # Annotate top movers
        rank_diff = {f: abs(rank1.get(f, 0) - rank2.get(f, 0)) for f in feature_names}
        top_movers = sorted(rank_diff.items(), key=lambda x: x[1], reverse=True)[:5]
        for f, _ in top_movers:
            ax.annotate(f, (rank1.get(f, 0), rank2.get(f, 0)), fontsize=8, alpha=0.8)

        ax.set_xlabel(f'{model_names[0]} Rank', fontsize=11)
        ax.set_ylabel(f'{model_names[1]} Rank', fontsize=11)
        ax.set_title('Feature Rank Agreement', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'importance_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_importance_heatmap(results, feature_names, output_dir):
    """Plot per-head importance heatmap for each model."""
    for name, data in results['models'].items():
        per_head = data.get('per_head_importances', {})
        if not per_head:
            continue

        heads = list(per_head.keys())
        # Top 20 features by average importance
        avg_imp = {}
        for f in feature_names:
            avg_imp[f] = np.mean([per_head[h].get(f, 0) for h in heads])
        top_features = sorted(avg_imp.keys(), key=lambda f: avg_imp[f], reverse=True)[:20]

        matrix = np.array([[per_head[h].get(f, 0) for h in heads] for f in top_features])

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(heads)))
        ax.set_xticklabels(heads, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)

        plt.colorbar(im, ax=ax, label='Importance (%)')
        ax.set_title(f'{name}: Per-Head Feature Importance', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'importance_heatmap_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def plot_divergence_metrics(results, output_dir):
    """Plot divergence metrics visualization."""
    if 'divergence' not in results or not results['divergence']:
        return

    pairs = list(results['divergence'].keys())
    n_pairs = len(pairs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Divergence metrics bar chart
    ax = axes[0, 0]
    metrics = ['cosine_similarity', 'spearman_correlation', 'kendall_tau', 'top5_overlap', 'top10_overlap']
    x = np.arange(len(metrics))
    width = 0.8 / n_pairs

    for i, pair in enumerate(pairs):
        values = [results['divergence'][pair].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=pair, color=COLORS[i % len(COLORS)])

    ax.set_xticks(x + width * (n_pairs - 1) / 2)
    ax.set_xticklabels(['Cosine\nSim', 'Spearman\nρ', 'Kendall\nτ', 'Top-5\nOverlap', 'Top-10\nOverlap'], fontsize=9)
    ax.set_ylabel('Score (higher = more similar)')
    ax.set_title('Similarity Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    ax.grid(axis='y', alpha=0.3)

    # 2. Distance metrics bar chart
    ax = axes[0, 1]
    dist_metrics = ['jensen_shannon_divergence', 'wasserstein_distance']
    x = np.arange(len(dist_metrics))

    for i, pair in enumerate(pairs):
        values = [results['divergence'][pair].get(m, 0) for m in dist_metrics]
        ax.bar(x + i * width, values, width, label=pair, color=COLORS[i % len(COLORS)])

    ax.set_xticks(x + width * (n_pairs - 1) / 2)
    ax.set_xticklabels(['Jensen-Shannon\nDivergence', 'Wasserstein\nDistance'], fontsize=9)
    ax.set_ylabel('Distance (lower = more similar)')
    ax.set_title('Distance Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 3. Largest importance shifts
    ax = axes[1, 0]
    if pairs:
        pair = pairs[0]
        shifts = results['divergence'][pair].get('largest_shifts', [])
        if shifts:
            features = [s[0] for s in shifts[:10]]
            values = [s[1] for s in shifts[:10]]
            colors = ['#E74C3C' if v > 0 else '#3498DB' for v in values]
            ax.barh(features, values, color=colors, alpha=0.8)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel('Importance Shift (%)')
            ax.set_title(f'Largest Importance Shifts ({pair})', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

    # 4. Largest rank changes
    ax = axes[1, 1]
    if pairs:
        pair = pairs[0]
        rank_changes = results['divergence'][pair].get('largest_rank_changes', [])
        if rank_changes:
            features = [r[0] for r in rank_changes[:10]]
            values = [r[1] for r in rank_changes[:10]]
            colors = ['#27AE60' if v > 0 else '#8E44AD' for v in values]
            ax.barh(features, values, color=colors, alpha=0.8)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel('Rank Change (positive = moved up)')
            ax.set_title(f'Largest Rank Changes ({pair})', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'divergence_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_output_stability(results, output_dir):
    """Plot output stability analysis."""
    if 'stability' not in results or not results['stability']:
        return

    pairs = list(results['stability'].keys())

    for pair in pairs:
        metrics = results['stability'][pair]
        heads = list(metrics.keys())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Correlation per head
        ax = axes[0, 0]
        corrs = [metrics[h]['correlation'] for h in heads]
        colors = ['#27AE60' if c > 0.9 else '#F39C12' if c > 0.7 else '#E74C3C' for c in corrs]
        ax.bar(heads, corrs, color=colors, alpha=0.8)
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5)
        ax.set_ylabel('Correlation')
        ax.set_title('Output Correlation per Head', fontsize=11, fontweight='bold')
        ax.set_xticklabels(heads, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)

        # 2. MSD per head
        ax = axes[0, 1]
        msds = [metrics[h]['msd'] for h in heads]
        ax.bar(heads, msds, color=COLORS[1], alpha=0.8)
        ax.set_ylabel('Mean Squared Difference')
        ax.set_title('Prediction Energy Difference', fontsize=11, fontweight='bold')
        ax.set_xticklabels(heads, rotation=45, ha='right', fontsize=8)

        # 3. Variance ratio
        ax = axes[0, 2]
        var_ratios = [metrics[h]['variance_ratio'] for h in heads]
        colors = ['#27AE60' if 0.8 < v < 1.2 else '#E74C3C' for v in var_ratios]
        ax.bar(heads, var_ratios, color=colors, alpha=0.8)
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=1.2, color='green', linestyle='--', alpha=0.3)
        ax.set_ylabel('Variance Ratio')
        ax.set_title('Amplitude Stability', fontsize=11, fontweight='bold')
        ax.set_xticklabels(heads, rotation=45, ha='right', fontsize=8)

        # 4. KS statistic
        ax = axes[1, 0]
        ks_stats = [metrics[h]['ks_statistic'] for h in heads]
        ks_pvals = [metrics[h]['ks_pvalue'] for h in heads]
        colors = ['#E74C3C' if p < 0.01 else '#F39C12' if p < 0.05 else '#27AE60' for p in ks_pvals]
        ax.bar(heads, ks_stats, color=colors, alpha=0.8)
        ax.set_ylabel('KS Statistic')
        ax.set_title('Distribution Shift (KS Test)', fontsize=11, fontweight='bold')
        ax.set_xticklabels(heads, rotation=45, ha='right', fontsize=8)

        # 5. Wasserstein distance
        ax = axes[1, 1]
        wass = [metrics[h]['wasserstein'] for h in heads]
        ax.bar(heads, wass, color=COLORS[3], alpha=0.8)
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Earth Mover Distance', fontsize=11, fontweight='bold')
        ax.set_xticklabels(heads, rotation=45, ha='right', fontsize=8)

        # 6. Summary radar chart
        ax = axes[1, 2]
        # Normalize metrics to [0, 1] for radar
        norm_corr = np.mean(corrs)
        norm_var = 1 - np.mean([abs(v - 1) for v in var_ratios])
        norm_ks = 1 - np.mean(ks_stats)
        norm_wass = 1 / (1 + np.mean(wass))

        categories = ['Correlation', 'Variance\nStability', 'Distribution\nMatch', 'Wasserstein\nSimilarity']
        values = [norm_corr, max(0, norm_var), max(0, norm_ks), norm_wass]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[0])
        ax.fill(angles, values, alpha=0.25, color=COLORS[0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Stability Score', fontsize=11, fontweight='bold')

        plt.suptitle(f'Output Stability: {pair}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stability_{pair.replace(" ", "_").replace("/", "-")}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_gradient_saliency(results, feature_names, output_dir):
    """Plot gradient saliency comparison."""
    model_names = list(results['models'].keys())

    fig, axes = plt.subplots(len(model_names), 2, figsize=(14, 5 * len(model_names)))
    if len(model_names) == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(model_names):
        grads = results['models'][name].get('gradient_saliency')
        if grads is None:
            continue

        # Feature saliency
        ax = axes[i, 0]
        saliency = grads['feature_saliency']
        sorted_idx = np.argsort(saliency)[::-1][:20]
        sorted_features = [feature_names[j] for j in sorted_idx]
        sorted_values = saliency[sorted_idx]

        ax.barh(sorted_features[::-1], sorted_values[::-1], color=COLORS[i % len(COLORS)], alpha=0.8)
        ax.set_xlabel('Mean |Gradient|')
        ax.set_title(f'{name}: Feature Gradient Saliency', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Temporal saliency
        ax = axes[i, 1]
        temporal = grads['temporal_saliency']
        ax.plot(temporal, color=COLORS[i % len(COLORS)], linewidth=1.5)
        ax.fill_between(range(len(temporal)), temporal, alpha=0.3, color=COLORS[i % len(COLORS)])
        ax.set_xlabel('Timestep (t)')
        ax.set_ylabel('Mean |Gradient|')
        ax.set_title(f'{name}: Temporal Attention Pattern', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

        # Mark recent vs. distant past
        seq_len = len(temporal)
        ax.axvline(x=seq_len * 0.8, color='red', linestyle='--', alpha=0.5, label='Recent 20%')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_saliency.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_gradient_alignment(results, output_dir):
    """Plot gradient alignment between models."""
    if 'gradient_alignment' not in results or not results['gradient_alignment']:
        return

    pairs = list(results['gradient_alignment'].keys())

    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
    if len(pairs) == 1:
        axes = [axes]

    for i, pair in enumerate(pairs):
        align = results['gradient_alignment'][pair]
        if align is None:
            continue

        ax = axes[i]

        # Alignment scores
        scores = {
            'Feature\nAlignment': align['feature_alignment'],
            'Temporal\nAlignment': align['temporal_alignment'],
        }

        colors = ['#27AE60' if v > 0.8 else '#F39C12' if v > 0.5 else '#E74C3C' for v in scores.values()]
        ax.bar(scores.keys(), scores.values(), color=colors, alpha=0.8)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Cosine Alignment')
        ax.set_title(f'Gradient Alignment: {pair}', fontsize=11, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_alignment.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_fisher_comparison(results, output_dir):
    """Plot Fisher Information comparison."""
    if 'fisher_comparison' not in results or not results['fisher_comparison']:
        return

    pairs = list(results['fisher_comparison'].keys())

    for pair in pairs:
        ratios = results['fisher_comparison'][pair]
        layers = list(ratios.keys())
        values = list(ratios.values())

        # Cap extreme values for visualization
        values = [min(max(v, 0.1), 10) for v in values]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#E74C3C' if v > 2 or v < 0.5 else '#27AE60' for v in values]
        ax.bar(layers, values, color=colors, alpha=0.8)
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Equal sensitivity')
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

        ax.set_ylabel('Fisher Information Ratio (Model1 / Model2)')
        ax.set_xlabel('Layer')
        ax.set_title(f'Parameter Sensitivity Comparison: {pair}', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fisher_comparison_{pair.replace(" ", "_")}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_hessian_spectrum(results, output_dir):
    """Plot Hessian eigenvalue spectrum comparison."""
    model_names = list(results['models'].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Eigenvalue spectrum
    ax = axes[0]
    for i, name in enumerate(model_names):
        eigs = results['models'][name].get('hessian_eigenvalues')
        if eigs is not None:
            ax.semilogy(range(1, len(eigs) + 1), np.abs(eigs) + EPS, 'o-',
                       label=name, color=COLORS[i % len(COLORS)], markersize=5)

    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('|Eigenvalue| (log scale)')
    ax.set_title('Hessian Eigenvalue Spectrum', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Spectral comparison metrics
    ax = axes[1]
    if 'hessian_comparison' in results:
        for pair, metrics in results['hessian_comparison'].items():
            labels = ['Spectral\nRatio', 'Trace\nRatio', 'Eigenvalue\nCorrelation']
            values = [
                min(metrics['spectral_ratio'], 10),
                min(metrics['trace_ratio'], 10),
                metrics['eigenvalue_correlation']
            ]

            x = np.arange(len(labels))
            ax.bar(x, values, alpha=0.8, label=pair)

    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Ratio / Correlation')
    ax.set_title('Loss Landscape Curvature Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hessian_spectrum.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_comparison(results, feature_names, output_dir):
    """Plot SHAP value comparison."""
    model_names = list(results['models'].keys())

    # Global SHAP comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # 1. Top SHAP features bar chart
    ax = axes[0]
    shap_data = {}
    for name in model_names:
        shap = results['models'][name].get('shap_values', {})
        if shap:
            shap_data[name] = shap.get('mean_abs_shap', {})

    if shap_data:
        # Get top features from first model
        first_shap = list(shap_data.values())[0]
        top_features = sorted(first_shap.keys(), key=lambda k: first_shap[k], reverse=True)[:15]

        x = np.arange(len(top_features))
        width = 0.8 / len(shap_data)

        for i, (name, shap) in enumerate(shap_data.items()):
            values = [shap.get(f, 0) for f in top_features]
            ax.barh(x + i * width, values, width, label=name, color=COLORS[i % len(COLORS)], alpha=0.8)

        ax.set_yticks(x + width * (len(shap_data) - 1) / 2)
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title('SHAP Feature Attribution Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    # 2. Per-head SHAP heatmap (first model)
    ax = axes[1]
    if model_names:
        per_head = results['models'][model_names[0]].get('shap_values', {})
        if per_head and 'per_head_shap' in per_head:
            per_head_data = per_head['per_head_shap']
            heads = list(per_head_data.keys())
            # Top 15 features
            all_shap = per_head.get('mean_abs_shap', {})
            top_features = sorted(all_shap.keys(), key=lambda k: all_shap[k], reverse=True)[:15]

            matrix = np.array([[per_head_data[h][feature_names.index(f)] if f in feature_names else 0
                               for h in heads] for f in top_features])

            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(np.arange(len(heads)))
            ax.set_xticklabels(heads, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(np.arange(len(top_features)))
            ax.set_yticklabels(top_features, fontsize=9)
            plt.colorbar(im, ax=ax, label='Mean |SHAP|')
            ax.set_title(f'{model_names[0]}: Per-Head SHAP', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mutual_information(results, feature_names, output_dir):
    """Plot mutual information matrix."""
    mi_matrix = results.get('mutual_information')
    if mi_matrix is None:
        return

    # Select top features for readability
    mi_var = np.var(mi_matrix, axis=1)
    top_idx = np.argsort(mi_var)[::-1][:25]
    top_features = [feature_names[i] for i in top_idx]
    mi_subset = mi_matrix[np.ix_(top_idx, top_idx)]

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(mi_subset, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)

    plt.colorbar(im, ax=ax, label='Normalized Mutual Information')
    ax.set_title('Feature Mutual Information Matrix', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mutual_information.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_decision_trees(results, feature_cols, output_dir):
    """Plot surrogate decision trees."""
    for name, data in results['models'].items():
        tree = data.get('tree_object')
        if tree is None:
            continue

        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(tree, feature_names=feature_cols, filled=True, rounded=True,
                 fontsize=8, ax=ax, proportion=True)
        ax.set_title(f'{name}: Surrogate Decision Tree (ROI Head)\nR² = {data.get("tree_r2", 0):.3f}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'decision_tree_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def plot_output_distributions(results, output_dir):
    """Plot output distribution comparison per head."""
    model_names = list(results['models'].keys())

    for head_idx, head_name in enumerate(HEAD_NAMES[:10]):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Distribution histograms
        ax = axes[0]
        for i, name in enumerate(model_names):
            outputs = results['models'][name].get('raw_outputs')
            if outputs is not None and outputs.shape[1] > head_idx:
                ax.hist(outputs[:, head_idx], bins=50, alpha=0.5, label=name,
                       color=COLORS[i % len(COLORS)], density=True)

        ax.set_xlabel(f'{head_name} Output Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{head_name}: Output Distribution', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Q-Q plot (if 2 models)
        ax = axes[1]
        if len(model_names) >= 2:
            o1 = results['models'][model_names[0]].get('raw_outputs')
            o2 = results['models'][model_names[1]].get('raw_outputs')
            if o1 is not None and o2 is not None:
                n = min(len(o1), len(o2), 1000)
                q1 = np.sort(o1[:n, head_idx])
                q2 = np.sort(o2[:n, head_idx])

                ax.scatter(q1, q2, alpha=0.5, s=20, c=COLORS[0])
                lims = [min(q1.min(), q2.min()), max(q1.max(), q2.max())]
                ax.plot(lims, lims, 'k--', alpha=0.5)

                ax.set_xlabel(f'{model_names[0]}')
                ax.set_ylabel(f'{model_names[1]}')
                ax.set_title(f'{head_name}: Q-Q Plot', fontsize=11, fontweight='bold')
                ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'output_dist_{head_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def plot_pairwise_matrix(results, output_dir):
    """Plot pairwise comparison matrices for 3+ models."""
    model_names = list(results['models'].keys())
    n_models = len(model_names)

    if n_models < 2:
        return

    # Metrics to display in matrices
    metrics_config = [
        ('cosine_similarity', 'Cosine Similarity', 'RdYlGn', True),
        ('spearman_correlation', 'Spearman Correlation', 'RdYlGn', True),
        ('jensen_shannon_divergence', 'Jensen-Shannon Divergence', 'RdYlGn_r', False),
        ('wasserstein_distance', 'Wasserstein Distance', 'RdYlGn_r', False),
        ('top5_overlap', 'Top-5 Feature Overlap', 'RdYlGn', True),
        ('top10_overlap', 'Top-10 Feature Overlap', 'RdYlGn', True),
    ]

    # Build matrices
    matrices = {}
    for metric_key, _, _, _ in metrics_config:
        matrix = np.eye(n_models)
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    pair_name = f"{m1} vs {m2}"
                    if pair_name not in results['divergence']:
                        pair_name = f"{m2} vs {m1}"
                    if pair_name in results['divergence']:
                        val = results['divergence'][pair_name].get(metric_key, 0)
                        matrix[i, j] = val
                        matrix[j, i] = val
        matrices[metric_key] = matrix

    # Plot 2x3 grid of matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric_key, title, cmap, higher_better) in enumerate(metrics_config):
        ax = axes[idx]
        matrix = matrices[metric_key]

        mask = np.eye(n_models, dtype=bool)
        off_diag = matrix[~mask]
        if len(off_diag) > 0:
            vmin, vmax = off_diag.min(), off_diag.max()
            margin = (vmax - vmin) * 0.1 + 0.01
            vmin, vmax = vmin - margin, vmax + margin
        else:
            vmin, vmax = 0, 1

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    val = matrix[i, j]
                    text_color = 'white' if (higher_better and val < (vmin + vmax)/2) or \
                                           (not higher_better and val > (vmin + vmax)/2) else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                           fontsize=9, color=text_color, fontweight='bold')
                else:
                    ax.text(j, i, '—', ha='center', va='center', fontsize=12, color='gray')

        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels([n[:15] for n in model_names], rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([n[:15] for n in model_names], fontsize=9)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        indicator = "↑ better" if higher_better else "↓ better"
        ax.set_title(f'{title}\n({indicator})', fontsize=11, fontweight='bold')

    plt.suptitle('Pairwise Model Comparison Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_radar_comparison(results, output_dir):
    """Plot radar chart comparing all models on key metrics."""
    model_names = list(results['models'].keys())
    n_models = len(model_names)

    if n_models < 2:
        return {}

    model_scores = {}

    for name in model_names:
        data = results['models'][name]
        scores = {}

        # 1. Feature balance
        imp = data.get('importances', {})
        if imp:
            top_val = max(imp.values())
            scores['Feature\nBalance'] = max(0, 1 - top_val / 50)
        else:
            scores['Feature\nBalance'] = 0.5

        # 2. Interpretability
        tree_r2 = data.get('tree_r2', 0) or 0
        scores['Interpretability'] = min(tree_r2 * 2, 1.0)

        # 3. Output stability
        stats_data = data.get('output_stats', {})
        if stats_data:
            avg_skew = np.mean([abs(s.get('skew', 0)) for s in stats_data.values()])
            scores['Output\nStability'] = max(0, 1 - avg_skew / 5)
        else:
            scores['Output\nStability'] = 0.5

        # 4. Trading signal usage
        trading_features = {'rsi', 'adx', 'ivr', 'vix', 'atr', 'bb_squeeze', 'stoch_k', 'macd', 'cmf'}
        if imp:
            top10 = set(list(sorted(imp.keys(), key=lambda k: imp[k], reverse=True))[:10])
            overlap = len(trading_features & top10)
            scores['Trading\nSignals'] = overlap / 5
        else:
            scores['Trading\nSignals'] = 0

        # 5. Gradient focus
        grads = data.get('gradient_saliency')
        if grads and 'temporal_saliency' in grads:
            temporal = grads['temporal_saliency']
            if len(temporal) > 10:
                recent = np.mean(temporal[-int(len(temporal)*0.2):])
                distant = np.mean(temporal[:int(len(temporal)*0.2)])
                balance = 1 - abs(recent - distant) / (recent + distant + 1e-6)
                scores['Temporal\nBalance'] = balance
            else:
                scores['Temporal\nBalance'] = 0.5
        else:
            scores['Temporal\nBalance'] = 0.5

        # 6. Attribution consistency
        shap = data.get('shap_values')
        if shap and imp:
            shap_imp = shap.get('mean_abs_shap', {})
            if shap_imp:
                shap_top5 = set(sorted(shap_imp.keys(), key=lambda k: shap_imp[k], reverse=True)[:5])
                perm_top5 = set(sorted(imp.keys(), key=lambda k: imp[k], reverse=True)[:5])
                scores['Attribution\nConsistency'] = len(shap_top5 & perm_top5) / 5
            else:
                scores['Attribution\nConsistency'] = 0.5
        else:
            scores['Attribution\nConsistency'] = 0.5

        model_scores[name] = scores

    # Create radar chart
    categories = list(list(model_scores.values())[0].keys())
    n_cats = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (name, scores) in enumerate(model_scores.items()):
        values = [scores[cat] for cat in categories]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=COLORS[i % len(COLORS)])
        ax.fill(angles, values, alpha=0.15, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.title('Model Quality Radar Comparison', fontsize=14, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return model_scores


def plot_ranking_summary(results, output_dir):
    """Plot overall model ranking summary."""
    model_names = list(results['models'].keys())
    n_models = len(model_names)

    if n_models < 2:
        return []

    rankings = {}

    for name in model_names:
        data = results['models'][name]
        score = 0
        breakdown = {}

        n_pros = len(data.get('pros', []))
        n_cons = len(data.get('cons', []))
        breakdown['Pros'] = n_pros * 2
        breakdown['Cons'] = -n_cons
        score += n_pros * 2 - n_cons

        tree_r2 = data.get('tree_r2', 0) or 0
        if tree_r2 > 0.4:
            breakdown['Interpretability'] = 3
            score += 3
        elif tree_r2 > 0.25:
            breakdown['Interpretability'] = 1
            score += 1
        else:
            breakdown['Interpretability'] = 0

        imp = data.get('importances', {})
        if imp:
            top = max(imp.values())
            if top < 20:
                breakdown['Feature Balance'] = 2
                score += 2
            elif top < 30:
                breakdown['Feature Balance'] = 1
                score += 1
            else:
                breakdown['Feature Balance'] = 0

        similarities = []
        for other in model_names:
            if other != name:
                pair1 = f"{name} vs {other}"
                pair2 = f"{other} vs {name}"
                pair = pair1 if pair1 in results.get('divergence', {}) else pair2
                if pair in results.get('divergence', {}):
                    sim = results['divergence'][pair].get('cosine_similarity', 0)
                    similarities.append(sim)

        if similarities:
            avg_sim = np.mean(similarities)
            if avg_sim > 0.8:
                breakdown['Consistency'] = 2
                score += 2
            elif avg_sim > 0.6:
                breakdown['Consistency'] = 1
                score += 1
            else:
                breakdown['Consistency'] = 0

        rankings[name] = {'total': score, 'breakdown': breakdown}

    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1]['total'], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Total scores bar chart
    ax = axes[0]
    names = [r[0] for r in sorted_rankings]
    scores = [r[1]['total'] for r in sorted_rankings]
    colors = [COLORS[model_names.index(n) % len(COLORS)] for n in names]

    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], alpha=0.8)
    ax.set_xlabel('Total Score', fontsize=11)
    ax.set_title('Overall Model Rankings', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
               f'{score}', va='center', fontsize=10, fontweight='bold')

    # 2. Score breakdown stacked bar
    ax = axes[1]
    categories = ['Pros', 'Interpretability', 'Feature Balance', 'Consistency']

    x = np.arange(len(names))
    bottom = np.zeros(len(names))

    color_map = {'Pros': '#27AE60', 'Interpretability': '#3498DB',
                 'Feature Balance': '#9B59B6', 'Consistency': '#F39C12', 'Cons': '#E74C3C'}

    for cat in categories:
        values = [rankings[n]['breakdown'].get(cat, 0) for n in names]
        ax.bar(x, values, bottom=bottom, label=cat, color=color_map.get(cat, 'gray'), alpha=0.8)
        bottom += np.array(values)

    cons_values = [rankings[n]['breakdown'].get('Cons', 0) for n in names]
    ax.bar(x, cons_values, bottom=np.zeros(len(names)), label='Cons', color=color_map['Cons'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([n[:15] for n in names], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score Contribution', fontsize=11)
    ax.set_title('Score Breakdown by Category', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ranking_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return sorted_rankings
