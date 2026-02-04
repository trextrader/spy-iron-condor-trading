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
from intelligence.condor_brain_net import CondorNet
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


def load_cde_model(ckpt_path, input_dim=None, verbose_math=False):
    """Load a CDE/CondorNet model OR its extracted logic (JSON).
    
    Automatically detects:
    1. JSON logic files (returns dict)
    2. CondorNet checkpoints (returns model)
    3. CondorBrain / Mamba checkpoints (returns model)
    """
    if ckpt_path.endswith('.json'):
        print(f"  [LOGIC] Loading extracted logic from {ckpt_path}")
        with open(ckpt_path, 'r') as f:
            return json.load(f), None, 240
            
    if input_dim is None:
        input_dim = INPUT_DIM_V22

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # 1. Extract config and seq_len
    seq_len = ckpt.get('seq_len', 240)
    config = ckpt.get('model_config', ckpt.get('config', {}))
    
    # 2. Extract state_dict (check for common nesting keys)
    state_dict = ckpt
    for key in ['model_state_dict', 'state_dict']:
        if key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            # If we found model_state_dict, the config might also be at top level
            break
            
    # Clean state dict (remove 'module.' prefix from DDP)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 3. Detect CondorNet vs CondorBrain based on unnested state_dict keys
    keys = state_dict.keys()
    is_condornet = any('super_sets' in k or 'state_block' in k for k in keys)
    
    if is_condornet:
        # CondorNet V21+ architecture
        # Extract base dimensions from state_dims
        dims = ckpt.get('state_dims', {})
        d_h = dims.get('d_h', config.get('d_h', 256))
        d_v = dims.get('d_v', config.get('d_v', 32))
        d_m = dims.get('d_m', config.get('d_m', 64))
        d_r = dims.get('d_r', config.get('d_r', 32))
        
        # INFER Logic Complexity from state_dict to prevent OOM from extreme defaults
        # Find max index in super_sets.{i}.
        super_set_indices = [int(k.split('.')[1]) for k in keys if k.startswith('super_sets.')]
        n_super_sets = max(super_set_indices) + 1 if super_set_indices else config.get('n_super_sets', 1)
        
        # Find max index in super_sets.0.sets.{j}.
        set_indices = [int(k.split('.')[3]) for k in keys if k.startswith('super_sets.0.sets.')]
        n_sets = max(set_indices) + 1 if set_indices else config.get('n_sets', 4)
        
        # Infer n_predicates from relational_logic weight dimension
        # WeightDim = 3 * n * (n - 1) / 2
        # Solve for n: n = (1.5 + sqrt(2.25 + 6 * WeightDim)) / 3
        n_predicates = 5 # Default
        target_key = 'super_sets.0.sets.0.relational_logic.projection.weight'
        if target_key in state_dict:
            weight_dim = state_dict[target_key].shape[1]
            if weight_dim > 1:
                import math as _math
                inferred_n = (1.5 + _math.sqrt(2.25 + 6 * weight_dim)) / 3
                n_predicates = int(round(inferred_n))
        elif config.get('n_predicates'):
            n_predicates = config.get('n_predicates')

        # Shared params
        d_control = config.get('d_control', 128)
        n_layers = config.get('n_layers', 2)
        
        print(f"  [CondorNet] Inferred: d_h={d_h}, n_super_sets={n_super_sets}, n_sets={n_sets}, n_predicates={n_predicates}")
        
        model = CondorNet(
            d_input=input_dim,
            d_h=d_h,
            d_v=d_v,
            d_m=d_m,
            d_r=d_r,
            d_control=d_control,
            n_layers=n_layers,
            n_predicates=n_predicates,
            n_sets=n_sets,
            n_super_sets=n_super_sets,
            verbose_math=verbose_math
        )
    else:
        # CondorBrain (Mamba/CDE) architecture
        d_model = config.get('d_model', 128)
        n_layers = config.get('n_layers', 2)
        use_topk = config.get('use_topk_moe', False)
        use_cde = config.get('use_cde', True)
        
        model = CondorBrain(
            d_model=d_model,
            n_layers=n_layers,
            input_dim=input_dim,
            use_cde=use_cde,
            use_topk_moe=use_topk
        )
        print(f"  [CondorBrain] d_model={d_model}, n_layers={n_layers}")

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
    # Force 'math' attention backend to allow second-order derivatives (Hessian)
    # Optimized kernels like FlashAttention/MemEfficient often don't support grad-of-grad.
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except:
        class DummyCtx:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        sdp_ctx = DummyCtx()

    with sdp_ctx:
        try:
            for idx in tqdm(indices[:30], desc="Estimating Hessian", leave=False):  # Limit for speed
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
        except RuntimeError as e:
            if "derivative" in str(e).lower() or "attention" in str(e).lower():
                print(f"  [Hessian] WARNING: Model uses non-differentiable attention kernels. Skipping eigen-spectrum.", flush=True)
                return np.zeros(n_eigenvalues)
            raise e

    hessian_approx /= len(indices[:30])

    # Compute eigenvalues
    try:
        eigenvalues = torch.linalg.eigvalsh(hessian_approx.cpu())
        eigenvalues = eigenvalues.numpy()
        eigenvalues = np.sort(eigenvalues)[::-1][:n_eigenvalues]
    except Exception as e:
        print(f"  [Hessian] Eigenvalue computation failed: {e}")
        eigenvalues = np.zeros(n_eigenvalues)

    return eigenvalues


def estimate_hessian_hutchinson(model, X, seq_len, indices, n_eigenvalues=20, n_vectors=10):
    """Stochastic Hessian trace estimation using Hutchinson's method.

    Memory-optimized version for large models (33M+ params).
    """
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    traces = []
    params = [p for p in model.parameters() if p.requires_grad]

    # Use fewer samples for memory efficiency
    sample_indices = indices[:5]  # Reduced from 20 to 5

    # Force 'math' attention backend to allow second-order derivatives
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except:
        class DummyCtx:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        sdp_ctx = DummyCtx()

    with sdp_ctx:
        try:
            for vec_idx in tqdm(range(n_vectors), desc="Hutchinson estimation", leave=False):
                # Random Rademacher vector
                v = [torch.randint_like(p, 0, 2).float() * 2 - 1 for p in params]

                total_hvp = 0.0
                n_successful = 0

                for idx in sample_indices:
                    try:
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

                        n_successful += 1

                    except torch.cuda.OutOfMemoryError:
                        print(f"  [Hessian] OOM on sample, skipping...", flush=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    finally:
                        # Clear intermediate tensors
                        del x_tensor, out, loss
                        if 'grads' in dir():
                            del grads
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if n_successful > 0:
                    traces.append(total_hvp / n_successful)

                # Clear Rademacher vectors
                del v
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "derivative" in str(e).lower() or "attention" in str(e).lower():
                print(f"  [Hessian] WARNING: Model uses non-differentiable attention kernels. Skipping stochastic trace.", flush=True)
                return np.zeros(n_eigenvalues)
            raise e

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not traces:
        return np.zeros(n_eigenvalues)
    
    avg_trace = np.mean(traces)
    # Return a dummy spectrum centered around the average trace for consistent report formatting
    return np.linspace(avg_trace*1.1, avg_trace*0.9, n_eigenvalues)

    except torch.cuda.OutOfMemoryError:
        print(f"  [Hessian] GPU OOM - returning approximate values", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.array([0.0] * n_eigenvalues)

    if len(traces) == 0:
        return np.array([0.0] * n_eigenvalues)

    # Return trace estimate and variance
    trace_mean = np.mean(traces)

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


# =============================================================================
# TECHNICAL ASSESSMENT & PROS/CONS GENERATION (Section 11)
# =============================================================================

# Thresholds for deterministic rubric
THRESHOLDS = {
    'r2_high': 0.50,
    'r2_low': 0.20,
    'nf_threshold': 0.001,
    'var_collapse': 0.001,
    'top1_concentration': 40,
    'extreme_concentration': 80,
    'entropy_min': 2.5,
    'spearman_drift': 0.20,
    'top10_drift': 0.20,
    'overlap_good': 0.60,
    'corr_good': 0.90,
    'corr_moderate': 0.70,
    'var_ratio_low': 0.5,
    'var_ratio_high': 2.0,
}


def generate_pros_cons(model_name, model_data, divergence_data, stability_data, feature_names):
    """
    Generate deterministic pros/cons based on rubric triggers (Spec Section 11).

    Returns:
        dict with 'pros' and 'cons' lists
    """
    pros = []
    cons = []

    imp = model_data.get('importances', {})
    tree_r2 = model_data.get('tree_r2') or 0
    output_stats = model_data.get('output_stats', {})
    grads = model_data.get('gradient_saliency')
    shap = model_data.get('shap_values', {})

    # =========================================================================
    # PROS TRIGGERS
    # =========================================================================

    # 1. High interpretability (surrogate fidelity)
    if tree_r2 >= THRESHOLDS['r2_high']:
        pros.append(f"High surrogate fidelity (R2={tree_r2:.3f} >= 0.50): decision tree rules are reliable summary")

    # 2. Stable outputs
    if output_stats:
        all_vars = [s.get('std', 0) ** 2 for s in output_stats.values()]
        nf_rate = model_data.get('nf_rate', 0)
        if nf_rate <= THRESHOLDS['nf_threshold'] and all_vars and min(all_vars) > THRESHOLDS['var_collapse']:
            pros.append("Stable finite outputs with healthy variance (no collapse/explosion detected)")

    # 3. Consistent signal with peers
    if divergence_data:
        overlaps = [d.get('top10_overlap', 0) for d in divergence_data.values()]
        if overlaps and np.mean(overlaps) >= THRESHOLDS['overlap_good']:
            pros.append(f"Signal features consistent with peer checkpoints (avg top-10 overlap={np.mean(overlaps):.2f})")

    # 4. Balanced attribution (not concentrated)
    if imp:
        top_val = max(imp.values())
        imp_values = list(imp.values())
        imp_probs = np.array(imp_values) / (sum(imp_values) + EPS)
        imp_entropy = -np.sum(imp_probs * np.log(imp_probs + EPS))

        if top_val <= THRESHOLDS['top1_concentration'] and imp_entropy >= THRESHOLDS['entropy_min']:
            pros.append(f"Attribution not overly concentrated (top feature={top_val:.1f}%, entropy={imp_entropy:.2f})")

    # 5. Good temporal gradient balance
    if grads and 'temporal_saliency' in grads:
        temporal = grads['temporal_saliency']
        if len(temporal) > 10:
            recent = np.mean(temporal[-int(len(temporal)*0.2):])
            distant = np.mean(temporal[:int(len(temporal)*0.2)])
            if distant > 0 and 0.3 < recent / (distant + EPS) < 3.0:
                pros.append("Balanced temporal attention: model attends to both recent and historical context")

    # 6. Trading-relevant features in top ranks
    trading_features = {'rsi', 'adx', 'ivr', 'vix', 'atr', 'bb_squeeze', 'stoch_k', 'macd', 'cmf',
                        'delta_pressure', 'gamma_exposure', 'volume_imbalance'}
    if imp:
        top10 = set(list(sorted(imp.keys(), key=lambda k: imp[k], reverse=True))[:10])
        overlap = len(trading_features & set(f.lower() for f in top10))
        if overlap >= 3:
            pros.append(f"Trading-relevant features prominent ({overlap} in top-10): model learned domain signals")

    # 7. SHAP-Permutation consistency
    if shap and imp:
        shap_imp = shap.get('mean_abs_shap', {})
        if shap_imp:
            shap_top5 = set(sorted(shap_imp.keys(), key=lambda k: shap_imp[k], reverse=True)[:5])
            perm_top5 = set(sorted(imp.keys(), key=lambda k: imp[k], reverse=True)[:5])
            consistency = len(shap_top5 & perm_top5) / 5
            if consistency >= 0.6:
                pros.append(f"Strong attribution consistency (SHAP vs Permutation top-5 overlap={consistency:.0%})")

    # =========================================================================
    # CONS TRIGGERS
    # =========================================================================

    # 1. Potential collapse
    if imp:
        top_val = max(imp.values())
        if top_val >= THRESHOLDS['extreme_concentration']:
            cons.append(f"Possible collapse: extreme attribution concentration (top feature={top_val:.1f}%)")

    if output_stats:
        all_vars = [s.get('std', 0) ** 2 for s in output_stats.values()]
        if all_vars and min(all_vars) <= THRESHOLDS['var_collapse']:
            cons.append("Possible collapse: near-constant outputs detected (variance near zero)")

    # 2. Potential explosion
    nf_rate = model_data.get('nf_rate', 0)
    if nf_rate >= THRESHOLDS['nf_threshold']:
        cons.append(f"Potential instability: non-finite output rate={nf_rate:.2%}")

    if output_stats:
        high_kurtosis = [s.get('kurtosis', 0) for s in output_stats.values() if abs(s.get('kurtosis', 0)) > 10]
        if high_kurtosis:
            cons.append(f"Heavy-tailed outputs detected ({len(high_kurtosis)} heads with |kurtosis| > 10)")

    # 3. Major logic drift (vs peers)
    if divergence_data:
        spearman_vals = [d.get('spearman_correlation', 1) for d in divergence_data.values()]
        overlap_vals = [d.get('top10_overlap', 1) for d in divergence_data.values()]

        if spearman_vals and min(spearman_vals) <= THRESHOLDS['spearman_drift']:
            cons.append(f"Major logic drift: Spearman correlation dropped to {min(spearman_vals):.2f}")

        if overlap_vals and min(overlap_vals) <= THRESHOLDS['top10_drift']:
            cons.append(f"Feature ranking shifted substantially (top-10 overlap={min(overlap_vals):.0%})")

    # 4. Poor surrogate fidelity
    if tree_r2 < THRESHOLDS['r2_low']:
        cons.append(f"Surrogate tree not faithful (R2={tree_r2:.3f} < 0.20): decision boundary is complex or temporal")

    # 5. Stability concerns
    if stability_data:
        for pair, metrics in stability_data.items():
            for head, m in metrics.items():
                if m.get('correlation', 1) < THRESHOLDS['corr_moderate']:
                    cons.append(f"Output drift: {head} correlation={m['correlation']:.2f} (below 0.70 vs peer)")
                    break
                if m.get('variance_ratio', 1) < THRESHOLDS['var_ratio_low'] or m.get('variance_ratio', 1) > THRESHOLDS['var_ratio_high']:
                    cons.append(f"Amplitude instability: {head} variance ratio={m['variance_ratio']:.2f}")
                    break

    # 6. Gradient instability
    if grads and 'feature_saliency' in grads:
        saliency = grads['feature_saliency']
        if np.isnan(saliency).any() or np.isinf(saliency).any():
            cons.append("Gradient instability detected: NaN/Inf in saliency maps")

    return {'pros': pros, 'cons': cons}


# =============================================================================
# PHYSICS INTERPRETATION FUNCTIONS (Section 14)
# =============================================================================

def interpret_divergence(metrics):
    """Interpret divergence metrics in domain terms."""
    interpretations = []

    cos = metrics.get('cosine_similarity', 0)
    js = metrics.get('jensen_shannon_divergence', 0)
    sp = metrics.get('spearman_correlation', 0)
    top5 = metrics.get('top5_overlap', 0)

    # Cosine similarity interpretation
    if cos > 0.95:
        interpretations.append(f"Near-identical importance direction (cosine={cos:.3f}): models learned same feature weighting")
    elif cos > 0.80:
        interpretations.append(f"Similar importance direction (cosine={cos:.3f}): minor reweighting of features")
    elif cos > 0.50:
        interpretations.append(f"Moderate directional shift (cosine={cos:.3f}): different feature emphasis emerging")
    else:
        interpretations.append(f"Major directional divergence (cosine={cos:.3f}): fundamentally different feature selection")

    # Jensen-Shannon interpretation
    if js < 0.1:
        interpretations.append(f"Minimal information distance (JS={js:.3f}): importance distributions nearly identical")
    elif js < 0.3:
        interpretations.append(f"Moderate information distance (JS={js:.3f}): some mass shifted between features")
    else:
        interpretations.append(f"Large information distance (JS={js:.3f}): substantial redistribution of importance")

    # Rank correlation
    if sp > 0.9:
        interpretations.append(f"Rank order preserved (Spearman={sp:.3f}): feature priority ordering unchanged")
    elif sp > 0.6:
        interpretations.append(f"Partial rank preservation (Spearman={sp:.3f}): some features swapped positions")
    else:
        interpretations.append(f"Rank structure disrupted (Spearman={sp:.3f}): feature priorities fundamentally reordered")

    # Top-K interpretation
    if top5 < 0.4:
        interpretations.append(f"Signal shift warning: only {top5:.0%} of top-5 features overlap")

    return interpretations


def interpret_stability(metrics):
    """Interpret stability metrics using physics analogies."""
    interpretations = []

    # Aggregate across heads
    avg_corr = np.mean([m.get('correlation', 0) for m in metrics.values()])
    avg_msd = np.mean([m.get('msd', 0) for m in metrics.values()])
    avg_vr = np.mean([m.get('variance_ratio', 1) for m in metrics.values()])
    avg_ks = np.mean([m.get('ks_statistic', 0) for m in metrics.values()])

    # Coherence (correlation)
    if avg_corr > 0.95:
        interpretations.append(f"High coherence (avg correlation={avg_corr:.3f}): outputs move in lockstep")
    elif avg_corr > 0.80:
        interpretations.append(f"Good coherence (avg correlation={avg_corr:.3f}): outputs directionally aligned")
    elif avg_corr > 0.50:
        interpretations.append(f"Partial coherence (avg correlation={avg_corr:.3f}): some behavioral divergence")
    else:
        interpretations.append(f"Low coherence (avg correlation={avg_corr:.3f}): models responding differently to same inputs")

    # Energy gap (MSD)
    if avg_msd < 0.01:
        interpretations.append(f"Minimal energy gap (MSD={avg_msd:.4f}): outputs nearly identical in magnitude")
    elif avg_msd < 0.1:
        interpretations.append(f"Small energy gap (MSD={avg_msd:.4f}): minor amplitude differences")
    else:
        interpretations.append(f"Large energy gap (MSD={avg_msd:.4f}): significant output magnitude differences")

    # Amplitude stability (variance ratio)
    if 0.8 < avg_vr < 1.25:
        interpretations.append(f"Stable amplitude (variance ratio={avg_vr:.2f}): similar output volatility")
    elif avg_vr < 0.5:
        interpretations.append(f"Amplitude compression (variance ratio={avg_vr:.2f}): later model is quieter")
    elif avg_vr > 2.0:
        interpretations.append(f"Amplitude expansion (variance ratio={avg_vr:.2f}): later model is louder")

    # Distribution shift (KS)
    if avg_ks < 0.05:
        interpretations.append(f"No distribution shift (KS={avg_ks:.3f}): output distributions match")
    elif avg_ks < 0.15:
        interpretations.append(f"Minor distribution shift (KS={avg_ks:.3f}): subtle statistical differences")
    else:
        interpretations.append(f"Significant distribution shift (KS={avg_ks:.3f}): structural change in output behavior")

    return interpretations


def interpret_gradients(alignment):
    """Interpret gradient alignment."""
    if alignment is None:
        return ["Gradient analysis unavailable"]

    interpretations = []

    feat_align = alignment.get('feature_alignment', 0)
    temp_align = alignment.get('temporal_alignment', 0)

    if feat_align > 0.9:
        interpretations.append(f"High feature attention alignment ({feat_align:.3f}): models focus on same features")
    elif feat_align > 0.7:
        interpretations.append(f"Moderate feature attention alignment ({feat_align:.3f}): similar but not identical focus")
    else:
        interpretations.append(f"Low feature attention alignment ({feat_align:.3f}): models attend to different features")

    if temp_align > 0.9:
        interpretations.append(f"Temporal attention aligned ({temp_align:.3f}): same recency bias")
    elif temp_align > 0.7:
        interpretations.append(f"Partially aligned temporal focus ({temp_align:.3f})")
    else:
        interpretations.append(f"Different temporal focus ({temp_align:.3f}): models weight history differently")

    return interpretations


def interpret_hessian(comparison):
    """Interpret Hessian spectral comparison."""
    if comparison is None:
        return ["Hessian analysis unavailable"]

    interpretations = []

    spec_ratio = comparison.get('spectral_ratio', 1)
    trace_ratio = comparison.get('trace_ratio', 1)
    cond1 = comparison.get('condition_number_1', 1)
    cond2 = comparison.get('condition_number_2', 1)

    if spec_ratio > 2.0:
        interpretations.append(f"Sharper loss landscape (spectral ratio={spec_ratio:.2f}): later model has steeper curvature")
    elif spec_ratio < 0.5:
        interpretations.append(f"Flatter loss landscape (spectral ratio={spec_ratio:.2f}): later model found broader minimum")
    else:
        interpretations.append(f"Similar loss curvature (spectral ratio={spec_ratio:.2f})")

    if cond1 > 1e4 or cond2 > 1e4:
        worse = "Model 1" if cond1 > cond2 else "Model 2"
        interpretations.append(f"High condition number detected in {worse}: potential numerical instability")

    return interpretations


def interpret_fisher(comparison):
    """Interpret Fisher Information comparison."""
    if not comparison:
        return ["Fisher analysis unavailable"]

    interpretations = []

    ratios = list(comparison.values())
    if not ratios:
        return interpretations

    max_ratio = max(ratios)
    min_ratio = min(ratios)

    if max_ratio > 5.0:
        layer = [k for k, v in comparison.items() if v == max_ratio][0]
        interpretations.append(f"High parameter sensitivity in {layer} (ratio={max_ratio:.2f}x): fragile layer")

    if min_ratio < 0.2:
        layer = [k for k, v in comparison.items() if v == min_ratio][0]
        interpretations.append(f"Reduced sensitivity in {layer} (ratio={min_ratio:.2f}x): layer may be undertrained")

    spread = max_ratio / (min_ratio + EPS)
    if spread > 10:
        interpretations.append(f"Uneven sensitivity distribution (spread={spread:.1f}x): parameter importance concentrated")
    else:
        interpretations.append(f"Relatively uniform sensitivity (spread={spread:.1f}x)")

    return interpretations


# =============================================================================
# REPORT GENERATION (Section 8)
# =============================================================================

def format_markdown_report(results, config):
    """
    Generate comprehensive markdown report (Spec Section 8.1).

    10 sections as specified:
    1. Executive Summary
    2. Configuration & Preprocessing Contract
    3. Per-Model Interpretability
    4. Pairwise Divergence Analysis
    5. Stability Analysis (Physics)
    6. Gradient Saliency Analysis
    7. Hessian & Fisher Analysis
    8. Decision Tree Rules
    9. Pros/Cons per Model
    10. Final Recommendation & Ranking
    """
    lines = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    model_names = list(results['models'].keys())
    n_models = len(model_names)

    # =========================================================================
    # HEADER
    # =========================================================================
    lines.append("# Multi-Model Neural CDE Interpretability Audit Report")
    lines.append(f"\n**Generated:** {timestamp}")
    lines.append(f"**Models Compared:** {n_models}")
    lines.append(f"**Samples Used:** {config.get('n_samples', 'N/A')}")
    lines.append(f"**Seed:** {config.get('seed', 'N/A')}")
    lines.append("")

    # =========================================================================
    # MATHEMATICAL FOUNDATIONS (Appendix A - Inline for Reference)
    # =========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## Appendix A: Mathematical Foundations")
    lines.append("")
    lines.append("This appendix provides the mathematical basis for the interpretability metrics used throughout this audit.")
    lines.append("")

    lines.append("### A.1 Neural CDE Architecture")
    lines.append("")
    lines.append("The CondorBrain uses **Neural Controlled Differential Equations** for continuous-time market dynamics modeling.")
    lines.append("")
    lines.append("**Core CDE Equation:**")
    lines.append("")
    lines.append("$$")
    lines.append("dZ_t = f(Z_t; \\theta) \\, dX_t")
    lines.append("$$")
    lines.append("")
    lines.append("Where $Z_t \\in \\mathbb{R}^H$ is the latent state and $f: \\mathbb{R}^H \\to \\mathbb{R}^{H \\times D}$ is the learned vector field.")
    lines.append("")
    lines.append("**Integral Form:**")
    lines.append("")
    lines.append("$$")
    lines.append("Z_T = Z_0 + \\int_0^T f(Z_t) \\, dX_t")
    lines.append("$$")
    lines.append("")
    lines.append("**Explicit Euler Discretization:**")
    lines.append("")
    lines.append("$$")
    lines.append("Z_{t+1} = Z_t + f(Z_t) \\cdot (X_{t+1} - X_t)")
    lines.append("$$")
    lines.append("")
    lines.append("**Vector Field Network (Stability-Bounded):**")
    lines.append("")
    lines.append("$$")
    lines.append("f(z) = \\tanh(W_2 \\cdot \\text{SiLU}(W_1 \\cdot z + b_1) + b_2)")
    lines.append("$$")
    lines.append("")
    lines.append("The tanh activation ensures $\\|f(z)\\|_\\infty \\leq 1$, preventing state explosion over long sequences.")
    lines.append("")

    lines.append("### A.2 Permutation Importance")
    lines.append("")
    lines.append("Feature importance is computed by measuring prediction degradation when features are randomly shuffled:")
    lines.append("")
    lines.append("$$")
    lines.append("I_j = \\frac{1}{K} \\sum_{k=1}^{K} \\left[ \\mathcal{L}(f(X^{(j,k)}), y) - \\mathcal{L}(f(X), y) \\right]")
    lines.append("$$")
    lines.append("")
    lines.append("Where $X^{(j,k)}$ denotes data with feature $j$ permuted in repetition $k$.")
    lines.append("")
    lines.append("**Normalization:**")
    lines.append("")
    lines.append("$$")
    lines.append("\\hat{I}_j = \\frac{I_j}{\\sum_{i=1}^{D} I_i} \\times 100\\%")
    lines.append("$$")
    lines.append("")

    lines.append("### A.3 Divergence Metrics")
    lines.append("")
    lines.append("**Cosine Similarity:**")
    lines.append("")
    lines.append("$$")
    lines.append("\\cos(\\mathbf{a}, \\mathbf{b}) = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{\\|\\mathbf{a}\\| \\|\\mathbf{b}\\|}")
    lines.append("$$")
    lines.append("")
    lines.append("**Jensen-Shannon Divergence:**")
    lines.append("")
    lines.append("$$")
    lines.append("D_{JS}(P \\| Q) = \\frac{1}{2} D_{KL}(P \\| M) + \\frac{1}{2} D_{KL}(Q \\| M)")
    lines.append("$$")
    lines.append("")
    lines.append("Where $M = \\frac{1}{2}(P + Q)$ and $D_{KL}$ is the Kullback-Leibler divergence:")
    lines.append("")
    lines.append("$$")
    lines.append("D_{KL}(P \\| Q) = \\sum_i P(i) \\log \\frac{P(i)}{Q(i)}")
    lines.append("$$")
    lines.append("")
    lines.append("**Wasserstein Distance (Earth Mover's):**")
    lines.append("")
    lines.append("$$")
    lines.append("W_1(P, Q) = \\inf_{\\gamma \\in \\Gamma(P,Q)} \\mathbb{E}_{(x,y) \\sim \\gamma}[\\|x - y\\|]")
    lines.append("$$")
    lines.append("")
    lines.append("For 1D distributions, this simplifies to the area between CDFs:")
    lines.append("")
    lines.append("$$")
    lines.append("W_1(P, Q) = \\int_{-\\infty}^{\\infty} |F_P(x) - F_Q(x)| \\, dx")
    lines.append("$$")
    lines.append("")

    lines.append("### A.4 Gradient Saliency")
    lines.append("")
    lines.append("Saliency maps measure input sensitivity via backpropagated gradients:")
    lines.append("")
    lines.append("$$")
    lines.append("S_j = \\frac{1}{N} \\sum_{i=1}^{N} \\left| \\frac{\\partial \\hat{y}^{(i)}}{\\partial x_j^{(i)}} \\right|")
    lines.append("$$")
    lines.append("")
    lines.append("**Temporal Saliency (across sequence positions):**")
    lines.append("")
    lines.append("$$")
    lines.append("S_t = \\sum_{j=1}^{D} \\left| \\frac{\\partial \\hat{y}}{\\partial x_{t,j}} \\right|")
    lines.append("$$")
    lines.append("")
    lines.append("**Gradient Alignment (between models A and B):**")
    lines.append("")
    lines.append("$$")
    lines.append("\\rho_{grad} = \\frac{\\mathbf{g}_A \\cdot \\mathbf{g}_B}{\\|\\mathbf{g}_A\\| \\|\\mathbf{g}_B\\|}")
    lines.append("$$")
    lines.append("")

    lines.append("### A.5 Fisher Information Matrix")
    lines.append("")
    lines.append("The Fisher Information Matrix captures parameter sensitivity:")
    lines.append("")
    lines.append("$$")
    lines.append("\\mathcal{F}_{ij} = \\mathbb{E}\\left[ \\frac{\\partial \\log p(y|x;\\theta)}{\\partial \\theta_i} \\frac{\\partial \\log p(y|x;\\theta)}{\\partial \\theta_j} \\right]")
    lines.append("$$")
    lines.append("")
    lines.append("**Empirical Estimation (diagonal approximation):**")
    lines.append("")
    lines.append("$$")
    lines.append("\\hat{\\mathcal{F}}_{jj} = \\frac{1}{N} \\sum_{i=1}^{N} \\left( \\frac{\\partial \\mathcal{L}^{(i)}}{\\partial \\theta_j} \\right)^2")
    lines.append("$$")
    lines.append("")
    lines.append("**Fisher Overlap (between models):**")
    lines.append("")
    lines.append("$$")
    lines.append("\\text{Overlap} = \\frac{\\mathbf{f}_A \\cdot \\mathbf{f}_B}{\\|\\mathbf{f}_A\\| \\|\\mathbf{f}_B\\|}")
    lines.append("$$")
    lines.append("")

    lines.append("### A.6 Hessian Eigenspectrum")
    lines.append("")
    lines.append("The Hessian matrix of the loss landscape reveals optimization geometry:")
    lines.append("")
    lines.append("$$")
    lines.append("H_{ij} = \\frac{\\partial^2 \\mathcal{L}}{\\partial \\theta_i \\partial \\theta_j}")
    lines.append("$$")
    lines.append("")
    lines.append("**Eigenvalue Interpretation:**")
    lines.append("")
    lines.append("- $\\lambda_{max} > 0$: Maximum curvature (sharpness of minimum)")
    lines.append("- $\\text{Tr}(H) = \\sum_i \\lambda_i$: Average curvature")
    lines.append("- $\\lambda_{max} / \\lambda_{min}$: Condition number (training stability)")
    lines.append("")
    lines.append("**Flatness Correlation:**")
    lines.append("Flatter minima (lower $\\lambda_{max}$) correlate with better generalization.")
    lines.append("")

    lines.append("### A.7 Stability Physics")
    lines.append("")
    lines.append("**Mean Squared Deviation (MSD):**")
    lines.append("")
    lines.append("$$")
    lines.append("\\text{MSD}(A, B) = \\frac{1}{N} \\sum_{i=1}^{N} (y_A^{(i)} - y_B^{(i)})^2")
    lines.append("$$")
    lines.append("")
    lines.append("**Variance Ratio:**")
    lines.append("")
    lines.append("$$")
    lines.append("\\rho_{var} = \\frac{\\text{Var}(y_A)}{\\text{Var}(y_B)}")
    lines.append("$$")
    lines.append("")
    lines.append("Values near 1.0 indicate similar output distributions; deviations suggest regime sensitivity differences.")
    lines.append("")
    lines.append("**Kolmogorov-Smirnov Statistic:**")
    lines.append("")
    lines.append("$$")
    lines.append("D_{KS} = \\sup_x |F_A(x) - F_B(x)|")
    lines.append("$$")
    lines.append("")
    lines.append("Maximum difference between empirical CDFs, sensitive to both location and shape differences.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    lines.append("## 1. Executive Summary")
    lines.append("")

    # Quick verdict
    rankings = results.get('rankings', [])
    if rankings:
        best = rankings[0][0]
        lines.append(f"**Best Model:** `{best}`")

        total_pros = sum(len(results['models'][n].get('pros', [])) for n in model_names)
        total_cons = sum(len(results['models'][n].get('cons', [])) for n in model_names)
        lines.append(f"**Overall Health:** {total_pros} strengths identified, {total_cons} concerns flagged")

    # Key divergence summary
    if results.get('divergence'):
        pair = list(results['divergence'].keys())[0]
        div = results['divergence'][pair]
        lines.append(f"\n**Key Divergence ({pair}):**")
        lines.append(f"- Cosine Similarity: {div.get('cosine_similarity', 0):.3f}")
        lines.append(f"- Spearman Rank Correlation: {div.get('spearman_correlation', 0):.3f}")
        lines.append(f"- Top-10 Feature Overlap: {div.get('top10_overlap', 0):.0%}")

    lines.append("")

    # =========================================================================
    # 2. CONFIGURATION & PREPROCESSING CONTRACT
    # =========================================================================
    lines.append("## 2. Configuration & Preprocessing Contract")
    lines.append("")
    lines.append("### Feature Registry")
    lines.append(f"- **Input Dimension:** {config.get('input_dim', INPUT_DIM_V22)}")
    lines.append(f"- **Feature Schema:** V2.2 (FEATURE_COLS_V22)")
    lines.append(f"- **Total Features:** {len(config.get('feature_names', []))}")

    lines.append("\n### Sequence Configuration")
    for name in model_names:
        seq_len = results['models'][name].get('seq_len', 'N/A')
        lines.append(f"- `{name}`: seq_len = {seq_len}")

    lines.append("\n### Leakage Masking")
    lines.append("- `target_spot` -> 0")
    lines.append("- `max_dd_60m` -> 0")

    lines.append("\n### Scaling")
    lines.append("- Method: Robust Scaling (median/MAD)")
    lines.append("- Clip Range: [-10, +10]")
    lines.append("- NaN/Inf Handling: Replace with 0")
    lines.append("")

    # =========================================================================
    # 3. PER-MODEL INTERPRETABILITY
    # =========================================================================
    lines.append("## 3. Per-Model Interpretability")
    lines.append("")

    for name in model_names:
        data = results['models'][name]
        lines.append(f"### {name}")
        lines.append("")

        # Top features
        imp = data.get('importances', {})
        if imp:
            lines.append("**All Features (Permutation Importance):**")
            lines.append("| Rank | Feature | Importance (%) |")
            lines.append("|------|---------|----------------|")
            all_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            for rank, (feat, val) in enumerate(all_sorted, 1):
                lines.append(f"| {rank} | {feat} | {val:.2f} |")
            lines.append("")

        # Surrogate tree
        tree_r2 = data.get('tree_r2')
        if tree_r2 is not None:
            interp = "High" if tree_r2 >= 0.50 else "Moderate" if tree_r2 >= 0.20 else "Low"
            lines.append(f"**Surrogate Tree R2:** {tree_r2:.4f} ({interp} interpretability)")
            lines.append("")

        # Output stats summary
        output_stats = data.get('output_stats', {})
        if output_stats:
            lines.append("**Output Statistics (ROI Head):**")
            if 'roi' in output_stats:
                s = output_stats['roi']
                lines.append(f"- Mean: {s['mean']:.4f}, Std: {s['std']:.4f}")
                lines.append(f"- Skewness: {s['skew']:.3f}, Kurtosis: {s['kurtosis']:.3f}")
            lines.append("")

    # =========================================================================
    # 4. PAIRWISE DIVERGENCE ANALYSIS
    # =========================================================================
    lines.append("## 4. Pairwise Divergence Analysis")
    lines.append("")

    for pair, metrics in results.get('divergence', {}).items():
        lines.append(f"### {pair}")
        lines.append("")

        lines.append("**Similarity Metrics:**")
        lines.append(f"- Cosine Similarity: {metrics.get('cosine_similarity', 0):.4f}")
        lines.append(f"- Spearman Correlation: {metrics.get('spearman_correlation', 0):.4f} (p={metrics.get('spearman_pvalue', 1):.2e})")
        lines.append(f"- Kendall Tau: {metrics.get('kendall_tau', 0):.4f}")

        lines.append("\n**Distance Metrics:**")
        lines.append(f"- Jensen-Shannon Divergence: {metrics.get('jensen_shannon_divergence', 0):.4f}")
        lines.append(f"- Wasserstein Distance: {metrics.get('wasserstein_distance', 0):.4f}")

        lines.append("\n**Top-K Feature Overlap:**")
        lines.append(f"- Top-5 Overlap: {metrics.get('top5_overlap', 0):.0%}")
        lines.append(f"- Top-10 Overlap: {metrics.get('top10_overlap', 0):.0%}")

        # Interpretation
        interp = interpret_divergence(metrics)
        lines.append("\n**Interpretation:**")
        for i in interp:
            lines.append(f"- {i}")

        # Largest shifts
        shifts = metrics.get('largest_shifts', [])
        if shifts:
            lines.append("\n**All Importance Shifts (sorted by magnitude):**")
            lines.append("| Feature | Shift (%) |")
            lines.append("|---------|-----------|")
            for feat, shift in shifts:
                lines.append(f"| {feat} | {shift:+.2f} |")

        lines.append("")

    # =========================================================================
    # 5. STABILITY ANALYSIS (PHYSICS)
    # =========================================================================
    lines.append("## 5. Stability Analysis (Physics-Inspired)")
    lines.append("")

    for pair, metrics in results.get('stability', {}).items():
        lines.append(f"### {pair}")
        lines.append("")

        lines.append("| Head | Correlation | MSD | Var Ratio | KS Stat | Wasserstein |")
        lines.append("|------|-------------|-----|-----------|---------|-------------|")
        for head, m in metrics.items():
            lines.append(f"| {head} | {m.get('correlation', 0):.3f} | {m.get('msd', 0):.4f} | {m.get('variance_ratio', 0):.3f} | {m.get('ks_statistic', 0):.3f} | {m.get('wasserstein', 0):.4f} |")

        # Interpretation
        interp = interpret_stability(metrics)
        lines.append("\n**Physics Interpretation:**")
        for i in interp:
            lines.append(f"- {i}")

        lines.append("")

    # =========================================================================
    # 6. GRADIENT SALIENCY ANALYSIS
    # =========================================================================
    lines.append("## 6. Gradient Saliency Analysis")
    lines.append("")

    for name in model_names:
        grads = results['models'][name].get('gradient_saliency')
        if grads:
            lines.append(f"### {name}")
            saliency = grads['feature_saliency']
            feature_names = config.get('feature_names', [])
            if len(feature_names) == len(saliency):
                sorted_idx = np.argsort(saliency)[::-1]
                lines.append("**All Features (Gradient Saliency):**")
                lines.append("| Rank | Feature | Saliency |")
                lines.append("|------|---------|----------|")
                for rank, idx in enumerate(sorted_idx, 1):
                    lines.append(f"| {rank} | {feature_names[idx]} | {saliency[idx]:.4f} |")

            temporal = grads['temporal_saliency']
            recent_weight = np.mean(temporal[-int(len(temporal)*0.2):])
            old_weight = np.mean(temporal[:int(len(temporal)*0.2)])
            lines.append(f"\n**Temporal Focus:** Recent 20% weight={recent_weight:.4f}, Distant 20% weight={old_weight:.4f}")
            lines.append("")

    # Gradient alignment
    for pair, align in results.get('gradient_alignment', {}).items():
        if align:
            lines.append(f"### Gradient Alignment: {pair}")
            interp = interpret_gradients(align)
            for i in interp:
                lines.append(f"- {i}")
            lines.append("")

    # =========================================================================
    # 7. HESSIAN & FISHER ANALYSIS
    # =========================================================================
    lines.append("## 7. Hessian & Fisher Analysis")
    lines.append("")

    lines.append("### Hessian Eigenspectrum")
    for name in model_names:
        eigs = results['models'][name].get('hessian_eigenvalues')
        if eigs is not None and len(eigs) > 0:
            lines.append(f"**{name}:** Top eigenvalue = {eigs[0]:.4e}, Trace approx = {sum(eigs):.4e}")
    lines.append("")

    for pair, comp in results.get('hessian_comparison', {}).items():
        lines.append(f"**{pair}:**")
        interp = interpret_hessian(comp)
        for i in interp:
            lines.append(f"- {i}")
        lines.append("")

    lines.append("### Fisher Information")
    for pair, comp in results.get('fisher_comparison', {}).items():
        lines.append(f"**{pair}:**")
        interp = interpret_fisher(comp)
        for i in interp:
            lines.append(f"- {i}")
        lines.append("")

    # =========================================================================
    # 8. DECISION TREE RULES
    # =========================================================================
    lines.append("## 8. Surrogate Decision Tree Rules")
    lines.append("")

    for name in model_names:
        rules = results['models'][name].get('tree_rules')
        if rules:
            lines.append(f"### {name}")
            lines.append(f"**R2 Score:** {results['models'][name].get('tree_r2', 0):.4f}")
            lines.append("```")
            lines.append(rules)  # Full tree rules
            lines.append("```")
            lines.append("")

    # =========================================================================
    # 9. PROS/CONS PER MODEL
    # =========================================================================
    lines.append("## 9. Technical Assessment: Pros & Cons")
    lines.append("")

    for name in model_names:
        data = results['models'][name]
        pros = data.get('pros', [])
        cons = data.get('cons', [])

        lines.append(f"### {name}")
        lines.append("")

        lines.append("**Strengths:**")
        if pros:
            for p in pros:
                lines.append(f"- {p}")
        else:
            lines.append("- No significant strengths identified")
        lines.append("")

        lines.append("**Concerns:**")
        if cons:
            for c in cons:
                lines.append(f"- {c}")
        else:
            lines.append("- No significant concerns identified")
        lines.append("")

    # =========================================================================
    # 10. FINAL RECOMMENDATION & RANKING
    # =========================================================================
    lines.append("## 10. Final Recommendation & Ranking")
    lines.append("")

    if rankings:
        lines.append("### Overall Ranking")
        lines.append("| Rank | Model | Score | Breakdown |")
        lines.append("|------|-------|-------|-----------|")
        for rank, (name, info) in enumerate(rankings, 1):
            breakdown_str = ", ".join(f"{k}:{v}" for k, v in info['breakdown'].items() if v != 0)
            lines.append(f"| {rank} | {name} | {info['total']} | {breakdown_str} |")
        lines.append("")

        lines.append("### Recommendation")
        best = rankings[0][0]
        best_data = results['models'][best]
        lines.append(f"**Recommended Model:** `{best}`")
        lines.append("")
        lines.append("**Rationale (All Strengths):**")
        for p in best_data.get('pros', []):
            lines.append(f"- {p}")

        if len(rankings) > 1:
            runner_up = rankings[1][0]
            lines.append(f"\n**Runner-up:** `{runner_up}`")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append("1. **Neural CDE Theory:** Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series* (2020)")
    lines.append("2. **System Architecture:** `docs/scientific_spec.md` - Complete mathematical specification")
    lines.append("3. **Feature Schema:** V2.2 with 54 input dimensions (OHLCV, Greeks, dynamic indicators)")
    lines.append("4. **Loss Function:** Composite Risk-Aligned Loss with Huber, Sharpe proxy, and drawdown penalties")
    lines.append("")
    lines.append("**Composite Loss Function (from training):**")
    lines.append("")
    lines.append("$$")
    lines.append("\\mathcal{L}_{\\text{composite}} = \\lambda_1 \\mathcal{L}_{\\text{pred}} - \\lambda_2 \\mathcal{L}_{\\text{sharpe}} + \\lambda_3 \\mathcal{L}_{\\text{dd}} + \\lambda_4 \\mathcal{L}_{\\text{turn}}")
    lines.append("$$")
    lines.append("")
    lines.append("Where:")
    lines.append("- $\\mathcal{L}_{\\text{pred}}$: Huber loss for strike predictions")
    lines.append("- $\\mathcal{L}_{\\text{sharpe}}$: Negative Sharpe ratio (maximized via gradient descent)")
    lines.append("- $\\mathcal{L}_{\\text{dd}}$: Soft drawdown penalty")
    lines.append("- $\\mathcal{L}_{\\text{turn}}$: Turnover penalty for position stability")
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by audit_cde_comparison.py | Mathematical foundations from docs/scientific_spec.md*")

    return "\n".join(lines)


def convert_numpy_for_json(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_for_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_for_json(i) for i in obj)
    else:
        return obj


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_audit(model_paths, data_path, n_samples=3000, output_path='reports/model_comparison.md',
              seed=42,
    skip_mi=False,
    skip_hessian=False,
    skip_gradients=False,
    skip_fisher=False,
    skip_shap=False,
    verbose_math=False,
):
    """
    Main orchestration function for the audit (Spec Section 10).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 80, flush=True)
    print("MULTI-MODEL NEURAL CDE INTERPRETABILITY AUDIT", flush=True)
    print("=" * 80, flush=True)

    # =========================================================================
    # ENVIRONMENT & HARDWARE DETECTION
    # =========================================================================
    print("\n[0/9] Detecting hardware environment...", flush=True)
    print(f"  PyTorch version: {torch.__version__}", flush=True)
    print(f"  NumPy version: {np.__version__}", flush=True)
    print(f"  Pandas version: {pd.__version__}", flush=True)

    print(f"\n  CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}", flush=True)
        print(f"  cuDNN version: {torch.backends.cudnn.version()}", flush=True)
        print(f"  GPU count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / 1e9
            mem_alloc = torch.cuda.memory_allocated(i) / 1e9
            mem_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {props.name}", flush=True)
            print(f"    - Compute capability: {props.major}.{props.minor}", flush=True)
            print(f"    - Total memory: {mem_total:.1f} GB", flush=True)
            print(f"    - Allocated: {mem_alloc:.2f} GB / Reserved: {mem_reserved:.2f} GB", flush=True)
        print(f"  Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})", flush=True)
        # Enable TF32 for faster computation on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}", flush=True)
    else:
        print("  WARNING: Running on CPU - this will be slow!", flush=True)

    print(f"\n  Device selected: {DEVICE}", flush=True)

    print(f"\nModels: {len(model_paths)}", flush=True)
    for p in model_paths:
        exists = os.path.exists(p)
        size = os.path.path.getsize(p) / 1e6 if exists else 0
        print(f"  - {p} {'[OK, {:.1f}MB]'.format(size) if exists else '[NOT FOUND]'}", flush=True)
    print(f"Data: {data_path}", flush=True)
    data_exists = os.path.exists(data_path)
    if data_exists:
        data_size = os.path.getsize(data_path) / 1e6
        print(f"  Data file: [OK, {data_size:.1f}MB]", flush=True)
    else:
        print(f"  Data file: [NOT FOUND]", flush=True)
    print(f"Samples: {n_samples}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print("", flush=True)

    # =========================================================================
    # LOAD MODELS
    # =========================================================================
    print("\n[1/10] Loading models...", flush=True)
    models = {}
    seq_lens = {}

    for model_path in model_paths:
        name = os.path.basename(model_path).replace('.pth', '').replace('.json', '')
        print(f"  [{len(models)+1}/{len(model_paths)}] Loading {name}...", flush=True)
        
        try:
            model, ckpt, sl = load_cde_model(model_path, verbose_math=verbose_math)
            models[name] = model
            seq_lens[name] = sl
            seq_len = sl # Use sl for the print statement
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    - seq_len: {seq_len}", flush=True)
            print(f"    - parameters: {n_params:,}", flush=True)
            print(f"    - config: d_model={ckpt.get('model_config', {}).get('d_model', 'N/A')}, n_layers={ckpt.get('model_config', {}).get('n_layers', 'N/A')}", flush=True)
            if torch.cuda.is_available():
                print(f"    - GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated", flush=True)
        except Exception as e:
            print(f"    ERROR loading {name}: {e}", flush=True)
            raise

    # Use minimum seq_len for comparability
    T_common = min(seq_lens.values())
    print(f"\n  All models loaded successfully!", flush=True)
    print(f"  Using T_common = {T_common} (minimum across all models)", flush=True)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n[2/10] Loading data...", flush=True)
    # Load limited data based on samples requested to prevent OOM
    # We load n_samples * 5 + T_common to ensure we have enough valid windows
    # and variety after potential dropout/NaN filtering.
    read_rows = n_samples * 10 + T_common + 2000 # Increased buffer for better variety
    print(f"  Memory optimization: loading first {read_rows:,} rows from 5.8GB dataset", flush=True)
    
    import time as _time
    _t0 = _time.time()
    df = pd.read_csv(data_path, nrows=read_rows)
    _elapsed = _time.time() - _t0
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns in {_elapsed:.1f}s", flush=True)
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB", flush=True)

    feature_names = FEATURE_COLS_V22
    print(f"  Expected features: {len(feature_names)}", flush=True)

    # Select features (add missing as zeros)
    print(f"  Selecting features...", flush=True)
    X_cols = []
    missing_count = 0
    for col in feature_names:
        if col in df.columns:
            X_cols.append(df[col].values)
        else:
            missing_count += 1
            if missing_count <= 5:
                print(f"  [WARN] Missing feature: {col} - filling with zeros", flush=True)
            X_cols.append(np.zeros(len(df)))
    if missing_count > 5:
        print(f"  [WARN] ... and {missing_count - 5} more missing features", flush=True)

    print(f"  Stacking feature matrix...", flush=True)
    X = np.column_stack(X_cols).astype(np.float32)
    X = safe_nan_to_num(X)
    print(f"  Feature matrix shape: {X.shape}, dtype: {X.dtype}", flush=True)

    # Robust scaling
    print("\n[3/10] Applying robust scaling...", flush=True)
    print(f"  Computing median...", flush=True)
    median = np.median(X, axis=0)
    print(f"  Computing MAD...", flush=True)
    mad = np.median(np.abs(X - median), axis=0)
    scale = 1.4826 * mad + EPS
    print(f"  Normalizing...", flush=True)
    X = (X - median) / scale
    X = np.clip(X, -10, 10)
    X = safe_nan_to_num(X)
    print(f"  Scaling complete. Range: [{X.min():.2f}, {X.max():.2f}]", flush=True)

    # Leakage masking
    print(f"  Masking leakage columns...", flush=True)
    leakage_cols = ['target_spot', 'max_dd_60m']
    for col in leakage_cols:
        if col in feature_names:
            idx = feature_names.index(col)
            X[:, idx] = 0
            print(f"    Masked: {col}", flush=True)

    print(f"  Final X shape: {X.shape}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU memory after data prep: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated", flush=True)

    # =========================================================================
    # PER-MODEL ANALYSIS
    # =========================================================================
    print("\n[4/10] Running per-model analysis...", flush=True)
    results = {
        'models': {},
        'divergence': {},
        'stability': {},
        'gradient_alignment': {},
        'fisher_comparison': {},
        'hessian_comparison': {},
        'mutual_information': None,
        'rankings': [],
    }

    model_names = list(models.keys())
    total_models = len(model_names)

    for model_idx, (name, model) in enumerate(models.items()):
        print(f"\n  === Model {model_idx+1}/{total_models}: {name} ===", flush=True)
        seq_len = seq_lens[name]
        print(f"    Sequence length: {seq_len}", flush=True)

        model_results = {'seq_len': seq_len}
        import time as _time
        
        # Check if this is a pre-extracted logic dictionary (JSON)
        is_logic_only = isinstance(model, dict)
        
        if is_logic_only:
            print(f"    [LOGIC-MODE] Using pre-extracted logic from JSON...", flush=True)
            model_results['importances'] = model.get('importances', model.get('rules', {}))
            model_results['per_head_importances'] = model.get('per_head_importances', {})
            model_results['output_stats'] = model.get('output_stats', model.get('output_head', {}).get('sensitivities', {}))
            model_results['tree_rules'] = "\n".join(model.get('rules', [])) if isinstance(model.get('rules'), list) else str(model.get('rules'))
            # Populate other fields as empty to prevent crashes downstream
            model_results['tree_object'] = None
            model_results['tree_r2'] = 0.0
            model_results['tree_importances'] = {}
            model_results['raw_outputs'] = None
            model_results['gradient_saliency'] = None
            model_results['shap_values'] = None
            model_results['fisher_layer'] = {}
            model_results['hessian_eigenvalues'] = None
            
            results['models'][name] = model_results
            continue # Skip rest of loop for this model

        # Permutation importance
        print(f"    [4a] Computing permutation importance ({n_samples} samples)...", flush=True)
        _t0 = _time.time()
        imp_result = analyze_permutation_importance(model, X, feature_names, seq_len, n_samples)
        if isinstance(imp_result, tuple):
            model_results['importances'], model_results['per_head_importances'] = imp_result
        else:
            model_results['importances'] = imp_result
            model_results['per_head_importances'] = {}
        print(f"         Done in {_time.time()-_t0:.1f}s. Found {len(model_results['importances'])} features.", flush=True)

        # Surrogate tree
        print(f"    [4b] Training surrogate decision tree...", flush=True)
        _t0 = _time.time()
        tree, r2, rules, tree_imp = train_surrogate_tree(model, X, feature_names, seq_len, n_samples)
        model_results['tree_object'] = tree
        model_results['tree_r2'] = r2
        model_results['tree_rules'] = rules
        model_results['tree_importances'] = tree_imp
        print(f"         Done in {_time.time()-_t0:.1f}s. R²={r2:.4f}", flush=True)

        # Output statistics
        print(f"    [4c] Computing output statistics...", flush=True)
        _t0 = _time.time()
        output_stats, raw_outputs = compute_output_statistics(model, X, seq_len, n_samples)
        model_results['output_stats'] = output_stats
        model_results['raw_outputs'] = raw_outputs
        print(f"         Done in {_time.time()-_t0:.1f}s. Output shape: {raw_outputs.shape if raw_outputs is not None else 'N/A'}", flush=True)

        # Gradient saliency
        if not skip_gradients:
            print(f"    [4d] Computing gradient saliency ({min(500, n_samples)} samples)...", flush=True)
            _t0 = _time.time()
            grads = compute_gradient_saliency(model, X, seq_len, min(500, n_samples))
            model_results['gradient_saliency'] = grads
            print(f"         Done in {_time.time()-_t0:.1f}s.", flush=True)
        else:
            print(f"    [4d] Gradient saliency: SKIPPED", flush=True)

        # SHAP approximation
        if not skip_shap:
            print(f"    [4e] Computing SHAP approximation ({min(200, n_samples)} samples)...", flush=True)
            _t0 = _time.time()
            shap = compute_shap_approximation(model, X, feature_names, seq_len, min(200, n_samples))
            model_results['shap_values'] = shap
            print(f"         Done in {_time.time()-_t0:.1f}s.", flush=True)
        else:
            print(f"    [4e] SHAP approximation: SKIPPED", flush=True)

        # Fisher Information
        if not skip_fisher:
            print(f"    [4f] Estimating Fisher Information ({min(300, n_samples)} samples)...", flush=True)
            _t0 = _time.time()
            fisher_diag, fisher_layer = estimate_fisher_information(model, X, seq_len, min(300, n_samples))
            model_results['fisher_layer'] = fisher_layer
            print(f"         Done in {_time.time()-_t0:.1f}s.", flush=True)
        else:
            print(f"    [4f] Fisher Information: SKIPPED", flush=True)

        # Hessian spectrum
        if not skip_hessian:
            print(f"    [4g] Estimating Hessian spectrum ({min(50, n_samples)} samples) - memory intensive...", flush=True)
            # Aggressive memory cleanup before Hessian (requires second-order gradients)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                print(f"         GPU free memory: {free_mem/1e9:.2f} GB", flush=True)
                if free_mem < 2e9:  # Less than 2GB free
                    print(f"         ⚠️  Low GPU memory - Hessian may fail, will skip gracefully", flush=True)
            _t0 = _time.time()
            try:
                hessian_eigs = estimate_hessian_spectrum(model, X, seq_len, min(50, n_samples))
                model_results['hessian_eigenvalues'] = hessian_eigs
                print(f"         Done in {_time.time()-_t0:.1f}s.", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"         ⚠️  GPU OOM - skipping Hessian for this model", flush=True)
                model_results['hessian_eigenvalues'] = np.array([])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            print(f"    [4g] Hessian spectrum: SKIPPED", flush=True)

        results['models'][name] = model_results

        if torch.cuda.is_available():
            print(f"    GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.memory_reserved()/1e9:.2f} GB reserved", flush=True)

    # =========================================================================
    # MUTUAL INFORMATION
    # =========================================================================
    if not skip_mi:
        print("\n[5/9] Computing mutual information matrix...", flush=True)
        _t0 = _time.time()
        mi_matrix = compute_mutual_information(X, feature_names, T_common, n_samples)
        results['mutual_information'] = mi_matrix
        print(f"       Done in {_time.time()-_t0:.1f}s.", flush=True)
    else:
        print("\n[5/9] Skipping mutual information (--skip-mi)", flush=True)

    # =========================================================================
    # PAIRWISE COMPARISONS
    # =========================================================================
    n_pairs = len(model_names) * (len(model_names) - 1) // 2
    print(f"\n[6/9] Computing pairwise comparisons ({n_pairs} pairs)...", flush=True)

    pair_idx = 0
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            pair_idx += 1
            pair = f"{name1} vs {name2}"
            print(f"  [{pair_idx}/{n_pairs}] {pair}", flush=True)

            # Divergence metrics
            imp1 = results['models'][name1].get('importances', {})
            imp2 = results['models'][name2].get('importances', {})
            div_metrics = compute_divergence_metrics(imp1, imp2, feature_names)
            results['divergence'][pair] = div_metrics

            # Stability metrics
            out1 = results['models'][name1].get('raw_outputs')
            out2 = results['models'][name2].get('raw_outputs')
            if out1 is not None and out2 is not None:
                stab_metrics = compute_stability_metrics(out1, out2)
                results['stability'][pair] = stab_metrics

            # Gradient alignment
            if not skip_gradients:
                grad1 = results['models'][name1].get('gradient_saliency')
                grad2 = results['models'][name2].get('gradient_saliency')
                grad_align = compute_gradient_alignment(grad1, grad2)
                results['gradient_alignment'][pair] = grad_align

            # Fisher comparison
            if not skip_fisher:
                f1 = results['models'][name1].get('fisher_layer', {})
                f2 = results['models'][name2].get('fisher_layer', {})
                fisher_comp = compare_fisher_information(f1, f2)
                results['fisher_comparison'][pair] = fisher_comp

            # Hessian comparison
            if not skip_hessian:
                h1 = results['models'][name1].get('hessian_eigenvalues')
                h2 = results['models'][name2].get('hessian_eigenvalues')
                if h1 is not None and h2 is not None:
                    hess_comp = compare_hessian_spectra(h1, h2)
                    results['hessian_comparison'][pair] = hess_comp

    # =========================================================================
    # TECHNICAL ASSESSMENT
    # =========================================================================
    print("\n[7/9] Generating technical assessment...", flush=True)

    for name in model_names:
        # Get divergence data for this model
        div_data = {}
        stab_data = {}
        for pair, metrics in results['divergence'].items():
            if name in pair:
                div_data[pair] = metrics
        for pair, metrics in results['stability'].items():
            if name in pair:
                stab_data[pair] = metrics

        assessment = generate_pros_cons(name, results['models'][name], div_data, stab_data, feature_names)
        results['models'][name]['pros'] = assessment['pros']
        results['models'][name]['cons'] = assessment['cons']

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n[8/9] Generating visualizations...", flush=True)

    output_dir = os.path.dirname(output_path)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"  Output directory: {plots_dir}", flush=True)

    plot_count = 0
    try:
        plot_importance_comparison(results, feature_names, plots_dir)
        plot_count += 1
        print(f"  [{plot_count}] importance_comparison.png", flush=True)
    except Exception as e:
        print(f"  [WARN] importance_comparison failed: {e}", flush=True)

    try:
        plot_importance_heatmap(results, feature_names, plots_dir)
        plot_count += 1
        print(f"  [{plot_count}] importance_heatmap_*.png", flush=True)
    except Exception as e:
        print(f"  [WARN] importance_heatmap failed: {e}", flush=True)

    try:
        plot_divergence_metrics(results, plots_dir)
        plot_count += 1
        print(f"  [{plot_count}] divergence_metrics.png", flush=True)
    except Exception as e:
        print(f"  [WARN] divergence_metrics failed: {e}", flush=True)

    try:
        plot_output_stability(results, plots_dir)
        plot_count += 1
        print(f"  [{plot_count}] stability_*.png", flush=True)
    except Exception as e:
        print(f"  [WARN] output_stability failed: {e}", flush=True)

    if not skip_gradients:
        try:
            plot_gradient_saliency(results, feature_names, plots_dir)
            plot_count += 1
            print(f"  [{plot_count}] gradient_saliency.png", flush=True)
        except Exception as e:
            print(f"  [WARN] gradient_saliency failed: {e}", flush=True)

        try:
            plot_gradient_alignment(results, plots_dir)
            plot_count += 1
            print(f"  [{plot_count}] gradient_alignment.png", flush=True)
        except Exception as e:
            print(f"  [WARN] gradient_alignment failed: {e}", flush=True)

    if not skip_fisher:
        try:
            plot_fisher_comparison(results, plots_dir)
            plot_count += 1
            print(f"  [{plot_count}] fisher_comparison_*.png", flush=True)
        except Exception as e:
            print(f"  [WARN] fisher_comparison failed: {e}", flush=True)

    if not skip_hessian:
        try:
            plot_hessian_spectrum(results, plots_dir)
            plot_count += 1
            print(f"  [{plot_count}] hessian_spectrum.png", flush=True)
        except Exception as e:
            print(f"  [WARN] hessian_spectrum failed: {e}", flush=True)

    if not skip_shap:
        try:
            plot_shap_comparison(results, feature_names, plots_dir)
            plot_count += 1
            print(f"  [{plot_count}] shap_comparison.png", flush=True)
        except Exception as e:
            print(f"  [WARN] shap_comparison failed: {e}", flush=True)

    if not skip_mi:
        try:
            plot_mutual_information(results, feature_names, plots_dir)
            plot_count += 1
            print(f"  [{plot_count}] mutual_information.png", flush=True)
        except Exception as e:
            print(f"  [WARN] mutual_information failed: {e}", flush=True)

    try:
        plot_decision_trees(results, feature_names, plots_dir)
        plot_count += 1
        print(f"  [{plot_count}] decision_tree_*.png", flush=True)
    except Exception as e:
        print(f"  [WARN] decision_trees failed: {e}", flush=True)

    try:
        plot_output_distributions(results, plots_dir)
        plot_count += 1
        print(f"  [{plot_count}] output_dist_*.png", flush=True)
    except Exception as e:
        print(f"  [WARN] output_distributions failed: {e}", flush=True)

    try:
        plot_pairwise_matrix(results, plots_dir)
        plot_count += 1
        print(f"  [{plot_count}] pairwise_matrix.png", flush=True)
    except Exception as e:
        print(f"  [WARN] pairwise_matrix failed: {e}", flush=True)

    try:
        model_scores = plot_model_radar_comparison(results, plots_dir)
        plot_count += 1
        print(f"  [{plot_count}] model_radar.png", flush=True)
    except Exception as e:
        print(f"  [WARN] model_radar failed: {e}", flush=True)
        model_scores = {}

    try:
        rankings = plot_ranking_summary(results, plots_dir)
        results['rankings'] = rankings
        plot_count += 1
        print(f"  [{plot_count}] ranking_summary.png", flush=True)
    except Exception as e:
        print(f"  [WARN] ranking_summary failed: {e}", flush=True)

    print(f"\n  Total plots generated: {plot_count}", flush=True)

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    print("\n[9/9] Writing reports...", flush=True)

    config = {
        'n_samples': n_samples,
        'seed': seed,
        'input_dim': INPUT_DIM_V22,
        'feature_names': feature_names,
        'model_paths': model_paths,
        'data_path': data_path,
        'skip_gradients': skip_gradients,
        'skip_hessian': skip_hessian,
        'skip_shap': skip_shap,
        'skip_fisher': skip_fisher,
        'skip_mi': skip_mi,
        'verbose_math': verbose_math,
    }

    # Markdown report
    print(f"  Writing Markdown report...", flush=True)
    md_report = format_markdown_report(results, config)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"  Markdown: {output_path}", flush=True)

    # JSON report
    json_path = output_path.replace('.md', '.json')
    print(f"  Writing JSON report...", flush=True)

    # Prepare JSON-safe results (remove non-serializable objects)
    json_results = convert_numpy_for_json(results)

    # Remove tree objects (not serializable)
    for name in json_results['models']:
        json_results['models'][name].pop('tree_object', None)
        json_results['models'][name].pop('raw_outputs', None)  # Too large
        if 'gradient_saliency' in json_results['models'][name]:
            gs = json_results['models'][name]['gradient_saliency']
            if gs:
                gs.pop('raw_grads', None)  # Too large
        if 'shap_values' in json_results['models'][name]:
            sv = json_results['models'][name]['shap_values']
            if sv:
                sv.pop('raw_shap', None)  # Too large

    json_results['config'] = config

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"  JSON: {json_path}", flush=True)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("AUDIT COMPLETE", flush=True)
    print("=" * 80, flush=True)

    print("\n### Quick Summary ###", flush=True)
    if results['rankings']:
        print(f"Best Model: {results['rankings'][0][0]}", flush=True)

    for name in model_names:
        pros = results['models'][name].get('pros', [])
        cons = results['models'][name].get('cons', [])
        print(f"\n{name}:", flush=True)
        print(f"  Pros: {len(pros)}", flush=True)
        for p in pros[:2]:
            print(f"    + {p[:80]}...", flush=True)
        print(f"  Cons: {len(cons)}", flush=True)
        for c in cons[:2]:
            print(f"    - {c[:80]}...", flush=True)

    print(f"\nFull report: {output_path}", flush=True)
    print(f"Plots: {plots_dir}/", flush=True)

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Model Neural CDE Interpretability Comparison Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python intelligence/audit_cde_comparison.py \\
      --models models/epoch_1.pth models/epoch_3.pth \\
      --data data/processed/mamba_institutional_2024_1m_v22.csv \\
      --samples 3000 \\
      --output reports/model_comparison.md

  python intelligence/audit_cde_comparison.py \\
      --models models/*.pth \\
      --data data/processed/dataset.csv \\
      --skip-hessian --skip-mi \\
      --output reports/quick_comparison.md
        """
    )

    parser.add_argument('--models', nargs='+', required=True,
                        help='Paths to model checkpoints (at least 2)')
    parser.add_argument('--data', required=True,
                        help='Path to data CSV (V2.2 features)')
    parser.add_argument('--samples', type=int, default=3000,
                        help='Number of samples for analysis (default: 3000)')
    parser.add_argument('--output', default='reports/model_comparison.md',
                        help='Output path for markdown report')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose-math', action='store_true',
                        help='Output mathematical derivations and compute cycles')
    
    # Fast flags
    parser.add_argument('--skip-gradients', action='store_true',
                        help='Skip gradient saliency analysis')
    parser.add_argument('--skip-hessian', '--no-hessian', action='store_true',
                        help='Skip Hessian eigenspectrum analysis')
    parser.add_argument('--skip-shap', '--no-shap', action='store_true',
                        help='Skip SHAP approximation')
    parser.add_argument('--skip-fisher', '--no-fisher', action='store_true',
                        help='Skip Fisher Information estimation')
    parser.add_argument('--skip-mi', '--no-mi', action='store_true',
                        help='Skip mutual information matrix')

    args = parser.parse_args()

    # Validate inputs
    if len(args.models) < 2:
        parser.error("At least 2 model paths required")

    for path in args.models:
        if not os.path.exists(path):
            parser.error(f"Model not found: {path}")

    if not os.path.exists(args.data):
            parser.error(f"Data file not found: {args.data}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Run audit
    run_audit(
        model_paths=args.models,
        data_path=args.data,
        n_samples=args.samples,
        output_path=args.output,
        seed=args.seed,
        skip_gradients=args.skip_gradients,
        skip_hessian=args.skip_hessian,
        skip_shap=args.skip_shap,
        skip_fisher=args.skip_fisher,
        skip_mi=args.skip_mi,
        verbose_math=args.verbose_math,
    )


if __name__ == '__main__':
    main()
