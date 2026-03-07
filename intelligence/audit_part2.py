import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import warnings
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional

try:
    from tqdm import tqdm
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.metrics import r2_score
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    pass

def load_audit_data(csv_path: str, feature_names: List[str], max_samples: int = 3000, seq_len: int = 10) -> torch.Tensor:
    print(f"[AUDIT] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure all features exist, fill missing with 0
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0.0
            
    data = df[feature_names].values
    data = data[-max_samples:] if len(data) > max_samples else data
    
    # Create sequence tensor [B, T, F]
    N_samples = len(data) - seq_len + 1
    if N_samples <= 0:
        raise ValueError("Not enough data for sequence length.")
        
    seqs = []
    for i in range(N_samples):
        seqs.append(data[i:i+seq_len])
        
    X = torch.tensor(np.array(seqs), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[AUDIT] Built sequence tensor: {X.shape}")
    return X

class CondorRunner:
    """Wrapper to run model forward passes robustly."""
    def __init__(self, model, device, strategy_types, pivot_horizons):
        self.model = model
        self.device = device
        self.strategy_types = strategy_types
        self.pivot_horizons = pivot_horizons
        
    def _extract_dict(self, out) -> Dict[str, torch.Tensor]:
        res = {}
        for k in ['entry_signal', 'pop', 'ev', 'max_loss', 'var_95', 'cvar_95', 'position_size', 'spot_pred']:
            val = getattr(out, k, None)
            if val is not None:
                res[k] = val.squeeze(-1) if val.dim() > 1 else val
                
        s_logits = getattr(out, 'strategy_logits', None)
        if s_logits is not None:
            for i, st in enumerate(self.strategy_types):
                if i < s_logits.shape[-1]:
                    res[f'strategy_{st}'] = s_logits[..., i]
                    
        p_high = getattr(out, 'pivot_high_logits', None)
        if p_high is not None:
            for i, h in enumerate(self.pivot_horizons):
                if i < p_high.shape[-1]:
                    res[f'pivot_h{h}'] = p_high[..., i]
        return res

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            x_in = x.to(self.device)
            out = self.model.forward_compat(x_in)
            return self._extract_dict(out)

    def forward_with_grad(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        self.model.eval()
        x_in = x.to(self.device).requires_grad_(True)
        out = self.model.forward_compat(x_in)
        return self._extract_dict(out), x_in

def analyze_permutation_importance(runner: CondorRunner, X: torch.Tensor, feature_names: List[str], n_samples: int = 500) -> dict:
    print("[AUDIT] Running Permutation Importance...")
    X_sub = X[-n_samples:]
    base_out = runner.forward(X_sub)
    importance = {head: np.zeros(len(feature_names)) for head in base_out.keys()}
    
    for feat_idx in tqdm(range(len(feature_names)), desc="Permuting features", leave=False):
        X_perm = X_sub.clone()
        perm = torch.randperm(X_sub.shape[0])
        X_perm[:, :, feat_idx] = X_perm[perm, :, feat_idx]
        
        perm_out = runner.forward(X_perm)
        for head, base_val in base_out.items():
            diff = torch.abs(base_val - perm_out[head]).mean().item()
            importance[head][feat_idx] = diff
            
    results = {}
    for head, imps in importance.items():
        total = np.sum(imps) + 1e-8
        imps_norm = imps / total
        top_indices = np.argsort(imps_norm)[-10:][::-1]
        results[head] = [
            {"feature": feature_names[i], "importance": float(imps_norm[i])}
            for i in top_indices
        ]
    return results

def analyze_gradient_saliency(runner: CondorRunner, X: torch.Tensor, feature_names: List[str], n_samples: int = 100) -> dict:
    print("[AUDIT] Running Gradient Saliency...")
    X_sub = X[-n_samples:]
    out_dict, X_req = runner.forward_with_grad(X_sub)
    saliency = {}
    
    for head in tqdm(out_dict.keys(), desc="Computing gradients", leave=False):
        head_val = out_dict[head].sum()
        head_val.backward(retain_graph=True)
        grads = X_req.grad.abs().mean(dim=0).mean(dim=0)
        
        total = grads.sum().item() + 1e-8
        grads_norm = grads / total
        
        top_indices = torch.argsort(grads_norm, descending=True)[:10]
        saliency[head] = [
            {"feature": feature_names[i], "saliency": float(grads_norm[i].item())}
            for i in top_indices
        ]
        X_req.grad.zero_()
        
    return saliency

def analyze_output_statistics(runner: CondorRunner, X: torch.Tensor) -> dict:
    print("[AUDIT] Running Output Statistics...")
    out_dict = runner.forward(X)
    stats_dict = {}
    
    for head, val in out_dict.items():
        val_np = val.cpu().numpy()
        stats_dict[head] = {
            "mean": float(np.mean(val_np)),
            "std": float(np.std(val_np)),
            "skew": float(stats.skew(val_np.flatten())),
            "kurtosis": float(stats.kurtosis(val_np.flatten())),
            "min": float(np.min(val_np)),
            "max": float(np.max(val_np)),
        }
    return stats_dict

def analyze_surrogate_trees(runner: CondorRunner, X: torch.Tensor, feature_names: List[str]) -> dict:
    print("[AUDIT] Training Surrogate Decision Trees...")
    out_dict = runner.forward(X)
    X_np = X[:, -1, :].cpu().numpy()
    tree_results = {}
    
    for head, val in tqdm(out_dict.items(), desc="Fitting trees", leave=False):
        y_np = val.cpu().numpy().flatten()
        dt = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt.fit(X_np, y_np)
        
        preds = dt.predict(X_np)
        r2 = r2_score(y_np, preds)
        rules = export_text(dt, feature_names=feature_names, spacing=2, decimals=3)
        
        tree_results[head] = {
            "fidelity_r2": float(r2),
            "rules": rules
        }
    return tree_results

def analyze_mutual_information(runner: CondorRunner, X: torch.Tensor, feature_names: List[str]) -> dict:
    print("[AUDIT] Running Mutual Information...")
    out_dict = runner.forward(X)
    X_np = X[:, -1, :].cpu().numpy()
    mi_results = {}
    
    heads_to_run = ['pop', 'ev', 'strategy_abstain', 'strategy_iron_condor']
    for head in heads_to_run:
        if head in out_dict:
            y_np = out_dict[head].cpu().numpy().flatten()
            mi = mutual_info_regression(X_np, y_np, random_state=42)
            total = np.sum(mi) + 1e-8
            mi_norm = mi / total
            top_indices = np.argsort(mi_norm)[-10:][::-1]
            mi_results[head] = [
                {"feature": feature_names[i], "mi_score": float(mi_norm[i])}
                for i in top_indices
            ]
    return mi_results

def run_full_data_audit(model, csv_path: str, seq_len: int, max_samples: int, feature_names: List[str], strategy_types: List[str], pivot_horizons: List[int], device: torch.device) -> dict:
    X = load_audit_data(csv_path, feature_names, max_samples, seq_len)
    runner = CondorRunner(model, device, strategy_types, pivot_horizons)
    
    results = {}
    results["output_statistics"] = analyze_output_statistics(runner, X)
    results["permutation_importance"] = analyze_permutation_importance(runner, X, feature_names)
    results["gradient_saliency"] = analyze_gradient_saliency(runner, X, feature_names)
    results["mutual_information"] = analyze_mutual_information(runner, X, feature_names)
    results["surrogate_trees"] = analyze_surrogate_trees(runner, X, feature_names)
    
    return results
