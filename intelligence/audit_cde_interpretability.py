import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import sys
import os
from sklearn.tree import DecisionTreeRegressor, export_text
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.getcwd())

from intelligence.condor_brain import CondorBrain
from intelligence.canonical_feature_registry import FEATURE_COLS_V22

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 256

def load_cde_model(ckpt_path, input_dim=52):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    config = ckpt.get('config', {})
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
    
    # Strip 'module.' prefix if trained with DataParallel
    state_dict = ckpt['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def analyze_permutation_importance(model, dataset, feature_names, n_samples=1000):
    """
    Computes feature importance via Permutation Importance (Robust to Gradient issues).
    
    Method:
    1. Measure baseline error/output magnitude.
    2. For each feature:
       - Shuffle that feature across the batch (breaking its signal).
       - Measure the change in output (magnitude of deviation from baseline).
       - Higher deviation = Higher importance.
    """
    print(f"\n🔬 Running Permutation Importance on {n_samples} samples...")
    
    # Get a batch
    # Ensure n_samples does not exceed available sequences in dataset
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    batch_X = []
    
    # Collect batch
    for idx in indices:
        # data is raw (N, D), slice sequence manually
        x_seq = dataset[idx : idx+32] # SEQ_LEN is 32 globally
        batch_X.append(torch.tensor(x_seq, dtype=torch.float32))
        
    X_base = torch.stack(batch_X).to(DEVICE) # (B, T, D)
    
    X_base = torch.stack(batch_X).to(DEVICE) # (B, T, D)
    
    # Baseline forward pass
    model.eval()
    with torch.no_grad():
        base_out = model(X_base)
        
    # FILTER: Keep only samples with finite outputs
    # base_out: (B, Hidden)
    valid_mask = torch.isfinite(base_out).all(dim=1) # (B,)
    n_valid = valid_mask.sum().item()
    
    if n_valid == 0:
        print("⚠️ Warning: All baseline samples produced NaN outputs! Model may be unstable.")
        return {}
        
    if n_valid < n_samples:
        print(f"⚠️ Note: Filtered {n_samples - n_valid} NaN samples. using {n_valid} valid samples.")
        
    # Keep only valid
    X_base = X_base[valid_mask]
    base_out = base_out[valid_mask]
    
    # We use strict output magnitude or just the output tensor itself to compare
    # Let's track mean absolute change in the output vector
    
    importances = {}
    
    for i, fname in enumerate(tqdm(feature_names, desc="Perturbing features")):
        # Create perturbed batch
        X_perm = X_base.clone()
        
        # Shuffle feature i across batch dimension (only valid samples)
        perm_indices = torch.randperm(n_valid)
        X_perm[:, :, i] = X_perm[perm_indices, :, i]
        
        with torch.no_grad():
            perm_out = model(X_perm)
            
        # Measure impact: Mean Absolute Difference between base_out and perm_out
        # This captures how much the output *changes* when the feature is destroyed
        # Handle valid outputs only (perm_out might become NaN due to perturbation)
        diff_tensor = torch.abs(base_out - perm_out)
        
        # Mean over features (Hidden dim)
        diff_per_sample = diff_tensor.mean(dim=1)
        
        # Ignore new NaNs
        idx_valid_perm = torch.isfinite(diff_per_sample)
        if idx_valid_perm.sum() > 0:
            diff = diff_per_sample[idx_valid_perm].mean().item()
        else:
            diff = 0.0 # If perturbation causes 100% NaNs, effectively it broke the model (high impact?)
            # Or 0.0 to be safe
        
        importances[fname] = diff

    # Normalize to 0-100
    total_impact = sum(importances.values())
    if total_impact > 0:
        for k in importances:
            importances[k] = (importances[k] / total_impact) * 100.0
            
    # Sort
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🌟 CDE Feature Importance (Permutation Method):")
    print(f"{'Feature':<30} | {'Impact Score':<10}")
    print("-" * 45)
    for name, score in sorted_feats[:15]:
        print(f"{name:<30} | {score:6.2f}")
        
    return importances

def train_surrogate_tree(model, X, feature_cols, n_samples=5000):
    print(f"\n🌳 Training Surrogate Decision Tree on {n_samples} samples...")
    from sklearn.tree import DecisionTreeRegressor, export_text
    
    n_samples = min(n_samples, len(X) - SEQ_LEN)
    idxs = np.random.randint(0, len(X) - SEQ_LEN, size=n_samples)
    input_states = []
    targets = []
    
    with torch.no_grad():
        for idx in tqdm(idxs):
            seq = X[idx : idx+SEQ_LEN]
            last_step = seq[-1] # Interpret based on current state
            
            x_tensor = torch.tensor(seq, device=DEVICE).unsqueeze(0).float()
            # Predict
            with torch.no_grad():
                out = model(x_tensor)
                # CondorBrain forward returns (outputs, regime, horizon, features, ...)
                # Primary logic uses index 0 (regression outputs)
                if isinstance(out, tuple):
                    out = out[0]
                pred = out[0, 5].item() # index 5 is expected_roi in CondorBrain
            
            # CRITICAL: Skip if scalar prediction is NaN or Inf
            if not np.isfinite(pred):
                continue
                
            input_states.append(last_step)
            targets.append(pred)
            
    if not input_states:
        print("⚠️ Warning: No valid samples collected for surrogate tree (all NaNs).")
        return
            
    input_states = np.array(input_states)
    targets = np.array(targets)
    
    # Check if we have enough samples
    if len(targets) < 50:
         print(f"⚠️ Warning: Only {len(targets)} valid samples. Tree might be unstable.")
    
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50)
    tree.fit(input_states, targets)
    
    r2 = tree.score(input_states, targets)
    print(f"Surrogate Tree R2 Score: {r2:.3f}")
    
    rules = export_text(tree, feature_names=feature_cols)
    print("\n📜 Extracted Trading Rules (Surrogate):")
    print(rules)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading Data: {args.data}")
    df = pd.read_csv(args.data)
    feature_cols = FEATURE_COLS_V22
    for c in feature_cols:
        if c not in df.columns: df[c] = 0.0
            
    X = df[feature_cols].values.astype(np.float32)
    # Simple robust scaler (match training logic)
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad = np.maximum(mad, 1e-6)
    X = (X - median) / (1.4826 * mad)
    X = np.clip(X, -10.0, 10.0)
    
    # Mask leakage
    leakage = ['target_spot', 'max_dd_60m']
    for c in leakage:
        if c in feature_cols:
            idx = feature_cols.index(c)
            X[:, idx] = 0.0
            
    # 2. Load Model
    model = load_cde_model(args.model, input_dim=len(feature_cols))
    
    # 3. Analyze (Permutation Importance)
    analyze_permutation_importance(model, X, feature_cols, n_samples=args.samples)
    train_surrogate_tree(model, X, feature_cols, n_samples=args.samples * 5)

if __name__ == "__main__":
    main()
