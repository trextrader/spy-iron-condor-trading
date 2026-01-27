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

from intelligence.models.neural_cde import NeuralCDE
from intelligence.canonical_feature_registry import FEATURE_COLS_V22

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 256

def load_cde_model(ckpt_path, input_dim=52):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if 'config' in ckpt:
        config = ckpt['config']
        hidden = config['hidden']
        layers = config['layers']
    else:
        # Fallback defaults if config missing
        hidden = 128
        layers = 2
        
    model = NeuralCDE(input_dim, hidden, 2, n_layers=layers)
    model.load_state_dict(ckpt['state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

def analyze_sensitivity(model, X, feature_cols, n_samples=1000):
    print(f"\n🔬 Running Sensitivity Analysis on {n_samples} samples...")
    idxs = np.random.randint(0, len(X) - SEQ_LEN, size=n_samples)
    grads_sum = np.zeros(X.shape[1])
    
    for idx in tqdm(idxs):
        # (1, T, D)
        x_seq = X[idx : idx+SEQ_LEN]
        x_tensor = torch.tensor(x_seq, device=DEVICE).unsqueeze(0).float()
        x_tensor.requires_grad = True # Enable gradients w.r.t input
        
        # Forward
        z_out = model(x_tensor)
        preds = model.decoder(z_out) # (1, 2) -> [Return, Vol]
        
        # Backward on Predicted Return (Index 0)
        score = preds[0, 0] 
        score.backward()
        
        # Gradients: Max absolute grad over time window
        grad = x_tensor.grad.abs().squeeze(0).cpu().numpy() # (T, D)
        time_max_grad = grad.max(axis=0) # (D,)
        
        grads_sum += time_max_grad
        
    avg_sensitivity = grads_sum / n_samples
    # Normalize
    if avg_sensitivity.sum() > 0:
        avg_sensitivity = 100 * avg_sensitivity / avg_sensitivity.sum()
    
    sens_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': avg_sensitivity
    }).sort_values('Importance', ascending=False)
    
    print("\n🏆 Top 10 Most Important Features (Gradient Sensitivity):")
    print(sens_df.head(10).to_string(index=False))

def train_surrogate_tree(model, X, feature_cols, n_samples=5000):
    print(f"\n🌳 Training Surrogate Decision Tree on {n_samples} samples...")
    from sklearn.tree import DecisionTreeRegressor, export_text
    
    idxs = np.random.randint(0, len(X) - SEQ_LEN, size=n_samples)
    input_states = []
    targets = []
    
    with torch.no_grad():
        for idx in tqdm(idxs):
            seq = X[idx : idx+SEQ_LEN]
            last_step = seq[-1] # Interpret based on current state
            
            x_tensor = torch.tensor(seq, device=DEVICE).unsqueeze(0).float()
            # Predict Return
            pred = model.decoder(model(x_tensor))[0, 0].item()
            
            input_states.append(last_step)
            targets.append(pred)
            
    input_states = np.array(input_states)
    targets = np.array(targets)
    
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
    
    # 3. Analyze
    analyze_sensitivity(model, X, feature_cols, n_samples=args.samples)
    train_surrogate_tree(model, X, feature_cols, n_samples=args.samples * 5)

if __name__ == "__main__":
    main()
