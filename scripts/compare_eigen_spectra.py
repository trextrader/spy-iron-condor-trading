
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain_net_v42 import CondorNet
from intelligence.canonical_feature_registry import D_INPUT

def load_and_extract_A(path, device='cpu'):
    print(f"Loading {path}...")
    try:
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # clean state dict keys
        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_state[k[7:]] = v
            else:
                clean_state[k] = v
                
        # Attempt to infer args from checkpoint
        state_dims = checkpoint.get('state_dims', {})
        hparams = checkpoint.get('hyperparameters', {})
        
        # Defaults if missing (V4.2 Standard)
        d_h = state_dims.get('d_h', 256)
        d_v = state_dims.get('d_v', 32)
        d_m = state_dims.get('d_m', 64)
        d_r = state_dims.get('d_r', 32)
        
        n_predicates = hparams.get('n_predicates', 512)
        n_sets = hparams.get('n_sets', 256)
        n_super_sets = hparams.get('n_super_sets', 128)
        n_layers = hparams.get('n_layers', 2)
        d_control = hparams.get('d_control', 128)
        
        print(f"  Config inferred: d_h={d_h}, d_v={d_v}, predicates={n_predicates}")
        
        model = CondorNet(
            d_h=d_h, d_v=d_v, d_m=d_m, d_r=d_r,
            d_control=d_control,
            n_layers=n_layers,
            n_predicates=n_predicates,
            n_sets=n_sets,
            n_super_sets=n_super_sets,
            verbose_math=False
        )
        
        model.load_state_dict(clean_state, strict=False)
        
        if hasattr(model, 'A_theta'):
             # Use the method that constructs it from weights
             return model.A_theta.full_matrix().detach().cpu().numpy()
        else:
            print(f"  Error: Model does not have A_theta")
            return None

    except Exception as e:
        print(f"  Failed to load {path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare CondorNet v4.2 A-Matrix Spectra")
    parser.add_argument("--dir", type=str, help="Directory containing .pth files")
    parser.add_argument("models", nargs='*', help="Explicit paths to .pth files")
    parser.add_argument("--output", default="v42_eigen_spectra.png", help="Output plot path")
    args = parser.parse_args()
    
    # Collect files
    files = []
    if args.dir:
        import glob
        files.extend(glob.glob(os.path.join(args.dir, "*.pth")))
    files.extend(args.models)
    
    if not files:
        print("No models provided. Usage: --dir <path> or list .pth files")
        return

    plt.figure(figsize=(10, 10))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    
    plotted = 0
    for i, path in enumerate(files):
        A = load_and_extract_A(path)
        if A is None:
            continue
            
        print(f"  A Matrix Shape: {A.shape}")
        eigvals = np.linalg.eigvals(A)
        
        label = os.path.basename(path).replace('.pth', '')
        plt.scatter(eigvals.real, eigvals.imag, alpha=0.6, s=15, label=label, c=colors[i % len(colors)])
        plotted += 1
            
    if plotted == 0:
        print("No valid models found.")
        return

    plt.title(f"CondorNet v4.2 A-Matrix Spectra (Seed Stability)")
    plt.xlabel("Real Part (Re > 0 indicates growth/instability if continuous)") 
    plt.ylabel("Imaginary Part (Oscillation)")
    
    # Draw unit circle equivalent or stability line
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, label="Re=0 (Stability Boundary)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
