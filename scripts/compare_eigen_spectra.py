
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain_net import CondorNet
from intelligence.canonical_feature_registry import D_INPUT

def load_model(path, device='cpu'):
    print(f"Loading {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    # Infer args from checkpoint or defaults
    # We need to instantiate CondorNet to load state_dict
    # For structure analysis, we just need the right dimensions
    
    # Try to find config in checkpoint
    if 'config' in checkpoint:
        conf = checkpoint['config']
        d_h = conf.get('d_h', 256) # Default from v4.2 spec if missing?
        # Actually CondorNet default is different.
        # Let's try to infer or use standard v4.2 params
        # If training script saved 'model_args' or similar.
        pass
    
    # Standard V4.2 dimensions used in training script:
    # d_model=256, layers=32 (from args default)
    # But CondorNet init signature is:
    # d_h=256, d_v=32, d_m=64, d_r=32 ...
    # Wait, train_condor_brain uses CondorBrain, which WRAPS CondorNet (or NeuralCDE).
    # IF CondorBrain is the top level, the checkpoint contains CondorBrain state.
    # CondorBrain has 'backbone' which is CondorNet (if CDE mode).
    
    # Let's inspect keys
    keys = list(checkpoint['state_dict'].keys())
    # Look for 'backbone.A_theta.blocks.0' or similar
    
    # We can reconstruct A without full model if we identify the weights.
    # But cleaner to load model.
    
    # Assumption: User used defaults d_model=256
    model = CondorNet(d_h=64, d_v=16, d_m=32, d_r=16) # Minimal init just to see
    
    # We might need to adjust mapping if keys don't match
    pass

    return checkpoint['state_dict']

def extract_A_from_state(state_dict):
    # A_theta in CondorNet is likely implemented as a structured matrix parameter.
    # We need to find the keys corresponding to A_theta.
    # In v4.2, A is likely a 'BlockMatrix' or similar.
    
    # Let's look for 'backbone.A_theta' prefix (if wrapped in CondorBrain)
    # or just 'A_theta' if CondorNet saved directly.
    
    prefixes = ['backbone.A_theta.', 'net.A_theta.', 'A_theta.']
    
    # We can't easily reconstruct the class structure from state_dict alone 
    # if it involves complex masking/parametrization.
    # However, if we can find the explicit matrix or the params to rebuild it.
    
    # Fallback: If we can't load the class, we can't run .full_matrix().
    # Detailed Plan: 
    # 1. Instantiate CondorBrain (wrapper) with correct args.
    # 2. Load state_dict.
    # 3. Access model.backbone.get_A_matrix().
    
    pass

def main():
    parser = argparse.ArgumentParser(description="Compare A-Matrix Eigen Spectra")
    parser.add_argument("models", nargs='+', help="Paths to .pth model files")
    parser.add_argument("--output", default="eigen_spectra_comparison.png", help="Output plot path")
    args = parser.parse_args()
    
    plt.figure(figsize=(10, 10))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    for i, path in enumerate(args.models):
        print(f"Analyzing {path}...")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # Heuristic to find A matrix directly in weights if possible
            # Or assume we need to init the model. 
            # Given the complexity, let's try to instantiate CondorBrain.
            
            from intelligence.condor_brain import CondorBrain
            from intelligence.canonical_feature_registry import INPUT_DIM_V42
            
            # Default args from training script
            model = CondorBrain(d_model=256, n_layers=32, input_dim=INPUT_DIM_V42, use_vol_gated_attn=True)
            
            # Strict=False to handle potential minor mismatches (e.g. if saved with DataParallel)
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            
            if hasattr(model, 'cde_backbone') and hasattr(model.cde_backbone, 'get_A_matrix'):
                A = model.cde_backbone.get_A_matrix().detach().numpy()
            elif hasattr(model, 'backbone') and hasattr(model.backbone, 'get_A_matrix'):
                A = model.backbone.get_A_matrix().detach().numpy()
            elif hasattr(model, 'net') and hasattr(model.net, 'get_A_matrix'): # Mamba/Linear path?
                 # Mamba path might not have A matrix in the same sense
                 print(f"Skipping {path}: No A-matrix found (Mamba backbone?)")
                 continue
            else:
                 print(f"Skipping {path}: Structure unknown")
                 continue
            
            print(f"A Matrix Shape: {A.shape}")
            eigvals = np.linalg.eigvals(A)
            
            label = os.path.basename(path).replace('.pth', '')
            plt.scatter(eigvals.real, eigvals.imag, alpha=0.5, s=10, label=label, c=colors[i % len(colors)])
            
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            import traceback
            traceback.print_exc()
    
    plt.title(f"A Matrix Eigenvalues (Spectrum Verification)")
    plt.xlabel("Real Part (Stability < 0 for continuous, |z| < 1 for discrete?)") 
    # Note: If Neural CDE uses exp(A*dt), then real part < 0 implies stability.
    # If discrete step x_{k+1} = A x_k, then |radius| < 1.
    # CondorNet uses Matrix Exponential? 
    # condor_brain_net.py: F_k, phi1 = etd1_kernel(A_full, dt) -> This implies Continuous
    # ETD1 implies A is the derivative. Stable if Re(eig) < 0.
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(args.output)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
