import torch
import torch.nn as nn
import numpy as np
from intelligence.condor_brain_net import CondorNet
from intelligence.canonical_feature_registry import INPUT_DIM_V22

def test_bct():
    print("Starting Bar Consistency Test (BCT)...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondorNet(
        d_input=INPUT_DIM_V22,
        d_h=64, d_v=16, d_m=32, d_r=16
    ).to(device).eval()

    def run_with_candidates(n_candidates):
        torch.manual_seed(42)
        # Create a single bar sequence: 1 initial state + n_candidates steps (redundant)
        # Row 0 stays, Row 1 is repeated n_candidates times
        x_base = torch.randn(1, 2, INPUT_DIM_V22).to(device)
        
        row_0 = x_base[:, 0:1, :]
        row_1 = x_base[:, 1:2, :]
        
        # Candidate Noise: Indices 5-13 (Options)
        # We want the PHYSICS to be invariant even if options are different.
        candidates = []
        for i in range(n_candidates):
            c_row = row_1.clone()
            # Inject noise ONLY into candidate-specific features
            c_row[:, :, 5:14] += torch.randn(1, 1, 9).to(device) * 0.1
            candidates.append(c_row)
            
        x = torch.cat([row_0] + candidates, dim=1)
        
        # Scenario: BCS (mask = 1 only on the first step, then 0s)
        mask = torch.zeros(1, x.size(1) - 1).to(device)
        mask[0, 0] = 1.0 
        
        # We want to see if the state change [0] -> [T] is the same for any n_candidates
        with torch.no_grad():
            outputs, diag = model(x, timestep_mask=mask, return_diagnostics=True)
            
        final_state = diag['z_final'][0] # (batch, d_x)
        return final_state

    # Test with 100, 250, 500 candidates
    state_100 = run_with_candidates(100)
    state_250 = run_with_candidates(250)
    state_500 = run_with_candidates(500)

    # They should be IDENTICAL because we only advanced once in all cases
    diff_250 = torch.norm(state_100 - state_250).item()
    diff_500 = torch.norm(state_100 - state_500).item()

    print(f"Diff 100 vs 250: {diff_250:.8f}")
    print(f"Diff 100 vs 500: {diff_500:.8f}")

    if diff_250 < 1e-6 and diff_500 < 1e-6:
        print("BCT PASSED: State evolution is invariant to candidate count.")
    else:
        print("BCT FAILED: State evolution varies with candidate count.")

if __name__ == "__main__":
    test_bct()
