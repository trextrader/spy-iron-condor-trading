
import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain_net import CondorNet
from intelligence.canonical_feature_registry import D_INPUT, FEATURE_LIST

def validate_v42_structure():
    print("=" * 60)
    print("CONDORNET V4.2 STRUCTURAL VALIDATION")
    print("=" * 60)
    
    # Check Registry
    print(f"Registry D_INPUT: {D_INPUT}")
    print(f"Registry FEATURE_LIST Len: {len(FEATURE_LIST)}")
    assert D_INPUT == len(FEATURE_LIST), "Registry Mismatch!"
    assert D_INPUT == 91, f"Expected 91 features, got {D_INPUT}"
    print("[PASS] Registry Consistency")

    # Initialize Model
    print("\nInitializing CondorNet...")
    model = CondorNet(
        d_h=64, d_v=16, d_m=32, d_r=16, # Small for test
        d_control=32,
        n_layers=1,
        verbose_math=True
    )
    
    # Verify Attributes
    print(f"Model.d_input: {model.d_input}")
    print(f"Model.d_pivot_raw: {getattr(model, 'd_pivot_raw', 'MISSING')}")
    print(f"Model.d_encoded_input: {getattr(model, 'd_encoded_input', 'MISSING')}")
    
    assert model.d_input == 91, "Model checked incorrect D_INPUT"
    assert model.d_pivot_raw == 4, "Incorrect d_pivot_raw"
    assert model.d_encoded_input == (91 - 4) + 32, f"Incorrect d_encoded_input: {model.d_encoded_input}"
    print("[PASS] Model Dimensions")

    # Verify Pivot Encoder Existence
    assert hasattr(model, 'pivot_encoder'), "Missing pivot_encoder"
    print("[PASS] Pivot Encoder Existence")

    # Forward Pass Trace
    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, D_INPUT)
    
    print(f"\nRunning Forward Pass with Input: {x.shape}")
    try:
        out, aux = model(x, return_diagnostics=False) # aux is pivot_preds
        print(f"Output Shape: {out.shape}")
        if isinstance(aux, tuple):
             print(f"Aux Head (Pivots) Output: {aux[0].shape}, {aux[1].shape}")
        else:
             print(f"Aux Head Output: {aux.shape}")
             
        assert out.shape == (batch_size, 10), "Output shape mismatch"
        print("[PASS] Forward Pass Success")
    except Exception as e:
        print(f"[FAIL] Forward Pass Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initial State Trace
    print("\nChecking get_initial_state...")
    x_init = x[:, 0, :]
    try:
        z0 = model.get_initial_state(x_init)
        print(f"Initial State Shape: {z0.shape}")
        assert z0.shape == (batch_size, model.spec.d_x), "Initial state shape mismatch"
        print("[PASS] Initial State Success")
    except Exception as e:
        print(f"[FAIL] get_initial_state Error: {e}")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("ALL CHECKS PASSED")
    print("="*60)

if __name__ == "__main__":
    validate_v42_structure()
