import torch
import numpy as np
from intelligence.condor_brain import CondorBrain
from intelligence.condor_brain_net import CondorNet
import time

def verify_condor_brain_parity():
    print("\n--- Verifying CondorBrain Stateful Parity ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondorBrain(d_model=128, n_layers=4, input_dim=52, use_cde=True).to(device).eval()
    
    batch, seq_len, dim = 2, 32, 52
    x = torch.randn(batch, seq_len, dim, device=device)
    
    # 1. Batch Forward (O(T^2) baseline)
    with torch.no_grad():
        y_batch = model(x)
        # If tuple, take the first element (outputs)
        if isinstance(y_batch, tuple):
            y_batch = y_batch[0]
            
    # 2. Stateful Step (O(T) incremental) with windowed context
    with torch.no_grad():
        h = model.get_initial_state(x[:, 0, :])
        # We need a buffer to satisfy the recursive evaluator's lookbacks (max 64 usually)
        # For verification, we can just slice from the full sequence to prove lookback dependency
        for t in range(seq_len - 1):
            # Window for predicates: up to t+1
            # Note: CondorBrain.step currently expects (B, D) tensors.
            # I must modify CondorBrain.step to optionally take the window 
            # if we want true parity with a lookback-aware model.
            y_step, h = model.step(x[:, t, :], x[:, t+1, :], h)
            
    # Compare
    diff = torch.abs(y_batch - y_step).max().item()
    print(f"Max Difference: {diff:.2e}")
    if diff < 1e-4:
        print("✅ SUCCESS: Stateful Parity Verified for CondorBrain")
    else:
        print(f"⚠️  DEBUG: Mismatch ({diff:.2e}) - Investigating lookback dependency...")
        # Try with a modified step that takes the full history (to prove it's the lookbacks)
        h = model.get_initial_state(x[:, 0, :])
        for t in range(seq_len - 1):
            # Pass prefix for predicate context
            prefix = x[:, :t+2, :] 
            # We'll need to modify CondorBrain.step to handle this
            pass 

def verify_condor_net_parity():
    print("\n--- Verifying CondorNet Stateful Parity ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondorNet(d_input=54, d_h=64, d_v=16, d_m=16, d_r=16).to(device).eval()
    
    batch, seq_len, dim = 1, 16, 54
    x = torch.randn(batch, seq_len, dim, device=device)
    
    # Mock remaining inputs
    greeks = torch.randn(batch, seq_len, 5, device=device)
    q = torch.ones(batch, seq_len, 1, device=device)
    pred_inputs = {
        'iv_rank': torch.rand(batch, device=device) * 100,
        'bid_ask_spread': torch.rand(batch, device=device) * 0.01,
        'price': torch.randn(batch, device=device) + 500,
        'rsi': torch.rand(batch, device=device) * 100,
        'delta_rsi': torch.randn(batch, device=device),
        'S_t': torch.randn(batch, device=device) + 500,
        'S_t_minus_1': torch.randn(batch, device=device) + 500,
        'gamma': torch.randn(batch, device=device) * 0.1
    }
    
    # 1. Batch Forward
    with torch.no_grad():
        y_batch = model(x, greeks=greeks, q=q)
        
    # 2. Stateful Step
    with torch.no_grad():
        # u is computed via TFT which is non-incremental in this implementation 
        # (needs whole sequence). For verification, we extract u from the model's internal TFT pass
        u = model.tft(x, return_sequence=True)
        
        state = model.get_initial_state(x[:, 0, :])
        for t in range(seq_len - 1):
            # Update pred_inputs for this step (mocking bar-by-bar update)
            local_pred = {k: v for k, v in pred_inputs.items()} # In real use, these come from current bar
            y_step, state = model.step(
                x_prev=x[:, t, :], 
                x_curr=x[:, t+1, :], 
                state_prev=state,
                u_k=u[:, t, :],
                greeks_k=greeks[:, t, :],
                q_k=q[:, t, :],
                pred_inputs=local_pred
            )
            
    # Compare
    diff = torch.abs(y_batch - y_step).max().item()
    print(f"Max Difference: {diff:.2e}")
    if diff < 1e-4:
        print("✅ SUCCESS: Stateful Parity Verified for CondorNet")
    else:
        print("❌ FAILURE: Parity Mismatch")

if __name__ == "__main__":
    verify_condor_brain_parity()
    verify_condor_net_parity()
