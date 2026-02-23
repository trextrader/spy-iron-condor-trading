
import torch
import numpy as np
from intelligence.condor_brain import CondorBrain, CondorLoss

def test_brain_hardening():
    print("=== Sub-Phase 2.2: MoE Hardening Verification (V47) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondorBrain(
        d_model=128, 
        n_layers=4, 
        input_dim=54,
        use_topk_moe=True,
        use_diffusion=True,
        use_cde=True
    ).to(device)
    
    loss_fn = CondorLoss()
    
    batch = 16
    seq = 32
    x = torch.randn(batch, seq, 54).to(device)
    target = torch.randn(batch, 10).to(device)
    
    # 1. MoE Routing & Entropy Check
    print("\n[1] MoE Routing Facets")
    model.train()
    # returns: (outputs, regime, horizon, diffusion, routing)
    res = model(x, return_experts=True)
    outputs, regime, _, _, routing = res
    
    probs = routing['routing_weights']
    
    # Internal methods check
    bal_l = model.moe_head.get_load_balancing_loss()
    ent_l = model.moe_head.get_routing_entropy_loss()
    print(f"  - Quadratic Balancing Loss: {bal_l.item():.6f}")
    print(f"  - Routing Entropy Loss: {ent_l.item():.6f}")
    
    # 2. Temperature Annealing Check
    print("\n[2] Temperature Annealing")
    t_start = model.update_moe_temperature(step=0, max_steps=1000, T_start=2.0, T_end=0.5)
    print(f"  - Step 0 Temp: {t_start:.4f}")
    
    t_mid = model.update_moe_temperature(step=500, max_steps=1000, T_start=2.0, T_end=0.5)
    print(f"  - Step 500 Temp: {t_mid:.4f} (Expected: ~1.0)")
    
    t_end = model.update_moe_temperature(step=1000, max_steps=1000, T_start=2.0, T_end=0.5)
    print(f"  - Step 1000 Temp: {t_end:.4f} (Expected: 0.5)")
    
    # 3. Composite Loss Integration
    print("\n[3] CondorLoss Integration")
    total_l = loss_fn(outputs, target, model=model)
    print(f"  - Total Composite Loss (with bal + ent): {total_l.item():.4f}")
    
    # Verify that the term is actually included
    loss_fn.moe_bal_weight = 1000.0
    total_l_heavy = loss_fn(outputs, target, model=model)
    print(f"  - Heavy-Weighted Bal Loss: {total_l_heavy.item():.4f} (Expected: significantly higher)")

if __name__ == "__main__":
    test_brain_hardening()
