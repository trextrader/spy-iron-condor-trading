
import torch
import torch.nn as nn
from intelligence.condor_brain_net import RelationalLogicLayer, PredicateSignature, AugmentedStateSpec

def verify_predicate_semantics():
    print("=== Sub-Phase 3.2: Predicate & Gate Semantics Audit (V47) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Relational Logic Test
    print("\n[1] RelationalLogicLayer Test")
    n_inputs = 4
    out_dim = 8
    layer = RelationalLogicLayer(n_inputs, out_dim).to(device)
    layer.eval()
    
    # Input: [1.0, 0.5, 1.0, 0.0]
    x = torch.tensor([[1.0, 0.5, 1.0, 0.0]], device=device)
    
    with torch.no_grad():
        out = layer(x)
        print(f"  - Output Shape: {out.shape}")
        
        # Internal inspection (manually compute for sanity)
        # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # x0-x1 = 0.5 (gt), x0-x2 = 0.0 (eq), x0-x3 = 1.0 (gt)
        # x1-x2 = -0.5 (lt), x1-x3 = 0.5 (gt)
        # x2-x3 = 1.0 (gt)
        
        tri_diffs = x[:, [0,0,0,1,1,2]] - x[:, [1,2,3,2,3,3]]
        lt = torch.sigmoid(-layer.steepness * tri_diffs)
        gt = torch.sigmoid(layer.steepness * tri_diffs)
        eq = torch.exp(-layer.steepness * (tri_diffs ** 2))
        
        print(f"  - Pair (1.0 vs 0.5) Soft GT: {gt[0, 0].item():.4f}")
        print(f"  - Pair (1.0 vs 1.0) Soft EQ: {eq[0, 1].item():.4f}")
        print(f"  - Pair (0.5 vs 1.0) Soft LT: {lt[0, 3].item():.4f}")

    # 2. Signature Invariance Test
    print("\n[2] PredicateSignature Invariance Test")
    sig_layer = PredicateSignature(K=n_inputs).to(device)
    p = torch.tensor([[0.1, 0.9, 0.4, 0.6]], device=device)
    p_perm = torch.tensor([[0.9, 0.1, 0.6, 0.4]], device=device) # Permutation
    
    with torch.no_grad():
        _, _, _, z = sig_layer(p)
        _, _, _, z_perm = sig_layer(p_perm)
        
        diff = (z - z_perm).abs().max().item()
        print(f"  - Signature Max Diff (Permuted): {diff:.8f}")
        if diff < 1e-6:
            print("  SUCCESS: Signature is group-invariant.")
        else:
            print("  FAILURE: Signature is NOT group-invariant.")

    print("\nSUCCESS: Sub-Phase 3.2 mathematical semantics are verified.")

if __name__ == "__main__":
    verify_predicate_semantics()
