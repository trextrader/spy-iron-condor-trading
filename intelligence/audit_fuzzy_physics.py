
import numpy as np
import torch
from intelligence.fuzzy_engine import (
    compute_position_size, 
    calculate_rsi_membership, 
    calculate_adx_membership,
    calculate_cmf_membership,
    calculate_model_membership
)

def audit_fuzzy_physics():
    print("=== Phase 4: Fuzzy Sizing Engine Physics Audit (V47) ===")
    
    # 1. Monotonicity Proof: RSI (Risk Factor)
    print("\n[1] Prove Membership Monotonicity: RSI (Risk ↑ => Size ↓)")
    # Ideal IC zone is 40-60. 
    # Moving from 60 -> 100 should be monotonic (bad for range trading)
    rsi_range = np.linspace(60, 100, 20)
    memberships = [calculate_rsi_membership(r) for r in rsi_range]
    
    # Check if each step is non-increasing
    is_monotone = all(x >= y for x, y in zip(memberships, memberships[1:]))
    print(f"  - RSI [60->100] Monotone Decay: {'SUCCESS' if is_monotone else 'FAILURE'}")
    print(f"  - End points: RSI=60 => {memberships[0]:.2f}, RSI=100 => {memberships[-1]:.2f}")

    # 2. Hard Risk Veto Audit
    print("\n[2] Prove Hard Risk Veto (T-norm Aggregation)")
    # ADX is a veto factor. ADX=100 is "extreme trend", should kill trade.
    weights = {"adx": 0.5, "rsi": 0.5}
    benign_memberships = {
        "rsi": 1.0,   # Perfect range
        "adx": 0.0    # Extreme trend (veto target)
    }
    
    # Without veto, weighted sum would be 0.5. 
    # With veto, it must be 0.0.
    from intelligence.fuzzy_engine import compute_fuzzy_confidence
    ft_veto = compute_fuzzy_confidence(benign_memberships, weights, veto_factors=["adx"])
    
    print(f"  - Veto test (ADX=0.0, RSI=1.0): Ft = {ft_veto:.2f}")
    if ft_veto == 0.0:
        print("  SUCCESS: Hard Risk Veto confirmed.")
    else:
        print("  FAILURE: Aggregation logic leaked risk.")

    # 3. Geometric Mean Stability Audit (Model Factor)
    print("\n[3] Model Membership Proof (Geometric Mean Strictness)")
    # calculate_model_membership(confidence, prob_profit)
    # If confidence is 0.8 but prob_profit is 0.4 (unprofitable), score should be 0.0
    m_fail = calculate_model_membership(0.8, 0.4)
    m_pass = calculate_model_membership(0.8, 0.6)
    
    print(f"  - Model Fail (prob=0.4): {m_fail:.2f}")
    print(f"  - Model Pass (prob=0.6): {m_pass:.2f}")
    
    if m_fail == 0.0 and m_pass > 0.0:
        print("  SUCCESS: Model strictness verified.")
    else:
        print("  FAILURE: Relaxed model probability detected.")

    # 4. Ceiling & Lipschitz Integrity
    print("\n[4] Defuzzification Lipschitz Check")
    # Small change in input should lead to small change in size
    equity = 100000.0
    max_loss = 500.0
    weights_full = {"rsi": 1.0}
    
    # Base case
    q1 = compute_position_size(equity, max_loss, {"rsi": 1.0}, weights_full, 15.0, 10.0, 30.0)
    # Small perturbation (RSI 50 -> 50.01)
    q2 = compute_position_size(equity, max_loss, {"rsi": 0.9999}, weights_full, 15.0, 10.0, 30.0)
    
    print(f"  - q1 (RSI=1.0): {q1}")
    print(f"  - q2 (RSI=0.9999): {q2}")
    if abs(q1 - q2) <= 1:
        print("  SUCCESS: Local Lipschitz continuous.")
    else:
        print("  WARNING: Sizing jump detected.")

if __name__ == "__main__":
    audit_fuzzy_physics()
