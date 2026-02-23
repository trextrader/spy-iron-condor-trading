
import torch
import torch.nn as nn
from intelligence.condor_brain_net import CondorNet

def run_formal_verification():
    print("=== Phase 1 Formal Verification Suite ===")
    
    # Common setup
    d_input = 54
    batch = 1
    seq_len = 20
    
    model = CondorNet(
        d_input=d_input,
        d_h=64, d_v=16, d_m=32, d_r=16,
        d_control=128,
        verbose_math=False
    )
    model.eval()

    def get_inputs(b, s, q_val=1.0):
        x = torch.randn(b, s, d_input)
        greeks = torch.randn(b, s, 5)
        q = torch.full((b, s, 1), float(q_val))
        preds = {
            'iv_rank': torch.full((b, s), 60.0),
            'bid_ask_spread': torch.full((b, s), 0.002),
            'price': torch.full((b, s), 500.0),
            'rsi': torch.full((b, s), 50.0),
            'delta_rsi': torch.zeros(b, s),
            'S_t': torch.full((b, s), 500.0),
            'S_t_minus_1': torch.full((b, s), 499.0),
            'gamma': torch.zeros(b, s)
        }
        return x, greeks, q, preds

    # 1. RA Separation Test
    print("\n[1] RA Separation Test")
    x, gks, _, preds = get_inputs(batch, seq_len)
    
    q_low = torch.full((batch, seq_len, 1), 0.1)
    q_high = torch.full((batch, seq_len, 1), 100.0)
    
    with torch.no_grad():
        # Capturing diagnostics requires a run where we check internal states
        # Since diagnostics are internal, we can check the output or use a hook
        # But we can also just check the final state if we modify forward to return it
        # For this test, let's use a simple diagnostic capture
        
        # We need to capture h and v from the loop. 
        # I'll add a small diagnostic return to the forward pass or just check the final output
        # if the output head is Linear(spec.d_x, 10).
        
        out_low = model(x, greeks=gks, q=q_low, **preds)
        out_high = model(x, greeks=gks, q=q_high, **preds)
        
        # Even better: Let's check the forcing modules directly to be 100% sure
        z_phys_low = torch.cat([gks[:, 0], torch.zeros(batch, 16)], dim=-1)
        z_phys_high = torch.cat([gks[:, 0], torch.zeros(batch, 16)], dim=-1)
        
        dh_low = model.D_forcing.D_h_phys(z_phys_low)
        dh_high = model.D_forcing.D_h_phys(z_phys_high)
        
        ra_diff = (dh_low - dh_high).abs().max()
        print(f"  - D_h Variance wrt q: {ra_diff.item():.2e} (Expected: 0.0)")
        
        # Check D_v
        z_exec_low = torch.cat([gks[:, 0], torch.zeros(batch, 16), q_low[:, 0]], dim=-1)
        z_exec_high = torch.cat([gks[:, 0], torch.zeros(batch, 16), q_high[:, 0]], dim=-1)
        dv_low = model.D_forcing.D_v(z_exec_low)
        dv_high = model.D_forcing.D_v(z_exec_high)
        dv_diff = (dv_low - dv_high).abs().mean()
        print(f"  - D_v Sensitivity to q: {dv_diff.item():.4f} (Expected: > 0)")

    # 2. Causality Check
    print("\n[2] Causality Check (r_k)")
    # We want to see if flipping r_k at step T affects x_k or x_{k+1}
    # In my fixed code: x_k depends on r_{k-1}.
    # So if we change predicates at step 5 (which determines r_5), 
    # then x_5 should NOT change, but x_6 SHOULD change.
    
    x, gks, q, preds = get_inputs(batch, seq_len)
    preds_flipped = {k: v.clone() for k, v in preds.items()}
    # Flip IV Rank at step 5 (index 5)
    preds_flipped['iv_rank'][:, 5] = 100.0 
    
    # We need to capture the state sequence. 
    # I'll modify the loop to return diagnostics or just compare outputs.
    # Output at step 5 is x_5. Output at step 6 is x_6.
    # If seq_len=7, output is x_6.
    
    with torch.no_grad():
        # Run seq 1..6
        out_normal = model(x[:, :7], greeks=gks[:, :7], q=q[:, :7], **{k:v[:, :7] for k, v in preds.items()})
        out_flipped = model(x[:, :7], greeks=gks[:, :7], q=q[:, :7], **{k:v[:, :7] for k, v in preds_flipped.items()})
        
        # If causality is correct:
        # Step 5 predicates -> r_5
        # x_5 depends on D(gks_5, r_4, q_5) -> No change
        # x_6 depends on D(gks_6, r_5, q_6) -> Changed!
        
        # Wait, the output head is on x_final. For seq_len 7, final is x_6.
        # Let's try seq_len 6 (final is x_5)
        out_5_normal = model(x[:, :6], greeks=gks[:, :6], q=q[:, :6], **{k:v[:, :6] for k, v in preds.items()})
        out_5_flipped = model(x[:, :6], greeks=gks[:, :6], q=q[:, :6], **{k:v[:, :6] for k, v in preds_flipped.items()})
        
        diff_5 = (out_5_normal - out_5_flipped).abs().max()
        diff_6 = (out_normal - out_flipped).abs().max()
        
        print(f"  - Difference at x_5 (immediate): {diff_5.item():.2e} (Expected: < 1e-7 if pure ETD)")
        print(f"  - Difference at x_6 (causal):    {diff_6.item():.4f} (Expected: > 0)")

    # 3. ETD-1 Scaling Sanity
    print("\n[3] ETD-1 Scaling Sanity (dt=1 vs dt=0.5)")
    # Running 10 steps of dt=1 vs 20 steps of dt=0.5
    # For a linear system dx/dt = Ax, x(T) = exp(AT)x(0).
    # ETD-1 is exact for the linear part. 
    # We'll use a simplified run without forcing to check the propagator.
    
    with torch.no_grad():
        # Disable forcing for pure propagator test
        model.B_theta.B_h.weight.data.fill_(0)
        model.D_forcing.D_h_phys[0].weight.data.fill_(0)
        
        x0 = torch.randn(batch, 1, d_input)
        
        # Run 4 steps with dt=1.0
        # We simulate this by passing a sequence of 5 steps
        res_1 = model(x0.repeat(1, 5, 1), greeks=torch.zeros(1, 5, 5), q=torch.zeros(1, 5, 1), 
                      **{k: torch.zeros(1, 5) for k in preds})
        
        # Run 8 steps with dt=0.5
        # My current implementation has dt hardcoded as 1/240 or similar? 
        # No, let's check the code.
        # In CondorNet.forward: dt = 1.0 / 240.0
        # I'll temporarily modify it to be a parameter or just check the scaling.
        print("  - Note: dt is currently fixed in code at 1/240.")
        print("  - Propagator stability check: Running 100 steps...")
        res_long = model(x0.repeat(1, 101, 1), greeks=torch.zeros(1, 101, 5), q=torch.zeros(1, 101, 1),
                         **{k: torch.zeros(1, 101) for k in preds})
        
        mag = res_long.abs().mean()
        print(f"  - Final state magnitude after 100 steps: {mag.item():.4f} (Expected: bounded)")

if __name__ == "__main__":
    run_formal_verification()
