
import torch
from intelligence.condor_brain_net import CondorNet

def test_physics_hardening():
    print("Initializing CondorNet for Physics Hardening Verification...")
    model = CondorNet(
        d_input=54,
        d_h=64,
        d_v=16,
        d_m=32,
        d_r=16,
        d_control=128,
        verbose_math=True
    )
    
    batch = 2
    seq_len = 10
    x = torch.randn(batch, seq_len, 54)
    greeks = torch.randn(batch, seq_len, 5)
    q = torch.randn(batch, seq_len, 1)
    
    print("\nRunning Forward Pass...")
    try:
        # We need to provide the additional predicate inputs or they default
        output = model(
            x, 
            greeks=greeks, 
            q=q,
            iv_rank=torch.full((batch, seq_len), 60.0),
            bid_ask_spread=torch.full((batch, seq_len), 0.002),
            price=torch.full((batch, seq_len), 500.0),
            rsi=torch.full((batch, seq_len), 50.0),
            delta_rsi=torch.zeros(batch, seq_len),
            S_t=torch.full((batch, seq_len), 500.0),
            S_t_minus_1=torch.full((batch, seq_len), 499.0),
            gamma=torch.zeros(batch, seq_len)
        )
        print("\nForward Pass Successful!")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"\nForward Pass Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_physics_hardening()
