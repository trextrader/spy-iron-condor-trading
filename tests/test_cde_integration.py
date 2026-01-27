import torch
import sys
import os

# Add project root
sys.path.append(os.getcwd())

from intelligence.condor_brain import CondorBrain

def test_cde_integration():
    print("🚀 Testing CondorBrain with Neural CDE Backbone...")
    
    bs = 2
    seq_len = 32
    input_dim = 52
    d_model = 32 # Small for speed
    
    try:
        model = CondorBrain(
            d_model=d_model,
            input_dim=input_dim,
            n_layers=2, # Small
            use_cde=True # FORCE CDE
        )
        print("✅ CondorBrain initialized with use_cde=True")
        
        if hasattr(model, 'cde_backbone'):
            print("✅ cde_backbone found.")
        else:
            print("❌ cde_backbone MISSING!")
            return
            
        # Test Forward
        x = torch.randn(bs, seq_len, input_dim)
        print(f"🔄 Running forward pass on input {x.shape}...")
        
        output = model(x)
        
        if isinstance(output, tuple):
            print("✅ Forward pass returned tuple (Output, Regime, Forecast, etc.)")
            main_out = output[0]
        else:
            main_out = output
            
        print(f"✅ Output shape: {main_out.shape} (Expected {bs}, 10)")
        assert main_out.shape == (bs, 10)
        
        print("\n🎉 INTEGRATION TEST PASSED!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cde_integration()
