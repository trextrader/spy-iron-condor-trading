
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain import CondorBrain

def verify_inference():
    print("🚀 Verifying CondorBrain Sequence Model...")
    
    # Config (Must match training)
    D_MODEL = 512
    N_LAYERS = 12
    INPUT_DIM = 24
    SEQ_LEN = 256
    MODEL_PATH = "models/condor_brain_seq_e1.pth"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Initialize
    print("   Initializing Model Architecture...")
    model = CondorBrain(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        input_dim=INPUT_DIM,
        use_vol_gated_attn=True,
        use_topk_moe=True,
        moe_n_experts=3,
        moe_k=1
    ).to(device)
    
    # Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return False
        
    print(f"   Loading weights from {MODEL_PATH}...")
    try:
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Handle DataParallel keys if present (remove 'module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        model.eval()
        print("✅ Weights Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False
        
    # Inference Pass
    print("   Running Forward Pass (Synthetic Data)...")
    dummy_input = torch.randn(1, SEQ_LEN, INPUT_DIM).to(device)
    
    try:
        with torch.no_grad():
            # Run with all returns enabled
            res = model(dummy_input, return_regime=True, return_experts=True, return_features=True)
            
            # Unpack
            # (outputs, regime, horizon, features, experts)
            outputs = res[0]
            feat_pred = res[3]
            experts = res[4]
            
            print(f"   Shape Check:")
            print(f"   - Policy Output: {outputs.shape} (Expected: [1, 8])")
            print(f"   - Feature Pred:  {feat_pred.shape} (Expected: [1, 4])")
            
            if outputs.shape == (1, 8) and feat_pred.shape == (1, 4):
                print("✅ Forward Pass Successful")
                
                # Check Expert Usage
                if experts:
                    print(f"   - Experts Active: {list(experts.keys())}")
                
                return True
            else:
                print("❌ Shape Mismatch")
                return False
                
    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_inference()
    sys.exit(0 if success else 1)
