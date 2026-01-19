
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain import CondorBrain

def verify_inference():
    print("🚀 Verifying CondorBrain Sequence Model...")
    
    try:
        # Config defaults
        D_MODEL = 512
        N_LAYERS = 12
        INPUT_DIM = 24
        SEQ_LEN = 256
        
        # Check for the rescued model first
        if os.path.exists("models/condor_brain_retrain_e1.pth"):
            MODEL_PATH = "models/condor_brain_retrain_e1.pth"
        else:
            MODEL_PATH = "models/condor_brain_seq_e1.pth"
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Path(s)
        model_path = os.environ.get("CONDOR_MODEL", MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"[verify_inference] loading checkpoint: {model_path}")

        # Load checkpoint (supports raw state_dict OR dict checkpoint)
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            feature_cols = ckpt.get("feature_cols", None)
            median = ckpt.get("median", None)
            mad = ckpt.get("mad", None)
            seq_len = int(ckpt.get("seq_len", 256))
            input_dim = int(ckpt.get("input_dim", 24))
            use_diffusion = bool(ckpt.get("use_diffusion", False))
            diffusion_steps = int(ckpt.get("diffusion_steps", 0))
        else:
            state_dict = ckpt
            feature_cols = None
            median = None
            mad = None
            seq_len = 256
            input_dim = 24
            use_diffusion = False
            diffusion_steps = 0

        # Build model (must match training config)
        model = CondorBrain(
            d_model=512,
            n_layers=12,
            input_dim=input_dim,
            use_vol_gated_attn=True,
            use_topk_moe=True,
            moe_n_experts=3,
            moe_k=1,
            use_diffusion=use_diffusion,
            diffusion_steps=diffusion_steps if use_diffusion else 0,
        ).to(device)

        print("[verify_inference] loading weights (strict=True)...")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("[verify_inference] model loaded and set to eval")

        # Create dummy input (batch=2, seq=seq_len, dim=input_dim)
        x = torch.randn(2, seq_len, input_dim, device=device)

        with torch.no_grad():
            # Request all outputs
            res = model(x, return_regime=True, return_experts=True, return_features=True)

            if not isinstance(res, tuple):
                raise TypeError(f"Expected tuple output, got: {type(res)}")
            if len(res) < 4:
                raise ValueError(f"Expected at least 4 outputs, got {len(res)}")

            # Safe unpacking (handles diffusion-enabled tuples)
            policy = res[0]
            regime = res[1]
            horizon = res[2]
            feat = res[3] if len(res) > 3 else None

            print("[verify_inference] policy:", tuple(policy.shape))
            print("[verify_inference] regime:", "None" if regime is None else tuple(regime.shape))
            print("[verify_inference] horizon:", "None" if horizon is None else type(horizon))
            print("[verify_inference] feat:", "None" if feat is None else tuple(feat.shape))

            # Basic sanity checks
            if policy.ndim != 2 or policy.shape[-1] != 8:
                raise AssertionError(f"Policy head should be [B,8], got {tuple(policy.shape)}")

            outputs = res[0]
            feat_pred = res[3] if len(res) > 3 else None
            
            print(f"   Shape Check:")
            print(f"   - Policy Output: {outputs.shape} (Expected: [B, 8])")
            if feat_pred is not None:
                print(f"   - Feature Pred:  {feat_pred.shape} (Expected: [B, 4])")
            
            # ---- V2.1 & COLLAPSE CHECKS ----
            if input_dim == 32:
                print("✅ Input Dim = 32 (Matches V2.1 Schema)")
            else:
                print(f"⚠️  Input Dim = {input_dim} (Likely Legacy V2.0)")

            # Collapse Check
            pol_std = outputs.std(dim=0).mean().item()
            print(f"   Policy output std dev: {pol_std:.6f}")
            
            if pol_std < 0.001:
                print("❌ FAIL: Policy outputs are collapsed (constant). Retraining failed.")
                return False
            else:
                print("✅ PASS: Policy outputs show variance (model is active).")

            if outputs.shape[1] == 8:
                print("✅ Forward Pass Successful")
                return True
            else:
                print("❌ Shape Mismatch")
                return False
                
    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_inference()
    sys.exit(0 if success else 1)
