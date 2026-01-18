
import torch
import sys
import os
import pandas as pd
import numpy as np

def inspect_checkpoint():
    print("🔍 Inspecting Checkpoint Stats...")
    
    # Paths to check
    paths = [
        "condor_brain_retrain_e1.pth",
        "models/condor_brain_retrain_e1.pth",
        "/kaggle/working/condor_brain_retrain_e1.pth",
        "/kaggle/input/condor-brain-seq-e1/condor_brain_retrain_e1.pth"
    ]
    
    model_path = None
    for p in paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        print("❌ Could not find checkpoint file.")
        return

    print(f"📂 Loading {model_path}...")
    ckpt = torch.load(model_path, map_location='cpu')
    
    if isinstance(ckpt, dict):
        median = ckpt.get("median", None)
        mad = ckpt.get("mad", None)
        cols = ckpt.get("feature_cols", [])
    else:
        print("❌ Checkpoint is not a dictionary (Old format?). Stats not found.")
        return

    if median is None or mad is None:
        print("❌ 'median' or 'mad' keys missing in checkpoint.")
        return

    print(f"\n📊 Normalization Stats ({len(cols)} features):")
    print(f"{'Feature':<20} | {'Median':>10} | {'MAD':>10} | {'Scale Factor (1/MAD)':>20}")
    print("-" * 70)
    
    # Convert to numpy for easy printing
    if isinstance(median, torch.Tensor): median = median.numpy()
    if isinstance(mad, torch.Tensor): mad = mad.numpy()
    
    # Squeeze if needed (1, 1, 24) -> (24,)
    median = median.flatten()
    mad = mad.flatten()
    
    for i, col in enumerate(cols):
        m = median[i]
        d = mad[i]
        scale = 1.0 / (d + 1e-8)
        print(f"{col:<20} | {m:10.4f} | {d:10.4f} | {scale:20.4f}")

    print("-" * 70)
    
    # Check for "Death by Scaling"
    # If MAD is huge, Scale Factor is tiny -> Inputs become 0 -> Outputs become 0
    huge_mad = mad > 1000
    if np.any(huge_mad):
        print(f"\n⚠️ WARNING: {np.sum(huge_mad)} features have HUGE MAD (>1000).")
        print("   This compresses inputs to zero, causing Model Collapse.")
    else:
        print("\n✅ Scaling looks reasonable (No massive outliers stored).")

if __name__ == "__main__":
    inspect_checkpoint()
