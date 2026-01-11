# CondorBrain A100 Training Sweep
# Copy this to a Colab notebook (.ipynb) or run cells sequentially

#@title 1. Setup & Installation
!pip install -q mamba-ssm causal-conv1d pandas_ta torch

import os
import time
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# A100 TF32 optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#@title 2. Upload & Extract Data
from google.colab import files
import zipfile

# Upload your zip file
print("Please upload mamba_institutional_1m.zip (or targets version)")
uploaded = files.upload()

# Extract
for fn in uploaded.keys():
    if fn.endswith('.zip'):
        with zipfile.ZipFile(fn, 'r') as z:
            z.extractall('data/processed/')
        print(f"Extracted: {fn}")

!ls -la data/processed/

#@title 3. Clone/Upload Repository
# Option A: Mount Drive with repo
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/SPYOptionTrader_test/* .

# Option B: Direct upload of intelligence folder
# !mkdir -p intelligence
# files.upload()  # Upload condor_brain.py, train_condor_brain.py

#@title 4. Configuration Matrix
SWEEP_CONFIGS = [
    # (epochs, batch_size, d_model, layers, lr, name)
    (50, 256, 512, 16, 1e-4, "sweep1_baseline"),
    (50, 256, 1024, 32, 1e-4, "sweep2_scale"),
    (100, 512, 1024, 32, 5e-5, "sweep3_production"),
    (100, 512, 1024, 48, 5e-5, "sweep4_deep"),
    (200, 256, 1024, 32, 1e-5, "sweep5_finetune"),
]

# Which sweeps to run (edit this to skip)
RUN_SWEEPS = [1, 2, 3]  # Run sweeps 1-3

#@title 5. Automated Sweep Runner
import subprocess
import json

results = {}
!mkdir -p models logs

for idx, (epochs, batch, d_model, layers, lr, name) in enumerate(SWEEP_CONFIGS):
    sweep_num = idx + 1
    if sweep_num not in RUN_SWEEPS:
        print(f"⏭ Skipping Sweep {sweep_num}: {name}")
        continue
    
    print(f"\n{'='*60}")
    print(f"🚀 SWEEP {sweep_num}: {name}")
    print(f"   Epochs: {epochs}, Batch: {batch}, d_model: {d_model}, Layers: {layers}, LR: {lr}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run training
    cmd = f"""python intelligence/train_condor_brain.py \
        --local-data data/processed/mamba_institutional_1m.csv \
        --epochs {epochs} \
        --batch-size {batch} \
        --d-model {d_model} \
        --layers {layers} \
        --lr {lr} \
        2>&1 | tee logs/{name}.log"""
    
    !{cmd}
    
    elapsed = time.time() - start_time
    
    # Save model with sweep name
    !cp models/condor_brain.pth models/condor_brain_{name}.pth
    
    # Extract final val loss from log
    try:
        with open(f"logs/{name}.log", 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Val:" in line:
                    val_loss = float(line.split("Val:")[1].split()[0])
                    break
    except:
        val_loss = float('inf')
    
    results[name] = {
        'epochs': epochs,
        'batch_size': batch,
        'd_model': d_model,
        'layers': layers,
        'lr': lr,
        'val_loss': val_loss,
        'time_min': elapsed / 60
    }
    
    print(f"\n✅ Sweep {sweep_num} complete: val_loss={val_loss:.4f}, time={elapsed/60:.1f}min\n")

#@title 6. Results Summary
import pandas as pd

if results:
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('val_loss')
    print("\n📊 SWEEP RESULTS (sorted by val_loss):")
    print(df_results.to_string())
    
    # Save results
    df_results.to_csv('logs/sweep_results.csv')
    
    # Best model
    best = df_results.index[0]
    print(f"\n🏆 BEST MODEL: {best} (val_loss={df_results.loc[best, 'val_loss']:.4f})")
    
    # Copy best to main model
    !cp models/condor_brain_{best}.pth models/condor_brain.pth
    print(f"✅ Copied {best} to models/condor_brain.pth")

#@title 7. Download All Models
from google.colab import files

# Zip all models together
!zip -r all_sweep_models.zip models/

# Download the complete package
files.download('all_sweep_models.zip')
files.download('logs/sweep_results.csv')

print("✅ Downloaded all sweep models + results CSV")
