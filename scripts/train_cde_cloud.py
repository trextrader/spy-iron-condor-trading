#!/usr/bin/env python
"""
CondorBrain Neural CDE Training Script for Cloud Environments
(Lightning AI, Google Colab, Kaggle)

Usage:
    # Lightning AI
    lightning run app scripts/train_cde_cloud.py --cloud --gpus 1

    # Colab/Kaggle (cell)
    !python scripts/train_cde_cloud.py --data /content/data.csv --epochs 50

    # Local with larger dataset
    python scripts/train_cde_cloud.py --data data/processed/data_3mil.csv --epochs 100

Features:
    - Auto-detects GPU and enables BF16 for Ampere+
    - Gradient checkpointing for memory efficiency
    - TensorBoard logging
    - Early stopping
    - Per-epoch checkpoints
"""

import os
import sys
import argparse
import subprocess

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def get_optimal_config():
    """Detect hardware and return optimal training config."""
    import torch

    config = {
        "batch_size": 64,
        "d_model": 512,
        "layers": 12,
        "accum_steps": 1,
        "grad_checkpoint": True,
        "materialize_seqs": False,
    }

    if not torch.cuda.is_available():
        print("[CONFIG] CPU mode - using minimal settings")
        config["batch_size"] = 16
        config["d_model"] = 128
        config["layers"] = 4
        return config

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    compute_cap = torch.cuda.get_device_capability(0)

    print(f"[CONFIG] GPU: {gpu_name}")
    print(f"[CONFIG] Memory: {gpu_mem:.1f} GB")
    print(f"[CONFIG] Compute: {compute_cap[0]}.{compute_cap[1]}")

    # T4 (16GB, Colab free)
    if "T4" in gpu_name or gpu_mem < 20:
        config["batch_size"] = 128
        config["d_model"] = 512
        config["layers"] = 12
        config["accum_steps"] = 2
        print("[CONFIG] T4 profile: batch=128, accum=2, d_model=512")

    # A100/V100 (40-80GB, Colab Pro / Lightning)
    elif gpu_mem >= 40:
        config["batch_size"] = 512
        config["d_model"] = 1024
        config["layers"] = 24
        config["accum_steps"] = 1
        config["materialize_seqs"] = True
        config["grad_checkpoint"] = False
        print("[CONFIG] A100 profile: batch=512, d_model=1024, materialize=True")

    # L4/A10 (24GB, Colab Pro)
    elif gpu_mem >= 20:
        config["batch_size"] = 256
        config["d_model"] = 768
        config["layers"] = 16
        print("[CONFIG] L4/A10 profile: batch=256, d_model=768")

    return config


def main():
    parser = argparse.ArgumentParser(description="Train CondorBrain Neural CDE")
    parser.add_argument("--data", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--output", type=str, default="models/condor_brain_cde_prod.pth",
                        help="Output model path")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation, don't train")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    # Step 1: Validate setup
    print("\n" + "="*60)
    print("STEP 1: Pre-Training Validation")
    print("="*60)

    val_result = subprocess.run([
        sys.executable, "intelligence/validate_training_setup.py",
        "--data", args.data,
        "--rows", "50000"
    ])

    if val_result.returncode != 0:
        print("\n[ERROR] Validation failed. Fix issues before training.")
        return 1

    if args.validate_only:
        print("\n[INFO] Validation complete. Exiting (--validate-only)")
        return 0

    # Step 2: Get optimal config
    print("\n" + "="*60)
    print("STEP 2: Hardware Detection")
    print("="*60)

    config = get_optimal_config()

    # Step 3: Build training command
    print("\n" + "="*60)
    print("STEP 3: Starting Training")
    print("="*60)

    cmd = [
        sys.executable, "intelligence/train_condor_brain.py",
        "--local-data", args.data,
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--batch-size", str(config["batch_size"]),
        "--d-model", str(config["d_model"]),
        "--layers", str(config["layers"]),
        "--accum-steps", str(config["accum_steps"]),
        "--lr", str(args.lr),
        "--cde",  # Neural CDE backbone
        "--early-stop",
        "--patience", str(args.patience),
        "--monitor",
        "--monitor-every", "5",
    ]

    if config["grad_checkpoint"]:
        cmd.append("--grad-checkpoint")

    if config["materialize_seqs"]:
        cmd.append("--materialize-seqs")

    if not args.no_tensorboard:
        cmd.extend(["--tensorboard", "--tb-logdir", "runs/cde_training"])

    print(f"[CMD] {' '.join(cmd)}")
    print()

    # Run training
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {args.output}")

        if not args.no_tensorboard:
            print("\nView TensorBoard logs:")
            print("  tensorboard --logdir runs/cde_training --port 6006")

        print("\nNext steps:")
        print("  1. Run interpretability audit:")
        print(f"     python intelligence/audit_cde_interpretability.py --model {args.output} --data {args.data}")
        print("  2. Deploy for live trading")
    else:
        print("\n[ERROR] Training failed with exit code", result.returncode)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
