"""
Pre-Training Validation Script for CondorBrain Neural CDE

Run this BEFORE training to verify:
1. Dataset integrity (features, IVR variation, no NaN)
2. Model architecture compatibility
3. Memory requirements estimation
4. CDE backbone availability

Usage:
    python intelligence/validate_training_setup.py --data path/to/data.csv --rows 100000

For Lightning AI / Colab / Kaggle:
    !python intelligence/validate_training_setup.py --data data.csv --rows 0
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())


def validate_dataset(data_path: str, max_rows: int = 0) -> dict:
    """Validate dataset structure and quality."""
    print(f"\n{'='*60}")
    print("DATASET VALIDATION")
    print(f"{'='*60}")

    results = {"passed": True, "warnings": [], "errors": []}

    # Load dataset
    print(f"Loading: {data_path}")
    if max_rows > 0:
        df = pd.read_csv(data_path, nrows=max_rows)
    else:
        df = pd.read_csv(data_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Check required feature columns
    from intelligence.canonical_feature_registry import FEATURE_COLS_V22

    missing = [c for c in FEATURE_COLS_V22 if c not in df.columns]
    if missing:
        results["errors"].append(f"Missing {len(missing)} required columns: {missing[:5]}...")
        results["passed"] = False
    else:
        print(f"[OK] All 52 V2.2 feature columns present")

    # Check IVR variation
    if 'ivr' in df.columns:
        ivr = df['ivr'].values
        ivr_std = np.nanstd(ivr)
        ivr_unique = len(np.unique(ivr[~np.isnan(ivr)]))

        print(f"\nIVR Statistics:")
        print(f"  Min: {np.nanmin(ivr):.4f}")
        print(f"  Max: {np.nanmax(ivr):.4f}")
        print(f"  Std: {ivr_std:.4f}")
        print(f"  Unique: {ivr_unique}")

        if ivr_std < 1.0:
            results["errors"].append("IVR has near-zero variance - regime detection will fail!")
            results["passed"] = False
        elif ivr_std < 10.0:
            results["warnings"].append("IVR has low variance - regime detection may be poor")
        else:
            print(f"[OK] IVR has good variation (std={ivr_std:.2f})")

    # Check for NaN/Inf
    feature_df = df[FEATURE_COLS_V22] if not missing else df[[c for c in FEATURE_COLS_V22 if c in df.columns]]
    nan_counts = feature_df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        nan_pct = (nan_cols.sum() / (len(df) * len(feature_df.columns))) * 100
        print(f"\nNaN found in {len(nan_cols)} columns ({nan_pct:.2f}% of data)")
        if nan_pct > 10:
            results["warnings"].append(f"High NaN rate ({nan_pct:.1f}%) - check data quality")
        else:
            print(f"[OK] NaN rate acceptable ({nan_pct:.2f}%)")
    else:
        print(f"[OK] No NaN values in feature columns")

    # Memory estimation
    bytes_per_row = len(FEATURE_COLS_V22) * 4 + 10 * 4  # features + targets (float32)
    total_bytes = len(df) * bytes_per_row
    gpu_mem_gb = total_bytes / 1e9 * 3  # 3x for model + gradients

    print(f"\nMemory Estimation:")
    print(f"  Data size: {total_bytes / 1e9:.2f} GB")
    print(f"  Est. GPU needed: {gpu_mem_gb:.1f} GB (with gradients)")

    if gpu_mem_gb > 40:
        results["warnings"].append(f"Large memory requirement ({gpu_mem_gb:.1f}GB) - use gradient checkpointing")

    return results


def validate_model() -> dict:
    """Validate model can be instantiated."""
    print(f"\n{'='*60}")
    print("MODEL VALIDATION")
    print(f"{'='*60}")

    results = {"passed": True, "warnings": [], "errors": []}

    try:
        from intelligence.condor_brain import CondorBrain, HAS_CDE

        if not HAS_CDE:
            results["errors"].append("Neural CDE not available - check neural_cde.py")
            results["passed"] = False
            return results

        print(f"[OK] HAS_CDE = True")

        # Try instantiating model
        import torch
        model = CondorBrain(
            d_model=128,  # Small for validation
            n_layers=2,
            input_dim=52,
            use_cde=True,
            use_topk_moe=False
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model instantiated: {n_params:,} parameters")

        # Test forward pass
        x = torch.randn(2, 256, 52)
        with torch.no_grad():
            out = model(x, return_regime=True)

        if isinstance(out, tuple):
            outputs, regime = out[0], out[1]
            print(f"[OK] Forward pass: outputs shape = {outputs.shape}")
            print(f"[OK] Output heads: {outputs.shape[1]} (expected 10)")

            if outputs.shape[1] != 10:
                results["errors"].append(f"Expected 10 output heads, got {outputs.shape[1]}")
                results["passed"] = False

    except Exception as e:
        results["errors"].append(f"Model validation failed: {e}")
        results["passed"] = False

    return results


def validate_cuda() -> dict:
    """Validate CUDA availability and capabilities."""
    print(f"\n{'='*60}")
    print("CUDA VALIDATION")
    print(f"{'='*60}")

    results = {"passed": True, "warnings": [], "errors": []}

    import torch

    if not torch.cuda.is_available():
        results["warnings"].append("CUDA not available - training will be slow on CPU")
        print("[WARN] CUDA not available")
        return results

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    compute_cap = torch.cuda.get_device_capability(0)

    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_mem:.1f} GB")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

    # BF16 support
    use_bf16 = compute_cap[0] >= 8
    print(f"BF16 Support: {'Yes' if use_bf16 else 'No (will use FP16)'}")

    if gpu_mem < 8:
        results["warnings"].append("Low GPU memory - use small batch size and gradient checkpointing")
    elif gpu_mem >= 40:
        print("[OK] High-memory GPU - can use large batches and --materialize-seqs")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate training setup")
    parser.add_argument("--data", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--rows", type=int, default=100000, help="Max rows to check (0=all)")
    args = parser.parse_args()

    print("="*60)
    print("CONDORBRAIN PRE-TRAINING VALIDATION")
    print("="*60)

    all_results = []

    # 1. Dataset
    if os.path.exists(args.data):
        all_results.append(validate_dataset(args.data, args.rows))
    else:
        print(f"\n[ERROR] Data file not found: {args.data}")
        all_results.append({"passed": False, "errors": ["Data file not found"]})

    # 2. Model
    all_results.append(validate_model())

    # 3. CUDA
    all_results.append(validate_cuda())

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    all_passed = all(r["passed"] for r in all_results)
    all_warnings = [w for r in all_results for w in r.get("warnings", [])]
    all_errors = [e for r in all_results for e in r.get("errors", [])]

    if all_errors:
        print("\nERRORS:")
        for e in all_errors:
            print(f"  [X] {e}")

    if all_warnings:
        print("\nWARNINGS:")
        for w in all_warnings:
            print(f"  [!] {w}")

    if all_passed:
        print("\n" + "="*60)
        print("VALIDATION PASSED - Ready to train!")
        print("="*60)
        print("\nSuggested training command:")
        print(f"  python intelligence/train_condor_brain.py \\")
        print(f"    --local-data {args.data} \\")
        print(f"    --epochs 50 \\")
        print(f"    --batch-size 256 \\")
        print(f"    --d-model 512 \\")
        print(f"    --layers 12 \\")
        print(f"    --cde \\")
        print(f"    --tensorboard")
        return 0
    else:
        print("\n[FAILED] Fix errors before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
