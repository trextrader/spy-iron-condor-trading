#!/usr/bin/env python3
"""
Export CondorNet B matrix (input/control matrix) for Mathematica analysis.

Outputs:
  - CSV file(s) of the full B matrix  (Epoch{N}_B_Matrix.csv)
  - JSON with metadata (dimensions, simple norms)
  - NPY (optional)

Usage (Lightning AI / Colab / local):
  python scripts/export_B_matrix_for_mathematica.py                           # auto-find all epochs
  python scripts/export_B_matrix_for_mathematica.py --checkpoint models/condor_net_epoch3.pth
  python scripts/export_B_matrix_for_mathematica.py --checkpoint-dir models/ --all
  python scripts/export_B_matrix_for_mathematica.py --epochs 10               # expect 10 epochs
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path (script-relative, works on Lightning AI / Colab / local)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.condor_brain_net import BlockMatrixB, AugmentedStateSpec


def _infer_epoch_number(path: Path) -> int:
    """Extract epoch number from checkpoint filename."""
    stem = path.stem.lower()
    m = re.search(r'epoch[_]?(\d+)', stem)
    if m:
        return int(m.group(1))
    m = re.search(r'_e(\d+)', stem)
    if m:
        return int(m.group(1))
    return 0


def load_model_and_extract_B(checkpoint_path: Path, device: str = "cpu") -> tuple:
    """
    Load a CondorNet checkpoint and extract the full B matrix.

    Uses BlockMatrixB directly to avoid full model instantiation issues.

    Returns:
        (B_matrix, spec, metadata)
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
    else:
        state_dict = checkpoint
        epoch = "unknown"

    # Infer dimensions from A/B-related keys (fallback to defaults)
    d_h = d_v = d_m = d_r = 16  # defaults

    for key, value in state_dict.items():
        if "A_theta.A_hh.weight" in key:
            d_h = value.shape[0]
        elif "A_theta.A_vv.weight" in key:
            d_v = value.shape[0]
        elif "A_theta.A_mm.weight" in key:
            d_m = value.shape[0]
        elif "A_theta.A_rr.weight" in key:
            d_r = value.shape[0]

    print(f"  Inferred dimensions: d_h={d_h}, d_v={d_v}, d_m={d_m}, d_r={d_r}")

    # Create spec
    spec = AugmentedStateSpec(d_h=d_h, d_v=d_v, d_m=d_m, d_r=d_r)

    # Create BlockMatrixB directly
    B_theta = BlockMatrixB(spec)

    # Extract only the B_theta weights from state_dict
    B_theta_state = {}
    for key, value in state_dict.items():
        if key.startswith("B_theta."):
            # Remove "B_theta." prefix for loading into BlockMatrixB
            new_key = key[8:]  # len("B_theta.") = 8
            B_theta_state[new_key] = value

    # Load weights
    try:
        B_theta.load_state_dict(B_theta_state, strict=False)
        print(f"  Loaded {len(B_theta_state)} B_theta weight tensors")
    except Exception as e:
        print(f"  Warning: Partial load - {e}")

    B_theta.eval()

    # Extract full B matrix
    with torch.no_grad():
        B_full = B_theta.full_matrix().cpu().numpy()

    print(f"  B matrix shape: {B_full.shape}")

    # Simple metadata / norms
    fro_norm = float(np.linalg.norm(B_full, ord="fro"))
    max_abs = float(np.max(np.abs(B_full)))

    metadata = {
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "d_h": d_h,
        "d_v": d_v,
        "d_m": d_m,
        "d_r": d_r,
        "d_x": spec.d_x,
        "B_shape": list(B_full.shape),
        "frobenius_norm": fro_norm,
        "max_abs_entry": max_abs,
    }

    print(f"  ||B||_F: {fro_norm:.6e}")
    print(f"  max |B_ij|: {max_abs:.6e}")

    return B_full, spec, metadata


def export_to_csv(B: np.ndarray, output_path: Path):
    """Export B matrix to CSV (Mathematica-friendly)."""
    np.savetxt(output_path, B, delimiter=",", fmt="%.12e")
    print(f"  Saved CSV: {output_path}")


def export_to_json(B: np.ndarray, metadata: dict, output_path: Path):
    """Export B matrix and metadata to JSON."""
    data = {
        "metadata": metadata,
        "B_matrix": B.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON: {output_path}")


def export_to_npy(B: np.ndarray, output_path: Path):
    """Export B matrix to NumPy binary format."""
    np.save(output_path, B)
    print(f"  Saved NPY: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export CondorNet B matrix for Mathematica")
    parser.add_argument("--checkpoint", type=str, help="Path to single checkpoint file")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory with multiple checkpoints")
    parser.add_argument("--all", action="store_true", help="Export all checkpoints in directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: models/b_matrix)")
    parser.add_argument("--format", type=str, default="csv",
                        choices=["csv", "json", "npy", "all"],
                        help="Output format (default: csv)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Expected number of epochs (for validation)")

    args = parser.parse_args()

    # Default output mirrors A matrix convention: models/b_matrix/
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "models" / "b_matrix"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = []

    if args.checkpoint:
        checkpoints.append(Path(args.checkpoint))
    elif args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        if args.all:
            checkpoints = list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.pt"))
        else:
            checkpoints = sorted(ckpt_dir.glob("*.pth"))[-1:]
    else:
        # Auto-discover: check models/ dir relative to project root
        models_dir = PROJECT_ROOT / "models"
        found = (list(models_dir.glob("condor_net_epoch*.pth"))
                 + list(models_dir.glob("condor_net_best*.pth"))
                 + list(models_dir.glob("condor_net_*.pth")))
        # Deduplicate while preserving order
        seen = set()
        for cp in found:
            if cp not in seen:
                seen.add(cp)
                checkpoints.append(cp)

    if not checkpoints:
        print(f"No checkpoints found in {PROJECT_ROOT / 'models'}")
        print(f"  Looked for: condor_net_epoch*.pth, condor_net_best*.pth, condor_net_*.pth")
        return

    # Sort by epoch number
    checkpoints = sorted(checkpoints, key=_infer_epoch_number)

    print(f"Found {len(checkpoints)} checkpoint(s)")
    if args.epochs and len(checkpoints) != args.epochs:
        print(f"  Warning: Expected {args.epochs} epochs, found {len(checkpoints)}")

    all_metadata = []

    for ckpt_path in checkpoints:
        epoch_num = _infer_epoch_number(ckpt_path)
        print(f"\n{'='*60}")
        B, spec, metadata = load_model_and_extract_B(ckpt_path, args.device)
        all_metadata.append(metadata)

        # Use Epoch{N}_B_Matrix naming convention (matches A matrix pattern)
        if epoch_num > 0:
            base_name = f"Epoch{epoch_num}_B_Matrix"
        else:
            base_name = f"{ckpt_path.stem}_B_Matrix"

        if args.format in ["csv", "all"]:
            export_to_csv(B, output_dir / f"{base_name}.csv")

        if args.format in ["json", "all"]:
            export_to_json(B, metadata, output_dir / f"{base_name}.json")

        if args.format in ["npy", "all"]:
            export_to_npy(B, output_dir / f"{base_name}.npy")

    # Summary report
    print(f"\n{'='*60}")
    print(f"SUMMARY - {len(all_metadata)} B matrices exported")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")

    summary_path = output_dir / "B_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"  Summary: {summary_path}")

    # Print import hints
    print(f"""
{'='*60}
IMPORT INSTRUCTIONS
{'='*60}

Python:
  import numpy as np
  B = np.loadtxt("{output_dir.as_posix()}/Epoch1_B_Matrix.csv", delimiter=",")

Batch (all epochs):
  from pathlib import Path
  Bs = [np.loadtxt(f, delimiter=",") for f in sorted(Path("{output_dir.as_posix()}").glob("Epoch*_B_Matrix.csv"))]

Mathematica:
  B = Import["{output_dir.as_posix()}/Epoch1_B_Matrix.csv", "CSV"];
  files = FileNames["Epoch*_B_Matrix.csv", "{output_dir.as_posix()}"];
  Bs = Import[#, "CSV"] & /@ files;
""")

if __name__ == "__main__":
    main()
