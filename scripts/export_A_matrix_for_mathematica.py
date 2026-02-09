#!/usr/bin/env python3
"""
Export CondorNet A matrix (dynamics matrix) for Mathematica stability analysis.

Outputs:
  - CSV file(s) of the full A matrix
  - JSON with metadata (dimensions, eigenvalue summary)

Usage:
  python scripts/export_A_matrix_for_mathematica.py --checkpoint models/condor_net_epoch3.pth
  python scripts/export_A_matrix_for_mathematica.py --checkpoint-dir models/checkpoints --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.condor_brain_net import CondorNetV4, AugmentedStateSpec


def load_model_and_extract_A(checkpoint_path: Path, device: str = "cpu") -> tuple:
    """
    Load a CondorNet checkpoint and extract the full A matrix.

    Returns:
        (A_matrix, spec, metadata)
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

    # Infer spec from state_dict keys
    # Look for A_theta.A_hh.weight to get d_h
    d_h = d_v = d_m = d_r = 16  # defaults

    for key in state_dict.keys():
        if "A_theta.A_hh.weight" in key:
            d_h = state_dict[key].shape[0]
        elif "A_theta.A_vv.weight" in key:
            d_v = state_dict[key].shape[0]
        elif "A_theta.A_mm.weight" in key:
            d_m = state_dict[key].shape[0]
        elif "A_theta.A_rr.weight" in key:
            d_r = state_dict[key].shape[0]

    print(f"  Inferred dimensions: d_h={d_h}, d_v={d_v}, d_m={d_m}, d_r={d_r}")

    # Create spec
    spec = AugmentedStateSpec(d_h=d_h, d_v=d_v, d_m=d_m, d_r=d_r)

    # Infer other model params from state_dict
    # Count predicates from predicate layer
    n_predicates = 5  # default
    for key in state_dict.keys():
        if "predicate_layer" in key and "weight" in key:
            n_predicates = state_dict[key].shape[0]
            break

    # Create minimal model to load weights
    model = CondorNetV4(
        spec=spec,
        d_input=64,  # Will be overridden by state_dict
        n_predicates=n_predicates,
        n_sets=32,
        n_super_sets=8,
        d_control=64,
        enforce_sparsity=False,
    )

    # Load state dict (strict=False to handle any mismatches)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  Warning: Partial load - {e}")

    model.eval()

    # Extract full A matrix
    with torch.no_grad():
        A_full = model.A_theta.full_matrix().cpu().numpy()

    print(f"  A matrix shape: {A_full.shape}")

    # Compute basic stability metrics
    eigenvalues = np.linalg.eigvals(A_full)
    max_real_eig = np.max(np.real(eigenvalues))
    spectral_radius = np.max(np.abs(eigenvalues))

    # Gershgorin bound
    row_radii = np.sum(np.abs(A_full), axis=1) - np.abs(np.diag(A_full))
    gershgorin_alpha = -np.max(np.diag(A_full) + row_radii)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "d_h": d_h,
        "d_v": d_v,
        "d_m": d_m,
        "d_r": d_r,
        "d_x": spec.d_x,
        "max_real_eigenvalue": float(max_real_eig),
        "spectral_radius": float(spectral_radius),
        "gershgorin_alpha": float(gershgorin_alpha),
        "is_hurwitz": bool(max_real_eig < 0),
    }

    print(f"  Max Re(λ): {max_real_eig:.6f}")
    print(f"  ρ(A): {spectral_radius:.6f}")
    print(f"  Gershgorin α: {gershgorin_alpha:.6f}")
    print(f"  Hurwitz stable: {metadata['is_hurwitz']}")

    return A_full, spec, metadata


def export_to_csv(A: np.ndarray, output_path: Path):
    """Export A matrix to CSV (Mathematica-friendly)."""
    np.savetxt(output_path, A, delimiter=",", fmt="%.12e")
    print(f"  Saved CSV: {output_path}")


def export_to_json(A: np.ndarray, metadata: dict, output_path: Path):
    """Export A matrix and metadata to JSON."""
    data = {
        "metadata": metadata,
        "A_matrix": A.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON: {output_path}")


def export_to_npy(A: np.ndarray, output_path: Path):
    """Export A matrix to NumPy binary format."""
    np.save(output_path, A)
    print(f"  Saved NPY: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export CondorNet A matrix for Mathematica")
    parser.add_argument("--checkpoint", type=str, help="Path to single checkpoint file")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory with multiple checkpoints")
    parser.add_argument("--all", action="store_true", help="Export all checkpoints in directory")
    parser.add_argument("--output-dir", type=str, default="exports/A_matrices",
                        help="Output directory")
    parser.add_argument("--format", type=str, default="csv",
                        choices=["csv", "json", "npy", "all"],
                        help="Output format (default: csv)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading")

    args = parser.parse_args()

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
            # Just get the latest
            checkpoints = sorted(ckpt_dir.glob("*.pth"))[-1:]
    else:
        # Default: look in models/
        models_dir = Path("models")
        checkpoints = list(models_dir.glob("condor_net_*.pth"))

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoint(s)")

    all_metadata = []

    for ckpt_path in sorted(checkpoints):
        print(f"\n{'='*60}")
        A, spec, metadata = load_model_and_extract_A(ckpt_path, args.device)
        all_metadata.append(metadata)

        # Generate output filename
        stem = ckpt_path.stem

        if args.format in ["csv", "all"]:
            export_to_csv(A, output_dir / f"{stem}_A.csv")

        if args.format in ["json", "all"]:
            export_to_json(A, metadata, output_dir / f"{stem}_A.json")

        if args.format in ["npy", "all"]:
            export_to_npy(A, output_dir / f"{stem}_A.npy")

    # Summary report
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary_path = output_dir / "stability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # Print Mathematica import instructions
    print(f"""
{'='*60}
MATHEMATICA IMPORT INSTRUCTIONS
{'='*60}

Option 1: Import CSV directly
  A = Import["{output_dir.as_posix()}/{checkpoints[0].stem}_A.csv", "CSV"];
  dt = 1.0;
  stabilityReport[A, dt]

Option 2: Import JSON with metadata
  data = Import["{output_dir.as_posix()}/{checkpoints[0].stem}_A.json", "JSON"];
  A = data["A_matrix"];
  metadata = data["metadata"];
  Print["Epoch: ", metadata["epoch"]];
  dt = 1.0;
  stabilityReport[A, dt]

Option 3: Batch import all matrices
  files = FileNames["*_A.csv", "{output_dir.as_posix()}"];
  As = Import[#, "CSV"] & /@ files;
  dt = 1.0;
  batchStabilityReport[As, dt]

Cloud Upload (if needed):
  cloudObj = CloudExport[A, "CSV", "A_matrix.csv"];
  (* Then use CloudImport in Mathematica Cloud *)
""")


if __name__ == "__main__":
    main()
