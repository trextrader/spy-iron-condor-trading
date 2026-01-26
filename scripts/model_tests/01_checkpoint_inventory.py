import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--out", default="", help="Optional JSON output path")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[FAIL] model not found: {model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        print("[FAIL] checkpoint is not a dict")
        sys.exit(1)

    keys = sorted(list(ckpt.keys()))
    feature_cols = ckpt.get("feature_cols", [])
    input_dim = ckpt.get("input_dim", None)
    version = ckpt.get("version", None)
    median = ckpt.get("median", None)
    mad = ckpt.get("mad", None)

    summary = {
        "path": str(model_path),
        "keys": keys,
        "feature_cols_len": len(feature_cols) if feature_cols is not None else None,
        "input_dim": input_dim,
        "version": version,
        "median_shape": list(getattr(median, "shape", [])),
        "mad_shape": list(getattr(mad, "shape", [])),
    }

    print(json.dumps(summary, indent=2))

    required = ["state_dict", "feature_cols", "input_dim", "median", "mad"]
    missing = [k for k in required if k not in ckpt]
    if missing:
        print(f"[FAIL] missing keys: {missing}")
        sys.exit(1)

    if input_dim is not None and feature_cols is not None:
        if int(input_dim) != len(feature_cols):
            print(f"[FAIL] input_dim {input_dim} != feature_cols {len(feature_cols)}")
            sys.exit(1)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] checkpoint inventory complete")


if __name__ == "__main__":
    main()
