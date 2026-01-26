import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--tol", type=float, default=1e-4, help="Tolerance for rule feature MAD check")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[FAIL] model not found: {model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        print("[FAIL] checkpoint not a dict")
        sys.exit(1)

    median = np.asarray(ckpt.get("median", None))
    mad = np.asarray(ckpt.get("mad", None))
    feature_cols = ckpt.get("feature_cols", [])
    if median.size == 0 or mad.size == 0:
        print("[FAIL] missing median/mad")
        sys.exit(1)

    median = np.squeeze(median).reshape(-1)
    mad = np.squeeze(mad).reshape(-1)
    if len(median) != len(mad):
        print("[FAIL] median/mad length mismatch")
        sys.exit(1)

    if not np.isfinite(median).all() or not np.isfinite(mad).all():
        print("[FAIL] non-finite values in median/mad")
        sys.exit(1)

    if np.any(mad <= 0):
        print("[FAIL] mad contains non-positive values")
        sys.exit(1)

    # Rule feature normalization protection
    rule_features = ["rule_long_consensus", "rule_short_consensus", "rule_exit_consensus", "rule_block_any"]
    if feature_cols:
        for rf in rule_features:
            if rf in feature_cols:
                idx = feature_cols.index(rf)
                expected_mad = 1.0 / 1.4826
                if abs(median[idx]) > args.tol or abs(mad[idx] - expected_mad) > args.tol:
                    print(f"[FAIL] rule feature norm mismatch: {rf} median={median[idx]} mad={mad[idx]}")
                    sys.exit(1)

    print("[OK] normalization stats look sane")


if __name__ == "__main__":
    main()
