import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    args = ap.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        print(f"[FAIL] model not found: {model_path}")
        sys.exit(1)
    if not data_path.exists():
        print(f"[FAIL] data not found: {data_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "feature_cols" not in ckpt:
        print("[FAIL] checkpoint missing feature_cols")
        sys.exit(1)
    feature_cols = list(ckpt["feature_cols"])

    df = pd.read_csv(data_path, nrows=5)
    data_cols = list(df.columns)

    missing = [c for c in feature_cols if c not in data_cols]
    extra = [c for c in data_cols if c not in feature_cols]

    if missing:
        print(f"[FAIL] missing in data: {missing}")
    if extra:
        print(f"[WARN] extra columns in data (ignored by model): {extra[:20]}")

    if not missing:
        # Only check ordering if all features exist
        data_order = [c for c in data_cols if c in feature_cols]
        if data_order != feature_cols:
            print("[FAIL] feature order mismatch between data and checkpoint")
            sys.exit(1)

    print("[OK] feature schema alignment checks complete")


if __name__ == "__main__":
    main()
