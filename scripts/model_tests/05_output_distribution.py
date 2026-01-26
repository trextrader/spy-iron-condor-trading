import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from intelligence.condor_brain import CondorBrain
from intelligence.canonical_feature_registry import apply_semantic_nan_fill, get_neutral_fill_value_v22


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    ap.add_argument("--sample-n", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; Mamba requires GPU for forward.")
        sys.exit(0)

    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        print(f"[FAIL] model not found: {model_path}")
        sys.exit(1)
    if not data_path.exists():
        print(f"[FAIL] data not found: {data_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        print("[FAIL] checkpoint not a dict")
        sys.exit(1)
    feature_cols = list(ckpt.get("feature_cols", []))
    input_dim = ckpt.get("input_dim", None)
    if not feature_cols or input_dim is None:
        print("[FAIL] checkpoint missing feature_cols/input_dim")
        sys.exit(1)

    df = pd.read_csv(data_path)
    missing = [c for c in feature_cols if c not in df.columns]
    for col in missing:
        df[col] = get_neutral_fill_value_v22(col)

    X_np = df[feature_cols].values.astype(np.float32)
    X_np = np.where(np.isfinite(X_np), X_np, np.nan)
    X_np = apply_semantic_nan_fill(X_np, feature_cols)

    if "median" in ckpt and "mad" in ckpt:
        mu = np.asarray(ckpt["median"], dtype=np.float32).squeeze()
        mad = np.asarray(ckpt["mad"], dtype=np.float32).squeeze()
        if mu.ndim == 1 and mad.ndim == 1 and mu.shape[0] == X_np.shape[1]:
            mad = np.maximum(mad, 1e-6)
            X_np = (X_np - mu) / (1.4826 * mad)
    X_np = np.clip(X_np, -10.0, 10.0)

    max_start = len(X_np) - args.seq_len
    if max_start <= 0:
        print("[FAIL] dataset too small for seq_len")
        sys.exit(1)

    rng = np.random.RandomState(42)
    idx = rng.choice(max_start, size=min(args.sample_n, max_start), replace=False)
    seqs = np.stack([X_np[i : i + args.seq_len] for i in idx], axis=0)

    model = CondorBrain(
        d_model=512,
        n_layers=12,
        input_dim=input_dim,
        use_vol_gated_attn=True,
        use_topk_moe=True,
        moe_n_experts=3,
        moe_k=1,
        use_diffusion=True,
    ).cuda()
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()

    outputs = []
    for start in range(0, len(seqs), args.batch):
        end = min(start + args.batch, len(seqs))
        xb = torch.tensor(seqs[start:end], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            out = model(xb)[0]
        outputs.append(out.detach().cpu().numpy())
        del xb
    pol = np.concatenate(outputs, axis=0)

    stats = {
        "call_off": (pol[:, 0].mean(), pol[:, 0].std()),
        "put_off": (pol[:, 1].mean(), pol[:, 1].std()),
        "width": (pol[:, 2].mean(), pol[:, 2].std()),
        "te": (pol[:, 3].mean(), pol[:, 3].std()),
        "prob_profit": (pol[:, 4].mean(), pol[:, 4].std()),
        "expected_roi": (pol[:, 5].mean(), pol[:, 5].std()),
        "max_loss_pct": (pol[:, 6].mean(), pol[:, 6].std()),
        "confidence": (pol[:, 7].mean(), pol[:, 7].std()),
    }
    print("[OK] output distribution (mean, std):")
    for k, (m, s) in stats.items():
        print(f"  {k:12s}  mean={m:.6f}  std={s:.6f}")


if __name__ == "__main__":
    main()
