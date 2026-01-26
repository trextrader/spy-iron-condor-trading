import argparse
import sys
from pathlib import Path

import torch

from intelligence.condor_brain import CondorBrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; Mamba requires GPU for forward.")
        sys.exit(0)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[FAIL] model not found: {model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        print("[FAIL] checkpoint not a dict")
        sys.exit(1)

    input_dim = ckpt.get("input_dim", None)
    if input_dim is None:
        print("[FAIL] checkpoint missing input_dim")
        sys.exit(1)

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

    x = torch.zeros((args.batch, args.seq_len, input_dim), device="cuda", dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    if not isinstance(out, (tuple, list)):
        print("[FAIL] model forward did not return tuple/list")
        sys.exit(1)
    pol = out[0]
    if pol.ndim != 2 or pol.shape[1] != 8:
        print(f"[FAIL] policy head shape mismatch: {tuple(pol.shape)}")
        sys.exit(1)

    print("[OK] forward contract valid (policy head shape Bx8)")


if __name__ == "__main__":
    main()
