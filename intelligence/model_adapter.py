# intelligence/model_adapter.py
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from intelligence.condor_brain import CondorBrain
from intelligence.canonical_feature_registry import INPUT_DIM_V22

@dataclass
class ModelInfo:
    name: str
    ckpt_path: str
    seq_len: int
    use_cde: bool
    d_model: int
    n_layers: int
    n_params: int

class ModelAdapter:
    """
    Wraps CondorBrain checkpoints (CDE or Mamba2) into a standard interface:
        forward(x) -> {"y": Tensor[B,H], "pred_logits": Optional[Tensor[B,K]], "extras": dict}
    """
    def __init__(self, model: torch.nn.Module, info: ModelInfo, device: torch.device):
        self.model = model
        self.info = info
        self.device = device

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        x: (B, T, D)
        """
        # --- 2026-02-02 UPDATE: Use explicit return_predicates flag if supported ---
        try:
            out = self.model(x, return_predicates=True)
        except TypeError:
            # Fallback for models without the new flag
            out = self.model(x)

        pred_logits = None
        extras: Dict[str, Any] = {}

        if isinstance(out, tuple):
            # Convention in your codebase: first element is outputs
            y = out[0]
            # If we used return_predicates=True, it's the last element
            pred_logits = out[-1] if torch.is_tensor(out[-1]) and out[-1].ndim == 2 and out[-1].shape[1] > 10 else None
            
            # Heuristic fallback if last item wasn't it or flag failed
            if pred_logits is None:
                for item in out[1:]:
                    if torch.is_tensor(item) and item.ndim == 2 and item.shape[0] == y.shape[0]:
                        if item.shape[1] > 10:
                            pred_logits = item
                            break
            extras["tuple_len"] = len(out)
        else:
            y = out

        return {"y": y, "pred_logits": pred_logits, "extras": extras}

def _infer_use_cde_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[bool]:
    keys = list(state_dict.keys())
    # Heuristics: adjust based on your actual module names if needed.
    if any("cde" in k.lower() or "vector_field" in k.lower() for k in keys):
        return True
    if any("mamba" in k.lower() or "ssm" in k.lower() or "selective_scan" in k.lower() for k in keys):
        return False
    return None

def load_model_any(ckpt_path: str, device: torch.device, input_dim: int = INPUT_DIM_V22) -> Tuple[ModelAdapter, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    seq_len = int(ckpt.get("seq_len", 256))
    config = ckpt.get("model_config", ckpt.get("config", {})) or {}

    d_model = int(config.get("d_model", 128))
    n_layers = int(config.get("n_layers", 2))

    # Determine architecture
    use_cde = config.get("use_cde", None)
    if use_cde is None:
        use_cde = config.get("cde", None)
    if use_cde is None:
        # infer from weights
        state_dict = ckpt.get("state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        inferred = _infer_use_cde_from_state_dict(state_dict)
        use_cde = True if inferred is None else inferred

    model = CondorBrain(
        d_model=d_model,
        n_layers=n_layers,
        input_dim=input_dim,
        use_cde=bool(use_cde),
        use_topk_moe=bool(config.get("use_topk_moe", config.get("use_topk", False))),
    )

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    info = ModelInfo(
        name=os.path.basename(ckpt_path).replace(".pth", ""),
        ckpt_path=ckpt_path,
        seq_len=seq_len,
        use_cde=bool(use_cde),
        d_model=d_model,
        n_layers=n_layers,
        n_params=n_params,
    )
    return ModelAdapter(model, info, device), ckpt
