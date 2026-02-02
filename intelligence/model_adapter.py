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
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    config = ckpt.get("model_config", ckpt.get("config", {})) or {}
    
    # --- ARCHAEOLOGICAL INFERENCE ---
    # We prioritize state_dict shapes over config to avoid OOM/RuntimeErrors on mismatch
    
    # 1. Architecture: CDE vs Mamba
    use_cde = _infer_use_cde_from_state_dict(state_dict)
    if use_cde is None:
        use_cde = bool(config.get("use_cde", True))

    # 2. Predicate Discovery
    use_pred = any(k.startswith("predicate_selector") for k in state_dict.keys())
    
    # Initialize defaults
    d_model = config.get("d_model", 1024)
    n_layers = int(config.get("n_layers", 32))
    max_active = config.get("max_active_predicates", 512)
    n_slots = config.get("n_predicate_slots", 2048)
    input_dim = config.get("input_dim", INPUT_DIM_V22)

    # 3. Precise Hyperparam Reconstruction
    if use_pred:
        # predicate_embeddings: [n_slots, d_embed]
        p_embed = state_dict.get("predicate_selector.predicate_embeddings")
        if p_embed is not None:
            n_slots = p_embed.shape[0]
            # d_embed = p_embed.shape[1] (not used for CondorBrain init, but good to know)
        
        # left_field1.weight: [n_fields + n_slots, d_embed * 2]
        # This is the gold standard for recovering original input_dim
        lf1 = state_dict.get("predicate_selector.left_field1.weight")
        if lf1 is not None:
            input_dim = lf1.shape[0] - n_slots
        
        # logic_gates: [max_active, max_active, 4]
        gates = state_dict.get("predicate_combiner.logic_gates")
        if gates is not None:
            max_active = gates.shape[0]
    else:
        # Non-predicate model: check input_proj or backbone encoder
        if "input_proj.weight" in state_dict:
            # Mamba: [d_model, input_dim]
            input_dim = state_dict["input_proj.weight"].shape[1]
        elif "cde_backbone.encoder.weight" in state_dict:
            # CDE: [hidden_dim, input_dim + 2*max_active]
            # If no predicates, backbone_in_dim == input_dim
            input_dim = state_dict["cde_backbone.encoder.weight"].shape[1]

    # 4. d_model
    if "norm.weight" in state_dict:
        d_model = state_dict["norm.weight"].shape[0]
    elif "cde_backbone.encoder.weight" in state_dict:
        d_model = state_dict["cde_backbone.encoder.weight"].shape[0]

    # 5. n_layers
    if not use_cde:
        mamba_layers = [k for k in state_dict.keys() if k.startswith("layers.") and ".mamba." in k]
        if mamba_layers:
            n_layers = max([int(k.split(".")[1]) for k in mamba_layers]) + 1
    else:
        # CDE n_layers refers to vector field blocks if applicable, or stay with config
        pass

    print(f"  [Inferred] Arch={'CDE' if use_cde else 'Mamba'}, d_model={d_model}, layers={n_layers}, input_dim={input_dim}, slots={n_slots}, max_active={max_active}", flush=True)

    model = CondorBrain(
        d_model=d_model,
        n_layers=n_layers,
        input_dim=input_dim,
        use_cde=use_cde,
        use_predicate_discovery=use_pred,
        n_predicate_slots=n_slots,
        max_active_predicates=max_active,
        use_topk_moe=bool(config.get("use_topk_moe", config.get("use_topk", False))),
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    info = ModelInfo(
        name=os.path.basename(ckpt_path).replace(".pth", ""),
        ckpt_path=ckpt_path,
        seq_len=int(ckpt.get("seq_len", 256)),
        use_cde=use_cde,
        d_model=d_model,
        n_layers=n_layers,
        n_params=n_params,
    )
    return ModelAdapter(model, info, device), ckpt
