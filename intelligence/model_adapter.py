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
    # Robust Mamba Detection: Mamba2 models have A_log, dt_proj, or layers.X.mixer
    if any(".A_log" in k or ".dt_proj" in k or ".mixer" in k for k in keys):
        return False
    # Layers prefix check for Mamba/Transfomers
    if any(k.startswith("layers.") for k in keys) and not any(k.startswith("cde_backbone") for k in keys):
        return False
    # Neural CDE pillars
    if any("cde_backbone" in k or "vector_field" in k for k in keys):
        return True
    return None

def load_model_any(ckpt_path: str, device: torch.device, input_dim: int = INPUT_DIM_V22) -> Tuple[ModelAdapter, Dict[str, Any]]:
    print(f"  [Load] {os.path.basename(ckpt_path)}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    config = ckpt.get("model_config", ckpt.get("config", {})) or {}
    
    # --- ARCHAEOLOGICAL INFERENCE ---
    it_use_cde = _infer_use_cde_from_state_dict(state_dict)
    if it_use_cde is None:
        it_use_cde = bool(config.get("use_cde", True))

    it_use_pred = any(k.startswith("predicate_selector") for k in state_dict.keys())
    
    # Initialize working variables
    it_d_model = config.get("d_model", 1024)
    it_n_layers = int(config.get("n_layers", 32))
    it_max_active = config.get("max_active_predicates", 512)
    it_n_slots = config.get("n_predicate_slots", 2048)
    it_input_dim = input_dim # passed default

    # Hyperparam Recovery
    if it_use_pred:
        # Recover slots and input_dim from predicate shapes
        p_embed = state_dict.get("predicate_selector.predicate_embeddings")
        if p_embed is not None:
            it_n_slots = p_embed.shape[0]
        
        # left_field1 weights are [n_fields + n_slots, d_embed * 2]
        lf1 = state_dict.get("predicate_selector.left_field1.weight")
        if lf1 is not None and p_embed is not None:
            it_input_dim = lf1.shape[0] - it_n_slots
        
        gates = state_dict.get("predicate_combiner.logic_gates")
        if gates is not None:
            it_max_active = gates.shape[0]
    else:
        # Recover from projection weights
        if "input_proj.weight" in state_dict:
            it_input_dim = state_dict["input_proj.weight"].shape[1]
        elif "cde_backbone.encoder.weight" in state_dict:
            it_input_dim = state_dict["cde_backbone.encoder.weight"].shape[1]

    # Recover d_model
    if "norm.weight" in state_dict:
        it_d_model = state_dict["norm.weight"].shape[0]
    elif "cde_backbone.encoder.weight" in state_dict:
        it_d_model = state_dict["cde_backbone.encoder.weight"].shape[0]

    # Recover n_layers for Mamba
    if not it_use_cde:
        m_layers = [k for k in state_dict.keys() if k.startswith("layers.") and ".A_log" in k]
        if m_layers:
            it_n_layers = max([int(k.split(".")[1]) for k in m_layers]) + 1

    print(f"    - Inferred: Arch={'CDE' if it_use_cde else 'Mamba'}, d_model={it_d_model}, layers={it_n_layers}, input_dim={it_input_dim}, slots={it_n_slots}", flush=True)

    model = CondorBrain(
        d_model=it_d_model,
        n_layers=it_n_layers,
        input_dim=it_input_dim,
        use_cde=it_use_cde,
        use_predicate_discovery=it_use_pred,
        n_predicate_slots=it_n_slots,
        max_active_predicates=it_max_active,
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
