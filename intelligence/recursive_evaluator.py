
import torch

@torch.jit.script
def evaluate_predicates_recursive(
    data: torch.Tensor,
    active_params: torch.Tensor,
    active_importance: torch.Tensor,
    max_active: int = 128
) -> torch.Tensor:
    """
    Evaluate discovered predicates using RECURSIVE 'Sets of Sets' logic.
    
    Each pass allows predicates to reference outputs of the previous pass.
    Depth 0: Raw Data -> Predicates
    Depth 1: (Raw + Depth 0) -> Predicates (e.g., AND/OR logic)
    Depth 3: Deep nested logic (Sets of Sets)
    
    Args:
        data: Input tensor (Batch, SeqLen, N_Fields)
        active_params: Discovered rules (TopK, ParamDim)
        active_importance: Importance scores (TopK,)
        max_active: Limit on output features (output dimension K)
        
    Returns:
        Tensor (Batch, SeqLen, TopK)
    """
    # 1. Setup
    batch, seq_len, n_fields_raw = data.shape
    device = data.device
    top_k = active_params.shape[0]

    # --- FP32 Internal Enforcement ---
    working_data = data.to(torch.float32)
    working_eps = 1e-6 

    # Unpack Indices
    l_f1 = active_params[:, 0].long()
    l_lb1 = active_params[:, 1].long()
    l_op = active_params[:, 2].long()
    l_f2 = active_params[:, 3].long()
    l_lb2 = active_params[:, 4].long()
    
    cmp_op = active_params[:, 5].long()
    
    r_f1 = active_params[:, 6].long()
    r_lb1 = active_params[:, 7].long()
    r_op = active_params[:, 8].long()
    r_f2 = active_params[:, 9].long()
    r_lb2 = active_params[:, 10].long()
    
    # Template & Threshold
    has_templates = active_params.shape[1] >= 13
    if has_templates:
        templates = active_params[:, 11].long()
        thresholds = active_params[:, 12]
    else:
        templates = torch.zeros(top_k, device=device, dtype=torch.long)
        thresholds = torch.zeros(top_k, device=device)

    # -------------------------------------------------------------------------
    # RECURSIVE EVALUATION (The "Gödel Upgrade")
    # -------------------------------------------------------------------------
    recursion_depth = 4
    
    # Initialize state: (B, S, K)
    # At step 0, virtual inputs are 0.
    current_preds = torch.zeros(batch, seq_len, top_k, device=device, dtype=torch.float32)
    
    # Pre-compute indices for gathering (1, S, K)
    t_seq = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    
    # Expand indices to (B, S, K) for use in loop
    l_f1_exp = l_f1.view(1,1,top_k).expand(batch, seq_len, top_k)
    l_lb1_exp = l_lb1.view(1,1,top_k)
    l_f2_exp = l_f2.view(1,1,top_k).expand(batch, seq_len, top_k)
    l_lb2_exp = l_lb2.view(1,1,top_k)
    
    r_f1_exp = r_f1.view(1,1,top_k).expand(batch, seq_len, top_k)
    r_lb1_exp = r_lb1.view(1,1,top_k)
    r_f2_exp = r_f2.view(1,1,top_k).expand(batch, seq_len, top_k)
    r_lb2_exp = r_lb2.view(1,1,top_k)

    # LOOP
    for step in range(recursion_depth):
        
        # NOTE: TorchScript doesn't like defining functions inside loops if variables are captured.
        # But we can write the logic directly or use a helper outside.
        # Writing inline for JIT safety.
        
        
        # --- Left Side ---
        # Get Vals (Inline)
        f_ids_l = l_f1_exp[0,0,:]
        is_raw_l = f_ids_l < n_fields_raw
        raw_ids_l = torch.where(is_raw_l, f_ids_l, torch.zeros_like(f_ids_l))
        raw_val_l = working_data.index_select(2, raw_ids_l)
        slot_ids_l = f_ids_l - n_fields_raw
        active_idx_l = slot_ids_l % top_k
        safe_idx_l = active_idx_l.clamp(min=0, max=top_k-1)
        virt_val_l = current_preds.index_select(2, safe_idx_l)
        val_l = torch.where(is_raw_l.view(1,1,top_k), raw_val_l, virt_val_l)
        t_gather_l = (t_seq - l_lb1_exp).clamp(min=0).expand(batch, seq_len, top_k)
        lhs = torch.gather(val_l, 1, t_gather_l)
        
        cmp_mask_l = (l_op > 0).view(1,1,top_k)
        if cmp_mask_l.any():
             # Get Vals for Op
            f_ids_l2 = l_f2_exp[0,0,:]
            is_raw_l2 = f_ids_l2 < n_fields_raw
            raw_ids_l2 = torch.where(is_raw_l2, f_ids_l2, torch.zeros_like(f_ids_l2))
            raw_val_l2 = working_data.index_select(2, raw_ids_l2)
            slot_ids_l2 = f_ids_l2 - n_fields_raw
            active_idx_l2 = slot_ids_l2 % top_k
            safe_idx_l2 = active_idx_l2.clamp(min=0, max=top_k-1)
            virt_val_l2 = current_preds.index_select(2, safe_idx_l2)
            val_l2 = torch.where(is_raw_l2.view(1,1,top_k), raw_val_l2, virt_val_l2)
            t_gather_l2 = (t_seq - l_lb2_exp).clamp(min=0).expand(batch, seq_len, top_k)
            v2 = torch.gather(val_l2, 1, t_gather_l2)
            
            denom = v2 + working_eps
            op_v = l_op.view(1,1,top_k)
            lhs = torch.where(op_v==1, lhs+v2, lhs)
            lhs = torch.where(op_v==2, lhs-v2, lhs)
            lhs = torch.where(op_v==3, lhs*v2, lhs)
            lhs = torch.where(op_v==4, lhs/denom, lhs)
            
        lhs = torch.clamp(lhs, -1e4, 1e4)

        # --- Right Side ---
        f_ids_r = r_f1_exp[0,0,:]
        is_raw_r = f_ids_r < n_fields_raw
        raw_ids_r = torch.where(is_raw_r, f_ids_r, torch.zeros_like(f_ids_r))
        raw_val_r = working_data.index_select(2, raw_ids_r)
        slot_ids_r = f_ids_r - n_fields_raw
        active_idx_r = slot_ids_r % top_k
        safe_idx_r = active_idx_r.clamp(min=0, max=top_k-1)
        virt_val_r = current_preds.index_select(2, safe_idx_r)
        val_r = torch.where(is_raw_r.view(1,1,top_k), raw_val_r, virt_val_r)
        t_gather_r = (t_seq - r_lb1_exp).clamp(min=0).expand(batch, seq_len, top_k)
        rhs = torch.gather(val_r, 1, t_gather_r)
        
        is_thresh = (templates == 4).view(1,1,top_k)
        rhs = torch.where(is_thresh, thresholds.view(1,1,top_k).float(), rhs)
        
        cmp_mask_r = (r_op > 0).view(1,1,top_k) & (~is_thresh)
        if cmp_mask_r.any():
             # Get Vals for Op
            f_ids_r2 = r_f2_exp[0,0,:]
            is_raw_r2 = f_ids_r2 < n_fields_raw
            raw_ids_r2 = torch.where(is_raw_r2, f_ids_r2, torch.zeros_like(f_ids_r2))
            raw_val_r2 = working_data.index_select(2, raw_ids_r2)
            slot_ids_r2 = f_ids_r2 - n_fields_raw
            active_idx_r2 = slot_ids_r2 % top_k
            safe_idx_r2 = active_idx_r2.clamp(min=0, max=top_k-1)
            virt_val_r2 = current_preds.index_select(2, safe_idx_r2)
            val_r2 = torch.where(is_raw_r2.view(1,1,top_k), raw_val_r2, virt_val_r2)
            t_gather_r2 = (t_seq - r_lb2_exp).clamp(min=0).expand(batch, seq_len, top_k)
            v2 = torch.gather(val_r2, 1, t_gather_r2)

            denom = v2 + working_eps
            op_v = r_op.view(1,1,top_k)
            rhs = torch.where(op_v==1, rhs+v2, rhs)
            rhs = torch.where(op_v==2, rhs-v2, rhs)
            rhs = torch.where(op_v==3, rhs*v2, rhs)
            rhs = torch.where(op_v==4, rhs/denom, rhs)
            
        rhs = torch.clamp(rhs, -1e4, 1e4)

        # Compare
        diff = lhs - rhs
        steepness = 10.0
        op_view = cmp_op.view(1, 1, top_k)
        
        sig_pos = torch.sigmoid(steepness * diff)
        sig_neg = torch.sigmoid(-steepness * diff)
        eq_val = torch.exp(-steepness * diff.abs())
        
        res = torch.zeros_like(diff)
        res = torch.where((op_view == 0) | (op_view == 2), sig_pos, res)
        res = torch.where((op_view == 1) | (op_view == 3), sig_neg, res)
        res = torch.where(op_view == 4, eq_val, res)
        
        current_preds = res

    # Final Result
    result = current_preds

    # Logic: Mask if lookback invalid
    # We use active params lookback, ignoring recursive depth for simplicity of masking
    max_lb_per_k = torch.max(torch.stack([l_lb1, r_lb1, l_lb2, r_lb2]), dim=0)[0]
    max_lb_view = max_lb_per_k.view(1, 1, top_k)
    time_mask = t_seq >= max_lb_view
    
    result = result * time_mask.float()
    
    # Importance weighting
    result = result * active_importance.view(1, 1, top_k).to(torch.float32)
    
    return result

def trace_recursive_rule(
    rule_idx: int,
    active_params: torch.Tensor,
    feature_names: list,
    pred_names: list,
    depth: int = 0,
    max_depth: int = 4
) -> str:
    """
    Recursively unroll a rule definition (Slot ID -> Logic string).
    
    Args:
        rule_idx: Index in active_params (0..top_k-1)
        active_params: Tensor of shape (TopK, ParamDim)
        feature_names: List of raw feature names
        pred_names: List of strings for simple predicates (optional cache)
        depth: Current recursion depth
        
    Returns:
        String representation like "((Open > Close) > 0.5)"
    """
    if depth > max_depth:
        return "..."
        
    # Unpack
    p = active_params[rule_idx].cpu().numpy()
    n_fields = len(feature_names)
    
    # Helpers
    def get_atom_str(f_idx, lb_idx, arith_op, f2_idx, lb2_idx):
        f_idx = int(f_idx)
        
        # Raw Field
        if f_idx < n_fields:
            name = feature_names[f_idx]
            if lb_idx > 0: name += f"[-{int(lb_idx)}]"
            base = name
        else:
            # Virtual Field (Recursive)
            slot_id = f_idx - n_fields
            # Assuming robust mapping used in eval: slot_id % top_k
            # We need to know the 'top_k' count here.
            top_k = active_params.shape[0]
            recurse_idx = slot_id % top_k
            
            # Recurse!
            base = f"{{{trace_recursive_rule(recurse_idx, active_params, feature_names, pred_names, depth+1)}}}"
            
        # Arith Op
        op = int(arith_op)
        if op > 0:
            f2_idx = int(f2_idx)
            if f2_idx < n_fields:
                n2 = feature_names[f2_idx]
                if lb2_idx > 0: n2 += f"[-{int(lb2_idx)}]"
                b2 = n2
            else:
                s2 = f2_idx - n_fields
                top_k = active_params.shape[0]
                r2 = s2 % top_k
                b2 = f"{{{trace_recursive_rule(r2, active_params, feature_names, pred_names, depth+1)}}}"
            
            syms = ["", "+", "-", "*", "/"]
            if op < len(syms):
                base = f"({base} {syms[op]} {b2})"
                
        return base

    left = get_atom_str(p[0], p[1], p[2], p[3], p[4])
    
    # Template/Threshold
    if p.shape[0] >= 13:
        template = int(p[11])
        if template == 4: # THRESHOLD
            right = f"{p[12]:.3f}"
        else:
            right = get_atom_str(p[6], p[7], p[8], p[9], p[10])
    else:
        right = get_atom_str(p[6], p[7], p[8], p[9], p[10])
        
    op_syms = [">", "<", ">=", "<=", "=="]
    op = int(p[5])
    sym = op_syms[op] if 0 <= op < len(op_syms) else "?"
    
    return f"{left} {sym} {right}"

