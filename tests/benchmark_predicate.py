import torch
import time
import torch.nn.functional as F

def current_implementation(data, active_params, batch, seq_len, top_k, device):
    # Mocking the loop from the file
    l_f1 = active_params[:, 0].long()
    l_lb1 = active_params[:, 1].long()
    
    # Init output
    result = torch.zeros(batch, seq_len, top_k, device=device)
    
    # Loop over time
    max_lb = int(torch.max(l_lb1).item())
    
    start = time.time()
    for t in range(max_lb, seq_len):
        l_v1 = torch.zeros(batch, top_k, device=device)
        for k in range(top_k):
            t_l1 = t - l_lb1[k].item()
            l_v1[:, k] = data[:, t_l1, l_f1[k].item()]
        
        # Mock logic
        result[:, t, :] = l_v1
        
    torch.cuda.synchronize()
    return time.time() - start

def vectorized_implementation(data, active_params, batch, seq_len, top_k, device):
    l_f1 = active_params[:, 0].long() # (K,)
    l_lb1 = active_params[:, 1].long() # (K,)
    
    start = time.time()
    
    # Create time indices (Seq, 1) - (1, K) = (Seq, K)
    t_range = torch.arange(seq_len, device=device).unsqueeze(1) # (Seq, 1)
    
    # Indices for lookback: (Seq, K)
    # We want T >= LB, so mask invalid later or clamp
    # For now, just clamp to 0 (since data is padded or we ignore early steps)
    t_indices = (t_range - l_lb1.unsqueeze(0)).clamp(min=0) 
    
    # We need to gather from data: (Batch, Seq, Feats)
    # Target indices:
    # Batch: All (implicit broadcasting if we index carefully)
    # Seq: t_indices (Seq, K)
    # Feat: l_f1 (K,) -> Broadcast to (Seq, K)
    
    # Expand field indices to (Seq, K)
    feat_indices = l_f1.unsqueeze(0).expand(seq_len, top_k)
    
    # Prepare gather indices. 
    # Data is (B, S, F). We want result (B, S, K).
    # Since we want to gather specific (t, f) for each k, across all B...
    
    # Method 1: Advanced Indexing
    # data[:, t_indices, feat_indices] -> This might try to broadcast B...
    # Let's index one sample and broadcast? No, B varies.
    
    # Reshaping for gather is complex. 
    # Alternative: Unfold approach manually via advanced indexing
    
    # Let's try direct indexing if B is first dim.
    # data: (B, S, F)
    # We want: output[b, t, k] = data[b, t_idx[t,k], f_idx[k]]
    
    # We can permute data to (S, F, B) maybe?
    # Or just gather on S dim if we fix F?
    
    # Let's simple index:
    # Gather requires indices to match dims.
    
    # F-gathering:
    # First gather the specific fields for each K.
    # data_k = data[..., l_f1]  -> (Batch, Seq, K)  <-- This assumes distinct fields per K
    data_k = data.index_select(2, l_f1) # (Batch, Seq, K)
    
    # Now we just need to shift in time dim.
    # For each k, we want to roll by l_lb1[k].
    
    # But different K have different shifts.
    # We can gather on dim 1 (Seq).
    # Indices must be (Batch, Seq, K).
    t_gather_idx = t_indices.unsqueeze(0).expand(batch, seq_len, top_k)
    
    final = torch.gather(data_k, 1, t_gather_idx) # (Batch, Seq, K)
    
    torch.cuda.synchronize()
    return time.time() - start

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, S, F = 128, 240, 54
    K = 256
    
    data = torch.randn(B, S, F, device=device)
    # Random params: Field, Lookback
    params = torch.zeros(K, 2, device=device)
    params[:, 0] = torch.randint(0, F, (K,))
    params[:, 1] = torch.randint(0, 50, (K,))
    
    print(f"Benchmarking: B={B}, S={S}, K={K}")
    
    # Warmup
    vectorized_implementation(data, params, B, S, K, device)
    
    t_vec = vectorized_implementation(data, params, B, S, K, device)
    print(f"Vectorized Time: {t_vec:.4f}s")
    
    t_loop = current_implementation(data, params, B, S, K, device)
    print(f"Loop Time:       {t_loop:.4f}s")
    
    print(f"Speedup: {t_loop / t_vec:.1f}x")
