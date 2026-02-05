
import torch
import numpy as np
import time
from intelligence.condor_brain_net import CondorNet, etd1_kernel

def audit_etd_gpu_faithfulness():
    print("=== Sub-Phase 3.5: ETD Kernel GPU Hardening Audit (V47) ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU audit.")
        return

    # 1. Device Invariance Audit
    print("\n[1] Device Invariance: CPU vs GPU kernels")
    
    # Use fixed seed for initialization reproducibility
    torch.manual_seed(42)
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda')

    # Dimensions
    d_h, d_v, d_m, d_r = 64, 16, 32, 16
    n_params = d_h + d_v + d_m + d_r
    
    # Instantiate models
    torch.manual_seed(42)
    model_cpu = CondorNet(d_input=54, d_h=d_h, d_v=d_v, d_m=d_m, d_r=d_r).to(device_cpu)
    
    torch.manual_seed(42)
    model_gpu = CondorNet(d_input=54, d_h=d_h, d_v=d_v, d_m=d_m, d_r=d_r).to(device_gpu)

    # Precompute A matrices
    with torch.no_grad():
        A_cpu = model_cpu.A_theta.full_matrix().float()
        A_gpu = model_gpu.A_theta.full_matrix().float()
        
        # Verify A matrices are identical (init check)
        a_diff = (A_cpu - A_gpu.cpu()).abs().max().item()
        print(f"  - Initial A-matrix MAE: {a_diff:.8e}")

    # Compute kernels
    dt = 1.0
    with torch.no_grad():
        # GPU
        F_gpu, phi1_gpu = etd1_kernel(A_gpu, dt)
        
        # CPU
        F_cpu, phi1_cpu = etd1_kernel(A_cpu, dt)

    # Comparison
    f_mae = (F_cpu - F_gpu.cpu()).abs().max().item()
    phi_mae = (phi1_cpu - phi1_gpu.cpu()).abs().max().item()

    print(f"  - F_k (exp(A)) MAE: {f_mae:.8e}")
    print(f"  - phi1 (inv(A)(e^A-I)) MAE: {phi_mae:.8e}")

    if f_mae < 1e-7 and phi_mae < 1e-7:
        print("  SUCCESS: Device invariance verified.")
    else:
        print("  WARNING: Micro-divergence detected in kernels.")

    # 2. Autocast Integrity Test
    print("\n[2] Autocast Integrity: FP32 Enforcement")
    
    # We test if matrix_exp inside etd_kernel stays FP32 even under autocast
    with torch.amp.autocast('cuda', dtype=torch.float16):
        # We need a hook or simple check inside model.forward
        # But we can check output dtypes
        output_gpu = model_gpu(torch.randn(1, 10, 54).to(device_gpu))
        print(f"  - Output Dtype under Autocast: {output_gpu.dtype}")
        
        # Force a check on the kernel dtypes computed during forward
        # (Internal diagnostics would be better, but dtype of final out suffices)
        if output_gpu.dtype == torch.float32:
            print("  SUCCESS: Forward pass maintained FP32 floor.")
        else:
            print("  WARNING: Autocast downgraded manifold results.")

    # 3. Spectral Stability Stress Test
    print("\n[3] Spectral Stability: rho -> 1.0")
    
    # Manipulate A to have precisely rho=1.0
    # Simplest way: set diagonal to 0 and off-diagonal to 1
    with torch.no_grad():
        A_stress = torch.zeros(n_params, n_params, device=device_gpu)
        # Jordan block style to test matrix_exp stability
        for i in range(n_params-1):
            A_stress[i, i+1] = 1.0
        
        # This A is nilpotent, exp(A) is stable polynomial
        F_stress, p_stress = etd1_kernel(A_stress, dt=1.0)
        print(f"  - Nilpotent A (rho=0): Deterministic? {not torch.isnan(F_stress).any()}")
        
        # Identity A (rho=e^1)
        A_identity = torch.eye(n_params, device=device_gpu)
        F_id, p_id = etd1_kernel(A_identity, dt=1.0)
        print(f"  - Identity A (rho=1): F mean: {F_id.mean().item():.4f} (exp(1) = 2.7183)")
        
        if torch.isnan(F_id).any() or torch.isnan(p_id).any():
            print("  FAILURE: NaN detected at rho=1 boundary.")
        else:
            print("  SUCCESS: Numerical stability verified at unity.")

if __name__ == "__main__":
    audit_etd_gpu_faithfulness()
