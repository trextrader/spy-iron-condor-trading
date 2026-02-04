# tools/profile_etd1.py
"""
ETD-1 Kernel Performance Profiler

Profiles the ETD-1 kernel (matrix exponential + φ₁) across:
- Different state sizes
- Different dtypes (fp32, fp16, bf16)
- CPU vs CUDA

Usage:
    python tools/profile_etd1.py --device cuda --sizes 64,128,256,512 --bf16
    python tools/profile_etd1.py --device cpu --sizes 32,64,128
"""

import sys
import os
import time
import argparse
from contextlib import contextmanager

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from intelligence.condor_brain_net import (
    AugmentedStateSpec,
    BlockMatrixA,
    etd1_kernel,
)


@contextmanager
def cuda_timer(label: str):
    """Context manager for CUDA-synchronized timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"  {label}: {(end - start)*1000:.3f} ms")


def profile_etd1_once(d_x: int, dt: float, device: torch.device, dtype: torch.dtype, n_iters: int = 20):
    """Profile ETD-1 kernel for a single configuration."""
    # Create spec with approximate block sizes
    d_h = max(4, d_x // 2)
    d_v = max(2, d_x // 4)
    d_m = max(2, d_x // 4)
    d_r = max(2, d_x - d_h - d_v - d_m)
    
    spec = AugmentedStateSpec(d_h=d_h, d_v=d_v, d_m=d_m, d_r=d_r)
    A_theta = BlockMatrixA(spec, enforce_sparsity=True).to(device=device, dtype=dtype)

    # Build full matrix
    with torch.no_grad():
        A_full = A_theta.full_matrix()
    
    print(f"\n[ETD-1] d_x={spec.d_x}, dt={dt}, device={device}, dtype={dtype}")
    
    # Warmup
    for _ in range(3):
        F, phi1 = etd1_kernel(A_full, dt)
    
    # Timed run
    with cuda_timer(f"{n_iters} iterations"):
        for _ in range(n_iters):
            F, phi1 = etd1_kernel(A_full, dt)
    
    # Sanity checks
    assert torch.isfinite(F).all(), "F contains non-finite values!"
    assert torch.isfinite(phi1).all(), "phi1 contains non-finite values!"
    
    # Report per-iteration time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return F, phi1


def profile_full_step(d_x: int, d_input: int, d_control: int, batch: int, 
                      device: torch.device, dtype: torch.dtype, n_iters: int = 10):
    """Profile full condornet_master_step."""
    from intelligence.condor_brain_net import (
        BlockVectorB, CDEResponseG, FullForcingD, condornet_master_step
    )
    
    d_h = max(4, d_x // 2)
    d_v = max(2, d_x // 4)
    d_m = max(2, d_x // 4)
    d_r = max(2, d_x - d_h - d_v - d_m)
    
    spec = AugmentedStateSpec(d_h=d_h, d_v=d_v, d_m=d_m, d_r=d_r)
    n_greeks = 5
    d_q = 1
    
    A_theta = BlockMatrixA(spec, enforce_sparsity=True).to(device=device, dtype=dtype)
    B_theta = BlockVectorB(spec, d_control).to(device=device, dtype=dtype)
    G_theta = CDEResponseG(spec, d_input, d_control).to(device=device, dtype=dtype)
    D_forcing = FullForcingD(spec, n_greeks=n_greeks, d_q=d_q).to(device=device, dtype=dtype)
    
    x_prev = torch.randn(batch, spec.d_x, device=device, dtype=dtype)
    u_k = torch.randn(batch, d_control, device=device, dtype=dtype)
    dX_k = torch.randn(batch, d_input, device=device, dtype=dtype)
    greeks_k = torch.randn(batch, n_greeks, device=device, dtype=dtype)
    r_prev = torch.randn(batch, spec.d_r, device=device, dtype=dtype)
    q_k = torch.randn(batch, d_q, device=device, dtype=dtype)
    dt_k = 1.0
    
    print(f"\n[Master Step] d_x={spec.d_x}, batch={batch}, device={device}, dtype={dtype}")
    
    # Warmup
    for _ in range(3):
        x_k = condornet_master_step(
            spec, A_theta, B_theta, G_theta, D_forcing,
            x_prev, u_k, dX_k, greeks_k, r_prev, q_k, dt_k
        )
    
    # Timed run
    with cuda_timer(f"{n_iters} iterations"):
        for _ in range(n_iters):
            x_k = condornet_master_step(
                spec, A_theta, B_theta, G_theta, D_forcing,
                x_prev, u_k, dX_k, greeks_k, r_prev, q_k, dt_k
            )
    
    return x_k


def main():
    parser = argparse.ArgumentParser(description="Profile ETD-1 kernel performance")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sizes", type=str, default="64,128,256,512", 
                        help="Comma-separated state dimensions to test")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step")
    parser.add_argument("--bf16", action="store_true", help="Include bfloat16 tests")
    parser.add_argument("--fp16", action="store_true", help="Include float16 tests")
    parser.add_argument("--n-iters", type=int, default=20, help="Iterations per test")
    parser.add_argument("--full-step", action="store_true", help="Also profile full master step")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for full step")
    args = parser.parse_args()

    device = torch.device(args.device)
    sizes = [int(s) for s in args.sizes.split(",")]

    dtypes = [torch.float32]
    if args.fp16:
        dtypes.append(torch.float16)
    if args.bf16 and device.type == 'cuda':
        dtypes.append(torch.bfloat16)

    print("=" * 60)
    print("CondorNet ETD-1 Kernel Profiler")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Sizes: {sizes}")
    print(f"Dtypes: {dtypes}")
    print(f"Iterations: {args.n_iters}")
    print("=" * 60)

    # Profile ETD-1 kernel
    for d_x in sizes:
        for dtype in dtypes:
            try:
                profile_etd1_once(d_x, args.dt, device, dtype, args.n_iters)
            except Exception as e:
                print(f"  ERROR: {e}")

    # Profile full master step
    if args.full_step:
        print("\n" + "=" * 60)
        print("Full Master Step Profiling")
        print("=" * 60)
        
        d_input = 54
        d_control = 128
        
        for d_x in sizes:
            for dtype in dtypes:
                try:
                    profile_full_step(d_x, d_input, d_control, args.batch, 
                                     device, dtype, args.n_iters)
                except Exception as e:
                    print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Profiling complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
