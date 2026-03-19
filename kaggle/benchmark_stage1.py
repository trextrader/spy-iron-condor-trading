#!/usr/bin/env python3
"""
benchmark_stage1.py — Stage 1 selector & mark-to-market timing harness
=======================================================================
Measures:
  - CPU _find_best_structure_iron_condor for K=8,32,128,512 / S=50,200,500
  - GPU select_entry_for_bar for same K/S
  - GPU mark_to_market_gpu for K=8,32,128,512

Outputs a console table and writes:
  reports/gpu_transition/stage1_benchmarks.json
  reports/gpu_transition/stage1_benchmarks.md

Usage (from kaggle/ dir or project root):
  python kaggle/benchmark_stage1.py
  python kaggle/benchmark_stage1.py --iters 200 --warmup 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gpu_strike_selector import (
    select_entry_for_bar,
    mark_to_market_gpu,
)
from optimizer_engine import _find_best_structure_iron_condor

# ── Chain builder ─────────────────────────────────────────────────────────────

def build_chain_np(S: int, spot: float = 500.0, seed: int = 42) -> tuple:
    """
    Build a synthetic chain with S options (S/2 calls + S/2 puts) near spot.
    Returns (right, strike, dte, delta, bid, ask) as float32 numpy arrays.
    """
    rng = np.random.default_rng(seed)
    n_per_side = S // 2
    # Strikes: spread ± 12% around spot in n_per_side increments
    strikes_all = np.linspace(spot * 0.88, spot * 1.12, n_per_side).astype(np.float32)
    dte_all     = np.full(n_per_side, 21.0, dtype=np.float32)

    # Call deltas: roughly 0.9 ITM → 0.05 OTM as strike increases
    call_deltas = np.linspace(0.92, 0.05, n_per_side).astype(np.float32)
    put_deltas  = (1.0 - call_deltas).astype(np.float32)

    # Rough mid price ~ delta * vol * spot * sqrt(T)
    vol = 0.20
    T   = 21.0 / 252.0
    call_mid = np.maximum(0.05, call_deltas * vol * spot * np.sqrt(T)).astype(np.float32)
    put_mid  = np.maximum(0.05, put_deltas  * vol * spot * np.sqrt(T)).astype(np.float32)
    spread   = 0.05  # 5% half-spread

    # Calls: right=0
    rights  = np.concatenate([np.zeros(n_per_side, np.int8),
                               np.ones(n_per_side,  np.int8)])
    strikes = np.tile(strikes_all, 2)
    dtes    = np.tile(dte_all, 2)
    deltas  = np.concatenate([call_deltas, put_deltas])
    bids    = np.concatenate([call_mid * (1 - spread), put_mid * (1 - spread)])
    asks    = np.concatenate([call_mid * (1 + spread), put_mid * (1 + spread)])
    return rights, strikes, dtes, deltas, bids, asks


def build_chain_torch(S, spot=500.0, device=None, seed=42):
    """Build chain as torch tensors on device."""
    right_np, stk_np, dte_np, dlt_np, bid_np, ask_np = build_chain_np(S, spot, seed)
    dev = device or torch.device("cpu")
    return (
        torch.as_tensor(right_np, device=dev, dtype=torch.int8),
        torch.as_tensor(stk_np,   device=dev, dtype=torch.float32),
        torch.as_tensor(dte_np,   device=dev, dtype=torch.float32),
        torch.as_tensor(dlt_np,   device=dev, dtype=torch.float32),
        torch.as_tensor(bid_np,   device=dev, dtype=torch.float32),
        torch.as_tensor(ask_np,   device=dev, dtype=torch.float32),
    )


def build_params_torch(K, device=None, seed=0):
    """Build K candidate parameter tensors on device."""
    rng = np.random.default_rng(seed)
    dev = device or torch.device("cpu")
    t_dte   = torch.full((K,), 21.0, dtype=torch.float32, device=dev)
    t_sd    = torch.as_tensor(
        rng.uniform(0.10, 0.30, K).astype(np.float32), device=dev
    )
    t_sw    = torch.full((K,), 5.0, dtype=torch.float32, device=dev)
    return t_dte, t_sd, t_sw


# ── Timing helpers ─────────────────────────────────────────────────────────────

def time_cpu_fn(fn, *args, warmup=5, iters=50):
    """Wall time per call (seconds) for a CPU function."""
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters


def time_gpu_fn(fn, *args, warmup=10, iters=100, device=None):
    """Wall time per call with CUDA sync if device is CUDA."""
    for _ in range(warmup):
        fn(*args)
    if device and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if device and device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ── Main benchmark ─────────────────────────────────────────────────────────────

def run_benchmark(iters=100, warmup=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"CUDA  : {torch.version.cuda}")
    print(f"Torch : {torch.__version__}")
    print(f"Iters : {iters} (warmup={warmup})")
    print()

    K_vals = [8, 32, 128, 512]
    S_vals = [50, 200, 500]
    SPOT   = 500.0

    rows = []

    # ── Selector benchmarks ──────────────────────────────────────────────────
    print("=" * 80)
    print(f"{'Strategy':<20} {'K':>5} {'S':>5} {'CPU ms':>9} {'GPU ms':>9} {'Speedup':>9}")
    print("=" * 80)

    for S in S_vals:
        right_np, stk_np, dte_np, dlt_np, bid_np, ask_np = build_chain_np(S, SPOT)
        t_chain = build_chain_torch(S, SPOT, device=device)

        for K in K_vals:
            tdte_np  = np.full(K, 21.0, dtype=np.float32)
            sd_np    = np.full(K, 0.20, dtype=np.float32)
            sw_np    = np.full(K, 5.0,  dtype=np.float32)
            t_tdte, t_sd, t_sw = build_params_torch(K, device=device)

            # CPU
            cpu_t = time_cpu_fn(
                _find_best_structure_iron_condor,
                SPOT, right_np, stk_np, dte_np, dlt_np, bid_np, ask_np,
                tdte_np, sd_np, sw_np,
                warmup=warmup, iters=iters,
            )

            # GPU
            t_r, t_s, t_d, t_dt, t_b, t_a = t_chain
            gpu_t = time_gpu_fn(
                select_entry_for_bar,
                t_r, t_s, t_d, t_dt, t_b, t_a, SPOT,
                t_tdte, t_sd, t_sw, "iron_condor",
                warmup=warmup, iters=iters, device=device,
            )

            speedup = cpu_t / gpu_t if gpu_t > 0 else float("nan")
            tag = "iron_condor"
            print(f"{tag:<20} {K:>5} {S:>5} {cpu_t*1e3:>9.3f} {gpu_t*1e3:>9.3f} {speedup:>9.2f}x")
            rows.append({
                "strategy": tag, "K": K, "S": S,
                "cpu_ms": round(cpu_t * 1e3, 4),
                "gpu_ms": round(gpu_t * 1e3, 4),
                "speedup": round(speedup, 3),
                "device": str(device),
            })

    # ── MtM benchmark ────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"{'mark_to_market':<20} {'K':>5} {'S':>5} {'ms (GPU)':>9}")
    print("=" * 80)

    S_mtm = 200
    t_chain_mtm = build_chain_torch(S_mtm, SPOT, device=device)
    t_r, t_s, t_d, _, t_b, t_a = t_chain_mtm

    for K in K_vals:
        entry_ssc = torch.full((K,), 510.0, dtype=torch.float32, device=device)
        entry_ssp = torch.full((K,), 485.0, dtype=torch.float32, device=device)
        entry_sw  = torch.full((K,), 5.0,   dtype=torch.float32, device=device)
        open_mask = torch.ones(K, dtype=torch.bool, device=device)
        entry_dte = torch.full((K,), 21.0,  dtype=torch.float32, device=device)

        mtm_t = time_gpu_fn(
            mark_to_market_gpu,
            t_r, t_s, t_d, t_b, t_a,
            entry_ssc, entry_ssp, entry_sw, open_mask, entry_dte,
            "iron_condor",
            warmup=warmup, iters=iters, device=device,
        )
        print(f"{'mark_to_market':<20} {K:>5} {S_mtm:>5} {mtm_t*1e3:>9.3f}")
        rows.append({
            "strategy": "mark_to_market", "K": K, "S": S_mtm,
            "cpu_ms": None,
            "gpu_ms": round(mtm_t * 1e3, 4),
            "speedup": None,
            "device": str(device),
        })

    # ── Save results ──────────────────────────────────────────────────────────
    meta = {
        "torch_version": torch.__version__,
        "cuda_version":  torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name":      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device":        str(device),
        "iters":         iters,
        "warmup":        warmup,
    }

    out_dir = _PROJECT_ROOT / "reports" / "gpu_transition"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "stage1_benchmarks.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Markdown summary
    md_lines = [
        "# Stage 1 Benchmarks",
        "",
        f"Device: {device}",
        f"GPU: {meta['gpu_name'] or 'N/A'}",
        f"Torch: {meta['torch_version']}  CUDA: {meta['cuda_version'] or 'N/A'}",
        f"Iterations: {iters} (warmup={warmup})",
        "",
        "## Selector (iron_condor)",
        "",
        "| Strategy | K | S | CPU ms | GPU ms | Speedup |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["strategy"] != "iron_condor":
            continue
        md_lines.append(
            f"| {r['strategy']} | {r['K']} | {r['S']} "
            f"| {r['cpu_ms']:.3f} | {r['gpu_ms']:.3f} | {r['speedup']:.2f}x |"
        )
    md_lines += [
        "",
        "## Mark-to-market (GPU only)",
        "",
        "| K | S | GPU ms |",
        "|---|---|---|",
    ]
    for r in rows:
        if r["strategy"] != "mark_to_market":
            continue
        md_lines.append(f"| {r['K']} | {r['S']} | {r['gpu_ms']:.3f} |")

    md_path = out_dir / "stage1_benchmarks.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved: {md_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 performance benchmark")
    parser.add_argument("--iters",  type=int, default=100,
                        help="Timing iterations per measurement (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    args = parser.parse_args()
    run_benchmark(iters=args.iters, warmup=args.warmup)
