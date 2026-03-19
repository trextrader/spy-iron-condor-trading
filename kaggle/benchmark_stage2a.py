#!/usr/bin/env python3
"""
benchmark_stage2a.py — Stage 2A engine timing harness
======================================================
Measures end-to-end run_backtest_optimizer_batch wall time for:
  - CPU numpy path (gpu_k_threshold = K+1)
  - GPU tensor path (gpu_k_threshold = K, CUDA required)

Metrics reported:
  - wall_sec
  - bars/sec        = T / wall_sec
  - candidates/sec  = K × T / wall_sec

Outputs:
  reports/gpu_transition/stage2a_benchmarks.json
  reports/gpu_transition/stage2a_benchmarks.md

Usage (from project root):
  python kaggle/benchmark_stage2a.py
  python kaggle/benchmark_stage2a.py --T 1000 --iters 3
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_THIS_DIR     = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from optimizer_engine import run_backtest_optimizer_batch, ObjectiveSpec
from optimizer_prep   import OptimizerContext
from candidate_codec  import CandidateBatch

_OBJ_SPEC = ObjectiveSpec()


# ── Context builder ───────────────────────────────────────────────────────────

def build_ctx(device: torch.device, T: int, M: int = 36,
              entry_pct: float = 0.20, seed: int = 42) -> OptimizerContext:
    """
    Build a synthetic OptimizerContext with T bars and M options per bar.

    entry_pct controls the fraction of bars with gate signals active.
    Uses 5-minute timestamps to match production cadence.
    """
    rng = np.random.default_rng(seed)

    # 5-minute timestamps starting 2025-01-02 09:30
    t0  = int(datetime.datetime(2025, 1, 2, 9, 30).timestamp())
    ts  = np.arange(T, dtype=np.int64) * 300 + t0

    spot = (500.0 + np.cumsum(rng.normal(0, 0.2, T))).astype(np.float32)
    spot = np.clip(spot, 400.0, 650.0)

    # Gate signals: entry_pct fraction active, spread across bars
    gate_entry   = np.full(T, 0.10, np.float32)
    gate_pop     = np.full(T, 0.10, np.float32)
    strategy_idx = np.zeros(T, np.int16)
    abstain      = np.ones(T, bool)

    n_entry = max(1, int(T * entry_pct))
    entry_bars = np.linspace(0, T - 1, n_entry, dtype=int)
    for b in entry_bars:
        gate_entry[b]   = 0.70
        gate_pop[b]     = 0.60
        strategy_idx[b] = 8
        abstain[b]      = False

    # Chain: same options at every bar (M//2 strikes × 2 rights, DTE=21)
    n_strikes   = M // 2
    sp_mean     = float(spot.mean())
    strikes     = np.linspace(sp_mean * 0.88, sp_mean * 1.12, n_strikes, np.float32)
    call_deltas = np.linspace(0.92, 0.05, n_strikes, np.float32)
    put_deltas  = (1.0 - call_deltas).astype(np.float32)
    call_mid    = np.maximum(0.10, call_deltas * 15.0).astype(np.float32)
    put_mid     = np.maximum(0.10, put_deltas  * 15.0).astype(np.float32)

    bar_right   = np.concatenate([np.zeros(n_strikes, np.int8),
                                   np.ones(n_strikes,  np.int8)])
    bar_strike  = np.tile(strikes, 2)
    bar_dte     = np.full(M, 21.0, np.float32)
    bar_delta   = np.concatenate([call_deltas, put_deltas])
    bar_bid     = np.concatenate([call_mid * 0.9, put_mid * 0.9])
    bar_ask     = np.concatenate([call_mid * 1.1, put_mid * 1.1])
    bar_mid_arr = np.concatenate([call_mid, put_mid])

    bar_offsets = np.arange(T + 1, dtype=np.int64) * M
    right_flat  = np.tile(bar_right,   T)
    strike_flat = np.tile(bar_strike,  T)
    dte_flat    = np.tile(bar_dte,     T)
    delta_flat  = np.tile(bar_delta,   T)
    bid_flat    = np.tile(bar_bid,     T)
    ask_flat    = np.tile(bar_ask,     T)
    mid_flat    = np.tile(bar_mid_arr, T)

    base = datetime.date(2025, 1, 2)
    bar_dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(T)]

    def _t(arr, dtype):
        return torch.as_tensor(arr, device=device, dtype=dtype)

    return OptimizerContext(
        device       = device,
        T            = T,
        timestamps   = _t(ts,           torch.int64),
        spot         = _t(spot,         torch.float32),
        gate_entry   = _t(gate_entry,   torch.float32),
        gate_pop     = _t(gate_pop,     torch.float32),
        strategy_idx = _t(strategy_idx, torch.int16),
        abstain      = _t(abstain,      torch.bool),
        bar_offsets  = _t(bar_offsets,  torch.int64),
        option_right = _t(right_flat,   torch.int8),
        option_strike= _t(strike_flat,  torch.float32),
        option_dte   = _t(dte_flat,     torch.float32),
        option_delta = _t(delta_flat,   torch.float32),
        opt_bid      = _t(bid_flat,     torch.float32),
        opt_ask      = _t(ask_flat,     torch.float32),
        opt_mid      = _t(mid_flat,     torch.float32),
        fast_end     = max(1, T // 4),
        medium_end   = max(1, T * 3 // 5),
        bar_dates    = bar_dates,
    )


def build_candidates(K: int) -> CandidateBatch:
    return CandidateBatch(K=K, params={
        "target_dte":       np.full(K, 21.0, np.float64),
        "short_delta":      np.full(K, 0.45, np.float64),
        "spread_width":     np.full(K,  5.0, np.float64),
        "stop_loss_dollar": np.full(K, 600.0, np.float64),
        "profit_target":    np.full(K, 1500.0, np.float64),
        "max_dte_exit":     np.zeros(K, np.float64),
        "hold_days":        np.full(K,  7.0, np.float64),
    })


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(T_default=2000, iters=3):
    device_cpu  = torch.device("cpu")
    device_cuda = torch.device("cuda") if torch.cuda.is_available() else None

    print(f"\nStage 2A Engine Benchmark")
    print(f"CPU device: {device_cpu}")
    if device_cuda:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA      : {torch.version.cuda}")
    else:
        print("GPU       : not available (CPU-only benchmark)")
    print(f"Torch     : {torch.__version__}")
    print(f"T bars    : {T_default}")
    print(f"Iters     : {iters} (best-of for wall time)")

    K_vals = [8, 32, 128]
    T_vals = [500, T_default]

    rows = []

    print()
    print("=" * 90)
    print(f"{'Path':<8} {'K':>5} {'T':>6} {'wall_s':>8} "
          f"{'bars/s':>10} {'cands/s':>12} {'trades':>8}")
    print("=" * 90)

    for T in T_vals:
        ctx_cpu = build_ctx(device_cpu, T)
        if device_cuda:
            ctx_gpu = build_ctx(device_cuda, T)

        for K in K_vals:
            cands = build_candidates(K)

            # ── CPU path ──────────────────────────────────────────────────
            cpu_times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                r_cpu = run_backtest_optimizer_batch(
                    ctx_cpu, cands,
                    base_config={},
                    objective_spec=_OBJ_SPEC,
                    strategy_idx_filter=8,
                    strategy_family="iron_butterfly",
                    gpu_k_threshold=K + 1,
                )
                cpu_times.append(time.perf_counter() - t0)
            cpu_wall = min(cpu_times)
            cpu_bars = T / cpu_wall
            cpu_cands = K * T / cpu_wall
            total_trades_cpu = int(r_cpu.total.sum())

            print(f"{'CPU':<8} {K:>5} {T:>6} {cpu_wall:>8.3f} "
                  f"{cpu_bars:>10.0f} {cpu_cands:>12.0f} {total_trades_cpu:>8}")
            rows.append({
                "path": "CPU", "K": K, "T": T,
                "wall_s":     round(cpu_wall, 4),
                "bars_per_s": round(cpu_bars, 1),
                "cands_per_s": round(cpu_cands, 1),
                "total_trades": total_trades_cpu,
            })

            # ── GPU path ──────────────────────────────────────────────────
            if device_cuda:
                gpu_times = []
                for _ in range(iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    r_gpu = run_backtest_optimizer_batch(
                        ctx_gpu, cands,
                        base_config={},
                        objective_spec=_OBJ_SPEC,
                        strategy_idx_filter=8,
                        strategy_family="iron_butterfly",
                        gpu_k_threshold=K,
                    )
                    torch.cuda.synchronize()
                    gpu_times.append(time.perf_counter() - t0)
                gpu_wall  = min(gpu_times)
                gpu_bars  = T / gpu_wall
                gpu_cands = K * T / gpu_wall
                total_trades_gpu = int(r_gpu.total.sum())
                speedup = cpu_wall / gpu_wall

                print(f"{'GPU':<8} {K:>5} {T:>6} {gpu_wall:>8.3f} "
                      f"{gpu_bars:>10.0f} {gpu_cands:>12.0f} {total_trades_gpu:>8}"
                      f"  [{speedup:.2f}x]")
                rows.append({
                    "path": "GPU", "K": K, "T": T,
                    "wall_s":     round(gpu_wall, 4),
                    "bars_per_s": round(gpu_bars, 1),
                    "cands_per_s": round(gpu_cands, 1),
                    "total_trades": total_trades_gpu,
                    "speedup_vs_cpu": round(speedup, 3),
                })
        print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    meta = {
        "torch_version": torch.__version__,
        "cuda_version":  torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name":      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "T_default":     T_default,
        "iters":         iters,
    }

    out_dir = _PROJECT_ROOT / "reports" / "gpu_transition"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "stage2a_benchmarks.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    print(f"Saved: {json_path}")

    md_lines = [
        "# Stage 2A Engine Benchmarks",
        "",
        f"GPU: {meta['gpu_name'] or 'N/A'}  "
        f"Torch: {meta['torch_version']}  CUDA: {meta['cuda_version'] or 'N/A'}",
        f"T_default: {T_default}  Iters: {iters} (best-of)",
        "",
        "| Path | K | T | wall_s | bars/s | cands/s | trades | speedup |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        speedup_str = f"{r.get('speedup_vs_cpu', 'N/A')}"
        md_lines.append(
            f"| {r['path']} | {r['K']} | {r['T']} "
            f"| {r['wall_s']:.3f} | {r['bars_per_s']:.0f} "
            f"| {r['cands_per_s']:.0f} | {r['total_trades']} | {speedup_str} |"
        )

    md_path = out_dir / "stage2a_benchmarks.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2A engine benchmark")
    parser.add_argument("--T",     type=int, default=2000,
                        help="Number of bars for large benchmark (default: 2000)")
    parser.add_argument("--iters", type=int, default=3,
                        help="Repetitions per config (best-of, default: 3)")
    args = parser.parse_args()
    run_benchmark(T_default=args.T, iters=args.iters)
