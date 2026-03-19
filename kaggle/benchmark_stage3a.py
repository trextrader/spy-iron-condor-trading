#!/usr/bin/env python3
"""
benchmark_stage3a.py — Stage 3A compile benchmark
==================================================
Measures wall time and bars/sec for three execution modes:

  1. Stage 2A engine eager   — run_backtest_optimizer_batch (existing GPU path)
  2. step_bar_gpu eager      — new compile-friendly loop, no torch.compile
  3. step_bar_gpu compiled   — same loop with torch.compile(mode="reduce-overhead")

Reports
  - Bars/sec and relative speedup vs Stage 2A baseline
  - Compile warm-up cost (time-to-first-result)
  - Steady-state performance (iters > 1 benchmark)

Outputs
  reports/gpu_transition/stage3a_compile_benchmark.json
  reports/gpu_transition/stage3a_compile_benchmark.md

Usage (from project root):
  python kaggle/benchmark_stage3a.py
  python kaggle/benchmark_stage3a.py --T 2000 --iters 3
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

from optimizer_engine  import run_backtest_optimizer_batch, ObjectiveSpec, STARTING_EQUITY
from optimizer_prep    import OptimizerContext
from candidate_codec   import CandidateBatch
from optimizer_stage3a import (
    BarState, init_bar_state, step_bar_gpu, make_compiled_step,
    FAMILY_TO_CODE, CODE_TO_STR, FAMILY_IRON_BUTTERFLY,
)

_OBJ_SPEC = ObjectiveSpec()


# ── Context builder ────────────────────────────────────────────────────────────

def build_ctx(device: torch.device, T: int, M: int = 36,
              entry_pct: float = 0.20, seed: int = 42) -> OptimizerContext:
    """Synthetic OptimizerContext — mirrors benchmark_stage2a.py exactly."""
    rng = np.random.default_rng(seed)
    t0  = int(datetime.datetime(2025, 1, 2, 9, 30).timestamp())
    ts  = np.arange(T, dtype=np.int64) * 300 + t0

    spot = (500.0 + np.cumsum(rng.normal(0, 0.2, T))).astype(np.float32)
    spot = np.clip(spot, 400.0, 650.0)

    gate_entry   = np.full(T, 0.10, np.float32)
    gate_pop     = np.full(T, 0.10, np.float32)
    strategy_idx = np.zeros(T, np.int16)
    abstain      = np.ones(T, bool)

    n_entry    = max(1, int(T * entry_pct))
    entry_bars = np.linspace(0, T - 1, n_entry, dtype=int)
    for b in entry_bars:
        gate_entry[b]   = 0.70
        gate_pop[b]     = 0.60
        strategy_idx[b] = 8
        abstain[b]      = False

    n_strikes   = M // 2
    sp_mean     = float(spot.mean())
    strikes     = np.linspace(sp_mean * 0.88, sp_mean * 1.12, n_strikes, np.float32)
    call_deltas = np.linspace(0.92, 0.05, n_strikes, np.float32)
    put_deltas  = (1.0 - call_deltas).astype(np.float32)
    call_mid    = np.maximum(0.10, call_deltas * 15.0).astype(np.float32)
    put_mid     = np.maximum(0.10, put_deltas  * 15.0).astype(np.float32)

    bar_right  = np.concatenate([np.zeros(n_strikes, np.int8),  np.ones(n_strikes, np.int8)])
    bar_strike = np.tile(strikes, 2)
    bar_dte    = np.full(M, 21.0, np.float32)
    bar_delta  = np.concatenate([call_deltas, put_deltas])
    bar_bid    = np.concatenate([call_mid * 0.9, put_mid * 0.9])
    bar_ask    = np.concatenate([call_mid * 1.1, put_mid * 1.1])

    offsets    = np.arange(T + 1, dtype=np.int64) * M
    right_flat = np.tile(bar_right,  T)
    stk_flat   = np.tile(bar_strike, T)
    dte_flat   = np.tile(bar_dte,    T)
    dlt_flat   = np.tile(bar_delta,  T)
    bid_flat   = np.tile(bar_bid,    T)
    ask_flat   = np.tile(bar_ask,    T)

    base      = datetime.date(2025, 1, 2)
    bar_dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(T)]

    def _t(arr, dtype):
        return torch.as_tensor(arr, device=device, dtype=dtype)

    return OptimizerContext(
        device        = device,
        T             = T,
        timestamps    = _t(ts,         torch.int64),
        spot          = _t(spot,       torch.float32),
        gate_entry    = _t(gate_entry, torch.float32),
        gate_pop      = _t(gate_pop,   torch.float32),
        strategy_idx  = _t(strategy_idx, torch.int16),
        abstain       = _t(abstain,    torch.bool),
        bar_offsets   = _t(offsets,    torch.int64),
        option_right  = _t(right_flat, torch.int8),
        option_strike = _t(stk_flat,   torch.float32),
        option_dte    = _t(dte_flat,   torch.float32),
        option_delta  = _t(dlt_flat,   torch.float32),
        opt_bid       = _t(bid_flat,   torch.float32),
        opt_ask       = _t(ask_flat,   torch.float32),
        fast_end      = max(1, T // 4),
        medium_end    = max(1, T * 3 // 5),
        bar_dates     = bar_dates,
    )


def build_candidates(K: int) -> CandidateBatch:
    return CandidateBatch(K=K, params={
        "target_dte":       np.full(K, 21.0,   np.float64),
        "short_delta":      np.full(K, 0.45,   np.float64),
        "spread_width":     np.full(K,  5.0,   np.float64),
        "stop_loss_dollar": np.full(K, 600.0,  np.float64),
        "profit_target":    np.full(K, 1500.0, np.float64),
        "max_dte_exit":     np.zeros(K,         np.float64),
        "hold_days":        np.full(K,  7.0,   np.float64),
    })


# ── step_bar_gpu runner ────────────────────────────────────────────────────────

def run_step_bar_loop(ctx: OptimizerContext, candidates: CandidateBatch,
                      step_fn=None) -> float:
    """
    Run full T-bar loop using step_bar_gpu (or compiled version).
    Returns wall time in seconds. Includes force-close at end.
    """
    if step_fn is None:
        step_fn = step_bar_gpu

    K   = candidates.K
    dev = ctx.device
    fam = FAMILY_IRON_BUTTERFLY

    spot_np    = ctx.spot.cpu().numpy()
    gate_e_np  = ctx.gate_entry.cpu().numpy()
    gate_p_np  = ctx.gate_pop.cpu().numpy()
    sidx_np    = ctx.strategy_idx.cpu().numpy()
    abstain_np = ctx.abstain.cpu().numpy().astype(bool)
    offsets_np = ctx.bar_offsets.cpu().numpy()
    ts_t = torch.tensor(ctx.timestamps.cpu().numpy().astype(np.float64),
                        dtype=torch.float64, device=dev)

    p     = candidates.params
    td_t  = torch.from_numpy(p["target_dte"].astype(np.float32)).to(dev)
    sd_t  = torch.from_numpy(p["short_delta"].astype(np.float32)).to(dev)
    sw_t  = torch.from_numpy(p["spread_width"].astype(np.float32)).to(dev)
    sl_t  = torch.tensor(p["stop_loss_dollar"],  dtype=torch.float64, device=dev)
    pt_t  = torch.tensor(p["profit_target"],     dtype=torch.float64, device=dev)
    mde_t = torch.tensor(p["max_dte_exit"],      dtype=torch.float64, device=dev)
    hd_t  = torch.tensor(p["hold_days"],         dtype=torch.float64, device=dev)

    right_gpu  = ctx.option_right
    stk_gpu    = ctx.option_strike
    dte_gpu    = ctx.option_dte
    dlt_gpu    = ctx.option_delta
    bid_gpu    = ctx.opt_bid
    ask_gpu    = ctx.opt_ask

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    state = init_bar_state(K, dev)

    for i in range(ctx.T):
        s_off   = int(offsets_np[i])
        e_off   = int(offsets_np[i + 1])
        gate_ok = (
            float(gate_e_np[i]) > 0.55
            and float(gate_p_np[i]) > 0.50
            and not bool(abstain_np[i])
            and int(sidx_np[i]) == 8
        )
        state = step_fn(
            state      = state,
            bar_idx    = i,
            spot       = float(spot_np[i]),
            ts_t       = ts_t,
            chain_right   = right_gpu[s_off:e_off],
            chain_strike  = stk_gpu[s_off:e_off],
            chain_dte     = dte_gpu[s_off:e_off],
            chain_bid     = bid_gpu[s_off:e_off],
            chain_ask     = ask_gpu[s_off:e_off],
            chain_delta   = dlt_gpu[s_off:e_off],
            td_t=td_t, sd_t=sd_t, sw_t=sw_t,
            sl_t=sl_t, pt_t=pt_t, mde_t=mde_t, hd_t=hd_t,
            gate_ok=gate_ok, cooldown_bars=5,
            family_code=fam, K=K,
        )

    # Force-close
    if state.open_mask.any():
        fc_pnl   = torch.where(state.open_mask, -sl_t.abs() * 0.5,
                               torch.zeros_like(state.equity))
        equity_t = torch.where(state.open_mask, state.equity + fc_pnl, state.equity)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t_start


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_benchmark(T_default: int = 2000, iters: int = 3):
    device_gpu  = torch.device("cuda") if torch.cuda.is_available() else None
    device_cpu  = torch.device("cpu")

    print(f"\nStage 3A Compile Benchmark")
    if device_gpu:
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"CUDA  : {torch.version.cuda}")
    else:
        print("GPU   : not available — CPU-only benchmark")
    print(f"Torch : {torch.__version__}")
    print(f"T bars: {T_default}")
    print(f"Iters : {iters} (best-of for steady-state)")

    K_vals = [32, 128]
    T_vals = [500, T_default]

    rows = []

    print()
    print("=" * 100)
    print(f"{'Mode':<22} {'K':>5} {'T':>6} {'wall_s':>8} "
          f"{'bars/s':>10} {'vs Stage2A':>12} {'trades':>8}")
    print("=" * 100)

    for T in T_vals:
        # Stage 2A baseline (existing GPU engine)
        stage2a_wall: dict[int, float] = {}

        for K in K_vals:
            cands = build_candidates(K)

            if device_gpu:
                ctx_gpu = build_ctx(device_gpu, T)
                # ── Stage 2A baseline ────────────────────────────────────────
                times_2a = []
                for _ in range(iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    r = run_backtest_optimizer_batch(
                        ctx_gpu, cands,
                        base_config={},
                        objective_spec=_OBJ_SPEC,
                        strategy_idx_filter=8,
                        strategy_family="iron_butterfly",
                        gpu_k_threshold=K,   # GPU path
                    )
                    torch.cuda.synchronize()
                    times_2a.append(time.perf_counter() - t0)
                w2a     = min(times_2a)
                bars2a  = T / w2a
                trades2a = int(r.total.sum())
                stage2a_wall[K] = w2a

                print(f"{'Stage2A GPU eager':<22} {K:>5} {T:>6} {w2a:>8.3f} "
                      f"{bars2a:>10.0f} {'(baseline)':>12} {trades2a:>8}")
                rows.append({
                    "mode": "stage2a_gpu_eager", "K": K, "T": T,
                    "wall_s": round(w2a, 4),
                    "bars_per_s": round(bars2a, 1),
                    "speedup_vs_stage2a": 1.0,
                    "trades": trades2a,
                })

                # ── step_bar_gpu eager ───────────────────────────────────────
                times_eager = []
                for _ in range(iters):
                    times_eager.append(run_step_bar_loop(ctx_gpu, cands, step_fn=step_bar_gpu))
                w_eager  = min(times_eager)
                bars_eg  = T / w_eager
                speedup_eg = w2a / w_eager

                print(f"{'step_bar eager':<22} {K:>5} {T:>6} {w_eager:>8.3f} "
                      f"{bars_eg:>10.0f} {speedup_eg:>11.2f}x {trades2a:>8}")
                rows.append({
                    "mode": "step_bar_eager", "K": K, "T": T,
                    "wall_s": round(w_eager, 4),
                    "bars_per_s": round(bars_eg, 1),
                    "speedup_vs_stage2a": round(speedup_eg, 3),
                    "trades": trades2a,
                })

                # ── step_bar_gpu compiled — warm-up cost ─────────────────────
                compiled_step = make_compiled_step()

                t_warmup_start = time.perf_counter()
                run_step_bar_loop(ctx_gpu, cands, step_fn=compiled_step)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                w_warmup = time.perf_counter() - t_warmup_start

                print(f"{'step_bar compile(1st)':<22} {K:>5} {T:>6} {w_warmup:>8.3f} "
                      f"{'(warm-up)':>10} {w2a/w_warmup:>11.2f}x {trades2a:>8}")
                rows.append({
                    "mode": "step_bar_compile_warmup", "K": K, "T": T,
                    "wall_s": round(w_warmup, 4),
                    "bars_per_s": round(T / w_warmup, 1),
                    "speedup_vs_stage2a": round(w2a / w_warmup, 3),
                    "trades": trades2a,
                    "note": "includes JIT compilation cost",
                })

                # ── step_bar_gpu compiled — steady-state ─────────────────────
                times_comp = []
                for _ in range(iters):
                    times_comp.append(run_step_bar_loop(ctx_gpu, cands, step_fn=compiled_step))
                w_comp   = min(times_comp)
                bars_c   = T / w_comp
                speedup_c = w2a / w_comp

                print(f"{'step_bar compile(SS)':<22} {K:>5} {T:>6} {w_comp:>8.3f} "
                      f"{bars_c:>10.0f} {speedup_c:>11.2f}x {trades2a:>8}")
                rows.append({
                    "mode": "step_bar_compile_steady", "K": K, "T": T,
                    "wall_s": round(w_comp, 4),
                    "bars_per_s": round(bars_c, 1),
                    "speedup_vs_stage2a": round(speedup_c, 3),
                    "trades": trades2a,
                    "note": "steady-state (post-warmup)",
                })

            else:
                # CPU-only path — compare stage2a CPU vs step_bar_gpu eager
                ctx_cpu = build_ctx(device_cpu, T)

                times_2a = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    r = run_backtest_optimizer_batch(
                        ctx_cpu, cands,
                        base_config={},
                        objective_spec=_OBJ_SPEC,
                        strategy_idx_filter=8,
                        strategy_family="iron_butterfly",
                        gpu_k_threshold=K + 1,  # CPU path
                    )
                    times_2a.append(time.perf_counter() - t0)
                w2a = min(times_2a)
                bars2a = T / w2a
                stage2a_wall[K] = w2a
                print(f"{'Stage2A CPU eager':<22} {K:>5} {T:>6} {w2a:>8.3f} "
                      f"{bars2a:>10.0f} {'(baseline)':>12} {int(r.total.sum()):>8}")
                rows.append({
                    "mode": "stage2a_cpu_eager", "K": K, "T": T,
                    "wall_s": round(w2a, 4),
                    "bars_per_s": round(bars2a, 1),
                    "speedup_vs_stage2a": 1.0,
                    "trades": int(r.total.sum()),
                })

                times_eg = []
                for _ in range(iters):
                    times_eg.append(run_step_bar_loop(ctx_cpu, cands, step_fn=step_bar_gpu))
                w_eg = min(times_eg)
                bars_eg = T / w_eg
                print(f"{'step_bar eager (CPU)':<22} {K:>5} {T:>6} {w_eg:>8.3f} "
                      f"{bars_eg:>10.0f} {w2a/w_eg:>11.2f}x {int(r.total.sum()):>8}")
                rows.append({
                    "mode": "step_bar_eager_cpu", "K": K, "T": T,
                    "wall_s": round(w_eg, 4),
                    "bars_per_s": round(bars_eg, 1),
                    "speedup_vs_stage2a": round(w2a / w_eg, 3),
                    "trades": int(r.total.sum()),
                })

        print()

    # ── Save outputs ───────────────────────────────────────────────────────────
    meta = {
        "torch_version": torch.__version__,
        "cuda_version":  torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name":      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "T_default":     T_default,
        "iters":         iters,
    }

    out_dir = _PROJECT_ROOT / "reports" / "gpu_transition"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "stage3a_compile_benchmark.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    print(f"Saved: {json_path}")

    md_lines = [
        "# Stage 3A Compile Benchmark",
        "",
        f"GPU: {meta['gpu_name'] or 'N/A'}  "
        f"Torch: {meta['torch_version']}  CUDA: {meta['cuda_version'] or 'N/A'}",
        f"T_default: {T_default}  Iters: {iters} (best-of steady-state)",
        "",
        "| Mode | K | T | wall_s | bars/s | vs_Stage2A | trades |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        speedup = r.get("speedup_vs_stage2a", "N/A")
        speedup_str = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else speedup
        note = f" _{r['note']}_" if "note" in r else ""
        md_lines.append(
            f"| {r['mode']}{note} | {r['K']} | {r['T']} "
            f"| {r['wall_s']:.3f} | {r['bars_per_s']:.0f} "
            f"| {speedup_str} | {r['trades']} |"
        )

    md_path = out_dir / "stage3a_compile_benchmark.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3A compile benchmark")
    parser.add_argument("--T",     type=int, default=2000,
                        help="Number of bars for large benchmark (default: 2000)")
    parser.add_argument("--iters", type=int, default=3,
                        help="Steady-state repetitions (best-of, default: 3)")
    args = parser.parse_args()
    run_benchmark(T_default=args.T, iters=args.iters)
