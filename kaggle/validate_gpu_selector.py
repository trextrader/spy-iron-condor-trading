"""
validate_gpu_selector.py
========================
Step B: dual-run validation — compare GPU strike selector against existing
CPU implementation on real bars from the loaded chain.

Usage (run on Lightning AI after `git pull`):

    python kaggle/validate_gpu_selector.py \
        --bars 200 --K 16 --family iron_butterfly \
        2>&1 | tee validate_gpu.log

If all bars pass, runs a quick benchmark and prints per-bar timings.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from optimizer_prep import OptimizerContext, build_optimizer_context
from candidate_codec import CandidateBatch
import gpu_strike_selector as _gss

# CPU functions for reference
from optimizer_engine import (
    _find_best_structure_iron_bfly,
    _find_best_structure_iron_condor,
    _find_best_structure_short_call,
    _find_best_structure_short_put,
)


def _make_random_candidates(K: int, device: torch.device) -> tuple:
    """Return (td_t, sd_t, sw_t, td_np, sd_np, sw_np) for K candidates."""
    rng = np.random.default_rng(42)
    td_np = rng.integers(7, 29, K).astype(np.float32)
    sd_np = rng.uniform(0.15, 0.45, K).astype(np.float32)
    sw_np = rng.integers(5, 21, K).astype(np.float32)
    td_t  = torch.from_numpy(td_np).to(device)
    sd_t  = torch.from_numpy(sd_np).to(device)
    sw_t  = torch.from_numpy(sw_np).to(device)
    return td_t, sd_t, sw_t, td_np, sd_np, sw_np


def validate_bar(ctx, bar_idx, td_t, sd_t, sw_t,
                 td_np, sd_np, sw_np, family, spot, K, device):
    s_off = int(ctx.bar_offsets[bar_idx].item())
    e_off = int(ctx.bar_offsets[bar_idx + 1].item())
    if s_off >= e_off:
        return None  # no chain for this bar

    # GPU slices
    r_t  = ctx.option_right[s_off:e_off]
    s_t  = ctx.option_strike[s_off:e_off]
    d_t  = ctx.option_dte[s_off:e_off]
    da_t = ctx.option_delta[s_off:e_off]
    b_t  = ctx.opt_bid[s_off:e_off]
    a_t  = ctx.opt_ask[s_off:e_off]

    # CPU slices (for reference CPU call)
    r_np  = r_t.cpu().numpy()
    s_np  = s_t.cpu().numpy()
    d_np  = d_t.cpu().numpy()
    da_np = da_t.cpu().numpy()
    b_np  = b_t.cpu().numpy()
    a_np  = a_t.cpu().numpy()

    # ── GPU result ──────────────────────────────────────────────────────────
    gpu = _gss.select_entry_for_bar(
        r_t, s_t, d_t, da_t, b_t, a_t, spot,
        td_t, sd_t, sw_t, family,
    )

    # ── CPU reference result ────────────────────────────────────────────────
    if family == "iron_butterfly":
        cpu_cr, cpu_ssc, cpu_aw, cpu_val, cpu_dte = _find_best_structure_iron_bfly(
            spot, r_np, s_np, d_np, da_np, b_np, a_np, td_np, sd_np, sw_np
        )
        cpu_ssp = cpu_ssc.copy()
    elif family == "iron_condor":
        cpu_cr, cpu_ssc, cpu_ssp, cpu_aw, cpu_val, cpu_dte = _find_best_structure_iron_condor(
            spot, r_np, s_np, d_np, da_np, b_np, a_np, td_np, sd_np, sw_np
        )
    elif family == "short_call":
        cpu_cr, cpu_ssc, cpu_val, cpu_dte = _find_best_structure_short_call(
            spot, r_np, s_np, d_np, da_np, b_np, a_np, td_np, sd_np
        )
        cpu_ssp = cpu_ssc.copy()
        cpu_aw  = np.zeros(K, np.float32)
    elif family == "short_put":
        cpu_cr, cpu_ssp, cpu_val, cpu_dte = _find_best_structure_short_put(
            spot, r_np, s_np, d_np, da_np, b_np, a_np, td_np, sd_np
        )
        cpu_ssc = cpu_ssp.copy()
        cpu_aw  = np.zeros(K, np.float32)
    else:
        return None

    ok = _gss.validate_against_cpu(
        gpu, cpu_cr, cpu_ssc, cpu_ssp, cpu_aw, cpu_val, cpu_dte
    )
    return ok


def bench(ctx, bar_idx, td_t, sd_t, sw_t, td_np, sd_np, sw_np, family, spot, K, device):
    s_off = int(ctx.bar_offsets[bar_idx].item())
    e_off = int(ctx.bar_offsets[bar_idx + 1].item())
    if s_off >= e_off:
        return None, None

    r_t  = ctx.option_right[s_off:e_off]
    s_t  = ctx.option_strike[s_off:e_off]
    d_t  = ctx.option_dte[s_off:e_off]
    da_t = ctx.option_delta[s_off:e_off]
    b_t  = ctx.opt_bid[s_off:e_off]
    a_t  = ctx.opt_ask[s_off:e_off]
    r_np = r_t.cpu().numpy(); s_np = s_t.cpu().numpy(); d_np = d_t.cpu().numpy()
    da_np = da_t.cpu().numpy(); b_np = b_t.cpu().numpy(); a_np = a_t.cpu().numpy()

    gpu_time = _gss.bench_gpu(
        _gss.select_entry_for_bar,
        r_t, s_t, d_t, da_t, b_t, a_t, spot, td_t, sd_t, sw_t, family,
        warmup=5, iters=50,
    )

    if family == "iron_butterfly":
        cpu_fn = lambda: _find_best_structure_iron_bfly(
            spot, r_np, s_np, d_np, da_np, b_np, a_np, td_np, sd_np, sw_np)
    elif family == "short_call":
        cpu_fn = lambda: _find_best_structure_short_call(
            spot, r_np, s_np, d_np, da_np, b_np, a_np, td_np, sd_np)
    else:
        cpu_fn = lambda: _find_best_structure_iron_condor(
            spot, r_np, s_np, d_np, da_np, b_np, a_np, td_np, sd_np, sw_np)

    import time
    cpu_times = []
    for _ in range(50):
        t0 = time.perf_counter(); cpu_fn(); cpu_times.append(time.perf_counter() - t0)
    cpu_time = sum(cpu_times) / len(cpu_times)

    return gpu_time, cpu_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars",   type=int, default=200)
    ap.add_argument("--K",      type=int, default=16)
    ap.add_argument("--family", default="iron_butterfly",
                    choices=["iron_butterfly", "iron_condor", "short_call", "short_put"])
    ap.add_argument("--data-dir", default="data/Datasetv4/v43")
    ap.add_argument("--v43-model", default="models/condor_net_v43_run18.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[validate] device={device}  K={args.K}  bars={args.bars}  family={args.family}")

    # Build ctx — needs v43_outputs, bundle, and chain_df_by_date.
    # Use dummy v43 outputs (all zeros) since validation only tests strike search.
    from backtest_v45_data import load_multi_tf_bundle
    bundle = load_multi_tf_bundle(args.data_dir)

    T = len(bundle.m5_df_raw)
    dummy_v43 = {
        "entry_signal": np.ones(T, dtype=np.float32) * 0.8,   # always above threshold
        "pop":          np.ones(T, dtype=np.float32) * 0.8,
        "strategy_idx": np.full(T, 8, dtype=np.int16),         # iron_butterfly class
        "abstain":      np.zeros(T, dtype=bool),
    }

    # chain_df_by_date from bundle
    chain_df_by_date = bundle.chain_df_by_date

    ctx = build_optimizer_context(dummy_v43, bundle, chain_df_by_date, device=device)

    td_t, sd_t, sw_t, td_np, sd_np, sw_np = _make_random_candidates(args.K, device)

    # Sample bars from the chain
    bar_offsets_np = ctx.bar_offsets.cpu().numpy()
    has_chain = (bar_offsets_np[1:] - bar_offsets_np[:-1]) > 0
    chain_bars = np.where(has_chain)[0]
    sample = chain_bars[np.linspace(0, len(chain_bars)-1, args.bars, dtype=int)]

    spot = float(ctx.spot[sample[0]].item())

    # ── Validation pass ────────────────────────────────────────────────────
    n_pass = 0; n_fail = 0; n_skip = 0
    for bi in sample:
        sp = float(ctx.spot[bi].item())
        ok = validate_bar(ctx, bi, td_t, sd_t, sw_t, td_np, sd_np, sw_np,
                          args.family, sp, args.K, device)
        if ok is None:
            n_skip += 1
        elif ok:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n[validate] Results: PASS={n_pass}  FAIL={n_fail}  SKIP(no chain)={n_skip}")

    if n_fail > 0:
        print("[validate] ❌  GPU selector disagrees with CPU on some bars — check output above")
        sys.exit(1)
    else:
        print("[validate] ✅  All bars match CPU reference")

    # ── Benchmark pass ─────────────────────────────────────────────────────
    bench_bars = sample[:10]
    gpu_times = []; cpu_times = []
    for bi in bench_bars:
        sp = float(ctx.spot[bi].item())
        gt, ct = bench(ctx, bi, td_t, sd_t, sw_t, td_np, sd_np, sw_np,
                       args.family, sp, args.K, device)
        if gt is not None:
            gpu_times.append(gt); cpu_times.append(ct)

    if gpu_times:
        avg_gpu = sum(gpu_times) / len(gpu_times) * 1e6
        avg_cpu = sum(cpu_times) / len(cpu_times) * 1e6
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else float("inf")
        print(f"\n[bench]  K={args.K}  family={args.family}")
        print(f"  CPU per bar: {avg_cpu:.1f} µs")
        print(f"  GPU per bar: {avg_gpu:.1f} µs")
        print(f"  Speedup:     {speedup:.1f}×")


if __name__ == "__main__":
    main()
