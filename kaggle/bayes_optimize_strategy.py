"""
bayes_optimize_strategy.py — Bayesian Optimisation Driver
==========================================================
Multi-fidelity BO loop:
  1. Sobol warmup  (bo_init_trials candidates, fast fidelity)
  2. BO rounds     (bo_rounds × bo_batch_size, medium fidelity)
  3. Final eval    (top-k candidates, full fidelity)

BoTorch (SingleTaskGP + qLogNEI) is used when available.
Falls back to local perturbation around the current best if not.

CLI usage (via condor_brain_backtest_v45.py --optimize --optimize-mode bayes):
  --bo-init-trials 32
  --bo-batch-size  16
  --bo-rounds      8
  --bo-min-trades  5
  --bo-max-dd-cap  10.0
  --bo-min-pf      1.0
"""
from __future__ import annotations

import copy
import csv
import datetime
import io
import contextlib
import os
import time
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from optimizer_prep     import OptimizerContext, build_optimizer_context
from candidate_codec    import (SearchSpaceSpec, CandidateBatch, build_search_space,
                                 encode_config, decode_candidate_tensor,
                                 sobol_candidates, perturb_best, candidates_to_configs)
from optimizer_engine   import (run_backtest_optimizer_batch, ObjectiveSpec,
                                 BatchEvalResult, ENTRY_THRESHOLD, POP_THRESHOLD)


# ── BoTorch availability ──────────────────────────────────────────────────────
_BOTORCH_OK = False
try:
    from botorch.models              import SingleTaskGP
    from botorch.fit                 import fit_gpytorch_mll
    from botorch.acquisition.logei   import qLogNoisyExpectedImprovement
    from botorch.optim               import optimize_acqf
    from gpytorch.mlls               import ExactMarginalLogLikelihood
    _BOTORCH_OK = True
except ImportError:
    pass


# ── Result record ─────────────────────────────────────────────────────────────

def _make_row(rank: int, cfg: Dict, res: BatchEvalResult, k: int) -> Dict:
    row = {"rank": rank}
    row.update({key: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating, float)) else v)
                for key, v in cfg.items()})
    row["net_pnl"]       = round(float(res.net_pnl[k]),   2)
    row["net_pct"]       = round(float(res.net_pct[k]),   2)
    row["max_dd"]        = round(float(res.max_dd[k]),    2)
    row["objective"]     = round(float(res.objective[k]), 4)
    row["wins"]          = int(res.wins[k])
    row["losses"]        = int(res.losses[k])
    row["win_rate"]      = round(float(res.win_rate[k]),  4)
    row["profit_factor"] = round(float(res.profit_factor[k]), 3)
    row["fidelity"]      = res.fidelity
    return row


def _save_results(rows: List[Dict], csv_path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fit_gp_and_suggest(
    X_obs: torch.Tensor,   # [N, D] in [0,1]
    Y_obs: torch.Tensor,   # [N, 1]
    K:     int,
    space: SearchSpaceSpec,
) -> torch.Tensor:
    """Fit GP and return K new candidates via qLogNEI. Returns [K, D]."""
    from botorch.models import SingleTaskGP
    from botorch.fit    import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.optim  import optimize_acqf
    from gpytorch.mlls  import ExactMarginalLogLikelihood

    bounds = torch.zeros(2, space.dim, dtype=torch.double)
    bounds[1] = 1.0

    model = SingleTaskGP(X_obs.double(), Y_obs.double())
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    acqf = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=X_obs.double(),
    )
    candidates, _ = optimize_acqf(
        acqf,
        bounds=bounds.double(),
        q=K,
        num_restarts=5,
        raw_samples=128,
    )
    return candidates.float()          # [K, D] in [0,1]


# ── Main entry point ──────────────────────────────────────────────────────────

def run_bayes_optimizer(
    run_backtest_fn,           # original run_backtest for parity / full eval fallback
    args,                      # parsed argparse Namespace
    v43_outputs: Dict,
    bundle,                    # MultiTFDataBundle
    chain_df_by_date: Dict,
    strategy_configs: Dict,
    allowed_template_ids,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """
    Entry point called from condor_brain_backtest_v45.py when
    --optimize --optimize-mode bayes is passed.
    """
    # ── 0. Resolve template ───────────────────────────────────────────────
    if allowed_template_ids and len(allowed_template_ids) == 1:
        template_id = list(allowed_template_ids)[0]
    elif allowed_template_ids:
        print("[bayes_opt] Multiple templates. Using first:", sorted(allowed_template_ids)[0])
        template_id = sorted(allowed_template_ids)[0]
    else:
        print("[bayes_opt] ERROR: --strategies must specify a template (e.g. --strategies iron_butterfly)")
        return

    base_cfg = strategy_configs.get(template_id, {})
    space    = build_search_space(template_id)
    obj_spec = ObjectiveSpec(
        min_trades  = getattr(args, 'bo_min_trades',  5),
        max_dd_cap  = getattr(args, 'bo_max_dd_cap',  10.0),
        min_pf      = getattr(args, 'bo_min_pf',      1.0),
    )

    bo_init   = getattr(args, 'bo_init_trials', 32)
    bo_batch  = getattr(args, 'bo_batch_size',  16)
    bo_rounds = getattr(args, 'bo_rounds',       8)

    print()
    print("=" * 68)
    print(f"  BAYESIAN OPTIMIZER  [{template_id}]")
    print(f"  search_dim={space.dim}  init={bo_init}  batch={bo_batch}  rounds={bo_rounds}")
    print(f"  BoTorch: {'available ✓' if _BOTORCH_OK else 'NOT FOUND — using local perturbation fallback'}")
    print("=" * 68)

    # ── 1. Build immutable context ────────────────────────────────────────
    print("[bayes_opt] Building OptimizerContext (CSR tensor prep)…")
    ctx = build_optimizer_context(v43_outputs, bundle, chain_df_by_date, device)
    print(f"[bayes_opt] Fidelity bar counts:")
    print(f"  fast   (25%): {ctx.fast_end:>6} bars")
    print(f"  medium (60%): {ctx.medium_end:>6} bars")
    print(f"  full  (100%): {ctx.T:>6} bars")

    # ── 2. Output CSV ─────────────────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    ts_tag   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("reports", f"bo_{template_id}_{ts_tag}.csv")
    print(f"[bayes_opt] Results → {csv_path}")

    all_rows: List[Dict] = []
    X_obs_list: List[np.ndarray] = []   # [D] normalised
    Y_obs_list: List[float]      = []   # scalar objective (full fidelity)

    # Seed GP with current base config
    x0 = encode_config(base_cfg, space)
    X_obs_list.append(x0)

    # ── 3. Phase 1: Sobol warmup (fast fidelity) ─────────────────────────
    print(f"\n[Phase 1] Sobol warmup — {bo_init} candidates at FAST fidelity")
    sobol_batch = sobol_candidates(bo_init, space, seed=42)
    configs_p1  = candidates_to_configs(sobol_batch, base_cfg)

    # Print first 3 Sobol candidates so user can see the decoded param spread
    print(f"  First 3 Sobol candidates (sample of search coverage):")
    p1_cfgs_preview = candidates_to_configs(sobol_batch, base_cfg)
    for ki in range(min(3, bo_init)):
        cfg_preview = {k: v for k, v in p1_cfgs_preview[ki].items() if k in [p.name for p in space.params]}
        print(f"    cand[{ki}]: " + "  ".join(f"{k}={v}" for k, v in cfg_preview.items()))

    t0 = time.time()
    res_p1 = run_backtest_optimizer_batch(
        ctx, sobol_batch,
        base_config=base_cfg, objective_spec=obj_spec, fidelity="fast",
        strategy_idx_filter=_class_idx_for_template(template_id, strategy_configs),
        verbose=False,   # suppress bar-level detail for warmup speed
    )
    elapsed_p1 = time.time() - t0
    best_k_p1  = int(np.nanargmax(res_p1.objective))
    print(f"  done in {elapsed_p1:.1f}s  best_obj={res_p1.objective[best_k_p1]:.3f}  "
          f"(cand[{best_k_p1}]: trades={res_p1.total[best_k_p1]}  "
          f"net={res_p1.net_pct[best_k_p1]:+.1f}%  dd={res_p1.max_dd[best_k_p1]:.2f}%)")
    print(f"  Phase 1 full results:")
    print(f"  {'k':>3}  {'obj':>8}  {'trades':>6}  {'wins':>5}  {'loss':>5}  "
          f"{'net%':>7}  {'dd%':>6}")
    for ki in range(bo_init):
        marker = " <-- best" if ki == best_k_p1 else ""
        print(f"  {ki:>3}  {res_p1.objective[ki]:>8.3f}  {res_p1.total[ki]:>6}  "
              f"{res_p1.wins[ki]:>5}  {res_p1.losses[ki]:>5}  "
              f"{res_p1.net_pct[ki]:>7.2f}  {res_p1.max_dd[ki]:>6.2f}{marker}")

    for k in range(bo_init):
        row = _make_row(0, configs_p1[k], res_p1, k)
        all_rows.append(row)
        x_enc = encode_config(configs_p1[k], space)
        X_obs_list.append(x_enc)
        Y_obs_list.append(float(res_p1.objective[k]))

    _save_results(sorted(all_rows, key=lambda r: -r["objective"])[:100], csv_path)

    # ── 4. Phase 2: BO rounds (medium fidelity) ──────────────────────────
    print(f"\n[Phase 2] BO rounds — {bo_rounds} rounds × {bo_batch} candidates at MEDIUM fidelity")

    best_x = X_obs_list[1 + int(np.nanargmax(res_p1.objective))]  # best from warmup

    for rnd in range(bo_rounds):
        seed_rnd = 1000 + rnd
        if _BOTORCH_OK and len(Y_obs_list) >= 4:
            try:
                X_t = torch.tensor(np.array(X_obs_list), dtype=torch.float32)
                Y_t = torch.tensor(Y_obs_list, dtype=torch.float32).unsqueeze(-1)
                # Normalise Y
                y_mean, y_std = Y_t.mean(), Y_t.std().clamp_min(1e-6)
                Y_norm = (Y_t - y_mean) / y_std
                x_next = _fit_gp_and_suggest(X_t, Y_norm, bo_batch, space)
                cand_batch = decode_candidate_tensor(x_next, space)
            except Exception as e:
                print(f"  [BO round {rnd}] GP fit failed ({e}), falling back to perturbation")
                cand_batch = perturb_best(best_x, bo_batch, space, sigma=0.15, seed=seed_rnd)
        else:
            cand_batch = perturb_best(best_x, bo_batch, space, sigma=0.20, seed=seed_rnd)

        configs_rnd = candidates_to_configs(cand_batch, base_cfg)
        t0 = time.time()
        res_rnd = run_backtest_optimizer_batch(
            ctx, cand_batch,
            base_config=base_cfg, objective_spec=obj_spec, fidelity="medium",
            strategy_idx_filter=_class_idx_for_template(template_id, strategy_configs),
            verbose=False,
        )
        elapsed_rnd = time.time() - t0
        best_k   = int(np.nanargmax(res_rnd.objective))
        best_obj = float(res_rnd.objective[best_k])
        print(f"  Round {rnd+1:>2}/{bo_rounds}  best_obj={best_obj:.3f}  "
              f"net_pct={res_rnd.net_pct[best_k]:+.1f}%  "
              f"dd={res_rnd.max_dd[best_k]:.1f}%  "
              f"trades={res_rnd.total[best_k]}  "
              f"({elapsed_rnd:.1f}s)")
        print(f"    All {bo_batch} candidates this round:")
        print(f"    {'k':>3}  {'obj':>8}  {'trades':>6}  {'wins':>5}  {'loss':>5}  {'net%':>7}  {'dd%':>6}")
        for ki in range(bo_batch):
            marker = " <--" if ki == best_k else ""
            print(f"    {ki:>3}  {res_rnd.objective[ki]:>8.3f}  {res_rnd.total[ki]:>6}  "
                  f"{res_rnd.wins[ki]:>5}  {res_rnd.losses[ki]:>5}  "
                  f"{res_rnd.net_pct[ki]:>7.2f}  {res_rnd.max_dd[ki]:>6.2f}{marker}")

        for k in range(bo_batch):
            row = _make_row(0, configs_rnd[k], res_rnd, k)
            all_rows.append(row)
            x_enc = encode_config(configs_rnd[k], space)
            X_obs_list.append(x_enc)
            Y_obs_list.append(float(res_rnd.objective[k]))

        # Update best_x
        best_global_idx = int(np.nanargmax(Y_obs_list))
        best_x = X_obs_list[best_global_idx]

        _save_results(sorted(all_rows, key=lambda r: -r["objective"])[:100], csv_path)

    # ── 5. Phase 3: Full-fidelity eval of top-k ───────────────────────────
    top_k = min(bo_batch, len(all_rows))
    sorted_rows = sorted(all_rows, key=lambda r: -r["objective"])[:top_k]
    print(f"\n[Phase 3] Full-fidelity re-evaluation of top {top_k} candidates")

    # Re-encode top configs back into CandidateBatch
    top_cfgs   = [r for r in sorted_rows]
    top_params: Dict[str, np.ndarray] = {}
    for p in space.params:
        top_params[p.name] = np.array(
            [float(cfg.get(p.name, p.lo)) for cfg in top_cfgs], dtype=np.float32
        )
    top_batch = CandidateBatch(K=len(top_cfgs), params=top_params)

    t0 = time.time()
    res_full = run_backtest_optimizer_batch(
        ctx, top_batch,
        base_config=base_cfg, objective_spec=obj_spec, fidelity="full",
        strategy_idx_filter=_class_idx_for_template(template_id, strategy_configs),
        verbose=verbose,   # full diagnostic when --verbose flag is set
    )
    print(f"  Full-fidelity done in {time.time()-t0:.1f}s")

    final_rows = []
    for k in range(len(top_cfgs)):
        row = _make_row(k + 1, top_cfgs[k], res_full, k)
        row["fidelity"] = "full"
        final_rows.append(row)

    final_rows.sort(key=lambda r: -r["objective"])
    for i, r in enumerate(final_rows):
        r["rank"] = i + 1

    # Merge into all_rows for final CSV
    final_csv = os.path.join("reports", f"bo_{template_id}_{ts_tag}_final.csv")
    _save_results(final_rows, final_csv)
    print(f"\n[bayes_opt] Final results → {final_csv}")

    # ── 6. Print leaderboard ──────────────────────────────────────────────
    print()
    print("=" * 68)
    print(f"  TOP 10 RESULTS  [{template_id}]  (full fidelity)")
    print("=" * 68)
    _header_keys = [p.name for p in space.params] + ["net_pnl", "net_pct", "max_dd", "objective", "wins", "losses"]
    print("  " + "  ".join(f"{k:>16}" for k in _header_keys))
    for row in final_rows[:10]:
        vals = []
        for k in _header_keys:
            v = row.get(k, "?")
            if isinstance(v, float):
                vals.append(f"{v:>16.3f}")
            else:
                vals.append(f"{str(v):>16}")
        print("  " + "".join(vals))

    # ── 7. Offer to apply best config ────────────────────────────────────
    if final_rows:
        best = final_rows[0]
        print()
        print(f"  Best config (rank 1):  obj={best['objective']:.4f}  "
              f"net={best['net_pct']:+.1f}%  dd={best['max_dd']:.1f}%")
        print()

        # Guard: never apply when all candidates got the "no trades" penalty
        all_penalty = all(r.get("objective", -100) <= -100 for r in final_rows)
        if all_penalty:
            print("  [WARN] All candidates scored -100 (no trades executed — chain data missing?).")
            print("         Skipping apply to avoid overwriting strategy file with invalid params.")
        elif sys.stdin.isatty():
            # Only prompt when running interactively in a real terminal
            print("  Apply best config to strategy file? [y/N]: ", end="", flush=True)
            try:
                ans = input().strip().lower()
            except EOFError:
                ans = ""
            if ans == "y":
                _apply_best_config(template_id, best, space)
        else:
            # Non-interactive mode: print the best config but don't apply automatically
            print("  [non-interactive] Best config NOT auto-applied.")
            print("  To apply manually, run with an interactive terminal.")

    return final_rows


def _class_idx_for_template(template_id: str, strategy_configs: Dict) -> int:
    """Look up the class_idx for the given template from strategy_configs."""
    cfg = strategy_configs.get(template_id, {})
    return int(cfg.get("class_idx", 8))   # 8 = custom_multi_leg / iron_butterfly default


def _apply_best_config(template_id: str, best_row: Dict, space: SearchSpaceSpec):
    """Write best parameters back to the strategy .py file."""
    # Find the strategy file
    candidates = [
        os.path.join("kaggle", "strategies", f"{template_id}.py"),
        os.path.join("strategies", f"{template_id}.py"),
        os.path.join(os.path.dirname(__file__), "strategies", f"{template_id}.py"),
    ]
    strat_file = next((p for p in candidates if os.path.isfile(p)), None)
    if strat_file is None:
        print(f"  [apply] Strategy file not found for '{template_id}'")
        return

    with open(strat_file) as f:
        src = f.read()

    import re
    for p in space.params:
        v = best_row.get(p.name)
        if v is None:
            continue
        # Replace pattern:  "param_name":  <value>,
        pattern = rf'("{p.name}"\s*:\s*)([^,\n]+)'
        if p.dtype == int or isinstance(v, (int, np.integer)):
            repl = rf'\g<1>{int(v)}'
        else:
            repl = rf'\g<1>{float(v):.4f}'
        src = re.sub(pattern, repl, src)

    with open(strat_file, "w") as f:
        f.write(src)
    print(f"  [apply] Updated {strat_file} with best params.")
