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
  --bo-min-trades  12
  --bo-max-dd-cap  10.0
  --bo-min-pf      1.0
"""
from __future__ import annotations

import copy
import csv
import datetime
import hashlib
import io
import json
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
from optimization_intensity      import get_intensity_policy, IntensityPolicy
from adaptive_search_space_builder import (build_adaptive_search_space,
                                            AdaptiveParamManifestRow)

# Register GPU-specific intensity policies (t4-med, a100-sobol, etc.) so that
# build_adaptive_search_space() can resolve them via get_intensity_policy().
try:
    from gpu_profiles import register_gpu_intensity_policies as _reg_gpu_policies
    _reg_gpu_policies()
except ImportError:
    pass


# ── Cross-sweep best-objective registry ──────────────────────────────────────
# Prevents any single sweep from overwriting a better result found in a prior
# sweep.  Updated by apply_best_params.py (authoritative) and by auto-apply
# when a new run genuinely beats the stored best.
_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reports", "best_obj_registry.json"
)


def _load_best_registry() -> dict:
    """Return {strategy_name: {"obj": float, "intensity": str}} or {} if missing."""
    if os.path.exists(_REGISTRY_PATH):
        try:
            with open(_REGISTRY_PATH, "r") as _f:
                return json.load(_f)
        except Exception:
            pass
    return {}


def _save_best_registry(registry: dict) -> None:
    os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
    with open(_REGISTRY_PATH, "w") as _f:
        json.dump(registry, _f, indent=2, sort_keys=True)


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
    row["np_dd"]         = round(float(res.np_dd[k]),     3)
    row["eligible"]      = bool(res.eligible[k])
    row["legacy_objective"] = round(float(res.legacy_objective[k]), 4)
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
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Fit GP and return K new candidates via qLogNEI. Returns [K, D] on CPU."""
    from botorch.models import SingleTaskGP
    from botorch.fit    import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.optim  import optimize_acqf
    from gpytorch.mlls  import ExactMarginalLogLikelihood

    # Move GP tensors to device (GPU if available) — numpy backtest is always CPU
    tkw = dict(dtype=torch.double, device=device)
    X = X_obs.to(**tkw)
    Y = Y_obs.to(**tkw)
    bounds = torch.zeros(2, space.dim, **tkw)
    bounds[1] = 1.0

    import time as _time
    _t0 = _time.time()
    mem_mb = (torch.cuda.memory_allocated(device) // 1024 // 1024
              if device and device.type == "cuda" else 0)
    print(f"  [GP] device={X.device}  N={X.shape[0]} D={X.shape[1]} q={K}  "
          f"vram_used={mem_mb}MB")

    model = SingleTaskGP(X, Y)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    print(f"  [GP] fit done in {_time.time()-_t0:.1f}s")

    _t1 = _time.time()
    acqf = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=X,
    )
    candidates, _ = optimize_acqf(
        acqf,
        bounds=bounds,
        q=K,
        num_restarts=5,
        raw_samples=128,
    )
    print(f"  [GP] acqf optimized in {_time.time()-_t1:.1f}s  "
          f"(total {_time.time()-_t0:.1f}s)")
    return candidates.float().cpu()    # [K, D] in [0,1], always back to CPU


# ── Main entry point ──────────────────────────────────────────────────────────

def _print_adaptive_manifest(
    *,
    template_id: str,
    strategy_family: str,
    policy: IntensityPolicy,
    manifest_rows: List[AdaptiveParamManifestRow],
    search_dim: int,
) -> None:
    """Print the BO banner + parameter manifest for an adaptive run."""
    botorch_str = "available ✓" if _BOTORCH_OK else "NOT FOUND — using local perturbation fallback"

    # Build granularity summary from optimized rows
    seen_kinds: set = set()
    gran_parts: List[str] = []
    for row in manifest_rows:
        if row.status != "OPTIMIZED" or not row.kind or row.inferred_step is None:
            continue
        if row.kind in seen_kinds:
            continue
        seen_kinds.add(row.kind)
        label = {"dollar": "dollar", "integer": "int", "width": "width",
                 "delta": "delta", "percent": "pct", "continuous": "cont"}.get(row.kind, row.kind)
        s = row.inferred_step
        s_str = str(int(s)) if float(s).is_integer() else f"{s:g}"
        gran_parts.append(f"{s_str}-{label}")
    gran_summary = " / ".join(gran_parts) if gran_parts else "auto"

    print()
    print("=" * 72)
    print(f"  BAYESIAN OPTIMIZER  [{template_id}]")
    print(f"  intensity={policy.name}  search_dim={search_dim}  "
          f"init={policy.bo_init_trials}  batch={policy.bo_batch_size}  rounds={policy.bo_rounds}")
    print(f"  granularity={gran_summary}  simulation_family={strategy_family}")
    print(f"  fidelity={policy.fidelity_schedule}  BoTorch: {botorch_str}")
    print("=" * 72)

    print(f"\n[bayes_opt] Parameter manifest for [{template_id}]:")
    print(f"  {'PARAMETER':<22}  {'PRIORITY':<8}  {'STATUS':<14}  {'CURRENT':<10}  RANGE / GRID")
    print(f"  {'-'*22}  {'-'*8}  {'-'*14}  {'-'*10}  {'-'*38}")

    for row in manifest_rows:
        pr_str = "" if row.priority_rank is None else str(row.priority_rank)
        cur_str = str(row.current_value)[:10]

        if row.status == "OPTIMIZED":
            if row.kind in {"delta", "percent", "continuous"}:
                step_s = f"  step≈{row.inferred_step:g}" if row.inferred_step else ""
                rng = f"continuous [{row.lo:g}, {row.hi:g}]{step_s}"
            elif row.grid is not None:
                def _fv(x):
                    return str(int(x)) if float(x).is_integer() else f"{x:g}"
                if len(row.grid) <= 10:
                    rng = f"grid [{', '.join(_fv(x) for x in row.grid)}]"
                else:
                    head = ", ".join(_fv(x) for x in row.grid[:6])
                    s_s = f"(step={int(row.inferred_step)})" if row.inferred_step and float(row.inferred_step).is_integer() else f"(step={row.inferred_step:g})" if row.inferred_step else ""
                    rng = f"grid [{head}, ...]  {s_s}  ({len(row.grid)} pts)"
            else:
                rng = f"[{row.lo:g}, {row.hi:g}]"
        else:
            rng = f"note: {row.notes}" if row.notes else ""

        print(f"  {row.param_name:<22}  {pr_str:<8}  {row.status:<14}  {cur_str:<10}  {rng}")
    print()


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
    auto_apply: bool = False,      # If True: skip prompt and auto-apply rank-1 config
    intensity_name: str = "med",   # Optimizer intensity level
    min_apply_obj: float = 0.0,    # Auto-apply only if rank-1 obj >= this threshold
    gpu_k_threshold: int = 32,     # passed through to run_backtest_optimizer_batch
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

    base_cfg        = strategy_configs.get(template_id, {})
    strategy_family = _strategy_family_for_template(template_id)

    # ── Build adaptive search space (intensity-aware) ─────────────────────
    # Preserve calibrated expert bounds from candidate_codec as explicit_param_bounds;
    # intensity policy controls step sizes and which params are included.
    _old_space = build_search_space(template_id)
    _explicit_bounds = {p.name: (p.lo, p.hi) for p in _old_space.params}

    space, manifest_rows, policy = build_adaptive_search_space(
        strategy_id=template_id,
        strategy_config=base_cfg,
        intensity_name=intensity_name,
        explicit_param_bounds=_explicit_bounds,
    )

    # BO budget comes from policy (overrides manual --bo-* args when intensity is set)
    bo_init   = policy.bo_init_trials
    bo_batch  = policy.bo_batch_size
    bo_rounds = policy.bo_rounds

    obj_spec = ObjectiveSpec(
        min_trades  = getattr(args, 'bo_min_trades',  20),
        max_dd_cap  = getattr(args, 'bo_max_dd_cap',  25.0),
        min_pf      = getattr(args, 'bo_min_pf',      1.1),
    )

    _print_adaptive_manifest(
        template_id=template_id,
        strategy_family=strategy_family,
        policy=policy,
        manifest_rows=manifest_rows,
        search_dim=space.dim,
    )

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
    X_obs_fast: List[np.ndarray] = []   # Phase 1: fast fidelity (25% bars)
    Y_obs_fast: List[float]      = []
    X_obs_med:  List[np.ndarray] = []   # Phase 2: medium fidelity (60% bars)
    Y_obs_med:  List[float]      = []

    # Deterministic per-strategy seed (stable across Python restarts)
    _tmpl_seed = int(hashlib.md5(template_id.encode()).hexdigest(), 16) % 99991

    # ── 3. Phase 1: Sobol warmup (fast fidelity) ─────────────────────────
    print(f"\n[Phase 1] Sobol warmup — {bo_init} candidates at FAST fidelity")
    sobol_seed  = _tmpl_seed
    sobol_batch = sobol_candidates(bo_init, space, seed=sobol_seed)
    print(f"  [seed] Sobol seed={sobol_seed} (from template_id='{template_id}')")
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
        strategy_family=strategy_family,
        verbose=False,   # suppress bar-level detail for warmup speed
        gpu_k_threshold=gpu_k_threshold,
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
        X_obs_fast.append(x_enc)
        Y_obs_fast.append(float(res_p1.objective[k]))

    _save_results(sorted(all_rows, key=lambda r: -r["objective"])[:100], csv_path)

    # ── Early exit: dead strategy (gate never fires) ──────────────────────
    _p1_best = float(np.nanmax(res_p1.objective))
    _p1_trades = int(np.nanmax(res_p1.total))
    if _p1_trades == 0:
        print(f"\n[bayes_opt] SKIP — gate_eligible_bars=0 in Phase 1 fast fidelity.")
        print(f"[bayes_opt] v43 model does not predict class_idx="
              f"{_class_idx_for_template(template_id, strategy_configs)} on this dataset.")
        print(f"[bayes_opt] Skipping Phases 2 & 3 — no trades possible for [{template_id}].\n")
        return []

    # ── 4. Phase 2: BO rounds (medium fidelity) ──────────────────────────
    print(f"\n[Phase 2] BO rounds — {bo_rounds} rounds × {bo_batch} candidates at MEDIUM fidelity")

    best_x = X_obs_fast[int(np.nanargmax(res_p1.objective))]  # best from warmup

    for rnd in range(bo_rounds):
        seed_rnd = _tmpl_seed + rnd * 1000
        if _BOTORCH_OK and len(Y_obs_med) >= 4:
            try:
                X_t = torch.tensor(np.array(X_obs_med), dtype=torch.float32)
                Y_t = torch.tensor(Y_obs_med, dtype=torch.float32).unsqueeze(-1)
                # Normalise Y
                y_mean, y_std = Y_t.mean(), Y_t.std().clamp_min(1e-6)
                Y_norm = (Y_t - y_mean) / y_std
                x_next = _fit_gp_and_suggest(X_t, Y_norm, bo_batch, space, device=device)
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
            strategy_family=strategy_family,
            verbose=False,
            gpu_k_threshold=gpu_k_threshold,
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
            X_obs_med.append(x_enc)
            Y_obs_med.append(float(res_rnd.objective[k]))

        # Update best_x
        if Y_obs_med:
            best_global_idx = int(np.nanargmax(Y_obs_med))
            best_x = X_obs_med[best_global_idx]
        else:
            # No medium obs yet (GP fallback before first round completes)
            best_x = X_obs_fast[int(np.nanargmax(Y_obs_fast))]

        _save_results(sorted(all_rows, key=lambda r: -r["objective"])[:100], csv_path)

    # ── 5. Phase 3: Full-fidelity eval of top-k ───────────────────────────
    # Evaluate at least 10 candidates at full fidelity (or all available if <10)
    # so the TOP 10 leaderboard is always populated.
    top_k = min(max(bo_batch, 10), len(all_rows))
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
        strategy_family=strategy_family,
        verbose=verbose,   # full diagnostic when --verbose flag is set
        gpu_k_threshold=gpu_k_threshold,
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
    n_results = len(final_rows)
    print()
    print("=" * 68)
    print(f"  TOP {n_results} RESULTS  [{template_id}]  (full fidelity)")
    if n_results < 10:
        print(f"  NOTE: only {n_results} candidates evaluated total. "
              f"Use --bo-init-trials 10+ to see 10 results.")
    print("=" * 68)
    # Data keys for row lookup; display labels shown in header (rename wins/losses)
    _data_keys = [p.name for p in space.params] + ["net_pnl", "net_pct", "max_dd", "np_dd", "eligible", "objective", "legacy_objective", "wins", "losses"]
    _hdr_labels = [p.name for p in space.params] + ["net_pnl", "net_pct", "max_dd", "np_dd", "eligible", "objective", "legacy_obj", "#wins", "#losses"]
    print("  " + f"{'#':>4}  " + "  ".join(f"{h:>16}" for h in _hdr_labels))
    for idx, row in enumerate(final_rows, 1):
        vals = []
        for k in _data_keys:
            v = row.get(k, "?")
            if isinstance(v, float):
                vals.append(f"{v:>16.3f}")
            else:
                vals.append(f"{str(v):>16}")
        print(f"  {idx:>4}  " + "".join(vals))

    # ── 7. Offer to apply a config (numbered selection) ───────────────────
    if final_rows:
        print()
        all_penalty = all(not r.get("eligible", False) for r in final_rows)
        if all_penalty:
            print("  [WARN] All candidates failed the ranking guardrails.")
            print("         Skipping apply to avoid overwriting strategy file with invalid params.")
        elif auto_apply:
            # autoall mode: silently apply rank 1 without prompting,
            # but only if it clears the intensity floor AND beats the all-time
            # best objective stored in best_obj_registry.json.
            chosen = final_rows[0]
            _new_obj = chosen['objective']
            _registry = _load_best_registry()
            _prev_best = _registry.get(template_id, {}).get("obj", 0.0)
            wins = int(chosen.get("wins", 0))
            losses = int(chosen.get("losses", 0))
            print("  [audit] rank-1 selection metrics:")
            print(f"          objective     = {chosen.get('objective', 0.0):.4f}")
            print(f"          legacy_obj    = {chosen.get('legacy_objective', 0.0):.4f}")
            print(f"          net_pct       = {chosen.get('net_pct', 0.0):+.2f}%")
            print(f"          max_dd        = {abs(float(chosen.get('max_dd', 0.0))):.2f}%")
            print(f"          np_dd         = {chosen.get('np_dd', 0.0):.3f}")
            print(f"          profit_factor = {chosen.get('profit_factor', 0.0):.3f}")
            print(f"          eligible      = {chosen.get('eligible', False)}")
            print(f"          wins          = {wins}")
            print(f"          losses        = {losses}")
            print(f"          trades        = {wins + losses}")
            if not chosen.get("eligible", False):
                print("  [autoall] Skipping apply - rank 1 failed guardrails.")
            elif _new_obj < min_apply_obj:
                print(f"  [autoall] Skipping apply — obj={_new_obj:.4f} < "
                      f"min_apply_obj={min_apply_obj:.2f}  (existing params preserved)")
            elif _new_obj <= _prev_best:
                print(f"  [autoall] Skipping apply — obj={_new_obj:.4f} ≤ "
                      f"registry best {_prev_best:.4f}  (existing params preserved)")
            else:
                print(f"  [autoall] Auto-applying rank 1:  obj={_new_obj:.4f}  "
                      f"net={chosen['net_pct']:+.1f}%  dd={chosen['max_dd']:.1f}%"
                      + (f"  [NEW BEST vs {_prev_best:.4f}]" if _prev_best > 0 else ""))
                _apply_best_config(template_id, chosen, space)
                _registry[template_id] = {"obj": _new_obj, "intensity": intensity_name}
                _save_best_registry(_registry)
        elif sys.stdin.isatty():
            print(f"  Apply which config? [1–{n_results}, Enter to skip]: ", end="", flush=True)
            try:
                ans = input().strip()
            except EOFError:
                ans = ""
            if ans.isdigit():
                sel = int(ans) - 1   # convert to 0-indexed
                if 0 <= sel < n_results:
                    chosen = final_rows[sel]
                    print(f"  Applying rank {sel + 1}:  obj={chosen['objective']:.4f}  "
                          f"net={chosen['net_pct']:+.1f}%  dd={chosen['max_dd']:.1f}%")
                    _apply_best_config(template_id, chosen, space)
                else:
                    print(f"  '{ans}' out of range 1–{n_results}. Skipped.")
            elif ans == "":
                print("  Skipped.")
            else:
                print(f"  Unrecognised input '{ans}'. Skipped.")
        else:
            # Non-interactive (e.g. piped / notebook): print best but don't auto-apply
            best = final_rows[0]
            print(f"  [non-interactive] Best config (rank 1) NOT auto-applied.")
            print(f"  obj={best['objective']:.4f}  net={best['net_pct']:+.1f}%  dd={best['max_dd']:.1f}%")
            print("  Re-run in an interactive terminal and enter a rank number to apply.")

    return final_rows


# ── Strategy-family map: template_id → simulation engine family ──────────────
# "iron_butterfly" = ATM short call+put + wings (same strike for both shorts)
# "iron_condor"    = OTM short call+put at SEPARATE strikes + wings
# "short_call"     = single-leg naked short call
# "short_put"      = single-leg naked short put
_STRATEGY_FAMILY_MAP: Dict[str, str] = {
    # ── Iron butterfly (ATM same-strike shorts / straddle-based) ──────────
    "iron_butterfly":           "iron_butterfly",
    "inverse_iron_butterfly":   "iron_butterfly",   # long straddle + short wings
    "short_call_butterfly":     "iron_butterfly",
    "short_put_butterfly":      "iron_butterfly",
    "long_call_butterfly":      "iron_butterfly",
    "long_put_butterfly":       "iron_butterfly",
    "short_straddle":           "iron_butterfly",   # ATM call + put shorts
    "straddle_long":            "iron_butterfly",   # ATM call + put longs
    "covered_short_straddle":   "iron_butterfly",   # ATM straddle-like exposure
    "short_guts":               "iron_butterfly",   # ITM straddle short
    "guts":                     "iron_butterfly",   # ITM straddle long
    "strap":                    "iron_butterfly",   # 1P + 2C ATM
    "strip":                    "iron_butterfly",   # 2P + 1C ATM

    # ── Iron condor (OTM separate call/put shorts / strangle-based) ───────
    "iron_condor":              "iron_condor",
    "inverse_iron_condor":      "iron_condor",      # debit condor (OTM body)
    "short_call_condor":        "iron_condor",
    "short_put_condor":         "iron_condor",
    "long_call_condor":         "iron_condor",
    "long_put_condor":          "iron_condor",
    "double_diagonal":          "iron_condor",
    "diagonal_call":            "iron_condor",      # vertical + time spread → IC approx
    "diagonal_put":             "iron_condor",
    "calendar_call":            "iron_condor",      # time spread → IC approx
    "calendar_put":             "iron_condor",
    "collar":                   "iron_condor",      # long put + short call
    "jade_lizard":              "iron_condor",      # short put + short call spread
    "reverse_jade_lizard":      "iron_condor",      # long call + long put spread
    "covered_short_strangle":   "iron_condor",      # OTM call + put shorts
    "short_strangle":           "iron_condor",      # OTM call + put shorts
    "strangle_long":            "iron_condor",      # OTM call + put longs

    # ── Short call (call-side / directional single-leg approx) ────────────
    "short_call":               "short_call",
    "covered_call":             "short_call",
    "bear_call_ladder":         "short_call",
    "bull_call_ladder":         "short_call",       # debit call ladder
    "long_call":                "short_call",
    "call_ratio_backspread":    "short_call",       # sell 1, buy 2 calls
    "call_ratio_spread":        "short_call",       # buy 1, sell 2 calls
    "call_broken_wing":         "short_call",       # asymmetric call butterfly
    "inverse_call_broken_wing": "short_call",       # inverted call BWB
    "bear_call_spread_credit":  "short_call",       # sell call spread
    "bull_call_spread":         "short_call",       # buy call spread
    "long_combo":               "short_call",       # long call + short put (synthetic long)
    "short_combo":              "short_call",       # short call + long put (synthetic short)
    "long_synthetic_future":    "short_call",       # = long_combo
    "short_synthetic_future":   "short_call",       # = short_combo

    # ── Short put (put-side / directional single-leg approx) ──────────────
    "short_put":                "short_put",
    "cash_secured_put":         "short_put",
    "bull_put_ladder":          "short_put",
    "bear_put_ladder":          "short_put",        # debit put ladder
    "long_put":                 "short_put",
    "put_ratio_backspread":     "short_put",        # sell 1, buy 2 puts
    "put_ratio_spread":         "short_put",        # buy 1, sell 2 puts
    "put_broken_wing":          "short_put",        # asymmetric put butterfly
    "inverse_put_broken_wing":  "short_put",        # inverted put BWB
    "bear_put_spread":          "short_put",        # buy put spread
    "bull_put_spread_credit":   "short_put",        # sell put spread
    "protective_put":           "short_put",        # long stock + long put
    "synthetic_put":            "short_put",        # long call + short stock
}


def _strategy_family_for_template(template_id: str) -> str:
    """Return the simulation engine family for a given template."""
    fam = _STRATEGY_FAMILY_MAP.get(template_id)
    if fam is None:
        raise KeyError(
            f"[bayes_opt] No simulation family for {template_id!r}. "
            f"Add it to _STRATEGY_FAMILY_MAP in bayes_optimize_strategy.py."
        )
    return fam


def _class_idx_for_template(template_id: str, strategy_configs: Dict) -> int:
    """Look up the class_idx for the given template from strategy_configs."""
    cfg = strategy_configs.get(template_id, {})
    idx = int(cfg.get("class_idx", 8))   # 8 = custom_multi_leg / iron_butterfly default
    print(f"[bayes_opt] template={template_id!r}  class_idx={idx}  "
          f"(gate: sidx == {idx} in v43 predictions)")
    return idx


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
