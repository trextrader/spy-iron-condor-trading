#!/usr/bin/env python3
"""
audit_condornet_interpretability_v43.py — CondorNet v4.3 Full Interpretability Audit
==================================================================================

This script has two modes:

A) Weight-only audit (fast; no data required)
   - Predicate thresholds (v42 backbone canonical gates)
   - SuperSet relational logic (operator mix + top feature-pair comparisons)
   - Strategy / risk / pivot head weight diagnostics
   - Multi-TF projector contributions
   - A-matrix and B-matrix stability summaries (+ optional CSV export)
   - Strategy template catalog introspection (58+ templates)

B) Data-based audit (optional; requires --data CSV)
   - Permutation importance (feature → output sensitivity)
   - Gradient saliency (∂output/∂input feature)
   - Mutual information (feature → output dependency, if sklearn available)
   - Surrogate decision tree (approximate decision rules; if sklearn available)
   - Fisher-diagonal proxy (parameter sensitivity; lightweight)
   - Hessian top-eigen proxy (loss curvature; lightweight, optional)

Outputs:
   - JSON summary (always)
   - Optional matrices CSVs (--export-matrices)
   - Optional plots + Markdown report (--output-dir)

Usage examples:

  # Weight-only:
  python intelligence/audit_condornet_interpretability_v43.py \
      --model models/condornet_v43_epoch27.pth \
      --output-json reports/v43_logic.json \
      --export-matrices --verbose

  # Full audit with data (single-TF compatibility path):
  python intelligence/audit_condornet_interpretability_v43.py \
      --model models/condornet_v43_epoch27.pth \
      --data data/Datasetv4/v43/m5_dataset_v43_final.csv \
      --samples 3000 --seq-len 16 \
      --output-dir reports/v43_audit_epoch27 \
      --output-json reports/v43_audit_epoch27/logic.json \
      --export-matrices --verbose

Notes
-----
- The data-based path uses CondorNetV43.forward_compat(x) so it works with TF-only datasets.
- If sklearn isn't installed, the script will skip MI + decision tree (still runs the rest).

Version: 4.3.1 (expanded)
Date: 2026-02-27
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# Ensure repo root import works when script is called as `python intelligence/...`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from intelligence.condor_brain_net_v43 import CondorNetV43, build_condornet_v43
from intelligence.schema_v43 import (
    SCHEMA_VERSION,
    STRATEGY_TYPES,
    N_STRATEGY_TYPES,
    TF_FEATURE_NAMES,
    FRICTION_FEATURE_NAMES,
    TOD_FEATURE_NAMES,
    REGIME_PERSISTENCE_FEATURE,
    IVR_REVERSAL_FEATURE_NAMES,
    N_PIVOT_FEATURES,
    CHAIN_GRID_CONFIG,
)

try:
    from intelligence.strategy_templates_v43 import STRATEGY_TEMPLATES  # type: ignore
except Exception:
    STRATEGY_TEMPLATES = {}

# Optional heavy deps (guarded)
try:
    import pandas as pd  # type: ignore
except Exception as e:
    pd = None

try:
    from sklearn.tree import DecisionTreeRegressor  # type: ignore
    from sklearn.metrics import r2_score  # type: ignore
    from sklearn.feature_selection import mutual_info_regression  # type: ignore
except Exception:
    DecisionTreeRegressor = None
    r2_score = None
    mutual_info_regression = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

# Full 64-entry feature list matching training.
_FULL_FEATURE_NAMES: List[str] = (
    TF_FEATURE_NAMES
    + FRICTION_FEATURE_NAMES
    + TOD_FEATURE_NAMES
    + [REGIME_PERSISTENCE_FEATURE]
    + IVR_REVERSAL_FEATURE_NAMES[:4]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIVOT_HORIZONS = [5, 10, 20, 35, 70]
EPS = 1e-8


# =============================================================================
# Small utilities
# =============================================================================

def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(x: Any) -> float:
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return float("nan")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _topk_pairs(values: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    idx = np.argsort(values)[::-1][:k]
    return [(int(i), float(values[i])) for i in idx]


def _tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _flatten_outputs(out: "CondorNetOutputLike") -> Dict[str, np.ndarray]:
    """
    Convert CondorNetOutput to a dict of 1D arrays (shape [B]).
    We audit a few scalar heads for stability: entry_signal, pop, ev, max_loss, position_size, spot_pred.
    """
    d: Dict[str, np.ndarray] = {}
    d["entry_signal"] = _tensor_to_np(out.entry_signal).reshape(-1)
    d["pop"] = _tensor_to_np(out.pop).reshape(-1)
    d["ev"] = _tensor_to_np(out.ev).reshape(-1)
    d["max_loss"] = _tensor_to_np(out.max_loss).reshape(-1)
    d["position_size"] = _tensor_to_np(out.position_size).reshape(-1)
    d["spot_pred"] = _tensor_to_np(out.spot_pred).reshape(-1)
    # Strategy logits (10): keep as mean confidence for top class.
    logits = _tensor_to_np(out.strategy_logits)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / (probs.sum(axis=1, keepdims=True) + EPS)
    d["strategy_top_prob"] = probs.max(axis=1)
    d["strategy_top_idx"] = probs.argmax(axis=1)
    return d


class CondorNetOutputLike:
    # structural type; actual class is defined in condor_brain_net_v43.py
    strategy_logits: torch.Tensor
    entry_signal: torch.Tensor
    pop: torch.Tensor
    ev: torch.Tensor
    max_loss: torch.Tensor
    position_size: torch.Tensor
    spot_pred: torch.Tensor


# =============================================================================
# Model loading
# =============================================================================

def load_condornet_v43(ckpt_path: str) -> Tuple[CondorNetV43, Dict[str, Any]]:
    print(f"[AUDIT] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    config = ckpt.get("config", {}) or {}
    if not config:
        # Reasonable defaults if older ckpt
        config = {
            "d_tf_in": len(TF_FEATURE_NAMES),
            "d_joint": 256,
            "d_chain": CHAIN_GRID_CONFIG.get("d_chain", 128),
            "d_pivot": 16,
            "d_fused": 256,
            "n_pivot_features": N_PIVOT_FEATURES,
            "n_strategy_types": N_STRATEGY_TYPES,
            "chain_in_features": CHAIN_GRID_CONFIG["in_features"],
            "chain_d_model": CHAIN_GRID_CONFIG["d_model"],
            "chain_n_heads": CHAIN_GRID_CONFIG["n_heads"],
            "chain_n_layers": CHAIN_GRID_CONFIG["n_layers"],
            "chain_d_ff": CHAIN_GRID_CONFIG["d_ff"],
        }
        print("  [AUDIT] No 'config' found — using schema defaults")

    # build_condornet_v43 expects kwargs; filter out non-arch keys
    skip = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "model_state_dict",
        "normalization",
        "epoch",
        "train_loss",
        "val_loss",
        "best_val_loss",
        "version",
        "schema_version",
    }
    build_cfg = {k: v for k, v in config.items() if k not in skip}

    model = build_condornet_v43(**build_cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [AUDIT] WARNING: {len(missing)} missing keys (strict=False)")
    if unexpected:
        print(f"  [AUDIT] WARNING: {len(unexpected)} unexpected keys (strict=False)")

    model.to(DEVICE).eval()

    meta = {
        "loaded_at": _now_iso(),
        "checkpoint_path": ckpt_path,
        "epoch": ckpt.get("epoch", "?"),
        "train_loss": ckpt.get("train_loss", float("nan")),
        "val_loss": ckpt.get("val_loss", float("nan")),
        "best_val_loss": ckpt.get("best_val_loss", float("nan")),
        "version": ckpt.get("version", "?"),
        "schema_version": ckpt.get("schema_version", "?"),
        "config": config,
    }
    return model, meta


# =============================================================================
# Weight-based interpretability
# =============================================================================

def _build_pair_lookup(n_features: int, feature_names: List[str]) -> Dict[int, Tuple[str, str]]:
    iu = np.triu_indices(n_features, k=1)
    lookup: Dict[int, Tuple[str, str]] = {}
    for pair_idx in range(len(iu[0])):
        i, j = int(iu[0][pair_idx]), int(iu[1][pair_idx])
        name_i = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        name_j = feature_names[j] if j < len(feature_names) else f"feat_{j}"
        lookup[pair_idx] = (name_i, name_j)
    return lookup


def extract_learned_logic_v43(model: CondorNetV43, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    if feature_names is None:
        feature_names = _FULL_FEATURE_NAMES

    logic: Dict[str, Any] = {
        "predicates": {},
        "super_set": {},
        "strategy_head": {},
        "risk_head": {},
        "pivot_head": {},
        "fuzzy_gates": {},
        "a_matrix": {},
        "b_matrix": {},
        "strategy_templates": {},
    }

    backbone = getattr(model, "condor_core", getattr(model, "backbone", model))

    # 1) Predicate gates (v42 backbone)
    preds: Dict[str, float] = {}
    if hasattr(backbone, "pred_gates"):
        pg = backbone.pred_gates
        for attr in [
            "iv_rank_thresh",
            "spread_frac_thresh",
            "rsi_thresh",
            "gap_frac_thresh",
            "gamma_thresh",
            "iv_regime_frac_thresh",
            "put_flow_thresh",
            "spread_stress_mult_thresh",
        ]:
            if hasattr(pg, attr):
                preds[attr.replace("_thresh", "_threshold")] = _safe_float(getattr(pg, attr))
        if hasattr(pg, "steepness"):
            preds["steepness"] = _safe_float(pg.steepness)
    logic["predicates"] = preds

    # 2) Relational logic (super sets → sets → top comparisons)
    if hasattr(backbone, "super_sets") and len(backbone.super_sets) > 0:
        pair_lookup: Dict[int, Tuple[str, str]] = {}
        try:
            ss0 = backbone.super_sets[0]
            rl0 = getattr(ss0.sets[0], "relational_logic", None) if hasattr(ss0, "sets") and ss0.sets else None
            if rl0 is not None:
                n_inputs = int(getattr(rl0, "n_inputs", len(feature_names)))
                pair_lookup = _build_pair_lookup(n_inputs, feature_names)
        except Exception:
            pair_lookup = {}

        ss_list: List[Dict[str, Any]] = []
        for ss_idx, ss in enumerate(backbone.super_sets):
            ss_info: Dict[str, Any] = {"index": ss_idx, "n_sets": len(ss.sets) if hasattr(ss, "sets") else 0, "sets": []}
            if hasattr(ss, "sets"):
                for i, pset in enumerate(ss.sets):
                    set_info: Dict[str, Any] = {"index": i}
                    rl = getattr(pset, "relational_logic", None)
                    if rl is None or not hasattr(rl, "projection"):
                        ss_info["sets"].append(set_info)
                        continue

                    if hasattr(rl, "steepness"):
                        set_info["steepness"] = _safe_float(rl.steepness)

                    w = rl.projection.weight.detach().cpu().numpy()
                    n_pairs = int(getattr(rl, "n_pairs", w.shape[1] // 3))
                    if w.shape[1] == n_pairs * 3:
                        w_r = w.reshape(w.shape[0], n_pairs, 3)
                        op_w = np.abs(w_r).sum(axis=0)  # [n_pairs, 3]
                        total = float(op_w.sum()) + EPS
                        set_info["operator_weights_pct"] = {
                            "<": round(float(op_w[:, 0].sum()) / total * 100, 1),
                            ">": round(float(op_w[:, 1].sum()) / total * 100, 1),
                            "=": round(float(op_w[:, 2].sum()) / total * 100, 1),
                        }
                        importance = op_w.sum(axis=1)
                        top_idx = np.argsort(importance)[-10:][::-1]
                        comps = []
                        for p_idx in top_idx:
                            ops_row = op_w[p_idx]
                            dom_op = ["<", ">", "="][int(np.argmax(ops_row))]
                            feat_a, feat_b = pair_lookup.get(int(p_idx), (f"feat_{p_idx}a", f"feat_{p_idx}b"))
                            comps.append(
                                {
                                    "pair_idx": int(p_idx),
                                    "comparison": f"{feat_a} {dom_op} {feat_b}",
                                    "dominant_op": dom_op,
                                    "weights": {"<": float(ops_row[0]), ">": float(ops_row[1]), "=": float(ops_row[2])},
                                }
                            )
                        set_info["top_comparisons"] = comps
                    else:
                        set_info["projection_weight_norm"] = float(np.linalg.norm(w))
                    ss_info["sets"].append(set_info)

            ss_list.append(ss_info)
        logic["super_set"] = {"n_super_sets": len(ss_list), "super_sets": ss_list}

    # 3) Strategy head norms
    if hasattr(model, "strategy_head"):
        sh = model.strategy_head
        out: Dict[str, Any] = {}

        head_seq = getattr(sh, "strategy_head", None)
        if head_seq is not None:
            last_linear = next((m for m in reversed(list(head_seq.modules())) if hasattr(m, "weight") and m.weight.dim() == 2), None)
            if last_linear is not None:
                w = last_linear.weight.detach().cpu().numpy()
                norms = np.linalg.norm(w, axis=1)
                class_norms = {STRATEGY_TYPES[i]: float(norms[i]) for i in range(min(len(norms), N_STRATEGY_TYPES))}
                ranked = sorted(class_norms.items(), key=lambda x: x[1], reverse=True)
                out["output_class_norms"] = class_norms
                out["ranked_classes"] = [cls for cls, _ in ranked]
                out["dominant_class"] = ranked[0][0] if ranked else "unknown"

        leg_seq = getattr(sh, "leg_head", None)
        if leg_seq is not None:
            last_l = next((m for m in reversed(list(leg_seq.modules())) if hasattr(m, "weight") and m.weight.dim() == 2), None)
            if last_l is not None:
                out["leg_head_weight_norm"] = float(last_l.weight.norm().item())
        logic["strategy_head"] = out

    # 4) Risk head norms
    if hasattr(model, "risk_head"):
        rh = model.risk_head
        out: Dict[str, Any] = {}
        shared = getattr(rh, "shared", None)
        if shared is not None:
            first_l = next((m for m in shared.modules() if hasattr(m, "weight") and m.weight.dim() == 2), None)
            if first_l is not None:
                out["shared_input_frobenius"] = float(np.linalg.norm(first_l.weight.detach().cpu().numpy(), "fro"))
        for head_name in ["pop_head", "ev_head", "max_loss_head", "var_head", "cvar_offset_head"]:
            head = getattr(rh, head_name, None)
            if head is not None and hasattr(head, "weight"):
                out[f"{head_name}_frobenius"] = float(head.weight.detach().cpu().float().norm().item())
        logic["risk_head"] = out

    # 5) Pivot head norms (horizons)
    if hasattr(model, "pivot_pred_head"):
        ph = model.pivot_pred_head
        out: Dict[str, Any] = {}

        for attr_name, key in [("pivot_high_head", "pivot_high_horizon_norms"), ("pivot_low_head", "pivot_low_horizon_norms")]:
            head_seq = getattr(ph, attr_name, None)
            if head_seq is None:
                continue
            last_l = next((m for m in reversed(list(head_seq.modules())) if hasattr(m, "weight") and m.weight.dim() == 2), None)
            if last_l is None:
                continue
            w = last_l.weight.detach().cpu().numpy()  # [5, d]
            norms = np.linalg.norm(w, axis=1)
            out[key] = {f"h{h}": float(norms[i]) for i, h in enumerate(PIVOT_HORIZONS[: len(norms)])}

        strength_head = getattr(ph, "strength_head", None)
        if strength_head is not None and hasattr(strength_head, "weight"):
            out["pivot_strength_head_frobenius"] = float(strength_head.weight.detach().cpu().float().norm().item())
        logic["pivot_head"] = out

    # 6) TF projector contribution norms
    if hasattr(model, "tf_projector"):
        tfp = model.tf_projector
        contrib: Dict[str, float] = {}
        for tf_name in ["m1", "m5", "m15", "h1"]:
            proj = getattr(tfp, f"proj_{tf_name}", None)
            if proj is None:
                continue
            first_l = next((m for m in proj.modules() if hasattr(m, "weight") and m.weight.dim() == 2), None)
            if first_l is None:
                continue
            contrib[tf_name] = float(np.linalg.norm(first_l.weight.detach().cpu().numpy(), "fro"))
        ranked = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
        logic["fuzzy_gates"]["tf_projector_frobenius"] = contrib
        logic["fuzzy_gates"]["tf_ranked"] = [k for k, _ in ranked]

    # 7) A matrix stability
    try:
        with torch.no_grad():
            A = model.get_A_matrix().detach().cpu().float().numpy()
        eigvals = np.linalg.eigvals(A)
        mags = np.abs(eigvals)
        logic["a_matrix"] = {
            "shape": list(A.shape),
            "spectral_radius": float(np.max(mags)),
            "frobenius_norm": float(np.linalg.norm(A, "fro")),
            "top_8_eigenvalue_magnitudes": [float(x) for x in sorted(mags, reverse=True)[:8]],
        }
    except Exception as e:
        logic["a_matrix"] = {"error": str(e)}

    # 8) B matrix stability
    try:
        b_theta = getattr(backbone, "B_theta", None)
        B = None
        if b_theta is not None:
            with torch.no_grad():
                if hasattr(b_theta, "full_matrix"):
                    B = b_theta.full_matrix().detach().cpu().float().numpy()
                elif hasattr(b_theta, "weight"):
                    B = b_theta.weight.detach().cpu().float().numpy()
        if B is not None:
            u, s, vt = np.linalg.svd(B, full_matrices=False)
            logic["b_matrix"] = {
                "shape": list(B.shape),
                "frobenius_norm": float(np.linalg.norm(B, "fro")),
                "spectral_norm": float(s.max()) if len(s) else float("nan"),
                "condition_number": float(s.max() / (s.min() + EPS)) if len(s) else float("nan"),
                "column_norms_max": float(np.linalg.norm(B, axis=0).max()),
                "column_norms_mean": float(np.linalg.norm(B, axis=0).mean()),
                "top_8_singular_values": [float(x) for x in s[:8]],
            }
    except Exception as e:
        logic["b_matrix"] = {"error": str(e)}

    # 9) Strategy templates introspection
    try:
        if isinstance(STRATEGY_TEMPLATES, dict) and STRATEGY_TEMPLATES:
            logic["strategy_templates"] = {
                "n_templates": int(len(STRATEGY_TEMPLATES)),
                "names": sorted(list(STRATEGY_TEMPLATES.keys()))[:200],
            }
    except Exception:
        pass

    return logic


def generate_trading_rules_v43(logic: Dict[str, Any], verbose: bool = False) -> List[str]:
    rules: List[str] = []

    p = logic.get("predicates", {})
    if p:
        rules += ["=" * 70, "CANONICAL PREDICATE THRESHOLDS (v42 backbone)", "=" * 70]
        label_map = {
            "iv_rank_threshold": "Entry if IVR >",
            "spread_frac_threshold": "Block if Spread/Price >",
            "rsi_threshold": "Reversal signal if RSI <",
            "gap_frac_threshold": "Guard if 1m gap >",
            "gamma_threshold": "Hedge if |Gamma| >",
            "iv_regime_frac_threshold": "IV regime alert if IV_mid > IV_high x",
            "put_flow_threshold": "Heavy put flow if Put_Vol/Total >",
            "spread_stress_mult_threshold": "Microstructure stress if spread >",
        }
        for k, label in label_map.items():
            if k in p:
                rules.append(f"  {label} {p[k]:.6f}")
        if "steepness" in p:
            rules.append(f"  Sigmoid steepness: {p['steepness']:.3f}")

    sh = logic.get("strategy_head", {})
    if sh:
        rules += ["", "=" * 70, "STRATEGY HEAD — Class Norm Ranking", "=" * 70]
        ranked = sh.get("ranked_classes", [])
        norms = sh.get("output_class_norms", {})
        for i, cls in enumerate(ranked[: (len(ranked) if verbose else 10)]):
            rules.append(f"  #{i+1:2d} {cls:<28} norm={norms.get(cls, 0):.6f}")
        rules.append(f"  Dominant class: {sh.get('dominant_class', '?')}")

    fg = logic.get("fuzzy_gates", {})
    if fg and fg.get("tf_projector_frobenius"):
        rules += ["", "=" * 70, "MULTI-TF PROJECTOR — Contribution Ranking", "=" * 70]
        tf_ranked = fg.get("tf_ranked", [])
        tf_frob = fg.get("tf_projector_frobenius", {})
        for tf in tf_ranked:
            rules.append(f"  {tf.upper():<4} frob={tf_frob[tf]:.6f}")

    ss = logic.get("super_set", {})
    if ss.get("super_sets"):
        rules += ["", "=" * 70, "LEARNED RELATIONAL LOGIC (Predicate Sets)", "=" * 70]
        ss_show = ss["super_sets"] if verbose else ss["super_sets"][:4]
        for ss_info in ss_show:
            rules.append(f"\nSUPER_SET {ss_info['index']}  (n_sets={ss_info.get('n_sets', 0)})")
            sets = ss_info.get("sets", [])
            sets_show = sets if verbose else sets[:3]
            for set_info in sets_show:
                ops = set_info.get("operator_weights_pct", {})
                if ops:
                    rules.append(f"  SET {set_info['index']:>3}: (<:{ops.get('<',0)}%  >:{ops.get('>',0)}%  =:{ops.get('=',0)}%)")
                    for comp in set_info.get("top_comparisons", [])[: (len(set_info.get("top_comparisons", [])) if verbose else 3)]:
                        rules.append(f"      -> {comp['comparison']}")
                else:
                    rules.append(f"  SET {set_info['index']:>3}: (no relational weights)")

    rules += ["", "=" * 70, "STATE MATRIX STABILITY", "=" * 70]
    am = logic.get("a_matrix", {})
    if "spectral_radius" in am:
        rho = am["spectral_radius"]
        stable = "STABLE" if rho <= 1.0 else "UNSTABLE (rho > 1)"
        rules.append(f"  A_matrix rho={rho:.6f}  [{stable}]  frob={am.get('frobenius_norm', float('nan')):.6f}")
    bm = logic.get("b_matrix", {})
    if "frobenius_norm" in bm:
        rules.append(
            f"  B_matrix frob={bm['frobenius_norm']:.6f}  "
            f"spec={bm.get('spectral_norm', float('nan')):.6f}  "
            f"cond≈{bm.get('condition_number', float('nan')):.3f}"
        )

    return rules


def export_matrices_csv(model: CondorNetV43, out_dir: Path) -> Dict[str, Any]:
    _mkdir(out_dir)
    results: Dict[str, Any] = {}

    # A
    try:
        A = model.get_A_matrix().detach().cpu().float().numpy()
        a_path = out_dir / "audit_A_matrix.csv"
        np.savetxt(str(a_path), A, delimiter=",", fmt="%.8f")
        results["a_path"] = str(a_path)
        results["a_shape"] = list(A.shape)
    except Exception as e:
        results["a_error"] = str(e)

    # B (from backbone)
    try:
        backbone = getattr(model, "condor_core", getattr(model, "backbone", model))
        b_theta = getattr(backbone, "B_theta", None)
        B = None
        if b_theta is not None:
            if hasattr(b_theta, "full_matrix"):
                B = b_theta.full_matrix().detach().cpu().float().numpy()
            elif hasattr(b_theta, "weight"):
                B = b_theta.weight.detach().cpu().float().numpy()
        if B is not None:
            b_path = out_dir / "audit_B_matrix.csv"
            np.savetxt(str(b_path), B, delimiter=",", fmt="%.8f")
            results["b_path"] = str(b_path)
            results["b_shape"] = list(B.shape)
    except Exception as e:
        results["b_error"] = str(e)

    return results


# =============================================================================
# Data-based analytics
# =============================================================================



def _infer_model_tf_in_features(model: nn.Module) -> int:
    """Infer d_tf_in expected by forward_compat() and MultiTFProjector."""
    for attr in ["d_tf_in", "D_TF_IN", "d_input", "D_INPUT"]:
        if hasattr(model, attr):
            try:
                v = int(getattr(model, attr))
                if v > 0:
                    return v
            except Exception:
                pass
    tfp = getattr(model, "tf_projector", None)
    if tfp is not None and hasattr(tfp, "proj_m1"):
        proj = getattr(tfp, "proj_m1")
        if proj is not None:
            first_l = next((m for m in proj.modules() if isinstance(m, nn.Linear)), None)
            if first_l is not None:
                return int(first_l.in_features)
    return len(TF_FEATURE_NAMES)

@dataclass
class DataAuditConfig:
    samples: int = 2000
    seq_len: int = 16
    seed: int = 42
    batch_size: int = 256
    # which scalar output to analyze as "primary"
    primary_target: str = "entry_signal"  # entry_signal|pop|ev|max_loss|position_size|spot_pred
    # expensive options
    fisher_batches: int = 6
    hessian_batches: int = 2
    hessian_iters: int = 12


def _load_csv_sequences(csv_path: str, feature_names: List[str], cfg: DataAuditConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataframe and return:
      X_seq: [N, T, F]
      y_dummy: [N] (placeholder; we do output-based metrics)
    """
    if pd is None:
        raise RuntimeError("pandas is required for --data mode (pip install pandas).")

    df = pd.read_csv(csv_path)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"--data is missing {len(missing)} required feature columns, e.g. {missing[:10]}")

    X = df[feature_names].to_numpy(dtype=np.float32)
    n = X.shape[0]
    T = cfg.seq_len
    if n < T + 1:
        raise ValueError(f"Not enough rows for seq_len={T}: rows={n}")

    rng = np.random.default_rng(cfg.seed)
    # sample starting indices uniformly
    max_start = n - T
    starts = rng.integers(low=0, high=max_start, size=min(cfg.samples, max_start), endpoint=False)

    X_seq = np.stack([X[s : s + T] for s in starts], axis=0)  # [N, T, F]
    y_dummy = np.zeros((X_seq.shape[0],), dtype=np.float32)
    return X_seq, y_dummy


@torch.no_grad()
def _model_outputs(model: CondorNetV43, X_seq: np.ndarray, cfg: DataAuditConfig) -> Dict[str, np.ndarray]:
    """
    Run forward_compat in batches. Returns dict of scalar arrays with shape [N].
    """
    model.eval()
    N, T, F = X_seq.shape
    outs: Dict[str, List[np.ndarray]] = {}
    bs = cfg.batch_size
    tf_in = _infer_model_tf_in_features(model)
    if F < tf_in:
        raise ValueError(f"X_seq has {F} features but model expects {tf_in}.")

    for i in range(0, N, bs):
        xb = torch.from_numpy(X_seq[i : i + bs]).to(DEVICE)
        if xb.shape[-1] != tf_in:
            xb = xb[..., :tf_in]
        out = model.forward_compat(xb, return_diagnostics=False)  # type: ignore
        flat = _flatten_outputs(out)  # numpy
        for k, v in flat.items():
            outs.setdefault(k, []).append(np.asarray(v))
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def permutation_importance(
    model: CondorNetV43,
    X_seq: np.ndarray,
    baseline: Dict[str, np.ndarray],
    feature_names: List[str],
    cfg: DataAuditConfig,
) -> Dict[str, Any]:
    """
    Permute each feature across samples (preserving temporal structure per sample) and measure
    mean absolute deviation on primary_target vs baseline.
    """
    target = cfg.primary_target
    y0 = baseline[target].astype(np.float64)
    N, T, F = X_seq.shape
    rng = np.random.default_rng(cfg.seed)

    scores = np.zeros((F,), dtype=np.float64)
    for j in range(F):
        Xp = X_seq.copy()
        perm = rng.permutation(N)
        # permute the entire feature-trajectory across samples
        Xp[:, :, j] = X_seq[perm, :, j]
        y1 = _model_outputs(model, Xp, cfg)[target].astype(np.float64)
        scores[j] = float(np.mean(np.abs(y1 - y0)))
    order = np.argsort(scores)[::-1]
    top = [{"feature": feature_names[int(i)], "mad": float(scores[int(i)])} for i in order[:50]]
    return {"metric": "mean_abs_dev", "target": target, "top_50": top, "all": {feature_names[i]: float(scores[i]) for i in range(F)}}


def gradient_saliency(
    model: CondorNetV43,
    X_seq: np.ndarray,
    feature_names: List[str],
    cfg: DataAuditConfig,
) -> Dict[str, Any]:
    """
    Compute mean |∂target/∂x_{t,f}| aggregated over t and samples.
    """
    target = cfg.primary_target
    model.eval()

    # Use a single medium batch (or two) to keep it light
    N = min(X_seq.shape[0], cfg.batch_size)
    xb = torch.from_numpy(X_seq[:N]).to(DEVICE)
    xb.requires_grad_(True)

    out = model.forward_compat(xb, return_diagnostics=False)  # type: ignore
    y = getattr(out, target)
    if y.dim() == 2 and y.shape[1] == 1:
        y = y[:, 0]
    # scalar objective = mean
    obj = y.mean()
    obj.backward()

    g = xb.grad.detach().abs().mean(dim=0)  # [T, F]
    g_f = g.mean(dim=0).detach().cpu().numpy()  # [F]
    order = np.argsort(g_f)[::-1]
    top = [{"feature": feature_names[int(i)], "saliency": float(g_f[int(i)])} for i in order[:50]]
    return {"target": target, "aggregation": "mean_abs_grad_over_time_and_batch", "top_50": top, "all": {feature_names[i]: float(g_f[i]) for i in range(len(feature_names))}}


def mutual_information_analysis(
    X_seq: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg: DataAuditConfig,
) -> Dict[str, Any]:
    """
    MI between per-sample time-averaged feature and target (sklearn).
    """
    if mutual_info_regression is None:
        return {"skipped": True, "reason": "sklearn not available"}
    X_bar = X_seq.mean(axis=1)  # [N, F]
    y = y.astype(np.float64)
    mi = mutual_info_regression(X_bar, y, random_state=cfg.seed)
    order = np.argsort(mi)[::-1]
    top = [{"feature": feature_names[int(i)], "mi": float(mi[int(i)])} for i in order[:50]]
    return {"target": cfg.primary_target, "top_50": top, "all": {feature_names[i]: float(mi[i]) for i in range(len(feature_names))}}


def surrogate_tree_analysis(
    X_seq: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg: DataAuditConfig,
) -> Dict[str, Any]:
    if DecisionTreeRegressor is None or r2_score is None:
        return {"skipped": True, "reason": "sklearn not available"}

    X_bar = X_seq.mean(axis=1)  # [N, F]
    y = y.astype(np.float64)

    # simple train/test split
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    tr, te = idx[:split], idx[split:]

    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=40, random_state=cfg.seed)
    dt.fit(X_bar[tr], y[tr])
    pred = dt.predict(X_bar[te])
    r2 = float(r2_score(y[te], pred))

    # feature importances
    imp = dt.feature_importances_
    order = np.argsort(imp)[::-1]
    top = [{"feature": feature_names[int(i)], "importance": float(imp[int(i)])} for i in order[:25]]

    return {
        "target": cfg.primary_target,
        "r2_test": r2,
        "top_25_features": top,
        "notes": "Tree trained on time-averaged features (mean over seq_len).",
    }


def fisher_diagonal_proxy(
    model: CondorNetV43,
    X_seq: np.ndarray,
    cfg: DataAuditConfig,
) -> Dict[str, Any]:
    """
    Lightweight Fisher-diagonal proxy:
        F_ii ≈ E[ (∂L/∂θ_i)^2 ]
    using L = mean(entry_signal) (or primary_target) over a few batches.
    Reports per-module aggregated norms (NOT full per-parameter dump).
    """
    target = cfg.primary_target
    model.train(False)

    # choose a small subset of parameters to report (by module name prefix)
    buckets: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    def bucket_name(param_name: str) -> str:
        for pref in ["condor_core", "tf_projector", "chain_encoder", "strategy_head", "risk_head", "pivot_pred_head", "position_head"]:
            if param_name.startswith(pref):
                return pref
        return "other"

    N = X_seq.shape[0]
    bs = cfg.batch_size
    n_batches = min(cfg.fisher_batches, max(1, math.ceil(N / bs)))

    for b in range(n_batches):
        xb = torch.from_numpy(X_seq[b * bs : (b + 1) * bs]).to(DEVICE)
        xb.requires_grad_(False)

        model.zero_grad(set_to_none=True)
        out = model.forward_compat(xb, return_diagnostics=False)  # type: ignore
        y = getattr(out, target)
        if y.dim() == 2 and y.shape[1] == 1:
            y = y[:, 0]
        loss = (-y.mean())  # maximize target => negative mean as loss
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g2 = float((p.grad.detach() ** 2).mean().item())
            bn = bucket_name(name)
            buckets[bn] = buckets.get(bn, 0.0) + g2
            counts[bn] = counts.get(bn, 0) + 1

    agg = {k: float(v / max(1, counts[k])) for k, v in buckets.items()}
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return {"target": target, "metric": "mean_grad2", "ranked_modules": ranked, "per_module": agg, "batches": n_batches}


def _hvp(loss: torch.Tensor, params: List[torch.Tensor], v: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Hessian-vector product via autograd:
        H v = ∂/∂θ ( <∇L, v> )
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    dot = 0.0
    for g, vi in zip(grads, v):
        if g is None:
            continue
        dot = dot + (g * vi).sum()
    hv = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
    out: List[torch.Tensor] = []
    for h, p in zip(hv, params):
        if h is None:
            out.append(torch.zeros_like(p))
        else:
            out.append(h.detach())
    return out


def hessian_top_eigen_proxy(
    model: CondorNetV43,
    X_seq: np.ndarray,
    cfg: DataAuditConfig,
) -> Dict[str, Any]:
    """
    Approximate the top Hessian eigenvalue for a scalar loss using power iteration.
    This is intentionally lightweight and intended for trend monitoring, not exact curvature science.
    """
    target = cfg.primary_target
    model.train(False)

    # Use a subset of params (last-layer weights only) to keep compute bounded.
    params: List[torch.Tensor] = []
    param_names: List[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ["strategy_head", "risk_head", "pivot_pred_head", "position_head"]):
            if p.dim() >= 2:  # weights
                params.append(p)
                param_names.append(name)
    if not params:
        return {"skipped": True, "reason": "no eligible params found"}

    # Build a small batch
    N = min(X_seq.shape[0], cfg.batch_size)
    xb = torch.from_numpy(X_seq[:N]).to(DEVICE)

    out = model.forward_compat(xb, return_diagnostics=False)  # type: ignore
    y = getattr(out, target)
    if y.dim() == 2 and y.shape[1] == 1:
        y = y[:, 0]
    loss = (-y.mean())

    # init random v
    v: List[torch.Tensor] = [torch.randn_like(p) for p in params]
    # normalize
    def normalize(vlist: List[torch.Tensor]) -> List[torch.Tensor]:
        nrm = math.sqrt(sum(float((vi**2).sum().item()) for vi in vlist)) + 1e-12
        return [vi / nrm for vi in vlist]

    v = normalize(v)

    lam = float("nan")
    for _ in range(cfg.hessian_iters):
        hv = _hvp(loss, params, v)
        # Rayleigh quotient
        num = sum(float((hvi * vi).sum().item()) for hvi, vi in zip(hv, v))
        den = sum(float((vi * vi).sum().item()) for vi in v) + 1e-12
        lam = num / den
        v = normalize(hv)

    return {
        "target": target,
        "approx_top_eigenvalue": float(lam),
        "param_tensors": len(params),
        "example_params": param_names[:10],
        "iters": cfg.hessian_iters,
        "notes": "Power iteration on Hessian-vector product (subset of head weights).",
    }


def plot_top_bars(items: List[Dict[str, Any]], value_key: str, title: str, out_path: Path, top_n: int = 25) -> None:
    if plt is None:
        return
    items = items[:top_n]
    labels = [it["feature"] for it in items][::-1]
    vals = [it[value_key] for it in items][::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(labels, vals)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def write_markdown_report(out_dir: Path, meta: Dict[str, Any], logic: Dict[str, Any], data_audit: Optional[Dict[str, Any]]) -> str:
    md: List[str] = []
    md.append(f"# CondorNet v4.3 — Interpretability Audit\n")
    md.append(f"- Generated: `{_now_iso()}`")
    md.append(f"- Schema: `{SCHEMA_VERSION}`")
    md.append(f"- Checkpoint epoch: `{meta.get('epoch')}`  val_loss: `{meta.get('val_loss')}`")
    md.append("")

    md.append("## Weight-based summary\n")
    md.append("### Predicate thresholds\n")
    for k, v in logic.get("predicates", {}).items():
        md.append(f"- `{k}`: `{v}`")
    md.append("")

    md.append("### Strategy head ranking (by output weight norm)\n")
    ranked = logic.get("strategy_head", {}).get("ranked_classes", [])
    norms = logic.get("strategy_head", {}).get("output_class_norms", {})
    for i, cls in enumerate(ranked[:10]):
        md.append(f"- {i+1}. `{cls}` — norm `{norms.get(cls, 0)}`")
    md.append("")

    md.append("### Multi-TF projector contribution\n")
    tf_ranked = logic.get("fuzzy_gates", {}).get("tf_ranked", [])
    tf_frob = logic.get("fuzzy_gates", {}).get("tf_projector_frobenius", {})
    for tf in tf_ranked:
        md.append(f"- `{tf}`: frob `{tf_frob.get(tf)}`")
    md.append("")

    md.append("### State matrices\n")
    md.append(f"- A: `{logic.get('a_matrix', {})}`")
    md.append(f"- B: `{logic.get('b_matrix', {})}`")
    md.append("")

    if logic.get("strategy_templates", {}).get("n_templates") is not None:
        md.append("### Strategy template catalog\n")
        md.append(f"- n_templates: `{logic['strategy_templates']['n_templates']}`")
        md.append("")

    if data_audit:
        md.append("## Data-based analytics\n")
        md.append(f"- Target: `{data_audit.get('primary_target')}`\n")

        if "permutation_importance" in data_audit:
            md.append("### Permutation importance (top 25)\n")
            for it in data_audit["permutation_importance"]["top_50"][:25]:
                md.append(f"- `{it['feature']}`: MAD `{it['mad']:.6g}`")
            md.append("")

        if "gradient_saliency" in data_audit:
            md.append("### Gradient saliency (top 25)\n")
            for it in data_audit["gradient_saliency"]["top_50"][:25]:
                md.append(f"- `{it['feature']}`: |grad| `{it['saliency']:.6g}`")
            md.append("")

        if "mutual_information" in data_audit and not data_audit["mutual_information"].get("skipped"):
            md.append("### Mutual information (top 25)\n")
            for it in data_audit["mutual_information"]["top_50"][:25]:
                md.append(f"- `{it['feature']}`: MI `{it['mi']:.6g}`")
            md.append("")

        if "surrogate_tree" in data_audit and not data_audit["surrogate_tree"].get("skipped"):
            md.append("### Surrogate decision tree\n")
            md.append(f"- Test R²: `{data_audit['surrogate_tree']['r2_test']:.4f}`")
            md.append("- Top features:")
            for it in data_audit["surrogate_tree"]["top_25_features"][:10]:
                md.append(f"  - `{it['feature']}`: importance `{it['importance']:.6g}`")
            md.append("")

        if "fisher_proxy" in data_audit:
            md.append("### Fisher-diagonal proxy (module ranking)\n")
            for name, val in data_audit["fisher_proxy"]["ranked_modules"]:
                md.append(f"- `{name}`: mean_grad² `{val:.6g}`")
            md.append("")

        if "hessian_proxy" in data_audit and not data_audit["hessian_proxy"].get("skipped"):
            md.append("### Hessian top-eigen proxy\n")
            md.append(f"- Approx top eigenvalue: `{data_audit['hessian_proxy']['approx_top_eigenvalue']:.6g}`\n")

    md_path = out_dir / "audit_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return str(md_path)


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to CondorNetV43 checkpoint .pth")
    p.add_argument("--output-json", required=True, help="Path to write audit JSON")
    p.add_argument("--output-dir", default="", help="Directory for plots/markdown (optional)")
    p.add_argument("--export-matrices", action="store_true", help="Export A/B matrices to CSV in output-dir")
    p.add_argument("--verbose", action="store_true", help="Verbose printing")

    # data-based
    p.add_argument("--data", default="", help="CSV path for data-based audit (optional)")
    p.add_argument("--samples", type=int, default=2000, help="Number of sampled sequences")
    p.add_argument("--seq-len", type=int, default=16, help="Sequence length T")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for model inference")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--primary-target", default="entry_signal", help="Primary scalar output to analyze")
    p.add_argument("--no-hessian", action="store_true", help="Skip Hessian proxy")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model, meta = load_condornet_v43(args.model)

    logic = extract_learned_logic_v43(model, feature_names=_FULL_FEATURE_NAMES)
    rules = generate_trading_rules_v43(logic, verbose=args.verbose)

    out_json_path = Path(args.output_json)
    _mkdir(out_json_path.parent)

    out_dir: Optional[Path] = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        _mkdir(out_dir)

    matrices_info: Dict[str, Any] = {}
    if args.export_matrices:
        if out_dir is None:
            out_dir = out_json_path.parent
        matrices_info = export_matrices_csv(model, out_dir)

    data_audit: Optional[Dict[str, Any]] = None

    if args.data:
        cfg = DataAuditConfig(
            samples=args.samples,
            seq_len=args.seq_len,
            seed=args.seed,
            batch_size=args.batch_size,
            primary_target=args.primary_target,
        )

        data_feature_names = _FULL_FEATURE_NAMES
        tf_in = _infer_model_tf_in_features(model)
        if tf_in == len(TF_FEATURE_NAMES):
            data_feature_names = TF_FEATURE_NAMES
        elif tf_in == len(_FULL_FEATURE_NAMES):
            data_feature_names = _FULL_FEATURE_NAMES
        else:
            # Use prefix subset as a pragmatic fallback
            data_feature_names = _FULL_FEATURE_NAMES[:tf_in]
            print(f"[AUDIT] WARNING: model expects {tf_in} tf features; using first {tf_in} feature names for --data.")

        X_seq, _ = _load_csv_sequences(args.data, data_feature_names, cfg)
        baseline = _model_outputs(model, X_seq, cfg)

        data_audit = {"primary_target": cfg.primary_target, "n_sequences": int(X_seq.shape[0]), "seq_len": int(X_seq.shape[1])}
        data_audit["baseline_stats"] = {
            cfg.primary_target: {
                "mean": float(np.mean(baseline[cfg.primary_target])),
                "std": float(np.std(baseline[cfg.primary_target])),
                "min": float(np.min(baseline[cfg.primary_target])),
                "max": float(np.max(baseline[cfg.primary_target])),
            }
        }

        # Permutation importance (most expensive) — keep, but can be slow for 64 features.
        print("[AUDIT] Permutation importance...")
        data_audit["permutation_importance"] = permutation_importance(model, X_seq, baseline, _FULL_FEATURE_NAMES, cfg)

        # Gradient saliency
        print("[AUDIT] Gradient saliency...")
        data_audit["gradient_saliency"] = gradient_saliency(model, X_seq, _FULL_FEATURE_NAMES, cfg)

        # MI + Tree (if available)
        y = baseline[cfg.primary_target]
        print("[AUDIT] Mutual information (optional)...")
        data_audit["mutual_information"] = mutual_information_analysis(X_seq, y, _FULL_FEATURE_NAMES, cfg)

        print("[AUDIT] Surrogate tree (optional)...")
        data_audit["surrogate_tree"] = surrogate_tree_analysis(X_seq, y, _FULL_FEATURE_NAMES, cfg)

        # Fisher proxy
        print("[AUDIT] Fisher proxy...")
        data_audit["fisher_proxy"] = fisher_diagonal_proxy(model, X_seq, cfg)

        # Hessian proxy
        if not args.no_hessian:
            print("[AUDIT] Hessian proxy (optional)...")
            data_audit["hessian_proxy"] = hessian_top_eigen_proxy(model, X_seq, cfg)
        else:
            data_audit["hessian_proxy"] = {"skipped": True, "reason": "--no-hessian"}

        # plots
        if out_dir is not None:
            try:
                pi_top = data_audit["permutation_importance"]["top_50"]
                plot_top_bars(pi_top, "mad", f"Permutation importance ({cfg.primary_target})", out_dir / "perm_importance.png", top_n=25)
                gs_top = data_audit["gradient_saliency"]["top_50"]
                plot_top_bars(gs_top, "saliency", f"Gradient saliency ({cfg.primary_target})", out_dir / "grad_saliency.png", top_n=25)

                mi = data_audit["mutual_information"]
                if mi and not mi.get("skipped"):
                    plot_top_bars(mi["top_50"], "mi", f"Mutual information ({cfg.primary_target})", out_dir / "mutual_info.png", top_n=25)
            except Exception as e:
                print(f"[AUDIT] Plotting skipped due to error: {e}")

    payload: Dict[str, Any] = {
        "meta": meta,
        "logic": logic,
        "rules": rules,
        "matrices_export": matrices_info,
        "data_audit": data_audit,
    }

    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[AUDIT] Wrote JSON: {out_json_path}")

    if out_dir is not None:
        md_path = write_markdown_report(out_dir, meta, logic, data_audit)
        print(f"[AUDIT] Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()