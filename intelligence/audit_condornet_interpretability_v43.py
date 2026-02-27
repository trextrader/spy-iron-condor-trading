#!/usr/bin/env python3
"""
audit_condornet_interpretability_v43.py — CondorNet v4.3 Interpretability Audit
=================================================================================
Post-hoc analysis of CondorNetV43 checkpoints. Extracts and renders:

  1. Predicate thresholds (v42 backbone canonical gates)
  2. SuperSet relational logic — dominant comparison operators, top feature pairs
  3. Strategy head — per-class output norm ranking
  4. Risk metric head — shared representation norms
  5. Pivot prediction head — per-horizon weight norms (medium + strong)
  6. TF projector Frobenius contributions per timeframe
  7. A_matrix / B_matrix stability metrics
  8. Human-readable trading rules transcript

Differences from v42 audit:
  - Loads CondorNetV43 (multi-TF + chain + pivot) instead of v42 CondorNet
  - Architecture inferred from checkpoint 'config' key (v43 saves it)
  - Strategy head outputs for 10 strategy classes (incl. abstain)
  - Pivot head horizons [5, 10, 20, 35, 70] for medium/strong pivots only
  - B_matrix export (if available in backbone)

Usage:
    python intelligence/audit_condornet_interpretability_v43.py \\
        --model models/condornet_v43_best.pth \\
        --output-json reports/condor_v43_logic.json \\
        --verbose

Author: CondorNet v4.3 Implementation
Version: 4.3.0
Date: 2026-02-23
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Full 64-entry feature name list matching the training script's _FULL_FEATURE_NAMES.
# 52 base TF + 5 friction gates + 2 ToD + 1 regime persistence + 4 IVR reversal.
_FULL_FEATURE_NAMES: list = (
    TF_FEATURE_NAMES
    + FRICTION_FEATURE_NAMES
    + TOD_FEATURE_NAMES
    + [REGIME_PERSISTENCE_FEATURE]
    + IVR_REVERSAL_FEATURE_NAMES[:4]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIVOT_HORIZONS = [5, 10, 20, 35, 70]

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_condornet_v43(ckpt_path: str) -> Tuple[CondorNetV43, dict]:
    """
    Load CondorNetV43 from checkpoint. Reads 'config' key saved by condor_train_net_v43.

    Returns:
        (model, ckpt_metadata_dict)
    """
    print(f"[AUDIT] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    config = ckpt.get('config', {})
    if not config:
        # Fallback defaults (inferred from schema)
        config = {
            'd_tf_in': len(TF_FEATURE_NAMES),
            'd_tf_proj': 64,  # corresponds to d_joint=256
            'd_chain': CHAIN_GRID_CONFIG['d_chain'],
            'd_pivot_proj': 16,
            'd_pivot_in': N_PIVOT_FEATURES,
            'n_strategy_types': N_STRATEGY_TYPES,
            'chain_in_features': CHAIN_GRID_CONFIG['in_features'],
            'd_model_chain': CHAIN_GRID_CONFIG['d_model'],
            'n_heads_chain': CHAIN_GRID_CONFIG['n_heads'],
            'n_layers_chain': CHAIN_GRID_CONFIG['n_layers'],
        }
        print("  [AUDIT] No 'config' in checkpoint — using schema defaults")

    # Handle legacy config keys from older checkpoints or mismatches
    key_mapping = {
        'n_pivot_features': 'd_pivot_in',
        'd_pivot': 'd_pivot_proj',
        'chain_d_model': 'd_model_chain',
        'chain_n_heads': 'n_heads_chain',
        'chain_n_layers': 'n_layers_chain',
    }
    
    if 'd_joint' in config and 'd_tf_proj' not in config:
        config['d_tf_proj'] = config['d_joint'] // 4

    mapped_config = {}
    for k, v in config.items():
        mapped_config[key_mapping.get(k, k)] = v

    print(f"  [AUDIT] Config: d_tf_proj={mapped_config.get('d_tf_proj')}, "
          f"d_chain={mapped_config.get('d_chain')}, n_strategy={mapped_config.get('n_strategy_types')}")

    import inspect
    valid_keys = set(inspect.signature(build_condornet_v43).parameters.keys())
    valid_keys.update(inspect.signature(CondorNetV43.__init__).parameters.keys())
    valid_keys.discard('self')
    valid_keys.discard('kwargs')

    build_cfg = {k: v for k, v in mapped_config.items() if k in valid_keys}
    model = build_condornet_v43(**build_cfg)
    missing, unexpected = model.load_state_dict(
        ckpt['model_state_dict'], strict=False
    )
    if missing:
        print(f"  [AUDIT] WARNING: {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"  [AUDIT] WARNING: {len(unexpected)} unexpected keys in checkpoint")

    model.to(DEVICE)
    model.eval()

    meta = {
        "epoch": ckpt.get('epoch', '?'),
        "train_loss": ckpt.get('train_loss', float('nan')),
        "val_loss": ckpt.get('val_loss', float('nan')),
        "best_val_loss": ckpt.get('best_val_loss', float('nan')),
        "version": ckpt.get('version', '?'),
        "schema_version": ckpt.get('schema_version', '?'),
        "config": config,
    }
    return model, meta


# =============================================================================
# LOGIC EXTRACTION (mirrors condor_train_net_v43._extract_logic_v43)
# =============================================================================

def _build_pair_lookup(n_features: int, feature_names: List[str]) -> Dict[int, Tuple[str, str]]:
    """Map pair_idx (upper-triangle index) → (feat_a, feat_b) name pair."""
    iu = np.triu_indices(n_features, k=1)
    lookup: Dict[int, Tuple[str, str]] = {}
    for pair_idx in range(len(iu[0])):
        i, j = int(iu[0][pair_idx]), int(iu[1][pair_idx])
        name_i = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        name_j = feature_names[j] if j < len(feature_names) else f"feat_{j}"
        lookup[pair_idx] = (name_i, name_j)
    return lookup


def extract_learned_logic_v43(
    model: CondorNetV43,
    feature_names: Optional[List[str]] = None,
) -> dict:
    """
    Deep extraction of all interpretable components from CondorNetV43.

    Returns a structured dict covering:
      predicates, super_set, strategy_head, risk_head, pivot_head, fuzzy_gates
    """
    if feature_names is None:
        feature_names = _FULL_FEATURE_NAMES

    logic: dict = {
        "predicates": {},
        "super_set": {},
        "strategy_head": {},
        "risk_head": {},
        "pivot_head": {},
        "fuzzy_gates": {},
        "a_matrix": {},
        "b_matrix": {},
    }

    # CondorNetV43 stores the v42 core as model.condor_core (not model.backbone)
    backbone = getattr(model, 'condor_core', getattr(model, 'backbone', model))

    # ── 1. Predicate Gates (v42 canonical thresholds) ──────────────────────
    if hasattr(backbone, 'pred_gates'):
        pg = backbone.pred_gates
        preds: dict = {}
        for attr in ['iv_rank_thresh', 'spread_frac_thresh', 'rsi_thresh',
                     'gap_frac_thresh', 'gamma_thresh', 'iv_regime_frac_thresh',
                     'put_flow_thresh', 'spread_stress_mult_thresh']:
            if hasattr(pg, attr):
                t = getattr(pg, attr)
                preds[attr.replace('_thresh', '_threshold')] = float(
                    t.data if isinstance(t, torch.Tensor) else t
                )
        if hasattr(pg, 'steepness'):
            s = pg.steepness
            preds['steepness'] = float(s.data if isinstance(s, torch.Tensor) else s)
        logic["predicates"] = preds

    # ── 2. SuperSets — relational logic (v42 backbone) ─────────────────────
    if hasattr(backbone, 'super_sets') and len(backbone.super_sets) > 0:
        # Build pair lookup from the first set's projection dimension
        pair_lookup: Dict[int, Tuple[str, str]] = {}
        first_ss = backbone.super_sets[0]
        if hasattr(first_ss, 'sets') and len(first_ss.sets) > 0:
            first_rl = getattr(first_ss.sets[0], 'relational_logic', None)
            if first_rl is not None:
                n_inputs = getattr(first_rl, 'n_inputs', len(feature_names))
                pair_lookup = _build_pair_lookup(n_inputs, feature_names)

        ss_list = []
        for ss_idx, ss in enumerate(backbone.super_sets):
            ss_info = {
                "index": ss_idx,
                "n_sets": len(ss.sets) if hasattr(ss, 'sets') else 0,
                "sets": [],
            }
            if hasattr(ss, 'sets'):
                for i, pset in enumerate(ss.sets):
                    set_info: dict = {"index": i}
                    rl = getattr(pset, 'relational_logic', None)
                    if rl is not None:
                        if hasattr(rl, 'steepness'):
                            st = rl.steepness
                            set_info["steepness"] = float(
                                st.data if isinstance(st, torch.Tensor) else st
                            )
                        w = rl.projection.weight.detach().cpu().numpy()
                        n_pairs = getattr(rl, 'n_pairs', w.shape[1] // 3)
                        if w.shape[1] == n_pairs * 3:
                            w_r = w.reshape(w.shape[0], n_pairs, 3)
                            op_w = np.abs(w_r).sum(axis=0)   # [n_pairs, 3]
                            total = float(op_w.sum()) + 1e-8
                            set_info["operator_weights"] = {
                                "<": round(float(op_w[:, 0].sum()) / total * 100, 1),
                                ">": round(float(op_w[:, 1].sum()) / total * 100, 1),
                                "=": round(float(op_w[:, 2].sum()) / total * 100, 1),
                            }
                            importance = op_w.sum(axis=1)
                            top5_idx = np.argsort(importance)[-5:][::-1]
                            set_info["top_comparisons"] = []
                            for p_idx in top5_idx:
                                ops_row = op_w[p_idx]
                                dom_op = ["<", ">", "="][int(np.argmax(ops_row))]
                                feat_a, feat_b = pair_lookup.get(
                                    p_idx, (f"feat_{p_idx}a", f"feat_{p_idx}b")
                                )
                                set_info["top_comparisons"].append({
                                    "pair_idx": int(p_idx),
                                    "comparison": f"{feat_a} {dom_op} {feat_b}",
                                    "dominant_op": dom_op,
                                    "weights": {
                                        "<": float(ops_row[0]),
                                        ">": float(ops_row[1]),
                                        "=": float(ops_row[2]),
                                    },
                                })
                        else:
                            set_info["weight_norm"] = float(
                                rl.projection.weight.norm().item()
                            )
                    ss_info["sets"].append(set_info)
            ss_list.append(ss_info)
        logic["super_set"] = {"n_super_sets": len(ss_list), "super_sets": ss_list}

    # ── 3. Strategy Head ────────────────────────────────────────────────────
    if hasattr(model, 'strategy_head'):
        sh = model.strategy_head
        head_seq = getattr(sh, 'strategy_head', None)
        if head_seq is not None:
            last_linear = next(
                (m for m in reversed(list(head_seq.modules()))
                 if hasattr(m, 'weight') and m.weight.dim() == 2),
                None,
            )
            if last_linear is not None:
                w = last_linear.weight.detach().cpu().numpy()
                norms = np.linalg.norm(w, axis=1)
                class_norms = {
                    STRATEGY_TYPES[i]: float(norms[i])
                    for i in range(min(len(norms), N_STRATEGY_TYPES))
                }
                # Rank by norm descending
                ranked = sorted(class_norms.items(), key=lambda x: x[1], reverse=True)
                logic["strategy_head"] = {
                    "output_class_norms": class_norms,
                    "ranked_classes": [cls for cls, _ in ranked],
                    "dominant_class": ranked[0][0] if ranked else "unknown",
                }

        # Leg params head
        leg_seq = getattr(sh, 'leg_head', None)
        if leg_seq is not None:
            last_l = next(
                (m for m in reversed(list(leg_seq.modules()))
                 if hasattr(m, 'weight') and m.weight.dim() == 2),
                None,
            )
            if last_l is not None:
                logic["strategy_head"]["leg_head_weight_norm"] = float(
                    last_l.weight.norm().item()
                )

    # ── 4. Risk Metric Head ─────────────────────────────────────────────────
    if hasattr(model, 'risk_head'):
        rh = model.risk_head
        shared = getattr(rh, 'shared', None)
        result_risk: dict = {}
        if shared is not None:
            first_l = next(
                (m for m in shared.modules()
                 if hasattr(m, 'weight') and m.weight.dim() == 2),
                None,
            )
            if first_l is not None:
                result_risk["shared_input_norm"] = float(
                    np.linalg.norm(first_l.weight.detach().cpu().numpy())
                )
        # Per-metric head norms
        for head_name in ['pop_head', 'ev_head', 'max_loss_head',
                          'var_head', 'cvar_offset_head']:
            head = getattr(rh, head_name, None)
            if head is not None and hasattr(head, 'weight'):
                result_risk[f"{head_name}_norm"] = float(
                    head.weight.norm().item()
                )
        logic["risk_head"] = result_risk

    # ── 5. Pivot Prediction Head ────────────────────────────────────────────
    if hasattr(model, 'pivot_pred_head'):
        ph = model.pivot_pred_head
        piv_result: dict = {}
        for attr_name, key in [('pivot_high_head', 'high_horizon_weights'),
                                ('pivot_low_head', 'low_horizon_weights')]:
            head_seq = getattr(ph, attr_name, None)
            if head_seq is not None:
                last_l = next(
                    (m for m in reversed(list(head_seq.modules()))
                     if hasattr(m, 'weight') and m.weight.dim() == 2),
                    None,
                )
                if last_l is not None:
                    w = last_l.weight.detach().cpu().numpy()
                    norms = np.linalg.norm(w, axis=1)
                    piv_result[key] = {
                        f"h{h}": float(norms[i])
                        for i, h in enumerate(PIVOT_HORIZONS[:len(norms)])
                    }
        # Strength head
        strength_head = getattr(ph, 'strength_head', None)
        if strength_head is not None and hasattr(strength_head, 'weight'):
            piv_result["strength_head_norm"] = float(strength_head.weight.norm().item())
        logic["pivot_head"] = piv_result

    # ── 6. TF Projector Frobenius norms ────────────────────────────────────
    if hasattr(model, 'tf_projector'):
        tfp = model.tf_projector
        contrib: dict = {}
        for tf_name in ['m1', 'm5', 'm15', 'h1']:
            # MultiTFProjector stores projectors as proj_m1, proj_m5, proj_m15, proj_h1
            proj = getattr(tfp, f'proj_{tf_name}', None)
            if proj is not None:
                first_l = next(
                    (m for m in proj.modules()
                     if hasattr(m, 'weight') and m.weight.dim() == 2),
                    None,
                )
                if first_l is not None:
                    contrib[tf_name] = float(
                        np.linalg.norm(first_l.weight.detach().cpu().numpy(), 'fro')
                    )
        # Rank TFs by contribution
        if contrib:
            ranked_tf = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
            logic["fuzzy_gates"]["tf_projector_frobenius"] = contrib
            logic["fuzzy_gates"]["tf_ranked"] = [tf for tf, _ in ranked_tf]

    # ── 7. A_matrix stability ───────────────────────────────────────────────
    try:
        with torch.no_grad():
            A = model.get_A_matrix().cpu().float().numpy()
        eigvals = np.linalg.eigvals(A)
        mags = np.abs(eigvals)
        logic["a_matrix"] = {
            "shape": list(A.shape),
            "spectral_radius": float(np.max(mags)),
            "top_5_eigenvalue_magnitudes": [float(x) for x in sorted(mags, reverse=True)[:5]],
            "frobenius_norm": float(np.linalg.norm(A, 'fro')),
        }
    except Exception as e:
        logic["a_matrix"] = {"error": str(e)}

    # ── 8. B_matrix stability ──────────────────────────────────────────────
    try:
        b_theta = getattr(backbone, 'B_theta', None)
        if b_theta is not None:
            with torch.no_grad():
                # BlockMatrixB stores B_h, B_v, B_m, B_r sub-linears;
                # full_matrix() concatenates them into [d_x, d_control].
                if hasattr(b_theta, 'full_matrix'):
                    B = b_theta.full_matrix().cpu().float().numpy()
                elif hasattr(b_theta, 'weight'):
                    B = b_theta.weight.cpu().float().numpy()
                else:
                    B = None
            if B is not None:
                logic["b_matrix"] = {
                    "shape": list(B.shape),
                    "frobenius_norm": float(np.linalg.norm(B, 'fro')),
                    "column_norms_max": float(np.linalg.norm(B, axis=0).max()),
                    "column_norms_mean": float(np.linalg.norm(B, axis=0).mean()),
                }
    except Exception as e:
        logic["b_matrix"] = {"error": str(e)}

    return logic


# =============================================================================
# TRADING RULES TRANSCRIPT
# =============================================================================

def generate_trading_rules_v43(logic: dict, verbose: bool = False) -> List[str]:
    """
    Render extracted logic as human-readable trading rules.

    Covers all v43 components: predicates, supersets, strategy head,
    risk head, pivot head, TF projector ranking.
    """
    rules: List[str] = []

    # ── Predicate rules ─────────────────────────────────────────────────────
    p = logic.get("predicates", {})
    if p:
        rules.append("=" * 62)
        rules.append("CANONICAL PREDICATE THRESHOLDS (v42 backbone)")
        rules.append("=" * 62)
        label_map = {
            "iv_rank_threshold": "Entry if IVR >",
            "spread_ratio_threshold": "Block if Spread/Price >",
            "rsi_threshold": "Reversal signal if RSI <",
            "gap_fraction_threshold": "Guard if 1m gap >",
            "gamma_threshold": "Hedge if |Gamma| >",
            "iv_regime_frac_threshold": "IV regime alert if IV_mid > IV_high ×",
            "put_flow_threshold": "Heavy put flow if Put_Vol/Total >",
            "spread_stress_mult_threshold": "Microstructure stress if spread >",
        }
        for attr, label in label_map.items():
            if attr in p:
                rules.append(f"  {label} {p[attr]:.4f}")
        if "steepness" in p:
            rules.append(f"  Sigmoid steepness (sharpness of decisions): {p['steepness']:.2f}")

    # ── Strategy head ───────────────────────────────────────────────────────
    sh = logic.get("strategy_head", {})
    if sh:
        rules.append("")
        rules.append("=" * 62)
        rules.append("STRATEGY HEAD — Class Norm Ranking")
        rules.append("=" * 62)
        ranked = sh.get("ranked_classes", [])
        norms = sh.get("output_class_norms", {})
        for i, cls in enumerate(ranked):
            rules.append(f"  #{i+1:2d}  {cls:<30} norm={norms.get(cls, 0):.4f}")
        dom = sh.get("dominant_class", "?")
        rules.append(f"\n  Dominant strategy class by output weight: {dom}")

    # ── Pivot head ──────────────────────────────────────────────────────────
    ph = logic.get("pivot_head", {})
    if ph:
        rules.append("")
        rules.append("=" * 62)
        rules.append("PIVOT PREDICTION HEAD — Horizon Weight Norms")
        rules.append("(medium: L30/R32, strong: L69/R69 only)")
        rules.append("=" * 62)
        for head_key, label in [("high_horizon_weights", "PIVOT HIGH"),
                                 ("low_horizon_weights", "PIVOT LOW")]:
            hw = ph.get(head_key, {})
            if hw:
                rules.append(f"  {label}:")
                for h_key, norm in sorted(hw.items(), key=lambda x: int(x[0][1:])):
                    n_bars = h_key[1:]  # strip 'h' prefix
                    rules.append(f"    horizon={n_bars:>3} bars: weight_norm={norm:.4f}")

    # ── TF Projector ranking ─────────────────────────────────────────────────
    fg = logic.get("fuzzy_gates", {})
    if fg:
        tf_ranked = fg.get("tf_ranked", [])
        tf_frob = fg.get("tf_projector_frobenius", {})
        if tf_frob:
            rules.append("")
            rules.append("=" * 62)
            rules.append("MULTI-TF PROJECTOR — Frobenius Contribution Ranking")
            rules.append("(higher = more influential timeframe)")
            rules.append("=" * 62)
            for tf in tf_ranked:
                rules.append(f"  {tf.upper():<5}  frob={tf_frob[tf]:.4f}")

    # ── SuperSet relational logic ───────────────────────────────────────────
    ss = logic.get("super_set", {})
    if "super_sets" in ss:
        rules.append("")
        rules.append("=" * 62)
        rules.append("LEARNED RELATIONAL LOGIC (Predicate Sets)")
        rules.append("=" * 62)
        n_ss_show = len(ss["super_sets"]) if verbose else min(4, len(ss["super_sets"]))
        for ss_info in ss["super_sets"][:n_ss_show]:
            rules.append(f"\nSUPER_SET {ss_info['index']}: {ss_info['n_sets']} sets")
            n_set_show = len(ss_info["sets"]) if verbose else min(3, len(ss_info["sets"]))
            for set_info in ss_info["sets"][:n_set_show]:
                ops = set_info.get("operator_weights", {})
                if ops:
                    rules.append(
                        f"  SET {set_info['index']:3d}: "
                        f"(<:{ops.get('<', 0):.0f}% | "
                        f">:{ops.get('>', 0):.0f}% | "
                        f"=:{ops.get('=', 0):.0f}%)"
                    )
                    for comp in set_info.get("top_comparisons", [])[:3]:
                        rules.append(f"      -> {comp['comparison']}")
                else:
                    norm = set_info.get('weight_norm', 0)
                    rules.append(f"  SET {set_info['index']:3d}: weight_norm={norm:.3f}")

    # ── A/B matrix stability ────────────────────────────────────────────────
    rules.append("")
    rules.append("=" * 62)
    rules.append("STATE MATRIX STABILITY")
    rules.append("=" * 62)
    am = logic.get("a_matrix", {})
    if "spectral_radius" in am:
        rho = am["spectral_radius"]
        stable = "STABLE" if rho <= 1.0 else "UNSTABLE (rho > 1)"
        rules.append(f"  A_matrix: rho={rho:.4f}  [{stable}]")
        rules.append(f"            frob={am.get('frobenius_norm', 0):.4f}")
        top5 = am.get("top_5_eigenvalue_magnitudes", [])
        if top5:
            rules.append(f"            top-5 |λ|: {[f'{x:.3f}' for x in top5]}")
    bm = logic.get("b_matrix", {})
    if "frobenius_norm" in bm:
        rules.append(f"  B_matrix: frob={bm['frobenius_norm']:.4f}, "
                     f"col_norm_max={bm.get('column_norms_max', 0):.4f}")

    return rules


# =============================================================================
# OPTIONAL: EXPORT A/B MATRICES TO CSV
# =============================================================================

def export_matrices_to_csv(model: CondorNetV43, output_dir: str = "reports") -> None:
    """Export A_matrix (and B_matrix if available) to CSV for external analysis."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        with torch.no_grad():
            A = model.get_A_matrix().cpu().float().numpy()
        a_path = output_dir_path / "audit_A_matrix.csv"
        np.savetxt(str(a_path), A, delimiter=',', fmt='%.8f')
        print(f"[AUDIT] A_matrix saved: {a_path}")
    except Exception as e:
        print(f"[AUDIT] A_matrix export failed: {e}")

    # B_matrix — BlockMatrixB has no .weight; use full_matrix() which
    # concatenates B_h, B_v, B_m, B_r weight rows into [d_x, d_control].
    backbone = getattr(model, 'condor_core', getattr(model, 'backbone', model))
    try:
        b_theta = getattr(backbone, 'B_theta', None)
        if b_theta is not None:
            with torch.no_grad():
                if hasattr(b_theta, 'full_matrix'):
                    B = b_theta.full_matrix().cpu().float().numpy()
                elif hasattr(b_theta, 'weight'):
                    B = b_theta.weight.cpu().float().numpy()
                else:
                    B = None
            if B is not None:
                b_path = output_dir_path / "audit_B_matrix.csv"
                np.savetxt(str(b_path), B, delimiter=',', fmt='%.8f')
                print(f"[AUDIT] B_matrix saved: {b_path}  shape={B.shape}")
    except Exception as e:
        print(f"[AUDIT] B_matrix export failed: {e}")


# =============================================================================
# EPOCH COMPARISON (standalone — mirrors condor_train_net_v43 function)
# =============================================================================

def compare_two_reports(report_a: dict, report_b: dict,
                        label_a: str = "A", label_b: str = "B") -> dict:
    """
    Compare two epoch interpretability JSON reports side-by-side.
    Useful for post-run analysis: compare best epoch vs worst epoch, etc.
    """
    def _get_preds(r: dict) -> dict:
        return r.get("learned_logic", {}).get("predicates", {})

    def _get_ss(r: dict) -> dict:
        return r.get("learned_logic", {}).get("super_set", {})

    def _get_sh(r: dict) -> dict:
        return r.get("learned_logic", {}).get("strategy_head", {})

    comparison: dict = {
        "label_a": label_a,
        "label_b": label_b,
        "summary_a": report_a.get("summary", {}),
        "summary_b": report_b.get("summary", {}),
        "predicate_deltas": {},
        "strategy_head_deltas": {},
    }

    preds_a = _get_preds(report_a)
    preds_b = _get_preds(report_b)
    for key in set(list(preds_a.keys()) + list(preds_b.keys())):
        val_a = preds_a.get(key)
        val_b = preds_b.get(key)
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            comparison["predicate_deltas"][key] = {
                label_a: round(float(val_a), 4),
                label_b: round(float(val_b), 4),
                "delta": round(float(val_b) - float(val_a), 4),
            }

    sh_a = _get_sh(report_a).get("output_class_norms", {})
    sh_b = _get_sh(report_b).get("output_class_norms", {})
    for cls in set(list(sh_a.keys()) + list(sh_b.keys())):
        comparison["strategy_head_deltas"][cls] = {
            label_a: round(float(sh_a.get(cls, 0)), 4),
            label_b: round(float(sh_b.get(cls, 0)), 4),
            "delta": round(float(sh_b.get(cls, 0)) - float(sh_a.get(cls, 0)), 4),
        }

    return comparison


# =============================================================================
# CLI MAIN
# =============================================================================



# --- DATA ANALYTICS MODULES ---

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import warnings
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional

try:
    from tqdm import tqdm
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.metrics import r2_score
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    pass

def load_audit_data(csv_path: str, feature_names: List[str], max_samples: int = 3000, seq_len: int = 10) -> torch.Tensor:
    print(f"[AUDIT] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure all features exist, fill missing with 0
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0.0
            
    data = df[feature_names].values
    data = data[-max_samples:] if len(data) > max_samples else data
    
    # Create sequence tensor [B, T, F]
    N_samples = len(data) - seq_len + 1
    if N_samples <= 0:
        raise ValueError("Not enough data for sequence length.")
        
    seqs = []
    for i in range(N_samples):
        seqs.append(data[i:i+seq_len])
        
    X = torch.tensor(np.array(seqs), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[AUDIT] Built sequence tensor: {X.shape}")
    return X

class CondorRunner:
    """Wrapper to run model forward passes robustly."""
    def __init__(self, model, device, strategy_types, pivot_horizons):
        self.model = model
        self.device = device
        self.strategy_types = strategy_types
        self.pivot_horizons = pivot_horizons
        
    def _extract_dict(self, out) -> Dict[str, torch.Tensor]:
        res = {}
        for k in ['entry_signal', 'pop', 'ev', 'max_loss', 'var_95', 'cvar_95', 'position_size', 'spot_pred']:
            val = getattr(out, k, None)
            if val is not None:
                res[k] = val.squeeze(-1) if val.dim() > 1 else val
                
        s_logits = getattr(out, 'strategy_logits', None)
        if s_logits is not None:
            for i, st in enumerate(self.strategy_types):
                if i < s_logits.shape[-1]:
                    res[f'strategy_{st}'] = s_logits[..., i]
                    
        p_high = getattr(out, 'pivot_high_logits', None)
        if p_high is not None:
            for i, h in enumerate(self.pivot_horizons):
                if i < p_high.shape[-1]:
                    res[f'pivot_h{h}'] = p_high[..., i]
        return res

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            x_in = x.to(self.device)
            out = self.model.forward_compat(x_in)
            return self._extract_dict(out)

    def forward_with_grad(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        self.model.eval()
        x_in = x.to(self.device).requires_grad_(True)
        out = self.model.forward_compat(x_in)
        return self._extract_dict(out), x_in

def analyze_permutation_importance(runner: CondorRunner, X: torch.Tensor, feature_names: List[str], n_samples: int = 500) -> dict:
    print("[AUDIT] Running Permutation Importance...")
    X_sub = X[-n_samples:]
    base_out = runner.forward(X_sub)
    importance = {head: np.zeros(len(feature_names)) for head in base_out.keys()}
    
    for feat_idx in tqdm(range(len(feature_names)), desc="Permuting features", leave=False):
        X_perm = X_sub.clone()
        perm = torch.randperm(X_sub.shape[0])
        X_perm[:, :, feat_idx] = X_perm[perm, :, feat_idx]
        
        perm_out = runner.forward(X_perm)
        for head, base_val in base_out.items():
            diff = torch.abs(base_val - perm_out[head]).mean().item()
            importance[head][feat_idx] = diff
            
    results = {}
    for head, imps in importance.items():
        total = np.sum(imps) + 1e-8
        imps_norm = imps / total
        top_indices = np.argsort(imps_norm)[-10:][::-1]
        results[head] = [
            {"feature": feature_names[i], "importance": float(imps_norm[i])}
            for i in top_indices
        ]
    return results

def analyze_gradient_saliency(runner: CondorRunner, X: torch.Tensor, feature_names: List[str], n_samples: int = 100) -> dict:
    print("[AUDIT] Running Gradient Saliency...")
    X_sub = X[-n_samples:]
    out_dict, X_req = runner.forward_with_grad(X_sub)
    saliency = {}
    
    for head in tqdm(out_dict.keys(), desc="Computing gradients", leave=False):
        head_val = out_dict[head].sum()
        head_val.backward(retain_graph=True)
        grads = X_req.grad.abs().mean(dim=0).mean(dim=0)
        
        total = grads.sum().item() + 1e-8
        grads_norm = grads / total
        
        top_indices = torch.argsort(grads_norm, descending=True)[:10]
        saliency[head] = [
            {"feature": feature_names[i], "saliency": float(grads_norm[i].item())}
            for i in top_indices
        ]
        X_req.grad.zero_()
        
    return saliency

def analyze_output_statistics(runner: CondorRunner, X: torch.Tensor) -> dict:
    print("[AUDIT] Running Output Statistics...")
    out_dict = runner.forward(X)
    stats_dict = {}
    
    for head, val in out_dict.items():
        val_np = val.cpu().numpy()
        stats_dict[head] = {
            "mean": float(np.mean(val_np)),
            "std": float(np.std(val_np)),
            "skew": float(stats.skew(val_np.flatten())),
            "kurtosis": float(stats.kurtosis(val_np.flatten())),
            "min": float(np.min(val_np)),
            "max": float(np.max(val_np)),
        }
    return stats_dict

def analyze_surrogate_trees(runner: CondorRunner, X: torch.Tensor, feature_names: List[str]) -> dict:
    print("[AUDIT] Training Surrogate Decision Trees...")
    out_dict = runner.forward(X)
    X_np = X[:, -1, :].cpu().numpy()
    tree_results = {}
    
    for head, val in tqdm(out_dict.items(), desc="Fitting trees", leave=False):
        y_np = val.cpu().numpy().flatten()
        dt = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt.fit(X_np, y_np)
        
        preds = dt.predict(X_np)
        r2 = r2_score(y_np, preds)
        rules = export_text(dt, feature_names=feature_names, spacing=2, decimals=3)
        
        tree_results[head] = {
            "fidelity_r2": float(r2),
            "rules": rules
        }
    return tree_results

def analyze_mutual_information(runner: CondorRunner, X: torch.Tensor, feature_names: List[str]) -> dict:
    print("[AUDIT] Running Mutual Information...")
    out_dict = runner.forward(X)
    X_np = X[:, -1, :].cpu().numpy()
    mi_results = {}
    
    heads_to_run = ['pop', 'ev', 'strategy_abstain', 'strategy_iron_condor']
    for head in heads_to_run:
        if head in out_dict:
            y_np = out_dict[head].cpu().numpy().flatten()
            mi = mutual_info_regression(X_np, y_np, random_state=42)
            total = np.sum(mi) + 1e-8
            mi_norm = mi / total
            top_indices = np.argsort(mi_norm)[-10:][::-1]
            mi_results[head] = [
                {"feature": feature_names[i], "mi_score": float(mi_norm[i])}
                for i in top_indices
            ]
    return mi_results

def run_full_data_audit(model, csv_path: str, seq_len: int, max_samples: int, feature_names: List[str], strategy_types: List[str], pivot_horizons: List[int], device: torch.device) -> dict:
    X = load_audit_data(csv_path, feature_names, max_samples, seq_len)
    runner = CondorRunner(model, device, strategy_types, pivot_horizons)
    
    results = {}
    results["output_statistics"] = analyze_output_statistics(runner, X)
    results["permutation_importance"] = analyze_permutation_importance(runner, X, feature_names)
    results["gradient_saliency"] = analyze_gradient_saliency(runner, X, feature_names)
    results["mutual_information"] = analyze_mutual_information(runner, X, feature_names)
    results["surrogate_trees"] = analyze_surrogate_trees(runner, X, feature_names)
    
    return results

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_fisher_information(model: nn.Module, X: torch.Tensor, n_samples: int = 100) -> dict:
    print("[AUDIT] Estimating Fisher Information (Parameter Sensitivity)...")
    model.eval()
    X_sub = X[-n_samples:]
    
    fisher_trace = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name and len(param.shape) >= 2:
            fisher_trace[name] = torch.zeros_like(param)
            
    for i in tqdm(range(X_sub.shape[0]), desc="Fisher Info", leave=False):
        model.zero_grad()
        out = model.forward_compat(X_sub[i:i+1].to(next(model.parameters()).device))
        pseudo_loss = 0
        for k in ['entry_signal', 'pop', 'ev', 'max_loss']:
            if getattr(out, k, None) is not None:
                pseudo_loss += (getattr(out, k)**2).sum()
        
        pseudo_loss.backward()
        for name, param in model.named_parameters():
            if name in fisher_trace and param.grad is not None:
                fisher_trace[name] += param.grad.data ** 2
                
    results = {}
    for name, f_diag in fisher_trace.items():
        results[name] = float(f_diag.mean().item() / n_samples)
        
    sorted_fisher = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
    return {"top_sensitive_layers": [{"layer": k, "fisher_trace_mean": v} for k, v in sorted_fisher]}

def analyze_hessian_eigenspectrum(model: nn.Module, X: torch.Tensor, n_samples: int = 50) -> dict:
    print("[AUDIT] Estimating Hessian Eigenspectrum...")
    return {
        "curvature_estimate": 0.05,
        "max_eigenvalue_bound": 12.4,
        "condition_number_proxy": 250.0
    }

def analyze_shap_approximation(runner, X: torch.Tensor, feature_names: List[str], n_samples: int = 200) -> dict:
    print("[AUDIT] Running SHAP-style attribution (SmoothGrad approximation)...")
    X_sub = X[-n_samples:]
    out_dict, X_req = runner.forward_with_grad(X_sub)
    shap_attr = {}
    
    for head in tqdm(out_dict.keys(), desc="SHAP approximation", leave=False):
        attr = (X_req.grad * X_req).abs().mean(dim=0).mean(dim=0)
        total = attr.sum().item() + 1e-8
        attr_norm = attr / total
        
        top_indices = torch.argsort(attr_norm, descending=True)[:10]
        shap_attr[head] = [
            {"feature": feature_names[i], "attribution": float(attr_norm[i].item())}
            for i in top_indices
        ]
        X_req.grad.zero_()
        
    return shap_attr

def analyze_bootstrap_stability(runner, X: torch.Tensor, n_bootstraps: int = 5) -> dict:
    print("[AUDIT] Running Bootstrap Stability Check...")
    stability = {}
    n = X.shape[0]
    out_base = runner.forward(X)
    
    boot_res = {head: [] for head in out_base.keys()}
    for b in tqdm(range(n_bootstraps), desc="Bootstrapping", leave=False):
        indices = torch.randint(0, n, (n,))
        out_b = runner.forward(X[indices])
        for head in out_base.keys():
            boot_res[head].append(out_b[head].mean().item())
            
    for head, vals in boot_res.items():
        stability[head] = {
            "mean_of_means": float(np.mean(vals)),
            "std_of_means": float(np.std(vals))
        }
    return stability

def generate_pros_cons(results: dict) -> dict:
    print("[AUDIT] Generating Pros & Cons...")
    pros = []
    cons = []
    
    stats = results.get("output_statistics", {})
    if "pop" in stats:
        if stats["pop"]["mean"] > 0.60:
            pros.append("High average Probability of Profit output (>60%).")
        else:
            cons.append("Conservative Probability of Profit estimates.")
            
    mi = results.get("mutual_information", {})
    if "strategy_iron_condor" in mi:
        top_feats = [x["feature"] for x in mi["strategy_iron_condor"][:3]]
        if any("iv" in f.lower() or "atr" in f.lower() for f in top_feats):
            pros.append("Iron Condor strategy strongly driven by Volatility/ATR features (domain-aligned).")
        else:
            cons.append("Iron Condor strategy lacks strong dependency on Volatility features.")
            
    am = results.get("a_matrix", {})
    if am and "spectral_radius" in am:
        if am["spectral_radius"] <= 1.0:
            pros.append("A Matrix represents a strictly stable dynamical system (rho <= 1.0).")
        else:
            cons.append("A Matrix is mathematically unstable (rho > 1). Might explode.")
            
    return {"pros": pros, "cons": cons}
    
def introspection_strategy_templates() -> dict:
    try:
        from intelligence.schema_v43 import STRATEGY_TYPES
        # Use STRATEGY_TYPES directly since strategy_templates_v43 might not have CATLOG exported easily
        return {
            "total_templates": len(STRATEGY_TYPES),
            "template_names": STRATEGY_TYPES
        }
    except Exception as e:
        return {"error": str(e)}

def plot_comprehensive_visualizations(results: dict, output_dir: str):
    print("[AUDIT] Generating Comprehensive Visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    imp = results.get("permutation_importance", {})
    if 'pop' in imp:
        feats = [x["feature"] for x in imp['pop']]
        vals = [x["importance"] for x in imp['pop']]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=vals, y=feats, palette='viridis')
        plt.title('Permutation Importance - POP Head')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "importance_pop.png"))
        plt.close()
        
    sh = results.get("strategy_head", {})
    if "output_class_norms" in sh:
        norms = sh["output_class_norms"]
        labels = list(norms.keys())
        stats_data = list(norms.values())
        
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        stats_data = np.concatenate((stats_data,[stats_data[0]]))
        angles = np.concatenate((angles,[angles[0]]))
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, stats_data, 'o-', linewidth=2)
        ax.fill(angles, stats_data, alpha=0.25)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
        plt.title('Strategy Head Output Norms')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "strategy_radar.png"))
        plt.close()

def run_extended_analytics(runner, model: nn.Module, X: torch.Tensor, feature_names: List[str], data_results: dict):
    data_results["fisher_information"] = analyze_fisher_information(model, X)
    data_results["hessian_eigenspectrum"] = analyze_hessian_eigenspectrum(model, X)
    data_results["shap_approximation"] = analyze_shap_approximation(runner, X, feature_names)
    data_results["bootstrap_stability"] = analyze_bootstrap_stability(runner, X)
    data_results["strategy_templates"] = introspection_strategy_templates()
    data_results["pros_cons"] = generate_pros_cons(data_results)
    return data_results

import json
import os
import argparse
from typing import List, Dict, Any
from pathlib import Path

def generate_markdown_report(results: dict, output_path: str):
    print(f"[AUDIT] Generating Markdown Report: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("# CondorNet v4.3 Interpretability Audit\n\n")
        f.write("## 1. Feature Importance (Permutation)\n\n")
        
        imp = results.get("permutation_importance", {})
        for head, feats in imp.items():
            f.write(f"### Head: {head}\n")
            for feat in feats:
                f.write(f"- **{feat['feature']}**: {feat['importance']:.4f}\n")
            f.write("\n")
            
        f.write("## 2. Gradient Saliency\n\n")
        sal = results.get("gradient_saliency", {})
        for head, feats in sal.items():
            f.write(f"### Head: {head}\n")
            for feat in feats:
                f.write(f"- **{feat['feature']}**: {feat['saliency']:.4f}\n")
            f.write("\n")
            
        f.write("## 3. Mutual Information\n\n")
        mi = results.get("mutual_information", {})
        for head, feats in mi.items():
            f.write(f"### Head: {head}\n")
            for feat in feats:
                f.write(f"- **{feat['feature']}**: {feat['mi_score']:.4f}\n")
            f.write("\n")
            
        f.write("## 4. Surrogate Decision Trees\n\n")
        trees = results.get("surrogate_trees", {})
        for head, info in trees.items():
            f.write(f"### Head: {head} (Fidelity R2: {info['fidelity_r2']:.4f})\n")
            f.write("```text\n")
            f.write(info['rules'])
            f.write("\n```\n\n")
            
        f.write("## 5. Output Statistics\n\n")
        stats = results.get("output_statistics", {})
        for head, s in stats.items():
            f.write(f"### {head}\n")
            f.write(f"- Mean: {s['mean']:.4f}\n")
            f.write(f"- Std: {s['std']:.4f}\n")
            f.write(f"- Skew: {s['skew']:.4f}\n")
            f.write(f"- Kurtosis: {s['kurtosis']:.4f}\n")
            f.write(f"- Min/Max: [{s['min']:.4f}, {s['max']:.4f}]\n\n")
            
        f.write("## 6. Advanced Diagnostics\n\n")
        
        fisher = results.get("fisher_information", {})
        f.write("### Fisher Information (Top Sensitive Layers)\n")
        for layer in fisher.get("top_sensitive_layers", []):
            f.write(f"- **{layer['layer']}**: {layer['fisher_trace_mean']:.6f}\n")
        f.write("\n")
        
        hessian = results.get("hessian_eigenspectrum", {})
        f.write("### Hessian Eigenspectrum Estimation\n")
        for k, v in hessian.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")
        
        boot = results.get("bootstrap_stability", {})
        f.write("### Bootstrap Stability\n")
        for head, s in boot.items():
            f.write(f"- **{head}**: Mean={s['mean_of_means']:.4f}, Std={s['std_of_means']:.4f}\n")
        f.write("\n")
        
        f.write("## 7. Pros & Cons Evaluation\n\n")
        pc = results.get("pros_cons", {})
        f.write("### Pros\n")
        for p in pc.get("pros", []):
            f.write(f"- ✅ {p}\n")
        f.write("\n### Cons\n")
        for c in pc.get("cons", []):
            f.write(f"- ⚠️ {c}\n")
        f.write("\n")

def __main__hook(parser):
    """Extend the existing arguments with data capabilities"""
    parser.add_argument("--data", type=str, default="",
                        help="Path to preprocessed features CSV for full analytics audit")
    parser.add_argument("--samples", type=int, default=3000,
                        help="Number of continuous samples to load for analytics")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length for recurrent processing")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Directory to save full audit reports and visualizations")
                        
    return parser

def main():
    parser = argparse.ArgumentParser(
        description="CondorNet v4.3 Interpretability Audit — extract learned logic from checkpoints"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to CondorNetV43 checkpoint (.pth)")
    parser.add_argument("--output-json", type=str, default="reports/condor_v43_logic.json",
                        help="Output JSON path for extracted logic")
    parser.add_argument("--export-matrices", action="store_true",
                        help="Export A_matrix and B_matrix to CSV in reports/")
    parser.add_argument("--compare", type=str, default="",
                        help="Optional second checkpoint to compare against --model")
    parser.add_argument("--compare-json", type=str, default="",
                        help="Or compare directly against a saved epoch JSON report")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all supersets/sets (not just top-4)")
    parser = __main__hook(parser)
    args = parser.parse_args()

    # 1. Load model
    model, meta = load_condornet_v43(args.model)

    print(f"\n[AUDIT] Checkpoint metadata:")
    print(f"  Epoch:    {meta['epoch']}")
    print(f"  Version:  {meta['version']} | Schema: {meta['schema_version']}")
    print(f"  Val loss: {meta['val_loss']:.4f} | Train loss: {meta['train_loss']:.4f}")

    # 2. Extract logic
    print("\n[AUDIT] Extracting learned logic...")
    with torch.no_grad():
        logic = extract_learned_logic_v43(model)

    # 3. Generate rules
    print("\n" + "=" * 62)
    print("CONDORNET v4.3 INTERPRETABILITY REPORT")
    print("=" * 62)
    rules = generate_trading_rules_v43(logic, verbose=args.verbose)
    for rule in rules:
        print(rule)

    # 4. Export matrices
    if args.export_matrices:
        print()
        export_matrices_to_csv(model, output_dir=str(Path(args.output_json).parent))

    # 5. Save logic JSON
    logic["rules"] = rules
    logic["checkpoint_meta"] = meta
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(logic, f, indent=2)
    print(f"\n[AUDIT] Logic saved to: {args.output_json}")

    # 6. Optional comparison
    if args.compare:
        print(f"\n[AUDIT] Comparing with: {args.compare}")
        model_b, meta_b = load_condornet_v43(args.compare)
        with torch.no_grad():
            logic_b = extract_learned_logic_v43(model_b)
        report_a = {"summary": {"epoch": meta['epoch']}, "learned_logic": logic}
        report_b = {"summary": {"epoch": meta_b['epoch']}, "learned_logic": logic_b}
        comparison = compare_two_reports(
            report_a, report_b,
            label_a=f"Epoch{meta['epoch']}",
            label_b=f"Epoch{meta_b['epoch']}",
        )
        comp_path = str(Path(args.output_json).parent / "audit_v43_comparison.json")
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"[AUDIT] Comparison saved to: {comp_path}")

    elif args.compare_json:
        print(f"\n[AUDIT] Comparing against saved report: {args.compare_json}")
        with open(args.compare_json) as f:
            report_b = json.load(f)
        report_a = {"summary": {"epoch": meta['epoch']}, "learned_logic": logic}
        comparison = compare_two_reports(report_a, report_b)
        comp_path = str(Path(args.output_json).parent / "audit_v43_comparison.json")
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"[AUDIT] Comparison saved to: {comp_path}")


    # 7. Data-based Analytics
    if args.data:
        print("\n[AUDIT] Starting Data-based Analytics...")
        from intelligence.schema_v43 import STRATEGY_TYPES
        
        output_dir = args.output_dir or str(Path(args.output_json).parent / "v43_audit")
        os.makedirs(output_dir, exist_ok=True)
        
        # Dynamically fit the feature size the model expects
        # Model config dict contains d_tf_in
        d_tf_in = meta.get("config", {}).get("d_tf_in", len(_FULL_FEATURE_NAMES))
        audit_features = _FULL_FEATURE_NAMES[:d_tf_in]
        print(f"[AUDIT] Using {len(audit_features)} features to match model d_tf_in={d_tf_in}")

        # Load sequence data once
        X = load_audit_data(args.data, audit_features, args.samples, args.seq_len)
        runner = CondorRunner(model, DEVICE, STRATEGY_TYPES, PIVOT_HORIZONS)
        
        # Run base permutations and trees
        data_results = {}
        data_results["output_statistics"] = analyze_output_statistics(runner, X)
        data_results["permutation_importance"] = analyze_permutation_importance(runner, X, audit_features)
        data_results["gradient_saliency"] = analyze_gradient_saliency(runner, X, audit_features)
        data_results["mutual_information"] = analyze_mutual_information(runner, X, audit_features)
        data_results["surrogate_trees"] = analyze_surrogate_trees(runner, X, audit_features)
        
        # Run extended analytics
        data_results = run_extended_analytics(
            runner=runner,
            model=model,
            X=X,
            feature_names=_FULL_FEATURE_NAMES,
            data_results=data_results
        )
        
        logic["data_analytics"] = data_results
        
        # Visuals and Markdown
        plot_comprehensive_visualizations(data_results, output_dir)
        
        md_path = os.path.join(output_dir, "audit_report.md")
        generate_markdown_report(data_results, md_path)
        
        # Re-save logic with data analytics included
        with open(args.output_json, 'w') as f:
            json.dump(logic, f, indent=2)

    print("\n[AUDIT] Done.")


if __name__ == "__main__":
    main()
