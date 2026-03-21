"""
       audit_condornet_interpretability_v43.py — CondorNet v4.3 Full Interpretability Audit
       ======================================================================================
       Comprehensive post-hoc analysis of CondorNetV43 checkpoints. Extracts and renders:

         WEIGHT-BASED (no data required):
         1.  Predicate thresholds (v42 backbone canonical gates)
         2.  SuperSet relational logic — dominant comparison operators, top feature pairs
         3.  Strategy head — per-class output norm ranking
         4.  Risk metric head — shared representation norms
         5.  Pivot prediction head — per-horizon weight norms
         6.  TF projector Frobenius contributions per timeframe
         7.  A_matrix / B_matrix stability metrics (Jacobian, SVD, spectral radius)
         8.  Human-readable trading rules transcript
         9.  All 58+ strategy template catalog introspection
         10. B-matrix extended analysis (SVD, Jacobian approximation)

         DATA-BASED (requires --data):
         11. Permutation importance rankings (all features, all heads)
         12. Gradient saliency analysis (feature + temporal)
         13. Fisher Information estimation (per-layer parameter sensitivity)
         14. Hessian eigenspectrum (loss landscape curvature)
         15. SHAP-style attribution approximation (per-feature, per-head)
         16. Mutual Information matrix (pairwise feature dependencies)
         17. Surrogate decision tree analysis (R2 fidelity + rules)
         18. Output statistics per head (mean, std, skew, kurtosis)
         19. Bootstrap stability analysis (self-consistency)
         20. Pros/Cons assessment (deterministic rubric)
         21. Comprehensive visualizations (PNG plots)
         22. Full Markdown audit report

       Usage:
           # Weight-only audit (fast, no data needed):
           python intelligence/audit_condornet_interpretability_v43.py \\
               --model models/condornet_v43_epoch27.pth \\
               --output-json reports/v43_logic.json \\
               --export-matrices --verbose

           # Full analytics audit with data:
           python intelligence/audit_condornet_interpretability_v43.py \\
               --model models/condornet_v43_epoch27.pth \\
               --data data/processed/spy_features_v43.csv \\
               --samples 3000 --seq-len 10 \\
               --output-dir reports/v43_audit/ \\
               --output-json reports/v43_audit/logic.json \\
               --export-matrices --verbose

       Author: CondorNet v4.3 Implementation
       Version: 4.3.1
       Date: 2026-02-27
       """

       from __future__ import annotations

       import argparse
       import json
       import os
       import sys
       import warnings
       from datetime import datetime
       from pathlib import Path
       from typing import Dict, List, Optional, Tuple

       import numpy as np
       import torch
       import torch.nn as nn

       warnings.filterwarnings('ignore')

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

       # Full 64-entry feature name list matching _FULL_FEATURE_NAMES in training script.
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
       EPS = 1e-8

       # Output head names for analytics
       OUTPUT_HEAD_NAMES = [
           'entry_signal', 'pop', 'ev', 'max_loss', 'var_95', 'cvar_95',
           'position_size', 'spot_pred',
           'pivot_h5', 'pivot_h10', 'pivot_h20', 'pivot_h35', 'pivot_h70',
       ] + [f'strategy_{st}' for st in STRATEGY_TYPES[:N_STRATEGY_TYPES]]

       COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95C623']


       # =============================================================================
       # MODEL LOADING
       # =============================================================================

       def load_condornet_v43(ckpt_path: str) -> Tuple[CondorNetV43, dict]:
           """Load CondorNetV43 from checkpoint."""
           print(f"[AUDIT] Loading: {ckpt_path}")
           ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

           config = ckpt.get('config', {})
           if not config:
               config = {
                   'd_tf_in': len(TF_FEATURE_NAMES),
                   'd_joint': 256,
                   'd_chain': CHAIN_GRID_CONFIG['d_chain'],
                   'd_pivot': 16,
                   'd_fused': 256,
                   'n_pivot_features': N_PIVOT_FEATURES,
                   'n_strategy_types': N_STRATEGY_TYPES,
                   'chain_in_features': CHAIN_GRID_CONFIG['in_features'],
                   'chain_d_model': CHAIN_GRID_CONFIG['d_model'],
                   'chain_n_heads': CHAIN_GRID_CONFIG['n_heads'],
                   'chain_n_layers': CHAIN_GRID_CONFIG['n_layers'],
                   'chain_d_ff': CHAIN_GRID_CONFIG['d_ff'],
               }
               print("  [AUDIT] No 'config' in checkpoint — using schema defaults")

           print(f"  [AUDIT] Config: d_joint={config.get('d_joint')}, "
                 f"d_chain={config.get('d_chain')}, n_strategy={config.get('n_strategy_types')}")

           _skip = {'optimizer_state_dict', 'scheduler_state_dict', 'model_state_dict',
                    'normalization', 'epoch', 'train_loss', 'val_loss', 'best_val_loss',
                    'version', 'schema_version'}
           build_cfg = {k: v for k, v in config.items() if k not in _skip}
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
       # LOGIC EXTRACTION
       # =============================================================================

       def _build_pair_lookup(n_features: int, feature_names: List[str]) -> Dict[int, Tuple[str, str]]:
           """Map pair_idx (upper-triangle index) to (feat_a, feat_b) name pair."""
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
           """Deep extraction of all interpretable components from CondorNetV43."""
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

           backbone = getattr(model, 'condor_core', getattr(model, 'backbone', model))

           # 1. Predicate Gates
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

           # 2. SuperSets — relational logic
           if hasattr(backbone, 'super_sets') and len(backbone.super_sets) > 0:
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
                                   op_w = np.abs(w_r).sum(axis=0)
                                   total = float(op_w.sum()) + EPS
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

           # 3. Strategy Head
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
                       ranked = sorted(class_norms.items(), key=lambda x: x[1], reverse=True)
                       logic["strategy_head"] = {
                           "output_class_norms": class_norms,
                           "ranked_classes": [cls for cls, _ in ranked],
                           "dominant_class": ranked[0][0] if ranked else "unknown",
                       }
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

           # 4. Risk Metric Head
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
               for head_name in ['pop_head', 'ev_head', 'max_loss_head',
                                 'var_head', 'cvar_offset_head']:
                   head = getattr(rh, head_name, None)
                   if head is not None and hasattr(head, 'weight'):
                       result_risk[f"{head_name}_norm"] = float(
                           head.weight.norm().item()
                       )
               logic["risk_head"] = result_risk

           # 5. Pivot Prediction Head
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
               strength_head = getattr(ph, 'strength_head', None)
               if strength_head is not None and hasattr(strength_head, 'weight'):
                   piv_result["strength_head_norm"] = float(strength_head.weight.norm().item())
               logic["pivot_head"] = piv_result

           # 6. TF Projector Frobenius norms
           if hasattr(model, 'tf_projector'):
               tfp = model.tf_projector
               contrib: dict = {}
               for tf_name in ['m1', 'm5', 'm15', 'h1']:
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
               if contrib:
                   ranked_tf = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
                   logic["fuzzy_gates"]["tf_projector_frobenius"] = contrib
                   logic["fuzzy_gates"]["tf_ranked"] = [tf for tf, _ in ranked_tf]

           # 7. A_matrix stability
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

           # 8. B_matrix stability
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
           """Render extracted logic as human-readable trading rules."""
           rules: List[str] = []

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
                   "iv_regime_frac_threshold": "IV regime alert if IV_mid > IV_high x",
                   "put_flow_threshold": "Heavy put flow if Put_Vol/Total >",
                   "spread_stress_mult_threshold": "Microstructure stress if spread >",
               }
               for attr, label in label_map.items():
                   if attr in p:
                       rules.append(f"  {label} {p[attr]:.4f}")
               if "steepness" in p:
                   rules.append(f"  Sigmoid steepness (sharpness of decisions): {p['steepness']:.2f}")

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
                           n_bars = h_key[1:]
                           rules.append(f"    horizon={n_bars:>3} bars: weight_norm={norm:.4f}")

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
                   rules.append(f"            top-5 |lambda|: {[f'{x:.3f}' for x in top5]}")
           bm = logic.get("b_matrix", {})
           if "frobenius_norm" in bm:
               rules.append(f"  B_matrix: frob={bm['frobenius_norm']:.4f}, "
                            f"col_norm_max={bm.get('column_norms_max', 0):.4f}")

           return rules