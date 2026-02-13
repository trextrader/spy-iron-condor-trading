#!/usr/bin/env python3
"""
CondorNet Interpretability Audit Tool

Extracts learned thresholds, relational logic, and trading rules from
trained CondorNet checkpoints. Provides human-readable interpretation
of the neural model's decision logic.

Usage:
    python intelligence/audit_condornet_interpretability.py \
        --model models/condor_net_e5_dh64_lr1e04.pth \
        --data data/processed/mamba_institutional_2024_1m_v22.csv \
        --samples 2000 \
        --output-json reports/condor_logic.json
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import sys
import os
import json
import re
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.getcwd())

from intelligence.condor_brain_net import CondorNet
from intelligence.canonical_feature_registry import FEATURE_COLS_V22, FEATURE_COLS_V30, select_feature_frame

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 240  # CondorNet standard
EPS = 1e-6


def infer_architecture_from_state_dict(state_dict: dict) -> dict:
    """
    Infer CondorNet architecture parameters from state_dict keys and tensor shapes.

    This is used when checkpoints don't include hyperparameters metadata.
    Parses keys like 'super_sets.7.sets.31.relational_logic.steepness' to
    determine n_super_sets=8 and n_sets=32.

    Also infers n_features from relational_logic.projection.weight shapes.
    """
    n_super_sets = 0
    n_sets = 0
    n_predicates = 32  # Default
    n_features = None

    super_set_pattern = re.compile(r'^super_sets\.(\d+)\.')
    sets_pattern = re.compile(r'\.sets\.(\d+)\.')

    for key in state_dict.keys():
        # Find max super_set index
        ss_match = super_set_pattern.match(key)
        if ss_match:
            ss_idx = int(ss_match.group(1))
            n_super_sets = max(n_super_sets, ss_idx + 1)

        # Find max sets index within super_sets
        sets_match = sets_pattern.search(key)
        if sets_match:
            set_idx = int(sets_match.group(1))
            n_sets = max(n_sets, set_idx + 1)

    # Infer n_features (d_input) from initial_encoder or tft.input_proj weights
    # These layers have shape [hidden_dim, d_input]
    for key, tensor in state_dict.items():
        if ('initial_encoder.0.weight' in key or 'tft.input_proj.weight' in key) and tensor.dim() == 2:
            n_features = tensor.shape[1]  # Second dimension is input features
            break

    # Infer n_predicates from pred_gates.extra_heads shape
    # extra_heads[0].weight shape is (n_extra_preds, 11)
    # n_extra_preds = n_predicates - 8 (8 canonical gates in V3.0)
    n_base_predicates = 8  # V3.0: 5 original + 3 V3.0 canonical gates
    for key, tensor in state_dict.items():
        if 'pred_gates.extra_heads.0.weight' in key:
            n_extra = tensor.shape[0]
            n_predicates = n_extra + n_base_predicates
            break

    return {
        'n_super_sets': max(n_super_sets, 8),
        'n_sets': max(n_sets, 16),
        'n_predicates': n_predicates,
        'n_features': n_features
    }


def build_pair_lookup(n_features: int, feature_names: list) -> dict:
    """
    Build a lookup table mapping pair_idx -> (feature_a, feature_b).

    The RelationalLogicLayer uses upper triangle indices (i < j).
    pair_idx corresponds to torch.triu_indices ordering.
    """
    iu = np.triu_indices(n_features, k=1)
    pair_lookup = {}
    for pair_idx in range(len(iu[0])):
        i, j = iu[0][pair_idx], iu[1][pair_idx]
        name_i = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        name_j = feature_names[j] if j < len(feature_names) else f"feat_{j}"
        pair_lookup[pair_idx] = (name_i, name_j)
    return pair_lookup


def format_comparison(pair_idx: int, dominant_op: str, pair_lookup: dict) -> str:
    """Format a comparison as 'feature_a < feature_b' or similar."""
    if pair_idx not in pair_lookup:
        return f"Pair_{pair_idx} {dominant_op} ?"
    feat_a, feat_b = pair_lookup[pair_idx]
    return f"{feat_a} {dominant_op} {feat_b}"


def safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_condor_model(ckpt_path):
    """
    Load CondorNet model from checkpoint, inferring architecture from state_dict
    when hyperparameters are not saved in the checkpoint.
    """
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    sd = ckpt['state_dims']
    feature_cols = ckpt.get('feature_cols', FEATURE_COLS_V22)
    state_dict = ckpt['model_state_dict']

    # Try to read hyperparameters from checkpoint metadata
    hp = ckpt.get('hyperparameters', ckpt.get('model_config', {}))

    # Always infer from state_dict to get correct dimensions
    print("  Inferring architecture from state_dict...")
    inferred = infer_architecture_from_state_dict(state_dict)

    # Use checkpoint hyperparameters if available, otherwise use inferred
    if hp and 'n_sets' in hp:
        n_predicates = hp.get('n_predicates', inferred['n_predicates'])
        n_sets = hp.get('n_sets', inferred['n_sets'])
        n_super_sets = hp.get('n_super_sets', inferred['n_super_sets'])
    else:
        n_predicates = inferred['n_predicates']
        n_sets = inferred['n_sets']
        n_super_sets = inferred['n_super_sets']

    # CRITICAL: Use inferred n_features if available, not len(feature_cols)
    # This ensures model dimensions match the checkpoint
    if inferred.get('n_features'):
        d_input = inferred['n_features']
        print(f"  Using inferred d_input={d_input} (checkpoint was trained with {d_input} features)")
        # Adjust feature_cols to match
        if len(feature_cols) > d_input:
            feature_cols = feature_cols[:d_input]
        elif len(feature_cols) < d_input:
            # Pad with generic names
            feature_cols = list(feature_cols) + [f'feat_{i}' for i in range(len(feature_cols), d_input)]
    else:
        d_input = len(feature_cols)

    print(f"  Architecture: d_input={d_input}, d_h={sd['d_h']}, d_v={sd['d_v']}, d_m={sd['d_m']}, d_r={sd['d_r']}")
    print(f"  Logic: n_predicates={n_predicates}, n_sets={n_sets}, n_super_sets={n_super_sets}")

    model = CondorNet(
        d_input=d_input,
        d_h=sd['d_h'], d_v=sd['d_v'], d_m=sd['d_m'], d_r=sd['d_r'],
        n_predicates=n_predicates,
        n_sets=n_sets,
        n_super_sets=n_super_sets,
    )

    # Load with strict=False to handle minor mismatches, then report
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  Warning: {len(missing)} missing keys (model has extra params)")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys in checkpoint")

    model.to(DEVICE)
    model.eval()
    return model, feature_cols, ckpt.get('normalization', None)


def extract_learned_logic(model, feature_names):
    """Extracts internal weights and thresholds from the CondorNet modules."""
    logic = {
        "predicates": {},
        "super_set": {},
        "output_head": {}
    }

    # 1. Extract Predicate Thresholds (V2.2 + V3.0)
    if hasattr(model, 'pred_gates'):
        pg = model.pred_gates
        logic["predicates"] = {
            "iv_rank_threshold": float(pg.iv_rank_thresh.data),
            "spread_ratio_threshold": float(pg.spread_frac_thresh.data),
            "rsi_threshold": float(pg.rsi_thresh.data),
            "gap_fraction_threshold": float(pg.gap_frac_thresh.data),
            "gamma_threshold": float(pg.gamma_thresh.data),
            "steepness": pg.steepness,
        }
        # V3.0 thresholds (if available)
        if hasattr(pg, 'iv_regime_frac_thresh'):
            logic["predicates"]["iv_regime_frac_threshold"] = float(pg.iv_regime_frac_thresh.data)
        if hasattr(pg, 'put_flow_thresh'):
            logic["predicates"]["put_flow_threshold"] = float(pg.put_flow_thresh.data)
        if hasattr(pg, 'spread_stress_mult_thresh'):
            logic["predicates"]["spread_stress_mult_threshold"] = float(pg.spread_stress_mult_thresh.data)

    # 2. Extract SuperSet Logic (V21+: super_sets is a ModuleList)
    if hasattr(model, 'super_sets') and len(model.super_sets) > 0:
        logic["super_set"]["n_super_sets"] = len(model.super_sets)
        logic["super_set"]["super_sets"] = []

        # Build pair lookup for first super_set
        first_ss = model.super_sets[0]
        if hasattr(first_ss, 'sets') and len(first_ss.sets) > 0:
            first_rl = first_ss.sets[0].relational_logic if hasattr(first_ss.sets[0], 'relational_logic') else None
            if first_rl and hasattr(first_rl, 'n_inputs'):
                n_inputs = first_rl.n_inputs
                pair_lookup = build_pair_lookup(n_inputs, feature_names)
            else:
                pair_lookup = {}
        else:
            pair_lookup = {}

        for ss_idx, ss in enumerate(model.super_sets):
            ss_info = {
                "index": ss_idx,
                "n_sets": len(ss.sets) if hasattr(ss, 'sets') else 0,
                "sets": []
            }
            if hasattr(ss, 'sets'):
                for i, pset in enumerate(ss.sets):
                    set_info = {"index": i}
                    if hasattr(pset, 'relational_logic'):
                        rl = pset.relational_logic
                        set_info["steepness"] = float(rl.steepness.data) if hasattr(rl, 'steepness') else 10.0

                        w = rl.projection.weight.detach().cpu().numpy()
                        n_pairs = rl.n_pairs if hasattr(rl, 'n_pairs') else w.shape[1] // 3

                        if w.shape[1] == n_pairs * 3:
                            w_reshaped = w.reshape(w.shape[0], n_pairs, 3)
                            op_weights = np.abs(w_reshaped).sum(axis=0)
                            total_lt = float(op_weights[:, 0].sum())
                            total_gt = float(op_weights[:, 1].sum())
                            total_eq = float(op_weights[:, 2].sum())

                            total = total_lt + total_gt + total_eq + 1e-8
                            set_info["operator_weights"] = {
                                "<": round(100 * total_lt / total, 1),
                                ">": round(100 * total_gt / total, 1),
                                "=": round(100 * total_eq / total, 1)
                            }

                            pair_importance = op_weights.sum(axis=1)
                            top_pairs = np.argsort(pair_importance)[-5:][::-1]
                            set_info["top_comparisons"] = []
                            for p_idx in top_pairs:
                                ops = op_weights[p_idx]
                                dominant_op = ["<", ">", "="][np.argmax(ops)]
                                comparison_str = format_comparison(p_idx, dominant_op, pair_lookup)
                                set_info["top_comparisons"].append({
                                    "pair_idx": int(p_idx),
                                    "comparison": comparison_str,
                                    "dominant_op": dominant_op,
                                    "weights": {"<": float(ops[0]), ">": float(ops[1]), "=": float(ops[2])}
                                })
                        else:
                            set_info["projection_weight_norm"] = float(rl.projection.weight.norm().item())
                    ss_info["sets"].append(set_info)
            logic["super_set"]["super_sets"].append(ss_info)

    # Legacy fallback
    elif hasattr(model, 'super_set'):
        ss = model.super_set
        logic["super_set"]["n_sets"] = ss.n_sets
        logic["super_set"]["sets"] = []
        for i, pset in enumerate(ss.sets):
            weights = torch.softmax(pset.membership, dim=0).detach().cpu().numpy().tolist()
            logic["super_set"]["sets"].append({
                "index": i,
                "aggregation": pset.aggregation,
                "predicate_weights": {
                    "vol_spike": weights[0],
                    "liq_lock": weights[1],
                    "mom_rev": weights[2],
                    "gap_gate": weights[3],
                    "gamma_gate": weights[4]
                }
            })

    # 3. Extract Output Head Focus (Sensitivities)
    if hasattr(model, 'output_head'):
        weights = model.output_head.weight.detach().cpu().numpy()
        spec = model.spec

        target_names = [
            'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
            'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target',
            'entry_target', 'exit_target'
        ]

        logic["output_head"]["sensitivities"] = {}
        for i, target in enumerate(target_names):
            if i >= weights.shape[0]:
                break
            w = weights[i]
            h_w = np.linalg.norm(w[:spec.d_h])
            v_w = np.linalg.norm(w[spec.d_h : spec.d_h+spec.d_v])
            m_w = np.linalg.norm(w[spec.d_h+spec.d_v : spec.d_h+spec.d_v+spec.d_m])
            r_w = np.linalg.norm(w[spec.d_h+spec.d_v+spec.d_m :])

            logic["output_head"]["sensitivities"][target] = {
                "market_physics_h": float(h_w),
                "portfolio_v": float(v_w),
                "momentum_m": float(m_w),
                "regime_r": float(r_w)
            }

    return logic


def generate_trading_rules(logic, max_super_sets=None, max_sets=None, max_comparisons=None):
    """Transcribes extracted logic into human-readable rules.

    Args:
        logic: Extracted logic dictionary
        max_super_sets: Max super_sets to show (None = all)
        max_sets: Max sets per super_set to show (None = all)
        max_comparisons: Max comparisons per set to show (None = all)
    """
    rules = []

    p = logic.get("predicates", {})
    if p:
        rules.append(f"RULE 1 (Vol): Enter if IVR > {p.get('iv_rank_threshold', 0.75):.2f}")
        rules.append(f"RULE 2 (Liq): Block if Spread/Price > {p.get('spread_ratio_threshold', 0.004):.4%}")
        rules.append(f"RULE 3 (Trend): Signal Reversal if RSI < {p.get('rsi_threshold', 25):.2f} and delta_RSI < 0")
        rules.append(f"RULE 4 (Gap): Guard if 1m price jump > {p.get('gap_fraction_threshold', 0.012):.2%}")
        rules.append(f"RULE 5 (Greeks): Hedge if |Gamma| > {p.get('gamma_threshold', 0.01):.4f}")
        # V3.0 rules
        if 'iv_regime_frac_threshold' in p:
            rules.append(f"RULE 6 (IV Regime): Alert if IV_Mid > IV_High * {p['iv_regime_frac_threshold']:.2f}")
        if 'put_flow_threshold' in p:
            rules.append(f"RULE 7 (Flow): Heavy Put if Put_Vol/Total_Vol > {p['put_flow_threshold']:.2f}")
        if 'spread_stress_mult_threshold' in p:
            rules.append(f"RULE 8 (Microstructure): Stress if quote_spread > {p['spread_stress_mult_threshold']:.1f}x median")

    # Analyze SuperSet (V21+ format)
    ss = logic.get("super_set", {})
    if "super_sets" in ss:
        rules.append("")
        rules.append("=" * 60)
        rules.append("LEARNED RELATIONAL LOGIC (Feature Comparisons)")
        rules.append("=" * 60)
        super_sets_list = ss["super_sets"][:max_super_sets] if max_super_sets else ss["super_sets"]
        for ss_info in super_sets_list:
            rules.append(f"\nSUPER_SET {ss_info['index']}: {ss_info['n_sets']} predicate logic sets")
            sets_list = ss_info.get("sets", [])[:max_sets] if max_sets else ss_info.get("sets", [])
            for set_info in sets_list:
                steepness = set_info.get('steepness', 10)

                if "operator_weights" in set_info:
                    ops = set_info["operator_weights"]
                    rules.append(f"  SET {set_info['index']}: (<:{ops['<']:.0f}% | >:{ops['>']:.0f}% | =:{ops['=']:.0f}%)")

                    if "top_comparisons" in set_info:
                        comps_list = set_info["top_comparisons"][:max_comparisons] if max_comparisons else set_info["top_comparisons"]
                        for comp in comps_list:
                            comparison_str = comp.get("comparison", f"Pair_{comp['pair_idx']}")
                            rules.append(f"    → {comparison_str}")
                else:
                    norm = set_info.get('projection_weight_norm', 0)
                    rules.append(f"  SET {set_info['index']}: weight_norm={norm:.3f}, steepness={steepness:.1f}")
    elif "sets" in ss:
        for s in ss["sets"]:
            if "predicate_weights" in s:
                top_pred = max(s["predicate_weights"].items(), key=lambda x: x[1])[0]
                rules.append(f"SET {s['index']} Focus: {top_pred} (Weight: {s['predicate_weights'][top_pred]:.2f})")

    return rules


def main():
    parser = argparse.ArgumentParser(
        description="CondorNet Interpretability Audit - Extract learned trading rules from checkpoints"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to CondorNet checkpoint (.pth)")
    parser.add_argument("--data", type=str, required=True, help="Path to data CSV (for feature names)")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to load (default: 5000)")
    parser.add_argument("--output-json", type=str, default="models/condor_logic.json", help="Output JSON path")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading Data: {args.data}")
    df = pd.read_csv(args.data, nrows=args.samples + SEQ_LEN + 100)
    X_raw, _, _, _, _ = (lambda x: (x, 0, 0, 0, 0))(df[FEATURE_COLS_V22].values.astype(np.float32))

    # 2. Load Model
    model, feature_cols, norm = load_condor_model(args.model)

    # 3. Extract Logic
    print("\n🧠 Extracting Learned Logic...")
    logic = extract_learned_logic(model, feature_cols)

    # 4. Generate Signal Summary
    print("\n📜 Learned Trading Rules & Signals:")
    rules = generate_trading_rules(logic)
    for r in rules:
        print(f"  * {r}")

    # 5. Output Sensitivities
    if logic["output_head"].get("sensitivities"):
        print("\n🎯 Neural Sensitivity (Target -> State Block):")
        for target, sens in logic["output_head"]["sensitivities"].items():
            primary = max(sens.items(), key=lambda x: x[1])[0]
            print(f"  - {target:<20}: Driven by {primary:<20} (score: {sens[primary]:.4f})")

    # 6. Save to JSON
    logic["rules"] = rules
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(logic, f, indent=4)
    print(f"\n✅ Logic exported to {args.output_json}")


if __name__ == "__main__":
    main()
