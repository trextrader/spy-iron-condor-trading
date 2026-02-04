import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import sys
import os
import json
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.getcwd())

from intelligence.condor_brain_net import CondorNet
from intelligence.canonical_feature_registry import FEATURE_COLS_V22, select_feature_frame

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 240 # CondorNet standard
EPS = 1e-6

def safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def load_condor_model(ckpt_path):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    sd = ckpt['state_dims']
    feature_cols = ckpt.get('feature_cols', FEATURE_COLS_V22)
    
    # V24: Read hyperparameters from checkpoint to match saved architecture
    hp = ckpt.get('hyperparameters', {})
    n_predicates = hp.get('n_predicates', 32)
    n_sets = hp.get('n_sets', 16)
    n_super_sets = hp.get('n_super_sets', 8)
    
    print(f"  Architecture: d_h={sd['d_h']}, d_v={sd['d_v']}, d_m={sd['d_m']}, d_r={sd['d_r']}")
    print(f"  Logic: n_predicates={n_predicates}, n_sets={n_sets}, n_super_sets={n_super_sets}")
    
    model = CondorNet(
        d_input=len(feature_cols),
        d_h=sd['d_h'], d_v=sd['d_v'], d_m=sd['d_m'], d_r=sd['d_r'],
        n_predicates=n_predicates,
        n_sets=n_sets,
        n_super_sets=n_super_sets,
    )
    
    model.load_state_dict(ckpt['model_state_dict'])
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

    # 1. Extract Predicate Thresholds
    if hasattr(model, 'pred_gates'):
        pg = model.pred_gates
        logic["predicates"] = {
            "iv_rank_threshold": float(pg.iv_rank_thresh.data),
            "spread_ratio_threshold": float(pg.spread_frac_thresh.data),
            "rsi_threshold": float(pg.rsi_thresh.data),
            "gap_fraction_threshold": float(pg.gap_frac_thresh.data),
            "gamma_threshold": float(pg.gamma_thresh.data),
            "steepness": pg.steepness
        }

    # 2. Extract SuperSet Logic (V21+: super_sets is a ModuleList)
    if hasattr(model, 'super_sets') and len(model.super_sets) > 0:
        logic["super_set"]["n_super_sets"] = len(model.super_sets)
        logic["super_set"]["super_sets"] = []
        for ss_idx, ss in enumerate(model.super_sets):
            ss_info = {
                "index": ss_idx,
                "n_sets": len(ss.sets) if hasattr(ss, 'sets') else 0,
                "sets": []
            }
            if hasattr(ss, 'sets'):
                for i, pset in enumerate(ss.sets):
                    set_info = {"index": i}
                    # V21+: PredicateSet uses relational_logic
                    if hasattr(pset, 'relational_logic'):
                        rl = pset.relational_logic
                        set_info["steepness"] = float(rl.steepness.data) if hasattr(rl, 'steepness') else 10.0
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
        # Weight matrix: (10, d_x)
        # d_x is split into (h, v, m, r)
        weights = model.output_head.weight.detach().cpu().numpy() # (10, d_x)
        spec = model.spec
        
        target_names = [
            'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
            'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target',
            'entry_target', 'exit_target'
        ]
        
        logic["output_head"]["sensitivities"] = {}
        for i, target in enumerate(target_names):
            w = weights[i]
            # split into block norm magnitudes
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

def generate_trading_rules(logic):
    """Transcribes extracted logic into human-readable rules."""
    rules = []
    
    p = logic.get("predicates", {})
    if p:
        rules.append(f"RULE 1 (Vol): Enter if IVR > {p.get('iv_rank_threshold', 0.75):.2f}")
        rules.append(f"RULE 2 (Liq): Block if Spread/Price > {p.get('spread_ratio_threshold', 0.004):.4%}")
        rules.append(f"RULE 3 (Trend): Signal Reversal if RSI < {p.get('rsi_threshold', 25):.2f} and delta_RSI < 0")
        rules.append(f"RULE 4 (Gap): Guard if 1m price jump > {p.get('gap_fraction_threshold', 0.012):.2%}")
        rules.append(f"RULE 5 (Greeks): Hedge if |Gamma| > {p.get('gamma_threshold', 0.01):.4f}")
    
    # Analyze SuperSet (V21+ format)
    ss = logic.get("super_set", {})
    if "super_sets" in ss:
        for ss_info in ss["super_sets"]:
            rules.append(f"SUPER_SET {ss_info['index']}: {ss_info['n_sets']} logic sets")
            for set_info in ss_info.get("sets", [])[:3]:  # Show first 3 sets per super-set
                norm = set_info.get('projection_weight_norm', 0)
                steepness = set_info.get('steepness', 10)
                rules.append(f"  └─ SET {set_info['index']}: weight_norm={norm:.3f}, steepness={steepness:.1f}")
    # Legacy format
    elif "sets" in ss:
        for s in ss["sets"]:
            if "predicate_weights" in s:
                top_pred = max(s["predicate_weights"].items(), key=lambda x: x[1])[0]
                rules.append(f"SET {s['index']} Focus: {top_pred} (Weight: {s['predicate_weights'][top_pred]:.2f})")
        
    return rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output-json", type=str, default="models/condor_logic.json")
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
    print("\n🎯 Neural Sensitivity (Target -> State Block):")
    for target, sens in logic["output_head"]["sensitivities"].items():
        primary = max(sens.items(), key=lambda x: x[1])[0]
        print(f"  - {target:<20}: Driven by {primary:<20} (score: {sens[primary]:.4f})")
        
    # 6. Save to JSON
    logic["rules"] = rules
    with open(args.output_json, 'w') as f:
        json.dump(logic, f, indent=4)
    print(f"\n✅ Logic exported to {args.output_json}")

if __name__ == "__main__":
    main()
