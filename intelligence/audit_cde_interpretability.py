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
    
    model = CondorNet(
        d_input=len(feature_cols),
        d_h=sd['d_h'], d_v=sd['d_v'], d_m=sd['d_m'], d_r=sd['d_r'],
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

    # 2. Extract SuperSet Logic
    if hasattr(model, 'super_set'):
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
    
    p = logic["predicates"]
    rules.append(f"RULE 1 (Vol): Enter if IVR > {p['iv_rank_threshold']:.2f}")
    rules.append(f"RULE 2 (Liq): Block if Spread/Price > {p['spread_ratio_threshold']:.4%}")
    rules.append(f"RULE 3 (Trend): Signal Reversal if RSI < {p['rsi_threshold']:.2f} and delta_RSI < 0")
    rules.append(f"RULE 4 (Gap): Guard if 1m price jump > {p['gap_fraction_threshold']:.2%}")
    rules.append(f"RULE 5 (Greeks): Hedge if |Gamma| > {p['gamma_threshold']:.4f}")
    
    # Analyze SuperSet
    ss = logic["super_set"]
    for s in ss["sets"]:
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
