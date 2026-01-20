"""
CondorBrain V2.2 Backtest (with Rule Engine)
============================================
Integrates CondorBrain V2.2 Model (52 features) with Rule Engine V2.5.
Vectorized execution for Rules + Model Inference.

Usage:
1. Upload to Kaggle/Colab.
2. Ensure `condor_brain_retrain_v22_e3.pth` (or similar) is present.
3. Ensure `intelligence/` and `docs/Complete_Ruleset_DSL.yaml` are present.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add repo to path
# 0. Setup Path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
except NameError:
    pass

sys.path.insert(0, '/content/spy-iron-condor-trading')
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')
sys.path.insert(0, os.getcwd())

from intelligence.condor_brain import CondorBrain
from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V22, INPUT_DIM_V22, VERSION_V22
)
from intelligence.features.dynamic_features import (
    compute_all_dynamic_features,
    compute_all_primitive_features_v22
)
from intelligence.rule_engine.dsl_parser import RuleDSLParser
from intelligence.rule_engine.executor import RuleExecutionEngine

# --- CONFIG ---
MODEL_PATH = "condor_brain_retrain_v22_e3.pth" # Default
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
RULESET_PATH = "docs/Complete_Ruleset_DSL.yaml"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_and_features(data_path, rows=None):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if rows is not None:
        df = df.iloc[-rows:].reset_index(drop=True)
    
    # 1. Base Features (V2.1)
    print("Computing V2.1 Dynamic Features...")
    df = compute_all_dynamic_features(df, close_col="close", high_col="high", low_col="low")
    
    # 2. Primitive Features (V2.2)
    print("Computing V2.2 Primitive Features...")
    # Ensure Spread/Lag cols exist (or fallback handled in script)
    df = compute_all_primitive_features_v22(
        df,
        close_col="close", high_col="high", low_col="low",
        volume_col="volume",
        spread_col="spread_ratio" if "spread_ratio" in df.columns else "close",
        inplace=True
    )
    
    return df

def run_rule_engine(df, ruleset_path):
    print(f"Initializing Rule Engine from {ruleset_path}...")
    if not os.path.exists(ruleset_path):
        # Try local path variations
        if os.path.exists(f"spy-iron-condor-trading/{ruleset_path}"):
            ruleset_path = f"spy-iron-condor-trading/{ruleset_path}"
        else:
            print(f"⚠️ Warning: Ruleset not found at {ruleset_path}. Skipping rules.")
            return df, None
            
    parser = RuleDSLParser(ruleset_path)
    try:
        ruleset = parser.load()
    except Exception as e:
        print(f"❌ Error loading ruleset: {e}")
        return df, None
        
    engine = RuleExecutionEngine(ruleset)
    
    print("Executing Rules Vectorized...")
    # Engine requires Dict[str, Series]. df matches this interface (mostly).
    # We pass 'data' as df (which behaves like dict of Series).
    
    # NOTE: Engine.execute() might expect a dict of Series explicitly if strict.
    # But df keys() works.
    
    # To be safe, let's pass df (it has __getitem__).
    # Engine logic: self._compute_primitives(data) -> data[p_spec.inputs]
    
    results = engine.execute(df)
    
    # Parse Results into DataFrame Columns
    # results is Dict[rule_id, {'signals': Series, 'blocked': Series}]
    
    rule_signals = pd.DataFrame(index=df.index)
    
    for rule_id, rule in ruleset.rules.items():
        r_res = results.get(rule_id)
        if r_res is None:
            continue
        
        # Entry Signal (1=Long, -1=Short, 0=None)
        long_s = r_res.get('entry_long', pd.Series(False, index=df.index))
        short_s = r_res.get('entry_short', pd.Series(False, index=df.index))
        
        if hasattr(long_s, 'fillna'):
            long_s = long_s.fillna(False).astype(int)
        else:
            long_s = pd.Series(0, index=df.index)
        if hasattr(short_s, 'fillna'):
            short_s = short_s.fillna(False).astype(int)
        else:
            short_s = pd.Series(0, index=df.index)
        
        # Combine: Long=1, Short=-1
        sig = long_s - short_s
        rule_signals[f"{rule_id}_signal"] = sig
            
    print(f"Generated signals for {len(ruleset.rules)} rules.")
    return df, rule_signals


def run_backtest(df, rule_signals, model, feature_cols, device):
    print("Starting Backtest Simulation...")
    
    # Pre-process Features (Robust Norm same as training)
    X_np = df[feature_cols].values.astype(np.float32)
    X_np = np.nan_to_num(X_np, nan=0.0)
    
    # Normalize (approximate robust norm)
    mu = np.median(X_np, axis=0) if len(X_np) > 0 else 0
    mad = np.median(np.abs(X_np - mu), axis=0)
    mad = np.maximum(mad, 1e-6)
    
    X_norm = (X_np - mu) / (1.4826 * mad)
    X_norm = np.clip(X_norm, -10.0, 10.0)
    
    X_tensor = torch.tensor(X_norm, device=device)
    
    # Settings
    SEQ_LEN = 256
    capital = 100_000.0
    position = 0 # 0=None, 1=Long, -1=Short
    equity_curve = []
    trades = []
    
    model.eval()
    
    # Iterate
    print(f"Simulating {len(df)} bars...")
    # Start from SEQ_LEN
    for i in tqdm(range(SEQ_LEN, len(df) - 1)):
        # 1. State
        if i >= len(X_tensor): break
        x_seq = X_tensor[i-SEQ_LEN : i].unsqueeze(0) # [1, 256, 52]
        
        # 2. Model Inference
        with torch.no_grad():
            outputs = model(x_seq)
            
        pol = outputs[0].cpu().numpy()
        # [call_off, put_off, width, ..., prob_profit, ..., confidence]
        prob_profit = pol[0, 4]
        confidence = pol[0, 7]
        
        # DEBUG: Print model outputs for first 10 bars
        if i < SEQ_LEN + 10:
            print(f"Bar {i}: conf={confidence:.4f}, prob_profit={prob_profit:.4f}, rule_signal={rule_signals.iloc[i].sum() if rule_signals is not None else 0}")
        
        # 3. Rule Signal Check (Look up pre-computed signal)
        action = 0
        net_rule_signal = 0
        if rule_signals is not None:
             # Sum of signals for this bar
             # rule_signals is aligned with df. Index i matches.
             net_rule_signal = rule_signals.iloc[i].sum()
        
        if position == 0:
            # Hybrid Entry: Lower thresholds for testing
            # Original: confidence > 0.6 and prob_profit > 0.6
            if confidence > 0.3 or prob_profit > 0.3:  # Relaxed for testing
                if net_rule_signal >= 0: 
                    action = 1
                    trades.append({'idx': i, 'type': 'LONG', 'price': df['close'].iloc[i], 'conf': float(confidence), 'rules': float(net_rule_signal)})
                    position = 1
        
        elif position == 1:
            # Hybrid Exit: Model Loss of Confidence OR Rule Bearish Signal
            if confidence < 0.3 or net_rule_signal < 0:
                action = -1
                trades.append({'idx': i, 'type': 'EXIT_LONG', 'price': df['close'].iloc[i], 'pnl': 0}) 
                position = 0
            
        # 4. Simulation (PnL - Simplified: Underlying Return)
        # Using Underlying Return is proxy for Delta 1 Long.
        # Condor is delta neutral. This simulation logic needs refinement for Options PnL.
        # But for Signal Verification, this suffices.
        ret = df['close'].iloc[i+1] / df['close'].iloc[i] - 1.0
        
        if position == 1:
            capital *= (1.0 + ret)
        
        equity_curve.append(capital)
        
    return equity_curve, trades

def main():
    # 0. Data Path Detection
    possible_data_paths = [
        MODEL_PATH, # Kaggle default config (Line 42 might be overwritten)
        "/kaggle/input/spy-options-data/mamba_institutional_1m.csv",
        "/content/spy-iron-condor-trading/data/processed/mamba_institutional_1m.csv",
        "/content/spy-iron-condor-trading/data/mamba_institutional_1m_500k.csv",
        "data/processed/mamba_institutional_1m.csv",
        "data/mamba_institutional_1m_500k.csv",
        "mamba_institutional_1m.csv",
        "mamba_institutional_1m_500k.csv"
    ]
    use_data_path = DATA_PATH
    for p in possible_data_paths:
        if os.path.exists(p):
            use_data_path = p
            print(f"Found data at: {use_data_path}")
            break
            
    # 1. Pipeline
    # Load limited rows for test? Or all.
    df = load_data_and_features(use_data_path)  # Load ALL rows
    
    # 2. Rules
    df, rule_signals = run_rule_engine(df, RULESET_PATH)
    
    # 3. Model
    POSSIBLE_PATHS = [
        "condor_brain_retrain_v22_e3.pth", # 🚀 Final Model
        "condor_brain_retrain_v22_e2.pth", 
        "condor_brain_retrain_v22_e1.pth", 
        "condor_brain_retrain_e3.pth",
        "/kaggle/working/condor_brain_retrain_e3.pth",
        "/kaggle/working/condor_brain_retrain_v22_e3.pth", # Colab E3
        "/kaggle/working/condor_brain_retrain_v22_e1.pth", 
        "/kaggle/working/condor_brain_retrain_v22_e2.pth"
    ]
    model_path = MODEL_PATH
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            model_path = p
            break
            
    print(f"Loading Model from {model_path}...")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        
        # V2.2 Model
        model = CondorBrain(
            d_model=512, n_layers=12,
            input_dim=INPUT_DIM_V22, # 52
            use_vol_gated_attn=True, use_topk_moe=True, moe_n_experts=3, moe_k=1,
            use_diffusion=True
        ).to(DEVICE)
        
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded.")
        except Exception as e:
            print(f"Model load failed: {e}")
            return
            
        # 4. Backtest
        equity, trades = run_backtest(df, rule_signals, model, FEATURE_COLS_V22, DEVICE)
        
        # 5. Report
        print(f"Final Capital: ${equity[-1]:,.2f}")
        print(f"Trades: {len(trades)}")
        plt.figure(figsize=(10, 6))
        plt.plot(equity)
        plt.title(f"Equity Curve (Hybrid V2.2) - {len(trades)} Trades")
        plt.xlabel("Bars")
        plt.ylabel("Capital")
        plt.grid(True)
        plt.savefig("backtest_v2_result.png")
        print("Saved plot to backtest_v2_result.png")
        
        # Save Trades CSV
        if trades:
            pd.DataFrame(trades).to_csv("trades_v2.csv", index=False)
            print("Saved trades to trades_v2.csv")
        
    else:
        print(f"Model not found at {model_path}.")

if __name__ == "__main__":
    main()


