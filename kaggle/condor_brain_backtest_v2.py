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
    print("=" * 80)
    print("TRADE DECISION LOG (First 50 bars after warmup)")
    print("=" * 80)
    
    # Open log file for writing
    log_file = open("trade_decisions.log", "w")
    log_file.write("=" * 80 + "\n")
    log_file.write("TRADE DECISION LOG (ALL BARS)\n")
    log_file.write("=" * 80 + "\n\n")
    
    logged_count = 0
    MAX_LOGS = 50  # Console only
    
    # Start from SEQ_LEN
    for i in tqdm(range(SEQ_LEN, len(df) - 1)):
        # 1. State
        if i >= len(X_tensor): break
        x_seq = X_tensor[i-SEQ_LEN : i].unsqueeze(0) # [1, 256, 52]
        
        # 2. Model Inference
        with torch.no_grad():
            outputs = model(x_seq)
            
        pol = outputs[0].cpu().numpy().flatten()
        
        # Extract ALL policy outputs for logging
        # Policy head: [call_off, put_off, width, te, prob_profit, stop_mult, direction, confidence]
        all_outputs = {
            'call_off': pol[0] if len(pol) > 0 else None,
            'put_off': pol[1] if len(pol) > 1 else None,
            'width': pol[2] if len(pol) > 2 else None,
            'te': pol[3] if len(pol) > 3 else None,
            'prob_profit': pol[4] if len(pol) > 4 else None,
            'stop_mult': pol[5] if len(pol) > 5 else None,
            'direction': pol[6] if len(pol) > 6 else None,
            'confidence': pol[7] if len(pol) > 7 else None,
        }
        
        prob_profit = all_outputs['prob_profit'] or 0
        confidence = all_outputs['confidence'] or 0
        
        # 3. Rule Signal Check
        net_rule_signal = 0
        if rule_signals is not None:
            net_rule_signal = rule_signals.iloc[i].sum()
        
        # 4. Decision Logic with explicit reasoning
        action = 0
        rejection_reason = None
        
        if position == 0:
            # Check entry conditions
            conf_check = confidence > 0.3
            prob_check = prob_profit > 0.3
            rule_check = net_rule_signal >= 0
            
            if conf_check or prob_check:
                if rule_check:
                    action = 1
                    spot = df['close'].iloc[i]
                    trade_num = len(trades) + 1
                    
                    # TRADE STATS - Print immediately
                    trade_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🔔 TRADE #{trade_num}: ENTER LONG @ Bar {i}
╠══════════════════════════════════════════════════════════════════════════════╣
║ Spot Price:    ${spot:.2f}
║ Confidence:    {confidence:.4f}  (threshold: 0.3)
║ Prob Profit:   {prob_profit:.4f}  (threshold: 0.3)
║ Rule Signal:   {net_rule_signal:.2f}  (must be >= 0)
║ All Pol Out:   {[f'{x:.4f}' for x in pol[:8]]}
╚══════════════════════════════════════════════════════════════════════════════╝"""
                    print(trade_msg)
                    log_file.write(trade_msg + "\n")
                    
                    trades.append({
                        'idx': i, 
                        'type': 'LONG', 
                        'price': spot, 
                        'conf': float(confidence), 
                        'prob': float(prob_profit),
                        'rules': float(net_rule_signal)
                    })
                    position = 1
                else:
                    rejection_reason = f"RULES_BEARISH (signal={net_rule_signal:.2f})"
            else:
                rejection_reason = f"LOW_CONFIDENCE (conf={confidence:.4f}, prob={prob_profit:.4f})"
        
        elif position == 1:
            # Exit logic
            if confidence < 0.3 or net_rule_signal < 0:
                action = -1
                spot = df['close'].iloc[i]
                exit_reason = "Low Confidence" if confidence < 0.3 else "Bearish Rules"
                
                # EXIT STATS - Print immediately
                exit_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🔔 TRADE EXIT: Close LONG @ Bar {i}
╠══════════════════════════════════════════════════════════════════════════════╣
║ Spot Price:    ${spot:.2f}
║ Exit Reason:   {exit_reason}
║ Confidence:    {confidence:.4f}
║ Rule Signal:   {net_rule_signal:.2f}
╚══════════════════════════════════════════════════════════════════════════════╝"""
                print(exit_msg)
                log_file.write(exit_msg + "\n")
                
                trades.append({'idx': i, 'type': 'EXIT_LONG', 'price': spot, 'pnl': 0}) 
                position = 0
        
        # LOG: ALL bars to file, first N to console
        spot = df['close'].iloc[i]
        log_lines = [
            f"\n--- Bar {i} | Spot: ${spot:.2f} | Position: {position} ---",
            f"  Model Outputs: {pol[:8]}",
            f"  Confidence: {confidence:.4f} | Prob_Profit: {prob_profit:.4f}",
            f"  Rule Signal: {net_rule_signal:.2f}",
        ]
        if action == 1:
            log_lines.append(f"  >> ACTION: ENTER LONG")
        elif action == -1:
            log_lines.append(f"  >> ACTION: EXIT LONG")
        elif rejection_reason:
            log_lines.append(f"  >> REJECTED: {rejection_reason}")
        else:
            log_lines.append(f"  >> NO ACTION")
        
        # Write to file (ALL bars)
        for line in log_lines:
            log_file.write(line + "\n")
        
        # Print to console (first 50 only)
        if logged_count < MAX_LOGS:
            for line in log_lines:
                print(line)
            logged_count += 1
            
        # 5. PnL Simulation
        ret = df['close'].iloc[i+1] / df['close'].iloc[i] - 1.0
        
        if position == 1:
            capital *= (1.0 + ret)
        
        equity_curve.append(capital)
    
    # Close log file
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write(f"LOG END. Total Trades: {len(trades)}\n")
    log_file.write("=" * 80 + "\n")
    log_file.close()
    
    print("=" * 80)
    print(f"LOG END. Total Trades: {len(trades)}")
    print(f"Full log saved to: trade_decisions.log")
    print("=" * 80)
        
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


