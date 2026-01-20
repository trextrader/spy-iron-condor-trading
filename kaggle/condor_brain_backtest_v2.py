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

# Output directory
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Iron Condor P&L Config
IC_CREDIT_PER_SPREAD = 1.50  # $1.50 credit per spread (typical)
IC_CONTRACTS = 10  # Number of contracts per trade
IC_MULTIPLIER = 100  # Options multiplier

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
    position = 0 # 0=None, 1=Long Iron Condor
    equity_curve = []
    trades = []
    
    # Trade state tracking (for expiration)
    trade_entry_bar = None
    trade_dte = None  # Days to expiration
    BARS_PER_DAY = 390  # 1-min bars per trading day (6.5 hours)
    DEFAULT_DTE = 14  # Default 14 DTE if model output is invalid
    
    model.eval()
    
    # Iterate
    print(f"Simulating {len(df)} bars...")
    print("=" * 80)
    print("TRADE DECISION LOG (First 50 bars after warmup)")
    print("=" * 80)
    
    # Open log file for writing
    log_file = open(os.path.join(REPORTS_DIR, "trade_decisions.log"), "w")
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
        
        # 4. FUZZY LOGIC Entry Decision
        # Score-based entry: accumulate evidence from multiple sources
        # This allows entry even if not ALL conditions are perfect
        action = 0
        rejection_reason = None
        
        if position == 0:
            # Calculate fuzzy entry score (0-100)
            entry_score = 0
            entry_factors = []
            
            # Factor 1: Model Confidence (0-30 points)
            if confidence > 0.7:
                entry_score += 30
                entry_factors.append(f"Confidence HIGH ({confidence:.2f}) +30")
            elif confidence > 0.5:
                entry_score += 20
                entry_factors.append(f"Confidence MED ({confidence:.2f}) +20")
            elif confidence > 0.3:
                entry_score += 10
                entry_factors.append(f"Confidence LOW ({confidence:.2f}) +10")
            else:
                entry_factors.append(f"Confidence WEAK ({confidence:.2f}) +0")
            
            # Factor 2: Probability of Profit (0-30 points)
            if prob_profit > 0.6:
                entry_score += 30
                entry_factors.append(f"ProbProfit HIGH ({prob_profit:.2f}) +30")
            elif prob_profit > 0.45:
                entry_score += 20
                entry_factors.append(f"ProbProfit MED ({prob_profit:.2f}) +20")
            elif prob_profit > 0.3:
                entry_score += 10
                entry_factors.append(f"ProbProfit LOW ({prob_profit:.2f}) +10")
            else:
                entry_factors.append(f"ProbProfit WEAK ({prob_profit:.2f}) +0")
            
            # Factor 3: Rule Engine Signal (0-30 points)
            if net_rule_signal > 0.5:
                entry_score += 30
                entry_factors.append(f"Rules BULLISH ({net_rule_signal:.2f}) +30")
            elif net_rule_signal >= 0:
                entry_score += 15
                entry_factors.append(f"Rules NEUTRAL ({net_rule_signal:.2f}) +15")
            else:
                entry_factors.append(f"Rules BEARISH ({net_rule_signal:.2f}) +0")
            
            # Factor 4: Direction alignment (0-10 points)
            direction = all_outputs['direction'] or 0
            if abs(direction) < 0.5:  # Iron Condor likes neutral
                entry_score += 10
                entry_factors.append(f"Direction NEUTRAL ({direction:.2f}) +10")
            else:
                entry_factors.append(f"Direction BIASED ({direction:.2f}) +0")
            
            # Entry threshold: 40+ points (fuzzy, not all-or-nothing)
            ENTRY_THRESHOLD = 40
            
            if entry_score >= ENTRY_THRESHOLD:
                action = 1
                spot = df['close'].iloc[i]
                trade_num = len([t for t in trades if t.get('action') == 'OPEN']) + 1
                
                # Extract OPTIONS parameters from policy head
                call_offset = all_outputs['call_off'] or 0
                put_offset = all_outputs['put_off'] or 0
                width = all_outputs['width'] or 5
                te_suggested = all_outputs['te'] or 30
                direction = all_outputs['direction'] or 0
                
                # Compute suggested strikes (offset from ATM)
                short_call_strike = spot + (call_offset * spot * 0.01)  # ~1% per unit
                long_call_strike = short_call_strike + width
                short_put_strike = spot - (put_offset * spot * 0.01)
                long_put_strike = short_put_strike - width
                
                # Calculate Iron Condor credit (max profit)
                credit_received = IC_CREDIT_PER_SPREAD * IC_CONTRACTS * IC_MULTIPLIER
                max_loss = (width - IC_CREDIT_PER_SPREAD) * IC_CONTRACTS * IC_MULTIPLIER
                
                # TRADE STATS with fuzzy score breakdown
                factors_str = "\\n║   ".join(entry_factors)
                trade_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🦅 IRON CONDOR #{trade_num} ENTRY @ Bar {i}
╠══════════════════════════════════════════════════════════════════════════════╣
║ SPOT:          ${spot:.2f}
║ FUZZY SCORE:   {entry_score}/100 (threshold: {ENTRY_THRESHOLD})
╠─────────────────────────── DECISION FACTORS ─────────────────────────────────╣
║   {factors_str}
╠─────────────────────────── IRON CONDOR SETUP ────────────────────────────────╣
║   Short Call:  ${short_call_strike:.2f}  |  Long Call:  ${long_call_strike:.2f}
║   Short Put:   ${short_put_strike:.2f}  |  Long Put:   ${long_put_strike:.2f}
║   Width:       ${width:.2f}  |  DTE: {te_suggested:.0f} days
╠─────────────────────────── P&L POTENTIAL ────────────────────────────────────╣
║   Credit:      ${credit_received:,.2f} (Max Profit)
║   Max Loss:    ${max_loss:,.2f}
║   Contracts:   {IC_CONTRACTS}
╚══════════════════════════════════════════════════════════════════════════════╝"""
                print(trade_msg)
                log_file.write(trade_msg + "\n")
                
                # Set DTE for expiration tracking
                trade_entry_bar = i
                trade_dte = te_suggested if te_suggested > 0 else DEFAULT_DTE
                trade_credit = credit_received  # Store for P&L calc
                trade_max_loss = max_loss
                
                trades.append({
                    'idx': i, 
                    'type': 'IRON_CONDOR', 
                    'action': 'OPEN',
                    'spot': spot,
                    'short_call': short_call_strike,
                    'long_call': long_call_strike,
                    'short_put': short_put_strike,
                    'long_put': long_put_strike,
                    'width': width,
                    'dte': trade_dte,
                    'credit': credit_received,
                    'max_loss': max_loss,
                    'entry_score': entry_score,
                    'conf': float(confidence), 
                    'prob': float(prob_profit),
                    'rules': float(net_rule_signal)
                })
                position = 1
            else:
                # Rejection: entry score below threshold
                rejection_reason = f"SCORE_TOO_LOW ({entry_score}/{ENTRY_THRESHOLD})"
                rejection_factors = entry_factors
        
        elif position == 1:
            # Calculate remaining DTE
            bars_held = i - trade_entry_bar if trade_entry_bar else 0
            days_held = bars_held / BARS_PER_DAY
            remaining_dte = trade_dte - days_held if trade_dte else 0
            
            # Exit conditions: 1) Expiration, 2) Low confidence, 3) Bearish rules
            exit_reason = None
            if remaining_dte <= 0:
                exit_reason = f"EXPIRATION (DTE={remaining_dte:.1f})"
            elif confidence < 0.3:
                exit_reason = f"Low Confidence ({confidence:.4f})"
            elif net_rule_signal < 0:
                exit_reason = f"Bearish Rules ({net_rule_signal:.2f})"
            
            if exit_reason:
                action = -1
                spot = df['close'].iloc[i]
                
                # Calculate simple P&L (spot change proxy)
                entry_spot = trades[-1]['spot'] if trades and 'spot' in trades[-1] else spot
                pnl_pct = (spot - entry_spot) / entry_spot * 100
                
                # EXIT STATS - Print immediately
                exit_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🔔 IRON CONDOR EXIT @ Bar {i}
╠══════════════════════════════════════════════════════════════════════════════╣
║ Spot Price:    ${spot:.2f} (Entry: ${entry_spot:.2f})
║ Exit Reason:   {exit_reason}
║ Days Held:     {days_held:.1f}
║ Spot Change:   {pnl_pct:+.2f}%
║ Confidence:    {confidence:.4f}
║ Rule Signal:   {net_rule_signal:.2f}
╚══════════════════════════════════════════════════════════════════════════════╝"""
                print(exit_msg)
                log_file.write(exit_msg + "\n")
                
                trades.append({
                    'idx': i, 
                    'type': 'IRON_CONDOR',
                    'action': 'CLOSE',
                    'spot': spot,
                    'reason': exit_reason,
                    'days_held': days_held,
                    'pnl_pct': pnl_pct
                }) 
                position = 0
                trade_entry_bar = None
                trade_dte = None
        
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
            
        # 5. Iron Condor P&L Simulation
        # IC P&L: Collect theta (time decay) while in position, lose if breached
        spot = df['close'].iloc[i]
        
        if position == 1 and trade_entry_bar is not None:
            # Get open trade data
            open_trade = [t for t in trades if t.get('action') == 'OPEN'][-1] if trades else None
            if open_trade:
                short_call = open_trade.get('short_call', spot + 10)
                short_put = open_trade.get('short_put', spot - 10)
                credit = open_trade.get('credit', IC_CREDIT_PER_SPREAD * IC_CONTRACTS * IC_MULTIPLIER)
                max_loss = open_trade.get('max_loss', 5 * IC_CONTRACTS * IC_MULTIPLIER)
                
                # Check if breached (spot outside short strikes)
                if spot >= short_call:
                    # Call side breached - full loss on call spread
                    daily_pnl = -max_loss / (trade_dte * BARS_PER_DAY) if trade_dte else 0
                elif spot <= short_put:
                    # Put side breached - full loss on put spread  
                    daily_pnl = -max_loss / (trade_dte * BARS_PER_DAY) if trade_dte else 0
                else:
                    # Safe zone - collect theta (proportional time decay)
                    bars_remaining = (trade_dte * BARS_PER_DAY) - (i - trade_entry_bar)
                    if bars_remaining > 0:
                        daily_pnl = credit / (trade_dte * BARS_PER_DAY)  # Steady theta collection
                    else:
                        daily_pnl = 0
                
                capital += daily_pnl
        
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
        # Colab paths (prioritize the 500k trained model)
        "/content/spy-iron-condor-trading/condor_brain_retrain_e3+500k.pth",  # 🚀 500K Trained Model
        "/content/spy-iron-condor-trading/condor_brain_retrain_e3.pth",
        "condor_brain_retrain_e3+500k.pth",
        "condor_brain_retrain_v22_e3.pth",
        "condor_brain_retrain_v22_e2.pth", 
        "condor_brain_retrain_v22_e1.pth", 
        "condor_brain_retrain_e3.pth",
        "/kaggle/working/condor_brain_retrain_e3.pth",
        "/kaggle/working/condor_brain_retrain_v22_e3.pth",
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
        plt.figure(figsize=(12, 6))
        plt.plot(equity)
        plt.title(f"Equity Curve (Iron Condor V2.2) - {len(trades)} Trades")
        plt.xlabel("Bars")
        plt.ylabel("Capital ($)")
        plt.grid(True)
        plot_path = os.path.join(REPORTS_DIR, "backtest_v2_result.png")
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        
        # Save Trades CSV
        if trades:
            csv_path = os.path.join(REPORTS_DIR, "trades_v2.csv")
            pd.DataFrame(trades).to_csv(csv_path, index=False)
            print(f"Saved trades to {csv_path}")
            
            # 6. FACTOR ATTRIBUTION ANALYSIS
            print("\n" + "=" * 80)
            print("FACTOR ATTRIBUTION ANALYSIS")
            print("=" * 80)
            
            # Separate opens and closes
            opens = [t for t in trades if t.get('action') == 'OPEN']
            closes = [t for t in trades if t.get('action') == 'CLOSE']
            
            # Match opens with closes to get P&L
            trade_results = []
            for i, open_t in enumerate(opens):
                if i < len(closes):
                    close_t = closes[i]
                    # Simple P&L: if spot stayed inside strikes, profitable
                    entry_spot = open_t.get('spot', 0)
                    exit_spot = close_t.get('spot', 0)
                    short_call = open_t.get('short_call', entry_spot + 10)
                    short_put = open_t.get('short_put', entry_spot - 10)
                    
                    # Profitable if spot stayed within short strikes throughout
                    profitable = short_put < exit_spot < short_call
                    
                    trade_results.append({
                        'trade_num': i + 1,
                        'entry_score': open_t.get('entry_score', 0),
                        'conf': open_t.get('conf', 0),
                        'prob': open_t.get('prob', 0),
                        'rules': open_t.get('rules', 0),
                        'profitable': profitable,
                        'reason': close_t.get('reason', 'Unknown')
                    })
            
            if trade_results:
                results_df = pd.DataFrame(trade_results)
                winners = results_df[results_df['profitable'] == True]
                losers = results_df[results_df['profitable'] == False]
                
                print(f"\n📊 TRADE SUMMARY:")
                print(f"   Total Trades: {len(trade_results)}")
                print(f"   Winners: {len(winners)} ({100*len(winners)/len(trade_results):.1f}%)")
                print(f"   Losers: {len(losers)} ({100*len(losers)/len(trade_results):.1f}%)")
                
                print(f"\n📈 WINNING TRADES - Factor Averages:")
                if not winners.empty:
                    print(f"   Avg Entry Score: {winners['entry_score'].mean():.1f}")
                    print(f"   Avg Confidence:  {winners['conf'].mean():.4f}")
                    print(f"   Avg Prob Profit: {winners['prob'].mean():.4f}")
                    print(f"   Avg Rule Signal: {winners['rules'].mean():.2f}")
                else:
                    print("   (No winning trades)")
                
                print(f"\n📉 LOSING TRADES - Factor Averages:")
                if not losers.empty:
                    print(f"   Avg Entry Score: {losers['entry_score'].mean():.1f}")
                    print(f"   Avg Confidence:  {losers['conf'].mean():.4f}")
                    print(f"   Avg Prob Profit: {losers['prob'].mean():.4f}")
                    print(f"   Avg Rule Signal: {losers['rules'].mean():.2f}")
                else:
                    print("   (No losing trades)")
                
                # Save factor analysis
                analysis_path = os.path.join(REPORTS_DIR, "factor_attribution.csv")
                results_df.to_csv(analysis_path, index=False)
                print(f"\n💾 Saved factor attribution to {analysis_path}")
                
                # Key Insights
                print("\n🔑 KEY INSIGHTS:")
                if not winners.empty and not losers.empty:
                    if winners['entry_score'].mean() > losers['entry_score'].mean():
                        print("   ✅ Higher entry scores correlate with winning trades")
                    if winners['conf'].mean() > losers['conf'].mean():
                        print("   ✅ Higher model confidence correlates with winning trades")
                    if winners['prob'].mean() > losers['prob'].mean():
                        print("   ✅ Higher prob_profit correlates with winning trades")
                    if winners['rules'].mean() > losers['rules'].mean():
                        print("   ✅ Higher rule signals correlate with winning trades")
                
            print("=" * 80)
        
    else:
        print(f"Model not found at {model_path}.")

if __name__ == "__main__":
    main()


