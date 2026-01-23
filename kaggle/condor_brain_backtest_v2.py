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
    
    # --- STANDARDIZE DATE COLUMN ---
    if 'dt' not in df.columns and 'timestamp' in df.columns:
        print("   ⚠️ Standardizing 'timestamp' -> 'dt'...")
        df.rename(columns={'timestamp': 'dt'}, inplace=True)

    if 'dt' in df.columns:
        # Ensure UTC standard
        df['dt'] = pd.to_datetime(df['dt'], utc=True)
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
    return df, rule_signals, ruleset


def run_backtest(df, rule_signals, model, feature_cols, device, ruleset=None):
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
    
    # Running Stats
    stats = {
        'total_trades': 0,
        'winners': 0,
        'losers': 0,
        'total_pnl_dollar': 0.0,
        'peak_capital': capital,
        'max_dd_pct': 0.0,
        'win_streak': 0,
        'lose_streak': 0
    }
    
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
                # Identify Triggering Rules
                active_rules = []
                if rule_signals is not None:
                    for col in rule_signals.columns:
                        if "signal" in col and rule_signals[col].iloc[i] != 0:
                            r_id = col.replace("_signal", "")
                            active_rules.append(r_id)
                
                # Format Primitives & Reasons
                reasoning = []
                # 1. Primitives (from Triggering Rules)
                if ruleset and active_rules:
                    reasoning.append("  1) Primitives:")
                    for r_id in active_rules:
                        if r_id in ruleset.rules:
                            rule = ruleset.rules[r_id]
                            # Check features/primitives required
                            reqs = rule.requires.features if hasattr(rule.requires, 'features') else []
                            vals = []
                            for f in reqs:
                                if f in df.columns:
                                    vals.append(f"{f}={df[f].iloc[i]:.4f}")
                            if vals:
                                reasoning.append(f"     Rule {r_id}: " + ", ".join(vals))
                
                # 2. Trade Rules
                reasoning.append("  2) Trade Rules:")
                if active_rules:
                    reasoning.append(f"     Triggered: {', '.join(active_rules)}")
                else:
                    reasoning.append("     Triggered: None (Model Force?)")
                    
                # 3. Diffusion/Model
                reasoning.append("  3) Diffusion/Model:")
                reasoning.append(f"     Call Offset: {call_offset:.2f} | Put Offset: {put_offset:.2f}")
                reasoning.append(f"     Width: {width:.2f} | DTE Pred: {te_suggested:.2f}")
                reasoning.append(f"     Prob Profit: {prob_profit:.4f} | Confidence: {confidence:.4f}")
                
                # 4. Fuzzy Logic
                reasoning.append("  4) Fuzzy Logic Sizing:")
                reasoning.append(f"     Base Score: {entry_score:.1f}/100")
                reasoning.append(f"     Breakdown: Prob({prob_profit:.2f})*40 + Conf({confidence:.2f})*20 + Vol({vol_score:.2f})*20 + Trend({trend_score:.2f})*20")
                if 'position_size_multiplier' in df.columns:
                     reasoning.append(f"     Chaos Dampener: {df['position_size_multiplier'].iloc[i]:.2f}")
                
                reasoning_str = "\\n".join(reasoning)

                trade_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🦅 IRON CONDOR #{trade_num} ENTRY @ Bar {i}
╠══════════════════════════════════════════════════════════════════════════════╣
║ SPOT:          ${spot:.2f}
║ FUZZY SCORE:   {entry_score}/100 (threshold: {ENTRY_THRESHOLD})
╠─────────────────────────── DETAILED REASONING ───────────────────────────────╣
{reasoning_str}
╠─────────────────────────── IRON CONDOR SETUP ────────────────────────────────╣
║   Short Call:  ${short_call_strike:.2f}  |  Long Call:  ${long_call_strike:.2f}
║   Short Put:   ${short_put_strike:.2f}  |  Long Put:   ${long_put_strike:.2f}
║   Width:       ${width:.2f}  |  DTE: {trade_dte:.1f} days
╠─────────────────────────── P&L POTENTIAL ────────────────────────────────────╣
║   Credit:      ${credit_received:,.2f} (Max Profit)
║   Max Loss:    ${max_loss:,.2f}
║   Contracts:   {IC_CONTRACTS}
╚══════════════════════════════════════════════════════════════════════════════╝"""
                print(trade_msg)
                log_file.write(trade_msg + "\n")
                
                # Set DTE for expiration tracking
                trade_entry_bar = i
                # Fix: Model trained with TE=0 target predicts ~0. Enforce min DTE.
                if te_suggested < 1.0:
                    te_suggested = DEFAULT_DTE
                trade_dte = float(te_suggested)

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
                
                # Calculates Running Metrics
                pnl_dollar = (pnl_pct / 100.0) * trade_max_loss # Approx PnL dollars based on risk? Or based on credit?
                # Actually PnL % is usually on margin. Let's use simple approx.
                # If win, +credit. If loss, -Loss.
                # But pnl_pct is spot change? No, pnl_pct was defined as spot change proxy.
                # Let's refine PnL logic for Condor:
                trade_res_dollar = 0.0
                if pnl_pct > 0: # Proxy for win (should check strikes!)
                     # Wait, we checked strikes below in loop (lines 460+).
                     # Current logic checks exit reason. 
                     # If Expiration, we check strikes.
                     # If Early Exit, we need PnL.
                     # Let's use the 'Simulation' calculation logic which is more accurate?
                     # Limitation: The simulation loop updates 'capital' daily.
                     # Here at exit signal, we don't have exact option price.
                     # We'll rely on the existing PnL pct proxy for reporting, 
                     # BUT update the stats based on win/loss flag.
                     pass

                # Deterministic Win/Loss check (Spot vs Strikes)
                short_c = trades[-1]['short_call']
                short_p = trades[-1]['short_put']
                is_win = (spot < short_c) and (spot > short_p)
                status_icon = "✅ WIN" if is_win else "❌ LOSS"
                
                # PnL Realization (Approx)
                if is_win:
                    realized_pnl = trade_credit
                else:
                    # Loss amount depends on how far OTM. 
                    # Simulating max loss for pessimistic reporting if breached.
                    realized_pnl = -trade_max_loss

                # Update Stats
                stats['total_trades'] += 1
                if is_win:
                    stats['winners'] += 1
                    stats['total_win_dollar'] = stats.get('total_win_dollar', 0.0) + realized_pnl
                else:
                    stats['losers'] += 1
                    stats['total_loss_dollar'] = stats.get('total_loss_dollar', 0.0) + abs(realized_pnl)
                
                stats['total_pnl_dollar'] += realized_pnl
                
                # Check DD
                stats['peak_capital'] = max(stats['peak_capital'], capital + stats['total_pnl_dollar']) # Approx equity
                curr_equity = capital + stats['total_pnl_dollar']
                curr_dd_dollar = stats['peak_capital'] - curr_equity
                curr_dd_pct = (curr_dd_dollar / stats['peak_capital']) * 100 if stats['peak_capital'] > 0 else 0
                stats['max_dd_pct'] = max(stats['max_dd_pct'], curr_dd_pct)
                
                # Metrics Calculation
                win_count = stats['winners']
                loss_count = stats['losers']
                total_count = stats['total_trades']
                win_rate = (win_count / total_count) if total_count > 0 else 0
                
                avg_win = stats.get('total_win_dollar', 0.0) / win_count if win_count > 0 else 0
                avg_loss = stats.get('total_loss_dollar', 0.0) / loss_count if loss_count > 0 else 0
                
                # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                
                # Running Sharpe (Trade-based)
                # Mean(PnL) / Std(PnL) * sqrt(TradesPerYear?) -> Just use trade sharpe
                # Not accurately tracking per-trade PnL list here for StdDev.
                # using simplified Sharpe proxy: Expectancy / AvgLoss (sort of E-Ratio)
                # Proper Sharpe requires variance tracking.
                # Let's verify if we can list it.
                sharpe_proxy = expectancy / avg_loss if avg_loss > 0 else 0.0

                exit_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🔔 IRON CONDOR EXIT @ Bar {i}
╠══════════════════════════════════════════════════════════════════════════════╣
║ Spot Price:    ${spot:.2f} (Entry: ${entry_spot:.2f})
║ Exit Reason:   {exit_reason}
║ Result:        {status_icon} (PnL: ${realized_pnl:,.0f})
╠─────────────────────────── PERFORMANCE METRICS ──────────────────────────────╣
║ 1) Trades:     {total_count} (W: {win_count} {win_rate*100:.1f}% | L: {loss_count})
║ 2) Net P&L:    ${stats['total_pnl_dollar']:,.2f}
║ 3) Drawdown:   {curr_dd_pct:.2f}% (Max: {stats['max_dd_pct']:.2f}%)
║ 4) NP/DD:      {stats['total_pnl_dollar'] / curr_dd_dollar if curr_dd_dollar > 1 else 0:.2f}
║ 5) Expectancy: ${expectancy:.2f}
║ 6) Sharpe (Tr):{sharpe_proxy:.2f} (Proxy)
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
    import argparse
    parser = argparse.ArgumentParser(description="CondorBrain Backtest V2.2")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV data")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--ruleset", type=str, default=None, help="Path to ruleset YAML")
    args = parser.parse_args()

    # 0. Data Path Detection
    possible_data_paths = [
        args.input, # CLI override first
        MODEL_PATH, # Kaggle default config (Line 42 might be overwritten)
        "/kaggle/input/spy-options-data/mamba_institutional_1m.csv",
        "/content/spy-iron-condor-trading/data/processed/mamba_institutional_2024_1m_last 500k.csv",
        "data/processed/mamba_institutional_2024_1m_last 500k.csv",
        "/content/spy-iron-condor-trading/data/processed/mamba_institutional_1m.csv",
        "/content/spy-iron-condor-trading/data/mamba_institutional_1m_500k.csv",
        "data/processed/mamba_institutional_1m.csv",
        "data/mamba_institutional_1m_500k.csv",
        "mamba_institutional_1m.csv",
        "mamba_institutional_1m_500k.csv"
    ]
    use_data_path = DATA_PATH
    for p in possible_data_paths:
        if p and os.path.exists(p):
            use_data_path = p
            print(f"Found data at: {use_data_path}")
            break
            
    # 1. Pipeline
    # Load limited rows for test? Or all.
    df = load_data_and_features(use_data_path)  # Load ALL rows
    
    # 2. Rules
    ruleset_path = args.ruleset if args.ruleset else RULESET_PATH
    df, rule_signals, ruleset = run_rule_engine(df, ruleset_path)
    
    # 3. Model
    POSSIBLE_PATHS = [
        args.model, # CLI override first
        # Colab 500K trained models (E3 first)
        "/content/spy-iron-condor-trading/condor_brain_retrain_e3+500k.pth",  # 🚀 E3 500K
        "/content/spy-iron-condor-trading/condor_brain_retrain_e2_500k.pth",
        "/content/spy-iron-condor-trading/condor_brain_retrain_e1_500K.pth",
        "/content/spy-iron-condor-trading/condor_brain_retrain_e3.pth",
        # Local paths
        "condor_brain_retrain_e3+500k.pth",
        "condor_brain_retrain_e2_500k.pth",
        "condor_brain_retrain_e1_500K.pth",
        "condor_brain_retrain_v22_e3.pth",
        "condor_brain_retrain_e3.pth",
        # Kaggle paths
        "/kaggle/working/condor_brain_retrain_e3.pth",
        "/kaggle/working/condor_brain_retrain_v22_e3.pth",
    ]
    model_path = MODEL_PATH
    for p in POSSIBLE_PATHS:
        if p and os.path.exists(p):
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
        equity, trades = run_backtest(df, rule_signals, model, FEATURE_COLS_V22, DEVICE, ruleset)
        
        # 5. Report
        print(f"Final Capital: ${equity[-1]:,.2f}")
        print(f"Trades: {len(trades)}")
        
        # Calculate metrics for enhanced chart
        equity_arr = np.array(equity)
        starting_balance = equity_arr[0]
        
        # Calculate running max and drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max * 100  # Percentage
        max_drawdown = np.min(drawdown)
        
        # Calculate total return
        total_return = (equity_arr[-1] - starting_balance) / starting_balance * 100
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f"Iron Condor V2.2 Backtest Results - {len(trades)} Trades", fontsize=14, fontweight='bold')
        
        # --- Subplot 1: Equity Curve with Balance Reference ---
        ax1 = axes[0]
        ax1.plot(equity_arr, label='Equity', color='#2E86AB', linewidth=1.5)
        ax1.axhline(y=starting_balance, color='#E94F37', linestyle='--', linewidth=1, label=f'Starting Balance (${starting_balance:,.0f})')
        ax1.fill_between(range(len(equity_arr)), starting_balance, equity_arr, 
                         where=equity_arr >= starting_balance, alpha=0.3, color='green', label='Profit Zone')
        ax1.fill_between(range(len(equity_arr)), starting_balance, equity_arr, 
                         where=equity_arr < starting_balance, alpha=0.3, color='red', label='Loss Zone')
        ax1.set_ylabel('Capital ($)', fontsize=10)
        ax1.set_title(f'Equity Curve | Final: ${equity_arr[-1]:,.2f} | Return: {total_return:+.2f}%', fontsize=11)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(equity_arr))
        
        # --- Subplot 2: Drawdown ---
        ax2 = axes[1]
        ax2.fill_between(range(len(drawdown)), 0, drawdown, color='#E94F37', alpha=0.7)
        ax2.axhline(y=max_drawdown, color='darkred', linestyle='--', linewidth=1, label=f'Max DD: {max_drawdown:.2f}%')
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_title(f'Drawdown | Max: {max_drawdown:.2f}%', fontsize=11)
        ax2.legend(loc='lower left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, len(equity_arr))
        ax2.set_ylim(min(drawdown) * 1.1, 5)
        
        # --- Subplot 3: Trade P&L Markers ---
        ax3 = axes[2]
        # Mark trade points and P&L
        closes = [t for t in trades if t.get('action') == 'CLOSE']
        trade_bars = []
        trade_pnls = []
        for t in closes:
            pnl_pct = t.get('pnl_pct', 0)
            if pnl_pct != 0:
                trade_bars.append(t.get('idx', 0))
                trade_pnls.append(pnl_pct)
        
        colors = ['green' if p > 0 else 'red' for p in trade_pnls]
        ax3.bar(trade_bars, trade_pnls, color=colors, alpha=0.7, width=max(1, len(equity_arr)//200))
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.set_ylabel('Trade P&L (%)', fontsize=10)
        ax3.set_xlabel('Bar Index', fontsize=10)
        win_count = sum(1 for p in trade_pnls if p > 0)
        total_count = len(trade_pnls)
        win_rate = win_count / total_count * 100 if total_count > 0 else 0
        ax3.set_title(f'Individual Trade P&L | Win Rate: {win_rate:.1f}% ({win_count}/{total_count})', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, len(equity_arr))
        
        plt.tight_layout()
        plot_path = os.path.join(REPORTS_DIR, "backtest_v2_result.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved enhanced plot to {plot_path}")
        
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


