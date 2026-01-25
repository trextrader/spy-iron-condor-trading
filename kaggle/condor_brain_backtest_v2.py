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
import json
import uuid
import hashlib
import subprocess
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
from torch.utils.tensorboard import SummaryWriter # Added for TB support
try:
    from audit.decision_trace_logger import DecisionTraceLogger, TraceConfig
    HAS_TRACE_LOGGER = True
except Exception:
    HAS_TRACE_LOGGER = False

# --- SCALING HELPERS (Matched to training) ---
def robust_zscore_fit(X):
    median = np.nanmedian(X, axis=0)
    diff = np.abs(X - median)
    mad = np.nanmedian(diff, axis=0)
    return median, mad

def robust_zscore_transform(X, median, mad, clip_val=10.0):
    mad = np.where(mad < 1e-6, 1.0, mad) # Avoid div0
    z = (X - median) / (mad * 1.4826)
    return np.clip(z, -clip_val, clip_val)

# --- CONFIG ---
MODEL_PATH = "condor_brain_retrain_v22_e3.pth" # Default
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
RULESET_PATH = "docs/Complete_Ruleset_DSL.yaml"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
DECISION_TRACE_PATH = os.path.join(REPORTS_DIR, "decision_trace.jsonl")

# Iron Condor P&L Config
IC_CREDIT_PER_SPREAD = 1.50  # $1.50 credit per spread (typical)
IC_CONTRACTS = 10  # Number of contracts per trade
IC_MULTIPLIER = 100  # Options multiplier

def _sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"

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
    # CHECK if features already exist (from pre-compute pipeline)
    # V2.1 adds: 'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k'
    expected_v21 = ['rsi', 'atr', 'adx', 'stoch_k']
    if all(col in df.columns for col in expected_v21):
        print("✅ V2.1 Features already present. Skipping computation.")
    else:
        print("Computing V2.1 Dynamic Features...")
        df = compute_all_dynamic_features(df, close_col="close", high_col="high", low_col="low")
    
    # 2. Primitive Features (V2.2)
    # V2.2 adds: 'sma', 'psar', 'psar_mark' etc.
    expected_v22 = ['sma', 'psar', 'psar_mark']
    if all(col in df.columns for col in expected_v22):
         print("✅ V2.2 Features already present. Skipping computation.")
    else:
        print("Computing V2.2 Primitive Features...")
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


# --- P&L ESTIMATION ---
def estimate_condor_pnl(spot, short_call, long_call, short_put, long_put, credit_received, max_loss, days_held, total_dte):
    """
    Estimate P&L for Iron Condor based on Linear Theta Decay and Intrinsic Value.
    """
    # 1. Theta Decay (Profit)
    time_frac = min(max(days_held / total_dte, 0.0), 1.0) if total_dte > 0 else 1.0
    
    # 2. Intrinsic Value (Loss)
    call_spread_width = long_call - short_call
    intrinsic_call = max(0, spot - short_call)
    real_call_loss = min(intrinsic_call, call_spread_width)
    
    put_spread_width = short_put - long_put
    intrinsic_put = max(0, short_put - spot)
    real_put_loss = min(intrinsic_put, put_spread_width)
    
    total_intrinsic_loss = (real_call_loss + real_put_loss) * 100 * IC_CONTRACTS
    credit_dollar = credit_received * 100 * IC_CONTRACTS
    
    potential_profit = credit_dollar * time_frac
    net_pnl = potential_profit - total_intrinsic_loss
    
    # Cap
    actual_max_loss = -max_loss 
    net_pnl = max(net_pnl, actual_max_loss)
    net_pnl = min(net_pnl, credit_dollar)
    
    return net_pnl

def run_backtest(df, rule_signals, model, feature_cols, device, ruleset=None, model_path=None, data_path=None):
    print("Starting Backtest Simulation...")
    
    # Pre-process Features (Robust Norm same as training)
    X_np = df[feature_cols].values.astype(np.float32)
    X_np = np.nan_to_num(X_np, nan=0.0)
    
    # MATCH TRAINING: Log-transform Volume (Index 4)
    # Assuming 'volume' is index 4 in FEATURE_COLS_V22. 
    # Let's check column name to be safe.
    try:
        vol_idx = feature_cols.index('volume')
        print(f"Applying Log1p to Volume at index {vol_idx}...")
        X_np[:, vol_idx] = np.log1p(np.clip(X_np[:, vol_idx], 0.0, 1e9))
    except ValueError:
        print("Warning: 'volume' column not found in feature_cols. Skipping log transform.")

    # Normalize (approximate robust norm)
    mu = np.median(X_np, axis=0) if len(X_np) > 0 else 0
    mad = np.median(np.abs(X_np - mu), axis=0)
    mad = np.maximum(mad, 1e-6)
    
    X_norm = (X_np - mu) / (1.4826 * mad)
    X_norm = np.clip(X_norm, -10.0, 10.0)
    
    print(f"Feature Statistics check: Mean={np.mean(X_norm):.4f}, Std={np.std(X_norm):.4f}")
    if np.abs(np.mean(X_norm)) > 1.0:
        print("⚠️ WARNING: Features are not centered near 0. Model inputs might be drifted.")
    
    X_tensor = torch.tensor(X_norm, device=device)
    
    # Settings
    SEQ_LEN = 256
    capital = 100_000.0
    position = 0 # 0=None, 1=Long Iron Condor
    equity_curve = []
    trades = []
    open_trade = None
    
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

    # TensorBoard Writer
    tb_writer = SummaryWriter(log_dir=os.path.join(REPORTS_DIR, "tensorboard"))

    # Decision trace logger
    trace_logger = None
    if HAS_TRACE_LOGGER:
        trace_cfg = TraceConfig(
            output_path=DECISION_TRACE_PATH,
            model_id="CondorBrain",
            model_version=VERSION_V22,
            model_hash=_sha256_file(model_path) if model_path else "unknown",
            code_commit=_git_commit(),
            run_id=str(uuid.uuid4()),
            dataset_id=os.path.basename(data_path) if data_path else "unknown",
            dataset_path=data_path,
        )
        trace_logger = DecisionTraceLogger(trace_cfg)

    def _feature_snapshot(x_seq_tensor):
        vals = x_seq_tensor[0, -1, :].detach().cpu().numpy().tolist()
        return {feature_cols[j]: float(vals[j]) for j in range(len(feature_cols))}

    def _build_rule_factors(active_rules_list, net_rule_signal_val):
        rule_items = []
        for r_id in active_rules_list:
            rule_items.append({
                "rule_id": f"RULE:{r_id}",
                "rule_type": "SOFT_RULE",
                "passed": True,
                "value": 1.0,
                "threshold": 0.0,
                "weight": 1.0,
                "notes": ""
            })
        if not active_rules_list:
            rule_items.append({
                "rule_id": "RULE:NET_SIGNAL",
                "rule_type": "SOFT_RULE",
                "passed": bool(net_rule_signal_val >= 0),
                "value": float(net_rule_signal_val),
                "threshold": 0.0,
                "weight": 1.0,
                "notes": "Aggregated rule signal."
            })
        return rule_items

    def _emit_trace_event(
        scope,
        decision_type,
        intent,
        trade_id,
        spot_val,
        legs,
        entry_score_val=None,
        prob_val=None,
        conf_val=None,
        net_rule_signal_val=0.0,
        dte_entry_val=None,
        dte_remaining_val=None,
        pnl_val=None,
        max_loss_val=None,
        pos_size_pct_val=None,
        active_rules_list=None,
        feature_map=None,
        reason_text="",
    ):
        if trace_logger is None:
            return
        if active_rules_list is None:
            active_rules_list = []
        if feature_map is None:
            feature_map = {}
        r_mult = None
        if pnl_val is not None and max_loss_val:
            r_mult = float(pnl_val / max_loss_val) if max_loss_val > 0 else 0.0
        record = {
            "schema_version": "1.0",
            "event_id": str(uuid.uuid4()),
            "instrument": {
                "symbol": "SPY",
                "venue": "SIM",
                "asset_class": "OPTION",
                "contract": {
                    "expiry": "",
                    "right": "MULTI",
                    "strike": 0.0,
                    "multiplier": IC_MULTIPLIER
                }
            },
            "decision": {
                "trade_id": trade_id,
                "decision_id": str(uuid.uuid4()),
                "scope": scope,
                "decision_type": decision_type,
                "intent": intent,
                "timeframe": "1m",
                "horizon_bars": 0
            },
            "state": {
                "position": {
                    "side": "SHORT" if position == 1 else "FLAT",
                    "qty": float(IC_CONTRACTS) if position == 1 else 0.0,
                    "contracts": int(IC_CONTRACTS) if position == 1 else 0,
                    "avg_price": 0.0,
                    "greeks": {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0},
                    "margin_used": 0.0,
                    "risk_unit_R": float(max_loss_val) if max_loss_val is not None else 0.0
                },
                "market": {
                    "mid": float(spot_val),
                    "bid": 0.0,
                    "ask": 0.0,
                    "spread": 0.0,
                    "iv": 0.0,
                    "ivr": 0.0,
                    "volume": float(df["volume"].iloc[i]) if "volume" in df.columns else 0.0,
                    "liquidity_flags": []
                }
            },
            "inputs": {
                "feature_vector": {
                    "feature_schema_id": VERSION_V22,
                    "T": int(SEQ_LEN),
                    "D": int(len(feature_cols)),
                    "aggregation": "LAST",
                    "values": feature_map
                },
                "indicators": {},
                "engineered": {}
            },
            "model": {
                "outputs": {
                    "entry_logit": float(entry_score_val) if entry_score_val is not None else 0.0,
                    "exit_logit": float(net_rule_signal_val) if net_rule_signal_val is not None else 0.0,
                    "size_score": float(pos_size_pct_val) if pos_size_pct_val is not None else 0.0,
                    "uncertainty": {"sigma": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
                }
            },
            "decision_factors": {
                "rules": _build_rule_factors(active_rules_list, net_rule_signal_val),
                "learned_patterns": [],
                "attribution": [
                    {
                        "factor_id": f"FEAT:{k}",
                        "factor_kind": "FEATURE",
                        "value": float(v),
                        "contribution": 0.0,
                        "importance": 0.0,
                        "method": "NOT_AVAILABLE"
                    } for k, v in feature_map.items()
                ],
                "diffusion": {
                    "enabled": False,
                    "model_id": "",
                    "summary": {"path_mean": 0.0, "path_var": 0.0, "tail_risk": 0.0}
                },
                "fuzzy": {
                    "enabled": True,
                    "system_id": "ENTRY_SCORE_V1",
                    "memberships": {"score": float(entry_score_val) if entry_score_val is not None else 0.0},
                    "rules_fired": [],
                    "defuzz_output": {"size_multiplier": float(pos_size_pct_val) if pos_size_pct_val is not None else 0.0}
                }
            },
            "action": {
                "requested": {
                    "order_type": "SIM",
                    "side": "SELL" if decision_type == "OPEN" else "BUY",
                    "qty": float(IC_CONTRACTS),
                    "contracts": int(IC_CONTRACTS),
                    "limit_price": 0.0,
                    "tif": "DAY",
                    "legs": legs
                },
                "executed": {
                    "status": "FILLED",
                    "fill_qty": float(IC_CONTRACTS),
                    "fill_price": 0.0,
                    "slippage": 0.0,
                    "fees": 0.0,
                    "latency_ms": 0
                }
            },
            "outcome": {
                "labeling_policy_id": "WINDEF:v1",
                "evaluation_window_bars": 0,
                "win": bool(pnl_val > 0) if pnl_val is not None else False,
                "loss": bool(pnl_val < 0) if pnl_val is not None else False,
                "neutral": pnl_val is None,
                "r_multiple_final": float(r_mult) if r_mult is not None else 0.0,
                "mfe_r": 0.0,
                "mae_r": 0.0,
                "notes": reason_text
            },
            "governance": {
                "risk_gates": [],
                "overrides": [],
                "data_provenance": {
                    "dataset_id": os.path.basename(data_path) if data_path else "unknown",
                    "dataset_hash": _sha256_file(data_path) if data_path else None,
                    "bar_source": "SIM",
                    "timezone": "UTC"
                }
            }
        }
        record["state"]["position"]["dte_entry"] = float(dte_entry_val) if dte_entry_val is not None else None
        record["state"]["position"]["dte_remaining"] = float(dte_remaining_val) if dte_remaining_val is not None else None
        trace_logger.append(record)
    
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
        active_rules = []
        if rule_signals is not None:
            for col in rule_signals.columns:
                if "signal" in col and rule_signals[col].iloc[i] != 0:
                    active_rules.append(col.replace("_signal", ""))
        
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
                    
                # 3. Diffusion/Model (360 Degree View)
                # Helper for interpretation
                def interpret(val, low, high):
                    return "Bullish" if val > high else ("Bearish" if val < low else "Neutral")
                
                # Sequence Info
                # STANDARDIZED: use 'dt' (or 'timestamp' fallback)
                time_col = 'dt' if 'dt' in df.columns else 'timestamp'
                seq_start_time = df[time_col].iloc[i-SEQ_LEN] if time_col in df.columns else "N/A"
                seq_end_time = df[time_col].iloc[i] if time_col in df.columns else "N/A"

                reasoning.append(f"  3) 360° PREDICTOR VIEW (Model Output):")
                reasoning.append(f"     {'Predictor':<15} | {'Value':<10} | {'Interpretation'}")
                reasoning.append(f"     {'-'*15}-+-{'-'*10}-+-{'-'*20}")
                reasoning.append(f"     {'Confidence':<15} | {confidence:<10.4f} | {interpret(confidence, 0.3, 0.7)} (Threshold > 0.4)")
                reasoning.append(f"     {'Prob Profit':<15} | {prob_profit:<10.4f} | {interpret(prob_profit, 0.3, 0.6)} (Threshold > 0.4)")
                reasoning.append(f"     {'Direction':<15} | {direction:<10.4f} | {interpret(direction, -0.5, 0.5)} (Prefers Neutral)")
                reasoning.append(f"     {'TE (DTE)':<15} | {te_suggested:<10.4f} | {'Short Term' if te_suggested < 7 else 'Standard'}")
                reasoning.append(f"     {'Width':<15} | {width:<10.4f} | {'Wide' if width > 5 else 'Narrow'}")
                reasoning.append(f"     {'Call Offset':<15} | {call_offset:<10.4f} | {call_offset:.1f}% OTM")
                reasoning.append(f"     {'Put Offset':<15} | {put_offset:<10.4f} | {put_offset:.1f}% OTM")
                
                # 4. Fuzzy Logic & Sizing
                reasoning.append(f"  4) DECISION LOGIC:")
                reasoning.append(f"     Base Score:      {entry_score:.1f}/100 (Need {ENTRY_THRESHOLD})")
                
                # Position Sizing
                pos_size_pct = 100.0
                if 'position_size_multiplier' in df.columns:
                    dampener = df['position_size_multiplier'].iloc[i]
                    pos_size_pct = dampener * 100
                    reasoning.append(f"     Chaos Dampener:  {dampener:.4f} (Adjusts Size)")
                else:
                    reasoning.append(f"     Chaos Dampener:  1.00 (No Adjustment)")
                
                reasoning.append(f"     Final Sizing:    {pos_size_pct:.1f}% of Max Allocation")
                
                reasoning_str = "\\n".join(reasoning)

                # Fix: Model trained with TE=0 target predicts ~0. Enforce min DTE.
                DEFAULT_DTE = 14 # Hardcode default if not in config
                if te_suggested < 1.0:
                    te_suggested = DEFAULT_DTE
                trade_dte = float(te_suggested)

                trade_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🦅 IRON CONDOR #{trade_num} ENTRY @ Bar {i}
║ Time:          {seq_end_time}
╠══════════════════════════════════════════════════════════════════════════════╣
║ SPOT:          ${spot:.2f}
║ SEQUENCE:      {seq_start_time} -> {seq_end_time} ({SEQ_LEN} bars)
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

                trade_credit = credit_received  # Store for P&L calc
                trade_max_loss = max_loss
                
                trade_id = f"IC-{trade_num}"
                open_trade = {
                    'idx': i, 
                    'type': 'IRON_CONDOR', 
                    'action': 'OPEN',
                    'trade_id': trade_id,
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
                }
                trades.append(open_trade)
                legs = [
                    {"right": "C", "side": "SELL", "strike": float(short_call_strike), "qty": int(IC_CONTRACTS)},
                    {"right": "C", "side": "BUY", "strike": float(long_call_strike), "qty": int(IC_CONTRACTS)},
                    {"right": "P", "side": "SELL", "strike": float(short_put_strike), "qty": int(IC_CONTRACTS)},
                    {"right": "P", "side": "BUY", "strike": float(long_put_strike), "qty": int(IC_CONTRACTS)},
                ]
                feature_map = _feature_snapshot(x_seq)
                _emit_trace_event(
                    scope="ENTRY",
                    decision_type="OPEN",
                    intent="OPEN_CONDOR",
                    trade_id=trade_id,
                    spot_val=spot,
                    legs=legs,
                    entry_score_val=entry_score,
                    prob_val=prob_profit,
                    conf_val=confidence,
                    net_rule_signal_val=net_rule_signal,
                    dte_entry_val=trade_dte,
                    dte_remaining_val=trade_dte,
                    pnl_val=None,
                    max_loss_val=max_loss,
                    pos_size_pct_val=pos_size_pct,
                    active_rules_list=active_rules,
                    feature_map=feature_map,
                    reason_text="entry"
                )
                _emit_trace_event(
                    scope="SIZING",
                    decision_type="SIZE_ONLY",
                    intent="SIZE_ONLY",
                    trade_id=trade_id,
                    spot_val=spot,
                    legs=legs,
                    entry_score_val=entry_score,
                    prob_val=prob_profit,
                    conf_val=confidence,
                    net_rule_signal_val=net_rule_signal,
                    dte_entry_val=trade_dte,
                    dte_remaining_val=trade_dte,
                    pnl_val=None,
                    max_loss_val=max_loss,
                    pos_size_pct_val=pos_size_pct,
                    active_rules_list=active_rules,
                    feature_map=feature_map,
                    reason_text="sizing"
                )
                position = 1
            else:
                # Rejection: entry score below threshold
                rejection_reason = f"SCORE_TOO_LOW ({entry_score}/{ENTRY_THRESHOLD})"
                rejection_factors = entry_factors
        
        elif position == 1:
            # Calculate remaining DTE
            bars_held = i - trade_entry_bar if trade_entry_bar else 0
            days_held = bars_held / BARS_PER_DAY
            entry_dte = open_trade.get('dte') if open_trade else trade_dte
            remaining_dte = entry_dte - days_held if entry_dte else 0
            
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
                entry = open_trade if open_trade else trades[-1]
                time_col = 'dt' if 'dt' in df.columns else 'timestamp'
                entry_date = df[time_col].iloc[trade_entry_bar] if time_col in df.columns else "N/A"
                
                # Estimate P&L using Option Logic
                days_elapsed = (i - trade_entry_bar) / BARS_PER_DAY
                
                pnl_dollar = estimate_condor_pnl(
                    spot=spot,
                    short_call=entry['short_call'],
                    long_call=entry['long_call'],
                    short_put=entry['short_put'],
                    long_put=entry['long_put'],
                    credit_received=entry['credit'],
                    max_loss=entry['max_loss'],
                    days_held=days_elapsed,
                    total_dte=entry['dte']
                )
                
                realized_pnl = pnl_dollar # for exit logic below
                
                # Update status based on PnL
                if pnl_dollar > 0:
                    status_icon = "✅ WIN"
                    stats['winners'] += 1
                    stats['total_win_dollar'] = stats.get('total_win_dollar', 0) + pnl_dollar
                else:
                    status_icon = "❌ LOSS"
                    stats['losers'] += 1
                    stats['total_loss_dollar'] = stats.get('total_loss_dollar', 0) + abs(pnl_dollar)
                # Define Return on Risk (ROI)
                pnl_pct = (realized_pnl / entry['max_loss']) * 100 if entry['max_loss'] > 0 else 0.0

                # Deterministic Win/Loss check (Spot vs Strikes)
                realized_pnl = estimate_condor_pnl(
                    spot=spot,
                    short_call=trades[-1]['short_call'],
                    long_call=trades[-1]['long_call'],
                    short_put=trades[-1]['short_put'],
                    long_put=trades[-1]['long_put'],
                    credit_received=trades[-1]['credit'],
                    max_loss=trades[-1]['max_loss'],
                    days_held=trades[-1]['dte'], # Full Expiration
                    total_dte=trades[-1]['dte']
                )
                
                is_win = realized_pnl > 0
                status_icon = "✅ WIN" if is_win else "❌ LOSS"

                # Define Return on Risk (ROI) for Expiration
                pnl_pct = (realized_pnl / entry['max_loss']) * 100 if entry['max_loss'] > 0 else 0.0

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
                if avg_loss > 0:
                    sharpe_proxy = expectancy / avg_loss 
                    sharpe_str = f"{sharpe_proxy:.2f}"
                else:
                    sharpe_proxy = 99.99
                    sharpe_str = "Inf"

                # --- TENSORBOARD LOGGING ---
                step_idx = stats['total_trades']
                tb_writer.add_scalar("Performance/Equity", curr_equity, step_idx)
                tb_writer.add_scalar("Performance/Drawdown_Pct", curr_dd_pct, step_idx)
                tb_writer.add_scalar("Performance/Win_Rate", win_rate, step_idx)
                tb_writer.add_scalar("Trades/PnL_Dollar", realized_pnl, step_idx)
                tb_writer.add_scalar("Trades/Expectancy", expectancy, step_idx)
                tb_writer.add_scalar("Debug/Entry_Conf", trades[-1]['conf'], step_idx)

                legs = [
                    {"right": "C", "side": "BUY", "strike": float(entry.get('short_call', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "C", "side": "SELL", "strike": float(entry.get('long_call', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "P", "side": "BUY", "strike": float(entry.get('short_put', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "P", "side": "SELL", "strike": float(entry.get('long_put', 0.0)), "qty": int(IC_CONTRACTS)}
                ]
                feature_map = _feature_snapshot(x_seq)
                _emit_trace_event(
                    scope="EXIT",
                    decision_type="CLOSE",
                    intent="CLOSE_CONDOR",
                    trade_id=entry.get('trade_id', f"IC-{stats['total_trades']}"),
                    spot_val=spot,
                    legs=legs,
                    entry_score_val=entry.get('entry_score', 0.0),
                    prob_val=entry.get('prob', 0.0),
                    conf_val=entry.get('conf', 0.0),
                    net_rule_signal_val=net_rule_signal,
                    dte_entry_val=entry.get('dte', None),
                    dte_remaining_val=remaining_dte,
                    pnl_val=realized_pnl,
                    max_loss_val=entry.get('max_loss', None),
                    pos_size_pct_val=None,
                    active_rules_list=active_rules,
                    feature_map=feature_map,
                    reason_text=exit_reason
                )

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
║ 6) Sharpe (Tr):{sharpe_str:>5} (Proxy)
╚══════════════════════════════════════════════════════════════════════════════╝"""
                print(exit_msg)
                log_file.write(exit_msg + "\n")
                
                close_trade = {
                    'idx': i, 
                    'type': 'IRON_CONDOR',
                    'action': 'CLOSE',
                    'spot': spot,
                    'reason': exit_reason,
                    'days_held': days_held,
                    'dte_entry': entry.get('dte', None),
                    'dte_remaining': remaining_dte,
                    'short_call': entry.get('short_call'),
                    'long_call': entry.get('long_call'),
                    'short_put': entry.get('short_put'),
                    'long_put': entry.get('long_put'),
                    'width': entry.get('width'),
                    'credit': entry.get('credit'),
                    'max_loss': entry.get('max_loss'),
                    'pnl_pct': pnl_pct
                }
                trades.append(close_trade)
                position = 0
                trade_entry_bar = None
                trade_dte = None
                open_trade = None
        
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
    if 'tb_writer' in locals():
        tb_writer.close()
        print(f"TensorBoard logs saved to: {os.path.join(REPORTS_DIR, 'tensorboard')}")
    print("=" * 80)
        
    return equity_curve, trades

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CondorBrain Backtest V2.2")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV data")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--ruleset", type=str, default=None, help="Path to ruleset YAML")
    args = parser.parse_args()

    # --- GPU CHECK ---
    print("="*60)
    print("HARDWARE STATUS CHECK")
    print("="*60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU DETECTED: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️ GPU NOT DETECTED! Running on CPU (Will be slow).")
    print("="*60)

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
        equity, trades = run_backtest(
            df,
            rule_signals,
            model,
            FEATURE_COLS_V22,
            DEVICE,
            ruleset,
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
        )
        
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
                        'reason': close_t.get('reason', 'Unknown'),
                        'dte_entry': open_t.get('dte', None),
                        'dte_remaining': close_t.get('dte_remaining', None),
                        'short_call': open_t.get('short_call', None),
                        'long_call': open_t.get('long_call', None),
                        'short_put': open_t.get('short_put', None),
                        'long_put': open_t.get('long_put', None),
                        'width': open_t.get('width', None),
                        'credit': open_t.get('credit', None),
                        'max_loss': open_t.get('max_loss', None)
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


