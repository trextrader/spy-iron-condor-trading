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
    FEATURE_COLS_V22,
    INPUT_DIM_V22,
    VERSION_V22,
    apply_semantic_nan_fill,
    get_neutral_fill_value_v22,
)
from intelligence.features.dynamic_features import (
    compute_all_dynamic_features,
    compute_all_primitive_features_v22
)
from intelligence.rule_engine.dsl_parser import RuleDSLParser
from intelligence.rule_engine.executor import RuleExecutionEngine
try:
    from torch.utils.tensorboard import SummaryWriter # Added for TB support
except ImportError:
    class SummaryWriter:
        def __init__(self, log_dir=None): pass
        def add_scalar(self, tag, scalar_value, global_step=None): pass
        def close(self): pass
try:
    from audit.decision_trace_logger import DecisionTraceLogger, TraceConfig
    HAS_TRACE_LOGGER = True
except Exception:
    HAS_TRACE_LOGGER = False
from audit.contract_snapshot import generate_contract_snapshot
from audit.generate_decision_factor_attribution import generate_attribution_csv
from audit.schema.validate_decision_factor_attribution import validate_decision_factor_attribution
from audit.schema.validate_decision_trace import validate_decision_trace

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
BAR_TRACE_PATH = os.path.join(REPORTS_DIR, "bar_trace.jsonl")

# Iron Condor P&L Config
IC_CREDIT_PER_SPREAD = 1.50  # $1.50 credit per spread (typical)
IC_CONTRACTS = 10  # Number of contracts per trade
IC_MULTIPLIER = 100  # Options multiplier
RULE_FEATURES = ["rule_long_consensus", "rule_short_consensus", "rule_exit_consensus", "rule_block_any"]

# Trace + outcome labeling config
TRACE_PER_BAR = True
TOP_N_FEATURES = 20
BAR_TRACE_ENABLED = True
BAR_TRACE_INCLUDE_ROW = True
ENTRY_ALPHA_R = 0.25
ENTRY_BETA_R = 0.25
EXIT_EVAL_HOLD_BARS = 30
EXIT_RISK_LAMBDA = 0.5
SIZING_RISK_LAMBDA = 0.5
CONF_ENTRY_MIN = 0.35      # Lowered from 0.4 - model outputs ~0.5-0.65
PROB_ENTRY_MIN = 0.40      # Lowered from 0.45 - main blocker for entries
MIN_HOLD_BARS = 30

# === DEBUG INFERENCE FLAGS ===
DEBUG_INFERENCE = True      # Set to True to see detailed feature/model output analysis
DEBUG_SAMPLE_EVERY = 500    # Print debug info every N bars (reduce noise)
DEBUG_FIRST_N = 20          # Always print first N bars for initial analysis
DEBUG_LOG_FILE = "debug_inference.log"  # Output file for debug logs
MIN_BARS_BETWEEN_TRADES = 5

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

def _append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

def _json_safe(val):
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, (pd.Timestamp,)):
        return str(val)
    return val

def _json_safe_dict(d):
    return {k: _json_safe(v) for k, v in d.items()}

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
    
    # Compute missing V2.2 features (dynamic + primitives) on unique spot bars, then merge back.
    missing_v22 = [c for c in FEATURE_COLS_V22 if c not in df.columns]

    # SMART CHECK: Also detect if existing V2.2 columns have suspicious constant values
    # These key features should have variance if properly computed
    validation_cols = ['exec_allow', 'friction_ratio', 'gap_risk_score', 'rsi_dyn', 'adx_adaptive']
    suspicious_cols = []
    for col in validation_cols:
        if col in df.columns:
            col_std = df[col].std()
            col_unique = df[col].nunique()
            if col_std < 0.001 or col_unique <= 2:
                suspicious_cols.append(f"{col}(std={col_std:.4f}, unique={col_unique})")

    if suspicious_cols and not missing_v22:
        print(f"⚠️ V2.2 columns exist but have SUSPICIOUS constant values:")
        for s in suspicious_cols:
            print(f"   - {s}")
        print(f"   FORCING RECOMPUTATION of primitives...")
        # Force recomputation by pretending these are missing
        missing_v22 = validation_cols

    if not missing_v22:
        print("✅ V2.2 Features already present with valid variance. Skipping computation.")
        return df

    print(f"⚠️ Missing/Invalid {len(missing_v22)} V2.2 features; computing on spot bars...")

    # Determine datetime column for spot keying
    dt_col = None
    for c in ['dt', 'timestamp', 'datetime', 'date']:
        if c in df.columns:
            dt_col = c
            break

    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    aux_cols = [c for c in ["spread_ratio", "lag_minutes"] if c in df.columns]

    if dt_col is None:
        print("⚠️ No datetime column found; computing on full dataframe.")
        df = compute_all_dynamic_features(df, close_col="close", high_col="high", low_col="low")
        df = compute_all_primitive_features_v22(
            df,
            close_col="close", high_col="high", low_col="low",
            volume_col="volume",
            spread_col="spread_ratio" if "spread_ratio" in df.columns else "close",
            inplace=True
        )
        return df

    spot_key_cols = ['symbol', dt_col] if 'symbol' in df.columns else [dt_col]
    spot_df = df.drop_duplicates(subset=spot_key_cols)[spot_key_cols + ohlcv_cols + aux_cols].copy()
    spot_df = spot_df.sort_values(spot_key_cols).reset_index(drop=True)

    print("   Computing dynamic features on spot bars...")
    spot_df = compute_all_dynamic_features(spot_df, close_col="close", high_col="high", low_col="low")

    print("   Computing V2.2 primitive features on spot bars...")
    spot_df = compute_all_primitive_features_v22(
        spot_df,
        close_col="close", high_col="high", low_col="low",
        volume_col="volume",
        spread_col="spread_ratio" if "spread_ratio" in spot_df.columns else "close",
        inplace=True
    )

    # Merge computed columns back without clobbering existing values
    exclude_cols = spot_key_cols + ohlcv_cols + aux_cols
    computed_cols = [c for c in spot_df.columns if c not in exclude_cols]
    merge_df = spot_df[spot_key_cols + computed_cols]
    df = df.merge(merge_df, on=spot_key_cols, how='left', suffixes=('', '_calc'))
    for col in computed_cols:
        calc_col = f"{col}_calc"
        if calc_col in df.columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[calc_col])
            else:
                df[col] = df[calc_col]
            df.drop(columns=[calc_col], inplace=True)
    
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
    
    # If options rows include duplicates per timestamp, execute rules on unique spot bars.
    dt_col = None
    for c in ['dt', 'timestamp', 'datetime', 'date']:
        if c in df.columns:
            dt_col = c
            break
    spot_key_cols = ['symbol', dt_col] if dt_col and 'symbol' in df.columns else ([dt_col] if dt_col else None)
    if spot_key_cols and df.duplicated(subset=spot_key_cols).any():
        use_spot_df = df.drop_duplicates(subset=spot_key_cols).copy()
        use_spot_df = use_spot_df.sort_values(spot_key_cols).reset_index(drop=True)
    else:
        use_spot_df = df

    results = engine.execute(use_spot_df)
    
    # Parse Results into DataFrame Columns
    
    rule_signals = pd.DataFrame(index=use_spot_df.index)
    long_signals = []
    short_signals = []
    exit_signals = []
    block_signals = []
    
    for rule_id, rule in ruleset.rules.items():
        r_res = results.get(rule_id)
        if r_res is None:
            continue
        
        # Entry Signal (1=Long, -1=Short, 0=None)
        long_s = r_res.get('signal_long', pd.Series(False, index=use_spot_df.index))
        short_s = r_res.get('signal_short', pd.Series(False, index=use_spot_df.index))
        exit_s = r_res.get('signal_exit', pd.Series(False, index=use_spot_df.index))
        block_s = r_res.get('blocked', pd.Series(False, index=use_spot_df.index))
        
        if hasattr(long_s, 'fillna'):
            long_s = long_s.fillna(False).astype(int)
        else:
            long_s = pd.Series(0, index=use_spot_df.index)
        if hasattr(short_s, 'fillna'):
            short_s = short_s.fillna(False).astype(int)
        else:
            short_s = pd.Series(0, index=use_spot_df.index)
        if hasattr(exit_s, 'fillna'):
            exit_s = exit_s.fillna(False).astype(int)
        else:
            exit_s = pd.Series(0, index=use_spot_df.index)
        if hasattr(block_s, 'fillna'):
            block_s = block_s.fillna(False).astype(int)
        else:
            block_s = pd.Series(0, index=use_spot_df.index)
        
        # Combine: Long=1, Short=-1
        sig = long_s - short_s
        rule_signals[f"{rule_id}_signal"] = sig
        long_signals.append(long_s.values)
        short_signals.append(short_s.values)
        exit_signals.append(exit_s.values)
        block_signals.append(block_s.values)
            
    if long_signals:
        spot_consensus = pd.DataFrame({
            "rule_long_consensus": np.mean(long_signals, axis=0),
            "rule_short_consensus": np.mean(short_signals, axis=0),
            "rule_exit_consensus": np.mean(exit_signals, axis=0),
            "rule_block_any": np.max(block_signals, axis=0),
        }, index=use_spot_df.index)
    else:
        spot_consensus = pd.DataFrame({
            "rule_long_consensus": 0.0,
            "rule_short_consensus": 0.0,
            "rule_exit_consensus": 0.0,
            "rule_block_any": 0.0,
        }, index=use_spot_df.index)

    # If we used spot_df, merge signals back to full df
    if spot_key_cols and len(use_spot_df) != len(df):
        spot_join = use_spot_df[spot_key_cols].copy()
        spot_join = spot_join.join(spot_consensus)
        df = df.merge(spot_join, on=spot_key_cols, how='left')

        rule_signals_full = use_spot_df[spot_key_cols].copy()
        rule_signals_full = rule_signals_full.join(rule_signals)
        rule_signals_full = df[spot_key_cols].merge(rule_signals_full, on=spot_key_cols, how='left')
        rule_signals_full = rule_signals_full.drop(columns=spot_key_cols)
        rule_signals_full.index = df.index
    else:
        df["rule_long_consensus"] = spot_consensus["rule_long_consensus"].values
        df["rule_short_consensus"] = spot_consensus["rule_short_consensus"].values
        df["rule_exit_consensus"] = spot_consensus["rule_exit_consensus"].values
        df["rule_block_any"] = spot_consensus["rule_block_any"].values
        rule_signals_full = rule_signals

    print(f"Generated signals for {len(ruleset.rules)} rules.")
    return df, rule_signals_full, ruleset


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
    
    # credit_received and max_loss are total dollars for the full position
    total_intrinsic_loss = (real_call_loss + real_put_loss) * IC_MULTIPLIER * IC_CONTRACTS
    credit_dollar = credit_received
    
    potential_profit = credit_dollar * time_frac
    net_pnl = potential_profit - total_intrinsic_loss
    
    # Cap to max loss / max profit
    actual_max_loss = -max_loss
    net_pnl = max(net_pnl, actual_max_loss)
    net_pnl = min(net_pnl, credit_dollar)
    
    return net_pnl

def find_best_legs(chain_df, spot, call_off, put_off, width):
    """
    Search the 100-row options chain for the 4 legs matching model suggestions.
    Uses close prices for pricing.
    """
    if chain_df.empty:
        return None

    # Suggested strikes
    s_call_target = spot + (call_off * spot * 0.01)
    s_put_target = spot - (put_off * spot * 0.01)
    
    # Filter by CP
    calls = chain_df[chain_df['call_put'] == 'C']
    puts = chain_df[chain_df['call_put'] == 'P']
    
    if calls.empty or puts.empty:
        return None
        
    # Find closest short strikes
    short_call_row = calls.iloc[(calls['strike'] - s_call_target).abs().argsort()[:1]]
    short_put_row = puts.iloc[(puts['strike'] - s_put_target).abs().argsort()[:1]]
    
    s_call = short_call_row['strike'].values[0]
    s_put = short_put_row['strike'].values[0]
    
    # Find matching long strikes (strike + width)
    l_call_target = s_call + width
    l_put_target = s_put - width
    
    long_call_row = calls.iloc[(calls['strike'] - l_call_target).abs().argsort()[:1]]
    long_put_row = puts.iloc[(puts['strike'] - l_put_target).abs().argsort()[:1]]
    
    l_call = long_call_row['strike'].values[0]
    l_put = long_put_row['strike'].values[0]
    
    # Package legs
    return {
        'short_call': s_call, 'short_call_close': short_call_row['close'].values[0], 'short_call_symbol': short_call_row['option_symbol'].values[0],
        'long_call': l_call, 'long_call_close': long_call_row['close'].values[0], 'long_call_symbol': long_call_row['option_symbol'].values[0],
        'short_put': s_put, 'short_put_close': short_put_row['close'].values[0], 'short_put_symbol': short_put_row['option_symbol'].values[0],
        'long_put': l_put, 'long_put_close': long_put_row['close'].values[0], 'long_put_symbol': long_put_row['option_symbol'].values[0],
        'width': abs(l_call - s_call) 
    }

# --- M1 BACKTEST CORE ---

class Trade:
    def __init__(self, trade_id, entry_dt, legs, entry_credit, max_loss, rules=None, scores=None, dte=None):
        self.trade_id = trade_id
        self.entry_dt = entry_dt
        self.legs = legs
        self.net_credit = entry_credit
        self.max_loss = max_loss
        self.qty = IC_CONTRACTS
        
        self.realized_pnl = 0.0
        self.exit_dt = None
        self.exit_reason = None
        self.is_closed = False
        self.dte_entry = dte
        
        self.rules = rules or {}
        self.scores = scores or {}
        
        self.unrealized_pnl = 0.0
        self.pnl_pct = 0.0
        self.max_dd_pct = 0.0

    def update_mark(self, current_dt, marks_dict):
        """Update unrealized PnL using current O(1) mark cache."""
        # Price 4 legs
        sc = marks_dict.get(self.legs['short_call_symbol'])
        lc = marks_dict.get(self.legs['long_call_symbol'])
        sp = marks_dict.get(self.legs['short_put_symbol'])
        lp = marks_dict.get(self.legs['long_put_symbol'])
        
        # If any leg missing, carry forward logic (simplified: skip update)
        if None in [sc, lc, sp, lp]:
            return
            
        # Cost to close = Buy Shorts - Sell Longs
        debit = (sc + sp) - (lc + lp)
        
        # PnL = Credit - Debit
        pnl_per_share = self.net_credit - debit
        self.unrealized_pnl = pnl_per_share * self.qty * IC_MULTIPLIER
        
        # DD tracking
        if self.max_loss > 0:
            self.pnl_pct = self.unrealized_pnl / self.max_loss
            self.max_dd_pct = min(self.max_dd_pct, self.pnl_pct)

    def should_risk_close(self, current_equity):
        # HARD STOP: 5% of Account Equity
        limit = -0.05 * current_equity
        return self.unrealized_pnl < limit

def build_ts_ranges(df, time_col='dt'):
    """Build O(1) lookup index for timestamp ranges."""
    print("Building O(1) option chain index...")
    ts_ranges = {}
    cur_ts = None
    start = 0
    vals = df[time_col].values
    n = len(df)
    
    for idx, ts in enumerate(tqdm(vals, desc="Indexing")):
        if cur_ts is None:
            cur_ts = ts
            start = idx
        elif ts != cur_ts:
            ts_ranges[cur_ts] = (start, idx)
            cur_ts = ts
            start = idx
    if cur_ts is not None:
        ts_ranges[cur_ts] = (start, n)
    return ts_ranges


def run_backtest(df, rule_signals, model, feature_cols, device, ruleset=None, model_path=None, data_path=None, norm_stats=None, 
                 use_fuzzy_sizing=False, use_trade_rules=True, use_diffusion=False, limit=None):
    
    # 1. PREPARE DATA (M1-Centric)
    print("Sorting and Indexing Data...")
    time_col = 'dt' if 'dt' in df.columns else 'timestamp'
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Build O(1) Chain Index
    ts_ranges = build_ts_ranges(df, time_col)
    
    # Build Spot Bars (1 row per minute)
    # Correctness: Use FIRST row for OHLCV (or explicit spot cols if robust)
    spot_bars = df.drop_duplicates(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    
    if limit:
        spot_bars = spot_bars.iloc[:limit+256] # approx buffer
        print(f"Limiting to {len(spot_bars)} bars.")

    # 2. PRE-COMPUTE MODEL OUTPUTS
    # We need features for every unique bar.
    # We'll re-extract features from spot_bars to ensure alignment.
    print("Preparing Features...")
    # Fill missing columns in spot_bars if any (usually duplicates of option row 0)
    missing_cols = [c for c in feature_cols if c not in spot_bars.columns]
    if missing_cols:
         # Attempt to merge from first occurrence in df? No, spot_bars IS first occurrence.
         # Just fill defaults if missing.
         for c in missing_cols: spot_bars[c] = 0.0
         
    X_np = spot_bars[feature_cols].values.astype(np.float32)
    X_np = np.where(np.isfinite(X_np), X_np, 0.0) # Simple sanitize
    
    # Normalize
    if norm_stats:
        mu = np.array(norm_stats['median']).reshape(-1)
        mad = np.array(norm_stats['mad']).reshape(-1)
        # Handle shape mismatch if any
        if len(mu) != X_np.shape[1]: 
             # fallback
             mu = np.median(X_np, axis=0)
             mad = np.median(np.abs(X_np - mu), axis=0)
    else:
        mu = np.median(X_np, axis=0)
        mad = np.median(np.abs(X_np - mu), axis=0)
    
    mad = np.maximum(mad, 1e-6)
    X_norm = (X_np - mu) / (1.4826 * mad)
    X_norm = np.clip(X_norm, -10.0, 10.0)
    
    # Batch Inference
    print("Running Batched Inference...")
    SEQ_LEN = 256
    BATCH_SIZE = 128
    num_bars = len(spot_bars)
    all_policy = []
    
    X_tensor = torch.tensor(X_norm, device=device)
    
    for b_start in tqdm(range(SEQ_LEN, num_bars, BATCH_SIZE), desc="Inference"):
        b_end = min(b_start + BATCH_SIZE, num_bars)
        batch_seqs = []
        valid_indices = []
        
        for i in range(b_start, b_end):
            # Sequence: [i-SEQ_LEN : i]
            # Warning: checks bounds
            batch_seqs.append(X_tensor[i-SEQ_LEN : i])
            valid_indices.append(i)
            
        if not batch_seqs: continue
            
        batch_input = torch.stack(batch_seqs)
        with torch.no_grad():
            # Model returns tuple, 0-index is policy
            out = model(batch_input)[0].cpu().numpy()
            all_policy.append(out)
            
    # Concatenate all batches
    if all_policy:
        policy_matrix = np.concatenate(all_policy, axis=0)
    else:
        policy_matrix = np.zeros((0, 10))
        
    # Map back to bar index: policy_matrix[k] corresponds to spot_bars.iloc[SEQ_LEN + k]
    
    # 3. SIMULATION LOOP
    print("Starting M1 Simulation Loop...")
    
    equity = 100_000.0
    starting_equity = equity
    open_trades = [] # List of Trade objects
    closed_trades = [] # List of dicts
    equity_curve = [] # List of dicts
    
    # Warmup
    sim_start_idx = SEQ_LEN
    
    # Helpers
    def _sigmoid(x): return 1 / (1 + np.exp(-x))
    
    for i in tqdm(range(sim_start_idx, num_bars), desc="Simulating"):
         # Adjust index for policy
         pol_idx = i - SEQ_LEN
         if pol_idx >= len(policy_matrix): break
         
         # 1. Get Bar Data
         ts = spot_bars[time_col].iloc[i]
         spot = float(spot_bars['close'].iloc[i])
         
         # 2. Get Option Chain (O(1))
         s, e = ts_ranges.get(ts, (None, None))
         if s is None: 
             continue # No chain this minute?
             
         chain_slice = df.iloc[s:e]
         
         # 3. Build Marks (Transient)
         # dict comprehension is fast enough for ~100 items
         marks = dict(zip(chain_slice['option_symbol'], chain_slice['close']))
         
         # 4. Update Open Trades
         active_trades = []
         for tr in open_trades:
             tr.update_mark(ts, marks)
             
             # Check Risk Stop (5%)
             if tr.should_risk_close(equity):
                 tr.exit_dt = ts
                 tr.exit_reason = "RISK_5PCT"
                 tr.realized_pnl = tr.unrealized_pnl
                 tr.is_closed = True
                 
                 equity += tr.realized_pnl
                 # Log close
                 closed_trades.append({
                     'trade_id': tr.trade_id,
                     'entry_dt': tr.entry_dt,
                     'exit_dt': ts,
                     'pnl': tr.realized_pnl,
                     'pnl_pct': tr.pnl_pct * 100, # % of max risk
                     'reason': "RISK_STOP",
                     'max_dd': tr.max_dd_pct * 100,
                     'held_bars': -1 # TODO
                 })
                 continue # Trade is gone
            
             # Check Model Exit Signal (optional)
             # pol = policy_matrix[pol_idx]
             # exit_logit = pol[9]
             # if exit_logit > 1.0: ...
             
             # Check Expiration (DTE < 0.1)
             # Approximation: if we hold > DTE (in bars) ??
             # Better: check days passed.
             days_held = (pd.Timestamp(ts) - pd.Timestamp(tr.entry_dt)).total_seconds() / 86400
             if days_held > tr.dte_entry:
                 # Expired (Exit at mark)
                 tr.exit_dt = ts
                 tr.exit_reason = "EXPIRED"
                 tr.realized_pnl = tr.unrealized_pnl
                 tr.is_closed = True
                 equity += tr.realized_pnl
                 closed_trades.append({
                     'trade_id': tr.trade_id,
                     'entry_dt': tr.entry_dt,
                     'exit_dt': ts,
                     'pnl': tr.realized_pnl,
                     'pnl_pct': tr.pnl_pct * 100,
                     'reason': "EXPIRED",
                     'max_dd': tr.max_dd_pct * 100
                 })
                 continue
                 
             active_trades.append(tr)
         
         open_trades = active_trades
         
         # 5. Entry Logic
         # Only if no open positions (simplification for robustness first)
         if len(open_trades) == 0:
             pol = policy_matrix[pol_idx]
             entry_logit = pol[8]
             prob = _sigmoid(pol[4])
             conf = _sigmoid(pol[7])
             
             # Min thresholds
             if entry_logit > 0.0 and prob > 0.4 and conf > 0.35:
                 # Attempt Entry
                 # Decode Model Params
                 call_off = pol[0] * 5.0
                 put_off = pol[1] * 5.0
                 width = pol[2] * 10.0
                 dte = pol[3] * 45.0 # Denormalized
                 
                 # Leg Selection
                 legs = find_best_legs(chain_slice, spot, call_off, put_off, width)
                 
                 if legs:
                    # Calculate Credit
                    # Short Call + Short Put - Long Call - Long Put
                    cre = legs['short_call_close'] + legs['short_put_close'] - legs['long_call_close'] - legs['long_put_close']
                    
                    if cre > 0.10: # Min credit filter
                        max_loss = (legs['width'] - cre) * IC_MULTIPLIER * IC_CONTRACTS
                        net_credit = cre
                        trade_id = f"TR_{i}"
                        
                        new_trade = Trade(
                            trade_id, ts, legs, net_credit, max_loss, 
                            dte=dte,
                            scores={'entry': entry_logit, 'prob': prob}
                        )
                        open_trades.append(new_trade)
         
         # 6. Record Equity Curve
         unrealized = sum(t.unrealized_pnl for t in open_trades)
         equity_curve.append({
             'dt': ts,
             'equity': equity + unrealized,
             'cash': equity,
             'open_pnl': unrealized,
             'open_count': len(open_trades)
         })
         
    # End Simulation
    print(f"Simulation Complete. Final Equity: ${equity:,.2f}")
    
    # Helper to return list of values for legacy main compatibility?
    # Old main expects (equity_list, trades_list)
    # We should return simple lists or update main.
    # The old main did `equity[-1]`. 
    # Let's return just the equity values list for plot, and full trades dicts.
    equity_vals = [e['equity'] for e in equity_curve]
    
    # Save CSVs (GUI Artifacts) inside run_backtest?
    # Or rely on main. New Plan says "Implement GUI-Ready Exports".
    # Let's save them here to be sure.
    pd.DataFrame(equity_curve).to_csv(os.path.join(REPORTS_DIR, "equity_curve.csv"), index=False)
    pd.DataFrame(closed_trades).to_csv(os.path.join(REPORTS_DIR, "trades.csv"), index=False)
    
    return equity_vals, closed_trades

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CondorBrain Backtest V2.2")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV data")
    parser.add_argument("--data", type=str, default=None, help="Alias for --input")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--ruleset", type=str, default=None, help="Path to ruleset YAML")
    parser.add_argument("--rules-only", action="store_true", help="Run in logic-only mode without neural model")
    
    # New Toggles for Hybrid Logic
    parser.add_argument("--use-fuzzy-sizing", action="store_true", default=False, help="Enable 11-factor fuzzy position sizing")
    parser.add_argument("--use-trade-rules", action="store_true", default=False, help="Enable rule-based entry/exit logic")
    parser.add_argument("--use-diffusion", action="store_true", default=False, help="Enable diffusion-based parameter refinement")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of bars to simulate")
    
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
    input_override = args.data or args.input
    possible_data_paths = [
        input_override, # CLI override first
        DATA_PATH,
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
    use_data_path = None
    if input_override:
        if os.path.exists(input_override):
            use_data_path = input_override
            print(f"✅ Using CLI Data Override: {use_data_path}")
        else:
            print(f"❌ CLI Data Override not found: {input_override}")
    
    if not use_data_path:
        for p in possible_data_paths:
            if p and os.path.exists(p):
                use_data_path = p
                print(f"✅ Found Data: {p}")
                break
    
    if not use_data_path:
        print("❌ CRITICAL: No data found. Please upload `mamba_institutional_1m.csv`.")
        return

    # 1. Load Data & Compute Features
    df = load_data_and_features(use_data_path, rows=args.limit)
    if df is None or df.empty:
        print("❌ Data load failed.")
        return

    feature_cols = FEATURE_COLS_V22
    print(f"Feature Columns ({len(feature_cols)}): {feature_cols[:5]} ...")

    # 2. Rule Engine
    rule_signals = None
    ruleset = None
    if args.ruleset or os.path.exists(RULESET_PATH):
        r_path = args.ruleset if args.ruleset else RULESET_PATH
        df, rule_signals, ruleset = run_rule_engine(df, r_path)

    # 3. Model Load
    model = None
    model_path = args.model if args.model else MODEL_PATH
    norm_stats = {}
    
    if not args.rules_only:
        if not os.path.exists(model_path):
            # Try variations
            if os.path.exists(f"spy-iron-condor-trading/{model_path}"):
                model_path = f"spy-iron-condor-trading/{model_path}"
            elif os.path.exists("condor_brain.pth"):
                model_path = "condor_brain.pth"
            else:
                print(f"❌ Model not found at {model_path}")
                return
        
        print(f"Loading Model: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Determine State Dict format
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Check for normalization stats
        if isinstance(checkpoint, dict):
            # Check for scaler state
            ckpt_feature_cols = checkpoint.get("feature_cols")
            if ckpt_feature_cols:
                feature_cols = list(ckpt_feature_cols)
        # Dynamic Attributes
        model_input_dim = checkpoint.get("input_dim", len(feature_cols)) if isinstance(checkpoint, dict) else len(feature_cols)
        
        # Extract model config if available
        ckpt_config = {}
        training_config = {}
        if isinstance(checkpoint, dict):
            if "model_config" in checkpoint:
                ckpt_config = checkpoint["model_config"]
            elif "config" in checkpoint:
                ckpt_config = checkpoint["config"]
            if "training_config" in checkpoint:
                training_config = checkpoint["training_config"]
                
        # Default to 12 layers, 512 dim if not specified
        n_layers = int(ckpt_config.get("n_layers", ckpt_config.get("layers", 12)))
        d_model = int(ckpt_config.get("d_model", ckpt_config.get("dim", 512)))
        
        # Predicate Discovery Flags
        use_pred = training_config.get('use_predicate_discovery', False)
        if not use_pred: use_pred = ckpt_config.get('use_predicate_discovery', False)
        n_slots = training_config.get('predicate_slots', 2048)
        max_active = training_config.get('max_active_predicates', 256)

        if model_input_dim != len(feature_cols):
            print(f"⚠️ input_dim mismatch: checkpoint={model_input_dim} vs features={len(feature_cols)}; using checkpoint value.")
        
        print(f"Initializing CondorBrain: d_model={d_model}, n_layers={n_layers}, input_dim={model_input_dim}, predicates={use_pred}")
        
        # V2.2 Model with Neural CDE backbone
        model = CondorBrain(
            d_model=d_model, n_layers=n_layers,
            input_dim=model_input_dim,
            use_cde=True,            # Explicit: Use Neural CDE backbone
            use_vol_gated_attn=True, use_topk_moe=True, moe_n_experts=3, moe_k=1,
            use_predicate_discovery=use_pred,
            n_predicate_slots=n_slots,
            max_active_predicates=max_active
        )
            use_diffusion=True,
            diffusion_steps=50,      # Match training
            diffusion_horizon=1,     # Match training (was 32)
            diffusion_input_dim=10   # Match training (targets=10)
        ).to(DEVICE)
        
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded.")
        except Exception as e:
            print(f"Model load failed: {e}")
            return
        
        if isinstance(checkpoint, dict):
            if "median" in checkpoint and "mad" in checkpoint:
                norm_stats["median"] = checkpoint["median"]
                norm_stats["mad"] = checkpoint["mad"]

    # 4. Backtest
    generate_contract_snapshot(
        os.path.join(REPORTS_DIR, "..", "audit", "contract_snapshot.json"),
        repo_root if "repo_root" in globals() else os.getcwd(),
        feature_cols=feature_cols,
        checkpoint_path=model_path,
        extra={"mode": "backtest", "data_path": DATA_PATH},
    )

    equity, trades = run_backtest(
        df,
        rule_signals,
        model,
        feature_cols,
        DEVICE,
        ruleset,
        model_path=model_path,
        data_path=use_data_path,
        norm_stats=norm_stats,
        use_fuzzy_sizing=args.use_fuzzy_sizing,
        use_trade_rules=args.use_trade_rules,
        use_diffusion=args.use_diffusion,
        limit=args.limit
    )
    
    # 5. Report
    if not equity:
        print("⚠️ Simulation produced no data (0 bars simulated). Check data/limit settings.")
        return [], []

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

        try:
            attribution_path = os.path.join(REPORTS_DIR, "decision_factor_attribution.csv")
            if os.path.exists(DECISION_TRACE_PATH):
                generate_attribution_csv(DECISION_TRACE_PATH, attribution_path)
                validate_decision_factor_attribution(attribution_path)
                validate_decision_trace(DECISION_TRACE_PATH)
                print(f"Saved decision factor attribution to {attribution_path}")
            else:
                print("⚠️ decision_trace.jsonl not found; skipping attribution export.")
        except Exception as e:
            print(f"⚠️ Attribution export/validation failed: {e}")
        
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
if __name__ == "__main__":
    main()
