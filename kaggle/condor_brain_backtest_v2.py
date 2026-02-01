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
    # Engine requires Dict[str, Series]. df matches this interface (mostly).
    # We pass 'data' as df (which behaves like dict of Series).
    
    # NOTE: Engine.execute() might expect a dict of Series explicitly if strict.
    # But df keys() works.
    
    # To be safe, let's pass df (it has __getitem__).
    # Engine logic: self._compute_primitives(data) -> data[p_spec.inputs]
    
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
    # results is Dict[rule_id, {'signals': Series, 'blocked': Series}]
    
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

def run_backtest(df, rule_signals, model, feature_cols, device, ruleset=None, model_path=None, data_path=None, norm_stats=None, 
                 use_fuzzy_sizing=False, use_trade_rules=True, use_diffusion=False, limit=None):
    print("Starting Backtest Simulation (Unique-Bar Time Alignment)...")
    
    # Identify unique bars
    time_col = 'dt' if 'dt' in df.columns else 'timestamp'
    spot_key_cols = [time_col]
    unique_bars = df.drop_duplicates(subset=spot_key_cols).sort_values(time_col).reset_index(drop=True)
    num_bars = len(unique_bars)
    
    if limit:
        num_bars = min(num_bars, limit + SEQ_LEN)
        unique_bars = unique_bars.iloc[:num_bars].reset_index(drop=True)
        print(f"Limiting to {num_bars} unique bars.")

    # Pre-process Features (Robust Norm same as training) on UNIQUE BARS
    missing_cols = [c for c in feature_cols if c not in unique_bars.columns]
    if missing_cols:
        print(f"⚠️ Missing {len(missing_cols)} feature columns in unique_bars; filling with neutral defaults.")
        for col in missing_cols:
            unique_bars[col] = get_neutral_fill_value_v22(col)
            
    X_np = unique_bars[feature_cols].values.astype(np.float32)
    # Sanitize inf/-inf before semantic fill
    X_np = np.where(np.isfinite(X_np), X_np, np.nan)
    X_np = apply_semantic_nan_fill(X_np, feature_cols)

    # Normalize (match training if stats available)
    if norm_stats is not None and "median" in norm_stats and "mad" in norm_stats:
        mu = np.asarray(norm_stats["median"], dtype=np.float32)
        mad = np.asarray(norm_stats["mad"], dtype=np.float32)
        mu = np.squeeze(mu)
        mad = np.squeeze(mad)
        if mu.ndim != 1: mu = mu.reshape(-1)
        if mad.ndim != 1: mad = mad.reshape(-1)
    else:
        mu = np.median(X_np, axis=0) if len(X_np) > 0 else 0
        mad = np.median(np.abs(X_np - mu), axis=0)
        
    mad = np.maximum(mad, 1e-6)

    # Protect rule features to match training normalization
    rule_idx = [feature_cols.index(c) for c in RULE_FEATURES if c in feature_cols]
    if rule_idx and hasattr(mu, "__len__") and len(mu) == X_np.shape[1]:
        for idx in rule_idx:
            mu[idx] = 0.0
            mad[idx] = 1.0 / 1.4826
    
    X_norm = (X_np - mu) / (1.4826 * mad)
    X_norm = np.clip(X_norm, -10.0, 10.0)
    
    print(f"Feature Statistics check: Mean={np.mean(X_norm):.4f}, Std={np.std(X_norm):.4f}")
    if np.abs(np.mean(X_norm)) > 1.0:
        print("⚠️ WARNING: Features are not centered near 0. Model inputs might be drifted.")
    
    X_tensor = torch.tensor(X_norm, device=device)
    
    # Settings
    SEQ_LEN = 256
    capital = 100_000.0
    starting_capital = capital
    position = 0 # 0=None, 1=Long Iron Condor
    equity_curve = []
    trades = []
    open_trade = None
    last_exit_bar = None
    
    # Trade state tracking (for expiration)
    trade_entry_bar = None
    trade_dte = None  # Days to expiration
    BARS_PER_DAY = 390  # 1-min bars per trading day (6.5 hours)
    DEFAULT_DTE = 14  # Default 14 DTE if model output is invalid
    
    model.eval()
    
    # Iterate
    num_bars = len(unique_bars)
    
    print(f"Simulating {num_bars} unique bars...")
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

    # Bar-level trace file
    if BAR_TRACE_ENABLED:
        with open(BAR_TRACE_PATH, "w", encoding="utf-8") as f:
            f.write("")

    def _feature_snapshot(x_seq_tensor, top_n=TOP_N_FEATURES):
        vals = x_seq_tensor[0, -1, :].detach().cpu().numpy()
        if top_n is not None and top_n > 0:
            idx = np.argsort(np.abs(vals))[-top_n:]
        else:
            idx = np.arange(len(vals))
        snap = {feature_cols[j]: float(vals[j]) for j in idx}
        return snap

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

    def _entry_win_from_path(r_path, alpha_r=ENTRY_ALPHA_R, beta_r=ENTRY_BETA_R):
        tau_pos = None
        tau_neg = None
        for idx, r_val in enumerate(r_path):
            if tau_pos is None and r_val >= alpha_r:
                tau_pos = idx
            if tau_neg is None and r_val <= -beta_r:
                tau_neg = idx
            if tau_pos is not None and tau_neg is not None:
                break
        if tau_pos is None and tau_neg is None:
            return None
        if tau_pos is None:
            return False
        if tau_neg is None:
            return True
        return tau_pos < tau_neg

    def _exit_win(r_exit, r_future, dd_future, lam=EXIT_RISK_LAMBDA):
        if r_future is None:
            return None
        u_exit = r_exit
        u_hold = r_future - lam * max(0.0, r_exit - dd_future)
        return u_exit > u_hold

    def _sizing_win(r_exit, min_r, size_mult, lam=SIZING_RISK_LAMBDA):
        if size_mult is None:
            return None
        dd = max(0.0, -min_r)
        u_size = r_exit * size_mult - lam * dd * size_mult
        u_base = r_exit - lam * dd
        if size_mult == 1.0:
            return None
        return u_size > u_base
    
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
    
    # --- BATCHED OPTIMIZATION ---
    print(f"Pre-computing model outputs for {num_bars - SEQ_LEN} UNIQUE bars (Batch Size: 64)...")
    BATCH_SIZE = 64
    all_policy_outputs = []
    
    # We slice the tensor into batches of sequences
    # Sequence indices start from SEQ_LEN to num_bars-1
    for b_start in tqdm(range(SEQ_LEN, num_bars - 1, BATCH_SIZE), desc="Batched Inference"):
        b_end = min(b_start + BATCH_SIZE, num_bars - 1)
        batch_seqs = []
        for j in range(b_start, b_end):
            batch_seqs.append(X_tensor[j - SEQ_LEN : j])
        
        batch_x = torch.stack(batch_seqs).to(device)
        with torch.no_grad():
            # CondorBrain forward returns (policy_outputs, regime, horizon, features, ...)
            out_tuple = model(batch_x)
            policy_batch = out_tuple[0].cpu().numpy()
            all_policy_outputs.append(policy_batch)
            
    all_policy_outputs = np.concatenate(all_policy_outputs, axis=0)
    print(f"✅ Pre-computation complete. Starting simulation loop...")

    # === DEBUG: Open log file and create helper ===
    debug_log = None
    if DEBUG_INFERENCE:
        debug_log = open(DEBUG_LOG_FILE, 'w', encoding='utf-8')
        debug_log.write(f"DEBUG INFERENCE LOG - {pd.Timestamp.now()}\n")
        debug_log.write(f"Model: {model_path}\n")
        debug_log.write(f"Unique bars: {len(unique_bars)}\n")
        debug_log.write("=" * 80 + "\n\n")
        print(f"📝 Debug output will be written to: {DEBUG_LOG_FILE}")

    def debug_print(msg):
        """Print to console and write to debug log file."""
        print(msg)
        if debug_log:
            debug_log.write(msg + "\n")

    # === DEBUG: Feature statistics collection ===
    if DEBUG_INFERENCE:
        debug_stats = {
            'prob_profit': [], 'confidence': [], 'entry_prob': [], 'exit_prob': [],
            'ivr': [], 'rsi': [], 'adx': [], 'friction_ratio': [], 'exec_allow': [], 'gap_risk': [],
            'gate_reasons': {},  # Count of each gate reason
            'total_bars': 0, 'bars_with_position_0': 0, 'bars_blocked': 0
        }

    # === CACHE OPTION CHAINS FOR FAST LOOKUP ===
    print("   Caching option chains for high-fidelity lookup...")
    chains_by_ts = {ts: group for ts, group in df.groupby(time_col)}

    # === CACHE OPTION PRICES FOR O(1) LOOKUP ===
    print("   Building P&L mark cache (O(1) lookup)...")
    # Mapping (timestamp, symbol) -> close price for instant MTM
    mark_cache = {}
    for row in tqdm(df[[time_col, 'option_symbol', 'close']].itertuples(index=False), total=len(df), desc="Pricing Cache"):
        mark_cache[(getattr(row, time_col), row.option_symbol)] = row.close

    # Track last known price per symbol for gaps
    last_known_mark = {}

    # Start from SEQ_LEN
    for i in tqdm(range(SEQ_LEN, num_bars - 1), desc="Simulating"):
        # 1. State
        if i >= len(all_policy_outputs): break
        pol = all_policy_outputs[i - SEQ_LEN]
        ts = unique_bars[time_col].iloc[i]
        spot = unique_bars['close'].iloc[i]
        
        all_outputs = {
            'call_off': pol[0], 'put_off': pol[1], 'width': pol[2], 
            'te': pol[3] * 45.0,  # DENORMALIZE DTE: 0-1 -> 0-45 days
            'prob_profit': pol[4], 'expected_roi': pol[5], 'max_loss_pct': pol[6],
            'confidence': pol[7], 'entry_logit': pol[8], 'exit_logit': pol[9],
        }
        
        def _sigmoid(v): return 1.0 / (1.0 + np.exp(-v))
        prob_profit = float(all_outputs['prob_profit'])
        confidence = float(all_outputs['confidence'])
        if prob_profit < 0 or prob_profit > 1: prob_profit = _sigmoid(prob_profit)
        if confidence < 0 or confidence > 1: confidence = _sigmoid(confidence)

        # Rule Engine Signal
        rule_block = float(unique_bars["rule_block_any"].iloc[i]) if "rule_block_any" in unique_bars.columns else 0.0
        rule_long = float(unique_bars["rule_long_consensus"].iloc[i]) if "rule_long_consensus" in unique_bars.columns else 0.0
        rule_short = float(unique_bars["rule_short_consensus"].iloc[i]) if "rule_short_consensus" in unique_bars.columns else 0.0
        rule_exit = float(unique_bars["rule_exit_consensus"].iloc[i]) if "rule_exit_consensus" in unique_bars.columns else 0.0
        net_rule_signal = rule_long - rule_short

        action = 0
        rejection_reason = None
        gate_reasons = []
        entry_score = 0
        pnl_dollar = 0.0
        r_normalized = 0.0
        entry_factors = []

        if position == 0:
            # 2. Fuzzy Entry Scoring
            # confidence (30), prob_profit (30), rules (30), logit (10)
            if confidence > 0.6:   entry_score += 30; entry_factors.append("Conf:HIGH")
            elif confidence > 0.4: entry_score += 20; entry_factors.append("Conf:MED")
            
            if prob_profit > 0.5:   entry_score += 30; entry_factors.append("Prob:HIGH")
            elif prob_profit > 0.4: entry_score += 20; entry_factors.append("Prob:MED")
            
            if net_rule_signal > 0.4:  entry_score += 30; entry_factors.append("Rules:BULL")
            elif net_rule_signal >= 0: entry_score += 15; entry_factors.append("Rules:NEUT")
            
            if _sigmoid(all_outputs['entry_logit'] or 0) > 0.5: entry_score += 10

            ENTRY_THRESHOLD = 40
            if entry_score < ENTRY_THRESHOLD: gate_reasons.append(f"SCORE_LOW({entry_score})")
            if rule_block > 0: gate_reasons.append("BLOCK_SIGNAL")
            if last_exit_bar and (i - last_exit_bar) < 15: gate_reasons.append("COOLDOWN")

            if not gate_reasons:
                chain = chains_by_ts.get(ts)
                if chain is not None:
                    legs = find_best_legs(chain, spot, all_outputs['call_off'], all_outputs['put_off'], all_outputs['width'])
                    if legs:
                        # Entry pricing from actual close prices
                        entry_credit = (legs['short_call_close'] + legs['short_put_close'] - 
                                        legs['long_call_close'] - legs['long_put_close'])
                        if entry_credit > 0.10:
                            trade_num = len([t for t in trades if t.get('action') == 'OPEN']) + 1
                            open_trade = {
                                'idx': i, 'type': 'IRON_CONDOR', 'action': 'OPEN', 'trade_id': f"IC-{trade_num}",
                                'ts': ts, 'spot': spot, 'entry_credit_per_leg': entry_credit,
                                'credit_total': entry_credit * IC_CONTRACTS * IC_MULTIPLIER,
                                'max_loss': (legs['width'] - entry_credit) * IC_CONTRACTS * IC_MULTIPLIER,
                                'dte': float(all_outputs['te'] or 14.0), 'entry_score': entry_score,
                                'sc_sym': legs['short_call_symbol'], 'lc_sym': legs['long_call_symbol'],
                                'sp_sym': legs['short_put_symbol'], 'lp_sym': legs['long_put_symbol'],
                                'short_call': legs['short_call'], 'short_put': legs['short_put'],
                                'long_call': legs['long_call'], 'long_put': legs['long_put'],
                                'conf': confidence, 'prob': prob_profit, 'r_path': []
                            }
                            trades.append(open_trade)
                            position, trade_entry_bar, action = 1, i, 1
                            print(f"  >> ENTER #{trade_num} @ {ts}: Spot ${spot:.2f} | Credit ${entry_credit:.2f} | DTE {open_trade['dte']:.1f}")
                        else: rejection_reason = "LOW_CREDIT"
                    else: rejection_reason = "NO_LEGS_MATCH"
                else: rejection_reason = "CHAIN_MISSING"

        elif position == 1:
            # 3. O(1) Pricing via mark_cache
            entry = open_trade
            symbols = [entry['sc_sym'], entry['lc_sym'], entry['sp_sym'], entry['lp_sym']]
            marks = []
            for s in symbols:
                m = mark_cache.get((ts, s))
                if m is None: m = last_known_mark.get(s, 0.0)
                else: last_known_mark[s] = m
                marks.append(m)
            
            sc_m, lc_m, sp_m, lp_m = marks
            current_cost = (sc_m + sp_m - lc_m - lp_m)
            pnl_dollar = (entry['entry_credit_per_leg'] - current_cost) * IC_CONTRACTS * IC_MULTIPLIER
            r_normalized = pnl_dollar / entry['max_loss'] if entry['max_loss'] > 0 else 0.0
            entry['r_path'].append(float(r_normalized))

            # Exit check
            days_held = (i - trade_entry_bar) / BARS_PER_DAY
            remaining_dte = entry['dte'] - days_held
            exit_reason = None
            ep = _sigmoid(all_outputs['exit_logit'] or 0)
            
            if remaining_dte <= 0: exit_reason = "EXPIRATION"
            elif rule_block > 0.5: exit_reason = "RULE_BLOCK"
            elif rule_exit > 0.5:  exit_reason = "RULE_EXIT"
            elif ep > 0.6:         exit_reason = f"MODEL_SIGNAL({ep:.2f})"
            elif r_normalized < -0.8: exit_reason = "STOP_LOSS"
            elif r_normalized > 0.5:  exit_reason = "TAKE_PROFIT"

            if exit_reason:
                capital += pnl_dollar
                stats['total_trades'] += 1
                if pnl_dollar > 0: stats['winners'] += 1; stats['total_win_dollar'] += pnl_dollar
                else: stats['losers'] += 1; stats['total_loss_dollar'] += abs(pnl_dollar)
                
                stats['peak_capital'] = max(stats['peak_capital'], capital)
                stats['max_dd_pct'] = max(stats['max_dd_pct'], (stats['peak_capital']-capital)/stats['peak_capital']*100 if stats['peak_capital']>0 else 0)
                
                print(f"  << EXIT @ {ts}: PnL ${pnl_dollar:,.0f} | Reason: {exit_reason} | WR: {stats['winners']/stats['total_trades']*100:.1f}%")
                
                close_rec = entry.copy()
                close_rec.update({'action': 'CLOSE', 'exit_ts': ts, 'exit_spot': spot, 'exit_reason': exit_reason, 'realized_pnl': pnl_dollar})
                trades.append(close_rec)
                position, trade_entry_bar, action = 0, None, -1
                last_exit_bar = i
        spot = df['close'].iloc[i]
        entry_logit_val = all_outputs.get('entry_logit')
        exit_logit_val = all_outputs.get('exit_logit')
        entry_exit_str = f"{entry_logit_val:.3f} / {exit_logit_val:.3f}" if entry_logit_val is not None else "N/A (legacy)"
        log_lines = [
            f"\n--- Bar {i} | Spot: ${spot:.2f} | Position: {position} ---",
            f"  Model Outputs (10): {pol[:10]}",
            f"  Confidence: {confidence:.4f} | Prob_Profit: {prob_profit:.4f}",
            f"  Entry/Exit Logits: {entry_exit_str}",
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
        
        # Print to console (always log if limit is set, otherwise first 50)
        show_console = (limit is not None) or (logged_count < MAX_LOGS)
        if show_console:
            for line in log_lines:
                print(line)
            logged_count += 1

        # Full bar trace (JSONL)
        if BAR_TRACE_ENABLED:
            time_col = 'dt' if 'dt' in df.columns else ('timestamp' if 'timestamp' in df.columns else None)
            bar_ts = str(df[time_col].iloc[i]) if time_col else None
            row_data = _json_safe_dict(df.iloc[i].to_dict()) if BAR_TRACE_INCLUDE_ROW else {}
            legs = []
            if position == 1 and open_trade:
                legs = [
                    {"right": "C", "side": "SELL", "strike": float(open_trade.get('short_call', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "C", "side": "BUY", "strike": float(open_trade.get('long_call', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "P", "side": "SELL", "strike": float(open_trade.get('short_put', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "P", "side": "BUY", "strike": float(open_trade.get('long_put', 0.0)), "qty": int(IC_CONTRACTS)}
                ]
            bar_record = {
                "schema_version": "1.0",
                "idx": int(i),
                "ts": bar_ts,
                "spot": float(spot),
                "position": int(position),
                "action": "ENTER" if action == 1 else ("EXIT" if action == -1 else "HOLD"),
                "rejection_reason": rejection_reason or "",
                "gate_reasons": gate_reasons if position == 0 else [],
                "entry_score": float(entry_score) if position == 0 else (open_trade.get("entry_score") if open_trade else None),
                "entry_factors": entry_factors if position == 0 else [],
                "model_outputs": {
                    "call_off": float(all_outputs['call_off'] or 0.0),
                    "put_off": float(all_outputs['put_off'] or 0.0),
                    "width": float(all_outputs['width'] or 0.0),
                    "te": float(all_outputs['te'] or 0.0),
                    "prob_profit": float(prob_profit),
                    "expected_roi": float(all_outputs['expected_roi'] or 0.0),
                    "max_loss_pct": float(all_outputs['max_loss_pct'] or 0.0),
                    "confidence": float(confidence),
                },
                "rule_signals": {
                    "net_rule_signal": float(net_rule_signal),
                    "rule_long_consensus": float(rule_long),
                    "rule_short_consensus": float(rule_short),
                    "rule_exit_consensus": float(rule_exit),
                    "rule_block_any": float(rule_block),
                    "active_rules": active_rules,
                },
                "options_legs": legs,
                "feature_snapshot": _feature_snapshot(x_seq),
                "row": row_data,
                "learned_conditions": {
                    "surrogate_rules": None,
                    "attribution_method": "NOT_AVAILABLE_IN_BACKTEST",
                },
            }
            _append_jsonl(BAR_TRACE_PATH, bar_record)

        # Decision trace for every bar
        if trace_logger is not None and TRACE_PER_BAR:
            legs = []
            trade_id = f"BAR-{i}"
            dte_entry_val = None
            dte_remaining_val = None
            entry_score_val = None
            pos_size_pct_val = None
            if position == 1 and open_trade:
                trade_id = open_trade.get('trade_id', trade_id)
                legs = [
                    {"right": "C", "side": "SELL", "strike": float(open_trade.get('short_call', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "C", "side": "BUY", "strike": float(open_trade.get('long_call', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "P", "side": "SELL", "strike": float(open_trade.get('short_put', 0.0)), "qty": int(IC_CONTRACTS)},
                    {"right": "P", "side": "BUY", "strike": float(open_trade.get('long_put', 0.0)), "qty": int(IC_CONTRACTS)}
                ]
                dte_entry_val = open_trade.get('dte', None)
                bars_held = i - trade_entry_bar if trade_entry_bar else 0
                days_held = bars_held / BARS_PER_DAY
                dte_remaining_val = dte_entry_val - days_held if dte_entry_val else None
                entry_score_val = open_trade.get('entry_score', None)
                pos_size_pct_val = open_trade.get('pos_size_pct', None)

            scope = "EXIT" if position == 1 else "ENTRY"
            feature_map = _feature_snapshot(x_seq)
            _emit_trace_event(
                scope=scope,
                decision_type="HOLD",
                intent="HOLD",
                trade_id=trade_id,
                spot_val=spot,
                legs=legs,
                entry_score_val=entry_score_val,
                prob_val=prob_profit,
                conf_val=confidence,
                net_rule_signal_val=net_rule_signal,
                dte_entry_val=dte_entry_val,
                dte_remaining_val=dte_remaining_val,
                pnl_val=None,
                max_loss_val=open_trade.get('max_loss', None) if open_trade else None,
                pos_size_pct_val=pos_size_pct_val,
                active_rules_list=active_rules,
                feature_map=feature_map,
                reason_text="bar_trace"
            )
            
        # 5. Iron Condor P&L Simulation
        # IC P&L: track mark-to-market for diagnostics only
        spot = df['close'].iloc[i]
        
        if position == 1 and trade_entry_bar is not None:
            # Get open trade data
            open_trade = [t for t in trades if t.get('action') == 'OPEN'][-1] if trades else None
            if open_trade:
                # Update r_path for entry/exit/sizing outcomes
                days_elapsed = (i - trade_entry_bar) / BARS_PER_DAY
                mtm_pnl = estimate_condor_pnl(
                    spot=spot,
                    short_call=open_trade['short_call'],
                    long_call=open_trade['long_call'],
                    short_put=open_trade['short_put'],
                    long_put=open_trade['long_put'],
                    credit_received=open_trade['credit'],
                    max_loss=open_trade['max_loss'],
                    days_held=days_elapsed,
                    total_dte=open_trade['dte']
                )
                r_val = mtm_pnl / open_trade['max_loss'] if open_trade['max_loss'] > 0 else 0.0
                open_trade['r_path'].append(float(r_val))
        
        stats['total_pnl_dollar'] = capital - starting_capital
        mtm_equity = capital
        if position == 1 and trade_entry_bar is not None and open_trade:
            days_elapsed = (i - trade_entry_bar) / BARS_PER_DAY
            mtm_pnl = estimate_condor_pnl(
                spot=spot,
                short_call=open_trade['short_call'],
                long_call=open_trade['long_call'],
                short_put=open_trade['short_put'],
                long_put=open_trade['long_put'],
                credit_received=open_trade['credit'],
                max_loss=open_trade['max_loss'],
                days_held=days_elapsed,
                total_dte=open_trade['dte']
            )
            mtm_equity = capital + mtm_pnl
        equity_curve.append(mtm_equity)
    
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

    # === DEBUG: Print Feature & Model Output Summary ===
    if DEBUG_INFERENCE and debug_stats['total_bars'] > 0:
        debug_print("\n" + "=" * 80)
        debug_print("DEBUG INFERENCE SUMMARY")
        debug_print("=" * 80)

        debug_print(f"\n  SIMULATION STATS:")
        debug_print(f"    Total bars analyzed (position=0): {debug_stats['bars_with_position_0']:,}")
        debug_print(f"    Bars blocked (no entry):          {debug_stats['bars_blocked']:,}")
        debug_print(f"    Bars with entry:                  {debug_stats['bars_with_position_0'] - debug_stats['bars_blocked']:,}")
        block_rate = debug_stats['bars_blocked'] / max(1, debug_stats['bars_with_position_0']) * 100
        debug_print(f"    Block rate:                       {block_rate:.1f}%")

        debug_print(f"\n  GATE REASON BREAKDOWN:")
        if debug_stats['gate_reasons']:
            sorted_reasons = sorted(debug_stats['gate_reasons'].items(), key=lambda x: -x[1])
            for reason, count in sorted_reasons:
                pct = count / debug_stats['bars_blocked'] * 100 if debug_stats['bars_blocked'] > 0 else 0
                debug_print(f"    {reason:<20}: {count:>8,} ({pct:>5.1f}%)")
        else:
            debug_print(f"    (No blocks recorded)")

        def _debug_print_dist(name, values, threshold=None, threshold_name="threshold"):
            if values:
                arr = np.array(values)
                below = np.sum(arr < threshold) if threshold else 0
                pct_below = below / len(arr) * 100 if threshold else 0
                debug_print(f"    {name:<18}: min={arr.min():.4f}, mean={arr.mean():.4f}, max={arr.max():.4f}, std={arr.std():.4f}")
                if threshold:
                    debug_print(f"      Below {threshold_name} ({threshold}): {below:,} ({pct_below:.1f}%)")
            else:
                debug_print(f"    {name:<18}: N/A (not in data)")

        debug_print(f"\n  MODEL OUTPUT DISTRIBUTIONS:")
        _debug_print_dist("prob_profit", debug_stats['prob_profit'], PROB_ENTRY_MIN, "PROB_ENTRY_MIN")
        _debug_print_dist("confidence", debug_stats['confidence'], CONF_ENTRY_MIN, "CONF_ENTRY_MIN")
        _debug_print_dist("entry_prob", debug_stats['entry_prob'], 0.48, "MED threshold")
        _debug_print_dist("exit_prob", debug_stats['exit_prob'])

        debug_print(f"\n  KEY INPUT FEATURE DISTRIBUTIONS:")
        _debug_print_dist("ivr", debug_stats['ivr'], 30, "IVR>30 gate")
        _debug_print_dist("rsi", debug_stats['rsi'])
        _debug_print_dist("adx", debug_stats['adx'], 30, "ADX<30 gate")
        _debug_print_dist("friction_ratio", debug_stats['friction_ratio'], 1.0, "friction<1.0 gate")
        _debug_print_dist("exec_allow", debug_stats['exec_allow'], 0.5, "exec>0.5 gate")
        _debug_print_dist("gap_risk", debug_stats['gap_risk'], 0.8, "gap<0.8 gate")

        # Diagnosis
        debug_print(f"\n  DIAGNOSIS:")
        issues = []
        if debug_stats['prob_profit']:
            prob_mean = np.mean(debug_stats['prob_profit'])
            if prob_mean < PROB_ENTRY_MIN:
                issues.append(f"prob_profit mean ({prob_mean:.4f}) is BELOW threshold ({PROB_ENTRY_MIN})")
        if debug_stats['confidence']:
            conf_mean = np.mean(debug_stats['confidence'])
            if conf_mean < CONF_ENTRY_MIN:
                issues.append(f"confidence mean ({conf_mean:.4f}) is BELOW threshold ({CONF_ENTRY_MIN})")
        if debug_stats['exec_allow']:
            exec_blocked = np.sum(np.array(debug_stats['exec_allow']) <= 0.5) / len(debug_stats['exec_allow']) * 100
            if exec_blocked > 50:
                issues.append(f"exec_allow blocks {exec_blocked:.1f}% of bars (friction gate)")
        if debug_stats['gap_risk']:
            gap_blocked = np.sum(np.array(debug_stats['gap_risk']) >= 0.8) / len(debug_stats['gap_risk']) * 100
            if gap_blocked > 20:
                issues.append(f"gap_risk blocks {gap_blocked:.1f}% of bars")
        if debug_stats['friction_ratio']:
            friction_blocked = np.sum(np.array(debug_stats['friction_ratio']) >= 1.0) / len(debug_stats['friction_ratio']) * 100
            if friction_blocked > 30:
                issues.append(f"friction_ratio blocks {friction_blocked:.1f}% of bars")

        if issues:
            for issue in issues:
                debug_print(f"    WARNING: {issue}")
        else:
            debug_print(f"    OK: No obvious issues detected. Check ENTRY_THRESHOLD or scoring logic.")

        debug_print("=" * 80 + "\n")

    # Close debug log file
    if debug_log:
        debug_log.write(f"\nLog closed at {pd.Timestamp.now()}\n")
        debug_log.close()
        print(f"Debug log saved to: {DEBUG_LOG_FILE}")

    return equity_curve, trades

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
            print(f"❌ ERROR: CLI Data path not found: {input_override}")
            return

    if not use_data_path:
        for p in possible_data_paths[1:]: # Skip the first which was input_override
            if p and os.path.exists(p):
                use_data_path = p
                print(f"Found data at: {use_data_path}")
                break
    
    if not use_data_path:
        use_data_path = DATA_PATH
        print(f"Using default data path: {use_data_path}")
            
    # 1. Pipeline
    # Load limited rows for test? Or all.
    df = load_data_and_features(use_data_path)  # Load ALL rows
    
    # 2. Rules
    ruleset_path = args.ruleset if args.ruleset else RULESET_PATH
    df, rule_signals, ruleset = run_rule_engine(df, ruleset_path)
    
    # 3. Model
    # 3. Model
    POSSIBLE_PATHS = [
        args.model, # CLI override first
        # Epoch 4 (New Request)
        "models/old models/condor_brain_retrain_e4.pth",
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
    
    if args.rules_only:
        print("⚠️ RULES-ONLY MODE: Skipping model search.")
        model_path = "RULES_ONLY"
    else:
        model_path = MODEL_PATH
        for p in POSSIBLE_PATHS:
            if p and os.path.exists(p):
                model_path = p
                break
            
    # Align feature schema with training (52 base + 4 rule consensus)
    feature_cols = FEATURE_COLS_V22 + RULE_FEATURES

    norm_stats = {}

    if args.rules_only:
        print("🛠️ RULES-ONLY MODE: Initializing Dummy CondorBrain...")
        n_layers, d_model, model_input_dim = 1, 32, len(feature_cols)
        # Minimal dummy model for compatibility (with CDE backbone)
        model = CondorBrain(d_model=d_model, n_layers=n_layers, input_dim=model_input_dim, use_cde=True).to(DEVICE)
    else:
        print(f"Loading Model from {model_path}...")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}.")
            return
            
        # Set weights_only=False to support numpy/legacy checkpoints
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        if isinstance(checkpoint, dict):
            ckpt_feature_cols = checkpoint.get("feature_cols")
            if ckpt_feature_cols:
                feature_cols = list(ckpt_feature_cols)
        # Dynamic Attributes
        model_input_dim = checkpoint.get("input_dim", len(feature_cols)) if isinstance(checkpoint, dict) else len(feature_cols)
        
        # Extract model config if available
        ckpt_config = {}
        if isinstance(checkpoint, dict):
            if "model_config" in checkpoint:
                ckpt_config = checkpoint["model_config"]
            elif "config" in checkpoint:
                ckpt_config = checkpoint["config"]
                
        # Default to 12 layers, 512 dim if not specified
        n_layers = int(ckpt_config.get("n_layers", ckpt_config.get("layers", 12)))
        d_model = int(ckpt_config.get("d_model", ckpt_config.get("dim", 512)))

        if model_input_dim != len(feature_cols):
            print(f"⚠️ input_dim mismatch: checkpoint={model_input_dim} vs features={len(feature_cols)}; using checkpoint value.")
        
        print(f"Initializing CondorBrain: d_model={d_model}, n_layers={n_layers}, input_dim={model_input_dim}")
        
        # V2.2 Model with Neural CDE backbone
        model = CondorBrain(
            d_model=d_model, n_layers=n_layers,
            input_dim=model_input_dim,
            use_cde=True,            # Explicit: Use Neural CDE backbone (default, but be explicit)
            use_vol_gated_attn=True, use_topk_moe=True, moe_n_experts=3, moe_k=1,
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


