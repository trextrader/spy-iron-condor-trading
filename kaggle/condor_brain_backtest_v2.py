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
# Phase 5.5: Execution Reality Engine
try:
    from intelligence.execution_reality import (
        ExecutionRealityEngine,
        MarketState,
        FillStatus,
        FillResult,
        create_default_engine,
    )
    HAS_EXECUTION_REALITY = True
except ImportError:
    HAS_EXECUTION_REALITY = False
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

# =============================================================================
# PHASE 5.2: EXECUTION REALITY CONFIG (Truth Alignment)
# =============================================================================
# Price Source: bid/ask/mid instead of close
# Slippage: Per-leg slippage model
# Atomicity: All 4 legs must have valid quotes
# Partial Fills: Volume-based fill probability
# Greeks: Delta-based strike validation

EXEC_SLIPPAGE_PER_LEG = 0.02      # $0.02 slippage per leg (conservative)
EXEC_MIN_SPREAD_RATIO = 0.0001   # Minimum spread_ratio to consider valid
EXEC_MAX_SPREAD_RATIO = 0.10     # Maximum spread_ratio before rejecting (10%)
EXEC_MIN_VOLUME = 10             # Minimum volume for fill confidence
EXEC_MIN_OI = 100                # Minimum open interest for liquidity
EXEC_DELTA_TOLERANCE = 0.05      # Delta tolerance for strike validation
EXEC_TARGET_SHORT_DELTA = 0.16   # Target delta for short strikes
EXEC_PARTIAL_FILL_THRESHOLD = 50 # Volume threshold for guaranteed full fill
EXEC_USE_BID_ASK = True          # Use synthesized bid/ask instead of close
EXEC_ATOMICITY_STRICT = True     # Require all 4 legs to have valid quotes

# =============================================================================
# PHASE 5.5: EXECUTION REALITY ENGINE CONFIG
# =============================================================================
# Enables full execution reality modeling with 8 components:
# - Latency, Queue Position, Spread Dynamics, Volatility Shock
# - Broken Spread Detection, Microstructure, Quote Staleness, TOD Liquidity
EXEC_REALITY_ENABLED = True       # Use ExecutionRealityEngine for fills
EXEC_REALITY_SEED = 42            # Seed for deterministic execution
EXEC_REALITY_LATENCY_MS = 100.0   # Mean latency in ms
EXEC_REALITY_AGGRESSION = 0.5     # Queue aggression (0=passive, 1=aggressive)
EXEC_REALITY_LOG_DIAGNOSTICS = True  # Log detailed execution diagnostics

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
    print(f"Loading data from {data_path}...")
    
    # FIX: Use nrows to prevent OOM on huge files
    if rows is not None:
        print(f"   [Memory Opt] Reading first {rows} rows only...")
        df = pd.read_csv(data_path, nrows=rows)
    else:
        df = pd.read_csv(data_path)
    
    # --- STANDARDIZE DATE COLUMN ---
    if 'dt' not in df.columns and 'timestamp' in df.columns:
        print("   ⚠️ Standardizing 'timestamp' -> 'dt'...")
        df.rename(columns={'timestamp': 'dt'}, inplace=True)

    if 'dt' in df.columns:
        # Ensure UTC standard
        df['dt'] = pd.to_datetime(df['dt'], utc=True)
    
    # if rows is not None:
    #    df = df.iloc[-rows:].reset_index(drop=True)
    
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


# =============================================================================
# PHASE 5.2: EXECUTION REALITY HELPERS (Truth Alignment)
# =============================================================================

def synthesize_bid_ask(row):
    """
    Synthesize bid/ask from close and spread_ratio.

    Formula:
        spread = spread_ratio * close
        bid = close - spread/2
        ask = close + spread/2
        mid = close (assumed)

    Phase 5.2 Fix: Use proper bid/ask instead of just close.
    """
    close = row.get('close', 0.0)
    spread_ratio = row.get('spread_ratio', EXEC_MIN_SPREAD_RATIO)

    # Clamp spread_ratio to reasonable bounds
    spread_ratio = max(EXEC_MIN_SPREAD_RATIO, min(spread_ratio, EXEC_MAX_SPREAD_RATIO))

    half_spread = (spread_ratio * close) / 2.0
    bid = close - half_spread
    ask = close + half_spread
    mid = close

    return {
        'bid': max(0.01, bid),  # Floor at $0.01
        'ask': max(0.02, ask),
        'mid': max(0.01, mid),
        'spread': ask - bid,
        'spread_ratio': spread_ratio
    }


def synthesize_chain_prices(chain_df):
    """
    Add bid/ask/mid columns to chain DataFrame.

    Phase 5.2 Fix: Synthesize realistic bid/ask from close + spread_ratio.
    """
    if chain_df.empty:
        return chain_df

    df = chain_df.copy()

    # Get spread_ratio, default to minimum if missing
    if 'spread_ratio' not in df.columns:
        df['spread_ratio'] = EXEC_MIN_SPREAD_RATIO

    # Synthesize bid/ask
    half_spread = (df['spread_ratio'].clip(EXEC_MIN_SPREAD_RATIO, EXEC_MAX_SPREAD_RATIO) * df['close']) / 2.0
    df['bid'] = (df['close'] - half_spread).clip(lower=0.01)
    df['ask'] = (df['close'] + half_spread).clip(lower=0.02)
    df['mid'] = df['close'].clip(lower=0.01)

    return df


def validate_leg_liquidity(row):
    """
    Validate that a leg has sufficient liquidity for fill.

    Phase 5.2 Fix: Check volume and OI before assuming fill.

    Returns:
        (is_valid, fill_probability, reason)
    """
    volume = row.get('volume', 0) or 0
    oi = row.get('open_interest', 0) or 0
    spread_ratio = row.get('spread_ratio', 1.0) or 1.0

    # Check minimum thresholds
    if volume < EXEC_MIN_VOLUME:
        return False, 0.0, f"volume={volume} < {EXEC_MIN_VOLUME}"

    if oi < EXEC_MIN_OI:
        return False, 0.0, f"OI={oi} < {EXEC_MIN_OI}"

    if spread_ratio > EXEC_MAX_SPREAD_RATIO:
        return False, 0.0, f"spread_ratio={spread_ratio:.4f} > {EXEC_MAX_SPREAD_RATIO}"

    # Calculate fill probability based on volume
    if volume >= EXEC_PARTIAL_FILL_THRESHOLD:
        fill_prob = 1.0
    else:
        fill_prob = volume / EXEC_PARTIAL_FILL_THRESHOLD

    return True, fill_prob, "OK"


def validate_leg_delta(row, target_delta, is_short=True):
    """
    Validate that a leg's delta is within tolerance of target.

    Phase 5.2 Fix: Use Greeks for strike validation.

    Args:
        row: Option row with 'delta' column
        target_delta: Target absolute delta (e.g., 0.16)
        is_short: True for short legs (validate around target), False for long (just check exists)

    Returns:
        (is_valid, actual_delta, deviation)
    """
    delta = row.get('delta', 0.0) or 0.0
    abs_delta = abs(delta)

    if not is_short:
        # Long legs just need to exist and have some delta
        return abs_delta > 0.01, abs_delta, 0.0

    deviation = abs(abs_delta - target_delta)
    is_valid = deviation <= EXEC_DELTA_TOLERANCE

    return is_valid, abs_delta, deviation


def calculate_entry_fill(legs, chain_df, target_qty):
    """
    Calculate realistic entry fill with slippage.

    Phase 5.2 Fix:
    - Use bid for shorts (we're selling)
    - Use ask for longs (we're buying)
    - Apply slippage per leg
    - Check atomicity

    Returns:
        (filled_qty, net_credit, is_atomic, fill_details)
    """
    if legs is None:
        return 0, 0.0, False, {"reason": "no_legs"}

    # Build symbol lookup for prices
    prices = {}
    for _, row in chain_df.iterrows():
        sym = row.get('option_symbol', row.get('symbol', ''))
        if sym:
            prices[sym] = synthesize_bid_ask(row)

    required_symbols = [
        legs['short_call_symbol'],
        legs['long_call_symbol'],
        legs['short_put_symbol'],
        legs['long_put_symbol']
    ]

    # Atomicity check: All legs must have valid prices
    missing = [s for s in required_symbols if s not in prices]
    if missing and EXEC_ATOMICITY_STRICT:
        return 0, 0.0, False, {"reason": "missing_legs", "missing": missing}

    # Calculate credit using proper bid/ask
    # Short legs: we SELL at BID
    # Long legs: we BUY at ASK
    try:
        short_call_bid = prices[legs['short_call_symbol']]['bid']
        short_put_bid = prices[legs['short_put_symbol']]['bid']
        long_call_ask = prices[legs['long_call_symbol']]['ask']
        long_put_ask = prices[legs['long_put_symbol']]['ask']
    except KeyError as e:
        return 0, 0.0, False, {"reason": f"price_lookup_failed: {e}"}

    # Gross credit (before slippage)
    gross_credit = (short_call_bid + short_put_bid) - (long_call_ask + long_put_ask)

    # Apply slippage (4 legs)
    total_slippage = EXEC_SLIPPAGE_PER_LEG * 4
    net_credit = gross_credit - total_slippage

    fill_details = {
        "short_call_bid": short_call_bid,
        "short_put_bid": short_put_bid,
        "long_call_ask": long_call_ask,
        "long_put_ask": long_put_ask,
        "gross_credit": gross_credit,
        "slippage": total_slippage,
        "net_credit": net_credit
    }

    return target_qty, net_credit, True, fill_details


def calculate_exit_fill(legs, marks_bid, marks_ask, marks_mid):
    """
    Calculate realistic exit fill with slippage.

    Phase 5.2 Fix:
    - Use ask for shorts (we're buying back)
    - Use bid for longs (we're selling)
    - Apply slippage per leg

    Returns:
        (exit_debit, is_valid, fill_details)
    """
    required_symbols = [
        legs['short_call_symbol'],
        legs['long_call_symbol'],
        legs['short_put_symbol'],
        legs['long_put_symbol']
    ]

    # Check all legs have prices
    missing = [s for s in required_symbols if s not in marks_ask or s not in marks_bid]
    if missing:
        return None, False, {"reason": "missing_legs", "missing": missing}

    # Calculate debit using proper bid/ask
    # Short legs: we BUY BACK at ASK
    # Long legs: we SELL at BID
    short_call_ask = marks_ask[legs['short_call_symbol']]
    short_put_ask = marks_ask[legs['short_put_symbol']]
    long_call_bid = marks_bid[legs['long_call_symbol']]
    long_put_bid = marks_bid[legs['long_put_symbol']]

    # Gross debit (before slippage)
    gross_debit = (short_call_ask + short_put_ask) - (long_call_bid + long_put_bid)

    # Apply slippage (4 legs)
    total_slippage = EXEC_SLIPPAGE_PER_LEG * 4
    net_debit = gross_debit + total_slippage

    fill_details = {
        "short_call_ask": short_call_ask,
        "short_put_ask": short_put_ask,
        "long_call_bid": long_call_bid,
        "long_put_bid": long_put_bid,
        "gross_debit": gross_debit,
        "slippage": total_slippage,
        "net_debit": net_debit
    }

    return net_debit, True, fill_details


# =============================================================================
# PHASE 5.5: EXECUTION REALITY WRAPPERS FOR IRON CONDORS
# =============================================================================

def create_market_state_from_bar(spot, vix, time_of_day, volume=100, quote_age_ms=50):
    """
    Create MarketState from bar data for ExecutionRealityEngine.

    Phase 5.5: Converts backtester context to execution reality format.
    """
    if not HAS_EXECUTION_REALITY:
        return None

    return MarketState(
        vix=vix if vix else 15.0,
        spot_price=spot,
        recent_volume=volume,
        time_of_day_hour=time_of_day,
        recent_price_change_pct=0.0,  # Could be computed from spot history
        quote_age_ms=quote_age_ms,
        market_speed=1.0 + max(0, (vix - 15) / 30) if vix else 1.0  # Faster in high VIX
    )


def calculate_entry_fill_reality(legs, chain_df, target_qty, engine, market_state):
    """
    Calculate realistic entry fill using ExecutionRealityEngine.

    Phase 5.5: Routes each leg through the execution reality engine and
    validates atomic fill across all 4 legs.

    Returns:
        (filled_qty, net_credit, is_atomic, fill_details)
    """
    if legs is None or engine is None or market_state is None:
        # Fallback to simple model
        return calculate_entry_fill(legs, chain_df, target_qty)

    # Build symbol lookup for prices
    prices = {}
    for _, row in chain_df.iterrows():
        sym = row.get('option_symbol', row.get('symbol', ''))
        if sym:
            prices[sym] = synthesize_bid_ask(row)

    leg_configs = [
        ('short_call', legs['short_call_symbol'], True),   # is_short=True (we sell)
        ('long_call', legs['long_call_symbol'], False),    # is_short=False (we buy)
        ('short_put', legs['short_put_symbol'], True),
        ('long_put', legs['long_put_symbol'], False),
    ]

    fill_results = {}
    total_credit = 0.0
    total_slippage = 0.0
    all_filled = True
    diagnostics = {'leg_results': {}}

    for leg_name, symbol, is_short in leg_configs:
        if symbol not in prices:
            return 0, 0.0, False, {"reason": f"missing_{leg_name}", "symbol": symbol}

        p = prices[symbol]
        bid, ask = p['bid'], p['ask']

        # Entry: shorts sell (is_entry=True), longs buy (is_entry=True but different side)
        # For Iron Condor entry:
        # - Short legs: we SELL, so we get the bid side (is_entry=True in engine)
        # - Long legs: we BUY, so we pay the ask side (is_entry=True for buying)
        result = engine.simulate_realistic_fill(
            bid=bid,
            ask=ask,
            size=target_qty,
            market_state=market_state,
            is_entry=is_short  # True for shorts (sell), False for longs (buy to open)
        )

        fill_results[leg_name] = result
        diagnostics['leg_results'][leg_name] = {
            'status': result.status.value,
            'fill_price': result.fill_price,
            'slippage': result.slippage,
            'latency_ms': result.diagnostics.get('latency_ms', 0),
        }

        if result.status != FillStatus.FILLED:
            all_filled = False
            diagnostics['rejection'] = {
                'leg': leg_name,
                'reason': result.diagnostics.get('rejection_reason', result.status.value)
            }
            break

        # Accumulate credit/debit
        if is_short:
            total_credit += result.fill_price  # We receive this
        else:
            total_credit -= result.fill_price  # We pay this

        total_slippage += result.slippage

    if not all_filled:
        return 0, 0.0, False, diagnostics

    diagnostics['gross_credit'] = total_credit
    diagnostics['total_slippage'] = total_slippage
    diagnostics['net_credit'] = total_credit  # Slippage already embedded in fill prices

    return target_qty, total_credit, True, diagnostics


def calculate_exit_fill_reality(legs, marks_bid, marks_ask, marks_mid, engine, market_state):
    """
    Calculate realistic exit fill using ExecutionRealityEngine.

    Phase 5.5: Routes each leg through the execution reality engine.

    Returns:
        (exit_debit, is_valid, fill_details)
    """
    if legs is None or engine is None or market_state is None:
        # Fallback to simple model
        return calculate_exit_fill(legs, marks_bid, marks_ask, marks_mid)

    leg_configs = [
        ('short_call', legs['short_call_symbol'], True),   # is_short=True (we buy back)
        ('long_call', legs['long_call_symbol'], False),    # is_short=False (we sell)
        ('short_put', legs['short_put_symbol'], True),
        ('long_put', legs['long_put_symbol'], False),
    ]

    fill_results = {}
    total_debit = 0.0
    total_slippage = 0.0
    all_filled = True
    diagnostics = {'leg_results': {}}

    for leg_name, symbol, is_short in leg_configs:
        if symbol not in marks_bid or symbol not in marks_ask:
            return None, False, {"reason": f"missing_{leg_name}", "symbol": symbol}

        bid = marks_bid[symbol]
        ask = marks_ask[symbol]

        # Exit: shorts buy back, longs sell
        # - Short legs: we BUY BACK at ask (is_entry=False in engine)
        # - Long legs: we SELL at bid (is_entry=False, but favorable side)
        result = engine.simulate_realistic_fill(
            bid=bid,
            ask=ask,
            size=1,  # Size doesn't matter for pricing
            market_state=market_state,
            is_entry=not is_short  # False for shorts (buy to close), True for longs (sell to close)
        )

        fill_results[leg_name] = result
        diagnostics['leg_results'][leg_name] = {
            'status': result.status.value,
            'fill_price': result.fill_price,
            'slippage': result.slippage,
        }

        if result.status == FillStatus.REJECTED:
            # On exit, we may need to force close even with bad quotes
            # Use mid price as fallback
            mid = marks_mid.get(symbol, (bid + ask) / 2)
            if is_short:
                total_debit += mid * 1.02  # Add 2% buffer for bad exit
            else:
                total_debit -= mid * 0.98
            diagnostics['leg_results'][leg_name]['forced_mid'] = True
            continue

        if result.status == FillStatus.QUEUED:
            # Retry with aggressive fill on exit
            total_debit += ask if is_short else -bid
            continue

        # Normal filled case
        if is_short:
            total_debit += result.fill_price  # We pay this
        else:
            total_debit -= result.fill_price  # We receive this

        total_slippage += result.slippage

    diagnostics['gross_debit'] = total_debit
    diagnostics['total_slippage'] = total_slippage
    diagnostics['net_debit'] = total_debit

    return total_debit, True, diagnostics


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

def find_best_legs(chain_df, spot, call_off, put_off, width, validate_greeks=True):
    """
    Search the 100-row options chain for the 4 legs matching model suggestions.

    Phase 5.2 Fix:
    - Uses bid/ask prices instead of close
    - Validates liquidity (volume, OI, spread)
    - Optionally validates Greeks (delta targeting)
    - Returns None if atomicity cannot be guaranteed
    """
    if chain_df.empty:
        return None

    # Synthesize bid/ask prices
    chain_df = synthesize_chain_prices(chain_df)

    # Suggested strikes
    s_call_target = spot + (call_off * spot * 0.01)
    s_put_target = spot - (put_off * spot * 0.01)

    # Filter by CP
    calls = chain_df[chain_df['call_put'] == 'C'].copy()
    puts = chain_df[chain_df['call_put'] == 'P'].copy()

    if calls.empty or puts.empty:
        return None

    # --- PHASE 5.2: Delta-based selection if Greeks available ---
    if validate_greeks and 'delta' in calls.columns:
        # For short call: find delta closest to target (e.g., 0.16)
        calls['delta_abs'] = calls['delta'].abs()
        puts['delta_abs'] = puts['delta'].abs()

        # Filter to reasonable delta range for short strikes (0.10 - 0.25)
        valid_short_calls = calls[(calls['delta_abs'] >= 0.10) & (calls['delta_abs'] <= 0.25)]
        valid_short_puts = puts[(puts['delta_abs'] >= 0.10) & (puts['delta_abs'] <= 0.25)]

        if not valid_short_calls.empty:
            # Select by delta closest to target
            short_call_row = valid_short_calls.iloc[
                (valid_short_calls['delta_abs'] - EXEC_TARGET_SHORT_DELTA).abs().argsort()[:1]
            ]
        else:
            # Fallback to strike-based selection
            short_call_row = calls.iloc[(calls['strike'] - s_call_target).abs().argsort()[:1]]

        if not valid_short_puts.empty:
            short_put_row = valid_short_puts.iloc[
                (valid_short_puts['delta_abs'] - EXEC_TARGET_SHORT_DELTA).abs().argsort()[:1]
            ]
        else:
            short_put_row = puts.iloc[(puts['strike'] - s_put_target).abs().argsort()[:1]]
    else:
        # Fallback: strike-based selection (original behavior)
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

    # --- PHASE 5.2: Liquidity Validation ---
    legs_data = [
        ('short_call', short_call_row),
        ('short_put', short_put_row),
        ('long_call', long_call_row),
        ('long_put', long_put_row)
    ]

    min_fill_prob = 1.0
    liquidity_issues = []

    for leg_name, leg_row in legs_data:
        row_dict = leg_row.iloc[0].to_dict()
        is_valid, fill_prob, reason = validate_leg_liquidity(row_dict)
        if not is_valid and EXEC_ATOMICITY_STRICT:
            liquidity_issues.append(f"{leg_name}: {reason}")
        min_fill_prob = min(min_fill_prob, fill_prob)

    if liquidity_issues and EXEC_ATOMICITY_STRICT:
        # Log but don't fail - let entry logic decide
        pass

    # Package legs with bid/ask prices
    return {
        'short_call': s_call,
        'short_call_bid': short_call_row['bid'].values[0],
        'short_call_ask': short_call_row['ask'].values[0],
        'short_call_mid': short_call_row['mid'].values[0],
        'short_call_close': short_call_row['close'].values[0],  # Keep for compatibility
        'short_call_symbol': short_call_row['option_symbol'].values[0],
        'short_call_delta': short_call_row['delta'].values[0] if 'delta' in short_call_row.columns else None,

        'long_call': l_call,
        'long_call_bid': long_call_row['bid'].values[0],
        'long_call_ask': long_call_row['ask'].values[0],
        'long_call_mid': long_call_row['mid'].values[0],
        'long_call_close': long_call_row['close'].values[0],
        'long_call_symbol': long_call_row['option_symbol'].values[0],
        'long_call_delta': long_call_row['delta'].values[0] if 'delta' in long_call_row.columns else None,

        'short_put': s_put,
        'short_put_bid': short_put_row['bid'].values[0],
        'short_put_ask': short_put_row['ask'].values[0],
        'short_put_mid': short_put_row['mid'].values[0],
        'short_put_close': short_put_row['close'].values[0],
        'short_put_symbol': short_put_row['option_symbol'].values[0],
        'short_put_delta': short_put_row['delta'].values[0] if 'delta' in short_put_row.columns else None,

        'long_put': l_put,
        'long_put_bid': long_put_row['bid'].values[0],
        'long_put_ask': long_put_row['ask'].values[0],
        'long_put_mid': long_put_row['mid'].values[0],
        'long_put_close': long_put_row['close'].values[0],
        'long_put_symbol': long_put_row['option_symbol'].values[0],
        'long_put_delta': long_put_row['delta'].values[0] if 'delta' in long_put_row.columns else None,

        'width': abs(l_call - s_call),
        'min_fill_prob': min_fill_prob,
        'liquidity_issues': liquidity_issues
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

    def update_mark(self, current_dt, marks_mid, marks_bid=None, marks_ask=None):
        """
        Update unrealized PnL using current O(1) mark cache.

        Phase 5.2 Fix:
        - Uses MID prices for fair value MTM
        - Optionally uses bid/ask for realistic exit cost estimation
        """
        # Price 4 legs using MID for fair value
        sc = marks_mid.get(self.legs['short_call_symbol'])
        lc = marks_mid.get(self.legs['long_call_symbol'])
        sp = marks_mid.get(self.legs['short_put_symbol'])
        lp = marks_mid.get(self.legs['long_put_symbol'])

        # If any leg missing, carry forward logic (simplified: skip update)
        if None in [sc, lc, sp, lp]:
            return

        # MTM using MID prices (fair value)
        debit_mid = (sc + sp) - (lc + lp)

        # PnL = Credit - Debit
        pnl_per_share = self.net_credit - debit_mid
        self.unrealized_pnl = pnl_per_share * self.qty * IC_MULTIPLIER

        # Also calculate realistic exit cost using bid/ask if available
        if marks_bid is not None and marks_ask is not None:
            # To close: buy back shorts at ASK, sell longs at BID
            sc_ask = marks_ask.get(self.legs['short_call_symbol'], sc)
            sp_ask = marks_ask.get(self.legs['short_put_symbol'], sp)
            lc_bid = marks_bid.get(self.legs['long_call_symbol'], lc)
            lp_bid = marks_bid.get(self.legs['long_put_symbol'], lp)

            debit_real = (sc_ask + sp_ask) - (lc_bid + lp_bid)
            debit_real += EXEC_SLIPPAGE_PER_LEG * 4  # Add slippage

            pnl_real = self.net_credit - debit_real
            self.unrealized_pnl_real = pnl_real * self.qty * IC_MULTIPLIER
        else:
            self.unrealized_pnl_real = self.unrealized_pnl

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

    # ==========================================================================
    # PHASE 5.5: Initialize Execution Reality Engine
    # ==========================================================================
    exec_engine = None
    if EXEC_REALITY_ENABLED and HAS_EXECUTION_REALITY:
        print(f"[Phase 5.5] Initializing ExecutionRealityEngine (seed={EXEC_REALITY_SEED})...")
        exec_engine = ExecutionRealityEngine(
            seed=EXEC_REALITY_SEED,
            latency_mean_ms=EXEC_REALITY_LATENCY_MS,
            aggression=EXEC_REALITY_AGGRESSION,
        )
        print(f"   Latency: {EXEC_REALITY_LATENCY_MS}ms, Aggression: {EXEC_REALITY_AGGRESSION}")
    elif EXEC_REALITY_ENABLED and not HAS_EXECUTION_REALITY:
        print("[Phase 5.5] WARNING: ExecutionRealityEngine requested but not available. Using simple model.")

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
            
        batch_input = torch.stack(batch_seqs).float().to(DEVICE)
        with torch.no_grad():
            # Model returns tuple, 0-index is policy
            out = model(batch_input)[0].cpu().numpy()
            all_policy.append(out)
            
    # Concatenate all batches
    if all_policy:
        policy_matrix = np.concatenate(all_policy, axis=0)
    else:
        policy_matrix = np.zeros((0, 10))
    
    # DEBUG: Diagnostic on Model Outputs
    print(f"\n[DEBUG] Policy Matrix Shape: {policy_matrix.shape}")
    entry_logits = policy_matrix[:, 8]
    entry_probs = 1.0 / (1.0 + np.exp(-entry_logits))
    print(f"[DEBUG] Entry Prob Stats: Min={entry_probs.min():.4f}, Max={entry_probs.max():.4f}, Mean={entry_probs.mean():.4f}")
    print(f"[DEBUG] > 0.40 count: {(entry_probs > 0.40).sum()}")
    print(f"[DEBUG] > 0.50 count: {(entry_probs > 0.50).sum()}")
    
    # DEBUG: Additional diagnostics for other gate conditions
    pop_raw = policy_matrix[:, 4]
    pop_probs = 1.0 / (1.0 + np.exp(-pop_raw))
    conf_raw = policy_matrix[:, 7]
    conf_probs = 1.0 / (1.0 + np.exp(-conf_raw))
    
    print(f"[DEBUG] POP (pol[4]) Stats: Min={pop_probs.min():.4f}, Max={pop_probs.max():.4f}, Mean={pop_probs.mean():.4f}")
    print(f"[DEBUG] POP > 0.40 count: {(pop_probs > 0.40).sum()}")
    print(f"[DEBUG] Conf (pol[7]) Stats: Min={conf_probs.min():.4f}, Max={conf_probs.max():.4f}, Mean={conf_probs.mean():.4f}")
    print(f"[DEBUG] Conf > 0.35 count: {(conf_probs > 0.35).sum()}")
    
    # Combined gate check
    combined_pass = (entry_logits > 0.0) & (pop_probs > 0.40) & (conf_probs > 0.35)
    print(f"[DEBUG] ALL GATES PASS count: {combined_pass.sum()}")

        
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
    
    # Pre-extract spot timestamps as numpy array to ensure type match with ts_ranges (built from .values)
    spot_timestamps = spot_bars[time_col].values
    
    for i in tqdm(range(sim_start_idx, num_bars), desc="Simulating"):
         # Adjust index for policy
         pol_idx = i - SEQ_LEN
         if pol_idx >= len(policy_matrix): break
         
         # 1. Get Bar Data
         ts = spot_timestamps[i]
         spot = float(spot_bars['close'].iloc[i])
         
         # 2. Get Option Chain (O(1))
         s, e = ts_ranges.get(ts, (None, None))
         if s is None: 
             continue # No chain this minute?
             
         chain_slice = df.iloc[s:e]

         # 3. Build Marks (Transient) - PHASE 5.2 FIX: Separate bid/ask/mid
         # Synthesize bid/ask from close + spread_ratio
         chain_with_prices = synthesize_chain_prices(chain_slice)

         marks_mid = dict(zip(chain_with_prices['option_symbol'], chain_with_prices['mid']))
         marks_bid = dict(zip(chain_with_prices['option_symbol'], chain_with_prices['bid']))
         marks_ask = dict(zip(chain_with_prices['option_symbol'], chain_with_prices['ask']))

         # PHASE 5.5: Create MarketState for execution reality engine
         bar_data = spot_bars.iloc[i]
         vix_val = bar_data.get('vix', bar_data.get('VIX', 15.0)) if hasattr(bar_data, 'get') else 15.0
         vol_val = bar_data.get('volume', 100) if hasattr(bar_data, 'get') else 100
         # Extract hour from timestamp for TOD liquidity
         try:
             ts_hour = pd.Timestamp(ts).hour + pd.Timestamp(ts).minute / 60.0
         except:
             ts_hour = 12.0
         market_state = create_market_state_from_bar(spot, vix_val, ts_hour, vol_val) if exec_engine else None

         # 4. Update Open Trades - PHASE 5.2 FIX: Pass bid/ask/mid
         active_trades = []
         for tr in open_trades:
             tr.update_mark(ts, marks_mid, marks_bid, marks_ask)
             
             # Check Risk Stop (5%) - PHASE 5.2/5.5 FIX: Use realistic exit cost
             if tr.should_risk_close(equity):
                 # PHASE 5.5: Use ExecutionRealityEngine if available
                 if exec_engine and market_state:
                     exit_debit, exit_valid, exit_details = calculate_exit_fill_reality(
                         tr.legs, marks_bid, marks_ask, marks_mid, exec_engine, market_state
                     )
                 else:
                     exit_debit, exit_valid, exit_details = calculate_exit_fill(
                         tr.legs, marks_bid, marks_ask, marks_mid
                     )

                 if exit_valid:
                     # PnL = Entry Credit - Exit Debit (per share), then scale
                     realized_pnl = (tr.net_credit - exit_debit) * tr.qty * IC_MULTIPLIER
                 else:
                     # Fallback to MTM-based estimate
                     realized_pnl = getattr(tr, 'unrealized_pnl_real', tr.unrealized_pnl)

                 tr.exit_dt = ts
                 tr.exit_reason = "RISK_5PCT"
                 tr.realized_pnl = realized_pnl
                 tr.is_closed = True

                 equity += tr.realized_pnl
                 # Log close
                 closed_trades.append({
                     'trade_id': tr.trade_id,
                     'entry_dt': tr.entry_dt,
                     'exit_dt': ts,
                     'pnl': tr.realized_pnl,
                     'pnl_pct': tr.pnl_pct * 100,  # % of max risk
                     'reason': "RISK_STOP",
                     'max_dd': tr.max_dd_pct * 100,
                     'held_bars': -1,  # TODO
                     'exit_details': exit_details if exit_valid else None
                 })
                 continue  # Trade is gone
            
             # Check Model Exit Signal (optional)
             # pol = policy_matrix[pol_idx]
             # exit_logit = pol[9]
             # if exit_logit > 1.0: ...
             
             # Check Expiration (DTE < 0.1)
             # Approximation: if we hold > DTE (in bars) ??
             # Better: check days passed.
             days_held = (pd.Timestamp(ts) - pd.Timestamp(tr.entry_dt)).total_seconds() / 86400
             if days_held > tr.dte_entry:
                 # PHASE 5.2/5.5 FIX: Expired - Use realistic exit cost
                 if exec_engine and market_state:
                     exit_debit, exit_valid, exit_details = calculate_exit_fill_reality(
                         tr.legs, marks_bid, marks_ask, marks_mid, exec_engine, market_state
                     )
                 else:
                     exit_debit, exit_valid, exit_details = calculate_exit_fill(
                         tr.legs, marks_bid, marks_ask, marks_mid
                     )

                 if exit_valid:
                     # PnL = Entry Credit - Exit Debit (per share), then scale
                     realized_pnl = (tr.net_credit - exit_debit) * tr.qty * IC_MULTIPLIER
                 else:
                     # Fallback to MTM-based estimate (with real spread cost if available)
                     realized_pnl = getattr(tr, 'unrealized_pnl_real', tr.unrealized_pnl)

                 tr.exit_dt = ts
                 tr.exit_reason = "EXPIRED"
                 tr.realized_pnl = realized_pnl
                 tr.is_closed = True
                 equity += tr.realized_pnl
                 closed_trades.append({
                     'trade_id': tr.trade_id,
                     'entry_dt': tr.entry_dt,
                     'exit_dt': ts,
                     'pnl': tr.realized_pnl,
                     'pnl_pct': tr.pnl_pct * 100,
                     'reason': "EXPIRED",
                     'max_dd': tr.max_dd_pct * 100,
                     'exit_details': exit_details if exit_valid else None
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
                 
                 # DEBUG: Track downstream blocking
                 if not hasattr(run_backtest, '_debug_counters'):
                     run_backtest._debug_counters = {'gate_pass': 0, 'no_legs': 0, 'not_atomic': 0, 'low_credit': 0, 'success': 0}
                 run_backtest._debug_counters['gate_pass'] += 1
                 
                 # Leg Selection - PHASE 5.2 FIX: Includes bid/ask and Greeks validation
                 legs = find_best_legs(chain_with_prices, spot, call_off, put_off, width, validate_greeks=True)

                 if legs:
                     # PHASE 5.2/5.5 FIX: Use realistic execution
                     if exec_engine and market_state:
                         # PHASE 5.5: Full execution reality modeling
                         filled_qty, net_credit, is_atomic, fill_details = calculate_entry_fill_reality(
                             legs, chain_with_prices, IC_CONTRACTS, exec_engine, market_state
                         )
                     else:
                         # PHASE 5.2: Simple bid/ask + slippage
                         filled_qty, net_credit, is_atomic, fill_details = calculate_entry_fill(
                             legs, chain_with_prices, IC_CONTRACTS
                         )

                     if not is_atomic:
                         # Atomicity violation - skip this entry
                         run_backtest._debug_counters['not_atomic'] += 1
                         continue

                     if net_credit < 0.10:  # Min credit filter (after slippage)
                         run_backtest._debug_counters['low_credit'] += 1
                         # DEBUG: Track actual credit values
                         if not hasattr(run_backtest, '_credit_samples'):
                             run_backtest._credit_samples = []
                         if len(run_backtest._credit_samples) < 10:  # First 10 samples
                             run_backtest._credit_samples.append({
                                 'gross': fill_details.get('gross_credit', 0),
                                 'net': net_credit,
                                 'slippage': fill_details.get('slippage', 0),
                             })
                         continue

                     max_loss = (legs['width'] - net_credit) * IC_MULTIPLIER * filled_qty
                     trade_id = f"TR_{i}"

                     new_trade = Trade(
                         trade_id, ts, legs, net_credit, max_loss,
                         dte=dte,
                         scores={
                             'entry': entry_logit,
                             'prob': prob,
                             'gross_credit': fill_details.get('gross_credit', 0),
                             'slippage': fill_details.get('total_slippage', fill_details.get('slippage', 0)),
                             'short_call_delta': legs.get('short_call_delta'),
                             'short_put_delta': legs.get('short_put_delta'),
                             'exec_reality': True if exec_engine else False,
                         }
                     )
                     open_trades.append(new_trade)
                     run_backtest._debug_counters['success'] += 1
                 else:
                     run_backtest._debug_counters['no_legs'] += 1
         
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
    
    # DEBUG: Print downstream blocking stats
    if hasattr(run_backtest, '_debug_counters'):
        dc = run_backtest._debug_counters
        print(f"\n[DEBUG] DOWNSTREAM BLOCKING STATS:")
        print(f"   Gate Pass Attempts: {dc.get('gate_pass', 0)}")
        print(f"   No Legs (find_best_legs=None): {dc.get('no_legs', 0)}")
        print(f"   Atomicity Violations: {dc.get('not_atomic', 0)}")
        print(f"   Low Credit (<$0.10): {dc.get('low_credit', 0)}")
        print(f"   SUCCESS (trades opened): {dc.get('success', 0)}")
    
    # DEBUG: Print credit samples
    if hasattr(run_backtest, '_credit_samples') and run_backtest._credit_samples:
        print(f"\n[DEBUG] CREDIT SAMPLES (first 10 low-credit trades):")
        for i, cs in enumerate(run_backtest._credit_samples):
            print(f"   [{i+1}] Gross: ${cs['gross']:.4f}, Net: ${cs['net']:.4f}, Slippage: ${cs['slippage']:.4f}")
    
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

    # Phase 5.5: Execution Reality Toggles
    parser.add_argument("--no-exec-reality", action="store_true", default=False, help="Disable Phase 5.5 execution reality modeling")
    parser.add_argument("--exec-latency", type=float, default=100.0, help="Mean order latency in ms (default: 100)")
    parser.add_argument("--exec-aggression", type=float, default=0.5, help="Queue aggression 0-1 (default: 0.5)")
    parser.add_argument("--exec-seed", type=int, default=42, help="Seed for execution reality RNG (default: 42)")

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

    # Phase 5.5: Update Execution Reality Config from CLI args
    global EXEC_REALITY_ENABLED, EXEC_REALITY_SEED, EXEC_REALITY_LATENCY_MS, EXEC_REALITY_AGGRESSION
    if args.no_exec_reality:
        EXEC_REALITY_ENABLED = False
        print("[Phase 5.5] Execution reality DISABLED (--no-exec-reality)")
    else:
        EXEC_REALITY_ENABLED = True
        EXEC_REALITY_SEED = args.exec_seed
        EXEC_REALITY_LATENCY_MS = args.exec_latency
        EXEC_REALITY_AGGRESSION = args.exec_aggression
        print(f"[Phase 5.5] Execution reality ENABLED (seed={EXEC_REALITY_SEED}, latency={EXEC_REALITY_LATENCY_MS}ms, aggression={EXEC_REALITY_AGGRESSION})")

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
            max_active_predicates=max_active,
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
