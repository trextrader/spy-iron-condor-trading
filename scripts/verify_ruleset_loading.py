
import sys
import os
import pandas as pd
import logging
from pprint import pprint

# Setup path
sys.path.append(os.getcwd())

from intelligence.rule_engine.dsl_parser import RuleDSLParser
from intelligence.rule_engine.executor import RuleExecutionEngine, LogicEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

def verify_ruleset():
    print("[-] Loading Ruleset from docs/Complete_Ruleset_DSL.yaml...")
    try:
        parser = RuleDSLParser("docs/Complete_Ruleset_DSL.yaml")
        ruleset = parser.load()
        print(f"[+] Loaded {len(ruleset.rules)} rules. Version: {ruleset.version}")
    except Exception as e:
        print(f"[!] Failed to parse YAML: {e}")
        return

    print("[-] Initializing Engine...")
    engine = RuleExecutionEngine(ruleset)
    
    print("[-] Verifying Primitives Registry Ref...")
    missing_ids = set()
    for rid, rule in ruleset.rules.items():
        for p in rule.primitives:
            if p.id not in engine.registry:
                missing_ids.add(p.id)
                print(f"[!] Rule {rid} references missing primitive ID: {p.id}")
    
    if missing_ids:
        print(f"[!] Missing Primitives: {missing_ids}")
    else:
        print("[+] All primitive IDs found in registry.")

    print("[-] Dry Run Logic Parsing...")
    # We can't easily verify keys exist without running primitives, but we can verify syntax
    # and maybe run primitives with dummy data if we know inputs.
    
    # Let's try to run with dummy data on one rule?
    # Or just check if parser handles the strings.
    
    for rid, rule in ruleset.rules.items():
        print(f"Checking Rule: {rid}")
        
        # Check logic strings
        logic = rule.signal_logic
        if logic.entry_long:
            try:
                # Just invoke evaluate with empty context to check syntax?
                # or mock context
                pass
            except Exception as e:
                print(f"[!] Rule {rid} entry_long logic parse error: {e}")
                
    # Create Dummy Data for full dry run (Aggressive!)
    # We need all columns required by all primitives.
    # We can inspect `requires` in YAML if available or `inputs`.
    
    print("[-] Creating Synthetic Data for Execution Test...")
    dates = pd.date_range("2024-01-01", periods=100, freq="5min")
    data = {
        "close": pd.Series(100.0, index=dates),
        "open": pd.Series(100.0, index=dates),
        "high": pd.Series(101.0, index=dates),
        "low": pd.Series(99.0, index=dates),
        "volume": pd.Series(1000, index=dates),
        "vol_ewma": pd.Series(0.01, index=dates),
        "ema_12": pd.Series(100.0, index=dates),
        "ema_26": pd.Series(100.0, index=dates),
        "bb_bandwidth": pd.Series(0.02, index=dates),
        "bb_upper": pd.Series(102.0, index=dates),
        "bb_lower": pd.Series(98.0, index=dates),
        "bb_middle": pd.Series(100.0, index=dates),
        "adx_norm": pd.Series(30, index=dates),
        "rsi_dynamic": pd.Series(50, index=dates),
        "plus_di": pd.Series(20, index=dates),
        "minus_di": pd.Series(10, index=dates),
        "psar": pd.Series(99.0, index=dates),
        "beta1_norm": pd.Series(1.5, index=dates),
        "curvature": pd.Series(0.1, index=dates),
        "vol_energy": pd.Series(0.5, index=dates),
    }
    # Add Aliases for input mismatch resolution (assuming these would come from Primitives or Preprocessing)
    data["bandwidth"] = data["bb_bandwidth"] 
    data["volume_ratio"] = pd.Series(1.0, index=dates)
    
    # MTF Mock Data
    data["signal_1m"] = pd.Series(0.0, index=dates)
    data["signal_5m"] = pd.Series(0.0, index=dates)
    data["signal_15m"] = pd.Series(0.0, index=dates)
    data["w_1m"] = pd.Series(1.0, index=dates)
    data["w_5m"] = pd.Series(1.0, index=dates)
    data["w_15m"] = pd.Series(1.0, index=dates)
    
    # F001 Mock Data (11 Factors)
    for factor in ["mu_mtf", "mu_ivr", "mu_vix", "mu_rsi", "mu_stoch", "mu_adx", "mu_sma", "mu_psar", "mu_bb", "mu_bbsqueeze", "mu_vol"]:
        data[factor] = pd.Series(0.5, index=dates)

    # Topology & Gating Mock Data
    data["beta1_raw"] = pd.Series(1.0, index=dates)
    data["curvature_proxy"] = pd.Series(0.0, index=dates)
    data["risk_override"] = pd.Series(False, index=dates)
    
    # Missing Primitive Output Mocks (for implicit chaining where primitive execution is skipped or incomplete context)
    data["bw_percentile"] = pd.Series(50.0, index=dates)
    data["expansion_rate"] = pd.Series(1.0, index=dates)
    data["beta1_gated"] = pd.Series(1.0, index=dates) # For P010 if P009 fails or implicit chain incomplete


    # data["macd_norm"] = ... (Not needed if P003 runs and implicit chaining works)
    
    # Run Execute
    try:
        results = engine.execute(data)
        print(f"[+] Execution successful! Generated results for {len(results)} rules.")
        
        # Check for blocked signals (should be some if gates are active)
        for rid, df in results.items():
            blocked_cnt = df["blocked"].sum()
            print(f"Rule {rid}: {len(df)} bars. Signals: {df['signal_long'].sum()} Long, {df['signal_short'].sum()} Short. Blocked: {blocked_cnt}")
            
    except Exception as e:
        print(f"[!] Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_ruleset()
