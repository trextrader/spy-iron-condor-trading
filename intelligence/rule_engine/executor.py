# intelligence/rule_engine/executor.py
"""
Rule execution engine implementing the 6-phase execution flow:
1. Precompute Primitives (parallelized)
2. Evaluate Signal Logic
3. Apply Gate Stack (short-circuit on BLOCK)
4. Compute Position Sizing
5. Risk Check
6. Execute/Log
"""

import re
import logging
from typing import Dict, Any, Callable, Optional, Union

import pandas as pd
import numpy as np

from intelligence.rule_engine.dsl_parser import Ruleset, RuleSpec, PrimitiveSpec
from intelligence import primitives as prim

logger = logging.getLogger(__name__)

# Registry mapping canonical IDs to functions
PRIMITIVE_REGISTRY = {
    # Bands/Microstructure (P001-P007)
    "P001": prim.compute_dynamic_bollinger_bands,
    "P002": prim.compute_bandwidth_percentile_and_expansion,
    "P003": prim.compute_volume_ratio,
    "P004": prim.compute_spread_friction_ratio,
    "P005": prim.compute_gap_risk_score,
    "P006": prim.compute_iv_confidence,
    "P007": prim.compute_mtf_consensus,
    
    # Momentum (M001-M004) -> Mapped as P008-P012 in some contexts, but sticking to module mapping
    "M001": prim.compute_vol_normalized_macd,
    "M002": prim.compute_vol_normalized_adx,
    "M003": prim.compute_dynamic_rsi,
    "M004": prim.compute_psar_flip_membership,
    
    # Topology (T001-T002)
    "T001": prim.compute_beta1_regime_score,
    "T002": prim.compute_chaos_membership,
    
    # Fuzzy (F001)
    "F001": prim.compute_fuzzy_reversion_score_11,
    
    # Gates (G003-G010)
    "G003": prim.compute_trend_strength_gate,
    "G004": prim.compute_reversion_fuzzy_gate,
    "G005": prim.compute_chaos_risk_gate,
    "G006": prim.compute_regime_score_gate,
    "G007": prim.compute_liquidity_gate,
    "G008": prim.compute_spread_liquidity_combo_gate,
    "G009": prim.compute_gap_override_gate,
    "G010": prim.compute_position_size_gate,
    
    # Signals (S001-S015)
    "S001": prim.compute_macd_trend_signal,
    "S002": prim.compute_band_squeeze_breakout_signal,
    "S003": prim.compute_rsi_reversion_signal,
    "S004": prim.compute_mtf_alignment_signal,
    "S005": prim.compute_fuzzy_reversion_signal,
    "S006": prim.compute_gap_event_signal,
    "S007": prim.compute_chaos_dampening_signal,
    "S008": prim.compute_regime_shift_signal,
    "S009": prim.compute_liquidity_exec_signal,
    "S010": prim.compute_trend_follow_entry_signal,
    "S011": prim.compute_reversion_vs_trend_conflict_signal,
    "S012": prim.compute_spread_block_signal,
    "S013": prim.compute_gap_exit_signal,
    "S014": prim.compute_size_adjustment_signal,
    "S015": prim.compute_final_execution_signal,
}


class LogicEvaluator:
    """Evaluates logical expressions like 'AND(a.b, c > 0.5)' against primitive outputs"""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def _get_value(self, token: str) -> pd.Series:
        token = token.strip()
        
        # Numeric literal
        if re.match(r"^-?\d+(\.\d+)?$", token):
            return float(token)
            
        # Attribute access: alias.key
        if "." in token:
            alias, key = token.split(".", 1)
            if alias in self.context and isinstance(self.context[alias], dict):
                val = self.context[alias].get(key)
                if val is not None:
                    return val
                logger.warning(f"Key '{key}' not found in primitive '{alias}'")
        
        # Direct alias access (returns full dict/series)
        if token in self.context:
            return self.context[token]
            
        logger.warning(f"Token '{token}' not found in context")
        # Try to find a valid index from context to return zeros
        idx = None
        for v in self.context.values():
            if hasattr(v, "index"):
                idx = v.index
                break
            if isinstance(v, dict):
                for sub_v in v.values():
                    if hasattr(sub_v, "index"):
                        idx = sub_v.index
                        break
            if idx is not None: break
            
        if idx is not None:
            return pd.Series(0, index=idx)
        return 0

    def evaluate(self, expr: str) -> pd.Series:
        if not expr:
            return None
        
        expr = expr.strip()

        # Logical operators (Check these FIRST to avoid greedy regex matching inside parens)
        # Note: This naive parser struggles with nested calls containing commas inside args
        # e.g. AND(OR(A,B), C) work if split(",") handles nested parens. 
        # For V2.5 MVP we assume flat logic at top level or handle carefully.
        
        if expr.startswith("AND(") and expr.endswith(")"):
            inner = expr[4:-1]
            # TODO: Robust split ignoring nested parens
            parts = [self.evaluate(p.strip()) for p in inner.split(",")]
            res = parts[0]
            for p in parts[1:]:
                res = res & p
            return res
            
        if expr.startswith("OR(") and expr.endswith(")"):
            inner = expr[3:-1]
            parts = [self.evaluate(p.strip()) for p in inner.split(",")]
            res = parts[0]
            for p in parts[1:]:
                res = res | p
            return res
            
        if expr.startswith("NOT(") and expr.endswith(")"):
            inner = expr[4:-1]
            val = self.evaluate(inner)
            return ~val
            
        # Comparison logic (simple support)
        # Matches: term > value, term < value, term >= value, term <= value, term == value
        comp_match = re.match(r"(.+?)\s*(>=|<=|>|<|==)\s*(.+)", expr)
        if comp_match:
            left, op, right = comp_match.groups()
            l_val = self._get_value(left)
            r_val = self._get_value(right)
            
            if op == ">": return l_val > r_val
            if op == "<": return l_val < r_val
            if op == ">=": return l_val >= r_val
            if op == "<=": return l_val <= r_val
            if op == "==": return l_val == r_val

        # Fallback to direct value lookup
        return self._get_value(expr)


class RuleExecutionEngine:
    """Executes rules from a Ruleset against market data."""

    def __init__(self, ruleset: Ruleset, registry: Dict[str, Callable] = PRIMITIVE_REGISTRY):
        self.ruleset = ruleset
        self.registry = registry

    def _compute_primitives(
        self, rule: RuleSpec, data: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Phase 1: Precompute primitives"""
        outputs = {}
        for p_spec in rule.primitives:
             # Resolve alias for function lookup or fallback to ID
            # Use 'alias' as the key in the context for logic lookup
            
            func_id = p_spec.id
            if func_id not in self.registry:
                logger.error(f"Function ID {func_id} not found in registry for rule {rule.id}")
                continue
                
            func = self.registry[func_id]
            
            # Map inputs
            kwargs = {}
            # Combine available data sources: input data + already computed outputs
            # Note: This allows chaining if primitives are ordered topologically in rule
            current_context = {**data, **outputs}
            
            mapped_inputs = {}
            for arg, src in p_spec.inputs.items():
                if src in current_context:
                    mapped_inputs[arg] = current_context[src]
                elif "." in src:
                    # Resolve alias.key from outputs
                    alias, key = src.split(".", 1)
                    if alias in outputs and isinstance(outputs[alias], dict):
                        val = outputs[alias].get(key)
                        if val is not None:
                            mapped_inputs[arg] = val
            
            # Combine mapped inputs, unmapped params, and remaining data (opportunistic)
            effective_kwargs = {**data, **mapped_inputs, **p_spec.params}
            
            try:
                # We need to filter this to valid args only to avoid TypeError
                import inspect
                sig = inspect.signature(func)
                valid_kwargs = {k: v for k, v in effective_kwargs.items() if k in sig.parameters}
                
                result = func(**valid_kwargs)
                outputs[p_spec.alias] = result
                
            except Exception as e:
                logger.error(f"Primitive {p_spec.id} ({p_spec.alias}) failed in {rule.id}: {e}")
                outputs[p_spec.alias] = {} # Empty dict to prevent downstream crash

        return outputs

    def _evaluate_signal_logic(
        self, rule: RuleSpec, primitives: Dict[str, Any]
    ) -> pd.DataFrame:
        """Phase 2: Signal Logic"""
        evaluator = LogicEvaluator(primitives)
        
        # Evaluate long/short entry logic
        long_sig = evaluator.evaluate(rule.signal_logic.entry_long)
        short_sig = evaluator.evaluate(rule.signal_logic.entry_short)
        exit_sig = evaluator.evaluate(rule.signal_logic.exit)
        
        # Normalize to False if None
        if long_sig is None: long_sig = pd.Series(False, index=next(iter(primitives.values())).index) if primitives else False
        if short_sig is None: short_sig = pd.Series(False, index=long_sig.index)
        if exit_sig is None: exit_sig = pd.Series(False, index=long_sig.index)
        
        df = pd.DataFrame({
            "entry_long": long_sig,
            "entry_short": short_sig,
            "exit": exit_sig
        })
        return df

    def _apply_gate_stack(
        self, rule: RuleSpec, primitives: Dict[str, Any], signals: pd.DataFrame
    ) -> pd.DataFrame:
        """Phase 3: Gates"""
        gated = signals.copy()
        
        # Initialize allow flags
        gated["blocked"] = False
        
        # Since we don't have gate logic fully integrated in primitives dict (they are computed as primitives!),
        # We need to find the gate outputs in `primitives`.
        # Gates in `rule.gate_stack` refer to IDs (G003). 
        # But `_compute_primitives` might have run them if they were listed in `primitives`. 
        # IF gates are separate in DSL, we need to run them here.
        
        # DSL V2.5 separates `gates` list. We need to compute them now or assume they were precomputed?
        # Update: `dsl_parser` loops over `primitives`. DSL V2.5 has `gates` list separately.
        # We should iterate `rule.gate_stack`, resolve the function, compute it, then apply logic.
        
        for gate_spec in rule.gate_stack:
            func = self.registry.get(gate_spec.id)
            if not func: continue
            
            # Map args
            import inspect
            sig = inspect.signature(func)
            # Gather inputs from `primitives` results or raw data?
            # Gates usually take primitive outputs (e.g. adx_norm)
            
            # Flatten primitive outputs for easier Arg matching
            flat_context = {}
            for p_alias, p_out in primitives.items():
                if isinstance(p_out, dict):
                    flat_context.update(p_out)
                else:
                    flat_context[p_alias] = p_out
            
            # Add params
            flat_context.update(gate_spec.params)
            
            valid_kwargs = {k: v for k, v in flat_context.items() if k in sig.parameters}
            
            try:
                res = func(**valid_kwargs)
                
                # Apply gate logic
                # Canonical gates usually return keys like "trend_ok", "chaos_block"
                
                # Block logic
                if gate_spec.action == "block":
                    # Check for explicit block keys
                    if "chaos_block" in res:
                        block = res["chaos_block"]
                        gated["entry_long"] &= (~block)
                        gated["entry_short"] &= (~block)
                        gated.loc[block, "blocked"] = True
                        
                    # Check for explicit Allow keys (AND logic)
                    if "trend_ok" in res:
                        allow = res["trend_ok"]
                        gated["entry_long"] &= allow
                        gated["entry_short"] &= allow
                        if (~allow).any():
                             gated.loc[~allow, "blocked"] = True
                             
                    # Liquidity gates
                    if "liquidity_ok" in res:
                        allow = res["liquidity_ok"]
                        gated["entry_long"] &= allow
                        gated["entry_short"] &= allow
                        
            except Exception as e:
                logger.error(f"Gate {gate_spec.id} failed: {e}")

        return gated

    def execute(self, data: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        """Phase 6: Full Execution"""
        results = {}
        for rule_id, rule in self.ruleset.rules.items():
            # 1. Primitives
            primitives = self._compute_primitives(rule, data)
            
            # 2. Signals
            signals = self._evaluate_signal_logic(rule, primitives)
            
            # 3. Gates
            gated_signals = self._apply_gate_stack(rule, primitives, signals)
            
            # 4. Sizing (Stub)
            size = pd.Series(1.0, index=signals.index)
            # TODO: Integrate Sizing logic
            
            # 5. Risk override (Stub)
            
            results[rule_id] = pd.DataFrame({
                "signal_long": gated_signals["entry_long"],
                "signal_short": gated_signals["entry_short"],
                "signal_exit": gated_signals["exit"],
                "blocked": gated_signals["blocked"],
                "size": size
            })
            
        return results
