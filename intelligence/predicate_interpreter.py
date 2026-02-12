"""
Predicate Interpretability Module for CondorBrain.

Converts cryptic F### notation to human-readable field names and exports
decision logic as structured decision trees.

Usage:
    from intelligence.predicate_interpreter import PredicateInterpreter
    
    interpreter = PredicateInterpreter()
    readable = interpreter.predicate_to_readable(pred_string)
    decision_tree = interpreter.export_decision_tree(predicates_json)
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import re
from dataclasses import dataclass

# Import the canonical feature registry
try:
    from .canonical_feature_registry import FEATURE_COLS_V22, FEATURE_COLS_V30
except ImportError:
    from intelligence.canonical_feature_registry import FEATURE_COLS_V22, FEATURE_COLS_V30


# =============================================================================
# FIELD NAME MAPPING (V3.0: 79 features, indices 0-78)
# =============================================================================

# Build comprehensive lookup: index -> human-readable name
FIELD_NAMES: Dict[int, str] = {i: name for i, name in enumerate(FEATURE_COLS_V30)}

# Reverse lookup: name -> index
NAME_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_COLS_V30)}


def field_to_name(idx: int, include_category: bool = False) -> str:
    """
    Convert field index to human-readable name.

    Args:
        idx: Field index (0-78 for raw features, 79+ for predicates)
        include_category: If True, prefix with category (e.g., "[PRICE]")

    Returns:
        Human-readable name like "close" or "P_312" for recursive predicates
    """
    if idx < len(FEATURE_COLS_V30):
        name = FEATURE_COLS_V30[idx]
        if include_category:
            cat = _get_category(name)
            return f"[{cat}]{name}"
        return name
    else:
        # Recursive predicate reference - mark clearly
        return f"P{idx}"  # "P" for Predicate (not "F" for Field)


def _get_category(field_name: str) -> str:
    """Get semantic category for a field name."""
    categories = {
        # Market Primitives
        "open": "PRICE", "high": "PRICE", "low": "PRICE", "close": "PRICE", "volume": "VOL",
        # Greeks
        "delta": "GREEK", "gamma": "GREEK", "vega": "GREEK", "theta": "GREEK",
        "rho": "GREEK",
        # Volatility
        "iv": "VOL", "ivr": "VOL", "vol_ewma": "VOL", "atr_pct": "VOL", "bb_sigma_dyn": "VOL",
        # Momentum
        "rsi_dyn": "MOM", "macd_norm": "MOM", "macd_signal_norm": "MOM", "macd_histogram": "MOM",
        "plus_di": "MOM", "minus_di": "MOM", "adx_adaptive": "MOM", "stoch_k_dyn": "MOM",
        # Bands
        "bb_mu_dyn": "BAND", "bb_lower_dyn": "BAND", "bb_upper_dyn": "BAND", "bb_percentile": "BAND",
        "bandwidth": "BAND",
        # PSAR
        "psar_adaptive": "PSAR", "psar_trend": "PSAR", "psar_reversion_mu": "PSAR",
        "psar_mark": "PSAR",
        # Risk
        "max_dd_60m": "RISK", "gap_risk_score": "RISK", "risk_override": "RISK",
        # Regime
        "consolidation_score": "REGIME", "breakout_score": "REGIME", "mtf_consensus": "REGIME",
        # Fuzzy
        "fuzzy_reversion_11": "FUZZY", "chaos_membership": "FUZZY",
        # V3.0: Price Context
        "underlying_price": "PRICE",
        # V3.0: Liquidity
        "open_interest": "LIQUIDITY",
        # V3.0: Trend Structure
        "sma": "TREND", "FRAMA": "TREND", "Anchored_VWAP": "TREND", "McClellanOsc": "BREADTH",
        # V3.0: IV Bands
        "IV_High": "VOL", "IV_Mid": "VOL", "IV_Low": "VOL",
        # V3.0: Chain Aggregates
        "Options_Total_Volume": "FLOW", "Options_Put_Volume": "FLOW",
        "Options_Call_Volume": "FLOW", "OSC_Volume": "FLOW",
        # V3.0: Flow Extensions
        "WeightedAlpha": "FLOW", "WilderAccSwingIndex": "FLOW",
        "AccDistWill": "FLOW", "AccDistWillMovAvg": "FLOW",
        # V3.0: Microstructure
        "aggregate_bid_count": "MICRO", "aggregate_ask_count": "MICRO",
        "mean_bid": "MICRO", "mean_ask": "MICRO", "quote_spread": "MICRO",
    }
    return categories.get(field_name, "OTHER")


# =============================================================================
# PREDICATE PARSER
# =============================================================================

class PredicateInterpreter:
    """
    Interprets and humanizes predicate expressions from training output.
    
    Converts:
        {F772[36]*F312[0]} <= F937[32] 
    To:
        {P772[36] × P312[0]} ≤ P937[32]  // Recursive predicate references
    Or for raw features:
        {minus_di[36] × close[0]} ≤ rsi_dyn[32]
    """
    
    # Regex to parse predicate strings
    FIELD_PATTERN = re.compile(r'F(\d+)\[(\d+)\]')
    NAMED_FIELD_PATTERN = re.compile(r'([a-z_]+)\[(\d+)\]')
    
    def __init__(self, predicate_registry: Optional[Dict[int, str]] = None):
        """
        Initialize interpreter.
        
        Args:
            predicate_registry: Optional mapping of predicate slot -> definition
                               Used to unroll recursive predicates
        """
        self.predicate_registry = predicate_registry or {}
        self.n_raw_features = len(FEATURE_COLS_V30)
    
    def parse_field(self, match_str: str) -> Tuple[str, int]:
        """
        Parse a field reference like "F312[0]" or "close[0]".
        
        Returns:
            (human_name, lookback)
        """
        # Try F### format
        m = self.FIELD_PATTERN.match(match_str)
        if m:
            idx = int(m.group(1))
            lookback = int(m.group(2))
            return (field_to_name(idx), lookback)
        
        # Try named format
        m = self.NAMED_FIELD_PATTERN.match(match_str)
        if m:
            return (m.group(1), int(m.group(2)))
        
        return (match_str, 0)
    
    def predicate_to_readable(self, pred_str: str) -> str:
        """
        Convert predicate string to human-readable format.
        
        Input:  "[GENERAL] {F772[36]*F312[0]} <= F937[32] (imp=0.007)"
        Output: "[GENERAL] {P772[36] × P312[0]} ≤ P937[32] (imp=0.007)"
                or with raw features:
                "[GENERAL] {minus_di[36] × close[0]} ≤ rsi_dyn[32] (imp=0.007)"
        """
        def replace_field(match):
            idx = int(match.group(1))
            lookback = match.group(2)
            name = field_to_name(idx)
            return f"{name}[{lookback}]"
        
        readable = self.FIELD_PATTERN.sub(replace_field, pred_str)
        
        # Humanize operators
        readable = readable.replace('*', ' × ')
        readable = readable.replace('<=', ' ≤ ')
        readable = readable.replace('>=', ' ≥ ')
        readable = readable.replace('==', ' = ')
        
        return readable
    
    def batch_to_readable(self, predicates: List[str]) -> List[str]:
        """Convert a batch of predicates to readable format."""
        return [self.predicate_to_readable(p) for p in predicates]
    
    def analyze_predicate_usage(self, predicates: List[str]) -> Dict[str, Any]:
        """
        Analyze which fields are most commonly used in predicates.
        
        Returns:
            {
                "field_frequency": {field_name: count},
                "lookback_histogram": {lookback: count},
                "operator_frequency": {op: count},
                "recursive_depth": max_recursive_depth,
            }
        """
        field_freq: Dict[str, int] = {}
        lookback_freq: Dict[int, int] = {}
        op_freq: Dict[str, int] = {}
        max_recursive = 0
        
        for pred in predicates:
            # Count field references
            for match in self.FIELD_PATTERN.finditer(pred):
                idx = int(match.group(1))
                lookback = int(match.group(2))
                
                name = field_to_name(idx)
                field_freq[name] = field_freq.get(name, 0) + 1
                lookback_freq[lookback] = lookback_freq.get(lookback, 0) + 1
                
                # Track recursive depth
                if idx >= self.n_raw_features:
                    max_recursive = max(max_recursive, 1)
            
            # Count operators
            for op in ['<', '>', '<=', '>=', '==']:
                if op in pred:
                    op_freq[op] = op_freq.get(op, 0) + 1
        
        return {
            "field_frequency": dict(sorted(field_freq.items(), key=lambda x: -x[1])[:20]),
            "lookback_histogram": dict(sorted(lookback_freq.items())),
            "operator_frequency": op_freq,
            "recursive_depth": max_recursive,
            "raw_feature_ratio": sum(1 for k in field_freq if not k.startswith('P')) / max(1, len(field_freq)),
        }


# =============================================================================
# DECISION TREE EXPORT
# =============================================================================

@dataclass
class DecisionNode:
    """A node in the decision tree."""
    condition: str                     # Human-readable condition
    importance: float                  # Feature importance
    true_branch: Optional['DecisionNode'] = None
    false_branch: Optional['DecisionNode'] = None
    action: Optional[str] = None       # Leaf action: "ENTER_LONG", "EXIT", etc.


def export_decision_tree_json(
    predicates: List[Dict[str, Any]],
    action_type: str = "entry",
    importance_threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Export predicates as a decision tree JSON structure.
    
    Args:
        predicates: List of predicate dicts with keys: template, expr, importance
        action_type: "entry", "exit", or "sizing"
        importance_threshold: Minimum importance to include
        
    Returns:
        JSON-serializable decision tree structure
    """
    interpreter = PredicateInterpreter()
    
    # Filter by importance
    filtered = [p for p in predicates if p.get('importance', 0) >= importance_threshold]
    
    # Sort by importance (most important first = root of tree)
    filtered.sort(key=lambda x: -x.get('importance', 0))
    
    tree = {
        "action_type": action_type,
        "generated_at": None,  # Will be filled
        "n_predicates": len(filtered),
        "rules": []
    }
    
    for i, pred in enumerate(filtered[:50]):  # Top 50
        expr = pred.get('expr', pred.get('predicate', str(pred)))
        readable = interpreter.predicate_to_readable(expr)
        
        tree["rules"].append({
            "rank": i + 1,
            "template": pred.get('template', 'GENERAL'),
            "condition": readable,
            "raw_condition": expr,
            "importance": pred.get('importance', 0),
            "category": _classify_predicate(expr)
        })
    
    return tree


def _classify_predicate(expr: str) -> str:
    """Classify predicate by its dominant features."""
    expr_lower = expr.lower()
    
    if any(x in expr_lower for x in ['rsi', 'macd', 'stoch', 'adx', 'plus_di', 'minus_di']):
        return "MOMENTUM"
    elif any(x in expr_lower for x in ['bb_', 'band']):
        return "BAND"
    elif any(x in expr_lower for x in ['psar', 'trend']):
        return "TREND"
    elif any(x in expr_lower for x in ['iv', 'vol', 'theta', 'vega']):
        return "VOLATILITY"
    elif any(x in expr_lower for x in ['high', 'low', 'close', 'open']):
        return "PRICE"
    elif any(x in expr_lower for x in ['fuzzy', 'chaos']):
        return "FUZZY"
    else:
        return "RECURSIVE"


def format_decision_tree_text(tree: Dict[str, Any], max_rules: int = 20) -> str:
    """
    Format decision tree as human-readable text.
    
    Example output:
    ═══════════════════════════════════════════════════════════════════
    ENTRY DECISION LOGIC (Top 20 Rules)
    ═══════════════════════════════════════════════════════════════════
    
    [MOMENTUM] #1 (imp=0.032)
    IF minus_di[36] × close[0] ≤ rsi_dyn[32]
    
    [BAND] #2 (imp=0.028)
    IF close[0] > bb_upper_dyn[0]
    ...
    """
    lines = []
    lines.append("═" * 70)
    lines.append(f"{tree['action_type'].upper()} DECISION LOGIC (Top {max_rules} Rules)")
    lines.append("═" * 70)
    lines.append("")
    
    for rule in tree["rules"][:max_rules]:
        lines.append(f"[{rule['category']}] #{rule['rank']} (imp={rule['importance']:.4f})")
        lines.append(f"  IF {rule['condition']}")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# QUICK CONVERSION UTILITY
# =============================================================================

def convert_epoch_output_to_readable(epoch_json: List[str]) -> str:
    """
    Quick converter for the epoch predicate output.
    
    Args:
        epoch_json: List of predicate strings like ["[GENERAL] {F772[36]*F312[0]} <= F937[32] (imp=0.007)"]
        
    Returns:
        Human-readable formatted string
    """
    interpreter = PredicateInterpreter()
    
    lines = ["=" * 70, "PREDICATE INTERPRETABILITY REPORT", "=" * 70, ""]
    
    # Analyze usage
    analysis = interpreter.analyze_predicate_usage(epoch_json)
    
    lines.append("📊 FIELD USAGE SUMMARY:")
    lines.append("-" * 40)
    for field, count in list(analysis["field_frequency"].items())[:10]:
        lines.append(f"  {field:25s}: {count:4d} references")
    
    lines.append("")
    lines.append(f"📈 Recursive predicate ratio: {1 - analysis['raw_feature_ratio']:.1%}")
    lines.append(f"📈 Max lookback used: {max(analysis['lookback_histogram'].keys()) if analysis['lookback_histogram'] else 0}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("TOP 20 PREDICATES (Human Readable):")
    lines.append("=" * 70)
    
    for i, pred in enumerate(epoch_json[:20]):
        readable = interpreter.predicate_to_readable(pred)
        lines.append(f"#{i+1:3d}: {readable}")
    
    return "\n".join(lines)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    # Test with sample predicates
    sample_predicates = [
        "[GENERAL] {F772[36]*F312[0]} <= F937[32] (imp=0.007)",
        "[GENERAL] {minus_di[36]*F809[0]} <= F937[32] (imp=0.007)",
        "[MOMENTUM] F772[0] <= F772[32] (imp=0.007)",
    ]
    
    print(convert_epoch_output_to_readable(sample_predicates))
    
    # Test individual field resolution
    print("\n--- Field Name Resolution ---")
    for idx in [0, 3, 10, 45, 53, 54, 312, 772, 937]:
        print(f"  Index {idx:4d} -> {field_to_name(idx)}")
