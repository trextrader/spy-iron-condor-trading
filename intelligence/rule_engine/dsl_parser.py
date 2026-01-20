# intelligence/rule_engine/dsl_parser.py
"""
DSL parser for rule YAML files (V2.5 Schema).
Loads Complete_Ruleset_DSL.yaml and constructs rule objects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class PrimitiveSpec:
    """Specification for a single primitive invocation"""
    id: str
    alias: str
    func: Optional[str] = None  # Resolved function name
    inputs: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SignalLogicSpec:
    """Specification for signal entry/exit logic"""
    entry_long: Optional[str] = None
    entry_short: Optional[str] = None
    exit: Optional[str] = None
    min_confidence: float = 0.5

@dataclass
class GateSpec:
    """Specification for a gate in the execution stack"""
    id: str
    params: Dict[str, Any] = field(default_factory=dict)
    type: str = "BLOCK"  # BLOCK, FILTER, ADJUST
    action: str = "block" # block, dampen

@dataclass
class RuleSpec:
    """Specification for a single rule"""
    id: str
    name: str
    category: str
    strategy_type: str
    primitives: List[PrimitiveSpec]
    signal_logic: SignalLogicSpec
    gate_stack: List[GateSpec]
    sizing_logic: Dict[str, Any]
    requires: Dict[str, List[str]]

@dataclass
class Ruleset:
    """Complete ruleset containing all rules"""
    version: str
    rules: Dict[str, RuleSpec]

class RuleDSLParser:
    """
    Parses YAML rule files into Ruleset objects.
    Handles V2.5 schema with nested primitive definitions.
    """

    def __init__(self, path: str):
        self.path = path

    def load(self) -> Ruleset:
        """Load and parse the YAML ruleset file"""
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        rules = {}
        metadata = data.get("ruleset_metadata", {})
        version = metadata.get("version", "unknown")

        # Iterate over all top-level keys that look like rules (RULE_*)
        for key, rule_def in data.items():
            if not key.startswith("RULE_"):
                continue

            rule_id = rule_def.get("id", key)
            
            # Parse Primitives
            primitives = []
            for p in rule_def.get("primitives", []):
                primitives.append(PrimitiveSpec(
                    id=p["id"],
                    alias=p.get("alias", p["id"]),
                    params=p.get("params", {})
                ))

            # Parse Signal Logic
            signals = rule_def.get("signals", {})
            entry = signals.get("entry", {})
            exit_logic = signals.get("exit", {}).get("logic")
            
            signal_spec = SignalLogicSpec(
                entry_long=entry.get("long", {}).get("logic"),
                entry_short=entry.get("short", {}).get("logic"),
                exit=exit_logic,
                min_confidence=entry.get("long", {}).get("min_confidence", 0.5) 
            )

            # Parse Gates
            gates = []
            for g in rule_def.get("gates", []):
                gates.append(GateSpec(
                    id=g["id"],
                    params=g.get("params", {}),
                    type=g.get("type", "BLOCK"),
                    action=g.get("action", "block")
                ))

            rules[rule_id] = RuleSpec(
                id=rule_id,
                name=rule_def.get("name", rule_id),
                category=rule_def.get("category", "unknown"),
                strategy_type=rule_def.get("strategy_type", "unknown"),
                primitives=primitives,
                signal_logic=signal_spec,
                gate_stack=gates,
                sizing_logic=rule_def.get("sizing", {}),
                requires=rule_def.get("requires", {})
            )

        logger.info(f"Loaded {len(rules)} rules from {self.path} (v{version})")
        return Ruleset(version=version, rules=rules)
