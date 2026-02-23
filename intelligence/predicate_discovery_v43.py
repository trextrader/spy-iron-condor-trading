"""
Differentiable Predicate Discovery Engine — V4.3

Learns which inequality patterns (from billions of possibilities)
are predictive for trading signals across ALL v4.3 input domains:

- 4 TF datasets (M1, M5, M15, H1) — 52 features each (schema_v43 Groups 1-6)
- 13 Pivot features (sparse, NaN-meaningful)
- 10 Options chain features per contract (greeks, IV, bid/ask, OI)

Total discoverable temporal fields: 221  (52 × 4 TFs + 13 pivots)
Chain features handled separately via ChainPredicateSelector.

Grammar:
  Atom := TF_Field[n]                        # e.g. m1.close[5]
        | {TF_Field1[n] op TF_Field2[m]}     # e.g. {m1.high[0] - m1.low[0]}

  Expr := Atom cmp Atom                       # Single inequality

  Predicate := Expr                            # Depth 1
             | Expr AND/OR Predicate          # Depth 2-4

  CrossTF := m1.Field[n] cmp h1.Field[m]     # Cross-timeframe

Templates:
  0: GENERAL     — Unconstrained
  1: MOMENTUM    — Self-comparison: F[0] cmp F[n]
  2: RELATIVE    — Peer comparison: F1[0] cmp F2[0]
  3: CROSSOVER   — Dual: {F[0]-F[n]} cmp {G[0]-G[m]}
  4: THRESHOLD   — Constant: F[0] cmp K
  5: CROSS_TF    — Cross-timeframe: m1.F[0] cmp m15.F[0]
  6: CHAIN_SPREAD — Chain domain: iv[contract_i] cmp iv[contract_j]

Author: CondorNet v4.3 Implementation
Version: 4.3.0
Date: 2026-02-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Set
from dataclasses import dataclass, field
from enum import IntEnum
import json
import warnings

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available — predicate evaluation will be slower")

from intelligence.schema_v43 import (
    TF_FEATURE_NAMES,
    TF_PIVOT_FEATURES,
    CHAIN_FEATURE_NAMES,
)

# =============================================================================
# V4.3 FIELD GRAMMAR — 221 temporal fields
# =============================================================================

TF_PREFIXES: List[str] = ["m1", "m5", "m15", "h1"]
"""Timeframe prefixes for namespaced field indexing."""

# Build namespaced field list: m1.open, m1.high, ..., h1.SlopeATR, pivot.PivotHigh, ...
_all_fields: List[str] = []
for prefix in TF_PREFIXES:
    for feat in TF_FEATURE_NAMES:
        _all_fields.append(f"{prefix}.{feat}")

for feat in TF_PIVOT_FEATURES:
    _all_fields.append(f"pivot.{feat}")

ALL_FIELDS_V43: List[str] = _all_fields
FIELD_TO_IDX_V43: Dict[str, int] = {f: i for i, f in enumerate(ALL_FIELDS_V43)}
IDX_TO_FIELD_V43: Dict[int, str] = {i: f for i, f in enumerate(ALL_FIELDS_V43)}
N_FIELDS_V43: int = len(ALL_FIELDS_V43)

# Derived constants for namespace ranges
N_TF_FEATURES: int = len(TF_FEATURE_NAMES)  # 51 per TF
N_PIVOT_FEATURES: int = len(TF_PIVOT_FEATURES)  # 13
N_CHAIN_FEATURES: int = len(CHAIN_FEATURE_NAMES)  # 10

# Index ranges per namespace
TF_FIELD_RANGES: Dict[str, Tuple[int, int]] = {}
for i, prefix in enumerate(TF_PREFIXES):
    start = i * N_TF_FEATURES
    end = (i + 1) * N_TF_FEATURES
    TF_FIELD_RANGES[prefix] = (start, end)
PIVOT_FIELD_RANGE: Tuple[int, int] = (
    len(TF_PREFIXES) * N_TF_FEATURES,
    len(TF_PREFIXES) * N_TF_FEATURES + N_PIVOT_FEATURES,
)

print(f"[PredicateDiscovery V43] Fields: {N_FIELDS_V43} "
      f"({N_TF_FEATURES}×{len(TF_PREFIXES)} TF + {N_PIVOT_FEATURES} pivot)")


# =============================================================================
# GRAMMAR DEFINITION
# =============================================================================

class ArithOp(IntEnum):
    """Arithmetic operators for compound atoms."""
    NONE = 0   # Simple atom
    ADD = 1    # +
    SUB = 2    # -
    MUL = 3    # *
    DIV = 4    # / (safe)


class CompareOp(IntEnum):
    """Comparison operators."""
    GT = 0    # >
    LT = 1    # <
    GTE = 2   # >=
    LTE = 3   # <=
    EQ = 4    # == (with epsilon)


class LogicOp(IntEnum):
    """Logical connectives."""
    AND = 0
    OR = 1


class TemplateType(IntEnum):
    """Structural templates for guided predicate discovery."""
    GENERAL = 0
    MOMENTUM = 1
    RELATIVE = 2
    CROSSOVER = 3
    THRESHOLD = 4
    CROSS_TF = 5       # NEW: Cross-timeframe comparison
    CHAIN_SPREAD = 6   # NEW: Options chain domain


COMPARE_SYMBOLS = ['>', '<', '>=', '<=', '==']
ARITH_SYMBOLS = ['', '+', '-', '*', '/']
LOGIC_SYMBOLS = ['AND', 'OR']
TEMPLATE_NAMES = ['GENERAL', 'MOMENTUM', 'RELATIVE', 'CROSSOVER',
                  'THRESHOLD', 'CROSS_TF', 'CHAIN_SPREAD']


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Atom:
    """
    Single or compound atom in an inequality.

    Simple: Field[lookback]     e.g. m1.close[5]
    Compound: {Field1[n] op Field2[m]}
    """
    field1: int            # Index into ALL_FIELDS_V43
    lookback1: int         # Lookback period for field1
    arith_op: ArithOp      # NONE for simple atom
    field2: int = 0        # Second field (only if arith_op != NONE)
    lookback2: int = 0

    def is_simple(self) -> bool:
        return self.arith_op == ArithOp.NONE

    def to_tuple(self) -> Tuple:
        return (self.field1, self.lookback1, int(self.arith_op),
                self.field2, self.lookback2)

    @staticmethod
    def from_tuple(t: Tuple) -> 'Atom':
        return Atom(t[0], t[1], ArithOp(t[2]), t[3], t[4])

    def __str__(self) -> str:
        f1 = IDX_TO_FIELD_V43.get(self.field1, f"F{self.field1}")
        if self.is_simple():
            return f"{f1}[{self.lookback1}]"
        f2 = IDX_TO_FIELD_V43.get(self.field2, f"F{self.field2}")
        op = ARITH_SYMBOLS[self.arith_op]
        return f"{{{f1}[{self.lookback1}]{op}{f2}[{self.lookback2}]}}"

    def max_lookback(self) -> int:
        if self.is_simple():
            return self.lookback1
        return max(self.lookback1, self.lookback2)


@dataclass
class Inequality:
    """Single inequality: LeftAtom cmp RightAtom."""
    left: Atom
    compare: CompareOp
    right: Atom

    def __str__(self) -> str:
        return f"{self.left} {COMPARE_SYMBOLS[self.compare]} {self.right}"

    def max_lookback(self) -> int:
        return max(self.left.max_lookback(), self.right.max_lookback())

    def to_encoding(self) -> np.ndarray:
        """Encode as fixed-size vector (11 floats)."""
        return np.array([
            self.left.field1, self.left.lookback1, self.left.arith_op,
            self.left.field2, self.left.lookback2,
            int(self.compare),
            self.right.field1, self.right.lookback1, self.right.arith_op,
            self.right.field2, self.right.lookback2
        ], dtype=np.float32)


@dataclass
class Predicate:
    """Compound predicate: 1-4 inequalities combined with AND/OR."""
    inequalities: List[Inequality]
    logic_ops: List[LogicOp] = field(default_factory=list)

    def __post_init__(self):
        if len(self.logic_ops) != max(0, len(self.inequalities) - 1):
            self.logic_ops = [LogicOp.AND] * (len(self.inequalities) - 1)

    def depth(self) -> int:
        return len(self.inequalities)

    def max_lookback(self) -> int:
        return max(ineq.max_lookback() for ineq in self.inequalities)

    def __str__(self) -> str:
        if self.depth() == 1:
            return str(self.inequalities[0])
        parts = [str(self.inequalities[0])]
        for ineq, op in zip(self.inequalities[1:], self.logic_ops):
            parts.append(f" {LOGIC_SYMBOLS[op]} {ineq}")
        return ''.join(parts)

    def to_encoding(self, max_depth: int = 4) -> np.ndarray:
        encoding = np.zeros(max_depth * 11 + (max_depth - 1), dtype=np.float32)
        for i, ineq in enumerate(self.inequalities):
            base = i * 11
            encoding[base:base + 11] = ineq.to_encoding()
        for i, op in enumerate(self.logic_ops):
            encoding[max_depth * 11 + i] = int(op)
        return encoding


# --- Paper-aligned structures ---

@dataclass
class InequalityChain:
    """Chained inequality: E₁ ○₁ E₂ ○₂ ... ○ₙ₋₁ Eₙ."""
    expressions: List[Atom]
    operators: List[CompareOp]

    def __post_init__(self):
        if len(self.operators) != max(0, len(self.expressions) - 1):
            raise ValueError(
                f"Operators ({len(self.operators)}) must be "
                f"len(expressions)-1 ({len(self.expressions) - 1})")

    def length(self) -> int:
        return len(self.expressions)

    def max_lookback(self) -> int:
        if not self.expressions:
            return 0
        return max(e.max_lookback() for e in self.expressions)

    def __str__(self) -> str:
        if not self.expressions:
            return "∅"
        parts = [str(self.expressions[0])]
        for expr, op in zip(self.expressions[1:], self.operators):
            parts.append(f" {COMPARE_SYMBOLS[op]} {expr}")
        return ''.join(parts)

    def to_encoding(self, max_length: int = 50) -> np.ndarray:
        encoding = np.zeros(max_length * 5 + (max_length - 1), dtype=np.float32)
        for i, expr in enumerate(self.expressions[:max_length]):
            base = i * 5
            encoding[base:base + 5] = [
                expr.field1, expr.lookback1, int(expr.arith_op),
                expr.field2, expr.lookback2
            ]
        for i, op in enumerate(self.operators[:max_length - 1]):
            encoding[max_length * 5 + i] = int(op)
        return encoding


@dataclass
class InequalitySet:
    """Set of chained inequalities combined with AND/OR."""
    chains: List[InequalityChain]
    logic_ops: List[LogicOp]
    set_id: int = 0

    def __post_init__(self):
        if self.chains and len(self.logic_ops) != len(self.chains) - 1:
            self.logic_ops = [LogicOp.AND] * (len(self.chains) - 1)

    def size(self) -> int:
        return len(self.chains)

    def max_lookback(self) -> int:
        if not self.chains:
            return 0
        return max(c.max_lookback() for c in self.chains)

    def __str__(self) -> str:
        if not self.chains:
            return f"Set{self.set_id}(∅)"
        if len(self.chains) == 1:
            return f"Set{self.set_id}({self.chains[0]})"
        parts = [f"({self.chains[0]})"]
        for chain, op in zip(self.chains[1:], self.logic_ops):
            parts.append(f" {LOGIC_SYMBOLS[op]} ({chain})")
        return f"Set{self.set_id}[{''.join(parts)}]"


@dataclass
class SuperSet:
    """Hierarchical comparison of inequality sets."""
    sets: List[InequalitySet]
    set_comparisons: List[CompareOp]
    aggregation: str = "pnorm"
    p_value: float = 1.0

    def __post_init__(self):
        if self.sets and len(self.set_comparisons) != len(self.sets) - 1:
            self.set_comparisons = [CompareOp.GT] * (len(self.sets) - 1)

    def depth(self) -> int:
        return len(self.sets)

    def __str__(self) -> str:
        if not self.sets:
            return "SuperSet(∅)"
        parts = [f"S{self.sets[0].set_id}"]
        for s, op in zip(self.sets[1:], self.set_comparisons):
            parts.append(f" {COMPARE_SYMBOLS[op]} S{s.set_id}")
        return f"SuperSet({' '.join(parts)})"

    def aggregate_set(self, truth_values: np.ndarray) -> float:
        if len(truth_values) == 0:
            return 0.0
        if self.aggregation == "pnorm":
            return self._pnorm_aggregate(truth_values, self.p_value)
        elif self.aggregation == "max":
            return float(np.max(truth_values))
        elif self.aggregation == "min":
            return float(np.min(truth_values))
        return float(np.mean(truth_values))

    def _pnorm_aggregate(self, values: np.ndarray, p: float) -> float:
        eps = 1e-8
        n = len(values)
        if n == 0:
            return 0.0
        if p > 50.0:
            return float(np.max(values))
        elif p < -50.0:
            return float(np.min(values))
        elif abs(p) < eps:
            return float(np.exp(np.mean(np.log(values + eps))))
        values = np.clip(values, eps, 1.0 - eps)
        return float(np.power(np.mean(np.power(values, p)), 1.0 / p))


# --- Chain domain data classes ---

@dataclass
class ChainAtom:
    """Atom referencing a feature across contracts in the options chain.

    field: index into CHAIN_FEATURE_NAMES (0-9)
    contract_idx: which contract to reference (0 = nearest ATM, etc.)
    """
    field: int
    contract_idx: int = 0

    def __str__(self) -> str:
        fname = CHAIN_FEATURE_NAMES[self.field] if self.field < len(CHAIN_FEATURE_NAMES) else f"CF{self.field}"
        return f"chain.{fname}[c{self.contract_idx}]"


@dataclass
class ChainInequality:
    """Inequality comparing features across contracts in the chain grid."""
    left: ChainAtom
    compare: CompareOp
    right: ChainAtom

    def __str__(self) -> str:
        return f"{self.left} {COMPARE_SYMBOLS[self.compare]} {self.right}"


# =============================================================================
# LEARNABLE AGGREGATION MODULES
# =============================================================================

class LearnablePNormAggregator(nn.Module):
    """Learnable p-norm aggregation for within-set truth value reduction."""

    def __init__(self, init_p: float = 1.0, min_p: float = -10.0,
                 max_p: float = 10.0):
        super().__init__()
        self.p_raw = nn.Parameter(torch.tensor(init_p))
        self.min_p = min_p
        self.max_p = max_p

    @property
    def p(self) -> torch.Tensor:
        return torch.clamp(self.p_raw, self.min_p, self.max_p)

    def forward(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        eps = 1e-8
        p = self.p
        values = torch.clamp(values, eps, 1.0 - eps)
        if p.abs() < eps:
            return torch.exp(torch.mean(torch.log(values), dim=dim))
        mean_powered = torch.mean(torch.pow(values, p), dim=dim)
        return torch.pow(mean_powered + eps, 1.0 / p)

    def extra_repr(self) -> str:
        return f"p={self.p.item():.2f}"


class LearnableOWAAggregator(nn.Module):
    """Learnable Ordered Weighted Averaging (OWA) aggregation."""

    def __init__(self, n_positions: int, init_mode: str = "mean"):
        super().__init__()
        self.n_positions = n_positions
        if init_mode == "max":
            logits = torch.linspace(2.0, -2.0, n_positions)
        elif init_mode == "min":
            logits = torch.linspace(-2.0, 2.0, n_positions)
        else:
            logits = torch.zeros(n_positions)
        self.weight_logits = nn.Parameter(logits)

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.weight_logits, dim=0)

    def forward(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        sorted_values, _ = torch.sort(values, dim=dim, descending=True)
        n = sorted_values.shape[dim]
        if n < self.n_positions:
            pad_shape = list(sorted_values.shape)
            pad_shape[dim] = self.n_positions - n
            padding = torch.zeros(pad_shape, device=sorted_values.device,
                                  dtype=sorted_values.dtype)
            sorted_values = torch.cat([sorted_values, padding], dim=dim)
        elif n > self.n_positions:
            sorted_values = sorted_values.narrow(dim, 0, self.n_positions)
        weights = self.weights
        if dim == -1:
            pass
        else:
            shape = [1] * sorted_values.dim()
            shape[dim] = self.n_positions
            weights = weights.view(shape)
        return torch.sum(sorted_values * weights, dim=dim)

    def extra_repr(self) -> str:
        w = self.weights.detach().cpu().numpy()
        return f"n_positions={self.n_positions}, weights=[{w[0]:.2f}...{w[-1]:.2f}]"


class HierarchicalSetAggregator(nn.Module):
    """Complete hierarchical aggregation for sets-of-sets."""

    def __init__(
        self,
        max_chain_length: int = 200,
        max_chains_per_set: int = 200,
        max_sets_per_super: int = 4,
        within_chain_mode: str = "pnorm",
        within_set_mode: str = "pnorm",
        between_set_mode: str = "pnorm",
        init_p_chain: float = 1.0,
        init_p_set: float = -1.0,
        init_p_super: float = 2.0,
    ):
        super().__init__()
        if within_chain_mode == "owa":
            self.chain_aggregator = LearnableOWAAggregator(
                max_chain_length, init_mode="mean")
        else:
            self.chain_aggregator = LearnablePNormAggregator(
                init_p=init_p_chain)

        if within_set_mode == "owa":
            self.set_aggregator = LearnableOWAAggregator(
                max_chains_per_set, init_mode="mean")
        else:
            self.set_aggregator = LearnablePNormAggregator(init_p=init_p_set)

        if between_set_mode == "owa":
            self.super_aggregator = LearnableOWAAggregator(
                max_sets_per_super, init_mode="mean")
        else:
            self.super_aggregator = LearnablePNormAggregator(
                init_p=init_p_super)

    def forward(self, chain_truths: torch.Tensor) -> torch.Tensor:
        set_truths = self.chain_aggregator(chain_truths, dim=-1)
        super_truths = self.set_aggregator(set_truths, dim=-1)
        return self.super_aggregator(super_truths, dim=-1)


# =============================================================================
# PREDICATE GRAMMAR V43 — Multi-TF Aware Sampling
# =============================================================================

class PredicateGrammarV43:
    """
    Search space definition for v4.3 multi-TF predicates.

    Supports cross-timeframe inequality sampling and pivot-aware templates.
    """

    FIBONACCI_LOOKBACKS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128]

    def __init__(
        self,
        max_lookback: int = 128,
        lookback_set: Optional[List[int]] = None,
        max_depth: int = 4,
        allow_compound: bool = True,
        allow_cross_tf: bool = True,
    ):
        self.max_lookback = max_lookback
        self.lookback_set = lookback_set or list(range(max_lookback + 1))
        self.max_depth = max_depth
        self.allow_compound = allow_compound
        self.allow_cross_tf = allow_cross_tf

        self.fields = ALL_FIELDS_V43
        self.n_fields = N_FIELDS_V43
        self.field_to_idx = FIELD_TO_IDX_V43
        self.n_lookbacks = len(self.lookback_set)

        self._compute_search_space()

    def _compute_search_space(self):
        n_simple = self.n_fields * self.n_lookbacks
        n_compound = 0
        if self.allow_compound:
            n_compound = (self.n_fields ** 2) * (self.n_lookbacks ** 2) * 4
        n_total = n_simple + n_compound
        n_single_ineq = (n_total ** 2) * 5

        self.search_space = {
            'simple_atoms': n_simple,
            'compound_atoms': n_compound,
            'total_atoms': n_total,
            'single_inequalities': n_single_ineq,
        }

        print(f"[PredicateGrammarV43] Search space:")
        print(f"  Fields: {self.n_fields} (4×{N_TF_FEATURES} TF + {N_PIVOT_FEATURES} pivot)")
        print(f"  Lookbacks: {self.n_lookbacks}")
        print(f"  Simple atoms: {n_simple:,}")
        print(f"  Compound atoms: {n_compound:,}")
        print(f"  Single inequalities: ~{n_single_ineq:,.0e}")

    def _get_tf_field(self, tf_prefix: Optional[str] = None) -> int:
        """Sample a field index, optionally restricted to a specific TF."""
        if tf_prefix and tf_prefix in TF_FIELD_RANGES:
            start, end = TF_FIELD_RANGES[tf_prefix]
            return np.random.randint(start, end)

        # Weight: 60% TF fields (any), 15% pivot, 25% OHLCV-biased
        r = np.random.random()
        if r < 0.25:
            # OHLCV bias: pick close/high/low/open from a random TF
            tf = np.random.choice(TF_PREFIXES)
            start, _ = TF_FIELD_RANGES[tf]
            return start + np.random.randint(4)  # OHLCV = indices 0-3
        elif r < 0.85:
            # Random TF field
            tf = np.random.choice(TF_PREFIXES)
            start, end = TF_FIELD_RANGES[tf]
            return np.random.randint(start, end)
        else:
            # Pivot field
            start, end = PIVOT_FIELD_RANGE
            return np.random.randint(start, end)

    def sample_atom(self, allow_compound: bool = True,
                    tf_prefix: Optional[str] = None) -> Atom:
        """Sample a random atom with multi-TF awareness."""
        if (allow_compound and self.allow_compound
                and np.random.random() < 0.3):
            arith_op = (ArithOp.SUB if np.random.random() < 0.7
                        else ArithOp(np.random.randint(1, 5)))
            return Atom(
                field1=self._get_tf_field(tf_prefix),
                lookback1=np.random.choice(self.lookback_set),
                arith_op=arith_op,
                field2=self._get_tf_field(tf_prefix),
                lookback2=np.random.choice(self.lookback_set),
            )
        return Atom(
            field1=self._get_tf_field(tf_prefix),
            lookback1=np.random.choice(self.lookback_set),
            arith_op=ArithOp.NONE,
        )

    def sample_inequality(self, template: Optional[TemplateType] = None
                          ) -> Inequality:
        """Sample a random inequality, optionally with template constraints."""
        if template == TemplateType.CROSS_TF and self.allow_cross_tf:
            tf_left = np.random.choice(TF_PREFIXES)
            tf_right = np.random.choice(
                [t for t in TF_PREFIXES if t != tf_left])
            # Same feature name, different TF
            feat_idx = np.random.randint(N_TF_FEATURES)
            left_start, _ = TF_FIELD_RANGES[tf_left]
            right_start, _ = TF_FIELD_RANGES[tf_right]
            left = Atom(left_start + feat_idx, 0, ArithOp.NONE)
            right = Atom(right_start + feat_idx,
                         np.random.choice(self.lookback_set), ArithOp.NONE)
            return Inequality(left, CompareOp(np.random.randint(5)), right)

        if template == TemplateType.MOMENTUM:
            field_idx = self._get_tf_field()
            left = Atom(field_idx, 0, ArithOp.NONE)
            right = Atom(field_idx,
                         max(1, np.random.choice(self.lookback_set)),
                         ArithOp.NONE)
            return Inequality(left, CompareOp(np.random.randint(5)), right)

        return Inequality(
            left=self.sample_atom(),
            compare=CompareOp(np.random.randint(5)),
            right=self.sample_atom(),
        )

    def sample_predicate(self, max_depth: Optional[int] = None) -> Predicate:
        depth = np.random.randint(1, (max_depth or self.max_depth) + 1)
        inequalities = []
        for i in range(depth):
            # 20% chance of cross-TF template
            if self.allow_cross_tf and np.random.random() < 0.2:
                inequalities.append(
                    self.sample_inequality(TemplateType.CROSS_TF))
            elif np.random.random() < 0.15:
                inequalities.append(
                    self.sample_inequality(TemplateType.MOMENTUM))
            else:
                inequalities.append(self.sample_inequality())
        logic_ops = [LogicOp(np.random.randint(2)) for _ in range(depth - 1)]
        return Predicate(inequalities, logic_ops)

    def sample_random(self, n: int) -> List[Predicate]:
        return [self.sample_predicate() for _ in range(n)]

    def sample_chain(self, max_length: int = 200) -> InequalityChain:
        length = np.random.randint(2, min(max_length + 1, 20))
        expressions = [self.sample_atom() for _ in range(length)]
        operators = [CompareOp(np.random.randint(5))
                     for _ in range(length - 1)]
        return InequalityChain(expressions=expressions, operators=operators)

    def sample_inequality_set(self, max_chains: int = 200,
                              max_chain_length: int = 200,
                              set_id: int = 0) -> InequalitySet:
        n_chains = np.random.randint(1, min(max_chains + 1, 10))
        chains = [self.sample_chain(max_chain_length)
                  for _ in range(n_chains)]
        logic_ops = [LogicOp(np.random.randint(2))
                     for _ in range(n_chains - 1)]
        return InequalitySet(chains=chains, logic_ops=logic_ops,
                             set_id=set_id)

    def sample_super_set(self, n_sets: int = 4) -> SuperSet:
        n_sets = min(n_sets, 4)
        sets = [self.sample_inequality_set(set_id=i) for i in range(n_sets)]
        comparisons = [CompareOp(np.random.randint(5))
                       for _ in range(n_sets - 1)]
        aggregation = np.random.choice(["mean", "max", "min"])
        return SuperSet(sets=sets, set_comparisons=comparisons,
                        aggregation=aggregation)


# =============================================================================
# VECTORIZED EVALUATORS (CPU)
# =============================================================================

def _evaluate_atom_vectorized_v43(
    data: np.ndarray,  # (N, N_FIELDS_V43)
    atom: Atom,
    eps: float = 1e-8
) -> np.ndarray:
    """Evaluate an atom over all timesteps. Returns (N,) array."""
    N = data.shape[0]
    if atom.lookback1 >= N:
        return np.zeros(N, dtype=np.float32)

    val1 = np.zeros(N, dtype=np.float32)
    val1[atom.lookback1:] = data[:N - atom.lookback1, atom.field1]

    if atom.is_simple():
        return val1

    val2 = np.zeros(N, dtype=np.float32)
    if atom.lookback2 < N:
        val2[atom.lookback2:] = data[:N - atom.lookback2, atom.field2]

    if atom.arith_op == ArithOp.ADD:
        return val1 + val2
    elif atom.arith_op == ArithOp.SUB:
        return val1 - val2
    elif atom.arith_op == ArithOp.MUL:
        return val1 * val2
    elif atom.arith_op == ArithOp.DIV:
        return val1 / (val2 + eps)
    return val1


def _evaluate_inequality_vectorized_v43(
    data: np.ndarray,
    ineq: Inequality,
    eps: float = 1e-6
) -> np.ndarray:
    """Evaluate an inequality over all timesteps. Returns (N,) boolean array."""
    left_val = _evaluate_atom_vectorized_v43(data, ineq.left)
    right_val = _evaluate_atom_vectorized_v43(data, ineq.right)

    if ineq.compare == CompareOp.GT:
        return (left_val > right_val).astype(np.float32)
    elif ineq.compare == CompareOp.LT:
        return (left_val < right_val).astype(np.float32)
    elif ineq.compare == CompareOp.GTE:
        return (left_val >= right_val).astype(np.float32)
    elif ineq.compare == CompareOp.LTE:
        return (left_val <= right_val).astype(np.float32)
    elif ineq.compare == CompareOp.EQ:
        return (np.abs(left_val - right_val) < eps).astype(np.float32)
    return np.zeros(data.shape[0], dtype=np.float32)


def evaluate_predicate_vectorized_v43(
    data: np.ndarray,  # (N, N_FIELDS_V43)
    pred: Predicate
) -> np.ndarray:
    """Evaluate a compound predicate over all timesteps."""
    if pred.depth() == 0:
        return np.zeros(data.shape[0], dtype=np.float32)

    result = _evaluate_inequality_vectorized_v43(data, pred.inequalities[0])
    for ineq, logic_op in zip(pred.inequalities[1:], pred.logic_ops):
        ineq_result = _evaluate_inequality_vectorized_v43(data, ineq)
        if logic_op == LogicOp.AND:
            result = result * ineq_result
        else:
            result = np.maximum(result, ineq_result)
    return result


def evaluate_predicates_v43(
    data: np.ndarray,           # (N, N_FIELDS_V43)
    predicates: List[Predicate]
) -> np.ndarray:
    """Evaluate multiple predicates over data. Returns (N, K)."""
    N = data.shape[0]
    K = len(predicates)
    result = np.zeros((N, K), dtype=np.float32)
    for k, pred in enumerate(predicates):
        result[:, k] = evaluate_predicate_vectorized_v43(data, pred)
    return result


def evaluate_predicates_batch_v43(
    data: np.ndarray,           # (N, N_FIELDS_V43)
    predicates: List[Predicate]
) -> np.ndarray:
    """Evaluate predicates over a single sequence. Returns (N, K)."""
    return evaluate_predicates_v43(data, predicates)


# =============================================================================
# GPU EVALUATOR — Differentiable soft-logic for training
# =============================================================================

def evaluate_predicates_gpu_v43(
    data: torch.Tensor,       # (batch, seq, N_FIELDS_V43) or (N, N_FIELDS_V43)
    params: torch.Tensor,     # (K, 13) predicate parameters
    importance: torch.Tensor, # (K,) importance weights
    max_active: int = 256,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Evaluate predicates on GPU using vectorized soft-logic.

    Operates on the concatenated (B, S, 217) multi-TF + pivot data tensor.
    Differentiable — intended for training.
    """
    if data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    batch, seq_len, n_fields = data.shape
    K = params.shape[0]
    device = data.device

    top_k = min(max_active, K)
    _, top_idx = torch.topk(importance, top_k)
    active_params = params[top_idx]
    active_importance = importance[top_idx]

    # Extract parameters
    l_f1 = active_params[:, 0].long()
    l_lb1 = active_params[:, 1].long()
    l_op = active_params[:, 2].long()
    l_f2 = active_params[:, 3].long()
    l_lb2 = active_params[:, 4].long()
    cmp_op = active_params[:, 5].long()
    r_f1 = active_params[:, 6].long()
    r_lb1 = active_params[:, 7].long()
    r_op = active_params[:, 8].long()
    r_f2 = active_params[:, 9].long()
    r_lb2 = active_params[:, 10].long()

    has_templates = active_params.shape[1] >= 13
    if has_templates:
        templates = active_params[:, 11].long()
        thresholds = active_params[:, 12]
    else:
        templates = torch.zeros(top_k, device=device, dtype=torch.long)
        thresholds = torch.zeros(top_k, device=device)

    # FP32 for stability
    working_data = data.to(torch.float32)
    working_eps = 1e-6

    t_seq = torch.arange(seq_len, device=device).view(1, seq_len, 1)

    # Field indices expanded for gathering
    l_f1_idx = l_f1.view(1, 1, top_k).expand(batch, seq_len, top_k)
    r_f1_idx = r_f1.view(1, 1, top_k).expand(batch, seq_len, top_k)
    l_f2_idx = l_f2.view(1, 1, top_k).expand(batch, seq_len, top_k)
    r_f2_idx = r_f2.view(1, 1, top_k).expand(batch, seq_len, top_k)

    l_lb1_long = l_lb1.view(1, 1, top_k)
    r_lb1_long = r_lb1.view(1, 1, top_k)
    l_lb2_long = l_lb2.view(1, 1, top_k)
    r_lb2_long = r_lb2.view(1, 1, top_k)

    def get_atom_values(field_idx_map, lb_map):
        """Gather values for B, S, K from data (B, S, F)."""
        return torch.gather(
            working_data.index_select(2, field_idx_map[0, 0, :]),
            1,
            (t_seq - lb_map).clamp(min=0).expand(batch, seq_len, top_k)
        )

    # Left side
    l_v1 = get_atom_values(l_f1_idx, l_lb1_long)
    left_val = l_v1

    compound_mask = (l_op > 0).view(1, 1, top_k)
    if compound_mask.any():
        l_v2 = get_atom_values(l_f2_idx, l_lb2_long)
        l_op_view = l_op.view(1, 1, top_k)
        left_val = torch.where(l_op_view == 1, left_val + l_v2, left_val)
        left_val = torch.where(l_op_view == 2, left_val - l_v2, left_val)
        left_val = torch.where(l_op_view == 3, left_val * l_v2, left_val)
        left_val = torch.where(l_op_view == 4,
                               left_val / (l_v2 + working_eps), left_val)

    left_val = torch.clamp(left_val, -1e4, 1e4)

    # Right side
    r_v1 = get_atom_values(r_f1_idx, r_lb1_long)
    right_val = r_v1

    template_view = templates.view(1, 1, top_k)
    thresh_view = thresholds.view(1, 1, top_k)
    right_val = torch.where(template_view == 4, thresh_view.float(), right_val)

    compound_mask_r = (r_op > 0).view(1, 1, top_k) & (template_view != 4)
    if compound_mask_r.any():
        r_v2 = get_atom_values(r_f2_idx, r_lb2_long)
        r_op_view = r_op.view(1, 1, top_k)
        right_val = torch.where(r_op_view == 1, right_val + r_v2, right_val)
        right_val = torch.where(r_op_view == 2, right_val - r_v2, right_val)
        right_val = torch.where(r_op_view == 3, right_val * r_v2, right_val)
        right_val = torch.where(r_op_view == 4,
                                right_val / (r_v2 + working_eps), right_val)

    right_val = torch.clamp(right_val, -1e4, 1e4)

    # Comparison with soft-logic
    diff = left_val - right_val
    steepness = 10.0
    op_view = cmp_op.view(1, 1, top_k)

    sig_pos = torch.sigmoid(steepness * diff)
    sig_neg = torch.sigmoid(-steepness * diff)
    eq_val = torch.exp(-steepness * diff.abs())

    result = torch.zeros_like(diff)
    result = torch.where((op_view == 0) | (op_view == 2), sig_pos, result)
    result = torch.where((op_view == 1) | (op_view == 3), sig_neg, result)
    result = torch.where(op_view == 4, eq_val, result)

    # Time mask
    max_lb_per_k = torch.max(
        torch.stack([l_lb1, r_lb1, l_lb2, r_lb2]), dim=0)[0]
    max_lb_view = max_lb_per_k.view(1, 1, top_k)
    time_mask = t_seq >= max_lb_view
    result = result * time_mask.float()

    # Importance weighting
    result = result * active_importance.view(1, 1, top_k).to(torch.float32)

    result = result.to(data.dtype)
    if squeeze_out:
        result = result.squeeze(0)
    return result


# =============================================================================
# PREDICATE SELECTOR V43 — Differentiable discovery with Gumbel-softmax
# =============================================================================

class PredicateSelectorV43(nn.Module):
    """
    Learns which predicates (from billions in the v4.3 search space)
    are useful for strategy selection across all 10 strategy types.

    Expanded for 217 temporal fields + 7 template types.
    """

    def __init__(
        self,
        n_slots: int = 4096,
        max_active: int = 512,
        d_embed: int = 128,
        n_fields: int = N_FIELDS_V43,
        max_lookback: int = 128,
        max_depth: int = 4,
        temperature: float = 1.0
    ):
        super().__init__()

        self.n_slots = n_slots
        self.max_active = max_active
        self.n_fields = n_fields
        self.max_lookback = max_lookback
        self.max_depth = max_depth
        self.temperature = temperature

        # Recursive logic: fields + predicate slot outputs
        self.total_input_fields = n_fields + n_slots

        # Learnable predicate embeddings
        self.predicate_embeddings = nn.Parameter(
            torch.randn(n_slots, d_embed) * 0.01)

        # Shared decoder backbone
        self.decoder = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.GELU(),
            nn.LayerNorm(d_embed * 2),
            nn.Linear(d_embed * 2, d_embed * 2),
            nn.GELU(),
        )

        # Parameter heads
        self.left_field1 = nn.Linear(d_embed * 2, self.total_input_fields)
        self.left_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.left_arith = nn.Linear(d_embed * 2, 5)
        self.left_field2 = nn.Linear(d_embed * 2, self.total_input_fields)
        self.left_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        self.compare_op = nn.Linear(d_embed * 2, 5)

        self.right_field1 = nn.Linear(d_embed * 2, self.total_input_fields)
        self.right_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.right_arith = nn.Linear(d_embed * 2, 5)
        self.right_field2 = nn.Linear(d_embed * 2, self.total_input_fields)
        self.right_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        # Importance scores — initialize sparse
        self.importance_logits = nn.Parameter(torch.ones(n_slots) * -5.0)

        # Template head — 7 types (v43: +CROSS_TF, +CHAIN_SPREAD)
        self.template_head = nn.Linear(d_embed * 2, 7)

        # Threshold head for THRESHOLD template
        self.threshold_head = nn.Linear(d_embed * 2, 1)

        self._cached_predicates = None
        self._cache_valid = False

    def _gumbel_sample(self, logits: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            indices = (probs * torch.arange(
                logits.size(-1), device=logits.device)).sum(-1)
        else:
            indices = logits.argmax(-1).float()
            probs = F.one_hot(indices.long(), logits.size(-1)).float()
        return probs, indices

    def forward(self, return_params: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute importance weights and optionally decode predicate parameters.

        Returns:
            importance: (n_slots,) importance weights
            params: (n_slots, 13) encoded parameters if return_params=True
        """
        importance = torch.sigmoid(self.importance_logits)

        if not return_params:
            return importance, None

        hidden = self.decoder(self.predicate_embeddings)

        # Left atom
        _, l_f1 = self._gumbel_sample(self.left_field1(hidden))
        _, l_n1 = self._gumbel_sample(self.left_lookback1(hidden))
        _, l_op = self._gumbel_sample(self.left_arith(hidden))
        _, l_f2 = self._gumbel_sample(self.left_field2(hidden))
        _, l_n2 = self._gumbel_sample(self.left_lookback2(hidden))

        _, cmp = self._gumbel_sample(self.compare_op(hidden))

        # Right atom
        _, r_f1 = self._gumbel_sample(self.right_field1(hidden))
        _, r_n1 = self._gumbel_sample(self.right_lookback1(hidden))
        _, r_op = self._gumbel_sample(self.right_arith(hidden))
        _, r_f2 = self._gumbel_sample(self.right_field2(hidden))
        _, r_n2 = self._gumbel_sample(self.right_lookback2(hidden))

        # Template selection
        _, templates = self._gumbel_sample(self.template_head(hidden))

        # --- Template constraints ---

        # MOMENTUM: same field, lookback 0 vs n
        is_mom = (templates == TemplateType.MOMENTUM)
        r_f1 = torch.where(is_mom, l_f1, r_f1)
        l_n1 = torch.where(is_mom, torch.zeros_like(l_n1), l_n1)
        l_op = torch.where(is_mom, torch.zeros_like(l_op), l_op)
        r_op = torch.where(is_mom, torch.zeros_like(r_op), r_op)

        # RELATIVE: same time, different fields
        is_rel = (templates == TemplateType.RELATIVE)
        l_n1 = torch.where(is_rel, torch.zeros_like(l_n1), l_n1)
        r_n1 = torch.where(is_rel, torch.zeros_like(r_n1), r_n1)
        l_op = torch.where(is_rel, torch.zeros_like(l_op), l_op)
        r_op = torch.where(is_rel, torch.zeros_like(r_op), r_op)

        # CROSSOVER: dual subtraction
        is_cross = (templates == TemplateType.CROSSOVER)
        l_op = torch.where(is_cross,
                           torch.full_like(l_op, ArithOp.SUB), l_op)
        r_op = torch.where(is_cross,
                           torch.full_like(r_op, ArithOp.SUB), r_op)
        l_f2 = torch.where(is_cross, l_f1, l_f2)
        r_f2 = torch.where(is_cross, r_f1, r_f2)
        l_n1 = torch.where(is_cross, torch.zeros_like(l_n1), l_n1)
        r_n1 = torch.where(is_cross, torch.zeros_like(r_n1), r_n1)

        # CROSS_TF: ensure left and right are from different TFs
        # (soft constraint — the field head will learn to select cross-TF)
        is_xtf = (templates == TemplateType.CROSS_TF)
        l_op = torch.where(is_xtf, torch.zeros_like(l_op), l_op)
        r_op = torch.where(is_xtf, torch.zeros_like(r_op), r_op)

        # THRESHOLD: right side is learned constant
        is_thresh = (templates == TemplateType.THRESHOLD)
        thresh_val = self.threshold_head(hidden).squeeze(-1)

        params = torch.stack([
            l_f1, l_n1, l_op, l_f2, l_n2,
            cmp,
            r_f1, r_n1, r_op, r_f2, r_n2,
            templates,
            thresh_val
        ], dim=-1)

        return importance, params

    def get_active_predicates(
        self,
        threshold: float = 0.1,
        max_return: Optional[int] = None
    ) -> Tuple[List[Predicate], np.ndarray, List[str]]:
        """Extract predicates with importance above threshold."""
        self.eval()
        with torch.no_grad():
            importance, params = self.forward(return_params=True)

            if max_return is None:
                max_return = self.max_active

            mask = importance > threshold
            active_idx = torch.where(mask)[0]

            if len(active_idx) == 0:
                top_k = min(self.n_slots, max_return)
                _, top_idx = torch.topk(importance, top_k)
                active_idx = top_idx
            else:
                active_importance_filtered = importance[active_idx]
                sort_idx = torch.argsort(
                    active_importance_filtered, descending=True)
                active_idx = active_idx[sort_idx]
                if len(active_idx) > max_return:
                    active_idx = active_idx[:max_return]

            if len(active_idx) == 0:
                return [], np.array([]), []

            active_params = params[active_idx].cpu().numpy()
            active_importance = importance[active_idx].cpu().numpy()

            predicates = []
            names = []

            for i, p in enumerate(active_params):
                left = Atom(int(p[0]), int(p[1]),
                            ArithOp(int(p[2])), int(p[3]), int(p[4]))
                right = Atom(int(p[6]), int(p[7]),
                             ArithOp(int(p[8])), int(p[9]), int(p[10]))
                ineq = Inequality(left, CompareOp(int(p[5])), right)
                pred = Predicate([ineq], [])

                t_idx = int(p[11]) if len(p) > 11 else 0
                t_name = (TEMPLATE_NAMES[t_idx]
                          if t_idx < len(TEMPLATE_NAMES) else "UNKNOWN")

                if t_idx == TemplateType.THRESHOLD:
                    t_val = p[12]
                    names.append(
                        f"[{t_name}] {left} "
                        f"{COMPARE_SYMBOLS[int(p[5])]} {t_val:.3f} "
                        f"(imp={active_importance[i]:.3f})")
                else:
                    names.append(
                        f"[{t_name}] {pred} "
                        f"(imp={active_importance[i]:.3f})")

                predicates.append(pred)

            return predicates, active_importance, names

    def sparsity_loss(self) -> torch.Tensor:
        """L1 regularization on importance weights."""
        return torch.sigmoid(self.importance_logits).sum()

    def diversity_loss(self) -> torch.Tensor:
        """Penalize predicates for using the same features."""
        hidden = self.decoder(self.predicate_embeddings)

        l1_logits = self.left_field1(hidden)
        l2_logits = self.left_field2(hidden)
        r1_logits = self.right_field1(hidden)
        r2_logits = self.right_field2(hidden)

        l1_probs = torch.softmax(l1_logits, dim=-1)
        l2_probs = torch.softmax(l2_logits, dim=-1)
        r1_probs = torch.softmax(r1_logits, dim=-1)
        r2_probs = torch.softmax(r2_logits, dim=-1)

        importance = torch.sigmoid(self.importance_logits).view(-1, 1)
        importance_norm = importance / (importance.sum() + 1e-8)

        avg_probs = (
            (l1_probs * importance_norm).sum(0) +
            (l2_probs * importance_norm).sum(0) +
            (r1_probs * importance_norm).sum(0) +
            (r2_probs * importance_norm).sum(0)
        ) / 4.0

        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
        return -entropy  # Minimize = maximize entropy = more diversity

    def entropy_loss(self) -> torch.Tensor:
        """Encourage certainty in parameter selection."""
        self.eval()
        hidden = self.decoder(self.predicate_embeddings)

        total_entropy = 0.0
        for head in [self.left_field1, self.left_lookback1, self.left_arith,
                     self.left_field2, self.left_lookback2, self.compare_op,
                     self.right_field1, self.right_lookback1, self.right_arith,
                     self.right_field2, self.right_lookback2]:
            logits = head(hidden)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
            total_entropy += entropy

        return total_entropy


# =============================================================================
# CHAIN PREDICATE SELECTOR — Options chain domain
# =============================================================================

class ChainPredicateSelector(nn.Module):
    """
    Discovers predicates within the options chain grid [B, N, 10].

    Learns cross-contract comparisons:
    - iv[contract_i] > iv[contract_j]
    - moneyness[i] < threshold
    - delta[near_ATM] > delta[OTM]
    - bid_ask_spread[i] < bid_ask_spread[j]

    These predicates help determine which strategy type is appropriate
    given the current chain structure (e.g., skew patterns favoring
    butterflies vs iron condors vs straddles).
    """

    def __init__(
        self,
        n_chain_features: int = N_CHAIN_FEATURES,  # 10
        n_slots: int = 256,
        max_active: int = 64,
        d_embed: int = 64,
        max_contracts: int = 120,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.n_chain_features = n_chain_features
        self.n_slots = n_slots
        self.max_active = max_active
        self.max_contracts = max_contracts
        self.temperature = temperature

        # Learnable predicate embeddings
        self.predicate_embeddings = nn.Parameter(
            torch.randn(n_slots, d_embed) * 0.01)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.GELU(),
            nn.LayerNorm(d_embed * 2),
            nn.Linear(d_embed * 2, d_embed * 2),
            nn.GELU(),
        )

        # Heads: select feature, contract pair, and comparison
        self.left_feature = nn.Linear(d_embed * 2, n_chain_features)
        self.left_contract = nn.Linear(d_embed * 2, max_contracts)
        self.right_feature = nn.Linear(d_embed * 2, n_chain_features)
        self.right_contract = nn.Linear(d_embed * 2, max_contracts)
        self.compare_op = nn.Linear(d_embed * 2, 5)

        # Threshold for constant comparison
        self.threshold_head = nn.Linear(d_embed * 2, 1)
        self.is_threshold = nn.Linear(d_embed * 2, 2)  # 0=pair, 1=threshold

        # Importance
        self.importance_logits = nn.Parameter(torch.ones(n_slots) * -5.0)

    def _gumbel_sample(self, logits: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            indices = (probs * torch.arange(
                logits.size(-1), device=logits.device)).sum(-1)
        else:
            indices = logits.argmax(-1).float()
            probs = F.one_hot(indices.long(), logits.size(-1)).float()
        return probs, indices

    def forward(
        self,
        chain: torch.Tensor,       # (B, N_contracts, n_chain_features)
        chain_mask: torch.Tensor,   # (B, N_contracts) True=padded
    ) -> torch.Tensor:
        """
        Evaluate chain predicates.

        Returns:
            (B, max_active) predicate gate activations
        """
        B, N_contracts, _ = chain.shape
        device = chain.device

        importance = torch.sigmoid(self.importance_logits)
        top_k = min(self.max_active, self.n_slots)
        _, top_idx = torch.topk(importance, top_k)

        hidden = self.decoder(self.predicate_embeddings[top_idx])

        # Decode predicate parameters
        _, l_feat = self._gumbel_sample(self.left_feature(hidden))
        _, l_cont = self._gumbel_sample(self.left_contract(hidden))
        _, r_feat = self._gumbel_sample(self.right_feature(hidden))
        _, r_cont = self._gumbel_sample(self.right_contract(hidden))
        _, cmp = self._gumbel_sample(self.compare_op(hidden))
        _, is_thresh = self._gumbel_sample(self.is_threshold(hidden))
        thresh_val = self.threshold_head(hidden).squeeze(-1)

        # Clamp contract indices to available contracts
        l_cont = l_cont.long().clamp(0, N_contracts - 1)  # (top_k,)
        r_cont = r_cont.long().clamp(0, N_contracts - 1)
        l_feat = l_feat.long().clamp(0, self.n_chain_features - 1)
        r_feat = r_feat.long().clamp(0, self.n_chain_features - 1)

        # Gather left values: chain[b, l_cont[k], l_feat[k]]
        # Shape: (B, top_k)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, top_k)
        l_vals = chain[b_idx, l_cont.unsqueeze(0).expand(B, top_k),
                        l_feat.unsqueeze(0).expand(B, top_k)]

        # Right: either from chain or threshold
        r_vals = chain[b_idx, r_cont.unsqueeze(0).expand(B, top_k),
                        r_feat.unsqueeze(0).expand(B, top_k)]

        # Replace with threshold where is_thresh == 1
        is_thresh_mask = (is_thresh == 1).unsqueeze(0).expand(B, top_k)
        r_vals = torch.where(is_thresh_mask,
                             thresh_val.unsqueeze(0).expand(B, top_k),
                             r_vals)

        # Mask out padded contracts
        l_pad = chain_mask[b_idx, l_cont.unsqueeze(0).expand(B, top_k)]
        r_pad = chain_mask[b_idx, r_cont.unsqueeze(0).expand(B, top_k)]
        valid_mask = (~l_pad & (~r_pad | is_thresh_mask)).float()

        # Soft comparison
        diff = l_vals - r_vals
        steepness = 10.0
        cmp_view = cmp.long().unsqueeze(0).expand(B, top_k)

        sig_pos = torch.sigmoid(steepness * diff)
        sig_neg = torch.sigmoid(-steepness * diff)
        eq_val = torch.exp(-steepness * diff.abs())

        result = torch.zeros_like(diff)
        result = torch.where((cmp_view == 0) | (cmp_view == 2),
                             sig_pos, result)
        result = torch.where((cmp_view == 1) | (cmp_view == 3),
                             sig_neg, result)
        result = torch.where(cmp_view == 4, eq_val, result)

        # Apply validity mask and importance
        active_imp = importance[top_idx].unsqueeze(0).expand(B, top_k)
        result = result * valid_mask * active_imp

        return result  # (B, top_k)

    def sparsity_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.importance_logits).sum()


# =============================================================================
# PREDICATE COMBINER V43 — Multi-domain fusion
# =============================================================================

class PredicateCombinerV43(nn.Module):
    """
    Neural network that learns to combine discovered predicates from
    both the temporal domain (TF predicates) and chain domain.

    Takes:
    - TF predicate values: (B, S, K_tf) from PredicateSelectorV43
    - Chain predicate values: (B, K_chain) from ChainPredicateSelector

    Produces combined features for strategy selection.
    """

    def __init__(
        self,
        n_tf_predicates: int,
        n_chain_predicates: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        max_chain_depth: int = 8,
        n_output_heads: int = 10,
        n_chains: int = 256,
    ):
        super().__init__()

        self.n_tf_predicates = n_tf_predicates
        self.n_chain_predicates = n_chain_predicates
        self.max_chain_depth = max_chain_depth
        self.n_chains = n_chains

        # TF predicate embedding
        self.tf_predicate_embed = nn.Embedding(n_tf_predicates, d_model)

        # Chain attention: which TF predicates to combine
        self.chain_attention = nn.Parameter(
            torch.randn(n_chains, max_chain_depth, n_tf_predicates) * 0.01)

        # Logic gates
        self.logic_gates = nn.Parameter(
            torch.zeros(n_chains, max_chain_depth - 1))

        # Temporal processing for TF predicates
        self.temporal_proj = nn.Linear(n_tf_predicates, d_model)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1)

        # Chain predicate integration
        self.chain_proj = nn.Linear(n_chain_predicates, d_model // 2)

        # Output: TF + chain combined
        self.output_proj = nn.Sequential(
            nn.Linear(d_model + n_chains + d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_output_heads),
        )

    def forward(
        self,
        tf_predicate_values: torch.Tensor,   # (B, S, K_tf)
        chain_predicate_values: torch.Tensor, # (B, K_chain)
    ) -> torch.Tensor:
        """Returns (B, S, n_output_heads) predictions."""
        batch, seq_len, n_pred = tf_predicate_values.shape
        device = tf_predicate_values.device

        # === TF Chain Evaluation ===
        chain_attn = F.softmax(self.chain_attention, dim=-1)

        if tf_predicate_values.dtype != chain_attn.dtype:
            tf_predicate_values = tf_predicate_values.to(chain_attn.dtype)

        chain_values = torch.einsum(
            'bsp,cdp->bscd', tf_predicate_values, chain_attn)

        # Apply logic gates
        logic_weights = torch.sigmoid(self.logic_gates)
        combined = chain_values[:, :, :, 0]

        for d in range(1, self.max_chain_depth):
            next_val = chain_values[:, :, :, d]
            and_result = combined * next_val
            or_result = torch.maximum(combined, next_val)
            w = logic_weights[:, d - 1].unsqueeze(0).unsqueeze(0)
            combined = w * or_result + (1 - w) * and_result

        # Temporal features
        temporal_features = self.temporal_proj(tf_predicate_values)
        temporal_out, _ = self.temporal_attn(
            temporal_features, temporal_features, temporal_features)

        # Chain features — broadcast to sequence length
        chain_feat = self.chain_proj(chain_predicate_values)  # (B, d_model//2)
        chain_feat_seq = chain_feat.unsqueeze(1).expand(
            batch, seq_len, chain_feat.shape[-1])

        # Concatenate all features
        combined_features = torch.cat(
            [temporal_out, combined, chain_feat_seq], dim=-1)

        return self.output_proj(combined_features)

    def get_logic_sets(self, predicate_names: List[str],
                       threshold: float = 0.5) -> List[str]:
        """Extract human-readable logical chains."""
        self.eval()
        with torch.no_grad():
            chain_attn = F.softmax(
                self.chain_attention, dim=-1).cpu().numpy()
            logic_weights = torch.sigmoid(self.logic_gates).cpu().numpy()

            logic_sets = []
            for c in range(self.n_chains):
                active_elements = []
                for d in range(self.max_chain_depth):
                    best_idx = np.argmax(chain_attn[c, d])
                    if chain_attn[c, d, best_idx] > 0.3:
                        name = (predicate_names[best_idx]
                                if best_idx < len(predicate_names)
                                else f"P{best_idx}")
                        active_elements.append((d, name))

                if len(active_elements) < 2:
                    continue

                expr = f"({active_elements[0][1]})"
                for i in range(1, len(active_elements)):
                    depth_idx, name = active_elements[i]
                    prev_depth = active_elements[i - 1][0]
                    op = ("OR" if logic_weights[c, prev_depth] > 0.5
                          else "AND")
                    expr += f" {op} ({name})"

                logic_sets.append(expr)
            return logic_sets


# =============================================================================
# INTEGRATED MODEL: Multi-TF + Chain Predicates
# =============================================================================

class PredicateAugmentedModelV43(nn.Module):
    """
    Neural model augmented with learned predicates for all v4.3 domains.

    Architecture:
    1. PredicateSelectorV43 discovers temporal predicates (217 fields)
    2. ChainPredicateSelector discovers options chain predicates (10 features)
    3. Both evaluated on respective data domains
    4. PredicateCombinerV43 fuses temporal + chain predicate features
    5. Combined with backbone for multi-strategy decision making

    Supports all 10 strategy types and learns which market conditions
    favor each type through predicate-gated logic.
    """

    def __init__(
        self,
        raw_input_dim: int,          # Per-TF feature dim
        n_predicate_slots: int = 2048,
        max_active_predicates: int = 512,
        n_chain_predicate_slots: int = 256,
        max_active_chain_predicates: int = 64,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_output_heads: int = 10,    # 10 strategy types
        field_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.raw_input_dim = raw_input_dim
        self.max_active_tf = max_active_predicates
        self.max_active_chain = max_active_chain_predicates
        self.field_names = field_names or ALL_FIELDS_V43
        self.n_fields = len(self.field_names)

        # TF predicate discovery
        self.predicate_selector = PredicateSelectorV43(
            n_slots=n_predicate_slots,
            max_active=max_active_predicates,
            n_fields=self.n_fields,
        )

        # Chain predicate discovery
        self.chain_predicate_selector = ChainPredicateSelector(
            n_slots=n_chain_predicate_slots,
            max_active=max_active_chain_predicates,
        )

        # Predicate combiner (multi-domain)
        self.predicate_combiner = PredicateCombinerV43(
            n_tf_predicates=max_active_predicates,
            n_chain_predicates=max_active_chain_predicates,
            d_model=d_model // 2,
            n_output_heads=d_model // 2,
        )

        # Raw feature pathway
        combined_dim = raw_input_dim + max_active_predicates
        self.input_proj = nn.Linear(combined_dim, d_model)

        # Transformer backbone
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4,
                batch_first=True, dropout=0.1),
            num_layers=n_layers,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Output heads
        self.output_heads = nn.Linear(d_model, n_output_heads)

    def evaluate_tf_predicates_on_data(
        self,
        data: torch.Tensor,  # (B, S, N_FIELDS_V43)
        predicates: List[Predicate],
    ) -> torch.Tensor:
        """Evaluate temporal predicates on concatenated multi-TF data."""
        batch, seq_len, _ = data.shape
        device = data.device
        n_pred = len(predicates)

        if n_pred == 0:
            return torch.zeros(
                batch, seq_len, self.max_active_tf, device=device)

        data_np = data.detach().cpu().numpy()
        results = []
        for b in range(batch):
            pred_vals = evaluate_predicates_batch_v43(data_np[b], predicates)
            results.append(pred_vals)

        result = torch.tensor(
            np.stack(results), device=device, dtype=torch.float32)

        if n_pred < self.max_active_tf:
            padding = torch.zeros(
                batch, seq_len, self.max_active_tf - n_pred, device=device)
            result = torch.cat([result, padding], dim=-1)

        return result

    def forward(
        self,
        raw_features: torch.Tensor,            # (B, S, raw_input_dim)
        field_data: Optional[torch.Tensor] = None,  # (B, S, N_FIELDS_V43)
        chain: Optional[torch.Tensor] = None,       # (B, N, 10)
        chain_mask: Optional[torch.Tensor] = None,  # (B, N)
    ) -> torch.Tensor:
        """
        Forward pass with multi-domain predicate discovery.

        Args:
            raw_features: Normalized TF features (projected)
            field_data: Raw 217-field concatenated data for predicate evaluation
            chain: Options chain grid
            chain_mask: Chain padding mask (True=padded)

        Returns:
            (B, n_output_heads) strategy predictions
        """
        batch, seq_len, _ = raw_features.shape
        device = raw_features.device

        # === TF Predicate Discovery ===
        predicates, importance, _ = \
            self.predicate_selector.get_active_predicates(
                threshold=0.1, max_return=self.max_active_tf)

        if field_data is not None:
            tf_pred_features = self.evaluate_tf_predicates_on_data(
                field_data, predicates)
        else:
            field_data = raw_features[:, :, :self.n_fields]
            tf_pred_features = self.evaluate_tf_predicates_on_data(
                field_data, predicates)

        # === Chain Predicate Discovery ===
        if chain is not None and chain_mask is not None:
            chain_pred_features = self.chain_predicate_selector(
                chain, chain_mask)
        else:
            chain_pred_features = torch.zeros(
                batch, self.max_active_chain, device=device)

        # === Combine ===
        combined_input = torch.cat(
            [raw_features, tf_pred_features], dim=-1)

        x = self.input_proj(combined_input)
        x = self.backbone(x)
        backbone_out = x[:, -1, :]  # (B, d_model)

        combiner_out = self.predicate_combiner(
            tf_pred_features, chain_pred_features)
        combiner_last = combiner_out[:, -1, :]

        fused = self.fusion(torch.cat([backbone_out, combiner_last], dim=-1))
        return self.output_heads(fused)

    def get_discovered_predicates(self) -> List[str]:
        """Return human-readable list of discovered predicates."""
        _, _, names = self.predicate_selector.get_active_predicates(
            threshold=0.1)
        return names

    def total_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sparsity_weight: float = 0.001,
        entropy_weight: float = 0.0001,
    ) -> torch.Tensor:
        """Combined loss with regularization."""
        pred_loss = F.mse_loss(pred, target)
        sparse_loss = (self.predicate_selector.sparsity_loss() +
                       self.chain_predicate_selector.sparsity_loss())
        entropy_loss = self.predicate_selector.entropy_loss()
        return (pred_loss + sparsity_weight * sparse_loss +
                entropy_weight * entropy_loss)


# =============================================================================
# UTILITIES: Export / Import
# =============================================================================

def export_predicates_to_json_v43(
    predicates: List[Predicate],
    importance: np.ndarray,
    output_path: str,
):
    """Export discovered predicates for external use."""
    export = {
        'version': '4.3',
        'n_fields': N_FIELDS_V43,
        'n_predicates': len(predicates),
        'predicates': [],
    }

    for pred, imp in zip(predicates, importance):
        export['predicates'].append({
            'expression': str(pred),
            'importance': float(imp),
            'depth': pred.depth(),
            'max_lookback': pred.max_lookback(),
            'encoding': pred.to_encoding().tolist(),
        })

    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"[export_predicates_to_json_v43] "
          f"Saved {len(predicates)} predicates to {output_path}")


def load_predicates_from_json_v43(
    path: str,
) -> Tuple[List[Predicate], np.ndarray]:
    """Load predicates from JSON export."""
    with open(path, 'r') as f:
        data = json.load(f)

    predicates = []
    importance = []

    for p in data['predicates']:
        enc = np.array(p['encoding'], dtype=np.float32)
        left = Atom(int(enc[0]), int(enc[1]),
                     ArithOp(int(enc[2])), int(enc[3]), int(enc[4]))
        right = Atom(int(enc[6]), int(enc[7]),
                      ArithOp(int(enc[8])), int(enc[9]), int(enc[10]))
        ineq = Inequality(left, CompareOp(int(enc[5])), right)
        predicates.append(Predicate([ineq], []))
        importance.append(p['importance'])

    return predicates, np.array(importance)


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PREDICATE DISCOVERY ENGINE V4.3 — Demo")
    print("=" * 60)

    print(f"\nField Grammar: {N_FIELDS_V43} fields")
    print(f"  TF fields: {N_TF_FEATURES} × {len(TF_PREFIXES)} = "
          f"{N_TF_FEATURES * len(TF_PREFIXES)}")
    print(f"  Pivot fields: {N_PIVOT_FEATURES}")
    print(f"  Chain features: {N_CHAIN_FEATURES} (separate domain)")

    # Show field ranges
    for prefix, (start, end) in TF_FIELD_RANGES.items():
        print(f"  {prefix}: indices {start}–{end - 1}")
    ps, pe = PIVOT_FIELD_RANGE
    print(f"  pivot: indices {ps}–{pe - 1}")

    # Grammar demo
    grammar = PredicateGrammarV43(
        max_lookback=128,
        lookback_set=PredicateGrammarV43.FIBONACCI_LOOKBACKS,
    )

    print("\nSample random predicates:")
    samples = grammar.sample_random(10)
    for p in samples:
        print(f"  {p}")

    # Selector demo
    print("\n" + "=" * 60)
    print("PredicateSelectorV43 Architecture:")

    selector = PredicateSelectorV43(
        n_slots=2048,
        max_active=256,
        n_fields=N_FIELDS_V43,
    )

    total_params = sum(p.numel() for p in selector.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Predicate slots: 2048")
    print(f"  Max active: 256")
    print(f"  Fields: {N_FIELDS_V43}")

    # Chain selector demo
    print("\nChainPredicateSelector Architecture:")
    chain_sel = ChainPredicateSelector(n_slots=256, max_active=64)
    chain_params = sum(p.numel() for p in chain_sel.parameters())
    print(f"  Total parameters: {chain_params:,}")

    # Full model demo
    print("\n" + "=" * 60)
    print("PredicateAugmentedModelV43 Architecture:")

    model = PredicateAugmentedModelV43(
        raw_input_dim=N_FIELDS_V43,
        n_predicate_slots=2048,
        max_active_predicates=256,
        n_chain_predicate_slots=256,
        max_active_chain_predicates=64,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("\nTest forward pass:")
    B, S = 4, 128
    raw_feat = torch.randn(B, S, N_FIELDS_V43)
    chain_data = torch.randn(B, 60, N_CHAIN_FEATURES)
    chain_mask = torch.zeros(B, 60, dtype=torch.bool)

    with torch.no_grad():
        output = model(raw_feat, raw_feat, chain_data, chain_mask)

    print(f"  Input: ({B}, {S}, {N_FIELDS_V43})")
    print(f"  Chain: ({B}, 60, {N_CHAIN_FEATURES})")
    print(f"  Output: {output.shape}")
    print("\n✓ V4.3 Predicate Discovery Engine ready")
