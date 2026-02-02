"""
Differentiable Predicate Discovery Engine

Learns which inequality patterns (from millions of possibilities)
are predictive for trading signals. Discovered predicates augment
the neural CDE and decision tree.

Grammar:
  Atom := Field[n]                          # Simple: close[5]
        | {Field1[n] op Field2[m]}          # Compound: {high[0] - low[0]}

  Expr := Atom cmp Atom                     # Single inequality: H[n] > L[m]

  Predicate := Expr                         # Depth 1
             | Expr AND/OR Predicate        # Depth 2-4

Search Space:
  - 54 fields × 129 lookbacks = ~7,000 simple atoms
  - Compound atoms with +/-: ~7000² × 2 = ~98 million
  - Inequalities: atoms² × 5 comparators = astronomical
  - Chained (up to 4): effectively infinite

Solution: Differentiable sparse selection from learned embeddings.

Author: Claude Code
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import IntEnum
import json
import warnings

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available - predicate evaluation will be slower")

from .canonical_feature_registry import FEATURE_COLS_V22


# =============================================================================
# GRAMMAR DEFINITION
# =============================================================================

class FieldType(IntEnum):
    """All available fields from V2.2 schema + OHLCV."""
    # Enumerated dynamically from FEATURE_COLS_V22
    pass


# Build field enum dynamically
ALL_FIELDS: List[str] = FEATURE_COLS_V22.copy()
FIELD_TO_IDX: Dict[str, int] = {f: i for i, f in enumerate(ALL_FIELDS)}
IDX_TO_FIELD: Dict[int, str] = {i: f for i, f in enumerate(ALL_FIELDS)}
N_FIELDS: int = len(ALL_FIELDS)


import torch.nn.functional as F


class ArithOp(IntEnum):
    """Arithmetic operators for compound atoms."""
    NONE = 0   # Simple atom (no operation)
    ADD = 1    # +
    SUB = 2    # -
    MUL = 3    # *
    DIV = 4    # / (safe division)


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
    GENERAL = 0    # No constraints: {F1[n] op F2[m]} cmp {F3[x] op F4[y]}
    MOMENTUM = 1   # Self-comparison: F1[0] cmp F1[n]
    RELATIVE = 2   # Peer comparison: F1[0] cmp F2[0]
    CROSSOVER = 3  # Dual comparison: {F1[0] - F1[n]} cmp {F2[0] - F2[m]}
    THRESHOLD = 4  # Constant compare: F1[0] cmp K


COMPARE_SYMBOLS = ['>', '<', '>=', '<=', '==']
ARITH_SYMBOLS = ['', '+', '-', '*', '/']
LOGIC_SYMBOLS = ['AND', 'OR']


@dataclass
class Atom:
    """
    Single or compound atom in an inequality.

    Simple: Field[lookback]
    Compound: {Field1[n] op Field2[m]}
    """
    field1: int            # Index into ALL_FIELDS
    lookback1: int         # Lookback period for field1
    arith_op: ArithOp      # NONE for simple atom
    field2: int = 0        # Second field (only if arith_op != NONE)
    lookback2: int = 0     # Lookback for field2

    def is_simple(self) -> bool:
        return self.arith_op == ArithOp.NONE

    def to_tuple(self) -> Tuple:
        return (self.field1, self.lookback1, int(self.arith_op), self.field2, self.lookback2)

    @staticmethod
    def from_tuple(t: Tuple) -> 'Atom':
        return Atom(t[0], t[1], ArithOp(t[2]), t[3], t[4])

    def __str__(self) -> str:
        f1 = IDX_TO_FIELD.get(self.field1, f"F{self.field1}")
        if self.is_simple():
            return f"{f1}[{self.lookback1}]"
        else:
            f2 = IDX_TO_FIELD.get(self.field2, f"F{self.field2}")
            op = ARITH_SYMBOLS[self.arith_op]
            return f"{{{f1}[{self.lookback1}]{op}{f2}[{self.lookback2}]}}"

    def max_lookback(self) -> int:
        if self.is_simple():
            return self.lookback1
        return max(self.lookback1, self.lookback2)


@dataclass
class Inequality:
    """
    Single inequality: LeftAtom cmp RightAtom

    Examples:
      H[0] > H[1]                          # Simple vs simple
      {H[0] - L[0]} > {H[1] - L[1]}        # Compound vs compound
      close[0] > {bb_upper_dyn[0] - bb_mu_dyn[0]}  # Mixed
    """
    left: Atom
    compare: CompareOp
    right: Atom

    def __str__(self) -> str:
        return f"{self.left} {COMPARE_SYMBOLS[self.compare]} {self.right}"

    def max_lookback(self) -> int:
        return max(self.left.max_lookback(), self.right.max_lookback())

    def to_encoding(self) -> np.ndarray:
        """Encode as fixed-size vector (11 floats)."""
        # Left atom: 5 values
        # Compare op: 1 value
        # Right atom: 5 values
        return np.array([
            self.left.field1, self.left.lookback1, self.left.arith_op,
            self.left.field2, self.left.lookback2,
            int(self.compare),
            self.right.field1, self.right.lookback1, self.right.arith_op,
            self.right.field2, self.right.lookback2
        ], dtype=np.float32)


@dataclass
class Predicate:
    """
    Compound predicate: 1-4 inequalities combined with AND/OR.

    Examples:
      H[0] > H[1]                                           # Depth 1
      H[0] > H[1] AND L[0] > L[1]                          # Depth 2
      H[0] > H[1] AND L[0] > L[1] OR C[0] > O[0]           # Depth 3
      {H[0]-L[0]} > {H[1]-L[1]} AND rsi_dyn[0] < 30        # With indicators
    """
    inequalities: List[Inequality]
    logic_ops: List[LogicOp] = field(default_factory=list)  # len = len(inequalities) - 1

    def __post_init__(self):
        if len(self.logic_ops) != max(0, len(self.inequalities) - 1):
            # Auto-fill with AND
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
        """Encode as fixed-size vector."""
        # Each inequality: 11 values
        # Logic ops: max_depth - 1 values
        encoding = np.zeros(max_depth * 11 + (max_depth - 1), dtype=np.float32)

        for i, ineq in enumerate(self.inequalities):
            base = i * 11
            encoding[base:base+11] = ineq.to_encoding()

        for i, op in enumerate(self.logic_ops):
            encoding[max_depth * 11 + i] = int(op)

        return encoding


# =============================================================================
# PAPER-ALIGNED STRUCTURES (Phase 2)
# =============================================================================
# Per "Recursive Inequality Logic for SPY Options Using Combinatorics and Group Theory"

@dataclass
class InequalityChain:
    """
    Chained inequality: E₁ ○₁ E₂ ○₂ E₃ ... ○ₙ₋₁ Eₙ
    
    Paper: "A chained inequality is a sequence of expressions...  
           with up to 200 such chained inequalities"
    
    Example: H[3]-H[2] > H[5]-H[3] < H[10]-C[8] > ...
    
    This represents a SINGLE chain where comparisons flow left-to-right,
    evaluating as: (E₁ ○₁ E₂) AND (E₂ ○₂ E₃) AND ...
    """
    expressions: List[Atom]  # Up to 200 expressions (paper spec)
    operators: List[CompareOp]  # len = len(expressions) - 1
    
    def __post_init__(self):
        if len(self.operators) != max(0, len(self.expressions) - 1):
            raise ValueError(f"Operators ({len(self.operators)}) must be len(expressions)-1 ({len(self.expressions)-1})")
    
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
        """Encode chain as fixed-size vector."""
        # Each expression (Atom): 5 values
        # Each operator: 1 value
        # Total: max_length * 5 + (max_length - 1) = 6*max_length - 1
        encoding = np.zeros(max_length * 5 + (max_length - 1), dtype=np.float32)
        
        for i, expr in enumerate(self.expressions[:max_length]):
            base = i * 5
            encoding[base:base+5] = [
                expr.field1, expr.lookback1, int(expr.arith_op),
                expr.field2, expr.lookback2
            ]
        
        for i, op in enumerate(self.operators[:max_length-1]):
            encoding[max_length * 5 + i] = int(op)
        
        return encoding


@dataclass
class InequalitySet:
    """
    Set of chained inequalities combined with AND/OR.
    
    Paper: "An inequality set S is a conjunction (AND) or disjunction (OR) 
           of up to 200 such chained inequalities"
    
    S = ⋀ⱼ₌₁²⁰⁰ Iⱼ  OR  S = ⋁ⱼ₌₁²⁰⁰ Iⱼ
    """
    chains: List[InequalityChain]  # Up to 200 chains
    logic_ops: List[LogicOp]  # len = len(chains) - 1
    set_id: int = 0  # Identifier for super-set comparisons
    
    def __post_init__(self):
        if self.chains and len(self.logic_ops) != len(self.chains) - 1:
            # Default to AND
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
    """
    Hierarchical comparison of inequality sets.
    
    Paper: "A super set S = (S₁ # S₂ # S₃ # S₄) with up to four sets Sᵢ,
           each compared using <, >, ="
    
    S = (S₁ < S₂) AND (S₂ > S₃) AND (S₃ = S₄)
    """
    sets: List[InequalitySet]  # Up to 4 sets
    set_comparisons: List[CompareOp]  # Compare operators between sets
    aggregation: str = "pnorm"  # "pnorm", "owa", "mean", "max", "min"
    p_value: float = 1.0  # For p-norm: p→∞=max, p=1=mean, p→-∞=min
    
    def __post_init__(self):
        if self.sets and len(self.set_comparisons) != len(self.sets) - 1:
            # Default to ">"
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
        """Reduce a set's truth values to a single scalar using chosen aggregation."""
        if len(truth_values) == 0:
            return 0.0
        if self.aggregation == "pnorm":
            return self._pnorm_aggregate(truth_values, self.p_value)
        elif self.aggregation == "max":
            return float(np.max(truth_values))
        elif self.aggregation == "min":
            return float(np.min(truth_values))
        else:  # "mean" or default
            return float(np.mean(truth_values))
    
    def _pnorm_aggregate(self, values: np.ndarray, p: float) -> float:
        """
        Generalized p-norm aggregation:
        A_p(t₁,...,tₙ) = (1/n Σᵢ tᵢᵖ)^(1/p)
        
        - p → +∞: behaves like max
        - p = 1: mean
        - p → -∞: behaves like min
        """
        eps = 1e-8
        n = len(values)
        if n == 0:
            return 0.0
        
        # Handle edge cases
        if p > 50.0:  # Approximate max
            return float(np.max(values))
        elif p < -50.0:  # Approximate min
            return float(np.min(values))
        elif abs(p) < eps:  # Geometric mean (p→0)
            return float(np.exp(np.mean(np.log(values + eps))))
        else:
            # Clamp values to avoid numerical issues
            values = np.clip(values, eps, 1.0 - eps)
            return float(np.power(np.mean(np.power(values, p)), 1.0 / p))


# =============================================================================
# LEARNABLE AGGREGATION MODULES (Phase 3)
# =============================================================================
# Per user feedback: "treat aggregation as a learnable or at least configurable 
# operator drawn from a small, well-structured family"

class LearnablePNormAggregator(nn.Module):
    """
    Learnable p-norm aggregation for within-set truth value reduction.
    
    A_p(t₁,...,tₙ) = (1/n Σᵢ tᵢᵖ)^(1/p)
    
    - p → +∞: max (disjunctive)
    - p = 1: mean (averaging)
    - p → -∞: min (conjunctive)
    
    The parameter p is learned per aggregation layer.
    """
    def __init__(self, init_p: float = 1.0, min_p: float = -10.0, max_p: float = 10.0):
        super().__init__()
        # Learnable parameter (unconstrained, then clamped)
        self.p_raw = nn.Parameter(torch.tensor(init_p))
        self.min_p = min_p
        self.max_p = max_p
    
    @property
    def p(self) -> torch.Tensor:
        """Return clamped p value."""
        return torch.clamp(self.p_raw, self.min_p, self.max_p)
    
    def forward(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Aggregate values along specified dimension.
        
        Args:
            values: Tensor of truth values in [0, 1]
            dim: Dimension to aggregate over
            
        Returns:
            Aggregated tensor with dim reduced
        """
        eps = 1e-8
        p = self.p
        
        # Clamp values to avoid numerical issues
        values = torch.clamp(values, eps, 1.0 - eps)
        
        # p-norm aggregation
        if p.abs() < eps:
            # Geometric mean (p → 0)
            return torch.exp(torch.mean(torch.log(values), dim=dim))
        else:
            mean_powered = torch.mean(torch.pow(values, p), dim=dim)
            return torch.pow(mean_powered + eps, 1.0 / p)
    
    def extra_repr(self) -> str:
        return f"p={self.p.item():.2f}"


class LearnableOWAAggregator(nn.Module):
    """
    Learnable Ordered Weighted Averaging (OWA) aggregation.
    
    Sort truth values t_(1) ≥ ... ≥ t_(n), then:
    A_w(t₁,...,tₙ) = Σᵢ wᵢ * t_(i)
    
    Weights wᵢ ≥ 0 with Σwᵢ = 1 (enforced via softmax).
    
    - Max-like: most weight on t_(1)
    - Min-like: most weight on t_(n)
    - Mean-like: uniform weights
    """
    def __init__(self, n_positions: int, init_mode: str = "mean"):
        super().__init__()
        self.n_positions = n_positions
        
        # Initialize weight logits
        if init_mode == "max":
            # Most weight on highest values
            logits = torch.linspace(2.0, -2.0, n_positions)
        elif init_mode == "min":
            # Most weight on lowest values
            logits = torch.linspace(-2.0, 2.0, n_positions)
        else:  # "mean"
            logits = torch.zeros(n_positions)
        
        self.weight_logits = nn.Parameter(logits)
    
    @property
    def weights(self) -> torch.Tensor:
        """Return normalized weights via softmax."""
        return torch.softmax(self.weight_logits, dim=0)
    
    def forward(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Aggregate values using OWA.
        
        Args:
            values: Tensor of truth values in [0, 1]
            dim: Dimension to aggregate over
            
        Returns:
            Aggregated tensor with dim reduced
        """
        # Sort values in descending order
        sorted_values, _ = torch.sort(values, dim=dim, descending=True)
        
        # Pad or truncate to n_positions
        n = sorted_values.shape[dim]
        if n < self.n_positions:
            # Pad with zeros
            pad_shape = list(sorted_values.shape)
            pad_shape[dim] = self.n_positions - n
            padding = torch.zeros(pad_shape, device=sorted_values.device, dtype=sorted_values.dtype)
            sorted_values = torch.cat([sorted_values, padding], dim=dim)
        elif n > self.n_positions:
            # Truncate
            sorted_values = sorted_values.narrow(dim, 0, self.n_positions)
        
        # Apply weights
        weights = self.weights.view([1] * dim + [self.n_positions] + [1] * (sorted_values.dim() - dim - 1))
        if dim == -1:
            weights = self.weights
        
        return torch.sum(sorted_values * weights, dim=dim)
    
    def extra_repr(self) -> str:
        w = self.weights.detach().cpu().numpy()
        return f"n_positions={self.n_positions}, weights=[{w[0]:.2f}...{w[-1]:.2f}]"


class HierarchicalSetAggregator(nn.Module):
    """
    Complete hierarchical aggregation for sets-of-sets.
    
    Level 1 (within chain): Aggregate inequalities in a chain
    Level 2 (within set): Aggregate chains in an inequality set  
    Level 3 (between sets): Aggregate sets in a super-set
    Level 4 (super-set comparison): Compare super-set scalars
    
    Each level uses a learnable aggregator (p-norm or OWA).
    """
    def __init__(
        self,
        max_chain_length: int = 200,
        max_chains_per_set: int = 200,
        max_sets_per_super: int = 4,
        within_chain_mode: str = "pnorm",
        within_set_mode: str = "pnorm", 
        between_set_mode: str = "pnorm",
        init_p_chain: float = 1.0,  # Mean
        init_p_set: float = -1.0,   # Slightly conjunctive
        init_p_super: float = 2.0,  # Slightly disjunctive
    ):
        super().__init__()
        
        # Level 1: Within chain (conjunctive by default - all inequalities must hold)
        if within_chain_mode == "owa":
            self.chain_aggregator = LearnableOWAAggregator(max_chain_length, init_mode="mean")
        else:
            self.chain_aggregator = LearnablePNormAggregator(init_p=init_p_chain)
        
        # Level 2: Within set (average of chains)
        if within_set_mode == "owa":
            self.set_aggregator = LearnableOWAAggregator(max_chains_per_set, init_mode="mean")
        else:
            self.set_aggregator = LearnablePNormAggregator(init_p=init_p_set)
        
        # Level 3: Between sets (compare 4 sets)
        if between_set_mode == "owa":
            self.super_aggregator = LearnableOWAAggregator(max_sets_per_super, init_mode="mean")
        else:
            self.super_aggregator = LearnablePNormAggregator(init_p=init_p_super)
    
    def forward(
        self,
        chain_truths: torch.Tensor,  # (B, S, N_chains, Chain_len) or list of tensors
    ) -> torch.Tensor:
        """
        Aggregate hierarchically.
        
        Args:
            chain_truths: Truth values for each inequality in each chain
            
        Returns:
            Single scalar per batch element
        """
        # Level 1: Aggregate within chains
        set_truths = self.chain_aggregator(chain_truths, dim=-1)  # (B, S, N_chains)
        
        # Level 2: Aggregate within sets
        super_truths = self.set_aggregator(set_truths, dim=-1)  # (B, S)
        
        # Level 3: Aggregate across super-set
        result = self.super_aggregator(super_truths, dim=-1)  # (B,)
        
        return result



class PredicateGrammar:
    """
    Defines the search space of all possible predicates.

    Parameters:
        max_lookback: Maximum lookback period (0 to max_lookback inclusive)
        lookback_set: If provided, restrict to these specific lookbacks
        max_depth: Maximum number of chained inequalities (1-4)
        fields: List of field names to use (defaults to all V2.2)
        allow_compound: Whether to allow arithmetic compound atoms
    """

    # Fibonacci-inspired lookbacks for efficient search
    FIBONACCI_LOOKBACKS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128]

    def __init__(
        self,
        max_lookback: int = 128,
        lookback_set: Optional[List[int]] = None,
        max_depth: int = 4,
        fields: Optional[List[str]] = None,
        allow_compound: bool = True
    ):
        self.max_lookback = max_lookback
        self.lookback_set = lookback_set or list(range(max_lookback + 1))
        self.max_depth = max_depth
        self.allow_compound = allow_compound

        # Field configuration
        if fields is None:
            self.fields = ALL_FIELDS
        else:
            self.fields = [f for f in fields if f in ALL_FIELDS]

        self.field_to_idx = {f: i for i, f in enumerate(self.fields)}
        self.n_fields = len(self.fields)
        self.n_lookbacks = len(self.lookback_set)

        # Compute search space size
        self._compute_search_space()

    def _compute_search_space(self):
        """Estimate total search space size."""
        # Simple atoms: n_fields × n_lookbacks
        n_simple_atoms = self.n_fields * self.n_lookbacks

        # Compound atoms: n_fields² × n_lookbacks² × 4 arith ops
        if self.allow_compound:
            n_compound_atoms = (self.n_fields ** 2) * (self.n_lookbacks ** 2) * 4
        else:
            n_compound_atoms = 0

        n_total_atoms = n_simple_atoms + n_compound_atoms

        # Single inequalities: atoms² × 5 comparators
        n_single_ineq = (n_total_atoms ** 2) * 5

        self.search_space = {
            'simple_atoms': n_simple_atoms,
            'compound_atoms': n_compound_atoms,
            'total_atoms': n_total_atoms,
            'single_inequalities': n_single_ineq,
            'estimate_depth_4': n_single_ineq ** 4 * 8,  # Very rough
        }

        print(f"[PredicateGrammar] Search space estimates:")
        print(f"  Fields: {self.n_fields}")
        print(f"  Lookbacks: {self.n_lookbacks}")
        print(f"  Simple atoms: {n_simple_atoms:,}")
        print(f"  Compound atoms: {n_compound_atoms:,}")
        print(f"  Single inequalities: ~{n_single_ineq:,.0e}")

    def sample_atom(self, allow_compound: bool = True) -> Atom:
        """Sample a random atom with priority for OHLCV features."""
        # OHLCV Priority (indices 0-4)
        def get_field():
            if np.random.random() < 0.5: # 50% chance for OHLCV
                return np.random.randint(min(5, self.n_fields))
            return np.random.randint(self.n_fields)

        if allow_compound and self.allow_compound and np.random.random() < 0.3:
            # Compound atom - prioritize SUB for spreads (H-L, C-O, etc.)
            arith_op = ArithOp.SUB if np.random.random() < 0.7 else ArithOp(np.random.randint(1, 5))
            return Atom(
                field1=get_field(),
                lookback1=np.random.choice(self.lookback_set),
                arith_op=arith_op,
                field2=get_field(),
                lookback2=np.random.choice(self.lookback_set),
            )
        else:
            # Simple atom
            return Atom(
                field1=get_field(),
                lookback1=np.random.choice(self.lookback_set),
                arith_op=ArithOp.NONE,
            )

    def sample_inequality(self) -> Inequality:
        """Sample a random inequality."""
        return Inequality(
            left=self.sample_atom(),
            compare=CompareOp(np.random.randint(5)),
            right=self.sample_atom(),
        )

    def sample_predicate(self, max_depth: Optional[int] = None) -> Predicate:
        """Sample a random predicate with balanced templates (avoiding THRESHOLD bias)."""
        depth = np.random.randint(1, (max_depth or self.max_depth) + 1)

        # Force a diverse template choice for the inequalities
        inequalities = []
        for _ in range(depth):
            ineq = self.sample_inequality()
            # If it's a THRESHOLD template, we might want to re-sample or modify it
            # though usually templates are selected in the differentiable layer.
            # Here we ensure the random INITIALIZATION is diverse.
            inequalities.append(ineq)
            
        logic_ops = [LogicOp(np.random.randint(2)) for _ in range(depth - 1)]

        return Predicate(inequalities, logic_ops)

    def sample_random(self, n: int) -> List[Predicate]:
        """Sample n random predicates."""
        return [self.sample_predicate() for _ in range(n)]
    
    # =========================================================================
    # PAPER-ALIGNED SAMPLING (Phase 2)
    # =========================================================================
    
    def sample_chain(self, max_length: int = 200) -> 'InequalityChain':
        """
        Sample a chained inequality: E₁ ○₁ E₂ ○₂ ... Eₙ
        
        Paper: Up to 200 expressions per chain.
        """
        length = np.random.randint(2, max_length + 1)
        expressions = [self.sample_atom() for _ in range(length)]
        operators = [CompareOp(np.random.randint(5)) for _ in range(length - 1)]
        return InequalityChain(expressions=expressions, operators=operators)
    
    def sample_inequality_set(
        self, 
        max_chains: int = 200,
        max_chain_length: int = 200,
        set_id: int = 0
    ) -> 'InequalitySet':
        """
        Sample an inequality set: conjunction/disjunction of chains.
        
        Paper: Up to 200 chained inequalities per set.
        """
        n_chains = np.random.randint(1, max_chains + 1)
        chains = [self.sample_chain(max_chain_length) for _ in range(n_chains)]
        logic_ops = [LogicOp(np.random.randint(2)) for _ in range(n_chains - 1)]
        return InequalitySet(chains=chains, logic_ops=logic_ops, set_id=set_id)
    
    def sample_super_set(self, n_sets: int = 4) -> 'SuperSet':
        """
        Sample a super-set: hierarchical comparison of inequality sets.
        
        Paper: Up to 4 sets compared using <, >, =.
        """
        n_sets = min(n_sets, 4)
        sets = [self.sample_inequality_set(set_id=i) for i in range(n_sets)]
        comparisons = [CompareOp(np.random.randint(5)) for _ in range(n_sets - 1)]
        aggregation = np.random.choice(["mean", "max", "min"])
        return SuperSet(sets=sets, set_comparisons=comparisons, aggregation=aggregation)


# =============================================================================
# PREDICATE EVALUATOR (Vectorized)
# =============================================================================

def _evaluate_atom_vectorized(
    data: np.ndarray,  # (N, n_fields)
    atom: Atom,
    eps: float = 1e-8
) -> np.ndarray:
    """Evaluate an atom over all timesteps. Returns (N,) array."""
    N = data.shape[0]

    # Get first field with lookback
    if atom.lookback1 >= N:
        return np.zeros(N, dtype=np.float32)

    val1 = np.zeros(N, dtype=np.float32)
    val1[atom.lookback1:] = data[:N - atom.lookback1, atom.field1]

    if atom.is_simple():
        return val1

    # Get second field for compound
    val2 = np.zeros(N, dtype=np.float32)
    if atom.lookback2 < N:
        val2[atom.lookback2:] = data[:N - atom.lookback2, atom.field2]

    # Apply arithmetic operation
    if atom.arith_op == ArithOp.ADD:
        return val1 + val2
    elif atom.arith_op == ArithOp.SUB:
        return val1 - val2
    elif atom.arith_op == ArithOp.MUL:
        return val1 * val2
    elif atom.arith_op == ArithOp.DIV:
        return val1 / (val2 + eps)

    return val1


def _evaluate_inequality_vectorized(
    data: np.ndarray,
    ineq: Inequality,
    eps: float = 1e-6
) -> np.ndarray:
    """Evaluate an inequality over all timesteps. Returns (N,) boolean array."""
    left_val = _evaluate_atom_vectorized(data, ineq.left)
    right_val = _evaluate_atom_vectorized(data, ineq.right)

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


def evaluate_predicate_vectorized(
    data: np.ndarray,
    pred: Predicate
) -> np.ndarray:
    """
    Evaluate a compound predicate over all timesteps.

    Args:
        data: (N, n_fields) array of feature values
        pred: Predicate to evaluate

    Returns:
        (N,) array of boolean values (0 or 1)
    """
    if pred.depth() == 0:
        return np.zeros(data.shape[0], dtype=np.float32)

    # Evaluate first inequality
    result = _evaluate_inequality_vectorized(data, pred.inequalities[0])

    # Chain with subsequent inequalities
    for ineq, logic_op in zip(pred.inequalities[1:], pred.logic_ops):
        ineq_result = _evaluate_inequality_vectorized(data, ineq)

        if logic_op == LogicOp.AND:
            result = result * ineq_result  # Boolean AND
        else:
            result = np.maximum(result, ineq_result)  # Boolean OR

    return result


def evaluate_predicates(
    data: np.ndarray,           # (N, n_fields)
    predicates: List[Predicate]
) -> np.ndarray:
    """
    Evaluate multiple predicates over data.

    Args:
        data: (N, n_fields) array
        predicates: List of K predicates

    Returns:
        (N, K) array of boolean values
    """
    N = data.shape[0]
    K = len(predicates)
    result = np.zeros((N, K), dtype=np.float32)

    for k, pred in enumerate(predicates):
        result[:, k] = evaluate_predicate_vectorized(data, pred)

    return result


def evaluate_predicates_gpu(
    data: torch.Tensor,  # (batch, seq, n_fields) or (N, n_fields)
    params: torch.Tensor,  # (K, 11-13) predicate parameters
    importance: torch.Tensor,  # (K,) importance weights
    max_active: int = 256,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Evaluate predicates on GPU using vectorized operations with soft-logic.
    This is differentiable and intended for use during training.
    """
    # Handle 2D vs 3D input
    if data.dim() == 2:
        data = data.unsqueeze(0)  # (1, N, F)
        squeeze_out = True
    else:
        squeeze_out = False

    batch, seq_len, n_fields = data.shape
    K = params.shape[0]
    device = data.device

    # Select top-K by importance
    top_k = min(max_active, K)
    _, top_idx = torch.topk(importance, top_k)
    active_params = params[top_idx]  
    active_importance = importance[top_idx]  # (top_k,)

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
    
    # Optional Template Info
    has_templates = active_params.shape[1] >= 13
    if has_templates:
        templates = active_params[:, 11].long()
        thresholds = active_params[:, 12] # Float constant for THRESHOLD
    else:
        templates = torch.zeros(top_k, device=device, dtype=torch.long)
        thresholds = torch.zeros(top_k, device=device)

    # Max lookback for this batch
    max_lb = int(torch.max(active_params[:, [1, 4, 7, 10]]).item())
    max_lb = min(max_lb, seq_len - 1)

    # --- FP32 Internal Enforcement ---
    # We cast to FP32 for arithmetic calculations because MUL/DIV in FP16 
    # easily exceeds 65504 (overflow) leading to Infs and NaNs.
    working_data = data.to(torch.float32)
    working_eps = 1e-6 # Slightly larger for stability

    # Initialize output in FP32
    result = torch.zeros(batch, seq_len, max_active, device=device, dtype=torch.float32)

    # Optimized Vectorized Evaluation (No Python Loops)
    # -------------------------------------------------------------------------
    # 1. Create indices for gathering
    # We want: result[b, t, k] based on lookbacks l_lb1[k]
    
    # Time indices (1, S, 1) to (1, S, K)
    t_seq = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    
    # Lookback adjustments (1, 1, K)
    # We use .long() for indexing
    l_lb1_long = l_lb1.view(1, 1, top_k)
    r_lb1_long = r_lb1.view(1, 1, top_k)
    l_lb2_long = l_lb2.view(1, 1, top_k)
    r_lb2_long = r_lb2.view(1, 1, top_k)
    
    # Calculate effective T indices for each atom: t - lookback
    # result shape: (1, S, K) -> broadcast to batch later or gather directly
    # But data has batch dim (B, S, F).
    # We need to gather (B, S, K). 
    
    # Field indices (1, 1, K)
    l_f1_idx = l_f1.view(1, 1, top_k).expand(batch, seq_len, top_k)
    r_f1_idx = r_f1.view(1, 1, top_k).expand(batch, seq_len, top_k)
    l_f2_idx = l_f2.view(1, 1, top_k).expand(batch, seq_len, top_k)
    r_f2_idx = r_f2.view(1, 1, top_k).expand(batch, seq_len, top_k)

    def get_atom_values(field_idx_map, lb_map):
        """Gather values for B, S, K from data (B, S, F)."""
        # Calculate time indices: (B, S, K)
        # Clamp to 0 to avoid index error (will mask later if needed, or assume padded)
        t_gather = (t_seq - lb_map).clamp(min=0).expand(batch, seq_len, top_k)
        
        # We need to gather from 3D tensor data[b, t_gather[b,t,k], field_idx[b,t,k]]
        # Use torch.gather? Gather works on one dim.
        # Direct indexing with gather is tricky for multi-dim.
        # Easier: Gather fields first -> (B, S, K) using field_idx per K
        # data_fields = data.index_select(2, ...)? No, diff fields per K.
        # We can use gather on dim 2.
        
        # 1. Gather fields at all time steps. 
        # But wait, field is constant per K.
        # data index select is fast.
        
        # Let's use the 'unfold' trick or just simple advanced indexing?
        # PyTorch advanced indexing: data[b_idx, t_idx, f_idx]
        
        # Create Batch indices (B, S, K)
        # b_idx = torch.arange(batch, device=device).view(batch, 1, 1).expand(batch, seq_len, top_k)
        
        # return working_data[b_idx, t_gather, field_idx_map] << Memory heavy?
        # (128 * 240 * 256) * 4 bytes = 30MB. Is fine.
        
        # Manual gathering to avoid massive index tensor creation if possible.
        # Check step 1: Gather fields
        # fields = active_params[:, field_col].long()
        # sub_data = working_data.index_select(2, fields) # (B, S, K) - this gets Columns for each K
        
        # Step 2: Shift in time (Gather on dim 1)
        # t_gather = (t_seq - lb_map).clamp(min=0).expand(batch, seq_len, top_k)
        # final = torch.gather(sub_data, 1, t_gather)
        
        return torch.gather(working_data.index_select(2, field_idx_map[0,0,:]), 1, (t_seq - lb_map).clamp(min=0).expand(batch, seq_len, top_k))

    # Evaluate Left Side
    # ------------------
    l_v1 = get_atom_values(l_f1_idx, l_lb1_long)
    left_val = l_v1
    
    # LogicOps (Vectorized)
    # Mask for compound ops: l_op > 0
    compound_mask = (l_op > 0).view(1, 1, top_k)
    if compound_mask.any():
        l_v2 = get_atom_values(l_f2_idx, l_lb2_long)
        
        # Apply ops carefully using masks to avoid execution on NONE atoms
        # 1: ADD, 2: SUB, 3: MUL, 4: DIV
        l_op_view = l_op.view(1, 1, top_k)
        
        left_val = torch.where(l_op_view == 1, left_val + l_v2, left_val)
        left_val = torch.where(l_op_view == 2, left_val - l_v2, left_val)
        left_val = torch.where(l_op_view == 3, left_val * l_v2, left_val)
        left_val = torch.where(l_op_view == 4, left_val / (l_v2 + working_eps), left_val)

    # Stability Clip
    left_val = torch.clamp(left_val, -1e4, 1e4)

    # Evaluate Right Side
    # -------------------
    r_v1 = get_atom_values(r_f1_idx, r_lb1_long)
    right_val = r_v1
    
    # Handle Templates
    # 4: THRESHOLD (Constant)
    template_view = templates.view(1, 1, top_k)
    thresh_view = thresholds.view(1, 1, top_k)
    
    right_val = torch.where(template_view == 4, thresh_view.float(), right_val)
    
    # Compound ops for Right Side (only if not Threshold)
    compound_mask_r = (r_op > 0).view(1, 1, top_k) & (template_view != 4)
    if compound_mask_r.any():
        r_v2 = get_atom_values(r_f2_idx, r_lb2_long)
        r_op_view = r_op.view(1, 1, top_k)
        
        right_val = torch.where(r_op_view == 1, right_val + r_v2, right_val)
        right_val = torch.where(r_op_view == 2, right_val - r_v2, right_val)
        right_val = torch.where(r_op_view == 3, right_val * r_v2, right_val)
        right_val = torch.where(r_op_view == 4, right_val / (r_v2 + working_eps), right_val)

    # Stability Clip
    right_val = torch.clamp(right_val, -1e4, 1e4)

    # Comparison (Vectorized)
    # -----------------------
    diff = left_val - right_val
    steepness = 10.0
    op_view = cmp_op.view(1, 1, top_k)
    
    # 0: GT, 1: LT, 2: GTE, 3: LTE, 4: EQ
    # GT/GTE: sigmoid(steep * diff)
    # LT/LTE: sigmoid(-steep * diff)
    # EQ: exp(-steep * abs(diff))
    
    # Pre-calculate common terms
    sig_pos = torch.sigmoid(steepness * diff)
    sig_neg = torch.sigmoid(-steepness * diff)
    eq_val = torch.exp(-steepness * diff.abs())
    
    result = torch.zeros_like(diff)
    result = torch.where((op_view == 0) | (op_view == 2), sig_pos, result)
    result = torch.where((op_view == 1) | (op_view == 3), sig_neg, result)
    result = torch.where(op_view == 4, eq_val, result)
    
    # Mask out early time steps where lb was invalid?
    # max_lb logic from loop: t < max_lb is 0.
    # We can apply a mask.
    max_lb_per_k = torch.max(torch.stack([l_lb1, r_lb1, l_lb2, r_lb2]), dim=0)[0] # (K,)
    max_lb_view = max_lb_per_k.view(1, 1, top_k)
    
    time_mask = t_seq >= max_lb_view
    result = result * time_mask.float()

    # Importance weighting
    result = result * active_importance.view(1, 1, top_k).to(torch.float32)

    # Cast back
    result = result.to(data.dtype)

    if squeeze_out:
        result = result.squeeze(0)
    return result


# =============================================================================
# NUMBA-ACCELERATED EVALUATOR (if available)
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _eval_simple_atoms_numba(
        data: np.ndarray,           # (N, n_fields)
        atom_params: np.ndarray,    # (K, 2) - field_idx, lookback
    ) -> np.ndarray:
        """Evaluate K simple atoms. Returns (N, K)."""
        N, _ = data.shape
        K = atom_params.shape[0]
        result = np.zeros((N, K), dtype=np.float32)

        for k in range(K):
            field_idx = int(atom_params[k, 0])
            lookback = int(atom_params[k, 1])

            for t in range(lookback, N):
                result[t, k] = data[t - lookback, field_idx]

        return result

    @jit(nopython=True, parallel=True, cache=True)
    def _eval_inequalities_numba(
        data: np.ndarray,           # (N, n_fields)
        ineq_params: np.ndarray,    # (K, 13) - encoded inequalities with template info
    ) -> np.ndarray:
        """Evaluate K single inequalities. Returns (N, K)."""
        N, n_fields = data.shape
        K = ineq_params.shape[0]
        result = np.zeros((N, K), dtype=np.float32)
        eps = 1e-8

        for k in prange(K):
            # Decode inequality
            l_f1 = int(ineq_params[k, 0])
            l_n1 = int(ineq_params[k, 1])
            l_op = int(ineq_params[k, 2])
            l_f2 = int(ineq_params[k, 3])
            l_n2 = int(ineq_params[k, 4])
            cmp_op = int(ineq_params[k, 5])
            r_f1 = int(ineq_params[k, 6])
            r_n1 = int(ineq_params[k, 7])
            r_op = int(ineq_params[k, 8])
            r_f2 = int(ineq_params[k, 9])
            r_n2 = int(ineq_params[k, 10])

            max_lb = max(l_n1, l_n2, r_n1, r_n2)

            for t in range(max_lb, N):
                # Evaluate left atom
                left_val = data[t - l_n1, l_f1] if l_n1 < N else 0.0
                if l_op == 1:  # ADD
                    left_val += data[t - l_n2, l_f2] if l_n2 < N else 0.0
                elif l_op == 2:  # SUB
                    left_val -= data[t - l_n2, l_f2] if l_n2 < N else 0.0
                elif l_op == 3:  # MUL
                    left_val *= data[t - l_n2, l_f2] if l_n2 < N else 1.0
                elif l_op == 4:  # DIV
                    denom = data[t - l_n2, l_f2] if l_n2 < N else 1.0
                    left_val /= (denom + eps)

                # Evaluate right atom
                # THRESHOLD logic: template_idx is at p[11], thresh_val at p[12]
                if int(ineq_params[k, 11]) == 4: # THRESHOLD
                    right_val = ineq_params[k, 12]
                else:
                    right_val = data[t - r_n1, r_f1] if r_n1 < N else 0.0
                    if r_op == 1:
                        right_val += data[t - r_n2, r_f2] if r_n2 < N else 0.0
                    elif r_op == 2:
                        right_val -= data[t - r_n2, r_f2] if r_n2 < N else 0.0
                    elif r_op == 3:
                        right_val *= data[t - r_n2, r_f2] if r_n2 < N else 1.0
                    elif r_op == 4:
                        denom = data[t - r_n2, r_f2] if r_n2 < N else 1.0
                        right_val /= (denom + eps)

                # Compare
                if cmp_op == 0:    # GT
                    result[t, k] = 1.0 if left_val > right_val else 0.0
                elif cmp_op == 1:  # LT
                    result[t, k] = 1.0 if left_val < right_val else 0.0
                elif cmp_op == 2:  # GTE
                    result[t, k] = 1.0 if left_val >= right_val else 0.0
                elif cmp_op == 3:  # LTE
                    result[t, k] = 1.0 if left_val <= right_val else 0.0
                elif cmp_op == 4:  # EQ
                    result[t, k] = 1.0 if abs(left_val - right_val) < 1e-6 else 0.0

        return result


# =============================================================================
# DIFFERENTIABLE PREDICATE SELECTOR
# =============================================================================

class PredicateSelector(nn.Module):
    """
    Learns which predicates (from millions) are useful.

    Architecture:
    - Maintains N_SLOTS learnable predicate embeddings
    - Each embedding decodes to inequality parameters
    - Importance weights determine which predicates are active
    - Sparsity regularization keeps selection focused

    Parameters:
        n_slots: Number of predicate slots to learn (default: 4096)
        max_active: Maximum active predicates at inference
        d_embed: Embedding dimension
        n_fields: Number of input fields
        max_lookback: Maximum lookback period
        temperature: Gumbel-softmax temperature
    """

    def __init__(
        self,
        n_slots: int = 4096,
        max_active: int = 512,
        d_embed: int = 128,
        n_fields: int = N_FIELDS,
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
        
        # RECURSIVE LOGIC:
        # The input space for a predicate includes raw fields AND output of all predicate slots.
        # This allows predicates to reference each other.
        self.total_input_kwargs = n_fields + n_slots

        # Learnable predicate embeddings
        self.predicate_embeddings = nn.Parameter(torch.randn(n_slots, d_embed) * 0.01)

        # Shared decoder backbone
        self.decoder = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.GELU(),
            nn.LayerNorm(d_embed * 2),
            nn.Linear(d_embed * 2, d_embed * 2),
            nn.GELU(),
        )

        # Parameter heads for each component of the predicate
        # We learn parameters for a single inequality (can extend to chains)
        # Left atom: field1, lookback1, arith_op, field2, lookback2
        # FIELDS now map to [0..total_input_kwargs-1]
        self.left_field1 = nn.Linear(d_embed * 2, self.total_input_kwargs)
        self.left_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.left_arith = nn.Linear(d_embed * 2, 5)  # NONE, ADD, SUB, MUL, DIV
        self.left_field2 = nn.Linear(d_embed * 2, self.total_input_kwargs)
        self.left_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        # Compare operator
        self.compare_op = nn.Linear(d_embed * 2, 5)  # GT, LT, GTE, LTE, EQ

        # Right atom
        self.right_field1 = nn.Linear(d_embed * 2, self.total_input_kwargs)
        self.right_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.right_arith = nn.Linear(d_embed * 2, 5)
        self.right_field2 = nn.Linear(d_embed * 2, self.total_input_kwargs)
        self.right_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        # Importance scores (learned sparsity)
        # Initialize to -5.0 (sigmoid(-5) ~= 0.006) to start sparse.
        # This prevents the model from being overwhelmed by noise and collapsing to "Always False" (EQ) 
        # just to silence the random predicates.
        self.importance_logits = nn.Parameter(torch.ones(n_slots) * -5.0)

        # Template Selection Head
        self.template_head = nn.Linear(d_embed * 2, 5)  # 5 TemplateTypes
        
        # Threshold Head (for THRESHOLD template)
        self.threshold_head = nn.Linear(d_embed * 2, 1)

        # Cache for decoded predicates
        self._cached_predicates = None
        self._cache_valid = False

    def _gumbel_sample(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from categorical using Gumbel-softmax."""
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            indices = (probs * torch.arange(logits.size(-1), device=logits.device)).sum(-1)
        else:
            indices = logits.argmax(-1).float()
            probs = F.one_hot(indices.long(), logits.size(-1)).float()
        return probs, indices

    def forward(self, return_params: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute importance weights and optionally decode predicate parameters.

        Returns:
            importance: (n_slots,) importance weights
            params: (n_slots, 11) encoded parameters if return_params=True
        """
        # Compute importance weights
        importance = torch.sigmoid(self.importance_logits)

        if not return_params:
            return importance, None

        # Decode all predicate embeddings
        hidden = self.decoder(self.predicate_embeddings)  # (n_slots, d*2)

        # Left atom
        _, l_f1 = self._gumbel_sample(self.left_field1(hidden))
        _, l_n1 = self._gumbel_sample(self.left_lookback1(hidden))
        _, l_op = self._gumbel_sample(self.left_arith(hidden))
        _, l_f2 = self._gumbel_sample(self.left_field2(hidden))
        _, l_n2 = self._gumbel_sample(self.left_lookback2(hidden))

        # Compare
        _, cmp = self._gumbel_sample(self.compare_op(hidden))

        # Right atom
        _, r_f1 = self._gumbel_sample(self.right_field1(hidden))
        _, r_n1 = self._gumbel_sample(self.right_lookback1(hidden))
        _, r_op = self._gumbel_sample(self.right_arith(hidden))
        _, r_f2 = self._gumbel_sample(self.right_field2(hidden))
        _, r_n2 = self._gumbel_sample(self.right_lookback2(hidden))

        # Template Selection
        _, templates = self._gumbel_sample(self.template_head(hidden))

        # --- TEMPLATE CONSTRAINTS (Differentiable Masking) ---
        # Note: We use the indices directly as this happens after hard-sampling
        
        # MOMENTUM: Fix r_f1=l_f1, l_n1=0, r_n1>1, ops=NONE
        is_mom = (templates == TemplateType.MOMENTUM)
        r_f1 = torch.where(is_mom, l_f1, r_f1)
        l_n1 = torch.where(is_mom, torch.zeros_like(l_n1), l_n1)
        l_op = torch.where(is_mom, torch.zeros_like(l_op), l_op)
        r_op = torch.where(is_mom, torch.zeros_like(r_op), r_op)

        # RELATIVE: Fix l_n1=0, r_n1=0, ops=NONE
        is_rel = (templates == TemplateType.RELATIVE)
        l_n1 = torch.where(is_rel, torch.zeros_like(l_n1), l_n1)
        r_n1 = torch.where(is_rel, torch.zeros_like(r_n1), r_n1)
        l_op = torch.where(is_rel, torch.zeros_like(l_op), l_op)
        r_op = torch.where(is_rel, torch.zeros_like(r_op), r_op)

        # CROSSOVER: dual comparison {F1[0]-F1[n]} cmp {F2[0]-F2[m]}
        # Fixes: atoms must be compound, ops=SUB, l_n1=0, r_n1=0, l_f1!=r_f1
        is_cross = (templates == TemplateType.CROSSOVER)
        l_op = torch.where(is_cross, torch.full_like(l_op, ArithOp.SUB), l_op)
        r_op = torch.where(is_cross, torch.full_like(r_op, ArithOp.SUB), r_op)
        l_f2 = torch.where(is_cross, l_f1, l_f2) # {F1[0] - F1[n]}
        r_f2 = torch.where(is_cross, r_f1, r_f2) # {F2[0] - F2[m]}
        l_n1 = torch.where(is_cross, torch.zeros_like(l_n1), l_n1)
        r_n1 = torch.where(is_cross, torch.zeros_like(r_n1), r_n1)

        # THRESHOLD: compare against learned constant
        is_thresh = (templates == TemplateType.THRESHOLD)
        thresh_val = self.threshold_head(hidden).squeeze(-1)
        # Note: In THRESHOLD mode, the right atom is effectively replaced by thresh_val in the evaluator
        
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
        """
        Extract predicates with importance above threshold.

        Returns:
            predicates: List of Predicate objects
            importance: Array of importance scores
            names: Human-readable predicate strings
        """
        self.eval()
        with torch.no_grad():
            importance, params = self.forward(return_params=True)

            # Determine which predicates to return
            if max_return is None:
                max_return = self.max_active # Use class default if not specified

            # Get indices above threshold
            mask = importance > threshold
            active_idx = torch.where(mask)[0]

            if len(active_idx) == 0:
                # If no predicates meet threshold, take the top `max_return` overall
                top_k = min(self.n_slots, max_return)
                _, top_idx = torch.topk(importance, top_k)
                active_idx = top_idx
            else:
                # Sort by importance if some predicates meet threshold
                active_importance_filtered = importance[active_idx]
                sort_idx = torch.argsort(active_importance_filtered, descending=True)
                active_idx = active_idx[sort_idx]

                if len(active_idx) > max_return:
                    active_idx = active_idx[:max_return]

            if len(active_idx) == 0:
                return [], np.array([]), []

            active_params = params[active_idx].cpu().numpy()
            active_importance = importance[active_idx].cpu().numpy()

            # Convert to Predicate objects
            predicates = []
            names = []

            for i, p in enumerate(active_params):
                # Decode inequality
                left = Atom(int(p[0]), int(p[1]), ArithOp(int(p[2])), int(p[3]), int(p[4]))
                right = Atom(int(p[6]), int(p[7]), ArithOp(int(p[8])), int(p[9]), int(p[10]))
                ineq = Inequality(left, CompareOp(int(p[5])), right)
                pred = Predicate([ineq], [])
                
                # Get template name if exists
                t_idx = int(p[11]) if len(p) > 11 else 0
                t_name = TemplateType(t_idx).name if t_idx < 5 else "UNKNOWN"
                
                # Format rule
                if t_idx == TemplateType.THRESHOLD:
                    t_val = p[12]
                    names.append(f"[{t_name}] {left} {COMPARE_SYMBOLS[int(p[5])]} {t_val:.3f} (imp={active_importance[i]:.3f})")
                else:
                    names.append(f"[{t_name}] {pred} (imp={active_importance[i]:.3f})")

                predicates.append(pred)

            return predicates, active_importance, names

    def sparsity_loss(self) -> torch.Tensor:
        """L1 regularization on importance weights."""
        return torch.sigmoid(self.importance_logits).sum()

    def diversity_loss(self) -> torch.Tensor:
        """
        Penalize predicates for using the same features.
        Encourages the engine to explore the full feature space.
        """
        hidden = self.decoder(self.predicate_embeddings)  # (n_slots, d*2)
        
        # Get feature selection logits for left and right atoms
        l1_logits = self.left_field1(hidden)  # (n_slots, total_input)
        l2_logits = self.left_field2(hidden)
        r1_logits = self.right_field1(hidden)
        r2_logits = self.right_field2(hidden)
        
        # Softmax to get probabilities
        l1_probs = torch.softmax(l1_logits, dim=-1)
        l2_probs = torch.softmax(l2_logits, dim=-1)
        r1_probs = torch.softmax(r1_logits, dim=-1)
        r2_probs = torch.softmax(r2_logits, dim=-1)
        
        # Weighted average probabilities by predicate importance
        importance = torch.sigmoid(self.importance_logits).view(-1, 1)
        # Normalize importance to sum to 1 for distribution analysis
        importance_norm = importance / (importance.sum() + 1e-8)
        
        # Average feature distribution across all slots (weighted by importance)
        avg_probs = (
            (l1_probs * importance_norm).sum(0) +
            (l2_probs * importance_norm).sum(0) +
            (r1_probs * importance_norm).sum(0) +
            (r2_probs * importance_norm).sum(0)
        ) / 4.0
        
        # Penalty: Sum of squares (minimizing this encourages uniform distribution)
        # or Entropy (maximizing this encourages uniform distribution)
        # Let's use negative entropy to minimize.
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
        
        return -entropy # Minimizing -entropy = Maximizing entropy = More diversity

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
# PREDICATE COMBINER (Learns chains of inequalities)
# =============================================================================

class PredicateCombiner(nn.Module):
    """
    Neural network that learns to combine discovered predicates.

    Takes evaluated boolean predicates and learns:
    - Which predicates to chain together (up to max_depth)
    - Whether to use AND or OR logic
    - Temporal patterns in predicate activations

    This runs AFTER PredicateSelector has identified useful single predicates.
    """

    def __init__(
        self,
        n_predicates: int,
        d_model: int = 128,
        n_heads: int = 4,
        max_chain_depth: int = 8,  # Increased for nested logic
        n_output_heads: int = 10,
        n_chains: int = 256
    ):
        super().__init__()

        self.n_predicates = n_predicates
        self.max_chain_depth = max_chain_depth
        self.n_chains = n_chains

        # Predicate embedding (learns semantic meaning of each predicate)
        self.predicate_embed = nn.Embedding(n_predicates, d_model)

        # Chain attention: which predicates to combine
        # Shape: (n_chains, max_depth, n_predicates) - attention over predicates per position
        self.n_chains = 256
        self.chain_attention = nn.Parameter(torch.randn(self.n_chains, max_chain_depth, n_predicates) * 0.01)

        # Logic gates: AND (0) vs OR (1) between positions
        self.logic_gates = nn.Parameter(torch.zeros(self.n_chains, max_chain_depth - 1))

        # Temporal processing
        self.temporal_proj = nn.Linear(n_predicates, d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(d_model + self.n_chains, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_output_heads)
        )

    def forward(self, predicate_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicate_values: (batch, seq_len, n_predicates) boolean values

        Returns:
            (batch, n_output_heads) predictions
        """
        batch, seq_len, n_pred = predicate_values.shape
        device = predicate_values.device

        # === Chain Evaluation ===
        # Soft attention over predicates per chain position
        chain_attn = F.softmax(self.chain_attention, dim=-1)  # (n_chains, max_depth, n_pred)

        # Weighted predicate selection per chain position
        # predicate_values: (batch, seq, n_pred)
        # chain_attn: (n_chains, max_depth, n_pred)
        # Result: (batch, seq, n_chains, max_depth)
        
        # Ensure dtype match (e.g. if predicate_values is float64/double but weights are float32)
        if predicate_values.dtype != chain_attn.dtype:
            predicate_values = predicate_values.to(chain_attn.dtype)
            
        chain_values = torch.einsum('bsp,cdp->bscd', predicate_values, chain_attn)

        # Apply logic gates (soft AND/OR)
        # AND: product, OR: max
        logic_weights = torch.sigmoid(self.logic_gates)  # (n_chains, max_depth-1)

        # Start with first position
        combined = chain_values[:, :, :, 0]  # (batch, seq, n_chains)

        for d in range(1, self.max_chain_depth):
            next_val = chain_values[:, :, :, d]
            # Soft blend between AND (product) and OR (max)
            and_result = combined * next_val
            or_result = torch.maximum(combined, next_val)

            w = logic_weights[:, d-1].unsqueeze(0).unsqueeze(0)  # (1, 1, n_chains)
            combined = w * or_result + (1 - w) * and_result

        # preservation of sequence dimension for backbone augmentation
        # combined: (batch, seq, n_chains)
        temporal_features = self.temporal_proj(predicate_values)  # (batch, seq, d_model)
        temporal_out, _ = self.temporal_attn(temporal_features, temporal_features, temporal_features) # (batch, seq, d_model)

        # Concatenate on feature dim (preserves B and S)
        # Result: (batch, seq, d_model + n_chains)
        combined_features = torch.cat([temporal_out, combined], dim=-1)
        
        # Output proj is linear, can handle (B, S, D) -> (B, S, Out)
        return self.output_proj(combined_features)

    def get_logic_sets(self, predicate_names: List[str], threshold: float = 0.5) -> List[str]:
        """Extract full human-readable logical chains."""
        self.eval()
        with torch.no_grad():
            chain_attn = F.softmax(self.chain_attention, dim=-1).cpu().numpy()
            logic_weights = torch.sigmoid(self.logic_gates).cpu().numpy()
            
            logic_sets = []
            for c in range(self.n_chains):
                active_elements = []
                for d in range(self.max_chain_depth):
                    best_pred_idx = np.argmax(chain_attn[c, d])
                    if chain_attn[c, d, best_pred_idx] > 0.3:
                        name = predicate_names[best_pred_idx] if best_pred_idx < len(predicate_names) else f"P{best_pred_idx}"
                        active_elements.append((d, name))
                
                if len(active_elements) < 2:
                    continue
                
                # Build string with operators only between active elements
                expr = f"({active_elements[0][1]})"
                for i in range(1, len(active_elements)):
                    depth_idx, name = active_elements[i]
                    # Use the logic gate corresponding to the PREVIOUS position
                    prev_depth = active_elements[i-1][0]
                    op = "OR" if logic_weights[c, prev_depth] > 0.5 else "AND"
                    expr += f" {op} ({name})"
                
                logic_sets.append(expr)
            
            return logic_sets


# =============================================================================
# INTEGRATED MODEL: CDE + Predicates
# =============================================================================

class PredicateAugmentedModel(nn.Module):
    """
    Neural model augmented with learned predicates.

    Architecture:
    1. PredicateSelector discovers useful inequality patterns
    2. Discovered predicates are evaluated on OHLCV + indicator data
    3. Predicate features concatenated with raw features
    4. Backbone processes combined features
    5. PredicateCombiner learns higher-order combinations

    This is designed to be integrated with CondorBrain.
    """

    def __init__(
        self,
        raw_input_dim: int,
        n_predicate_slots: int = 2048,
        max_active_predicates: int = 512,  # Paper: 800+
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_output_heads: int = 10,
        field_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.raw_input_dim = raw_input_dim
        self.max_active = max_active_predicates
        self.field_names = field_names or ALL_FIELDS
        self.n_fields = len(self.field_names)

        # Predicate discovery
        self.predicate_selector = PredicateSelector(
            n_slots=n_predicate_slots,
            max_active=max_active_predicates,
            n_fields=self.n_fields,
        )

        # Predicate combiner
        self.predicate_combiner = PredicateCombiner(
            n_predicates=max_active_predicates,
            d_model=d_model // 2,
            n_output_heads=d_model // 2,  # Intermediate features
        )

        # Raw feature pathway
        combined_dim = raw_input_dim + max_active_predicates
        self.input_proj = nn.Linear(combined_dim, d_model)

        # Transformer backbone
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True, dropout=0.1),
            num_layers=n_layers
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Output heads
        self.output_heads = nn.Linear(d_model, n_output_heads)

        # Field index mapping
        self._build_field_map()

    def _build_field_map(self):
        """Build mapping from field names to indices in input data."""
        self.field_map = {}
        for i, name in enumerate(self.field_names):
            if name in FIELD_TO_IDX:
                self.field_map[i] = FIELD_TO_IDX[name]

    def evaluate_predicates_on_data(
        self,
        data: torch.Tensor,  # (batch, seq, n_fields)
        predicates: List[Predicate]
    ) -> torch.Tensor:
        """Evaluate predicates on input data."""
        batch, seq_len, _ = data.shape
        device = data.device
        n_pred = len(predicates)

        if n_pred == 0:
            return torch.zeros(batch, seq_len, self.max_active, device=device)

        # Convert to numpy for evaluation
        data_np = data.detach().cpu().numpy()

        # Evaluate each batch item
        results = []
        for b in range(batch):
            pred_vals = evaluate_predicates_batch(data_np[b], predicates)
            results.append(pred_vals)

        result = torch.tensor(np.stack(results), device=device, dtype=torch.float32)

        # Pad to max_active
        if n_pred < self.max_active:
            padding = torch.zeros(batch, seq_len, self.max_active - n_pred, device=device)
            result = torch.cat([result, padding], dim=-1)

        return result

    def forward(
        self,
        raw_features: torch.Tensor,
        field_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            raw_features: (batch, seq_len, raw_input_dim) normalized features
            field_data: (batch, seq_len, n_fields) raw field values for predicate evaluation
                       If None, extracts from raw_features

        Returns:
            (batch, n_output_heads) predictions
        """
        batch, seq_len, _ = raw_features.shape
        device = raw_features.device

        # Get active predicates
        predicates, importance, _ = self.predicate_selector.get_active_predicates(
            threshold=0.1, max_return=self.max_active
        )

        # Evaluate predicates
        if field_data is not None:
            pred_features = self.evaluate_predicates_on_data(field_data, predicates)
        else:
            # Use first n_fields columns of raw_features
            field_data = raw_features[:, :, :self.n_fields]
            pred_features = self.evaluate_predicates_on_data(field_data, predicates)

        # Combine raw + predicate features
        combined = torch.cat([raw_features, pred_features], dim=-1)

        # Process through backbone
        x = self.input_proj(combined)
        x = self.backbone(x)
        backbone_out = x[:, -1, :]  # (batch, d_model)

        # Process predicates through combiner
        combiner_out = self.predicate_combiner(pred_features)  # (batch, d_model//2)

        # Fuse
        fused = self.fusion(torch.cat([backbone_out, combiner_out], dim=-1))

        return self.output_heads(fused)

    def get_discovered_predicates(self) -> List[str]:
        """Return human-readable list of discovered predicates."""
        _, _, names = self.predicate_selector.get_active_predicates(threshold=0.1)
        return names

    def total_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sparsity_weight: float = 0.001,
        entropy_weight: float = 0.0001
    ) -> torch.Tensor:
        """Combined loss with regularization."""
        pred_loss = F.mse_loss(pred, target)
        sparse_loss = self.predicate_selector.sparsity_loss()
        entropy_loss = self.predicate_selector.entropy_loss()

        return pred_loss + sparsity_weight * sparse_loss + entropy_weight * entropy_loss


# =============================================================================
# UTILITY: Export discovered predicates
# =============================================================================

def export_predicates_to_json(
    predicates: List[Predicate],
    importance: np.ndarray,
    output_path: str
):
    """Export discovered predicates for external use."""
    export = {
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

    print(f"[export_predicates_to_json] Saved {len(predicates)} predicates to {output_path}")


def load_predicates_from_json(path: str) -> Tuple[List[Predicate], np.ndarray]:
    """Load predicates from JSON export."""
    with open(path, 'r') as f:
        data = json.load(f)

    predicates = []
    importance = []

    for p in data['predicates']:
        enc = np.array(p['encoding'], dtype=np.float32)
        # Decode from encoding (depth-1 only for now)
        left = Atom(int(enc[0]), int(enc[1]), ArithOp(int(enc[2])), int(enc[3]), int(enc[4]))
        right = Atom(int(enc[6]), int(enc[7]), ArithOp(int(enc[8])), int(enc[9]), int(enc[10]))
        ineq = Inequality(left, CompareOp(int(enc[5])), right)
        predicates.append(Predicate([ineq], []))
        importance.append(p['importance'])

    return predicates, np.array(importance)


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PREDICATE DISCOVERY ENGINE - Demo")
    print("=" * 60)

    # Show grammar
    grammar = PredicateGrammar(
        max_lookback=128,
        lookback_set=PredicateGrammar.FIBONACCI_LOOKBACKS,
        max_depth=4,
        allow_compound=True,
    )

    print("\nSample random predicates:")
    samples = grammar.sample_random(10)
    for p in samples:
        print(f"  {p}")

    print("\n" + "=" * 60)
    print("Predicate Selector Architecture:")

    selector = PredicateSelector(
        n_slots=2048,
        max_active=256,
        n_fields=len(ALL_FIELDS),
    )

    total_params = sum(p.numel() for p in selector.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Predicate slots: 2048")
    print(f"  Max active: 256")
    print(f"  Fields: {len(ALL_FIELDS)}")

    print("\n" + "=" * 60)
    print("Full Model Architecture:")

    model = PredicateAugmentedModel(
        raw_input_dim=54,
        n_predicate_slots=2048,
        max_active_predicates=512,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test forward pass
    print("\nTest forward pass:")
    batch = 4
    seq_len = 128
    raw_features = torch.randn(batch, seq_len, 54)
    field_data = torch.randn(batch, seq_len, len(ALL_FIELDS))

    with torch.no_grad():
        output = model(raw_features, field_data)

    print(f"  Input: ({batch}, {seq_len}, 54)")
    print(f"  Output: {output.shape}")

    print("\n" + "=" * 60)
    print("Usage in training:")
    print("""
    from intelligence.predicate_discovery import PredicateAugmentedModel

    model = PredicateAugmentedModel(
        raw_input_dim=54,
        n_predicate_slots=2048,
        max_active_predicates=256,
    )

    # Training loop
    for batch in dataloader:
        raw_features = batch['features']
        field_data = batch['ohlc']  # Raw OHLCV + indicators
        targets = batch['targets']

        outputs = model(raw_features, field_data)
        loss = model.total_loss(outputs, targets, sparsity_weight=0.001)

        loss.backward()
        optimizer.step()

    # Extract discovered predicates
    predicates = model.get_discovered_predicates()
    for p in predicates[:20]:
        print(p)
    """)
