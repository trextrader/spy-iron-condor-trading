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
        """Sample a random atom."""
        if allow_compound and self.allow_compound and np.random.random() < 0.3:
            # Compound atom
            return Atom(
                field1=np.random.randint(self.n_fields),
                lookback1=np.random.choice(self.lookback_set),
                arith_op=ArithOp(np.random.randint(1, 5)),  # Skip NONE
                field2=np.random.randint(self.n_fields),
                lookback2=np.random.choice(self.lookback_set),
            )
        else:
            # Simple atom
            return Atom(
                field1=np.random.randint(self.n_fields),
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
        """Sample a random predicate with 1 to max_depth inequalities."""
        depth = np.random.randint(1, (max_depth or self.max_depth) + 1)

        inequalities = [self.sample_inequality() for _ in range(depth)]
        logic_ops = [LogicOp(np.random.randint(2)) for _ in range(depth - 1)]

        return Predicate(inequalities, logic_ops)

    def sample_random(self, n: int) -> List[Predicate]:
        """Sample n random predicates."""
        return [self.sample_predicate() for _ in range(n)]


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


def evaluate_predicates_batch(
    data: np.ndarray,
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
        ineq_params: np.ndarray,    # (K, 11) - encoded inequalities
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
        self.left_field1 = nn.Linear(d_embed * 2, n_fields)
        self.left_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.left_arith = nn.Linear(d_embed * 2, 5)  # NONE, ADD, SUB, MUL, DIV
        self.left_field2 = nn.Linear(d_embed * 2, n_fields)
        self.left_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        # Compare operator
        self.compare_op = nn.Linear(d_embed * 2, 5)  # GT, LT, GTE, LTE, EQ

        # Right atom
        self.right_field1 = nn.Linear(d_embed * 2, n_fields)
        self.right_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.right_arith = nn.Linear(d_embed * 2, 5)
        self.right_field2 = nn.Linear(d_embed * 2, n_fields)
        self.right_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        # Importance scores (learned sparsity)
        self.importance_logits = nn.Parameter(torch.zeros(n_slots))

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

        params = torch.stack([
            l_f1, l_n1, l_op, l_f2, l_n2,
            cmp,
            r_f1, r_n1, r_op, r_f2, r_n2
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

            # Get indices above threshold
            mask = importance > threshold
            active_idx = torch.where(mask)[0]

            if len(active_idx) == 0:
                return [], np.array([]), []

            # Sort by importance
            active_importance = importance[active_idx]
            sort_idx = torch.argsort(active_importance, descending=True)
            active_idx = active_idx[sort_idx]

            if max_return and len(active_idx) > max_return:
                active_idx = active_idx[:max_return]

            active_params = params[active_idx].cpu().numpy().astype(int)
            active_importance = importance[active_idx].cpu().numpy()

            # Convert to Predicate objects
            predicates = []
            names = []

            for i, p in enumerate(active_params):
                # Decode inequality
                left = Atom(p[0], p[1], ArithOp(p[2]), p[3], p[4])
                right = Atom(p[6], p[7], ArithOp(p[8]), p[9], p[10])
                ineq = Inequality(left, CompareOp(p[5]), right)
                pred = Predicate([ineq], [])

                predicates.append(pred)
                names.append(f"{pred} (imp={active_importance[i]:.3f})")

            return predicates, active_importance, names

    def sparsity_loss(self) -> torch.Tensor:
        """L1 regularization on importance weights."""
        return torch.sigmoid(self.importance_logits).sum()

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
        max_chain_depth: int = 4,
        n_output_heads: int = 10
    ):
        super().__init__()

        self.n_predicates = n_predicates
        self.max_chain_depth = max_chain_depth

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

        # Take last timestep of chain values
        chain_features = combined[:, -1, :]  # (batch, n_chains)

        # === Temporal Processing ===
        temporal_embed = self.temporal_proj(predicate_values)  # (batch, seq, d_model)
        temporal_out, _ = self.temporal_attn(temporal_embed, temporal_embed, temporal_embed)
        temporal_features = temporal_out[:, -1, :]  # (batch, d_model)

        # === Combine and Output ===
        combined_features = torch.cat([temporal_features, chain_features], dim=-1)
        return self.output_proj(combined_features)

    def get_top_chains(self, k: int = 10) -> List[Dict[str, Any]]:
        """Extract most focused chains for interpretability."""
        chain_attn = F.softmax(self.chain_attention, dim=-1).detach().cpu().numpy()
        logic_weights = torch.sigmoid(self.logic_gates).detach().cpu().numpy()

        # Score chains by attention concentration
        entropies = -np.sum(chain_attn * np.log(chain_attn + 1e-8), axis=-1).mean(axis=1)
        top_idx = np.argsort(entropies)[:k]

        chains = []
        for idx in top_idx:
            chain = {
                'chain_id': int(idx),
                'entropy': float(entropies[idx]),
                'positions': [],
            }
            for d in range(self.max_chain_depth):
                top_preds = np.argsort(chain_attn[idx, d])[-3:][::-1]
                chain['positions'].append({
                    'depth': d,
                    'top_predicates': top_preds.tolist(),
                    'attention_weights': chain_attn[idx, d, top_preds].tolist(),
                })
            chain['logic_ops'] = ['OR' if w > 0.5 else 'AND' for w in logic_weights[idx]]
            chains.append(chain)

        return chains


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
        max_active_predicates: int = 256,
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
        max_active_predicates=256,
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
