"""
Predicate Discovery - Colab Test Script

Upload this script and your data/models to Colab to test the predicate discovery engine.

Required files:
- models/epoch_1_012926.pth (or any checkpoint)
- data/processed/mamba_institutional_2024_1m_v22.csv

Usage in Colab:
  1. Upload files to /content/
  2. Run this script

Or run locally:
  python notebooks/test_predicate_discovery_colab.py --data data/processed/mamba_institutional_2024_1m_v22.csv --samples 5000
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
import time
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# V2.2 Feature Schema (must match your data)
FEATURE_COLS_V22 = [
    # Market Primitives (5)
    "open", "high", "low", "close", "volume",
    # Options / Chain State (9)
    "delta", "gamma", "vega", "theta", "iv", "ivr", "spread_ratio", "te", "strike",
    # Strategy / Targets / Risk (2)
    "target_spot", "max_dd_60m",
    # Dynamic / Regime-Aware (16)
    "log_return", "vol_ewma", "ret_z", "atr_pct", "kappa_proxy", "vol_energy",
    "rsi_dyn", "adx_adaptive", "psar_adaptive",
    "bb_mu_dyn", "bb_sigma_dyn", "bb_lower_dyn", "bb_upper_dyn",
    "stoch_k_dyn", "consolidation_score", "breakout_score",
    # V2.2 Primitive Outputs (22)
    "bb_percentile", "bw_expansion_rate", "cmf", "pressure_up", "pressure_down",
    "friction_ratio", "exec_allow", "gap_risk_score", "risk_override", "iv_confidence",
    "mtf_consensus", "macd_norm", "macd_signal_norm", "macd_histogram",
    "plus_di", "minus_di", "psar_trend", "psar_reversion_mu",
    "beta1_norm_stub", "chaos_membership", "position_size_mult", "fuzzy_reversion_11",
]

ALL_FIELDS = FEATURE_COLS_V22.copy()
FIELD_TO_IDX = {f: i for i, f in enumerate(ALL_FIELDS)}
IDX_TO_FIELD = {i: f for i, f in enumerate(ALL_FIELDS)}
N_FIELDS = len(ALL_FIELDS)


# =============================================================================
# GRAMMAR DEFINITION (Self-contained copy)
# =============================================================================

class ArithOp(IntEnum):
    NONE = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4


class CompareOp(IntEnum):
    GT = 0
    LT = 1
    GTE = 2
    LTE = 3
    EQ = 4


class LogicOp(IntEnum):
    AND = 0
    OR = 1


COMPARE_SYMBOLS = ['>', '<', '>=', '<=', '==']
ARITH_SYMBOLS = ['', '+', '-', '*', '/']
LOGIC_SYMBOLS = ['AND', 'OR']


@dataclass
class Atom:
    field1: int
    lookback1: int
    arith_op: ArithOp
    field2: int = 0
    lookback2: int = 0

    def is_simple(self) -> bool:
        return self.arith_op == ArithOp.NONE

    def __str__(self) -> str:
        f1 = IDX_TO_FIELD.get(self.field1, f"F{self.field1}")
        if self.is_simple():
            return f"{f1}[{self.lookback1}]"
        else:
            f2 = IDX_TO_FIELD.get(self.field2, f"F{self.field2}")
            op = ARITH_SYMBOLS[self.arith_op]
            return f"{{{f1}[{self.lookback1}]{op}{f2}[{self.lookback2}]}}"

    def max_lookback(self) -> int:
        return max(self.lookback1, self.lookback2) if not self.is_simple() else self.lookback1


@dataclass
class Inequality:
    left: Atom
    compare: CompareOp
    right: Atom

    def __str__(self) -> str:
        return f"{self.left} {COMPARE_SYMBOLS[self.compare]} {self.right}"

    def max_lookback(self) -> int:
        return max(self.left.max_lookback(), self.right.max_lookback())


@dataclass
class Predicate:
    inequalities: List[Inequality]
    logic_ops: List[LogicOp]

    def __post_init__(self):
        if len(self.logic_ops) != max(0, len(self.inequalities) - 1):
            self.logic_ops = [LogicOp.AND] * (len(self.inequalities) - 1)

    def depth(self) -> int:
        return len(self.inequalities)

    def __str__(self) -> str:
        if self.depth() == 1:
            return str(self.inequalities[0])
        parts = [str(self.inequalities[0])]
        for ineq, op in zip(self.inequalities[1:], self.logic_ops):
            parts.append(f" {LOGIC_SYMBOLS[op]} {ineq}")
        return ''.join(parts)


# =============================================================================
# PREDICATE EVALUATOR
# =============================================================================

def evaluate_atom(data: np.ndarray, atom: Atom, eps: float = 1e-8) -> np.ndarray:
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


def evaluate_inequality(data: np.ndarray, ineq: Inequality, eps: float = 1e-6) -> np.ndarray:
    """Evaluate an inequality over all timesteps. Returns (N,) boolean array."""
    left_val = evaluate_atom(data, ineq.left)
    right_val = evaluate_atom(data, ineq.right)

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


def evaluate_predicate(data: np.ndarray, pred: Predicate) -> np.ndarray:
    """Evaluate a compound predicate."""
    if pred.depth() == 0:
        return np.zeros(data.shape[0], dtype=np.float32)

    result = evaluate_inequality(data, pred.inequalities[0])

    for ineq, logic_op in zip(pred.inequalities[1:], pred.logic_ops):
        ineq_result = evaluate_inequality(data, ineq)
        if logic_op == LogicOp.AND:
            result = result * ineq_result
        else:
            result = np.maximum(result, ineq_result)

    return result


def evaluate_predicates_batch(data: np.ndarray, predicates: List[Predicate]) -> np.ndarray:
    """Evaluate multiple predicates. Returns (N, K)."""
    N = data.shape[0]
    K = len(predicates)
    result = np.zeros((N, K), dtype=np.float32)
    for k, pred in enumerate(predicates):
        result[:, k] = evaluate_predicate(data, pred)
    return result


# =============================================================================
# PREDICATE SELECTOR (Differentiable)
# =============================================================================

class PredicateSelector(nn.Module):
    def __init__(
        self,
        n_slots: int = 2048,
        max_active: int = 256,
        d_embed: int = 128,
        n_fields: int = N_FIELDS,
        max_lookback: int = 128,
        temperature: float = 1.0
    ):
        super().__init__()

        self.n_slots = n_slots
        self.max_active = max_active
        self.n_fields = n_fields
        self.max_lookback = max_lookback
        self.temperature = temperature

        self.predicate_embeddings = nn.Parameter(torch.randn(n_slots, d_embed) * 0.01)

        self.decoder = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.GELU(),
            nn.LayerNorm(d_embed * 2),
            nn.Linear(d_embed * 2, d_embed * 2),
            nn.GELU(),
        )

        self.left_field1 = nn.Linear(d_embed * 2, n_fields)
        self.left_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.left_arith = nn.Linear(d_embed * 2, 5)
        self.left_field2 = nn.Linear(d_embed * 2, n_fields)
        self.left_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.compare_op = nn.Linear(d_embed * 2, 5)
        self.right_field1 = nn.Linear(d_embed * 2, n_fields)
        self.right_lookback1 = nn.Linear(d_embed * 2, max_lookback + 1)
        self.right_arith = nn.Linear(d_embed * 2, 5)
        self.right_field2 = nn.Linear(d_embed * 2, n_fields)
        self.right_lookback2 = nn.Linear(d_embed * 2, max_lookback + 1)

        self.importance_logits = nn.Parameter(torch.zeros(n_slots))

    def forward(self, return_params: bool = False):
        importance = torch.sigmoid(self.importance_logits)
        if not return_params:
            return importance, None

        hidden = self.decoder(self.predicate_embeddings)

        def sample(head):
            logits = head(hidden)
            if self.training:
                probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
                indices = (probs * torch.arange(logits.size(-1), device=logits.device)).sum(-1)
            else:
                indices = logits.argmax(-1).float()
            return indices

        params = torch.stack([
            sample(self.left_field1), sample(self.left_lookback1), sample(self.left_arith),
            sample(self.left_field2), sample(self.left_lookback2),
            sample(self.compare_op),
            sample(self.right_field1), sample(self.right_lookback1), sample(self.right_arith),
            sample(self.right_field2), sample(self.right_lookback2)
        ], dim=-1)

        return importance, params

    def get_active_predicates(self, threshold: float = 0.1, max_return: int = None):
        self.eval()
        with torch.no_grad():
            importance, params = self.forward(return_params=True)
            mask = importance > threshold
            active_idx = torch.where(mask)[0]

            if len(active_idx) == 0:
                return [], np.array([]), []

            active_importance = importance[active_idx]
            sort_idx = torch.argsort(active_importance, descending=True)
            active_idx = active_idx[sort_idx]

            if max_return and len(active_idx) > max_return:
                active_idx = active_idx[:max_return]

            active_params = params[active_idx].cpu().numpy().astype(int)
            active_importance = importance[active_idx].cpu().numpy()

            predicates = []
            names = []

            for i, p in enumerate(active_params):
                left = Atom(p[0], p[1], ArithOp(p[2]), p[3], p[4])
                right = Atom(p[6], p[7], ArithOp(p[8]), p[9], p[10])
                ineq = Inequality(left, CompareOp(p[5]), right)
                pred = Predicate([ineq], [])
                predicates.append(pred)
                names.append(f"{pred} (imp={active_importance[i]:.3f})")

            return predicates, active_importance, names

    def sparsity_loss(self):
        return torch.sigmoid(self.importance_logits).sum()


# =============================================================================
# DEMO: RANDOM PREDICATE SEARCH (No training, just evaluation)
# =============================================================================

def random_predicate_search(
    data: np.ndarray,
    targets: np.ndarray,
    n_candidates: int = 10000,
    top_k: int = 100,
    fibonacci_lookbacks: bool = True
):
    """
    Exhaustive random search for predictive predicates.

    This is a brute-force baseline - samples random predicates and
    evaluates their correlation with targets.
    """
    print(f"\n{'='*60}")
    print(f"Random Predicate Search")
    print(f"{'='*60}")
    print(f"Data shape: {data.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Candidates to test: {n_candidates:,}")

    lookbacks = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128] if fibonacci_lookbacks else list(range(129))

    def sample_random_predicate():
        # Sample 1-4 inequalities
        n_ineq = np.random.choice([1, 1, 1, 2, 2, 3, 4], p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.03, 0.02])

        inequalities = []
        for _ in range(n_ineq):
            # Left atom
            l_field1 = np.random.randint(N_FIELDS)
            l_lb1 = np.random.choice(lookbacks)
            l_arith = np.random.choice([0, 0, 0, 1, 2, 3, 4], p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])
            l_field2 = np.random.randint(N_FIELDS) if l_arith > 0 else 0
            l_lb2 = np.random.choice(lookbacks) if l_arith > 0 else 0
            left = Atom(l_field1, l_lb1, ArithOp(l_arith), l_field2, l_lb2)

            # Compare op
            cmp = CompareOp(np.random.randint(5))

            # Right atom
            r_field1 = np.random.randint(N_FIELDS)
            r_lb1 = np.random.choice(lookbacks)
            r_arith = np.random.choice([0, 0, 0, 1, 2, 3, 4], p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])
            r_field2 = np.random.randint(N_FIELDS) if r_arith > 0 else 0
            r_lb2 = np.random.choice(lookbacks) if r_arith > 0 else 0
            right = Atom(r_field1, r_lb1, ArithOp(r_arith), r_field2, r_lb2)

            inequalities.append(Inequality(left, cmp, right))

        logic_ops = [LogicOp(np.random.randint(2)) for _ in range(n_ineq - 1)]
        return Predicate(inequalities, logic_ops)

    # Search
    print(f"\nSearching...")
    start_time = time.time()

    results = []
    for i in range(n_candidates):
        pred = sample_random_predicate()
        try:
            vals = evaluate_predicate(data, pred)
            # Skip predicates that are always 0 or always 1
            if vals.std() < 0.01:
                continue

            # Compute correlation with each target column
            corrs = []
            for t in range(targets.shape[1]):
                # Skip NaN targets
                valid = ~np.isnan(targets[:, t])
                if valid.sum() < 100:
                    continue
                corr = np.corrcoef(vals[valid], targets[valid, t])[0, 1]
                if not np.isnan(corr):
                    corrs.append(abs(corr))

            if corrs:
                avg_corr = np.mean(corrs)
                max_corr = np.max(corrs)
                results.append({
                    'predicate': pred,
                    'str': str(pred),
                    'avg_corr': avg_corr,
                    'max_corr': max_corr,
                    'activation_rate': vals.mean(),
                })

        except Exception as e:
            continue

        if (i + 1) % 1000 == 0:
            print(f"  Tested {i+1:,} candidates, found {len(results):,} valid predicates")

    elapsed = time.time() - start_time
    print(f"\nSearch completed in {elapsed:.1f}s")
    print(f"Found {len(results):,} valid predicates")

    if not results:
        print("No predictive predicates found!")
        return []

    # Sort by max correlation
    results.sort(key=lambda x: x['max_corr'], reverse=True)

    print(f"\nTop {min(top_k, len(results))} Predicates by Max Correlation:")
    print("-" * 80)
    for i, r in enumerate(results[:top_k]):
        print(f"{i+1:3d}. max_corr={r['max_corr']:.4f} avg_corr={r['avg_corr']:.4f} "
              f"act={r['activation_rate']:.3f} | {r['str']}")

    return results[:top_k]


# =============================================================================
# MAIN
# =============================================================================

def load_data(data_path: str, samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare data."""
    print(f"\nLoading data from {data_path}")
    df = pd.read_csv(data_path, nrows=samples)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Find available feature columns
    available_cols = [c for c in FEATURE_COLS_V22 if c in df.columns]
    missing_cols = [c for c in FEATURE_COLS_V22 if c not in df.columns]

    print(f"Available features: {len(available_cols)}/{len(FEATURE_COLS_V22)}")
    if missing_cols:
        print(f"Missing: {missing_cols[:5]}...")

    # Extract feature matrix
    data = df[available_cols].fillna(0).values.astype(np.float32)

    # Create targets (use forward returns if available, else use close)
    if 'target_spot' in df.columns:
        # Forward return
        targets = df[['target_spot']].fillna(0).values.astype(np.float32)
    else:
        # Compute forward returns from close
        close = df['close'].values
        targets = np.zeros((len(close), 3), dtype=np.float32)
        for i, horizon in enumerate([5, 15, 60]):
            targets[:-horizon, i] = (close[horizon:] - close[:-horizon]) / close[:-horizon]
        targets[-60:, :] = np.nan  # Mark forward-looking as NaN

    print(f"Data shape: {data.shape}")
    print(f"Targets shape: {targets.shape}")

    return data, targets


def main():
    parser = argparse.ArgumentParser(description="Predicate Discovery Test")
    parser.add_argument('--data', type=str, default='data/processed/mamba_institutional_2024_1m_v22.csv',
                        help='Path to data CSV')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of rows to load')
    parser.add_argument('--candidates', type=int, default=10000,
                        help='Number of random predicates to test')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Number of top predicates to display')
    parser.add_argument('--output', type=str, default='discovered_predicates.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # Load data
    data, targets = load_data(args.data, args.samples)

    # Run random search
    top_predicates = random_predicate_search(
        data, targets,
        n_candidates=args.candidates,
        top_k=args.top_k
    )

    # Save results
    if top_predicates:
        export = {
            'n_predicates': len(top_predicates),
            'data_samples': args.samples,
            'candidates_tested': args.candidates,
            'predicates': [
                {
                    'expression': r['str'],
                    'max_corr': float(r['max_corr']),
                    'avg_corr': float(r['avg_corr']),
                    'activation_rate': float(r['activation_rate']),
                }
                for r in top_predicates
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(export, f, indent=2)
        print(f"\nSaved {len(top_predicates)} predicates to {args.output}")

    print("\n" + "=" * 60)
    print("EXAMPLE COLAB USAGE:")
    print("=" * 60)
    print("""
# In Colab:
!pip install pandas numpy torch

# Upload your data file, then:
!python test_predicate_discovery_colab.py \\
    --data /content/mamba_institutional_2024_1m_v22.csv \\
    --samples 50000 \\
    --candidates 50000 \\
    --top-k 100 \\
    --output /content/discovered_predicates.json

# For a quick test:
!python test_predicate_discovery_colab.py --samples 5000 --candidates 5000
    """)


if __name__ == '__main__':
    main()
