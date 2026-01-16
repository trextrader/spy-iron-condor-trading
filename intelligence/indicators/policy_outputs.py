"""
Measure-Theoretic Policy Outputs

This module provides discretization and policy extraction utilities for
converting continuous state observations into interpretable action weights.

Mathematical Foundation:
- State space S is discretized into bins via StateBinner
- Q-tables map discrete states to action values
- Softmax converts Q-values to probability distributions
- This enables auditable, interpretable policy decisions

Usage:
1. Define bin edges for each state dimension
2. Load or learn Q-table (dict mapping state keys to action Q-values)
3. For each observation, encode state and compute policy vector
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Iron Condor action space
ACTIONS = ["buy_call", "sell_call", "buy_put", "sell_put", "hold"]


@dataclass
class StateBinner:
    """
    Discretizes continuous state variables into bin indices.
    
    Each state dimension is mapped to an integer bin index based on
    provided edge values. This enables Q-table lookup for policy extraction.
    
    Example:
        binner = StateBinner(
            vol_edges=[0.0, 0.5, 1.0, 2.0],  # 4 edges → 5 bins
            cons_edges=[0.0, 0.3, 0.7],      # 3 edges → 4 bins
            brk_edges=[0.0, 0.5],            # 2 edges → 3 bins
            ivr_edges=[0.0, 30.0, 70.0]      # 3 edges → 4 bins
        )
        state = binner.encode(0.75, 0.4, 0.1, 45.0)  # → (2, 2, 1, 2)
    """
    vol_edges: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    cons_edges: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.7])
    brk_edges: List[float] = field(default_factory=lambda: [0.0, 0.5])
    ivr_edges: List[float] = field(default_factory=lambda: [0.0, 30.0, 70.0])
    
    def encode(
        self, 
        vol_energy: float, 
        consolidation: float, 
        breakout: float, 
        ivr: float
    ) -> Tuple[int, int, int, int]:
        """
        Encode continuous state values into discrete bin indices.
        
        Args:
            vol_energy: Volatility energy (from manifold indicators)
            consolidation: Consolidation score (0-1)
            breakout: Breakout score (0-1)
            ivr: IV Rank (0-100)
            
        Returns:
            Tuple of 4 bin indices
        """
        def _bin(value: float, edges: List[float]) -> int:
            return int(np.searchsorted(edges, value, side="right"))
        
        return (
            _bin(vol_energy, self.vol_edges),
            _bin(consolidation, self.cons_edges),
            _bin(breakout, self.brk_edges),
            _bin(ivr, self.ivr_edges),
        )
    
    def state_key(
        self,
        vol_energy: float,
        consolidation: float,
        breakout: float,
        ivr: float
    ) -> str:
        """Get string key for Q-table lookup."""
        s = self.encode(vol_energy, consolidation, breakout, ivr)
        return ",".join(map(str, s))
    
    def n_states(self) -> int:
        """Total number of discrete states."""
        return (
            (len(self.vol_edges) + 1) *
            (len(self.cons_edges) + 1) *
            (len(self.brk_edges) + 1) *
            (len(self.ivr_edges) + 1)
        )


def policy_vector_from_row(
    row: pd.Series,
    binner: StateBinner,
    Q: Dict[str, Dict[str, float]],
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute softmax policy vector from observation and Q-table.
    
    The policy is computed as:
        π(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
    
    Args:
        row: DataFrame row with state columns
        binner: StateBinner instance for discretization
        Q: Q-table mapping state keys to action Q-values
        temperature: Softmax temperature (lower = more deterministic)
        
    Returns:
        Array of shape (5,) with action probabilities for ACTIONS
    """
    # Extract state values (with defaults)
    vol = float(row.get("vol_energy", 0.0))
    cons = float(row.get("consolidation_score", 0.0))
    brk = float(row.get("breakout_score", 0.0))
    ivr = float(row.get("ivr", 50.0))
    
    # Get state key
    s_key = binner.state_key(vol, cons, brk, ivr)
    
    # Lookup Q-values (default to 0 for unknown states)
    default_q = {a: 0.0 for a in ACTIONS}
    q = Q.get(s_key, default_q)
    
    # Convert to vector
    vec = np.array([q.get(a, 0.0) for a in ACTIONS], dtype=float)
    
    # Softmax with temperature
    vec = vec / max(temperature, 1e-6)
    vec = vec - vec.max()  # Numerical stability
    expv = np.exp(vec)
    
    return expv / (expv.sum() + 1e-12)


def compute_policy_for_dataframe(
    df: pd.DataFrame,
    binner: Optional[StateBinner] = None,
    Q: Optional[Dict[str, Dict[str, float]]] = None,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute policy vectors for all rows in a DataFrame.
    
    Args:
        df: DataFrame with state columns
        binner: StateBinner (uses default if None)
        Q: Q-table (uses uniform if None)
        temperature: Softmax temperature
        
    Returns:
        Array of shape (N, 5) with action probabilities
    """
    if binner is None:
        binner = StateBinner()
    
    if Q is None:
        Q = {}  # Will use default Q-values (uniform)
    
    policies = np.zeros((len(df), len(ACTIONS)), dtype=float)
    
    for i, (_, row) in enumerate(df.iterrows()):
        policies[i] = policy_vector_from_row(row, binner, Q, temperature)
    
    return policies


def create_uniform_q_table(binner: StateBinner) -> Dict[str, Dict[str, float]]:
    """Create a Q-table with uniform action values for all states."""
    Q = {}
    # This would enumerate all states - expensive for large state spaces
    # In practice, Q-tables are learned from experience
    return Q
