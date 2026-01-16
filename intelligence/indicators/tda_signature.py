"""
Persistent Homology Regime Signature (Topological Data Analysis)

This module provides TDA-based regime detection using persistent homology.
The key insight is that different market regimes (trending, ranging, volatile)
produce distinct topological signatures in their delay-embedded phase space.

Mathematical Foundation:
- Takens embedding: x(t) → [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
- Point cloud in R^m captures the "shape" of the price dynamics
- H1 persistent homology measures "holes" (loops) in the data
- High persistence indicates cyclic/ranging behavior
- Low persistence indicates trending behavior

Dependencies:
- ripser (optional): Fast Vietoris-Rips persistent homology
- Falls back gracefully if ripser is not installed
"""

import numpy as np
import pandas as pd
from typing import Optional

# Try to import ripser for fast TDA computation
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def _takens_embedding(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Construct Takens delay embedding of a 1D time series.
    
    Maps x(t) → [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
    
    This reconstructs the phase space of the underlying dynamical system.
    
    Args:
        series: 1D numpy array of observations
        m: Embedding dimension (number of delays + 1)
        tau: Time delay between observations
        
    Returns:
        Point cloud of shape (L, m) where L = len(series) - (m-1)*tau
    """
    n = len(series)
    L = n - (m - 1) * tau
    
    if L <= 10:
        # Not enough data for meaningful embedding
        return np.empty((0, m), dtype=float)
    
    out = np.zeros((L, m), dtype=float)
    for j in range(m):
        start_idx = (m - 1 - j) * tau
        out[:, j] = series[start_idx : start_idx + L]
    
    return out


def persistent_signature_ripser(point_cloud: np.ndarray, J: int = 2) -> float:
    """
    Compute persistent homology signature using ripser.
    
    We focus on H1 (1-dimensional holes, i.e., loops) because:
    - H0 mostly captures connectivity (less informative for regime)
    - H1 captures cyclic structure (ranging/mean-reverting behavior)
    
    The signature is the sum of the J longest lifetimes in H1.
    
    Args:
        point_cloud: (N, m) array of embedded points
        J: Number of top persistence bars to sum (default: 2)
        
    Returns:
        Sum of J longest H1 bar lifetimes (scalar)
    """
    if not HAS_RIPSER:
        return 0.0
    
    if point_cloud.shape[0] < 10:
        return 0.0
    
    # Run ripser with maxdim=1 (compute H0 and H1)
    res = ripser(point_cloud, maxdim=1)
    dgms = res.get("dgms", [])
    
    # H1 diagram is at index 1
    if len(dgms) < 2:
        return 0.0
    
    H1 = dgms[1]
    if H1.size == 0:
        return 0.0
    
    # Lifetimes = death - birth for each bar
    # Filter out infinite bars (birth, inf) if any
    finite_mask = np.isfinite(H1[:, 1])
    H1_finite = H1[finite_mask]
    
    if H1_finite.size == 0:
        return 0.0
    
    lifetimes = H1_finite[:, 1] - H1_finite[:, 0]
    lifetimes = np.sort(lifetimes)[::-1]  # Sort descending
    
    # Sum top J lifetimes
    return float(lifetimes[:J].sum())


def compute_pi_series(
    close: pd.Series,
    window: int = 256,
    m: int = 5,
    tau: int = 2,
    J: int = 2
) -> pd.Series:
    """
    Compute rolling persistent homology signature for price series.
    
    For each time t, we:
    1. Extract the last 'window' prices
    2. Construct Takens embedding with dimension m and delay tau
    3. Compute H1 persistent homology
    4. Return sum of top J persistence lifetimes
    
    Args:
        close: Price series
        window: Rolling window size (default: 256 bars)
        m: Embedding dimension (default: 5)
        tau: Time delay (default: 2)
        J: Number of top bars to sum (default: 2)
        
    Returns:
        Series of TDA signature values (NaN for warmup period)
    """
    c = close.astype(float).to_numpy()
    out = np.full_like(c, np.nan, dtype=float)
    
    for t in range(window - 1, len(c)):
        seg = c[t - window + 1 : t + 1]
        
        # Normalize segment to [0, 1] for numerical stability
        seg_min, seg_max = seg.min(), seg.max()
        if seg_max - seg_min < 1e-12:
            out[t] = 0.0
            continue
        seg_norm = (seg - seg_min) / (seg_max - seg_min)
        
        # Construct point cloud via Takens embedding
        pc = _takens_embedding(seg_norm, m=m, tau=tau)
        
        if pc.shape[0] == 0:
            out[t] = 0.0
            continue
        
        # Compute persistent signature
        if HAS_RIPSER:
            out[t] = persistent_signature_ripser(pc, J=J)
        else:
            # Fallback: return 0.0 if ripser not available
            out[t] = 0.0
    
    return pd.Series(out, index=close.index, name="pi_tda")


def is_ripser_available() -> bool:
    """Check if ripser is available for TDA computation."""
    return HAS_RIPSER
