"""
Manifold-Based Volatility and Topological Indicators

This module provides geometric and topological features derived from
price manifolds, including curvature-based volatility proxies and
dynamically-weighted momentum indicators.

Mathematical Foundation:
- Curvature κ(t) measures the "bendiness" of the return trajectory
- Volatility energy E(t) = log(1 + α|κ(t)|) provides scale-invariant regime signal
- Dynamic RSI weights gains/losses by volatility energy for regime-awareness
"""

import numpy as np
import pandas as pd
from typing import Optional


def curvature_proxy_from_returns(r: pd.Series, span: int = 64) -> pd.Series:
    """
    Compute a curvature proxy from returns using second-order differences.
    
    The curvature κ is approximated as:
        κ(t) ≈ d²r/dt² / |dr/dt| = (rₜ - 2rₜ₋₁ + rₜ₋₂) / scale
    
    This measures how quickly the return trajectory is "bending" - high curvature
    indicates regime transitions or trend reversals.
    
    Args:
        r: Series of log returns
        span: EWM span for scale normalization (default: 64 bars)
        
    Returns:
        Smoothed curvature proxy series (same index as input)
    """
    dr = r.diff()
    d2r = dr.diff()
    
    # Scale by local volatility (prevents division by near-zero)
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    kappa = d2r / scale
    
    # Smooth the curvature to reduce noise
    smoothing_span = max(8, span // 4)
    return kappa.ewm(span=smoothing_span, adjust=False).mean()


def volatility_energy_from_curvature(kappa: pd.Series, alpha: float = 1.0) -> pd.Series:
    """
    Convert curvature to a volatility "energy" measure.
    
    The energy is defined as:
        E(t) = log(1 + α|κ(t)|)
    
    This provides a scale-invariant, non-negative measure that:
    - Returns 0 when curvature is 0 (linear regime)
    - Grows logarithmically with curvature (bounded growth)
    - Is interpretable as "regime stress"
    
    Args:
        kappa: Curvature proxy from curvature_proxy_from_returns()
        alpha: Scaling factor (default: 1.0)
        
    Returns:
        Volatility energy series
    """
    u = kappa.abs().astype(float)
    return np.log1p(alpha * u)


def dynamic_rsi(
    close: pd.Series, 
    window: int = 14, 
    vol_energy: Optional[pd.Series] = None
) -> pd.Series:
    """
    Compute a volatility-weighted RSI (Relative Strength Index).
    
    Standard RSI uses equal weights for all gains/losses. This dynamic RSI
    weights each observation by (1 + vol_energy), giving more importance to
    price moves during high-volatility (high-curvature) periods.
    
    This makes RSI more responsive during regime transitions while remaining
    stable during low-volatility consolidation.
    
    Args:
        close: Price series
        window: RSI window (default: 14)
        vol_energy: Optional volatility energy weights. If None, uses standard RSI.
        
    Returns:
        Dynamic RSI series (0-100 scale)
    """
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    
    if vol_energy is None:
        # Standard RSI
        avg_gain = gains.rolling(window, min_periods=window).mean()
        avg_loss = losses.rolling(window, min_periods=window).mean()
    else:
        # Volatility-weighted RSI
        weights = (1.0 + vol_energy).astype(float)
        
        weighted_gains = gains * weights
        weighted_losses = losses * weights
        
        sum_weights = weights.rolling(window, min_periods=window).sum() + 1e-12
        avg_gain = weighted_gains.rolling(window, min_periods=window).sum() / sum_weights
        avg_loss = weighted_losses.rolling(window, min_periods=window).sum() / sum_weights
    
    # RS and RSI calculation
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def compute_manifold_features(close: pd.Series, span: int = 64) -> pd.DataFrame:
    """
    Convenience function to compute all manifold-based features at once.
    
    Args:
        close: Price series
        span: Base span for curvature calculation
        
    Returns:
        DataFrame with columns: log_return, curvature, vol_energy, dynamic_rsi
    """
    log_ret = np.log(close).diff()
    
    curvature = curvature_proxy_from_returns(log_ret, span=span)
    vol_energy = volatility_energy_from_curvature(curvature)
    dyn_rsi = dynamic_rsi(close, window=14, vol_energy=vol_energy)
    
    return pd.DataFrame({
        'log_return': log_ret,
        'curvature': curvature,
        'vol_energy': vol_energy,
        'dynamic_rsi': dyn_rsi
    }, index=close.index)
