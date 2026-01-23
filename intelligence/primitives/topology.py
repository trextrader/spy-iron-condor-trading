# intelligence/primitives/topology.py
"""
Topology and chaos primitives (T001-T002)
Exact canonical signatures - DO NOT MODIFY
"""

import numpy as np
import pandas as pd


def compute_beta1_regime_score(
    beta1_raw: pd.Series = None,
    window: int = 20,
    curvature_proxy: pd.Series = None,
    volatility_energy: pd.Series = None,
    alpha: float = 0.1,
    beta: float = 0.1,
    lag_minutes: pd.Series = None,
    lambda_decay: float = 0.1,
    # Aliases
    beta1_norm_stub: pd.Series = None,
) -> pd.DataFrame:
    """
    T001 - Normalized β₁ + Regime Score (Rule C2 v2.5)

    Returns DataFrame with columns:
        ['beta1_norm', 'beta1_gated', 'regime_score']
    """
    # Use alias if raw missing
    raw = beta1_raw if beta1_raw is not None else beta1_norm_stub
    if raw is None:
        # Fallback if neither exists
        return pd.DataFrame({
            "beta1_norm": pd.Series([0.0]),
            "beta1_gated": pd.Series([0.0]),
            "regime_score": pd.Series([0.0]),
        })
        
    mu = raw.rolling(window).mean()
    sigma = raw.rolling(window).std(ddof=0)
    beta1_norm = (raw - mu) / sigma.replace(0, np.nan)

    kappa = curvature_proxy.fillna(0.0) if curvature_proxy is not None else pd.Series(0, index=raw.index)
    vol_energy = volatility_energy.fillna(0.0) if volatility_energy is not None else pd.Series(0, index=raw.index)

    beta1_gated = beta1_norm * (1.0 + alpha * kappa) * (1.0 + beta * vol_energy)

    lag_minutes = lag_minutes.fillna(0.0) if lag_minutes is not None else pd.Series(0, index=raw.index)
    iv_conf = np.exp(-lambda_decay * lag_minutes)

    regime_score = beta1_gated * iv_conf

    return pd.DataFrame(
        {
            "beta1_norm": beta1_norm,
            "beta1_gated": beta1_gated,
            "regime_score": regime_score,
        }
    )


def compute_chaos_membership(
    beta1_gated: pd.Series = None,
    chaos_threshold: float = 0.5,
    relax_threshold: float = 0.3,
    # Aliases
    vol_energy: pd.Series = None,
) -> pd.DataFrame:
    """
    T002 - Chaos Membership (Rule E3 v2.5)

    Returns DataFrame with columns:
        ['chaos_membership', 'position_size_multiplier']
    """
    # Use vol_energy as proxy if beta1_gated is missing
    inp = beta1_gated if beta1_gated is not None else vol_energy
    if inp is None:
         return pd.DataFrame({
            "chaos_membership": pd.Series([0.0]),
            "position_size_multiplier": pd.Series([1.0]),
        })

    # sigmoid centered at chaos_threshold
    x = inp - chaos_threshold
    chaos_membership = 1.0 / (1.0 + np.exp(-x))
    chaos_membership = pd.Series(chaos_membership, index=inp.index)

    # clamp to 0 when below relax_threshold
    chaos_membership = np.where(
        inp < relax_threshold, 0.0, chaos_membership
    )
    chaos_membership = pd.Series(chaos_membership, index=inp.index)

    position_size_multiplier = 1.0 - chaos_membership
    position_size_multiplier = pd.Series(
        position_size_multiplier, index=inp.index
    )

    return pd.DataFrame(
        {
            "chaos_membership": chaos_membership,
            "position_size_multiplier": position_size_multiplier,
        }
    )
