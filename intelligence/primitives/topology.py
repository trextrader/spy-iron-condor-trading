# intelligence/primitives/topology.py
"""
Topology and chaos primitives (T001-T002)
Exact canonical signatures - DO NOT MODIFY
"""

import numpy as np
import pandas as pd


def compute_beta1_regime_score(
    beta1_raw: pd.Series,
    window: int = 20,
    curvature_proxy: pd.Series = None,
    volatility_energy: pd.Series = None,
    alpha: float = 0.1,
    beta: float = 0.1,
    lag_minutes: pd.Series = None,
    lambda_decay: float = 0.1,
) -> pd.DataFrame:
    """
    T001 - Normalized β₁ + Regime Score (Rule C2 v2.5)

    Returns DataFrame with columns:
        ['beta1_norm', 'beta1_gated', 'regime_score']
    """
    mu = beta1_raw.rolling(window).mean()
    sigma = beta1_raw.rolling(window).std(ddof=0)
    beta1_norm = (beta1_raw - mu) / sigma.replace(0, np.nan)

    kappa = curvature_proxy.fillna(0.0) if curvature_proxy is not None else pd.Series(0, index=beta1_raw.index)
    vol_energy = volatility_energy.fillna(0.0) if volatility_energy is not None else pd.Series(0, index=beta1_raw.index)

    beta1_gated = beta1_norm * (1.0 + alpha * kappa) * (1.0 + beta * vol_energy)

    lag_minutes = lag_minutes.fillna(0.0) if lag_minutes is not None else pd.Series(0, index=beta1_raw.index)
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
    beta1_gated: pd.Series,
    chaos_threshold: float = 0.5,
    relax_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    T002 - Chaos Membership (Rule E3 v2.5)

    Returns DataFrame with columns:
        ['chaos_membership', 'position_size_multiplier']
    """
    # sigmoid centered at chaos_threshold
    x = beta1_gated - chaos_threshold
    chaos_membership = 1.0 / (1.0 + np.exp(-x))
    chaos_membership = pd.Series(chaos_membership, index=beta1_gated.index)

    # clamp to 0 when below relax_threshold
    chaos_membership = np.where(
        beta1_gated < relax_threshold, 0.0, chaos_membership
    )
    chaos_membership = pd.Series(chaos_membership, index=beta1_gated.index)

    position_size_multiplier = 1.0 - chaos_membership
    position_size_multiplier = pd.Series(
        position_size_multiplier, index=beta1_gated.index
    )

    return pd.DataFrame(
        {
            "chaos_membership": chaos_membership,
            "position_size_multiplier": position_size_multiplier,
        }
    )
