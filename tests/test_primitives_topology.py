# tests/test_primitives_topology.py
"""
Unit tests for topology primitives (T001-T002)
"""

import numpy as np
import pandas as pd

from intelligence.primitives.topology import (
    compute_beta1_regime_score,
    compute_chaos_membership,
)


def test_beta1_regime_score_shapes():
    """Test T001 returns correct schema"""
    n = 200
    idx = pd.RangeIndex(n)
    beta1_raw = pd.Series(np.random.randint(0, 5, size=n), index=idx)
    curvature = pd.Series(0.1, index=idx)
    vol_energy = pd.Series(1.0, index=idx)
    lag = pd.Series(5.0, index=idx)

    out = compute_beta1_regime_score(
        beta1_raw,
        window=50,
        curvature_proxy=curvature,
        volatility_energy=vol_energy,
        alpha=0.2,
        beta=0.3,
        lag_minutes=lag,
        lambda_decay=0.05,
    )

    assert set(out.columns) == {
        "beta1_norm",
        "beta1_gated",
        "regime_score",
    }


def test_chaos_membership_bounds():
    """Test T002 chaos dampening"""
    beta1_gated = pd.Series(np.linspace(0, 5, 50))

    out = compute_chaos_membership(
        beta1_gated, chaos_threshold=2.5, relax_threshold=1.5
    )

    assert out["chaos_membership"].between(0, 1).all()
    assert out["position_size_multiplier"].between(0, 1).all()
    # At relax_threshold or below, chaos should be 0
    assert (out.loc[beta1_gated < 1.5, "chaos_membership"] == 0.0).all()
