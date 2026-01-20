# tests/test_primitives_momentum.py
"""
Unit tests for momentum primitives (M001-M004)
"""

import numpy as np
import pandas as pd

from intelligence.primitives.momentum import (
    compute_vol_normalized_macd,
    compute_vol_normalized_adx,
    compute_dynamic_rsi,
    compute_psar_flip_membership,
)


def test_vol_normalized_macd_shapes():
    """Test M001 returns correct schema"""
    close = pd.Series(np.linspace(100, 110, 100))
    vol_ewma = pd.Series(0.5, index=close.index)

    out = compute_vol_normalized_macd(close, 12, 26, 9, vol_ewma)

    assert set(out.columns) == {
        "macd",
        "signal",
        "hist",
        "macd_norm",
        "signal_norm",
    }


def test_vol_normalized_adx_range():
    """Test M002 ADX normalization"""
    n = 100
    idx = pd.RangeIndex(n)
    high = pd.Series(np.linspace(100, 110, n), index=idx)
    low = high - 1.0
    close = (high + low) / 2
    vol_energy = pd.Series(1.0, index=idx)

    adx_norm = compute_vol_normalized_adx(high, low, close, 14, 0.1, vol_energy)

    assert len(adx_norm) == n


def test_dynamic_rsi_basic():
    """Test M003 RSI calculation"""
    close = pd.Series(np.linspace(100, 110, 100))
    curvature = pd.Series(0.0, index=close.index)

    out = compute_dynamic_rsi(close, 14, curvature, 0.2)

    assert set(out.columns) == {"rsi", "rsi_dynamic"}
    assert out["rsi"].between(0, 100).all()


def test_psar_flip_membership_values():
    """Test M004 PSAR flip detection"""
    close = pd.Series([10, 11, 12, 11, 10])
    psar = pd.Series([9, 9.5, 10, 12, 13])

    out = compute_psar_flip_membership(close, psar)

    assert "psar_trend" in out.columns
    assert "psar_reversion_membership" in out.columns
    assert out["psar_trend"].isin([-1, 1]).all()
    assert out["psar_reversion_membership"].isin([0.0, 1.0]).all()
