# tests/test_primitives_bands.py
"""
Unit tests for bands primitives (P001-P007)
"""

import numpy as np
import pandas as pd

from intelligence.primitives.bands import (
    compute_dynamic_bollinger_bands,
    compute_bandwidth_percentile_and_expansion,
    compute_volume_ratio,
    compute_spread_friction_ratio,
    compute_gap_risk_score,
    compute_iv_confidence,
    compute_mtf_consensus,
)


def test_dynamic_bollinger_bands_basic():
    """Test P001 returns correct schema and basic properties"""
    close = pd.Series(np.linspace(100, 110, 100))
    vol_energy = pd.Series(0.5, index=close.index)
    out = compute_dynamic_bollinger_bands(close, 20, 2.0, 0.2, vol_energy)

    assert set(out.columns) == {
        "upper_band",
        "middle_band",
        "lower_band",
        "bandwidth",
    }
    assert out["upper_band"].iloc[-1] > out["middle_band"].iloc[-1]
    assert out["lower_band"].iloc[-1] < out["middle_band"].iloc[-1]


def test_bandwidth_percentile_and_expansion_shapes():
    """Test P002 returns correct columns and length"""
    bw = pd.Series(np.random.rand(200))
    out = compute_bandwidth_percentile_and_expansion(bw, 100, 5)

    assert set(out.columns) == {"bw_percentile", "expansion_rate"}
    assert len(out) == len(bw)


def test_volume_ratio_simple():
    """Test P003 basic calculation"""
    vol = pd.Series(np.ones(50))
    vr = compute_volume_ratio(vol, 10)

    assert np.isclose(vr.iloc[-1], 1.0)


def test_spread_friction_ratio_exec_allow():
    """Test P004 friction gate logic"""
    idx = pd.RangeIndex(50)
    spread = pd.Series(0.1, index=idx)
    high = pd.Series(1.0, index=idx)
    low = pd.Series(0.9, index=idx)
    atr_norm = pd.Series(0.01, index=idx)
    vol_ratio = pd.Series(1.0, index=idx)
    bw = pd.Series(0.02, index=idx)
    event = pd.Series(0.0, index=idx)

    out = compute_spread_friction_ratio(
        spread,
        high,
        low,
        n=20,
        atr_norm=atr_norm,
        vol_ratio=vol_ratio,
        bandwidth=bw,
        event_flag=event,
        theta0=1.0,
        a=0.1,
        b=0.1,
        c=0.15,
        d=0.2,
        theta_min=0.5,
        theta_max=1.5,
    )

    assert "friction_ratio" in out.columns
    assert "exec_allow" in out.columns
    assert out["exec_allow"].isin([0, 1]).all()


def test_gap_risk_score_bounds():
    """Test P005 gap risk scoring"""
    n = 20
    event = pd.Series(1.0, index=range(n))
    atr_spike = pd.Series(1.0, index=range(n))
    bw_exp = pd.Series(1.0, index=range(n))
    late = pd.Series(1.0, index=range(n))

    out = compute_gap_risk_score(
        event,
        atr_spike,
        bw_exp,
        late,
        w_event=0.25,
        w_atr=0.25,
        w_bw=0.25,
        w_late=0.25,
        g_crit=0.7,
    )

    assert out["gap_risk_score"].between(0, 1).all()
    assert out["risk_override"].isin([0, 1]).all()


def test_iv_confidence_monotonic():
    """Test P006 IV confidence decay"""
    lag = pd.Series([0, 5, 10])
    conf = compute_iv_confidence(lag, lambda_decay=0.05)

    assert conf.iloc[0] > conf.iloc[1] > conf.iloc[2]
    assert conf.iloc[0] == 1.0


def test_mtf_consensus_range():
    """Test P007 MTF consensus bounds"""
    s1 = pd.Series([1, 1, -1])
    s5 = pd.Series([1, -1, -1])
    s15 = pd.Series([1, 1, 1])

    c = compute_mtf_consensus(s1, s5, s15, 0.2, 0.3, 0.5)

    assert c.between(-1, 1).all()
