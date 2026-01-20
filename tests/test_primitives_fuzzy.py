# tests/test_primitives_fuzzy.py
"""
Unit tests for fuzzy logic primitives (F001)
"""

import numpy as np
import pandas as pd

from intelligence.primitives.fuzzy import compute_fuzzy_reversion_score_11


def test_fuzzy_reversion_score_range():
    """Test F001 fuzzy score bounds and weight normalization"""
    idx = pd.RangeIndex(20)
    ones = pd.Series(1.0, index=idx)
    zeros = pd.Series(0.0, index=idx)

    fs = compute_fuzzy_reversion_score_11(
        mu_mtf=ones,
        mu_ivr=ones,
        mu_vix=zeros,
        mu_rsi=ones,
        mu_stoch=zeros,
        mu_adx=ones,
        mu_sma=zeros,
        mu_psar=ones,
        mu_bb=ones,
        mu_bbsqueeze=zeros,
        mu_vol=ones,
        w_mtf=0.25,
        w_ivr=0.15,
        w_vix=0.10,
        w_rsi=0.15,
        w_stoch=0.05,
        w_adx=0.05,
        w_sma=0.05,
        w_psar=0.10,
        w_bb=0.05,
        w_bbsqueeze=0.03,
        w_vol=0.02,
    )

    assert fs.between(0, 1).all()
    # Since most memberships are 1.0 with weights summing to 1.0,
    # score should be high
    assert fs.iloc[0] > 0.5


def test_fuzzy_score_weight_normalization():
    """Test F001 auto-normalizes weights"""
    idx = pd.RangeIndex(10)
    ones = pd.Series(1.0, index=idx)

    # Pass unnormalized weights (sum = 2.0)
    fs = compute_fuzzy_reversion_score_11(
        mu_mtf=ones,
        mu_ivr=ones,
        mu_vix=ones,
        mu_rsi=ones,
        mu_stoch=ones,
        mu_adx=ones,
        mu_sma=ones,
        mu_psar=ones,
        mu_bb=ones,
        mu_bbsqueeze=ones,
        mu_vol=ones,
        w_mtf=0.5,  # 2x normal
        w_ivr=0.3,
        w_vix=0.2,
        w_rsi=0.3,
        w_stoch=0.1,
        w_adx=0.1,
        w_sma=0.1,
        w_psar=0.2,
        w_bb=0.1,
        w_bbsqueeze=0.06,
        w_vol=0.04,
    )

    # Should still be in [0, 1] due to normalization
    assert fs.between(0, 1).all()
    # All memberships = 1.0, so score should be 1.0
    assert np.allclose(fs, 1.0)
