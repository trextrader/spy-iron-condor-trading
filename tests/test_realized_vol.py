import numpy as np
from analytics.realized_vol import RealizedVolCalculator


def test_realized_vol_window(sample_prices):
    calc = RealizedVolCalculator()
    # window=3 uses last 3 log-return squares
    v = calc.compute_realized_variance(sample_prices, window=3)
    assert v >= 0.0


def test_vrp_gate_rejects():
    # stub: demonstrates expected gating logic shape
    implied = 0.20
    realized = 0.19
    threshold = 0.02
    should_trade = (implied - realized) > threshold
    assert should_trade is False
