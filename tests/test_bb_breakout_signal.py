import pandas as pd

from intelligence.primitives.signals import compute_bb_breakout_signal


def test_bb_breakout_signal_accepts_band_aliases():
    close = pd.Series([10.0, 11.0, 12.0])
    bb_upper = pd.Series([10.5, 10.5, 10.5])
    bb_lower = pd.Series([9.5, 9.5, 9.5])

    out = compute_bb_breakout_signal(
        close=close,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
    )

    assert out["bullish"].iloc[-1] is True
    assert out["bearish"].iloc[-1] is False


def test_bb_breakout_signal_handles_missing_bands():
    out = compute_bb_breakout_signal(close=None, upper_band=None, lower_band=None)
    assert out["bullish"].iloc[0] is False
    assert out["bearish"].iloc[0] is False
