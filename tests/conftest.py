import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices():
    # deterministic toy series
    return np.array([100, 101, 100.5, 102, 101.8, 103, 102.7], dtype=float)


@pytest.fixture
def sample_bars_5m():
    # Minimal OHLCV DataFrame typical of your 5-min bars
    idx = pd.date_range("2026-01-02 09:30:00", periods=60, freq="5min")
    close = np.linspace(100, 102, len(idx))
    high = close + 0.2
    low = close - 0.2
    open_ = close + np.random.default_rng(0).normal(0, 0.05, len(idx))
    vol = np.random.default_rng(1).integers(1000, 5000, len(idx))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx
    )


@pytest.fixture
def sample_spread_series():
    # Spread series for Z-score tests
    return np.array([0.1, 0.12, 0.09, 0.11, 0.13, 0.14, 0.10], dtype=float)
