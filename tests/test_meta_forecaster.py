
import numpy as np
import pandas as pd
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.meta_forecaster import MetaForecaster

logging.basicConfig(level=logging.DEBUG)

def create_dummy_ohlcv(n=600):
    """Create dummy OHLCV data"""
    dates = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.0005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0005, n)))
    open_ = close * (1 + np.random.normal(0, 0.0002, n)) # noisy
    volume = np.random.lognormal(10, 1, n)
    
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
    return df

def test_initialization():
    mf = MetaForecaster()
    assert mf.horizon == 5
    assert "YW" in mf.METHODS

def test_transform_features():
    mf = MetaForecaster()
    df = create_dummy_ohlcv(100)
    y = mf.transform_features(df)
    assert y.shape == (100, 4)
    # Check for NaNs (should be filled)
    assert not np.isnan(y).any()

def test_fit_predict_smoke():
    mf = MetaForecaster(fit_window=50, val_window=20)
    df = create_dummy_ohlcv(100)
    
    # Run fit_predict
    forecast_df = mf.fit_predict(df)
    
    assert isinstance(forecast_df, pd.DataFrame)
    assert len(forecast_df) == mf.horizon
    assert all(c in forecast_df.columns for c in ['open', 'high', 'low', 'close', 'volume'])
    
    # Check state updated
    assert mf.state.active_method is not None
    # Depending on random data, could be any method
    print(f"Selected Method: {mf.state.active_method} Order: {mf.state.active_order}")

def test_solvers():
    mf = MetaForecaster()
    x = np.random.randn(100)
    
    yw = mf._solve_yw(x, 2)
    assert len(yw) == 2
    
    burg = mf._solve_burg(x, 2)
    assert len(burg) == 2
    
    cov = mf._solve_cov(x, 2)
    assert len(cov) == 2
    
    mcov = mf._solve_mcov(x, 2)
    assert len(mcov) == 2

if __name__ == "__main__":
    test_fit_predict_smoke()
