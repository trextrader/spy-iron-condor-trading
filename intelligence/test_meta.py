
import sys
import os
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligence.meta_forecaster import MetaForecaster, transform_ohlcv, reconstruct_ohlcv

def test_meta_forecaster():
    print("Testing MetaForecaster...")
    
    # 1. Generate Synthetic OHLCV
    np.random.seed(42)
    T = 1000
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.001, T))
    highs = prices * (1 + np.abs(np.random.normal(0, 0.0005, T)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.0005, T)))
    opens = prices * (1 + np.random.normal(0, 0.0002, T)) # Noisy open
    volumes = np.abs(np.random.normal(1000, 200, T))
    
    # 2. Transform to features
    y = transform_ohlcv(opens, highs, lows, prices, volumes)
    print(f"Features shape: {y.shape}")
    
    # 3. Init Forecaster
    mf = MetaForecaster(
        condor_brain_model=None, # Test classical only for simplicity
        methods=["YW", "BURG", "COV"], 
        fit_window=100, 
        val_window=50,
        horizon=5
    )
    
    # 4. Run loop
    print("\nRunning Simulation:")
    history = []
    
    for t in range(200, 250):
        # Current view of history ending at t
        # (Mocking real-time feed)
        y_slice = y[:t] 
        
        pred, info = mf.step(y_slice)
        
        # Test reconstruction
        last_bar = {
            'close': prices[t-1], 'volume': volumes[t-1], 
            'open': opens[t-1], 'high': highs[t-1], 'low': lows[t-1]
        }
        reconstructed = reconstruct_ohlcv(last_bar, pred)
        
        print(f"T={t} | Selected: {info['method']} (p={info['order']}) | Err: {info['error']:.4f} | Next Close: {reconstructed[0]['close']:.2f}")
        
    print("\nTest Complete.")

if __name__ == "__main__":
    test_meta_forecaster()
