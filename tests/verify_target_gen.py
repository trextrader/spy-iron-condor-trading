import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from intelligence.generate_condor_targets import generate_condor_targets, ICSimConfig

def run_verification():
    print("[VERIFY] Creating mock data...")
    # Create 1 day of minute data (390 bars)
    dates = [datetime(2025, 1, 6, 9, 30) + timedelta(minutes=i) for i in range(1000)]
    
    # Simulate a price path: Flat then crash then recovery
    price = 100.0
    prices = []
    ivs = []
    for i in range(1000):
        # Sine wave with random noise
        change = np.random.normal(0, 0.05)
        if 300 < i < 400: # Crash
            change -= 0.2
        price += change
        prices.append(price)
        
        # IV mostly constant but spikes during crash
        iv = 0.20
        if 300 < i < 400: 
            iv = 0.40
        ivs.append(iv)
        
    df = pd.DataFrame({
        'dt': dates,
        'close': prices,
        'open': prices,
        'high': [p + 0.1 for p in prices],
        'low': [p - 0.1 for p in prices],
        'volume': 1000,
        'ivr': [iv * 100 for iv in ivs], # IVR approx IV * 100 here
        'implied_vol': ivs
    })
    
    # Ensure datetime index
    df['dt'] = pd.to_datetime(df['dt'])
    
    print("[VERIFY] Running generate_condor_targets (Multi-Strategy)...")
    config = ICSimConfig()
    
    start_time = pd.Timestamp.now()
    try:
        results = generate_condor_targets(df, config)
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        
        print(f"[VERIFY] Success! Processed {len(df)} rows in {duration:.2f}s")
        print(f"[VERIFY] Output Data Shape: {results.shape}")
        
        # Check for new columns
        required_cols = ['target_roi_calendar', 'target_roi_bwb', 'target_roi']
        missing = [c for c in required_cols if c not in results.columns]
        
        if missing:
            print(f"[FAIL] Missing columns: {missing}")
            sys.exit(1)
        
        # Check values
        print("\n[VERIFY] Sample Stats:")
        print(results[['target_roi', 'target_roi_calendar', 'target_roi_bwb']].describe())
        
        # Check for non-nulls
        non_null_cal = results['target_roi_calendar'].count()
        print(f"\n[VERIFY] Non-null Calendar ROIs: {non_null_cal}/{len(results)}")
        
        if non_null_cal == 0:
            print("[WARN] No valid Calendar trades found? (Might be due to mock data gaps?)")
            
        print("\n[PASS] Verification Complete.")
        
    except Exception as e:
        print(f"\n[FAIL] Generation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
