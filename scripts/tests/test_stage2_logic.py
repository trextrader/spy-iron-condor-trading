
import sys
import os
import unittest
import datetime as dt
from dataclasses import dataclass

# Fix path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.dto import OptionQuote
from strategies.options_strategy import (
    calculate_skew, nearest_by_delta, 
    regime_wing_width, build_condor, calculate_breach_probability
)
from intelligence.fuzzy_engine import classify_market_regime

class MockConfig:
    wing_width_min = 5.0
    wing_width_max = 20.0
    regime_iv_rank_widen = 50.0
    regime_vix_widen = 20.0
    width_widen_increment = 5.0
    
    target_short_delta_low = 0.12
    target_short_delta_high = 0.30

class TestStage2Logic(unittest.TestCase):
    
    def test_classify_market_regime(self):
        print("\n--- Testing Regime Classifier ---")
        # Volatile
        r = classify_market_regime(adx=15, rsi=50, vix=30)
        print(f"VIX=30 -> {r}")
        self.assertEqual(r, "VOLATILE")
        
        # Trending
        r = classify_market_regime(adx=30, rsi=70, vix=15)
        print(f"ADX=30, RSI=70 -> {r}")
        self.assertEqual(r, "TRENDING")
        
        # Ranging
        r = classify_market_regime(adx=15, rsi=50, vix=15)
        print(f"ADX=15, RSI=50 -> {r}")
        self.assertEqual(r, "RANGING")
        
        # Neutral
        r = classify_market_regime(adx=25, rsi=30, vix=15) # RSI low but ADX boundary
        print(f"ADX=25, RSI=30 -> {r}")
        
    def test_regime_wing_width(self):
        print("\n--- Testing Regime Wing Width ---")
        cfg = MockConfig()
        
        # Neutral (Base)
        w = regime_wing_width(cfg, iv_rank=10, vix=15, regime="NEUTRAL")
        print(f"Neutral: {w}")
        self.assertEqual(w, 5.0)
        
        # Volatile (Max)
        w = regime_wing_width(cfg, iv_rank=10, vix=15, regime="VOLATILE")
        print(f"Volatile: {w}")
        self.assertEqual(w, 20.0)
        
        # Trending (Widen)
        w = regime_wing_width(cfg, iv_rank=10, vix=15, regime="TRENDING")
        print(f"Trending: {w}")
        self.assertEqual(w, 10.0) # Base + Inc = 5+5=10
        
    def test_calculate_skew(self):
        print("\n--- Testing Skew Calculation (Put IV - Call IV) ---")
        # Create quotes
        # 25 Delta Put with IV 20%
        p25 = OptionQuote(strike=100, bid=1, ask=1, mid=1, iv=0.20, delta=-0.25, is_call=False)
        # 25 Delta Call with IV 15%
        c25 = OptionQuote(strike=110, bid=1, ask=1, mid=1, iv=0.15, delta=0.25, is_call=True)
        # Noise
        p10 = OptionQuote(strike=90, bid=1, ask=1, mid=1, iv=0.25, delta=-0.10, is_call=False)
        
        chain = [p25, c25, p10]
        
        skew = calculate_skew(chain)
        print(f"Skew: {skew:.4f}")
        self.assertAlmostEqual(skew, 0.05)
        
    def test_breach_probability(self):
        print("\n--- Testing Breach Probability ---")
        prob = calculate_breach_probability(delta=-0.30)
        self.assertEqual(prob, 0.30)
        
    def test_nearest_by_delta_skew_logic(self):
        print("\n--- Testing Skew Penalty in Strike Selection ---")
        # Scenario: High Skew (Puts Expensive). We want to sell Puts.
        # But we have a safety logic: if skew is high, avoid "risky" high delta puts.
        
        market_skew = 0.06 # High skew
        
        # Put A: 0.28 Delta (Risky), IV 0.22
        pA = OptionQuote(strike=100, bid=1, ask=1, mid=1, iv=0.22, delta=-0.28, is_call=False)
        # Put B: 0.15 Delta (Safe), IV 0.20
        pB = OptionQuote(strike=90, bid=1, ask=1, mid=1, iv=0.20, delta=-0.15, is_call=False)
        
        # Target range: 0.12 - 0.30. Center is 0.21.
        # Dist A: |0.28 - 0.21| = 0.07
        # Dist B: |0.15 - 0.21| = 0.06
        # Without skew, B is slightly closer to center (0.06 vs 0.07). B wins.
        # Wait, let's make A closer to center.
        
        # Put A: 0.22 Delta (Center-ish). Dist = 0.01.
        pA.delta = -0.22 
        # Put B: 0.15 Delta (Far). Dist = 0.06.
        # Normally A wins easily.
        
        # Apply Logic:
        # If market_skew > 0.05 and Is Put:
        # Penalize if abs(delta) > center (0.21).
        # A (0.22) > 0.21 -> Penalty applied.
        # B (0.15) < 0.21 -> No penalty.
        
        chain = [pA, pB]
        
        chosen = nearest_by_delta(chain, is_call=False, target_low=0.12, target_high=0.30, atm_iv=0.20, market_skew=market_skew)
        
        print(f"Chosen Delta for High Skew market: {chosen.delta}")
        # We expect B because A got penalized for being too aggressive in high skew env
        self.assertEqual(chosen, pB)
        

if __name__ == '__main__':
    unittest.main()
