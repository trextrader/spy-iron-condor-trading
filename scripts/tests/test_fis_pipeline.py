"""
scripts/tests/test_fis_pipeline.py

Integration test for the Fuzzy Inference System (FIS) Sizing Pipeline.
Verifies the orchestration between Fuzzifier, InferenceEngine, Defuzzifier, and Sizer.
"""
import unittest
import datetime as dt
import pandas as pd
import numpy as np

# Adjust path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.config import StrategyConfig
from core.dto import MarketSnapshot, TradeDecision
from intelligence.fis_sizer import FISSizer

class TestFISPipeline(unittest.TestCase):
    
    def setUp(self):
        self.cfg = StrategyConfig()
        # Ensure we have capital to trade
        self.cfg.risk_per_trade_pct = 0.02 # 2% risk
        self.cfg.max_contracts_per_trade = 10
        
        self.sizer = FISSizer(self.cfg)
        
    def test_full_sizing_flow(self):
        print("\n--- Testing FIS Sizing Pipeline (End-to-End) ---")
        
        # 1. Mock Market Snapshot (Favorable Conditions for IC)
        # Low volatility, Neutral RSI, Low ADX
        bars = pd.DataFrame({
            'high': [100]*50,
            'low': [99]*50,
            'close': [99.5]*50,
            'open': [99.5]*50,
            'volume': [1000]*50
        })
        
        snapshot = MarketSnapshot(
            ts=dt.datetime.now(),
            symbol="SPY",
            spot=400.0,
            bars=bars,
            option_chain=pd.DataFrame(), # Mock empty chain
            vix=15.0, # Low VIX
            es_price=4000.0,
        )
        # Inject mock chain for IV if needed by fuzzifier, but current fuzzifier might skip if empty.
        # Fuzzifier checks chain for ATM IV. Let's mock the internal call or just rely on defaults.
        
        # 2. Mock Trade Decision
        decision = TradeDecision(
            symbol="SPY",
            should_trade=True,
            structure="iron_condor",
            bias="neutral",
            rationale={"width": 5.0, "credit": 1.0} # Max Loss = 4.0 * 100 = $400
        )
        
        # 3. Run Sizer
        # Equity = $100,000. Risk 2% = $2,000.
        # Max Loss per contract = $400.
        # Base Qty (q0) = 2000 / 400 = 5 contracts.
        equity = 100000.0
        
        result = self.sizer.size_trade(decision, snapshot, equity)
        
        print(f"Result Contracts: {result.contracts}")
        print(f"Result Confidence: {result.confidence:.4f}")
        
        # Assertions
        self.assertTrue(result.contracts > 0, "Should size > 0 contracts in favorable conditions")
        self.assertTrue(result.contracts <= 5, "Should not exceed risk budget cap")
        self.assertTrue(0.0 <= result.confidence <= 1.0, "Confidence must be normalized")
        
    def test_high_volatility_scaling(self):
        print("\n--- Testing Volatility Scaling (high VIX) ---")
        
        # Mock High VIX Snapshot
        snapshot = MarketSnapshot(
            ts=dt.datetime.now(),
            symbol="SPY",
            spot=400.0,
            bars=pd.DataFrame({
                'high': [101]*50,
                'low': [99]*50,
                'close': [100]*50,
                'open': [100]*50,
                'volume': [2000]*50
            }), 
            option_chain=pd.DataFrame(),
            vix=40.0, # High VIX -> Classification: VOLATILE
            es_price=4000.0,
        )
        
        decision = TradeDecision(
            symbol="SPY",
            should_trade=True,
            structure="iron_condor",
            bias="neutral",
            rationale={"width": 5.0, "credit": 1.0}
        )
        
        result = self.sizer.size_trade(decision, snapshot, 100000.0)
        
        print(f"High VIX Contracts: {result.contracts}")
        # High VIX should reduce confidence and/or scaling factor
        # Defuzzifier: g = conf * (1 - sigma_star). VIX 40 -> Sigma=1.0 -> g=0.
        
        self.assertTrue(result.contracts < 5, "Should scale down in high volatility")
        # In extreme volatility (Sigma=1.0), it might even be 0.
        
if __name__ == '__main__':
    unittest.main()
