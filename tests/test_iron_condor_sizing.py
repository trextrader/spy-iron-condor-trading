"""
Regression tests for Iron Condor position sizing.

These tests ensure the 1-lot failure bug is never reintroduced.
The original bug: fuzzy scaling reduced quantity back to 1 even after
fallback to 2, which violated the Iron Condor 2-wing minimum requirement.
"""

import pytest
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtmf.facade import benchmark_and_size
from qtmf.models import TradeIntent


class TestIronCondorMinimumQuantity:
    """
    Test suite for the 2-contract minimum Iron Condor requirement.
    
    Iron Condors have 4 legs (2 wings: put spread + call spread).
    Each wing needs at least 1 contract, so total minimum is 2.
    """
    
    def test_fallback_respects_two_wing_minimum(self):
        """
        REGRESSION TEST: Original 1-lot failure.
        
        When equity is too small to compute a risk-based quantity,
        the system should fall back to the configured minimum (2),
        NOT reduce it to 1 via scaling.
        """
        intent = TradeIntent(
            symbol="SPY",
            action="SELL_CONDOR",
            gaussian_confidence=0.55,  # Above threshold
            current_price=550.0,
            vix=20.0,
            ivr=30.0,
            extras={
                'equity': 1000.0,  # Very small account
                'max_loss_per_contract': 500.0,  # Would compute to ~0 from risk
                'risk_fraction': 0.02,
                'min_gaussian_confidence': 0.50,
                'fallback_total_qty': 2,  # Iron Condor minimum
                'require_two_wings': True,
                'min_total_qty_for_two_wings': 2,
            }
        )
        
        plan = benchmark_and_size(intent)
        
        assert plan.approved, f"Plan should be approved, got: {plan.reason}"
        assert plan.total_qty >= 2, (
            f"Iron Condor must have at least 2 contracts, got {plan.total_qty}. "
            "This is a REGRESSION of the 1-lot failure bug!"
        )
    
    def test_scaling_never_reduces_below_floor(self):
        """
        Even with aggressive scaling (low confidence), the floor must hold.
        """
        intent = TradeIntent(
            symbol="SPY",
            action="SELL_CONDOR",
            gaussian_confidence=0.51,  # Just above threshold (aggressive scaling)
            current_price=550.0,
            vix=35.0,  # High volatility (should reduce sizing)
            ivr=20.0,  # Low IV (would reduce further)
            extras={
                'equity': 25000.0,
                'max_loss_per_contract': 500.0,
                'risk_fraction': 0.02,
                'min_gaussian_confidence': 0.50,
                'fallback_total_qty': 2,
                'require_two_wings': True,
                'min_total_qty_for_two_wings': 2,
            }
        )
        
        plan = benchmark_and_size(intent)
        
        assert plan.approved, f"Plan should be approved, got: {plan.reason}"
        assert plan.total_qty >= 2, (
            f"Scaling must never reduce below floor. Got {plan.total_qty}, expected >= 2"
        )
    
    def test_both_wings_get_at_least_one_contract(self):
        """
        With exactly 2 contracts, both put and call wings must get 1 each.
        """
        intent = TradeIntent(
            symbol="SPY",
            action="SELL_CONDOR",
            gaussian_confidence=0.51,  # Low confidence = aggressive scaling
            current_price=550.0,
            extras={
                'equity': 1000.0,  # Force fallback
                'max_loss_per_contract': 5000.0,  # Would compute to 0
                'risk_fraction': 0.01,
                'min_gaussian_confidence': 0.50,
                'fallback_total_qty': 2,
                'require_two_wings': True,
                'min_total_qty_for_two_wings': 2,
            }
        )
        
        plan = benchmark_and_size(intent)
        
        assert plan.approved
        assert plan.put_qty >= 1, f"Put wing must have at least 1 contract, got {plan.put_qty}"
        assert plan.call_qty >= 1, f"Call wing must have at least 1 contract, got {plan.call_qty}"
        assert plan.put_qty + plan.call_qty == plan.total_qty
    
    def test_rejection_when_confidence_too_low(self):
        """
        System should reject trades with insufficient gaussian confidence.
        """
        intent = TradeIntent(
            symbol="SPY",
            action="SELL_CONDOR",
            gaussian_confidence=0.40,  # Below threshold
            current_price=550.0,
            extras={
                'min_gaussian_confidence': 0.50,
                'fallback_total_qty': 2,
            }
        )
        
        plan = benchmark_and_size(intent)
        
        assert not plan.approved
        assert "gaussian_confidence" in plan.reason
    
    def test_configurable_minimum_is_respected(self):
        """
        The min_total_qty_for_two_wings config should be respected.
        """
        # Test with custom minimum of 4
        intent = TradeIntent(
            symbol="SPY",
            action="SELL_CONDOR",
            gaussian_confidence=0.55,
            current_price=550.0,
            extras={
                'equity': 1000.0,
                'max_loss_per_contract': 5000.0,
                'risk_fraction': 0.01,
                'min_gaussian_confidence': 0.50,
                'fallback_total_qty': 4,  # Custom minimum
                'require_two_wings': True,
                'min_total_qty_for_two_wings': 4,  # Custom floor
            }
        )
        
        plan = benchmark_and_size(intent)
        
        assert plan.approved
        assert plan.total_qty >= 4, f"Custom floor of 4 not respected, got {plan.total_qty}"


class TestSingleLegStrategies:
    """
    Ensure single-leg strategies (non-Iron Condor) can still use 1 contract.
    """
    
    def test_single_leg_allows_one_contract(self):
        """
        When require_two_wings=False, minimum should be 1.
        """
        intent = TradeIntent(
            symbol="SPY",
            action="SELL_PUT",  # Single leg
            gaussian_confidence=0.55,
            current_price=550.0,
            extras={
                'equity': 1000.0,
                'max_loss_per_contract': 5000.0,
                'risk_fraction': 0.01,
                'min_gaussian_confidence': 0.50,
                'fallback_total_qty': 1,
                'require_two_wings': False,  # Single leg strategy
                'min_total_qty_for_two_wings': 1,
            }
        )
        
        plan = benchmark_and_size(intent)
        
        assert plan.approved
        assert plan.total_qty >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
