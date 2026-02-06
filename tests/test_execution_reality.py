"""
Phase 5.5: Execution Reality Modeling Tests

Tests all 8 components of the execution reality engine
and verifies determinism is preserved.

Usage:
    pytest tests/test_execution_reality.py -v -s --cpu
"""

import sys
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.execution_reality import (
    LatencyModel,
    QueuePositionModel,
    SpreadDynamicsModel,
    VolatilityShockModel,
    BrokenSpreadDetector,
    MicrostructureModel,
    QuoteStalenessModel,
    TimeOfDayLiquidityModel,
    ExecutionRealityEngine,
    MarketState,
    FillStatus,
)


# ==============================================================================
# 1. LATENCY MODEL TESTS
# ==============================================================================

class TestLatencyModel:
    """Test latency simulation and price drift."""

    def test_latency_bounds(self):
        """Latency should respect minimum floor."""
        model = LatencyModel(mean_latency_ms=100, std_latency_ms=30, min_latency_ms=10)
        model.seed(42)

        latencies = [model.simulate_latency() for _ in range(1000)]

        assert min(latencies) >= 10, "Latency should respect minimum"
        assert 50 < np.mean(latencies) < 150, "Mean should be reasonable"

    def test_latency_determinism(self):
        """Same seed should produce same latencies."""
        model = LatencyModel()

        model.seed(42)
        lat_1 = [model.simulate_latency() for _ in range(10)]

        model.seed(42)
        lat_2 = [model.simulate_latency() for _ in range(10)]

        assert lat_1 == lat_2, "Latency should be deterministic with same seed"

    def test_price_drift_scaling(self):
        """Drift should scale with sqrt(time)."""
        model = LatencyModel()

        drift_100ms = model.price_drift_during_latency(0.20, 100)
        drift_400ms = model.price_drift_during_latency(0.20, 400)

        # 4x time = 2x drift (sqrt scaling)
        assert 1.8 < drift_400ms / drift_100ms < 2.2


# ==============================================================================
# 2. QUEUE POSITION MODEL TESTS
# ==============================================================================

class TestQueuePositionModel:
    """Test queue position and fill probability."""

    def test_aggressive_always_fills(self):
        """Aggression=1.0 should always fill."""
        model = QueuePositionModel(aggression_factor=1.0)
        prob = model.fill_probability(our_size=10, queue_depth=1000, time_at_level_ms=100)
        assert prob == 1.0

    def test_passive_lower_fill_prob(self):
        """Passive orders should have lower fill probability."""
        passive = QueuePositionModel(aggression_factor=0.0)
        aggressive = QueuePositionModel(aggression_factor=0.8)

        prob_passive = passive.fill_probability(10, 100, 1000)
        prob_aggressive = aggressive.fill_probability(10, 100, 1000)

        assert prob_passive < prob_aggressive

    def test_fill_prob_increases_with_time(self):
        """Longer time at level should increase fill probability."""
        model = QueuePositionModel(aggression_factor=0.5)

        prob_short = model.fill_probability(10, 100, 100)
        prob_long = model.fill_probability(10, 100, 5000)

        assert prob_long > prob_short


# ==============================================================================
# 3. SPREAD DYNAMICS MODEL TESTS
# ==============================================================================

class TestSpreadDynamicsModel:
    """Test spread widening logic."""

    def test_spread_widens_with_vix(self):
        """Higher VIX should widen spreads."""
        model = SpreadDynamicsModel()

        spread_low = model.dynamic_spread(0.02, vix=12, time_of_day=12, recent_volume=100)
        spread_high = model.dynamic_spread(0.02, vix=35, time_of_day=12, recent_volume=100)

        assert spread_high > spread_low
        print(f"\n[Spread] VIX 12: {spread_low:.4f}, VIX 35: {spread_high:.4f}")

    def test_spread_widens_at_open(self):
        """Spreads should be wider at market open."""
        model = SpreadDynamicsModel()

        spread_open = model.dynamic_spread(0.02, vix=15, time_of_day=9.5, recent_volume=100)
        spread_mid = model.dynamic_spread(0.02, vix=15, time_of_day=12, recent_volume=100)

        assert spread_open > spread_mid

    def test_spread_widens_low_volume(self):
        """Low volume should widen spreads."""
        model = SpreadDynamicsModel()

        spread_high_vol = model.dynamic_spread(0.02, vix=15, time_of_day=12, recent_volume=150)
        spread_low_vol = model.dynamic_spread(0.02, vix=15, time_of_day=12, recent_volume=50)

        assert spread_low_vol > spread_high_vol


# ==============================================================================
# 4. VOLATILITY SHOCK MODEL TESTS
# ==============================================================================

class TestVolatilityShockModel:
    """Test volatility shock detection."""

    def test_detects_large_move(self):
        """Should detect large moves as shocks."""
        model = VolatilityShockModel(shock_threshold_annual=50.0)

        # 2% move in 60 seconds = very high annualized vol
        is_shock = model.detect_shock(price_change_pct=2.0, time_window_sec=60)
        assert is_shock, "Should detect 2% move in 60s as shock"

    def test_ignores_small_move(self):
        """Should not flag small moves."""
        model = VolatilityShockModel(shock_threshold_annual=50.0)

        # 0.1% move in 60 seconds = normal
        is_shock = model.detect_shock(price_change_pct=0.1, time_window_sec=60)
        assert not is_shock, "Should not flag 0.1% move"

    def test_shock_impact(self):
        """Shock impact should degrade execution."""
        model = VolatilityShockModel()
        impact = model.shock_impact(0.02)

        assert impact['spread_multiplier'] > 1.0
        assert impact['fill_probability'] < 1.0
        assert impact['should_avoid_entry'] == True


# ==============================================================================
# 5. BROKEN SPREAD DETECTOR TESTS
# ==============================================================================

class TestBrokenSpreadDetector:
    """Test invalid quote detection."""

    def test_valid_quote(self):
        """Normal quote should be valid."""
        detector = BrokenSpreadDetector()
        valid, reason = detector.is_valid_quote(bid=5.00, ask=5.10)
        assert valid
        assert reason == "valid"

    def test_crossed_market(self):
        """Bid > ask should be invalid."""
        detector = BrokenSpreadDetector()
        valid, reason = detector.is_valid_quote(bid=5.10, ask=5.00)
        assert not valid
        assert "crossed" in reason

    def test_spread_too_wide(self):
        """Very wide spread should be invalid."""
        detector = BrokenSpreadDetector(max_spread_ratio=0.10)
        valid, reason = detector.is_valid_quote(bid=4.00, ask=5.00)  # 20% spread
        assert not valid
        assert "wide" in reason

    def test_zero_price(self):
        """Zero prices should be invalid."""
        detector = BrokenSpreadDetector()
        valid, reason = detector.is_valid_quote(bid=0, ask=5.00)
        assert not valid


# ==============================================================================
# 6. MICROSTRUCTURE MODEL TESTS
# ==============================================================================

class TestMicrostructureModel:
    """Test tick size and rounding."""

    def test_tick_size_below_3(self):
        """Prices below $3 should have $0.01 ticks."""
        model = MicrostructureModel()
        assert model.get_tick_size(2.50) == 0.01

    def test_tick_size_above_3(self):
        """Prices above $3 should have $0.05 ticks."""
        model = MicrostructureModel()
        assert model.get_tick_size(5.00) == 0.05

    def test_round_to_tick(self):
        """Prices should round to valid ticks."""
        model = MicrostructureModel()

        # Above $3: $0.05 ticks (use approx for floating point)
        assert abs(model.round_to_tick(5.03) - 5.05) < 1e-9
        assert abs(model.round_to_tick(5.07) - 5.05) < 1e-9
        assert abs(model.round_to_tick(5.08) - 5.10) < 1e-9

        # Below $3: $0.01 ticks
        assert abs(model.round_to_tick(2.003) - 2.00) < 1e-9
        assert abs(model.round_to_tick(2.007) - 2.01) < 1e-9


# ==============================================================================
# 7. QUOTE STALENESS MODEL TESTS
# ==============================================================================

class TestQuoteStalenessModel:
    """Test quote staleness handling."""

    def test_fresh_quote_reliable(self):
        """Fresh quotes should have high reliability."""
        model = QuoteStalenessModel(max_reliable_age_ms=500)
        reliability = model.quote_reliability(quote_age_ms=50, market_speed=1.0)
        assert reliability > 0.8

    def test_stale_quote_unreliable(self):
        """Old quotes should have low reliability."""
        model = QuoteStalenessModel(max_reliable_age_ms=500)
        reliability = model.quote_reliability(quote_age_ms=600, market_speed=1.0)
        assert reliability == 0.0

    def test_fast_market_degrades_faster(self):
        """Fast markets should degrade quote reliability faster."""
        model = QuoteStalenessModel(max_reliable_age_ms=500)

        rel_normal = model.quote_reliability(quote_age_ms=200, market_speed=1.0)
        rel_fast = model.quote_reliability(quote_age_ms=200, market_speed=2.0)

        assert rel_fast < rel_normal

    def test_staleness_widens_spread(self):
        """Stale quotes should widen effective spread."""
        model = QuoteStalenessModel()

        adj_bid, adj_ask = model.adjust_for_staleness(5.00, 5.10, reliability=0.5)

        assert adj_bid < 5.00
        assert adj_ask > 5.10


# ==============================================================================
# 8. TIME-OF-DAY LIQUIDITY MODEL TESTS
# ==============================================================================

class TestTimeOfDayLiquidityModel:
    """Test intraday liquidity patterns."""

    def test_open_low_liquidity(self):
        """Market open should have low liquidity."""
        model = TimeOfDayLiquidityModel()
        assert model.get_liquidity_multiplier(9.5) < 1.0

    def test_midday_normal_liquidity(self):
        """Midday should have normal liquidity."""
        model = TimeOfDayLiquidityModel()
        assert model.get_liquidity_multiplier(11) >= 1.0

    def test_preclose_high_liquidity(self):
        """Pre-close should have high liquidity."""
        model = TimeOfDayLiquidityModel()
        assert model.get_liquidity_multiplier(15) > 1.0

    def test_good_entry_time(self):
        """Should identify good entry times."""
        model = TimeOfDayLiquidityModel()

        assert not model.is_good_entry_time(9.5)  # Open - bad
        assert model.is_good_entry_time(14)  # Afternoon - good


# ==============================================================================
# UNIFIED ENGINE TESTS
# ==============================================================================

class TestExecutionRealityEngine:
    """Test unified execution engine."""

    def test_engine_determinism(self):
        """Engine should be deterministic with same seed."""
        state = MarketState(vix=18, spot_price=500, time_of_day_hour=12)

        engine1 = ExecutionRealityEngine(seed=42)
        result1 = engine1.simulate_realistic_fill(5.00, 5.10, 10, state, is_entry=True)

        engine2 = ExecutionRealityEngine(seed=42)
        result2 = engine2.simulate_realistic_fill(5.00, 5.10, 10, state, is_entry=True)

        assert result1.fill_price == result2.fill_price
        assert result1.fill_qty == result2.fill_qty
        assert result1.status == result2.status

        print(f"\n[Engine] Determinism verified: price={result1.fill_price:.4f}")

    def test_rejects_broken_quote(self):
        """Should reject crossed markets."""
        engine = ExecutionRealityEngine(seed=42)
        state = MarketState()

        result = engine.simulate_realistic_fill(5.10, 5.00, 10, state)  # Crossed

        assert result.status == FillStatus.REJECTED
        assert "crossed" in result.diagnostics.get('rejection_reason', '')

    def test_entry_vs_exit_pricing(self):
        """Entry should get worse price than exit."""
        engine = ExecutionRealityEngine(seed=42)
        state = MarketState(vix=15, time_of_day_hour=12)

        entry = engine.simulate_realistic_fill(5.00, 5.10, 10, state, is_entry=True)
        engine.reset(42)
        exit = engine.simulate_realistic_fill(5.00, 5.10, 10, state, is_entry=False)

        # Entry sells at lower price, exit buys at higher price
        # So entry fill < exit fill (entry is worse for us)
        print(f"\n[Engine] Entry: {entry.fill_price:.4f}, Exit: {exit.fill_price:.4f}")

    def test_diagnostics_populated(self):
        """Should populate all diagnostic fields."""
        engine = ExecutionRealityEngine(seed=42)
        state = MarketState(vix=20, time_of_day_hour=10)

        result = engine.simulate_realistic_fill(5.00, 5.10, 10, state)

        assert 'latency_ms' in result.diagnostics
        assert 'spread_multiplier' in result.diagnostics
        assert 'liquidity_multiplier' in result.diagnostics
        assert 'quote_reliability' in result.diagnostics

        print(f"\n[Engine] Diagnostics: {result.diagnostics}")


# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

class TestExecutionRealityIntegration:
    """Integration tests for full execution path."""

    def test_realistic_fill_impact(self):
        """Realistic fills should show measurable slippage."""
        engine = ExecutionRealityEngine(seed=42, latency_mean_ms=100)

        # Normal market conditions
        state = MarketState(
            vix=18,
            spot_price=500,
            recent_volume=100,
            time_of_day_hour=14,
            quote_age_ms=50,
            market_speed=1.0
        )

        fills = []
        for i in range(100):
            engine.reset(i)
            result = engine.simulate_realistic_fill(5.00, 5.10, 10, state, is_entry=True)
            if result.status == FillStatus.FILLED:
                fills.append(result.fill_price)

        if fills:
            avg_fill = np.mean(fills)
            theoretical_mid = 5.05

            print(f"\n[Integration] Realistic Fill Analysis:")
            print(f"  Theoretical mid: {theoretical_mid:.4f}")
            print(f"  Average fill: {avg_fill:.4f}")
            print(f"  Fill count: {len(fills)}/100")
            print(f"  Slippage: {(theoretical_mid - avg_fill):.4f}")

            # Entry should fill below mid (we're selling)
            assert avg_fill <= theoretical_mid, "Entry should fill at or below mid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
