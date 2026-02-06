"""
Phase 5.5: Execution Reality Modeling

Models the messiness of real market execution on top of the proven
deterministic foundation from Phases 5.1-5.4.

Components:
    1. LatencyModel - Order-to-fill delay
    2. QueuePositionModel - FIFO queue dynamics
    3. SpreadDynamicsModel - Volatility/time-dependent spreads
    4. VolatilityShockModel - Sudden spike handling
    5. BrokenSpreadDetector - Invalid quote detection
    6. MicrostructureModel - Tick sizes, rounding
    7. QuoteStalenessModel - Stale quote handling
    8. TimeOfDayLiquidityModel - Intraday liquidity regimes

All models are deterministic when seeded, preserving the Replay Theorem.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

class FillStatus(Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    QUEUED = "queued"


@dataclass
class MarketState:
    """Current market conditions for execution modeling."""
    vix: float = 15.0
    spot_price: float = 500.0
    recent_volume: float = 100.0
    time_of_day_hour: float = 12.0  # Hour in ET (9.5 = 9:30 AM)
    recent_price_change_pct: float = 0.0
    quote_age_ms: float = 50.0
    market_speed: float = 1.0  # 1.0 = normal, >1 = fast


@dataclass
class FillResult:
    """Result of execution simulation."""
    status: FillStatus
    fill_price: float
    fill_qty: int
    fill_time_ms: float
    slippage: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 1. LATENCY MODEL
# ==============================================================================

class LatencyModel:
    """
    Models order-to-fill delay and price drift during transit.

    Typical latencies:
        - Retail: 100-300ms
        - Institutional: 10-50ms
        - HFT: <1ms
    """

    def __init__(self, mean_latency_ms: float = 100.0, std_latency_ms: float = 30.0,
                 min_latency_ms: float = 10.0):
        self.mean = mean_latency_ms
        self.std = std_latency_ms
        self.min = min_latency_ms
        self._rng = np.random.default_rng()

    def seed(self, seed: int):
        """Set random seed for determinism."""
        self._rng = np.random.default_rng(seed)

    def simulate_latency(self) -> float:
        """Returns latency in milliseconds."""
        return max(self.min, self._rng.normal(self.mean, self.std))

    def price_drift_during_latency(self, volatility_annual: float, latency_ms: float) -> float:
        """
        Estimate expected absolute price movement during order transit.

        Uses sqrt(t) scaling for Brownian motion.

        Args:
            volatility_annual: Annualized volatility (e.g., 0.20 for 20%)
            latency_ms: Latency in milliseconds

        Returns:
            Expected absolute price drift as fraction of price
        """
        # Convert latency to fraction of trading year
        # 252 days * 6.5 hours * 60 min * 60 sec * 1000 ms
        trading_ms_per_year = 252 * 6.5 * 60 * 60 * 1000
        t_fraction = latency_ms / trading_ms_per_year

        # Expected absolute movement = vol * sqrt(t) * sqrt(2/pi) for half-normal
        return volatility_annual * np.sqrt(t_fraction) * np.sqrt(2 / np.pi)


# ==============================================================================
# 2. QUEUE POSITION MODEL
# ==============================================================================

class QueuePositionModel:
    """
    Models FIFO queue dynamics at a price level.

    Aggression factor:
        0.0 = passive (back of queue, best price)
        0.5 = mid (join queue midway)
        1.0 = aggressive (cross spread, guaranteed fill)
    """

    def __init__(self, aggression_factor: float = 0.5):
        self.aggression = max(0.0, min(1.0, aggression_factor))

    def fill_probability(self, our_size: int, queue_depth: int,
                         time_at_level_ms: float) -> float:
        """
        Estimate probability of fill given queue position.

        Args:
            our_size: Our order size
            queue_depth: Total size ahead of us in queue
            time_at_level_ms: Time price has been at this level

        Returns:
            Probability of fill (0-1)
        """
        if self.aggression >= 1.0:
            return 1.0  # Crossing spread always fills

        # Estimate our position in queue
        position_ahead = queue_depth * (1 - self.aggression)

        # Fill rate based on how much needs to trade before us
        if position_ahead <= 0:
            base_prob = 1.0
        else:
            base_prob = our_size / (our_size + position_ahead)

        # Time decay: longer at level = more likely to fill
        time_factor = min(1.0, time_at_level_ms / 5000)  # 5 sec to full prob

        return min(1.0, base_prob * (0.5 + 0.5 * time_factor))

    def expected_fill_time(self, our_size: int, queue_depth: int,
                           trade_rate_per_sec: float) -> float:
        """Estimate time to fill in milliseconds."""
        if self.aggression >= 1.0:
            return 50.0  # Immediate fill when crossing

        position_ahead = queue_depth * (1 - self.aggression)
        if trade_rate_per_sec <= 0:
            return 10000.0  # 10 sec max

        return (position_ahead / trade_rate_per_sec) * 1000


# ==============================================================================
# 3. SPREAD DYNAMICS MODEL
# ==============================================================================

class SpreadDynamicsModel:
    """
    Models how spreads change based on market conditions.

    Spread widens during:
        - High VIX
        - Market open/close
        - Low volume
        - News events
    """

    def __init__(self, base_spread_ratio: float = 0.02):
        self.base_spread = base_spread_ratio

    def dynamic_spread(self, base_spread: float, vix: float,
                       time_of_day: float, recent_volume: float) -> float:
        """
        Calculate spread based on current market conditions.

        Args:
            base_spread: Base spread ratio from quote
            vix: Current VIX level
            time_of_day: Hour in ET (9.5 = 9:30 AM, 16.0 = 4:00 PM)
            recent_volume: Recent volume as % of average (100 = normal)

        Returns:
            Adjusted spread ratio
        """
        # VIX multiplier: linear scaling above VIX 15
        vix_mult = 1.0 + max(0, (vix - 15) / 30)

        # Time-of-day multiplier
        if time_of_day < 9.75 or time_of_day > 15.75:  # First/last 15 min
            tod_mult = 2.0
        elif time_of_day < 10.0 or time_of_day > 15.5:  # First/last 30 min
            tod_mult = 1.5
        elif time_of_day < 10.5 or time_of_day > 15.0:
            tod_mult = 1.2
        else:
            tod_mult = 1.0

        # Volume multiplier: inverse relationship
        vol_mult = 1.0 / max(0.3, min(2.0, recent_volume / 100))

        return base_spread * vix_mult * tod_mult * vol_mult

    def spread_impact_on_credit(self, base_credit: float, spread_mult: float) -> float:
        """
        Calculate credit reduction due to spread widening.

        For a 4-leg Iron Condor, spread impact is on all legs.
        """
        # Each leg loses half-spread on entry
        # 4 legs = 2 full spreads of slippage
        spread_cost = base_credit * (spread_mult - 1) * 0.5
        return max(0, base_credit - spread_cost)


# ==============================================================================
# 4. VOLATILITY SHOCK MODEL
# ==============================================================================

class VolatilityShockModel:
    """
    Detects and handles sudden volatility spikes.

    During shocks:
        - Spreads blow out 3-10x
        - Quotes may be withdrawn
        - Fill probability drops
    """

    def __init__(self, shock_threshold_annual: float = 50.0):
        """
        Args:
            shock_threshold_annual: Annualized vol threshold for shock detection
        """
        self.threshold = shock_threshold_annual

    def detect_shock(self, price_change_pct: float, time_window_sec: float) -> bool:
        """
        Detect if a volatility shock is occurring.

        Args:
            price_change_pct: Recent price change as percentage
            time_window_sec: Time window for the change

        Returns:
            True if shock detected
        """
        if time_window_sec <= 0:
            return False

        # Annualize the move
        # Trading year = 252 days * 6.5 hours * 3600 seconds
        trading_sec_per_year = 252 * 6.5 * 3600
        annualized = abs(price_change_pct) * np.sqrt(trading_sec_per_year / time_window_sec)

        return annualized > self.threshold

    def shock_impact(self, normal_spread: float) -> Dict[str, Any]:
        """
        Return degraded execution conditions during shock.

        Returns:
            Dict with adjusted parameters
        """
        return {
            'spread_multiplier': 3.0,
            'fill_probability': 0.3,
            'quote_staleness_ms': 500,
            'should_avoid_entry': True,
            'reason': 'volatility_shock_detected'
        }


# ==============================================================================
# 5. BROKEN SPREAD DETECTOR
# ==============================================================================

class BrokenSpreadDetector:
    """
    Detects invalid or unreliable quotes.

    Invalid conditions:
        - Crossed market (bid > ask)
        - Zero prices
        - Spread too wide
        - Quote too far from last trade
    """

    def __init__(self, max_spread_ratio: float = 0.15, min_price: float = 0.01,
                 max_deviation_from_last: float = 0.20):
        self.max_spread = max_spread_ratio
        self.min_price = min_price
        self.max_deviation = max_deviation_from_last

    def is_valid_quote(self, bid: float, ask: float,
                       last_trade: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if a quote is valid for execution.

        Returns:
            (is_valid, reason)
        """
        if bid <= 0 or ask <= 0:
            return False, "zero_or_negative_price"

        if bid > ask:
            return False, "crossed_market"

        if ask < self.min_price:
            return False, "price_below_minimum"

        spread_ratio = (ask - bid) / ask
        if spread_ratio > self.max_spread:
            return False, f"spread_too_wide_{spread_ratio:.2%}"

        if last_trade is not None and last_trade > 0:
            mid = (bid + ask) / 2
            deviation = abs(mid - last_trade) / last_trade
            if deviation > self.max_deviation:
                return False, f"quote_stale_deviation_{deviation:.2%}"

        return True, "valid"


# ==============================================================================
# 6. MICROSTRUCTURE MODEL
# ==============================================================================

class MicrostructureModel:
    """
    Models tick sizes and price rounding.

    SPY Options tick sizes:
        - Below $3: $0.01 increments
        - Above $3: $0.05 increments
    """

    def __init__(self, tick_below_3: float = 0.01, tick_above_3: float = 0.05):
        self.tick_below = tick_below_3
        self.tick_above = tick_above_3

    def get_tick_size(self, price: float) -> float:
        """Get tick size for a given price level."""
        return self.tick_below if price < 3.0 else self.tick_above

    def round_to_tick(self, price: float, direction: str = 'nearest') -> float:
        """
        Round price to valid tick increment.

        Args:
            price: Price to round
            direction: 'nearest', 'up', or 'down'
        """
        tick = self.get_tick_size(price)

        if direction == 'down':
            return np.floor(price / tick) * tick
        elif direction == 'up':
            return np.ceil(price / tick) * tick
        else:  # nearest
            return round(price / tick) * tick

    def can_improve_price(self, current_bid: float, current_ask: float) -> bool:
        """Check if there's room to improve inside the spread."""
        tick = self.get_tick_size((current_bid + current_ask) / 2)
        return (current_ask - current_bid) > tick

    def minimum_credit_increment(self, price_level: float) -> float:
        """Minimum credit improvement possible at this price level."""
        return self.get_tick_size(price_level)


# ==============================================================================
# 7. QUOTE STALENESS MODEL
# ==============================================================================

class QuoteStalenessModel:
    """
    Models quote reliability based on age and market speed.

    Stale quotes require wider safety margins.
    """

    def __init__(self, max_reliable_age_ms: float = 500.0):
        self.max_age = max_reliable_age_ms

    def quote_reliability(self, quote_age_ms: float, market_speed: float = 1.0) -> float:
        """
        Return reliability score 0-1 based on quote age.

        Args:
            quote_age_ms: Age of quote in milliseconds
            market_speed: Market speed multiplier (>1 = fast market)

        Returns:
            Reliability score (1.0 = perfectly fresh, 0.0 = totally stale)
        """
        effective_age = quote_age_ms * market_speed
        return max(0, 1 - effective_age / self.max_age)

    def adjust_for_staleness(self, bid: float, ask: float,
                             reliability: float) -> Tuple[float, float]:
        """
        Widen spread to account for staleness uncertainty.

        Returns:
            (adjusted_bid, adjusted_ask)
        """
        uncertainty = (1 - reliability) * (ask - bid)
        return bid - uncertainty, ask + uncertainty


# ==============================================================================
# 8. TIME-OF-DAY LIQUIDITY MODEL
# ==============================================================================

class TimeOfDayLiquidityModel:
    """
    Models intraday liquidity variations.

    Liquidity is typically:
        - Low at open (high volatility)
        - Normal mid-morning
        - Lower at lunch
        - Highest in afternoon
        - Volatile at close
    """

    def __init__(self):
        # Liquidity multipliers by hour (ET)
        self.liquidity_curve = {
            9: 0.5,    # 9:00-9:30: Pre-market, very thin
            9.5: 0.6,  # 9:30-10:00: Open, volatile
            10: 0.9,   # 10:00-11:00: Settling
            11: 1.0,   # 11:00-12:00: Normal
            12: 0.8,   # 12:00-13:00: Lunch lull
            13: 0.9,   # 13:00-14:00: Afternoon pickup
            14: 1.0,   # 14:00-15:00: Normal
            15: 1.2,   # 15:00-16:00: Pre-close liquidity surge
            16: 0.4,   # 16:00+: After hours, very thin
        }

    def get_liquidity_multiplier(self, hour: float) -> float:
        """
        Get liquidity multiplier for given time.

        Args:
            hour: Hour in ET (9.5 = 9:30 AM)

        Returns:
            Liquidity multiplier (1.0 = normal)
        """
        # Find closest hour
        closest_hour = min(self.liquidity_curve.keys(),
                          key=lambda h: abs(h - hour))
        return self.liquidity_curve[closest_hour]

    def adjust_fill_expectations(self, base_fill_prob: float, hour: float) -> float:
        """Adjust fill probability based on time of day."""
        return min(1.0, base_fill_prob * self.get_liquidity_multiplier(hour))

    def is_good_entry_time(self, hour: float, min_liquidity: float = 0.7) -> bool:
        """Check if this is a good time for entries."""
        return self.get_liquidity_multiplier(hour) >= min_liquidity


# ==============================================================================
# EXECUTION REALITY ENGINE (UNIFIED)
# ==============================================================================

class ExecutionRealityEngine:
    """
    Unified execution reality modeling engine.

    Combines all 8 components into a single interface that preserves
    determinism when seeded.
    """

    def __init__(self,
                 seed: Optional[int] = None,
                 latency_mean_ms: float = 100.0,
                 aggression: float = 0.5,
                 base_spread: float = 0.02,
                 shock_threshold: float = 50.0):
        # Initialize all models
        self.latency = LatencyModel(mean_latency_ms=latency_mean_ms)
        self.queue = QueuePositionModel(aggression_factor=aggression)
        self.spread = SpreadDynamicsModel(base_spread_ratio=base_spread)
        self.shock = VolatilityShockModel(shock_threshold_annual=shock_threshold)
        self.broken = BrokenSpreadDetector()
        self.micro = MicrostructureModel()
        self.staleness = QuoteStalenessModel()
        self.tod = TimeOfDayLiquidityModel()

        self._seed = seed
        if seed is not None:
            self.reset(seed)

    def reset(self, seed: int):
        """Reset all random state for determinism."""
        self._seed = seed
        self.latency.seed(seed)
        self._rng = np.random.default_rng(seed)

    def simulate_realistic_fill(self,
                                bid: float,
                                ask: float,
                                size: int,
                                market_state: MarketState,
                                is_entry: bool = True) -> FillResult:
        """
        Simulate a realistic fill incorporating all reality factors.

        Args:
            bid: Current bid price
            ask: Current ask price
            size: Order size
            market_state: Current market conditions
            is_entry: True for entry, False for exit

        Returns:
            FillResult with all details
        """
        diagnostics = {}

        # 1. Check for broken quotes
        valid, reason = self.broken.is_valid_quote(bid, ask)
        if not valid:
            return FillResult(
                status=FillStatus.REJECTED,
                fill_price=0.0,
                fill_qty=0,
                fill_time_ms=0.0,
                slippage=0.0,
                diagnostics={'rejection_reason': reason}
            )
        diagnostics['quote_valid'] = True

        # 2. Check for volatility shock
        if self.shock.detect_shock(market_state.recent_price_change_pct, 60):
            shock_impact = self.shock.shock_impact(ask - bid)
            diagnostics['shock_detected'] = True
            if is_entry and shock_impact['should_avoid_entry']:
                return FillResult(
                    status=FillStatus.REJECTED,
                    fill_price=0.0,
                    fill_qty=0,
                    fill_time_ms=0.0,
                    slippage=0.0,
                    diagnostics={'rejection_reason': 'volatility_shock', **shock_impact}
                )

        # 3. Adjust for quote staleness
        reliability = self.staleness.quote_reliability(
            market_state.quote_age_ms,
            market_state.market_speed
        )
        adj_bid, adj_ask = self.staleness.adjust_for_staleness(bid, ask, reliability)
        diagnostics['quote_reliability'] = reliability

        # 4. Apply spread dynamics
        dynamic_spread_ratio = self.spread.dynamic_spread(
            (adj_ask - adj_bid) / adj_ask,
            market_state.vix,
            market_state.time_of_day_hour,
            market_state.recent_volume
        )
        diagnostics['spread_multiplier'] = dynamic_spread_ratio / ((ask - bid) / ask)

        # 5. Apply time-of-day liquidity
        liquidity_mult = self.tod.get_liquidity_multiplier(market_state.time_of_day_hour)
        diagnostics['liquidity_multiplier'] = liquidity_mult

        # 6. Simulate latency and drift
        latency_ms = self.latency.simulate_latency()
        iv_annual = market_state.vix / 100 * 1.5  # Rough IV estimate from VIX
        drift = self.latency.price_drift_during_latency(iv_annual, latency_ms)
        diagnostics['latency_ms'] = latency_ms
        diagnostics['drift_pct'] = drift * 100

        # 7. Calculate fill price
        mid = (adj_bid + adj_ask) / 2
        half_spread = (adj_ask - adj_bid) / 2

        if is_entry:
            # Entry: we sell, get hit on the bid side with slippage
            raw_fill = mid - half_spread * (1 - self.queue.aggression)
            raw_fill -= drift * market_state.spot_price  # Adverse drift
        else:
            # Exit: we buy, pay the ask side with slippage
            raw_fill = mid + half_spread * (1 - self.queue.aggression)
            raw_fill += drift * market_state.spot_price  # Adverse drift

        # 8. Round to tick
        fill_price = self.micro.round_to_tick(raw_fill, 'down' if is_entry else 'up')
        diagnostics['raw_fill'] = raw_fill
        diagnostics['tick_rounded'] = fill_price

        # 9. Calculate slippage
        theoretical_mid = (bid + ask) / 2
        slippage = abs(fill_price - theoretical_mid)
        diagnostics['slippage'] = slippage

        # 10. Determine fill probability
        base_fill_prob = self.queue.fill_probability(size, 100, 1000)
        adj_fill_prob = self.tod.adjust_fill_expectations(base_fill_prob,
                                                          market_state.time_of_day_hour)
        diagnostics['fill_probability'] = adj_fill_prob

        # Simulate whether we fill
        if self._rng.random() > adj_fill_prob:
            return FillResult(
                status=FillStatus.QUEUED,
                fill_price=fill_price,
                fill_qty=0,
                fill_time_ms=latency_ms,
                slippage=slippage,
                diagnostics={**diagnostics, 'queued_reason': 'fill_probability'}
            )

        return FillResult(
            status=FillStatus.FILLED,
            fill_price=fill_price,
            fill_qty=size,
            fill_time_ms=latency_ms,
            slippage=slippage,
            diagnostics=diagnostics
        )


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_default_engine(seed: int = 42) -> ExecutionRealityEngine:
    """Create engine with sensible defaults for retail execution."""
    return ExecutionRealityEngine(
        seed=seed,
        latency_mean_ms=100.0,
        aggression=0.5,
        base_spread=0.02,
        shock_threshold=50.0
    )


def create_institutional_engine(seed: int = 42) -> ExecutionRealityEngine:
    """Create engine modeling institutional execution quality."""
    return ExecutionRealityEngine(
        seed=seed,
        latency_mean_ms=25.0,
        aggression=0.7,
        base_spread=0.01,
        shock_threshold=75.0  # Higher threshold before avoiding
    )
