# core/risk_manager.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
from datetime import date

@dataclass
class PortfolioGreeks:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    
    def __add__(self, other):
        return PortfolioGreeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta
        )

class RiskManager:
    def __init__(self, config):
        self.cfg = config
        self.current_greeks = PortfolioGreeks()
        self.daily_start_equity = 0.0
        self.current_drawdown_pct = 0.0
        
    def set_daily_start_equity(self, equity: float):
        self.daily_start_equity = equity

    def update_portfolio_greeks(self, positions: List[any]):
        """
        Recalculate total portfolio greeks from open positions.
        Assumes positions have 'legs' with 'delta', 'gamma', etc.
        If greeks are not on option objects, we might need a pricer here.
        For now, we sum the greeks stored/snapshot at entry or last update.
        """
        total = PortfolioGreeks()
        for pos in positions:
            # Assuming PositionState -> legs -> [short_call, etc.]
            # We need to iterate the 4 legs
            qty = pos.quantity
            legs = pos.legs
            
            # Short Call (Negative qty)
            total.delta += legs.short_call.delta * -1 * qty * 100
            total.gamma += getattr(legs.short_call, 'gamma', 0.0) * -1 * qty * 100
            total.vega += getattr(legs.short_call, 'vega', 0.0) * -1 * qty * 100
            total.theta += getattr(legs.short_call, 'theta', 0.0) * -1 * qty * 100

            # Short Put (Negative qty)
            total.delta += legs.short_put.delta * -1 * qty * 100
            total.gamma += getattr(legs.short_put, 'gamma', 0.0) * -1 * qty * 100
            total.vega += getattr(legs.short_put, 'vega', 0.0) * -1 * qty * 100
            total.theta += getattr(legs.short_put, 'theta', 0.0) * -1 * qty * 100

            # Long Call (Positive qty)
            total.delta += legs.long_call.delta * 1 * qty * 100
            total.gamma += getattr(legs.long_call, 'gamma', 0.0) * 1 * qty * 100
            total.vega += getattr(legs.long_call, 'vega', 0.0) * 1 * qty * 100
            total.theta += getattr(legs.long_call, 'theta', 0.0) * 1 * qty * 100

            # Long Put (Positive qty)
            total.delta += legs.long_put.delta * 1 * qty * 100
            total.gamma += getattr(legs.long_put, 'gamma', 0.0) * 1 * qty * 100
            total.vega += getattr(legs.long_put, 'vega', 0.0) * 1 * qty * 100
            total.theta += getattr(legs.long_put, 'theta', 0.0) * 1 * qty * 100
            
        self.current_greeks = total
        return total

    def calculate_trade_greeks(self, legs, quantity: int) -> PortfolioGreeks:
        """Calculate greeks for a prospective trade (Iron Condor)"""
        g = PortfolioGreeks()
        # Shorts
        g.delta += (legs.short_call.delta + legs.short_put.delta) * -1 * quantity * 100
        g.gamma += (getattr(legs.short_call, 'gamma', 0.0) + getattr(legs.short_put, 'gamma', 0.0)) * -1 * quantity * 100
        g.vega += (getattr(legs.short_call, 'vega', 0.0) + getattr(legs.short_put, 'vega', 0.0)) * -1 * quantity * 100
        
        # Longs
        g.delta += (legs.long_call.delta + legs.long_put.delta) * 1 * quantity * 100
        g.gamma += (getattr(legs.long_call, 'gamma', 0.0) + getattr(legs.long_put, 'gamma', 0.0)) * 1 * quantity * 100
        g.vega += (getattr(legs.long_call, 'vega', 0.0) + getattr(legs.long_put, 'vega', 0.0)) * 1 * quantity * 100
        
        return g

    def check_drawdown_stop(self, current_equity: float, today: date) -> Tuple[bool, str]:
        """
        Check daily drawdown limit.
        Returns: (is_halted, message)
        """
        # Initialize on first run
        if self.daily_start_equity == 0.0:
            self.daily_start_equity = current_equity
            
        # Reset tracker on new day if we have date tracking
        # (This implies we need to track separate dates, simpler to just rely on caller to set_start on new day
        # or we track 'last_check_date' state)
        # Let's add last_check_date to state
        if not hasattr(self, 'last_check_date'):
            self.last_check_date = None
            
        if self.last_check_date != today:
            self.daily_start_equity = current_equity
            self.last_check_date = today
            self.current_drawdown_pct = 0.0
            return False, "New Day"
            
        if self.daily_start_equity <= 0:
            return False, "No Equity"
            
        dd_amt = self.daily_start_equity - current_equity
        self.current_drawdown_pct = dd_amt / self.daily_start_equity
        
        limit = getattr(self.cfg, 'max_daily_drawdown_pct', 0.02)
        if self.current_drawdown_pct > limit:
            return True, f"DAILY STOP: Drawdown {self.current_drawdown_pct*100:.2f}% > Limit {limit*100:.1f}%"
            
        return False, "OK"

    def check_new_trade(self, legs, quantity: int, current_equity: float, existing_greeks: Optional[PortfolioGreeks] = None) -> Tuple[bool, str]:
        """
        Validate a new trade against risk limits.
        """
        # 1. Drawdown Check (using current state)
        limit = getattr(self.cfg, 'max_daily_drawdown_pct', 0.02)
        if self.current_drawdown_pct > limit:
            return False, f"Daily Drawdown {self.current_drawdown_pct:.1%} > Limit"

        trade_greeks = self.calculate_trade_greeks(legs, quantity)
        
        # 2. Per-Trade Limits
        if abs(trade_greeks.delta) > getattr(self.cfg, 'max_delta_per_trade', 30.0):
            return False, f"Trade Delta {trade_greeks.delta:.1f} > Limit"
            
        if abs(trade_greeks.vega) > getattr(self.cfg, 'max_vega_per_trade', 100.0):
            return False, f"Trade Vega {trade_greeks.vega:.1f} > Limit"

        # 3. Portfolio Limits (Post-Trade)
        base_greeks = existing_greeks if existing_greeks is not None else self.current_greeks
        projected = base_greeks + trade_greeks
        
        if abs(projected.delta) > getattr(self.cfg, 'max_portfolio_delta', 200.0):
            return False, f"Projected Portfolio Delta {projected.delta:.1f} > Limit"
            
        if abs(projected.gamma) > getattr(self.cfg, 'max_portfolio_gamma', 50.0):
            return False, f"Projected Portfolio Gamma {projected.gamma:.1f} > Limit"
            
        if abs(projected.vega) > getattr(self.cfg, 'max_portfolio_vega', 500.0):
            return False, f"Projected Portfolio Vega {projected.vega:.1f} > Limit"
            
        return True, "OK"
