
# strategies/risk_manager.py
from dataclasses import dataclass
from typing import List, Tuple, Dict
import datetime as dt

# Import necessary types based on existing code structure
# Assuming PositionState and IronCondorLegs are in strategies.options_strategy
from core.dto import IronCondorLegs
from strategies.options_strategy import PositionState, net_position_delta

@dataclass
class PortfolioGreeks:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    
class RiskManager:
    """
    Stage 3 Risk Manager
    Enforces portfolio-level and trade-level risk limits (Greeks, Drawdown).
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.daily_start_equity = 0.0
        self.current_drawdown_pct = 0.0
        self.last_check_date = None

    def calculate_portfolio_greeks(self, positions: List[PositionState], 
                                   current_prices: Dict[str, IronCondorLegs] = None) -> PortfolioGreeks:
        """
        Aggregate Greeks across all open positions.
        """
        total = PortfolioGreeks()
        
        for pos in positions:
            # Use current legs if provided (market to market), else use stored legs (approx)
            legs = pos.legs
            
            # Simple aggregation (Strategy assumes IC legs have greek data)
            # Net Greek = (Long - Short) * Qty * 100 (for option multiplier)
            # Iron Condor: Short Call, Long Call, Short Put, Long Put
            # Signs: Long (+), Short (-)
            
            q = pos.quantity * 100 # Standard multiplier
            
            # Sum greeks for this position
            pos_delta = (legs.long_call.delta - legs.short_call.delta + 
                         legs.long_put.delta - legs.short_put.delta) * q
                         
            # Note: Gamma/Vega/Theta might be 0.0 in backtest if not loaded, 
            # but we implement logic for when they are available.
            pos_gamma = (legs.long_call.gamma - legs.short_call.gamma + 
                         legs.long_put.gamma - legs.short_put.gamma) * q
                         
            pos_vega = (legs.long_call.vega - legs.short_call.vega + 
                        legs.long_put.vega - legs.short_put.vega) * q
                        
            pos_theta = (legs.long_call.theta - legs.short_call.theta + 
                         legs.long_put.theta - legs.short_put.theta) * q
            
            total.delta += pos_delta
            total.gamma += pos_gamma
            total.vega += pos_vega
            total.theta += pos_theta
            
        return total

    def check_new_trade_risk(self, proposed_legs: IronCondorLegs, quantity: int, 
                             current_portfolio_greeks: PortfolioGreeks) -> Tuple[bool, str]:
        """
        Validate a SINGLE new trade against per-trade and portfolio limits.
        """
        q100 = quantity * 100
        
        # 1. Trade Limit Check
        trade_delta = (proposed_legs.long_call.delta - proposed_legs.short_call.delta + 
                       proposed_legs.long_put.delta - proposed_legs.short_put.delta) * q100
                       
        trade_vega = (proposed_legs.long_call.vega - proposed_legs.short_call.vega + 
                      proposed_legs.long_put.vega - proposed_legs.short_put.vega) * q100
                      
        if abs(trade_delta) > self.cfg.max_delta_per_trade:
            return False, f"Risk Reject: Trade Delta {trade_delta:.2f} > Limit {self.cfg.max_delta_per_trade}"
            
        if abs(trade_vega) > self.cfg.max_vega_per_trade:
            return False, f"Risk Reject: Trade Vega {trade_vega:.2f} > Limit {self.cfg.max_vega_per_trade}"
            
        # 2. Portfolio Impact Check
        new_port_delta = current_portfolio_greeks.delta + trade_delta
        new_port_gamma = current_portfolio_greeks.gamma + ((proposed_legs.long_call.gamma - proposed_legs.short_call.gamma + proposed_legs.long_put.gamma - proposed_legs.short_put.gamma) * q100)
        new_port_vega = current_portfolio_greeks.vega + trade_vega
        
        if abs(new_port_delta) > self.cfg.max_portfolio_delta:
             return False, f"Risk Reject: Portfolio Delta {new_port_delta:.2f} > Limit {self.cfg.max_portfolio_delta}"
             
        if abs(new_port_gamma) > self.cfg.max_portfolio_gamma:
             return False, f"Risk Reject: Portfolio Gamma {new_port_gamma:.2f} > Limit {self.cfg.max_portfolio_gamma}"

        if abs(new_port_vega) > self.cfg.max_portfolio_vega:
             return False, f"Risk Reject: Portfolio Vega {new_port_vega:.2f} > Limit {self.cfg.max_portfolio_vega}"
             
        return True, "OK"

    def check_drawdown_stop(self, current_equity: float, today: dt.date) -> Tuple[bool, str]:
        """
        Check daily drawdown limit.
        Returns: (is_halted, message)
        """
        # Reset tracker on new day
        if self.last_check_date != today:
            self.daily_start_equity = current_equity
            self.last_check_date = today
            return False, "New Day"
            
        if self.daily_start_equity <= 0:
            return False, "No Equity"
            
        dd_amt = self.daily_start_equity - current_equity
        dd_pct = dd_amt / self.daily_start_equity
        
        if dd_pct > self.cfg.max_daily_drawdown_pct:
            return True, f"DAILY STOP: Drawdown {dd_pct*100:.2f}% > Limit {self.cfg.max_daily_drawdown_pct*100:.1f}%"
            
        return False, "OK"
