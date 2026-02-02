"""
Strategy Suite for CondorBrain v3.0

This module provides simulation logic for alternative option strategies:
1. Calendar Spreads (Vega/Theta play)
2. Broken Wing Butterflies (Skew/Directional play)
3. Vertical Spreads (Directional play)

It uses Black-Scholes approximation for undefined-risk exits (like Calendar back-legs).
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import norm

# ============================================================================
# BLACK-SCHOLES PRICING ENGINE (Approximate)
# ============================================================================

class BlackScholes:
    """Vectorized Black-Scholes pricer for estimating option values."""
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Implied Volatility
        """
        # Avoid division by zero
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-3)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, T, r, sigma):
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-3)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ============================================================================
# STRATEGY SIMULATORS
# ============================================================================

class CalendarSimulator:
    """
    Simulates a Long Calendar Spread (Short Front / Long Back).
    Profit = Value(Long, T_front) - Intrinsic(Short, T_front) - Debit
    """
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    def simulate(self, entry_spot, exit_spot, entry_iv, exit_iv, strike, front_dte, back_dte, entry_debit=None):
        """
        Simulate outcome at Front Month Expiration.
        
        Args:
            entry_spot: Spot at entry
            exit_spot: Spot at front month expiry
            entry_iv: IV at entry
            exit_iv: IV at front month expiry (critical for profit!)
            strike: Strike price
            front_dte: Days to Expiration of front leg
            back_dte: Days to Expiration of back leg
            entry_debit: Cost to enter (estimated via BS if None)
        """
        T_front = front_dte / 365.0
        T_back = back_dte / 365.0
        dt_exit = T_front # Time elapsed
        T_remaining_back = T_back - dt_exit
        
        # 1. Estimate Entry Debit if not provided
        if entry_debit is None:
            # Price both legs at entry
            # Assuming Call Calendar for simplicity (Puts similar for ATM)
            short_opt = BlackScholes.call_price(entry_spot, strike, T_front, self.r, entry_iv)
            long_opt = BlackScholes.call_price(entry_spot, strike, T_back, self.r, entry_iv)
            entry_debit = long_opt - short_opt
            
        # 2. Calculate Exit Value
        # Short leg assumes held to expiry (Cash settlement of intrinsic value)
        short_value_exit = max(0, exit_spot - strike)
        
        # Long leg value at exit (priced using Exit Spot and Exit IV)
        long_value_exit = BlackScholes.call_price(exit_spot, strike, T_remaining_back, self.r, exit_iv)
        
        exit_credit = long_value_exit - short_value_exit
        
        # 3. Metrics
        pnl = exit_credit - entry_debit
        roi = pnl / entry_debit if entry_debit > 0 else 0.0
        
        return {
            'pnl': pnl,
            'roi': roi,
            'entry_debit': entry_debit,
            'exit_credit': exit_credit,
            'exit_iv': exit_iv
        }

class VerticalSimulator:
    """
    Simulates Credit Vertical Spreads (Bear Call / Bull Put).
    Defined Risk, Defined Reward.
    """
    def simulate(self, entry_spot, exit_spot, short_strike, long_strike, is_put=True, credit=0.50):
        wing_width = abs(short_strike - long_strike)
        max_loss = wing_width - credit
        
        # Check breaches
        if is_put:
            # Bull Put: Loss if Price < Short
            # Max Loss if Price < Long
            if exit_spot >= short_strike:
                pnl = credit # Full Profit
            elif exit_spot <= long_strike:
                pnl = -max_loss # Max Loss
            else:
                # Partial Loss
                intrinsic = short_strike - exit_spot
                pnl = credit - intrinsic
        else:
            # Bear Call: Loss if Price > Short
            if exit_spot <= short_strike:
                pnl = credit
            elif exit_spot >= long_strike:
                pnl = -max_loss
            else:
                intrinsic = exit_spot - short_strike
                pnl = credit - intrinsic
                
        roi = pnl / max_loss if max_loss > 0 else 0.0
        return {'pnl': pnl, 'roi': roi, 'max_loss': max_loss}

class BWBSimulator:
    """
    Simulates Broken Wing Butterfly (Put side standard).
    Buy 1 ITM (Higher Strike), Sell 2 ATM, Buy 1 OTM (Lower Strike, Skipped).
    Structure: Body is ATM. Upper Wing is +Width. Lower Wing is -Width*2 (Broken).
    This creates a "Free" upside trade usually, but adds risk to the downside.
    """
    def simulate(self, entry_spot, exit_spot, body_strike, width, skip_factor=2.0, net_credit=0.0):
        # Strikes (Put Butterfly)
        # Upper (Long ITM) > Body (Short ATM) > Lower (Long OTM)
        upper_strike = body_strike + width
        lower_strike = body_strike - (width * skip_factor) # Broken wing
        
        # PnL at expiry
        # 1. Long Upper Put
        val_upper = max(0, upper_strike - exit_spot)
        # 2. Short 2 Body Puts
        val_body = 2 * max(0, body_strike - exit_spot)
        # 3. Long Lower Put
        val_lower = max(0, lower_strike - exit_spot)
        
        # Net value at expiry (This is the DEBIT/CREDIT at exit, so positive is good for Long Fly)
        # Wait, normally we ENTER for a small credit or debit.
        # Let's assume we ENTER for net_credit. 
        # Exit Value = (val_upper + val_lower) - val_body
        # Total PnL = Exit_Value + net_credit (if credit) or Exit_Value - net_debit
        
        exit_value = val_upper + val_lower - val_body
        
        pnl = exit_value + net_credit
        
        # Risk is undefined? No, defined but "Broken".
        # Max Risk is usually on the downside below lower strike.
        # Max Risk = (Skip_Width - Width) - Credit.
        # E.g. Width 5, Skip 10. Risk = (10-5) - Credit = 5 - Credit.
        risk_width = (body_strike - lower_strike) - width
        max_risk = risk_width - net_credit
        
        roi = pnl / max_risk if max_risk > 0 else 0.0
        
        return {'pnl': pnl, 'roi': roi, 'max_risk': max_risk}
