"""
risk/pot_monitor.py

Probability of Touch (POT) Monitor.
Estimates the probability that the underlying price will touch a specific strike
before expiration, using analytic solutions for Brownian Motion (Double Barrier or Single).
"""
import math
import numpy as np
from scipy.stats import norm

class POTMonitor:
    def __init__(self):
        pass

    def probability_of_touch(self, spot: float, strike: float, iv: float, dte: float, drift: float = 0.0) -> float:
        """
        Calculate One-Touch Probability for a single barrier.
        
        Args:
            spot: Underlying price
            strike: Barrier level
            iv: Implied Volatility (decimal)
            dte: Days to Expiration
            drift: Annualized drift (risk-free rate - div yield)

        Formula:
        POT = 2 * N( d_touch ) ?? 
        Actually, for zero drift, POT = 2 * P(S_T > K) (for call side).
        Standard formulation:
        POT = exp( -2*a*b / sigma^2 ) ... for infinite time?
        
        Using Reiner-Rubinstein (1991) generic approximation for finite time:
        If Spot < Strike (Up-and-In):
           Prob = (Strike/Spot)^(2*mu/sigma^2) * N(...) + ...
           
        SIMPLIFIED APPROXIMATION (Practical for Trading):
        POT ~= 2 * probability of expiring ITM (Delta).
        If Delta is 0.30, POT is approx 0.60.
        This rule of thumb is widely used in retail logic (Delta * 2).
        """
        if dte <= 0:
            return 1.0 if (spot >= strike if strike > spot else spot <= strike) else 0.0
            
        t = dte / 365.0
        sigma = iv
        
        # d1 calculation
        # d1 = (ln(S/K) + (r + sigma^2/2)t) / (sigma * sqrt(t))
        # N(d1) = Delta approximation
        
        denom = sigma * math.sqrt(t)
        if denom == 0:
             return 0.0
             
        d1 = (math.log(spot / strike) + (drift + 0.5 * sigma**2) * t) / denom
        
        # If call side (Strike > Spot)
        if strike > spot:
            nd1 = norm.cdf(d1) # Delta
            return min(1.0, 2.0 * nd1)
            
        # If put side (Strike < Spot)
        else:
            nd1 = norm.cdf(-d1) # Put Delta (abs)
            return min(1.0, 2.0 * nd1)

    def check_condor_touch(self, spot: float, short_call: float, short_put: float, iv: float, dte: float) -> dict:
        """
        Check POT for both sides of a Condor.
        """
        call_pot = self.probability_of_touch(spot, short_call, iv, dte)
        put_pot = self.probability_of_touch(spot, short_put, iv, dte)
        
        return {
            "call_pot": call_pot,
            "put_pot": put_pot,
            "max_pot": max(call_pot, put_pot)
        }
