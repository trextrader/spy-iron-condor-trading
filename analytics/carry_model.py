"""
analytics/carry_model.py

Cost-of-carry fair value model for futures basis.
Follows Quantor-MTFuzz specification Section 3.1 (Carry).
"""

from __future__ import annotations
import math


class CostOfCarry:
    """
    Cost-of-carry fair value for futures pricing.
    
    Theory:
        F = S * exp((r - q) * τ)
        
    Where:
        F = Futures fair value
        S = Spot price
        r = Risk-free rate
        q = Dividend yield
        τ = Time to expiration (years)
    """

    def fair_value(self, spot: float, r: float, q: float, tau_years: float) -> float:
        """
        Compute F = S * exp((r - q) * tau).
        
        Parameters
        ----------
        spot : float
            Spot price (e.g., SPY).
        r : float
            Risk-free rate (annualized, e.g., 0.05 = 5%).
        q : float
            Dividend yield (annualized, e.g., 0.015 = 1.5%).
        tau_years : float
            Time to expiration in years.

        Returns
        -------
        float
            Futures fair value.
        """
        return spot * math.exp((r - q) * tau_years)
    
    def basis(self, futures_price: float, spot: float) -> float:
        """
        Compute futures basis = F - S.
        
        Parameters
        ----------
        futures_price : float
            Observed futures price.
        spot : float
            Spot price.

        Returns
        -------
        float
            Basis (positive = contango, negative = backwardation).
        """
        return futures_price - spot
