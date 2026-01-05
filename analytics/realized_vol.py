"""
analytics/realized_vol.py

Realized volatility and variance estimators from intraday bars.
Follows Quantor-MTFuzz specification Section 2.1 (VRP).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class RealizedVolCalculator:
    """
    Computes realized volatility using rolling sums of squared log returns.
    
    Theory:
        RV² = (252/N) * Σ(r_t²) where r_t = ln(P_t / P_{t-1})
        
    This is the foundation for Volatility Risk Premium (VRP) calculation:
        VRP = IV - RV
    """

    def compute_realized_variance(self, close_prices: pd.Series, window: int) -> float:
        """
        Compute annualized realized variance over `window` observations.

        Parameters
        ----------
        close_prices : pd.Series or array-like
            Close prices of bars (time-ordered).
        window : int
            Rolling window length.

        Returns
        -------
        float
            Annualized realized variance.
            
        Notes
        -----
        Uses 252 trading days for annualization.
        """
        if len(close_prices) < window + 1:
            return 0.0
            
        px = np.asarray(close_prices, dtype=float)
        r = np.log(px[1:] / px[:-1])
        r2 = r[-window:] ** 2
        
        return (252.0 / window) * float(np.sum(r2))

    def compute_realized_vol(self, close_prices: pd.Series, window: int) -> float:
        """
        Return annualized realized volatility (sqrt of variance).
        
        Parameters
        ----------
        close_prices : pd.Series or array-like
            Close prices of bars (time-ordered).
        window : int
            Rolling window length.

        Returns
        -------
        float
            Annualized realized volatility (sigma).
        """
        rv2 = self.compute_realized_variance(close_prices, window)
        return float(np.sqrt(rv2))
