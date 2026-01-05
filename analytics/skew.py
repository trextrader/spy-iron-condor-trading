"""
analytics/skew.py

Implied volatility skew estimators for penalty logic.
Follows Quantor-MTFuzz specification Section 4.2 (Skew Penalty).
"""

from __future__ import annotations


class SkewCalculator:
    """
    IV skew metrics for crash-risk suppression.
    
    Theory:
        Skew = (IV_put - IV_call) / IV_atm
        
    High skew (> 0.15) indicates expensive downside protection,
    suggesting crash risk. Strategy should widen put wings or reduce size.
    """

    def skew_metric(self, iv_put: float, iv_call: float, iv_atm: float) -> float:
        """
        Compute (IV_put - IV_call) / IV_atm with safe handling.
        
        Parameters
        ----------
        iv_put : float
            Implied volatility of OTM put.
        iv_call : float
            Implied volatility of OTM call.
        iv_atm : float
            Implied volatility of ATM option.

        Returns
        -------
        float
            Skew metric (positive = put skew, negative = call skew).
        """
        if iv_atm == 0:
            return 0.0
        return (iv_put - iv_call) / iv_atm
    
    def is_steep_skew(self, skew: float, threshold: float = 0.15) -> bool:
        """
        Check if skew exceeds threshold (crash risk indicator).
        
        Parameters
        ----------
        skew : float
            Skew metric from skew_metric().
        threshold : float, optional
            Skew threshold (default 0.15).

        Returns
        -------
        bool
            True if skew > threshold (steep put skew).
        """
        return skew > threshold
