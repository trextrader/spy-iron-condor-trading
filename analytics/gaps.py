"""
analytics/gaps.py

Gap sizing and classification for gap-fill module.
Follows Quantor-MTFuzz specification Section 3.3 (Gaps).
"""

from __future__ import annotations


class GapAnalyzer:
    """
    Gap classification for overnight mean-reversion edge.
    
    Theory:
        Small gaps (< 0.19%) tend to fill intraday.
        Large gaps indicate momentum continuation.
    """

    def gap_pct(self, open_price: float, prev_close: float) -> float:
        """
        Return absolute gap percentage = |open - prev_close| / prev_close.
        
        Parameters
        ----------
        open_price : float
            Today's open price.
        prev_close : float
            Previous day's close price.

        Returns
        -------
        float
            Absolute gap percentage (e.g., 0.0019 = 0.19%).
        """
        if prev_close == 0:
            return 0.0
        return abs(open_price - prev_close) / prev_close

    def is_small_gap(self, open_price: float, prev_close: float, threshold: float = 0.0019) -> bool:
        """
        Check if gap is "small" (likely to fill).
        
        Parameters
        ----------
        open_price : float
            Today's open price.
        prev_close : float
            Previous day's close price.
        threshold : float, optional
            Gap threshold (default 0.0019 = 0.19%).

        Returns
        -------
        bool
            True if gap <= threshold (small gap).
        """
        return self.gap_pct(open_price, prev_close) <= threshold
    
    def gap_direction(self, open_price: float, prev_close: float) -> str:
        """
        Determine gap direction.
        
        Parameters
        ----------
        open_price : float
            Today's open price.
        prev_close : float
            Previous day's close price.

        Returns
        -------
        str
            "gap_up" | "gap_down" | "no_gap"
        """
        if open_price > prev_close * 1.0001:  # > 0.01% threshold
            return "gap_up"
        elif open_price < prev_close * 0.9999:  # < -0.01% threshold
            return "gap_down"
        else:
            return "no_gap"
