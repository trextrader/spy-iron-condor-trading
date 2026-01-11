"""
analytics/divergence.py

SPY–ES divergence computation and Z-score normalization.
Follows Quantor-MTFuzz specification Section 3.2 (Divergence).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class DivergenceZScore:
    """
    Computes Z-score of SPY-QQQ price ratio for relative strength divergence.
    
    Theory:
        spread = SPY_price / QQQ_price
        z = (spread - μ) / σ
        
    Signal:
        z > +2 → SPY overvalued vs QQQ (bearish SPY bias)
        z < -2 → SPY undervalued vs QQQ (bullish SPY bias)
    """

    def zscore(self, spread_series: pd.Series, lookback: int) -> float:
        """
        Compute Z-score of the most recent spread value.
        
        Parameters
        ----------
        spread_series : pd.Series or array-like
            Historical spread values (SPY / QQQ ratio).
        lookback : int
            Rolling window for mean/std calculation.

        Returns
        -------
        float
            Z-score of most recent spread value.
        """
        s = np.asarray(spread_series, dtype=float)
        
        if len(s) < lookback:
            return 0.0
            
        x = s[-lookback:]
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if lookback > 1 else float(np.std(x))
        
        if sd == 0.0:
            return 0.0
            
        return (float(x[-1]) - mu) / sd
    
    def compute_spread(self, spy_price: float, qqq_price: float) -> float:
        """
        Compute SPY/QQQ ratio.
        
        Parameters
        ----------
        spy_price : float
            SPY spot price.
        qqq_price : float
            QQQ spot price.

        Returns
        -------
        float
            Spread = SPY / QQQ
        """
        if qqq_price == 0:
            return 1.0
        return spy_price / qqq_price
