"""
risk/expected_shortfall.py

Conditional Value at Risk (CVaR) / Expected Shortfall Calculator.
Implements Historical Simulation for tail risk estimation.
"""
import numpy as np
import pandas as pd
from typing import List

class ExpectedShortfall:
    def __init__(self, confidence_level: float = 0.95, lookback_window: int = 252):
        self.alpha = 1.0 - confidence_level # Tail probability (e.g., 0.05)
        self.lookback = lookback_window

    def calculate_cvar(self, returns: pd.Series) -> float:
        """
        Calculate Expected Shortfall (Average of losses exceeding VaR).
        Input: pd.Series of pct_change (fractional returns)
        Returns: Positive float representing expected loss % (e.g. 0.03 for 3%)
        """
        if len(returns) < 50:
            return 0.0 # Insufficient history
        
        # Focus on the tail (losses)
        # Sort returns ascending (losses are negative)
        sorted_rets = returns.sort_values(ascending=True)
        
        # Calculate VaR index
        var_index = int(len(sorted_rets) * self.alpha)
        if var_index == 0:
            var_index = 1
            
        # Tail slice
        tail_losses = sorted_rets.iloc[:var_index]
        
        # CVaR is the mean of the tail
        cvar = tail_losses.mean()
        
        # Return as positive percentage loss
        return abs(cvar)

    def estimate_portfolio_cvar(self, portfolio_value: float, benchmark_returns: pd.Series, beta: float) -> float:
        """
        Estimate Portfolio $ CVaR using Beta-adjusted Benchmark Returns.
        Simplified approach: Portfolio Returns ~ Beta * Benchmark Returns
        """
        adj_returns = benchmark_returns * beta
        cvar_pct = self.calculate_cvar(adj_returns)
        return portfolio_value * cvar_pct
