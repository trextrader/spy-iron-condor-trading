"""
risk/beta_weighting.py

Beta Weighting Module.
Calculates the Beta of an asset relative to SPY (the benchmark).
For SPY options, Beta is 1.0, but this infrastructure allows for multi-asset expansion.
"""
import pandas as pd
import numpy as np

class BetaCalculator:
    def __init__(self, benchmark_returns: pd.Series = None):
        self.benchmark_rets = benchmark_returns # SPY Returns

    def calculate_beta(self, asset_returns: pd.Series) -> float:
        """
        Calculate Beta = Cov(Asset, Bench) / Var(Bench)
        """
        if self.benchmark_rets is None or len(self.benchmark_rets) < 30:
            return 1.0 # Default fallback
            
        common_idx = asset_returns.index.intersection(self.benchmark_rets.index)
        if len(common_idx) < 30:
            return 1.0
            
        asset_slice = asset_returns.loc[common_idx]
        bench_slice = self.benchmark_rets.loc[common_idx]
        
        covariance = np.cov(asset_slice, bench_slice)[0][1]
        variance = np.var(bench_slice)
        
        if variance == 0:
            return 1.0
            
        return covariance / variance

    def beta_weight_delta(self, delta: float, beta: float) -> float:
        """
        Convert position delta to Beta-Weighted Delta (SPY-equivalent exposure).
        BWD = Delta * (Underlying Price / SPY Price) * Beta
        For SPY trading, Price Ratio and Beta are 1.0, so BWD = Delta.
        """
        return delta * beta
