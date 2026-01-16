"""
Composite Risk-Aligned Loss Function for CondorBrain

This module provides a multi-objective loss function that directly aligns
model optimization with trading objectives:

1. Predictive Fidelity (Huber): Accurate prediction of IC parameters
2. Sharpe Proxy: Maximize risk-adjusted returns
3. Soft Drawdown Penalty: Penalize strategies with deep drawdowns
4. Turnover Penalty: Discourage excessive position changes

Mathematical Foundation:
    L_total = λ₁·L_pred + λ₂·L_sharpe + λ₃·L_dd + λ₄·L_turn

Where:
    L_pred = HuberLoss(y_pred, y_true)
    L_sharpe = -μ(returns) / (σ(returns) + ε)
    L_dd = log(Σ exp((peak - cumsum) / τ))
    L_turn = mean(|weights_t - weights_{t-1}|)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class CompositeCondorLoss(nn.Module):
    """
    Multi-objective loss function for risk-aligned Iron Condor optimization.
    
    Combines prediction accuracy with trading-specific objectives:
    - Sharpe ratio maximization
    - Drawdown minimization
    - Turnover control
    
    Args:
        lambdas: Weight tuple (pred, sharpe, drawdown, turnover)
        huber_delta: Delta parameter for Huber loss
        dd_tau: Temperature for soft drawdown (higher = smoother)
    """
    
    def __init__(
        self,
        lambdas: Tuple[float, float, float, float] = (1.0, 0.5, 0.1, 0.1),
        huber_delta: float = 1.0,
        dd_tau: float = 0.02
    ):
        super().__init__()
        self.lambdas = lambdas
        self.dd_tau = dd_tau
        self.huber = nn.HuberLoss(delta=huber_delta, reduction='mean')
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        last_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute composite loss.
        
        Args:
            y_pred: Model predictions (B, 8) for IC parameters
            y_true: Ground truth targets (B, 8)
            returns: Realized returns (B,) or (B, T) for Sharpe/DD computation
            weights: Current position weights (B, K) for turnover
            last_weights: Previous position weights (B, K) for turnover
            
        Returns:
            Scalar loss tensor
        """
        device = y_pred.device
        
        # 1. Predictive Fidelity (always computed)
        l_pred = self.huber(y_pred, y_true)
        
        # Initialize optional loss components
        l_sharpe = torch.tensor(0.0, device=device)
        l_dd = torch.tensor(0.0, device=device)
        l_turn = torch.tensor(0.0, device=device)
        
        # 2. Sharpe Proxy (if returns provided)
        if returns is not None and self.lambdas[1] > 0:
            returns_flat = returns.flatten()
            if returns_flat.numel() > 1:
                mu = returns_flat.mean()
                sigma = returns_flat.std() + 1e-9
                l_sharpe = -mu / sigma  # Negative because we minimize
        
        # 3. Soft Drawdown Penalty (if returns provided)
        if returns is not None and self.lambdas[2] > 0:
            returns_flat = returns.flatten()
            if returns_flat.numel() > 1:
                cum_ret = returns_flat.cumsum(dim=0)
                peak = torch.cummax(cum_ret, dim=0).values
                drawdown = peak - cum_ret
                # Log-sum-exp gives smooth max approximation
                l_dd = torch.logsumexp(drawdown / self.dd_tau, dim=0)
        
        # 4. Turnover Penalty (if weights provided)
        if weights is not None and last_weights is not None and self.lambdas[3] > 0:
            l_turn = (weights - last_weights).abs().mean()
        
        # Weighted combination
        total = (
            self.lambdas[0] * l_pred +
            self.lambdas[1] * l_sharpe +
            self.lambdas[2] * l_dd +
            self.lambdas[3] * l_turn
        )
        
        return total
    
    def forward_decomposed(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        last_weights: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute composite loss with individual component values for logging.
        
        Returns:
            Dict with 'total', 'pred', 'sharpe', 'drawdown', 'turnover' keys
        """
        device = y_pred.device
        
        l_pred = self.huber(y_pred, y_true)
        
        l_sharpe = torch.tensor(0.0, device=device)
        l_dd = torch.tensor(0.0, device=device)
        l_turn = torch.tensor(0.0, device=device)
        
        if returns is not None and self.lambdas[1] > 0:
            returns_flat = returns.flatten()
            if returns_flat.numel() > 1:
                mu = returns_flat.mean()
                sigma = returns_flat.std() + 1e-9
                l_sharpe = -mu / sigma
        
        if returns is not None and self.lambdas[2] > 0:
            returns_flat = returns.flatten()
            if returns_flat.numel() > 1:
                cum_ret = returns_flat.cumsum(dim=0)
                peak = torch.cummax(cum_ret, dim=0).values
                drawdown = peak - cum_ret
                l_dd = torch.logsumexp(drawdown / self.dd_tau, dim=0)
        
        if weights is not None and last_weights is not None and self.lambdas[3] > 0:
            l_turn = (weights - last_weights).abs().mean()
        
        total = (
            self.lambdas[0] * l_pred +
            self.lambdas[1] * l_sharpe +
            self.lambdas[2] * l_dd +
            self.lambdas[3] * l_turn
        )
        
        return {
            'total': total,
            'pred': l_pred.detach(),
            'sharpe': l_sharpe.detach(),
            'drawdown': l_dd.detach(),
            'turnover': l_turn.detach()
        }
