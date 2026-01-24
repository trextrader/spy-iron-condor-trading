"""
Composite Risk-Aligned Loss Function for CondorBrain

This module provides a multi-objective loss function that directly aligns
model optimization with trading objectives:

1. Predictive Fidelity (Huber): Accurate prediction of IC parameters
2. Sharpe Proxy: Maximize risk-adjusted returns
3. Soft Drawdown Penalty: Penalize strategies with deep drawdowns
4. Turnover Penalty: Discourage excessive position changes
5. Rule Consistency Penalty: Penalize logic that violates institutional safety rules

Mathematical Foundation:
    L_total = λ₁·L_pred + λ₂·L_sharpe + λ₃·L_dd + λ₄·L_turn + λ₅·L_rule

Where:
    L_pred = HuberLoss(y_pred, y_true)
    L_sharpe = -μ(returns) / (σ(returns) + ε)
    L_dd = log(Σ exp((peak - cumsum) / τ))
    L_turn = mean(|weights_t - weights_{t-1}|)
    L_rule = L_block + 0.5 * L_consensus
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
    - Rule consistency matching
    
    Args:
        lambdas: Weight tuple (pred, sharpe, drawdown, turnover, rule)
        huber_delta: Delta parameter for Huber loss
        dd_tau: Temperature for soft drawdown (higher = smoother)
    """
    
    def __init__(
        self,
        # (pred, sharpe, drawdown, turnover, rule_consistency)
        lambdas: Tuple[float, float, float, float, float] = (1.0, 0.5, 0.1, 0.1, 1.0),
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
        last_weights: Optional[torch.Tensor] = None,
        rule_signals: Optional[torch.Tensor] = None # (B, 4) [long, short, exit, block]
    ) -> torch.Tensor:
        """
        Compute composite loss.
        
        Args:
            y_pred: Model predictions (B, 8) for IC parameters
            y_true: Ground truth targets (B, 8)
            returns: Realized returns (B,) or (B, T) for Sharpe/DD computation
            weights: Current position weights (B, K) for turnover
            last_weights: Previous position weights (B, K) for turnover
            rule_signals: Institutional rule signals for consistency (long, short, exit, block)
            
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
        l_rule = torch.tensor(0.0, device=device)
        
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

        # 5. Rule Consistency Loss (if rule_signals provided)
        # rule_signals indices: 0=long, 1=short, 2=exit, 3=block
        # pred index 7 is 'confidence'
        if rule_signals is not None and len(self.lambdas) > 4 and self.lambdas[4] > 0:
            # SAFETY: Clamp rule signals to [0, 1] to prevent explosion if normalization drifts
            rule_signals = torch.clamp(rule_signals, 0.0, 1.0)
            
            # Rule block flag (B,)
            block_flag = rule_signals[:, 3]
            # Model's predicted confidence (B,) - Index 7 in CondorExpertHead
            confidence = y_pred[:, 7]
            
            # A. Block Penalty: If rule says block, model confidence should be 0
            # Higher confidence in blocked state = Higher loss
            l_block = (block_flag * confidence).mean()
            
            # B. Consensus Penalty: Penalize mismatch between model confidence and rule long/short
            # If both long and short consensus are low (market consolidation), model should be low confidence
            rule_strength = torch.max(rule_signals[:, 0], rule_signals[:, 1])
            l_consensus = ((1.0 - rule_strength) * confidence).mean()
            
            l_rule = l_block + 0.5 * l_consensus
        
        # Weighted combination
        total = (
            self.lambdas[0] * l_pred +
            self.lambdas[1] * l_sharpe +
            self.lambdas[2] * l_dd +
            self.lambdas[3] * l_turn +
            (self.lambdas[4] * l_rule if len(self.lambdas) > 4 else 0.0)
        )
        
        return total
    
    def forward_decomposed(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        last_weights: Optional[torch.Tensor] = None,
        rule_signals: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute composite loss with individual component values for logging.
        
        Returns:
            Dict with 'total', 'pred', 'sharpe', 'drawdown', 'turnover', 'rule' keys
        """
        device = y_pred.device
        
        l_pred = self.huber(y_pred, y_true)
        
        l_sharpe = torch.tensor(0.0, device=device)
        l_dd = torch.tensor(0.0, device=device)
        l_turn = torch.tensor(0.0, device=device)
        l_rule = torch.tensor(0.0, device=device)
        
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

        if rule_signals is not None and len(self.lambdas) > 4 and self.lambdas[4] > 0:
            # SAFETY: Clamp rule signals to [0, 1] for logging consistency
            rule_signals = torch.clamp(rule_signals, 0.0, 1.0)
            
            block_flag = rule_signals[:, 3]
            confidence = y_pred[:, 7]
            l_block = (block_flag * confidence).mean()
            rule_strength = torch.max(rule_signals[:, 0], rule_signals[:, 1])
            l_consensus = ((1.0 - rule_strength) * confidence).mean()
            l_rule = l_block + 0.5 * l_consensus
        
        total = (
            self.lambdas[0] * l_pred +
            self.lambdas[1] * l_sharpe +
            self.lambdas[2] * l_dd +
            self.lambdas[3] * l_turn +
            (self.lambdas[4] * l_rule if len(self.lambdas) > 4 else 0.0)
        )
        
        return {
            'total': total,
            'pred': l_pred.detach(),
            'sharpe': l_sharpe.detach(),
            'drawdown': l_dd.detach(),
            'turnover': l_turn.detach(),
            'rule': l_rule.detach()
        }
