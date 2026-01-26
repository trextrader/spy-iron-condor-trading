"""
Regime-Routed Mixture-of-Experts (MoE)

This module provides sparse expert routing for regime-specialized processing.
Different market regimes (low/normal/high volatility) require different
trading strategies, so we train specialized experts for each.

Mathematical Foundation:
    logits = W_gate · E[:,-1,:]          # (B, n_experts)
    probs = softmax(logits)              # (B, n_experts)
    topv, topi = topk(probs, k)          # Select top-k experts
    output = Σᵢ (w_i · Expert_i(E))      # Weighted expert outputs

Key Insight:
- Sparse activation: only top-K experts are computed (efficiency)
- Each expert specializes in a volatility regime
- Gating learns to route based on market state
- Integrates with existing RegimeDetector for soft gating

Usage:
    Replace final output head with TopKMoE for regime-specialized predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TopKMoE(nn.Module):
    """
    Top-K Mixture of Experts for regime-specialized Iron Condor output.

    Args:
        d_model: Input embedding dimension
        output_dim: Output dimension (10 for IC parameters: 8 orig + entry/exit logits)
        n_experts: Number of expert networks (default: 3 for Low/Normal/High)
        k: Number of experts to activate per sample (default: 1)
        hidden_ratio: Expert hidden layer size ratio (default: 4)
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int = 10,  # Changed from 8: now includes entry_logit, exit_logit
        n_experts: int = 3,
        k: int = 1,
        hidden_ratio: int = 4
    ):
        super().__init__()
        
        self.d_model = d_model
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.k = min(k, n_experts)  # Can't activate more experts than exist
        
        # Gating network: produces routing probabilities
        self.gate = nn.Linear(d_model, n_experts)
        
        # Expert networks: each is a small MLP
        hidden_dim = d_model * hidden_ratio
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(n_experts)
        ])
        
    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with top-K expert routing.
        
        Args:
            E: Input embeddings (B, Seq, D) or (B, D)
            
        Returns:
            Expert-weighted output (B, output_dim)
        """
        # Handle both sequence and non-sequence inputs
        if E.dim() == 3:
            # Take last timestep for routing decision
            E_gate = E[:, -1, :]  # (B, D)
        else:
            E_gate = E  # (B, D)
        
        B = E_gate.shape[0]
        device = E_gate.device
        
        # Compute gating probabilities
        logits = self.gate(E_gate)  # (B, n_experts)
        probs = F.softmax(logits, dim=-1)  # (B, n_experts)
        
        # Select top-k experts
        topv, topi = torch.topk(probs, self.k, dim=-1)  # (B, k), (B, k)
        
        # Renormalize top-k weights to sum to 1
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)  # (B, k)
        
        # Compute weighted expert outputs
        output = torch.zeros(B, self.output_dim, device=device, dtype=E_gate.dtype)
        
        for j in range(self.k):
            # Get expert indices for this slot
            expert_idx = topi[:, j]  # (B,)
            weight = topv[:, j].unsqueeze(-1)  # (B, 1)
            
            # Compute expert outputs batch-wise
            # Note: This is a simplified loop; for large batches, 
            # consider batched expert execution
            for b in range(B):
                idx = expert_idx[b].item()
                expert_out = self.experts[idx](E_gate[b:b+1])  # (1, output_dim)
                output[b:b+1] += weight[b:b+1] * expert_out
        
        return output
    
    def forward_with_routing(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns routing info for logging.
        
        Returns:
            Tuple of (output, routing_probs, selected_experts)
        """
        if E.dim() == 3:
            E_gate = E[:, -1, :]
        else:
            E_gate = E
        
        B = E_gate.shape[0]
        device = E_gate.device
        
        logits = self.gate(E_gate)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.k, dim=-1)
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        
        output = torch.zeros(B, self.output_dim, device=device, dtype=E_gate.dtype)
        
        for j in range(self.k):
            expert_idx = topi[:, j]
            weight = topv[:, j].unsqueeze(-1)
            
            for b in range(B):
                idx = expert_idx[b].item()
                expert_out = self.experts[idx](E_gate[b:b+1])
                output[b:b+1] += weight[b:b+1] * expert_out
        
        return output, probs, topi


class BatchedTopKMoE(nn.Module):
    """
    Batch-efficient Top-K MoE using einsum for parallel expert computation.

    This version is more efficient for large batches by avoiding per-sample loops.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int = 10,  # Changed from 8: now includes entry_logit, exit_logit
        n_experts: int = 3,
        k: int = 1,
        hidden_ratio: int = 4
    ):
        super().__init__()
        
        self.d_model = d_model
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.k = min(k, n_experts)
        
        self.gate = nn.Linear(d_model, n_experts)
        
        # Batched expert weights
        hidden_dim = d_model * hidden_ratio
        self.expert_up = nn.Parameter(torch.randn(n_experts, d_model, hidden_dim) * 0.02)
        self.expert_down = nn.Parameter(torch.randn(n_experts, hidden_dim, output_dim) * 0.02)
        self.expert_bias_up = nn.Parameter(torch.zeros(n_experts, hidden_dim))
        self.expert_bias_down = nn.Parameter(torch.zeros(n_experts, output_dim))
        
    def forward(self, E: torch.Tensor) -> torch.Tensor:
        if E.dim() == 3:
            E_gate = E[:, -1, :]
        else:
            E_gate = E
        
        B = E_gate.shape[0]
        
        # Routing
        logits = self.gate(E_gate)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.k, dim=-1)
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Compute all expert outputs at once
        # E_gate: (B, D), expert_up: (E, D, H)
        h = torch.einsum('bd,edh->beh', E_gate, self.expert_up)  # (B, E, H)
        h = h + self.expert_bias_up.unsqueeze(0)  # (B, E, H)
        h = F.silu(h)
        
        out = torch.einsum('beh,eho->beo', h, self.expert_down)  # (B, E, O)
        out = out + self.expert_bias_down.unsqueeze(0)  # (B, E, O)
        
        # Select top-k and weight
        # Gather selected expert outputs
        topi_expanded = topi.unsqueeze(-1).expand(-1, -1, self.output_dim)  # (B, k, O)
        selected_out = torch.gather(out, dim=1, index=topi_expanded)  # (B, k, O)
        
        # Weight and sum
        output = (topv.unsqueeze(-1) * selected_out).sum(dim=1)  # (B, O)
        
        return output
