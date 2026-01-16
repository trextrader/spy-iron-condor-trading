"""
Volatility-Gated Attention Module

This module provides a dynamic attention mechanism that activates based on
volatility regime. In low-volatility periods, the model uses efficient local
processing. In high-volatility periods, it engages expensive global attention.

Mathematical Foundation:
    g = σ(W₂ · SiLU(W₁ · E[:,-1,:])) ∈ [0,1]    # Gating signal
    A = MultiHeadAttn(E, E, E)                    # Global context
    output = (1 - g) · E + g · A                  # Blended output

Key Insight:
- Gate g is computed from the last hidden state (most recent context)
- When g ≈ 0: output ≈ E (passthrough, efficient)
- When g ≈ 1: output ≈ A (full attention, expensive but informative)
- The model learns when global context is needed

Usage:
    Insert VolGatedAttn modules after Mamba layers 8, 16, 24 in CondorBrain.
"""

import torch
import torch.nn as nn
from typing import Optional


class VolGatedAttn(nn.Module):
    """
    Volatility-Gated Attention that dynamically blends local and global context.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads (default: 8)
        dropout: Dropout rate for attention (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating network: maps last hidden state to scalar gate
        # Uses bottleneck architecture for efficiency
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Layer norm for attention output (optional, improves stability)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        E: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with dynamic gating.
        
        Args:
            E: Input embeddings (B, Seq, D)
            key_padding_mask: Optional padding mask (B, Seq)
            
        Returns:
            Gated output (B, Seq, D)
        """
        B, S, D = E.shape
        
        # Compute gate from last timestep
        last_hidden = E[:, -1, :]  # (B, D)
        g = self.gate(last_hidden)  # (B, 1)
        g = g.unsqueeze(1)  # (B, 1, 1) for broadcasting
        
        # Self-attention (expensive)
        A, _ = self.attn(
            E, E, E,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        A = self.norm(A)
        
        # Blend: (1-g)*local + g*global
        output = (1.0 - g) * E + g * A
        
        return output
    
    def forward_with_gate(
        self,
        E: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass that also returns the gate value for logging.
        
        Returns:
            Tuple of (output, gate_value)
        """
        B, S, D = E.shape
        
        last_hidden = E[:, -1, :]
        g = self.gate(last_hidden)
        g_expanded = g.unsqueeze(1)
        
        A, _ = self.attn(E, E, E, key_padding_mask=key_padding_mask, need_weights=False)
        A = self.norm(A)
        
        output = (1.0 - g_expanded) * E + g_expanded * A
        
        return output, g.squeeze(-1)  # (B, S, D), (B,)


class VolGatedAttnBlock(nn.Module):
    """
    Complete residual block with VolGatedAttn and feedforward.
    
    Structure:
        x -> VolGatedAttn -> Add&Norm -> FFN -> Add&Norm -> output
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attn = VolGatedAttn(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = self.norm1(x + self.attn(x))
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x
