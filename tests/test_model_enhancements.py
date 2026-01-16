"""
Unit Tests for CondorBrain Model Enhancements

Tests for:
- CompositeCondorLoss (risk-aligned loss function)
- VolGatedAttn (volatility-gated attention)
- TopKMoE (mixture of experts)
- Manifold volatility indicators
- TDA signature (with ripser fallback)
- Policy outputs
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================================
# COMPOSITE LOSS TESTS
# ============================================================================

class TestCompositeCondorLoss:
    """Tests for CompositeCondorLoss."""
    
    def test_forward_basic(self):
        """Test basic forward pass without optional args."""
        from intelligence.condor_loss import CompositeCondorLoss
        
        loss_fn = CompositeCondorLoss(lambdas=(1.0, 0.0, 0.0, 0.0))
        
        y_pred = torch.randn(32, 8)
        y_true = torch.randn(32, 8)
        
        loss = loss_fn(y_pred, y_true)
        
        assert loss.shape == ()  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0  # Huber loss is non-negative
    
    def test_forward_with_returns(self):
        """Test Sharpe and drawdown computation with returns."""
        from intelligence.condor_loss import CompositeCondorLoss
        
        loss_fn = CompositeCondorLoss(lambdas=(1.0, 0.5, 0.1, 0.0))
        
        y_pred = torch.randn(32, 8)
        y_true = torch.randn(32, 8)
        returns = torch.randn(32)  # Simulated returns
        
        loss = loss_fn(y_pred, y_true, returns=returns)
        
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_forward_with_turnover(self):
        """Test turnover penalty computation."""
        from intelligence.condor_loss import CompositeCondorLoss
        
        loss_fn = CompositeCondorLoss(lambdas=(1.0, 0.0, 0.0, 0.5))
        
        y_pred = torch.randn(32, 8)
        y_true = torch.randn(32, 8)
        weights = torch.rand(32, 4)  # Current weights
        last_weights = torch.rand(32, 4)  # Previous weights
        
        loss = loss_fn(y_pred, y_true, weights=weights, last_weights=last_weights)
        
        assert loss.shape == ()
        assert not torch.isnan(loss)
    
    def test_decomposed_output(self):
        """Test decomposed loss returns all components."""
        from intelligence.condor_loss import CompositeCondorLoss
        
        loss_fn = CompositeCondorLoss(lambdas=(1.0, 0.5, 0.1, 0.1))
        
        y_pred = torch.randn(16, 8)
        y_true = torch.randn(16, 8)
        returns = torch.randn(16)
        weights = torch.rand(16, 4)
        last_weights = torch.rand(16, 4)
        
        result = loss_fn.forward_decomposed(
            y_pred, y_true, returns, weights, last_weights
        )
        
        assert 'total' in result
        assert 'pred' in result
        assert 'sharpe' in result
        assert 'drawdown' in result
        assert 'turnover' in result


# ============================================================================
# VOLATILITY-GATED ATTENTION TESTS
# ============================================================================

class TestVolGatedAttn:
    """Tests for VolGatedAttn module."""
    
    def test_forward_shapes(self):
        """Test input/output tensor shapes match."""
        from intelligence.vol_gated_attn import VolGatedAttn
        
        d_model = 256
        batch_size = 8
        seq_len = 64
        
        attn = VolGatedAttn(d_model=d_model, n_heads=4)
        x = torch.randn(batch_size, seq_len, d_model)
        
        out = attn(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_gate_range(self):
        """Test that gate values are in [0, 1]."""
        from intelligence.vol_gated_attn import VolGatedAttn
        
        d_model = 128
        attn = VolGatedAttn(d_model=d_model, n_heads=2)
        x = torch.randn(4, 32, d_model)
        
        out, gate = attn.forward_with_gate(x)
        
        assert gate.min() >= 0.0
        assert gate.max() <= 1.0
    
    def test_residual_block(self):
        """Test complete VolGatedAttnBlock."""
        from intelligence.vol_gated_attn import VolGatedAttnBlock
        
        d_model = 256
        block = VolGatedAttnBlock(d_model=d_model, n_heads=4)
        x = torch.randn(4, 32, d_model)
        
        out = block(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


# ============================================================================
# MIXTURE OF EXPERTS TESTS
# ============================================================================

class TestTopKMoE:
    """Tests for TopKMoE module."""
    
    def test_forward_shapes(self):
        """Test output shape is (B, output_dim)."""
        from intelligence.topk_moe import TopKMoE
        
        d_model = 256
        output_dim = 8
        batch_size = 16
        
        moe = TopKMoE(d_model=d_model, output_dim=output_dim, n_experts=3, k=1)
        x = torch.randn(batch_size, 64, d_model)  # Sequence input
        
        out = moe(x)
        
        assert out.shape == (batch_size, output_dim)
        assert not torch.isnan(out).any()
    
    def test_routing_probabilities(self):
        """Test that routing probabilities sum to 1."""
        from intelligence.topk_moe import TopKMoE
        
        d_model = 128
        moe = TopKMoE(d_model=d_model, output_dim=8, n_experts=3, k=2)
        x = torch.randn(8, 32, d_model)
        
        out, probs, selected = moe.forward_with_routing(x)
        
        # Full probabilities sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)
        
        # Selected experts are valid indices
        assert selected.min() >= 0
        assert selected.max() < 3
    
    def test_batched_moe(self):
        """Test BatchedTopKMoE for efficiency."""
        from intelligence.topk_moe import BatchedTopKMoE
        
        d_model = 256
        moe = BatchedTopKMoE(d_model=d_model, output_dim=8, n_experts=3, k=1)
        x = torch.randn(32, 64, d_model)
        
        out = moe(x)
        
        assert out.shape == (32, 8)
        assert not torch.isnan(out).any()


# ============================================================================
# MANIFOLD VOLATILITY TESTS
# ============================================================================

class TestManifoldVolatility:
    """Tests for manifold-based volatility indicators."""
    
    def test_curvature_proxy_no_nan(self):
        """Test curvature proxy produces finite values."""
        from intelligence.indicators.manifold_volatility import curvature_proxy_from_returns
        
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        
        kappa = curvature_proxy_from_returns(returns, span=64)
        
        # Skip warmup period
        assert not kappa.iloc[100:].isna().any()
        assert np.isfinite(kappa.iloc[100:]).all()
    
    def test_volatility_energy_non_negative(self):
        """Test volatility energy is non-negative."""
        from intelligence.indicators.manifold_volatility import (
            curvature_proxy_from_returns,
            volatility_energy_from_curvature
        )
        
        returns = pd.Series(np.random.randn(500) * 0.01)
        kappa = curvature_proxy_from_returns(returns)
        energy = volatility_energy_from_curvature(kappa)
        
        # Energy should be >= 0 (log1p of non-negative)
        valid_energy = energy.dropna()
        assert (valid_energy >= 0).all()
    
    def test_dynamic_rsi_range(self):
        """Test dynamic RSI is in [0, 100]."""
        from intelligence.indicators.manifold_volatility import dynamic_rsi
        
        close = pd.Series(100 + np.cumsum(np.random.randn(500) * 0.5))
        
        rsi = dynamic_rsi(close, window=14)
        valid_rsi = rsi.dropna()
        
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_compute_manifold_features(self):
        """Test convenience function returns all features."""
        from intelligence.indicators.manifold_volatility import compute_manifold_features
        
        close = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5))
        
        features = compute_manifold_features(close)
        
        assert 'log_return' in features.columns
        assert 'curvature' in features.columns
        assert 'vol_energy' in features.columns
        assert 'dynamic_rsi' in features.columns
        assert len(features) == len(close)


# ============================================================================
# TDA SIGNATURE TESTS
# ============================================================================

class TestTDASignature:
    """Tests for persistent homology TDA features."""
    
    def test_takens_embedding_shape(self):
        """Test Takens embedding output shape."""
        from intelligence.indicators.tda_signature import _takens_embedding
        
        series = np.random.randn(100)
        m = 5
        tau = 2
        
        pc = _takens_embedding(series, m=m, tau=tau)
        
        expected_length = len(series) - (m - 1) * tau
        assert pc.shape == (expected_length, m)
    
    def test_ripser_fallback(self):
        """Test TDA works without ripser (returns 0.0)."""
        from intelligence.indicators.tda_signature import compute_pi_series, is_ripser_available
        
        close = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5))
        
        pi = compute_pi_series(close, window=100, m=3, tau=1)
        
        # Should return values (0.0 if ripser not available, actual signature if available)
        assert len(pi) == len(close)
        
        if not is_ripser_available():
            # Without ripser, all values after warmup should be 0.0
            assert (pi.iloc[100:] == 0.0).all()
        else:
            # With ripser, values should be non-negative
            assert (pi.iloc[100:] >= 0).all()


# ============================================================================
# POLICY OUTPUTS TESTS
# ============================================================================

class TestPolicyOutputs:
    """Tests for measure-theoretic policy outputs."""
    
    def test_state_binner_encode(self):
        """Test StateBinner produces valid bin indices."""
        from intelligence.indicators.policy_outputs import StateBinner
        
        binner = StateBinner()
        
        state = binner.encode(0.5, 0.5, 0.25, 50.0)
        
        assert len(state) == 4
        assert all(isinstance(s, int) for s in state)
        assert all(s >= 0 for s in state)
    
    def test_state_binner_n_states(self):
        """Test state count calculation."""
        from intelligence.indicators.policy_outputs import StateBinner
        
        binner = StateBinner(
            vol_edges=[0.0, 1.0],      # 3 bins
            cons_edges=[0.0, 0.5],     # 3 bins
            brk_edges=[0.0],           # 2 bins
            ivr_edges=[0.0, 50.0]      # 3 bins
        )
        
        assert binner.n_states() == 3 * 3 * 2 * 3  # 54 states
    
    def test_policy_vector_sums_to_one(self):
        """Test policy vector is valid probability distribution."""
        from intelligence.indicators.policy_outputs import StateBinner, policy_vector_from_row, ACTIONS
        
        binner = StateBinner()
        Q = {}  # Empty Q-table (uniform policy)
        
        row = pd.Series({
            'vol_energy': 0.5,
            'consolidation_score': 0.3,
            'breakout_score': 0.1,
            'ivr': 45.0
        })
        
        policy = policy_vector_from_row(row, binner, Q)
        
        assert len(policy) == len(ACTIONS)
        assert np.isclose(policy.sum(), 1.0, atol=1e-6)
        assert (policy >= 0).all()
        assert (policy <= 1).all()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple modules."""
    
    def test_loss_with_moe_output(self):
        """Test CompositeCondorLoss with TopKMoE output."""
        from intelligence.condor_loss import CompositeCondorLoss
        from intelligence.topk_moe import TopKMoE
        
        d_model = 256
        batch_size = 16
        
        moe = TopKMoE(d_model=d_model, output_dim=8, n_experts=3, k=1)
        loss_fn = CompositeCondorLoss()
        
        x = torch.randn(batch_size, 64, d_model)
        y_true = torch.randn(batch_size, 8)
        
        y_pred = moe(x)
        loss = loss_fn(y_pred, y_true)
        
        assert loss.shape == ()
        assert not torch.isnan(loss)
    
    def test_attn_then_moe(self):
        """Test VolGatedAttn feeding into TopKMoE."""
        from intelligence.vol_gated_attn import VolGatedAttn
        from intelligence.topk_moe import TopKMoE
        
        d_model = 256
        batch_size = 8
        seq_len = 32
        
        attn = VolGatedAttn(d_model=d_model, n_heads=4)
        moe = TopKMoE(d_model=d_model, output_dim=8, n_experts=3, k=1)
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Attention then MoE
        h = attn(x)
        out = moe(h)
        
        assert out.shape == (batch_size, 8)
        assert not torch.isnan(out).any()
