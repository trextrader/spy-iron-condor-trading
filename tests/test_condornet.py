# tests/test_condornet.py
"""
Unit test suite for CondorNet™ - Mathematically Faithful Implementation

Tests cover:
- AugmentedStateSpec split/cat operations
- BlockMatrixA full matrix and forward consistency
- ETD-1 kernel properties
- condornet_master_step shapes
- CanonicalPredicateGates range [0,1]
- PredicateSignature permutation invariance
- RegimeCombinatoricsDynamics update
- CondorNet forward shapes and diagnostics
- ETD-1 tensor dt support
"""

import math
import torch
import pytest

from intelligence.condor_brain_net import (
    CondorNet,
    AugmentedStateSpec,
    BlockMatrixA,
    BlockVectorB,
    CDEResponseG,
    FullForcingD,
    CanonicalPredicateGates,
    PredicateSignature,
    RegimeCombinatoricsDynamics,
    etd1_kernel,
    condornet_master_step,
)


@pytest.mark.parametrize("d_h,d_v,d_m,d_r", [(16, 8, 8, 4)])
def test_augmented_state_spec_split_cat(d_h, d_v, d_m, d_r):
    """Test that split and cat are inverses."""
    spec = AugmentedStateSpec(d_h, d_v, d_m, d_r)
    x = torch.randn(3, spec.d_x)

    h, v, m, r = spec.split(x)
    assert h.shape == (3, d_h)
    assert v.shape == (3, d_v)
    assert m.shape == (3, d_m)
    assert r.shape == (3, d_r)

    x_recon = spec.cat(h, v, m, r)
    assert torch.allclose(x, x_recon, atol=1e-6)


def test_block_matrix_a_full_matrix_and_forward():
    """Test BlockMatrixA consistency: A(x) ≈ A_full @ x."""
    spec = AugmentedStateSpec(8, 4, 4, 2)
    A = BlockMatrixA(spec, enforce_sparsity=True)

    x = torch.randn(5, spec.d_x)
    y = A.forward_blocks(x)
    assert y.shape == x.shape

    A_full = A.full_matrix()
    assert A_full.shape == (spec.d_x, spec.d_x)

    # Check consistency: A(x) ≈ A_full @ x^T
    y_mat = x @ A_full.T
    assert torch.allclose(y, y_mat, atol=1e-5)


def test_etd1_kernel_basic_properties():
    """Test ETD-1 kernel approximates I + AΔt for small A."""
    d = 6
    A = torch.randn(d, d) * 0.01  # small for stability
    dt = 0.5

    F, phi1 = etd1_kernel(A, dt)
    assert F.shape == (d, d)
    assert phi1.shape == (d, d)

    # F should be close to I + A dt for small A
    I = torch.eye(d)
    approx = I + A * dt
    assert torch.allclose(F, approx, atol=1e-2)


def test_condornet_master_step_shapes():
    """Test condornet_master_step produces correct output shape."""
    spec = AugmentedStateSpec(8, 4, 4, 2)
    d_input = 10
    d_control = 12
    n_greeks = 5
    d_q = 1

    A_theta = BlockMatrixA(spec, enforce_sparsity=True)
    B_theta = BlockVectorB(spec, d_control)
    G_theta = CDEResponseG(spec, d_input, d_control)
    D_forcing = FullForcingD(spec, n_greeks=n_greeks, d_q=d_q)

    batch = 7
    x_prev = torch.randn(batch, spec.d_x)
    u_k = torch.randn(batch, d_control)
    dX_k = torch.randn(batch, d_input)
    greeks_k = torch.randn(batch, n_greeks)
    r_prev = torch.randn(batch, spec.d_r)
    q_k = torch.randn(batch, d_q)
    dt_k = 1.0

    x_k = condornet_master_step(
        spec, A_theta, B_theta, G_theta, D_forcing,
        x_prev, u_k, dX_k, greeks_k, r_prev, q_k, dt_k
    )
    assert x_k.shape == (batch, spec.d_x)


def test_canonical_predicate_gates_range():
    """Test CanonicalPredicateGates outputs are in [0, 1]."""
    gates = CanonicalPredicateGates()
    batch = 11

    iv_rank = torch.randn(batch) * 10 + 50
    bid_ask_spread = torch.rand(batch) * 0.5
    price = torch.rand(batch) * 500 + 10
    rsi = torch.rand(batch) * 100
    delta_rsi = torch.randn(batch)
    S_t = torch.rand(batch) * 500 + 10
    S_t_minus_1 = S_t + torch.randn(batch)
    gamma = torch.randn(batch) * 0.1

    p = gates(
        iv_rank=iv_rank,
        bid_ask_spread=bid_ask_spread,
        price=price,
        rsi=rsi,
        delta_rsi=delta_rsi,
        S_t=S_t,
        S_t_minus_1=S_t_minus_1,
        gamma=gamma,
    )
    assert p.shape == (batch, 5)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)


def test_predicate_signature_invariance():
    """Test PredicateSignature is permutation-invariant."""
    K = 5
    sig = PredicateSignature(K=K, R=4, M=16)
    batch = 4

    p = torch.rand(batch, K)
    p_perm = p[:, torch.randperm(K)]

    _, _, _, z = sig(p)
    _, _, _, z_perm = sig(p_perm)

    # Invariant part (moments + bloom) should be equal
    inv = sig.signature_only(p)
    inv_perm = sig.signature_only(p_perm)
    assert torch.allclose(inv, inv_perm, atol=1e-5)

    assert z.shape[0] == batch
    assert z_perm.shape[0] == batch


def test_regime_combinatorics_dynamics_update():
    """Test RegimeCombinatoricsDynamics produces valid update."""
    d_r = 6
    K = 5
    R = 4
    M = 16
    z_dim = K + R + M

    dyn = RegimeCombinatoricsDynamics(d_r=d_r, z_dim=z_dim)
    batch = 3

    r_prev = torch.randn(batch, d_r)
    z_pred = torch.randn(batch, z_dim)

    r_k = dyn(r_prev, z_pred)
    assert r_k.shape == (batch, d_r)

    # Check that alpha in (0,1)
    with torch.no_grad():
        alpha = torch.sigmoid(dyn.W_alpha(z_pred))
        assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)


def test_condornet_forward_shapes_and_diagnostics():
    """Test CondorNet forward pass produces correct shapes and diagnostics."""
    model = CondorNet(
        d_input=54,
        d_h=32,
        d_v=8,
        d_m=8,
        d_r=4,
        d_control=16,
        n_greeks=5,
        d_q=1,
        n_predicates=5,
        R_moments=4,
        M_bloom=8,
        n_layers=1,
        enforce_sparsity=True,
    )

    batch = 5
    seq = 20
    x = torch.randn(batch, seq, 54)

    outputs, diag = model(x, return_diagnostics=True)

    assert outputs.shape == (batch, 10)
    assert 'predicates' in diag
    assert 'z_final' in diag
    assert diag['predicates'].shape[0] == batch
    assert diag['z_final'].shape[0] == batch


def test_etd1_kernel_dt_tensor_support():
    """Test ETD-1 kernel accepts tensor dt."""
    d = 4
    A = torch.randn(d, d) * 0.01
    dt = torch.tensor(0.25, dtype=A.dtype, device=A.device)

    F, phi1 = etd1_kernel(A, dt)
    assert F.shape == (d, d)
    assert phi1.shape == (d, d)


def test_full_forcing_d_shapes():
    """Test FullForcingD outputs correct shape for all 4 blocks."""
    spec = AugmentedStateSpec(8, 4, 4, 2)
    n_greeks = 5
    d_q = 1
    D = FullForcingD(spec, n_greeks=n_greeks, d_q=d_q)
    
    batch = 6
    greeks = torch.randn(batch, n_greeks)
    r_prev = torch.randn(batch, spec.d_r)
    q = torch.randn(batch, d_q)
    
    out = D(greeks, r_prev, q)
    assert out.shape == (batch, spec.d_x)


def test_block_vector_b_shapes():
    """Test BlockVectorB outputs correct shape."""
    spec = AugmentedStateSpec(8, 4, 4, 2)
    d_control = 12
    B = BlockVectorB(spec, d_control)
    
    batch = 5
    u = torch.randn(batch, d_control)
    
    out = B(u)
    assert out.shape == (batch, spec.d_x)


def test_cde_response_g_shapes():
    """Test CDEResponseG outputs correct shape."""
    spec = AugmentedStateSpec(8, 4, 4, 2)
    d_input = 10
    d_control = 12
    G = CDEResponseG(spec, d_input, d_control)
    
    batch = 5
    x = torch.randn(batch, spec.d_x)
    u = torch.randn(batch, d_control)
    
    out = G(x, u)
    assert out.shape == (batch, spec.d_x, d_input)
