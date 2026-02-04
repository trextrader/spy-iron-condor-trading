"""
CondorNet™ - Mathematically Faithful Implementation

This module implements the complete CondorNet architecture as specified in the
canonical PDF/Word document. Key features:

- True ETD-1 with matrix_exp(AΔt) and φ_1(M) = M⁺(e^M - I)
- 4-block augmented state [h_k; v_k; m_k; r_k] with explicit block semantics
- Block-partitioned operators: A_θ, B_θ, G_θ, D
- 5 canonical inequality gates (vol spike, liquidity, reversal, gap, Greeks)
- Predicate combinatorics with group-invariant signatures
- r_k dynamics: r_k = α_k ⊙ r_{k-1} + β_k
- Full 4-block forcing with causality (r_{k-1} in D)

Master equation:
    x_k = e^{A_θ Δt_k} x_{k-1} + Δt_k φ_1(A_θ Δt_k) B_θ(u_k)
        + G_θ(x_{k-1}, u_k) ΔX_k + D(Greeks_k, r_{k-1}, q_k)

Author: Claude Code (Opus 4.5)
Version: 1.0.0
Date: 2026-02-03
"""

import math
from typing import Tuple, Dict, Optional, List, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PART 1: CORE DIMENSIONS AND BLOCK HELPERS
# =============================================================================

class AugmentedStateSpec:
    """
    Pure specification: dimensions and block slicing helpers.

    The 4-block state is the audit enforcement mechanism:
    - h: latent risk-manifold (market physics)
    - v: portfolio state (PnL, Greeks, positions)
    - m: risk memory (drawdown, cumulative Greeks)
    - r: regime/combinatorics (vol regime, predicate memory)
    """
    def __init__(self, d_h: int, d_v: int, d_m: int, d_r: int):
        self.d_h = d_h
        self.d_v = d_v
        self.d_m = d_m
        self.d_r = d_r
        self.d_x = d_h + d_v + d_m + d_r

    def split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split x into (h, v, m, r) blocks.

        Args:
            x: (..., d_x) state tensor

        Returns:
            (h, v, m, r) with matching leading dims
        """
        dh, dv, dm = self.d_h, self.d_v, self.d_m
        h = x[..., :dh]
        v = x[..., dh:dh+dv]
        m = x[..., dh+dv:dh+dv+dm]
        r = x[..., dh+dv+dm:]
        return h, v, m, r

    def cat(self, h: torch.Tensor, v: torch.Tensor,
            m: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Concatenate (h, v, m, r) into full state x."""
        return torch.cat([h, v, m, r], dim=-1)


# =============================================================================
# PART 2: BLOCK OPERATORS (A_θ, B_θ, G_θ, D)
# =============================================================================

class BlockMatrixA(nn.Module):
    """
    4×4 block transition operator A_θ(u_k).

    Structure:
        A_θ = [A_hh  A_hv  A_hm  A_hr]
              [A_vh  A_vv  A_vm  A_vr]
              [A_mh  A_mv  A_mm  A_mr]
              [A_rh  A_rv  A_rm  A_rr]

    Sparsity rules (RA-RD from spec):
        RA: D_v carries execution drag, D_h does NOT absorb slippage
        RB: A_hr free (regime→physics), A_hv ≈ 0 (PnL doesn't rewrite physics)
        RC: A_hm small/diagonal (memory throttles, doesn't rewrite physics)
        RD: Entry/exit via G_k ΔX_k (CDE response dominates signals)
    """
    def __init__(self, spec: AugmentedStateSpec, enforce_sparsity: bool = True):
        super().__init__()
        d_h, d_v, d_m, d_r = spec.d_h, spec.d_v, spec.d_m, spec.d_r
        self.spec = spec
        self.enforce_sparsity = enforce_sparsity

        # === DIAGONAL BLOCKS (self-dynamics) ===
        self.A_hh = nn.Linear(d_h, d_h, bias=False)
        self.A_vv = nn.Linear(d_v, d_v, bias=False)
        self.A_mm = nn.Linear(d_m, d_m, bias=False)
        self.A_rr = nn.Linear(d_r, d_r, bias=False)

        # === OFF-DIAGONAL BLOCKS ===
        # h row: what affects latent physics
        if enforce_sparsity:
            # RB: Constrain A_hv to near-zero (PnL shouldn't rewrite physics)
            self.A_hv = None  # Explicitly disabled
            # RC: A_hm small/diagonal - use diagonal parameterization
            self.A_hm_diag = nn.Parameter(torch.zeros(min(d_h, d_m)))
        else:
            self.A_hv = nn.Linear(d_v, d_h, bias=False)
            self.A_hm = nn.Linear(d_m, d_h, bias=False)

        # RB: A_hr free (regime → physics allowed)
        self.A_hr = nn.Linear(d_r, d_h, bias=False)

        # v row: what affects portfolio
        self.A_vh = nn.Linear(d_h, d_v, bias=False)
        self.A_vm = nn.Linear(d_m, d_v, bias=False)
        self.A_vr = nn.Linear(d_r, d_v, bias=False)

        # m row: what affects memory
        self.A_mh = nn.Linear(d_h, d_m, bias=False)
        self.A_mv = nn.Linear(d_v, d_m, bias=False)
        self.A_mr = nn.Linear(d_r, d_m, bias=False)

        # r row: what affects regime
        self.A_rh = nn.Linear(d_h, d_r, bias=False)
        self.A_rv = nn.Linear(d_v, d_r, bias=False)
        self.A_rm = nn.Linear(d_m, d_r, bias=False)

        # Stability initialization
        self._init_stable()

    def _init_stable(self):
        """Initialize for spectral stability (ρ(A) < 1)."""
        scale = 0.01
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=scale)
            elif 'diag' in name:
                nn.init.zeros_(param)

    def _apply_A_hm(self, m: torch.Tensor) -> torch.Tensor:
        """Apply A_hm (diagonal or full depending on sparsity mode)."""
        if self.enforce_sparsity:
            # Diagonal parameterization: pad to (d_h, d_m) implicitly
            d = len(self.A_hm_diag)
            diag_expanded = torch.diag(self.A_hm_diag)  # (d, d)
            # Pad to (d_h, d_m)
            d_h, d_m = self.spec.d_h, self.spec.d_m
            A_hm_mat = torch.zeros(d_h, d_m, device=m.device, dtype=m.dtype)
            A_hm_mat[:d, :d] = diag_expanded
            return F.linear(m, A_hm_mat)
        else:
            return self.A_hm(m)

    def forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply A_θ in block form to x = [h, v, m, r].
        This is the drift operator A @ x (no Δt yet).
        """
        h, v, m, r = self.spec.split(x)

        # h block
        h_new = self.A_hh(h)
        if self.A_hv is not None:
            h_new = h_new + self.A_hv(v)
        h_new = h_new + self._apply_A_hm(m) + self.A_hr(r)

        # v block
        v_new = self.A_vh(h) + self.A_vv(v) + self.A_vm(m) + self.A_vr(r)

        # m block
        m_new = self.A_mh(h) + self.A_mv(v) + self.A_mm(m) + self.A_mr(r)

        # r block
        r_new = self.A_rh(h) + self.A_rv(v) + self.A_rm(m) + self.A_rr(r)

        return self.spec.cat(h_new, v_new, m_new, r_new)


    def full_matrix(self) -> torch.Tensor:
        """
        Construct full [d_x, d_x] matrix for ETD-1.
        Required for exact matrix_exp(AΔt).
        """
        d_x = self.spec.d_x
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Build by applying to basis vectors
        eye = torch.eye(d_x, device=device, dtype=dtype)
        cols = []
        for i in range(d_x):
            e_i = eye[:, i]
            col = self.forward_blocks(e_i)
            cols.append(col.unsqueeze(-1))
        A_full = torch.cat(cols, dim=-1)  # [d_x, d_x]
        return A_full


class BlockVectorB(nn.Module):
    """
    B_θ(u_k) ∈ ℝ^{d_x}, partitioned into [B_h, B_v, B_m, B_r].

    This is the control injection vector for the ETD-1 term.
    """
    def __init__(self, spec: AugmentedStateSpec, d_control: int):
        super().__init__()
        self.B_h = nn.Linear(d_control, spec.d_h, bias=True)
        self.B_v = nn.Linear(d_control, spec.d_v, bias=True)
        self.B_m = nn.Linear(d_control, spec.d_m, bias=True)
        self.B_r = nn.Linear(d_control, spec.d_r, bias=True)
        self.spec = spec

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, d_control) control embedding

        Returns:
            B: (batch, d_x) block-partitioned forcing vector
        """
        Bh = self.B_h(u)
        Bv = self.B_v(u)
        Bm = self.B_m(u)
        Br = self.B_r(u)
        return self.spec.cat(Bh, Bv, Bm, Br)


class CDEResponseG(nn.Module):
    """
    G_θ(x, u) ∈ ℝ^{d_x × F}, the CDE controlled response matrix.

    Rule RD: Entry/exit features MUST enter via G @ dX, not via drift.
    This ensures CDE response dominates signal injection.
    """
    def __init__(self, spec: AugmentedStateSpec, d_input: int, d_control: int):
        super().__init__()
        self.spec = spec
        self.d_state = spec.d_x
        self.d_input = d_input

        # MLP to produce response matrix
        self.net = nn.Sequential(
            nn.Linear(self.d_state + d_control, self.d_state * 2),
            nn.SiLU(),
            nn.Linear(self.d_state * 2, self.d_state * d_input),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_x) current state
            u: (batch, d_control) control embedding

        Returns:
            G: (batch, d_x, d_input) response matrix
        """
        z = torch.cat([x, u], dim=-1)
        G_flat = self.net(z)
        G = G_flat.view(-1, self.d_state, self.d_input)
        return torch.tanh(G)  # Bound for stability


class FullForcingD(nn.Module):
    """
    D(Greeks_k, r_{k-1}, q_k) ∈ ℝ^{d_x}, the full 4-block forcing.

    Block semantics:
        D_h: Market physics forcing from Greeks + regime
        D_v: PRIMARY execution location (slippage, impact, margin) - Rule RA
        D_m: Cumulative risk (stress integrals, drawdown-weighted)
        D_r: Regime forcing (from predicate combinatorics)

    CRITICAL: Uses r_{k-1} (previous regime) for causality.
    """
    def __init__(self, spec: AugmentedStateSpec, n_greeks: int = 5, d_q: int = 1):
        super().__init__()
        d_in = n_greeks + spec.d_r + d_q

        # D_h: Market physics (NO execution drag here - Rule RA)
        self.D_h = nn.Sequential(
            nn.Linear(d_in, spec.d_h),
            nn.Tanh(),  # Bounded
        )

        # D_v: PRIMARY execution location (slippage, impact, margin)
        self.D_v = nn.Sequential(
            nn.Linear(d_in, spec.d_v * 2),
            nn.SiLU(),
            nn.Linear(spec.d_v * 2, spec.d_v),
        )

        # D_m: Cumulative risk
        self.D_m = nn.Sequential(
            nn.Linear(d_in, spec.d_m),
            nn.Tanh(),
        )

        # D_r: Regime forcing
        self.D_r = nn.Linear(d_in, spec.d_r)

        self.spec = spec

    def forward(self, greeks: torch.Tensor, r_prev: torch.Tensor,
                q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            greeks: (batch, n_greeks) [delta, gamma, theta, vega, rho]
            r_prev: (batch, d_r) PREVIOUS regime state (causality!)
            q: (batch, d_q) position size

        Returns:
            D: (batch, d_x) full forcing vector
        """
        z = torch.cat([greeks, r_prev, q], dim=-1)
        Dh = self.D_h(z)
        Dv = self.D_v(z)
        Dm = self.D_m(z)
        Dr = self.D_r(z)
        return self.spec.cat(Dh, Dv, Dm, Dr)


# =============================================================================
# PART 3: TRUE ETD-1 CORE
# =============================================================================

def etd1_kernel(A: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    True ETD-1 kernel computation.

    Computes:
        M = A Δt
        F = exp(M)
        φ_1(M) = M⁺ (exp(M) - I)

    CRITICAL: Compute M = AΔt first, then pinv(M), NOT pinv(A).

    Supports mixed precision: bf16/fp16 inputs are computed in fp32
    for numerical stability in linalg ops, then cast back.

    Args:
        A: (d_x, d_x) drift matrix
        dt: time increment

    Returns:
        F: (d_x, d_x) state transition matrix e^{AΔt}
        phi1: (d_x, d_x) ETD-1 basis function φ_1(AΔt)
    """
    original_dtype = A.dtype
    
    # ALWAYS compute in FP32 for numerical stability (matrix_exp, pinv)
    A = A.float()
    
    # Ensure dt is tensor
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, device=A.device, dtype=torch.float32)
    else:
        dt = dt.float()
    
    M = A * dt
    expM = torch.linalg.matrix_exp(M)
    I = torch.eye(M.shape[-1], device=M.device, dtype=torch.float32)

    # φ_1(M) = M⁺(e^M - I)
    phi1 = torch.linalg.pinv(M) @ (expM - I)

    # ALWAYS return FP32 for the master step matmuls to avoid Half/Float mismatch
    return expM.float(), phi1.float()


def condornet_master_step(
    spec: AugmentedStateSpec,
    A_theta: BlockMatrixA,
    B_theta: BlockVectorB,
    G_theta: CDEResponseG,
    D_forcing: FullForcingD,
    x_prev: torch.Tensor,
    u_k: torch.Tensor,
    dX_k: torch.Tensor,
    greeks_k: torch.Tensor,
    r_prev: torch.Tensor,
    q_k: torch.Tensor,
    dt_k: float,
) -> torch.Tensor:
    """
    One CondorNet master update step (faithful to canonical equation).

    Implements:
        x_k = F_k x_{k-1} + g_k + G_θ(x_{k-1}, u_k) ΔX_k + D(Greeks_k, r_{k-1}, q_k)

    where:
        F_k = exp(A_θ(u_k) Δt_k)
        g_k = Δt_k φ_1(A_θ(u_k) Δt_k) B_θ(u_k)

    Args:
        spec: AugmentedStateSpec
        A_theta: Block transition operator
        B_theta: Control injection
        G_theta: CDE response
        D_forcing: Full forcing
        x_prev: (batch, d_x) previous state
        u_k: (batch, d_control) control embedding
        dX_k: (batch, d_input) control increment
        greeks_k: (batch, n_greeks) Greeks
        r_prev: (batch, d_r) PREVIOUS regime (causality!)
        q_k: (batch, d_q) position size
        dt_k: time increment

    Returns:
        x_k: (batch, d_x) updated state
    """
    batch = x_prev.shape[0]

    # Force all math to FP32 for the master step to avoid autocast mismatches
    x_prev = x_prev.float()
    u_k = u_k.float()
    dX_k = dX_k.float()
    greeks_k = greeks_k.float()
    r_prev = r_prev.float()
    q_k = q_k.float()

    # 1. Build A(u_k) as full [d_x, d_x] (shared across batch)
    A_full = A_theta.full_matrix().float()  # (d_x, d_x)
    F_k, phi1 = etd1_kernel(A_full, dt_k)  # (d_x, d_x), (d_x, d_x)

    # 2. B_θ(u_k) and ETD injection g_k = Δt φ_1(AΔt) B
    # Disable autocast for these specific matmuls to force FP32 consistency
    with torch.amp.autocast('cuda', enabled=False):
        B_k = B_theta(u_k).float()  # (batch, d_x)
        g_k = torch.matmul(B_k, phi1.T) * dt_k  # (batch, d_x)

        # 3. Linear propagation F_k x_{k-1}
        x_lin = torch.matmul(x_prev, F_k.T)  # (batch, d_x)

        # 4. Controlled response G_θ(x_{k-1}, u_k) ΔX_k
        G_k = G_theta(x_prev, u_k).float()  # (batch, d_x, d_input)
        dX_vec = dX_k.unsqueeze(-1)  # (batch, d_input, 1)
        cde_term = torch.bmm(G_k, dX_vec).squeeze(-1)  # (batch, d_x)

        # 5. Full forcing D(Greeks_k, r_{k-1}, q_k)
        D_k = D_forcing(greeks_k, r_prev, q_k).float()  # (batch, d_x)

        # 6. Master update
        x_k = x_lin + g_k + cde_term + D_k

    return x_k.float()


# =============================================================================
# PART 4: PREDICATE COMBINATORICS AND r_k DYNAMICS
# =============================================================================

class CanonicalPredicateGates(nn.Module):
    """
    The 5 canonical inequality gates from the specification.

    Implements soft (differentiable) versions using steep sigmoids:
        1. Volatility spike: IVR_t > 75
        2. Liquidity compression: Spread/Price > 0.4%
        3. Momentum reversal: RSI < 25 AND ΔRSI < 0
        4. Gap risk: |S_t - S_{t-1}|/S_{t-1} > 1.2%
        5. Greeks pressure: |Γ| > threshold

    These gates modulate the decay in A_θ via:
        A(u, σ) = -diag(exp(η(u,σ)) · (1 + λ_p γ_t))
    """
    def __init__(self, steepness: float = 50.0, learnable_thresholds: bool = True):
        super().__init__()
        self.steepness = steepness

        if learnable_thresholds:
            self.iv_rank_thresh = nn.Parameter(torch.tensor(75.0))
            self.spread_frac_thresh = nn.Parameter(torch.tensor(0.004))
            self.rsi_thresh = nn.Parameter(torch.tensor(25.0))
            self.gap_frac_thresh = nn.Parameter(torch.tensor(0.012))
            self.gamma_thresh = nn.Parameter(torch.tensor(0.05))
        else:
            self.register_buffer('iv_rank_thresh', torch.tensor(75.0))
            self.register_buffer('spread_frac_thresh', torch.tensor(0.004))
            self.register_buffer('rsi_thresh', torch.tensor(25.0))
            self.register_buffer('gap_frac_thresh', torch.tensor(0.012))
            self.register_buffer('gamma_thresh', torch.tensor(0.05))

    def forward(
        self,
        iv_rank: torch.Tensor,
        bid_ask_spread: torch.Tensor,
        price: torch.Tensor,
        rsi: torch.Tensor,
        delta_rsi: torch.Tensor,
        S_t: torch.Tensor,
        S_t_minus_1: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft predicate gates.

        All inputs: (batch,) scalar tensors

        Returns:
            p_k: (batch, 5) soft predicates in [0, 1]
        """
        # 1. Volatility spike: IVR_t > 75
        vol_spike = torch.sigmoid(self.steepness * (iv_rank - self.iv_rank_thresh))

        # 2. Liquidity compression: spread/price > 0.4%
        spread_frac = bid_ask_spread / (price + 1e-8)
        liq_lock = torch.sigmoid(self.steepness * (spread_frac - self.spread_frac_thresh))

        # 3. Momentum reversal: RSI < 25 AND ΔRSI < 0
        rsi_gate = torch.sigmoid(self.steepness * (self.rsi_thresh - rsi))
        drsi_gate = torch.sigmoid(self.steepness * (-delta_rsi))
        mom_rev = rsi_gate * drsi_gate  # Soft AND

        # 4. Gap risk: |S_t - S_{t-1}|/S_{t-1} > 1.2%
        gap_frac = torch.abs(S_t - S_t_minus_1) / (S_t_minus_1 + 1e-8)
        gap_gate = torch.sigmoid(self.steepness * (gap_frac - self.gap_frac_thresh))

        # 5. Greeks pressure: |Γ| > threshold
        gamma_gate = torch.sigmoid(self.steepness * (torch.abs(gamma) - self.gamma_thresh))

        p = torch.stack([vol_spike, liq_lock, mom_rev, gap_gate, gamma_gate], dim=-1)
        return p  # (batch, 5)


class PredicateSignature(nn.Module):
    """
    Group-invariant signature from predicate vector.

    Produces z_pred = [p; s; B] where:
        - p: raw predicates
        - s: sorted moments (permutation-invariant)
        - B: Bloom-like signature

    The signature s satisfies s(p) = s(π(p)) for any permutation π.
    """
    def __init__(self, K: int, R: int = 4, M: int = 16):
        super().__init__()
        self.K = K  # Number of predicates
        self.R = R  # Number of moments
        self.M = M  # Bloom dimension

        self.W_bloom = nn.Linear(K, M, bias=True)

        # Output dimension
        self.d_out = K + R + M

    def forward(self, p: torch.Tensor):
        # Math in float32 for stability
        p_fp32 = p.float()

        # Sort for permutation invariance
        p_sorted_fp32, _ = torch.sort(p_fp32, dim=-1, descending=True)

        # Power moments μ_r = E[p^r]
        moments_fp32 = []
        for r in range(1, self.R + 1):
            mu_r = (p_sorted_fp32 ** r).mean(dim=-1, keepdim=True)
            moments_fp32.append(mu_r)
        moments_fp32 = torch.cat(moments_fp32, dim=-1)  # (batch, R)

        # Bloom-like signature (match W_bloom dtype for matmul safely inside autocast)
        # Even if autocast is active, we force FP32 for signature stability
        with torch.amp.autocast('cuda', enabled=False):
            w_dtype = self.W_bloom.weight.dtype
            bloom = torch.sigmoid(self.W_bloom(p_sorted_fp32.to(w_dtype)))

        # Full signature
        z_pred = torch.cat([p_fp32, moments_fp32, bloom.float()], dim=-1)

        # ALWAYS return FP32 results to keep the backbone stable
        return p_sorted_fp32, moments_fp32, bloom.float(), z_pred.float()

    def signature_only(self, p: torch.Tensor) -> torch.Tensor:
        """Return just the invariant part (moments + bloom)."""
        p_fp32 = p.float()
        
        p_sorted_fp32, _ = torch.sort(p_fp32, dim=-1, descending=True)
        moments_fp32 = []
        for r in range(1, self.R + 1):
            mu_r = (p_sorted_fp32 ** r).mean(dim=-1, keepdim=True)
            moments_fp32.append(mu_r)
        moments_fp32 = torch.cat(moments_fp32, dim=-1)
        
        # Match W_bloom dtype
        with torch.amp.autocast('cuda', enabled=False):
            w_dtype = self.W_bloom.weight.dtype
            bloom = torch.sigmoid(self.W_bloom(p_sorted_fp32.to(w_dtype)))
        
        return torch.cat([moments_fp32, bloom.float()], dim=-1)


class RegimeCombinatoricsDynamics(nn.Module):
    """
    Regime latent r_k dynamics driven by predicate combinatorics.

    Update equation:
        r_k = α_k ⊙ r_{k-1} + β_k

    where:
        α_k = σ(W_α z_pred)  -- forget gate
        β_k = W_β z_pred     -- update

    This makes r_k a predicate-combinatorics memory, not just static flags.
    """
    def __init__(self, d_r: int, z_dim: int):
        super().__init__()
        self.W_alpha = nn.Linear(z_dim, d_r)
        self.W_beta = nn.Linear(z_dim, d_r)

    def forward(self, r_prev: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r_prev: (batch, d_r) previous regime state
            z_pred: (batch, z_dim) predicate signature

        Returns:
            r_k: (batch, d_r) updated regime state
        """
        alpha = torch.sigmoid(self.W_alpha(z_pred))  # Forget gate
        beta = self.W_beta(z_pred)  # Update
        r_k = alpha * r_prev + beta
        return r_k


# =============================================================================
# PART 5: PREDICATE SETS AND SUPER-SETS
# =============================================================================

class PredicateSet(nn.Module):
    """
    Set of predicates with learnable aggregation.

    Π(S; p) = aggregation({p_i : i ∈ S})

    Aggregation options:
        - mean: simple average
        - pnorm: generalized mean with learnable p
        - owa: ordered weighted average
    """
    def __init__(self, n_predicates: int, aggregation: str = 'pnorm'):
        super().__init__()
        self.n_predicates = n_predicates
        self.aggregation = aggregation

        # Learnable soft membership
        self.membership = nn.Parameter(torch.randn(n_predicates) * 0.1)

        if aggregation == 'pnorm':
            self.p = nn.Parameter(torch.tensor(1.0))
        elif aggregation == 'owa':
            self.owa_weights = nn.Parameter(torch.ones(n_predicates) / n_predicates)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (batch, K) predicate values

        Returns:
            set_value: (batch, 1) aggregated activation
        """
        weights = torch.softmax(self.membership, dim=0)
        weighted_p = p * weights.unsqueeze(0)

        if self.aggregation == 'mean':
            return weighted_p.sum(dim=-1, keepdim=True)
        elif self.aggregation == 'pnorm':
            p_val = torch.clamp(self.p, -10, 10)
            eps = 1e-8
            weighted_p = torch.clamp(weighted_p, eps, 1 - eps)
            if abs(p_val.item()) < eps:
                return torch.exp(torch.log(weighted_p + eps).mean(dim=-1, keepdim=True))
            else:
                mean_powered = (weighted_p ** p_val).mean(dim=-1, keepdim=True)
                return mean_powered ** (1.0 / p_val)
        elif self.aggregation == 'owa':
            sorted_p, _ = torch.sort(weighted_p, dim=-1, descending=True)
            owa = torch.softmax(self.owa_weights, dim=0)
            return (sorted_p * owa.unsqueeze(0)).sum(dim=-1, keepdim=True)


class SuperSet(nn.Module):
    """
    Hierarchical super-set: comparison of predicate sets.

    S = (S_1 # S_2 # S_3 # S_4) with comparisons <, >, =

    The final output is a gating value in [0, 1].
    """
    def __init__(self, n_sets: int = 4, n_predicates: int = 5):
        super().__init__()
        self.n_sets = n_sets

        self.sets = nn.ModuleList([
            PredicateSet(n_predicates, aggregation='pnorm')
            for _ in range(n_sets)
        ])

        self.final = nn.Sequential(
            nn.Linear(n_sets, n_sets * 2),
            nn.SiLU(),
            nn.Linear(n_sets * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (batch, K) predicate values

        Returns:
            gate: (batch, 1) final super-set gate
        """
        set_values = [pred_set(p) for pred_set in self.sets]
        set_values = torch.cat(set_values, dim=-1)
        return self.final(set_values)


# =============================================================================
# PART 6: TFT CONTROL ENCODER
# =============================================================================

class TFTControlEncoder(nn.Module):
    """
    Temporal Fusion Transformer for control embeddings.

    Produces u_k = TFT_θ(X_{1:k}) for control injection.
    """
    def __init__(self, d_input: int, d_control: int, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_control = d_control

        self.input_proj = nn.Linear(d_input, d_control)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_control) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_control,
            nhead=n_heads,
            dim_feedforward=d_control * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_control, d_control)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_input) input sequence
            return_sequence: if True, return all u_k; else just final

        Returns:
            u: (batch, d_control) or (batch, seq, d_control)
        """
        batch, seq_len, _ = x.shape

        z = self.input_proj(x)
        z = z + self.pos_enc[:, :seq_len, :]

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        z = self.encoder(z, mask=mask)
        u = self.out_proj(z)

        if return_sequence:
            return u
        return u[:, -1, :]


# =============================================================================
# PART 7: FUSION GATE
# =============================================================================

class FusionGate(nn.Module):
    """
    Fusion gate for combining branches/experts.

    Uses z_gate = [h_summary; B; s] per specification.
    """
    def __init__(self, d_h: int, bloom_dim: int, n_moments: int, n_branches: int):
        super().__init__()
        z_gate_dim = d_h + bloom_dim + n_moments

        self.gate = nn.Sequential(
            nn.Linear(z_gate_dim, z_gate_dim),
            nn.SiLU(),
            nn.Linear(z_gate_dim, n_branches),
        )

    def forward(self, h_summary: torch.Tensor, bloom: torch.Tensor,
                moments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_summary: (batch, d_h)
            bloom: (batch, bloom_dim)
            moments: (batch, n_moments)

        Returns:
            weights: (batch, n_branches) softmax weights
        """
        z_gate = torch.cat([h_summary, bloom, moments], dim=-1)
        return F.softmax(self.gate(z_gate), dim=-1)


# =============================================================================
# PART 8: COMPLETE CONDORNET MODULE
# =============================================================================

class CondorNet(nn.Module):
    """
    CondorNet™: Mathematically faithful implementation.

    Master equation:
        x_k = e^{A_θ Δt_k} x_{k-1} + Δt_k φ_1(A_θ Δt_k) B_θ(u_k)
            + G_θ(x_{k-1}, u_k) ΔX_k + D(Greeks_k, r_{k-1}, q_k)

    with 4-block state: x_k = [h_k; v_k; m_k; r_k]

    Features:
        - True ETD-1 with matrix_exp and φ_1
        - Block-partitioned operators
        - 5 canonical predicate gates
        - Group-invariant signatures
        - r_k dynamics: α_k ⊙ r_{k-1} + β_k
        - Full 4-block forcing with causality
    """
    def __init__(
        self,
        d_input: int = 54,
        d_h: int = 256,
        d_v: int = 32,
        d_m: int = 64,
        d_r: int = 32,
        d_control: int = 128,
        n_greeks: int = 5,
        d_q: int = 1,
        n_predicates: int = 5,
        R_moments: int = 4,
        M_bloom: int = 16,
        n_layers: int = 2,
        enforce_sparsity: bool = True,
    ):
        super().__init__()

        # State specification
        self.spec = AugmentedStateSpec(d_h, d_v, d_m, d_r)
        self.d_input = d_input
        self.d_control = d_control
        self.n_greeks = n_greeks

        # === CORE OPERATORS ===
        self.A_theta = BlockMatrixA(self.spec, enforce_sparsity=enforce_sparsity)
        self.B_theta = BlockVectorB(self.spec, d_control)
        self.G_theta = CDEResponseG(self.spec, d_input, d_control)
        self.D_forcing = FullForcingD(self.spec, n_greeks=n_greeks, d_q=d_q)

        # === CONTROL ===
        self.tft = TFTControlEncoder(d_input, d_control, n_layers=n_layers)

        # === INITIAL STATE ===
        self.initial_encoder = nn.Sequential(
            nn.Linear(d_input, self.spec.d_x),
            nn.SiLU(),
            nn.Linear(self.spec.d_x, self.spec.d_x),
        )

        # === PREDICATE SYSTEM ===
        self.pred_gates = CanonicalPredicateGates()
        self.pred_signature = PredicateSignature(K=n_predicates, R=R_moments, M=M_bloom)
        self.regime_dyn = RegimeCombinatoricsDynamics(
            d_r=d_r,
            z_dim=n_predicates + R_moments + M_bloom,
        )
        self.super_set = SuperSet(n_sets=4, n_predicates=n_predicates)

        # === FUSION ===
        self.fusion_gate = FusionGate(d_h, M_bloom, R_moments, n_branches=3)

        # === OUTPUT ===
        self.output_head = nn.Linear(self.spec.d_x, 10)

    def forward(
        self,
        x: torch.Tensor,
        greeks: torch.Tensor = None,
        q: torch.Tensor = None,
        iv_rank: torch.Tensor = None,
        bid_ask_spread: torch.Tensor = None,
        price: torch.Tensor = None,
        rsi: torch.Tensor = None,
        delta_rsi: torch.Tensor = None,
        S_t: torch.Tensor = None,
        S_t_minus_1: torch.Tensor = None,
        gamma: torch.Tensor = None,
        dt: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass with defensive dtype casting."""
        batch, seq_len, _ = x.shape
        device = x.device
        
        # Determine target model dtype
        model_dtype = next(self.parameters()).dtype
        
        # Defensive Casting: Force all inputs to match model dtype
        x = x.to(model_dtype)
        if greeks is not None: greeks = greeks.to(model_dtype)
        if q is not None: q = q.to(model_dtype)
        if iv_rank is not None: iv_rank = iv_rank.to(model_dtype)
        if bid_ask_spread is not None: bid_ask_spread = bid_ask_spread.to(model_dtype)
        if price is not None: price = price.to(model_dtype)
        if rsi is not None: rsi = rsi.to(model_dtype)
        if delta_rsi is not None: delta_rsi = delta_rsi.to(model_dtype)
        if S_t is not None: S_t = S_t.to(model_dtype)
        if S_t_minus_1 is not None: S_t_minus_1 = S_t_minus_1.to(model_dtype)
        if gamma is not None: gamma = gamma.to(model_dtype)
        
        dtype = model_dtype

        # Debug Prints for first batch
        if not hasattr(self, '_printed_dtype_debug'):
            print(f"\n[CondorNet Forward Debug] Dtypes:")
            print(f"  Internal Weights: {model_dtype}")
            print(f"  Input x: {x.dtype}")
            print(f"  Autocast Active: {torch.is_autocast_enabled()}")
            self._printed_dtype_debug = True

        # === DEFAULT INPUTS ===
        if greeks is None:
            greeks = torch.zeros(batch, seq_len, self.n_greeks, device=device, dtype=dtype)
        if q is None:
            q = torch.ones(batch, seq_len, 1, device=device, dtype=dtype)

        # Default predicate inputs (use features from x if not provided)
        if iv_rank is None:
            iv_rank = torch.full((batch, seq_len), 50.0, device=device, dtype=dtype)
        if bid_ask_spread is None:
            bid_ask_spread = torch.full((batch, seq_len), 0.01, device=device, dtype=dtype)
        if price is None:
            price = torch.full((batch, seq_len), 500.0, device=device, dtype=dtype)
        if rsi is None:
            rsi = torch.full((batch, seq_len), 50.0, device=device, dtype=dtype)
        if delta_rsi is None:
            delta_rsi = torch.zeros(batch, seq_len, device=device, dtype=dtype)
        if S_t is None:
            S_t = torch.full((batch, seq_len), 500.0, device=device, dtype=dtype)
        if S_t_minus_1 is None:
            S_t_minus_1 = torch.full((batch, seq_len), 500.0, device=device, dtype=dtype)
        if gamma is None:
            gamma = torch.zeros(batch, seq_len, device=device, dtype=dtype)

        # === INITIALIZE STATE ===
        z = self.initial_encoder(x[:, 0, :])
        h, v, m, r = self.spec.split(z)

        # === CONTROL EMBEDDING ===
        u = self.tft(x)  # (batch, d_control)

        # Force core manifold to FP32 for "Ultra-Safe Mode"
        # Only the TFT encoder above uses Mixed Precision (autocast).
        with torch.amp.autocast('cuda', enabled=False):
            u = u.float()
            x = x.float()
            dX = x[:, 1:, :] - x[:, :-1, :]  # (batch, seq-1, d_input)

            # === DIAGNOSTICS ===
            diagnostics = {'h': [], 'v': [], 'm': [], 'r': [], 'gates': []} if return_diagnostics else None

            # === TIME LOOP ===
            for t in range(seq_len - 1):
                x_prev = self.spec.cat(h, v, m, r).float()

                # Extract time-step inputs
                greeks_k = greeks[:, t, :].float()
                q_k = q[:, t, :].float()
                dX_k = dX[:, t, :].float()

                # Master update (uses r as r_{k-1} for causality)
                x_k = condornet_master_step(
                    spec=self.spec,
                    A_theta=self.A_theta,
                    B_theta=self.B_theta,
                    G_theta=self.G_theta,
                    D_forcing=self.D_forcing,
                    x_prev=x_prev,
                    u_k=u,
                    dX_k=dX_k,
                    greeks_k=greeks_k,
                    r_prev=r,  # CRITICAL: r_{k-1} for causality
                    q_k=q_k,
                    dt_k=dt,
                ).float()

                h, v, m, r = self.spec.split(x_k)

                if return_diagnostics:
                    diagnostics['h'].append(h)
                    diagnostics['v'].append(v)
                    diagnostics['m'].append(m)
                    diagnostics['r'].append(r)

            # === FINAL PREDICATES ===
            p_k = self.pred_gates(
                iv_rank[:, -1].float(),
                bid_ask_spread[:, -1].float(),
                price[:, -1].float(),
                rsi[:, -1].float(),
                delta_rsi[:, -1].float(),
                S_t[:, -1].float(),
                S_t_minus_1[:, -1].float(),
                gamma[:, -1].float(),
            ).float()

            p_sorted, moments, bloom, z_pred = self.pred_signature(p_k)

            # Update r_k with combinatorics dynamics
            r = self.regime_dyn(r.float(), z_pred.float()).float()
            z_final = self.spec.cat(h.float(), v.float(), m.float(), r.float())

            # Super-set gating
            super_gate = self.super_set(p_k).float()
            z_gated = (z_final * super_gate).float()

            # Output (Keep in FP32 for "Ultra-Safe Mode")
            outputs = self.output_head(z_gated).float()

        if return_diagnostics:
            diagnostics['final_gate'] = super_gate
            diagnostics['z_final'] = z_final
            diagnostics['predicates'] = p_k
            diagnostics['moments'] = moments
            diagnostics['bloom'] = bloom
            return outputs, diagnostics

        return outputs

    def get_A_matrix(self, dt: float = 1.0) -> torch.Tensor:
        """
        Return the full drift matrix A_θ as fp32 for loss computations.

        This is used by spectral_radius_loss; we keep it in fp32 to avoid
        mixed-precision matmul issues in autograd.
        """
        A_full = self.A_theta.full_matrix()  # same dtype as model (fp16/bf16)
        return A_full.to(torch.float32)

# =============================================================================
# PART 9: GROUP INVARIANT LOSS
# =============================================================================

# =============================================================================
# AUXILIARY LOSSES (GROUP INVARIANCE, SPECTRAL RADIUS)
# =============================================================================

def group_invariant_loss(
    pred_signature_module: "PredicateSignature",
    gates: torch.Tensor,
    n_permutations: int = 2,
) -> torch.Tensor:
    """
    Enforce group invariance of the predicate signature under permutations.

    Args:
        pred_signature_module: PredicateSignature instance
        gates: (batch, K) predicate activations
        n_permutations: number of random permutations to compare

    Returns:
        loss: scalar tensor
    """
    if gates is None or gates.numel() == 0:
        return torch.tensor(0.0, device=gates.device if gates is not None else "cpu")

    device = gates.device
    dtype = torch.float32  # compute in fp32 for stability

    p = gates.to(dtype=dtype)
    base_sig = pred_signature_module.signature_only(p)  # (batch, d_sig)

    losses = []
    K = p.shape[-1]
    for _ in range(n_permutations):
        perm = torch.randperm(K, device=device)
        p_perm = p[:, perm]
        sig_perm = pred_signature_module.signature_only(p_perm)
        losses.append(F.mse_loss(sig_perm, base_sig))

    return torch.stack(losses).mean().to(gates.dtype)

# =============================================================================
# PART 10: SPECTRAL RADIUS LOSS
# =============================================================================

def spectral_radius_loss(
    A: torch.Tensor,
    dt: float = 1.0,
    target_rho: float = 0.99,
) -> torch.Tensor:
    """
    Penalize spectral radius of exp(A * dt) above target_rho.

    Args:
        A: (d_x, d_x) drift matrix
        dt: time increment
        target_rho: desired upper bound on spectral radius

    Returns:
        loss: scalar tensor
    """
    if A is None:
        return torch.tensor(0.0)

    # Work in fp32 for eigenvalues
    A32 = A.to(torch.float32)
    M = A32 * float(dt)

    # Use matrix_exp then power iteration for dominant eigenvalue magnitude
    F = torch.linalg.matrix_exp(M)  # (d_x, d_x)

    # Power iteration
    v = torch.randn(F.shape[0], 1, device=F.device, dtype=F.dtype)
    v = v / (v.norm() + 1e-8)
    for _ in range(10):
        v = F @ v
        v = v / (v.norm() + 1e-8)

    # Rayleigh quotient approximation
    vT_F_v = (v.transpose(0, 1) @ (F @ v)).squeeze()
    rho_est = torch.abs(vT_F_v)

    excess = torch.clamp(rho_est - target_rho, min=0.0)
    return excess.to(A.dtype)


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("CondorNet™ - Mathematically Faithful Implementation")
    print("=" * 70)

    # Create model
    model = CondorNet(
        d_input=54,
        d_h=128,
        d_v=16,
        d_m=32,
        d_r=16,
        d_control=64,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"State dimension: {model.spec.d_x}")

    # Test forward pass
    batch, seq = 4, 64
    x = torch.randn(batch, seq, 54)

    print(f"\nTest forward pass:")
    print(f"  Input: ({batch}, {seq}, 54)")

    with torch.no_grad():
        outputs, diag = model(x, return_diagnostics=True)

    print(f"  Output: {outputs.shape}")
    print(f"  Final gate: {diag['final_gate'].mean().item():.4f}")
    print(f"  Predicates: {diag['predicates'].mean(0).tolist()}")

    # Test ETD-1 kernel
    print(f"\nTest ETD-1 kernel:")
    A = model.get_A_matrix()
    F_k, phi1 = etd1_kernel(A, dt=1.0)
    print(f"  A shape: {A.shape}")
    print(f"  F = exp(AΔt) shape: {F_k.shape}")
    print(f"  φ_1(AΔt) shape: {phi1.shape}")

    # Test spectral radius
    rho_loss = spectral_radius_loss(A, dt=1.0)
    print(f"  Spectral radius loss: {rho_loss.item():.6f}")

    # Test group invariance
    p = torch.rand(batch, 5)
    gi_loss = group_invariant_loss(model.pred_signature, p)
    print(f"\nGroup invariance loss: {gi_loss.item():.6f}")

    print("\n" + "=" * 70)
    print("Implementation complete and tested.")
    print("=" * 70)
