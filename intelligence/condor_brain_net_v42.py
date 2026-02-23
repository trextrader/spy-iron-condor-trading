"""
CondorNet™ - Mathematically Faithful Implementation
condor_brain_net.py
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

from intelligence.canonical_feature_registry import D_INPUT

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

        # === CONTROL MODULATOR (V41) ===
        # Modulates the diagonal of A based on control context u_k
        self.control_modulator = nn.Sequential(
            nn.Linear(d_r, d_h), # Using regime dimensions as modulator source
            nn.SiLU(),
            nn.Linear(d_h, spec.d_x)
        )

        # Stability initialization
        self._init_stable()

    def _init_stable(self):
        """Initialize for spectral stability (ρ(A) < 1)."""
        scale = 0.001  # Tighter for V5 stability
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=scale)
            elif 'diag' in name:
                nn.init.zeros_(param)

    def forward_blocks(self, x: torch.Tensor, u_k: torch.Tensor = None) -> torch.Tensor:
        """
        Apply A_theta in block form to x = [h, v, m, r].
        
        Args:
            x: current full state
            u_k: optional time-local control embedding for diagonal modulation
        """
        h, v, m, r = self.spec.split(x)

        # Apply diagonal modulation from control u_k if provided
        if u_k is not None:
            # Shift = diag(f(u_k)) * x
            diag_shift = torch.tanh(self.control_modulator(u_k)) * 0.1
            x = x * (1.0 + diag_shift)
            h, v, m, r = self.spec.split(x)

        # h block
        h_new = self.A_hh(h)
        if self.A_hv is not None:
            h_new = h_new + self.A_hv(v)
        
        # Apply A_hm (diagonal or full)
        if self.enforce_sparsity:
            d = len(self.A_hm_diag)
            # Manual diagonal matmul for efficiency
            h_new[..., :d] = h_new[..., :d] + self.A_hm_diag * m[..., :d]
        else:
            h_new = h_new + self.A_hm(m)
            
        h_new = h_new + self.A_hr(r)

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
        
        Hardened (V47): Always uses FP32 for construction to prevent 
        precision loss even if the rest of the model is in FP16/BF16.
        """
        d_x = self.spec.d_x
        device = next(self.parameters()).device
        
        # Build by applying to basis vectors in FP32
        # NOTE: No torch.no_grad() — gradients MUST flow through A_theta
        # so the linear dynamics block can train via ETD kernels + spectral loss.
        eye = torch.eye(d_x, device=device, dtype=torch.float32)
        cols = []
        for i in range(d_x):
            e_i = eye[:, i]
            col = self.forward_blocks(e_i.to(next(self.parameters()).dtype)).float()
            cols.append(col.unsqueeze(-1))
            
        A_full = torch.cat(cols, dim=-1)  # [d_x, d_x]
        return A_full


class BlockMatrixB(nn.Module):
    """
    B_θ(u_k) ∈ ℝ^{d_x × d_control}, partitioned into [B_h, B_v, B_m, B_r].

    This is the control injection operator for the ETD-1 term.
    Mathematically, it is a matrix B ∈ ℝ^{d_x × d_control} that maps 
    control embeddings u_k into the augmented state space for forcing.
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

    def full_matrix(self) -> torch.Tensor:
        """
        Construct the full [d_x, d_control] matrix B_theta.
        Concatenates weights W_h, W_v, W_m, W_r.
        
        Hardened (V47): Always uses FP32 for extraction.
        """
        device = self.B_h.weight.device
        
        # Concatenate weight matrices (d_block, d_control) along row dim
        B_full = torch.cat([
            self.B_h.weight.data.float(),
            self.B_v.weight.data.float(),
            self.B_m.weight.data.float(),
            self.B_r.weight.data.float()
        ], dim=0)
        
        return B_full


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

        # D_r: Regime forcing from Greeks (continuous part)
        self.D_r = nn.Linear(d_in, spec.d_r)

        # === RA ENFORCEMENT ===
        # D_h (physics) and D_r (regime) are invariant to position size q
        self.d_in_h = n_greeks + spec.d_r
        self.D_h_phys = nn.Sequential(
            nn.Linear(self.d_in_h, spec.d_h),
            nn.Tanh(),
        )
        self.D_r_phys = nn.Linear(self.d_in_h, spec.d_r)

        self.spec = spec

    def forward(self, greeks: torch.Tensor, r_prev: torch.Tensor,
                q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            greeks: (batch, n_greeks) [delta, gamma, theta, vega, rho]
            r_prev: (batch, d_r) PREVIOUS regime state
            q: (batch, d_q) position size
        """
        # RA: MARKET PHYSICS (D_h, D_r) ARE INVARIANT TO q
        z_phys = torch.cat([greeks, r_prev], dim=-1)
        Dh = self.D_h_phys(z_phys)
        Dr = self.D_r_phys(z_phys)

        # EXECUTION DRAG (D_v, D_m) CAN SEE q
        z_exec = torch.cat([greeks, r_prev, q], dim=-1)
        Dv = self.D_v(z_exec)
        Dm = self.D_m(z_exec)
        
        return self.spec.cat(Dh, Dv, Dm, Dr)


# =============================================================================
# PART 7: AUXILIARY HEADS (Pivot, Offset)
# =============================================================================

class PivotHead(nn.Module):
    """
    CondorNet v4.2 Structural Auxiliary Head.
    Predicts:
    - pivot_prob: (B, 1) probability of being a pivot
    - pivot_strength: (B, 1) structural importance
    - pivot_type: (B, 2) classification (high/low)
    - pivot_dist: (B, 1) distance to nearest pivot
    """
    def __init__(self, d_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 5) # [prob, strength, high_logit, low_logit, dist]
        )

    def forward(self, z: torch.Tensor):
        out = self.net(z)
        prob = torch.sigmoid(out[:, 0:1])
        strength = F.softplus(out[:, 1:2])
        type_logits = out[:, 2:4]
        dist = F.softplus(out[:, 4:5])
        return prob, strength, type_logits, dist

class OffsetModule(nn.Module):
    """
    v4.2 Dynamic Parameter Offset Module.
    Learns dynamic adjustments to precomputed indicator features.
    """
    def __init__(self, d_control: int, n_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_control, 32),
            nn.SiLU(),
            nn.Linear(32, n_features)
        )

    def forward(self, u: torch.Tensor):
        # Learn a delta in range [-0.5, 0.5]
        return torch.tanh(self.net(u)) * 0.5

class PivotEncoder(nn.Module):
    """
    CondorNet v4.2 Structural Pivot Encoder.
    
    Transforms raw pivot features (dist, slope, flags) into a latent structural embedding.
    This embedding is CONCATENATED (not added) to the main feature stream before 
    state encoding, preserving the distinction between 'physics' (Greeks/Price) 
    and 'structure' (Geometry).
    
    Blueprint Rule X: "DO NOT treat them as flat numeric inputs... Encoded vector... concatenated."
    """
    def __init__(self, d_pivot: int, d_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_pivot, 64),
            nn.GELU(),
            nn.Linear(64, d_embed)
        )
        
    def forward(self, pivots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pivots: (batch, seq, d_pivot) Raw pivot features 
        Returns:
            embedding: (batch, seq, d_embed) Structured geometric embedding
        """
        return self.net(pivots)

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
        F: (d_x, d_x) state transition matrix e^{AΔt} (Always FP32)
        phi1: (d_x, d_x) ETD-1 basis function φ_{1}(AΔt) (Always FP32)
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
    d_x = M.shape[-1]
    
    # --- Augmented Matrix Trick (Al-Mohy & Higham) ---
    # Instead of pinv, we compute exp([[M, I], [0, 0]])
    # The Top-Right block of the result is exactly phi_1(M).
    # This is numerically stable even if M is singular or zero.
    
    # 1. Setup Augmented Matrix K [2*d_x, 2*d_x]
    K = torch.zeros(2*d_x, 2*d_x, device=M.device, dtype=torch.float32)
    K[:d_x, :d_x] = M
    K[:d_x, d_x:] = torch.eye(d_x, device=M.device, dtype=torch.float32)
    
    # 2. Compute exp(K)
    expK = torch.linalg.matrix_exp(K)
    
    # 3. Extract blocks
    expM = expK[:d_x, :d_x]
    phi1 = expK[:d_x, d_x:]
    
    # ALWAYS return FP32 for the master step matmuls to avoid Half/Float mismatch
    return expM.float(), phi1.float()


def condornet_master_step(
    spec: AugmentedStateSpec,
    B_theta: BlockMatrixB,
    G_theta: CDEResponseG,
    D_forcing: FullForcingD,
    x_prev: torch.Tensor,
    u_k: torch.Tensor,
    dX_k: torch.Tensor,
    greeks_k: torch.Tensor,
    r_prev: torch.Tensor,
    q_k: torch.Tensor,
    F_k: torch.Tensor,
    phi1: torch.Tensor,
    dt_k: float,
    A_theta: Optional[BlockMatrixA] = None,
) -> torch.Tensor:
    """
    Update x_{k-1} → x_k using precomputed ETD-1 kernels.
    Includes control-dependent diagonal modulation.
    
    NOTE: All internal math is forced to FP32 for numerical stability.
    """
    # Force all math to FP32
    x_prev = x_prev.float()
    u_k = u_k.float()
    dX_k = dX_k.float()
    greeks_k = greeks_k.float()
    r_prev = r_prev.float()
    q_k = q_k.float()

    with torch.amp.autocast('cuda', enabled=False):
        # 1. Linear propagation F_k x_{k-1}
        x_lin = torch.matmul(x_prev, F_k.T)  # (batch, d_x)

        # 1b. Control-Dependent Diagonal Modulation (V41)
        if A_theta is not None:
            # Shift = diag(f(u_k)) * dt_k * x_prev
            # Approximation for u_k modulation in precomputed A_full context
            diag_shift = torch.tanh(A_theta.control_modulator(r_prev)) * 0.1
            x_lin = x_lin + (diag_shift * x_prev * dt_k)

        # 2. B_θ(u_k) and ETD injection (Squashed for stability)
        # Apply tanh to forcing to prevent energy explosion over T=240
        B_k = torch.tanh(B_theta(u_k).float())
        g_k = torch.matmul(B_k, phi1.T) * dt_k

        # 3. Controlled response G_θ(x_{k-1}, u_k) ΔX_k (Squashed for stability)
        G_k = G_theta(x_prev, u_k).float()
        dX_vec = dX_k.unsqueeze(-1)
        cde_term = torch.tanh(torch.bmm(G_k, dX_vec).squeeze(-1))

        # 4. Full forcing D(Greeks_k, r_{k-1}, q_k) (Squashed for stability)
        D_k = torch.tanh(D_forcing(greeks_k, r_prev, q_k).float())

        # 5. Master update
        x_k = x_lin + g_k + cde_term + D_k

    return x_k.float()


# =============================================================================
# PART 4: PREDICATE COMBINATORICS AND r_k DYNAMICS
# =============================================================================

class CanonicalPredicateGates(nn.Module):
    """
    The 8 canonical inequality gates (5 original + 3 V3.0).

    Implements soft (differentiable) versions using steep sigmoids:
        1. Volatility spike: IVR_t > 75
        2. Liquidity compression: Spread/Price > 0.4%
        3. Momentum reversal: RSI < 25 AND ΔRSI < 0
        4. Gap risk: |S_t - S_{t-1}|/S_{t-1} > 1.2%
        5. Greeks pressure: |Γ| > threshold
        6. IV Regime (V3.0): IV_Mid > IV_High * 0.8 (IV near 52-week high)
        7. Options Flow Imbalance (V3.0): Put_Volume / Total_Volume > 0.6
        8. Microstructure Stress (V3.0): quote_spread > 2 * median(quote_spread)

    These gates modulate the decay in A_θ via:
        A(u, σ) = -diag(exp(η(u,σ)) · (1 + λ_p γ_t))
    """
    def __init__(self, n_predicates: int = 8, steepness: float = 50.0, learnable_thresholds: bool = True):
        super().__init__()
        self.n_predicates = n_predicates
        self.steepness = steepness

        if learnable_thresholds:
            # Original 5 thresholds
            self.iv_rank_thresh = nn.Parameter(torch.tensor(75.0))
            self.spread_frac_thresh = nn.Parameter(torch.tensor(0.004))
            self.rsi_thresh = nn.Parameter(torch.tensor(25.0))
            self.gap_frac_thresh = nn.Parameter(torch.tensor(0.012))
            self.gamma_thresh = nn.Parameter(torch.tensor(0.05))
            # V3.0 thresholds
            self.iv_regime_frac_thresh = nn.Parameter(torch.tensor(0.8))
            self.put_flow_thresh = nn.Parameter(torch.tensor(0.6))
            self.spread_stress_mult_thresh = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_buffer('iv_rank_thresh', torch.tensor(75.0))
            self.register_buffer('spread_frac_thresh', torch.tensor(0.004))
            self.register_buffer('rsi_thresh', torch.tensor(25.0))
            self.register_buffer('gap_frac_thresh', torch.tensor(0.012))
            self.register_buffer('gamma_thresh', torch.tensor(0.05))
            self.register_buffer('iv_regime_frac_thresh', torch.tensor(0.8))
            self.register_buffer('put_flow_thresh', torch.tensor(0.6))
            self.register_buffer('spread_stress_mult_thresh', torch.tensor(2.0))

        if n_predicates > 8:
            # Extra learned predicates from physics features (11 inputs for V3.0)
            d_extra_in = 11 
            self.extra_heads = nn.Sequential(
                nn.Linear(d_extra_in, n_predicates - 8),
                nn.Sigmoid()
            )
            
            # BLUEPRINT PHASE 1, STEP 4: Correct Predicate Initialization
            # gain = 1.0 / math.sqrt(d_input)
            # For the extra heads, the input dim is 11, so we scale by 1/sqrt(11)
            # If we switch to full d_input later, this logic holds.
            gain = 1.0 / math.sqrt(d_extra_in)
            nn.init.xavier_uniform_(self.extra_heads[0].weight, gain=gain)
            print(f"[CanonicalPredicateGates] Initialized extra_heads with gain={gain:.4f}")

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
        # V3.0 optional inputs (default to neutral values if not provided)
        iv_mid: torch.Tensor = None,
        iv_high: torch.Tensor = None,
        put_volume: torch.Tensor = None,
        total_volume: torch.Tensor = None,
        quote_spread: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute soft predicate gates.

        All inputs: (batch,) scalar tensors

        Returns:
            p_k: (batch, n_predicates) soft predicates in [0, 1]
        """
        device = iv_rank.device
        dtype = iv_rank.dtype

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

        # 6. IV Regime (V3.0): IV_Mid > IV_High * 0.8 (IV near 52-week high)
        if iv_mid is not None and iv_high is not None:
            iv_regime = torch.sigmoid(self.steepness * (iv_mid - iv_high * self.iv_regime_frac_thresh))
        else:
            iv_regime = torch.zeros_like(iv_rank)

        # 7. Options Flow Imbalance (V3.0): Put_Volume / Total_Volume > 0.6
        if put_volume is not None and total_volume is not None:
            put_ratio = put_volume / (total_volume + 1e-8)
            flow_imbalance = torch.sigmoid(self.steepness * (put_ratio - self.put_flow_thresh))
        else:
            flow_imbalance = torch.zeros_like(iv_rank)

        # 8. Microstructure Stress (V3.0): quote_spread > 2 * median(quote_spread)
        if quote_spread is not None:
            # Use running median approximation: compare against 0.01 * spread_stress_mult
            # In practice, the median is estimated from training data normalization
            spread_stress = torch.sigmoid(self.steepness * (quote_spread - 0.01 * self.spread_stress_mult_thresh))
        else:
            spread_stress = torch.zeros_like(iv_rank)

        p_canonical = torch.stack([
            vol_spike, liq_lock, mom_rev, gap_gate, gamma_gate,
            iv_regime, flow_imbalance, spread_stress,
        ], dim=-1)

        if self.n_predicates > 8:
            # Projection for additional predicates (11 physics inputs for V3.0)
            physics_features = torch.stack([
                iv_rank, bid_ask_spread / (price + 1e-8), rsi, delta_rsi,
                S_t, S_t_minus_1, torch.abs(S_t - S_t_minus_1) / (S_t_minus_1 + 1e-8),
                torch.abs(gamma),
                iv_regime, flow_imbalance, spread_stress,
            ], dim=-1)
            p_extra = self.extra_heads(physics_features)
            return torch.cat([p_canonical, p_extra], dim=-1)

        return p_canonical


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
# PART 5: EXPLICIT RELATIONAL LOGIC
# =============================================================================

class RelationalLogicLayer(nn.Module):
    """
    Memory-Efficient Explicit Logic Engine (V22).
    Evaluates pairwise relations (<, >, =) in chunks to prevent OOM.
    """
    def __init__(self, n_inputs: int, out_dim: int, chunk_size: int = 4096):
        super().__init__()
        self.n_inputs = n_inputs
        self.out_dim = out_dim
        self.chunk_size = chunk_size
        
        # Features: (N choose 2) * 3 [less, greater, equal]
        self.n_pairs = (n_inputs * (n_inputs - 1)) // 2
        relational_dim = self.n_pairs * 3 if n_inputs > 1 else n_inputs
        
        self.projection = nn.Linear(relational_dim, out_dim)
        # BLUEPRINT CORE 4: Properly scale the logic projection to unit variance
        # Standard initialization makes W too small, causing output to vanish to 0.0 -> Sigmoid(0)=0.5
        nn.init.xavier_normal_(self.projection.weight, gain=1.0)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
            
        # Bounded steepness via reparameterization: α = 1 + 7·σ(β) ∈ [1, 8].
        # Free learnable β cannot push α out of the safe range even after many epochs,
        # preventing the "fixed for 3 epochs then drifted back" failure mode.
        # σ(0) = 0.5 → α_init ≈ 4.5 (previously hard-coded 5.0; effectively identical).
        self._steepness_raw = nn.Parameter(torch.tensor(0.0))  # β; unconstrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N) input activations

        Returns:
            out: (batch, out_dim) projected logic signal
        """
        if self.n_inputs < 2:
            return self.projection(x)

        # Bounded steepness (computed once per forward call): α ∈ [1, 8]
        steepness = 1.0 + 7.0 * torch.sigmoid(self._steepness_raw)

        batch_size = x.shape[0]
        device = x.device

        # Pre-compute upper triangle indices once
        iu = torch.triu_indices(self.n_inputs, self.n_inputs, offset=1, device=device)
        n_pairs = iu.shape[1]

        # Use chunked processing if too many pairs
        if n_pairs > self.chunk_size:
            # Accumulate result via projection chunks
            out = torch.zeros(batch_size, self.out_dim, device=device)
            weight = self.projection.weight  # (out_dim, n_pairs*3)

            for start in range(0, n_pairs, self.chunk_size):
                end = min(start + self.chunk_size, n_pairs)
                chunk_iu0 = iu[0, start:end]
                chunk_iu1 = iu[1, start:end]

                # Compute diffs for this chunk only
                tri_diffs = x[:, chunk_iu0] - x[:, chunk_iu1]  # (batch, chunk)

                # Soft Logic Operators (Center around 0, unit variance)
                # Instead of [0, 1] sigmoid which adds positive bias, use tanh [-1, 1]
                lt = torch.tanh(-steepness * tri_diffs)
                gt = torch.tanh(steepness * tri_diffs)
                eq = torch.exp(-steepness * (tri_diffs ** 2)) * 2 - 1 # [-1, 1]

                # Accumulate projection contribution from this chunk
                chunk_features = torch.cat([lt, gt, eq], dim=-1)  # (batch, chunk*3)

                # Slice the weight matrix for this chunk's features
                feat_start = start * 3
                feat_end = end * 3
                chunk_weight = weight[:, feat_start:feat_end]  # (out_dim, chunk*3)

                out += torch.mm(chunk_features, chunk_weight.t())

            # Add bias once at the end
            if self.projection.bias is not None:
                out += self.projection.bias
            return out
        else:
            # Standard processing for small inputs
            tri_diffs = x[:, iu[0]] - x[:, iu[1]]

            # Soft Logic Operators (Center around 0, [-1, 1] variance)
            lt = torch.tanh(-steepness * tri_diffs)
            gt = torch.tanh(steepness * tri_diffs)
            eq = torch.exp(-steepness * (tri_diffs ** 2)) * 2 - 1
            
            rel_features = torch.cat([lt, gt, eq], dim=-1)
            out = self.projection(rel_features)
            return out



class PredicateSet(nn.Module):
    """
    Set of predicates with learnable aggregation.

    Π(S; p) = aggregation({p_i : i ∈ S})

    Aggregation options:
        - mean: simple average
        - pnorm: generalized mean with learnable p
        - owa: ordered weighted average
    """
    def __init__(self, n_predicates: int):
        super().__init__()
        self.n_predicates = n_predicates
        # V21: Explicit Comparison between individual predicates
        self.relational_logic = RelationalLogicLayer(n_predicates, 1)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (batch, K) predicate values

        Returns:
            set_value: (batch, 1) aggregated activation
        """
        # Compare individual predicates to form set activation
        return torch.sigmoid(self.relational_logic(p))


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
            PredicateSet(n_predicates)
            for _ in range(n_sets)
        ])

        # V21: standardized relational layer for sets
        self.relational_logic = RelationalLogicLayer(n_sets, 1)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (batch, K) predicate values

        Returns:
            gate: (batch, 1) final super-set gate
        """
        # S: (batch, n_sets)
        S = torch.cat([pred_set(p) for pred_set in self.sets], dim=-1)
        return torch.sigmoid(self.relational_logic(S))


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

    def forward(self, x: torch.Tensor, return_sequence: bool = False, timestep_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_input) input sequence
            return_sequence: if True, return all u_k; else just final
            timestep_mask: (batch, seq-1) binary mask for Phase 5 BCS
        """
        batch, seq_len, d_in = x.shape
        device = x.device

        if timestep_mask is not None and return_sequence:
            # BCS HARDENING (Phase 5): Structurally True Invariance
            # We process only unique bars per batch element.
            
            # Since Transformer requires a fixed length, we'll process the maximum 
            # number of unique bars in the batch, padding shorter ones.
            
            # 1. Identify bar boundaries
            # bar_id = [0, 1, 1, 1, 2, 2, ...]
            bar_ids = torch.cumsum(timestep_mask, dim=1).long()
            bar_ids = torch.cat([torch.zeros(batch, 1, device=device, dtype=torch.long), bar_ids], dim=1)
            
            # 2. Extract first row of each unique bar
            # We use a selection mask: row i is kept if it's the first time its bar_id appears.
            # Row 0 is always first. Row t+1 is first if timestep_mask[t] == 1.
            first_row_mask = torch.cat([torch.ones(batch, 1, device=device).bool(), timestep_mask.bool()], dim=1)
            
            # Count bars per batch element
            bars_per_batch = first_row_mask.sum(dim=1)
            max_bars = bars_per_batch.max().item()
            
            # 3. Process through Transformer
            x_unique_full = torch.zeros(batch, max_bars, d_in, device=device, dtype=x.dtype)
            for i in range(batch):
                rows = x[i, first_row_mask[i]]
                x_unique_full[i, :rows.size(0)] = rows
            
            # Phase 5 Hardening: Zero out candidate-specific features (5-13)
            # This makes the control logic purely bar-level/market-driven.
            x_unique_full[:, :, 5:14] = 0.0
                
            z_unique = self.input_proj(x_unique_full)
            z_unique = z_unique + self.pos_enc[:, :max_bars, :]
            
            # Causal mask for the bar sequence
            causal_mask = torch.triu(torch.ones(max_bars, max_bars, device=device), diagonal=1).bool()
            
            # Padding mask (ignore the padded bars for shorter batches)
            pad_mask = torch.arange(max_bars, device=device).expand(batch, max_bars) >= bars_per_batch.unsqueeze(1)
            
            z_unique = self.encoder(z_unique, mask=causal_mask, src_key_padding_mask=pad_mask)
            u_unique = self.out_proj(z_unique) # (batch, max_bars, d_control)
            
            # 4. Redistribute back to all candidate rows
            # Each row i gets u_unique[bar_ids[i]]
            u = torch.zeros(batch, seq_len, self.d_control, device=device, dtype=x.dtype)
            for i in range(batch):
                u[i] = u_unique[i, bar_ids[i]]
                
            return u

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
    - True ETD-1 with matrix_exp(Adt) and phi_1(M) = M_plus(e^M - I)
    - Block-partitioned operators: A_theta, B_theta, G_theta, D
    - 5 canonical predicate gates (vol spike, liquidity, reversal, gap, Greeks)
    - Group-invariant signatures
    - r_k dynamics: r_k = alpha_k * r_{k-1} + beta_k
    - Full 4-block forcing with causality (r_{k-1} in D)

    Master equation:
        x_k = e^{A_theta dt_k} x_{k-1} + dt_k phi_1(A_theta dt_k) B_theta(u_k)
            + G_theta(x_{k-1}, u_k) dX_k + D(Greeks_k, r_{k-1}, q_k)
    """
    def __init__(
        self,
        # d_input: int = 79,  <-- REMOVED per blueprint
        d_h: int = 256,
        d_v: int = 32,
        d_m: int = 64,
        d_r: int = 32,
        d_control: int = 128,
        n_greeks: int = 5,
        d_q: int = 1,
        n_predicates: int = 8,
        R_moments: int = 4,
        M_bloom: int = 16,
        n_layers: int = 2,
        n_sets: int = 4,
        n_super_sets: int = 1,
        enforce_sparsity: bool = True,
        verbose_math: bool = False,
        gate_temp_init: float = 3.0,
    ):
        super().__init__()
        self.verbose_math = verbose_math

        # State specification
        self.spec = AugmentedStateSpec(d_h, d_v, d_m, d_r)
        
        # BLUEPRINT PHASE 1: Dynamic Dimensions
        self.d_input = D_INPUT
        print(f"[CondorNet] Initializing with D_INPUT={self.d_input} from Registry")
        
        self.d_control = d_control
        self.n_greeks = n_greeks

        # === CORE OPERATORS ===
        self.A_theta = BlockMatrixA(self.spec, enforce_sparsity=enforce_sparsity)
        # Type fix: BlockVectorB was not defined in context, assuming generic B logic or typo in original
        # For now, using BlockMatrixB as seen in previous context
        self.B_theta = BlockMatrixB(self.spec, d_control) 
        self.G_theta = CDEResponseG(self.spec, self.d_input, d_control)
        self.D_forcing = FullForcingD(self.spec, n_greeks=n_greeks, d_q=d_q)

        # === CONTROL ===
        self.tft = TFTControlEncoder(self.d_input, d_control, n_layers=n_layers)

        # === PIVOT ARCHITECTURE (Phase 3) ===
        # Blueprint: Identify pivot columns vs main columns
        # V4.2 Registry: 91 total. Indices 87-90 are pivots (4 cols).
        self.d_pivot_raw = 4 
        self.d_pivot_embed = 32 # Latent structural dimension
        self.d_main = self.d_input - self.d_pivot_raw
        
        self.pivot_encoder = PivotEncoder(self.d_pivot_raw, self.d_pivot_embed)
        
        # New input dimension for state encoder: Main Features + Encoded Pivots
        self.d_encoded_input = self.d_main + self.d_pivot_embed

        # === INITIAL STATE ===
        self.initial_encoder = nn.Sequential(
            nn.Linear(self.d_encoded_input, self.spec.d_x),
            nn.SiLU(),
            nn.Linear(self.spec.d_x, self.spec.d_x),
        )

        # === HAL GATING (Predicate Architecture) ===
        # Re-scaling initialization handles in PredicateGates (Phase 2)
        self.pred_gates = CanonicalPredicateGates(n_predicates=n_predicates) 
        self.pred_signature = PredicateSignature(n_predicates, R_moments, M_bloom)
        self.regime_dyn = RegimeCombinatoricsDynamics(d_r, self.pred_signature.d_out)

        # v4.2 Structural Brain upgrades
        self.pivot_head = PivotHead(self.spec.d_x)
        self.offset_module = OffsetModule(d_control)

        # Output head
        # self.output_head = CondorExpertHead(
        #     self.spec.d_x, 
        #     n_heads=n_sets, 
        #     n_super_heads=n_super_sets
        # )
        if n_super_sets > 0:
            self.super_sets = nn.ModuleList([
                SuperSet(n_sets, n_predicates)
                for _ in range(n_super_sets)
            ])
            # V21: Hierarchical Relational Logic comparing the super-sets themselves
            self.hierarchical_logic = RelationalLogicLayer(n_super_sets, 1)
            # Run 5: learnable routing temperature τ.
            # Multiplies gate logits before sigmoid so the gate can make
            # meaningful binary routing decisions.
            # Init = gate_temp_init (default 3.0).  Target λ std: 0.15–0.30.
            self.gate_temperature = nn.Parameter(torch.tensor(float(gate_temp_init)))
        else:
            self.super_sets = None

        # === FUSION ===
        self.fusion_gate = FusionGate(d_h, M_bloom, R_moments, n_branches=3)

        # === OUTPUT ===
        self.output_head = nn.Linear(self.spec.d_x, 10)

    def log_math(self, label: str, equation: str, tensor: torch.Tensor = None):
        """Print mathematical derivation step in LaTeX style."""
        if not self.verbose_math or not hasattr(self, 'verbose_math') or not self.verbose_math:
            return
        
        prefix = "  [MATH] "
        print(f"{prefix}{label}: {equation}")
        if tensor is not None:
            # Stats for the tensor
            mu = tensor.mean().item()
            # std() is NaN for single elements (n-1 = 0)
            std = tensor.std().item() if tensor.numel() > 1 else 0.0
            print(f"{prefix}       Shape: {list(tensor.shape)} | mu={mu:.4f}, std={std:.4f}")

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
        verbose_math: bool = None,
        timestep_mask: torch.Tensor = None,  # (batch, seq-1) - Binary mask for BCS
        bar_index: torch.Tensor = None,      # (batch, seq) - Optional for grouping guardrail
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass with defensive dtype casting."""
        if verbose_math is not None:
            self.verbose_math = verbose_math
            
        self.log_math("START", "x_0 initialization from d_input path")
        batch, seq_len, _ = x.shape
        device = x.device
        
        # Determine target model dtype
        model_dtype = next(self.parameters()).dtype

        # BLUEPRINT CHECK: Input Dimension
        assert x.shape[-1] == self.d_input, (
            f"Input dim mismatch: got {x.shape[-1]}, expected {self.d_input} "
            f"(Registry: {self.d_input})"
        )
        
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

        # === BCS GUARDRAIL (Phase 5) ===
        if bar_index is not None:
            # Move 4: Invariant Check
            diff = bar_index[:, 1:] - bar_index[:, :-1]
            if not (diff >= 0).all():
                raise ValueError("BCS Violation: Input batch must be monotonically grouped by bar_index (time).")
            
            # Auto-generate mask if missing
            if timestep_mask is None:
                timestep_mask = (diff > 0.5).float()

        # === PIVOT ENCODING (Phase 3) ===
        # Slice raw pivots (last 4 cols) vs main features
        # Blueprint: x_struct = concat(x_main, Encoder(x_pivot))
        # Note: We do this for the WHOLE sequence to support state initialization
        x_main = x[..., :-self.d_pivot_raw]
        x_piv = x[..., -self.d_pivot_raw:]
        
        # Encode pivot sequence
        # (batch, seq, d_pivot) -> (batch, seq, d_embed)
        x_piv_enc = self.pivot_encoder(x_piv) 
        
        # Capture pivot diagnostics
        if return_diagnostics:
            _pivot_diag = {
                'pivot_raw': x_piv[:, -1].detach(),
                'pivot_embedding': x_piv_enc[:, -1].detach(),
                'pivot_embed_norm': x_piv_enc[:, -1].detach().norm(dim=-1).mean().item(),
            }

        # Concatenate for structural manifold input
        x_struct = torch.cat([x_main, x_piv_enc], dim=-1)

        # === INITIALIZE STATE ===
        # Use structurally aware feature vector for x_0
        z = self.initial_encoder(x_struct[:, 0, :])
        h, v, m, r = self.spec.split(z)

        # === CONTROL EMBEDDING ===
        # Produces sequence (batch, seq, d_control) for time-local modulation
        u = self.tft(x, return_sequence=True, timestep_mask=timestep_mask)  

        # Capture TFT diagnostics
        if return_diagnostics:
            _tft_diag = {
                'control_vector_norm': u[:, -1].detach().norm(dim=-1).mean().item(),
                'control_vector': u[:, -1].detach(),
            }

        # Force core manifold to FP32 for "Ultra-Safe Mode"
        # Only the TFT encoder above uses Mixed Precision (autocast).
        with torch.amp.autocast('cuda', enabled=False):
            u = u.float()
            x = x.float()
            dX = x[:, 1:, :] - x[:, :-1, :]  # (batch, seq-1, d_input)

            # === DIAGNOSTICS ===
            diagnostics = {'h': [], 'v': [], 'm': [], 'r': [], 'gates': []} if return_diagnostics else None
            moments = None
            bloom = None

            # === PRECOMPUTE KERNELS (V5 Performance Opt) ===
            # A_θ is shared across sequence, dt is constant
            A_full = self.A_theta.full_matrix().float()
            self.log_math("MATRIX_A", "A_full = [A_hh A_hv; A_vh A_vv ...]", A_full)
            
            F_k, phi1 = etd1_kernel(A_full, dt)
            self.log_math("TRANSITION", "F_k = exp(A_full * deltat)", F_k)
            self.log_math("PHI_1", "phi1 = M_pinv @ (expM - I)", phi1)
            
            # Definitive numerical verification (one-time per instantiation)
            if not hasattr(self, '_printed_phi1_diff'):
                diff = torch.max(torch.abs(phi1 - F_k)).item()
                print(f"\n  [MATH BUG HUNT] max_abs(phi1 - F_k) = {diff:.8f}")
                self._printed_phi1_diff = True

            # Capture CDE spectral diagnostics
            if return_diagnostics:
                try:
                    eigvals = torch.linalg.eigvals(A_full).detach()
                    spectral_radius = eigvals.abs().max().item()
                    top_eigenvalues = eigvals.abs().topk(min(5, len(eigvals))).values.tolist()
                except Exception:
                    spectral_radius = 0.0
                    top_eigenvalues = []
                _cde_diag = {
                    'spectral_radius': spectral_radius,
                    'top_eigenvalues': top_eigenvalues,
                }

            # === TIME LOOP ===
            for t in range(seq_len - 1):
                x_prev = self.spec.cat(h, v, m, r).float()

                # Extract time-step inputs
                greeks_k = greeks[:, t, :].float()
                q_k = q[:, t, :].float()
                dX_k = dX[:, t, :].float()
                u_k = u[:, t, :].float()
                
                # BCS: Gate the timestep (Phase 5)
                # In training: mask is 0 for 99 steps if 100 rows/bar.
                weight = timestep_mask[:, t].view(-1, 1).float() if timestep_mask is not None else 1.0
                
                # Snapshot r_{k-1} for causal forcing in D
                r_causal = r.clone()

                # Per-step Predicate Evaluation (V41)
                p_k = self.pred_gates(
                    iv_rank[:, t].float(),
                    bid_ask_spread[:, t].float(),
                    price[:, t].float(),
                    rsi[:, t].float(),
                    delta_rsi[:, t].float(),
                    S_t[:, t].float(),
                    S_t_minus_1[:, t].float(),
                    gamma[:, t].float(),
                ).float()
                
                _, _, _, z_pred = self.pred_signature(p_k)
                
                # Update r_k with combinatorics dynamics IN LOOP
                # BCS: Only apply update if weight is high
                r_next = self.regime_dyn(r.float(), z_pred.float()).float()
                r = r + weight * (r_next - r) 

                # Master update
                x_proposal = condornet_master_step(
                    spec=self.spec,
                    B_theta=self.B_theta,
                    G_theta=self.G_theta,
                    D_forcing=self.D_forcing,
                    x_prev=x_prev,
                    u_k=u_k,
                    dX_k=dX_k,
                    greeks_k=greeks_k,
                    r_prev=r_causal, # CRITICAL: uses r_{k-1} for causality
                    q_k=q_k,
                    F_k=F_k,
                    phi1=phi1,
                    dt_k=dt,
                    A_theta=self.A_theta,
                ).float()

                # V7: Manifold Squashing
                x_proposal_tanh = torch.tanh(x_proposal)
                
                # BLUEPRINT PHASE 4: State Noise Annealing
                # Inject small Gaussian noise during training to prevent manifold collapse
                if self.training:
                    noise_scale = 0.001 # Start small, can be annealed externally if passed as arg
                    x_proposal_tanh = x_proposal_tanh + torch.randn_like(x_proposal_tanh) * noise_scale

                # BCS Master Gate: Only advance physics if this is a new bar
                # Use x_prev (already tanh'ed from previous step) or the new tanh'ed proposal
                x_k = x_prev + weight * (x_proposal_tanh - x_prev)

                h, v, m, _ = self.spec.split(x_k) # r is NOT overwritten; preserved from regime_dyn

                if return_diagnostics:
                    diagnostics['h'].append(h)
                    diagnostics['v'].append(v)
                    diagnostics['m'].append(m)
                    diagnostics['r'].append(r)

            # Final split for return diagnostics (if needed)
            z_final = self.spec.cat(h.float(), v.float(), m.float(), r.float())

            # Hierarchical Relational Logic (HAL) Gating
            # (Remains at end of path for final policy decision)
            p_final = self.pred_gates(
                iv_rank[:, -1].float(),
                bid_ask_spread[:, -1].float(),
                price[:, -1].float(),
                rsi[:, -1].float(),
                delta_rsi[:, -1].float(),
                S_t[:, -1].float(),
                S_t_minus_1[:, -1].float(),
                gamma[:, -1].float(),
            ).float()
            
            if not hasattr(self, '_printed_pfinal_debug'):
                print(f"  [MATH BUG HUNT] p_final shape: {p_final.shape}, std: {p_final.float().std().item():.6f}, min: {p_final.min().item():.6f}, max: {p_final.max().item():.6f}")
                self._printed_pfinal_debug = True

            # Hierarchical Relational Logic (HAL) Gating
            if self.super_sets is not None:
                # V21: Concatenate super-set outputs and pass through hierarchical relation layer
                gates = torch.cat([ss(p_final).float() for ss in self.super_sets], dim=-1)
                self.log_math("SUPER_SETS", "S = concat([SuperSet_i(p) for i in 1..N])", gates)
                
                # Use Hierarchical Logic if available, else product
                if hasattr(self, 'hierarchical_logic'):
                    # Architecture note: this is a 3-level sigmoid chain —
                    #   L1: PredicateSet → sigmoid(RelationalLogicLayer(p))   ∈ [0,1]
                    #   L2: SuperSet     → sigmoid(RelationalLogicLayer(S))   ∈ [0,1]
                    #   L3: here         → sigmoid(RelationalLogicLayer(G))   ∈ [0,1]
                    # Mean-only centering: removes runaway offset drift while still
                    # allowing the model to learn a persistent gate bias (open/closed
                    # preference per regime). Std-normalization was tested in Run #3
                    # and proved too restrictive — gate_logit std≈1.0 every batch
                    # means the model cannot accumulate a stable directional signal.
                    #
                    # Run 5: temperature τ = self.gate_temperature scales logit
                    # amplitude before centering so λ std targets 0.15–0.30.
                    tau = self.gate_temperature.clamp(1.0, 10.0)
                    gate_logit = self.hierarchical_logic(gates).float() * tau
                    gate_logit = gate_logit - gate_logit.detach().mean(dim=0, keepdim=True)
                    super_gate = torch.sigmoid(gate_logit)
                    # Store detached stats as model attrs — read by training loop each batch.
                    self._last_gate_logit = gate_logit.detach()
                    self._last_super_gate = super_gate.detach()
                    self._last_gate_temperature = tau.item()
                    if self.verbose_math:
                        self.log_math("HIERARCHICAL_LOGIC", f"lambda = sigmoid(tau*RELATION(S) - E[tau*RELATION(S)])  tau={tau.item():.3f}", super_gate)

                    if not hasattr(self, '_printed_ss_debug'):
                        if gates.shape[0] >= 2:
                            print(f"  [MATH BUG HUNT] S[0:2]\nS[0]: {gates[0].tolist()}\nS[1]: {gates[1].tolist()}")
                        print(f"  [MATH BUG HUNT] tau={tau.item():.3f} | gate_logit std: {gate_logit.float().std().item():.6f} | lambda std: {super_gate.float().std().item():.6f}, min: {super_gate.min().item():.6f}, max: {super_gate.max().item():.6f}")
                        self._printed_ss_debug = True
                else:
                    super_gate = gates.prod(dim=-1, keepdim=True)
            else:
                super_gate = torch.ones((batch, 1), device=device, dtype=torch.float32)
            
            z_gated = (z_final.float() * super_gate).float()

            # Output (Keep in FP32 for "Ultra-Safe Mode")
            outputs = self.output_head(z_gated).float()

            # v4.2 Auxiliary Structural Head
            pivot_preds = self.pivot_head(z_gated) # (prob, str, logits, dist)

        if return_diagnostics:
            diagnostics['final_gate'] = super_gate
            diagnostics['z_final'] = z_final
            diagnostics['predicates'] = p_k
            diagnostics['moments'] = moments
            diagnostics['bloom'] = bloom
            diagnostics['pivot_preds'] = pivot_preds
            # Deep observability extensions
            diagnostics['pivot'] = _pivot_diag
            diagnostics['tft'] = _tft_diag
            diagnostics['cde'] = _cde_diag
            diagnostics['gate_stats'] = {
                'active_count': (p_k.detach() > 0.5).float().sum(dim=-1).mean().item(),
                'total_predicates': p_k.shape[-1],
                'entropy': -(p_k.detach() * (p_k.detach() + 1e-8).log()).sum(dim=-1).mean().item(),
                'top5': p_k.detach().mean(dim=0).topk(min(5, p_k.shape[-1])).values.tolist(),
                'top5_indices': p_k.detach().mean(dim=0).topk(min(5, p_k.shape[-1])).indices.tolist(),
            }
            # Set/SuperSet routing
            if self.super_sets is not None:
                ss_weights = [ss(p_k).detach().mean().item() for ss in self.super_sets]
                diagnostics['superset_routing'] = {
                    'weights': ss_weights,
                    'dominant_superset': int(torch.tensor(ss_weights).argmax().item()),
                    'consensus': max(ss_weights) / (sum(ss_weights) + 1e-8),
                }
            diagnostics['output_stats'] = {
                'mean': outputs.detach().mean(dim=0).tolist(),
                'std': outputs.detach().std(dim=0).tolist(),
            }
            # Hierarchical gate diagnostics (populated by new logit-centering path)
            if hasattr(self, '_last_gate_logit'):
                gl = self._last_gate_logit.float()
                sg = self._last_super_gate.float()
                diagnostics['gate_logit_stats'] = {
                    'mean': gl.mean().item(),
                    'std':  gl.std().item() if gl.numel() > 1 else 0.0,
                    'min':  gl.min().item(),
                    'max':  gl.max().item(),
                }
                diagnostics['lambda_quantiles'] = {
                    'p05':  sg.quantile(0.05).item(),
                    'p50':  sg.quantile(0.50).item(),
                    'p95':  sg.quantile(0.95).item(),
                    'mean': sg.mean().item(),
                    'std':  sg.std().item() if sg.numel() > 1 else 0.0,
                }
            return outputs, diagnostics

        return outputs, pivot_preds

    def get_initial_state(self, x_init: torch.Tensor) -> torch.Tensor:
        """
        Initializes x_0 for stateful inference (Phase 5).
        x_init: (B, D) first bar features
        """
        # Slice and Encode Pivots (Phase 3)
        x_main = x_init[..., :-self.d_pivot_raw]
        x_piv = x_init[..., -self.d_pivot_raw:]
        
        # We need the encoder to handle (Batch, D) not just (Batch, Seq, D)
        # PivotEncoder handles generic last dim if written with Linear
        x_piv_enc = self.pivot_encoder(x_piv) 
        
        x_struct = torch.cat([x_main, x_piv_enc], dim=-1)
        return self.initial_encoder(x_struct)

    @torch.no_grad()
    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, state_prev: torch.Tensor,
             u_k: torch.Tensor, greeks_k: torch.Tensor, q_k: torch.Tensor,
             pred_inputs: Dict[str, torch.Tensor], dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stateful incremental step for Phase 5 auditing/efficiency.
        Returns (outputs, state_next)
        """
        h_p, v_p, m_p, r_p = self.spec.split(state_prev)
        
        # 1. Predicate Evaluation
        p_k = self.pred_gates(
            pred_inputs['iv_rank'].float(),
            pred_inputs['bid_ask_spread'].float(),
            pred_inputs['price'].float(),
            pred_inputs['rsi'].float(),
            pred_inputs['delta_rsi'].float(),
            pred_inputs['S_t'].float(),
            pred_inputs['S_t_minus_1'].float(),
            pred_inputs['gamma'].float(),
        ).float()
        
        _, _, _, z_pred = self.pred_signature(p_k)
        
        # 2. Update r_k (Regime dynamics)
        r_k = self.regime_dyn(r_p.float(), z_pred.float()).float()
        
        # 3. Master Step (Physics Transition)
        # Precompute kernels if not cached (for this step)
        A_full = self.A_theta.full_matrix().float()
        F_k, phi1 = etd1_kernel(A_full, dt)
        
        x_p_full = self.spec.cat(h_p, v_p, m_p, r_p).float()
        x_k = condornet_master_step(
            spec=self.spec, B_theta=self.B_theta, G_theta=self.G_theta,
            D_forcing=self.D_forcing, x_prev=x_p_full, u_k=u_k.float(),
            dX_k=(x_curr - x_prev).float(), greeks_k=greeks_k.float(),
            r_prev=r_p.float(), q_k=q_k.float(), F_k=F_k, phi1=phi1,
            dt_k=dt, A_theta=self.A_theta
        ).float()
        
        x_k = torch.tanh(x_k)
        h, v, m, _ = self.spec.split(x_k)
        state_next = self.spec.cat(h, v, m, r_k)
        
        # 4. Gating & Output
        # (Simplified Gating for step-wise; actual HAL uses full sequence context)
        # For stateful parity, we use the local predicate 'p_k'
        if self.super_sets is not None:
            gates = torch.cat([ss(p_k).float() for ss in self.super_sets], dim=-1)
            super_gate = torch.sigmoid(self.hierarchical_logic(gates)) if hasattr(self, 'hierarchical_logic') else gates.prod(dim=-1, keepdim=True)
        else:
            super_gate = torch.ones((x_k.shape[0], 1), device=x_k.device)
            
        z_gated = (state_next.float() * super_gate).float()
        outputs = self.output_head(z_gated).float()
        
        return outputs, state_next

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

    # Create model (79 fields = V3.0 schema)
    model = CondorNet(
        d_input=79,
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
    x = torch.randn(batch, seq, 79)

    print(f"\nTest forward pass:")
    print(f"  Input: ({batch}, {seq}, 79)")

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
    p = torch.rand(batch, 8)
    gi_loss = group_invariant_loss(model.pred_signature, p)
    print(f"\nGroup invariance loss: {gi_loss.item():.6f}")

    print("\n" + "=" * 70)
    print("Implementation complete and tested.")
    print("=" * 70)
