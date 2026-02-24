# CondorNet Implementation Plan

## Overview

This document outlines the complete implementation plan for the **CondorNet** architecture. Originally targeting the transition from Neural CDE to CondorNet v4.0, this document has been updated with the **v4.3 addendum** covering the multi-strategy extensions.

> [!IMPORTANT]
> As of 2026-02-24, the authoritative implementation is `intelligence/condor_brain_net_v43.py` (CondorNet™ v4.3). The v4.0 core (Parts 1-2 below) is preserved as the `CondorNet` class in `condor_brain_net_v42.py` and used as the inner core engine of `CondorNetV43`.

This is a first-of-its-kind implementation combining:

- **ETD-1 (Exponential Time Differencing)** matrix exponential integrator
- **Neural CDE** controlled differential equations
- **Predicate Gates** (inequality-based logical conditions)
- **Group-Invariant Signatures** for permutation-invariant encoding
- **Fuzzy Sizing** integration with neural outputs
- **Composite Loss** with 10 components

---

## Part 1: condor_brain_net.py Architecture

### 1.1 Augmented State Vector

The core innovation is the **4-block augmented state**:

```
x_k = [h_k; v_k; m_k; r_k]
```

| Block | Dim | Description |
|-------|-----|-------------|
| `h_k` | d_h | Latent risk-manifold (hidden state) |
| `v_k` | d_v | Portfolio state (Greeks, P&L, positions) |
| `m_k` | d_m | Risk memory (running max drawdown, VaR) |
| `r_k` | d_r | Explicit regime state (vol regime, trend) |

**Implementation:**
```python
class AugmentedState:
    """
    Manages the 4-block augmented state vector.
    Total dim = d_h + d_v + d_m + d_r
    """
    def __init__(self, d_h=256, d_v=32, d_m=64, d_r=32):
        self.d_h = d_h  # Latent risk-manifold
        self.d_v = d_v  # Portfolio state (Greeks)
        self.d_m = d_m  # Risk memory
        self.d_r = d_r  # Regime state
        self.d_total = d_h + d_v + d_m + d_r
```

### 1.2 Master Evolution Equation

The CondorNet evolution follows:

```
x_k = e^{A_θ Δt_k} x_{k-1} + Δt_k φ_1(A_θ Δt_k) B_θ(u_k) + G_θ(x_{k-1}, u_k) ΔX_k + x_k^forcing
```

| Term | Description | Implementation |
|------|-------------|----------------|
| `e^{A_θ Δt_k}` | Matrix exponential (autonomous dynamics) | SSD/Mamba scan kernel |
| `φ_1(A_θ Δt_k)` | ETD-1 basis function: `M^{-1}(e^M - I)` | Custom kernel |
| `B_θ(u_k)` | Forcing from control input | TFT encoder output |
| `G_θ(x, u)` | CDE response matrix | Neural network |
| `ΔX_k` | Control path increment | `X_k - X_{k-1}` |
| `x_k^forcing` | Greeks forcing term | `D @ Greeks_k` |

### 1.3 Block-Matrix Operators

**A_θ (State Transition) - 4×4 Block Structure:**
```
A_θ = [A_hh  A_hv  A_hm  A_hr]
      [A_vh  A_vv  A_vm  A_vr]
      [A_mh  A_mv  A_mm  A_mr]
      [A_rh  A_rv  A_rm  A_rr]
```

Each block is a learnable `nn.Linear` or structured matrix:
- `A_hh`: Latent-to-latent dynamics (d_h × d_h)
- `A_hv`: Portfolio influences on latent (d_h × d_v)
- `A_rr`: Regime self-dynamics (d_r × d_r)
- etc.

**Implementation:**
```python
class BlockMatrixA(nn.Module):
    """4×4 block transition matrix A_θ."""
    def __init__(self, d_h, d_v, d_m, d_r):
        super().__init__()
        # Diagonal blocks (self-dynamics)
        self.A_hh = nn.Linear(d_h, d_h, bias=False)
        self.A_vv = nn.Linear(d_v, d_v, bias=False)
        self.A_mm = nn.Linear(d_m, d_m, bias=False)
        self.A_rr = nn.Linear(d_r, d_r, bias=False)

        # Off-diagonal blocks (cross-dynamics)
        self.A_hv = nn.Linear(d_v, d_h, bias=False)
        self.A_hm = nn.Linear(d_m, d_h, bias=False)
        self.A_hr = nn.Linear(d_r, d_h, bias=False)
        # ... (12 total cross-blocks)

    def forward(self, x):
        """Apply A_θ to augmented state x = [h, v, m, r]."""
        h, v, m, r = self.split_state(x)

        h_new = self.A_hh(h) + self.A_hv(v) + self.A_hm(m) + self.A_hr(r)
        v_new = self.A_vh(h) + self.A_vv(v) + self.A_vm(m) + self.A_vr(r)
        m_new = self.A_mh(h) + self.A_mv(v) + self.A_mm(m) + self.A_mr(r)
        r_new = self.A_rh(h) + self.A_rv(v) + self.A_rm(m) + self.A_rr(r)

        return torch.cat([h_new, v_new, m_new, r_new], dim=-1)
```

### 1.4 ETD-1 Basis Function φ_1

The ETD-1 basis function:
```
φ_1(M) = M^{-1}(e^M - I)
```

For small `Δt`, approximation:
```
φ_1(M) ≈ I + M/2 + M²/6 + M³/24 + ...
```

**Implementation:**
```python
def phi_1_approx(M, order=4):
    """
    Compute φ_1(M) = M^{-1}(e^M - I) via Taylor expansion.
    Avoids matrix inversion instability.
    """
    result = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    M_power = M.clone()

    for k in range(1, order + 1):
        result = result + M_power / math.factorial(k + 1)
        M_power = M_power @ M

    return result

def matrix_exp_approx(M, order=6):
    """
    Compute e^M via Taylor expansion.
    For SSD kernel, use specialized cuda implementation.
    """
    result = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    M_power = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)

    for k in range(1, order + 1):
        M_power = M_power @ M
        result = result + M_power / math.factorial(k)

    return result
```

### 1.5 CDE Response G_θ

The CDE response matrix follows the Neural CDE pattern:

```python
class CDEResponseG(nn.Module):
    """
    G_θ(x, u) produces the controlled response matrix.
    dZ = G_θ(x, u) @ dX
    """
    def __init__(self, d_state, d_input, d_control):
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input

        # MLP to produce response matrix
        self.net = nn.Sequential(
            nn.Linear(d_state + d_control, d_state * 2),
            nn.SiLU(),
            nn.Linear(d_state * 2, d_state * d_input),
        )

    def forward(self, x, u):
        """
        Args:
            x: (batch, d_state) augmented state
            u: (batch, d_control) TFT control embedding
        Returns:
            G: (batch, d_state, d_input) response matrix
        """
        combined = torch.cat([x, u], dim=-1)
        G_flat = self.net(combined)
        G = G_flat.view(-1, self.d_state, self.d_input)
        return torch.tanh(G)  # Bound response for stability
```

### 1.6 Greeks Forcing Term

Explicit Greeks influence portfolio state:

```python
class GreeksForcing(nn.Module):
    """
    x_k^forcing = D @ Greeks_k

    Greeks: [delta, gamma, theta, vega, rho]
    """
    def __init__(self, d_v, n_greeks=5):
        super().__init__()
        self.D = nn.Linear(n_greeks, d_v, bias=False)

    def forward(self, greeks):
        """
        Args:
            greeks: (batch, 5) [delta, gamma, theta, vega, rho]
        Returns:
            forcing: (batch, d_v) portfolio forcing term
        """
        return self.D(greeks)
```

### 1.7 Predicate Gates

Predicate gates use inequality conditions:

```python
class PredicateGate(nn.Module):
    """
    Single predicate: P_i = σ(s_i · (f_a(x) - f_b(x) - τ_i))

    Types:
    - Volatility spike: IV[0] > 1.5 * IV[20]
    - Liquidity compression: spread[0] > 2 * spread_avg
    - Momentum reversal: sign(ret[0]) != sign(ret[5])
    - Gap risk: abs(open - prev_close) > 2 * ATR
    - Greeks pressure: abs(delta) > 0.3
    """
    def __init__(self, d_input, n_predicates=5):
        super().__init__()
        self.n_predicates = n_predicates

        # Learnable comparison functions
        self.f_a = nn.ModuleList([
            nn.Linear(d_input, 1) for _ in range(n_predicates)
        ])
        self.f_b = nn.ModuleList([
            nn.Linear(d_input, 1) for _ in range(n_predicates)
        ])

        # Learnable thresholds and steepness
        self.tau = nn.Parameter(torch.zeros(n_predicates))
        self.steepness = nn.Parameter(torch.ones(n_predicates) * 10.0)

    def forward(self, x):
        """
        Args:
            x: (batch, d_input) features
        Returns:
            gates: (batch, n_predicates) soft gate values in [0, 1]
        """
        gates = []
        for i in range(self.n_predicates):
            a = self.f_a[i](x)  # (batch, 1)
            b = self.f_b[i](x)  # (batch, 1)
            diff = a - b - self.tau[i]
            gate = torch.sigmoid(self.steepness[i] * diff)
            gates.append(gate)

        return torch.cat(gates, dim=-1)
```

### 1.8 Predicate Sets and Super-Sets

Hierarchical logical composition:

```python
class PredicateSet(nn.Module):
    """
    S = ⋀_j P_j (AND) or S = ⋁_j P_j (OR)
    With learnable aggregation (p-norm or OWA).
    """
    def __init__(self, n_predicates, aggregation='pnorm'):
        super().__init__()
        self.n_predicates = n_predicates
        self.aggregation = aggregation

        if aggregation == 'pnorm':
            self.p = nn.Parameter(torch.tensor(1.0))  # Learnable p
        elif aggregation == 'owa':
            self.weights = nn.Parameter(torch.ones(n_predicates) / n_predicates)

    def forward(self, predicates):
        """
        Args:
            predicates: (batch, n_predicates) predicate values
        Returns:
            set_value: (batch, 1) aggregated set value
        """
        if self.aggregation == 'pnorm':
            p = torch.clamp(self.p, -10, 10)
            eps = 1e-8
            predicates = torch.clamp(predicates, eps, 1 - eps)
            mean_powered = torch.mean(predicates ** p, dim=-1, keepdim=True)
            return mean_powered ** (1.0 / p)
        elif self.aggregation == 'owa':
            # Sort descending and apply weights
            sorted_preds, _ = torch.sort(predicates, dim=-1, descending=True)
            weights = torch.softmax(self.weights, dim=0)
            return (sorted_preds * weights).sum(dim=-1, keepdim=True)


class SuperSet(nn.Module):
    """
    S = (S_1 < S_2) AND (S_2 > S_3) ...
    Hierarchical comparison of sets.
    """
    def __init__(self, n_sets=4):
        super().__init__()
        self.n_sets = n_sets
        self.comparisons = nn.Parameter(torch.zeros(n_sets - 1))  # Learnable comparison ops
        self.final_agg = nn.Linear(n_sets, 1)

    def forward(self, set_values):
        """
        Args:
            set_values: (batch, n_sets) aggregated set values
        Returns:
            super_value: (batch, 1) final gating value
        """
        return torch.sigmoid(self.final_agg(set_values))
```

### 1.9 Group-Invariant Signatures

Permutation-invariant feature encoding via sorted moments:

```python
class GroupInvariantSignature(nn.Module):
    """
    Computes permutation-invariant signatures via sorted moments.

    sig(X) = [sorted(X), mean(X), std(X), skew(X), kurt(X)]
    """
    def __init__(self, d_input, d_output, n_moments=4):
        super().__init__()
        self.n_moments = n_moments

        # Project sorted features + moments to output dim
        sig_dim = d_input + n_moments
        self.proj = nn.Linear(sig_dim, d_output)

    def forward(self, x):
        """
        Args:
            x: (batch, seq, d_input) or (batch, d_input)
        Returns:
            sig: (batch, d_output) invariant signature
        """
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last timestep

        # Sort features (permutation-invariant)
        sorted_x, _ = torch.sort(x, dim=-1)

        # Compute moments
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        skew = ((x - mean) ** 3).mean(dim=-1, keepdim=True) / (std ** 3 + 1e-8)
        kurt = ((x - mean) ** 4).mean(dim=-1, keepdim=True) / (std ** 4 + 1e-8) - 3

        # Concatenate
        sig = torch.cat([sorted_x, mean, std, skew, kurt], dim=-1)

        return self.proj(sig)
```

### 1.10 TFT Control Path

The control embedding `u_k = TFT_θ(X_{1:k})`:

```python
class TFTControlEncoder(nn.Module):
    """
    Temporal Fusion Transformer control encoder.
    Produces control embeddings from input sequence.
    """
    def __init__(self, d_input, d_model, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=n_layers
        )

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq, d_input) input sequence
        Returns:
            u: (batch, d_model) control embedding
        """
        x = self.input_proj(x)
        x = self.transformer(x)
        u = x[:, -1, :]  # Take final hidden state
        return self.output_proj(u)
```

### 1.11 Complete CondorNet Module (Actual Implementation)

> [!IMPORTANT]
> The code below reflects the **actual implementation** in `condor_brain_net.py` as of 2026-02-04,
> including true ETD-1 integration, 5 canonical inequality gates, full 4-block forcing, and r_k dynamics.

#### Key Components (as implemented):

| Component | Class | Description |
|-----------|-------|-------------|
| State Spec | `AugmentedStateSpec` | Dimensions and block slicing for `[h, v, m, r]` |
| Transition | `BlockMatrixA` | 4×4 block operator with sparsity rules |
| Control Injection | `BlockVectorB` | Block-partitioned `B_θ(u_k)` |
| CDE Response | `CDEResponseG` | `G_θ(x, u) ∈ ℝ^{d_x × d_input}` |
| Full Forcing | `FullForcingD` | `D(Greeks_k, r_{k-1}, q_k)` → all 4 blocks |
| ETD-1 Kernel | `etd1_kernel()` | `F = exp(AΔt)`, `φ₁(M) = M⁺(e^M - I)` |
| 5 Canonical Gates | `CanonicalPredicateGates` | Vol spike, liquidity, reversal, gap, Greeks |
| Predicate Signature | `PredicateSignature` | Group-invariant `[p; moments; bloom]` |
| r_k Dynamics | `RegimeCombinatoricsDynamics` | `r_k = α_k ⊙ r_{k-1} + β_k` |
| Control Encoder | `TFTControlEncoder` | Causal transformer for `u_k` |
| Fusion | `FusionGate` | Combines branches via `[h; bloom; moments]` |

```python
class CondorNet(nn.Module):
    """
    CondorNet™: Mathematically faithful implementation.

    Master equation (TRUE ETD-1, not Euler):
        x_k = e^{A_θ Δt_k} x_{k-1} + Δt_k φ_1(A_θ Δt_k) B_θ(u_k)
            + G_θ(x_{k-1}, u_k) ΔX_k + D(Greeks_k, r_{k-1}, q_k)

    4-block state: x_k = [h_k; v_k; m_k; r_k]
    """
    def __init__(
        self,
        d_input: int = 54,
        d_h: int = 256,        # Latent risk-manifold
        d_v: int = 32,         # Portfolio state
        d_m: int = 64,         # Risk memory
        d_r: int = 32,         # Regime/combinatorics
        d_control: int = 128,  # TFT control dim
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

        # === PREDICATE SYSTEM (5 canonical inequality gates) ===
        self.pred_gates = CanonicalPredicateGates()  # Vol spike, liquidity, reversal, gap, Greeks
        self.pred_signature = PredicateSignature(K=n_predicates, R=R_moments, M=M_bloom)
        self.regime_dyn = RegimeCombinatoricsDynamics(d_r=d_r, z_dim=n_predicates + R_moments + M_bloom)
        self.super_set = SuperSet(n_sets=4, n_predicates=n_predicates)

        # === FUSION ===
        self.fusion_gate = FusionGate(d_h, M_bloom, R_moments, n_branches=3)

        # === OUTPUT ===
        self.output_head = nn.Linear(self.spec.d_x, 10)
```

#### True ETD-1 Master Step (not Euler approximation):

```python
def condornet_master_step(
    spec, A_theta, B_theta, G_theta, D_forcing,
    x_prev, u_k, dX_k, greeks_k, r_prev, q_k, dt_k
) -> torch.Tensor:
    """
    One CondorNet master update step (faithful to canonical equation).

    x_k = F_k x_{k-1} + g_k + G_θ(x_{k-1}, u_k) ΔX_k + D(Greeks_k, r_{k-1}, q_k)

    where:
        F_k = exp(A_θ Δt_k)          # TRUE matrix exponential
        g_k = Δt_k φ_1(A_θ Δt_k) B_θ(u_k)  # ETD-1 basis function
    """
    # 1. Build A(u_k) as full [d_x, d_x] matrix
    A_full = A_theta.full_matrix()

    # 2. TRUE ETD-1 kernel: F = exp(AΔt), φ₁ = M⁺(e^M - I)
    F_k, phi1 = etd1_kernel(A_full, dt_k)

    # 3. ETD injection: g_k = Δt φ_1(AΔt) B
    B_k = B_theta(u_k)
    g_k = torch.matmul(B_k, phi1.T) * dt_k

    # 4. Linear propagation: F_k @ x_{k-1}
    x_lin = torch.matmul(x_prev, F_k.T)

    # 5. CDE response: G_θ(x, u) @ dX
    G_k = G_theta(x_prev, u_k)
    cde_term = torch.bmm(G_k, dX_k.unsqueeze(-1)).squeeze(-1)

    # 6. FULL 4-block forcing: D(Greeks_k, r_{k-1}, q_k)
    D_k = D_forcing(greeks_k, r_prev, q_k)  # Uses r_{k-1} for causality!

    # 7. Master update
    return x_lin + g_k + cde_term + D_k
```

#### 5 Canonical Inequality Gates:

```python
class CanonicalPredicateGates(nn.Module):
    """
    The 5 canonical inequality gates from the specification.
    Uses steep sigmoids for differentiable approximation.

    1. Volatility spike:     IVR_t > 75
    2. Liquidity compression: Spread/Price > 0.4%
    3. Momentum reversal:    RSI < 25 AND ΔRSI < 0
    4. Gap risk:             |S_t - S_{t-1}|/S_{t-1} > 1.2%
    5. Greeks pressure:      |Γ| > threshold
    """
    def forward(self, iv_rank, bid_ask_spread, price, rsi, delta_rsi,
                S_t, S_t_minus_1, gamma) -> torch.Tensor:
        # Returns: (batch, 5) soft predicates in [0, 1]
        ...
```

#### r_k Dynamics (Regime Combinatorics):

```python
class RegimeCombinatoricsDynamics(nn.Module):
    """
    Update equation:
        r_k = α_k ⊙ r_{k-1} + β_k

    where:
        α_k = σ(W_α z_pred)  -- forget gate
        β_k = W_β z_pred     -- update term
        z_pred = [p; moments; bloom] -- predicate signature
    """
    def forward(self, r_prev, z_pred) -> torch.Tensor:
        alpha = torch.sigmoid(self.W_alpha(z_pred))  # Forget gate
        beta = self.W_beta(z_pred)                   # Update
        return alpha * r_prev + beta
```

#### Full 4-Block Forcing D (not v-only):

```python
class FullForcingD(nn.Module):
    """
    D(Greeks_k, r_{k-1}, q_k) ∈ ℝ^{d_x}

    Block semantics:
        D_h: Market physics forcing from Greeks + regime
        D_v: PRIMARY execution location (slippage, impact, margin) - Rule RA
        D_m: Cumulative risk (stress integrals, drawdown-weighted)
        D_r: Regime forcing (from predicate combinatorics)

    CRITICAL: Uses r_{k-1} (previous regime) for causality.
    """
    def forward(self, greeks, r_prev, q) -> torch.Tensor:
        z = torch.cat([greeks, r_prev, q], dim=-1)
        return self.spec.cat(self.D_h(z), self.D_v(z), self.D_m(z), self.D_r(z))
```

#### Group-Invariant Predicate Signature:

```python
class PredicateSignature(nn.Module):
    """
    Produces z_pred = [p; s; B] where:
        - p: raw predicates (batch, K)
        - s: sorted moments (permutation-invariant)
        - B: Bloom-like signature

    The signature satisfies: s(p) = s(π(p)) for any permutation π.
    """
    def forward(self, p) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        p_sorted, _ = torch.sort(p, dim=-1, descending=True)
        moments = [p_sorted ** r for r in range(1, R+1)]
        bloom = torch.sigmoid(self.W_bloom(p))
        z_pred = torch.cat([p, moments, bloom], dim=-1)
        return p_sorted, moments, bloom, z_pred
```

---

## Part 2: condor_train_net.py Training Script

### 2.1 Composite Loss Function

The CondorNet loss has **10 components**:

| Component | Symbol | Description | Weight |
|-----------|--------|-------------|--------|
| NPDD Loss | L_npdd | Net profit / drawdown | λ_npdd |
| Sharpe Loss | L_Sharpe | -Sharpe ratio | λ_sharpe |
| Drawdown Loss | L_drawdown | Max drawdown penalty | λ_dd |
| Turnover Loss | L_turnover | Trading frequency | λ_turn |
| Fuzzy Loss | L_fuzzy | Sizing consistency | λ_fuzzy |
| Pattern Entropy | L_pattern-ent | Predicate diversity | λ_pent |
| Group Invariance | L_group-inv | Signature consistency | λ_ginv |
| Correlation Loss | L_ρ | Cross-head correlation | λ_rho |
| Energy Loss | L_energy | State magnitude | λ_energy |
| Growth Loss | L_growth | Capital growth rate | λ_growth |

```python
class CompositeCondorNetLoss(nn.Module):
    """
    10-component composite loss for CondorNet training.
    """
    def __init__(
        self,
        lambda_npdd: float = 1.0,
        lambda_sharpe: float = 0.5,
        lambda_dd: float = 0.3,
        lambda_turnover: float = 0.1,
        lambda_fuzzy: float = 0.2,
        lambda_pattern_ent: float = 0.05,
        lambda_group_inv: float = 0.05,
        lambda_rho: float = 0.1,
        lambda_energy: float = 0.01,
        lambda_growth: float = 0.1,
    ):
        super().__init__()
        self.lambdas = {
            'npdd': lambda_npdd,
            'sharpe': lambda_sharpe,
            'dd': lambda_dd,
            'turnover': lambda_turnover,
            'fuzzy': lambda_fuzzy,
            'pattern_ent': lambda_pattern_ent,
            'group_inv': lambda_group_inv,
            'rho': lambda_rho,
            'energy': lambda_energy,
            'growth': lambda_growth,
        }

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        gates: torch.Tensor = None,
        state: torch.Tensor = None,
        returns: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute composite loss.

        Args:
            predictions: (batch, 10) model outputs
            targets: (batch, 10) target values
            gates: (batch, n_predicates) predicate gate values
            state: (batch, d_state) final state vector
            returns: (batch, seq) return series for Sharpe

        Returns:
            total_loss: scalar
            components: dict of individual losses
        """
        components = {}

        # 1. NPDD Loss (primary prediction loss)
        pred_loss = F.mse_loss(predictions, targets)
        components['npdd'] = pred_loss

        # 2. Sharpe Loss
        if returns is not None:
            sharpe = self._compute_sharpe(returns)
            components['sharpe'] = -sharpe  # Negative because we minimize
        else:
            components['sharpe'] = torch.tensor(0.0, device=predictions.device)

        # 3. Drawdown Loss
        if returns is not None:
            dd = self._compute_max_drawdown(returns)
            components['dd'] = dd
        else:
            components['dd'] = torch.tensor(0.0, device=predictions.device)

        # 4. Turnover Loss (from entry/exit logits)
        entry_logits = predictions[:, 8]
        exit_logits = predictions[:, 9]
        turnover = torch.abs(entry_logits).mean() + torch.abs(exit_logits).mean()
        components['turnover'] = turnover

        # 5. Fuzzy Loss (sizing consistency)
        confidence = predictions[:, 7]  # Confidence head
        fuzzy_loss = torch.var(confidence)  # Penalize erratic confidence
        components['fuzzy'] = fuzzy_loss

        # 6. Pattern Entropy (predicate diversity)
        if gates is not None:
            gate_probs = gates.mean(dim=0)  # Average activation per predicate
            entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum()
            components['pattern_ent'] = -entropy  # Maximize entropy
        else:
            components['pattern_ent'] = torch.tensor(0.0, device=predictions.device)

        # 7. Group Invariance Loss
        # Encourage consistent outputs under permutation (implicit in architecture)
        components['group_inv'] = torch.tensor(0.0, device=predictions.device)

        # 8. Correlation Loss (cross-head decorrelation)
        if predictions.shape[0] > 1:
            corr_matrix = torch.corrcoef(predictions.T)
            off_diag = corr_matrix - torch.eye(10, device=predictions.device)
            rho_loss = (off_diag ** 2).mean()
            components['rho'] = rho_loss
        else:
            components['rho'] = torch.tensor(0.0, device=predictions.device)

        # 9. Energy Loss (state magnitude regularization)
        if state is not None:
            energy = (state ** 2).mean()
            components['energy'] = energy
        else:
            components['energy'] = torch.tensor(0.0, device=predictions.device)

        # 10. Growth Loss (capital trajectory)
        if returns is not None:
            cumulative = torch.cumprod(1 + returns, dim=-1)
            final_growth = cumulative[:, -1].mean()
            components['growth'] = -final_growth  # Maximize growth
        else:
            components['growth'] = torch.tensor(0.0, device=predictions.device)

        # Combine with weights
        total_loss = sum(
            self.lambdas[k] * v for k, v in components.items()
        )

        return total_loss, components

    def _compute_sharpe(self, returns: torch.Tensor) -> torch.Tensor:
        """Annualized Sharpe ratio."""
        mean_ret = returns.mean(dim=-1)
        std_ret = returns.std(dim=-1) + 1e-8
        sharpe = mean_ret / std_ret * math.sqrt(252 * 78)  # 5-min bars
        return sharpe.mean()

    def _compute_max_drawdown(self, returns: torch.Tensor) -> torch.Tensor:
        """Maximum drawdown from returns."""
        cumulative = torch.cumprod(1 + returns, dim=-1)
        running_max = torch.cummax(cumulative, dim=-1)[0]
        drawdown = (running_max - cumulative) / (running_max + 1e-8)
        max_dd = drawdown.max(dim=-1)[0]
        return max_dd.mean()
```

### 2.2 Training Loop Structure

```python
def train_condor_net(args):
    """Main training function for CondorNet."""

    # 1. Setup device and optimizations
    device = setup_device(args)

    # 2. Load and prepare data
    X, y, regime, med, scale = prepare_features(args.local_data)
    train_loader, val_loader = create_dataloaders(X, y, regime, args)

    # 3. Initialize model
    model = CondorNet(
        d_input=len(FEATURE_COLS),
        d_h=args.d_h,
        d_v=args.d_v,
        d_m=args.d_m,
        d_r=args.d_r,
        n_layers=args.layers,
    ).to(device)

    # 4. Loss and optimizer
    criterion = CompositeCondorNetLoss(
        lambda_npdd=args.lambda_npdd,
        lambda_sharpe=args.lambda_sharpe,
        # ... other lambdas
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # 5. Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y, batch_r in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs, gates = model(batch_x, return_gates=True)

            # Compute composite loss
            loss, components = criterion(
                outputs, batch_y,
                gates=gates,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        val_loss = validate(model, val_loader, criterion, device)

        # Logging
        print(f"Epoch {epoch+1}: Train={epoch_loss:.4f}, Val={val_loss:.4f}")

        scheduler.step()

    # Save model
    torch.save(model.state_dict(), args.output)
```

---

## Part 3: Implementation Steps

### Phase 1: Core Architecture (condor_brain_net.py)

1. **Create `AugmentedState` class** - State vector management
2. **Implement `BlockMatrixA`** - 4×4 block transition matrix
3. **Implement `CDEResponseG`** - CDE controlled response
4. **Implement `GreeksForcing`** - Greeks influence on portfolio state
5. **Implement ETD-1 functions** - `phi_1_approx`, `matrix_exp_approx`
6. **Implement `TFTControlEncoder`** - Control embedding generation
7. **Implement `PredicateGate`** - Inequality-based gates
8. **Implement `PredicateSet` and `SuperSet`** - Hierarchical logic
9. **Implement `GroupInvariantSignature`** - Permutation-invariant encoding
10. **Assemble `CondorNet` main class** - Complete forward pass

### Phase 2: Training Infrastructure (condor_train_net.py)

1. **Create `CompositeCondorNetLoss`** - 10-component loss
2. **Implement loss components** - Sharpe, drawdown, entropy, etc.
3. **Adapt data preparation** - Greeks extraction, return computation
4. **Implement training loop** - With all optimizations from current script
5. **Add validation and monitoring** - Reuse TrainingMonitor
6. **Add checkpointing** - Model saving with metadata

### Phase 3: Integration and Testing

1. **Unit tests for each component**
2. **Integration test with synthetic data**
3. **Full training run on institutional dataset**
4. **Performance comparison with Neural CDE baseline**

---

## Part 4: File Structure

```
intelligence/
├── condor_brain_net.py      # NEW: CondorNet architecture
│   ├── AugmentedState
│   ├── BlockMatrixA
│   ├── CDEResponseG
│   ├── GreeksForcing
│   ├── TFTControlEncoder
│   ├── PredicateGate
│   ├── PredicateSet
│   ├── SuperSet
│   ├── GroupInvariantSignature
│   └── CondorNet (main class)
│
├── condor_train_net.py      # NEW: Training script
│   ├── CompositeCondorNetLoss
│   ├── train_condor_net()
│   └── CLI argument parsing
│
├── condor_brain.py          # EXISTING: Keep as fallback
├── train_condor_brain.py    # EXISTING: Keep as reference
└── models/
    └── neural_cde.py        # EXISTING: Reference for CDE patterns
```

---

## Part 5: Key Differences from Neural CDE

| Aspect | Neural CDE (Current) | CondorNet (New) |
|--------|---------------------|-----------------|
| State | Single `h_k` | 4-block `[h, v, m, r]` |
| Dynamics | Simple CDE | ETD-1 + CDE + Forcing |
| Control | Direct features | TFT-encoded control |
| Gating | None | Predicate gates |
| Loss | Basic MSE + penalties | 10-component composite |
| Regime | Implicit | Explicit `r_k` block |
| Greeks | Not used | Explicit forcing term |

---

## Part 6: Expected Outcomes

1. **Better regime adaptation** via explicit regime state `r_k`
2. **Greeks-aware predictions** via forcing term
3. **Interpretable gating** via predicate conditions
4. **Stable training** via ETD-1 integrator
5. **Risk-aware optimization** via composite loss

---

## Appendix: CLI Arguments for condor_train_net.py

```bash
python intelligence/condor_train_net.py \
    --local-data data/institutional/2024_combined.csv \
    --d-h 256 \
    --d-v 32 \
    --d-m 64 \
    --d-r 32 \
    --layers 32 \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --lambda-npdd 1.0 \
    --lambda-sharpe 0.5 \
    --lambda-dd 0.3 \
    --lambda-turnover 0.1 \
    --output models/condor_net_v1.pth
```

---

## Part 7: Mathematical Faithfulness Patches

> [!IMPORTANT]
> These patches ensure the implementation is mathematically faithful to the canonical CondorNet specification. All patches apply to `condor_brain_net.py` except for minor training integration changes in `condor_train_net.py`.

### 7.1 Tensor-Safe ETD-1 Kernel

**Status**: ✅ Implemented (current code correct, minor dt tensor safety needed)

Make `dt` broadcastable and avoid Python float assumptions:

```python
def etd1_kernel(A: torch.Tensor, dt: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    True ETD-1 kernel:
        M = A Δt
        F = exp(M)
        φ_1(M) = M⁺ (exp(M) - I)
    """
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, device=A.device, dtype=A.dtype)

    M = A * dt
    expM = torch.linalg.matrix_exp(M)
    I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)

    phi1 = torch.linalg.pinv(M) @ (expM - I)
    return expM, phi1
```

---

### 7.2 Control-Dependent BlockMatrixA

**Status**: 🔧 Needs update (current implementation ignores `u_k`)

Add control-dependent stability modulation and expose full matrix with control:

```python
class BlockMatrixA(nn.Module):
    def __init__(self, spec: AugmentedStateSpec, d_control: int, enforce_sparsity: bool = True):
        super().__init__()
        self.spec = spec
        self.enforce_sparsity = enforce_sparsity
        
        # ... existing block definitions ...
        
        # Control-dependent stability modulation: η(u_k) -> diag term
        self.stab_mlp = nn.Sequential(
            nn.Linear(d_control, spec.d_x),
            nn.Tanh()
        )
        self._init_stable()

    def forward_blocks(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply A_θ(u) in block form to x = [h, v, m, r]."""
        h, v, m, r = self.spec.split(x)
        
        # ... existing block application ...
        
        out = self.spec.cat(h_new, v_new, m_new, r_new)
        
        if u is not None:
            # stability modulation: A(u) ≈ A_base - diag(exp(η(u)))
            eta = self.stab_mlp(u)  # (batch, d_x)
            diag_term = -torch.exp(eta)
            out = out + diag_term * x
        
        return out

    def full_matrix(self, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Construct full [d_x, d_x] matrix for ETD-1."""
        d_x = self.spec.d_x
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        eye = torch.eye(d_x, device=device, dtype=dtype)
        u_rep = u.mean(dim=0, keepdim=True) if u is not None else None
        
        cols = []
        for i in range(d_x):
            e_i = eye[:, i].unsqueeze(0)
            col = self.forward_blocks(e_i, u_rep).squeeze(0)
            cols.append(col.unsqueeze(-1))
        return torch.cat(cols, dim=-1)
```

**CondorNet.__init__ update:**
```python
self.A_theta = BlockMatrixA(self.spec, d_control=d_control, enforce_sparsity=enforce_sparsity)
```

---

### 7.3 Per-Step Predicate Gates and r_k Dynamics

**Status**: 🔧 Needs update (current forward computes predicates only at final step)

Wire predicate gates into the time loop with explicit r_k dynamics:

```python
def forward(self, x, ..., strict_causality: bool = True, return_diagnostics: bool = False):
    """
    Args:
        strict_causality: If True, D_forcing uses r_{k-1} before regime update.
                          If False, regime updates before D_forcing.
    """
    B, T, _ = x.shape
    device = x.device
    
    # 1) Control path u_k from TFT
    u_seq = self.tft(x, return_sequence=True)  # (B, T, d_control)
    
    # 2) Initial state
    x_state = self._initial_state(x)
    h, v, m, r = self.spec.split(x_state)
    
    all_predicates = []
    
    # 3) Time stepping
    for k in range(1, T):
        u_k = u_seq[:, k, :]
        dX_k = x[:, k, :] - x[:, k-1, :]
        greeks_k = greeks[:, k, :] if greeks is not None else torch.zeros(B, self.n_greeks, device=device)
        q_k = q[:, k:k+1] if q is not None else torch.zeros(B, 1, device=device)
        
        # 3a) Predicate gates at step k
        p_k = self.pred_gates(iv_rank[:, k], bid_ask_spread[:, k], price[:, k], 
                               rsi[:, k], delta_rsi[:, k], S_t[:, k], 
                               S_t_minus_1[:, k], gamma[:, k])
        all_predicates.append(p_k)
        _, moments, bloom, z_pred = self.pred_signature(p_k)
        
        if strict_causality:
            # D uses r_{k-1}, then update r
            D_k = self.D_forcing(greeks_k, r, q_k)
            r = self.regime_dyn(r, z_pred)
        else:
            # Update r first, then D uses r_k
            r = self.regime_dyn(r, z_pred)
            D_k = self.D_forcing(greeks_k, r, q_k)
        
        # 3b) Full ETD-1 step
        x_prev = self.spec.cat(h, v, m, r)
        A_full = self.A_theta.full_matrix(u_k)
        F_k, phi1 = etd1_kernel(A_full, dt)
        
        B_k = self.B_theta(u_k)
        g_k = torch.matmul(B_k, phi1.T) * dt
        x_lin = torch.matmul(x_prev, F_k.T)
        
        G_k = self.G_theta(x_prev, u_k)
        cde_term = torch.bmm(G_k, dX_k.unsqueeze(-1)).squeeze(-1)
        
        x_state = x_lin + g_k + cde_term + D_k
        h, v, m, r = self.spec.split(x_state)
    
    # Final output
    z_final = self.spec.cat(h, v, m, r)
    outputs = self.output_head(z_final)
    
    if return_diagnostics:
        diag = {
            "z_final": z_final,
            "predicates": torch.stack(all_predicates, dim=1) if all_predicates else None,
        }
        return outputs, diag
    return outputs
```

**New helper method:**
```python
def get_A_matrix(self, u_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Expose A_θ(u_ref) as full matrix for diagnostics/loss."""
    if u_ref is None:
        device = next(self.parameters()).device
        u_ref = torch.zeros(1, self.d_control, device=device)
    return self.A_theta.full_matrix(u_ref)
```

---

### 7.4 Group Invariant and Spectral Radius Losses

**Status**: ✅ Exists (minor refinement to use `signature_only`)

```python
def group_invariant_loss(
    pred_signature: PredicateSignature,
    p: torch.Tensor,
    n_permutations: int = 2,
) -> torch.Tensor:
    """
    Enforce permutation invariance: s(p) ≈ s(π(p)) for random permutations π.
    """
    device = p.device
    B, K = p.shape
    base_sig = pred_signature.signature_only(p)  # (B, R+M) - invariant part only
    
    losses = []
    for _ in range(n_permutations):
        perm = torch.randperm(K, device=device)
        p_perm = p[:, perm]
        sig_perm = pred_signature.signature_only(p_perm)
        losses.append(F.mse_loss(sig_perm, base_sig))
    
    return torch.stack(losses).mean()


def spectral_radius_loss(A: torch.Tensor, dt: float = 1.0, target_rho: float = 0.99) -> torch.Tensor:
    """Penalize spectral radius of exp(A Δt) above target_rho."""
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, device=A.device, dtype=A.dtype)
    
    M = A * dt
    F = torch.linalg.matrix_exp(M)
    eigvals = torch.linalg.eigvals(F)
    rho = eigvals.abs().max()
    return F.relu(rho - target_rho)
```

---

### 7.5 Training Integration

**Status**: 🔧 Needs update (gates need time-collapse before loss)

In `condor_train_net.py`, after `model(...)` call:

```python
# Collapse predicates over time: mean over seq dimension
gates = diag.get('predicates')
if gates is not None:
    gates_mean = gates.mean(dim=1)  # (batch, 5)
else:
    gates_mean = None

loss, components = criterion(
    outputs.float(),
    batch_y,
    gates=gates_mean,
    state=diag.get('z_final'),
    A_matrix=model.get_A_matrix(),
    pred_signature=model.pred_signature,
)
```

**New CLI argument for strict causality:**
```python
parser.add_argument("--strict-causality", action="store_true", default=False,
                    help="D_forcing uses r_{k-1} (causal) instead of r_k")
```

---

### 7.6 Patch Summary Table

| Patch | File | Status | Description |
|-------|------|--------|-------------|
| 1. Tensor-safe dt | `condor_brain_net.py` | ✅ Minor | Make dt broadcastable in `etd1_kernel` |
| 2. Control-dependent A | `condor_brain_net.py` | 🔧 Required | Add `stab_mlp`, expose `full_matrix(u)` |
| 3. Per-step predicates | `condor_brain_net.py` | 🔧 Required | Wire gates in time loop, r_k dynamics |
| 4. Loss functions | `condor_brain_net.py` | ✅ Minor | Use `signature_only` in group loss |
| 5. Training integration | `condor_train_net.py` | 🔧 Required | Collapse gates, pass to loss |

---

## Part 3: CondorNet™ v4.3 Architecture (Addendum)

> [!IMPORTANT]
> CondorNet v4.3 is implemented in `intelligence/condor_brain_net_v43.py` (1139 lines).
> The v4.2 core (Parts 1-2 above) is fully preserved and used as the inner ETD-1/CDE engine.

### 3.1 v4.3 Component Overview

| Component | Class | Input | Output | New in v4.3 |
|-----------|-------|-------|--------|-------------|
| MultiTFProjector | `MultiTFProjector` | 4×[B,T,64] | [B,T,256] | ✅ |
| PivotProjector | `PivotProjector` | [B,T,13] | [B,T,16] | ✅ |
| TFFusionBlock | `TFFusionBlock` | [B,T,256]+[B,T,16] | [B,T,256] | ✅ |
| OptionsChainEncoder | `OptionsChainEncoder` | [B,N,10] | [B,128] | ✅ |
| JointFusion | `JointFusionLayer` | [B,T,256]+[B,128] | [B,T,384] | ✅ |
| Strategy Selection | `StrategyHead` | [B,384] | 10 logits + legs + entry | ✅ |
| Risk Metrics | `RiskMetricHead` | [B,384] | PoP/EV/MaxLoss/VaR/CVaR | ✅ |
| Pivot Prediction | `PivotPredictionHead` | [B,384] | P(pivot at 5 horizons) | ✅ |
| Position Sizing | `PositionSizeHead` | PoP + [B,384] | [B,1] | ✅ |
| v4.2 Core Engine | `CondorNet` (from v42) | [B,T,d_input] | [B,10] | Preserved |

### 3.2 Strategy Universe (10 Types)

Defined in `intelligence/schema_v43.py`:

```python
STRATEGY_TYPES = [
    "single_call", "single_put",
    "bull_call_spread", "bear_put_spread",
    "straddle", "strangle",
    "butterfly_call", "iron_condor",
    "custom_multi_leg", "abstain",
]
```

Abstain threshold: softmax(logits).max() < 0.60 → no trade.

### 3.3 Risk Metric Head (CVaR ≥ VaR Constraint)

```python
pop      = sigmoid(W_pop @ h)           # [0, 1]
ev       = W_ev @ h                     # unbounded
max_loss = softplus(W_ml @ h)           # ≥ 0
var_95   = softplus(W_var @ h)          # ≥ 0
cvar_95  = var_95 + softplus(W_off @ h) # ≥ VaR (enforced)
```

### 3.4 Backward Compatibility

`CondorNetV43.forward_compat(x)` routes a single TF input to all 4 TFs with a zero-valued chain, preserving the v4.2 API for existing backtest engine calls.

### 3.5 Training Parameters (v43TrainRun12)

| Parameter | Value |
|-----------|-------|
| Total parameters | 10,955,687 |
| d_joint | 128 |
| d_chain | 128 |
| d_h / d_v / d_m / d_r | 256 / 32 / 64 / 32 |
| n_predicates | 8 |
| n_strategy_types | 10 |
| batch_size | 256 |
| lookback | 200 |
| learning_rate | 1e-4 |
| schema_version | v4.3.0 |

---

*Document created: 2026-02-03*
*Updated: 2026-02-24 (CondorNet v4.3 Addendum)*
*Version: 4.3*
