# Legacy Documentation (Mamba-2 → Neural CDE → CondorNet™ Era)

**Archived:** 2026-02-04

This directory contains documentation from the previous Mamba-2 SSM and Neural CDE backbone architectures, prior to the **CondorNet™ v4.0** unified architecture (February 2026).

## Status

**ARCHIVED** - For historical reference only

## Superseded By

- **CondorNet™ backbone:** `intelligence/condor_brain_net.py`
- **Current architecture docs:** `docs/condor_brain_architecture.md`
- **Current spec:** `docs/scientific_spec.md` (updated with CondorNet™ mathematics)

## Architecture Evolution

| Version | Architecture | Status |
|---------|--------------|--------|
| v1.x | TFT (Temporal Fusion Transformer) | Failed to converge |
| v2.x | Mamba-2 SSM | NaN explosions during training |
| v3.x | Neural CDE | Overfitting, insufficient for complex regime dynamics |
| **v4.0** | **CondorNet™** | Unified fusion of all three paradigms with ETD-1 stability |

## Why The Switch?

The Mamba-2 State Space Model and Neural CDE were replaced with **CondorNet™** due to:

| Issue | Mamba-2 | Neural CDE | CondorNet™ |
|-------|---------|------------|------------|
| **Gradient Stability** | Exploding/vanishing gradients, NaN explosions | Tanh-bounded vector field | ETD-1 exponential integrator guarantees stability |
| **Feature Collapse** | Frequent constant-output predictions | Occasional overfitting | TFT control synthesis prevents collapse |
| **Regime Dynamics** | Implicit only | Insufficient | Explicit r_k combinatorics with 5 predicate gates |
| **Time Modeling** | Discrete steps | Continuous integral | Unified: TFT + CDE path response + ETD-1 |

## Archived Files

| File | Original Purpose |
|------|-----------------|
| `mamba_engine_logic.dot/png` | Mamba block diagram showing selective scan kernel |
| `Selective SSM Block Mamba2.dot/png` | Single SSM block internals |
| `Mamba-2 SSD Block Matrix.dot/png` | Semiseparable matrix decomposition |
| `Mamba-2 Semiseparable SSD Matrix Block Decomposition.dot/png` | Multi-panel SSM/SSD visualization |
| `Mamba2_MultiPanel_SSM_SSD.dot` | Publication-quality multi-panel diagram |
| `SSD_Banded_Outer_Product.dot/png` | Banded matrix visualization |
| `MAMBA2_FIX_VERIFICATION.md` | Model collapse fix verification report |

## Mathematical Reference (Archived)

The Mamba-2 SSM used the following formulation:

```
Continuous Form:
  ḣ(t) = Ah(t) + Bx(t)
  y(t) = Ch(t) + Dx(t)

Discretization (Zero-Order Hold):
  Ā = exp(ΔA)
  B̄ = ΔB

Selective Scan:
  h_t = Ā_t h_{t-1} + B̄_t x_t
  y_t = Ch_t
```

This has been replaced by the **CondorNet™** formulation:

```
Master Equation (ETD-1):
x_k = e^{A_θ(u_k)Δt_k} x_{k-1} + Δt_k φ₁(A_θ(u_k)Δt_k) B_θ(u_k) + G_θ(x_{k-1}, u_k) ΔX_k + D(Greeks_k, r_{k-1}, q_k)

Where:
- φ₁(M) = M⁻¹(e^M - I) is the ETD-1 basis function
- u_k = TFT(X_{1:k}) is the TFT control embedding
- G_θ(x, u) · ΔX_k is the Neural CDE path response
- D is the 4-block forcing term
```

See `docs/scientific_spec.md` and `docs/condor_brain_architecture.md` for current mathematical specification.
