# Legacy Documentation (Mamba-2 Era)

**Archived:** 2026-01-27

This directory contains documentation from the previous Mamba-2 SSM backbone architecture used prior to the Neural CDE switch (January 2026).

## Status

**ARCHIVED** - For historical reference only

## Superseded By

- **Neural CDE backbone:** `intelligence/models/neural_cde.py`
- **Current architecture docs:** `docs/architecture/cde_engine_logic.dot`
- **Current spec:** `docs/scientific_spec.md` (updated with CDE mathematics)

## Why The Switch?

The Mamba-2 State Space Model was replaced with Neural Controlled Differential Equations (CDE) due to:

| Issue | Mamba-2 | Neural CDE |
|-------|---------|------------|
| **Gradient Stability** | Exploding/vanishing gradients, required clip=1.0 | Tanh-bounded vector field, inherently stable |
| **Feature Collapse** | Frequent constant-output predictions | Continuous dynamics prevent collapse |
| **Time Modeling** | Discrete steps (ignores market gaps) | Continuous integral (natural time handling) |
| **Code Complexity** | Hardware-specific CUDA kernels | Simple PyTorch ops (50 lines) |

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

This has been replaced by the Neural CDE formulation:

```
dZ_t = f(Z_t) dX_t

Z_T = Z_0 + ∫₀ᵀ f(Z_t) dX_t
```

See `docs/scientific_spec.md` for current mathematical specification.
