# CondorBrain Architecture (v3.0 - Neural CDE)

This document visualizes the **CondorBrain v3.0** model architecture. It features **Neural Controlled Differential Equations (CDE)** as the backbone, **Volatility-Gated Attention**, and a **Sparse Top-K Mixture-of-Experts** head.

> **Migration Note:** This replaces the Mamba-2 SSM backbone (v2.x). See `docs/legacy/` for archived Mamba documentation.

## Architecture Flowchart

```mermaid
graph TD
    subgraph Inputs
    Input[Input Tensor<br/>(Batch, Seq=256, Feat=54)]
    Norm[Robust Normalization<br/>(Median/MAD)]
    X0[First Observation X₀]
    end

    subgraph "Neural CDE Backbone"
    Encoder[Initial Encoder<br/>Linear + Softplus<br/>Z₀ = softplus(W·X₀)]

    subgraph "CDE Integration Loop"
    dX[Control Increments<br/>dX_t = X_{t+1} - X_t]
    VecField[<b>Vector Field f(Z)</b><br/>Linear → SiLU → Linear → Tanh<br/>‖f(z)‖∞ ≤ 1]
    Euler[<b>Euler Integration</b><br/>Z_{t+1} = Z_t + f(Z_t)·dX_t]
    end

    FinalState[Final Hidden State<br/>Z_T ∈ (Batch, 512)]
    RMSNorm[RMSNorm]
    end

    subgraph "Intelligent Output Head"
    VolAttn[<b>VolGatedAttn</b><br/>(Volatility Awareness)]
    Regime[Regime Detector<br/>(Low/Normal/High Vol)]

    subgraph "Top-K Mixture of Experts"
    Router[<b>Router Gate</b><br/>Select Top-1 Expert]
    Experts[<b>3 Expert Heads</b><br/>(Specialized Networks)]
    end

    Output[<b>Final Output</b><br/>(10 Iron Condor Params)]
    end

    %% Flow Connections
    Input --> Norm
    Norm --> X0
    Norm --> dX
    X0 --> Encoder
    Encoder --> Euler
    dX --> VecField
    VecField --> Euler
    Euler -.->|t=0 to T-1| Euler
    Euler --> FinalState
    FinalState --> RMSNorm

    RMSNorm --> VolAttn
    VolAttn --> Regime
    VolAttn --> Router
    Router -- "Routing Weights" --> Experts
    Experts --> Output

    %% Styling
    style VecField fill:#48bb78,stroke:#333,color:black
    style Euler fill:#68d391,stroke:#333,color:black
    style VolAttn fill:#ff9900,stroke:#333,color:black
    style Router fill:#00ccff,stroke:#333,color:black
    style Output fill:#66ff66,stroke:#333,color:black
```

## Key Components

### 1. Neural CDE Backbone

The core innovation replacing Mamba-2. The CDE treats the input sequence as a **control path** that drives latent state evolution through a learned vector field.

**Mathematical Formulation:**

$$dZ_t = f(Z_t; \theta) \, dX_t$$

$$Z_T = Z_0 + \int_0^T f(Z_t) \, dX_t$$

**Why CDE over Mamba-2:**
| Aspect | Mamba-2 | Neural CDE |
|--------|---------|------------|
| Stability | Gradient clipping required | Tanh-bounded (inherently stable) |
| Feature Collapse | Frequent | Prevented |
| Time Modeling | Discrete steps | Continuous integral |
| Interpretability | Black-box SSM | Vector field Jacobian |

### 2. Vector Field Network

The heart of the CDE, parameterized as a 2-layer MLP:

```
f(z) = tanh(W₂ · SiLU(W₁ · z + b₁) + b₂)
```

- **Tanh stabilization** bounds outputs to [-1, 1], preventing state explosion
- **SiLU activation** provides smooth gradients

### 3. Volatility-Gated Attention (`VolGatedAttn`)

Applied to the final CDE hidden state, this module allows the model to attend to specific high-volatility events globally, providing regime-aware context blending.

### 4. Top-K Mixture of Experts (`TopKMoE`)

Instead of a single dense output layer, we use a sparse router that selects the single best "Expert" network for the current market condition. Three experts specialize in:
- **Expert 1:** Low volatility regimes (tight wings)
- **Expert 2:** Normal volatility (standard IC)
- **Expert 3:** High volatility (wide wings, premium collection)

### 5. Regime Detector

A parallel auxiliary head that explicitly classifies the market state (Low Vol, Normal, High Vol). This provides:
- Interpretability of model decisions
- Guidance for the MoE router
- Explicit regime probabilities for position sizing

## Output Specification (10 Parameters)

| Index | Output | Range | Purpose |
|-------|--------|-------|---------|
| 0 | `short_call_offset` | 0-5% | ATM distance for short call |
| 1 | `short_put_offset` | 0-5% | ATM distance for short put |
| 2 | `wing_width` | 0-$10 | Long strike offset |
| 3 | `dte_selection` | 2-45 days | Optimal days to expiry |
| 4 | `prob_profit` | 0-1 | Estimated win probability |
| 5 | `expected_roi` | -50% to +50% | Return on risk |
| 6 | `max_loss_pct` | 0-1 | Max loss fraction |
| 7 | `confidence` | 0-1 | Model certainty |
| 8 | `entry_logit` | raw | Entry signal |
| 9 | `exit_logit` | raw | Exit signal |

---

## Repository Sync Addendum (2026-01-27)

This document is part of the synchronized documentation set. The authoritative engineering spec and audit references are:

- `docs/INTEGRATION_PLAN_MASTER.md`
- `docs/INTERFACE_CATALOG.md`
- `docs/scientific_spec.md`

Key alignment requirements:
1. Feature schema selection by **name** (V2.2) only; no CSV order dependence.
2. Dataset column order differs across years; schema validation must be strict.
3. Model config metadata (layers/heads/input_dim) must match deployed checkpoints.
4. **CDE backbone is now default** - use `--cde` flag (enabled by default in training).

If this document conflicts with the master spec, the master spec governs implementation.

---

## Model Profile Alignment (Addendum)

This document describes the *conceptual* architecture. Deployed configurations may differ by profile. The authoritative model metadata comes from checkpoint fields:

- `feature_cols` (V2.2: 54 features)
- `input_dim` (54)
- `seq_len` (256)
- `model_config.use_cde` (True)

Always prefer checkpoint metadata over static defaults in docs or code.
