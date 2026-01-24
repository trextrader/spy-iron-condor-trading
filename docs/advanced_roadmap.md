# Advanced Forecasting Strategies for CondorBrain (v3.0 Roadmap)

**Status:** Proposal / Research Phase
**Target:** 32-step OHLCV Prediction & Robust Risk Alignment

To improve 32-step OHLCV predictions, we propose a multi-pronged approach: enhancing the Mamba/MoE architecture, integrating generative and geometric methods, using robust/risk-aware losses, and explicitly modeling volatility regimes and uncertainty.

## 1. Enhancing Mamba/MoE Architecture

The Mamba SSM (and its successor Mamba-2) already achieves state-of-the-art accuracy and efficiency on sequence tasks. We can further specialize it via Mixture-of-Experts (MoE) gating on market regimes.

### Volatility-Gated MoE
Use an adaptive gating network that routes inputs (or volatility indicators) to different experts.
*   **Concept**: One expert (e.g., RNN/SSM) handles high-volatility regimes, another (linear) handles calm periods.
*   **Evidence**: Volatility-gated MoE achieved ~33% MSE reduction on volatile stocks.
*   **Efficiency**: MoE-Mamba reaches accuracy in ≈2.2× fewer steps.

**Implementation Pattern:**

```python
class VolatilityGatedMoE(nn.Module):
    """Mixture-of-Experts where gate weights depend on input (e.g. volatility feature)."""
    def __init__(self, input_dim, out_dim, n_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, n_experts)      # gating network
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, out_dim))
            for _ in range(n_experts)
        ])
    def forward(self, x):
        logits = self.gate(x)                     # (batch, n_experts)
        weights = F.softmax(logits, dim=-1)       # expert weights
        outs = torch.stack([exp(x) for exp in self.experts], dim=2)
        return torch.sum(outs * weights.unsqueeze(1), dim=2)
```

## 2. Generative and Geometric Integrations

### Diffusion Heads
Treat the 32-future bars as a "latent" series and train a score-based model to sample from the conditional distribution.
*   **Advantage**: Stable training compared to GANs, smooth trajectories (TransFusion).
*   **Action**: Train a small score-network to refine Mamba outputs by denoising.

### Topological Data Analysis (TDA) Loss
Penalize differences in persistent diagrams using a "Topological Consistency Loss":
$$ L_{topo} = || \Phi(y_{pred}) - \Phi(y_{true}) || $$
Where $\Phi$ maps time series to persistence diagrams (loop lifetimes). This forces preservation of high-level cycle structures.

### Manifold-Based Denoising
Project input/predicted trajectories onto a learned smooth manifold to filter noise.
*   **Technique**: Autoencoder or Diffusion Map on lookback windows.
*   **Constraint**: Impose smoothness/curvature bounds (Riemannian regularization) to ensure $M^* \in C^2$.

## 3. Robust and Risk-Aware Loss Functions

Standard MSE explodes on outliers. We adopt financially-informed objectives:

### Robust Losses
*   **Huber Loss**: Quadratic for small errors, linear for large deviations.
*   **Quantile (Pinball) Loss**: Learn prediction intervals (e.g., 10th/90th percentiles) directly.

### Sharpe Ratio Loss
Embed the Sharpe ratio directly into the optimization objective:
$$ \mathcal{L}_{sharpe} = - \frac{\mathbb{E}[R]}{\text{Std}(R) + \epsilon} $$
This forces the network to prefer paths that yield high risk-adjusted returns, not just low MSE.

## 4. Adapting to Volatility Regimes

### Online/Adaptive Training (AdaVol)
Implementing online learning rules to update parameters more frequently during crises.
*   **Mechanism**: Use an "AdaVol" estimator of conditional variance.
*   **Response**: Double-weight loss during FOMC/News events ($d = \log(H/L)$ spikes).

### Time-Varying SSM State
$$ h_{t+1} = A_t h_t + B_t x_t $$
Where $A_t, B_t$ adapt based on recent error or volatility.

## 5. Probabilistic Forecasting

Move beyond point forecasts to predictive distributions.

### Neural Likelihood (Gaussian NLL)
Dual-headed network predicting Mean ($\mu$) and Log-Variance ($\log \sigma^2$).
Optimization via Negative Log Likelihood:
$$ \mathcal{L}_{NLL} = \frac{1}{2} \sum_t \left[ (y_t - \mu_t)^2 e^{-\log \sigma^2_t} + \log \sigma^2_t \right] $$

### Generative Sampling
Draw multiple sample futures via Diffusion head to form a non-parametric confidence band.

---

**Summary Recommendation for v3.0:**
1.  **Architecture**: MoE-Mamba-2 with Volatility Gating.
2.  **Generative**: Add Diffusion refinement head.
3.  **Losses**: Composite Huber + Sharpe + Quantile.
4.  **Adaptation**: Online learning on high-vol triggers.
5.  **Output**: Probabilistic (Mean + Variance).


---

## Repository Sync Addendum (2026-01-24)

This document is part of the synchronized documentation set. The authoritative engineering spec and audit references are:

- `docs/INTEGRATION_PLAN_MASTER.md`
- `docs/INTERFACE_CATALOG.md`

Key alignment requirements:
1. Feature schema selection by **name** (V2.2) only; no CSV order dependence.
2. Dataset column order differs across years; schema validation must be strict.
3. Model config metadata (layers/heads/input_dim) must match deployed checkpoints.

If this document conflicts with the master spec, the master spec governs implementation.
