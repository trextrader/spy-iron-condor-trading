# CondorBrain Training Dashboard Guide

## Overview

The Training Dashboard provides real-time visualization of CondorNet model training. This guide explains how to interpret each metric, chart, and visualization.

---

## Fuzzy Gate Heatmap

The heatmap displays **predicate gate activations** in real-time, showing which trading conditions the model detects.

### Color Scale (0.0 to 1.0)

| Value | Color | Meaning |
|-------|-------|---------|
| **0.0** | Dark (black/purple) | Gate is **OFF** - condition not met |
| **0.5** | Mid (orange) | Gate is **uncertain** - near decision threshold |
| **1.0** | Bright (yellow/white) | Gate is **ON** - condition strongly met |

### Gate Definitions (Rows 0-9)

| Gate | Name | Learned Rule | Trading Implication |
|------|------|--------------|---------------------|
| 0 | Vol Spike | `IVR > 75` | High IV environment - favorable for selling premium |
| 1 | Liquidity Lock | `Spread/Price > 1.12%` | Wide spreads - block entry to avoid slippage |
| 2 | Trend Reversal | `RSI < 25` | Oversold condition - potential mean reversion |
| 3 | Gap Guard | `1m price jump > 1.12%` | Sudden move detected - increase caution |
| 4 | Gamma Hedge | `|Γ| > 0.011` | High gamma exposure - consider hedging |
| 5-9 | Learned Predicates | Data-driven | Model-discovered patterns from training data |

### Reading the Heatmap

- **Columns** = Time steps (sliding window, newest on right)
- **Rows** = Individual predicate gates (0-9)
- **Bright horizontal bands** = Gate fires consistently across time
- **Vertical bright stripes** = Multiple gates firing simultaneously (high-signal moment)
- **Dark regions** = Calm market conditions, few triggers active

### Soft Sigmoid Activation

The gates use a soft sigmoid with **steepness=20**, meaning:
- Values transition smoothly between 0 and 1
- You can see "almost triggered" states (0.3-0.7 range)
- This provides early warning before full activation

---

## Loss Metrics (12 Components)

The MetricGrid displays all 12 loss components with live sparklines.

### Primary Loss

| Metric | Range | Target | Description |
|--------|-------|--------|-------------|
| **loss** | Any | ↓ Lower | Combined weighted loss (sum of all components) |

### Prediction Quality

| Metric | Range | Target | Description |
|--------|-------|--------|-------------|
| **mse** | 0 - ∞ | ↓ Lower | Mean Squared Error between predictions and targets |

### Trading Performance

| Metric | Range | Target | Description |
|--------|-------|--------|-------------|
| **npdd** | -1 to 1 | ↑ Higher | **Normalized Profit per Dollar of Drawdown** - risk-adjusted return |
| **sharpe** | -∞ to +∞ | ↑ Higher | **Sharpe Ratio** - return per unit of volatility (annualized) |
| **dd** | 0 to 1 | ↓ Lower | **Drawdown** - maximum peak-to-trough decline |
| **turnover** | 0 to ∞ | ↓ Lower | **Portfolio Turnover** - trading frequency penalty |
| **growth** | Any | ↑ Higher | **Capital Growth** - cumulative return trajectory |

### Model Regularization

| Metric | Range | Target | Description |
|--------|-------|--------|-------------|
| **fuzzy** | 0 to 1 | → Stable | **Fuzzy Gate Variance** - encourages consistent gate behavior |
| **pattern_ent** | -∞ to 0 | ↓ Lower | **Pattern Entropy** - encourages diverse pattern recognition |
| **group_inv** | 0 to ∞ | ↓ Lower | **Group Invariance** - penalizes sensitivity to input permutations |
| **rho** | 0 to 1 | → ~0.01 | **Spectral Radius** - stability of learned dynamics matrix A |
| **energy** | 0 to ∞ | ↓ Lower | **State Energy** - prevents exploding hidden states |

### Interpreting Sparklines

Each metric card shows a mini sparkline (last ~50 values):
- **Downward trend** (for loss, mse, dd, turnover): Good - model improving
- **Upward trend** (for sharpe, npdd, growth): Good - performance improving
- **Flat line**: Model may be converged or stuck
- **Oscillating wildly**: Learning rate may be too high

---

## Loss Trajectory Chart

The streaming line chart shows loss components over training steps.

### Reading the Chart

- **X-axis**: Training steps (global step count)
- **Y-axis**: Loss value
- **Multiple lines**: Each loss component in different color

### Component Colors

| Component | Color | Priority |
|-----------|-------|----------|
| loss | Blue | Primary - watch this most |
| mse | Orange | Prediction accuracy |
| sharpe | Green | Trading performance |
| npdd | Red | Risk-adjusted returns |
| dd | Purple | Drawdown control |

### Toggle Controls

Click component names to show/hide individual lines. Useful for:
- Isolating specific metrics
- Reducing visual clutter
- Comparing related components

### Healthy Training Patterns

- **Steep initial drop**: Model learning quickly from random init
- **Gradual descent**: Stable convergence
- **Plateaus**: May need learning rate adjustment
- **Spikes up then down**: Normal - recovering from bad batch

---

## Diagnostics Panel

Mini charts showing optimizer and training health.

### Learning Rate (LR)

| Pattern | Meaning |
|---------|---------|
| Decreasing curve | Cosine annealing schedule (normal) |
| Flat line | Constant LR (if configured) |
| Step drops | Step LR schedule |

**Current default**: Cosine annealing from initial LR to near-zero over epochs.

### Gradient Norm

| Value | Status | Action |
|-------|--------|--------|
| < 0.1 | Low | May be vanishing gradients |
| 0.1 - 1.0 | Healthy | Normal training |
| 1.0 - 10.0 | Elevated | Watch for instability |
| > 10.0 | High | Gradient clipping active |

**Note**: Gradient clipping at 1.0 is applied, so you shouldn't see values above 1.0 after clipping.

### Scaler Scale (AMP)

For mixed-precision (FP16) training:

| Value | Status |
|-------|--------|
| 65536 | Initial scale (normal) |
| Decreasing | Scaler reducing due to overflow detection |
| Stable | Healthy training |
| Very low (< 1) | May indicate numerical issues |

---

## Training Header

The top banner shows overall progress.

### Metrics Displayed

| Element | Description |
|---------|-------------|
| **Epoch X/Y** | Current epoch out of total |
| **Step N** | Global step count |
| **Progress Ring** | Visual completion percentage |
| **Progress Bar** | Linear completion indicator |
| **ETA** | Estimated time remaining |

### ETA Calculation

```
ETA = (elapsed_time / steps_completed) × steps_remaining
```

Updates every 10 batches based on actual throughput.

---

## WebSocket Status

The dashboard connects via WebSocket for real-time updates.

### Connection States

| Indicator | Meaning |
|-----------|---------|
| Connected | Receiving live data |
| Reconnecting | Temporary disconnection, auto-retry |
| Disconnected | Check backend is running |

### Telemetry Frequency

| Endpoint | Frequency | Data |
|----------|-----------|------|
| `/telemetry/step` | Every 10 batches | All 12 loss components |
| `/telemetry/status` | Every 10 batches | Progress, ETA |
| `/telemetry/fuzzy` | Every 10 batches | Gate activations (10 values) |
| `/telemetry/epoch` | End of epoch | Epoch summary |
| `/telemetry/complete` | End of training | Final results |

---

## Interpreting Training Progress

### Epoch 1 (Initialization)

- Loss starts high/negative (due to Sharpe component)
- Gates may be random (uniform heatmap)
- Rapid improvement expected

### Epochs 2-3 (Learning)

- Loss should decrease significantly
- Sharpe component often improves dramatically
- Gate patterns start emerging in heatmap

### Epochs 4-5 (Refinement)

- Slower improvements
- Gates should show clear patterns
- Watch for overfitting (val loss increasing)

### Signs of Good Training

- [ ] Total loss trending down
- [ ] Sharpe trending up (less negative → positive)
- [ ] Drawdown (dd) staying low
- [ ] Clear patterns in heatmap (not random noise)
- [ ] Gradient norm stable (0.1 - 1.0)

### Signs of Problems

- [ ] Loss increasing or oscillating wildly
- [ ] Gradient norm at 0 (vanishing) or clipped every step
- [ ] All heatmap values stuck at 0.5 (gates not learning)
- [ ] Scaler scale dropping rapidly (numerical instability)

---

## Quick Reference

### Metrics Cheat Sheet

| Metric | Good Direction | Healthy Range |
|--------|----------------|---------------|
| loss | ↓ Down | Depends on config |
| mse | ↓ Down | < 1.0 |
| sharpe | ↑ Up | > 0 (ideally > 1) |
| npdd | ↑ Up | > 0 |
| dd | ↓ Down | < 0.2 |
| turnover | ↓ Down | < 0.5 |
| fuzzy | → Stable | 0.001 - 0.01 |
| rho | → Stable | ~0.01 |
| energy | ↓ Down | < 0.01 |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Esc` | Close any modal |
| `F12` | Open browser dev tools |

---

## Troubleshooting

### Heatmap Not Updating

1. Check backend logs for `/telemetry/fuzzy` requests
2. Verify training script has `--gui-telemetry` flag
3. Check browser console for WebSocket errors

### Metrics Stuck

1. Training may be paused or crashed
2. Check `tail -f train.log` for errors
3. Verify GPU memory isn't exhausted

### High Loss Values

1. Check learning rate (may be too high)
2. Verify data normalization
3. Check for NaN/Inf in debug output

---

*Generated: 2026-02-08*
*CondorBrain GUI v2.3.0*
