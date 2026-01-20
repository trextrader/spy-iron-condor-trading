# Canonical Rule Primitive Library (v2.5)
## For CondorBrain/DeepMamba & AntiGrav Integration

**Version:** 2.5
**Last Updated:** 2026-01-19
**Alignment:** DeepMamba v2.5, Phase 2.5 Lag-Alignment, FIS 11-Factor System

---

## Table of Contents

1. [Overview](#overview)
2. [Primitive Categories](#primitive-categories)
3. [Core Computational Primitives](#core-computational-primitives)
4. [Signal Generation Primitives](#signal-generation-primitives)
5. [Gate Primitives](#gate-primitives)
6. [Membership Function Primitives](#membership-function-primitives)
7. [Composite Primitives](#composite-primitives)
8. [Integration with AntiGrav](#integration-with-antigrav)

---

## Overview

This library provides **52 reusable primitives** that compose all 13 trading rules. Instead of encoding thousands of brittle if/then statements, these primitives can be combined declaratively via the Rule DSL (see Part C).

### Design Principles

1. **Composability**: Each primitive is self-contained and can be combined with others
2. **Parameter Transparency**: All thresholds and coefficients are explicit parameters
3. **Regime Awareness**: Primitives adapt to volatility/curvature/topological regime
4. **Differentiability**: Compatible with neural integration (fuzzy logic, gradient-based)
5. **Auditability**: Clear mathematical formulas and deterministic outputs

---

## Primitive Categories

| Category | Count | Purpose |
|----------|-------|---------|
| Core Computational | 12 | Base indicators (RSI, ADX, MACD, BB, PSAR, TDA) |
| Signal Generation | 15 | Entry/exit signal logic |
| Gate Primitives | 10 | Execution/risk/portfolio filters |
| Membership Functions | 11 | Fuzzy logic memberships |
| Composite Primitives | 4 | High-level rule compositions |
| **TOTAL** | **52** | |

---

## Core Computational Primitives

These primitives compute base indicators and features. All are regime-aware and volatility-adaptive.

### P001: Dynamic Bollinger Bands

**Description**: Volatility-adaptive Bollinger Bands with geometric energy modulation.

**Formula**:
```
Upper_Band(t) = SMA_n(t) + k(t) · σ_n(t)
Middle_Band(t) = SMA_n(t)
Lower_Band(t) = SMA_n(t) - k(t) · σ_n(t)

k(t) = k_0 · (1 + α · VolatilityEnergy(t))
VolatilityEnergy(t) = ATR(t) / SMA_50(ATR(t))
```

**Parameters**:
- `n`: Period (default: 20)
- `k_0`: Base multiplier (default: 2.0)
- `α`: Sensitivity (default: 0.2)

**Output**: `{upper, middle, lower, bandwidth, percentile}`

**References**: Rule A1, B1, C1, D1, E2

---

### P002: Bollinger Band Bandwidth Percentile

**Description**: Percentile ranking of bandwidth for regime-invariant squeeze/expansion detection.

**Formula**:
```
Bandwidth(t) = (Upper_Band(t) - Lower_Band(t)) / Middle_Band(t)
BW_percentile(t, w) = PercentileRank(Bandwidth(t), Bandwidth[t-w:t])
```

**Parameters**:
- `w`: Rolling window (default: 100)

**Output**: `{bandwidth, percentile}`

**References**: Rule A2, C1, E2

---

### P003: Volatility-Normalized MACD

**Description**: MACD normalized by vol_ewma for regime adaptability.

**Formula**:
```
MACD(t) = EMA_12(t) - EMA_26(t)
Signal(t) = EMA_9(MACD(t))
vol_ewma(t) = EWMA_α(|returns(t)|)

MACD_norm(t) = MACD(t) / vol_ewma(t)
Signal_norm(t) = Signal(t) / vol_ewma(t)
```

**Parameters**:
- `fast`: Fast EMA (default: 12)
- `slow`: Slow EMA (default: 26)
- `signal`: Signal EMA (default: 9)
- `α`: EWMA factor (default: 0.94)

**Output**: `{macd_norm, signal_norm, histogram_norm, crossover}`

**References**: Rule A2

---

### P004: Volatility-Normalized ADX

**Description**: ADX normalized by volatility energy for trend strength.

**Formula**:
```
ADX_raw(t) = standard ADX computation
ADX_norm(t) = ADX_raw(t) / (1 + β · VolatilityEnergy(t))
```

**Parameters**:
- `period`: ADX period (default: 14)
- `β`: Normalization coefficient (default: 0.15)

**Output**: `{adx_norm, +DI, -DI, trend_direction}`

**References**: Rule A1, A2, A3, B1

---

### P005: Dynamic RSI

**Description**: RSI weighted by volatility energy and geometric curvature.

**Formula**:
```
RS(t) = Average_Gain_14(t) / Average_Loss_14(t)
RSI_raw(t) = 100 - (100 / (1 + RS(t)))
RSI_dynamic(t) = RSI_raw(t) × (1 + γ · CurvatureProxy(t))

CurvatureProxy(t) = |r(t+1) - 2·r(t) + r(t-1)| / Δt²
```

**Parameters**:
- `period`: RSI period (default: 14)
- `γ`: Curvature weight (default: 0.1)

**Output**: `{rsi_dynamic, overbought, oversold, divergence}`

**References**: Rule A3, B1, B2, D1

---

### P006: Parabolic SAR

**Description**: ATR-scaled PSAR with adaptive acceleration.

**Formula**:
```
PSAR(t) = PSAR(t-1) + AF(t) · (EP(t) - PSAR(t-1))
AF(t) = min(AF_max, AF(t-1) + AF_step · ATR_scale(t))
```

**Parameters**:
- `AF_start`: Initial acceleration (default: 0.02)
- `AF_step`: Increment (default: 0.02)
- `AF_max`: Maximum (default: 0.2)

**Output**: `{psar, position, flip_signal}`

**References**: Rule B1, PSAR Reversion

---

### P007: Volume Ratio

**Description**: Volume relative to rolling mean, regime-aware.

**Formula**:
```
VolumeRatio(t) = Volume(t) / SMA_20(Volume)
VolumeRatio_dynamic(t) = VolumeRatio(t) / (1 + γ · VolEnergy(t))
```

**Parameters**:
- `period`: SMA period (default: 20)
- `γ`: Vol dampening (default: 0.3)

**Output**: `{volume_ratio, spike_flag}`

**References**: Rule A1, C1, D2, E1

---

### P008: IV Confidence (Lag-Aware)

**Description**: Exponential decay based on IV data staleness.

**Formula**:
```
IV_confidence(t) = exp(-λ · lag_minutes(t))
```

**Parameters**:
- `λ`: Decay rate (default: 0.05, 5-minute half-life)

**Output**: `{iv_confidence, stale_flag}`

**References**: Rules A2, B1, C1, C2, D2, E2

---

### P009: Persistent Homology β₁ (Normalized)

**Description**: Topological regime signature via Takens embedding and persistent homology.

**Formula**:
```
X(t) = Takens_Embed(P[t-d·τ:t], τ, d)
dgm1 = Ripser(X, maxdim=1)
β₁_raw(t) = count(dgm1 with persistence > θ · max_persistence)
β₁_norm(t) = (β₁_raw(t) - μ_β₁(w)) / σ_β₁(w)
```

**Parameters**:
- `τ`: Time delay (default: 1)
- `d`: Embedding dimension (default: 3)
- `θ`: Persistence threshold (default: 0.1)
- `w`: Normalization window (default: 100)

**Output**: `{beta1_norm, breakout_flag, consolidation_flag}`

**References**: Rules C2, E3

---

### P010: Curvature Proxy

**Description**: Second derivative of log-returns for geometric regime detection.

**Formula**:
```
κ(t) = |r(t+1) - 2·r(t) + r(t-1)| / Δt²
```

**Output**: `{curvature}`

**References**: Rules C2, E3, P005

---

### P011: Multi-Timeframe Consensus

**Description**: Weighted consensus across 1m, 5m, 15m timeframes.

**Formula**:
```
MTF_consensus(t) = (w_1m · Signal_1m + w_5m · Signal_5m + w_15m · Signal_15m)
Signal_tf ∈ {-1, 0, +1}  # bearish, neutral, bullish
```

**Parameters**:
- `w_1m`: 1-minute weight (default: 0.2)
- `w_5m`: 5-minute weight (default: 0.3)
- `w_15m`: 15-minute weight (default: 0.5)

**Output**: `{mtf_consensus, alignment_flag}`

**References**: Rules A1, B1, C1, E2

---

### P012: IV Rank (Percentile)

**Description**: IV percentile rank over lookback window.

**Formula**:
```
IVR(t) = PercentileRank(IV(t), IV[t-n:t]) × 100
```

**Parameters**:
- `n`: Lookback days (default: 252)

**Output**: `{ivr}`

**References**: Rule B1

---

## Signal Generation Primitives

These primitives generate binary or continuous entry/exit signals.

### S001: BB Breakout Signal

**Description**: Price closes outside dynamic Bollinger Band.

**Formula**:
```
Breakout_bullish(t) = Close(t) > Upper_Band(t)
Breakout_bearish(t) = Close(t) < Lower_Band(t)
```

**Output**: `{breakout_direction, strength}`

**References**: Rules A1, A2, C1, D1, D2

---

### S002: BB Reversion Signal

**Description**: Price touches band extreme with reversion confirmation.

**Formula**:
```
Reversion_bearish(t) = Close(t) >= Upper_Band(t) AND RSI(t) > 70
Reversion_bullish(t) = Close(t) <= Lower_Band(t) AND RSI(t) < 30
```

**Output**: `{reversion_direction, confidence}`

**References**: Rules B1, B2, PSAR Reversion

---

### S003: MACD Crossover Signal

**Description**: MACD line crosses signal line.

**Formula**:
```
Bullish_cross(t) = MACD_norm(t) > Signal_norm(t) AND MACD_norm(t-1) <= Signal_norm(t-1)
Bearish_cross(t) = MACD_norm(t) < Signal_norm(t) AND MACD_norm(t-1) >= Signal_norm(t-1)
```

**Output**: `{crossover_direction, histogram}`

**References**: Rule A2

---

### S004: ADX Trend Signal

**Description**: ADX confirms trend strength and direction.

**Formula**:
```
Trend_bullish(t) = ADX_norm(t) > 25 AND +DI(t) > -DI(t) AND ADX_norm(t) > ADX_norm(t-1)
Trend_bearish(t) = ADX_norm(t) > 25 AND -DI(t) > +DI(t) AND ADX_norm(t) > ADX_norm(t-1)
```

**Output**: `{trend_direction, strength}`

**References**: Rules A2, A3

---

### S005: RSI Divergence Signal

**Description**: Price makes new extreme but RSI doesn't confirm.

**Formula**:
```
Bearish_divergence(t) = High(t) > High(t_prev_swing) AND RSI(t) < RSI(t_prev_swing)
Bullish_divergence(t) = Low(t) < Low(t_prev_swing) AND RSI(t) > RSI(t_prev_swing)
```

**Output**: `{divergence_type, strength}`

**References**: Rule B2

---

### S006: BB Squeeze Signal

**Description**: Bandwidth compresses below percentile threshold.

**Formula**:
```
Squeeze(t) = BW_percentile(t, w) < 10
```

**Output**: `{squeeze_flag, duration}`

**References**: Rules C1, E2

---

### S007: BB Expansion Signal

**Description**: Bandwidth expands above percentile threshold.

**Formula**:
```
Expansion(t) = BW_percentile(t, w) > 75 AND Expansion_rate(t, 5) > 0
```

**Output**: `{expansion_flag, rate}`

**References**: Rules A2, E2

---

### S008: Volume Spike Signal

**Description**: Volume exceeds dynamic threshold.

**Formula**:
```
Spike(t) = VolumeRatio(t) > VolumeThreshold_dynamic(t)
VolumeThreshold_dynamic(t) = 1.5 · (1 + γ · max(0, VolEnergy(t) - 1))
```

**Output**: `{spike_flag, ratio}`

**References**: Rules A1, C1, D2

---

### S009: PSAR Flip Signal

**Description**: PSAR changes trend direction.

**Formula**:
```
Flip_bullish(t) = PSAR(t) < Close(t) AND PSAR(t-1) >= Close(t-1)
Flip_bearish(t) = PSAR(t) > Close(t) AND PSAR(t-1) <= Close(t-1)
```

**Output**: `{flip_direction}`

**References**: Rule B1, PSAR Reversion

---

### S010: Topological Breakout Signal

**Description**: Persistent homology detects cycle destruction (breakout).

**Formula**:
```
Breakout(t) = β₁_norm(t) < -1.0 AND Δβ₁_norm(t) < -0.5
```

**Output**: `{breakout_flag, regime_score}`

**References**: Rule C2

---

### S011: Topological Consolidation Signal

**Description**: Persistent homology detects cycle emergence (consolidation).

**Formula**:
```
Consolidation(t) = β₁_norm(t) > +2.0 AND Δβ₁_norm(t) > +0.5
```

**Output**: `{consolidation_flag, chaos_flag}`

**References**: Rules C2, E3

---

### S012: Chaos Detection Signal

**Description**: Sharp rise in β₁ indicates chaotic regime.

**Formula**:
```
Chaos(t) = β₁_gated(t) > +2.5 AND Δβ₁_gated(t) > +0.5
β₁_gated(t) = β₁_norm(t) · (1 + α·κ(t)) · (1 + β·VolEnergy(t))
```

**Output**: `{chaos_flag, membership}`

**References**: Rule E3

---

### S013: MTF Alignment Signal

**Description**: Multi-timeframe consensus exceeds threshold.

**Formula**:
```
Aligned_bullish(t) = MTF_consensus(t) > 0.7
Aligned_bearish(t) = MTF_consensus(t) < -0.7
```

**Output**: `{alignment_direction, strength}`

**References**: Rules A1, B1, C1, E2

---

### S014: Swing High/Low Detection

**Description**: Identifies local extrema for divergence analysis.

**Formula**:
```
Swing_high(t) = High(t) > High(t-k) AND High(t) > High(t+k) for k in window
Swing_low(t) = Low(t) < Low(t-k) AND Low(t) < Low(t+k)
```

**Output**: `{swing_type, timestamp}`

**References**: Rule B2

---

### S015: Reversal Candle Pattern

**Description**: Detects bullish/bearish reversal candle patterns.

**Formula**:
```
Bearish_reversal(t) = shooting_star OR bearish_engulfing OR evening_star
Bullish_reversal(t) = hammer OR bullish_engulfing OR morning_star
```

**Output**: `{pattern_type, reliability}`

**References**: Rules B1, B2

---

## Gate Primitives

These primitives implement execution, risk, and portfolio constraints.

### G001: Execution Friction Gate (EFG)

**Description**: Prevents entries/exits when spread cost exceeds realized range.

**Formula**:
```
AvgRange(n, t) = (1/n) · Σ (High(t-i) - Low(t-i))
Spread(t) = Ask(t) - Bid(t)  # per leg, sum for multi-leg
F(t) = Spread(t) / AvgRange(n, t)

ALLOW if F(t) < θ(t)
BLOCK if F(t) >= θ(t)

θ(t) = clamp(θ₀ + a·z(ATR_n) + b·z(VolRatio) - c·z(BW) - d·Event(t), θ_min, θ_max)
```

**Parameters**:
- `n`: Avg range period (default: 20)
- `θ₀`: Base threshold (default: 1.0)
- `θ_min`: Min threshold (default: 0.5)
- `θ_max`: Max threshold (default: 1.5)
- `a, b, c, d`: Regime coefficients

**Output**: `{allow_flag, friction_ratio, threshold}`

**References**: Rule E1

---

### G002: Gap Risk Override Gate (GRO)

**Description**: Forces exit regardless of friction when gap risk elevated.

**Formula**:
```
G(t) = w₁·EventFlag(t) + w₂·ATR_spike(t) + w₃·BW_expansion(t) + w₄·LateDay(t)

FORCE_EXIT if ExitSignal(t) = True AND G(t) >= G_crit
```

**Parameters**:
- `G_crit`: Critical gap risk (default: 0.7)
- `w₁, w₂, w₃, w₄`: Risk weights

**Output**: `{override_flag, gap_risk_score}`

**References**: Rule E1

---

### G003: IV Confidence Gate

**Description**: Defers entries when IV data stale.

**Formula**:
```
ALLOW if IV_confidence(t) > threshold
DEFER if IV_confidence(t) <= threshold
```

**Parameters**:
- `threshold`: Min confidence (default: 0.5)

**Output**: `{allow_flag, confidence}`

**References**: Rules A2, C1, D2, E2

---

### G004: MTF Consensus Gate

**Description**: Requires multi-timeframe alignment.

**Formula**:
```
ALLOW_LONG if MTF_consensus(t) > +0.7
ALLOW_SHORT if MTF_consensus(t) < -0.7
BLOCK otherwise
```

**Output**: `{allow_flag, consensus}`

**References**: Rules C1, E2

---

### G005: Volatility Regime Gate

**Description**: Filters entries based on volatility state.

**Formula**:
```
ALLOW if VolEnergy(t) < high_vol_threshold
BLOCK if VolEnergy(t) >= high_vol_threshold
```

**Parameters**:
- `high_vol_threshold`: Max vol energy (default: 2.0)

**Output**: `{allow_flag, vol_energy}`

**References**: Rule C2

---

### G006: IV Rank Gate

**Description**: Minimum IV rank for premium selling.

**Formula**:
```
ALLOW if IVR(t) > min_ivr
BLOCK otherwise
```

**Parameters**:
- `min_ivr`: Minimum IV rank (default: 20.0)

**Output**: `{allow_flag, ivr}`

**References**: Rule B1

---

### G007: Breakout Score Gate

**Description**: Composite breakout confidence.

**Formula**:
```
Breakout_score(t) = Signal_strength · Vol_confirmation · IV_confidence
ALLOW if Breakout_score(t) > threshold
```

**Parameters**:
- `threshold`: Min score (default: 0.7)

**Output**: `{allow_flag, score}`

**References**: Rules A2, D2

---

### G008: Fuzzy Consensus Gate

**Description**: 11-factor fuzzy logic consensus.

**Formula**:
```
FuzzyScore(t) = Σᵢ wᵢ · μᵢ(t)
ALLOW if FuzzyScore(t) > threshold
```

**Parameters**:
- `threshold`: Min fuzzy score (default: 0.7)
- `w₁...w₁₁`: Factor weights

**Output**: `{allow_flag, fuzzy_score, memberships}`

**References**: Rule B1

---

### G009: Position Size Dampening Gate

**Description**: Fuzzy chaos dampening (not hard block).

**Formula**:
```
ChaosMembership(t) = sigmoid(β₁_gated(t) - 2.0)
PositionSize_adjusted(t) = PositionSize_base · (1 - ChaosMembership(t))
```

**Output**: `{size_multiplier, chaos_membership}`

**References**: Rule E3

---

### G010: Portfolio Greeks Gate

**Description**: Delta/gamma/vega limits.

**Formula**:
```
ALLOW if |NetDelta + TradeDelta| < DeltaBudget
BLOCK otherwise
```

**Parameters**:
- `DeltaBudget`: Max net delta (default: 1000)

**Output**: `{allow_flag, net_greeks}`

**References**: CondorBrain RiskManager

---

## Membership Function Primitives

These primitives compute fuzzy memberships for the 11-factor FIS system.

### M001: MTF Consensus Membership

**Formula**:
```
μ_MTF(t) = |MTF_consensus(t)|  # normalized 0-1
```

**References**: Rule B1

---

### M002: IV Rank Membership

**Formula**:
```
μ_IVR(t) = IVR(t) / 100
```

**References**: Rule B1

---

### M003: VIX Regime Membership

**Formula**:
```
μ_VIX(t) = 1 - sigmoid((VIX(t) - 20) / 5)  # low VIX = high membership
```

**References**: Rule B1

---

### M004: RSI Membership

**Formula**:
```
μ_RSI_overbought(t) = max(0, (RSI(t) - 70) / 30)
μ_RSI_oversold(t) = max(0, (30 - RSI(t)) / 30)
```

**References**: Rule B1

---

### M005: Stochastic Membership

**Formula**:
```
μ_Stoch(t) = 1 - |Stoch(t) - 50| / 50  # neutral = high membership
```

**References**: Rule B1

---

### M006: ADX Membership

**Formula**:
```
μ_ADX(t) = 1 - ADX_norm(t) / 50  # weak trend = high membership for IC
```

**References**: Rule B1

---

### M007: SMA Distance Membership

**Formula**:
```
μ_SMA(t) = 1 - min(|Close(t) - SMA(t)| / SMA(t), 0.05) / 0.05
```

**References**: Rule B1

---

### M008: PSAR Membership

**Formula**:
```
μ_PSAR_reversion(t) = 1 if PSAR suggests reversal direction, 0 otherwise
```

**References**: Rule B1

---

### M009: Bollinger Band Position Membership

**Formula**:
```
μ_BB(t) = |Close(t) - Middle(t)| / |Upper(t) - Middle(t)|
```

**References**: Rule B1

---

### M010: Bollinger Band Squeeze Membership

**Formula**:
```
μ_BBsqueeze(t) = 1 - BW_percentile(t, 100) / 100
```

**References**: Rule B1

---

### M011: Volume Membership

**Formula**:
```
μ_Vol(t) = min(1.0, VolumeRatio(t) / 0.8)
```

**References**: Rule B1

---

## Composite Primitives

These primitives combine multiple lower-level primitives for complex logic.

### C001: Iron Condor Entry Filter

**Description**: Complete filter for iron condor entries.

**Formula**:
```
IC_entry_allowed(t) =
    MTF_consensus(t) > 0.9 AND
    FuzzyScore(t) > 0.8 AND
    IVR(t) > 20 AND
    β₁_norm(t) < 1.5 AND  # not consolidating
    IV_confidence(t) > 0.5
```

**Output**: `{allow_flag, composite_score}`

---

### C002: Breakout Confirmation Composite

**Description**: Multi-factor breakout validation.

**Formula**:
```
Breakout_confirmed(t) =
    S001(t) AND  # BB breakout
    S008(t) AND  # Volume spike
    S013(t) AND  # MTF alignment
    G003(t) AND  # IV confidence
    G007(t)      # Breakout score
```

**Output**: `{confirmed_flag, strength}`

---

### C003: Mean Reversion Composite

**Description**: Multi-factor reversion validation.

**Formula**:
```
Reversion_confirmed(t) =
    S002(t) AND  # BB reversion signal
    S005(t) AND  # RSI divergence
    G008(t)      # Fuzzy consensus
```

**Output**: `{confirmed_flag, confidence}`

---

### C004: Regime Shift Composite

**Description**: Topological + geometric regime detection.

**Formula**:
```
Regime_shift(t) =
    S010(t) OR S011(t) AND  # TDA signal
    VolEnergy(t) > 1.2 AND  # Volatility spike
    κ(t) > κ_threshold      # Curvature anomaly
```

**Output**: `{shift_type, strength}`

---

## Integration with AntiGrav

### Primitive Encoding

Each primitive is encoded in the Rule DSL (Part C) with:
- **ID**: Unique identifier (e.g., "P001", "S003", "G001")
- **Name**: Human-readable name
- **Category**: Core/Signal/Gate/Membership/Composite
- **Parameters**: Default values and ranges
- **Dependencies**: Required primitives/features
- **Output Schema**: Return value structure

### Feature Pipeline Mapping

All primitives map to pre-computed features in the CondorBrain v2.1 feature schema:

| Primitive | Feature Column |
|-----------|----------------|
| P001 | `bb_upper`, `bb_middle`, `bb_lower`, `bb_bandwidth` |
| P002 | `bb_percentile` |
| P003 | `macd_norm`, `macd_signal_norm` |
| P004 | `adx_norm`, `plus_di`, `minus_di` |
| P005 | `rsi_dynamic` |
| P006 | `psar`, `psar_position` |
| P007 | `volume_ratio` |
| P008 | `iv_confidence` |
| P009 | `beta1_norm` |
| P010 | `curvature_proxy` |
| P011 | `mtf_consensus` |
| P012 | `ivr` |

### Model Training Integration

Primitives are used in three ways:

1. **Feature Engineering**: Primitives compute input features for neural network
2. **Auxiliary Targets**: Gate outputs (ALLOW/BLOCK) become auxiliary training targets
3. **Fuzzy Constraints**: Membership functions constrain neural outputs via differentiable fuzzy layers

### Rule Composition

Rules are composed by combining primitives:

```yaml
rule_a1:
  name: "Dynamic BB Breakout with Volume"
  signals:
    - S001  # BB Breakout
    - S008  # Volume Spike
  gates:
    - G003  # IV Confidence
    - G004  # MTF Consensus
  entry_logic: "AND(signals) AND AND(gates)"
```

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| **Total Primitives** | 52 |
| Core Computational | 12 |
| Signal Generation | 15 |
| Gate Primitives | 10 |
| Membership Functions | 11 |
| Composite Primitives | 4 |
| **Rules Covered** | 13 |
| **Feature Columns Mapped** | 45+ |

---

## Next Steps

1. **Part C**: Build Rule DSL (YAML/JSON) for AntiGrav ingestion
2. **Part D**: Map primitives to CondorBrain feature pipeline and training targets
3. Implement primitives in `intelligence/primitives/` module
4. Generate test cases for each primitive
5. Build primitive composition validator

---

**End of Canonical Rule Primitive Library v2.5**
