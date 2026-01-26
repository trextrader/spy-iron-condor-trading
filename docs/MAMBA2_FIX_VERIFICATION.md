# Mamba2 Model Collapse Fix - Verification Report

**Date:** 2026-01-26
**Purpose:** Document verification of all fixes to address model collapse issues

## 1. Feature Verification

### Data File: `data/processed/mamba_institutional_2024_1m_last 1mil.csv`
- **Total Rows:** 1,000,000
- **Total Columns:** 65

### V22 Feature Schema (52 features) - ALL PRESENT
| Category | Count | Features |
|----------|-------|----------|
| Market Primitives | 5 | open, high, low, close, volume |
| Greeks/Options | 9 | delta, gamma, vega, theta, iv, ivr, spread_ratio, te, strike |
| Targets/Risk | 2 | target_spot, max_dd_60m |
| Dynamic Regime | 16 | log_return, vol_ewma, ret_z, atr_pct, kappa_proxy, vol_energy, rsi_dyn, adx_adaptive, psar_adaptive, bb_mu_dyn, bb_sigma_dyn, bb_lower_dyn, bb_upper_dyn, stoch_k_dyn, consolidation_score, breakout_score |
| V2.2 Primitives | 20 | bb_percentile, bw_expansion_rate, volume_ratio, **friction_ratio**, **exec_allow**, gap_risk_score, risk_override, iv_confidence, mtf_consensus, macd_norm, macd_signal_norm, macd_histogram, plus_di, minus_di, psar_trend, psar_reversion_mu, beta1_norm_stub, chaos_membership, position_size_mult, fuzzy_reversion_11 |

**Status:** ALL 52 features confirmed present in data file.

---

## 2. Rule Verification

### Complete Ruleset: `docs/Complete_Ruleset_DSL.yaml`
- **Total Rules:** 14
- **Categories:**
  - Trend Following: A1, A2, A3
  - Mean Reversion: B1, B2, PSAR Reversion
  - Breakout: C1, C2
  - Momentum: D1, D2
  - Volatility: E2, E3
  - Execution Filter: E1, **RULE_SPREAD (Rule #14)**

### Friction Gate (Rule #14 - RULE_SPREAD)
**Location:** `docs/Complete_Ruleset_DSL.yaml:496-583`

**Verified Components:**
- `friction_ratio` = Spread / AvgRange
- `exec_allow` = Friction gate output (1 = allow, 0 = block)
- `gap_risk_score` = Gap risk [0,1]
- `risk_override` = Gap risk override flag

**Entry Logic:** `friction_ratio < dynamic_threshold.theta`
**Exit Logic:** `OR(friction_ratio >= theta, gap_risk >= 0.8)`

**Status:** Friction gate fully integrated into entry/exit target generation.

---

## 3. Model Output Verification

### CondorExpertHead (10 outputs)
| Index | Output | Activation | Range |
|-------|--------|-----------|-------|
| 0 | short_call_offset | Sigmoid * 5 | 0-5% |
| 1 | short_put_offset | Sigmoid * 5 | 0-5% |
| 2 | wing_width | Sigmoid * 10 | 0-$10 |
| 3 | dte_selection | Sigmoid * 43 + 2 | 2-45 days |
| 4 | prob_profit | Sigmoid | 0-1 |
| 5 | expected_roi | Tanh * 0.5 | -50% to +50% |
| 6 | max_loss_pct | Sigmoid | 0-1 |
| 7 | confidence | Sigmoid | 0-1 |
| 8 | **entry_logit** (NEW) | Raw | -inf to +inf |
| 9 | **exit_logit** (NEW) | Raw | -inf to +inf |

---

## 4. Separate Conditions Learning

### Entry Conditions (index 8)
- Supervised with explicit entry_target
- Incorporates: RSI neutral, IVR > 30, ADX < 30
- **Friction Gate:** exec_allow > 0.5, friction_ratio < 1.0, gap_risk < 0.8

### Exit Conditions (index 9)
- Supervised with explicit exit_target
- Incorporates: Extreme RSI, high volatility, strong trend
- **Friction Gate:** gap_risk >= 0.8 forces exit, high friction suggests exit

### Sizing Conditions (index 5: expected_roi)
- Supervised with realized_roi target
- Fuzzy logic via position_size_mult and fuzzy_reversion_11 features
- Separate decision tree learned in export_learned_conditions.py

---

## 5. Diffusion Head Verification

### ConditionalDiffusionHead
- **Input Dim:** 4 features (r, rho, d, v)
- **Condition Dim:** d_model (from Mamba backbone)
- **Horizon:** 32 steps
- **Diffusion Steps:** Configurable (default 50)

### Forward Predictors When Diffusion Enabled:
1. **Policy Head (10):** call/put offsets, width, dte, pop, roi, max_loss, conf, entry, exit
2. **Regime Detector (3):** low/normal/high volatility probabilities
3. **HorizonForecaster (6):**
   - daily_forecast: (B, num_days, 4) for [close, high, low, vol]
   - max_range: (B, 2) for [max_high_pct, max_low_pct]
4. **Diffusion Head (4):** trajectory refinement for (r, rho, d, v)

---

## 6. Anti-Collapse Fixes Implemented

### Fix 1: Explicit ENTRY/EXIT Heads
- **File:** `intelligence/condor_brain.py:227-276`
- Outputs 8→10: Added entry_logit (idx 8), exit_logit (idx 9)
- No longer derived from confidence scalar

### Fix 2: Realistic Training Targets
- **File:** `intelligence/train_condor_brain.py:136-248`
- Targets vary based on IVR, RSI, ADX (not constants)
- Entry/exit incorporate friction gate signals

### Fix 3: Anti-Collapse Loss
- **File:** `intelligence/condor_brain.py:638-760`
- Entropy regularization for probability heads
- Variance penalty if batch variance too low

### Fix 4: Feature-Group Dropout
- **File:** `intelligence/condor_brain.py:73-136`
- Drops entire feature groups during training
- Groups: market_primitives, greeks, targets_risk, dynamic_regime, v22_primitives

### Fix 5: Variance Monitoring
- **File:** `intelligence/training_monitor.py:630-740`
- Per-head variance tracking
- Collapse warning if variance < threshold

### Fix 6: Temporal Attribution
- **File:** `audit/export_learned_conditions.py:379-435`
- 7 temporal aggregations: last, mean, std, min, max, last5, slope
- Better captures sequence model learning

---

## 7. Training Configuration (Lightning AI T4)

### Script: `scripts/lightning_train_4runs.sh`

| Run | Diffusion | Config |
|-----|-----------|--------|
| 1 | OFF | d_model=512, layers=16, epochs=50 |
| 2 | OFF | d_model=512, layers=16, epochs=50 |
| 3 | ON | d_model=512, layers=16, epochs=50, diffusion_steps=50 |
| 4 | ON | d_model=512, layers=16, epochs=50, diffusion_steps=50 |

**Common Settings:**
- Max Rows: 100,000
- Batch Size: 128 (T4 16GB VRAM)
- Feature Group Dropout: 0.15
- Composite Loss: Enabled
- VolGatedAttn: Enabled
- TopKMoE: Enabled
- Early Stopping: Patience 10

---

## 8. Output Files

### Training Outputs:
- `models/condor_run{1-4}_*.pth` - Trained models
- `runs/run{1-4}_*/` - TensorBoard logs
- `logs/run{1-4}_*.log` - Training logs

### Learned Conditions (from export_learned_conditions.py):
- `artifacts/learned_conditions/entry_rules.md` - Decision tree for ENTRY
- `artifacts/learned_conditions/exit_rules.md` - Decision tree for EXIT
- `artifacts/learned_conditions/sizing_rules.md` - Decision tree for SIZING
- `artifacts/learned_conditions/attribution_*.csv` - Feature importances

---

## Summary

All requested verifications complete:
1. All 52 V22 features present in data file
2. All 14 rules loaded from Complete_Ruleset_DSL.yaml
3. Friction gate (Rule #14) integrated into entry/exit targets
4. Diffusion head configured with forward predictors
5. Separate conditions learned for ENTRY, EXIT, and SIZING
6. Model uses all inputs (52 features) via select_feature_frame()
7. Decision trees/inequalities learned from temporal feature aggregations
