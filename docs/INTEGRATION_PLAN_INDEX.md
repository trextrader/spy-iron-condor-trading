# Integration Plan Index: HJB Framework & Robust Pricing-Hedging Duality

**Quick Reference for CondorBrain V2.2 Integration**

---

## Document Map

| Document | Content | Location |
|----------|---------|----------|
| Part 1 | Phases 1-4, Current Architecture | `INTEGRATION_PLAN_HJB_ROBUST_PRICING.md` |
| Part 2 | Phases 5-8, Testing, Rollback | `INTEGRATION_PLAN_HJB_ROBUST_PRICING_PART2.md` |
| Source Papers | Research foundations | `/docs/*.pdf` |

---

## 8-Phase Integration Summary

| Phase | Name | Key Deliverables | Risk Level |
|-------|------|------------------|------------|
| **1** | Prediction Sets | `prediction_set.py`, `martingale_measure.py` | LOW |
| **2** | Enhanced Loss | `EnhancedCompositeCondorLoss` (8 components) | LOW |
| **3** | HJB Solver | `value_function.py`, `nt_zones.py`, 5-region classifier | MEDIUM |
| **4** | Alpha Predictors | `ou_process.py`, α-features (32→35) | LOW |
| **5** | Execution Costs | `cost_model.py`, `fill_probability.py`, order router | MEDIUM |
| **6** | Fuzzy Measure | `FuzzyMeasure` class, prediction set integration | LOW |
| **7** | Physics Features | κ_t, E_t, Π_t (35→38 features) | LOW |
| **8** | Governance Layer | 5-level risk cascade, unified decision flow | HIGH |

---

## Mathematical Foundations Integration Map

### From Robust Pricing-Hedging Duality (Hou-Obłój)

| Concept | Mathematical Form | Integration Point |
|---------|------------------|-------------------|
| Prediction Set | I ⊂ Ω | `prediction_set.py` → Phase 1 |
| Martingale Measure | M^I = {P : P(I)=1} | `martingale_measure.py` → Phase 1 |
| Superhedging Bound | V_{X,P,I}(G) | `EnhancedCompositeCondorLoss.λ₆` → Phase 2 |
| Pricing-Hedging Duality | V = sup E_P[G] | Calibration validation → Phase 1 |
| Fuzzy as Measure | μ_fuzzy ∈ [0,1] | `FuzzyMeasure` class → Phase 6 |

### From Passerini-Vázquez HJB Framework

| Concept | Mathematical Form | Integration Point |
|---------|------------------|-------------------|
| Value Function | V(t,x,q) | `value_function.py` → Phase 3 |
| NT Zone | b±(t,x) = q* ± (g-C)/(2T-t) | `nt_zones.py` → Phase 3 |
| 5 Trading Regions | Market/Limit/Hold | `trading_regions.py` → Phase 3 |
| OU Alpha | dx = -κx·dt + η·dZ | `ou_process.py` → Phase 4 |
| Integrated Gain | g(t,x) = ∫x_s ds | `HJBValueFunction.integrated_gain()` → Phase 3 |
| Fill Probability | P± | `fill_probability.py` → Phase 5 |

---

## Critical Integration Hooks

### CondorBrain Model (`condor_brain.py`)

```python
# Line 431: After last_hidden extraction
last_hidden = x[:, -1, :]  # (B, d_model)

# NEW: Add prediction set membership output
ps_membership = self.ps_head(last_hidden)  # (B, 1)

# NEW: Add HJB state features
hjb_features = self.hjb_head(last_hidden)  # (B, 3) [g, dV/dq, nt_prob]
```

### Loss Function (`condor_loss.py`)

```python
# Current: 5 components
# L = λ₁·L_pred + λ₂·L_sharpe + λ₃·L_dd + λ₄·L_turn + λ₅·L_rule

# NEW: 8 components
# L = ... + λ₆·L_superhedge + λ₇·L_martingale + λ₈·L_prediction_set
```

### QTMF Facade (`qtmf/facade.py`)

```python
# Line 126: After confidence gate
if ti.gaussian_confidence < min_gaussian_conf:
    return SizingPlan(approved=False, ...)

# NEW: Prediction set gate
ps_membership = prediction_set.membership(market_state)
if ps_membership < 0.1:
    return SizingPlan(approved=False, reason='outside_prediction_set', ...)

# NEW: HJB no-trade zone gate
if nt_computer.classify_position(t, x, q) == 'nt_zone':
    return SizingPlan(approved=False, reason='hjb_no_trade_zone', ...)
```

### Rule Engine (`rule_engine/executor.py`)

```python
# Gate stack: Add G011 (HJB NT Zone Gate)
GATE_REGISTRY = {
    ...
    'G011': compute_nt_zone_gate,  # NEW
}
```

---

## Feature Registry Evolution

| Version | Feature Count | New Features |
|---------|---------------|--------------|
| V2.1 | 32 | Base dynamic + primitive |
| V2.2-alpha | 35 | +ou_state, +integrated_gain, +time_to_close |
| V2.2 | 38 | +kappa_proxy, +vol_energy, +homology_sig |

---

## Testing Quickstart

```bash
# Run all unit tests for new modules
python -m pytest tests/unit/test_prediction_set.py tests/unit/test_hjb_*.py -v

# Verify baseline preservation
python -m pytest tests/regression/test_baseline_parity.py -v

# Full validation suite
python -m pytest tests/ -v --tb=short

# Backtest with governance (final validation)
python core/main.py --mode backtest --use-governance --bt-samples 0
```

---

## Rollback Quick Reference

```bash
# Disable all new features (immediate)
# Edit core/config.py:
USE_PREDICTION_SETS = False
USE_ENHANCED_LOSS = False
USE_HJB_CONTROL = False
...

# Full code rollback
git checkout pre-integration-backup
```

---

## Key Invariants (DO NOT BREAK)

1. **CondorExpertHead activations** (indices 0-7 constrained to valid ranges)
2. **CompositeCondorLoss clamping** (rule_signals in [0,1])
3. **Minimum 2 contracts** for Iron Condor (`assert total_qty >= 2`)
4. **MTFSyncEngine indicator order** (RSI, ADX, BBands, etc.)
5. **Robust Z-score normalization** (median + 1.4826×MAD)

---

## Next Steps After Plan Approval

1. **Create branch**: `git checkout -b feature/hjb-robust-integration`
2. **Implement Phase 1**: `intelligence/robust/` module
3. **Run Phase 1 tests**: `pytest tests/unit/test_prediction_set.py`
4. **Verify no regression**: `pytest tests/regression/`
5. **Proceed to Phase 2** if all tests pass

---

## Contact Points

- **Loss Function Changes**: Modify `condor_loss.py`, test with `test_enhanced_loss.py`
- **Model Architecture**: Modify `condor_brain.py`, test with `test_condor_brain_v22.py`
- **Sizing Logic**: Modify `qtmf/facade.py`, test with `test_qtmf_v2.py`
- **Rule Engine**: Modify `rule_engine/executor.py`, test with `test_governance_layer.py`

---

## Addendum (2026-01-24): Master Spec + Audit Alignment

This index now has a master engineering spec that unifies all integration details, derivations, interface tables, and audit findings:

- `INTEGRATION_PLAN_MASTER.md` (primary engineering spec)
- `INTERFACE_CATALOG.md` (autogenerated interface signatures)

Key alignment requirements (must be enforced in code):
1. Feature schema must be selected by **column name** (V2.2) and **ordered explicitly**.
2. Dataset column order varies between 2024 and 2025 institutional CSVs; order-based selection is a bug.
3. Model config metadata (layers/heads/input_dim) must be consistent across docs, checkpoints, and inference engines.

Use this index to navigate. The master spec is now the single source of truth for integration flow and validation.


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
