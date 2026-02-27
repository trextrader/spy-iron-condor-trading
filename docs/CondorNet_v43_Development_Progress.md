# CondorNet™ v4.3 — Development Progress & Engineering Log

**Project**: SPY Iron Condor Algorithmic Trading System
**Module**: CondorNet v4.3 Intelligence Pipeline
**Status**: 99% Complete — Awaiting Final Retrain with Enriched Dataset
**Date**: 2026-02-27

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Data Pipeline (v4.3)](#3-data-pipeline-v43)
4. [Strategy Template Catalog](#4-strategy-template-catalog)
5. [Options Features Module](#5-options-features-module)
6. [Training Run History](#6-training-run-history)
7. [Interpretability Audit System](#7-interpretability-audit-system)
8. [Bug Fix History](#8-bug-fix-history)
9. [Remaining Work](#9-remaining-work)

---

## 1. Project Overview

CondorNet v4.3 is the neural backbone for the SPY Iron Condor trading system. It replaces the earlier
heuristic-only labeling approach with a full **multi-strategy classification pipeline** that:

- Labels every M5 bar with the optimal options strategy from a 58-template catalog
- Integrates real market options data (2.32M-row CSV) for IC economics rather than BSM proxies
- Trains a multi-task neural network (ETD-1 × TFT × Neural CDE fusion) on the enriched dataset
- Produces a checkpoint interpretable via the audit script (predicates, relational logic, A/B matrices)

The system uses a **dual-data engine**:
1. **Strategy clock**: SPY M1/M5/M15/H1 bars — drives entry/exit timing
2. **Options chain**: `data/Datasetv4/v43/options_2025_v43.csv` — 2.32M rows, real IC economics

---

## 2. Architecture

```
CondorNetV43
├── MultiTFProjector        [M1+M5+M15+H1 → d_joint=256]
│   ├── proj_m1  (Sequential: Linear→LayerNorm→GELU)
│   ├── proj_m5
│   ├── proj_m15
│   └── proj_h1
├── PivotProjector          [13 pivot features → 16]
├── TFFusionBlock           [256+16 → 256]
├── OptionsChainEncoder     [N_contracts × 10 → d_chain=128]
├── JointFusionLayer        [256+128 → d_joint=384]
├── condor_core (CondorNet) [v42 backbone]
│   ├── CanonicalPredicateGates  [8+ soft predicates]
│   ├── SuperSets (16×)
│   │   └── PredicateSets (32×)
│   │       └── RelationalLogicLayer(n_predicates=64, 1)
│   ├── A_theta (BlockMatrixA)   [ETD-1 state dynamics, 192×192]
│   ├── B_theta (BlockMatrixB)   [Control injection]
│   │   └── B_h, B_v, B_m, B_r sub-linears
│   └── G_theta (CDEResponseG)
├── StrategyHead            [384 → 10 strategy classes]
├── RiskMetricHead          [384 → pop/ev/max_loss/var/cvar]
├── PivotPredictionHead     [384 → 5 horizons × (high+low)]
└── PositionSizeHead        [384 → size multiplier]
```

**Key dimensions** (v4.3 run 13–15 config):
- `d_tf_in = 52` (base TF), `d_tf_proj = 64`, `d_joint = 256+128 = 384`
- `d_h=128, d_v=16, d_m=32, d_r=16` → `d_x = 192`
- `n_predicates = 64`, `n_sets = 32`, `n_super_sets = 16`
- A_matrix shape: `[192, 192]`
- B_matrix shape: `[192, d_control=64]`

**Feature vector** (64 total):
| Index | Source | Names |
|-------|--------|-------|
| 0–51  | TF_FEATURE_NAMES | open, high, low, close, MA, …, Slope, SlopeATR |
| 52–56 | FRICTION_FEATURE_NAMES | friction_ok_5/10/20/40/60 |
| 57–58 | TOD_FEATURE_NAMES | tod_sin, tod_cos |
| 59    | REGIME_PERSISTENCE_FEATURE | regime_persistence |
| 60–63 | IVR_REVERSAL_FEATURE_NAMES[:4] | price_stretch, ivr_zone, stretch_zone, reversal_score |

---

## 3. Data Pipeline (v4.3)

**File**: `intelligence/data_pipeline_v43.py`

### 3.1 Core Function: `compute_multitask_labels()`

Assigns one of 10 strategy classes to every M5 bar:

```
0  single_call      — Bullish directional, low IV
1  single_put       — Bearish directional, low IV
2  bull_call_spread — Bullish debit spread
3  bear_put_spread  — Bearish debit spread
4  straddle         — Long ATM vol, no directional bias
5  strangle         — Long OTM vol, no directional bias
6  butterfly_call   — Low IV, pinning expected
7  iron_condor      — Neutral credit, IVR neutral zone
8  custom_multi_leg — All other multi-leg strategies
9  abstain          — No eligible template (low score)
```

### 3.2 Scoring Formula

For each eligible template at each bar:

```
score = 0.40 × pop_k  +  0.25 × ev_k  +  0.20 × dte_affinity  +  0.15 × family_bonus
```

where `pop_k = clip(pop × pop_scale + pop_offset, 0.01, 0.99)`.

Winner = template with highest score. Ties broken by `np.argmax` (first occurrence).

If `best_score < PIPELINE_SCORE_CUTOFF (0.15)` → class 9 (abstain).

### 3.3 Real Options Economics (Step 4b)

When `--options-path` is supplied, real IC economics from the options CSV **override** BSM proxies:

```python
pop_arr[real_mask]  = pop_real          # 1 - sc_delta - |sp_delta|
ml_arr[real_mask]   = ml_real / close   # max_loss / close
ev_arr[real_mask]   = (pop × credit - (1-pop) × max_loss) / close
```

Run 15 result: **18,494/18,494 bars (100%)** covered with real economics.
- ic_pop_real mean: **0.7555**
- credit mean: **$0.069**
- max_loss mean: **$1.624**

### 3.4 Observed Distribution (Run 15 Enriched Dataset)

| Class | Strategy | Count | % |
|-------|----------|-------|---|
| 0 | single_call | 11 | 0.1% |
| 1 | single_put | 5 | 0.0% |
| 2 | bull_call_spread | 5,894 | 31.9% |
| 3 | bear_put_spread | 1,606 | 8.7% |
| 4 | straddle | ~active | — |
| 5 | strangle | ~active | — |
| 6 | butterfly_call | ~active | — |
| 7 | iron_condor | 2,301 | 12.4% |
| 8 | custom_multi_leg | 8,677 | 46.9% |
| 9 | abstain | 0 | 0% |

---

## 4. Strategy Template Catalog

**File**: `intelligence/strategy_templates_v43.py`

58 templates across 4 families, each with:
- `template_id` — unique name
- `v43_class` — maps to one of the 10 STRATEGY_TYPES
- `family` — ShortVolIncome / LongVolConvex / DirectionalDefinedRisk / TermStructure
- `legs` — tuple of LegTemplate (opt_type, side, qty, strike_rank, expiry_mode)
- `pop_scale`, `pop_offset`, `ev_scale`, `max_loss_scale` — economics multipliers
- `pred` — vectorized predicate lambda over `compute_predicate_atoms(df)`

### 4.1 Template Counts by Family

| Family | Count | Examples |
|--------|-------|---------|
| ShortVolIncome (SVI) | 19 | iron_condor, iron_butterfly, short_straddle, short_strangle, jade_lizard |
| LongVolConvex (LVC) | 13 | straddle_long, strangle_long, inverse_iron_butterfly, guts, strip, strap |
| DirectionalDefinedRisk (DDR) | 23 | long_call, bear_put_spread, bull_call_butterfly, collar, diagonal_call |
| TermStructure (TS) | 3 | calendar_call, calendar_put, double_diagonal |

### 4.2 IV-Regime Zone Design (Exclusive)

Templates are designed with mutually exclusive IV-regime zones to prevent score crowding:

| IV Zone | Primary Strategies |
|---------|--------------------|
| `ivr_high + consol_vhigh` | short_straddle, iron_butterfly, short_guts, covered_short_straddle |
| `ivr_high + consol_high` | short_strangle, covered_short_strangle |
| `ivr_neutral + range-bound` | **iron_condor** (exclusive owner) |
| `ivr_low + consol_vhigh` | long_call_butterfly (pop_scale=0.72 to beat long_call_condor) |
| `bw_expanding + no_direction` | straddle_long, strangle_long |

### 4.3 DTE_AFFINITY Matrix (schema_v43.py)

Recalibrated for 0-DTE SPY use. Key change: `iron_condor "0-2"` raised from 0.2 → **0.8** to
match the 0-DTE primary use case.

---

## 5. Options Features Module

**File**: `intelligence/options_features_v43.py`

New module added to compute real IC economics from the options CSV.

### 5.1 Function: `build_options_daily_summary()`

**Input**: `data/Datasetv4/v43/options_2025_v43.csv` (2.32M rows, 21 columns)

**Processing**:
1. Parse `timestamp` + `expiration` (explicit `pd.to_datetime()` — pandas 2.x fix)
2. Force numeric dtypes for all 15 numeric columns (mixed str/int CSV fix)
3. Compute DTE vectorially: `(expiration - timestamp).dt.days` (not `.apply(lambda)`)
4. Filter near-DTE rows (0–7 days)
5. For each trading date: find 4 IC legs at target deltas (0.175 short, 0.10 long)
6. Compute IC economics: credit, max_loss, pop (delta-based), EV

**Output**: `pd.DataFrame` indexed by `"YYYY-MM-DD"` with columns:
```
atm_iv, ic_pop_real, ic_credit_raw, ic_max_loss_raw, ic_ev_raw,
sc_strike, sp_strike, lc_strike, lp_strike, expiry_used, dte_used
```

**Run 15 stats**:
- 2.32M rows loaded, 238 dates OK, 0 skipped
- DTE distribution: 0-DTE dominant (0-DTE expirations preferred)

---

## 6. Training Run History

All runs use `condor_train_net_v43.py` on Lightning AI (T4 GPU).

| Run | Epochs | Best Epoch | Best Val Loss | Train Loss | Notes |
|-----|--------|------------|---------------|------------|-------|
| Run 2  | 11 | 6  | 1.55967 | 1.57515 | First v4.3 attempts |
| Run 3  | 11 | 2  | 2.15718 | 2.31981 | Gate saturation issues |
| Run 4  | 11 | 2  | 2.09898 | 2.20477 | Continued saturation |
| Run 13 | 31 | 5  | **0.98954** | 0.92278 | Post gate-fix; best single-session result |
| Run 15 | 41 | 27 | 1.55983 | 1.21553 | Run with enriched dataset (TF projector bug present) |

**Run 13** (best) checkpoint config: `n_predicates=64, n_sets=32, n_super_sets=16, d_h=128`

### 6.1 Gate Saturation Fix History (Runs 2–13)

| Commit | Fix |
|--------|-----|
| `27d46b8` | Fix lambda zero-variance — missing kwargs feature slice, variance initialization |
| `6b34eee` | Prevent hierarchical λ saturation — steepness 20→5, logit centering, per-epoch diagnostics |
| `12327a9` | Harden λ saturation — bounded steepness, std-norm, 3-mode warnings |
| `9887cf5` | Gate attr path fix, disable intra-epoch truncation, mean-only gate centering |

---

## 7. Interpretability Audit System

**File**: `intelligence/audit_condornet_interpretability_v43.py`

### 7.1 Output JSON Structure

Each epoch generates `Epoch{N}_Audit_Interpretation_output.json` with these top-level keys:

```json
{
  "predicates":       { 8 canonical threshold values },
  "super_set":        { 16 super-sets × 32 sets × top-5 feature comparisons },
  "strategy_head":    { output_class_norms, ranked_classes, dominant_class },
  "risk_head":        { shared_input_norm, pop/ev/max_loss/var/cvar head norms },
  "pivot_head":       { high/low horizon weights for h5/h10/h20/h35/h70 },
  "fuzzy_gates":      { tf_projector_frobenius (m1/m5/m15/h1), tf_ranked },
  "a_matrix":         { shape [192,192], spectral_radius, top-5 eigenvalues, frob_norm },
  "b_matrix":         { shape [192,64], frobenius_norm, column_norms_max/mean },
  "rules":            [ human-readable trading rules transcript ],
  "checkpoint_meta":  { epoch, val_loss, train_loss, config }
}
```

### 7.2 Feature Name Mapping (64 features)

The `_FULL_FEATURE_NAMES` list used for `pair_idx → (feat_a, feat_b)` lookup:

```
[0–51]  TF_FEATURE_NAMES (52 base features, e.g. rsi_dyn, bw_expansion_rate, etc.)
[52]    friction_ok_5
[53]    friction_ok_10
[54]    friction_ok_20
[55]    friction_ok_40
[56]    friction_ok_60
[57]    tod_sin
[58]    tod_cos
[59]    regime_persistence
[60]    price_stretch
[61]    ivr_zone
[62]    stretch_zone
[63]    reversal_score
```

With `n_predicates=64` the RelationalLogicLayer has **2016 pairs** (C(64,2)). All resolve to
human-readable names when `_FULL_FEATURE_NAMES` (64 entries) is used.

---

## 8. Bug Fix History

### 8.1 Data Pipeline Bugs

| Bug | Root Cause | Fix | Commit |
|-----|-----------|-----|--------|
| 100% abstain labels | `ABSTAIN_LABEL_SCORE_CUTOFF=0.40` unreachable with normalized EV ≈ 1e-4 | Added `PIPELINE_SCORE_CUTOFF=0.15` | `f87022c` |
| Iron condor = 0% (DTE affinity) | `DTE_AFFINITY["iron_condor"]["0-2"]=0.2` vs custom_multi_leg=0.5 → 0.030 gap | Recalibrated IC to 0.8 | `7461c10` |
| Iron condor = 0% (predicate) | `consol_high AND adx_weak` rarely both true in volatile 2025 SPY | Changed to `consol_high OR adx_weak` | `85872d3` |
| Straddle/strangle/butterfly = 0% | IC predicate included `ivr_high` so IC won over short_straddle on all `ivr_high` bars | IC restricted to `ivr_neutral` only | `2039ce6` |
| Butterfly = 0% | `long_call_butterfly` pop_scale=0.60 tied with `long_call_condor` 0.60 → argmax gave condor | Butterfly pop_scale 0.60→0.72 | `2039ce6` |

### 8.2 Options Features Bugs (3 iterations)

| Bug | Root Cause | Fix | Commit |
|-----|-----------|-----|--------|
| `Can only use .dt accessor with datetimelike values` | `pd.read_csv(parse_dates=[...])` fails in pandas 2.x | Explicit `pd.to_datetime(col, errors='coerce')` | `7461c10` |
| `'>' not supported between str and int` (iter 1) | `.apply(lambda x: x.days)` on object-dtype → mixed int/None types | `(expiration - timestamp).dt.days` vectorized | `85872d3` |
| `'>' not supported between str and int` (iter 2) | CSV mixes "400.0" (float-string) and "915" (int-string) → object dtype on strike/delta | `pd.to_numeric(errors='coerce')` for 15 numeric columns | `cd5ffb3` |

### 8.3 Interpretability Audit Bugs (Fixed 2026-02-27)

| Bug | Root Cause | Fix | Commit |
|-----|-----------|-----|--------|
| `feat_52`–`feat_63` unresolved in top_comparisons | Audit JSON generated with old script using only TF_FEATURE_NAMES (52 entries); current code already has 64-entry `_FULL_FEATURE_NAMES` | Re-run audit script on checkpoint to regenerate | `909e98a` |
| `b_matrix: {}` (empty) | `export_matrices_csv` checked `hasattr(b_theta, 'weight')` — False for `BlockMatrixB` (has B_h/B_v/B_m/B_r sub-linears, not `.weight`) | Changed to call `b_theta.full_matrix()` which concatenates sub-matrix weights → shape `[d_x, d_control]` | `909e98a` |
| `fuzzy_gates: {}` (empty) | Both scripts looked for `m1_proj`, `m5_proj`, etc. on `MultiTFProjector` but actual attributes are `proj_m1`, `proj_m5`, `proj_m15`, `proj_h1` | Changed `f'{tf_name}_proj'` → `f'proj_{tf_name}'` in both scripts | `909e98a` |

---

## 9. Remaining Work

### 9.1 Immediate — Run Next Training

The dataset is now fully enriched with real options economics. The next training run should produce
correct fuzzy_gates (TF projector Frobenius contributions) and b_matrix metrics in every epoch JSON.

**Recommended training invocation** (Lightning AI):
```bash
python intelligence/condor_train_net_v43.py \
  --data-path data/Datasetv4/v43/spy_m5_v43.csv \
  --options-path data/Datasetv4/v43/options_2025_v43.csv \
  --epochs 40 \
  --n-predicates 64 \
  --n-sets 32 \
  --n-super-sets 16 \
  --report-dir reports/v43TrainRunNext
```

### 9.2 Regenerate Epoch 27 Audit JSON

With the fixed audit script, re-run on run 15 epoch 27 checkpoint:
```bash
python intelligence/audit_condornet_interpretability_v43.py \
  --model models/condornet_v43_best.pth \
  --output-json reports/v43TrainRun15/Epoch27_Audit_Fixed.json \
  --verbose
```

Expected: feat_54=`friction_ok_20`, feat_62=`stretch_zone`, feat_63=`reversal_score` now resolved.
Expected: `fuzzy_gates.tf_projector_frobenius` populated with m1/m5/m15/h1 Frobenius norms.
Expected: `b_matrix` populated with shape `[192, 64]`, frobenius_norm, column_norms.

### 9.3 Final Deployment Checklist

- [ ] Retrain with enriched M5 + real options dataset
- [ ] Verify all 10 strategy classes appear in training distribution
- [ ] Confirm `fuzzy_gates`, `b_matrix` sections populated in new run JSONs
- [ ] Review run N epoch comparison JSON for val_loss < 1.0 sustained
- [ ] Connect trained model to live paper trading engine

---

## Appendix A — Key File Locations

| File | Purpose |
|------|---------|
| `intelligence/data_pipeline_v43.py` | ETL + multi-strategy labeling (Step 4b: real options override) |
| `intelligence/options_features_v43.py` | Options CSV → per-date IC economics |
| `intelligence/strategy_templates_v43.py` | 58-template catalog (predicates, economics, leg structures) |
| `intelligence/schema_v43.py` | STRATEGY_TYPES, DTE_AFFINITY, _FULL_FEATURE_NAMES constants |
| `intelligence/condor_brain_net_v43.py` | CondorNetV43 + MultiTFProjector + BlockMatrixB architecture |
| `intelligence/condor_train_net_v43.py` | Training loop + per-epoch interpretability extraction |
| `intelligence/audit_condornet_interpretability_v43.py` | Standalone audit/interpretability tool |
| `data/Datasetv4/v43/options_2025_v43.csv` | 2.32M-row real options data (daily snapshots) |
| `reports/v43TrainRun13/` | Best checkpoint (val_loss=0.98954, epoch 5) |
| `reports/v43TrainRun15/` | Most recent run (41 epochs, best epoch 27) |

## Appendix B — Commit Map

```
909e98a  fix(audit+train): resolve feat_52-63, B_matrix empty, fuzzy_gates empty
f5b4d26  fix(audit): pass full config through **kwargs
e6806b0  fix(audit): unpack config dict as **kwargs to build_condornet_v43
3a96851  fix(strangle): boost pop_offset for strangle_long scoring
a123f06  fix(ivr): percentile-based IVR zones + boost straddle/strangle scoring
5ac422a  fix(ivr+scoring): lower IVR thresholds + boost long-vol pop_scale for classes 4/5/6
57d47c5  fix(templates): loosen predicates for classes 4/5/6
2039ce6  fix(templates): exclusive IV-regime zones — straddle/strangle/butterfly fixed
cd5ffb3  fix(options): force numeric dtype for 15 columns
85872d3  fix(options+ic): vectorised DTE calc + loosen IC predicate (AND→OR)
7461c10  fix(options+schema): datetime parse fix + DTE_AFFINITY recalibration
1ddf129  feat(pipeline): real options economics via options_features_v43
f87022c  fix(pipeline): PIPELINE_SCORE_CUTOFF=0.15
f7018d8  feat(pipeline): full 58-template multi-strategy labeling catalog
```
