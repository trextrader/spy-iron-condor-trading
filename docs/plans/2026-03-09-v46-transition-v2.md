# CondorNet v45 → v46 Institutional Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transition CondorNet from v45 (2025-only, Iron Condor) to v46 (2020–2024 training across 63 discrete input datasets, 58-strategy optimization, causal-clean features, live Alpaca paper trading) with zero data leakage, full reproducibility, and ablation-traceable model improvement.

**Architecture:** Governance-first. Schema contract and causality policy are frozen before any feature is generated. All 58 strategy output datasets are bar-level, lagged, and masked. The v46 model fuses base features + strategy signals + optional position state through a StrategyOutputEncoder with explicit mask support. 2025 is permanent holdout. Live inference stack is validated offline via historical replay before any Alpaca paper order is placed.

**Tech Stack:** Python 3.12, PyTorch, pandas, numpy, scipy, `kaggle/condor_brain_backtest_v45.py` (active backtester), Lightning AI T4, Alpaca API (paper only).

**Execution rule:** Nothing enters code before its governing document exists. No optimization before data contract is signed off. No live trading before offline replay passes.

---

## PHASE 1 — Baseline Freeze and Reproducibility Lock

> Before changing a single line, lock the current "last known good" baseline: file hashes, software versions, random seeds, and a smoke-test reference run. This is the anchor every later comparison is measured against.

### Task 1.1: Record software environment snapshot

**Files:**
- Create: `docs/baseline/v45_environment.txt`

**Step 1: Capture on Lightning AI**
```bash
python3 - <<'EOF'
import sys, torch, pandas, numpy, scipy, yaml
env = {
    "python":  sys.version,
    "torch":   torch.__version__,
    "pandas":  pandas.__version__,
    "numpy":   numpy.__version__,
    "scipy":   scipy.__version__,
    "cuda":    torch.version.cuda,
    "gpu":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
}
for k, v in env.items():
    print(f"{k}: {v}")
EOF
```
Paste output into `docs/baseline/v45_environment.txt`.

**Step 2: Lock random seeds policy**

Add to `docs/baseline/v45_environment.txt`:
```
RANDOM_SEED_POLICY:
  python_seed: 42
  numpy_seed:  42
  torch_seed:  42
  deterministic_cudnn: True
  comment: All experiments must set these at process start before any data load
```

**Step 3: Commit**
```bash
git add docs/baseline/v45_environment.txt
git commit -m "chore(baseline): record v45 software environment snapshot"
```

---

### Task 1.2: Hash all baseline input files

**Files:**
- Create: `docs/baseline/v45_input_manifest.json`

**Step 1: Write hash script**
```python
#!/usr/bin/env python3
"""Hash all 2025 dataset files and key source files."""
import hashlib, json, os
from pathlib import Path

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

manifest = {}

# Dataset files
for fn in sorted(Path("data/Datasetv4/v43/2025").iterdir()):
    if fn.suffix == '.csv':
        manifest[str(fn)] = sha256(str(fn))

# Key source files
for fn in [
    "kaggle/condor_brain_backtest_v45.py",
    "intelligence/condor_brain_net_v43.py",
    "intelligence/schema_v43.py",
    "intelligence/data_pipeline_v43.py",
]:
    if os.path.exists(fn):
        manifest[fn] = sha256(fn)

with open("docs/baseline/v45_input_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Hashed {len(manifest)} files")
for k, v in list(manifest.items())[:5]:
    print(f"  {k}: {v[:16]}...")
```

**Step 2: Run**
```bash
python3 docs/baseline/hash_baseline.py
```

**Step 3: Commit**
```bash
git add docs/baseline/hash_baseline.py docs/baseline/v45_input_manifest.json
git commit -m "chore(baseline): hash all v45/2025 input files and source files"
```

---

### Task 1.3: Run and record v45 reference smoke test

**Step 1: Run with fixed seed**
```bash
python3 kaggle/condor_brain_backtest_v45.py \
    --limit 500 \
    --seed 42 \
    2>&1 | tee docs/baseline/v45_smoke_500bar.log
```

**Step 2: Extract and record key metrics**
```bash
grep -E "(total_return|max_drawdown|sharpe|win_rate|trades_opened|entry_gate|pop_gate|ic_gate|atomicity)" \
    docs/baseline/v45_smoke_500bar.log \
    > docs/baseline/v45_smoke_500bar_metrics.txt
cat docs/baseline/v45_smoke_500bar_metrics.txt
```

**Step 3: Commit baseline artifact**
```bash
git add docs/baseline/v45_smoke_500bar.log docs/baseline/v45_smoke_500bar_metrics.txt
git commit -m "chore(baseline): record v45 reference smoke-test output (500 bars, seed=42)"
```

**Invariant:** Any later run with identical code + data + seed must reproduce these metrics. If it does not, investigate before proceeding.

---

## PHASE 2 — v46 Schema Contract, Causality Policy, and Manifest

> Schema governance comes **before feature code**. Every proposed column must be classified before a single line of feature-generation code is written.

### Task 2.1: Write LEAKAGE_POLICY.md

**Files:**
- Create: `docs/v46/LEAKAGE_POLICY.md`

**Step 1: Write**

```markdown
# v46 Leakage Policy

## Temporal Execution Contract (frozen, non-negotiable)

- Features are computed from information available at the **close** of bar t.
- Model decision is emitted at the **end** of bar t.
- Orders are assumed filled at the **open** of bar t+1 (or next available mid snapshot).
- Exit is evaluated at the close of bar t using same-bar close price.
- No feature may use ANY information from bar t+1 onward.

## Feature Classification

Every column in v46 datasets is classified as exactly one of:

| Class | Code | Training input? | Live inference input? | Notes |
|---|---|---|---|---|
| Causal feature | `CF` | YES | YES | Available at bar-t close from bar-t-or-earlier data |
| Lagged strategy signal | `LS` | YES | YES | Must be lagged by ≥1 bar before merge |
| Offline training label | `OL` | As label only | NO | Used only in loss function |
| Offline diagnostic | `OD` | NO | NO | QA/audit use only |
| Forward-looking | `FL` | NO | NO | Strictly prohibited from encoder input |

## Specific Column Rulings

| Column | Class | Reason |
|---|---|---|
| `bars_to_next_pivot` | FL | Explicitly forward-looking. Offline label only or excluded entirely. |
| `UpperCloseBackInsideBandFlag` | CF | Uses prior-bar break + current close. Causal IF decision after bar close. |
| `LowerCloseBackInsideBandFlag` | CF | Same as above. |
| `eq_price`, `gamma_net`, `gamma_flip` | CF | Causal if computed from contemporaneous chain data at bar-t. |
| `pinning_bias` | CF | Derived from contemporaneous chain. |
| `bars_since_band_break` | CF | Backward-looking counter. Causal. |
| `bars_since_psar_flip` | CF | Backward-looking counter. Causal. |
| strategy pnl_pct (same bar) | FL | Realized outcome — must be lagged ≥1 bar. |
| strategy rolling_expectancy_lag1 | LS | Rolling up to t-1. Causal. |
| `exit_signal` (model output) | OL | Training label for exit_bce loss only. |
| `pop`, `ev`, `max_loss` | OL | Strategy simulation labels. |

## Fill Policy

- NaN in CF columns: forward-fill up to 5 bars, then fill with column median computed on train years only.
- NaN in LS columns: fill with 0.0 (neutral signal). Must not forward-fill.
- Mask channel: if LS column was originally NaN (no strategy activity), mask=0.

## Normalization Policy

- All CF and LS columns are normalized using statistics computed on **train years only** (2020–2024).
- Normalization params are saved to `models/norm_stats_v46.json` at ETL time.
- Live inference loads `norm_stats_v46.json` — never recomputes from live stream.
- After normalization: clip to [-5, 5].
- Bounded columns (flags, probabilities): no normalization; clip to [0, 1].
```

**Step 2: Commit**
```bash
git add docs/v46/LEAKAGE_POLICY.md
git commit -m "docs(governance): add v46 leakage policy and temporal execution contract"
```

---

### Task 2.2: Write FEATURE_ROLE_MAP.json

**Files:**
- Create: `docs/v46/FEATURE_ROLE_MAP.json`

**Step 1: Write (all v43 + all proposed v46 columns classified)**

```json
{
  "_doc": "v46 feature role map. Every column must appear here. class codes: CF/LS/OL/OD/FL",
  "_schema_version": "v46.0",
  "timestamp":                    { "class": "OD", "live": false, "train_input": false, "notes": "index key, not a feature" },
  "open":                         { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "ffill5" },
  "high":                         { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "ffill5" },
  "low":                          { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "ffill5" },
  "close":                        { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "ffill5" },
  "volume":                       { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "log_return":                   { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ret_z":                        { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "atr_pct":                      { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "median" },
  "bb_lower_dyn":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "ffill5" },
  "bb_upper_dyn":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "ffill5" },
  "bb_sigma_dyn":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "median" },
  "bandwidth":                    { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "median" },
  "bw_expansion_rate":            { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "bb_percentile":                { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "0.5" },
  "psar_trend":                   { "class": "CF", "live": true,  "train_input": true,  "bounded": [-1,1],"fill": "zero" },
  "psar_mark":                    { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "ffill5" },
  "psar_adaptive":                { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "ffill5" },
  "psar_reversion_mu":            { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "rsi_dyn":                      { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,100],"fill": "50" },
  "stoch_k_dyn":                  { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,100],"fill": "50" },
  "adx_adaptive":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,100],"fill": "25" },
  "PivotHigh":                    { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "PivotLow":                     { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "Slope":                        { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "PivotResidual":                { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "PivotResidualZ":               { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "PivotCurvatureProxy":          { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "PivotSegmentLengthBars":       { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "PivotSegmentLengthMinutes":    { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "PivotSegmentResidualStd":      { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "PivotSegmentVolatility":       { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "tod_sin":                      { "class": "CF", "live": true,  "train_input": true,  "bounded": [-1,1],"fill": "zero" },
  "tod_cos":                      { "class": "CF", "live": true,  "train_input": true,  "bounded": [-1,1],"fill": "zero" },
  "regime_persistence":           { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "chaos_membership":             { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "0" },
  "consolidation_score":          { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "0" },
  "breakout_score":               { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "0" },
  "pressure_up":                  { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "pressure_down":                { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "UpperBandOvershootATR":        { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "LowerBandOvershootATR":        { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "UpperTailRatio":               { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "zero" },
  "LowerTailRatio":               { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "zero" },
  "UpperCloseBackInsideBandFlag": { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "zero", "notes": "causal only if decision made after bar close" },
  "LowerCloseBackInsideBandFlag": { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "zero", "notes": "causal only if decision made after bar close" },
  "BearBreakPressure_10":         { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "zero" },
  "BullBreakPressure_10":         { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "zero" },
  "NetReversalPressure_10":       { "class": "CF", "live": true,  "train_input": true,  "bounded": [-1,1],"fill": "zero" },
  "bars_since_band_break":        { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "-1" },
  "bars_since_psar_flip":         { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "-1" },
  "bars_to_next_pivot":           { "class": "FL", "live": false, "train_input": false, "notes": "FORWARD-LOOKING. Offline label/diagnostic only. Never in encoder." },
  "eq_price":                     { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "close", "notes": "computed from contemporaneous chain proxy" },
  "eq_distance_pct":              { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "gamma_net":                    { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "gamma_flip":                   { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "close" },
  "zone_tight_upper":             { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "close" },
  "zone_tight_lower":             { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "close" },
  "zone_full_upper":              { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "close" },
  "zone_full_lower":              { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "close" },
  "pinning_bias":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,2], "fill": "zero" },
  "ps_pnl_pct":                   { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ps_credit_norm":               { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ps_bars_held":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ps_dte_frac":                  { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "zero" },
  "ps_delta_exp":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ps_gamma_exp":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ps_theta_pos":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ps_iv_change":                 { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ps_high_water":                { "class": "CF", "live": true,  "train_input": true,  "bounded": [0,1], "fill": "zero" },
  "ps_mae":                       { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "ps_unrealized_norm":           { "class": "CF", "live": true,  "train_input": true,  "bounded": false, "fill": "zero" },
  "pop":                          { "class": "OL", "live": false, "train_input": false, "notes": "training label only" },
  "ev":                           { "class": "OL", "live": false, "train_input": false, "notes": "training label only" },
  "max_loss":                     { "class": "OL", "live": false, "train_input": false, "notes": "training label only" },
  "var_95":                       { "class": "OL", "live": false, "train_input": false, "notes": "training label only" },
  "cvar_95":                      { "class": "OL", "live": false, "train_input": false, "notes": "training label only" },
  "exit_signal":                  { "class": "OL", "live": false, "train_input": false, "notes": "exit_bce label only" },
  "strategy_label":               { "class": "OL", "live": false, "train_input": false, "notes": "strategy_idx label only" },
  "target_spot":                  { "class": "OL", "live": false, "train_input": false, "notes": "spot prediction label" },
  "position_size_mult":           { "class": "OL", "live": false, "train_input": false, "notes": "sizing label" }
}
```

**Step 2: Write validation script for the map**
```python
# docs/v46/validate_role_map.py
import json, sys

with open("docs/v46/FEATURE_ROLE_MAP.json") as f:
    role_map = json.load(f)

valid_classes = {"CF", "LS", "OL", "OD", "FL"}
errors = []
for col, spec in role_map.items():
    if col.startswith("_"): continue
    if spec.get("class") not in valid_classes:
        errors.append(f"{col}: invalid class '{spec.get('class')}'")
    if spec.get("live") and spec.get("class") in ("FL", "OL"):
        errors.append(f"{col}: class {spec['class']} cannot have live=true")
    if spec.get("train_input") and spec.get("class") == "FL":
        errors.append(f"{col}: FL class cannot have train_input=true")

if errors:
    print("ROLE MAP ERRORS:")
    for e in errors: print(f"  {e}")
    sys.exit(1)
else:
    cf = sum(1 for s in role_map.values() if not str(s).startswith("_") and isinstance(s, dict) and s.get("class")=="CF")
    print(f"Role map valid. CF={cf}, total={len([k for k in role_map if not k.startswith('_')])}")
```

**Step 3: Run validator**
```bash
python3 docs/v46/validate_role_map.py
```
Expected: `Role map valid.`

**Step 4: Commit**
```bash
git add docs/v46/FEATURE_ROLE_MAP.json docs/v46/validate_role_map.py
git commit -m "docs(governance): add v46 feature role map with causal/leakage classification"
```

---

### Task 2.3: Write schema_v46.py with metadata

**Files:**
- Create: `intelligence/schema_v46.py`

**Step 1: Write**

```python
"""
v46 feature schema with full metadata.
RULE: FEATURE_COLS_V46 = only columns where class=CF or class=LS in FEATURE_ROLE_MAP.
      Forward-looking (FL) and offline labels (OL) are never in this list.
"""
from intelligence.schema_v43 import (
    TF_LABEL_NAMES, STRATEGY_TYPES, ABSTAIN_IDX, POS_STATE_NAMES, N_POS_STATE,
    get_dte_affinity,
)
import json, os

# Load role map at import time for validation
_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'docs', 'v46', 'FEATURE_ROLE_MAP.json')
try:
    with open(_MAP_PATH) as f:
        FEATURE_ROLE_MAP: dict = json.load(f)
except FileNotFoundError:
    FEATURE_ROLE_MAP = {}

def get_role(col: str) -> dict:
    return FEATURE_ROLE_MAP.get(col, {})

def is_live_eligible(col: str) -> bool:
    return FEATURE_ROLE_MAP.get(col, {}).get("live", False)

def is_train_input(col: str) -> bool:
    return FEATURE_ROLE_MAP.get(col, {}).get("train_input", False)

# Canonical encoder input list — only CF + LS, ordered, no FL/OL
# NOTE: bars_to_next_pivot is explicitly EXCLUDED (FL class)
CAUSAL_FEATURE_COLS_V46 = [
    col for col, spec in FEATURE_ROLE_MAP.items()
    if not col.startswith("_")
    and isinstance(spec, dict)
    and spec.get("train_input") is True
]

# Separate: strategy-output signal slots (58 channels, each lagged)
N_STRATEGY_OUTPUTS = 58

STRATEGY_OUTPUT_NAMES = [
    'iron_condor','iron_butterfly','short_straddle','bull_put_spread_credit',
    'short_strangle','bear_call_spread_credit','short_put','jade_lizard',
    'reverse_jade_lizard','short_call_condor','short_put_condor','short_guts',
    'call_ratio_spread','put_ratio_spread','cash_secured_put',
    'covered_short_straddle','covered_short_strangle','straddle_long',
    'call_ratio_backspread','strangle_long','put_ratio_backspread',
    'inverse_iron_butterfly','inverse_iron_condor','short_call_butterfly',
    'short_put_butterfly','long_call_condor','long_put_condor',
    'strip','strap','guts','long_call','long_put','bull_call_spread',
    'bear_put_spread','long_call_butterfly','long_put_butterfly',
    'protective_put','long_synthetic_future','short_synthetic_future',
    'long_combo','short_combo','collar','diagonal_call','diagonal_put',
    'put_broken_wing','call_broken_wing','inverse_call_broken_wing',
    'inverse_put_broken_wing','bull_call_ladder','bear_call_ladder',
    'bull_put_ladder','bear_put_ladder','synthetic_put','calendar_call',
    'calendar_put','double_diagonal','short_call','covered_call',
]
assert len(STRATEGY_OUTPUT_NAMES) == N_STRATEGY_OUTPUTS, \
    f"Expected 58 strategy names, got {len(STRATEGY_OUTPUT_NAMES)}"

# Strategy universe governance
RESEARCH_UNIVERSE    = set(STRATEGY_OUTPUT_NAMES)   # all 58 used in optimization
LIVE_ELIGIBLE_UNIVERSE = {                            # conservative whitelist for paper trading
    'iron_condor', 'iron_butterfly', 'short_straddle',
    'bull_put_spread_credit', 'bear_call_spread_credit',
    'short_strangle', 'jade_lizard',
}

# Offline-only diagnostics (never model inputs)
OFFLINE_DIAGNOSTIC_COLS = ['bars_to_next_pivot']

# Offline training labels
OFFLINE_LABEL_COLS = ['pop', 'ev', 'max_loss', 'var_95', 'cvar_95',
                       'exit_signal', 'strategy_label', 'target_spot', 'position_size_mult']

VERSION = "v46"
INPUT_DIM_V46 = len(CAUSAL_FEATURE_COLS_V46)
```

**Step 2: Write validation test**
```python
# tests/test_schema_v46.py
from intelligence.schema_v46 import (
    CAUSAL_FEATURE_COLS_V46, STRATEGY_OUTPUT_NAMES, N_STRATEGY_OUTPUTS,
    OFFLINE_DIAGNOSTIC_COLS, OFFLINE_LABEL_COLS, is_live_eligible,
)

def test_strategy_count():
    assert len(STRATEGY_OUTPUT_NAMES) == 58

def test_no_fl_in_causal_features():
    # bars_to_next_pivot must never be in CAUSAL_FEATURE_COLS_V46
    assert 'bars_to_next_pivot' not in CAUSAL_FEATURE_COLS_V46

def test_no_labels_in_causal_features():
    for col in OFFLINE_LABEL_COLS:
        assert col not in CAUSAL_FEATURE_COLS_V46, f"{col} must not be in encoder inputs"

def test_bars_to_next_pivot_not_live():
    assert not is_live_eligible('bars_to_next_pivot')

def test_causal_features_nonempty():
    assert len(CAUSAL_FEATURE_COLS_V46) > 50  # sanity lower bound
```

**Step 3: Run tests**
```bash
python3 -m pytest tests/test_schema_v46.py -v
```
Expected: 5 PASS.

**Step 4: Commit**
```bash
git add intelligence/schema_v46.py tests/test_schema_v46.py
git commit -m "feat(schema): add schema_v46.py with full metadata and causal/live eligibility"
```

---

### Task 2.4: Write v46 MANIFEST.json

**Files:**
- Create: `data/Datasetv4/v46/MANIFEST.json`

**Step 1: Write**
```json
{
  "_schema_version": "v46.0",
  "_doc": "Authoritative manifest for all v46 training datasets. All ETL and training code reads this.",
  "governance": {
    "train_years": [2020, 2021, 2022, 2023, 2024],
    "holdout_year": 2025,
    "holdout_policy": "2025 data NEVER touches training, optimization, or parameter selection. Validation only after training is complete.",
    "causality_policy": "docs/v46/LEAKAGE_POLICY.md",
    "feature_role_map": "docs/v46/FEATURE_ROLE_MAP.json",
    "schema": "intelligence/schema_v46.py",
    "random_seed": 42,
    "normalization_stats": "models/norm_stats_v46.json",
    "normalization_source": "train years only — never fitted on 2025"
  },
  "base_datasets": {
    "count_per_year": 5,
    "timeframes": ["m1", "m5", "m15", "h1", "options"],
    "path_template": "data/Datasetv4/v43/{year}/{tf}_dataset_v43_{year}.csv",
    "options_path_template": "data/Datasetv4/v43/{year}/options_{year}_v43.csv",
    "notes": "Options files are REAL historical SPY options data, not synthetic."
  },
  "strategy_output_datasets": {
    "count": 58,
    "path_template": "data/Datasetv4/v46/strategy_outputs/{year}/{strategy}_trajectory.csv",
    "bar_level_signal": "rolling_expectancy_lag1",
    "lag_policy": "strategy signal lagged by 1 bar before merge — never same-bar realized outcome",
    "mask_policy": "mask=1 where signal available, mask=0 where no strategy activity on that bar",
    "missing_bar_fill": 0.0,
    "aggregation": "if multiple trades at same timestamp: mean of pnl_pct, then lag"
  },
  "processed_datasets": {
    "path_template": "data/Datasetv4/v46/processed/{year}/{tf}_dataset_v46_{year}.csv",
    "feature_cols": "CAUSAL_FEATURE_COLS_V46 from schema_v46.py",
    "excluded": ["bars_to_next_pivot", "pop", "ev", "max_loss", "var_95", "cvar_95", "exit_signal", "strategy_label"]
  },
  "tick_dataset": {
    "status": "PENDING — awaiting historical tick data acquisition",
    "planned_slot": 64
  },
  "totals": {
    "base_inputs": 5,
    "strategy_inputs": 58,
    "current_total": 63,
    "future_total_with_tick": 64
  }
}
```

**Step 2: Commit**
```bash
git add data/Datasetv4/v46/MANIFEST.json
git commit -m "docs(manifest): write v46 authoritative training manifest (63 datasets, 2025 holdout)"
```

---

## PHASE 3 — 2020–2024 Schema Parity Audit

> Validate existing 2020–2024 files before generating anything new.

### Task 3.1: Run schema parity audit

**Files:**
- Create: `data/Datasetv4/v43/scripts/audit_schema_parity_v46.py`

**Step 1: Write audit using FEATURE_ROLE_MAP as source of truth**

```python
#!/usr/bin/env python3
"""
Audit 2020-2024 dataset columns against v46 schema requirements.
Reports: present, missing from required CF columns, extra columns, NaN density.
"""
import pandas as pd, json, os, sys

BASE  = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024]
TFS   = ['m1', 'm5', 'm15', 'h1']

with open("docs/v46/FEATURE_ROLE_MAP.json") as f:
    role_map = json.load(f)

# Required CF columns that must be present in train files
required_cf = sorted([
    col for col, spec in role_map.items()
    if not col.startswith("_") and isinstance(spec, dict) and spec.get("train_input") is True
])

report = {}
for year in YEARS:
    report[year] = {}
    for tf in TFS:
        # Check both old 2025-style name and year-specific name
        candidates = [
            f"{BASE}/{year}/{tf}_dataset_v43_{year}.csv",
            f"{BASE}/{year}/{tf}_dataset_v43_final.csv",  # fallback
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path:
            report[year][tf] = {"status": "MISSING_FILE"}
            continue

        df = pd.read_csv(path, nrows=200, low_memory=False)
        n_rows = sum(1 for _ in open(path)) - 1
        cols   = set(df.columns)

        missing_cf = sorted(set(required_cf) - cols)
        nan_density = {c: df[c].isna().mean() for c in df.columns if df[c].isna().sum() > 0}
        high_nan    = {c: v for c, v in nan_density.items() if v > 0.05}

        report[year][tf] = {
            "path":       path,
            "rows":       n_rows,
            "cols":       len(cols),
            "missing_cf": missing_cf,
            "high_nan":   high_nan,
        }

        status = "OK" if not missing_cf else f"MISSING {len(missing_cf)} CF cols"
        nan_warn = f" | HIGH_NAN={len(high_nan)}" if high_nan else ""
        print(f"[{status}] {year}/{tf} ({n_rows} rows, {len(cols)} cols){nan_warn}")
        if missing_cf:
            print(f"  MISSING: {missing_cf}")

with open("data/Datasetv4/v46/schema_parity_audit_v46.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print("\nAudit saved to data/Datasetv4/v46/schema_parity_audit_v46.json")
```

**Step 2: Run**
```bash
python3 data/Datasetv4/v43/scripts/audit_schema_parity_v46.py 2>&1 | tee docs/baseline/schema_parity_report.txt
```

**Step 3: Commit script + report**
```bash
git add data/Datasetv4/v43/scripts/audit_schema_parity_v46.py docs/baseline/schema_parity_report.txt
git commit -m "feat(audit): run v46 schema parity audit on 2020-2024 datasets"
```

**Step 4: Triage output.** Missing CF columns from audit become the work list for Phase 4.

---

## PHASE 4 — Causal Feature Generation (2020–2024)

> Only causal (CF class) features are generated here. Forward-looking (FL) columns are stored separately as offline diagnostics if needed at all.

### Task 4.1: Add band-break reversal block (9 CF columns)

**Files:**
- Create: `data/Datasetv4/v43/scripts/add_bandbreak_features.py`
- Create: `tests/test_bandbreak_features.py`

**Step 1: Write compute_bandbreak (causal version)**

```python
#!/usr/bin/env python3
"""
Add 9 band-break reversal features (all CF class per FEATURE_ROLE_MAP).
CAUSAL NOTE: All features use only bar-t-and-earlier data.
  UpperCloseBackInsideBandFlag uses prior-bar break + current-bar close.
  This is causal if and only if model decisions are made after bar-t closes.
  This is enforced by temporal execution contract (docs/v46/LEAKAGE_POLICY.md).
"""
import pandas as pd, numpy as np, os

PRESSURE_WINDOW = 10

BANDBREAK_CF_COLS = [
    'UpperBandOvershootATR', 'LowerBandOvershootATR',
    'UpperTailRatio', 'LowerTailRatio',
    'UpperCloseBackInsideBandFlag', 'LowerCloseBackInsideBandFlag',
    'BearBreakPressure_10', 'BullBreakPressure_10', 'NetReversalPressure_10',
]

def compute_bandbreak(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['bb_upper_dyn', 'bb_lower_dyn', 'atr_pct', 'close', 'high', 'low', 'open']:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    atr_abs = (df['atr_pct'] * df['close']).replace(0, np.nan).ffill().fillna(1e-6)

    df['UpperBandOvershootATR'] = ((df['high']  - df['bb_upper_dyn']) / atr_abs).clip(lower=0)
    df['LowerBandOvershootATR'] = ((df['bb_lower_dyn'] - df['low'])   / atr_abs).clip(lower=0)

    body       = (df['close'] - df['open']).abs().replace(0, np.nan)
    upper_wick = df['high'] - df[['open','close']].max(axis=1)
    lower_wick = df[['open','close']].min(axis=1) - df['low']
    df['UpperTailRatio'] = (upper_wick / (upper_wick + body.fillna(upper_wick))).fillna(0).clip(0,1)
    df['LowerTailRatio'] = (lower_wick / (lower_wick + body.fillna(lower_wick))).fillna(0).clip(0,1)

    broke_upper = df['close'] > df['bb_upper_dyn']
    broke_lower = df['close'] < df['bb_lower_dyn']
    # Causal: prior bar broke, current bar closed back inside
    df['UpperCloseBackInsideBandFlag'] = (broke_upper.shift(1).fillna(False) & (df['close'] <= df['bb_upper_dyn'])).astype(float)
    df['LowerCloseBackInsideBandFlag'] = (broke_lower.shift(1).fillna(False) & (df['close'] >= df['bb_lower_dyn'])).astype(float)

    df['BearBreakPressure_10'] = broke_upper.rolling(PRESSURE_WINDOW, min_periods=1).mean().fillna(0)
    df['BullBreakPressure_10'] = broke_lower.rolling(PRESSURE_WINDOW, min_periods=1).mean().fillna(0)
    df['NetReversalPressure_10'] = df['BullBreakPressure_10'] - df['BearBreakPressure_10']

    return df

BASE  = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024]
TFS   = ['m1', 'm5', 'm15', 'h1']

if __name__ == '__main__':
    for year in YEARS:
        for tf in TFS:
            path = f"{BASE}/{year}/{tf}_dataset_v43_{year}.csv"
            if not os.path.exists(path):
                print(f"[SKIP] {path}"); continue
            df = pd.read_csv(path, low_memory=False)
            if all(c in df.columns for c in BANDBREAK_CF_COLS):
                print(f"[SKIP] {year}/{tf} already has bandbreak cols"); continue
            try:
                df = compute_bandbreak(df)
                df.to_csv(path, index=False)
                print(f"[OK]   {year}/{tf} → {len(df)} rows, added 9 bandbreak cols")
            except ValueError as e:
                print(f"[ERR]  {year}/{tf}: {e}")
```

**Step 2: Write tests**

```python
# tests/test_bandbreak_features.py
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from data.Datasetv4.v43.scripts.add_bandbreak_features import compute_bandbreak, BANDBREAK_CF_COLS

def _make_df(n=60, seed=42):
    rng = np.random.default_rng(seed)
    close = 500 + np.cumsum(rng.standard_normal(n) * 2)
    high  = close + np.abs(rng.standard_normal(n))
    low   = close - np.abs(rng.standard_normal(n))
    return pd.DataFrame({
        'open': close - 0.3, 'high': high, 'low': low, 'close': close,
        'bb_upper_dyn': close + 5, 'bb_lower_dyn': close - 5,
        'atr_pct': np.full(n, 0.005),
    })

def test_all_cols_present():
    df = compute_bandbreak(_make_df())
    for c in BANDBREAK_CF_COLS:
        assert c in df.columns

def test_no_nans():
    df = compute_bandbreak(_make_df())
    for c in BANDBREAK_CF_COLS:
        assert df[c].isna().sum() == 0, f"NaN in {c}"

def test_pressure_bounded():
    df = compute_bandbreak(_make_df())
    assert df['BearBreakPressure_10'].between(0,1).all()
    assert df['BullBreakPressure_10'].between(0,1).all()

def test_overshoot_nonnegative():
    df = compute_bandbreak(_make_df())
    assert (df['UpperBandOvershootATR'] >= 0).all()
    assert (df['LowerBandOvershootATR'] >= 0).all()

def test_flags_binary():
    df = compute_bandbreak(_make_df())
    assert set(df['UpperCloseBackInsideBandFlag'].unique()).issubset({0.0, 1.0})
    assert set(df['LowerCloseBackInsideBandFlag'].unique()).issubset({0.0, 1.0})

def test_net_reversal_range():
    df = compute_bandbreak(_make_df())
    assert df['NetReversalPressure_10'].between(-1,1).all()
```

**Step 3: Run tests**
```bash
python3 -m pytest tests/test_bandbreak_features.py -v
```
Expected: 6 PASS.

**Step 4: Run on 2020–2024**
```bash
python3 data/Datasetv4/v43/scripts/add_bandbreak_features.py 2>&1 | tee logs/bandbreak_add.log
```
Expected: 20 [OK] lines (5 years × 4 TFs).

**Step 5: Commit**
```bash
git add data/Datasetv4/v43/scripts/add_bandbreak_features.py tests/test_bandbreak_features.py
git commit -m "feat(data): add 9 causal band-break reversal cols to 2020-2024 v43 datasets"
```

---

### Task 4.2: Add causal pivot diagnostics (2 CF columns, 1 FL stored separately)

> `bars_since_band_break` and `bars_since_psar_flip` are CF. `bars_to_next_pivot` is FL and stored in a **separate** offline diagnostic file, NEVER in the main dataset.

**Files:**
- Create: `data/Datasetv4/v43/scripts/add_pivot_diagnostics.py`

**Step 1: Write — CF columns only in main CSV, FL in sidecar**

```python
#!/usr/bin/env python3
"""
Causal pivot diagnostics:
  CF: bars_since_band_break, bars_since_psar_flip  → written to main CSV
  FL: bars_to_next_pivot → written to SIDECAR file only, never main CSV
"""
import pandas as pd, numpy as np, os

BASE  = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024]
TFS   = ['m1', 'm5', 'm15', 'h1']
CF_COLS = ['bars_since_band_break', 'bars_since_psar_flip']
FL_COLS = ['bars_to_next_pivot']   # sidecar only

def _bars_since(event_series: pd.Series) -> np.ndarray:
    out, cnt = np.full(len(event_series), np.nan), np.nan
    for i in range(len(event_series)):
        if event_series.iloc[i] == 1.0: cnt = 0
        elif not np.isnan(cnt): cnt += 1
        out[i] = cnt
    return out

def compute_causal_diag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'UpperCloseBackInsideBandFlag' in df.columns and 'LowerCloseBackInsideBandFlag' in df.columns:
        any_break = (df['UpperCloseBackInsideBandFlag'] + df['LowerCloseBackInsideBandFlag']).clip(0,1)
    else:
        any_break = pd.Series(np.zeros(len(df)), index=df.index)
    df['bars_since_band_break'] = _bars_since(any_break)
    df['bars_since_band_break'] = df['bars_since_band_break'].fillna(-1)

    if 'psar_trend' in df.columns:
        trend = df['psar_trend'].fillna(0)
        flip  = (trend != trend.shift(1)).astype(float); flip.iloc[0] = 0
        df['bars_since_psar_flip'] = _bars_since(flip)
        df['bars_since_psar_flip'] = df['bars_since_psar_flip'].fillna(-1)
    else:
        df['bars_since_psar_flip'] = -1.0
    return df

def compute_fl_sidecar(df: pd.DataFrame, path: str):
    """Compute forward-looking bars_to_next_pivot and save to sidecar ONLY."""
    if 'PivotHigh' not in df.columns and 'PivotLow' not in df.columns:
        return
    is_pivot = ((df.get('PivotHigh', 0).fillna(0) != 0) |
                (df.get('PivotLow',  0).fillna(0) != 0)).astype(float)
    btn = np.full(len(df), np.nan)
    nxt = np.nan
    for i in range(len(df)-1, -1, -1):
        if is_pivot.iloc[i]: nxt = 0
        elif not np.isnan(nxt): nxt += 1
        btn[i] = nxt
    sidecar = pd.DataFrame({'timestamp': df['timestamp'], 'bars_to_next_pivot': btn})
    sidecar_path = path.replace('.csv', '_FL_sidecar.csv')
    sidecar.to_csv(sidecar_path, index=False)
    print(f"  [FL sidecar] {sidecar_path}")

for year in YEARS:
    for tf in TFS:
        path = f"{BASE}/{year}/{tf}_dataset_v43_{year}.csv"
        if not os.path.exists(path): print(f"[SKIP] {path}"); continue
        df = pd.read_csv(path, low_memory=False)
        if all(c in df.columns for c in CF_COLS):
            print(f"[SKIP] {year}/{tf} already has CF diag cols"); continue
        df = compute_causal_diag(df)
        df.to_csv(path, index=False)
        compute_fl_sidecar(df, path)
        print(f"[OK]   {year}/{tf} → added {CF_COLS}")
```

**Step 2: Run**
```bash
python3 data/Datasetv4/v43/scripts/add_pivot_diagnostics.py
```

**Step 3: Verify `bars_to_next_pivot` is NOT in main CSV**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/Datasetv4/v43/2020/m5_dataset_v43_2020.csv', nrows=1)
assert 'bars_to_next_pivot' not in df.columns, 'LEAKAGE: bars_to_next_pivot in main CSV!'
print('OK — bars_to_next_pivot not in main CSV')
"
```

**Step 4: Commit**
```bash
git add data/Datasetv4/v43/scripts/add_pivot_diagnostics.py
git commit -m "feat(data): add causal pivot diagnostics; FL bars_to_next_pivot to sidecar only"
```

---

### Task 4.3: Add payout equilibrium features (9 CF columns)

**Files:**
- Create: `data/Datasetv4/v43/scripts/add_payout_equilibrium.py`

> These are computed from the contemporaneous bar's close price and a synthetic OI proxy — causal as of bar close. Labeled as proxy features, not true dealer positioning.

**Step 1:** (Implementation as in v1 plan but add explicit header comment)

```python
#!/usr/bin/env python3
"""
Add 9 payout equilibrium features (all CF class).
PROXY NOTE: OI distribution is synthetic (distance-weighted BSM model),
NOT real dealer positioning. Live behavior may differ materially.
Labeled 'eq_' to distinguish from real chain equilibrium features.
All computed from bar-t close — causal.
"""
# ... (implementation from v1 plan, unchanged)
```

**Step 2: Run on m5/m15/h1 only (m1 too slow for prototype)**
```bash
python3 data/Datasetv4/v43/scripts/add_payout_equilibrium.py \
    --timeframes m5 m15 h1 \
    2>&1 | tee logs/eq_features.log
```

**Step 3: Commit**
```bash
git add data/Datasetv4/v43/scripts/add_payout_equilibrium.py
git commit -m "feat(data): add 9 equilibrium proxy features to 2020-2024 m5/m15/h1"
```

---

## PHASE 5 — Feature QA and Live Eligibility Audit

> Every generated feature column is inspected: NaN density, range, sparsity, correlation with near-duplicates, and live-eligibility tag.

### Task 5.1: Write comprehensive feature QA script

**Files:**
- Create: `data/Datasetv4/v43/scripts/feature_qa_v46.py`

**Step 1: Write**

```python
#!/usr/bin/env python3
"""
Feature QA for v46 columns on 2020-2024 datasets.
For each CF column: NaN density, range, sparsity, correlation warnings.
Outputs: data/Datasetv4/v46/feature_qa_report.json
"""
import pandas as pd, numpy as np, json, os

BASE  = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024]
TF    = "m5"  # anchor timeframe for QA

with open("docs/v46/FEATURE_ROLE_MAP.json") as f:
    role_map = json.load(f)

cf_cols = [c for c, s in role_map.items() if not c.startswith("_") and isinstance(s,dict) and s.get("train_input")]

qa = {}
for year in YEARS:
    path = f"{BASE}/{year}/{TF}_dataset_v43_{year}.csv"
    if not os.path.exists(path): continue
    df = pd.read_csv(path, low_memory=False)
    qa[year] = {}
    for col in cf_cols:
        if col not in df.columns:
            qa[year][col] = {"status": "MISSING"}
            continue
        s = df[col]
        qa[year][col] = {
            "status":        "OK",
            "nan_pct":       round(s.isna().mean(), 4),
            "zero_pct":      round((s == 0).mean(), 4),
            "min":           round(float(s.min()), 6) if not s.isna().all() else None,
            "max":           round(float(s.max()), 6) if not s.isna().all() else None,
            "mean":          round(float(s.mean()), 6) if not s.isna().all() else None,
            "std":           round(float(s.std()),  6) if not s.isna().all() else None,
            "first_valid":   int(s.first_valid_index()) if s.first_valid_index() is not None else None,
        }

# Print summary
print(f"{'Col':<35} {'Y':<6} {'NaN%':<7} {'Zero%':<7} {'Min':<12} {'Max':<12}")
print("-"*80)
for year, cols in qa.items():
    for col, info in cols.items():
        if info.get("status") == "MISSING":
            print(f"{'*** MISSING ***':<35} {year:<6} {col}")
        elif info.get("nan_pct", 0) > 0.05:
            print(f"{'[HIGH NaN]':<35} {year:<6} {info['nan_pct']:.1%}  {col}")

with open("data/Datasetv4/v46/feature_qa_report.json", "w") as f:
    json.dump(qa, f, indent=2)
print("\nQA report saved.")
```

**Step 2: Run**
```bash
python3 data/Datasetv4/v43/scripts/feature_qa_v46.py 2>&1 | tee docs/baseline/feature_qa_v46.txt
```

**Step 3: Fix any HIGH NaN or MISSING columns before proceeding**

**Step 4: Commit**
```bash
git add data/Datasetv4/v43/scripts/feature_qa_v46.py docs/baseline/feature_qa_v46.txt
git commit -m "feat(qa): feature QA report for all v46 CF columns on 2020-2024"
```

---

## PHASE 6 — Backtester Event-Schema Hardening

> Define and enforce bar-timing rules, event atomicity, and the OPEN/CLOSE join contract before any optimization run.

### Task 6.1: Write event schema contract document

**Files:**
- Create: `docs/v46/EVENT_SCHEMA_CONTRACT.md`

```markdown
# v46 Event Schema Contract

## Bar Timing Rules (frozen)
- Features at bar-t close: computed from OHLCV of bar t.
- Entry decision: made at end of bar t.
- Fill: at open of bar t+1 (or next available mid snapshot).
- Exit evaluation: at close of bar t using same-bar close price.
- NO intrabar decisions. NO lookahead beyond bar-t close.

## OPEN Event Required Fields
| Field | Type | Notes |
|---|---|---|
| action | str | Must be "OPEN" |
| trade_id | str | UUID, unique per trade |
| bar_idx | int | Index of decision bar |
| dt | str | Timestamp of decision bar t |
| fill_dt | str | Expected fill at bar t+1 open |
| spot | float | Close price of bar t |
| entry_credit | float | Net credit after slippage |
| trade_margin | float | Collateral required |
| template_id | str | Strategy name from STRATEGY_OUTPUT_NAMES |
| strategy_class | str | Category (single_call, iron_condor, etc.) |
| dte_entry | int | DTE at entry |
| pop_prob | float | Model P(profit) estimate |

## CLOSE Event Required Fields
| Field | Type | Notes |
|---|---|---|
| action | str | Must be "CLOSE" |
| trade_id | str | Must match corresponding OPEN |
| bar_idx | int | Bar index of close decision |
| dt | str | Timestamp of close decision |
| reason | str | One of: hard_max_loss / delta_violation / neural_exit / profit_target / expiry / manual |
| pnl | float | Realized P&L in dollars |
| pnl_pct | float | P&L as fraction of credit received |
| held_bars | int | Number of bars from entry to exit |
| exit_details | dict | {cost_at_close, spot_at_close, fill_price} |

## Atomicity Invariant
- Every OPEN must have exactly one matching CLOSE (matched by trade_id).
- Partial closes are not permitted for defined-risk structures.
- Orphaned OPENs are a hard error in any trajectory export.
```

**Step 2: Commit**
```bash
git add docs/v46/EVENT_SCHEMA_CONTRACT.md
git commit -m "docs(contract): add v46 event schema — OPEN/CLOSE atomicity and bar-timing rules"
```

---

### Task 6.2: Complete StrategyRegistry, ParameterGrid, OptimizationRun

> These were in v1 plan (Tasks 2.1–2.3). Implement now, before optimization. Same code, same tests. Reference v1 plan Tasks 2.1–2.3 for exact implementation.

**Files:**
- Create: `kaggle/core/strategy_registry.py`
- Create: `kaggle/core/param_grid.py`
- Create: `kaggle/core/optimization_record.py`

**Tests:** `tests/test_strategy_registry.py`, `tests/test_param_grid.py`, `tests/test_optimization_record.py`

Run all three test files:
```bash
python3 -m pytest tests/test_strategy_registry.py tests/test_param_grid.py tests/test_optimization_record.py -v
```
Expected: all PASS.

```bash
git add kaggle/core/ tests/test_strategy_registry.py tests/test_param_grid.py tests/test_optimization_record.py
git commit -m "feat(registry): add StrategyRegistry, ParameterGrid, OptimizationRun"
```

---

### Task 6.3: Port ExitDecisionStack into v45 + standardize CLOSE events

> Reference v1 plan Tasks 2.4–2.5 for implementation. After porting:

**Verify OPEN/CLOSE atomicity:**
```bash
python3 - <<'EOF'
import json, collections
opens, closes = collections.defaultdict(int), collections.defaultdict(int)
with open('bar_trace.jsonl') as f:
    for line in f:
        e = json.loads(line)
        if e.get('action') == 'OPEN':  opens[e['trade_id']]  += 1
        if e.get('action') == 'CLOSE': closes[e['trade_id']] += 1
orphans = [tid for tid in opens if opens[tid] != closes.get(tid, 0)]
print(f"Opens={sum(opens.values())} Closes={sum(closes.values())} Orphans={len(orphans)}")
assert len(orphans) == 0, f"ATOMICITY VIOLATION: {orphans}"
EOF
```

```bash
git add kaggle/condor_brain_backtest_v45.py
git commit -m "feat(backtest): port ExitDecisionStack + enforce OPEN/CLOSE atomicity"
```

---

## PHASE 7 — Strategy-Output Bar-Level Contract

> The most important missing piece from v1. Before running 116+ optimization jobs, define exactly what value goes into the model at each bar for each strategy.

### Task 7.1: Write STRATEGY_OUTPUT_CONTRACT.md

**Files:**
- Create: `docs/v46/STRATEGY_OUTPUT_CONTRACT.md`

```markdown
# Strategy Output Bar-Level Contract

## Primary Key
Each trajectory CSV has one row per bar where a decision was made.
Timestamp is the join key for merging with base datasets.
Primary key: (timestamp, trade_id).

## Trajectory CSV Required Columns
| Column | Type | Notes |
|---|---|---|
| timestamp | str/datetime | Bar-t decision timestamp |
| trade_id | str | UUID, links to OPEN event |
| action | str | OPEN or CLOSE |
| template_id | str | Strategy name |
| pnl_pct | float | Realized P&L as pct of credit. CLOSE rows only. |
| held_bars | int | Bars held. CLOSE rows only. |
| entry_credit | float | Credit received at OPEN. |
| spot | float | Close price at decision bar. |

## Bar-Level Model Signal (what enters the model)

The signal fed to the model at bar t for strategy s is:
  `rolling_expectancy_lag1_s` = rolling mean of `pnl_pct` for CLOSE events
                                 up to and including bar t-1 (LAG 1).

This is causal: it uses only realized outcomes from before bar t.

### Computation
```python
# Per strategy s:
close_rows = traj_df[traj_df['action'] == 'CLOSE'].copy()
close_rows = close_rows.sort_values('timestamp')
close_rows['rolling_expectancy'] = close_rows['pnl_pct'].expanding().mean()
# Lag by 1 to ensure no same-bar leakage
close_rows['rolling_expectancy_lag1'] = close_rows['rolling_expectancy'].shift(1)
# Merge onto base m5 bars
merged = base_m5.merge(
    close_rows[['timestamp', 'rolling_expectancy_lag1']].rename(
        columns={'rolling_expectancy_lag1': f'strat_{s}'}),
    on='timestamp', how='left'
)
merged[f'strat_{s}'] = merged[f'strat_{s}'].fillna(0.0)  # neutral = 0
```

## Mask Channel
For each strategy s at bar t:
- mask=1 if strategy has at least one historical CLOSE event prior to bar t
- mask=0 if no history yet (strategy has never completed a trade)

Model receives: values tensor [B, T, 58] + mask tensor [B, T, 58].

## No-Trade Bars
Bars where the strategy has no open or closed trade: signal = 0.0, mask = 0.

## Multiple Trades at Same Timestamp
If multiple CLOSE events share the same timestamp: take mean of pnl_pct, then lag.

## Year Isolation
Rolling expectancy is reset at the start of each year.
Cross-year leakage is not permitted (train year 2020 state does not initialize year 2021).
```

**Step 2: Commit**
```bash
git add docs/v46/STRATEGY_OUTPUT_CONTRACT.md
git commit -m "docs(contract): define strategy output bar-level contract with lag and mask policy"
```

---

### Task 7.2: Write and test trajectory merger

**Files:**
- Create: `intelligence/trajectory_merger_v46.py`

**Step 1: Write**

```python
"""
Merge 58 strategy trajectory CSVs into bar-level signal vectors.
Implements the STRATEGY_OUTPUT_CONTRACT spec exactly.
"""
import pandas as pd, numpy as np, os, json
from intelligence.schema_v46 import STRATEGY_OUTPUT_NAMES

def compute_strategy_signal(traj_path: str, strategy_name: str,
                             base_timestamps: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (signal_series, mask_series) aligned to base_timestamps.
    Both series are indexed by timestamp.
    Signal: rolling_expectancy_lag1 (causal, lagged).
    Mask:   1 if history available, 0 if no history yet.
    """
    if not os.path.exists(traj_path):
        null = pd.Series(0.0, index=base_timestamps)
        return null, null.copy()

    df = pd.read_csv(traj_path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    close_rows = df[df['action'] == 'CLOSE'][['timestamp','pnl_pct']].copy()
    close_rows = close_rows.sort_values('timestamp').drop_duplicates('timestamp', keep='last')

    # Rolling mean up to each event
    close_rows['rolling_exp'] = close_rows['pnl_pct'].expanding().mean()
    # Lag by 1 bar
    close_rows['rolling_exp_lag1'] = close_rows['rolling_exp'].shift(1).fillna(0.0)
    # Clip to [-5, 5] to prevent outlier trades from dominating the signal
    close_rows['rolling_exp_lag1'] = close_rows['rolling_exp_lag1'].clip(-5.0, 5.0)
    # Build mask: 1 once first CLOSE event has been seen (and lagged)
    close_rows['mask'] = (close_rows['rolling_exp'].notna().cumsum().shift(1) > 0).astype(float).fillna(0.0)

    base_ts = pd.Series(pd.to_datetime(base_timestamps.values), name='timestamp')
    signal = (pd.merge_asof(
        base_ts.to_frame(), close_rows[['timestamp','rolling_exp_lag1']],
        on='timestamp', direction='backward')
        ['rolling_exp_lag1'].fillna(0.0))
    mask   = (pd.merge_asof(
        base_ts.to_frame(), close_rows[['timestamp','mask']],
        on='timestamp', direction='backward')
        ['mask'].fillna(0.0))

    return signal, mask


def build_strategy_tensors(base_df: pd.DataFrame, year: int,
                            traj_base: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values [N,58], masks [N,58]) for all 58 strategies."""
    N = len(base_df)
    values = np.zeros((N, 58), dtype=np.float32)
    masks  = np.zeros((N, 58), dtype=np.float32)
    base_ts = base_df['timestamp']

    for idx, strat in enumerate(STRATEGY_OUTPUT_NAMES):
        path = f"{traj_base}/{year}/{strat}_trajectory.csv"
        sig, msk = compute_strategy_signal(path, strat, base_ts)
        values[:, idx] = sig.values
        masks[:,  idx] = msk.values

    return values, masks
```

**Step 2: Write tests**

```python
# tests/test_trajectory_merger.py
import pandas as pd, numpy as np, os, tempfile
from intelligence.trajectory_merger_v46 import compute_strategy_signal

def _make_traj(n_trades=10, seed=1):
    rng = np.random.default_rng(seed)
    ts  = pd.date_range("2023-01-03 09:30", periods=n_trades*20, freq="5min")
    close_ts = ts[::20][:n_trades]
    df = pd.DataFrame({
        'timestamp': close_ts,
        'action':    'CLOSE',
        'pnl_pct':   rng.uniform(-0.2, 0.5, n_trades),
    })
    return df

def test_signal_is_lagged():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "traj.csv")
        df = _make_traj()
        df.to_csv(path, index=False)
        base_ts = df['timestamp']
        sig, msk = compute_strategy_signal(path, "test", base_ts)
        # First bar must be 0 (no prior history)
        assert sig.iloc[0] == 0.0

def test_mask_starts_zero():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "traj.csv")
        _make_traj().to_csv(path, index=False)
        base_ts = _make_traj()['timestamp']
        sig, msk = compute_strategy_signal(path, "test", base_ts)
        assert msk.iloc[0] == 0.0  # no history before first event

def test_missing_traj_returns_zeros():
    base_ts = pd.date_range("2023-01-03", periods=10, freq="5min")
    sig, msk = compute_strategy_signal("/nonexistent/path.csv", "x", pd.Series(base_ts))
    assert (sig == 0.0).all()
    assert (msk == 0.0).all()

def test_no_forward_leakage():
    """Signal at bar t must only reflect events BEFORE bar t."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "traj.csv")
        # Single close event at bar 5
        ts = pd.date_range("2023-01-03 09:30", periods=10, freq="5min")
        df = pd.DataFrame({'timestamp': [ts[5]], 'action': ['CLOSE'], 'pnl_pct': [1.0]})
        df.to_csv(path, index=False)
        sig, _ = compute_strategy_signal(path, "x", pd.Series(ts))
        # Bars 0-5: signal must be 0 (event not yet seen as of bar-t-1)
        assert sig.iloc[5] == 0.0  # bar 5 is when event happened; lag means it's not available yet
```

**Step 3: Run**
```bash
python3 -m pytest tests/test_trajectory_merger.py -v
```
Expected: 4 PASS.

**Step 4: Commit**
```bash
git add intelligence/trajectory_merger_v46.py tests/test_trajectory_merger.py
git commit -m "feat(pipeline): add trajectory merger with lag-1 signal and mask per STRATEGY_OUTPUT_CONTRACT"
```

---

## PHASE 8 — Strategy Optimization Runs

> Rollout order: smoke → one strategy all years → five strategies two years → all 58 × 2023–2024.

### Task 8.1: Optimization smoke test (iron_condor × 2025, --limit 500)

```bash
python3 kaggle/run_strategy_optimization.py \
    --years 2025 \
    --strategies iron_condor \
    --limit 500 \
    --seed 42 \
    2>&1 | tee logs/opt_smoke_ic_2025.log
```
Expected: `opt_results.csv` written, trajectory CSV written, 0 errors.

**Verify trajectory lag:**
```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("data/Datasetv4/v46/strategy_outputs/2025/iron_condor_trajectory.csv")
close_rows = df[df['action']=='CLOSE']
print(f"CLOSE events: {len(close_rows)}")
print(close_rows[['timestamp','pnl_pct']].head(5))
EOF
```

---

### Task 8.2: Single strategy × all years (iron_condor × 2020–2024)

```bash
python3 kaggle/run_strategy_optimization.py \
    --years 2020 2021 2022 2023 2024 \
    --strategies iron_condor \
    --seed 42 \
    2>&1 | tee logs/opt_ic_all_years.log
```

**Year-robust ranking check:**
```bash
python3 - <<'EOF'
import pandas as pd, glob
dfs = []
for path in glob.glob("reports/optimization/*/iron_condor/opt_results.csv"):
    year = path.split("/")[2]
    df   = pd.read_csv(path)
    df['year'] = year
    dfs.append(df)
all_df = pd.concat(dfs)
# Rank by median NP/DD across years
by_param = all_df.groupby('param_hash')['np_dd_ratio']
robust = by_param.median() - 0.5 * by_param.std()
print("Top 5 by robust score:")
print(robust.sort_values(ascending=False).head())
EOF
```

---

### Task 8.3: Scale to 5 strategies × 2023–2024

```bash
python3 kaggle/run_strategy_optimization.py \
    --years 2023 2024 \
    --strategies iron_condor iron_butterfly short_straddle bull_put_spread_credit short_strangle \
    --seed 42 \
    2>&1 | tee logs/opt_5strat_2year.log
```
Expected: 10 opt CSVs + 10 trajectory CSVs.

---

### Task 8.4: Full 58 strategies × 2023–2024

```bash
python3 kaggle/run_strategy_optimization.py \
    --years 2023 2024 \
    --seed 42 \
    2>&1 | tee logs/opt_58_2023_2024.log
```
Expected: 116 opt CSVs + 116 trajectory CSVs.

**Minimum trade count validation (≥5 per year, not just total):**
```bash
python3 - <<'EOF'
import pandas as pd, glob

failures = []
for path in glob.glob("reports/optimization/*/*/opt_results.csv"):
    year     = path.split("/")[2]
    df       = pd.read_csv(path)
    strategy = path.split("/")[3]
    # trades_total is per-year (one opt run = one year), so this is already per-year
    low_rows = df[df['trades_total'] < 5]
    if not low_rows.empty:
        for _, row in low_rows.iterrows():
            failures.append(
                f"[{year}][{strategy}] param_hash={row.get('param_hash','?')} "
                f"trades={row['trades_total']} — BELOW 5 per year minimum"
            )

print(f"Low-trade-count configs: {len(failures)}")
for f in failures[:20]:
    print(f"  {f}")

if failures:
    print("\nACTION: Exclude these configs from ranking. Do not use params with <5 trades/year.")
else:
    print("All configs meet >=5 trades/year minimum.")
EOF
```

> **Rule:** Any param combination with `trades_total < 5` in ANY single year is excluded from the multi-year robust ranking, even if it looks strong in other years. One bad year of thin data is enough to disqualify.

**Step: Commit logs**
```bash
git add logs/opt_58_2023_2024.log
git commit -m "data: 58-strategy optimization complete for 2023-2024"
```

---

## PHASE 9 — v46 ETL Pipeline

### Task 9.1: Build data_pipeline_v46.py with three namespaces

**Files:**
- Create: `intelligence/data_pipeline_v46.py`

Three feature namespaces enforced:
1. `BASE`: causal market features from `CAUSAL_FEATURE_COLS_V46`
2. `STRATEGY`: 58-dim lagged signals + 58-dim masks from trajectory merger
3. `OFFLINE`: diagnostics stored separately, never included in train tensor

```python
#!/usr/bin/env python3
"""
v46 ETL pipeline — assembles 63-dataset training CSVs.
Enforces LEAKAGE_POLICY.md: no FL features in output, strategy signals lagged.
Outputs:
  data/Datasetv4/v46/processed/{year}/{tf}_dataset_v46_{year}.csv
  models/norm_stats_v46.json (fitted on train years only)
"""
import argparse, os, json
import pandas as pd, numpy as np
from intelligence.schema_v46 import (
    CAUSAL_FEATURE_COLS_V46, STRATEGY_OUTPUT_NAMES, OFFLINE_DIAGNOSTIC_COLS,
    OFFLINE_LABEL_COLS,
)
from intelligence.trajectory_merger_v46 import build_strategy_tensors

def load_base_df(year: int, tf: str, data_base: str) -> pd.DataFrame:
    path = f"{data_base}/{year}/{tf}_dataset_v43_{year}.csv"
    df = pd.read_csv(path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def extract_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Extract offline label columns into separate DataFrame for loss computation."""
    label_cols = [c for c in OFFLINE_LABEL_COLS if c in df.columns]
    return df[['timestamp'] + label_cols].copy()

def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only causal features. Drop FL and OL columns."""
    drop = set(OFFLINE_DIAGNOSTIC_COLS) | set(OFFLINE_LABEL_COLS)
    keep = [c for c in df.columns if c not in drop]
    return df[keep].copy()

def fit_normalization_stats(years: list, tf: str, data_base: str,
                             feature_cols: list) -> dict:
    """Fit mean/std on train years only."""
    frames = []
    for year in years:
        df = load_base_df(year, tf, data_base)
        cf_present = [c for c in feature_cols if c in df.columns]
        frames.append(df[cf_present])
    combined = pd.concat(frames, ignore_index=True)
    stats = {}
    for col in feature_cols:
        if col not in combined.columns: continue
        s = combined[col].dropna()
        stats[col] = {"mean": float(s.mean()), "std": max(float(s.std()), 1e-8)}
    return stats

def normalize_df(df: pd.DataFrame, stats: dict, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns or col not in stats: continue
        df[col] = (df[col] - stats[col]['mean']) / stats[col]['std']
        df[col] = df[col].clip(-5, 5)
    return df

def build_v46_dataset(year: int, tf: str, args, norm_stats: dict):
    df = load_base_df(year, tf, args.data_base)
    print(f"[ETL] {year}/{tf}: {len(df)} rows base")

    # Labels saved separately
    label_df = extract_labels(df)
    label_path = f"{args.out_base}/{year}/{tf}_labels_v46_{year}.csv"
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    label_df.to_csv(label_path, index=False)

    # Drop non-causal cols
    df = build_feature_df(df)

    # Normalize CF columns
    df = normalize_df(df, norm_stats, CAUSAL_FEATURE_COLS_V46)

    # Strategy signals: 58 values + 58 masks
    values, masks = build_strategy_tensors(df, year, args.traj_base)
    for idx, strat in enumerate(STRATEGY_OUTPUT_NAMES):
        df[f'strat_val_{strat}']  = values[:, idx]
        df[f'strat_mask_{strat}'] = masks[:,  idx]

    out_path = f"{args.out_base}/{year}/{tf}_dataset_v46_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK]  {out_path}: {len(df)} rows x {len(df.columns)} cols")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-years',  nargs='+', type=int, default=[2020,2021,2022,2023,2024])
    ap.add_argument('--timeframes',   nargs='+', default=['m5'])
    ap.add_argument('--data-base',    default='data/Datasetv4/v43')
    ap.add_argument('--traj-base',    default='data/Datasetv4/v46/strategy_outputs')
    ap.add_argument('--out-base',     default='data/Datasetv4/v46/processed')
    ap.add_argument('--norm-out',     default='models/norm_stats_v46.json')
    ap.add_argument('--force',        action='store_true')
    args = ap.parse_args()

    # Fit normalization on train years only
    print("[ETL] Fitting normalization statistics on train years...")
    norm_stats = fit_normalization_stats(args.train_years, args.timeframes[0],
                                         args.data_base, CAUSAL_FEATURE_COLS_V46)
    os.makedirs(os.path.dirname(args.norm_out), exist_ok=True)
    with open(args.norm_out, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"[ETL] Norm stats saved → {args.norm_out} ({len(norm_stats)} cols)")

    for year in args.train_years:
        for tf in args.timeframes:
            build_v46_dataset(year, tf, args, norm_stats)

if __name__ == '__main__':
    main()
```

**Step 2: Run on 2023–2024**
```bash
python3 intelligence/data_pipeline_v46.py \
    --train-years 2023 2024 \
    --timeframes m5 \
    2>&1 | tee logs/pipeline_v46.log
```

**Step 3: Verify no FL cols in output**
```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("data/Datasetv4/v46/processed/2023/m5_dataset_v46_2023.csv", nrows=1)
assert 'bars_to_next_pivot' not in df.columns, "LEAKAGE: FL column in v46 processed output"
print(f"OK — {len(df.columns)} cols, no FL leakage")
EOF
```

**Step 4: Commit**
```bash
git add intelligence/data_pipeline_v46.py
git commit -m "feat(pipeline): add v46 ETL — 63-dataset, lagged strategy signals, norm-on-train-only"
```

---

## PHASE 10 — CondorNet v46 Architecture

> Architecture is defined after the data contract is locked.

### Task 10.1: Build condor_brain_net_v46.py

**Files:**
- Create: `intelligence/condor_brain_net_v46.py`

**Key additions over v43:**
1. `StrategyOutputEncoder` with mask support
2. Input dim = `INPUT_DIM_V46` (causal features only)
3. Gate logit clamp `[-20, 20]` baked in

```python
class StrategyOutputEncoder(nn.Module):
    """
    Encodes 58 strategy signals + 58 masks into d_tf_joint.
    values: [B, T, 58] — lagged rolling expectancy
    masks:  [B, T, 58] — 1 where signal available, 0 where no history
    """
    def __init__(self, n_strategies: int, d_out: int):
        super().__init__()
        self.val_proj  = nn.Linear(n_strategies, d_out)
        self.mask_proj = nn.Linear(n_strategies, d_out)
        nn.init.zeros_(self.val_proj.weight);  nn.init.zeros_(self.val_proj.bias)
        nn.init.zeros_(self.mask_proj.weight); nn.init.zeros_(self.mask_proj.bias)

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Masked weighted projection. values and masks: [B, T, 58]"""
        masked_vals = values * masks  # zero out unavailable signals — mask=0 → no signal contribution
        # val_proj(masked_vals): weighted strategy outcomes
        # mask_proj(masks):      learned availability prior ("this strategy is active/inactive")
        # With all-zero mask and zero-init weights: output = 0 (neutral, no injection)
        return self.val_proj(masked_vals) + self.mask_proj(masks)
```

Gate logit clamp in forward (after gating logit computation):
```python
gate_logits = gate_logits.clamp(-20, 20)
```

**Step 2: Smoke tests**
```python
# tests/test_condornet_v46_shape.py
def test_with_strategy_outputs():
    model = CondorNetV46(input_dim=80, n_strategy_outputs=58)
    B, T = 2, 32
    x      = torch.randn(B, T, 80)
    vals   = torch.randn(B, T, 58)
    masks  = torch.randint(0, 2, (B, T, 58)).float()
    out    = model(x, strategy_values=vals, strategy_masks=masks)
    assert out.entry_signal.shape == (B, 1)
    assert out.exit_signal.shape  == (B, 1)

def test_without_strategy_outputs():
    model = CondorNetV46(input_dim=80, n_strategy_outputs=58)
    out   = model(torch.randn(2, 32, 80))
    assert out.entry_signal.shape == (2, 1)

def test_mask_zeros_suppress_signal():
    """Zero mask must suppress strategy signal contribution regardless of value."""
    model = CondorNetV46(input_dim=80, n_strategy_outputs=58)
    B, T = 1, 8
    x     = torch.randn(B, T, 80)
    vals  = torch.ones(B, T, 58) * 999.0   # extreme values
    # all-zero mask: network must produce same output as no strategy input
    mask_zero = torch.zeros(B, T, 58)
    mask_one  = torch.ones(B, T, 58)
    out_zero  = model(x, strategy_values=vals, strategy_masks=mask_zero)
    out_none  = model(x)                   # no strategy inputs at all
    # With all-zero mask the encoder output is self.mask_proj(zeros) = zeros (zero-init)
    # so output must equal no-strategy forward pass
    assert torch.allclose(out_zero.entry_signal, out_none.entry_signal, atol=1e-5), \
        "All-zero mask must produce same output as no strategy input"

def test_gate_logit_clamp():
    """Gate logits must not exceed [-20, 20] after clamp."""
    # Inject extreme weights, verify output remains bounded
    model = CondorNetV46(input_dim=80, n_strategy_outputs=58)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1e6)
    out = model(torch.randn(1, 8, 80))
    # If clamp works, no NaN or Inf in output
    assert not torch.isnan(out.entry_signal).any()
    assert not torch.isinf(out.entry_signal).any()
```

```bash
python3 -m pytest tests/test_condornet_v46_shape.py -v
git add intelligence/condor_brain_net_v46.py tests/test_condornet_v46_shape.py
git commit -m "feat(model): CondorNetV46 with masked StrategyOutputEncoder and gate clamp"
```

---

## PHASE 11 — v46 Training with Ablations

### Task 11.1: Ablation plan — 6 configs

Train each ablation config and record val loss and key metrics. Use 2023–2024 train, 2025 holdout for every run.

| Ablation | Features | Strategy Outputs | Tag |
|---|---|---|---|
| A0 | v43 baseline only | No | `v46_ablation_A0` |
| A1 | v43 + band-break | No | `v46_ablation_A1` |
| A2 | v43 + pivot diagnostics | No | `v46_ablation_A2` |
| A3 | v43 + equilibrium | No | `v46_ablation_A3` |
| A4 | v43 + all new features | No | `v46_ablation_A4` |
| A5 | Full v46 | Yes (58 strategies) | `v46_full` |

**Training command template:**
```bash
python3 intelligence/condor_train_net_v46.py \
    --train-years 2023 2024 \
    --val-year 2025 \
    --epochs 80 \
    --batch-size 32 \
    --lookback 64 \
    --lr 5e-5 \
    --patience 15 \
    --min-delta 0.001 \
    --gate-temp 3.0 \
    --seed 42 \
    --ablation A5 \
    --save models/condornet_v46_full_best.pth \
    2>&1 | tee logs/train_v46_A5.log
```

**After each run, record:**
```
Ablation | Best val loss | exit_bce | entry_bce | Epoch | Improvement vs A0
A0       | ...           | ...      | ...       | ...   | baseline
A1       | ...           | ...      | ...       | ...   | ±...
...
```

**Rule:** Only proceed to A5 (full v46) if A4 shows improvement over A0. If not, investigate before adding strategy outputs.

### Task 11.2: Per-year validation (not just aggregate)

Training loop must emit per-year validation metrics after each epoch:
```
[EPOCH 10] train_loss=1.432 | val_2025=1.891
[PER-YEAR] val_2025_q1=1.81 val_2025_q2=1.90 val_2025_q3=1.95 val_2025_q4=1.88
```

### Task 11.3: Calibration check after training

```bash
python3 - <<'EOF'
import torch, pandas as pd, numpy as np
# Load model and run on val set
# Plot histogram of entry_signal and exit_signal outputs
# Well-calibrated model: entry_signal should be roughly uniformly distributed
# (not collapsed to 0 or 1)
# Flag if P(entry_signal > 0.5) < 0.05 or > 0.95 — likely collapsed gate
EOF
```

---

### Task 11.4: Temporal Permutation Leakage Test

> This is the single most powerful leakage detector in financial ML.
> Any legitimate market signal depends on temporal structure.
> Destroy time ordering — if the model remains predictive, something leaks.

**Principle:**

A causal model learns `y_{t+1} = f(x_t)`. After permuting, `x_t → x_π(t)`, the causal
relationship breaks and performance must collapse to random baseline.

**If permuted performance stays high → you have hidden look-ahead bias.**

This test catches what normal train/test splits miss:
- Improperly lagged indicators
- Strategy outputs containing realized outcomes
- Normalization fitted on full dataset
- Rolling windows with look-ahead
- `bars_to_next_pivot` accidentally present despite FL classification

**Files:**
- Create: `tests/test_temporal_permutation.py`

**Step 1: Write the test (window-level permutation for sequence model)**

```python
"""
Temporal Permutation Leakage Test for CondorNet v46.

IMPORTANT: CondorNet is a sequence model (lookback windows). We must permute
at the WINDOW level, not the bar level. Permuting individual rows would still
allow the model to use within-window temporal structure. We must shuffle
which windows are fed to the model, breaking cross-window temporal order.

Two variants:
  1. Full window permutation — destroys all temporal structure
  2. Daily block permutation — shuffles days, preserves intraday structure

Expected result after permutation:
  entry AUC < 0.55 (random = 0.50)
  exit  AUC < 0.55
  entry Sharpe ≈ 0

If result is AUC > 0.55 on permuted data → STOP. Investigate leakage before proceeding.
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

SEED = 42
PLAN_VERSION = "v46"

# ─── Helpers ────────────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC. Returns 0.5 if only one class present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


def load_windows(processed_path: str, label_col: str = 'exit_signal',
                 lookback: int = 64,
                 feature_cols: list = None) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Load a processed v46 CSV and slice into (N, lookback, n_features) windows.
    Labels are the value at the LAST bar of each window (bar t).
    Returns (X [N, lookback, F], y [N], timestamps [full bar-level Series]).

    `timestamps` is the full bar-level datetime Series before windowing.
    For window i, its terminal bar is at timestamps.iloc[lookback + i].
    Pass `timestamps` to permute_windows_daily_blocks() for timestamp-based
    daily grouping (avoids hard-coded bars_per_day assumption).
    """
    from intelligence.schema_v46 import CAUSAL_FEATURE_COLS_V46
    df = pd.read_csv(processed_path, low_memory=False)
    feature_cols = feature_cols or [c for c in CAUSAL_FEATURE_COLS_V46 if c in df.columns]

    # Also include strategy signal columns if present
    strat_cols = [c for c in df.columns if c.startswith('strat_val_') or c.startswith('strat_mask_')]
    all_feature_cols = feature_cols + strat_cols

    X_raw = df[all_feature_cols].fillna(0.0).values.astype(np.float32)

    # Preserve bar-level timestamps for daily-block grouping
    ts_col = next((c for c in ('timestamp', 'datetime', 'date') if c in df.columns), None)
    timestamps: pd.Series = pd.to_datetime(df[ts_col]) if ts_col else pd.Series(
        pd.date_range("2020-01-01", periods=len(df), freq="5min")
    )

    # Get label: exit_signal from labels CSV (offline label, not in processed CSV)
    label_path = processed_path.replace('_dataset_v46_', '_labels_v46_')
    if os.path.exists(label_path):
        labels = pd.read_csv(label_path)
        if label_col in labels.columns:
            y_raw = labels[label_col].fillna(0).values.astype(np.float32)
        else:
            y_raw = np.zeros(len(df), dtype=np.float32)
    else:
        y_raw = np.zeros(len(df), dtype=np.float32)

    # Slice into windows
    N = len(X_raw) - lookback
    X = np.stack([X_raw[i:i+lookback] for i in range(N)])
    y = y_raw[lookback:]
    return X, y, timestamps


def train_simple_model(X: np.ndarray, y: np.ndarray,
                        n_epochs: int = 5, batch_size: int = 64,
                        seed: int = SEED) -> tuple[float, float]:
    """
    Train a lightweight 1-layer GRU classifier on windows.
    Returns (AUC, final loss).
    We use a simple model (not full CondorNet) to isolate data leakage
    from model capacity effects.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    n_features = X.shape[2]
    model = nn.Sequential(
        # Wrap GRU: input [B, T, F] → last hidden [B, 64] → sigmoid
    )
    # Use a minimal GRU to avoid capacity overfitting on noise
    gru   = nn.GRU(n_features, 64, batch_first=True).to(device)
    head  = nn.Linear(64, 1).to(device)
    opt   = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            _, h = gru(xb)
            logit = head(h.squeeze(0))
            loss  = loss_fn(logit.squeeze(1), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"  epoch {epoch+1}/{n_epochs} loss={total_loss/len(dl):.4f}")

    # Evaluate
    gru.eval(); head.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(X_t, y_t), batch_size=256):
            xb = xb.to(device)
            _, h = gru(xb)
            score = torch.sigmoid(head(h.squeeze(0))).squeeze(1).cpu().numpy()
            all_scores.append(score)
            all_labels.append(yb.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    auc = compute_auc(labels, scores)
    return auc, total_loss / len(dl)


def permute_windows_full(X: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Randomly shuffle all windows. Destroys all temporal structure."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx]


def permute_windows_daily_blocks(X: np.ndarray, timestamps: pd.Series,
                                  lookback: int = 64, seed: int = SEED) -> np.ndarray:
    """
    Shuffle at the daily-block level using calendar dates derived from each window's
    terminal bar timestamp (index lookback + i).  Preserves intraday order within a
    day, destroys cross-day order.  More conservative than full permutation — if this
    still leaks, it is a serious structural issue.

    Args:
        X:          windows array, shape (N, lookback, n_features)
        timestamps: bar-level timestamp Series aligned to the *full* bar index;
                    terminal bar for window i is at position (lookback + i).
        lookback:   sequence length used when building windows (default 64).
        seed:       RNG seed.
    """
    rng = np.random.default_rng(seed)
    N   = len(X)

    # Derive calendar date of each window's terminal bar
    terminal_ts   = pd.to_datetime(timestamps.iloc[lookback : lookback + N].values)
    terminal_dates = terminal_ts.normalize()               # date portion only

    # Group window indices by calendar day
    from collections import defaultdict
    day_groups: dict = defaultdict(list)
    for win_idx, day in enumerate(terminal_dates):
        day_groups[day].append(win_idx)

    # Shuffle the day groups, then flatten to new index order
    day_keys = list(day_groups.keys())
    rng.shuffle(day_keys)
    new_idx = []
    for day in day_keys:
        new_idx.extend(day_groups[day])

    return X[np.array(new_idx)]


# ─── Main Test ──────────────────────────────────────────────────────────────

def run_temporal_permutation_test(
        processed_path: str,
        auc_threshold: float = 0.55,
        n_epochs: int = 5,
        lookback: int = 64,
        seed: int = SEED,
        ablation: str = "A5",
        ablate_block: str | None = None,
):
    """
    Full temporal permutation leakage test.
    PASS criteria: permuted AUC < auc_threshold.
    FAIL: permuted AUC >= auc_threshold → STOP, investigate leakage.
    """
    print("=" * 70)
    print("TEMPORAL PERMUTATION LEAKAGE TEST")
    print(f"  Dataset: {processed_path}")
    print(f"  AUC threshold (permuted must be below): {auc_threshold}")
    print("=" * 70)

    X, y, timestamps = load_windows(processed_path, lookback=lookback)
    print(f"  Loaded {len(X)} windows, {X.shape[2]} features, "
          f"label rate={y.mean():.3f}")

    # 1. Baseline: train on real data
    print("\n[1/3] Training on REAL temporal data...")
    auc_real, loss_real = train_simple_model(X, y, n_epochs=n_epochs)
    print(f"  Real AUC = {auc_real:.4f}")

    # 2. Full permutation
    print("\n[2/3] Training on FULL PERMUTATION (all windows shuffled)...")
    X_perm_full = permute_windows_full(X)
    auc_perm_full, _ = train_simple_model(X_perm_full, y, n_epochs=n_epochs)
    print(f"  Permuted (full) AUC = {auc_perm_full:.4f}")

    # 3. Block permutation (timestamp-based day grouping — no hardcoded bars_per_day)
    print("\n[3/3] Training on DAILY BLOCK PERMUTATION...")
    X_perm_block = permute_windows_daily_blocks(X, timestamps=timestamps, lookback=lookback)
    auc_perm_block, _ = train_simple_model(X_perm_block, y, n_epochs=n_epochs)
    print(f"  Permuted (block) AUC = {auc_perm_block:.4f}")

    # Label permutation sanity check
    print("\n[sanity] Label permutation (labels shuffled, features intact)...")
    rng    = np.random.default_rng(SEED + 1)
    y_perm = y[rng.permutation(len(y))]
    auc_label_perm, _ = train_simple_model(X, y_perm, n_epochs=n_epochs)
    print(f"  Label-permuted AUC = {auc_label_perm:.4f}")

    # Results table
    print("\n" + "=" * 70)
    print(f"{'Test':<35} {'AUC':>8} {'Status'}")
    print("-" * 70)
    print(f"{'Real temporal data':<35} {auc_real:>8.4f}  (should be > 0.50 if any signal exists)")
    print(f"{'Full window permutation':<35} {auc_perm_full:>8.4f}  "
          f"{'PASS' if auc_perm_full < auc_threshold else '*** FAIL — LEAKAGE SUSPECTED ***'}")
    print(f"{'Daily block permutation':<35} {auc_perm_block:>8.4f}  "
          f"{'PASS' if auc_perm_block < auc_threshold else '*** FAIL — LEAKAGE SUSPECTED ***'}")
    print(f"{'Label permutation (sanity)':<35} {auc_label_perm:>8.4f}  "
          f"{'PASS' if auc_label_perm < auc_threshold else '*** FAIL — CATASTROPHIC ***'}")
    print("=" * 70)

    passed = auc_perm_full < auc_threshold and auc_perm_block < auc_threshold
    if not passed:
        print("\n*** TEMPORAL PERMUTATION TEST FAILED ***")
        print("Performance remains high after destroying time structure.")
        print("This is strong evidence of data leakage. Investigate before proceeding.")
        print("Most likely causes:")
        print("  1. Forward-looking feature accidentally in CAUSAL_FEATURE_COLS_V46")
        print("  2. Strategy signal not properly lagged")
        print("  3. Normalization statistics fitted on full dataset (including 2025)")
        print("  4. Rolling window look-ahead in feature generation")
        print("  5. Label column leaking into feature columns")
    else:
        print("\nTEMPORAL PERMUTATION TEST PASSED.")
        print(f"Real AUC={auc_real:.4f} vs Permuted AUC={auc_perm_full:.4f} — "
              f"temporal structure required for signal ({auc_real - auc_perm_full:.4f} drop).")

    # ── Machine-readable summary for Phase 12 gating ────────────────────────
    summary = {
        # self-describing provenance fields
        "plan_version":    PLAN_VERSION,
        "seed":            seed,
        "ablation":        ablation,
        "dataset_path":    processed_path,
        "timestamp_utc":   datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        # gate fields
        "passed":          passed,
        "auc_real":        round(auc_real,        4),
        "auc_perm_full":   round(auc_perm_full,   4),
        "auc_perm_block":  round(auc_perm_block,  4),
        "auc_label_perm":  round(auc_label_perm,  4),
        "leakage_gap":     round(auc_real - auc_perm_full, 4),
        "auc_threshold":   auc_threshold,
        "ablate_block":    ablate_block,
    }
    summary_dir = Path("reports/leakage_test")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "temporal_permutation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written → {summary_path}")

    assert passed, (
        f"Temporal permutation test FAILED: "
        f"full_perm AUC={auc_perm_full:.4f}, "
        f"block_perm AUC={auc_perm_block:.4f} — "
        f"threshold={auc_threshold}"
    )
    return summary


# ─── Block ablation support ──────────────────────────────────────────────────

# Maps ablation block name → column prefixes / names to KEEP (others dropped)
ABLATION_BLOCK_KEEP = {
    "bandbreak":        ["UpperBandOvershootATR","LowerBandOvershootATR",
                         "UpperTailRatio","LowerTailRatio",
                         "UpperCloseBackInsideBandFlag","LowerCloseBackInsideBandFlag",
                         "BearBreakPressure_10","BullBreakPressure_10","NetReversalPressure_10",
                         "bars_since_band_break"],
    "pivot_diag":       ["bars_since_psar_flip","PivotHigh","PivotLow","Slope",
                         "PivotResidual","PivotResidualZ","PivotCurvatureProxy",
                         "PivotSegmentLengthBars","PivotSegmentLengthMinutes",
                         "PivotSegmentResidualStd","PivotSegmentVolatility"],
    "equilibrium":      ["eq_price","eq_distance_pct","gamma_net","gamma_flip",
                         "zone_tight_upper","zone_tight_lower",
                         "zone_full_upper","zone_full_lower","pinning_bias"],
    "strategy_outputs": [],   # matched by prefix strat_val_ / strat_mask_
    "ps_state":         ["ps_pnl_pct","ps_credit_norm","ps_bars_held","ps_dte_frac",
                         "ps_delta_exp","ps_gamma_exp","ps_theta_pos",
                         "ps_iv_change","ps_high_water","ps_mae","ps_unrealized_norm"],
}

def drop_block_columns(X: np.ndarray, feature_cols: list, block: str) -> tuple[np.ndarray, list]:
    """Return X and feature_cols with the named block's columns zeroed out (not dropped,
    to preserve tensor shape). Zeroing preserves model structure while isolating signal."""
    keep = ABLATION_BLOCK_KEEP.get(block, [])
    if block == "strategy_outputs":
        zero_mask = [i for i, c in enumerate(feature_cols)
                     if c.startswith("strat_val_") or c.startswith("strat_mask_")]
    else:
        zero_mask = [i for i, c in enumerate(feature_cols) if c in keep]
    if not zero_mask:
        print(f"  [WARN] No columns found for block '{block}'")
        return X, feature_cols
    X_ablated = X.copy()
    X_ablated[:, :, zero_mask] = 0.0
    return X_ablated, feature_cols


def run_block_ablation_diagnosis(processed_path: str, threshold: float = 0.55,
                                  n_epochs: int = 5, lookback: int = 64):
    """
    Run temporal permutation test with each feature block zeroed out in turn.
    The block whose removal drops permuted AUC below threshold is the leakage source.
    """
    from intelligence.schema_v46 import CAUSAL_FEATURE_COLS_V46
    df = pd.read_csv(processed_path, low_memory=False)
    strat_cols   = [c for c in df.columns if c.startswith("strat_val_") or c.startswith("strat_mask_")]
    all_feat_cols = [c for c in CAUSAL_FEATURE_COLS_V46 if c in df.columns] + strat_cols
    X, y = load_windows(processed_path, lookback=lookback, feature_cols=all_feat_cols)

    print("\n" + "=" * 70)
    print("BLOCK ABLATION DIAGNOSIS (temporal permutation)")
    print(f"{'Block':<20} {'Real AUC':>10} {'Perm AUC':>10} {'Status'}")
    print("-" * 70)

    for block in ABLATION_BLOCK_KEEP:
        X_abl, _ = drop_block_columns(X, all_feat_cols, block)
        auc_real, _  = train_simple_model(X_abl,    y, n_epochs=n_epochs)
        X_perm = permute_windows_full(X_abl)
        auc_perm, _  = train_simple_model(X_perm,   y, n_epochs=n_epochs)
        status = "PASS" if auc_perm < threshold else "*** SOURCE OF LEAKAGE ***"
        print(f"  zeroed {block:<16} {auc_real:>10.4f} {auc_perm:>10.4f}  {status}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',     required=True,
                    help='Processed v46 CSV path')
    ap.add_argument('--threshold',   type=float, default=0.55)
    ap.add_argument('--epochs',      type=int,   default=5)
    ap.add_argument('--lookback',    type=int,   default=64)
    ap.add_argument('--seed',        type=int,   default=42)
    ap.add_argument('--ablation',    default='A5',
                    help='Ablation config label for provenance (e.g. A0, A4, A5)')
    ap.add_argument('--ablate-block', dest='ablate_block', default=None,
                    choices=list(ABLATION_BLOCK_KEEP.keys()),
                    help='Run full temporal permutation test with this block zeroed out')
    ap.add_argument('--diagnose',    action='store_true',
                    help='Run block ablation diagnosis across all blocks')
    args = ap.parse_args()

    if args.diagnose:
        run_block_ablation_diagnosis(args.dataset, args.threshold, args.epochs, args.lookback)
    else:
        run_temporal_permutation_test(
            args.dataset, args.threshold, args.epochs, args.lookback,
            seed=args.seed, ablation=args.ablation,
            ablate_block=args.ablate_block,
        )
```

**Step 2: Run after Phase 11 training (on 2023 processed dataset)**
```bash
python3 tests/test_temporal_permutation.py \
    --dataset data/Datasetv4/v46/processed/2023/m5_dataset_v46_2023.csv \
    --threshold 0.55 \
    --epochs 5 \
    2>&1 | tee logs/temporal_permutation_test.log
```

**Expected output pattern:**
```
Real temporal data              AUC = 0.61    (model learned something)
Full window permutation         AUC = 0.51    PASS
Daily block permutation         AUC = 0.52    PASS
Label permutation (sanity)      AUC = 0.50    PASS
TEMPORAL PERMUTATION TEST PASSED.
Real AUC=0.61 vs Permuted AUC=0.51 — temporal structure required for signal (0.10 drop).
```

**Step 3: If test FAILS — diagnosis protocol**

```bash
# Identify which feature block is responsible by ablating
for block in bandbreak pivot_diag equilibrium strategy_outputs ps_state; do
    python3 tests/test_temporal_permutation.py \
        --dataset data/Datasetv4/v46/processed/2023/m5_dataset_v46_2023.csv \
        --ablate-block $block \
        --threshold 0.55 2>&1 | grep "Permuted (full) AUC"
done
```
The block whose removal drops permuted AUC below 0.55 is the leakage source.

**Step 4: Commit**
```bash
git add tests/test_temporal_permutation.py
git commit -m "test(leakage): add temporal permutation leakage test — row + block + label variants"
```

---

---

### Task 11.5: Leave-One-Year-Out Regime Generalization Test

> The temporal permutation test asks: "Is this leakage?"
> This test asks: "Is this durable, or just regime memorization?"
>
> A model can pass permutation testing yet still collapse when one regime year is
> removed from training. That is regime overfitting — learning a shortcut that
> works only because the training mixture includes the right vol regime mix.
>
> For v46 (trained on 2020–2024) this is critical: each year is a structurally
> different market regime. The model must generalize across all five, not merely
> memorize the pooled mixture.

**Objective:**

Run five training folds, holding out one year at a time. Confirm:
1. No single year is load-bearing for model performance
2. Performance degrades gracefully across all regime transitions
3. Strategy-output signals and new feature blocks are regime-stable

**Outputs:**
- `reports/regime_generalization/loyo_summary.csv` — per-fold metrics
- `reports/regime_generalization/loyo_summary.json` — aggregate stats + stability scores
- `logs/regime_loyo_matrix.log` — full training output for each fold

**Pass criteria:**
- No held-out year: entry AUC < 0.53
- No held-out year: exit AUC < 0.53
- Every held-out year: Sharpe ≥ 0, OR Sharpe ≥ that year's non-neural baseline Sharpe
  (the comparator is the held-out year's own baseline strategy, not 2025 — those are separate evaluation problems)
- Regime stability score ≥ 0 for both entry and exit AUC
- Regime fragility gap < 0.10

**Files:**
- Create: `tests/test_regime_generalization_loyo.py`

**Step 1: Write the test**

```python
"""
Leave-One-Year-Out (LOYO) Regime Generalization Test for CondorNet v46.

Two variants:
  1. LOYO matrix: hold out each 2020-2024 year, train on the other four.
     Tests: is any single year load-bearing for predictive power?
  2. Forward-drift: train 2020-2022, val 2023. Train 2020-2023, val 2024.
     Tests: does the model generalize to a future regime it has not seen?

Key statistics:
  regime_fragility_gap = pooled_metric - min(loyo_metrics)
  regime_stability_score = median(loyo_metrics) - 0.5 * std(loyo_metrics)

If any LOYO fold collapses → run feature-block attribution to identify culprit.
"""
import os, sys, json, datetime, subprocess, argparse
import numpy as np
import pandas as pd
from pathlib import Path

PLAN_VERSION = "v46"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_YEARS_ALL = [2020, 2021, 2022, 2023, 2024]

# Pass thresholds
MIN_ENTRY_AUC = 0.53
MIN_EXIT_AUC  = 0.53
MAX_FRAGILITY_GAP = 0.10

# ─── Training runner ──────────────────────────────────────────────────────────

def run_training_fold(train_years: list[int], val_year: int,
                       ablation: str = "A5", seed: int = 42,
                       epochs: int = 40, out_dir: str = None) -> dict:
    """
    Call condor_train_net_v46.py with the specified train/val split.
    Returns metrics dict parsed from the training log.
    """
    tag = f"loyo_holdout{val_year}_train{'_'.join(map(str, train_years))}"
    out_dir = out_dir or f"reports/regime_generalization/folds/{tag}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = f"{out_dir}/train.log"

    cmd = [
        sys.executable, "intelligence/condor_train_net_v46.py",
        "--train-years", *[str(y) for y in train_years],
        "--val-year",    str(val_year),
        "--epochs",      str(epochs),
        "--batch-size",  "32",
        "--lookback",    "64",
        "--lr",          "5e-5",
        "--patience",    "10",
        "--min-delta",   "0.001",
        "--seed",        str(seed),
        "--ablation",    ablation,
        "--save",        f"{out_dir}/best.pth",
    ]
    print(f"\n[LOYO] Running: train={train_years} | val={val_year}")
    print(f"  cmd: {' '.join(cmd)}")

    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        print(f"  [WARN] Training exited with code {result.returncode}")

    metrics = load_metrics_json(out_dir)   # hard error if missing — no silent fallback
    metrics.update({"held_out_year": val_year, "train_years": train_years,
                    "ablation": ablation, "log": log_path, "metrics_json": out_dir})
    return metrics


def load_metrics_json(out_dir: str) -> dict:
    """
    Load structured metrics written by condor_train_net_v46.py at end of training.

    CONTRACT: condor_train_net_v46.py MUST write this file at training end:
      {out_dir}/metrics.json
    with exactly these keys:
      {
        "best_epoch":  int,
        "best_val_loss": float,
        "entry_auc":   float,   # AUC of entry_signal on val set at best epoch
        "exit_auc":    float,   # AUC of exit_signal on val set at best epoch
        "sharpe":      float,   # annualised Sharpe of val set pseudo-trades
        "win_rate":    float,
        "trades":      int
      }

    If this file is missing, training either failed or the training script does not
    implement the output contract. This is a hard error — do NOT silently fall back
    to 0.5/0.0, because that would produce fake LOYO "passes" masking actual failures.
    """
    metrics_path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"metrics.json not found at {metrics_path}. "
            "condor_train_net_v46.py must write a structured metrics.json at end of training. "
            "Add this to the training script:\n"
            "  with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:\n"
            "      json.dump({'best_epoch':..., 'entry_auc':..., 'exit_auc':...,\n"
            "                 'sharpe':..., 'win_rate':..., 'trades':...}, f)"
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    required = ["entry_auc", "exit_auc", "sharpe", "win_rate", "trades"]
    missing  = [k for k in required if k not in metrics]
    if missing:
        raise KeyError(f"metrics.json at {metrics_path} is missing keys: {missing}")
    return metrics


# ─── Statistics ───────────────────────────────────────────────────────────────

def regime_fragility_gap(pooled: float, loyo_list: list[float]) -> float:
    return pooled - min(loyo_list)

def regime_stability_score(loyo_list: list[float], alpha: float = 0.5) -> float:
    arr = np.array(loyo_list)
    return float(np.median(arr) - alpha * np.std(arr))


# ─── LOYO matrix ─────────────────────────────────────────────────────────────

def run_loyo_matrix(ablation: str = "A5", seed: int = 42, epochs: int = 40,
                    pooled_entry_auc: float = None,
                    data_base: str = "data/Datasetv4/v46/processed",
                    ) -> tuple[pd.DataFrame, dict]:
    """
    Primary test: hold out each year 2020-2024 in turn.
    """
    rows = []
    for held_out in TRAIN_YEARS_ALL:
        train_years = [y for y in TRAIN_YEARS_ALL if y != held_out]
        metrics = run_training_fold(train_years, held_out, ablation, seed, epochs)
        rows.append(metrics)
        print(f"  entry_auc={metrics['entry_auc']:.4f}  exit_auc={metrics['exit_auc']:.4f}"
              f"  sharpe={metrics['sharpe']:.3f}  trades={metrics['trades']}")

    df = pd.DataFrame(rows).sort_values("held_out_year")
    out_dir = "reports/regime_generalization"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/loyo_summary.csv", index=False)

    entry_list  = df["entry_auc"].tolist()
    exit_list   = df["exit_auc"].tolist()
    sharpe_list = df["sharpe"].tolist()

    min_entry = round(float(np.min(entry_list)), 4)
    min_exit  = round(float(np.min(exit_list)),  4)
    frag_gap  = (round(regime_fragility_gap(pooled_entry_auc, entry_list), 4)
                 if pooled_entry_auc is not None else None)
    loyo_passed = (
        min_entry >= MIN_ENTRY_AUC
        and min_exit >= MIN_EXIT_AUC
        and (frag_gap is None or frag_gap < MAX_FRAGILITY_GAP)
    )

    summary = {
        # ── Self-describing provenance ─────────────────────────────────────
        "plan_version":         PLAN_VERSION,
        "seed":                 seed,
        "ablation":             ablation,
        "dataset_path":         data_base,
        "timestamp_utc":        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        # ── Phase 12 gate fields (canonical names for structured gating) ──
        "passed":               loyo_passed,
        "entry_auc_min_loyo":   min_entry,      # alias used by Phase 12 gate
        "exit_auc_min_loyo":    min_exit,        # alias used by Phase 12 gate
        "regime_fragility_gap": frag_gap,        # None if pooled_entry_auc not supplied
        # ── Full statistics ────────────────────────────────────────────────
        "variant":              "LOYO",
        "folds":                len(df),
        "entry_auc_median":     round(float(np.median(entry_list)), 4),
        "entry_auc_std":        round(float(np.std(entry_list)),    4),
        "exit_auc_median":      round(float(np.median(exit_list)),  4),
        "exit_auc_std":         round(float(np.std(exit_list)),     4),
        "sharpe_median":        round(float(np.median(sharpe_list)),4),
        "sharpe_std":           round(float(np.std(sharpe_list)),   4),
        "min_entry_auc":        min_entry,       # kept for backwards compat
        "min_exit_auc":         min_exit,        # kept for backwards compat
        "min_sharpe":           round(float(np.min(sharpe_list)),   4),
        "entry_stability_score":round(regime_stability_score(entry_list),  4),
        "exit_stability_score": round(regime_stability_score(exit_list),   4),
        "sharpe_stability_score":round(regime_stability_score(sharpe_list),4),
    }
    if pooled_entry_auc is not None:
        summary["pooled_entry_auc"] = pooled_entry_auc

    with open(f"{out_dir}/loyo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  LOYO summary written → {out_dir}/loyo_summary.json")
    return df, summary


# ─── Forward-drift variant ───────────────────────────────────────────────────

def run_forward_drift_test(ablation: str = "A5", seed: int = 42,
                            epochs: int = 40) -> pd.DataFrame:
    """
    Variant 2: train on earlier years, validate on the immediately following year.
    Tests whether the model generalizes forward in time (regime drift direction).
      fold A: train 2020-2022, val 2023
      fold B: train 2020-2023, val 2024
    These are the strictest folds — closest to real deployment chronology.
    """
    out_dir = "reports/regime_generalization"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    forward_folds = [
        ([2020, 2021, 2022], 2023),
        ([2020, 2021, 2022, 2023], 2024),
    ]
    rows = []
    for train_years, val_year in forward_folds:
        metrics = run_training_fold(train_years, val_year, ablation, seed, epochs)
        rows.append(metrics)
        print(f"  [Forward drift] train={train_years} → val={val_year} | "
              f"entry_auc={metrics['entry_auc']:.4f} sharpe={metrics['sharpe']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/forward_drift_summary.csv", index=False)
    return df


# ─── Feature-block attribution ───────────────────────────────────────────────

def run_loyo_block_attribution(worst_held_out_year: int, seed: int = 42,
                                epochs: int = 20) -> pd.DataFrame:
    """
    If LOYO fails for a specific year, run LOYO for that year under each
    feature ablation to identify which block is regime-overfit.
    """
    out_dir = "reports/regime_generalization"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ablation_blocks = ["A0", "A1", "A2", "A3", "A4", "A5"]
    block_labels    = [
        "v43 baseline only",
        "+ band-break",
        "+ pivot diagnostics",
        "+ equilibrium",
        "+ all CF features",
        "full v46 (+ strategy outputs)",
    ]
    train_years = [y for y in TRAIN_YEARS_ALL if y != worst_held_out_year]
    rows = []
    print(f"\n[LOYO Attribution] Held-out year: {worst_held_out_year}")
    for ablation, label in zip(ablation_blocks, block_labels):
        metrics = run_training_fold(train_years, worst_held_out_year,
                                    ablation=ablation, seed=seed, epochs=epochs)
        metrics["block_label"] = label
        rows.append(metrics)
        flag = ""
        if metrics["entry_auc"] < MIN_ENTRY_AUC:
            flag = "  ← REGIME COLLAPSE"
        print(f"  [{ablation}] {label:<40} entry_auc={metrics['entry_auc']:.4f}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/loyo_attribution_holdout{worst_held_out_year}.csv", index=False)
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation",       default="A5")
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--epochs",         type=int, default=40)
    ap.add_argument("--data-base",      default="data/Datasetv4/v46/processed",
                    help="Root directory of processed v46 datasets (for provenance)")
    ap.add_argument("--pooled-entry-auc", type=float, default=None,
                    help="AUC from pooled 2020-2024 training (for fragility gap calc)")
    ap.add_argument("--forward-drift-only", action="store_true",
                    help="Skip 5-fold LOYO matrix; run only the two forward-drift folds")
    ap.add_argument("--attribution-only-year", type=int, default=None,
                    help="Skip LOYO, run block attribution for this held-out year only")
    args = ap.parse_args()

    if args.attribution_only_year:
        run_loyo_block_attribution(args.attribution_only_year, args.seed, args.epochs)
        return

    if args.forward_drift_only:
        print("\n" + "=" * 70)
        print("FORWARD DRIFT TEST ONLY (chronological folds)")
        print("=" * 70)
        run_forward_drift_test(args.ablation, args.seed, args.epochs)
        return

    # ── Primary LOYO matrix ──
    print("\n" + "=" * 70)
    print("LEAVE-ONE-YEAR-OUT REGIME GENERALIZATION TEST")
    print(f"Ablation: {args.ablation} | Seed: {args.seed} | Epochs: {args.epochs}")
    print("=" * 70)

    df_loyo, summary = run_loyo_matrix(
        ablation=args.ablation, seed=args.seed, epochs=args.epochs,
        pooled_entry_auc=args.pooled_entry_auc,
        data_base=args.data_base)

    # ── Forward-drift test (always runs after full LOYO matrix) ──
    print("\n" + "-" * 70)
    print("FORWARD DRIFT TEST (earlier years → later year)")
    df_fwd = run_forward_drift_test(args.ablation, args.seed, args.epochs)

    # ── Results table ──
    print("\n" + "=" * 70)
    print("LOYO RESULTS MATRIX")
    print("-" * 70)
    print(df_loyo[["held_out_year","entry_auc","exit_auc","sharpe","trades"]].to_string(index=False))
    print("\nSUMMARY STATISTICS")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ── Pass/fail checks ──
    failures = []
    if summary["min_entry_auc"] < MIN_ENTRY_AUC:
        worst_year = int(df_loyo.loc[df_loyo["entry_auc"].idxmin(), "held_out_year"])
        failures.append(
            f"Entry AUC collapse: held_out={worst_year} "
            f"AUC={summary['min_entry_auc']:.4f} < {MIN_ENTRY_AUC}")
    if summary["min_exit_auc"] < MIN_EXIT_AUC:
        worst_year = int(df_loyo.loc[df_loyo["exit_auc"].idxmin(), "held_out_year"])
        failures.append(
            f"Exit AUC collapse: held_out={worst_year} "
            f"AUC={summary['min_exit_auc']:.4f} < {MIN_EXIT_AUC}")
    if args.pooled_entry_auc and summary.get("regime_fragility_gap", 0) > MAX_FRAGILITY_GAP:
        failures.append(
            f"Fragility gap too large: {summary['regime_fragility_gap']:.4f} > {MAX_FRAGILITY_GAP}")

    if failures:
        print("\n*** LOYO TEST FAILED ***")
        for f in failures:
            print(f"  FAIL: {f}")
        print("\nDiagnosis: re-run with --attribution-only-year <worst_year>")
        print("That will show which feature block is regime-overfit.")
        # Raise assertion for CI-style usage
        assert False, "LOYO test failed: " + "; ".join(failures)
    else:
        print("\nLOYO TEST PASSED.")
        print(f"  Entry AUC range: [{summary['min_entry_auc']:.4f}, "
              f"{summary['entry_auc_median'] + summary['entry_auc_std']:.4f}]")
        print(f"  Regime stability score: entry={summary['entry_stability_score']:.4f}  "
              f"exit={summary['exit_stability_score']:.4f}")

if __name__ == "__main__":
    main()
```

**Step 2: Run primary LOYO matrix**

First, record pooled training AUC from Task 11.1 (ablation A5):
```bash
# Extract pooled AUC from full-training log (set --pooled-entry-auc accordingly)
grep "entry_auc" logs/train_v46_A5.log | tail -1
```

Then run LOYO:
```bash
python3 tests/test_regime_generalization_loyo.py \
    --ablation A5 \
    --seed 42 \
    --epochs 40 \
    --pooled-entry-auc 0.612 \
    2>&1 | tee logs/regime_loyo_matrix.log
```

**Step 3: Run forward-drift test only (strictest variant)**
```bash
python3 tests/test_regime_generalization_loyo.py \
    --ablation A5 \
    --seed 42 \
    --epochs 40 \
    --forward-drift-only \
    2>&1 | tee logs/regime_forward_drift.log
```
`--forward-drift-only` skips the full 5-fold LOYO matrix and runs only the two
chronological forward-drift folds (`2020–2022 → 2023` and `2020–2023 → 2024`).
Run this independently if you want the forward-drift result without repeating the full matrix.

Inspect `reports/regime_generalization/forward_drift_summary.csv`. Pay particular attention to:
- `train 2020-2022 → val 2023`: 2022 was high-vol (Fed tightening); 2023 was recovery
- `train 2020-2023 → val 2024`: 2024 was AI-driven bull market

**Step 4: If LOYO fails — run block attribution for worst year**
```bash
# Example: 2024 collapsed
python3 tests/test_regime_generalization_loyo.py \
    --attribution-only-year 2024 \
    --seed 42 \
    --epochs 20 \
    2>&1 | tee logs/loyo_attribution_2024.log
```

Expected output:
```
[A0] v43 baseline only            entry_auc=0.57
[A1] + band-break                 entry_auc=0.58
[A2] + pivot diagnostics          entry_auc=0.57
[A3] + equilibrium                entry_auc=0.53  ← REGIME COLLAPSE
[A4] + all CF features            entry_auc=0.52  ← REGIME COLLAPSE
[A5] full v46 (+ strategy)        entry_auc=0.49  ← REGIME COLLAPSE
```
→ In this example the equilibrium proxy block is regime-overfit to 2024. Investigate and either remove it or re-engineer as a relative-distance feature rather than absolute level.

**Step 5: Commit**
```bash
git add tests/test_regime_generalization_loyo.py
git commit -m "test(regime): add LOYO regime generalization test — 5-fold + forward-drift + block attribution"
```

---

**Expected healthy result:**
```
Held-out  Entry AUC  Exit AUC  Sharpe   Trades
2020      0.58       0.60      0.82     240     pass
2021      0.56       0.58      0.67     198     pass
2022      0.57       0.59      0.74     312     pass
2023      0.55       0.57      0.61     287     pass
2024      0.56       0.58      0.69     274     pass

Regime stability score: entry=0.54  exit=0.57
Regime fragility gap:   0.06 < 0.10 threshold
LOYO TEST PASSED.
```

**Danger sign:**
Any year where Sharpe < 0 or AUC < 0.53 while pooled training AUC was > 0.60
→ that year's regime was providing load-bearing signal in training.

---

## PHASE 12 — 2025 Holdout Backtest

### Task 12.0: Normalization drift check (train distribution vs 2025 holdout)

> Run before the backtest. Detects whether 2025 market regime is within the distribution the model was normalized against.

**Files:**
- Create: `docs/baseline/normalization_drift_report.txt`

```python
# Run on Lightning AI before loading v46 model for inference
import pandas as pd, numpy as np, json

with open("models/norm_stats_v46.json") as f:
    stats = json.load(f)

# Load 2025 holdout
m5_2025 = pd.read_csv("data/Datasetv4/v43/2025/m5_dataset_v43_2025.csv", low_memory=False)

print(f"{'Feature':<35} {'train_mean':>12} {'2025_mean':>12} {'drift_sigma':>12} {'WARNING'}")
print("-" * 80)
warnings = []
for col, s in stats.items():
    if col not in m5_2025.columns: continue
    val_2025 = m5_2025[col].dropna()
    if len(val_2025) < 10: continue
    mean_2025 = val_2025.mean()
    drift_sigma = abs(mean_2025 - s['mean']) / s['std']
    warn = " *** HIGH DRIFT" if drift_sigma > 2.0 else ""
    if drift_sigma > 2.0:
        warnings.append((col, drift_sigma))
    print(f"{col:<35} {s['mean']:>12.4f} {mean_2025:>12.4f} {drift_sigma:>12.2f}{warn}")

print(f"\nFeatures with >2σ drift: {len(warnings)}")
for col, d in sorted(warnings, key=lambda x: -x[1])[:10]:
    print(f"  {col}: {d:.2f}σ")
```

**Interpretation:**
- Drift < 1σ: normal
- 1σ–2σ: expected year-to-year variation, note it
- > 2σ: investigate before trusting model calibration on 2025

**Step: Commit report**
```bash
python3 docs/baseline/check_normalization_drift.py > docs/baseline/normalization_drift_report.txt 2>&1
git add docs/baseline/normalization_drift_report.txt
git commit -m "chore(qa): normalization drift report — train distribution vs 2025 holdout"
```

---

### Task 12.1: Run v46 on 2025

```bash
python3 kaggle/condor_brain_backtest_v45.py \
    --data-path data/Datasetv4/v43/2025/m5_dataset_v43_2025.csv \
    --options-path data/Datasetv4/v43/2025/options_2025_v43.csv \
    --v46-model models/condornet_v46_full_best.pth \
    --use-v46 \
    --seed 42 \
    2>&1 | tee logs/backtest_v46_2025.log
```

### Task 12.2: Three-way comparison table

| Metric | v45 baseline | Best non-neural strategy | v46 full |
|---|---|---|---|
| Total return | | | |
| Max drawdown | | | |
| Sharpe | | | |
| Win rate | | | |
| Trades | | | |
| Avg hold bars | | | |
| Tail loss (worst 5%) | | | |

"Best non-neural strategy" = best iron_condor config from Phase 8 optimization, no model.

**Any regression vs v45 baseline blocks release to Alpaca.**

---

## PHASE 13 — Offline Live-Stack Replay

> Before any Alpaca paper order, validate that the live inference stack produces identical feature vectors and model outputs as the offline ETL for the same historical windows.

### Task 13.1: Write train/live parity harness

**Files:**
- Create: `tests/test_live_parity.py`

```python
"""
Parity harness: offline ETL features vs live rolling computation on same historical window.
Both must produce the same feature vector within numerical tolerance.
"""
import pandas as pd, numpy as np, torch
from intelligence.data_pipeline_v46 import build_feature_df, normalize_df
from intelligence.schema_v46 import CAUSAL_FEATURE_COLS_V46
from live.alpaca_data_bridge import featurize_live_bar  # Phase 14 module

TOLERANCE = 1e-4

def test_feature_parity():
    # Load a 128-bar window from historical 2025 data (holdout — safe to use for testing only)
    df = pd.read_csv("data/Datasetv4/v43/2025/m5_dataset_v43_2025.csv",
                     nrows=200, low_memory=False)

    # Offline ETL path
    import json
    with open("models/norm_stats_v46.json") as f:
        norm_stats = json.load(f)
    offline_df = build_feature_df(df)
    offline_df = normalize_df(offline_df, norm_stats, CAUSAL_FEATURE_COLS_V46)
    offline_vec = offline_df[CAUSAL_FEATURE_COLS_V46].iloc[128].values

    # Live rolling path — uses same 128-bar history
    live_vec = featurize_live_bar(df.iloc[:129], norm_stats)

    diff = np.abs(offline_vec - live_vec)
    max_diff = diff.max()
    assert max_diff < TOLERANCE, \
        f"Parity failure: max diff={max_diff:.6f} at col {CAUSAL_FEATURE_COLS_V46[diff.argmax()]}"
    print(f"Parity OK: max diff={max_diff:.2e}")
```

**Step 2: Run**
```bash
python3 -m pytest tests/test_live_parity.py -v
```
Expected: PASS with max diff < 1e-4.

**Step 3: Commit**
```bash
git add tests/test_live_parity.py
git commit -m "test(live): add train/live feature parity harness"
```

---

### Task 13.2: Historical replay through live inference stack

```bash
python3 live/historical_replay.py \
    --data-path data/Datasetv4/v43/2025/m5_dataset_v43_2025.csv \
    --model models/condornet_v46_full_best.pth \
    --start 2025-01-02 --end 2025-01-31 \
    --compare-to logs/backtest_v46_2025.log \
    2>&1 | tee logs/replay_2025_jan.log
```

**Pass criteria:**
- Entry signal values within 1e-3 of offline backtest for same timestamps
- No NaN or Inf in live output
- Execution decisions match offline decisions for same inputs

**Any discrepancy blocks Alpaca deployment.**

---

## PHASE 14 — Alpaca Paper Trading

### Task 14.1: Alpaca options data schema alignment

**Step 1: Map Alpaca options chain fields → v43 schema**

| Alpaca field | v43 column | Notes |
|---|---|---|
| `greeks.delta` | `delta` | Direct map |
| `greeks.gamma` | `gamma` | Direct map |
| `greeks.theta` | `theta` | Direct map |
| `greeks.vega` | `vega` | Direct map |
| `implied_volatility` | `iv` | Direct map |
| `bid_price` | `bid` | Direct map |
| `ask_price` | `ask` | Direct map |
| `(bid+ask)/2` | `mid` | Computed |
| `strike_price` | `strike` | Direct map |
| `expiration_date` | DTE computed | `(exp_date - today).days` |
| `type` | `option_type` | `'call'` or `'put'` |

**Step 2: Write `live/alpaca_data_bridge.py`** (skeleton to be fleshed out with real Alpaca SDK)

### Task 14.2: Live strategy whitelist enforcement

Only `LIVE_ELIGIBLE_UNIVERSE` from `schema_v46.py` may generate paper orders:
```python
from intelligence.schema_v46 import LIVE_ELIGIBLE_UNIVERSE
if template_id not in LIVE_ELIGIBLE_UNIVERSE:
    print(f"[LIVE] Skipping {template_id} — not in live-eligible whitelist")
    continue
```

### Task 14.3: Circuit breakers (paper trading only)

```python
PAPER_CIRCUIT_BREAKERS = {
    "max_daily_loss_pct": -0.03,      # halt if equity drops >3% in a day
    "max_open_positions": 5,           # never more than 5 paper trades open
    "stale_data_seconds": 60,          # halt if market data is >60s old
    "min_bar_history_bars": 64,        # refuse inference if less than 64 bars of history
}
```

---

## PHASE 15 — Tick Data Integration (Future)

> When full historical tick data (2020–2025) becomes available:

- Aggregate tick → OHLCV bars with real `bid`, `ask`, `spread_mean`, `spread_std` per bar
- Add as 64th training input dataset
- Replace synthetic ATR-based IV proxy with actual realized spread
- Update `MANIFEST.json` total from 63 → 64
- New columns classified as CF in `FEATURE_ROLE_MAP.json`

---

## CRITICAL INVARIANTS (All Phases)

These are checked at every phase boundary. Any violation is a hard stop:

1. **2025 never in training.** `assert 2025 not in train_years`
2. **58 strategies exactly.** `assert len(STRATEGY_OUTPUT_NAMES) == 58`
3. **`bars_to_next_pivot` not in encoder input.** Verified by `test_schema_v46.py`
4. **Strategy signals lagged ≥1 bar.** Verified by `test_trajectory_merger.py::test_no_forward_leakage`
5. **OPEN/CLOSE atomicity.** Zero orphaned OPEN events in any trajectory
6. **Gate logit clamp.** `gate_logits.clamp(-20, 20)` present in `condor_brain_net_v46.py`
7. **Normalization fitted on train years only.** `norm_stats_v46.json` never touched after initial fit
8. **Live parity.** `test_live_parity.py` passes before any Alpaca deployment
9. **Min trade count.** No optimization result with fewer than 5 trades is used for ranking
10. **Full console output.** No suppressed logs. Every epoch, every trade, every loss component printed

---

## ARTIFACT BUNDLE (emitted at each phase boundary)

Each major phase must emit to `artifacts/{phase}/`:
```
config.json       — args used for this phase
code_hash.txt     — sha256 of key source files
input_hashes.json — sha256 of all input CSVs
output_hashes.json— sha256 of all output CSVs
metrics.json      — summary metrics for this phase
qa_report.txt     — any warnings or anomalies
```

Script template:
```python
# utils/emit_artifact_bundle.py
import hashlib, json, os, datetime

def emit_bundle(phase: str, config: dict, input_files: list,
                output_files: list, metrics: dict, warnings: list):
    out_dir = f"artifacts/phase_{phase}"
    os.makedirs(out_dir, exist_ok=True)
    def sha256(p):
        h = hashlib.sha256()
        with open(p,'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
        return h.hexdigest()
    bundle = {
        "phase": phase, "timestamp": datetime.datetime.utcnow().isoformat(),
        "config": config,
        "input_hashes":  {p: sha256(p) for p in input_files if os.path.exists(p)},
        "output_hashes": {p: sha256(p) for p in output_files if os.path.exists(p)},
        "metrics": metrics, "warnings": warnings,
    }
    out = f"{out_dir}/bundle.json"
    with open(out, 'w') as f: json.dump(bundle, f, indent=2)
    print(f"[ARTIFACT] {out}")
```
