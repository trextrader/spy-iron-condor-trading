# Optimizer Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix six diagnosed defects in the Bayesian optimizer that produce identical results across strategy groups, statistically unreliable objectives, and wasted GPU cycles on permanently dead strategies.

**Architecture:** All fixes are targeted surgical edits to four existing files — no new modules needed. Each fix is independent and addresses a specific root cause identified from three optimization sweeps (min / med / med-max). The changes are ordered from lowest-risk / highest-impact to higher-risk / medium-impact.

**Tech Stack:** Python 3.12, PyTorch/BoTorch, numpy — all in `kaggle/` directory.

Status legend:

- `✅` completed
- `🟡` started / in progress
- `[ ]` not started

---

## Diagnosis Summary

From analysis of `kaggle/reports/optimizer_sweep_comparison_all_58_strategies.csv` and the three sweep logs:

| # | Root Cause | Evidence |
|---|-----------|---------|
| 1 | 4 strategies with `class_idx=6` never predicted by v43 → always 0 trades | call/long/short_butterfly: "no results" in every sweep |
| 2 | All strategies use same Sobol `seed=42` → identical-family strategies converge to identical params | med-max: bear_call_ladder, bear_call_spread_credit, bull_call_ladder, inverse_call_broken_wing all get sl=775, pt=850, delta=0.4225 — exact same params |
| 3 | `min_trades=5` is too permissive — 6-10 trade results rank highly and are statistically noise | calendar_call at 24 trades wins over iron_condor at 36 trades in min sweep; short_call jumps 283→48→52 across sweeps |
| 4 | `min_pf` field in `ObjectiveSpec` exists but is **never applied** in `compute()` — dead code | `optimizer_engine.py:69` declares `min_pf=1.0`, `compute()` never references it |
| 5 | Parameter floor sl=300/pt=300 → degenerate configs where near-zero credits prevent meaningful stop-loss | iron_condor med: sl=300, pt=300 (both at exact floor); guts med: pt=300 at floor |
| 6 | GP trains on mixed fast+medium fidelity observations — fast (25% bars) produces systematically higher objectives than medium (60%) for same params → GP surface is biased | Phase 2 GP sees Phase 1 fast-fidelity Y values alongside medium Y values — different scales |

---

## ✅ Task 1: Skip permanently dead strategies in autoall

**Status:** Completed in current repo state. `condor_brain_backtest_v45.py` now skips `_AUTOALL_SKIP_TEMPLATES` before BO startup and records them as `skipped:class_idx=6`.

**Files:**
- Modify: `kaggle/condor_brain_backtest_v45.py:3793` (autoall template setup)

These 4 strategies have `class_idx=6` (butterfly_call), which is never predicted by the v43 model. The early-exit in Phase 1 already catches them, but they still waste 30-60 seconds of startup/teardown per strategy × 4 = ~3 minutes per sweep. More importantly they pollute the summary table with "no results" entries.

**Step 1: Add the skip constant** near the top of the autoall section (find line with `_strat_filter_raw == "autoall"`, around line 3793):

```python
# Strategies permanently dead under v43 (class_idx=6 never predicted).
# Skip entirely in autoall — no BO budget wasted.
_AUTOALL_SKIP_TEMPLATES: frozenset = frozenset({
    "call_broken_wing",
    "long_call_butterfly",
    "short_call_butterfly",
    "short_put_butterfly",
})
```

Place this constant at module level near line 490 of `bayes_optimize_strategy.py`, since the autoall loop is in `condor_brain_backtest_v45.py` which imports from it. Actually easier: place it as a local constant in `condor_brain_backtest_v45.py` right after the `elif _strat_filter_raw == "autoall":` block (around line 3793):

```python
elif _strat_filter_raw == "autoall":
    ALLOWED_STRATEGY_IDXS = None
    ALLOWED_TEMPLATE_IDS  = set(_STRATEGY_CONFIGS.keys())
    # Dead under v43 model (class_idx=6, never predicted): skip entirely.
    _AUTOALL_SKIP_TEMPLATES = frozenset({
        "call_broken_wing",
        "long_call_butterfly",
        "short_call_butterfly",
        "short_put_butterfly",
    })
    print(f"[strategies] autoall mode — {len(ALLOWED_TEMPLATE_IDS)} templates queued for sequential BO")
    print(f"[strategies] autoall skip (class_idx=6): {sorted(_AUTOALL_SKIP_TEMPLATES)}")
```

**Step 2: Skip in the autoall loop** (find the `for _tmpl in sorted(ALLOWED_TEMPLATE_IDS):` loop, around line 4128):

```python
for _tmpl in sorted(ALLOWED_TEMPLATE_IDS):
    # ── Dead strategy: skip before spending any GPU time ──
    if _tmpl in _AUTOALL_SKIP_TEMPLATES:
        print(f"\n[autoall] SKIP [{_ti}/{_n_total}: {_tmpl}] — class_idx=6 never predicted by v43.")
        _autoall_summary.append((_tmpl, None, "skipped:class_idx=6"))
        _ti += 1
        continue
    # ... rest of loop unchanged
```

**Step 3: Update summary display** so "skipped" shows distinctly from "no results" and "error". Find the summary print loop around line 4156:

```python
for _tmpl, _r, _err in _autoall_summary:
    if _err and _err.startswith("skipped:"):
        # Show as skipped row — no obj/net/dd
        _reason = _err.split(":", 1)[1]
        print(f"  {_tmpl:<32}  {'SKIPPED':<8}  {_reason}")
    elif _err:
        print(f"  {_tmpl:<32}  ERROR: {_err}")
    elif _r is None:
        print(f"  {_tmpl:<32}  {'':>8}  {'':>7}  {'':>6}  {'':>7}  no results")
    else:
        # ... existing display code
```

Also exclude skipped strategies from the params table at the bottom (the one starting around line 4174).

**Step 4: Commit**
```bash
git add kaggle/condor_brain_backtest_v45.py
git commit -m "feat(autoall): skip class_idx=6 dead strategies before BO startup"
```

---

## ✅ Task 2: Strategy-specific Sobol seed (fix identical convergence)

**Status:** Completed in current repo state. Implemented with a deterministic `md5(template_id)`-derived seed plus strategy-specific per-round perturbation seeds.

**Files:**
- Modify: `kaggle/bayes_optimize_strategy.py:272` and `:329`

**Root cause:** All strategies use `seed=42` for Sobol initialization, and `seed_rnd = 1000 + rnd` for perturbation. When two strategies share the same simulation family (same class_idx gate + same engine), the identical seed → identical quasi-random exploration → GP converges to identical optima.

**Step 1: Replace hardcoded seed=42** at line 272:

```python
# OLD:
sobol_batch = sobol_candidates(bo_init, space, seed=42)

# NEW — use strategy-specific seed so different strategies explore different regions:
sobol_seed  = abs(hash(template_id)) % 99991
sobol_batch = sobol_candidates(bo_init, space, seed=sobol_seed)
print(f"  [seed] Sobol seed={sobol_seed} (from template_id hash)")
```

**Step 2: Update perturbation seeds** at line 329:

```python
# OLD:
seed_rnd = 1000 + rnd

# NEW — incorporate template hash so perturbations are strategy-specific:
seed_rnd = (abs(hash(template_id)) % 99991) + rnd * 1000
```

**Step 3: Verify fix** — after running, check that the two previously-identical groups in med-max now produce different params:
- bear_call_ladder vs bear_call_spread_credit vs bull_call_ladder vs inverse_call_broken_wing
- call_ratio_backspread vs call_ratio_spread vs long_combo vs long_synthetic_future

These groups will still use the same *simulation* (same class_idx + family), so P&L will be similar, but they'll explore different regions of parameter space → different final params.

**Step 4: Commit**
```bash
git add kaggle/bayes_optimize_strategy.py
git commit -m "fix(bayes): strategy-specific Sobol seed prevents identical-group param convergence"
```

---

## ✅ Task 3: Raise min_trades threshold from 5 to 12

**Status:** Completed in intent and later tightened further. The CLI default in `condor_brain_backtest_v45.py` is now `12`, while the newer guarded optimizer score in `bayes_optimize_strategy.py` currently uses a stricter default of `20`.

**Files:**
- Modify: `kaggle/condor_brain_backtest_v45.py:3721` (CLI default)
- Modify: `kaggle/bayes_optimize_strategy.py:239` (ObjectiveSpec construction)

**Root cause:** With min_trades=5, a strategy that fires 6 trades (purely on lucky early-period data) can score obj=0.5+ and get applied. With only 6 trades, one unlucky trade changes drawdown by ~16%. This creates unstable params across sweeps.

**Context:** Most strategies generate 15-60 trades per year. Single-leg strategies (short_call) generate 50-283. A threshold of 12 strikes the right balance — eliminates noise from tiny samples while keeping strategies that genuinely fire rarely.

**Step 1: Update CLI default** at line 3721 in `condor_brain_backtest_v45.py`:

```python
# OLD:
parser.add_argument("--bo-min-trades", type=int, default=5,
# NEW:
parser.add_argument("--bo-min-trades", type=int, default=12,
```

**Step 2: Update ObjectiveSpec construction** at line 239 in `bayes_optimize_strategy.py`:

```python
# OLD:
obj_spec = ObjectiveSpec(
    min_trades  = getattr(args, 'bo_min_trades',  5),
# NEW:
obj_spec = ObjectiveSpec(
    min_trades  = getattr(args, 'bo_min_trades',  12),
```

**Step 3: Commit**
```bash
git add kaggle/condor_brain_backtest_v45.py kaggle/bayes_optimize_strategy.py
git commit -m "fix(objective): raise min_trades from 5 to 12 to eliminate low-sample noise"
```

---

## ✅ Task 4: Activate profit-factor penalty in ObjectiveSpec (fix dead code)

**Status:** Completed in intent and superseded by the newer score model. The original dead-code issue is resolved: the current guarded objective uses `profit_factor >= 1.1` as an eligibility gate, and the retained `legacy_objective` still applies the PF penalty for audit comparison.

**Files:**
- Modify: `kaggle/optimizer_engine.py:71`

**Root cause:** `ObjectiveSpec` has `min_pf=1.0` declared at line 69, but `compute()` never references `self.min_pf`. This means strategies with profit_factor < 1.0 (losing more than they win in gross terms) receive no penalty. The comment `# min profit factor (soft penalty)` describes intent, not reality.

**Step 1: Add min_pf check to compute()** — replace lines 71-89:

```python
def compute(self, net_pct: np.ndarray, max_dd: np.ndarray,
            total: np.ndarray, pf: np.ndarray) -> np.ndarray:
    """Return objective[K] — higher is better.

    Objective = net_pct / (effective_dd + 2.0)
      - effective_dd = max(max_dd, 1.0)  — floor prevents denominator gaming
        when a candidate has 0% drawdown (all-win streaks get no free lunch)
      - Additional penalties for insufficient trades, extreme drawdown, or
        profit factor below threshold
    """
    # Clamp max_dd to minimum 1.0% so 0-drawdown strategies aren't
    # artificially rewarded with a near-zero denominator
    eff_dd = np.maximum(max_dd, 1.0)
    obj = net_pct / (eff_dd + 2.0)

    # Hard constraint: insufficient trades → big negative penalty
    obj = np.where(total < self.min_trades, -100.0 + net_pct * 0.01, obj)
    # Soft constraint: extreme drawdown cap
    obj = np.where(max_dd > self.max_dd_cap, obj * 0.5, obj)
    # Soft constraint: low profit factor (was declared but never applied before)
    obj = np.where(pf < self.min_pf, obj * 0.7, obj)
    return obj
```

**Step 2: Commit**
```bash
git add kaggle/optimizer_engine.py
git commit -m "fix(objective): activate min_pf penalty that was declared but never applied"
```

---

## ✅ Task 5: Raise parameter floor bounds (prevent degenerate configs)

**Status:** Completed in current repo state. `candidate_codec.py` now uses the raised floor bounds described here for iron butterfly, iron condor, and the generic fallback search space.

**Files:**
- Modify: `kaggle/candidate_codec.py`

**Root cause:** Several multi-leg strategies (iron_condor, inverse_iron_butterfly, etc.) converge to stop_loss_dollar=300 (the floor) and profit_target=300 (the floor). When both are set at 300: any trade that loses $300 exits immediately, and any trade that profits $300 exits immediately. This creates very clean metrics from a tiny number of trades rather than representative full-period performance.

The single-leg strategies (short_call, short_put) have higher floors already (sl=200, pt=100) relative to typical credit — these are fine as-is.

**Step 1: Update `build_iron_butterfly_search_space()`** — raise stop_loss_dollar and profit_target floors:

```python
def build_iron_butterfly_search_space() -> SearchSpaceSpec:
    return SearchSpaceSpec(
        template_id="iron_butterfly",
        params=[
            ParamSpec("stop_loss_dollar", 600,  1500, "grid", 50,  int),   # was 500 → 600
            ParamSpec("profit_target",    500,  2000, "grid", 50,  int),   # was 500, keep
            ParamSpec("max_dte_exit",     0,    7,    "grid", 1,   int),
            ParamSpec("spread_width",     5,    20,   "grid", 1,   int),
            ParamSpec("target_dte",       7,    21,   "grid", 1,   int),
            ParamSpec("short_delta",      0.40, 0.50, "continuous"),
            ParamSpec("hold_days",        5,    21,   "grid", 1,   int),
        ],
    )
```

**Step 2: Update `build_iron_condor_search_space()`**:

```python
def build_iron_condor_search_space() -> SearchSpaceSpec:
    return SearchSpaceSpec(
        template_id="iron_condor",
        params=[
            ParamSpec("stop_loss_dollar", 400,  1500, "grid", 50,  int),   # was 300 → 400
            ParamSpec("profit_target",    400,  2000, "grid", 50,  int),   # was 300 → 400
            ParamSpec("max_dte_exit",     0,    5,    "grid", 1,   int),
            ParamSpec("spread_width",     5,    20,   "grid", 1,   int),
            ParamSpec("target_dte",       7,    21,   "grid", 1,   int),
            ParamSpec("short_delta",      0.15, 0.30, "continuous"),
            ParamSpec("hold_days",        5,    21,   "grid", 1,   int),
        ],
    )
```

**Step 3: Update generic fallback** at end of `build_search_space()`:

```python
# Generic fallback: fine resolution across all exit/structure params.
return SearchSpaceSpec(
    template_id=template_id,
    params=[
        ParamSpec("stop_loss_dollar", 400,  1500, "grid", 50,  int),   # was 300 → 400
        ParamSpec("profit_target",    400,  3000, "grid", 50,  int),   # was 300 → 400
        ParamSpec("target_dte",       7,    28,   "grid", 1,   int),
        ParamSpec("short_delta",      0.15, 0.45, "continuous"),
        ParamSpec("spread_width",     5,    20,   "grid", 1,   int),
        ParamSpec("hold_days",        5,    21,   "grid", 1,   int),
        ParamSpec("max_dte_exit",     0,    5,    "grid", 1,   int),
    ],
)
```

**Step 4: Commit**
```bash
git add kaggle/candidate_codec.py
git commit -m "fix(codec): raise parameter floors to prevent degenerate low-trade configs"
```

---

## ✅ Task 6: Separate fast/medium fidelity in GP observations

**Status:** Completed in current repo state. `bayes_optimize_strategy.py` now tracks `X_obs_fast/Y_obs_fast` separately from `X_obs_med/Y_obs_med`, and the GP trains only on medium-fidelity observations.

**Files:**
- Modify: `kaggle/bayes_optimize_strategy.py:267`

**Root cause:** `X_obs_list` and `Y_obs_list` accumulate ALL observations across Phase 1 (fast, 25% bars) and Phase 2 (medium, 60% bars). For the same parameter vector, fast fidelity typically returns a **higher** objective than medium fidelity because there's less time for drawdown to accumulate. The GP sees these as the same function at different (x, y) pairs — so it fits a noisy surface where the "noise" is systematic fidelity bias. This causes the GP to underestimate the objective of candidates that were only evaluated at medium fidelity.

**Fix:** Keep Phase 1 observations separate from Phase 2. The GP in Phase 2 trains ONLY on medium-fidelity observations. Phase 1 `best_x` is still used as the warm start point.

**Step 1: Split obs lists** at line 267:

```python
# OLD:
X_obs_list: List[np.ndarray] = []   # [D] normalised
Y_obs_list: List[float]      = []   # scalar objective (full fidelity)

# NEW:
X_obs_fast: List[np.ndarray] = []   # [D] from Phase 1 (fast fidelity, 25% bars)
Y_obs_fast: List[float]      = []
X_obs_med:  List[np.ndarray] = []   # [D] from Phase 2 (medium fidelity, 60% bars)
Y_obs_med:  List[float]      = []
```

**Step 2: Update Phase 1 append** (lines 308-309):

```python
# OLD:
X_obs_list.append(x_enc)
Y_obs_list.append(float(res_p1.objective[k]))

# NEW:
X_obs_fast.append(x_enc)
Y_obs_fast.append(float(res_p1.objective[k]))
```

**Step 3: Update best_x initialization** (line 326):

```python
# OLD:
best_x = X_obs_list[int(np.nanargmax(res_p1.objective))]

# NEW:
best_x = X_obs_fast[int(np.nanargmax(res_p1.objective))]
```

**Step 4: Update Phase 2 GP training** (lines 330-333):

```python
# OLD:
if _BOTORCH_OK and len(Y_obs_list) >= 4:
    X_t = torch.tensor(np.array(X_obs_list), dtype=torch.float32)
    Y_t = torch.tensor(Y_obs_list, dtype=torch.float32).unsqueeze(-1)

# NEW — GP trains only on medium-fidelity observations:
if _BOTORCH_OK and len(Y_obs_med) >= 4:
    X_t = torch.tensor(np.array(X_obs_med), dtype=torch.float32)
    Y_t = torch.tensor(Y_obs_med, dtype=torch.float32).unsqueeze(-1)
```

**Step 5: Update Phase 2 append** (lines 374-375):

```python
# OLD:
X_obs_list.append(x_enc)
Y_obs_list.append(float(res_rnd.objective[k]))

# NEW:
X_obs_med.append(x_enc)
Y_obs_med.append(float(res_rnd.objective[k]))
```

**Step 6: Update best_global_idx** (lines 378-379):

```python
# OLD:
best_global_idx = int(np.nanargmax(Y_obs_list))
best_x = X_obs_list[best_global_idx]

# NEW:
if Y_obs_med:
    best_global_idx = int(np.nanargmax(Y_obs_med))
    best_x = X_obs_med[best_global_idx]
else:
    # Fallback: no medium obs yet (first round before GP), use fast best
    best_x = X_obs_fast[int(np.nanargmax(Y_obs_fast))]
```

**Step 7: Commit**
```bash
git add kaggle/bayes_optimize_strategy.py
git commit -m "fix(bayes): separate fast/medium fidelity GP observations to remove fidelity bias"
```

---

## Verification Run

After all 6 tasks are committed, push and run on Lightning AI:

```bash
git push origin main

# Quick sanity check (single strategy, fast):
python kaggle/condor_brain_backtest_v45.py --use-v43 --strategies iron_condor \
  --optimize --optimize-mode bayes --optimize-intensity min --verbose \
  2>&1 | tee bo_fix_verify_iron_condor.log

# Check that:
# 1. No "seed=42" appears in output — verify strategy-specific seed is logged
# 2. objective values reflect min_trades=12 threshold (6-11 trade configs score -100)
# 3. parameter bounds don't hit sl=300 or pt=300 (iron_condor floor now 400)

# Full autoall verification:
python kaggle/condor_brain_backtest_v45.py --use-v43 --strategies autoall \
  --optimize --optimize-mode bayes --optimize-intensity min \
  2>&1 | tee autoall_post_fixes.log

# Check that:
# 4. call_broken_wing/long_call_butterfly/short_call_butterfly/short_put_butterfly
#    appear as "SKIPPED" not "no results"
# 5. Previously-identical groups (bear_call_ladder vs bear_call_spread_credit)
#    now show DIFFERENT params (Sobol seed fix working)
# 6. pf penalty visible: strategies with gross_loss > gross_win get objective × 0.7
```

---

## Expected Impact

| Fix | Expected Improvement |
|-----|---------------------|
| Task 1 (dead skip) | ~3-4 min saved per autoall run; cleaner summary table |
| Task 2 (Sobol seed) | Eliminates identical-param groups; each strategy gets unique exploration path |
| Task 3 (min_trades=12) | Reduces sweep-to-sweep variance; fewer statistically-unreliable results ranked high |
| Task 4 (min_pf penalty) | Strategies with gross_loss > gross_win get 30% obj penalty — prevents theta-positive but gross-loser configs |
| Task 5 (floor bounds) | Eliminates degenerate sl=300/pt=300 configs; optimizer forced into meaningful parameter regions |
| Task 6 (fidelity split) | GP surface more accurate in Phase 2 — better candidates proposed each round |

The combination of Tasks 2 + 5 + 6 should most visibly improve the "identical group" and "stale" patterns in the comparison CSV.
