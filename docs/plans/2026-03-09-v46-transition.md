# CondorNet v45 → v46 Transition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transition CondorNet from v45 (single-year 2025 training, 1 Iron Condor strategy) to v46 (5-year 2020–2024 training, 58 strategies, 63 discrete input datasets, new feature columns, live Alpaca demo trading).

**Architecture:** v46 trains on 2020–2024 (5 base market datasets × 5 years = 25 files, plus 58 strategy-output datasets per year = 290 strategy files, total ≤ 315 discrete CSVs depending on GPU budget). 2025 is the permanent holdout — never touches training. The model learns from strategy P&L outcomes across years, infers which decision patterns worked, and generalizes to live trading. v46 adds new feature columns (band-break reversal, pivot diagnostics, cross-TF alignment, payout equilibrium), a new multi-dataset fusion encoder in CondorNet, and an Alpaca real-time inference stack.

**Tech Stack:** Python 3.12, PyTorch, pandas, numpy, backtrader (legacy), kaggle/condor_brain_backtest_v45.py (active backtester), Lightning AI T4 GPU, Alpaca API (paper trading).

---

## PHASE 0 — Current-State Validation (v43/2025, test baseline)

> Run before touching ANY new code. Establishes a clean baseline on the existing 2025 dataset.

### Task 0.1: Verify 2025 dataset integrity

**Files:**
- Read: `data/Datasetv4/v43/2025/m5_dataset_v43_2025.csv`
- Read: `data/Datasetv4/v43/2025/options_2025_v43.csv`

**Step 1: Count columns and rows in each 2025 file**

Run on Lightning AI:
```bash
python3 - <<'EOF'
import pandas as pd, os
base = "data/Datasetv4/v43/2025"
for fn in sorted(os.listdir(base)):
    if fn.endswith('.csv'):
        df = pd.read_csv(f"{base}/{fn}", nrows=3)
        nrows = sum(1 for _ in open(f"{base}/{fn}")) - 1
        print(f"{fn}: {nrows} rows x {len(df.columns)} cols")
        print(f"  first ts: {df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'NO TS'}")
        print(f"  last check: cols[-5:] = {list(df.columns[-5:])}")
EOF
```
Expected: m1 ~92k rows, m5 ~18.5k rows, m15 ~6.2k rows, h1 ~1.7k rows, options varies.

**Step 2: Verify all 91+ expected columns present in m5**

```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("data/Datasetv4/v43/2025/m5_dataset_v43_2025.csv", nrows=1)
cols = set(df.columns)
required = {
    'timestamp','open','high','low','close','volume',
    'log_return','ret_z','atr_pct','bb_lower_dyn','bb_upper_dyn',
    'psar_trend','psar_mark','psar_adaptive',
    'PivotHigh','PivotLow','Slope','PivotResidual',
    'ps_pnl_pct','ps_credit_norm','ps_bars_held','ps_dte_frac',
    'exit_signal','strategy_label','pop','ev','max_loss',
    'tod_sin','tod_cos','regime_persistence',
}
missing = required - cols
print(f"Total cols: {len(cols)}")
print(f"Missing required: {sorted(missing)}")
EOF
```
Expected: 0 missing.

**Step 3: Commit nothing — this is read-only validation**

Record output. If any required columns are missing, fix in dataset before proceeding.

---

### Task 0.2: Run v45 backtester smoke test on 2025 data

**Files:**
- Run: `kaggle/condor_brain_backtest_v45.py`

**Step 1: Smoke test with --limit 500 on Lightning AI**

```bash
cd ~/spy-iron-condor-trading
python kaggle/condor_brain_backtest_v45.py \
    --limit 500 \
    --data-path data/Datasetv4/v43/2025/m5_dataset_v43_2025.csv \
    --use-v43 2>&1 | tee logs/smoke_v45_2025.log
```

**Step 2: Verify gate summary is printed**

```bash
grep -E "(entry_gate|pop_gate|ic_gate|atomicity|trades_opened)" logs/smoke_v45_2025.log
```
Expected (approximate): entry_gate_fail ~6%, pop_gate_fail ~35%, ic_gate_fail ~57%, atomicity_fail=0.

**Step 3: Verify no Python exceptions in log**

```bash
grep -E "(Traceback|Error:|Exception)" logs/smoke_v45_2025.log | head -20
```
Expected: 0 lines.

---

## PHASE 1 — 2020–2024 Dataset Completion

> 2020–2024 have m1/m5/m15/h1 but are missing: (a) options files, (b) new v46 feature columns.
> Work in Lightning AI. Local repo tracks the scripts; data files stay in Lightning AI workspace.

### Task 1.1: Audit 2020–2024 column schema vs 2025 baseline

**Files:**
- Create: `data/Datasetv4/v43/scripts/audit_schema_parity.py`

**Step 1: Write audit script**

```python
#!/usr/bin/env python3
"""Audit column parity between 2025 baseline and 2020-2024 datasets."""
import pandas as pd, os, json

BASE = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TFS   = ['m1', 'm5', 'm15', 'h1']

# Load 2025 as reference
ref = {}
for tf in TFS:
    p = f"{BASE}/2025/{tf}_dataset_v43_2025.csv"
    ref[tf] = set(pd.read_csv(p, nrows=1).columns)

report = {}
for year in YEARS[:-1]:  # 2020-2024
    report[year] = {}
    for tf in TFS:
        p = f"{BASE}/{year}/{tf}_dataset_v43_{year}.csv"
        if not os.path.exists(p):
            report[year][tf] = {"status": "MISSING_FILE"}
            continue
        cols = set(pd.read_csv(p, nrows=1).columns)
        missing = sorted(ref[tf] - cols)
        extra   = sorted(cols - ref[tf])
        report[year][tf] = {
            "ncols": len(cols),
            "ref_ncols": len(ref[tf]),
            "missing": missing,
            "extra": extra,
        }

with open("data/Datasetv4/v43/schema_parity_audit.json", "w") as f:
    json.dump(report, f, indent=2)

# Print summary
for year, tfs in report.items():
    for tf, info in tfs.items():
        if info.get("status") == "MISSING_FILE":
            print(f"[{year}][{tf}] MISSING FILE")
        elif info["missing"]:
            print(f"[{year}][{tf}] MISSING {len(info['missing'])} cols: {info['missing'][:5]}...")
        else:
            print(f"[{year}][{tf}] OK ({info['ncols']} cols)")
```

**Step 2: Run audit**
```bash
python3 data/Datasetv4/v43/scripts/audit_schema_parity.py
```
Expected: lists all missing columns per year/timeframe. Save this output — it drives Tasks 1.2–1.4.

**Step 3: Commit script**
```bash
git add data/Datasetv4/v43/scripts/audit_schema_parity.py
git commit -m "feat(data): add schema parity audit script for 2020-2024 vs 2025"
```

---

### Task 1.2: Add band-break reversal feature block to 2020–2024

New columns (9 total):
- `UpperBandOvershootATR`, `LowerBandOvershootATR`
- `UpperTailRatio`, `LowerTailRatio`
- `UpperCloseBackInsideBandFlag`, `LowerCloseBackInsideBandFlag`
- `BearBreakPressure_10`, `BullBreakPressure_10`, `NetReversalPressure_10`

**Files:**
- Create: `data/Datasetv4/v43/scripts/add_bandbreak_features.py`

**Step 1: Write the feature computation script**

```python
#!/usr/bin/env python3
"""
Add band-break reversal features to 2020-2024 v43 datasets.
Computes 9 new columns from existing bb_upper_dyn, bb_lower_dyn, atr_pct, close, high, low.
"""
import pandas as pd
import numpy as np
import os

BASE  = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024]
TFS   = ['m1', 'm5', 'm15', 'h1']

PRESSURE_WINDOW = 10

def compute_bandbreak(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 9 band-break reversal columns. Input df must have:
    bb_upper_dyn, bb_lower_dyn, atr_pct, close, high, low, open."""
    df = df.copy()

    # Guard: require all input cols
    required = ['bb_upper_dyn', 'bb_lower_dyn', 'atr_pct', 'close', 'high', 'low']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    atr_abs = df['atr_pct'] * df['close']
    atr_abs = atr_abs.replace(0, np.nan).fillna(method='ffill').fillna(1e-6)

    # Overshoot in ATR units
    df['UpperBandOvershootATR'] = ((df['high']  - df['bb_upper_dyn']) / atr_abs).clip(lower=0)
    df['LowerBandOvershootATR'] = ((df['bb_lower_dyn'] - df['low'])   / atr_abs).clip(lower=0)

    # Wick-to-body tail ratios (0-1)
    body = (df['close'] - df['open']).abs().replace(0, np.nan)
    upper_wick = df['high']  - df[['open','close']].max(axis=1)
    lower_wick = df[['open','close']].min(axis=1) - df['low']
    df['UpperTailRatio'] = (upper_wick / (upper_wick + body.fillna(upper_wick))).fillna(0).clip(0, 1)
    df['LowerTailRatio'] = (lower_wick / (lower_wick + body.fillna(lower_wick))).fillna(0).clip(0, 1)

    # Close-back-inside flags (candle closed above band but next close back inside)
    broke_upper = df['close'] > df['bb_upper_dyn']
    broke_lower = df['close'] < df['bb_lower_dyn']
    back_inside_upper = (broke_upper.shift(1)) & (df['close'] <= df['bb_upper_dyn'])
    back_inside_lower = (broke_lower.shift(1)) & (df['close'] >= df['bb_lower_dyn'])
    df['UpperCloseBackInsideBandFlag'] = back_inside_upper.astype(float).fillna(0)
    df['LowerCloseBackInsideBandFlag'] = back_inside_lower.astype(float).fillna(0)

    # Rolling pressure (fraction of last N bars that broke the band)
    df['BearBreakPressure_10'] = broke_upper.rolling(PRESSURE_WINDOW, min_periods=1).mean().fillna(0)
    df['BullBreakPressure_10'] = broke_lower.rolling(PRESSURE_WINDOW, min_periods=1).mean().fillna(0)
    df['NetReversalPressure_10'] = df['BullBreakPressure_10'] - df['BearBreakPressure_10']

    return df

BANDBREAK_COLS = [
    'UpperBandOvershootATR', 'LowerBandOvershootATR',
    'UpperTailRatio', 'LowerTailRatio',
    'UpperCloseBackInsideBandFlag', 'LowerCloseBackInsideBandFlag',
    'BearBreakPressure_10', 'BullBreakPressure_10', 'NetReversalPressure_10',
]

for year in YEARS:
    for tf in TFS:
        path = f"{BASE}/{year}/{tf}_dataset_v43_{year}.csv"
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        df = pd.read_csv(path, low_memory=False)
        existing = [c for c in BANDBREAK_COLS if c in df.columns]
        if existing:
            print(f"[SKIP] {year}/{tf} already has {len(existing)} bandbreak cols")
            continue
        try:
            df = compute_bandbreak(df)
            df.to_csv(path, index=False)
            print(f"[OK]   {year}/{tf} → added 9 bandbreak cols ({len(df)} rows)")
        except ValueError as e:
            print(f"[ERR]  {year}/{tf}: {e}")
```

**Step 2: Write test for compute_bandbreak**

Create `tests/test_bandbreak_features.py`:
```python
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
from data.Datasetv4.v43.scripts.add_bandbreak_features import compute_bandbreak, BANDBREAK_COLS

def make_df(n=50):
    np.random.seed(42)
    close = 500 + np.cumsum(np.random.randn(n) * 2)
    high  = close + np.abs(np.random.randn(n))
    low   = close - np.abs(np.random.randn(n))
    bb_u  = close + 5
    bb_l  = close - 5
    return pd.DataFrame({
        'open': close - 0.5, 'high': high, 'low': low, 'close': close,
        'bb_upper_dyn': bb_u, 'bb_lower_dyn': bb_l,
        'atr_pct': np.full(n, 0.005),
    })

def test_all_cols_present():
    df = compute_bandbreak(make_df())
    for c in BANDBREAK_COLS:
        assert c in df.columns, f"Missing: {c}"

def test_no_nans():
    df = compute_bandbreak(make_df())
    for c in BANDBREAK_COLS:
        assert df[c].isna().sum() == 0, f"NaNs in {c}"

def test_pressure_in_range():
    df = compute_bandbreak(make_df())
    assert df['BearBreakPressure_10'].between(0, 1).all()
    assert df['BullBreakPressure_10'].between(0, 1).all()

def test_overshoot_nonnegative():
    df = compute_bandbreak(make_df())
    assert (df['UpperBandOvershootATR'] >= 0).all()
    assert (df['LowerBandOvershootATR'] >= 0).all()

def test_flag_is_binary():
    df = compute_bandbreak(make_df())
    assert set(df['UpperCloseBackInsideBandFlag'].unique()).issubset({0.0, 1.0})
    assert set(df['LowerCloseBackInsideBandFlag'].unique()).issubset({0.0, 1.0})
```

**Step 3: Run tests**
```bash
python3 -m pytest tests/test_bandbreak_features.py -v
```
Expected: 5 PASS.

**Step 4: Run on 2020–2024**
```bash
python3 data/Datasetv4/v43/scripts/add_bandbreak_features.py 2>&1 | tee logs/bandbreak_add.log
grep -E "(OK|ERR|SKIP)" logs/bandbreak_add.log
```
Expected: 20 [OK] lines (5 years × 4 TFs).

**Step 5: Commit**
```bash
git add data/Datasetv4/v43/scripts/add_bandbreak_features.py tests/test_bandbreak_features.py
git commit -m "feat(data): add 9 band-break reversal columns to 2020-2024 v43 datasets"
```

---

### Task 1.3: Add pivot confirmation diagnostics (3 new cols)

New columns: `bars_since_band_break`, `bars_since_psar_flip`, `bars_to_next_pivot`

**Files:**
- Create: `data/Datasetv4/v43/scripts/add_pivot_diagnostics.py`

**Step 1: Write script**

```python
#!/usr/bin/env python3
"""Add pivot confirmation timing diagnostics to 2020-2024 v43 datasets."""
import pandas as pd, numpy as np, os

BASE  = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024]
TFS   = ['m1', 'm5', 'm15', 'h1']
NEW_COLS = ['bars_since_band_break', 'bars_since_psar_flip', 'bars_to_next_pivot']

def compute_pivot_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # bars_since_band_break: bars since last UpperCloseBackInsideBandFlag or LowerCloseBackInsideBandFlag
    if 'UpperCloseBackInsideBandFlag' in df.columns and 'LowerCloseBackInsideBandFlag' in df.columns:
        any_break = (df['UpperCloseBackInsideBandFlag'] + df['LowerCloseBackInsideBandFlag']).clip(0, 1)
    elif 'bb_upper_dyn' in df.columns:
        # fallback: compute from raw price
        broke_u = (df['close'] > df['bb_upper_dyn']).astype(float)
        broke_l = (df['close'] < df['bb_lower_dyn']).astype(float)
        any_break = (broke_u + broke_l).clip(0, 1)
    else:
        any_break = pd.Series(np.nan, index=df.index)

    bars_since = np.full(len(df), np.nan)
    counter = np.nan
    for i in range(len(df)):
        if any_break.iloc[i] == 1.0:
            counter = 0
        elif not np.isnan(counter):
            counter += 1
        bars_since[i] = counter
    df['bars_since_band_break'] = bars_since

    # bars_since_psar_flip: bars since psar_trend changed sign
    if 'psar_trend' in df.columns:
        trend = df['psar_trend'].fillna(0)
        flip = (trend != trend.shift(1)).astype(float)
        flip.iloc[0] = 0
        bsf = np.full(len(df), np.nan)
        cnt = np.nan
        for i in range(len(df)):
            if flip.iloc[i] == 1.0:
                cnt = 0
            elif not np.isnan(cnt):
                cnt += 1
            bsf[i] = cnt
        df['bars_since_psar_flip'] = bsf
    else:
        df['bars_since_psar_flip'] = np.nan

    # bars_to_next_pivot: forward-looking distance to next PivotHigh or PivotLow
    # Uses future data — label only for training datasets (not live)
    if 'PivotHigh' in df.columns and 'PivotLow' in df.columns:
        is_pivot = ((df['PivotHigh'].fillna(0) != 0) | (df['PivotLow'].fillna(0) != 0)).astype(float)
        btn = np.full(len(df), np.nan)
        # reverse scan
        next_dist = np.nan
        for i in range(len(df) - 1, -1, -1):
            if is_pivot.iloc[i] == 1.0:
                next_dist = 0
            elif not np.isnan(next_dist):
                next_dist += 1
            btn[i] = next_dist
        df['bars_to_next_pivot'] = btn
    else:
        df['bars_to_next_pivot'] = np.nan

    return df

for year in YEARS:
    for tf in TFS:
        path = f"{BASE}/{year}/{tf}_dataset_v43_{year}.csv"
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found"); continue
        df = pd.read_csv(path, low_memory=False)
        if all(c in df.columns for c in NEW_COLS):
            print(f"[SKIP] {year}/{tf} already has pivot diagnostics"); continue
        df = compute_pivot_diagnostics(df)
        df.to_csv(path, index=False)
        print(f"[OK]   {year}/{tf} → added {NEW_COLS}")
```

**Step 2: Run**
```bash
python3 data/Datasetv4/v43/scripts/add_pivot_diagnostics.py 2>&1 | tee logs/pivot_diag.log
```

**Step 3: Commit**
```bash
git add data/Datasetv4/v43/scripts/add_pivot_diagnostics.py
git commit -m "feat(data): add bars_since_band_break, bars_since_psar_flip, bars_to_next_pivot to 2020-2024"
```

---

### Task 1.4: Add payout equilibrium features to 2020–2024

From `docs/PAYOUT_EQUILIBRIUM_MODEL.md` — columns:
`eq_price`, `eq_distance_pct`, `gamma_net`, `gamma_flip`, `zone_tight_upper`, `zone_tight_lower`, `zone_full_upper`, `zone_full_lower`, `pinning_bias`

**Files:**
- Create: `data/Datasetv4/v43/scripts/add_payout_equilibrium.py`

**Step 1: Write script (uses Black-Scholes gamma kernel, synthetic OI)**

```python
#!/usr/bin/env python3
"""
Compute payout equilibrium features per bar using synthetic OI model.
Outputs 9 new columns per timeframe for 2020-2024.
See docs/PAYOUT_EQUILIBRIUM_MODEL.md for math.
"""
import pandas as pd, numpy as np, os
from scipy.stats import norm

BASE  = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024]
TFS   = ['m5', 'm15', 'h1']   # m1 is too granular; skip unless memory permits

NEW_COLS = ['eq_price','eq_distance_pct','gamma_net','gamma_flip',
            'zone_tight_upper','zone_tight_lower','zone_full_upper','zone_full_lower','pinning_bias']

def _bsm_gamma(S, K, T, sigma):
    """Black-Scholes gamma (vectorized). T in years."""
    T = np.clip(T, 1e-6, None)
    d1 = (np.log(S/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def compute_eq_features(spot: float, sigma: float, dte_frac: float,
                         n_strikes: int = 30) -> dict:
    """Compute equilibrium features for a single bar."""
    T = max(dte_frac, 1/252)
    sigma = max(sigma, 0.05)

    pct_range = 0.10
    K_min = spot * (1 - pct_range)
    K_max = spot * (1 + pct_range)
    strikes = np.linspace(K_min, K_max, n_strikes)

    # Synthetic OI: distance-weighted decay
    d = np.abs(strikes - spot) / spot
    f = 1 / (1 + 8*d)
    g = 0.8 ** (15*d)
    oi_call = f * g
    oi_put  = f * g * 1.05  # slight put skew

    # Payout functional P(x)
    x_grid = strikes.copy()
    P = np.array([
        np.sum(oi_call * np.maximum(strikes - x, 0) +
               oi_put  * np.maximum(x - strikes, 0))
        for x in x_grid
    ])

    eq_idx   = np.argmin(P)
    eq_price = x_grid[eq_idx]
    eq_dist  = (eq_price - spot) / spot * 100

    # Gamma exposure per strike
    gamma    = _bsm_gamma(spot, strikes, T, sigma)
    gex      = (oi_call - oi_put) * gamma
    gamma_net  = float(np.sum(gex))
    # Gamma flip: strike where gex crosses zero
    gex_sign_changes = np.where(np.diff(np.sign(gex)))[0]
    gamma_flip = float(strikes[gex_sign_changes[0]]) if len(gex_sign_changes) else spot

    # Band (modeling band)
    B = spot * 0.005 * np.sqrt(T) * 0.25   # compressed 0.25x
    zone_tight_upper = eq_price + B
    zone_tight_lower = eq_price - B
    zone_full_upper  = eq_price + 3*B
    zone_full_lower  = eq_price - 3*B

    pin_dist = abs(eq_dist)
    if pin_dist < 2.0:   pinning_bias = 2.0
    elif pin_dist < 5.0: pinning_bias = 1.0
    else:                pinning_bias = 0.0

    return {
        'eq_price': eq_price, 'eq_distance_pct': eq_dist,
        'gamma_net': gamma_net, 'gamma_flip': gamma_flip,
        'zone_tight_upper': zone_tight_upper, 'zone_tight_lower': zone_tight_lower,
        'zone_full_upper': zone_full_upper,   'zone_full_lower': zone_full_lower,
        'pinning_bias': pinning_bias,
    }

def process_tf(path: str):
    df = pd.read_csv(path, low_memory=False)
    if all(c in df.columns for c in NEW_COLS):
        print(f"[SKIP] {path} already has eq cols"); return

    # Estimate IV from ATR (rough proxy when options chain unavailable)
    sigma_col = df.get('iv_rank', df.get('atr_pct', pd.Series(np.full(len(df), 0.20))))
    if hasattr(sigma_col, 'values'):
        sigma_arr = sigma_col.fillna(0.20).clip(0.05, 2.0).values
    else:
        sigma_arr = np.full(len(df), 0.20)

    dte_frac_col = df.get('ps_dte_frac', pd.Series(np.full(len(df), 21/252)))
    if hasattr(dte_frac_col, 'values'):
        dte_arr = dte_frac_col.fillna(21/252).clip(1e-6, 1.0).values
    else:
        dte_arr = np.full(len(df), 21/252)

    results = []
    spot_arr = df['close'].values
    for i in range(len(df)):
        results.append(compute_eq_features(spot_arr[i], sigma_arr[i], dte_arr[i]))

    for col in NEW_COLS:
        df[col] = [r[col] for r in results]

    df.to_csv(path, index=False)
    print(f"[OK]   {path} → added {len(NEW_COLS)} eq cols")

for year in YEARS:
    for tf in TFS:
        path = f"{BASE}/{year}/{tf}_dataset_v43_{year}.csv"
        if os.path.exists(path):
            process_tf(path)
        else:
            print(f"[SKIP] {path} not found")
```

**Step 2: Run (slow — ~10 min per year on CPU)**
```bash
python3 data/Datasetv4/v43/scripts/add_payout_equilibrium.py 2>&1 | tee logs/eq_features.log
```

**Step 3: Commit**
```bash
git add data/Datasetv4/v43/scripts/add_payout_equilibrium.py
git commit -m "feat(data): add 9 payout equilibrium features to 2020-2024 m5/m15/h1 datasets"
```

---

### Task 1.5: ✅ ALREADY DONE — Real historical options data confirmed for all 6 years

**IMPORTANT:** These are NOT synthetic Black-Scholes generated files.
They are REAL historical SPY options market data for 2020–2025.

Files confirmed present:
- `data/Datasetv4/v43/2020/options_2020_v43.csv`
- `data/Datasetv4/v43/2021/options_2021_v43.csv`
- `data/Datasetv4/v43/2022/options_2022_v43.csv`
- `data/Datasetv4/v43/2023/options_2023_v43.csv`
- `data/Datasetv4/v43/2024/options_2024_v43.csv`
- `data/Datasetv4/v43/2025/options_2025_v43.csv`

All schema alignment work (Task 1.1 audit) still applies — verify real columns match
exactly what the model expects before proceeding. Do NOT regenerate or overwrite these files.

**Files:**
- Read: `data/Datasetv4/v43/2025/options_2025_v43.csv` (to get schema)
- Create: `data/Datasetv4/v43/scripts/generate_options_per_year.py`

**Step 1: Inspect 2025 options schema**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/Datasetv4/v43/2025/options_2025_v43.csv', nrows=3)
print('Cols:', list(df.columns))
print('Sample row:', df.iloc[0].to_dict())
"
```
Record exact column names before writing generator.

**Step 2: ~~Write generator~~ — NOT NEEDED (real data already present)**

All options files for 2020–2024 are real historical data, already present.
The schema audit in Step 1 is all that is needed here.
Remaining steps in Task 1.5 are skipped.
import pandas as pd, numpy as np, os
from scipy.stats import norm

BASE  = "data/Datasetv4/v43"
YEARS = [2020, 2021, 2022, 2023, 2024]

# Read 2025 schema reference
REF_OPTIONS = pd.read_csv(f"{BASE}/2025/options_2025_v43.csv", nrows=1)
OPTIONS_COLS = list(REF_OPTIONS.columns)  # exact schema

def bsm_price(S, K, T, r, sigma, opt_type='call'):
    T = max(T, 1e-6)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bsm_greeks(S, K, T, r, sigma, opt_type='call'):
    T = max(T, 1e-6)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_base = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    if opt_type == 'call':
        delta = norm.cdf(d1)
        theta = (theta_base - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (theta_base + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def generate_chain_for_bar(timestamp, spot, iv_approx, dte_days, r=0.0):
    """Generate a mini options chain for one bar: 10 strikes, calls + puts."""
    rows = []
    T = dte_days / 365
    for offset_pct in np.arange(-0.10, 0.11, 0.02):
        K = round(spot * (1 + offset_pct), 0)
        for opt_type in ['call', 'put']:
            mid = bsm_price(spot, K, T, r, iv_approx, opt_type)
            spread = max(0.02, mid * 0.05)
            bid = max(0.01, mid - spread/2)
            ask = mid + spread/2
            greeks = bsm_greeks(spot, K, T, r, iv_approx, opt_type)
            rows.append({
                'timestamp': timestamp,
                'strike': K,
                'option_type': opt_type,
                'dte': dte_days,
                'iv': iv_approx,
                'bid': round(bid, 4),
                'ask': round(ask, 4),
                'mid': round(mid, 4),
                'delta': round(greeks['delta'], 6),
                'gamma': round(greeks['gamma'], 6),
                'theta': round(greeks['theta'], 6),
                'vega':  round(greeks['vega'],  6),
                'spot':  spot,
            })
    return rows

for year in YEARS:
    out_path = f"{BASE}/{year}/options_{year}_v43.csv"
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists"); continue
    m5_path = f"{BASE}/{year}/m5_dataset_v43_{year}.csv"
    if not os.path.exists(m5_path):
        print(f"[ERR]  {m5_path} not found"); continue

    m5 = pd.read_csv(m5_path, low_memory=False)
    print(f"[GEN]  {year}: {len(m5)} bars → generating options chain...")

    all_rows = []
    iv_col = m5.get('atr_pct', pd.Series(np.full(len(m5), 0.18)))
    for i, row in m5.iterrows():
        spot = row['close']
        iv   = max(0.10, float(iv_col.iloc[i]) * 40) if 'atr_pct' in m5.columns else 0.18
        dte  = 21  # default 21-DTE chain
        all_rows.extend(generate_chain_for_bar(row['timestamp'], spot, iv, dte))
        if i % 1000 == 0:
            print(f"  {i}/{len(m5)}...")

    df_opts = pd.DataFrame(all_rows)
    # Reorder to match 2025 schema where possible
    for col in OPTIONS_COLS:
        if col not in df_opts.columns:
            df_opts[col] = np.nan
    df_opts = df_opts[[c for c in OPTIONS_COLS if c in df_opts.columns]]
    df_opts.to_csv(out_path, index=False)
    print(f"[OK]   {out_path}: {len(df_opts)} rows")
```

**Step 3: Run**
```bash
python3 data/Datasetv4/v43/scripts/generate_options_per_year.py 2>&1 | tee logs/options_gen.log
```

**Step 4: Verify output**
```bash
python3 -c "
import pandas as pd, os
for year in [2020,2021,2022,2023,2024]:
    p = f'data/Datasetv4/v43/{year}/options_{year}_v43.csv'
    if os.path.exists(p):
        df = pd.read_csv(p, nrows=3)
        n = sum(1 for _ in open(p)) - 1
        print(f'{year}: {n} rows, {len(df.columns)} cols, bid range: {df.bid.min():.2f}-{df.bid.max():.2f}')
    else:
        print(f'{year}: MISSING')
"
```

**Step 5: Commit**
```bash
git add data/Datasetv4/v43/scripts/generate_options_per_year.py
git commit -m "feat(data): generate synthetic options chain CSVs for 2020-2024"
```

---

### Task 1.6: Final schema parity validation

**Step 1: Re-run audit**
```bash
python3 data/Datasetv4/v43/scripts/audit_schema_parity.py
```
Expected: all [OK] — no missing columns in 2020–2024 vs 2025 baseline.

**Step 2: Cross-check row counts**
```bash
python3 - <<'EOF'
import pandas as pd, os
BASE = "data/Datasetv4/v43"
for year in [2020,2021,2022,2023,2024,2025]:
    for tf in ['m1','m5','m15','h1']:
        p = f"{BASE}/{year}/{tf}_dataset_v43_{year}.csv"
        if os.path.exists(p):
            n = sum(1 for _ in open(p)) - 1
            print(f"  {year}/{tf}: {n:>7} rows")
EOF
```

**Step 3: Commit validation log**
```bash
git add data/Datasetv4/v43/schema_parity_audit.json
git commit -m "data: schema parity audit complete — 2020-2024 aligned to 2025 v43 schema"
```

---

## PHASE 2 — v45 Backtester Completion

> v45 backtester must be feature-complete before strategy optimization runs. Completes BACKTEST_V45_EVOLUTION_PLAN.md Phases 2, 3, 4, 6, 7.

### Task 2.1: StrategyRegistry + OptionsStrategy base interface

**Files:**
- Create: `kaggle/core/strategy_registry.py`

**Step 1: Write**

```python
"""Strategy registry — maps strategy name to class and config."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Type
from dataclasses import dataclass
import pandas as pd

@dataclass
class StrategyMeta:
    name:        str
    cls:         type
    legs:        int        # number of legs
    directional: bool       # True = directional bias required
    defined_risk: bool      # True = max loss is known at entry

class OptionsStrategy:
    """Base interface every strategy must implement."""
    name: str = "base"

    def build_structure(self, chain: pd.DataFrame, params: dict, spot: float) -> Optional[dict]:
        """Return leg dict {call_short, call_long, put_short, put_long, ...} or None if invalid."""
        raise NotImplementedError

    def validate_structure(self, legs: dict) -> bool:
        """Return True if all legs have valid bid/ask/mid."""
        raise NotImplementedError

    def estimate_risk(self, legs: dict, credit: float) -> dict:
        """Return {'margin': float, 'max_loss': float, 'pop': float}."""
        raise NotImplementedError

class StrategyRegistry:
    def __init__(self):
        self._registry: Dict[str, StrategyMeta] = {}

    def register(self, name: str, cls: type, legs: int = 4,
                 directional: bool = False, defined_risk: bool = True):
        self._registry[name] = StrategyMeta(name, cls, legs, directional, defined_risk)

    def get(self, name: str) -> Optional[StrategyMeta]:
        return self._registry.get(name)

    def all(self) -> List[Tuple[str, StrategyMeta]]:
        return list(self._registry.items())

    def names(self) -> List[str]:
        return list(self._registry.keys())
```

**Step 2: Write test**

```python
# tests/test_strategy_registry.py
from kaggle.core.strategy_registry import StrategyRegistry, OptionsStrategy

def test_register_and_get():
    reg = StrategyRegistry()
    class MyStrat(OptionsStrategy):
        name = "test"
    reg.register("test", MyStrat, legs=4)
    meta = reg.get("test")
    assert meta is not None
    assert meta.name == "test"
    assert meta.legs == 4

def test_all_returns_all():
    reg = StrategyRegistry()
    class A(OptionsStrategy): name = "a"
    class B(OptionsStrategy): name = "b"
    reg.register("a", A); reg.register("b", B)
    assert len(reg.all()) == 2

def test_get_missing_returns_none():
    reg = StrategyRegistry()
    assert reg.get("nonexistent") is None
```

**Step 3: Run**
```bash
python3 -m pytest tests/test_strategy_registry.py -v
```

**Step 4: Commit**
```bash
git add kaggle/core/strategy_registry.py tests/test_strategy_registry.py
git commit -m "feat(registry): add StrategyRegistry + OptionsStrategy base interface"
```

---

### Task 2.2: StrategyParameterGrid

**Files:**
- Create: `kaggle/core/param_grid.py`

**Step 1: Write**

```python
"""Cartesian parameter grid generator for strategy optimization."""
import itertools
from typing import Any, Dict, Generator, List

class StrategyParameterGrid:
    """Generates all combinations from a dict of param → list of values."""
    def __init__(self, config: Dict[str, List[Any]]):
        self.config = config
        self._keys   = sorted(config.keys())
        self._values = [config[k] for k in self._keys]

    def __len__(self) -> int:
        n = 1
        for v in self._values:
            n *= len(v)
        return n

    def generate(self) -> Generator[Dict[str, Any], None, None]:
        for combo in itertools.product(*self._values):
            yield dict(zip(self._keys, combo))

    def param_hash(self, params: Dict[str, Any]) -> str:
        import hashlib, json
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
```

**Step 2: Test**

```python
# tests/test_param_grid.py
from kaggle.core.param_grid import StrategyParameterGrid

def test_count():
    grid = StrategyParameterGrid({'a': [1,2], 'b': [10,20,30]})
    assert len(grid) == 6

def test_all_combos():
    grid = StrategyParameterGrid({'x': [1,2], 'y': ['a','b']})
    combos = list(grid.generate())
    assert len(combos) == 4
    assert {'x': 1, 'y': 'a'} in combos

def test_hash_is_stable():
    grid = StrategyParameterGrid({'a': [1]})
    p = {'a': 1}
    assert grid.param_hash(p) == grid.param_hash(p)
```

**Step 3: Run + commit**
```bash
python3 -m pytest tests/test_param_grid.py -v
git add kaggle/core/param_grid.py tests/test_param_grid.py
git commit -m "feat(optimizer): add StrategyParameterGrid"
```

---

### Task 2.3: OptimizationRun record + CSV output

**Files:**
- Create: `kaggle/core/optimization_record.py`

**Step 1: Write**

```python
"""OptimizationRun — typed result record for each grid search iteration."""
from dataclasses import dataclass, asdict, field
from typing import Optional
import pandas as pd, os, csv

@dataclass
class OptimizationRun:
    strategy_name:      str
    param_hash:         str
    params:             dict
    total_return:       float
    max_drawdown:       float
    sharpe:             float
    win_rate:           float
    avg_hold_bars:      float
    avg_credit:         float
    capital_efficiency: float
    trades_total:       int
    year:               int = 0
    timeframe:          str = "m5"
    np_dd_ratio:        float = field(init=False)

    def __post_init__(self):
        dd = abs(self.max_drawdown)
        self.np_dd_ratio = self.total_return / dd if dd > 1e-9 else 0.0

class OptimizationLog:
    def __init__(self, output_path: str):
        self.path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._rows = []

    def record(self, run: OptimizationRun):
        self._rows.append(run)
        d = asdict(run)
        # Flatten params dict into columns
        for k, v in d.pop('params', {}).items():
            d[f"param_{k}"] = v
        # Print to console — every iteration, full transparency
        print(f"[OPT] {run.strategy_name} | {run.param_hash} | "
              f"Return={run.total_return:.1f}% | DD={run.max_drawdown:.1f}% | "
              f"Sharpe={run.sharpe:.3f} | WR={run.win_rate:.1%} | "
              f"NP/DD={run.np_dd_ratio:.3f} | trades={run.trades_total}")

    def save(self):
        if not self._rows:
            print("[OPT] No results to save."); return
        rows = []
        for run in self._rows:
            d = asdict(run)
            for k, v in d.pop('params', {}).items():
                d[f"param_{k}"] = v
            rows.append(d)
        df = pd.DataFrame(rows).sort_values('np_dd_ratio', ascending=False)
        df.to_csv(self.path, index=False)
        print(f"[OPT] Saved {len(df)} results → {self.path}")
        print(f"[OPT] Top-1: {df.iloc[0].to_dict()}")
```

**Step 2: Test**

```python
# tests/test_optimization_record.py
from kaggle.core.optimization_record import OptimizationRun, OptimizationLog
import os, tempfile, pandas as pd

def test_np_dd_ratio():
    r = OptimizationRun("ic","abc",{},10.0,-5.0,1.2,0.6,15,0.5,0.4,10)
    assert abs(r.np_dd_ratio - 2.0) < 1e-9

def test_zero_dd():
    r = OptimizationRun("ic","abc",{},10.0,0.0,1.2,0.6,15,0.5,0.4,10)
    assert r.np_dd_ratio == 0.0

def test_save_creates_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out/test.csv")
        log = OptimizationLog(path)
        log.record(OptimizationRun("ic","a",{"dte":21},5.0,-2.5,1.0,0.6,10,0.3,0.3,5))
        log.save()
        df = pd.read_csv(path)
        assert len(df) == 1
        assert 'param_dte' in df.columns
```

**Step 3: Run + commit**
```bash
python3 -m pytest tests/test_optimization_record.py -v
git add kaggle/core/optimization_record.py tests/test_optimization_record.py
git commit -m "feat(optimizer): add OptimizationRun record + OptimizationLog CSV writer"
```

---

### Task 2.4: Port ExitDecisionStack into v45 backtester

**Files:**
- Modify: `kaggle/condor_brain_backtest_v45.py` (search for `# Phase 6` / close-trade block)
- Read: `intelligence/exit_stack.py`

**Step 1: Locate exit decision point in v45**
```bash
grep -n "close_trade\|should_exit\|profit_take\|loss_close\|Phase 6" \
    kaggle/condor_brain_backtest_v45.py | head -30
```

**Step 2: Import ExitDecisionStack at top of v45**

Add after existing imports:
```python
from intelligence.exit_stack import ExitDecisionStack, HardExitRules, CapitalConstraintEngine
```

**Step 3: Instantiate in run_backtest() setup block**

```python
exit_stack = ExitDecisionStack(
    p_exit_threshold=0.70,
    dte_protected_bars=14,
    theta_floor=0.005,
    high_water_floor=0.80,
)
```

**Step 4: Replace inline exit checks with stack call in per-trade update loop**

```python
dec = exit_stack.evaluate(
    credit_received=trade['credit'],
    current_cost=current_cost,
    net_delta=trade.get('net_delta', 0.0),
    dte_remaining=trade.get('dte_remaining', 99),
    ps_theta_pos=trade.get('ps_theta_pos', 0.0),
    ps_high_water=trade.get('ps_high_water', 0.0),
    p_exit=float(bar.get('exit_signal', 0.5)),
    equity_peak=equity_peak,
    equity_now=equity,
    pivot_high=bar.get('PivotHigh', None),
    pivot_low=bar.get('PivotLow', None),
    spot=spot,
)
if dec.should_exit:
    close_trade(trade, reason=dec.reason, bar_idx=i, ts=ts)
```

**Step 5: Smoke test with exit stack active**
```bash
python3 kaggle/condor_brain_backtest_v45.py --limit 500 2>&1 | grep -E "(exit|CLOSE|hard_)"
```

**Step 6: Commit**
```bash
git add kaggle/condor_brain_backtest_v45.py
git commit -m "feat(backtest): port ExitDecisionStack into v45 per-trade update loop"
```

---

### Task 2.5: CLOSE event standardization

Every CLOSE append must have: `trade_id`, `action='CLOSE'`, `bar_idx`, `dt`, `reason`, `pnl`, `pnl_pct`, `held_bars`, `exit_details`.

**Step 1: Find all close appends**
```bash
grep -n "closed_trades.append\|action.*CLOSE" kaggle/condor_brain_backtest_v45.py
```

**Step 2: Audit each one against the required schema — fix any missing fields**

Required CLOSE dict structure:
```python
{
    'action':       'CLOSE',
    'trade_id':     str,      # UUID from OPEN event
    'bar_idx':      int,
    'dt':           str,      # timestamp
    'reason':       str,      # 'hard_max_loss' | 'neural_exit' | 'profit_target' | etc.
    'pnl':          float,    # dollars
    'pnl_pct':      float,    # pct of credit
    'held_bars':    int,
    'credit':       float,
    'exit_details': dict,     # {cost_at_close, spot_at_close}
}
```

**Step 3: Verify no orphaned OPEN rows**
```bash
python3 - <<'EOF'
import json, collections
opens  = collections.defaultdict(int)
closes = collections.defaultdict(int)
with open('bar_trace.jsonl') as f:
    for line in f:
        e = json.loads(line)
        if e.get('action') == 'OPEN':  opens[e['trade_id']] += 1
        if e.get('action') == 'CLOSE': closes[e['trade_id']] += 1
orphans = [tid for tid in opens if opens[tid] != closes.get(tid, 0)]
print(f"Opens: {sum(opens.values())} | Closes: {sum(closes.values())} | Orphans: {len(orphans)}")
for o in orphans[:5]: print(f"  {o}")
EOF
```
Expected: Orphans: 0.

**Step 4: Commit**
```bash
git add kaggle/condor_brain_backtest_v45.py
git commit -m "fix(backtest): standardize all CLOSE event dicts — trade_id join key enforced"
```

---

## PHASE 3 — 58-Strategy Optimization Runs on 2020–2024

> Run each strategy through the backtester optimizer across 2020–2024 datasets. Output: 58 CSV files per year = 290 optimization result files. These also serve as the 58 strategy-output training datasets.

### Task 3.1: Build strategy optimization runner script

**Files:**
- Create: `kaggle/run_strategy_optimization.py`

**Step 1: Write**

```python
#!/usr/bin/env python3
"""
Run all 58 strategies through the v45 backtester optimizer.
For each strategy × year: optimize parameters, record best config, save trajectory CSV.

Output structure:
  reports/optimization/{year}/{strategy_name}/best_params.json
  reports/optimization/{year}/{strategy_name}/opt_results.csv
  data/Datasetv4/v46/strategy_outputs/{year}/{strategy_name}_trajectory.csv
"""
import os, json, sys, argparse, subprocess
from pathlib import Path

STRATEGIES = [
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

assert len(STRATEGIES) == 58, f"Expected 58 strategies, got {len(STRATEGIES)}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years',      nargs='+', type=int, default=[2020,2021,2022,2023,2024])
    ap.add_argument('--strategies', nargs='+', default=STRATEGIES)
    ap.add_argument('--data-base',  default='data/Datasetv4/v43')
    ap.add_argument('--out-base',   default='reports/optimization')
    ap.add_argument('--traj-base',  default='data/Datasetv4/v46/strategy_outputs')
    ap.add_argument('--limit',      type=int, default=0, help='0=full year')
    ap.add_argument('--timeframe',  default='m5')
    args = ap.parse_args()

    total = len(args.years) * len(args.strategies)
    done  = 0

    for year in args.years:
        data_path = f"{args.data_base}/{year}/{args.timeframe}_dataset_v43_{year}.csv"
        opts_path = f"{args.data_base}/{year}/options_{year}_v43.csv"
        if not os.path.exists(data_path):
            print(f"[SKIP] {data_path} not found"); continue

        for strat in args.strategies:
            done += 1
            out_dir  = Path(f"{args.out_base}/{year}/{strat}")
            traj_dir = Path(f"{args.traj_base}/{year}")
            out_dir.mkdir(parents=True, exist_ok=True)
            traj_dir.mkdir(parents=True, exist_ok=True)

            opt_csv  = out_dir / "opt_results.csv"
            traj_csv = traj_dir / f"{strat}_trajectory.csv"

            print(f"\n[{done}/{total}] {year}/{strat}")

            if opt_csv.exists() and traj_csv.exists():
                print(f"  [SKIP] already done"); continue

            cmd = [
                sys.executable, 'kaggle/condor_brain_backtest_v45.py',
                '--data-path',     data_path,
                '--options-path',  opts_path,
                '--strategies',    strat,
                '--optimize',
                '--export-trajectories',
                '--traj-output',   str(traj_csv),
                '--opt-output',    str(opt_csv),
                '--year',          str(year),
            ]
            if args.limit:
                cmd += ['--limit', str(args.limit)]

            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"  [ERR] exit code {result.returncode}")
            else:
                print(f"  [OK]  {opt_csv}")

if __name__ == '__main__':
    main()
```

**Step 2: Run validation pass — single strategy, single year**
```bash
python3 kaggle/run_strategy_optimization.py \
    --years 2025 \
    --strategies iron_condor \
    --limit 500 \
    2>&1 | tee logs/opt_smoke.log
```
Expected: opt_results.csv written, trajectory CSV written, 0 errors.

**Step 3: Commit**
```bash
git add kaggle/run_strategy_optimization.py
git commit -m "feat(optimizer): add 58-strategy batch optimization runner for 2020-2024"
```

---

### Task 3.2: Add --optimize and --export-trajectories flags to v45 backtester

**Files:**
- Modify: `kaggle/condor_brain_backtest_v45.py`

**Step 1: Add CLI args**
```python
ap.add_argument('--optimize',            action='store_true')
ap.add_argument('--export-trajectories', action='store_true')
ap.add_argument('--opt-output',          default=None)
ap.add_argument('--traj-output',         default=None)
ap.add_argument('--strategies',          nargs='+', default=['iron_condor'])
ap.add_argument('--year',                type=int,  default=2025)
```

**Step 2: Wire optimize mode** — calls `StrategyParameterGrid` for the selected strategy, loops through combos, calls `run_backtest()`, records each `OptimizationRun`, saves to `--opt-output`.

**Step 3: Wire trajectory export** — when `--export-trajectories` flag active, `TrajectoryRecorder.save(args.traj_output)` after simulation.

**Step 4: Commit**
```bash
git add kaggle/condor_brain_backtest_v45.py
git commit -m "feat(backtest): add --optimize and --export-trajectories CLI flags to v45"
```

---

### Task 3.3: Run full 58-strategy optimization on 2023–2024 (GPU priority)

> Run 2023 and 2024 first (most recent, highest signal). Add 2020–2022 if GPU budget allows.

**Step 1: Launch on Lightning AI T4**
```bash
python3 kaggle/run_strategy_optimization.py \
    --years 2023 2024 \
    --strategies $(python3 -c "from kaggle.run_strategy_optimization import STRATEGIES; print(' '.join(STRATEGIES))") \
    2>&1 | tee logs/opt_run_2023_2024.log
```

**Step 2: Verify output files**
```bash
find reports/optimization/2023 reports/optimization/2024 -name "opt_results.csv" | wc -l
```
Expected: 116 files (58 strategies × 2 years).

**Step 3: Verify trajectory files**
```bash
find data/Datasetv4/v46/strategy_outputs -name "*_trajectory.csv" | wc -l
```
Expected: 116.

**Step 4: Commit logs (not data)**
```bash
git add logs/opt_run_2023_2024.log
git commit -m "data: 58-strategy optimization complete for 2023-2024"
```

---

## PHASE 4 — v46 Dataset Schema Design

### Task 4.1: Define v46 training dataset manifest

**Files:**
- Create: `data/Datasetv4/v46/MANIFEST.json`

**Step 1: Write manifest**

```json
{
  "version": "v46",
  "description": "63-dataset training manifest for CondorNet v46",
  "train_years": [2020, 2021, 2022, 2023, 2024],
  "holdout_year": 2025,
  "base_datasets": {
    "count": 5,
    "files": ["m1", "m5", "m15", "h1", "options"],
    "per_year": true
  },
  "strategy_output_datasets": {
    "count": 58,
    "source": "data/Datasetv4/v46/strategy_outputs/{year}/{strategy}_trajectory.csv",
    "strategies": "__ALL_58__"
  },
  "tick_dataset": {
    "count": 1,
    "status": "PENDING — awaiting tick data acquisition",
    "notes": "Will replace m1/m5/m15/h1 derivation when available"
  },
  "total_training_inputs": 63,
  "new_feature_cols": [
    "UpperBandOvershootATR", "LowerBandOvershootATR",
    "UpperTailRatio", "LowerTailRatio",
    "UpperCloseBackInsideBandFlag", "LowerCloseBackInsideBandFlag",
    "BearBreakPressure_10", "BullBreakPressure_10", "NetReversalPressure_10",
    "bars_since_band_break", "bars_since_psar_flip", "bars_to_next_pivot",
    "eq_price", "eq_distance_pct", "gamma_net", "gamma_flip",
    "zone_tight_upper", "zone_tight_lower", "zone_full_upper", "zone_full_lower",
    "pinning_bias"
  ]
}
```

**Step 2: Commit**
```bash
git add data/Datasetv4/v46/MANIFEST.json
git commit -m "data: define v46 training manifest — 63 input datasets"
```

---

## PHASE 5 — CondorNet v46 Architecture

### Task 5.1: Update schema_v43.py → schema_v46.py

**Files:**
- Create: `intelligence/schema_v46.py` (copy of v43 + additions)

**Step 1: Copy and extend**

```python
# intelligence/schema_v46.py
"""v46 feature schema — extends v43 with band-break, pivot diagnostics, equilibrium."""
from intelligence.schema_v43 import (
    FEATURE_COLS as FEATURE_COLS_V43,
    TF_LABEL_NAMES,
    STRATEGY_TYPES,
    ABSTAIN_IDX,
    POS_STATE_NAMES,
    N_POS_STATE,
    get_dte_affinity,
)

# New v46 feature columns (appended after v43 features)
BANDBREAK_COLS = [
    'UpperBandOvershootATR', 'LowerBandOvershootATR',
    'UpperTailRatio', 'LowerTailRatio',
    'UpperCloseBackInsideBandFlag', 'LowerCloseBackInsideBandFlag',
    'BearBreakPressure_10', 'BullBreakPressure_10', 'NetReversalPressure_10',
]

PIVOT_DIAG_COLS = [
    'bars_since_band_break', 'bars_since_psar_flip', 'bars_to_next_pivot',
]

EQUILIBRIUM_COLS = [
    'eq_price', 'eq_distance_pct', 'gamma_net', 'gamma_flip',
    'zone_tight_upper', 'zone_tight_lower', 'zone_full_upper', 'zone_full_lower',
    'pinning_bias',
]

NEW_V46_COLS = BANDBREAK_COLS + PIVOT_DIAG_COLS + EQUILIBRIUM_COLS

FEATURE_COLS_V46 = FEATURE_COLS_V43 + NEW_V46_COLS

N_NEW_V46 = len(NEW_V46_COLS)

# Strategy output feature dim (one float per strategy: best NP/DD ratio)
N_STRATEGY_OUTPUTS = 58

VERSION = "v46"
```

**Step 2: Commit**
```bash
git add intelligence/schema_v46.py
git commit -m "feat(schema): add v46 schema — 21 new feature cols + 58 strategy output slots"
```

---

### Task 5.2: Update CondorNet model — add v46 input encoders

**Files:**
- Create: `intelligence/condor_brain_net_v46.py` (copy of v43 + extensions)

Key changes:
1. `StrategyOutputEncoder` — encodes 58-dim strategy output vector
2. Input dim = `len(FEATURE_COLS_V46)` (expanded)
3. Fusion of strategy outputs into `tf_fused` (additive, like PosStateProjector)

**Step 1: Add StrategyOutputEncoder class**

```python
class StrategyOutputEncoder(nn.Module):
    """Encodes 58-strategy output signals into d_tf_joint space."""
    def __init__(self, n_strategies: int, d_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_strategies, d_out),
            nn.Tanh(),
        )
        # Zero-init: starts neutral, learns from training
        nn.init.zeros_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 58] → [B, T, d_out]"""
        return self.proj(x)
```

**Step 2: Wire into forward() — inject after PosStateProjector, before ETD-1**

```python
# In forward():
if strategy_outputs is not None:
    tf_fused = tf_fused + self.strategy_output_enc(strategy_outputs)
```

**Step 3: Update __init__ signature**
```python
def __init__(self, ..., n_strategy_outputs: int = 58):
    ...
    self.strategy_output_enc = StrategyOutputEncoder(n_strategy_outputs, d_tf_joint)
```

**Step 4: Smoke test**
```python
# tests/test_condornet_v46_shape.py
import torch
from intelligence.condor_brain_net_v46 import CondorNetV46

def test_forward_with_strategy_outputs():
    model = CondorNetV46(input_dim=200, n_strategy_outputs=58)
    B, T = 2, 32
    x   = torch.randn(B, T, 200)
    strat = torch.randn(B, T, 58)
    out = model(x, strategy_outputs=strat)
    assert out.entry_signal.shape == (B, 1)
    assert out.exit_signal.shape  == (B, 1)

def test_forward_without_strategy_outputs():
    model = CondorNetV46(input_dim=200, n_strategy_outputs=58)
    x = torch.randn(2, 32, 200)
    out = model(x)  # strategy_outputs=None → skip injection
    assert out.entry_signal.shape == (2, 1)
```

**Step 5: Run + commit**
```bash
python3 -m pytest tests/test_condornet_v46_shape.py -v
git add intelligence/condor_brain_net_v46.py intelligence/schema_v46.py \
        tests/test_condornet_v46_shape.py
git commit -m "feat(model): add CondorNetV46 with StrategyOutputEncoder (58-dim injection)"
```

---

## PHASE 6 — v46 ETL Pipeline

### Task 6.1: Build v46 data pipeline

**Files:**
- Create: `intelligence/data_pipeline_v46.py`

Key changes vs v43 pipeline:
1. Loads `FEATURE_COLS_V46` (includes 21 new cols)
2. Loads strategy output trajectories → merges into per-bar feature vector
3. Outputs v46-format CSVs to `data/Datasetv4/v46/processed/`

**Step 1: Write pipeline skeleton**

```python
#!/usr/bin/env python3
"""v46 ETL pipeline — multi-year, 63-dataset ingestion."""
import argparse, os
import pandas as pd, numpy as np
from intelligence.schema_v46 import FEATURE_COLS_V46, N_STRATEGY_OUTPUTS
from intelligence.schema_v43 import TF_LABEL_NAMES, POS_STATE_NAMES

STRATEGY_NAMES = [...]  # import from run_strategy_optimization.py

def load_strategy_outputs(year: int, strategies: list, traj_base: str) -> pd.DataFrame:
    """Load all 58 strategy trajectory CSVs and merge into a single per-bar features DataFrame.
    Returns DataFrame with timestamp + 58 columns (one per strategy: best NP/DD for that bar)."""
    frames = []
    for s in strategies:
        path = f"{traj_base}/{year}/{s}_trajectory.csv"
        if not os.path.exists(path):
            print(f"[WARN] Missing trajectory: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        if 'timestamp' not in df.columns:
            continue
        # Aggregate per bar: use np_dd_ratio or pnl_pct as the signal
        metric = df.groupby('timestamp')['pnl_pct'].mean().rename(s)
        frames.append(metric)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=1).reset_index()
    return merged

def build_v46_dataset(year: int, timeframe: str, args):
    """Build one v46 processed CSV for a given year/TF."""
    # Load base m5 dataset
    base_path = f"{args.data_base}/{year}/{timeframe}_dataset_v43_{year}.csv"
    df = pd.read_csv(base_path, low_memory=False)
    print(f"[ETL] {year}/{timeframe}: {len(df)} bars, {len(df.columns)} cols base")

    # Load and merge strategy outputs
    strat_df = load_strategy_outputs(year, STRATEGY_NAMES, args.traj_base)
    if not strat_df.empty:
        df = df.merge(strat_df, on='timestamp', how='left')
        print(f"[ETL] After strategy merge: {len(df.columns)} cols")

    # Fill missing strategy columns with 0 (neutral)
    for s in STRATEGY_NAMES:
        if s not in df.columns:
            df[s] = 0.0

    # Validate v46 feature cols present
    v46_cols = FEATURE_COLS_V46
    missing = [c for c in v46_cols if c not in df.columns]
    if missing:
        print(f"[WARN] {len(missing)} v46 feature cols missing: {missing[:5]}...")

    out_path = f"{args.out_base}/{year}/{timeframe}_dataset_v46_{year}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK]  {out_path}: {len(df)} rows x {len(df.columns)} cols")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years',       nargs='+', type=int, default=[2023, 2024])
    ap.add_argument('--timeframes',  nargs='+', default=['m5'])
    ap.add_argument('--data-base',   default='data/Datasetv4/v43')
    ap.add_argument('--traj-base',   default='data/Datasetv4/v46/strategy_outputs')
    ap.add_argument('--out-base',    default='data/Datasetv4/v46/processed')
    args = ap.parse_args()

    for year in args.years:
        for tf in args.timeframes:
            build_v46_dataset(year, tf, args)

if __name__ == '__main__':
    main()
```

**Step 2: Run on 2023–2024**
```bash
python3 intelligence/data_pipeline_v46.py \
    --years 2023 2024 \
    --timeframes m5 m15 h1 \
    2>&1 | tee logs/pipeline_v46.log
```

**Step 3: Commit**
```bash
git add intelligence/data_pipeline_v46.py
git commit -m "feat(pipeline): add v46 ETL — 63-dataset ingestion with strategy output merge"
```

---

## PHASE 7 — v46 Model Training

### Task 7.1: Write v46 training script

**Files:**
- Create: `intelligence/condor_train_net_v46.py`

Key differences from v43:
- Loads `FEATURE_COLS_V46` (extended schema)
- Loads 58-dim strategy output tensor per bar
- `--train-years` / `--val-year 2025` enforced separation
- Full console output: every epoch prints all loss components

**Training command (Lightning AI):**
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
    --logit-var-alpha 0.1 --logit-var-target 2.0 \
    --logit-iqr-alpha 0.08 --logit-iqr-target 1.0 \
    --logit-mad-alpha 0.05 --logit-mad-target 0.9 \
    --save models/condornet_v46_best.pth \
    2>&1 | tee logs/train_v46.log
```

**Step 1: Commit training script**
```bash
git add intelligence/condor_train_net_v46.py
git commit -m "feat(training): add v46 training loop — 2023-2024 train, 2025 holdout"
```

---

### Task 7.2: Monitor training — gate logit saturation guard

Per known issue from MEMORY.md: add clamp on gate logits before training.

In `condor_brain_net_v46.py` forward(), after gate logit computation:
```python
gate_logits = gate_logits.clamp(-20, 20)  # prevent saturation (-90 seen in v43)
```

Commit:
```bash
git add intelligence/condor_brain_net_v46.py
git commit -m "fix(model): clamp gate logits to [-20,20] to prevent saturation"
```

---

## PHASE 8 — v46 Backtest on 2025 (Holdout Validation)

### Task 8.1: Run v46 backtester on 2025

```bash
python3 kaggle/condor_brain_backtest_v45.py \
    --data-path data/Datasetv4/v43/2025/m5_dataset_v43_2025.csv \
    --options-path data/Datasetv4/v43/2025/options_2025_v43.csv \
    --v46-model models/condornet_v46_best.pth \
    --use-v46 \
    2>&1 | tee logs/backtest_v46_2025.log
```

### Task 8.2: Compare v46 vs v45 metrics

```bash
python3 - <<'EOF'
import re

def extract_metrics(log_path):
    metrics = {}
    with open(log_path) as f:
        txt = f.read()
    for key in ['total_return','max_drawdown','sharpe','win_rate','trades']:
        m = re.search(rf'{key}[:\s]+([0-9\.\-]+)', txt, re.IGNORECASE)
        if m: metrics[key] = float(m.group(1))
    return metrics

v45 = extract_metrics('logs/smoke_v45_2025.log')
v46 = extract_metrics('logs/backtest_v46_2025.log')
print("Metric      | v45      | v46")
print("-" * 40)
for k in ['total_return','max_drawdown','sharpe','win_rate']:
    print(f"{k:<12}| {v45.get(k,'N/A'):<8} | {v46.get(k,'N/A')}")
EOF
```

---

## PHASE 9 — Alpaca Live Demo Trading

### Task 9.1: Alpaca options data feed alignment

**Files:**
- Create: `live/alpaca_data_bridge.py`

The Alpaca options data schema must match the `options_YYYY_v43.csv` schema exactly so the model sees the same column structure in live trading as in training.

**Required columns from Alpaca:**
- `timestamp`, `strike`, `option_type`, `dte`, `iv`, `bid`, `ask`, `mid`, `delta`, `gamma`, `theta`, `vega`, `spot`

**Step 1: Write Alpaca polling loop**
```python
"""Poll Alpaca for SPY options chain snapshot and normalize to v43 schema."""
import alpaca_trade_api as tradeapi
import pandas as pd, numpy as np

class AlpacaDataBridge:
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url)

    def get_options_snapshot(self, symbol='SPY') -> pd.DataFrame:
        """Fetch current options chain and return as v43-schema DataFrame."""
        # TODO: use Alpaca Options API endpoint
        # Map Alpaca response fields → v43 column names
        raise NotImplementedError("Wire Alpaca options endpoint here")

    def get_ohlcv_bar(self, symbol='SPY', timeframe='5Min') -> dict:
        """Fetch latest OHLCV bar."""
        bars = self.api.get_bars(symbol, timeframe, limit=1).df
        return bars.iloc[-1].to_dict()
```

**Step 2: Compute all v46 features in real-time from live bar**

```python
from intelligence.features.dynamic_features import compute_all_dynamic_features
from data.Datasetv4.v43.scripts.add_bandbreak_features import compute_bandbreak
from data.Datasetv4.v43.scripts.add_payout_equilibrium import compute_eq_features

def featurize_live_bar(bar_history: pd.DataFrame) -> np.ndarray:
    """Convert live bar history → v46 feature vector (same order as FEATURE_COLS_V46)."""
    df = compute_all_dynamic_features(bar_history)
    df = compute_bandbreak(df)
    # ... compute equilibrium, pivot diagnostics
    from intelligence.schema_v46 import FEATURE_COLS_V46
    return df[FEATURE_COLS_V46].iloc[-1].values
```

**Step 3: Wire into inference loop**
```python
def run_live(model, bridge, device):
    bar_history = pd.DataFrame()
    while True:
        bar    = bridge.get_ohlcv_bar()
        bar_history = bar_history.append(bar, ignore_index=True).tail(128)
        chain  = bridge.get_options_snapshot()
        feat   = featurize_live_bar(bar_history)
        out    = model.forward(torch.tensor(feat).unsqueeze(0).unsqueeze(0).to(device))
        print(f"[LIVE] entry={out.entry_signal.item():.3f} exit={out.exit_signal.item():.3f}")
        # decision logic → place/close orders via Alpaca
        import time; time.sleep(300)  # 5-min bars
```

**Step 4: Commit**
```bash
git add live/alpaca_data_bridge.py
git commit -m "feat(live): add Alpaca data bridge skeleton for live v46 inference"
```

---

## PHASE 10 — Tick Data Integration (Future)

### Task 10.1: Tick data ingestion specification

When tick data (2020–2025) becomes available:

**New dataset:** `data/Datasetv4/v46/tick/SPY_tick_{year}.csv`

**Required columns:** `timestamp`, `price`, `bid`, `ask`, `size`

**Aggregation script to create:** `data/Datasetv4/v46/scripts/aggregate_tick_to_bars.py`

```python
"""Aggregate raw tick data into OHLCV bars with real bid/ask columns."""
# Input: tick CSV with timestamp, price, bid, ask, size
# Output: m1/m5/m15/h1 CSVs with real_bid_open, real_ask_open, real_bid_close, real_ask_close,
#         spread_mean, spread_std per bar

def aggregate(tick_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """freq: '1min', '5min', '15min', '60min'"""
    ohlcv = tick_df.resample(freq, on='timestamp').agg({
        'price': ['first','max','min','last'],
        'size':  'sum',
        'bid':   ['first','last','mean'],
        'ask':   ['first','last','mean'],
    })
    ohlcv.columns = ['open','high','low','close','volume',
                     'real_bid_open','real_bid_close','real_bid_mean',
                     'real_ask_open','real_ask_close','real_ask_mean']
    ohlcv['spread_mean'] = ohlcv['real_ask_mean'] - ohlcv['real_bid_mean']
    return ohlcv.dropna()
```

This adds 1 more dataset slot → total training inputs = **64**.

---

## TASKS CHECKLIST

```
PHASE 0 — Baseline Validation
[ ] 0.1  Verify 2025 dataset integrity (col count, row count, required cols)
[ ] 0.2  Smoke test v45 backtester on 2025 --limit 500

PHASE 1 — 2020-2024 Dataset Completion
[ ] 1.1  Audit schema parity 2020-2024 vs 2025
[ ] 1.2  Add 9 band-break reversal columns to 2020-2024
[ ] 1.3  Add 3 pivot confirmation diagnostic columns
[ ] 1.4  Add 9 payout equilibrium columns to m5/m15/h1
[x] 1.5  ~~Generate synthetic options CSVs~~ — ALREADY DONE (all 5 years present)
[ ] 1.6  Final schema parity validation — all [OK]

PHASE 2 — v45 Backtester Completion
[ ] 2.1  StrategyRegistry + OptionsStrategy base interface
[ ] 2.2  StrategyParameterGrid
[ ] 2.3  OptimizationRun record + OptimizationLog
[ ] 2.4  Port ExitDecisionStack into v45
[ ] 2.5  CLOSE event standardization — trade_id join key

PHASE 3 — 58-Strategy Optimization Runs
[ ] 3.1  Build batch optimization runner script
[ ] 3.2  Add --optimize and --export-trajectories to v45 CLI
[ ] 3.3  Run 58 strategies × 2023-2024 → 116 opt CSVs + 116 trajectory CSVs
[ ] 3.4  (Optional) Extend to 2020-2022 if GPU budget allows

PHASE 4 — v46 Dataset Manifest
[ ] 4.1  Write MANIFEST.json for 63-input training schema

PHASE 5 — CondorNet v46 Architecture
[ ] 5.1  Create schema_v46.py (21 new feature cols + N_STRATEGY_OUTPUTS=58)
[ ] 5.2  Create condor_brain_net_v46.py with StrategyOutputEncoder
[ ] 5.3  Smoke test v46 model shapes (forward pass with/without strategy outputs)

PHASE 6 — v46 ETL Pipeline
[ ] 6.1  Build data_pipeline_v46.py (63-dataset ingestion + merge)
[ ] 6.2  Run pipeline on 2023-2024 → v46 processed CSVs

PHASE 7 — v46 Training
[ ] 7.1  Write condor_train_net_v46.py
[ ] 7.2  Add gate logit clamp [-20,20]
[ ] 7.3  Train on 2023-2024, validate on 2025 → condornet_v46_best.pth

PHASE 8 — v46 Holdout Backtest
[ ] 8.1  Run v46 backtester on 2025 holdout
[ ] 8.2  Compare v45 vs v46 metrics table

PHASE 9 — Alpaca Live Demo
[ ] 9.1  Audit Alpaca options API — confirm field mapping to v43 schema
[ ] 9.2  Write AlpacaDataBridge (options snapshot + OHLCV bar)
[ ] 9.3  Write featurize_live_bar() → v46 feature vector
[ ] 9.4  Wire live inference loop → paper trading decisions

PHASE 10 — Tick Data (Future)
[ ] 10.1 Define tick data ingestion spec + aggregation script stub
[ ] 10.2 Acquire 2020-2025 tick data feed (external — iVolatility / Polygon)
[ ] 10.3 Run aggregation → real bid/ask columns per bar
[ ] 10.4 Add to v46 training as 64th dataset input
```

---

## Critical Constraints (Non-Negotiable)

1. **2025 is holdout only** — never appears in training data, optimization loops, or parameter selection. Only used for validation after training is complete.
2. **58 strategies = 58** — verified by `assert len(STRATEGIES) == 58` in runner script.
3. **Full console output always** — no suppressed logs, no truncated outputs. Every bar, every trade, every loss component printed.
4. **No in-trade partial exits** — 4-leg Iron Condor and all defined-risk strategies must close all legs simultaneously.
5. **Gate logit clamp [-20, 20]** — required in v46 model to prevent saturation seen in v43 runs.
6. **`--min-delta 0.001` patience** — prevents micro-gain patience reset seen in v43 training.
7. **Strategy output datasets are discrete** — each of 58 strategies writes its own trajectory CSV; never merged into a single file.
