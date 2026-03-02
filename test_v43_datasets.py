"""
test_v43_datasets.py — Quick validation of m1/m5/m15/h1 dataset integrity.

Checks:
  1. No duplicate columns in any file
  2. m1 / m15 / h1 have identical column sets
  3. m5 has all m1 columns + expected extras (labels + pos_state)
  4. Curvature columns have real data (not all-zero / all-NaN)
  5. Row counts match expected ranges
  6. Spot-check a few label columns on m5

Run on Lightning AI:
  python test_v43_datasets.py
"""
import sys
import pandas as pd
import numpy as np

DATA_DIR = "data/Datasetv4/v43"

FILES = {
    "m1":  f"{DATA_DIR}/m1_dataset_v43_final.csv",
    "m5":  f"{DATA_DIR}/m5_dataset_v43_final.csv",
    "m15": f"{DATA_DIR}/m15_dataset_v43_final.csv",
    "h1":  f"{DATA_DIR}/h1_dataset_v43_final.csv",
}

# Expected M5-only extras (labels + pos_state)
M5_EXTRA_LABELS = ["pop", "max_loss", "ev", "var_95", "cvar_95", "strategy_label", "exit_signal"]
M5_EXTRA_POSSTATE = [
    "ps_pnl_pct", "ps_credit_norm", "ps_bars_held", "ps_dte_frac",
    "ps_delta_exp", "ps_gamma_exp", "ps_theta_pos", "ps_iv_change",
    "ps_high_water", "ps_mae", "ps_unrealized_norm",
]
M5_EXTRAS = M5_EXTRA_LABELS + M5_EXTRA_POSSTATE

# Curvature / pivot columns that must have real data
CURVATURE_COLS = [
    "PivotHigh", "PivotLow", "Slope",
    "PivotResidual", "PivotResidualZ", "PivotCurvatureProxy",
    "SlopeATR", "PivotResidualATR", "PivotCurvatureATR",
    "PivotSegmentLengthBars", "PivotSegmentLengthMinutes",
    "PivotSegmentResidualStd", "PivotSegmentVolatility",
]

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"

failures = 0

def check(label, cond, detail=""):
    global failures
    status = PASS if cond else FAIL
    if not cond:
        failures += 1
    print(f"  [{status}] {label}" + (f"  → {detail}" if detail else ""))
    return cond

print("=" * 60)
print("CondorNet v4.3 Dataset Integrity Check")
print("=" * 60)

# ── Load headers only (fast) ──────────────────────────────────
print("\n[1] Loading headers...")
headers = {}
for tf, path in FILES.items():
    try:
        h = pd.read_csv(path, nrows=0).columns.tolist()
        headers[tf] = h
        print(f"    {tf:4s}: {len(h)} columns  ({path})")
    except FileNotFoundError:
        print(f"    {tf:4s}: FILE NOT FOUND — {path}")
        failures += 1

if len(headers) < 4:
    print("\nCannot continue — missing files.")
    sys.exit(1)

# ── Test 1: No duplicate columns ─────────────────────────────
print("\n[2] Duplicate column check...")
for tf, cols in headers.items():
    dupes = [c for c in set(cols) if cols.count(c) > 1]
    check(f"{tf}: no duplicate columns", not dupes,
          f"DUPLICATES: {dupes}" if dupes else "")

# ── Test 2: m1 / m15 / h1 identical columns ──────────────────
print("\n[3] m1 / m15 / h1 column-set parity...")
base = set(headers["m1"])
for tf in ("m15", "h1"):
    only_base = base - set(headers[tf])
    only_tf   = set(headers[tf]) - base
    check(f"{tf} has no extra cols vs m1",   not only_tf,   f"extra: {sorted(only_tf)}")
    check(f"{tf} missing no cols vs m1",     not only_base, f"missing: {sorted(only_base)}")

# ── Test 3: m5 has all m1 cols + extras ──────────────────────
print("\n[4] m5 superset check...")
m5_set   = set(headers["m5"])
m1_set   = set(headers["m1"])
missing_from_m5 = m1_set - m5_set
check("m5 contains all m1 columns", not missing_from_m5,
      f"missing: {sorted(missing_from_m5)}")

for col in M5_EXTRAS:
    check(f"m5 has {col}", col in m5_set)

extra_beyond_expected = m5_set - m1_set - set(M5_EXTRAS)
if extra_beyond_expected:
    print(f"  [{WARN}] m5 has unexpected extra cols: {sorted(extra_beyond_expected)}")

# ── Test 4: Curvature columns have real data ──────────────────
# Pivots are SPARSE by design (only set at pivot bars, NaN elsewhere).
# We sample 2000 rows from the MIDDLE of the file to catch real pivot events.
print("\n[5] Curvature data spot-check (2000 rows from file middle)...")
for tf in ("m1", "m5", "m15", "h1"):
    path = FILES[tf]
    # Count rows fast, then skip to middle
    total = sum(1 for _ in open(path)) - 1
    skip  = max(0, total // 2 - 1000)
    df = pd.read_csv(path, skiprows=range(1, skip + 1), nrows=2000,
                     usecols=lambda c: c in CURVATURE_COLS or c == "timestamp")
    present = [c for c in CURVATURE_COLS if c in df.columns]
    missing = [c for c in CURVATURE_COLS if c not in df.columns]
    if missing:
        check(f"{tf}: curvature cols present", False, f"missing cols: {missing}")
        continue
    all_nan  = [c for c in present if df[c].isna().all()]
    all_zero = [c for c in present if not df[c].isna().all()
                and df[c].fillna(0).abs().sum() == 0]
    check(f"{tf}: curvature cols not all-NaN (sparse=ok, sampled middle {skip}–{skip+2000})",
          not all_nan, f"all-NaN: {all_nan}" if all_nan else "")
    check(f"{tf}: curvature cols not all-zero", not all_zero,
          f"all-zero: {all_zero}" if all_zero else "")
    pcts = {c: f"{df[c].notna().mean()*100:.0f}%" for c in ["PivotCurvatureProxy","SlopeATR"]}
    print(f"      {tf}: non-null rates in sample → {pcts}")

# ── Test 5: Row counts ────────────────────────────────────────
print("\n[6] Row counts...")
EXPECTED = {"m1": (80_000, 110_000),  # 1-min bars: Jan 3 – Dec 12 2025 (~237 trading days × 390)
            "m5": (18_000, 20_000),   # 5-min bars: Jan 3 – Dec 12 2025
            "m15": (5_500, 7_500),    # 15-min bars: Jan 3 – Dec 12 2025
            "h1":  (1_200, 1_800)}    # hourly bars: Jan 3 – Dec 12 2025
for tf, path in FILES.items():
    n = sum(1 for _ in open(path)) - 1  # fast line count
    lo, hi = EXPECTED[tf]
    check(f"{tf}: {n:,} rows in expected range [{lo:,}–{hi:,}]",
          lo <= n <= hi, f"got {n:,}")

# ── Test 6: m5 label data spot-check ─────────────────────────
print("\n[7] m5 label column spot-check...")
m5_labels = pd.read_csv(FILES["m5"], nrows=500,
                         usecols=lambda c: c in M5_EXTRA_LABELS + ["timestamp"])
for col in M5_EXTRA_LABELS:
    if col not in m5_labels.columns:
        check(f"m5.{col} present", False); continue
    non_null = m5_labels[col].notna().sum()
    non_zero = (m5_labels[col].fillna(0) != 0).sum()
    check(f"m5.{col} has data (non-null={non_null}, non-zero={non_zero})",
          non_null > 10)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures == 0:
    print("\033[92m ALL CHECKS PASSED — datasets look intact\033[0m")
else:
    print(f"\033[91m {failures} CHECK(S) FAILED — review above\033[0m")
print("=" * 60)
sys.exit(failures)
