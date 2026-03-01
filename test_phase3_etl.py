"""test_phase3_etl.py — Validates Phase 3 Position State Vector ETL output.

Run AFTER the ETL pipeline:
    python intelligence/data_pipeline_v43.py --force --simexit-epsilon 0.02
    python test_phase3_etl.py

All tests should print PASS. Any FAIL means the ETL did not produce valid Phase 3 data.
"""

import sys
import traceback
from pathlib import Path

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
_failures = []

def check(label, condition):
    if condition:
        print(f"  [{PASS}] {label}")
    else:
        print(f"  [{FAIL}] {label}")
        _failures.append(label)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ──────────────────────────────────────────────────────────────────────────────
# Resolve dataset path
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/Datasetv4/v43")
M5_PATH  = DATA_DIR / "m5_dataset_v43_final.csv"

section("Phase 3 ETL — Pre-flight")
check("M5 dataset file exists", M5_PATH.exists())
if not M5_PATH.exists():
    print(f"\n  [ERROR] {M5_PATH} not found.")
    print("  Run: python intelligence/data_pipeline_v43.py --force --simexit-epsilon 0.02")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Load CSV
# ──────────────────────────────────────────────────────────────────────────────
section("Phase 3 ETL — CSV Load & Column Check")
try:
    import numpy as np
    import pandas as pd
    from intelligence.schema_v43 import POS_STATE_NAMES, N_POS_STATE, TF_LABEL_NAMES

    df = pd.read_csv(M5_PATH, nrows=None, low_memory=False)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # All 11 ps_* columns present
    for col in POS_STATE_NAMES:
        check(f"Column '{col}' present", col in df.columns)

    # Column count sanity: 90 (pre-Phase3) + 11 = 101
    # Or possibly 102 if there are extra ETL cols — just ensure >= 101
    check(f"Total columns >= 101 (90 base + 11 ps_*)", len(df.columns) >= 101)

    # exit_signal still present (Phase 2)
    check("Column 'exit_signal' still present (Phase 2 not clobbered)", "exit_signal" in df.columns)

    # strategy_label still present (Phase 1)
    check("Column 'strategy_label' still present", "strategy_label" in df.columns)

except Exception as exc:
    check(f"CSV load failed: {exc}", False)
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Position State Value Checks
# ──────────────────────────────────────────────────────────────────────────────
section("Phase 3 ETL — Position State Value Checks")

# Active bars: rows where ps_pnl_pct != 0 (position is open)
ps_pnl = df["ps_pnl_pct"].values
n_active = int((ps_pnl != 0.0).sum())
n_total  = len(df)
active_pct = n_active / n_total * 100 if n_total > 0 else 0
print(f"  Active position bars: {n_active:,} / {n_total:,} ({active_pct:.1f}%)")
check("Active bars > 0 (pos_state engine ran)", n_active > 0)
# Should be a meaningful fraction — not too sparse, not all bars
check("Active bars fraction > 5%",  active_pct > 5.0)
check("Active bars fraction < 90%", active_pct < 90.0)

# ps_bars_held: normalized age in [0, 1]
bh = df["ps_bars_held"].values
check("ps_bars_held min >= 0.0", float(bh.min()) >= -1e-6)
check("ps_bars_held max <= 1.0 + eps", float(bh.max()) <= 1.0 + 1e-4)

# ps_dte_frac: fraction of DTE remaining in [0, 1]
dte = df["ps_dte_frac"].values
check("ps_dte_frac min >= 0.0", float(dte.min()) >= -1e-6)
check("ps_dte_frac max <= 1.0 + eps", float(dte.max()) <= 1.0 + 1e-4)

# ps_theta_pos: should be >= 0 for IC seller
theta = df["ps_theta_pos"].values
check("ps_theta_pos min >= 0.0 (IC seller θ positive)", float(theta.min()) >= -1e-6)

# ps_high_water: running max of pnl_pct — should be >= ps_pnl_pct at every bar
hw  = df["ps_high_water"].values
pnl = df["ps_pnl_pct"].values
# High water must be >= pnl_pct element-wise (only for active bars to avoid noise)
active_mask = pnl != 0.0
if n_active > 0:
    hw_ok = (hw[active_mask] >= pnl[active_mask] - 1e-4).all()
    check("ps_high_water >= ps_pnl_pct at all active bars", hw_ok)

# ps_mae: max adverse excursion — should be >= 0
mae = df["ps_mae"].values
check("ps_mae >= 0.0 at all bars", float(mae.min()) >= -1e-6)

# ps_credit_norm: credit / spot — small positive number
cr = df["ps_credit_norm"].values
check("ps_credit_norm >= 0.0", float(cr.min()) >= -1e-6)

# No NaN in any ps_* column (these are always set, never sparse)
for col in POS_STATE_NAMES:
    n_nan = df[col].isna().sum()
    check(f"No NaN in {col}", n_nan == 0)

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Dataset Loader Smoke Test
# ──────────────────────────────────────────────────────────────────────────────
section("Phase 3 ETL — Dataset Loader Smoke Test")

try:
    import torch
    from intelligence.condor_train_net_v43 import _load_pos_state, N_POS_STATE as N_PS

    ps_arr = _load_pos_state(str(M5_PATH))
    check("_load_pos_state returns non-None", ps_arr is not None)

    if ps_arr is not None:
        check(f"pos_state shape is (N, 11)", ps_arr.ndim == 2 and ps_arr.shape[1] == N_PS)
        check(f"pos_state rows match M5 CSV rows ({len(df)})", ps_arr.shape[0] == len(df))
        check("pos_state dtype is float32", ps_arr.dtype == np.float32)
        check("pos_state has no NaN", not np.isnan(ps_arr).any())
        check("pos_state has active rows (nonzero pnl_pct col)",
              int((ps_arr[:, 0] != 0.0).sum()) > 0)
        print(f"  Active rows via _load_pos_state: {int((ps_arr[:, 0] != 0.0).sum()):,}")

except ImportError as exc:
    # PyTorch may not be in test environment
    print(f"  [{SKIP}] Dataset loader smoke test skipped ({exc})")
except Exception as exc:
    check(f"Dataset loader smoke test failed: {exc}", False)
    traceback.print_exc()

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Model Integration Check
# ──────────────────────────────────────────────────────────────────────────────
section("Phase 3 ETL — Model Integration Check")

try:
    import torch
    from intelligence.condor_brain_net_v43 import CondorNetV43, PosStateProjector
    from intelligence.schema_v43 import N_POS_STATE as NPS

    check("PosStateProjector importable", True)

    # Instantiate and verify zero-init
    proj = PosStateProjector(NPS, 256)
    w_norm = proj.proj.weight.abs().max().item()
    b_norm = proj.proj.bias.abs().max().item()
    check("PosStateProjector weight is zero-init", w_norm < 1e-9)
    check("PosStateProjector bias is zero-init", b_norm < 1e-9)

    # Forward with zeros → should produce zero delta
    ps_zeros = torch.zeros(2, 8, NPS)
    delta = proj(ps_zeros)
    check("Zero pos_state → zero delta from PosStateProjector",
          delta.abs().max().item() < 1e-9)

    # Forward with real data → tanh-bounded output
    ps_real = torch.randn(2, 8, NPS)
    # Manually set nonzero weights to test tanh bound
    with torch.no_grad():
        proj.proj.weight.fill_(0.1)
    delta_real = proj(ps_real)
    check("PosStateProjector output in (-1, 1) after tanh",
          delta_real.abs().max().item() < 1.0 + 1e-6)

except ImportError as exc:
    print(f"  [{SKIP}] Model integration check skipped ({exc})")
except Exception as exc:
    check(f"Model integration check failed: {exc}", False)
    traceback.print_exc()

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
section("Results")
print(f"\n  Failures: {len(_failures)}")
if _failures:
    for f in _failures:
        print(f"    ✗ {f}")
    print(f"\n  RESULT: {len(_failures)} test(s) FAILED")
    sys.exit(1)
else:
    print(f"\n  RESULT: ALL TESTS PASSED ✓")
    print(f"\n  Ready to retrain:")
    print(f"    python intelligence/condor_train_net_v43.py \\")
    print(f"        --epochs 60 --batch-size 64 --lookback 64 \\")
    print(f"        --lr 1e-4 --patience 10 --min-delta 0.001 \\")
    print(f"        --gate-temp 3.0 \\")
    print(f"        --logit-var-alpha 0.1  --logit-var-target 2.0 \\")
    print(f"        --logit-iqr-alpha 0.08 --logit-iqr-target 1.0 \\")
    print(f"        --logit-mad-alpha 0.05 --logit-mad-target 0.9 \\")
    print(f"        --resume models/condornet_v43_best.pth")
