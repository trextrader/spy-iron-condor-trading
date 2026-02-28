"""
test_exit_scaffold.py -- Exit Scaffold Validation (Phase 1 + Phase 2)
Run: python test_exit_scaffold.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch

# =============================================================================
# PHASE 1 — ExitHead model tests
# =============================================================================

from intelligence.condor_brain_net_v43 import build_condornet_v43

print("=" * 60)
print("PHASE 1 — MODEL TESTS")
print("=" * 60)

print("\nTEST 1: ExitHead forward pass")
m = build_condornet_v43()
m.eval()
B, T = 2, 16
x = torch.randn(B, T, 64)
with torch.no_grad():
    out = m.forward_compat(x)
print(f"  exit_signal shape  : {out.exit_signal.shape}")
print(f"  exit_signal values : {out.exit_signal.flatten().tolist()}")
assert out.exit_signal.shape == (B, 1), f"FAIL shape {out.exit_signal.shape}"
assert float(out.exit_signal.min()) >= 0.0
assert float(out.exit_signal.max()) <= 1.0
print("  TEST 1: PASS")

print("\nTEST 2: CondorNetOutput fields + to_dict()")
assert hasattr(out, 'exit_signal')
assert 'exit_signal' in out.to_dict()
print("  TEST 2: PASS")

print("\nTEST 3: forward() and forward_compat() both return exit_signal")
chain      = torch.zeros(B, 1, 10)
chain_mask = torch.ones(B, 1, dtype=torch.bool)
with torch.no_grad():
    out2 = m.forward(x_m1=x, x_m5=x, x_m15=x, x_h1=x,
                     chain=chain, chain_mask=chain_mask)
assert out2.exit_signal.shape == (B, 1)
print(f"  forward() exit_signal: {out2.exit_signal.shape}")
print("  TEST 3: PASS")

# =============================================================================
# PHASE 2 — SimExit labeling engine tests
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 2 — SIMEXIT LABELING ENGINE TESTS")
print("=" * 60)

from intelligence.data_pipeline_v43 import (
    compute_simexit_labels,
    _ic_value_scalar,
    _ic_value_vec,
    _bsm_call_vec,
    _bsm_put_vec,
)

print("\nTEST 4: BSM helpers — call/put prices")
S_arr   = np.array([500.0, 510.0, 490.0])
T_arr   = np.array([0.01,  0.005, 0.002])
sig_arr = np.array([0.20,  0.18,  0.22])
calls = _bsm_call_vec(S_arr, 505.0, T_arr, sig_arr)
puts  = _bsm_put_vec(S_arr,  495.0, T_arr, sig_arr)
assert calls.shape == (3,), f"FAIL call shape {calls.shape}"
assert puts.shape  == (3,), f"FAIL put shape  {puts.shape}"
assert (calls >= 0).all(), f"FAIL negative call prices"
assert (puts  >= 0).all(), f"FAIL negative put prices"
print(f"  calls={calls.round(4)}  puts={puts.round(4)}")
print("  TEST 4: PASS")

print("\nTEST 5: _ic_value_scalar — credit at entry > 0")
S0 = 500.0; sig0 = 0.20; T0 = 1.0/252.0
credit = _ic_value_scalar(S0, 510.0, 515.0, 490.0, 485.0, T0, sig0)
print(f"  IC credit (scalar): {credit:.4f}")
assert credit > 0, f"FAIL: credit={credit} should be > 0"
print("  TEST 5: PASS")

print("\nTEST 6: compute_simexit_labels — shape, range, non-trivial distribution")
# Build a synthetic 3-day M5 dataset (78 bars/day)
np.random.seed(42)
n_days, bars_per_day = 3, 78
N = n_days * bars_per_day
dates = pd.date_range("2024-01-02 09:30", periods=N, freq="5min")
closes = 500.0 + np.cumsum(np.random.randn(N) * 0.5)
df_test = pd.DataFrame({
    "timestamp": dates,
    "close":     closes.clip(400, 600),
    "high":      closes.clip(400, 600) + 0.5,
    "low":       closes.clip(400, 600) - 0.5,
    "vol_ewma":  np.full(N, 0.008),   # ~20% annualised
})
labels = compute_simexit_labels(df_test, target_delta=0.15,
                                 spread_atr_mult=2.0, epsilon=0.02)
print(f"  labels shape  : {labels.shape}")
print(f"  labels range  : [{labels.min():.1f}, {labels.max():.1f}]")
frac = float(labels.mean())
print(f"  exit=1 frac   : {frac:.3f}  ({frac*100:.1f}%)")
assert labels.shape == (N,),   f"FAIL shape {labels.shape}"
assert set(np.unique(labels)) <= {0.0, 1.0}, f"FAIL: labels not binary {np.unique(labels)}"
assert 0.05 < frac < 0.95,     f"FAIL: trivial label distribution frac={frac:.3f}"
print("  TEST 6: PASS")

print("\nTEST 7: Last bar of each day must be exit=1 (V_hold = V_exit at expiry)")
for day in range(n_days):
    last_bar = (day + 1) * bars_per_day - 1
    assert labels[last_bar] == 1.0, \
        f"FAIL: last bar of day {day} (idx={last_bar}) should be exit=1, got {labels[last_bar]}"
print("  Last bar of each day is exit=1 (confirmed)")
print("  TEST 7: PASS")

print("\nTEST 8: epsilon sensitivity — higher epsilon → more exit=1 bars")
labels_tight = compute_simexit_labels(df_test, epsilon=0.001)
labels_loose = compute_simexit_labels(df_test, epsilon=0.10)
assert labels_loose.mean() >= labels_tight.mean(), \
    f"FAIL: loose epsilon should give more exit=1 bars"
print(f"  epsilon=0.001: {labels_tight.mean()*100:.1f}%  "
      f"epsilon=0.10: {labels_loose.mean()*100:.1f}%")
print("  TEST 8: PASS")

# =============================================================================
print("\n" + "=" * 60)
print("ALL TESTS PASSED — Phase 1 + Phase 2")
print("=" * 60)
print("""
Next step — regenerate M5 labels with real SimExit labels:
  python intelligence/data_pipeline_v43.py --force --simexit-epsilon 0.02

Then retrain:
  python intelligence/condor_train_net_v43.py --epochs 30 [your usual args]
""")
