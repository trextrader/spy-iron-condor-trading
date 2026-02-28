"""
test_exit_scaffold.py -- Phase 1 Exit Scaffold Validation
Run: python test_exit_scaffold.py
"""
import sys
sys.path.insert(0, '.')

import torch
from intelligence.condor_brain_net_v43 import build_condornet_v43

print("=" * 60)
print("TEST 1: ExitHead forward pass")
print("=" * 60)

m = build_condornet_v43()
m.eval()

B, T = 2, 16
x = torch.randn(B, T, 64)

with torch.no_grad():
    out = m.forward_compat(x)

print(f"  exit_signal shape  : {out.exit_signal.shape}")
print(f"  exit_signal values : {out.exit_signal.flatten().tolist()}")
print(f"  entry_signal values: {out.entry_signal.flatten().tolist()}")

assert out.exit_signal.shape == (B, 1), \
    f"FAIL: expected shape ({B}, 1), got {out.exit_signal.shape}"
assert float(out.exit_signal.min()) >= 0.0, \
    f"FAIL: exit_signal below 0: {out.exit_signal.min()}"
assert float(out.exit_signal.max()) <= 1.0, \
    f"FAIL: exit_signal above 1: {out.exit_signal.max()}"

print("  TEST 1: PASS\n")

print("=" * 60)
print("TEST 2: CondorNetOutput has exit_signal field")
print("=" * 60)

assert hasattr(out, 'exit_signal'), "FAIL: CondorNetOutput missing exit_signal"
assert hasattr(out, 'entry_signal'), "FAIL: CondorNetOutput missing entry_signal"

d = out.to_dict()
assert 'exit_signal' in d, "FAIL: to_dict() missing exit_signal"
print(f"  to_dict() keys (subset): entry_signal, exit_signal both present")
print("  TEST 2: PASS\n")

print("=" * 60)
print("TEST 3: forward_compat and forward both return exit_signal")
print("=" * 60)

chain      = torch.zeros(B, 1, 10)
chain_mask = torch.ones(B, 1, dtype=torch.bool)

with torch.no_grad():
    out2 = m.forward(
        x_m1=x, x_m5=x, x_m15=x, x_h1=x,
        chain=chain,
        chain_mask=chain_mask,
    )

assert out2.exit_signal.shape == (B, 1), \
    f"FAIL: forward() exit_signal shape {out2.exit_signal.shape}"
print(f"  forward() exit_signal shape: {out2.exit_signal.shape}")
print("  TEST 3: PASS\n")

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
