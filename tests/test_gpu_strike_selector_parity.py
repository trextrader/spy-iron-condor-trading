"""
test_gpu_strike_selector_parity.py
===================================
Stage 1 parity tests: CPU _find_best_structure_* vs GPU select_entry_for_bar.

Tests run on CPU torch (no CUDA required) unless CUDA is available.
All four strategy families are covered: iron_condor, iron_butterfly,
short_call, short_put.

Edge cases:
  - empty chain
  - single option
  - all-NaN deltas (ATM fallback)
  - target delta outside available range

Usage:
    pytest tests/test_gpu_strike_selector_parity.py -v
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_KAGGLE_DIR   = _PROJECT_ROOT / "kaggle"
if str(_KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(_KAGGLE_DIR))

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from gpu_strike_selector import (
        select_entry_for_bar,
        nearest_delta_indices_bruteforce,
        nearest_delta_indices_sorted,
    )
    from optimizer_engine import (
        _find_best_structure_iron_condor,
        _find_best_structure_iron_bfly,
        _find_best_structure_short_call,
        _find_best_structure_short_put,
    )
    HAS_SELECTOR = True
except ImportError as e:
    HAS_SELECTOR = False
    _IMPORT_ERR = str(e)

pytestmark = pytest.mark.skipif(
    not HAS_SELECTOR,
    reason=f"kaggle imports unavailable: {'' if HAS_SELECTOR else _IMPORT_ERR}",
)

# ── Synthetic chain definition ─────────────────────────────────────────────────
#
# SPY-like chain: 9 strikes around spot=500, DTE=21
# (strike, call_delta, put_delta, call_mid, put_mid)
_SPOT = 500.0
_CHAIN_DEF = [
    (480, 0.95, 0.05, 22.0,  2.0),
    (485, 0.85, 0.15, 17.0,  3.5),
    (490, 0.70, 0.30, 12.5,  5.0),
    (495, 0.55, 0.45,  8.0,  7.5),
    (500, 0.45, 0.55,  5.5,  5.5),
    (505, 0.30, 0.70,  3.5,  8.0),
    (510, 0.20, 0.80,  2.0, 12.5),
    (515, 0.12, 0.88,  1.2, 17.0),
    (520, 0.05, 0.95,  0.7, 22.0),
]

# K=4 test candidates
_TARGET_DTE   = np.array([21.0, 21.0, 21.0, 21.0], dtype=np.float32)
_SHORT_DELTA  = np.array([0.20, 0.15, 0.30, 0.12], dtype=np.float32)
_SPREAD_WIDTH = np.array([5.0,  5.0,  5.0,  5.0],  dtype=np.float32)

ATOL = 1e-3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_chain_np():
    """Build numpy chain arrays from _CHAIN_DEF (calls + puts at DTE=21)."""
    right, strike, dte_arr, delta, bid, ask = [], [], [], [], [], []
    for stk, cd, pd, cmid, pmid in _CHAIN_DEF:
        # call
        right.append(0); strike.append(stk); dte_arr.append(21.0)
        delta.append(cd); bid.append(cmid * 0.9); ask.append(cmid * 1.1)
        # put
        right.append(1); strike.append(stk); dte_arr.append(21.0)
        delta.append(pd); bid.append(pmid * 0.9); ask.append(pmid * 1.1)
    return (
        np.array(right,    dtype=np.int8),
        np.array(strike,   dtype=np.float32),
        np.array(dte_arr,  dtype=np.float32),
        np.array(delta,    dtype=np.float32),
        np.array(bid,      dtype=np.float32),
        np.array(ask,      dtype=np.float32),
    )


def _chain_to_torch(right_np, strike_np, dte_np, delta_np, bid_np, ask_np,
                    device=None):
    """Convert numpy chain arrays to torch tensors on device."""
    if device is None:
        device = torch.device("cpu")
    return (
        torch.as_tensor(right_np,  device=device, dtype=torch.int8),
        torch.as_tensor(strike_np, device=device, dtype=torch.float32),
        torch.as_tensor(dte_np,    device=device, dtype=torch.float32),
        torch.as_tensor(delta_np,  device=device, dtype=torch.float32),
        torch.as_tensor(bid_np,    device=device, dtype=torch.float32),
        torch.as_tensor(ask_np,    device=device, dtype=torch.float32),
    )


def _params_to_torch(target_dte, short_delta, spread_width, device=None):
    if device is None:
        device = torch.device("cpu")
    return (
        torch.as_tensor(target_dte,   device=device, dtype=torch.float32),
        torch.as_tensor(short_delta,  device=device, dtype=torch.float32),
        torch.as_tensor(spread_width, device=device, dtype=torch.float32),
    )


def _assert_gpu_matches_cpu_ic(cpu_credit, cpu_ssc, cpu_ssp, cpu_aw, cpu_valid, cpu_dte,
                                gpu_result, atol=ATOL):
    """Compare iron_condor CPU result against GPU dict result."""
    gpu_valid = gpu_result["valid"].astype(bool)
    assert np.array_equal(cpu_valid, gpu_valid), \
        f"valid mismatch:\n  cpu={cpu_valid}\n  gpu={gpu_valid}"
    for k in range(len(cpu_valid)):
        if cpu_valid[k]:
            assert abs(float(cpu_credit[k]) - float(gpu_result["credit"][k])) < atol, \
                f"k={k} credit: cpu={cpu_credit[k]:.4f} gpu={gpu_result['credit'][k]:.4f}"
            assert abs(float(cpu_ssc[k]) - float(gpu_result["ss_call"][k])) < atol, \
                f"k={k} ss_call: cpu={cpu_ssc[k]} gpu={gpu_result['ss_call'][k]}"
            assert abs(float(cpu_ssp[k]) - float(gpu_result["ss_put"][k])) < atol, \
                f"k={k} ss_put: cpu={cpu_ssp[k]} gpu={gpu_result['ss_put'][k]}"
            assert abs(float(cpu_dte[k]) - float(gpu_result["actual_dte"][k])) < atol, \
                f"k={k} actual_dte: cpu={cpu_dte[k]} gpu={gpu_result['actual_dte'][k]}"


def _assert_gpu_matches_cpu_bfly(cpu_credit, cpu_ss, cpu_aw, cpu_valid, cpu_dte,
                                  gpu_result, atol=ATOL):
    """Compare iron_butterfly CPU (single ss) against GPU dict result."""
    gpu_valid = gpu_result["valid"].astype(bool)
    assert np.array_equal(cpu_valid, gpu_valid), \
        f"valid mismatch:\n  cpu={cpu_valid}\n  gpu={gpu_valid}"
    for k in range(len(cpu_valid)):
        if cpu_valid[k]:
            assert abs(float(cpu_credit[k]) - float(gpu_result["credit"][k])) < atol, \
                f"k={k} credit"
            assert abs(float(cpu_ss[k]) - float(gpu_result["ss_call"][k])) < atol, \
                f"k={k} ss_call"
            # GPU iron_butterfly: ss_call == ss_put (same ATM short)
            assert abs(float(gpu_result["ss_call"][k]) - float(gpu_result["ss_put"][k])) < atol, \
                f"k={k} ss_call != ss_put for iron_butterfly"
            assert abs(float(cpu_dte[k]) - float(gpu_result["actual_dte"][k])) < atol, \
                f"k={k} actual_dte"


def _assert_gpu_matches_cpu_single(cpu_credit, cpu_ss, cpu_valid, cpu_dte,
                                    gpu_result, ss_key="ss_call", atol=ATOL):
    """Compare short_call / short_put CPU against GPU dict result."""
    gpu_valid = gpu_result["valid"].astype(bool)
    assert np.array_equal(cpu_valid, gpu_valid), \
        f"valid mismatch:\n  cpu={cpu_valid}\n  gpu={gpu_valid}"
    for k in range(len(cpu_valid)):
        if cpu_valid[k]:
            assert abs(float(cpu_credit[k]) - float(gpu_result["credit"][k])) < atol, \
                f"k={k} credit"
            assert abs(float(cpu_ss[k]) - float(gpu_result[ss_key][k])) < atol, \
                f"k={k} {ss_key}"
            assert abs(float(cpu_dte[k]) - float(gpu_result["actual_dte"][k])) < atol, \
                f"k={k} actual_dte"


# ── Delta-search primitive tests ───────────────────────────────────────────────

class TestDeltaSearch:
    """nearest_delta_indices_bruteforce vs nearest_delta_indices_sorted."""

    def test_bruteforce_basic(self):
        deltas  = torch.tensor([0.05, 0.12, 0.20, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95])
        targets = torch.tensor([0.20, 0.15, 0.32, 0.60])
        idx = nearest_delta_indices_bruteforce(deltas, targets)
        # 0.20 → index 2; 0.15 → closest is 0.12 at 1; 0.32 → 0.30 at 3; 0.60 → 0.55 at 5
        assert idx.tolist() == [2, 1, 3, 5], f"got {idx.tolist()}"

    def test_sorted_matches_bruteforce(self):
        deltas  = torch.tensor([0.05, 0.12, 0.20, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95])
        targets = torch.tensor([0.20, 0.15, 0.32, 0.60, 0.01, 0.99])
        bf  = nearest_delta_indices_bruteforce(deltas, targets)
        srt = nearest_delta_indices_sorted(deltas, targets)
        assert torch.equal(bf, srt), f"bruteforce={bf.tolist()} sorted={srt.tolist()}"

    def test_bruteforce_with_valid_mask(self):
        """Valid mask should exclude masked entries."""
        deltas  = torch.tensor([0.05, 0.20, 0.30, 0.45])
        targets = torch.tensor([0.20])
        # Mask out index 1 (the exact match at 0.20)
        valid = torch.tensor([True, False, True, True])
        idx = nearest_delta_indices_bruteforce(deltas, targets, valid_mask=valid)
        # Nearest valid delta to 0.20 is 0.30 at index 2
        assert idx.item() == 2, f"expected 2, got {idx.item()}"

    def test_bruteforce_single_element(self):
        deltas  = torch.tensor([0.25])
        targets = torch.tensor([0.20, 0.30, 0.99])
        idx = nearest_delta_indices_bruteforce(deltas, targets)
        assert idx.tolist() == [0, 0, 0]

    def test_sorted_boundary_targets(self):
        """Targets below min and above max should clamp to endpoints."""
        deltas  = torch.tensor([0.10, 0.20, 0.30])
        targets = torch.tensor([0.01, 0.99])
        idx = nearest_delta_indices_sorted(deltas, targets)
        assert idx.tolist() == [0, 2], f"got {idx.tolist()}"


# ── Iron condor parity ─────────────────────────────────────────────────────────

class TestIronCondorParity:

    def test_standard_chain(self):
        """GPU and CPU iron_condor select the same strikes and credits."""
        np_chain = _build_chain_np()
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = np_chain

        # CPU
        cpu = _find_best_structure_iron_condor(
            _SPOT, right_np, strike_np, dte_np, delta_np, bid_np, ask_np,
            _TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH,
        )
        cpu_credit, cpu_ssc, cpu_ssp, cpu_aw, cpu_valid, cpu_dte = cpu

        # GPU (CPU torch device)
        t_right, t_stk, t_dte, t_delta, t_bid, t_ask = _chain_to_torch(*np_chain)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH)
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "iron_condor",
        )

        _assert_gpu_matches_cpu_ic(cpu_credit, cpu_ssc, cpu_ssp, cpu_aw,
                                   cpu_valid, cpu_dte, gpu)

    def test_k0_expected_values(self):
        """Spot-check k=0: sd=0.20, sw=5 → ssc=510, ssp=485, credit≈2.3."""
        np_chain = _build_chain_np()
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = np_chain

        cpu = _find_best_structure_iron_condor(
            _SPOT, right_np, strike_np, dte_np, delta_np, bid_np, ask_np,
            _TARGET_DTE[:1], _SHORT_DELTA[:1], _SPREAD_WIDTH[:1],
        )
        assert cpu[4][0], "k=0 expected valid=True"
        assert abs(cpu[1][0] - 510.0) < ATOL, f"ssc expected 510, got {cpu[1][0]}"
        assert abs(cpu[2][0] - 485.0) < ATOL, f"ssp expected 485, got {cpu[2][0]}"
        assert abs(cpu[0][0] - 2.3)   < 0.05, f"credit expected ~2.3, got {cpu[0][0]}"

    def test_empty_chain(self):
        """Empty chain → all valid=False for both CPU and GPU."""
        emp = np.array([], dtype=np.int8)
        emp_f = np.array([], dtype=np.float32)

        cpu = _find_best_structure_iron_condor(
            _SPOT, emp, emp_f, emp_f, emp_f, emp_f, emp_f,
            _TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH,
        )
        assert not cpu[4].any(), "CPU: empty chain should give valid=False"

        t_right = torch.zeros(0, dtype=torch.int8)
        t_stk   = torch.zeros(0, dtype=torch.float32)
        t_dte   = torch.zeros(0, dtype=torch.float32)
        t_delta = torch.zeros(0, dtype=torch.float32)
        t_bid   = torch.zeros(0, dtype=torch.float32)
        t_ask   = torch.zeros(0, dtype=torch.float32)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH)
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "iron_condor",
        )
        assert not gpu["valid"].any(), "GPU: empty chain should give valid=False"

    def test_nan_deltas_fallback(self):
        """Chain with all NaN deltas should fall back to ATM selection."""
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = _build_chain_np()
        delta_nan = np.full_like(delta_np, np.nan)

        # CPU
        cpu = _find_best_structure_iron_condor(
            _SPOT, right_np, strike_np, dte_np, delta_nan, bid_np, ask_np,
            _TARGET_DTE[:1], _SHORT_DELTA[:1], _SPREAD_WIDTH[:1],
        )
        cpu_credit, cpu_ssc, cpu_ssp, cpu_aw, cpu_valid, cpu_dte = cpu

        # GPU
        t_right, t_stk, t_dte, _, t_bid, t_ask = _chain_to_torch(
            right_np, strike_np, dte_np, delta_nan, bid_np, ask_np
        )
        t_tdte, t_sd, t_sw = _params_to_torch(
            _TARGET_DTE[:1], _SHORT_DELTA[:1], _SPREAD_WIDTH[:1]
        )
        t_delta_nan = torch.full_like(t_stk, float("nan"))
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta_nan, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "iron_condor",
        )
        _assert_gpu_matches_cpu_ic(cpu_credit, cpu_ssc, cpu_ssp, cpu_aw,
                                   cpu_valid, cpu_dte, gpu)


# ── Iron butterfly parity ─────────────────────────────────────────────────────

class TestIronButterflyParity:

    def test_standard_chain(self):
        np_chain = _build_chain_np()
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = np_chain

        cpu = _find_best_structure_iron_bfly(
            _SPOT, right_np, strike_np, dte_np, delta_np, bid_np, ask_np,
            _TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH,
        )
        cpu_credit, cpu_ss, cpu_aw, cpu_valid, cpu_dte = cpu

        t_right, t_stk, t_dte, t_delta, t_bid, t_ask = _chain_to_torch(*np_chain)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH)
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "iron_butterfly",
        )
        _assert_gpu_matches_cpu_bfly(cpu_credit, cpu_ss, cpu_aw, cpu_valid, cpu_dte, gpu)

    def test_ss_call_equals_ss_put(self):
        """For iron_butterfly GPU output, ss_call must equal ss_put."""
        np_chain = _build_chain_np()
        t_right, t_stk, t_dte, t_delta, t_bid, t_ask = _chain_to_torch(*np_chain)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH)
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "iron_butterfly",
        )
        valid_mask = gpu["valid"].astype(bool)
        sc = gpu["ss_call"][valid_mask]
        sp = gpu["ss_put"][valid_mask]
        assert np.allclose(sc, sp, atol=ATOL), \
            f"iron_butterfly: ss_call != ss_put: {sc} vs {sp}"

    def test_empty_chain(self):
        emp = np.array([], dtype=np.int8)
        emp_f = np.array([], dtype=np.float32)
        cpu = _find_best_structure_iron_bfly(
            _SPOT, emp, emp_f, emp_f, emp_f, emp_f, emp_f,
            _TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH,
        )
        assert not cpu[3].any()

        t_right = torch.zeros(0, dtype=torch.int8)
        t_f     = torch.zeros(0, dtype=torch.float32)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH)
        gpu = select_entry_for_bar(
            t_right, t_f, t_f, t_f, t_f, t_f, _SPOT,
            t_tdte, t_sd, t_sw, "iron_butterfly",
        )
        assert not gpu["valid"].any()


# ── Short call parity ─────────────────────────────────────────────────────────

class TestShortCallParity:

    def test_standard_chain(self):
        np_chain = _build_chain_np()
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = np_chain

        cpu = _find_best_structure_short_call(
            _SPOT, right_np, strike_np, dte_np, delta_np, bid_np, ask_np,
            _TARGET_DTE, _SHORT_DELTA,
        )
        cpu_credit, cpu_ss, cpu_valid, cpu_dte = cpu

        t_right, t_stk, t_dte, t_delta, t_bid, t_ask = _chain_to_torch(*np_chain)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA,
                                               np.zeros_like(_SPREAD_WIDTH))
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "short_call",
        )
        _assert_gpu_matches_cpu_single(cpu_credit, cpu_ss, cpu_valid, cpu_dte,
                                       gpu, ss_key="ss_call")

    def test_no_calls_in_chain(self):
        """Chain with only puts → valid=False for short_call."""
        puts_only_right = np.ones(10, dtype=np.int8)  # all puts
        stk = np.linspace(490, 540, 10, dtype=np.float32)
        dte = np.full(10, 21.0, dtype=np.float32)
        dlt = np.linspace(0.1, 0.5, 10, dtype=np.float32)
        bid = np.full(10, 1.0, dtype=np.float32)
        ask = np.full(10, 1.2, dtype=np.float32)

        cpu = _find_best_structure_short_call(
            _SPOT, puts_only_right, stk, dte, dlt, bid, ask,
            _TARGET_DTE[:1], _SHORT_DELTA[:1],
        )
        assert not cpu[2].any(), "CPU: no calls → valid=False"

        t_right = torch.ones(10, dtype=torch.int8)
        t_stk   = torch.as_tensor(stk)
        t_dte   = torch.as_tensor(dte)
        t_delta = torch.as_tensor(dlt)
        t_bid   = torch.as_tensor(bid)
        t_ask   = torch.as_tensor(ask)
        t_tdte, t_sd, t_sw = _params_to_torch(
            _TARGET_DTE[:1], _SHORT_DELTA[:1], np.zeros(1, dtype=np.float32)
        )
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "short_call",
        )
        assert not gpu["valid"].any(), "GPU: no calls → valid=False"

    def test_nan_deltas_fallback(self):
        """NaN deltas → ATM call selection; CPU and GPU should agree."""
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = _build_chain_np()
        delta_nan = np.full_like(delta_np, np.nan)

        cpu = _find_best_structure_short_call(
            _SPOT, right_np, strike_np, dte_np, delta_nan, bid_np, ask_np,
            _TARGET_DTE[:1], _SHORT_DELTA[:1],
        )
        t_right, t_stk, t_dte, _, t_bid, t_ask = _chain_to_torch(
            right_np, strike_np, dte_np, delta_nan, bid_np, ask_np
        )
        t_tdte, t_sd, t_sw = _params_to_torch(
            _TARGET_DTE[:1], _SHORT_DELTA[:1], np.zeros(1, dtype=np.float32)
        )
        t_delta_nan = torch.full_like(t_stk, float("nan"))
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta_nan, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "short_call",
        )
        if cpu[2][0]:  # if CPU produced a valid entry
            assert gpu["valid"][0], "GPU should agree on valid"
            assert abs(float(cpu[1][0]) - float(gpu["ss_call"][0])) < ATOL, \
                f"strike mismatch: cpu={cpu[1][0]} gpu={gpu['ss_call'][0]}"


# ── Short put parity ──────────────────────────────────────────────────────────

class TestShortPutParity:

    def test_standard_chain(self):
        np_chain = _build_chain_np()
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = np_chain

        cpu = _find_best_structure_short_put(
            _SPOT, right_np, strike_np, dte_np, delta_np, bid_np, ask_np,
            _TARGET_DTE, _SHORT_DELTA,
        )
        cpu_credit, cpu_ss, cpu_valid, cpu_dte = cpu

        t_right, t_stk, t_dte, t_delta, t_bid, t_ask = _chain_to_torch(*np_chain)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA,
                                               np.zeros_like(_SPREAD_WIDTH))
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "short_put",
        )
        _assert_gpu_matches_cpu_single(cpu_credit, cpu_ss, cpu_valid, cpu_dte,
                                       gpu, ss_key="ss_put")

    def test_no_puts_in_chain(self):
        """Chain with only calls → valid=False for short_put."""
        calls_only = np.zeros(10, dtype=np.int8)  # all calls
        stk = np.linspace(460, 500, 10, dtype=np.float32)
        dte = np.full(10, 21.0, dtype=np.float32)
        dlt = np.linspace(0.9, 0.3, 10, dtype=np.float32)
        bid = np.full(10, 2.0, dtype=np.float32)
        ask = np.full(10, 2.4, dtype=np.float32)

        cpu = _find_best_structure_short_put(
            _SPOT, calls_only, stk, dte, dlt, bid, ask,
            _TARGET_DTE[:1], _SHORT_DELTA[:1],
        )
        assert not cpu[2].any(), "CPU: no puts → valid=False"


# ── Dtype contract tests ───────────────────────────────────────────────────────

class TestDtypeContracts:
    """Verify that incorrect dtypes raise TypeError/ValueError."""

    def test_wrong_chain_right_dtype_raises(self):
        """chain_right as float32 should raise."""
        np_chain = _build_chain_np()
        t_right_bad = torch.zeros(len(np_chain[0]), dtype=torch.float32)  # wrong
        t_stk   = torch.as_tensor(np_chain[1], dtype=torch.float32)
        t_dte   = torch.as_tensor(np_chain[2], dtype=torch.float32)
        t_delta = torch.as_tensor(np_chain[3], dtype=torch.float32)
        t_bid   = torch.as_tensor(np_chain[4], dtype=torch.float32)
        t_ask   = torch.as_tensor(np_chain[5], dtype=torch.float32)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH)
        with pytest.raises(TypeError, match="chain_right must be an integer dtype"):
            select_entry_for_bar(
                t_right_bad, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
                t_tdte, t_sd, t_sw, "iron_condor",
            )

    def test_wrong_chain_strike_dtype_raises(self):
        """chain_strike as float64 should raise."""
        np_chain = _build_chain_np()
        t_right = torch.as_tensor(np_chain[0], dtype=torch.int8)
        t_stk_bad = torch.as_tensor(np_chain[1], dtype=torch.float64)  # wrong
        t_dte   = torch.as_tensor(np_chain[2], dtype=torch.float32)
        t_delta = torch.as_tensor(np_chain[3], dtype=torch.float32)
        t_bid   = torch.as_tensor(np_chain[4], dtype=torch.float32)
        t_ask   = torch.as_tensor(np_chain[5], dtype=torch.float32)
        t_tdte, t_sd, t_sw = _params_to_torch(_TARGET_DTE, _SHORT_DELTA, _SPREAD_WIDTH)
        with pytest.raises(TypeError, match="chain_strike must be"):
            select_entry_for_bar(
                t_right, t_stk_bad, t_dte, t_delta, t_bid, t_ask, _SPOT,
                t_tdte, t_sd, t_sw, "iron_condor",
            )

    def test_wrong_target_dte_dtype_raises(self):
        """target_dte as float64 should raise."""
        np_chain = _build_chain_np()
        t_right, t_stk, t_dte, t_delta, t_bid, t_ask = _chain_to_torch(*np_chain)
        t_tdte_bad = torch.as_tensor(_TARGET_DTE, dtype=torch.float64)  # wrong
        t_sd  = torch.as_tensor(_SHORT_DELTA,  dtype=torch.float32)
        t_sw  = torch.as_tensor(_SPREAD_WIDTH, dtype=torch.float32)
        with pytest.raises(TypeError, match="target_dte must be"):
            select_entry_for_bar(
                t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
                t_tdte_bad, t_sd, t_sw, "iron_condor",
            )


# ── Edge-case: K=1 ────────────────────────────────────────────────────────────

class TestKEquals1:

    def test_k1_iron_condor(self):
        """Single candidate (K=1) gives same result as CPU."""
        np_chain = _build_chain_np()
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = np_chain

        tdte1 = _TARGET_DTE[:1]
        sd1   = _SHORT_DELTA[:1]
        sw1   = _SPREAD_WIDTH[:1]

        cpu = _find_best_structure_iron_condor(
            _SPOT, right_np, strike_np, dte_np, delta_np, bid_np, ask_np,
            tdte1, sd1, sw1,
        )
        t_right, t_stk, t_dte, t_delta, t_bid, t_ask = _chain_to_torch(*np_chain)
        t_tdte, t_sd, t_sw = _params_to_torch(tdte1, sd1, sw1)
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "iron_condor",
        )
        _assert_gpu_matches_cpu_ic(cpu[0], cpu[1], cpu[2], cpu[3],
                                   cpu[4], cpu[5], gpu)

    def test_k1_short_call(self):
        np_chain = _build_chain_np()
        right_np, strike_np, dte_np, delta_np, bid_np, ask_np = np_chain

        cpu = _find_best_structure_short_call(
            _SPOT, right_np, strike_np, dte_np, delta_np, bid_np, ask_np,
            _TARGET_DTE[:1], _SHORT_DELTA[:1],
        )
        t_right, t_stk, t_dte, t_delta, t_bid, t_ask = _chain_to_torch(*np_chain)
        t_tdte, t_sd, t_sw = _params_to_torch(
            _TARGET_DTE[:1], _SHORT_DELTA[:1], np.zeros(1, dtype=np.float32)
        )
        gpu = select_entry_for_bar(
            t_right, t_stk, t_dte, t_delta, t_bid, t_ask, _SPOT,
            t_tdte, t_sd, t_sw, "short_call",
        )
        _assert_gpu_matches_cpu_single(cpu[0], cpu[1], cpu[2], cpu[3], gpu)
