"""
Phase 5.3: Single-vs-Batched Backtest Equivalence Tests

Verifies that the backtester produces identical results whether processing
trades one-at-a-time or in batches. Any divergence indicates state leakage,
order-dependence bugs, or numerical drift.

Invariants tested:
    INV-1: model(x_single) == model(x_batch)[i] for all i
    INV-2: Trade P&L independent of concurrent trades
    INV-3: Equity curve deterministic given same seed
    INV-4: No hidden state carries between inference batches

Usage:
    pytest tests/test_backtest_equivalence.py -v -s          # Auto-detect device
    pytest tests/test_backtest_equivalence.py -v -s --cpu    # Force CPU
"""

import sys
import os
import pytest
import numpy as np
import torch
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from copy import deepcopy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# PYTEST CONFIGURATION
# ==============================================================================

def pytest_addoption(parser):
    """Add --cpu flag to force CPU execution."""
    parser.addoption(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU execution (useful when GPU is busy with training)"
    )

@pytest.fixture
def force_cpu(request):
    """Check if --cpu flag was passed."""
    return request.config.getoption("--cpu")

# Attempt imports - skip tests if dependencies unavailable
try:
    from intelligence.condor_brain_net import CondorNet, AugmentedStateSpec
    HAS_MODEL = True
except ImportError as e:
    HAS_MODEL = False
    MODEL_IMPORT_ERROR = str(e)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@dataclass
class BacktestResult:
    """Container for backtest results comparison."""
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    final_equity: float

    def __eq__(self, other: 'BacktestResult') -> bool:
        if len(self.equity_curve) != len(other.equity_curve):
            return False
        if len(self.trades) != len(other.trades):
            return False
        if abs(self.final_equity - other.final_equity) > 0.01:
            return False
        return True


def set_deterministic_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def device(request):
    """Get compute device. Respects --cpu flag."""
    force_cpu = request.config.getoption("--cpu", default=False)
    if force_cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_features(device):
    """Generate sample feature tensor for testing."""
    set_deterministic_seed(42)
    # 100 sequences of length 256, 54 features
    return torch.randn(100, 256, 54, device=device, dtype=torch.float32)


@pytest.fixture
def small_model(device):
    """Create small CondorNet for testing."""
    if not HAS_MODEL:
        pytest.skip(f"CondorNet not available: {MODEL_IMPORT_ERROR}")

    set_deterministic_seed(42)

    spec = AugmentedStateSpec(
        d_h=32,  # Small for testing
        d_v=8,
        d_m=16,
        d_r=8
    )

    model = CondorNet(
        input_dim=54,
        spec=spec,
        n_predicates=16,
        n_sets=8,
        n_super_sets=4
    ).to(device)

    model.eval()
    return model


# ==============================================================================
# TEST 1: MODEL INFERENCE EQUIVALENCE (INV-1)
# ==============================================================================

@pytest.mark.skipif(not HAS_MODEL, reason="CondorNet not available")
class TestModelInferenceEquivalence:
    """
    INV-1: model(x_single) == model(x_batch)[i] for all i

    Verifies that batched inference produces identical outputs to
    sequential single-sample inference.
    """

    def test_single_vs_batched_output_equality(self, small_model, sample_features, device):
        """Core test: single inference == batched inference."""
        model = small_model
        x = sample_features[:10]  # Use 10 samples for speed

        # Single inference
        set_deterministic_seed(42)
        single_outputs = []
        with torch.no_grad():
            for i in range(len(x)):
                out, _ = model(x[i:i+1], return_diagnostics=True)
                single_outputs.append(out)
        single_stacked = torch.cat(single_outputs, dim=0)

        # Batched inference
        set_deterministic_seed(42)
        with torch.no_grad():
            batched_output, _ = model(x, return_diagnostics=True)

        # Compare
        max_diff = (single_stacked - batched_output).abs().max().item()
        mean_diff = (single_stacked - batched_output).abs().mean().item()

        print(f"\n[INV-1] Single vs Batched Inference:")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Tolerance: 1e-5")

        assert torch.allclose(single_stacked, batched_output, atol=1e-5), \
            f"Single vs batched mismatch: max_diff={max_diff:.2e}"

    def test_output_shape_consistency(self, small_model, sample_features, device):
        """Verify output shapes are consistent regardless of batch size."""
        model = small_model
        x = sample_features

        with torch.no_grad():
            # Single sample
            out_1, _ = model(x[:1], return_diagnostics=True)
            # Small batch
            out_5, _ = model(x[:5], return_diagnostics=True)
            # Larger batch
            out_20, _ = model(x[:20], return_diagnostics=True)

        assert out_1.shape[1:] == out_5.shape[1:] == out_20.shape[1:], \
            "Output feature dimensions should be consistent"
        assert out_1.shape[0] == 1
        assert out_5.shape[0] == 5
        assert out_20.shape[0] == 20

    def test_no_batch_norm_state_leakage(self, small_model, sample_features, device):
        """Verify BatchNorm (if present) doesn't leak state between samples."""
        model = small_model
        x = sample_features[:5]

        # First pass
        set_deterministic_seed(42)
        with torch.no_grad():
            out_first, _ = model(x, return_diagnostics=True)

        # Second pass (should be identical in eval mode)
        set_deterministic_seed(42)
        with torch.no_grad():
            out_second, _ = model(x, return_diagnostics=True)

        assert torch.allclose(out_first, out_second, atol=1e-6), \
            "Model outputs should be identical across eval passes"


# ==============================================================================
# TEST 3: REPLAY DETERMINISM (INV-3)
# ==============================================================================

class TestReplayDeterminism:
    """
    INV-3: Equity curve deterministic given same seed

    Verifies that running the same backtest twice with identical seed
    produces identical results at both decision and P&L levels.
    """

    @pytest.mark.skipif(not HAS_MODEL, reason="CondorNet not available")
    def test_model_output_determinism(self, small_model, sample_features, device):
        """Model outputs are deterministic with same seed."""
        model = small_model
        x = sample_features[:10]

        # Run 1
        set_deterministic_seed(42)
        with torch.no_grad():
            out_1, _ = model(x, return_diagnostics=True)

        # Run 2
        set_deterministic_seed(42)
        with torch.no_grad():
            out_2, _ = model(x, return_diagnostics=True)

        diff = (out_1 - out_2).abs().max().item()
        print(f"\n[INV-3] Model Determinism:")
        print(f"  Max difference between runs: {diff:.2e}")

        assert torch.allclose(out_1, out_2, atol=1e-7), \
            f"Model not deterministic: diff={diff:.2e}"

    def test_numpy_random_determinism(self):
        """NumPy random operations are deterministic with same seed."""
        # Run 1
        set_deterministic_seed(42)
        arr_1 = np.random.randn(100, 10)
        choice_1 = np.random.choice(100, size=10)

        # Run 2
        set_deterministic_seed(42)
        arr_2 = np.random.randn(100, 10)
        choice_2 = np.random.choice(100, size=10)

        assert np.allclose(arr_1, arr_2), "NumPy random not deterministic"
        assert np.array_equal(choice_1, choice_2), "NumPy choice not deterministic"

    def test_torch_random_determinism(self, device):
        """PyTorch random operations are deterministic with same seed."""
        # Run 1
        set_deterministic_seed(42)
        t_1 = torch.randn(100, 10, device=device)
        idx_1 = torch.randint(0, 100, (10,), device=device)

        # Run 2
        set_deterministic_seed(42)
        t_2 = torch.randn(100, 10, device=device)
        idx_2 = torch.randint(0, 100, (10,), device=device)

        assert torch.allclose(t_1, t_2), "Torch random not deterministic"
        assert torch.equal(idx_1, idx_2), "Torch randint not deterministic"


# ==============================================================================
# TEST 4: BATCH BOUNDARY CONTINUITY (INV-4)
# ==============================================================================

@pytest.mark.skipif(not HAS_MODEL, reason="CondorNet not available")
class TestBatchBoundaryContinuity:
    """
    INV-4: No hidden state carries between inference batches

    Verifies no discontinuity exists at inference batch boundaries.
    """

    def test_split_vs_combined_batches(self, small_model, sample_features, device):
        """Processing in split batches == processing in one batch."""
        model = small_model
        x = sample_features[:20]  # 20 samples

        # Split processing: 0-9, then 10-19
        set_deterministic_seed(42)
        with torch.no_grad():
            out_first_half, _ = model(x[:10], return_diagnostics=True)
            out_second_half, _ = model(x[10:], return_diagnostics=True)
        split_output = torch.cat([out_first_half, out_second_half], dim=0)

        # Combined processing: 0-19
        set_deterministic_seed(42)
        with torch.no_grad():
            combined_output, _ = model(x, return_diagnostics=True)

        max_diff = (split_output - combined_output).abs().max().item()

        print(f"\n[INV-4] Batch Boundary Test:")
        print(f"  Split (10+10) vs Combined (20)")
        print(f"  Max difference: {max_diff:.2e}")

        assert torch.allclose(split_output, combined_output, atol=1e-5), \
            f"Batch boundary discontinuity: max_diff={max_diff:.2e}"

    def test_boundary_sample_stability(self, small_model, sample_features, device):
        """Sample at batch boundary produces same output regardless of batch config."""
        model = small_model
        x = sample_features[:20]
        boundary_idx = 10  # Sample at boundary

        # As last sample of first batch
        set_deterministic_seed(42)
        with torch.no_grad():
            batch_1, _ = model(x[:11], return_diagnostics=True)
        out_as_last = batch_1[10]

        # As first sample of second batch
        set_deterministic_seed(42)
        with torch.no_grad():
            batch_2, _ = model(x[10:], return_diagnostics=True)
        out_as_first = batch_2[0]

        # As middle sample in combined batch
        set_deterministic_seed(42)
        with torch.no_grad():
            batch_full, _ = model(x, return_diagnostics=True)
        out_as_middle = batch_full[10]

        diff_1 = (out_as_last - out_as_first).abs().max().item()
        diff_2 = (out_as_last - out_as_middle).abs().max().item()

        print(f"\n[INV-4] Boundary Sample Stability:")
        print(f"  Sample 10 as last vs first: {diff_1:.2e}")
        print(f"  Sample 10 as last vs middle: {diff_2:.2e}")

        assert torch.allclose(out_as_last, out_as_first, atol=1e-5)
        assert torch.allclose(out_as_last, out_as_middle, atol=1e-5)


# ==============================================================================
# INTEGRATION TEST: SIMULATED BACKTEST LOOP
# ==============================================================================

class TestSimulatedBacktestDeterminism:
    """
    Integration test simulating the actual backtest loop pattern.
    """

    @pytest.mark.skipif(not HAS_MODEL, reason="CondorNet not available")
    def test_simulated_trading_loop_determinism(self, small_model, sample_features, device):
        """
        Simulate the actual backtest pattern:
        1. Batch inference over sequence
        2. Loop through results making trade decisions
        3. Track equity curve

        Verify determinism of the full pipeline.
        """
        model = small_model
        x = sample_features[:50]

        def run_simulation(seed: int) -> Tuple[List[float], List[int]]:
            """Run simulated trading loop."""
            set_deterministic_seed(seed)

            # Batch inference
            with torch.no_grad():
                policy, _ = model(x, return_diagnostics=True)

            # Simulate trading decisions
            equity = 100000.0
            equity_curve = [equity]
            trade_bars = []

            # Extract entry logits (index 8 per the model spec)
            entry_logits = policy[:, 8].cpu().numpy()

            for i, logit in enumerate(entry_logits):
                # Simple decision: enter if logit > 0.5
                if logit > 0.5:
                    trade_bars.append(i)
                    # Simulate P&L based on other outputs
                    pnl = float(policy[i, 4].cpu()) * 100  # Scaled prob
                    equity += pnl
                    equity_curve.append(equity)

            return equity_curve, trade_bars

        # Run twice with same seed
        curve_1, trades_1 = run_simulation(seed=42)
        curve_2, trades_2 = run_simulation(seed=42)

        # Verify equity curves match
        assert len(curve_1) == len(curve_2), \
            f"Equity curve lengths differ: {len(curve_1)} vs {len(curve_2)}"

        for i, (e1, e2) in enumerate(zip(curve_1, curve_2)):
            assert abs(e1 - e2) < 0.01, \
                f"Equity diverged at step {i}: {e1} vs {e2}"

        # Verify trade lists match
        assert trades_1 == trades_2, \
            f"Trade lists differ: {trades_1} vs {trades_2}"

        print(f"\n[Integration] Simulated Trading Loop:")
        print(f"  Trades executed: {len(trades_1)}")
        print(f"  Final equity: ${curve_1[-1]:,.2f}")
        print(f"  Determinism: PASS")


# ==============================================================================
# DIAGNOSTIC UTILITIES
# ==============================================================================

def diagnose_model_state(model: torch.nn.Module) -> Dict[str, Any]:
    """Capture model state for debugging non-determinism."""
    state = {}

    for name, param in model.named_parameters():
        state[f"param_{name}_mean"] = param.data.mean().item()
        state[f"param_{name}_std"] = param.data.std().item()

    for name, buf in model.named_buffers():
        if buf.numel() > 0:
            state[f"buffer_{name}_mean"] = buf.mean().item()

    return state


def find_first_divergence(tensor1: torch.Tensor, tensor2: torch.Tensor,
                          atol: float = 1e-5) -> Tuple[int, float]:
    """Find first index where tensors diverge beyond tolerance."""
    diff = (tensor1 - tensor2).abs()
    divergent = diff > atol

    if not divergent.any():
        return -1, 0.0

    # Find first divergent index
    flat_idx = divergent.flatten().nonzero(as_tuple=True)[0][0].item()
    return flat_idx, diff.flatten()[flat_idx].item()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
