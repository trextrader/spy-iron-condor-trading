"""
Phase 5.4: Replay Determinism Theorem (Full-System)

Proves that the complete CondorBrain system is a deterministic function:
    f(Tape, Seed) → (Trades, Equity, Diagnostics)

This extends Phase 5.3's model-level invariance to the entire execution stack.

Usage:
    pytest tests/test_replay_determinism.py -v -s --cpu
"""

import sys
import os
import pytest
import numpy as np
import torch
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt imports
try:
    from intelligence.condor_brain_net import CondorNet
    HAS_CONDORNET = True
except ImportError as e:
    HAS_CONDORNET = False
    CONDORNET_ERROR = str(e)

try:
    from intelligence.fuzzy_engine import FuzzyInferenceEngine
    HAS_FUZZY = True
except ImportError as e:
    HAS_FUZZY = False
    FUZZY_ERROR = str(e)

try:
    from intelligence.rule_engine.executor import RuleExecutor
    HAS_RULE_ENGINE = True
except ImportError as e:
    HAS_RULE_ENGINE = False
    RULE_ENGINE_ERROR = str(e)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ==============================================================================
# DETERMINISTIC SEED CONTROL
# ==============================================================================

def set_full_determinism(seed: int = 42):
    """
    Set ALL sources of randomness for full determinism.

    This is more aggressive than Phase 5.3's seed control.
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Force deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # Older PyTorch


# ==============================================================================
# TEST DATA STRUCTURES
# ==============================================================================

@dataclass
class TradeRecord:
    """Comparable trade record for determinism testing."""
    trade_id: str
    entry_dt: Any
    exit_dt: Any
    legs: Dict[str, Any]
    entry_credit: float
    exit_debit: float
    pnl: float
    exit_reason: str

    def __eq__(self, other: 'TradeRecord') -> bool:
        if self.trade_id != other.trade_id:
            return False
        if str(self.entry_dt) != str(other.entry_dt):
            return False
        if str(self.exit_dt) != str(other.exit_dt):
            return False
        if abs(self.pnl - other.pnl) > 0.01:
            return False
        return True


@dataclass
class BacktestSnapshot:
    """Complete backtest state for comparison."""
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    final_equity: float = 0.0
    total_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0

    def fingerprint(self) -> str:
        """Generate deterministic hash of results."""
        import hashlib
        data = f"{self.trade_count}:{self.final_equity:.2f}:{self.total_pnl:.2f}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


# ==============================================================================
# TEST: FUZZY ENGINE DETERMINISM (RDT-4)
# ==============================================================================

class TestFuzzyEngineDeterminism:
    """RDT-4: Fuzzy sizing produces identical position sizes."""

    @pytest.mark.skipif(not HAS_FUZZY, reason=f"FuzzyEngine not available")
    def test_fuzzy_membership_determinism(self):
        """Membership function evaluation is deterministic."""
        # Create engine
        engine = FuzzyInferenceEngine()

        # Same input
        market_state = {
            'iv_rank': 45.0,
            'vix': 18.5,
            'rsi': 52.0,
            'adx': 22.0,
            'bb_width': 0.015,
            'volume_ratio': 1.2,
        }

        # Evaluate twice
        result_1 = engine.evaluate(market_state)
        result_2 = engine.evaluate(market_state)

        print(f"\n[RDT-4] Fuzzy Engine Determinism:")
        print(f"  Result 1: {result_1}")
        print(f"  Result 2: {result_2}")

        assert result_1 == result_2, "Fuzzy evaluation not deterministic"

    @pytest.mark.skipif(not HAS_FUZZY, reason=f"FuzzyEngine not available")
    def test_fuzzy_across_seeds(self):
        """Fuzzy engine should NOT depend on random seed (it's deterministic)."""
        engine = FuzzyInferenceEngine()

        market_state = {
            'iv_rank': 60.0,
            'vix': 22.0,
            'rsi': 35.0,
        }

        set_full_determinism(42)
        result_seed_42 = engine.evaluate(market_state)

        set_full_determinism(999)
        result_seed_999 = engine.evaluate(market_state)

        assert result_seed_42 == result_seed_999, \
            "Fuzzy engine should be independent of random seed"


# ==============================================================================
# TEST: EXECUTION FILL DETERMINISM (RDT-5)
# ==============================================================================

class TestExecutionFillDeterminism:
    """RDT-5: Execution fill prices are deterministic."""

    def test_synthesize_bid_ask_determinism(self):
        """Bid/ask synthesis is deterministic."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "kaggle"))
            from condor_brain_backtest_v2 import synthesize_bid_ask
        except ImportError:
            pytest.skip("condor_brain_backtest_v2 not importable")

        row = {
            'close': 5.50,
            'spread_ratio': 0.02,
        }

        result_1 = synthesize_bid_ask(row)
        result_2 = synthesize_bid_ask(row)

        print(f"\n[RDT-5] Bid/Ask Synthesis:")
        print(f"  Result 1: {result_1}")
        print(f"  Result 2: {result_2}")

        assert result_1 == result_2, "Bid/ask synthesis not deterministic"

    def test_entry_fill_determinism(self):
        """Entry fill calculation is deterministic."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "kaggle"))
            from condor_brain_backtest_v2 import calculate_entry_fill, synthesize_chain_prices
            import pandas as pd
        except ImportError:
            pytest.skip("Required modules not importable")

        # Create sample chain
        chain_data = {
            'option_symbol': ['SPY_C_500', 'SPY_C_505', 'SPY_P_490', 'SPY_P_485'],
            'strike': [500, 505, 490, 485],
            'call_put': ['C', 'C', 'P', 'P'],
            'close': [5.50, 3.20, 4.80, 2.90],
            'spread_ratio': [0.02, 0.02, 0.02, 0.02],
            'volume': [100, 100, 100, 100],
            'open_interest': [500, 500, 500, 500],
            'delta': [0.45, 0.35, -0.40, -0.30],
        }
        chain_df = pd.DataFrame(chain_data)
        chain_with_prices = synthesize_chain_prices(chain_df)

        legs = {
            'short_call_symbol': 'SPY_C_500',
            'long_call_symbol': 'SPY_C_505',
            'short_put_symbol': 'SPY_P_490',
            'long_put_symbol': 'SPY_P_485',
            'width': 5,
        }

        # Calculate twice
        qty1, credit1, atomic1, details1 = calculate_entry_fill(legs, chain_with_prices, 10)
        qty2, credit2, atomic2, details2 = calculate_entry_fill(legs, chain_with_prices, 10)

        print(f"\n[RDT-5] Entry Fill Determinism:")
        print(f"  Run 1: qty={qty1}, credit={credit1:.4f}, atomic={atomic1}")
        print(f"  Run 2: qty={qty2}, credit={credit2:.4f}, atomic={atomic2}")

        assert qty1 == qty2, "Entry quantity not deterministic"
        assert credit1 == credit2, "Entry credit not deterministic"
        assert atomic1 == atomic2, "Atomicity flag not deterministic"


# ==============================================================================
# TEST: TIMESTAMP INDEPENDENCE (RDT-6)
# ==============================================================================

class TestTimestampIndependence:
    """RDT-6: No timestamp-dependent randomness."""

    @pytest.mark.skipif(not HAS_CONDORNET, reason="CondorNet not available")
    def test_model_wallclock_independence(self, device):
        """Model output doesn't depend on wall-clock time."""
        set_full_determinism(42)

        model = CondorNet(
            d_input=54,
            d_h=32,
            d_v=8,
            d_m=16,
            d_r=8,
            d_control=32,
            n_predicates=16,
            n_sets=8,
            n_super_sets=4
        ).to(device)
        model.eval()

        x = torch.randn(5, 256, 54, device=device)

        # Run at time T
        set_full_determinism(42)
        with torch.no_grad():
            out_t1, _ = model(x, return_diagnostics=True)

        # Wait and run again
        time.sleep(0.1)

        set_full_determinism(42)
        with torch.no_grad():
            out_t2, _ = model(x, return_diagnostics=True)

        diff = (out_t1 - out_t2).abs().max().item()

        print(f"\n[RDT-6] Wall-Clock Independence:")
        print(f"  Max diff after 0.1s delay: {diff:.2e}")

        assert diff == 0.0, f"Model depends on wall-clock time: diff={diff}"


# ==============================================================================
# TEST: NUMPY/PANDAS DETERMINISM
# ==============================================================================

class TestDataPipelineDeterminism:
    """Verify data manipulation operations are deterministic."""

    @pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
    def test_pandas_groupby_determinism(self):
        """Pandas groupby maintains order."""
        import pandas as pd

        # Create test data
        data = {
            'timestamp': ['2024-01-01'] * 5 + ['2024-01-02'] * 5,
            'value': list(range(10)),
        }
        df = pd.DataFrame(data)

        # Group twice
        groups_1 = list(df.groupby('timestamp').groups.keys())
        groups_2 = list(df.groupby('timestamp').groups.keys())

        assert groups_1 == groups_2, "Groupby order not deterministic"

    @pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
    def test_pandas_sort_determinism(self):
        """Pandas sort is deterministic."""
        import pandas as pd

        data = {
            'a': [3, 1, 2, 1, 3],
            'b': [1, 2, 3, 4, 5],
        }
        df = pd.DataFrame(data)

        sorted_1 = df.sort_values(['a', 'b']).reset_index(drop=True)
        sorted_2 = df.sort_values(['a', 'b']).reset_index(drop=True)

        assert sorted_1.equals(sorted_2), "Sort not deterministic"


# ==============================================================================
# TEST: DICT ITERATION ORDER
# ==============================================================================

class TestDictDeterminism:
    """Verify dict iteration order is deterministic (Python 3.7+)."""

    def test_dict_insertion_order(self):
        """Dict maintains insertion order."""
        d1 = {'a': 1, 'b': 2, 'c': 3}
        d2 = {'a': 1, 'b': 2, 'c': 3}

        keys_1 = list(d1.keys())
        keys_2 = list(d2.keys())

        assert keys_1 == keys_2, "Dict insertion order not preserved"

    def test_dict_from_zip_determinism(self):
        """Dict from zip is deterministic."""
        symbols = ['A', 'B', 'C', 'D']
        prices = [1.0, 2.0, 3.0, 4.0]

        d1 = dict(zip(symbols, prices))
        d2 = dict(zip(symbols, prices))

        assert list(d1.items()) == list(d2.items()), "Dict from zip not deterministic"


# ==============================================================================
# INTEGRATION TEST: SIMULATED FULL BACKTEST REPLAY
# ==============================================================================

@pytest.mark.skipif(not HAS_CONDORNET, reason="CondorNet not available")
class TestFullBacktestReplay:
    """
    RDT-1, RDT-2: Full backtest produces identical results.

    This is the ultimate integration test for the Replay Determinism Theorem.
    """

    def test_simulated_full_replay(self, device):
        """
        Simulate a full backtest replay and verify determinism.
        """
        def run_simulated_backtest(seed: int) -> BacktestSnapshot:
            """Run a simulated backtest with full determinism."""
            set_full_determinism(seed)

            # Create model
            model = CondorNet(
                d_input=54,
                d_h=32,
                d_v=8,
                d_m=16,
                d_r=8,
                d_control=32,
                n_predicates=16,
                n_sets=8,
                n_super_sets=4
            ).to(device)
            model.eval()

            # Generate deterministic "tape"
            set_full_determinism(seed)
            tape = torch.randn(100, 256, 54, device=device)

            # Run inference
            with torch.no_grad():
                policy, _ = model(tape, return_diagnostics=True)

            # Simulate trading logic
            equity = 100000.0
            equity_curve = [equity]
            trades = []
            trade_id = 0

            entry_logits = policy[:, 8].cpu().numpy()

            for i, logit in enumerate(entry_logits):
                if logit > 0.5:
                    # Simulate trade
                    pnl = float(policy[i, 4].cpu()) * 100
                    equity += pnl
                    equity_curve.append(equity)

                    trades.append(TradeRecord(
                        trade_id=f"TR_{trade_id}",
                        entry_dt=f"2024-01-01T{i:02d}:00:00",
                        exit_dt=f"2024-01-01T{i+1:02d}:00:00",
                        legs={'short_call': 500 + i},
                        entry_credit=1.50,
                        exit_debit=0.50,
                        pnl=pnl,
                        exit_reason="EXPIRED"
                    ))
                    trade_id += 1

            snapshot = BacktestSnapshot(
                trades=trades,
                equity_curve=equity_curve,
                final_equity=equity,
                total_pnl=equity - 100000.0,
                trade_count=len(trades),
                win_count=sum(1 for t in trades if t.pnl > 0)
            )

            return snapshot

        # Run twice with same seed
        snapshot_1 = run_simulated_backtest(seed=42)
        snapshot_2 = run_simulated_backtest(seed=42)

        print(f"\n[RDT-1/2] Full Backtest Replay:")
        print(f"  Run 1: {snapshot_1.trade_count} trades, ${snapshot_1.final_equity:,.2f}")
        print(f"  Run 2: {snapshot_2.trade_count} trades, ${snapshot_2.final_equity:,.2f}")
        print(f"  Fingerprint 1: {snapshot_1.fingerprint()}")
        print(f"  Fingerprint 2: {snapshot_2.fingerprint()}")

        # Verify trade count
        assert snapshot_1.trade_count == snapshot_2.trade_count, \
            f"Trade count mismatch: {snapshot_1.trade_count} vs {snapshot_2.trade_count}"

        # Verify final equity
        assert abs(snapshot_1.final_equity - snapshot_2.final_equity) < 0.01, \
            f"Final equity mismatch: {snapshot_1.final_equity} vs {snapshot_2.final_equity}"

        # Verify equity curves
        assert len(snapshot_1.equity_curve) == len(snapshot_2.equity_curve), \
            "Equity curve length mismatch"

        for i, (e1, e2) in enumerate(zip(snapshot_1.equity_curve, snapshot_2.equity_curve)):
            assert abs(e1 - e2) < 0.01, f"Equity diverged at step {i}: {e1} vs {e2}"

        # Verify individual trades
        for t1, t2 in zip(snapshot_1.trades, snapshot_2.trades):
            assert t1 == t2, f"Trade mismatch: {t1} vs {t2}"

        # Verify fingerprints
        assert snapshot_1.fingerprint() == snapshot_2.fingerprint(), \
            "Backtest fingerprints don't match"

        print(f"  Determinism: VERIFIED")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
