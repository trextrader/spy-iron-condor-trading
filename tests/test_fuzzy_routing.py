import pytest

from qtmf.models import TradeIntent
from qtmf.facade import benchmark_and_size


def _base_intent(extras=None):
    return TradeIntent(
        symbol="SPY",
        action="ENTER",
        gaussian_confidence=0.9,
        current_price=500.0,
        ivr=50.0,
        vix=18.0,
        realized_vol=15.0,
        mtf_snapshot=None,
        extras=extras or {},
    )


def test_fuzzy_mode_forced_from_fis_legacy():
    extras = {
        "min_gaussian_confidence": 0.1,
        "fuzzy_mode": "fis_legacy",
        "allow_fis_legacy": False,
    }
    plan = benchmark_and_size(_base_intent(extras))
    assert plan.approved
    assert plan.diagnostics.get("fuzzy_mode_forced") == "qtmf10"


def test_fuzzy_mode_forced_from_f001_without_allow():
    extras = {
        "min_gaussian_confidence": 0.1,
        "fuzzy_mode": "f001",
        "allow_f001_augment": False,
        "f001_score": 0.9,
    }
    plan = benchmark_and_size(_base_intent(extras))
    assert plan.approved
    assert plan.diagnostics.get("fuzzy_mode_forced") == "qtmf10"


def test_f001_augment_blend():
    extras = {
        "min_gaussian_confidence": 0.1,
        "allow_f001_augment": True,
        "f001_score": 0.8,
        "f001_alpha": 0.3,
    }
    plan = benchmark_and_size(_base_intent(extras))
    assert plan.approved
    assert plan.diagnostics.get("f001_used") is True
    assert abs(plan.diagnostics.get("f001_alpha") - 0.3) < 1e-6
