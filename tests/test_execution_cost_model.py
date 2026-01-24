from dataclasses import dataclass

from core.dto import IronCondorLegs, OptionQuote
from intelligence.execution.cost_model import estimate_entry_cost


@dataclass
class DummyRunConfig:
    slippage_per_contract: float = 0.02
    commission_per_contract: float = 0.65


def _legs_with_spread(bid: float, ask: float) -> IronCondorLegs:
    leg = OptionQuote(
        strike=100.0,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        iv=0.2,
        delta=0.1,
    )
    return IronCondorLegs(
        long_put=leg,
        short_put=leg,
        short_call=leg,
        long_call=leg,
        net_credit=1.0,
        max_loss=4.0,
    )


def test_estimate_entry_cost_uses_spread():
    cfg = DummyRunConfig(slippage_per_contract=0.02, commission_per_contract=0.65)
    legs = _legs_with_spread(1.00, 1.20)  # half-spread = 0.10
    result = estimate_entry_cost(legs, quantity=2, r_cfg=cfg, min_half_spread=0.01, impact_coeff=0.10)
    assert result.entry_slippage > 0.0
    assert result.entry_commission == 0.65 * 2 * 4
    assert len(result.per_leg) == 4
    assert result.per_leg[0]["half_spread"] >= 0.10 - 1e-12


def test_estimate_entry_cost_fallbacks_when_no_spread():
    cfg = DummyRunConfig(slippage_per_contract=0.05, commission_per_contract=0.65)
    leg = OptionQuote(
        strike=100.0,
        bid=None,
        ask=None,
        mid=1.0,
        iv=0.2,
        delta=0.1,
    )
    legs = IronCondorLegs(
        long_put=leg,
        short_put=leg,
        short_call=leg,
        long_call=leg,
        net_credit=1.0,
        max_loss=4.0,
    )
    result = estimate_entry_cost(legs, quantity=1, r_cfg=cfg, min_half_spread=0.01, impact_coeff=0.10)
    assert result.per_leg[0]["half_spread"] >= 0.05
