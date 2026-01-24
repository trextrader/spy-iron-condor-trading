from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExecutionCostResult:
    entry_slippage: float
    entry_commission: float
    total_cost: float
    per_leg: List[Dict[str, float]]
    details: Dict[str, float]


def _half_spread(bid: Optional[float], ask: Optional[float], min_half_spread: float) -> float:
    if bid is None or ask is None:
        return min_half_spread
    spread = max(0.0, ask - bid)
    return max(min_half_spread, 0.5 * spread)


def estimate_entry_cost(
    legs,
    quantity: int,
    r_cfg,
    min_half_spread: float = 0.01,
    impact_coeff: float = 0.10,
) -> ExecutionCostResult:
    """
    Estimate entry slippage + commissions for a 4-leg condor.

    - Uses bid/ask half-spread when available.
    - Falls back to slippage_per_contract if no spread data.
    - Adds a small impact term scaled by sqrt(quantity).
    """
    slippage_per_contract = getattr(r_cfg, "slippage_per_contract", 0.02)
    commission_per_contract = getattr(r_cfg, "commission_per_contract", 0.65)

    per_leg = []
    total_slippage = 0.0

    for leg in [legs.short_call, legs.long_call, legs.short_put, legs.long_put]:
        bid = getattr(leg, "bid", None)
        ask = getattr(leg, "ask", None)
        half_spread = _half_spread(bid, ask, min_half_spread)
        if bid is None or ask is None:
            half_spread = max(half_spread, slippage_per_contract)

        impact = impact_coeff * half_spread * (quantity ** 0.5)
        leg_slip = (half_spread + impact) * quantity
        total_slippage += leg_slip

        per_leg.append(
            {
                "bid": float(bid) if bid is not None else None,
                "ask": float(ask) if ask is not None else None,
                "half_spread": float(half_spread),
                "impact": float(impact),
                "slippage": float(leg_slip),
            }
        )

    entry_commission = float(commission_per_contract) * quantity * 4
    total_cost = total_slippage + entry_commission

    return ExecutionCostResult(
        entry_slippage=float(total_slippage),
        entry_commission=float(entry_commission),
        total_cost=float(total_cost),
        per_leg=per_leg,
        details={
            "min_half_spread": float(min_half_spread),
            "impact_coeff": float(impact_coeff),
            "slippage_per_contract": float(slippage_per_contract),
            "commission_per_contract": float(commission_per_contract),
        },
    )
