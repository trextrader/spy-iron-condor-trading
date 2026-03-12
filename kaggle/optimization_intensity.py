"""
optimization_intensity.py — Optimizer Intensity Policies
=========================================================
Controls breadth (how many params), granularity (step sizes),
and BO budget (init/batch/rounds) via a named intensity level.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class IntensityPolicy:
    """
    Controls:
      1) breadth_top_n     -> how many ranked params are exposed to optimization
      2) bo_init_trials    -> Sobol / random initial design size
      3) bo_batch_size     -> number of candidates proposed per BO round
      4) bo_rounds         -> number of BO rounds
      5) fidelity_schedule -> allowed progression of fidelity tiers
    """
    name: str
    breadth_top_n: int
    bo_init_trials: int
    bo_batch_size: int
    bo_rounds: int
    fidelity_schedule: List[str]

    def total_nominal_evals(self) -> int:
        return self.bo_init_trials + self.bo_batch_size * self.bo_rounds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_nominal_evals"] = self.total_nominal_evals()
        return d


INTENSITY_POLICIES: Dict[str, IntensityPolicy] = {
    "min": IntensityPolicy(
        name="min",
        breadth_top_n=7,
        bo_init_trials=8,
        bo_batch_size=4,
        bo_rounds=2,
        fidelity_schedule=["fast"],
    ),
    "min-med": IntensityPolicy(
        name="min-med",
        breadth_top_n=9,
        bo_init_trials=16,
        bo_batch_size=8,
        bo_rounds=3,
        fidelity_schedule=["fast", "medium"],
    ),
    "med": IntensityPolicy(
        name="med",
        breadth_top_n=11,
        bo_init_trials=24,
        bo_batch_size=12,
        bo_rounds=4,
        fidelity_schedule=["fast", "medium"],
    ),
    "med-max": IntensityPolicy(
        name="med-max",
        breadth_top_n=13,
        bo_init_trials=32,
        bo_batch_size=16,
        bo_rounds=6,
        fidelity_schedule=["fast", "medium", "full"],
    ),
    "max": IntensityPolicy(
        name="max",
        breadth_top_n=999,   # interpreted as "all optimizable"
        bo_init_trials=48,
        bo_batch_size=24,
        bo_rounds=8,
        fidelity_schedule=["fast", "medium", "full"],
    ),
}

DEFAULT_INTENSITY_NAME = "med"


def get_intensity_policy(name: str | None) -> IntensityPolicy:
    if not name:
        return INTENSITY_POLICIES[DEFAULT_INTENSITY_NAME]
    key = str(name).strip().lower()
    if key not in INTENSITY_POLICIES:
        allowed = ", ".join(INTENSITY_POLICIES.keys())
        raise KeyError(f"Unknown optimize intensity [{name!r}]. Allowed: {allowed}")
    return INTENSITY_POLICIES[key]


def get_intensity_names() -> List[str]:
    return list(INTENSITY_POLICIES.keys())
