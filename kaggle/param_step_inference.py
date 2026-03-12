"""
param_step_inference.py — Intensity-Aware Parameter Step / Grid Inference
==========================================================================
Classifies each parameter by kind, then returns intensity-appropriate
step sizes and semantic or arithmetic grids.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, List
import math


# Semantic grids for time-domain params (preferred over arithmetic steps).
SEMANTIC_INTEGER_GRIDS = {
    "target_dte": {
        "min":     [7, 14, 21],
        "min-med": [7, 14, 21, 28],
        "med":     [7, 10, 14, 17, 21, 28],
        "med-max": list(range(7, 29)),
        "max":     list(range(7, 61)),
    },
    "max_dte_exit": {
        "min":     [0, 1, 2, 3, 5, 7],
        "min-med": list(range(0, 8)),
        "med":     list(range(0, 11)),
        "med-max": list(range(0, 15)),
        "max":     list(range(0, 15)),
    },
    "hold_days": {
        "min":     [5, 9, 13, 17, 21],
        "min-med": [3, 5, 7, 9, 11, 14, 17, 21],
        "med":     list(range(3, 22, 2)),
        "med-max": list(range(1, 22)),
        "max":     list(range(1, 22)),
    },
}


def _to_decimal(x: Any) -> Decimal:
    return Decimal(str(x))


def _frange_inclusive(lo: float, hi: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    vals: List[float] = []
    cur = _to_decimal(lo)
    d_hi = _to_decimal(hi)
    d_step = _to_decimal(step)
    while cur <= d_hi + Decimal("1e-12"):
        vals.append(float(cur))
        cur += d_step
    return vals


def infer_param_kind(param_name: str, current_value: Any, lo: float, hi: float) -> str:
    """Heuristic classification of a parameter into a kind category."""
    name = str(param_name).lower()
    width = float(hi) - float(lo)

    if "delta" in name:
        return "delta"
    if "pct" in name or "percent" in name or "margin_pct" in name:
        return "percent"
    if "dollar" in name or "profit_target" in name or "stop_loss" in name:
        return "dollar"
    if "width" in name:
        return "width"
    if isinstance(current_value, int) or (
        isinstance(current_value, float) and float(lo).is_integer()
        and float(hi).is_integer() and width >= 1
    ):
        return "integer"
    return "continuous"


def infer_param_step(
    param_name: str,
    lo: float,
    hi: float,
    kind: str,
    intensity: str,
) -> float | None:
    """Return the appropriate numeric step for (kind, intensity)."""
    width = float(hi) - float(lo)

    if kind == "dollar":
        if intensity == "min":      return 250.0 if width >= 1000 else 100.0
        if intensity == "min-med":  return 100.0 if width >= 1000 else 50.0
        if intensity == "med":      return 50.0  if width >= 1000 else 25.0
        if intensity == "med-max":  return 25.0  if width >= 1000 else 10.0
        if intensity == "max":      return 5.0

    if kind == "delta":
        return {"min": 0.10, "min-med": 0.05, "med": 0.02, "med-max": 0.01, "max": 0.005}[intensity]

    if kind == "percent":
        return {"min": 0.05, "min-med": 0.02, "med": 0.01, "med-max": 0.005, "max": 0.001}[intensity]

    if kind == "integer":
        return {"min": 2.0 if width >= 6 else 1.0, "min-med": 1.0, "med": 1.0, "med-max": 1.0, "max": 1.0}[intensity]

    if kind == "width":
        return {"min": 5.0, "min-med": 2.0, "med": 1.0, "med-max": 1.0, "max": 0.5}[intensity]

    # continuous fallback
    return {"min": max(width / 4.0, 1e-3), "min-med": max(width / 8.0, 1e-3),
            "med": max(width / 16.0, 1e-3), "med-max": max(width / 32.0, 1e-3),
            "max": max(width / 64.0, 1e-4)}[intensity]


def build_semantic_or_arithmetic_grid(
    param_name: str,
    lo: float,
    hi: float,
    current_value: Any,
    intensity: str,
    kind: str | None = None,
) -> list[float]:
    """Return the candidate grid values for a param at the given intensity."""
    kind = kind or infer_param_kind(param_name, current_value, lo, hi)

    # Semantic grids for special time-domain params
    if param_name in SEMANTIC_INTEGER_GRIDS:
        grid = [x for x in SEMANTIC_INTEGER_GRIDS[param_name][intensity] if lo <= x <= hi]
        if grid:
            return [float(x) for x in grid]

    step = infer_param_step(param_name, lo, hi, kind, intensity)
    if step is None or step <= 0:
        return [float(current_value)]

    grid = _frange_inclusive(float(lo), float(hi), float(step))

    if kind == "integer":
        ints = sorted(set(int(round(x)) for x in grid if lo <= round(x) <= hi))
        return [float(x) for x in ints]

    # normalise tiny fp noise
    decimals = max(0, min(6, -Decimal(str(step)).as_tuple().exponent))
    return [round(float(x), decimals) for x in grid]


def infer_range_from_template_value(
    param_name: str,
    current_value: Any,
) -> tuple[float, float] | None:
    """
    Conservative auto-range from a single current value when no explicit
    bounds are available. Returns None for non-numeric / structural fields.
    """
    name = str(param_name).lower()
    v = current_value

    if v is None or isinstance(v, (bool, str, list, dict, tuple, set)):
        return None
    try:
        fv = float(v)
    except Exception:
        return None

    if "delta" in name:
        return max(0.01, round(fv - 0.10, 4)), min(0.99, round(fv + 0.10, 4))

    if "dte" in name or "hold_days" in name:
        lo = max(0.0, math.floor(fv * 0.5))
        hi = max(lo + 1.0, math.ceil(fv * 1.5))
        return lo, hi

    if "width" in name:
        lo = max(1.0, round(fv * 0.5))
        hi = max(lo + 1.0, round(fv * 2.0))
        return lo, hi

    if "pct" in name or "percent" in name:
        return round(fv - 0.10, 4), round(fv + 0.10, 4)

    if "dollar" in name or "profit_target" in name or "stop_loss" in name:
        lo = max(0.0, math.floor(fv * 0.5 / 5) * 5)
        hi = max(lo + 10.0, math.ceil(fv * 1.5 / 5) * 5)
        return lo, hi

    if float(fv).is_integer():
        lo = max(0.0, math.floor(fv * 0.5))
        hi = max(lo + 1.0, math.ceil(fv * 1.5))
        return lo, hi

    return round(fv * 0.5, 6), round(fv * 1.5, 6)
