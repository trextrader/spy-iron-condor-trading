"""
adaptive_search_space_builder.py — Intensity-Adaptive Search Space Construction
================================================================================
Builds a strategy-specific SearchSpaceSpec from:
  - strategy config current values (bounds via explicit_param_bounds or auto-inferred)
  - intensity policy (breadth + step granularity)
  - ranked parameter priorities (which params to include)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from candidate_codec import ParamSpec, SearchSpaceSpec
from optimization_intensity import IntensityPolicy, get_intensity_policy
from strategy_param_priority import (
    NON_OPTIMIZABLE_FIELDS,
    get_ranked_param_priority,
)
from param_step_inference import (
    infer_param_kind,
    infer_range_from_template_value,
    infer_param_step,
    build_semantic_or_arithmetic_grid,
)


@dataclass
class AdaptiveParamManifestRow:
    param_name: str
    priority_rank: Optional[int]
    status: str            # OPTIMIZED | fixed | non-optimizable | unsupported
    current_value: Any
    lo: Optional[float] = None
    hi: Optional[float] = None
    kind: Optional[str] = None
    inferred_step: Optional[float] = None
    grid: Optional[List[float]] = None
    notes: Optional[str] = None


def _is_scalar_optimizable(v: Any) -> bool:
    if v is None or isinstance(v, (bool, str, list, dict, tuple, set)):
        return False
    try:
        float(v)
        return True
    except Exception:
        return False


def _make_param_spec(
    param_name: str,
    current_value: Any,
    lo: float,
    hi: float,
    intensity_name: str,
) -> Tuple[str, ParamSpec, List[float]]:
    """
    Returns (kind, ParamSpec, grid_values) for one parameter.
    Maps internal kinds to the two kinds supported by candidate_codec:
      'grid' (discrete snapping) or 'continuous' (GP free).
    """
    kind = infer_param_kind(param_name, current_value, lo, hi)
    grid = build_semantic_or_arithmetic_grid(
        param_name=param_name, lo=lo, hi=hi,
        current_value=current_value, intensity=intensity_name, kind=kind,
    )
    step = infer_param_step(param_name, lo, hi, kind, intensity_name)

    if kind in {"delta", "percent", "continuous"}:
        # Keep as continuous in BO box; step is only for display
        spec = ParamSpec(
            name=param_name, lo=float(lo), hi=float(hi),
            kind="continuous",   # step defaults to 1.0 but is unused for continuous
        )
    elif kind == "integer":
        spec = ParamSpec(
            name=param_name, lo=float(lo), hi=float(hi),
            kind="grid", step=1.0, dtype=int,
        )
    else:
        # dollar, width, or other grid kind
        s = float(step) if step is not None and step > 0 else 1.0
        spec = ParamSpec(
            name=param_name, lo=float(lo), hi=float(hi),
            kind="grid", step=s,
        )

    return kind, spec, grid


def build_adaptive_search_space(
    *,
    strategy_id: str,
    strategy_config: Dict[str, Any],
    intensity_name: str,
    explicit_param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    non_optimizable_fields: Optional[Set[str]] = None,
) -> Tuple[SearchSpaceSpec, List[AdaptiveParamManifestRow], IntensityPolicy]:
    """
    Build an intensity-adaptive SearchSpaceSpec for one strategy.

    Returns
    -------
    (SearchSpaceSpec, manifest_rows, IntensityPolicy)
    """
    explicit_param_bounds = explicit_param_bounds or {}
    non_opt = set(non_optimizable_fields or set()) | NON_OPTIMIZABLE_FIELDS
    policy = get_intensity_policy(intensity_name)

    available_scalars = [
        k for k, v in strategy_config.items()
        if k not in non_opt and _is_scalar_optimizable(v)
    ]

    ranked = get_ranked_param_priority(
        strategy_id=strategy_id,
        available_param_names=available_scalars,
        non_optimizable_fields=non_opt,
    )

    top_n = policy.breadth_top_n
    optimized_names = set(ranked if top_n >= len(ranked) else ranked[:top_n])
    priority_map = {name: i + 1 for i, name in enumerate(ranked)}

    manifest_rows: List[AdaptiveParamManifestRow] = []
    param_specs: List[ParamSpec] = []

    for name, current in strategy_config.items():
        if name in non_opt:
            manifest_rows.append(AdaptiveParamManifestRow(
                param_name=name, priority_rank=None,
                status="non-optimizable", current_value=current,
                notes="metadata/structural field",
            ))
            continue

        if not _is_scalar_optimizable(current):
            manifest_rows.append(AdaptiveParamManifestRow(
                param_name=name, priority_rank=priority_map.get(name),
                status="unsupported", current_value=current,
                notes="non-scalar value type",
            ))
            continue

        if name not in optimized_names:
            manifest_rows.append(AdaptiveParamManifestRow(
                param_name=name, priority_rank=priority_map.get(name),
                status="fixed", current_value=current,
            ))
            continue

        # Resolve bounds
        if name in explicit_param_bounds:
            lo, hi = explicit_param_bounds[name]
        else:
            inferred = infer_range_from_template_value(name, current)
            if inferred is None:
                manifest_rows.append(AdaptiveParamManifestRow(
                    param_name=name, priority_rank=priority_map.get(name),
                    status="fixed", current_value=current,
                    notes="no bounds available — add to explicit_param_bounds",
                ))
                continue
            lo, hi = inferred

        kind, spec, grid = _make_param_spec(
            param_name=name, current_value=current,
            lo=float(lo), hi=float(hi), intensity_name=policy.name,
        )
        param_specs.append(spec)

        manifest_rows.append(AdaptiveParamManifestRow(
            param_name=name,
            priority_rank=priority_map.get(name),
            status="OPTIMIZED",
            current_value=current,
            lo=float(lo), hi=float(hi),
            kind=kind,
            inferred_step=infer_param_step(name, float(lo), float(hi), kind, policy.name),
            grid=grid,
        ))

    def _sort_key(row: AdaptiveParamManifestRow):
        status_rank = {"OPTIMIZED": 0, "fixed": 1, "unsupported": 2, "non-optimizable": 3}.get(row.status, 9)
        pr = row.priority_rank if row.priority_rank is not None else 9999
        return (status_rank, pr, row.param_name)

    manifest_rows.sort(key=_sort_key)

    return (
        SearchSpaceSpec(template_id=strategy_id, params=param_specs),
        manifest_rows,
        policy,
    )
