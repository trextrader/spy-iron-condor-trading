"""
strategy_param_priority.py — Strategy-Specific Parameter Importance Rankings
=============================================================================
Expert-prior ranked lists for every strategy. Used by the adaptive search
space builder to select the top-N parameters at a given intensity level.
Highest importance = index 0.
"""
from __future__ import annotations

from typing import Dict, List, Set


# Fields that should never be optimized (metadata / structural).
NON_OPTIMIZABLE_FIELDS: Set[str] = {
    "template_id", "class_name", "class_idx", "family", "legs",
}

# ── Common ranked lists (reused across structurally similar strategies) ──────

_IC_PRIORITY = [
    "short_delta", "target_dte", "spread_width", "profit_target", "stop_loss_dollar",
    "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
    "wing_delta", "call_offset_pct", "put_offset_pct", "margin_pct",
    "stop_loss_mult", "max_leg_spread",
]

_STRADDLE_PRIORITY = [
    "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
    "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
    "margin_pct", "stop_loss_mult", "max_leg_spread",
]

_STRANGLE_PRIORITY = [
    "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
    "call_offset_pct", "put_offset_pct", "max_dte_exit", "hold_days",
    "max_positions", "cooldown_bars", "max_contracts", "margin_pct",
    "stop_loss_mult", "max_leg_spread",
]

_BUTTERFLY_PRIORITY = [
    "spread_width", "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
    "hold_days", "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
    "margin_pct", "stop_loss_mult", "max_leg_spread",
]

_CALENDAR_PRIORITY = [
    "target_dte", "hold_days", "profit_target", "stop_loss_dollar",
    "spread_width", "max_dte_exit", "max_positions", "cooldown_bars",
    "max_contracts", "margin_pct", "stop_loss_mult", "max_leg_spread",
]

_SINGLE_CALL_PRIORITY = [
    "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
    "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
    "margin_pct", "stop_loss_mult",
]

_SINGLE_PUT_PRIORITY = [
    "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
    "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
    "margin_pct", "stop_loss_mult",
]

_SPREAD_CALL_PRIORITY = [
    "short_delta", "spread_width", "target_dte", "profit_target", "stop_loss_dollar",
    "hold_days", "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
    "margin_pct", "stop_loss_mult", "max_leg_spread",
]

_SPREAD_PUT_PRIORITY = [
    "short_delta", "spread_width", "target_dte", "profit_target", "stop_loss_dollar",
    "hold_days", "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
    "margin_pct", "stop_loss_mult", "max_leg_spread",
]

_RATIO_PRIORITY = [
    "spread_width", "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
    "hold_days", "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
]

_COMBO_PRIORITY = [
    "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
    "hold_days", "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
    "margin_pct", "stop_loss_mult",
]

_DIRECTIONAL_LONG_PRIORITY = [
    "target_dte", "hold_days", "profit_target", "stop_loss_dollar",
    "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
    "margin_pct", "stop_loss_mult",
]

# ── Per-strategy expert priority manifests ───────────────────────────────────

STRATEGY_PARAM_PRIORITY: Dict[str, List[str]] = {
    # Iron butterfly family
    "iron_butterfly": [
        "short_delta", "target_dte", "spread_width", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
        "wing_delta", "call_offset_pct", "put_offset_pct", "margin_pct",
        "stop_loss_mult", "max_leg_spread",
    ],
    "inverse_iron_butterfly": _IC_PRIORITY,
    "short_call_butterfly":   _BUTTERFLY_PRIORITY,
    "short_put_butterfly":    _BUTTERFLY_PRIORITY,
    "long_call_butterfly":    _BUTTERFLY_PRIORITY,
    "long_put_butterfly":     _BUTTERFLY_PRIORITY,
    "short_straddle":         _STRADDLE_PRIORITY,
    "straddle_long":          _STRADDLE_PRIORITY,
    "covered_short_straddle": _STRADDLE_PRIORITY,
    "short_guts":             _STRADDLE_PRIORITY,
    "guts":                   _STRADDLE_PRIORITY,
    "strap":                  _STRADDLE_PRIORITY,
    "strip":                  _STRADDLE_PRIORITY,

    # Iron condor family
    "iron_condor": [
        "short_delta", "target_dte", "spread_width", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
        "wing_delta", "call_offset_pct", "put_offset_pct", "margin_pct",
        "stop_loss_mult", "max_leg_spread",
    ],
    "inverse_iron_condor":    _IC_PRIORITY,
    "short_call_condor":      _IC_PRIORITY,
    "short_put_condor":       _IC_PRIORITY,
    "long_call_condor":       _IC_PRIORITY,
    "long_put_condor":        _IC_PRIORITY,
    "double_diagonal": [
        "target_dte", "hold_days", "spread_width", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
    ],
    "diagonal_call": [
        "target_dte", "hold_days", "spread_width", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
    ],
    "diagonal_put": [
        "target_dte", "hold_days", "spread_width", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "max_positions", "cooldown_bars", "max_contracts",
    ],
    "calendar_call":          _CALENDAR_PRIORITY,
    "calendar_put":           _CALENDAR_PRIORITY,
    "collar": [
        "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
        "margin_pct", "stop_loss_mult", "max_leg_spread",
    ],
    "jade_lizard": [
        "short_delta", "target_dte", "spread_width", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
        "call_offset_pct", "put_offset_pct", "wing_delta", "margin_pct",
        "stop_loss_mult", "max_leg_spread",
    ],
    "reverse_jade_lizard": [
        "short_delta", "target_dte", "spread_width", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
        "call_offset_pct", "put_offset_pct", "wing_delta", "margin_pct",
        "stop_loss_mult", "max_leg_spread",
    ],
    "covered_short_strangle": _STRANGLE_PRIORITY,
    "short_strangle":         _STRANGLE_PRIORITY,
    "strangle_long":          _STRANGLE_PRIORITY,

    # Short call family
    "short_call": [
        "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
    ],
    "covered_call":           _SINGLE_CALL_PRIORITY,
    "bear_call_ladder":       _SPREAD_CALL_PRIORITY,
    "bull_call_ladder":       _SPREAD_CALL_PRIORITY,
    "long_call":              _DIRECTIONAL_LONG_PRIORITY,
    "call_ratio_backspread":  _RATIO_PRIORITY,
    "call_ratio_spread":      _RATIO_PRIORITY,
    "call_broken_wing":       _SPREAD_CALL_PRIORITY,
    "inverse_call_broken_wing": _SPREAD_CALL_PRIORITY,
    "bear_call_spread_credit": _SPREAD_CALL_PRIORITY,
    "bull_call_spread":       _SPREAD_CALL_PRIORITY,
    "long_combo":             _COMBO_PRIORITY,
    "short_combo":            _COMBO_PRIORITY,
    "long_synthetic_future":  _COMBO_PRIORITY,
    "short_synthetic_future": _COMBO_PRIORITY,

    # Short put family
    "short_put": [
        "short_delta", "target_dte", "profit_target", "stop_loss_dollar",
        "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
    ],
    "cash_secured_put":       _SINGLE_PUT_PRIORITY,
    "bull_put_ladder":        _SPREAD_PUT_PRIORITY,
    "bear_put_ladder":        _SPREAD_PUT_PRIORITY,
    "long_put":               _DIRECTIONAL_LONG_PRIORITY,
    "put_ratio_backspread":   _RATIO_PRIORITY,
    "put_ratio_spread":       _RATIO_PRIORITY,
    "put_broken_wing":        _SPREAD_PUT_PRIORITY,
    "inverse_put_broken_wing": _SPREAD_PUT_PRIORITY,
    "bear_put_spread":        _SPREAD_PUT_PRIORITY,
    "bull_put_spread_credit": _SPREAD_PUT_PRIORITY,
    "protective_put":         _DIRECTIONAL_LONG_PRIORITY,
    "synthetic_put":          _DIRECTIONAL_LONG_PRIORITY,
}

# Default fallback for any strategy not explicitly listed above.
DEFAULT_PARAM_PRIORITY: List[str] = [
    "short_delta", "target_dte", "spread_width", "profit_target", "stop_loss_dollar",
    "max_dte_exit", "hold_days", "max_positions", "cooldown_bars", "max_contracts",
    "call_offset_pct", "put_offset_pct", "wing_delta", "margin_pct",
    "stop_loss_mult", "max_leg_spread",
]


def get_ranked_param_priority(
    strategy_id: str,
    available_param_names: List[str],
    non_optimizable_fields: Set[str] | None = None,
) -> List[str]:
    """
    Returns a ranked list filtered to the params that actually exist in
    the template config. Preserves expert-prior order, then appends any
    remaining unknown params at the end.
    """
    non_opt = set(non_optimizable_fields or set()) | NON_OPTIMIZABLE_FIELDS
    available = [p for p in available_param_names if p not in non_opt]

    base = STRATEGY_PARAM_PRIORITY.get(strategy_id, DEFAULT_PARAM_PRIORITY)
    ranked = [p for p in base if p in available]
    leftovers = [p for p in available if p not in ranked]
    ranked.extend(leftovers)
    return ranked


def select_top_priority_params(
    strategy_id: str,
    available_param_names: List[str],
    breadth_top_n: int,
    non_optimizable_fields: Set[str] | None = None,
) -> List[str]:
    ranked = get_ranked_param_priority(
        strategy_id=strategy_id,
        available_param_names=available_param_names,
        non_optimizable_fields=non_optimizable_fields,
    )
    if breadth_top_n >= len(ranked):
        return ranked
    return ranked[:breadth_top_n]
