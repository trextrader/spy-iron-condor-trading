"""
tensor_contracts.py
===================
Lightweight dtype/shape assertion helpers for GPU tensor operations.

Intended to be imported by gpu_strike_selector.py and the bar-loop engine
to enforce static dtype/shape contracts at runtime.

Design goals:
  - Zero overhead when disabled (set TENSOR_CONTRACTS_DISABLE=1)
  - Clear error messages: name, expected dtype, actual dtype
  - Composable: single-field helpers compose into bundle checks

Environment
-----------
TENSOR_CONTRACTS_DISABLE=1   Skip all assertions (production hot-path mode).
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# Set TENSOR_CONTRACTS_DISABLE=1 to skip assertions in production hot-paths.
_ENABLED = os.environ.get("TENSOR_CONTRACTS_DISABLE", "0") != "1"

_INT_DTYPES = frozenset({
    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
})


# ── Single-field helpers ───────────────────────────────────────────────────────

def assert_1d(name: str, t: torch.Tensor) -> None:
    """Assert tensor is exactly 1-dimensional."""
    if _ENABLED and t.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(t.shape)}")


def assert_dtype(name: str, t: torch.Tensor, dtype: torch.dtype) -> None:
    """Assert tensor has a specific dtype."""
    if _ENABLED and t.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {t.dtype}")


def assert_float32(name: str, t: torch.Tensor) -> None:
    """Assert tensor is float32."""
    assert_dtype(name, t, torch.float32)


def assert_float64(name: str, t: torch.Tensor) -> None:
    """Assert tensor is float64."""
    assert_dtype(name, t, torch.float64)


def assert_bool(name: str, t: torch.Tensor) -> None:
    """Assert tensor is bool."""
    assert_dtype(name, t, torch.bool)


def assert_int(name: str, t: torch.Tensor) -> None:
    """Assert tensor has any integer dtype (int8/16/32/64, uint8)."""
    if _ENABLED and t.dtype not in _INT_DTYPES:
        raise TypeError(f"{name} must be an integer dtype, got {t.dtype}")


# ── Bundle checks ──────────────────────────────────────────────────────────────

def assert_chain_tensors(
    chain_right:  torch.Tensor,              # [M] int   (0=call 1=put)
    chain_strike: torch.Tensor,              # [M] float32
    chain_dte:    torch.Tensor,              # [M] float32
    chain_bid:    torch.Tensor,              # [M] float32
    chain_ask:    torch.Tensor,              # [M] float32
    *,
    chain_delta:  Optional[torch.Tensor] = None,  # [M] float32 (may be NaN)
) -> None:
    """
    Assert dtype and 1D shape contracts for option chain tensors.

    chain_right   must be any integer dtype (int8 is typical)
    chain_strike,
    chain_dte,
    chain_bid,
    chain_ask     must be float32 and 1D
    chain_delta   if provided, must be float32 and 1D (NaN values are allowed)
    """
    if not _ENABLED:
        return
    assert_1d("chain_right", chain_right)
    assert_int("chain_right", chain_right)
    for name, t in [
        ("chain_strike", chain_strike),
        ("chain_dte",    chain_dte),
        ("chain_bid",    chain_bid),
        ("chain_ask",    chain_ask),
    ]:
        assert_1d(name, t)
        assert_float32(name, t)
    if chain_delta is not None:
        assert_1d("chain_delta", chain_delta)
        assert_float32("chain_delta", chain_delta)


def assert_candidate_params(
    target_dte:   torch.Tensor,   # [K] float32
    short_delta:  torch.Tensor,   # [K] float32
    spread_width: torch.Tensor,   # [K] float32
) -> None:
    """Assert dtype and 1D shape contracts for candidate parameter tensors."""
    if not _ENABLED:
        return
    for name, t in [
        ("target_dte",   target_dte),
        ("short_delta",  short_delta),
        ("spread_width", spread_width),
    ]:
        assert_1d(name, t)
        assert_float32(name, t)


# ── State tensor contracts ─────────────────────────────────────────────────────

_STATE_FLOAT64 = frozenset({
    "equity", "peak", "max_dd", "entry_credit",
    "entry_ss_call", "entry_ss_put", "entry_width", "entry_dte",
    "gross_win", "gross_loss",
})
_STATE_INT32 = frozenset({
    "entry_qty", "entry_bar", "last_entry", "wins", "losses",
})
_STATE_BOOL = frozenset({"open_mask"})


def assert_state_tensors(**named_tensors: torch.Tensor) -> None:
    """
    Assert dtype contracts for optimizer state tensors.

    float64 : equity, peak, max_dd, entry_credit,
               entry_ss_call, entry_ss_put, entry_width, entry_dte,
               gross_win, gross_loss
    int32   : entry_qty, entry_bar, last_entry, wins, losses
    bool    : open_mask
    """
    if not _ENABLED:
        return
    for name, t in named_tensors.items():
        if name in _STATE_FLOAT64:
            assert_float64(name, t)
        elif name in _STATE_INT32:
            if t.dtype != torch.int32:
                raise TypeError(f"{name} must be int32, got {t.dtype}")
        elif name in _STATE_BOOL:
            assert_bool(name, t)
        else:
            raise ValueError(
                f"Unknown state tensor '{name}'. Add it to _STATE_FLOAT64 / "
                f"_STATE_INT32 / _STATE_BOOL in tensor_contracts.py"
            )
