"""
core/types.py

Typed DTOs for pipeline stages to keep modules decoupled.
Follows Quantor-MTFuzz architectural specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import datetime as dt
import pandas as pd


@dataclass(frozen=True)
class MarketSnapshot:
    """
    Time-aligned market state snapshot.
    
    Contains all data needed for strategy evaluation, sizing, and risk checks.
    Immutable to prevent accidental state mutation across pipeline stages.
    """
    ts: dt.datetime
    symbol: str
    spot: float
    bars: pd.DataFrame              # OHLCV bars (recent history)
    option_chain: pd.DataFrame      # Option quotes for current date
    vix: float | None = None
    es_price: float | None = None
    prev_close: float | None = None
    open_price: float | None = None
    
    # Options alignment metadata (optional; populated by DataEngine when available)
    option_used_ts: Any = None
    option_lag_sec: float | None = None
    option_iv_conf: float | None = None
    option_align_mode: str | None = None


@dataclass(frozen=True)
class TradeDecision:
    """
    Strategy evaluation output (signal gating only).
    
    Indicates whether to trade and structural intent.
    Does NOT include sizing - that's handled by FISSizer.
    """
    symbol: str
    should_trade: bool
    structure: str                  # "iron_condor" | "calendar" | "butterfly"
    bias: str                       # "neutral" | "bullish" | "bearish"
    rationale: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SizedDecision:
    """
    FIS sizing output.
    
    Includes final contract count, confidence scalar, and resolved leg specs.
    """
    decision: TradeDecision
    contracts: int
    confidence: float               # [0, 1] from FIS defuzzification
    risk_budget: float              # Max loss per contract * contracts
    legs: list[dict[str, Any]]      # Resolved strikes, deltas, greeks


@dataclass(frozen=True)
class Approval:
    """
    Risk manager approval/rejection.
    
    If approved, includes OrderPlan for execution.
    """
    approved: bool
    reason: str
    order_plan: Optional["OrderPlan"] = None


@dataclass(frozen=True)
class OrderPlan:
    """
    Execution plan for approved trade.
    
    Contains broker order objects or dicts for TradeRouter.
    """
    orders: list[Any]               # Broker order objects or dicts
    metadata: dict[str, Any] = field(default_factory=dict)
