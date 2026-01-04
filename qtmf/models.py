from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class TradeIntent:
    """Input contract between Gaussian (or any caller) and QTMF.

    Keep this small and stable: callers can pass more fields via `extras`.
    """

    symbol: str
    action: str
    gaussian_confidence: float
    current_price: float

    # Optional context fields
    direction_probs: Optional[Sequence[float]] = None  # [down, neutral, up] (from Mamba/Neural)
    vix: Optional[float] = None
    ivr: Optional[float] = None
    realized_vol: Optional[float] = None
    mtf_snapshot: Optional[Dict[str, Any]] = None
    
    # Advanced Indicator Fields (9-Factor Fuzzy)
    rsi: Optional[float] = None
    adx: Optional[float] = None
    bb_position: Optional[float] = None
    bb_width: Optional[float] = None
    stoch_k: Optional[float] = None
    volume_ratio: Optional[float] = None
    sma_distance: Optional[float] = None
    
    # Neural Market State
    neural_forecast: Optional[Dict[str, Any]] = None  # {regime: 'volatile', trend: 'bull', conf: 0.8}

    # Optional suggested ceiling from caller (e.g., RL); QTMF can clamp
    suggested_total_qty: Optional[int] = None

    extras: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # drop None to reduce noise
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class SizingPlan:
    """Sizing output returned to the caller.

    For now we support wing-level weighting (put_qty/call_qty).
    Later we can extend with `leg_qty` if you decide to support ratio legs.
    """

    approved: bool
    reason: str

    total_qty: int
    put_qty: int
    call_qty: int

    put_weight: float = 0.5
    call_weight: float = 0.5

    # Optional structure metadata (expiry, strikes, credit, max loss, etc.)
    recipe: Optional[Dict[str, Any]] = None

    # Diagnostics for logging
    diagnostics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}
