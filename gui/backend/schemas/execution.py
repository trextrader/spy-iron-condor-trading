"""
Execution Reality Schemas
Phase 6.3 - Execution Reality Engine Panel
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class FillStatus(str, Enum):
    """Fill status from execution simulation."""
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    QUEUED = "queued"


# =============================================================================
# SIMULATION SCHEMAS
# =============================================================================

class MarketStateInput(BaseModel):
    """Market state for execution simulation."""
    vix: float = 15.0
    spot_price: float = 500.0
    recent_volume: float = 100.0
    time_of_day_hour: float = 12.0
    recent_price_change_pct: float = 0.0
    quote_age_ms: float = 50.0
    market_speed: float = 1.0


class ExecutionSimulateRequest(BaseModel):
    """Request to simulate a fill."""
    bid: float
    ask: float
    size: int
    market_state: MarketStateInput
    is_entry: bool = True
    seed: Optional[int] = None
    config_override: Optional[Dict[str, Any]] = None


class ExecutionDiagnostics(BaseModel):
    """Diagnostics from execution simulation."""
    quote_valid: bool
    quote_reliability: float
    spread_multiplier: float
    liquidity_multiplier: float
    latency_ms: float
    drift_pct: float
    raw_fill: float
    tick_rounded: float
    slippage: float
    fill_probability: float
    rejection_reason: Optional[str] = None
    shock_detected: bool = False


class ExecutionSimulateResponse(BaseModel):
    """Response from execution simulation."""
    status: FillStatus
    fill_price: float
    fill_qty: int
    fill_time_ms: float
    slippage: float
    diagnostics: ExecutionDiagnostics


# =============================================================================
# PROFILE SCHEMAS
# =============================================================================

class ExecutionProfile(BaseModel):
    """Named execution reality profile."""
    id: str
    name: str
    description: str
    created_at: datetime
    config: Dict[str, Any]
    is_default: bool = False


class ExecutionProfileListResponse(BaseModel):
    """Response containing list of profiles."""
    profiles: List[ExecutionProfile]
    active_profile_id: Optional[str] = None


class SaveProfileRequest(BaseModel):
    """Request to save a new profile."""
    name: str
    description: str
    config: Dict[str, Any]
    set_as_default: bool = False


class SaveProfileResponse(BaseModel):
    """Response after saving profile."""
    profile_id: str
    success: bool
    message: str
