"""
Backtest Schemas
Phase 6.1 - Core Infrastructure
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class BacktestStatus(str, Enum):
    """Backtest execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeviceType(str, Enum):
    """Compute device type."""
    CPU = "cpu"
    GPU = "gpu"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class BacktestRequest(BaseModel):
    """Request to run a backtest."""
    tape_id: str
    model_id: str
    seed: int = 42
    device: DeviceType = DeviceType.CPU
    batch_size: int = 512
    limit: Optional[int] = None  # Limit number of bars

    # Optional config overrides
    intelligence_config: Optional[Dict[str, Any]] = None
    execution_reality_config: Optional[Dict[str, Any]] = None

    # Toggles
    use_execution_reality: bool = True
    use_fuzzy_sizing: bool = False
    use_trade_rules: bool = True


class ReplayRequest(BaseModel):
    """Request to replay a previous backtest."""
    run_id: Optional[str] = None
    config_hash: Optional[str] = None


# =============================================================================
# RESULT SCHEMAS
# =============================================================================

class EquityCurvePoint(BaseModel):
    """Single point on equity curve."""
    timestamp: datetime
    equity: float
    cash: float
    open_pnl: float
    open_count: int


class TradeSummary(BaseModel):
    """Summary of a single trade."""
    trade_id: str
    entry_dt: datetime
    exit_dt: Optional[datetime] = None
    pnl: float
    pnl_pct: float
    reason: Optional[str] = None
    legs: Optional[Dict[str, Any]] = None


class BacktestMetrics(BaseModel):
    """Metrics from a backtest run."""
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    npdd: float = 0.0
    turnover: float = 0.0
    avg_hold_time: float = 0.0

    # Legacy aliases for compatibility
    net_pnl: Optional[float] = None
    npdd_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_trade_pnl: Optional[float] = None
    avg_trade_duration: Optional[float] = None


class BacktestProgress(BaseModel):
    """Progress update during backtest execution."""
    run_id: str
    status: BacktestStatus
    progress_pct: float
    bars_processed: int
    total_bars: int
    current_equity: float
    current_pnl: float
    trades_so_far: int
    eta_seconds: Optional[float] = None


class BacktestResult(BaseModel):
    """Complete backtest result."""
    run_id: str
    status: BacktestStatus
    config_hash: str = ""

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Input summary
    tape_id: str = ""
    model_id: str = ""
    seed: int = 42
    device: str = "cpu"
    total_bars: int = 0

    # Results
    metrics: Optional[BacktestMetrics] = None
    equity_curve: Optional[List[EquityCurvePoint]] = None
    trades: Optional[List[TradeSummary]] = None

    # Determinism
    determinism_verified: bool = False
    replay_fingerprint: Optional[str] = None

    # Error info (if failed)
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class BacktestStartResponse(BaseModel):
    """Response when starting a backtest."""
    run_id: str
    config_hash: str
    status: BacktestStatus
    message: str


class BacktestStatusResponse(BaseModel):
    """Response for backtest status check."""
    run_id: str
    status: BacktestStatus
    progress: Optional[BacktestProgress] = None
    result: Optional[BacktestResult] = None


class ReplayResponse(BaseModel):
    """Response from replay operation."""
    original_run_id: str
    replay_run_id: str
    determinism_verified: bool
    diff_fingerprint: str
    differences: Optional[List[str]] = None
