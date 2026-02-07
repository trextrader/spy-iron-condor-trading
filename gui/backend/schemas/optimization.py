"""
Optimization Schemas
Phase 6.8 - Optimization Suite
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class OptimizationMode(str, Enum):
    """Optimization algorithm mode."""
    ABLATION = "ablation"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    CERTIFICATION = "certification"


class OptimizationObjective(str, Enum):
    """Optimization objective function."""
    NPDD = "npdd"
    SHARPE = "sharpe"
    DRAWDOWN = "dd"
    COMPOSITE = "composite"


class OptimizationStatus(str, Enum):
    """Optimization run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class OptimizationRequest(BaseModel):
    """Request to start optimization."""
    mode: OptimizationMode
    objective: OptimizationObjective
    max_evaluations: int = 100

    # Base configuration to optimize from
    base_config_hash: Optional[str] = None

    # Search space definition
    search_space: Dict[str, Any] = Field(
        default_factory=lambda: {
            "components_to_search": [],  # List of component paths
            "param_ranges": {},  # Dict of param_path -> (min, max)
        }
    )

    # Composite objective weights (if objective=composite)
    objective_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "npdd": 0.4,
            "sharpe": 0.3,
            "dd": 0.2,
            "turnover": 0.1,
        }
    )

    # Backtest settings
    tape_id: str = ""
    model_id: str = ""
    seed: int = 42


# =============================================================================
# RESULT SCHEMAS
# =============================================================================

class ConfigCandidate(BaseModel):
    """A candidate configuration from optimization."""
    config_hash: str
    score: float
    metrics: Dict[str, float]
    components_active: List[str]
    params: Dict[str, Any]
    evaluation_number: int
    is_best: bool = False


class ComponentImportance(BaseModel):
    """Importance score for a component."""
    component_path: str
    importance: float  # 0-1
    direction: str  # positive, negative, neutral
    confidence: float


class ComponentSynergy(BaseModel):
    """Synergy between two components."""
    component_a: str
    component_b: str
    synergy_score: float  # negative = conflict, positive = synergy
    confidence: float


class OptimizationProgress(BaseModel):
    """Progress update during optimization."""
    optimization_id: str
    status: OptimizationStatus
    progress_pct: float
    evaluations_completed: int
    total_evaluations: int
    best_score: float
    best_config_hash: str
    current_evaluation: Optional[ConfigCandidate] = None
    eta_seconds: Optional[float] = None


class OptimizationResult(BaseModel):
    """Complete optimization result."""
    optimization_id: str
    status: OptimizationStatus
    mode: OptimizationMode
    objective: OptimizationObjective

    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Results
    best_config_hash: str
    best_score: float
    best_metrics: Dict[str, float]
    all_candidates: List[ConfigCandidate]

    # Analysis
    component_importance: Optional[List[ComponentImportance]] = None
    synergies: Optional[List[ComponentSynergy]] = None

    # Error info (if failed)
    error_message: Optional[str] = None


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class OptimizationStartResponse(BaseModel):
    """Response when starting optimization."""
    optimization_id: str
    status: OptimizationStatus
    message: str
    estimated_duration_seconds: Optional[float] = None


class OptimizationStatusResponse(BaseModel):
    """Response for optimization status check."""
    optimization_id: str
    status: OptimizationStatus
    progress: Optional[OptimizationProgress] = None
    result: Optional[OptimizationResult] = None
