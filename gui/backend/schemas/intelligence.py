"""
Intelligence Layer Schemas
Phase 6.2 - Intelligence Control Matrix
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# =============================================================================
# COMPONENT INFO SCHEMAS
# =============================================================================

class ComponentInfo(BaseModel):
    """Information about a single component."""
    id: str
    name: str
    description: str
    enabled: bool
    category: str  # model_core, fuzzy, indicators, diffusion
    params: Optional[Dict[str, Any]] = None
    dependencies: List[str] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)


class ComponentDependency(BaseModel):
    """Dependency relationship between components."""
    source: str
    target: str
    type: str  # requires, enables, conflicts


class IntelligenceState(BaseModel):
    """Current state of all intelligence components."""
    components: List[ComponentInfo]
    dependencies: List[ComponentDependency]
    active_count: int
    total_count: int


# =============================================================================
# MODEL LOGIC SCHEMAS
# =============================================================================

class PredicateInfo(BaseModel):
    """Information about a predicate."""
    id: int
    name: str
    description: Optional[str] = None
    feature_indices: List[int]
    threshold: Optional[float] = None


class RuleInfo(BaseModel):
    """Information about a rule."""
    id: str
    name: str
    dsl: str
    category: str  # entry, exit, sizing, filter


class ModelLogic(BaseModel):
    """Extracted model logic information."""
    model_id: str
    predicates: List[PredicateInfo]
    rules: List[RuleInfo]
    fuzzy_components: List[str]
    feature_names: List[str]
    input_dim: int
    param_count: int


# =============================================================================
# INTROSPECTION SCHEMAS
# =============================================================================

class PredicateActivation(BaseModel):
    """Predicate activation at a point in time."""
    timestamp: datetime
    predicate_id: int
    activation: float  # 0-1
    truth_value: bool


class FuzzyMembership(BaseModel):
    """Fuzzy set membership at a point in time."""
    timestamp: datetime
    component_id: str
    set_name: str
    membership: float  # 0-1
    contribution: float


class ETDState(BaseModel):
    """ETD hidden state at a point in time."""
    timestamp: datetime
    state_vector: List[float]
    derivative: List[float]


class LossComponent(BaseModel):
    """Individual loss component value."""
    name: str
    value: float
    weight: float
    weighted_value: float


class LossBreakdown(BaseModel):
    """Complete loss breakdown."""
    total_loss: float
    components: List[LossComponent]


# =============================================================================
# API RESPONSE SCHEMAS
# =============================================================================

class IntelligenceStateResponse(BaseModel):
    """Response containing intelligence state."""
    state: IntelligenceState
    config_hash: str


class ModelLogicResponse(BaseModel):
    """Response containing model logic."""
    logic: ModelLogic
    model_path: str


class IntrospectionDataResponse(BaseModel):
    """Response containing introspection data."""
    run_id: str
    start_time: datetime
    end_time: datetime
    predicates: Optional[List[PredicateActivation]] = None
    fuzzy: Optional[List[FuzzyMembership]] = None
    etd_states: Optional[List[ETDState]] = None
    loss: Optional[LossBreakdown] = None
