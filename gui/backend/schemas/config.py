"""
Configuration Schemas
Phase 6.1 - Core Infrastructure
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# =============================================================================
# INTELLIGENCE CONFIG SCHEMAS
# =============================================================================

class ModelCoreConfig(BaseModel):
    """Model core component toggles."""
    condor_net: bool = True
    predicate_layer: bool = True
    etd: bool = True
    cde: bool = True
    super_sets: bool = True
    group_invariance: bool = True


class FuzzyComponentConfig(BaseModel):
    """Fuzzy engine component configuration."""
    enabled: bool = True
    components: Dict[str, bool] = Field(
        default_factory=lambda: {
            "component_1": True,  # Model-driven
            "component_2": True,
            "component_3": True,
            "component_4": True,
            "component_5": True,
            "component_6": True,
            "component_7": True,
            "component_8": True,
            "component_9": True,
            "component_10": True,
            "component_11": True,
        }
    )


class IndicatorParams(BaseModel):
    """Parameters for a dynamic indicator."""
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)


class DynamicIndicatorsConfig(BaseModel):
    """Dynamic indicator toggles and parameters."""
    bb: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(
            enabled=True,
            params={"period": 20, "std": 2.0, "offset": 0}
        )
    )
    rsi: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(
            enabled=True,
            params={"length": 14}
        )
    )
    atr: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(
            enabled=True,
            params={"window": 14}
        )
    )
    macd: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(
            enabled=True,
            params={"fast": 12, "slow": 26, "signal": 9}
        )
    )
    manifold_volatility: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(enabled=True)
    )
    tda_signature: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(enabled=True)
    )
    realized_vol: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(enabled=True)
    )
    skew: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(enabled=True)
    )
    momentum_primitives: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(enabled=True)
    )
    topology_primitives: IndicatorParams = Field(
        default_factory=lambda: IndicatorParams(enabled=True)
    )


class DiffusionPredictorConfig(BaseModel):
    """Diffusion predictor configuration."""
    predictors: Dict[str, bool] = Field(
        default_factory=lambda: {
            "predictor_1": True,
            "predictor_2": True,
            "predictor_3": True,
            "predictor_4": True,
            "predictor_5": True,
            "predictor_6": True,
            "predictor_7": True,
        }
    )
    auto_select_best: bool = True


class IntelligenceConfig(BaseModel):
    """Complete intelligence layer configuration."""
    model_core: ModelCoreConfig = Field(default_factory=ModelCoreConfig)
    fuzzy: FuzzyComponentConfig = Field(default_factory=FuzzyComponentConfig)
    indicators: DynamicIndicatorsConfig = Field(default_factory=DynamicIndicatorsConfig)
    diffusion: DiffusionPredictorConfig = Field(default_factory=DiffusionPredictorConfig)


# =============================================================================
# EXECUTION REALITY CONFIG SCHEMAS
# =============================================================================

class LatencyModelConfig(BaseModel):
    """Latency model configuration."""
    enabled: bool = True
    mean_latency_ms: float = 100.0
    std_latency_ms: float = 30.0
    min_latency_ms: float = 10.0
    drift_model: str = "linear"  # linear, exponential, none


class QueuePositionConfig(BaseModel):
    """Queue position model configuration."""
    enabled: bool = True
    aggression_factor: float = 0.5
    partial_fill_enabled: bool = True


class SpreadDynamicsConfig(BaseModel):
    """Spread dynamics model configuration."""
    enabled: bool = True
    base_spread_ratio: float = 0.02
    vix_sensitivity: float = 0.8
    tod_curve: str = "default"  # default, aggressive, conservative


class VolatilityShockConfig(BaseModel):
    """Volatility shock model configuration."""
    enabled: bool = True
    shock_threshold_annual: float = 50.0
    avoid_entry_on_shock: bool = True


class BrokenSpreadConfig(BaseModel):
    """Broken spread detector configuration."""
    enabled: bool = True
    max_spread_ratio: float = 0.15
    min_price: float = 0.01


class MicrostructureConfig(BaseModel):
    """Microstructure model configuration."""
    enabled: bool = True
    tick_size_below_3: float = 0.01
    tick_size_above_3: float = 0.05
    rounding_mode: str = "nearest"  # nearest, up, down


class QuoteStalenessConfig(BaseModel):
    """Quote staleness model configuration."""
    enabled: bool = True
    max_reliable_age_ms: float = 500.0
    market_speed_sensitivity: float = 1.0


class TODLiquidityConfig(BaseModel):
    """Time of day liquidity model configuration."""
    enabled: bool = True
    profile: str = "default"  # default, open_heavy, close_heavy
    multipliers: Dict[str, float] = Field(
        default_factory=lambda: {
            "open": 0.6,
            "mid_morning": 0.9,
            "midday": 0.8,
            "afternoon": 1.0,
            "pre_close": 1.2,
            "close": 0.5,
        }
    )


class ExecutionRealityConfig(BaseModel):
    """Complete execution reality configuration."""
    enabled: bool = True
    seed: int = 42
    latency: LatencyModelConfig = Field(default_factory=LatencyModelConfig)
    queue: QueuePositionConfig = Field(default_factory=QueuePositionConfig)
    spread: SpreadDynamicsConfig = Field(default_factory=SpreadDynamicsConfig)
    shock: VolatilityShockConfig = Field(default_factory=VolatilityShockConfig)
    broken_spread: BrokenSpreadConfig = Field(default_factory=BrokenSpreadConfig)
    microstructure: MicrostructureConfig = Field(default_factory=MicrostructureConfig)
    staleness: QuoteStalenessConfig = Field(default_factory=QuoteStalenessConfig)
    tod_liquidity: TODLiquidityConfig = Field(default_factory=TODLiquidityConfig)


# =============================================================================
# FULL CONFIG SCHEMA
# =============================================================================

class FullConfig(BaseModel):
    """Complete system configuration."""
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    config_hash: Optional[str] = None

    # Backtest settings
    seed: int = 42
    device: str = "cpu"
    batch_size: int = 512

    # Data sources
    tape_id: Optional[str] = None
    model_id: Optional[str] = None

    # Component configs
    intelligence: IntelligenceConfig = Field(default_factory=IntelligenceConfig)
    execution_reality: ExecutionRealityConfig = Field(default_factory=ExecutionRealityConfig)


# =============================================================================
# API REQUEST/RESPONSE SCHEMAS
# =============================================================================

class ConfigResponse(BaseModel):
    """Response containing current configuration."""
    config: FullConfig
    config_hash: str


class ToggleComponentRequest(BaseModel):
    """Request to toggle a component."""
    component_path: str  # e.g., "intelligence.fuzzy.components.component_3"
    enabled: bool


class ToggleComponentResponse(BaseModel):
    """Response after toggling a component."""
    success: bool
    new_config_hash: str
    component_path: str
    new_value: bool


class ConfigHistoryEntry(BaseModel):
    """Entry in configuration history."""
    config_hash: str
    timestamp: datetime
    description: Optional[str] = None
    changes: Optional[Dict[str, Any]] = None


class ConfigHistoryResponse(BaseModel):
    """Response containing configuration history."""
    entries: List[ConfigHistoryEntry]
    total: int
