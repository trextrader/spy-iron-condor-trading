"""
Configuration Router
Phase 6.1 - Core Infrastructure
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from gui.backend.schemas.config import (
    FullConfig,
    ConfigResponse,
    ToggleComponentRequest,
    ToggleComponentResponse,
    ConfigHistoryResponse,
    IntelligenceConfig,
)
from gui.backend.services.config_engine import ConfigManager

router = APIRouter()

# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Dependency to get config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


@router.get("/current", response_model=ConfigResponse)
async def get_current_config(
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get the current system configuration."""
    config = config_manager.get_current_config()
    config_hash = config_manager.get_config_hash(config)
    return ConfigResponse(config=config, config_hash=config_hash)


@router.post("/update", response_model=ConfigResponse)
async def update_config(
    new_config: FullConfig,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Update the full system configuration."""
    config_manager.set_config(new_config)
    config_hash = config_manager.get_config_hash(new_config)
    return ConfigResponse(config=new_config, config_hash=config_hash)


@router.post("/toggle", response_model=ToggleComponentResponse)
async def toggle_component(
    request: ToggleComponentRequest,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Toggle a single component on/off."""
    try:
        success = config_manager.toggle_component(
            request.component_path,
            request.enabled
        )
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to toggle component: {request.component_path}"
            )

        new_config = config_manager.get_current_config()
        new_hash = config_manager.get_config_hash(new_config)

        return ToggleComponentResponse(
            success=True,
            new_config_hash=new_hash,
            component_path=request.component_path,
            new_value=request.enabled
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/history", response_model=ConfigHistoryResponse)
async def get_config_history(
    limit: int = 50,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get configuration change history."""
    history = config_manager.get_history(limit=limit)
    return ConfigHistoryResponse(entries=history, total=len(history))


@router.get("/intelligence", response_model=IntelligenceConfig)
async def get_intelligence_config(
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get current intelligence layer configuration."""
    config = config_manager.get_current_config()
    return config.intelligence


@router.post("/intelligence/update", response_model=ConfigResponse)
async def update_intelligence_config(
    intelligence_config: IntelligenceConfig,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Update intelligence layer configuration."""
    config = config_manager.get_current_config()
    config.intelligence = intelligence_config
    config_manager.set_config(config)
    config_hash = config_manager.get_config_hash(config)
    return ConfigResponse(config=config, config_hash=config_hash)


@router.get("/hash/{config_hash}")
async def get_config_by_hash(
    config_hash: str,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get a specific configuration by its hash."""
    config = config_manager.get_config_by_hash(config_hash)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Configuration not found: {config_hash}"
        )
    return ConfigResponse(config=config, config_hash=config_hash)
