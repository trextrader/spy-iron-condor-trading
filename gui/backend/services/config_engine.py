"""
Configuration Engine Service
Phase 6.1 - Core Infrastructure

Manages system configuration with versioning, hashing, and history.
"""

import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from copy import deepcopy

from gui.backend.schemas.config import FullConfig, IntelligenceConfig, ExecutionRealityConfig


class ConfigManager:
    """Manages system configuration with versioning and determinism guarantees."""

    def __init__(self):
        self._current_config: Optional[FullConfig] = None
        self._config_history: List[Dict[str, Any]] = []
        self._config_cache: Dict[str, FullConfig] = {}  # hash -> config

        # Initialize with default config
        self._current_config = self._create_default_config()
        self._store_config(self._current_config)

    def _create_default_config(self) -> FullConfig:
        """Create default configuration."""
        return FullConfig(
            tape_id="",
            model_id="",
            seed=42,
            device="cuda",
            batch_size=32,
            intelligence=IntelligenceConfig(
                condornet_enabled=True,
                fuzzy_enabled=True,
                indicators_enabled=True,
                diffusion_enabled=False,
                etd1_enabled=True,
                tft_enabled=True,
                cde_enabled=True,
                moe_enabled=True,
                model_id="",
                checkpoint_path="",
            ),
            execution_reality=ExecutionRealityConfig(
                enabled=True,
            ),
        )

    def get_current_config(self) -> FullConfig:
        """Get the current active configuration."""
        if self._current_config is None:
            self._current_config = self._create_default_config()
        return deepcopy(self._current_config)

    def set_config(self, config: FullConfig) -> str:
        """Set a new configuration and return its hash."""
        self._current_config = deepcopy(config)
        config_hash = self._store_config(config)
        self._add_history_entry(config_hash, "config_updated")
        return config_hash

    def toggle_component(self, component_path: str, enabled: bool) -> bool:
        """Toggle a specific component by its path (e.g., 'intelligence.condornet_enabled')."""
        if self._current_config is None:
            return False

        config_dict = self._current_config.model_dump()
        parts = component_path.split(".")

        # Navigate to parent
        current = config_dict
        for part in parts[:-1]:
            if part not in current:
                return False
            current = current[part]

        # Set the value
        final_key = parts[-1]
        if final_key not in current:
            return False

        current[final_key] = enabled

        # Update config
        self._current_config = FullConfig(**config_dict)
        self._store_config(self._current_config)
        self._add_history_entry(
            self.get_config_hash(self._current_config),
            f"toggled_{component_path}",
        )

        return True

    def get_config_hash(self, config: FullConfig) -> str:
        """Generate deterministic hash for a configuration."""
        # Convert to dict and sort keys for determinism
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def get_config_by_hash(self, config_hash: str) -> Optional[FullConfig]:
        """Retrieve a configuration by its hash."""
        return self._config_cache.get(config_hash)

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self._config_history[-limit:][::-1]

    def _store_config(self, config: FullConfig) -> str:
        """Store a config in the cache and return its hash."""
        config_hash = self.get_config_hash(config)
        self._config_cache[config_hash] = deepcopy(config)
        return config_hash

    def _add_history_entry(self, config_hash: str, action: str):
        """Add an entry to configuration history."""
        self._config_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "config_hash": config_hash,
            "action": action,
        })

    def export_config(self, config: Optional[FullConfig] = None) -> Dict[str, Any]:
        """Export configuration as a dictionary for JSON serialization."""
        cfg = config or self._current_config
        if cfg is None:
            return {}
        return cfg.model_dump()

    def import_config(self, config_dict: Dict[str, Any]) -> FullConfig:
        """Import configuration from a dictionary."""
        config = FullConfig(**config_dict)
        self.set_config(config)
        return config
