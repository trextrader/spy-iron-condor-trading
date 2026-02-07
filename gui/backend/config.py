"""
CondorBrain GUI Backend Configuration
Phase 6.1 - Core Infrastructure
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "CondorBrain GUI"
    app_version: str = "0.1.0"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent  # SPYOptionTrader_test
    models_dir: Path = project_root / "models"
    data_dir: Path = project_root / "data"
    reports_dir: Path = project_root / "reports"
    configs_dir: Path = project_root / "gui" / "configs"
    runs_dir: Path = project_root / "gui" / "runs"

    # Intelligence Module Paths
    intelligence_dir: Path = project_root / "intelligence"
    kaggle_dir: Path = project_root / "kaggle"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Default Backtest Settings
    default_seed: int = 42
    default_batch_size: int = 512
    default_device: str = "cpu"

    # Execution Reality Defaults
    exec_reality_enabled: bool = True
    exec_reality_latency_ms: float = 100.0
    exec_reality_aggression: float = 0.5

    class Config:
        env_prefix = "CONDOR_"
        env_file = ".env"


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.configs_dir.mkdir(parents=True, exist_ok=True)
settings.runs_dir.mkdir(parents=True, exist_ok=True)
