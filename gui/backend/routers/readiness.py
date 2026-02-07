"""
Readiness Router
Phase 6.1 - Core Infrastructure

Health and readiness checks for deployment validation.
"""

from fastapi import APIRouter
from typing import Dict, Any
import sys
from pathlib import Path

from gui.backend.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check - is the service running?"""
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check():
    """Detailed readiness check - are all dependencies available?"""
    checks: Dict[str, Any] = {
        "status": "ready",
        "checks": {},
    }

    # Check project root exists
    checks["checks"]["project_root"] = {
        "status": "ok" if settings.project_root.exists() else "fail",
        "path": str(settings.project_root),
    }

    # Check intelligence directory
    checks["checks"]["intelligence_dir"] = {
        "status": "ok" if settings.intelligence_dir.exists() else "warning",
        "path": str(settings.intelligence_dir),
    }

    # Check data directory
    data_dir = settings.project_root / "data"
    checks["checks"]["data_dir"] = {
        "status": "ok" if data_dir.exists() else "warning",
        "path": str(data_dir),
    }

    # Check models directory
    models_dir = settings.project_root / "models"
    checks["checks"]["models_dir"] = {
        "status": "ok" if models_dir.exists() else "warning",
        "path": str(models_dir),
    }

    # Check if any required check failed
    failed_checks = [
        k for k, v in checks["checks"].items()
        if v.get("status") == "fail"
    ]
    if failed_checks:
        checks["status"] = "not_ready"

    return checks


@router.get("/version")
async def version_info():
    """Get version and environment information."""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "python_version": sys.version,
        "environment": "development" if settings.debug else "production",
        "exec_reality_enabled": settings.exec_reality_enabled,
    }


@router.get("/dependencies")
async def check_dependencies():
    """Check availability of key dependencies."""
    deps = {}

    # Check torch
    try:
        import torch
        deps["torch"] = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    except ImportError:
        deps["torch"] = {"available": False}

    # Check numpy
    try:
        import numpy
        deps["numpy"] = {"available": True, "version": numpy.__version__}
    except ImportError:
        deps["numpy"] = {"available": False}

    # Check pandas
    try:
        import pandas
        deps["pandas"] = {"available": True, "version": pandas.__version__}
    except ImportError:
        deps["pandas"] = {"available": False}

    return {"dependencies": deps}
