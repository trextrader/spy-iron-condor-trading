"""
CondorBrain GUI Backend - Main Application
Phase 6.1 - Core Infrastructure

Entry point for the FastAPI server.

Usage:
    uvicorn gui.backend.main:app --reload --port 8000

Or:
    python -m gui.backend.main
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from gui.backend.config import settings
from gui.backend.routers import backtest, config, model, tape, trades, execution, optimization, readiness, websocket, training
from gui.backend.routers.websocket import get_ws_manager
from gui.backend.services.training_emitter import init_emitter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Project root: {settings.project_root}")
    print(f"Debug mode: {settings.debug}")

    # Verify critical paths
    if not settings.intelligence_dir.exists():
        print(f"WARNING: Intelligence directory not found: {settings.intelligence_dir}")
    if not settings.kaggle_dir.exists():
        print(f"WARNING: Kaggle directory not found: {settings.kaggle_dir}")

    # Initialize training emitter with WebSocket manager
    ws_manager = get_ws_manager()
    init_emitter(ws_manager)
    print("Training telemetry emitter initialized")

    yield

    # Shutdown
    print("Shutting down CondorBrain GUI Backend...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for CondorBrain/CondorNet Trading System GUI",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(config.router, prefix="/api/config", tags=["Configuration"])
app.include_router(model.router, prefix="/api/model", tags=["Model"])
app.include_router(tape.router, prefix="/api/tape", tags=["Tape"])
app.include_router(trades.router, prefix="/api/trades", tags=["Trades"])
app.include_router(execution.router, prefix="/api/execution", tags=["Execution"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization"])
app.include_router(readiness.router, prefix="/api/readiness", tags=["Readiness"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
app.include_router(training.router, prefix="/api/training", tags=["Training Diagnostics"])


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "project_root": str(settings.project_root),
        "intelligence_available": settings.intelligence_dir.exists(),
        "execution_reality_enabled": settings.exec_reality_enabled,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gui.backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
