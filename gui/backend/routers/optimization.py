"""
Optimization Router
Phase 6.1 - Core Infrastructure
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import uuid
from datetime import datetime

from gui.backend.schemas.optimization import (
    OptimizationRequest,
    OptimizationStartResponse,
    OptimizationStatusResponse,
    OptimizationStatus,
)

router = APIRouter()

# In-memory storage for optimization runs (replace with proper storage later)
_optimization_runs: dict = {}


@router.post("/run", response_model=OptimizationStartResponse)
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
):
    """Start an optimization run."""
    optimization_id = f"opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Store run info
    _optimization_runs[optimization_id] = {
        "status": OptimizationStatus.RUNNING,
        "request": request.model_dump(),
        "started_at": datetime.utcnow().isoformat(),
        "progress": 0,
        "best_score": None,
    }

    # TODO: Launch actual optimization in background
    # background_tasks.add_task(run_optimization, optimization_id, request)

    return OptimizationStartResponse(
        optimization_id=optimization_id,
        status=OptimizationStatus.RUNNING,
        message="Optimization started",
        estimated_duration_seconds=request.max_evaluations * 30,  # Rough estimate
    )


@router.get("/status/{optimization_id}", response_model=OptimizationStatusResponse)
async def get_optimization_status(optimization_id: str):
    """Get the status of an optimization run."""
    if optimization_id not in _optimization_runs:
        raise HTTPException(status_code=404, detail=f"Optimization not found: {optimization_id}")

    run = _optimization_runs[optimization_id]
    return OptimizationStatusResponse(
        optimization_id=optimization_id,
        status=run["status"],
        progress=None,  # TODO: Build proper progress object
        result=None,  # TODO: Build result if completed
    )


@router.get("/list")
async def list_optimizations(
    limit: int = 50,
    status_filter: Optional[OptimizationStatus] = None,
):
    """List optimization runs."""
    runs = list(_optimization_runs.values())
    if status_filter:
        runs = [r for r in runs if r["status"] == status_filter]
    return {
        "runs": runs[:limit],
        "total": len(runs),
    }


@router.delete("/{optimization_id}")
async def cancel_optimization(optimization_id: str):
    """Cancel a running optimization."""
    if optimization_id not in _optimization_runs:
        raise HTTPException(status_code=404, detail=f"Optimization not found: {optimization_id}")

    run = _optimization_runs[optimization_id]
    if run["status"] != OptimizationStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Optimization is not running")

    run["status"] = OptimizationStatus.CANCELLED
    return {"message": f"Optimization {optimization_id} cancelled"}
