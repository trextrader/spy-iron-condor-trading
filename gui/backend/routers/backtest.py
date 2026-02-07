"""
Backtest Router
Phase 6.1 - Core Infrastructure
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Optional
import uuid
from datetime import datetime

from gui.backend.schemas.backtest import (
    BacktestRequest,
    ReplayRequest,
    BacktestStartResponse,
    BacktestStatusResponse,
    BacktestResult,
    BacktestStatus,
    ReplayResponse,
)
from gui.backend.services.backtest_runner import BacktestRunner
from gui.backend.services.config_engine import ConfigManager

router = APIRouter()

# Global instances
_backtest_runner: Optional[BacktestRunner] = None
_config_manager: Optional[ConfigManager] = None


def get_backtest_runner() -> BacktestRunner:
    """Dependency to get backtest runner."""
    global _backtest_runner
    if _backtest_runner is None:
        _backtest_runner = BacktestRunner()
    return _backtest_runner


def get_config_manager() -> ConfigManager:
    """Dependency to get config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


@router.post("/run", response_model=BacktestStartResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    runner: BacktestRunner = Depends(get_backtest_runner),
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Start a new backtest run."""
    # Generate run ID
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Get or create config
    config = config_manager.get_current_config()
    config.tape_id = request.tape_id
    config.model_id = request.model_id
    config.seed = request.seed
    config.device = request.device.value
    config.batch_size = request.batch_size

    # Apply any overrides
    if request.intelligence_config:
        # Merge intelligence config overrides
        pass  # TODO: Implement merge logic

    if request.execution_reality_config:
        # Merge execution reality config overrides
        pass  # TODO: Implement merge logic

    config.execution_reality.enabled = request.use_execution_reality
    config_hash = config_manager.get_config_hash(config)

    # Start backtest in background
    background_tasks.add_task(
        runner.run_backtest,
        run_id=run_id,
        config=config,
        limit=request.limit,
    )

    return BacktestStartResponse(
        run_id=run_id,
        config_hash=config_hash,
        status=BacktestStatus.RUNNING,
        message="Backtest started successfully"
    )


@router.get("/status/{run_id}", response_model=BacktestStatusResponse)
async def get_backtest_status(
    run_id: str,
    runner: BacktestRunner = Depends(get_backtest_runner)
):
    """Get the status of a backtest run."""
    status = runner.get_status(run_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Backtest run not found: {run_id}"
        )

    progress = runner.get_progress(run_id)
    result = runner.get_result(run_id) if status == BacktestStatus.COMPLETED else None

    return BacktestStatusResponse(
        run_id=run_id,
        status=status,
        progress=progress,
        result=result
    )


@router.get("/results/{run_id}", response_model=BacktestResult)
async def get_backtest_results(
    run_id: str,
    runner: BacktestRunner = Depends(get_backtest_runner)
):
    """Get the full results of a completed backtest."""
    result = runner.get_result(run_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Backtest results not found: {run_id}"
        )
    return result


@router.post("/replay", response_model=ReplayResponse)
async def replay_backtest(
    request: ReplayRequest,
    background_tasks: BackgroundTasks,
    runner: BacktestRunner = Depends(get_backtest_runner),
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Replay a previous backtest to verify determinism."""
    if not request.run_id and not request.config_hash:
        raise HTTPException(
            status_code=400,
            detail="Either run_id or config_hash must be provided"
        )

    # Get original config
    if request.run_id:
        original_result = runner.get_result(request.run_id)
        if original_result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Original run not found: {request.run_id}"
            )
        config_hash = original_result.config_hash
        original_run_id = request.run_id
    else:
        config_hash = request.config_hash
        original_run_id = "unknown"

    config = config_manager.get_config_by_hash(config_hash)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Configuration not found: {config_hash}"
        )

    # Generate replay run ID
    replay_run_id = f"replay_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Run replay
    replay_result = await runner.run_replay(
        replay_run_id=replay_run_id,
        original_run_id=original_run_id,
        config=config
    )

    return ReplayResponse(
        original_run_id=original_run_id,
        replay_run_id=replay_run_id,
        determinism_verified=replay_result.get("verified", False),
        diff_fingerprint=replay_result.get("diff_fingerprint", "unknown"),
        differences=replay_result.get("differences")
    )


@router.get("/list")
async def list_backtests(
    limit: int = 50,
    status_filter: Optional[BacktestStatus] = None,
    runner: BacktestRunner = Depends(get_backtest_runner)
):
    """List recent backtest runs."""
    runs = runner.list_runs(limit=limit, status_filter=status_filter)
    return {"runs": runs, "total": len(runs)}


@router.delete("/{run_id}")
async def cancel_backtest(
    run_id: str,
    runner: BacktestRunner = Depends(get_backtest_runner)
):
    """Cancel a running backtest."""
    success = runner.cancel(run_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Could not cancel backtest: {run_id}"
        )
    return {"message": f"Backtest {run_id} cancelled"}
