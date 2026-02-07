"""
Training Diagnostics Router
Phase 6.7 - Model Introspection / Training Visualization

Provides API endpoints for accessing training diagnostics data.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import asyncio

router = APIRouter()

# Path to diagnostics files (configurable)
DIAGNOSTICS_DIR = Path(__file__).parent.parent.parent.parent / "models" / "diagnostics"


# ============================================================================
# Live Training Telemetry (receives from training script, broadcasts to WS)
# ============================================================================

@router.post("/telemetry/step")
async def receive_training_step(data: Dict[str, Any]):
    """Receive training step from external training script and broadcast to WebSocket."""
    from gui.backend.routers.websocket import get_ws_manager
    manager = get_ws_manager()

    message = {
        "type": "data",
        "channel": "training.step",
        "data": data,
    }
    await manager.broadcast_to_channel("training.step", message)
    return {"status": "broadcast"}


@router.post("/telemetry/status")
async def receive_training_status(data: Dict[str, Any]):
    """Receive training status from external training script and broadcast to WebSocket."""
    from gui.backend.routers.websocket import get_ws_manager
    manager = get_ws_manager()

    message = {
        "type": "data",
        "channel": "training.status",
        "data": data,
    }
    await manager.broadcast_to_channel("training.status", message)
    return {"status": "broadcast"}


@router.post("/telemetry/epoch")
async def receive_epoch_summary(data: Dict[str, Any]):
    """Receive epoch summary from external training script and broadcast to WebSocket."""
    from gui.backend.routers.websocket import get_ws_manager
    manager = get_ws_manager()

    message = {
        "type": "data",
        "channel": "training.epoch",
        "data": data,
    }
    await manager.broadcast_to_channel("training.epoch", message)
    return {"status": "broadcast"}


@router.post("/telemetry/complete")
async def receive_training_complete(data: Dict[str, Any]):
    """Receive training complete from external training script and broadcast to WebSocket."""
    from gui.backend.routers.websocket import get_ws_manager
    manager = get_ws_manager()

    message = {
        "type": "data",
        "channel": "training.complete",
        "data": data,
    }
    await manager.broadcast_to_channel("training.complete", message)
    return {"status": "broadcast"}


@router.post("/telemetry/fuzzy")
async def receive_fuzzy_activations(data: Dict[str, Any]):
    """Receive fuzzy gate activations from external training script and broadcast to WebSocket."""
    from gui.backend.routers.websocket import get_ws_manager
    manager = get_ws_manager()

    message = {
        "type": "data",
        "channel": "training.fuzzy",
        "data": data,
    }
    await manager.broadcast_to_channel("training.fuzzy", message)
    return {"status": "broadcast"}


@router.get("/diagnostics/list")
async def list_diagnostics():
    """List available training diagnostic files."""
    if not DIAGNOSTICS_DIR.exists():
        return {"files": [], "total": 0}

    files = []
    for f in DIAGNOSTICS_DIR.glob("*.json"):
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            meta = data.get("metadata", {})
            files.append({
                "filename": f.name,
                "path": str(f),
                "start_time": meta.get("start_time"),
                "end_time": meta.get("end_time"),
                "total_steps": meta.get("total_steps", 0),
            })
        except Exception:
            continue

    return {"files": files, "total": len(files)}


@router.get("/diagnostics/{filename}")
async def get_diagnostics(filename: str):
    """Get full diagnostics data for a training run."""
    filepath = DIAGNOSTICS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Diagnostics not found: {filename}")

    with open(filepath, "r") as f:
        data = json.load(f)

    return data


@router.get("/diagnostics/{filename}/loss-trajectory")
async def get_loss_trajectory(
    filename: str,
    window: int = Query(200, ge=1, le=1000),
    components: Optional[str] = None,  # comma-separated
):
    """Get smoothed loss component trajectories."""
    filepath = DIAGNOSTICS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Diagnostics not found: {filename}")

    with open(filepath, "r") as f:
        data = json.load(f)

    loss_logs = data.get("loss_logs", [])
    if not loss_logs:
        return {"steps": [], "components": {}}

    import pandas as pd
    df = pd.DataFrame(loss_logs)

    # Apply smoothing
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    smooth = df[numeric_cols].rolling(window, min_periods=1).mean()

    # Filter components if specified
    if components:
        requested = [c.strip() for c in components.split(",")]
        numeric_cols = [c for c in numeric_cols if c in requested or c in ["step", "epoch"]]

    return {
        "steps": df["step"].tolist(),
        "components": {
            col: smooth[col].tolist()
            for col in numeric_cols
            if col not in ["step", "epoch"]
        },
    }


@router.get("/diagnostics/{filename}/fuzzy-heatmap")
async def get_fuzzy_heatmap(filename: str):
    """Get fuzzy gate activation heatmap data."""
    filepath = DIAGNOSTICS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Diagnostics not found: {filename}")

    with open(filepath, "r") as f:
        data = json.load(f)

    fuzzy_logs = data.get("fuzzy_logs", [])
    if not fuzzy_logs:
        return {"steps": [], "gates": [], "values": []}

    import pandas as pd
    df = pd.DataFrame(fuzzy_logs)
    fuzzy_cols = [c for c in df.columns if c.startswith("fuzzy_")]

    return {
        "steps": df["step"].tolist(),
        "gates": [c.replace("fuzzy_", "F") for c in fuzzy_cols],
        "values": df[fuzzy_cols].values.tolist(),
    }


@router.get("/diagnostics/{filename}/epoch-summary")
async def get_epoch_summary(filename: str):
    """Get per-epoch aggregated metrics."""
    filepath = DIAGNOSTICS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Diagnostics not found: {filename}")

    with open(filepath, "r") as f:
        data = json.load(f)

    loss_logs = data.get("loss_logs", [])
    if not loss_logs:
        return {"epochs": []}

    import pandas as pd
    df = pd.DataFrame(loss_logs)
    numeric_cols = [c for c in df.columns if c not in ["step", "epoch"]]

    epoch_df = df.groupby("epoch")[numeric_cols].mean().reset_index()

    return {"epochs": epoch_df.to_dict("records")}


@router.get("/diagnostics/{filename}/gate-distribution")
async def get_gate_distribution(
    filename: str,
    epoch: int,
    bins: int = Query(30, ge=5, le=100),
):
    """Get fuzzy gate activation distributions for a specific epoch."""
    filepath = DIAGNOSTICS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Diagnostics not found: {filename}")

    with open(filepath, "r") as f:
        data = json.load(f)

    fuzzy_logs = data.get("fuzzy_logs", [])
    if not fuzzy_logs:
        return {"gates": {}}

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(fuzzy_logs)
    sub = df[df["epoch"] == epoch]

    if len(sub) == 0:
        raise HTTPException(status_code=404, detail=f"No data for epoch {epoch}")

    fuzzy_cols = [c for c in df.columns if c.startswith("fuzzy_")]
    distributions = {}

    for col in fuzzy_cols:
        values = sub[col].values
        hist, bin_edges = np.histogram(values, bins=bins)
        distributions[col.replace("fuzzy_", "F")] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {"epoch": epoch, "gates": distributions}


# ============================================================================
# Simulation Endpoints (for testing/demo)
# ============================================================================

_simulation_task = None
_simulation_running = False


@router.post("/simulate/start")
async def start_training_simulation(
    background_tasks: BackgroundTasks,
    epochs: int = Query(5, ge=1, le=20),
    steps_per_epoch: int = Query(100, ge=10, le=500),
    delay_ms: int = Query(100, ge=10, le=1000),
):
    """Start a simulated training run for testing the GUI."""
    global _simulation_running

    if _simulation_running:
        raise HTTPException(status_code=409, detail="Simulation already running")

    _simulation_running = True
    background_tasks.add_task(
        _run_simulation,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        delay_ms=delay_ms,
    )

    return {
        "status": "started",
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "delay_ms": delay_ms,
    }


@router.post("/simulate/stop")
async def stop_training_simulation():
    """Stop the current training simulation."""
    global _simulation_running
    _simulation_running = False
    return {"status": "stopping"}


@router.get("/simulate/status")
async def get_simulation_status():
    """Get current simulation status."""
    return {"running": _simulation_running}


async def _run_simulation(epochs: int, steps_per_epoch: int, delay_ms: int):
    """Run a simulated training loop."""
    global _simulation_running
    import random
    import math

    from gui.backend.services.training_emitter import get_emitter

    emitter = get_emitter()
    total_steps = epochs * steps_per_epoch

    # Emit training started
    emitter.emit_status(
        is_training=True,
        current_epoch=1,
        current_step=0,
        total_steps=total_steps,
        progress_pct=0,
        eta_seconds=int((total_steps * delay_ms) / 1000),
    )

    step = 0
    base_loss = 2.5
    base_mse = 1.2

    for epoch in range(1, epochs + 1):
        if not _simulation_running:
            break

        for ep_step in range(steps_per_epoch):
            if not _simulation_running:
                break

            step += 1

            # Simulate decreasing loss with noise
            progress = step / total_steps
            noise = random.gauss(0, 0.05)
            decay = math.exp(-progress * 2)

            loss = base_loss * decay + noise * decay
            mse = base_mse * decay + noise * 0.5 * decay

            # Emit step update
            emitter.emit_step(
                step=step,
                epoch=epoch,
                loss=max(0.01, loss),
                mse=max(0.01, mse),
                npdd=random.gauss(1.5, 0.3) * decay,
                sharpe=random.gauss(0.8, 0.2) * (1 - decay * 0.5),
                dd=random.gauss(0.5, 0.1) * decay,
                turnover=random.gauss(0.3, 0.05),
                fuzzy=random.gauss(0.2, 0.05) * decay,
                pattern_ent=-3.0 + random.gauss(0, 0.2),
                group_inv=random.gauss(0.1, 0.02),
                rho=random.gauss(0.5, 0.1),
                energy=random.gauss(0.05, 0.01) * decay,
                growth=random.gauss(0.02, 0.005),
                lr=1e-4 * (0.95 ** epoch),
                grad_norm=random.gauss(2.0, 0.5) * decay,
                scaler_scale=2 ** random.randint(10, 14),
            )

            # Emit fuzzy activations
            emitter.emit_fuzzy(
                step=step,
                epoch=epoch,
                activations=[random.random() for _ in range(10)],
            )

            # Emit progress status
            progress_pct = (step / total_steps) * 100
            remaining_steps = total_steps - step
            eta_seconds = int((remaining_steps * delay_ms) / 1000)

            emitter.emit_status(
                is_training=True,
                current_epoch=epoch,
                current_step=step,
                total_steps=total_steps,
                progress_pct=progress_pct,
                eta_seconds=eta_seconds,
            )

            await asyncio.sleep(delay_ms / 1000)

        # Emit epoch summary
        if _simulation_running:
            emitter.emit_epoch_summary(
                epoch=epoch,
                train_loss=loss,
                val_loss=loss * random.uniform(1.0, 1.3),
                metrics={
                    "mse": mse,
                    "sharpe": random.gauss(1.0, 0.2),
                    "pattern_ent": -3.0 + random.gauss(0, 0.2),
                },
                is_best=epoch == 1 or random.random() > 0.7,
            )

    # Emit training complete
    emitter.emit_status(
        is_training=False,
        current_epoch=epochs,
        current_step=total_steps,
        total_steps=total_steps,
        progress_pct=100,
    )

    emitter.emit_complete(
        epochs=epochs,
        final_loss=loss if _simulation_running else 0,
        best_val_loss=loss * 0.9 if _simulation_running else 0,
        duration_seconds=int((total_steps * delay_ms) / 1000),
    )

    _simulation_running = False
