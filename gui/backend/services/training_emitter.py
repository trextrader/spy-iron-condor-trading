"""
Training Telemetry Emitter
Phase 6.7 - Model Introspection

Provides functions to emit training updates via HTTP to backend.
Can be imported by training scripts to broadcast real-time metrics.
Uses HTTP POST to backend API, which then broadcasts to WebSocket clients.
"""

import requests
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading


class TrainingEmitter:
    """
    Emits training telemetry via HTTP to the backend server.
    The backend then broadcasts to connected WebSocket clients.

    Usage:
        from gui.backend.services.training_emitter import get_emitter

        emitter = get_emitter()
        emitter.emit_step(step=100, epoch=1, loss=0.5, mse=0.3, ...)
    """

    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self._session = requests.Session()

    def _post(self, endpoint: str, data: dict):
        """Send POST request to backend (non-blocking via thread)."""
        def _do_post():
            try:
                self._session.post(
                    f"{self.backend_url}/api/training{endpoint}",
                    json=data,
                    timeout=2,
                )
            except Exception:
                # Silently fail - training should continue even if telemetry fails
                pass

        # Run in thread to avoid blocking training
        thread = threading.Thread(target=_do_post, daemon=True)
        thread.start()

    def emit_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        mse: float,
        npdd: float = 0.0,
        sharpe: float = 0.0,
        dd: float = 0.0,
        turnover: float = 0.0,
        fuzzy: float = 0.0,
        pattern_ent: float = 0.0,
        group_inv: float = 0.0,
        rho: float = 0.0,
        energy: float = 0.0,
        growth: float = 0.0,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        scaler_scale: Optional[float] = None,
    ):
        """Emit a training step update."""
        data = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "mse": mse,
            "npdd": npdd,
            "sharpe": sharpe,
            "dd": dd,
            "turnover": turnover,
            "fuzzy": fuzzy,
            "pattern_ent": pattern_ent,
            "group_inv": group_inv,
            "rho": rho,
            "energy": energy,
            "growth": growth,
        }

        if lr is not None:
            data["lr"] = lr
        if grad_norm is not None:
            data["grad_norm"] = grad_norm
        if scaler_scale is not None:
            data["scaler_scale"] = scaler_scale

        self._post("/telemetry/step", data)

    def emit_fuzzy(
        self,
        step: int,
        epoch: int,
        activations: List[float],
    ):
        """Emit fuzzy gate activations."""
        self._post("/telemetry/fuzzy", {
            "step": step,
            "epoch": epoch,
            "activations": activations,
        })

    def emit_status(
        self,
        is_training: bool,
        current_epoch: int = 0,
        current_step: int = 0,
        total_steps: int = 0,
        progress_pct: float = 0.0,
        eta_seconds: Optional[int] = None,
    ):
        """Emit training status update."""
        data = {
            "isTraining": is_training,
            "currentEpoch": current_epoch,
            "currentStep": current_step,
            "totalSteps": total_steps,
            "progressPct": progress_pct,
        }

        if eta_seconds is not None:
            data["etaSeconds"] = eta_seconds

        self._post("/telemetry/status", data)

    def emit_complete(
        self,
        epochs: int,
        final_loss: float,
        best_val_loss: float,
        duration_seconds: int,
    ):
        """Emit training complete notification."""
        self._post("/telemetry/complete", {
            "epochs": epochs,
            "finalLoss": final_loss,
            "bestValLoss": best_val_loss,
            "durationSeconds": duration_seconds,
        })

    def emit_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Emit end-of-epoch summary."""
        self._post("/telemetry/epoch", {
            "epoch": epoch,
            "trainLoss": train_loss,
            "valLoss": val_loss,
            "metrics": metrics,
            "isBest": is_best,
        })


# Singleton instance
_emitter: Optional[TrainingEmitter] = None


def get_emitter(backend_url: str = "http://localhost:8000") -> TrainingEmitter:
    """Get or create the global training emitter instance.

    If backend_url differs from existing emitter, creates a new one.
    """
    global _emitter
    if _emitter is None or _emitter.backend_url != backend_url:
        _emitter = TrainingEmitter(backend_url=backend_url)
    return _emitter


def init_emitter(manager=None, backend_url: str = "http://localhost:8000"):
    """Initialize the emitter. Manager param kept for backwards compatibility."""
    return get_emitter(backend_url)
