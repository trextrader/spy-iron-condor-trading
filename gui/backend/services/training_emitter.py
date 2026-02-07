"""
Training Telemetry Emitter
Phase 6.7 - Model Introspection

Provides functions to emit training updates via WebSocket.
Can be imported by training scripts to broadcast real-time metrics.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime


class TrainingEmitter:
    """
    Emits training telemetry to connected WebSocket clients.

    Usage:
        from gui.backend.services.training_emitter import get_emitter

        emitter = get_emitter()
        emitter.emit_step(step=100, epoch=1, loss=0.5, mse=0.3, ...)
    """

    def __init__(self):
        self._manager = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_manager(self, manager):
        """Set the WebSocket connection manager."""
        self._manager = manager

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _broadcast(self, channel: str, data: dict):
        """Broadcast data to a channel (sync wrapper)."""
        if self._manager is None:
            return

        message = {
            "type": "data",
            "channel": channel,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            loop = self._get_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self._manager.broadcast_to_channel(channel, message),
                    loop=loop
                )
            else:
                loop.run_until_complete(
                    self._manager.broadcast_to_channel(channel, message)
                )
        except Exception as e:
            # Silently fail - training should continue even if broadcast fails
            pass

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

        self._broadcast("training.step", data)

    def emit_fuzzy(
        self,
        step: int,
        epoch: int,
        activations: List[float],
    ):
        """Emit fuzzy gate activations."""
        self._broadcast("training.fuzzy", {
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

        self._broadcast("training.status", data)

    def emit_complete(
        self,
        epochs: int,
        final_loss: float,
        best_val_loss: float,
        duration_seconds: int,
    ):
        """Emit training complete notification."""
        self._broadcast("training.complete", {
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
        self._broadcast("training.epoch", {
            "epoch": epoch,
            "trainLoss": train_loss,
            "valLoss": val_loss,
            "metrics": metrics,
            "isBest": is_best,
        })


# Singleton instance
_emitter: Optional[TrainingEmitter] = None


def get_emitter() -> TrainingEmitter:
    """Get the global training emitter instance."""
    global _emitter
    if _emitter is None:
        _emitter = TrainingEmitter()
    return _emitter


def init_emitter(manager):
    """Initialize the emitter with the WebSocket manager."""
    emitter = get_emitter()
    emitter.set_manager(manager)
    return emitter
