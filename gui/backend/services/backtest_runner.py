"""
Backtest Runner Service
Phase 6.5 - Core Infrastructure

Manages backtest execution with progress tracking and result storage.
Uses BacktestBridge to run the actual kaggle backtester.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from gui.backend.schemas.backtest import (
    BacktestStatus,
    BacktestResult,
    BacktestMetrics,
    BacktestProgress,
)
from gui.backend.schemas.config import FullConfig
from gui.backend.services.backtest_bridge import BacktestBridge


class BacktestRunner:
    """Manages backtest execution and result storage."""

    def __init__(self):
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, BacktestResult] = {}
        self._bridges: Dict[str, BacktestBridge] = {}
        self._progress_callbacks: Dict[str, Any] = {}

    def set_progress_callback(self, run_id: str, callback):
        """Set progress callback for a run (for WebSocket streaming)."""
        self._progress_callbacks[run_id] = callback

    async def _progress_handler(self, run_id: str, progress: BacktestProgress):
        """Handle progress updates from the bridge."""
        # Update internal state
        if run_id in self._runs:
            self._runs[run_id]["progress"] = progress.progress_pct
            self._runs[run_id]["bars_processed"] = progress.bars_processed
            self._runs[run_id]["current_pnl"] = progress.current_pnl
            self._runs[run_id]["current_equity"] = progress.current_equity
            self._runs[run_id]["trades_so_far"] = progress.trades_so_far

        # Call external callback if registered
        if run_id in self._progress_callbacks:
            callback = self._progress_callbacks[run_id]
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)

    async def run_backtest(
        self,
        run_id: str,
        config: FullConfig,
        limit: Optional[int] = None,
    ):
        """Run a backtest asynchronously using the real backtester."""
        # Initialize run state
        self._runs[run_id] = {
            "status": BacktestStatus.RUNNING,
            "progress": 0.0,
            "bars_processed": 0,
            "total_bars": limit or 10000,
            "current_pnl": 0.0,
            "current_equity": 100000.0,
            "trades_so_far": 0,
            "started_at": datetime.utcnow().isoformat(),
            "config": config.model_dump() if hasattr(config, 'model_dump') else {},
        }

        # Create bridge with progress callback
        bridge = BacktestBridge(
            progress_callback=lambda p: self._progress_handler(run_id, p)
        )
        self._bridges[run_id] = bridge

        try:
            # Run the real backtest
            result = await bridge.run_backtest(
                run_id=run_id,
                tape_id=config.tape_id,
                model_id=config.model_id,
                seed=config.seed,
                device=config.device,
                limit=limit,
                use_execution_reality=config.execution_reality.enabled if config.execution_reality else True,
                use_fuzzy_sizing=False,
                use_trade_rules=True,
            )

            # Store result
            self._results[run_id] = result
            self._runs[run_id]["status"] = result.status

            if result.status == BacktestStatus.COMPLETED:
                self._runs[run_id]["completed_at"] = datetime.utcnow().isoformat()

        except Exception as e:
            self._runs[run_id]["status"] = BacktestStatus.FAILED
            self._runs[run_id]["error"] = str(e)

            # Create error result
            self._results[run_id] = BacktestResult(
                run_id=run_id,
                status=BacktestStatus.FAILED,
                config_hash="",
                started_at=datetime.fromisoformat(self._runs[run_id]["started_at"]),
                tape_id=config.tape_id,
                model_id=config.model_id,
                seed=config.seed,
                device=config.device,
                total_bars=0,
                error_message=str(e),
            )

        finally:
            # Cleanup
            if run_id in self._bridges:
                del self._bridges[run_id]

    def get_status(self, run_id: str) -> Optional[BacktestStatus]:
        """Get the status of a backtest run."""
        if run_id not in self._runs:
            return None
        return self._runs[run_id]["status"]

    def get_progress(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information for a run."""
        if run_id not in self._runs:
            return None
        run = self._runs[run_id]
        return {
            "progress_pct": run.get("progress", 0),
            "bars_processed": run.get("bars_processed", 0),
            "total_bars": run.get("total_bars", 0),
            "current_pnl": run.get("current_pnl", 0),
            "current_equity": run.get("current_equity", 100000),
            "trades_so_far": run.get("trades_so_far", 0),
        }

    def get_result(self, run_id: str) -> Optional[BacktestResult]:
        """Get the result of a completed backtest."""
        return self._results.get(run_id)

    def list_runs(
        self,
        limit: int = 50,
        status_filter: Optional[BacktestStatus] = None,
    ) -> List[Dict[str, Any]]:
        """List recent backtest runs."""
        runs = []
        for run_id, run_data in self._runs.items():
            if status_filter and run_data["status"] != status_filter:
                continue
            runs.append({
                "run_id": run_id,
                **run_data,
            })
        # Sort by start time descending
        runs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return runs[:limit]

    def cancel(self, run_id: str) -> bool:
        """Cancel a running backtest."""
        if run_id not in self._runs:
            return False
        if self._runs[run_id]["status"] != BacktestStatus.RUNNING:
            return False

        # Signal cancellation to bridge
        if run_id in self._bridges:
            self._bridges[run_id].cancel()

        self._runs[run_id]["status"] = BacktestStatus.CANCELLED
        return True

    async def run_replay(
        self,
        replay_run_id: str,
        original_run_id: str,
        config: FullConfig,
    ) -> Dict[str, Any]:
        """Replay a backtest to verify determinism."""
        # Get original result
        original = self._results.get(original_run_id)
        if not original:
            return {
                "verified": False,
                "diff_fingerprint": "original_not_found",
                "differences": ["Original run not found"],
            }

        # Run replay
        await self.run_backtest(replay_run_id, config, limit=original.total_bars)
        replay = self._results.get(replay_run_id)

        if not replay or replay.status != BacktestStatus.COMPLETED:
            return {
                "verified": False,
                "diff_fingerprint": "replay_failed",
                "differences": ["Replay failed to complete"],
            }

        # Compare results
        differences = []

        if original.metrics and replay.metrics:
            if abs(original.metrics.total_pnl - replay.metrics.total_pnl) > 0.01:
                differences.append(f"PnL differs: {original.metrics.total_pnl:.2f} vs {replay.metrics.total_pnl:.2f}")
            if original.metrics.total_trades != replay.metrics.total_trades:
                differences.append(f"Trade count differs: {original.metrics.total_trades} vs {replay.metrics.total_trades}")

        return {
            "verified": len(differences) == 0,
            "diff_fingerprint": "match" if not differences else "mismatch",
            "differences": differences if differences else None,
        }
