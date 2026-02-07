"""
Backtest Runner Service
Phase 6.1 - Core Infrastructure

Manages backtest execution with progress tracking and result storage.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from gui.backend.schemas.backtest import BacktestStatus, BacktestResult, BacktestMetrics
from gui.backend.schemas.config import FullConfig


class BacktestRunner:
    """Manages backtest execution and result storage."""

    def __init__(self):
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, BacktestResult] = {}

    async def run_backtest(
        self,
        run_id: str,
        config: FullConfig,
        limit: Optional[int] = None,
    ):
        """Run a backtest asynchronously."""
        # Initialize run state
        self._runs[run_id] = {
            "status": BacktestStatus.RUNNING,
            "progress": 0.0,
            "bars_processed": 0,
            "total_bars": limit or 1000,  # TODO: Get from tape
            "current_pnl": 0.0,
            "started_at": datetime.utcnow().isoformat(),
            "config": config.model_dump(),
        }

        try:
            # TODO: Integrate with actual backtest engine
            # For now, simulate progress
            total_bars = limit or 1000
            for i in range(total_bars):
                if self._runs[run_id]["status"] == BacktestStatus.CANCELLED:
                    break

                # Simulate processing
                await asyncio.sleep(0.01)  # Placeholder

                # Update progress
                self._runs[run_id]["bars_processed"] = i + 1
                self._runs[run_id]["progress"] = (i + 1) / total_bars * 100

                # Simulate PnL changes
                import random
                self._runs[run_id]["current_pnl"] += random.uniform(-10, 15)

            # Mark as completed
            if self._runs[run_id]["status"] != BacktestStatus.CANCELLED:
                self._runs[run_id]["status"] = BacktestStatus.COMPLETED
                self._runs[run_id]["completed_at"] = datetime.utcnow().isoformat()

                # Create result
                self._results[run_id] = BacktestResult(
                    run_id=run_id,
                    config_hash="placeholder",  # TODO: Get from config manager
                    status=BacktestStatus.COMPLETED,
                    started_at=self._runs[run_id]["started_at"],
                    completed_at=self._runs[run_id]["completed_at"],
                    duration_seconds=10.0,  # TODO: Calculate actual duration
                    metrics=BacktestMetrics(
                        net_pnl=self._runs[run_id]["current_pnl"],
                        total_trades=50,  # Placeholder
                        win_rate=0.55,
                        sharpe_ratio=1.5,
                        max_drawdown=0.08,
                        npdd_ratio=2.1,
                        avg_trade_pnl=20.0,
                        avg_trade_duration=45.0,
                        profit_factor=1.8,
                    ),
                    equity_curve=[],
                    trades=[],
                )

        except Exception as e:
            self._runs[run_id]["status"] = BacktestStatus.FAILED
            self._runs[run_id]["error"] = str(e)

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
            "progress_pct": run["progress"],
            "bars_processed": run["bars_processed"],
            "total_bars": run["total_bars"],
            "current_pnl": run["current_pnl"],
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
        runs = list(self._runs.values())
        if status_filter:
            runs = [r for r in runs if r["status"] == status_filter]
        return runs[:limit]

    def cancel(self, run_id: str) -> bool:
        """Cancel a running backtest."""
        if run_id not in self._runs:
            return False
        if self._runs[run_id]["status"] != BacktestStatus.RUNNING:
            return False
        self._runs[run_id]["status"] = BacktestStatus.CANCELLED
        return True

    async def run_replay(
        self,
        replay_run_id: str,
        original_run_id: str,
        config: FullConfig,
    ) -> Dict[str, Any]:
        """Replay a backtest to verify determinism."""
        # TODO: Implement actual replay with comparison
        return {
            "verified": True,
            "diff_fingerprint": "match",
            "differences": None,
        }
