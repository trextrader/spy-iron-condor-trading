"""
core/engine.py

Top-level TradingEngine coordinating the data pipeline, strategy evaluation,
sizing, risk checks, and broker execution.

This is a stub implementation for Phase 2 trace harness.
Full implementation will come in Phase 6.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from core.config import RunConfig, StrategyConfig
from data_factory.data_engine import DataEngine
from core.dto import TradeDecision, SizedDecision, Approval


# Stub strategy that returns no-trade decisions
@dataclass
class StubStrategy:
    cfg: StrategyConfig

    def evaluate(self, snapshot) -> TradeDecision:
        return TradeDecision(
            symbol=snapshot.symbol,
            should_trade=False,
            structure="iron_condor",
            bias="neutral",
            rationale={"reason": "stub_strategy"}
        )


# Stub sizer
@dataclass
class StubSizer:
    cfg: StrategyConfig

    def size(self, decision: TradeDecision, snapshot) -> SizedDecision:
        return SizedDecision(
            decision=decision,
            contracts=0,
            confidence=0.5,
            risk_budget=0.0,
            legs=[]
        )


# Stub risk manager
@dataclass
class StubRiskManager:
    cfg: StrategyConfig

    def approve(self, sized: SizedDecision, snapshot) -> Approval:
        return Approval(
            approved=False,
            reason="stub_risk_manager"
        )


# Stub router
@dataclass
class StubRouter:
    cfg: RunConfig

    def execute(self, order_plan, snapshot) -> None:
        pass


@dataclass
class TradingEngine:
    cfg: RunConfig
    data: Optional[DataEngine] = None
    strategy: Optional[StubStrategy] = None
    sizer: Optional[StubSizer] = None
    risk: Optional[StubRiskManager] = None
    router: Optional[StubRouter] = None

    def __post_init__(self) -> None:
        s_cfg = StrategyConfig()  # Use default
        self.data = self.data or DataEngine(self.cfg)
        self.strategy = self.strategy or StubStrategy(s_cfg)
        self.sizer = self.sizer or StubSizer(s_cfg)
        self.risk = self.risk or StubRiskManager(s_cfg)
        self.router = self.router or StubRouter(self.cfg)

    def run(self) -> None:
        """
        Main loop for backtest/live.
        In backtest: iterates bars; in live: responds to streaming ticks/bars.
        """
        for snapshot in self.data.stream():
            decision = self.strategy.evaluate(snapshot)
            if not decision.should_trade:
                continue

            sized = self.sizer.size(decision, snapshot)
            approved = self.risk.approve(sized, snapshot)

            if not approved.approved:
                continue

            self.router.execute(approved.order_plan, snapshot)
