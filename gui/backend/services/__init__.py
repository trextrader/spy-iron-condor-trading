"""
CondorBrain GUI Backend Services
Phase 6.1 - Core Infrastructure
"""

from .config_engine import ConfigManager
from .backtest_runner import BacktestRunner

__all__ = ["ConfigManager", "BacktestRunner"]
