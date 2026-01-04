"""QTMF public package.

This package provides a stable facade that other systems (e.g., gaussian-system)
can import without depending on QTMF internal module layout.

Primary entrypoints:
- qtmf.benchmark_and_size(...)
- qtmf.SizingPlan
- qtmf.TradeIntent
"""

from .models import TradeIntent, SizingPlan
from .facade import benchmark_and_size

__all__ = ["TradeIntent", "SizingPlan", "benchmark_and_size"]
