"""
kaggle/strategies/__init__.py — Strategy Config Loader
======================================================
Auto-discovers all per-strategy .py files in this directory and returns
a unified STRATEGY_CONFIGS dict keyed by class_name.

Usage:
    from kaggle.strategies import load_strategy_configs
    STRATEGY_CONFIGS = load_strategy_configs()
    cfg = STRATEGY_CONFIGS.get("single_call")  # → dict or None
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, Optional

from strategies._defaults import DEFAULT_CONFIG

# Strategy class names (must match V43_STRATEGY_NAMES in the backtester)
_V43_CLASS_NAMES = [
    "single_call", "single_put", "bull_call_spread", "bear_put_spread",
    "straddle", "strangle", "butterfly_call", "iron_condor",
    "custom_multi_leg",
]

# Files to skip when scanning this directory
_SKIP_FILES = {"__init__.py", "_defaults.py", "__pycache__"}


def load_strategy_configs(verbose: bool = True) -> Dict[str, dict]:
    """
    Scan kaggle/strategies/ for per-strategy .py files.
    Each must export a CONFIG dict.
    Returns {class_name: merged_config} for all found strategies.
    Strategies without a config file get DEFAULT_CONFIG.
    """
    configs: Dict[str, dict] = {}
    strategies_dir = os.path.dirname(os.path.abspath(__file__))

    for fname in sorted(os.listdir(strategies_dir)):
        if fname in _SKIP_FILES or not fname.endswith(".py"):
            continue

        module_name = fname[:-3]  # strip .py
        try:
            mod = importlib.import_module(f"strategies.{module_name}")
            cfg = getattr(mod, "CONFIG", None)
            if cfg is None:
                if verbose:
                    print(f"  [strategies] SKIP {fname}: no CONFIG dict")
                continue

            class_name = cfg.get("class_name", module_name)

            # Merge with defaults (config overrides defaults)
            merged = {**DEFAULT_CONFIG, **cfg}
            configs[class_name] = merged

            if verbose:
                print(f"  [strategies] Loaded: {class_name} "
                      f"(max_qty={merged['max_contracts']}, "
                      f"stop={merged['stop_loss_mult']}×, "
                      f"target=${merged['profit_target']}, "
                      f"fallback={merged['fallback_template']})")

        except Exception as e:
            print(f"  [strategies] ERROR loading {fname}: {e}")

    if verbose and not configs:
        print("  [strategies] No per-strategy configs found — using defaults")

    return configs


def get_config(configs: Dict[str, dict],
               class_name: str) -> dict:
    """Get config for a strategy class, falling back to DEFAULT_CONFIG."""
    return configs.get(class_name, DEFAULT_CONFIG.copy())
