"""
kaggle/strategies/__init__.py — Auto-discovery loader for strategy configs
==========================================================================

Scans this directory for all .py files (excluding _ prefixed files),
imports each one's CONFIG dict, and merges with _defaults.py.

Usage in backtester:
    from strategies import load_strategy_configs
    configs = load_strategy_configs()  # dict: template_id -> merged config
"""
import os
import importlib.util

def _load_defaults():
    """Load DEFAULT_CONFIG from _defaults.py."""
    defaults_path = os.path.join(os.path.dirname(__file__), "_defaults.py")
    if not os.path.exists(defaults_path):
        return {}
    spec = importlib.util.spec_from_file_location("_defaults", defaults_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "DEFAULT_CONFIG", {})


def load_strategy_configs():
    """
    Scan strategies/ for all .py files and return {template_id: merged_config}.
    Each strategy file must define a CONFIG dict with at least 'template_id'.
    """
    strat_dir = os.path.dirname(__file__)
    defaults = _load_defaults()
    configs = {}

    for fname in sorted(os.listdir(strat_dir)):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        fpath = os.path.join(strat_dir, fname)
        try:
            spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cfg = getattr(mod, "CONFIG", None)
            if cfg and "template_id" in cfg:
                merged = {**defaults, **cfg}
                configs[cfg["template_id"]] = merged
        except Exception as e:
            print(f"  [strategies] SKIP {fname}: {e}")

    return configs


# Pre-load at import time for convenience
STRATEGY_CONFIGS = load_strategy_configs()
