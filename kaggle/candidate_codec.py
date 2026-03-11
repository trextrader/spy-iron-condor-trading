"""
candidate_codec.py — Parameter Search Space and Candidate Encoding
===================================================================
Defines the search space for each strategy template and provides
encode / decode between normalised [0,1] BO space and real parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ── Core data structures ─────────────────────────────────────────────────────

@dataclass
class ParamSpec:
    """Single optimisable parameter."""
    name:  str
    lo:    float
    hi:    float
    kind:  str          # "grid" (discrete) or "continuous"
    step:  float = 1.0
    dtype: type  = float

    @property
    def n_grid(self) -> int:
        if self.kind == "grid":
            return max(1, round((self.hi - self.lo) / self.step) + 1)
        return 0  # continuous


@dataclass
class SearchSpaceSpec:
    """Full search space for one strategy template."""
    template_id: str
    params: List[ParamSpec]

    @property
    def dim(self) -> int:
        return len(self.params)

    @property
    def bounds(self) -> np.ndarray:
        """Return [2, D] array of [lo, hi] for BoTorch."""
        lows  = np.array([p.lo for p in self.params], dtype=np.float64)
        highs = np.array([p.hi for p in self.params], dtype=np.float64)
        return np.stack([lows, highs], axis=0)


@dataclass
class CandidateBatch:
    """K decoded candidates ready for simulation."""
    K:      int
    params: Dict[str, np.ndarray]   # param_name -> [K] array of real values


# ── Search-space constructors ────────────────────────────────────────────────

def build_iron_butterfly_search_space() -> SearchSpaceSpec:
    # Bounds calibrated for SPY iron butterfly on 2025 data (SPY ~480–600).
    # Goal: generate simulation outputs for CN v46 training inputs.
    # Ranges reflect realistic SPY iron butterfly behaviour:
    #   - spread_width 5–20: $5 wings = tiny credit; $20 = max risk ~$1500/ct
    #   - target_dte 7–21: 0DTE/1DTE chains are deep ITM/OTM — need 1DTE+
    #   - short_delta 0.40–0.50: ATM to slight OTM; core iron butterfly zone
    #   - stop_loss_dollar 500–1500: 1–2× typical max credit ($750–$1200 for 10ct)
    #   - profit_target 500–2000: 50–80% of credit × 10 contracts
    #   - max_dte_exit 0–7: close early at 1 DTE (standard) or hold to expiry
    #   - hold_days 5–21: aligns with target_dte; no point holding longer than DTE
    return SearchSpaceSpec(
        template_id="iron_butterfly",
        params=[
            ParamSpec("stop_loss_dollar", 500,  1500, "grid", 250, int),   # 5 pts
            ParamSpec("profit_target",    500,  2000, "grid", 250, int),   # 7 pts
            ParamSpec("max_dte_exit",     0,    7,    "grid", 1,   int),   # 8 pts
            ParamSpec("spread_width",     5,    20,   "grid", 5,   int),   # 4 pts
            ParamSpec("target_dte",       7,    21,   "grid", 7,   int),   # 3 pts
            ParamSpec("short_delta",      0.40, 0.50, "continuous"),       # GP free
            ParamSpec("hold_days",        5,    21,   "grid", 4,   int),   # 5 pts
        ],
    )


def build_iron_condor_search_space() -> SearchSpaceSpec:
    # Bounds calibrated for SPY iron condor (OTM short strikes, class_idx=7).
    # Short delta 0.15-0.30: OTM calls/puts for wider profit zone vs butterfly.
    return SearchSpaceSpec(
        template_id="iron_condor",
        params=[
            ParamSpec("stop_loss_dollar", 300,  1500, "grid", 250, int),   # 5 pts
            ParamSpec("profit_target",    300,  2000, "grid", 250, int),   # 7 pts
            ParamSpec("max_dte_exit",     0,    5,    "grid", 1,   int),   # 6 pts
            ParamSpec("spread_width",     5,    20,   "grid", 5,   int),   # 4 pts
            ParamSpec("target_dte",       7,    21,   "grid", 7,   int),   # 3 pts
            ParamSpec("short_delta",      0.15, 0.30, "continuous"),       # GP free
            ParamSpec("hold_days",        5,    21,   "grid", 4,   int),   # 5 pts
        ],
    )


def build_short_call_search_space() -> SearchSpaceSpec:
    # Bounds for naked short call (single-leg, class_idx=0).
    # No spread_width (no wings). stop_loss is the key risk control.
    return SearchSpaceSpec(
        template_id="short_call",
        params=[
            ParamSpec("stop_loss_dollar", 200,  1000, "grid", 100, int),   # 9 pts
            ParamSpec("profit_target",    100,  1500, "grid", 100, int),   # 15 pts
            ParamSpec("max_dte_exit",     0,    7,    "grid", 1,   int),   # 8 pts
            ParamSpec("target_dte",       7,    21,   "grid", 7,   int),   # 3 pts
            ParamSpec("short_delta",      0.15, 0.35, "continuous"),       # GP free
            ParamSpec("hold_days",        3,    21,   "grid", 3,   int),   # 7 pts
        ],
    )


def build_short_put_search_space() -> SearchSpaceSpec:
    # Bounds for naked short put (single-leg, class_idx=1). Mirror of short_call.
    return SearchSpaceSpec(
        template_id="short_put",
        params=[
            ParamSpec("stop_loss_dollar", 200,  1000, "grid", 100, int),
            ParamSpec("profit_target",    100,  1500, "grid", 100, int),
            ParamSpec("max_dte_exit",     0,    7,    "grid", 1,   int),
            ParamSpec("target_dte",       7,    21,   "grid", 7,   int),
            ParamSpec("short_delta",      0.15, 0.35, "continuous"),
            ParamSpec("hold_days",        3,    21,   "grid", 3,   int),
        ],
    )


def build_search_space(template_id: str) -> SearchSpaceSpec:
    """Return a SearchSpaceSpec for any known template (generic fallback)."""
    _builders = {
        "iron_butterfly": build_iron_butterfly_search_space,
        "iron_condor":    build_iron_condor_search_space,
        "short_call":     build_short_call_search_space,
        "short_put":      build_short_put_search_space,
    }
    if template_id in _builders:
        return _builders[template_id]()
    # Generic fallback covers income strategies not yet given a specific builder.
    # Includes hold_days and max_dte_exit so K candidates have varying exit timing
    # (without these, all candidates default to hold_days=7 in the engine, making
    # results look identical across the K batch).
    return SearchSpaceSpec(
        template_id=template_id,
        params=[
            ParamSpec("stop_loss_dollar", 300,  1500, "grid", 100, int),
            ParamSpec("profit_target",    300,  3000, "grid", 250, int),
            ParamSpec("target_dte",       7,    28,   "grid", 7,   int),
            ParamSpec("short_delta",      0.15, 0.45, "continuous"),
            ParamSpec("spread_width",     5,    20,   "grid", 5,   int),
            ParamSpec("hold_days",        5,    21,   "grid", 4,   int),
            ParamSpec("max_dte_exit",     0,    5,    "grid", 1,   int),
        ],
    )


# ── Encode / Decode ──────────────────────────────────────────────────────────

def encode_config(config: Dict[str, Any], space: SearchSpaceSpec) -> np.ndarray:
    """
    Encode a strategy-config dict → normalised [0,1] vector of length D.
    Used to seed the GP surrogate from already-known good configs.
    """
    x = np.zeros(space.dim, dtype=np.float64)
    for i, p in enumerate(space.params):
        v = config.get(p.name, p.lo)
        if v is None:
            v = p.lo
        x[i] = float(np.clip((float(v) - p.lo) / max(p.hi - p.lo, 1e-9), 0.0, 1.0))
    return x


def decode_candidate_tensor(
    x_norm: torch.Tensor,      # [K, D] normalised in [0, 1]
    space:  SearchSpaceSpec,
) -> CandidateBatch:
    """
    Decode normalised tensor → CandidateBatch with real param values.
    Grid params are snapped to the nearest grid point.
    """
    x = x_norm.detach().cpu().numpy()   # [K, D]
    K = x.shape[0]
    params: Dict[str, np.ndarray] = {}

    for i, p in enumerate(space.params):
        col = x[:, i]                            # [K] in [0, 1]
        raw = col * (p.hi - p.lo) + p.lo         # [K] in [lo, hi]

        if p.kind == "grid":
            steps   = np.round((raw - p.lo) / p.step).astype(int)
            steps   = np.clip(steps, 0, p.n_grid - 1)
            snapped = p.lo + steps * p.step
            params[p.name] = snapped.astype(int) if p.dtype == int else snapped.astype(np.float32)
        else:
            params[p.name] = raw.astype(np.float32)

    return CandidateBatch(K=K, params=params)


def candidates_to_configs(
    batch:       CandidateBatch,
    base_config: Dict,
) -> List[Dict]:
    """Expand CandidateBatch into K config dicts merged with base_config."""
    configs = []
    for k in range(batch.K):
        cfg = dict(base_config)
        for name, arr in batch.params.items():
            v = arr[k]
            cfg[name] = int(v) if isinstance(arr.dtype, np.integer) or arr.dtype in (np.int32, np.int64) else float(v)
        configs.append(cfg)
    return configs


def sobol_candidates(
    K:     int,
    space: SearchSpaceSpec,
    seed:  int = 0,
) -> CandidateBatch:
    """Generate K quasi-random Sobol candidates covering the full search space."""
    try:
        from torch.quasirandom import SobolEngine
        eng = SobolEngine(dimension=space.dim, scramble=True, seed=seed)
        x_norm = eng.draw(K)                          # [K, D] in [0,1]
    except Exception:
        rng    = np.random.default_rng(seed)
        x_norm = torch.from_numpy(rng.random((K, space.dim)).astype(np.float32))

    return decode_candidate_tensor(x_norm, space)


def random_candidates(
    K:     int,
    space: SearchSpaceSpec,
    seed:  int = 0,
) -> CandidateBatch:
    """Generate K uniform-random candidates."""
    rng    = np.random.default_rng(seed)
    x_norm = torch.from_numpy(rng.random((K, space.dim)).astype(np.float32))
    return decode_candidate_tensor(x_norm, space)


def perturb_best(
    best_x: np.ndarray,        # [D] normalised best point so far
    K:      int,
    space:  SearchSpaceSpec,
    sigma:  float = 0.15,
    seed:   int   = 0,
) -> CandidateBatch:
    """
    Generate K candidates by perturbing the current best point with
    Gaussian noise (clipped to [0,1]).  Used as a fallback when BoTorch
    is unavailable.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=(K, space.dim)).astype(np.float32)
    x_norm = np.clip(best_x[None, :] + noise, 0.0, 1.0)
    return decode_candidate_tensor(torch.from_numpy(x_norm), space)
