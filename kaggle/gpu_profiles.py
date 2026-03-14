"""
gpu_profiles.py — GPU hardware profiles + intensity matrix
===========================================================
Maps --gputype × --gpuintensity to hardware-tuned IntensityPolicy objects.

GPU K threshold derived from crossover benchmark:
  GPU time is approximately flat per GPU model at moderate K.
  CPU scales linearly at ~97µs/candidate.
  crossover_K ≈ flat_gpu_us / 97us_per_candidate

  At high K the GPU becomes memory-bandwidth-bound and flat_latency_us rises.
  The effective peak K is chosen to stay within VRAM while saturating bandwidth.

Measured on A100-SXM4-40GB, iron_butterfly, ~757K options/bar:
  GPU flat latency: ~1935µs/bar  →  crossover K≈20  →  threshold=32

Memory estimate per K candidate:
  K × 757K options × 4 bytes (float32) ≈ K × 3.03 MB
  plus ~5.8 GB constant chain data (all 180M options rows on device).

VRAM budget (chain already loaded):
  T4  16 GB - 5.8 → ~10 GB free  → max K ≈  3300 candidates
  L40S 48 GB - 5.8 → ~42 GB free  → max K ≈ 13900 candidates
  A100 40 GB - 5.8 → ~34 GB free  → max K ≈ 11200 candidates
  H100 80 GB - 5.8 → ~74 GB free  → max K ≈ 24400 candidates
  H200 141 GB- 5.8 → ~135 GB free → max K ≈ 44500 candidates

Usage
-----
  # At program start (once):
  from gpu_profiles import register_gpu_intensity_policies, get_combined_policy
  register_gpu_intensity_policies()

  profile = get_gpu_profile("a100")       # GpuProfile(threshold=32, ...)
  policy  = get_combined_policy("a100", "med")  # IntensityPolicy(init=1024, ...)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List


# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuProfile:
    name: str
    gpu_k_threshold: int    # use GPU path only when K >= this value
    vram_gb: int
    bandwidth_gbs: int      # memory bandwidth GB/s (approx)
    flat_latency_us: float  # estimated flat GPU latency per bar at moderate K (µs)

    def crossover_k(self) -> int:
        """K value where GPU and CPU break even (~97µs/candidate on CPU)."""
        return int(self.flat_latency_us / 97.0)

    def estimated_speedup(self, K: int) -> float:
        cpu_us = K * 97.0
        return cpu_us / self.flat_latency_us if self.flat_latency_us > 0 else 1.0


GPU_PROFILES: Dict[str, GpuProfile] = {
    "t4": GpuProfile(
        name="t4",
        gpu_k_threshold=128,   # crossover K≈124  (12ms flat / 97µs/cand)
        vram_gb=16,
        bandwidth_gbs=300,
        flat_latency_us=12000.0,
    ),
    "l40s": GpuProfile(
        name="l40s",
        gpu_k_threshold=48,    # crossover K≈46  (4.5ms flat / 97µs/cand)
        vram_gb=48,
        bandwidth_gbs=864,
        flat_latency_us=4500.0,
    ),
    "a100": GpuProfile(
        name="a100",
        gpu_k_threshold=32,    # crossover K≈20  (1.9ms flat / 97µs/cand)
        vram_gb=40,            # A100-SXM4-40GB (Lightning AI lit-a100-40gb)
        bandwidth_gbs=2000,
        flat_latency_us=1935.0,
    ),
    "h100": GpuProfile(
        name="h100",
        gpu_k_threshold=16,    # crossover K≈15  (1.5ms flat / 97µs/cand)
        vram_gb=80,
        bandwidth_gbs=3350,
        flat_latency_us=1500.0,
    ),
    "h200": GpuProfile(
        name="h200",
        gpu_k_threshold=8,     # crossover K≈8   (~0.8ms flat / 97µs/cand)
        vram_gb=141,
        bandwidth_gbs=4800,
        flat_latency_us=800.0,
    ),
}

DEFAULT_GPU_PROFILE = GPU_PROFILES["t4"]


# ---------------------------------------------------------------------------
# Intensity matrix
# ---------------------------------------------------------------------------
# (gputype, intensity_level) -> (breadth_top_n, bo_init_trials, bo_batch_size,
#                                bo_rounds, fidelity_schedule)
#
# Design rules:
#   1. bo_init_trials MUST be >= gpu_k_threshold (Sobol phase hits GPU path).
#   2. For A100/H100/H200: scale init_trials aggressively — these GPUs can handle
#      K=1024–4096 without OOM; large K saturates the memory bus and pushes
#      GPU utilisation above 50%.
#   3. T4 caps at K=1024 (safe VRAM margin: 1024×3MB + 5.8GB = 8.9 GB of 16 GB).
#   4. L40S caps at K=2048 (2048×3MB + 5.8GB = 12.0 GB of 48 GB; leaves room).
#   5. A100-40GB caps at K=4096 (4096×3MB + 5.8GB = 18.2 GB of 40 GB).
#   6. H100/H200 can run K=4096+ comfortably.
#   7. bo_batch_size for GP rounds only needs to exceed gpu_k_threshold to use
#      the GPU path; larger values mean more parallel GP proposals.
# ---------------------------------------------------------------------------

GPU_INTENSITY_MATRIX: Dict[Tuple[str, str], Tuple] = {
    # ── T4 · 16 GB VRAM · threshold K≥128 · max safe K≈1024 ─────────────
    # At K=1024: GPU ~12ms; CPU would take 99ms → ~8× speedup.
    # ~100% GPU utilisation confirmed by user at K=1024 (gpu-sobol run).
    ("t4", "min"):     (7,    128,   8, 2, ["fast"]),
    ("t4", "min-med"): (9,    256,   8, 3, ["fast", "medium"]),
    ("t4", "med"):     (11,   512,  12, 4, ["fast", "medium"]),
    ("t4", "med-max"): (13,   768,  14, 5, ["fast", "medium", "full"]),
    ("t4", "max"):     (999, 1024,  16, 6, ["fast", "medium", "full"]),
    ("t4", "sobol"):   (11,  1024,  16, 0, ["fast", "medium"]),

    # ── L40S · 48 GB VRAM · threshold K≥48 · max safe K≈2048 ────────────
    # At K=2048: GPU ~4.5ms*2 ≈ 9ms (memory-bound); CPU 99ms → ~11× speedup.
    ("l40s", "min"):     (7,    128,   8, 2, ["fast"]),
    ("l40s", "min-med"): (9,    256,  12, 3, ["fast", "medium"]),
    ("l40s", "med"):     (11,   512,  16, 4, ["fast", "medium"]),
    ("l40s", "med-max"): (13,  1024,  24, 5, ["fast", "medium", "full"]),
    ("l40s", "max"):     (999, 2048,  32, 6, ["fast", "medium", "full"]),
    ("l40s", "sobol"):   (11,  2048,  32, 0, ["fast", "medium"]),

    # ── A100-40GB · 40 GB VRAM · threshold K≥32 · max safe K≈4096 ───────
    # At K=1024: GPU flat ~1935µs; CPU 99ms → ~51× speedup  (bandwidth ≈ 3.1 GB).
    # At K=2048: GPU ~3.8ms (memory-bound); CPU 198ms → ~52× speedup.
    # At K=4096: GPU ~7.6ms; CPU 397ms → ~52× speedup. VRAM: 18 GB of 40 GB.
    # med uses K=1024 — near bandwidth saturation, >50% GPU utilization.
    ("a100", "min"):     (7,    256,  16, 2, ["fast"]),
    ("a100", "min-med"): (9,    512,  16, 3, ["fast", "medium"]),
    ("a100", "med"):     (11,  1024,  24, 4, ["fast", "medium"]),
    ("a100", "med-max"): (13,  2048,  32, 5, ["fast", "medium", "full"]),
    ("a100", "max"):     (999, 4096,  48, 6, ["fast", "medium", "full"]),
    ("a100", "sobol"):   (11,  1024,  24, 0, ["fast", "medium"]),

    # ── H100 · 80 GB VRAM · threshold K≥16 · max safe K≈4096+ ───────────
    # 3350 GB/s bandwidth → K=1024 in ~0.93ms; K=2048 in ~1.85ms.
    # Large VRAM allows K=4096 (18.2 GB of 80 GB).
    ("h100", "min"):     (7,    256,  16, 2, ["fast"]),
    ("h100", "min-med"): (9,    512,  32, 3, ["fast", "medium"]),
    ("h100", "med"):     (11,  1024,  32, 4, ["fast", "medium"]),
    ("h100", "med-max"): (13,  2048,  64, 5, ["fast", "medium", "full"]),
    ("h100", "max"):     (999, 4096,  64, 6, ["fast", "medium", "full"]),
    ("h100", "sobol"):   (11,  2048,  32, 0, ["fast", "medium"]),

    # ── H200 · 141 GB VRAM · threshold K≥8 · max safe K≈8192+ ──────────
    # 4800 GB/s bandwidth → K=1024 in ~0.65ms; K=4096 in ~2.6ms.
    # VRAM allows K=8192 (30.6 GB of 141 GB).
    ("h200", "min"):     (7,    256,  16, 2, ["fast"]),
    ("h200", "min-med"): (9,    512,  32, 3, ["fast", "medium"]),
    ("h200", "med"):     (11,  1024,  32, 4, ["fast", "medium"]),
    ("h200", "med-max"): (13,  2048,  64, 5, ["fast", "medium", "full"]),
    ("h200", "max"):     (999, 4096, 128, 6, ["fast", "medium", "full"]),
    ("h200", "sobol"):   (11,  4096,  64, 0, ["fast", "medium"]),
}

_GPU_INTENSITY_REGISTERED = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_gpu_intensity_policies() -> None:
    """
    Insert all GPU-specific intensity policies into INTENSITY_POLICIES so that
    get_intensity_policy("t4-med"), get_intensity_policy("a100-sobol"), etc.
    resolve transparently. Idempotent — safe to call multiple times.
    """
    global _GPU_INTENSITY_REGISTERED
    if _GPU_INTENSITY_REGISTERED:
        return
    from optimization_intensity import IntensityPolicy, INTENSITY_POLICIES
    for (gtype, level), (breadth, init, batch, rounds, fidelity) in GPU_INTENSITY_MATRIX.items():
        key = f"{gtype}-{level}"
        INTENSITY_POLICIES[key] = IntensityPolicy(
            name=key,
            breadth_top_n=breadth,
            bo_init_trials=init,
            bo_batch_size=batch,
            bo_rounds=rounds,
            fidelity_schedule=list(fidelity),
        )
    _GPU_INTENSITY_REGISTERED = True


def get_gpu_profile(name: str | None) -> GpuProfile:
    if not name:
        return DEFAULT_GPU_PROFILE
    key = str(name).strip().lower()
    if key not in GPU_PROFILES:
        allowed = ", ".join(GPU_PROFILES)
        raise KeyError(f"Unknown --gputype {name!r}. Allowed: {allowed}")
    return GPU_PROFILES[key]


def get_combined_policy(gputype: str | None, intensity_level: str | None):
    """
    Return the IntensityPolicy for (gputype, intensity_level).
    Calls register_gpu_intensity_policies() if needed.
    Falls back to base intensity if gputype is None.
    """
    register_gpu_intensity_policies()
    from optimization_intensity import get_intensity_policy
    if not gputype or not intensity_level:
        return get_intensity_policy(intensity_level or "med")
    key = f"{str(gputype).strip().lower()}-{str(intensity_level).strip().lower()}"
    return get_intensity_policy(key)


def print_gpu_profile(profile: GpuProfile) -> None:
    print(f"[gpu_profile] type={profile.name}  "
          f"threshold=K≥{profile.gpu_k_threshold}  "
          f"vram={profile.vram_gb}GB  "
          f"bandwidth={profile.bandwidth_gbs}GB/s  "
          f"est_flat={profile.flat_latency_us:.0f}µs/bar  "
          f"crossover_K≈{profile.crossover_k()}")


def print_gpu_intensity_table(gputype: str) -> None:
    """Print the full intensity matrix for a given GPU type."""
    gtype = str(gputype).strip().lower()
    levels = ["min", "min-med", "med", "med-max", "max", "sobol"]
    print(f"[gpu_intensity] {gtype.upper()} intensity matrix:")
    print(f"  {'level':<10} {'init':>6} {'batch':>6} {'rounds':>7} {'breadth':>8}  fidelity")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*7} {'-'*8}  --------")
    for lv in levels:
        key = (gtype, lv)
        if key not in GPU_INTENSITY_MATRIX:
            continue
        breadth, init, batch, rounds, fidelity = GPU_INTENSITY_MATRIX[key]
        breadth_s = "all" if breadth >= 999 else str(breadth)
        fid_s = "+".join(f[0] for f in fidelity)   # f / f+m / f+m+F
        print(f"  {lv:<10} {init:>6} {batch:>6} {rounds:>7} {breadth_s:>8}  {fid_s}")
