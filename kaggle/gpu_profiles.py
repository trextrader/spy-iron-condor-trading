"""
gpu_profiles.py — GPU-specific optimizer tuning profiles
=========================================================
Maps --gputype values to hardware-tuned constants used throughout
the optimizer. All thresholds derived from the crossover benchmark:
  GPU time is flat per GPU model; CPU scales at ~97µs/candidate.
  crossover_K = flat_gpu_us / 97us_per_candidate

Benchmark base (A100, iron_butterfly, ~757K options/bar):
  GPU flat latency: ~1935µs/bar  →  crossover K≈20  →  threshold=32
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuProfile:
    name: str
    gpu_k_threshold: int    # use GPU path only when K >= this value
    vram_gb: int
    bandwidth_gbs: int      # memory bandwidth GB/s (approx)
    flat_latency_us: float  # estimated flat GPU latency per bar (µs)

    def crossover_k(self) -> int:
        """K value where GPU and CPU break even (~97µs/candidate on CPU)."""
        return int(self.flat_latency_us / 97.0)

    def estimated_speedup(self, K: int) -> float:
        cpu_us = K * 97.0
        return cpu_us / self.flat_latency_us if self.flat_latency_us > 0 else 1.0


GPU_PROFILES: dict[str, GpuProfile] = {
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
        vram_gb=80,
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
}

DEFAULT_GPU_PROFILE = GPU_PROFILES["a100"]


def get_gpu_profile(name: str | None) -> GpuProfile:
    if not name:
        return DEFAULT_GPU_PROFILE
    key = str(name).strip().lower()
    if key not in GPU_PROFILES:
        allowed = ", ".join(GPU_PROFILES)
        raise KeyError(f"Unknown --gputype {name!r}. Allowed: {allowed}")
    return GPU_PROFILES[key]


def print_gpu_profile(profile: GpuProfile) -> None:
    print(f"[gpu_profile] type={profile.name}  "
          f"threshold=K≥{profile.gpu_k_threshold}  "
          f"vram={profile.vram_gb}GB  "
          f"bandwidth={profile.bandwidth_gbs}GB/s  "
          f"est_flat={profile.flat_latency_us:.0f}µs/bar  "
          f"crossover_K≈{profile.crossover_k()}")
