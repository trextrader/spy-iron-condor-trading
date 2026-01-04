# core/benchmark_cpu.py
import json
import os
import platform
import sys
import time
from datetime import datetime

import psutil

from core.backtest_engine import run_backtest_headless
from core.config import StrategyConfig, RunConfig


def benchmark(save_json: bool = True, test_bars: int = 5000):
    """Run hardware benchmark and save results to JSON.
    
    Args:
        save_json: Whether to save results to reports/benchmark.json
        test_bars: Number of bars to test (default 5000)
        
    Returns:
        dict with benchmark results
    """
    print("==============================================")
    print(" HARDWARE & WORKLOAD BENCHMARK")
    print("==============================================\n")

    # 1. System Info
    cpu_count = psutil.cpu_count(logical=False) or 1
    logical_count = psutil.cpu_count(logical=True) or 1
    mem = psutil.virtual_memory()
    
    system_info = {
        "cpu_physical_cores": cpu_count,
        "cpu_logical_cores": logical_count,
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
        "python_version": platform.python_version(),
        "platform": platform.system(),
    }
    
    print(f"Processor: {cpu_count} Physical Cores / {logical_count} Logical Processors")
    print(f"Total RAM: {system_info['ram_total_gb']:.2f} GB")
    print(f"Available RAM: {system_info['ram_available_gb']:.2f} GB")
    print(f"Python: {system_info['python_version']}")
    print("-" * 40)

    # 2. Workload Test
    s_cfg = StrategyConfig()
    r_cfg = RunConfig()
    r_cfg.backtest_samples = test_bars
    
    print(f"Starting sample backtest ({test_bars:,} bars)...")
    start_time = time.time()
    
    # Run once to measure baseline
    run_backtest_headless(s_cfg, r_cfg)
    
    end_time = time.time()
    duration = end_time - start_time
    bars_per_second = test_bars / duration if duration > 0 else 0
    
    print(f"\nBenchmark Result:")
    print(f"Time for {test_bars:,} bars: {duration:.2f} seconds")
    print(f"Bars per second: {bars_per_second:,.0f}")
    
    # 3. Projection
    projected_serial = (duration * 100) / 60  # For 100 runs in minutes
    print(f"Projected time for 100 backtests (Serial): {projected_serial:.2f} minutes")
    
    if cpu_count > 1:
        projected_parallel = projected_serial / cpu_count
        print(f"Projected time for 100 backtests (Parallel - {cpu_count} cores): {projected_parallel:.2f} minutes")

    # 4. Build results
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_bars": test_bars,
        "runtime_seconds": round(duration, 3),
        "bars_per_second": round(bars_per_second, 0),
        "projected_100_runs_minutes": round(projected_serial, 2),
        "system": system_info,
    }
    
    # 5. Save to JSON
    if save_json:
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        json_path = os.path.join(reports_dir, "benchmark.json")
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Benchmark] Results saved to: {json_path}")
    
    # 6. Recommendation
    print("\nRecommendation:")
    if cpu_count <= 2:
        print("-> Use 2 parallel workers. This will saturate physical cores without locking the OS.")
    else:
        print(f"-> Use {cpu_count - 1} parallel workers.")
    
    return results


if __name__ == "__main__":
    benchmark()
