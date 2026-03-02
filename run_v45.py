"""
run_v45.py — launch condor_brain_backtest_v45.py with standard args.
Usage:
  python run_v45.py                   # default run
  python run_v45.py --verbose         # with verbose
  python run_v45.py --limit 500       # quick 500-bar test
  python run_v45.py --verbose --limit 200
"""
import sys, subprocess

BASE_ARGS = [
    sys.executable,
    "kaggle/condor_brain_backtest_v45.py",
    "--use-v43",
    "--v43-model",    "models/condor_net_v43_run18.pth",
    "--v43-data-dir", "data/Datasetv4/v43",
    "--strategyomit", "iron_condor",
    "--profittargets",
]

cmd = BASE_ARGS + sys.argv[1:]
print("Running:", " ".join(cmd))
subprocess.run(cmd)
