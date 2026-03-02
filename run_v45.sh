#!/usr/bin/env bash
nohup python kaggle/condor_brain_backtest_v45.py \
  --use-v43 \
  --v43-model models/condornet_v43_run18.pth \
  --v43-data-dir data/Datasetv4/v43 \
  --strategyomit "iron_condor" \
  --profittargets \
  "$@" \
  > backtest_run.log 2>&1 &
echo "Started PID $!"
echo "Tail with: tail -f backtest_run.log"
