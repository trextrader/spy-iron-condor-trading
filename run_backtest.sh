#!/usr/bin/env bash
# run_backtest.sh — convenience launcher for condor_brain_backtest_v45.py
# Run from repo root: bash run_backtest.sh [extra args]
#
# Examples:
#   bash run_backtest.sh                       # 500-bar quick test (default)
#   bash run_backtest.sh --limit 1000          # 1000-bar test
#   bash run_backtest.sh --limit 0             # full dataset
#   bash run_backtest.sh --verbose --limit 200
#   bash run_backtest.sh --max-positions 3

set -euo pipefail
cd "$(dirname "$0")"   # always run from repo root regardless of cwd

CHECKPOINT="models/condornet_v43_run18.pth"
DATA_DIR="data/Datasetv4/v43"

# Only inject --limit if the caller did not pass one
EXTRA_LIMIT=()
HAS_LIMIT=0
for arg in "$@"; do
  [[ "$arg" == "--limit" ]] && HAS_LIMIT=1
done
[[ $HAS_LIMIT -eq 0 ]] && EXTRA_LIMIT=(--limit 500)

python kaggle/condor_brain_backtest_v45.py \
  --use-v43 \
  --v43-model "$CHECKPOINT" \
  --v43-data-dir "$DATA_DIR" \
  "${EXTRA_LIMIT[@]}" \
  "$@"
