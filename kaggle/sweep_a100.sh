#!/usr/bin/env bash
# =============================================================================
# sweep_a100.sh  —  Full Bayesian optimization sweep: min-med → sobol
#
# Runs all 5 intensity levels sequentially on the 5-year (2020-2024) dataset.
# Survives SSH disconnects via nohup.  Supports resume after failure.
#
# USAGE
#   # From project root — launch and detach:
#   nohup bash kaggle/sweep_a100.sh 2>&1 | tee logs/sweep_launch_$(date +%Y%m%d).log &
#
#   # Monitor progress:
#   tail -f logs/sweep_<TIMESTAMP>/master.log
#
#   # Resume after crash (completed levels are skipped automatically):
#   nohup bash kaggle/sweep_a100.sh 2>&1 | tee logs/sweep_resume_$(date +%Y%m%d).log &
#
#   # Force-retry a specific level (e.g. med):
#   sed -i '/^med$/d' logs/.sweep_state_a100.txt
#   nohup bash kaggle/sweep_a100.sh 2>&1 | tee logs/sweep_resume_$(date +%Y%m%d).log &
#
#   # Start a completely fresh sweep:
#   rm -f logs/.sweep_state_a100.txt logs/.bench_a100_min-med_secs.txt
#   nohup bash kaggle/sweep_a100.sh 2>&1 | tee logs/sweep_fresh_$(date +%Y%m%d).log &
#
# OVERRIDE ENV VARS
#   GPUTYPE=h100 bash kaggle/sweep_a100.sh    # use H100 instead of A100
#   BENCH_STRATEGY=short_call bash kaggle/sweep_a100.sh
#   PYTHON=/opt/conda/bin/python bash kaggle/sweep_a100.sh
# =============================================================================
set -uo pipefail

# ── Configurable via environment ──────────────────────────────────────────────
GPUTYPE="${GPUTYPE:-a100}"
MODEL="${MODEL:-models/condor_net_v43_run18.pth}"
SCRIPT="kaggle/condor_brain_backtest_v45_gpu.py"
PYTHON="${PYTHON:-python}"
BENCH_STRATEGY="${BENCH_STRATEGY:-iron_butterfly}"   # single strategy for calibration
N_STRATEGIES=54                                       # active strategies (54/58; 4 skip class_idx=6)

# ── Persistent state (survives re-runs, shared across sessions) ───────────────
mkdir -p logs
STATE_FILE="logs/.sweep_state_${GPUTYPE}.txt"         # completed levels — edit to retry
BENCH_CACHE="logs/.bench_${GPUTYPE}_min-med_secs.txt" # cached benchmark time in seconds

# ── Per-run directories (unique timestamp so logs never overwrite) ────────────
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/sweep_${RUN_ID}"
ARCHIVE_DIR="reports/sweep_archive_${RUN_ID}"
MASTER_LOG="${LOG_DIR}/master.log"
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR"
touch "$STATE_FILE"

# ── Levels + work ratios ──────────────────────────────────────────────────────
# Ratios derived from: Phase1(init × 25% bars) + Phase2(rounds×batch × 60%) + Phase3(topK × 100%)
#   min-med : 512×0.25 + 3×64×0.60  + 9×1.00  = 128 + 115 + 9   = 252  → 1.00×
#   med     : 1024×0.25 + 4×128×0.60 + 11×1.00 = 256 + 307 + 11  = 574  → 2.28×
#   med-max : 2048×0.25 + 5×128×0.60 + 13×1.00 = 512 + 384 + 13  = 909  → 3.61×
#   max     : 4096×0.25 + 6×256×0.60 + 15×1.00 = 1024+ 922 + 15  = 1961 → 7.78×
#   sobol   : 1024×0.25 + 0          + 11×1.00 = 256 + 0   + 11  = 267  → 1.06×
# Override via env: LEVELS_OVERRIDE="min-med med" bash sweep_a100.sh
if [ -n "${LEVELS_OVERRIDE:-}" ]; then
    read -ra LEVELS <<< "$LEVELS_OVERRIDE"
else
    LEVELS=("min-med" "med" "med-max" "max" "sobol")
fi
declare -A WORK_RATIO
WORK_RATIO["min-med"]="1.00"
WORK_RATIO["med"]="2.28"
WORK_RATIO["med-max"]="3.61"
WORK_RATIO["max"]="7.78"
WORK_RATIO["sobol"]="1.06"

# ── Helper functions ──────────────────────────────────────────────────────────
log()     { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$MASTER_LOG"; }
hr()      { printf '─%.0s' {1..72} | tee -a "$MASTER_LOG"; printf '\n' | tee -a "$MASTER_LOG"; }
is_done() { grep -qxF "$1" "$STATE_FILE" 2>/dev/null; }
mark_done() { echo "$1" >> "$STATE_FILE"; }

fmt_time() {
    local secs=$1
    local h=$(( secs / 3600 ))
    local m=$(( (secs % 3600) / 60 ))
    printf '%dh %02dm' "$h" "$m"
}

# ── Banner ────────────────────────────────────────────────────────────────────
hr
log "A100 OPTIMIZATION SWEEP  (run_id=${RUN_ID})"
log "  gputype    : ${GPUTYPE}"
log "  model      : ${MODEL}"
log "  levels     : ${LEVELS[*]}"
log "  log dir    : ${LOG_DIR}/"
log "  archive    : ${ARCHIVE_DIR}/"
log "  state file : ${STATE_FILE}  (edit to retry a level)"
hr

# =============================================================================
# PHASE 0: BENCHMARK
# Run one strategy at min-med to get real per-strategy time on this machine.
# Result is cached in BENCH_CACHE; delete it to force re-calibration.
# =============================================================================
if [ -f "$BENCH_CACHE" ]; then
    BENCH_SECS=$(cat "$BENCH_CACHE")
    log "BENCHMARK — using cached result: ${BENCH_SECS}s/strategy at min-med on ${GPUTYPE}"
    log "  (delete ${BENCH_CACHE} to force re-calibration)"
else
    BENCH_LOG="${LOG_DIR}/benchmark_${BENCH_STRATEGY}_min-med.log"
    log "BENCHMARK — running '${BENCH_STRATEGY}' at min-med on ${GPUTYPE}..."
    log "  log: ${BENCH_LOG}"

    BENCH_T0=$(date +%s)
    $PYTHON $SCRIPT \
        --5years --no-friday-closeout \
        --v43-model "$MODEL" \
        --strategies "$BENCH_STRATEGY" \
        --optimize --optimize-mode bayes \
        --gputype "$GPUTYPE" \
        --gpuintensity min-med \
        > "$BENCH_LOG" 2>&1
    BENCH_EXIT=$?
    BENCH_T1=$(date +%s)
    BENCH_SECS=$(( BENCH_T1 - BENCH_T0 ))

    if [ $BENCH_EXIT -ne 0 ]; then
        log "WARNING: Benchmark exited with code ${BENCH_EXIT}"
        log "  Last 10 lines of ${BENCH_LOG}:"
        tail -10 "$BENCH_LOG" | tee -a "$MASTER_LOG"
        log "  Using fallback estimate of 120s/strategy — estimates will be inaccurate."
        BENCH_SECS=120
    else
        echo "$BENCH_SECS" > "$BENCH_CACHE"
        log "BENCHMARK — done: ${BENCH_SECS}s ($(fmt_time $BENCH_SECS)) for 1 strategy at min-med"
    fi
fi

# =============================================================================
# PHASE 0b: PRINT TIME ESTIMATES
# =============================================================================
hr
log "TIME ESTIMATES  (${N_STRATEGIES} strategies × ${BENCH_SECS}s calibration at min-med)"
log ""
log "  Level       Work ratio    Estimated time"
log "  ----------  ----------    --------------"

TOTAL_EST_SECS=0
for LV in "${LEVELS[@]}"; do
    RATIO="${WORK_RATIO[$LV]}"
    EST_SECS=$(awk "BEGIN {printf \"%d\", $BENCH_SECS * $N_STRATEGIES * $RATIO}")
    TOTAL_EST_SECS=$(( TOTAL_EST_SECS + EST_SECS ))
    DONE_MARKER=""
    is_done "$LV" && DONE_MARKER="  ✓ already done"
    printf '  %-10s  ×%-9s    ~%s%s\n' \
        "$LV" "$RATIO" "$(fmt_time $EST_SECS)" "$DONE_MARKER" \
        | tee -a "$MASTER_LOG"
done

log ""
log "  TOTAL                     ~$(fmt_time $TOTAL_EST_SECS)"
log ""
log "  Note: GP-fitting (CPU-bound) adds ~20-50% overhead at med-max and max."
log "  Actual time will vary; estimates assume linear scaling from calibration."
hr

# =============================================================================
# SWEEP LOOP
# =============================================================================
SWEEP_START=$(date +%s)

for LV in "${LEVELS[@]}"; do

    # ── Skip completed levels (resume mode) ───────────────────────────────────
    if is_done "$LV"; then
        log "SKIP  ${LV} — already in ${STATE_FILE}"
        continue
    fi

    RUN_LOG="${LOG_DIR}/run_${LV}.log"
    CKPT_FILE="logs/.ckpt_${GPUTYPE}_${LV}.json"   # persistent: survives re-runs

    hr
    RATIO="${WORK_RATIO[$LV]}"
    EST_SECS=$(awk "BEGIN {printf \"%d\", $BENCH_SECS * $N_STRATEGIES * $RATIO}")
    log "START  level=${LV}  gputype=${GPUTYPE}  est=$(fmt_time $EST_SECS)"
    log "  run log    : ${RUN_LOG}"
    log "  checkpoint : ${CKPT_FILE}  (strategy-level resume within this level)"
    log "  archive    : ${ARCHIVE_DIR}/${LV}/"

    # Timestamp reference for report archiving — created just before the run
    STAMP=$(mktemp)
    T0=$(date +%s)

    $PYTHON $SCRIPT \
        --5years --no-friday-closeout \
        --v43-model "$MODEL" \
        --strategies autoall \
        --optimize --optimize-mode bayes \
        --gputype "$GPUTYPE" \
        --gpuintensity "$LV" \
        --checkpoint "$CKPT_FILE" \
        > "$RUN_LOG" 2>&1

    EXIT_CODE=$?
    T1=$(date +%s)
    ELAPSED=$(( T1 - T0 ))

    # ── Archive outputs ────────────────────────────────────────────────────────
    LEVEL_ARCHIVE="${ARCHIVE_DIR}/${LV}"
    mkdir -p "$LEVEL_ARCHIVE"

    # BO CSV reports produced during this run
    find reports/ -maxdepth 1 -name "bo_*.csv" -newer "$STAMP" \
        -exec cp {} "$LEVEL_ARCHIVE/" \; 2>/dev/null || true

    # Strategy params snapshot (git-tracked, but snapshot for offline review)
    [ -d "kaggle/strategies" ] && \
        cp -r kaggle/strategies/ "$LEVEL_ARCHIVE/strategies_snapshot/" 2>/dev/null || true

    # Run log and checkpoint
    cp "$RUN_LOG" "$LEVEL_ARCHIVE/"
    [ -f "$CKPT_FILE" ] && cp "$CKPT_FILE" "$LEVEL_ARCHIVE/" || true

    rm -f "$STAMP"

    # ── Result ────────────────────────────────────────────────────────────────
    if [ $EXIT_CODE -eq 0 ]; then
        log "DONE   level=${LV}  wall=$(fmt_time $ELAPSED)  exit=0"
        log "  Archived → ${LEVEL_ARCHIVE}/"
        mark_done "$LV"
    else
        log "FAIL   level=${LV}  exit=${EXIT_CODE}  wall=$(fmt_time $ELAPSED)"
        log ""
        log "  Last 30 lines of ${RUN_LOG}:"
        tail -30 "$RUN_LOG" | tee -a "$MASTER_LOG"
        log ""
        log "  Archived partial outputs → ${LEVEL_ARCHIVE}/"
        log ""
        log "  TO RESUME from this level:"
        log "    nohup bash kaggle/sweep_a100.sh 2>&1 | tee logs/sweep_resume_\$(date +%Y%m%d).log &"
        log ""
        log "  TO RETRY this level from scratch:"
        log "    sed -i '/^${LV}\$/d' ${STATE_FILE}"
        log "    rm -f ${CKPT_FILE}"
        log "    nohup bash kaggle/sweep_a100.sh 2>&1 | tee logs/sweep_resume_\$(date +%Y%m%d).log &"
        exit $EXIT_CODE
    fi

done

# =============================================================================
# FINAL SUMMARY
# =============================================================================
SWEEP_END=$(date +%s)
SWEEP_TOTAL=$(( SWEEP_END - SWEEP_START ))

hr
log "SWEEP COMPLETE"
log "  Total wall time   : $(fmt_time $SWEEP_TOTAL)"
log "  Logs              : ${LOG_DIR}/"
log "  Archives          : ${ARCHIVE_DIR}/"
log "  Strategy files    : kaggle/strategies/  (auto-committed + pushed each run)"
log "  State file        : ${STATE_FILE}"
log ""
log "  Per-level wall times:"
for LV in "${LEVELS[@]}"; do
    LEVEL_LOG="${LOG_DIR}/run_${LV}.log"
    if [ -f "$LEVEL_LOG" ]; then
        # Extract actual duration from log if available, else show "skipped"
        printf '    %-10s  see %s\n' "$LV" "$LEVEL_LOG" | tee -a "$MASTER_LOG"
    else
        printf '    %-10s  skipped (was already done)\n' "$LV" | tee -a "$MASTER_LOG"
    fi
done
hr
