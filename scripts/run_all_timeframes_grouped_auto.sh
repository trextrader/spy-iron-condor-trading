#!/usr/bin/env bash
set -euo pipefail

TF_LIST=(
    "SPY_m1_clean_v43.csv:M1"
    "SPY_m5_clean_v43.csv:M5"
    "SPY_m15_clean_v43.csv:M15"
    "SPY_h1_clean_v43.csv:H1"
)

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$BASE_DIR/data/Datasetv4"
SCRIPT_DIR="$BASE_DIR/scripts/indicators"

show_tail() {
    tail -n 31 "$1"
}

banner() {
    echo ""
    echo "============================================"
    echo ">>> $1"
    echo "============================================"
}

run_groups_for_file() {
    INPUT_FILE="$1"
    PREFIX="$2"

    echo ""
    echo "====================================================="
    echo "PROCESSING TIMEFRAME: $PREFIX  ($INPUT_FILE)"
    echo "====================================================="

    WORK="${INPUT_FILE%.csv}_tmp.csv"
    FINAL="${INPUT_FILE%.csv}_enriched.csv"

    cp "$DATA_DIR/$INPUT_FILE" "$DATA_DIR/$WORK"

    # Normalize columns BEFORE indicators
    python "$BASE_DIR/scripts/normalize_columns.py" "$DATA_DIR/$WORK" "$DATA_DIR/$WORK"

    # GROUP 1
    banner "GROUP 1: Trend Indicators"
    python "$SCRIPT_DIR/sma.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    python "$SCRIPT_DIR/psar_adaptive.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 2
    banner "GROUP 2: Dynamic Bollinger"
    python "$SCRIPT_DIR/bb_dynamic.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 3
    banner "GROUP 3: Fractal Adaptive"
    python "$SCRIPT_DIR/fractal_adaptive.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 4
    banner "GROUP 4: Volume & Flow"
    python "$SCRIPT_DIR/volume_flow.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 5
    banner "GROUP 5: Breadth"
    python "$SCRIPT_DIR/mcclellan_osc.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 6
    banner "GROUP 6: Returns & Volatility"
    python "$SCRIPT_DIR/returns_volatility.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 7
    banner "GROUP 7: Pivots & Slopes"
    python "$SCRIPT_DIR/pivots_slopes.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK" --prefix "$PREFIX"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 8
    banner "GROUP 8: Dynamic Signals"
    python "$SCRIPT_DIR/dynamic_signals.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK" --prefix "$PREFIX"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 9
    banner "GROUP 9: Reversal Stack"
    python "$SCRIPT_DIR/reversal_stack.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 10
    banner "GROUP 10: Execution & Risk"
    python "$SCRIPT_DIR/execution_risk.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 11
    banner "GROUP 11: Chaos & Regime"
    python "$SCRIPT_DIR/chaos_regime.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 12
    banner "GROUP 12: Microstructure"
    python "$SCRIPT_DIR/microstructure.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    # GROUP 13
    banner "GROUP 13: Max Drawdown"
    python "$SCRIPT_DIR/max_dd_60m.py" --input "$DATA_DIR/$WORK" --output "$DATA_DIR/$WORK"
    show_tail "$DATA_DIR/$WORK"

    banner "FINALIZING TIMEFRAME $PREFIX"
    mv "$DATA_DIR/$WORK" "$DATA_DIR/$FINAL"
    echo "Enriched file written to: $DATA_DIR/$FINAL"
}

for ENTRY in "${TF_LIST[@]}"; do
    FILE="${ENTRY%%:*}"
    PREFIX="${ENTRY##*:}"
    run_groups_for_file "$FILE" "$PREFIX"
done

echo ""
echo "====================================================="
echo "ALL TIMEFRAMES COMPLETE"
echo "====================================================="
