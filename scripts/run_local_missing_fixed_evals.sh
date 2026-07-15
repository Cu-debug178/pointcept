#!/usr/bin/env bash

set -o pipefail

PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
FIXED_ROOT="${POINTCEPT_FIXED_ROOT:-$PROJECT/exp/fixed_protocol}"
PYTHON_BIN="${POINTCEPT_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
MANIFEST="$PROJECT/configs/s3dis/eval/kpconvx_fixed_protocol_local_available.json"
RESULT_ROOT="$FIXED_ROOT/results/kpconvx-fixed-server-a"
SCALE04_ROOT="$FIXED_ROOT/results/scale04-baseline-all-checkpoints"
ANALYSIS_ROOT="$FIXED_ROOT/analysis/checkpoint-comparison"
LOG_ROOT="$FIXED_ROOT/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/local_missing_fixed_${TIMESTAMP}.log"
EXIT_FILE="${RUN_LOG%.log}.exit"
LATEST_LOG="$LOG_ROOT/latest_local_missing_fixed.log"
PID_FILE="$FIXED_ROOT/local_missing_fixed.pid"
LOCK_FILE="$FIXED_ROOT/local_missing_fixed.lock"

mkdir -p "$LOG_ROOT" "$ANALYSIS_ROOT"
printf '%s\n' "$RUN_LOG"
exec >> "$RUN_LOG" 2>&1

echo "$(date '+%F %T') local missing fixed-protocol evaluations"
echo "manifest=$MANIFEST"
echo "result_root=$RESULT_ROOT"
echo "point_max=60000"
echo "fragment_batch_size=4"
echo "num_workers=12"

cd "$PROJECT" || exit 3
exec 9> "$LOCK_FILE"
if ! flock -n 9; then
  echo "Another local missing fixed evaluation holds $LOCK_FILE"
  exit 5
fi

ln -sfn "$RUN_LOG" "$LATEST_LOG"
printf '%s\n' "$$" > "$PID_FILE"

cleanup() {
  EXIT_CODE=$?
  rm -f "$PID_FILE"
  echo "$EXIT_CODE" > "$EXIT_FILE"
  echo "$(date '+%F %T') launcher_exit=$EXIT_CODE"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

POINTCEPT_FIXED_MANIFEST="$MANIFEST" \
POINTCEPT_FIXED_RESULT_ROOT="$RESULT_ROOT" \
  bash scripts/run_fixed_eval_safe.sh server-a screen 60000 4 12 all
TASK_EXIT=$?
if [ "$TASK_EXIT" -ne 0 ]; then
  exit "$TASK_EXIT"
fi

"$PYTHON_BIN" -u tools/analyze_s3dis_checkpoint_results.py \
  --fixed-root "$RESULT_ROOT" \
  --scale04-root "$SCALE04_ROOT" \
  --output-dir "$ANALYSIS_ROOT"
