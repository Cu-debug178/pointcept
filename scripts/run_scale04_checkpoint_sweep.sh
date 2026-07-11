#!/usr/bin/env bash

set -o pipefail

STAGE="${1:-screen}"
POINT_MAX="${2:-60000}"
FRAGMENT_BATCH_SIZE="${3:-2}"
NUM_WORKERS="${4:-6}"

if [[ "$STAGE" != "preflight" && "$STAGE" != "screen" ]]; then
  echo "Usage: $0 preflight|screen [point_max] [fragment_batch_size] [num_workers]"
  exit 2
fi

PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
PYTHON_BIN="${POINTCEPT_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
EXP_ROOT="${POINTCEPT_EXP_ROOT:-$PROJECT/exp}"
FIXED_ROOT="${POINTCEPT_FIXED_ROOT:-$EXP_ROOT/fixed_protocol}"
MANIFEST="$PROJECT/configs/s3dis/eval/kpconvx_scale04_baseline_checkpoints.json"
RESULT_ROOT="$FIXED_ROOT/results/scale04-baseline-all-checkpoints"
LOG_ROOT="$FIXED_ROOT/logs/scale04-baseline"
BUNDLE_PATH="$FIXED_ROOT/bundles/scale04-baseline-all-checkpoints-compact.zip"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/${STAGE}_${TIMESTAMP}.log"
EXIT_FILE="$LOG_ROOT/${STAGE}_${TIMESTAMP}.exit"
LATEST_LOG="$LOG_ROOT/latest_${STAGE}.log"
PID_FILE="$FIXED_ROOT/scale04_${STAGE}.pid"
LOCK_FILE="$FIXED_ROOT/scale04_${STAGE}.lock"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT" "$FIXED_ROOT/bundles"
printf '%s\n' "$RUN_LOG"
exec >> "$RUN_LOG" 2>&1

echo "$(date '+%F %T') scale04 checkpoint sweep"
echo "host=$(hostname)"
echo "stage=$STAGE"
echo "project=$PROJECT"
echo "manifest=$MANIFEST"
echo "result_root=$RESULT_ROOT"
echo "grid_size=0.04"
echo "point_max=$POINT_MAX"
echo "fragment_batch_size=$FRAGMENT_BATCH_SIZE"
echo "num_workers=$NUM_WORKERS"
echo "power_action=none"

cd "$PROJECT" || exit 3
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter is not executable: $PYTHON_BIN"
  exit 4
fi

exec 9> "$LOCK_FILE"
if ! flock -n 9; then
  echo "Another scale04 $STAGE task already holds $LOCK_FILE"
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

export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --manifest "$MANIFEST"
  --exp-root "$EXP_ROOT"
  --output-root "$RESULT_ROOT"
  --grid-size 0.04
  --point-max "$POINT_MAX"
  --fragment-batch-size-test "$FRAGMENT_BATCH_SIZE"
  --num-worker-test "$NUM_WORKERS"
  --strict-local
)

if [ "$STAGE" = "preflight" ]; then
  "$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
    --stage preflight \
    "${COMMON_ARGS[@]}" \
    --checkpoint-kinds best \
    --fallback-point-max 40000 \
    --fallback-fragment-batch-sizes 1
  exit $?
fi

"$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
  --stage screen \
  "${COMMON_ARGS[@]}"
TASK_EXIT=$?

if [ "$TASK_EXIT" -eq 0 ]; then
  "$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
    --stage bundle \
    --manifest "$MANIFEST" \
    --output-root "$RESULT_ROOT" \
    --bundle-path "$BUNDLE_PATH" || true
fi

exit "$TASK_EXIT"
