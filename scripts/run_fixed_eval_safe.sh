#!/usr/bin/env bash

set -o pipefail

SERVER_ID="${1:-}"
STAGE="${2:-}"
POINT_MAX="${3:-60000}"
FRAGMENT_BATCH_SIZE="${4:-4}"
NUM_WORKERS="${5:-6}"
CHECKPOINT_SELECTION="${6:-best}"
RUN_ID="${7:-}"

if [[ ! "$SERVER_ID" =~ ^server-[ab]$ ]]; then
  echo "Usage: $0 server-a|server-b preflight|screen [point_max] [fragment_batch] [workers] [best|last|all] [run_id]"
  exit 2
fi
if [[ "$STAGE" != "preflight" && "$STAGE" != "screen" ]]; then
  echo "Usage: $0 server-a|server-b preflight|screen [point_max] [fragment_batch] [workers] [best|last|all] [run_id]"
  exit 2
fi
if [[ "$CHECKPOINT_SELECTION" != "best" && "$CHECKPOINT_SELECTION" != "last" && "$CHECKPOINT_SELECTION" != "all" ]]; then
  echo "Checkpoint selection must be best, last, or all"
  exit 2
fi

PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
PYTHON_BIN="${POINTCEPT_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
EXP_ROOT="${POINTCEPT_EXP_ROOT:-$PROJECT/exp}"
FIXED_ROOT="${POINTCEPT_FIXED_ROOT:-$EXP_ROOT/fixed_protocol}"
LOG_ROOT="$FIXED_ROOT/logs"
RESULT_ROOT="$FIXED_ROOT/results/kpconvx-fixed-${SERVER_ID}"
BUNDLE_PATH="$FIXED_ROOT/bundles/kpconvx-fixed-${SERVER_ID}-compact.zip"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/fixed_${STAGE}_${SERVER_ID}_${TIMESTAMP}.log"
HEARTBEAT_LOG="$LOG_ROOT/fixed_${STAGE}_${SERVER_ID}_${TIMESTAMP}.heartbeat"
EXIT_FILE="$LOG_ROOT/fixed_${STAGE}_${SERVER_ID}_${TIMESTAMP}.exit"
PID_FILE="$FIXED_ROOT/fixed_${STAGE}_${SERVER_ID}.pid"
LOCK_FILE="$FIXED_ROOT/fixed_${STAGE}_${SERVER_ID}.lock"
LATEST_LOG="$LOG_ROOT/latest_${STAGE}_${SERVER_ID}.log"

mkdir -p "$LOG_ROOT" "$FIXED_ROOT/results" "$FIXED_ROOT/bundles"
printf '%s\n' "$RUN_LOG"
exec >> "$RUN_LOG" 2>&1

echo "$(date '+%F %T') safe fixed evaluation launcher"
echo "host=$(hostname)"
echo "server_id=$SERVER_ID"
echo "stage=$STAGE"
echo "project=$PROJECT"
echo "python=$PYTHON_BIN"
echo "point_max=$POINT_MAX"
echo "fragment_batch_size=$FRAGMENT_BATCH_SIZE"
echo "num_workers=$NUM_WORKERS"
echo "checkpoint_selection=$CHECKPOINT_SELECTION"
echo "run_id=${RUN_ID:-all}"
echo "power_action=none"

cd "$PROJECT" || exit 3
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter is not executable: $PYTHON_BIN"
  exit 4
fi

# Neutralize historical wrappers before starting any GPU work. This only stops
# stale processes and writes their old cancellation markers.
touch "$EXP_ROOT/cancel_server-a_shutdown" "$EXP_ROOT/cancel_server-b_shutdown"
touch "$FIXED_ROOT/cancel_server-a_shutdown" "$FIXED_ROOT/cancel_server-b_shutdown"
LEGACY_PIDS="$(pgrep -f '[r]un_fixed_screen_then_shutdown.sh|[w]atch_train_then_shutdown.sh' || true)"
if [ -n "$LEGACY_PIDS" ]; then
  echo "Stopping legacy wrapper PIDs: $LEGACY_PIDS"
  kill -TERM $LEGACY_PIDS 2>/dev/null || true
  sleep 2
  LEGACY_PIDS="$(pgrep -f '[r]un_fixed_screen_then_shutdown.sh|[w]atch_train_then_shutdown.sh' || true)"
  if [ -n "$LEGACY_PIDS" ]; then
    echo "Force-stopping legacy wrapper PIDs: $LEGACY_PIDS"
    kill -KILL $LEGACY_PIDS 2>/dev/null || true
  fi
fi

exec 9> "$LOCK_FILE"
if ! flock -n 9; then
  echo "Another $STAGE task already holds $LOCK_FILE"
  exit 5
fi

ln -sfn "$RUN_LOG" "$LATEST_LOG"
printf '%s\n' "$$" > "$PID_FILE"

(
  while kill -0 "$$" 2>/dev/null; do
    echo "$(date '+%F %T') alive host=$(hostname) pid=$$"
    sleep 30
  done
) >> "$HEARTBEAT_LOG" 2>&1 &
HEARTBEAT_PID=$!

cleanup() {
  EXIT_CODE=$?
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "$EXIT_CODE" > "$EXIT_FILE"
  echo "$(date '+%F %T') launcher_exit=$EXIT_CODE"
}
on_term() {
  echo "$(date '+%F %T') received TERM"
  exit 143
}
on_int() {
  echo "$(date '+%F %T') received INT"
  exit 130
}
trap cleanup EXIT
trap on_term TERM
trap on_int INT

export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --exp-root "$EXP_ROOT"
  --output-root "$RESULT_ROOT"
  --point-max "$POINT_MAX"
  --fragment-batch-size-test "$FRAGMENT_BATCH_SIZE"
  --num-worker-test "$NUM_WORKERS"
)

if [ "$CHECKPOINT_SELECTION" = "all" ]; then
  CHECKPOINT_ARGS=(--checkpoint-kinds best last)
else
  CHECKPOINT_ARGS=(--checkpoint-kinds "$CHECKPOINT_SELECTION")
fi
if [ -n "$RUN_ID" ]; then
  RUN_ARGS=(--run-ids "$RUN_ID")
else
  RUN_ARGS=()
fi

if [ "$STAGE" = "preflight" ]; then
  "$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
    --stage preflight \
    "${COMMON_ARGS[@]}" \
    --checkpoint-kinds best \
    --fallback-point-max 40000 \
    --fallback-fragment-batch-sizes 2 1
  exit $?
fi

"$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
  --stage screen \
  "${COMMON_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${RUN_ARGS[@]}"
TASK_EXIT=$?

if [ "$TASK_EXIT" -eq 0 ] && [ -d "$RESULT_ROOT" ]; then
  "$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
    --stage bundle \
    --output-root "$RESULT_ROOT" \
    --bundle-path "$BUNDLE_PATH" || true
fi

exit "$TASK_EXIT"
