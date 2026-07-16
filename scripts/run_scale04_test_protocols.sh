#!/usr/bin/env bash

set -o pipefail

PROTOCOL="${1:-all}"
MAX_ROOMS="${2:-}"
POINT_MAX="${POINT_MAX:-60000}"
FRAGMENT_BATCH_SIZE="${FRAGMENT_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-6}"

if [[ "$PROTOCOL" != "tta13" && "$PROTOCOL" != "official_vote10" && "$PROTOCOL" != "all" ]]; then
  echo "Usage: $0 tta13|official_vote10|all [max_rooms]"
  exit 2
fi
if [[ -n "$MAX_ROOMS" && ! "$MAX_ROOMS" =~ ^[1-9][0-9]*$ ]]; then
  echo "max_rooms must be a positive integer"
  exit 2
fi

PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
PYTHON_BIN="${POINTCEPT_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
EXP_ROOT="${POINTCEPT_EXP_ROOT:-$PROJECT/exp}"
FIXED_ROOT="${POINTCEPT_FIXED_ROOT:-$EXP_ROOT/fixed_protocol}"
MANIFEST="$PROJECT/configs/s3dis/eval/kpconvx_scale04_baseline_checkpoints.json"
RUN_ID="kpconvx-scale04-baseline_20260710_2057"
RESULT_NAME="scale04-baseline-epoch200-augmented-test"
if [[ -n "$MAX_ROOMS" ]]; then
  RESULT_NAME="${RESULT_NAME}-smoke-${MAX_ROOMS}rooms"
fi
RESULT_ROOT="$FIXED_ROOT/results/$RESULT_NAME"
LOG_ROOT="$FIXED_ROOT/logs/scale04-test-protocols"
BUNDLE_PATH="$FIXED_ROOT/bundles/${RESULT_NAME}-compact.zip"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/${PROTOCOL}_${TIMESTAMP}.log"
EXIT_FILE="$LOG_ROOT/${PROTOCOL}_${TIMESTAMP}.exit"
LATEST_LOG="$LOG_ROOT/latest_${PROTOCOL}.log"
LOCK_FILE="$FIXED_ROOT/${RESULT_NAME}.lock"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT" "$FIXED_ROOT/bundles"
printf '%s\n' "$RUN_LOG"
exec >> "$RUN_LOG" 2>&1

echo "$(date '+%F %T') scale04 augmented test"
echo "host=$(hostname)"
echo "protocol=$PROTOCOL"
echo "manifest=$MANIFEST"
echo "run_id=$RUN_ID"
echo "checkpoint=epoch_200"
echo "result_root=$RESULT_ROOT"
echo "grid_size=0.04"
echo "point_max=$POINT_MAX"
echo "fragment_batch_size=$FRAGMENT_BATCH_SIZE"
echo "num_workers=$NUM_WORKERS"
echo "max_rooms=${MAX_ROOMS:-all}"
echo "official_vote_count=10"
echo "official_vote_momentum=0.95"
echo "augmentation_seed=0"
echo "power_action=none"

cd "$PROJECT" || exit 3
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN"
  exit 4
fi

exec 9> "$LOCK_FILE"
if ! flock -n 9; then
  echo "Another augmented test already holds $LOCK_FILE"
  exit 5
fi

ln -sfn "$RUN_LOG" "$LATEST_LOG"

cleanup() {
  STATUS=$?
  echo "$STATUS" > "$EXIT_FILE"
  echo "$(date '+%F %T') launcher_exit=$STATUS"
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
  --fragment-log-interval-test 20
  --checkpoint-kinds epoch_200
  --run-ids "$RUN_ID"
  --vote-count 10
  --vote-momentum 0.95
  --augmentation-seed 0
  --strict-local
)
if [[ -n "$MAX_ROOMS" ]]; then
  COMMON_ARGS+=(--max-rooms "$MAX_ROOMS")
fi

run_protocol() {
  local stage="$1"
  "$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
    --stage "$stage" \
    "${COMMON_ARGS[@]}"
}

if [[ "$PROTOCOL" == "tta13" || "$PROTOCOL" == "all" ]]; then
  run_protocol tta13 || exit $?
fi
if [[ "$PROTOCOL" == "official_vote10" || "$PROTOCOL" == "all" ]]; then
  run_protocol official_vote10 || exit $?
fi

"$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
  --stage bundle \
  --manifest "$MANIFEST" \
  --output-root "$RESULT_ROOT" \
  --bundle-path "$BUNDLE_PATH" || true
