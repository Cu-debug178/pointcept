#!/usr/bin/env bash

set -o pipefail

SERVER_ID="${1:?Usage: $0 server-a 60000 4 6}"
POINT_MAX="${2:-60000}"
FRAGMENT_BATCH_SIZE="${3:-4}"
NUM_WORKERS="${4:-6}"
PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
EXP_ROOT="${POINTCEPT_EXP_ROOT:-$PROJECT/exp}"
FIXED_ROOT="${POINTCEPT_FIXED_ROOT:-$EXP_ROOT/fixed_protocol}"
LOG_ROOT="$FIXED_ROOT/logs"
OUTPUT_ROOT="$FIXED_ROOT/results/kpconvx-fixed-${SERVER_ID}"
BUNDLE_PATH="$FIXED_ROOT/bundles/kpconvx-fixed-${SERVER_ID}-compact.zip"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/fixed_screen_${SERVER_ID}_${TIMESTAMP}.log"
STATUS_LOG="$LOG_ROOT/fixed_screen_${SERVER_ID}_status.log"
LOCK_FILE="$FIXED_ROOT/run_${SERVER_ID}.lock"

mkdir -p "$LOG_ROOT" "$FIXED_ROOT/results" "$FIXED_ROOT/bundles"
cd "$PROJECT" || exit 2

exec 9> "$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date '+%F %T') ${SERVER_ID} 已有固定复评任务运行，拒绝重复启动" \
    | tee -a "$STATUS_LOG"
  exit 3
fi

{
  echo "$(date '+%F %T') 开始固定协议 screen: ${SERVER_ID}"
  echo "point_max: ${POINT_MAX}"
  echo "fragment_batch_size: ${FRAGMENT_BATCH_SIZE}"
  echo "num_workers: ${NUM_WORKERS}"
  echo "auto_shutdown: disabled"
  if source "$PROJECT/activate_env.sh"; then
    python tools/eval_s3dis_fixed_protocol.py \
      --stage screen \
      --exp-root "$EXP_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --point-max "$POINT_MAX" \
      --fragment-batch-size-test "$FRAGMENT_BATCH_SIZE" \
      --num-worker-test "$NUM_WORKERS"
    EXIT_CODE=$?
  else
    EXIT_CODE=$?
    echo "环境激活失败，退出码: ${EXIT_CODE}"
  fi
} > "$RUN_LOG" 2>&1

if [ -d "$OUTPUT_ROOT" ]; then
  python tools/eval_s3dis_fixed_protocol.py \
    --stage bundle \
    --output-root "$OUTPUT_ROOT" \
    --bundle-path "$BUNDLE_PATH" >> "$RUN_LOG" 2>&1 || true
fi

{
  echo "$(date '+%F %T') screen 结束，退出码: ${EXIT_CODE}"
  echo "运行日志: ${RUN_LOG}"
  echo "结果包: ${BUNDLE_PATH}"
  echo "自动关机已移除，服务器保持运行"
} | tee -a "$STATUS_LOG"

sync
exit "$EXIT_CODE"
