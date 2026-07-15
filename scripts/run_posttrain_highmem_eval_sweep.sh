#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
PYTHON_BIN="${POINTCEPT_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
GPU_INDEX="${GPU_INDEX:-0}"
MEMORY_FRACTION="${MEMORY_FRACTION:-0.80}"
MIN_FREE_MIB="${MIN_FREE_MIB:-1500}"
FRAGMENT_BATCH_SIZE="${FRAGMENT_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-6}"
RUN_ID="kpconvx-v17-scale04_20260714_1553"
MANIFEST="$PROJECT/configs/s3dis/eval/v17_scale04_all_saved_checkpoints.json"
OUTPUT_ROOT="$PROJECT/exp/fixed_protocol/results/v17-scale04-concurrent-screening"
LOG_ROOT="$PROJECT/exp/fixed_protocol/logs"
TELEMETRY_ROOT="$PROJECT/exp/fixed_protocol/telemetry"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/v17_scale04_posttrain_highmem_${TIMESTAMP}.log"
RUN_STATE="$LOG_ROOT/v17_scale04_posttrain_highmem_${TIMESTAMP}.state"
TELEMETRY="$TELEMETRY_ROOT/v17_scale04_posttrain_highmem_${TIMESTAMP}_gpu.csv"
LATEST_LOG="$LOG_ROOT/latest_v17_scale04_posttrain_highmem.log"
LATEST_STATE="$LOG_ROOT/latest_v17_scale04_posttrain_highmem.state"
LOCK_FILE="$PROJECT/exp/fixed_protocol/v17_scale04_posttrain_highmem.lock"
CHECKPOINT_KINDS=(
  epoch_70 epoch_50 epoch_30 epoch_10
  epoch_170 epoch_180 epoch_190 epoch_200 best last
)

mkdir -p "$LOG_ROOT" "$TELEMETRY_ROOT" "$OUTPUT_ROOT"
cd "$PROJECT"

if pgrep -af 'tools/train.py' | grep -q "$RUN_ID"; then
  echo "Training process is still running for $RUN_ID; refusing high-memory evaluation." >&2
  exit 3
fi
if [ ! -f "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 3
fi

GPU_FREE_MIB="$(nvidia-smi \
  --query-gpu=memory.free \
  --format=csv,noheader,nounits \
  --id="$GPU_INDEX" | tr -d ' ')"
if [ "$GPU_FREE_MIB" -lt 12000 ]; then
  echo "Expected at least 12000 MiB free before high-memory evaluation; found $GPU_FREE_MIB." >&2
  exit 4
fi

exec 9> "$LOCK_FILE"
if ! flock -n 9; then
  echo "Another post-training high-memory sweep holds $LOCK_FILE" >&2
  exit 5
fi

echo "$RUN_LOG"
ln -sfn "$RUN_LOG" "$LATEST_LOG"
ln -sfn "$RUN_STATE" "$LATEST_STATE"
echo "timestamp,memory_used_mib,memory_free_mib,gpu_util_percent,power_draw_w" > "$TELEMETRY"

setsid env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
    --stage screen \
    --manifest "$MANIFEST" \
    --exp-root "$PROJECT/exp" \
    --output-root "$OUTPUT_ROOT" \
    --grid-size 0.04 \
    --point-max 60000 \
    --fallback-point-max 60000 \
    --fragment-batch-size-test "$FRAGMENT_BATCH_SIZE" \
    --fallback-fragment-batch-sizes 2 1 \
    --fragment-log-interval-test 20 \
    --num-worker-test "$NUM_WORKERS" \
    --checkpoint-kinds "${CHECKPOINT_KINDS[@]}" \
    --run-ids "$RUN_ID" \
    --cuda-memory-fraction "$MEMORY_FRACTION" \
    > "$RUN_LOG" 2>&1 &
EVAL_PID=$!

{
  echo "wrapper_pid=$$"
  echo "evaluation_pid=$EVAL_PID"
  echo "memory_fraction=$MEMORY_FRACTION"
  echo "fragment_batch_size=$FRAGMENT_BATCH_SIZE"
  echo "num_workers=$NUM_WORKERS"
  echo "manifest=$MANIFEST"
  echo "log=$RUN_LOG"
  echo "telemetry=$TELEMETRY"
  echo "output_root=$OUTPUT_ROOT"
  echo "status=running"
} > "$RUN_STATE"

terminate_eval() {
  kill -TERM -- "-$EVAL_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$EVAL_PID" 2>/dev/null || return 0
    sleep 0.5
  done
  kill -KILL -- "-$EVAL_PID" 2>/dev/null || true
}

cleanup() {
  EXIT_CODE=$?
  if kill -0 "$EVAL_PID" 2>/dev/null; then
    terminate_eval
  fi
  echo "$(date '+%F %T') wrapper_exit=$EXIT_CODE" >> "$RUN_LOG"
  sed -i "s/^status=.*/status=finished/" "$RUN_STATE"
  echo "exit_code=$EXIT_CODE" >> "$RUN_STATE"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Post-training high-memory evaluation PID: $EVAL_PID"

while kill -0 "$EVAL_PID" 2>/dev/null; do
  GPU_ROW="$(nvidia-smi \
    --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader,nounits \
    --id="$GPU_INDEX")"
  MEMORY_USED="$(echo "$GPU_ROW" | awk -F',' '{gsub(/ /, "", $1); print $1}')"
  MEMORY_FREE="$(echo "$GPU_ROW" | awk -F',' '{gsub(/ /, "", $2); print $2}')"
  GPU_UTIL="$(echo "$GPU_ROW" | awk -F',' '{gsub(/ /, "", $3); print $3}')"
  POWER_DRAW="$(echo "$GPU_ROW" | awk -F',' '{gsub(/ /, "", $4); print $4}')"
  echo "$(date -Is),$MEMORY_USED,$MEMORY_FREE,$GPU_UTIL,$POWER_DRAW" >> "$TELEMETRY"

  if [ "$MEMORY_FREE" -lt "$MIN_FREE_MIB" ]; then
    echo "GPU free memory fell below ${MIN_FREE_MIB} MiB; stopping evaluation." >> "$RUN_LOG"
    terminate_eval
    exit 7
  fi
  sleep 2
done

set +e
wait "$EVAL_PID"
EVAL_EXIT=$?
set -e
echo "Evaluation exit: $EVAL_EXIT"
echo "Log: $RUN_LOG"
echo "Telemetry: $TELEMETRY"
exit "$EVAL_EXIT"
