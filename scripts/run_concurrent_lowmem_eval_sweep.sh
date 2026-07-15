#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
PYTHON_BIN="${POINTCEPT_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
TRAIN_PID="${TRAIN_PID:-2211}"
GPU_INDEX="${GPU_INDEX:-0}"
MEMORY_FRACTION="${MEMORY_FRACTION:-0.17}"
MIN_FREE_MIB="${MIN_FREE_MIB:-700}"
RUN_ID="kpconvx-v17-scale04_20260714_1553"
MANIFEST="$PROJECT/configs/s3dis/eval/v17_scale04_epoch_sweep.json"
OUTPUT_ROOT="$PROJECT/exp/fixed_protocol/results/v17-scale04-concurrent-screening"
LOG_ROOT="$PROJECT/exp/fixed_protocol/logs"
TELEMETRY_ROOT="$PROJECT/exp/fixed_protocol/telemetry"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/v17_scale04_concurrent_sweep_${TIMESTAMP}.log"
RUN_STATE="$LOG_ROOT/v17_scale04_concurrent_sweep_${TIMESTAMP}.state"
TELEMETRY="$TELEMETRY_ROOT/v17_scale04_concurrent_sweep_${TIMESTAMP}_gpu.csv"
LATEST_LOG="$LOG_ROOT/latest_v17_scale04_concurrent_sweep.log"
LATEST_STATE="$LOG_ROOT/latest_v17_scale04_concurrent_sweep.state"
LOCK_FILE="$PROJECT/exp/fixed_protocol/v17_scale04_concurrent_sweep.lock"
MODEL_DIR="$PROJECT/exp/s3dis/$RUN_ID/model"

mkdir -p "$LOG_ROOT" "$TELEMETRY_ROOT" "$OUTPUT_ROOT"
cd "$PROJECT"

if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "Training PID is not running: $TRAIN_PID" >&2
  exit 3
fi
TRAIN_START_TIME="$(awk '{print $22}' "/proc/$TRAIN_PID/stat")"
TRAIN_CMDLINE="$(tr '\0' ' ' < "/proc/$TRAIN_PID/cmdline")"
if [[ "$TRAIN_CMDLINE" != *"tools/train.py"* || "$TRAIN_CMDLINE" != *"$RUN_ID"* ]]; then
  echo "PID $TRAIN_PID is not the expected training process: $TRAIN_CMDLINE" >&2
  exit 3
fi
if [ ! -f "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 3
fi
if ! find "$MODEL_DIR" -maxdepth 1 -type f -name 'epoch_*.pth' -print -quit | grep -q .; then
  echo "No immutable epoch checkpoints found in: $MODEL_DIR" >&2
  exit 3
fi

exec 9> "$LOCK_FILE"
if ! flock -n 9; then
  echo "Another concurrent sweep holds $LOCK_FILE" >&2
  exit 5
fi

echo "$RUN_LOG"
ln -sfn "$RUN_LOG" "$LATEST_LOG"
ln -sfn "$RUN_STATE" "$LATEST_STATE"
echo "timestamp,memory_used_mib,memory_free_mib,gpu_util_percent,power_draw_w,training_alive" > "$TELEMETRY"

# Epoch checkpoints are immutable. Mutable model_best/model_last are intentionally excluded.
setsid nice -n 10 ionice -c 3 env \
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
    --fragment-batch-size-test 1 \
    --fallback-fragment-batch-sizes 1 \
    --fragment-log-interval-test 20 \
    --num-worker-test 2 \
    --run-ids "$RUN_ID" \
    --cuda-memory-fraction "$MEMORY_FRACTION" \
    > "$RUN_LOG" 2>&1 &
EVAL_PID=$!

{
  echo "wrapper_pid=$$"
  echo "evaluation_pid=$EVAL_PID"
  echo "training_pid=$TRAIN_PID"
  echo "training_start_time=$TRAIN_START_TIME"
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

echo "Concurrent checkpoint sweep PID: $EVAL_PID"
echo "Training PID (read-only monitor): $TRAIN_PID"

while kill -0 "$EVAL_PID" 2>/dev/null; do
  TRAINING_ALIVE=1
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    TRAINING_ALIVE=0
  elif [ "$(awk '{print $22}' "/proc/$TRAIN_PID/stat")" != "$TRAIN_START_TIME" ]; then
    TRAINING_ALIVE=0
  fi
  GPU_ROW="$(nvidia-smi \
    --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader,nounits \
    --id="$GPU_INDEX")"
  MEMORY_USED="$(echo "$GPU_ROW" | awk -F',' '{gsub(/ /, "", $1); print $1}')"
  MEMORY_FREE="$(echo "$GPU_ROW" | awk -F',' '{gsub(/ /, "", $2); print $2}')"
  GPU_UTIL="$(echo "$GPU_ROW" | awk -F',' '{gsub(/ /, "", $3); print $3}')"
  POWER_DRAW="$(echo "$GPU_ROW" | awk -F',' '{gsub(/ /, "", $4); print $4}')"
  echo "$(date -Is),$MEMORY_USED,$MEMORY_FREE,$GPU_UTIL,$POWER_DRAW,$TRAINING_ALIVE" >> "$TELEMETRY"

  if [ "$TRAINING_ALIVE" -ne 1 ]; then
    echo "Expected training PID disappeared; stopping evaluation only." >> "$RUN_LOG"
    terminate_eval
    exit 6
  fi
  if [ "$MEMORY_FREE" -lt "$MIN_FREE_MIB" ]; then
    echo "GPU free memory fell below ${MIN_FREE_MIB} MiB; stopping evaluation only." >> "$RUN_LOG"
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
