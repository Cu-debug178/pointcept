#!/usr/bin/env bash

set -u

WATCHED_PID="${1:?Usage: $0 watched_pid output_csv [interval_seconds]}"
OUTPUT_CSV="${2:?Usage: $0 watched_pid output_csv [interval_seconds]}"
INTERVAL="${3:-10}"

mkdir -p "$(dirname "$OUTPUT_CSV")"
if [ ! -s "$OUTPUT_CSV" ]; then
  echo "timestamp,index,name,pstate,memory_used_mib,memory_total_mib,gpu_util_percent,memory_util_percent,power_draw_w,power_limit_w,temperature_c,graphics_clock_mhz,memory_clock_mhz" > "$OUTPUT_CSV"
fi

while kill -0 "$WATCHED_PID" 2>/dev/null; do
  TIMESTAMP="$(date -Is)"
  nvidia-smi \
    --query-gpu=index,name,pstate,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,clocks.current.graphics,clocks.current.memory \
    --format=csv,noheader,nounits \
    | sed "s/^/${TIMESTAMP},/" >> "$OUTPUT_CSV"
  sleep "$INTERVAL"
done
