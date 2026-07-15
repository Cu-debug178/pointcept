#!/usr/bin/env bash

set -Eeuo pipefail

MODE="${1:-check}"
PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
PYTHON_BIN="${POINTCEPT_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
EXP_ROOT="${POINTCEPT_EXP_ROOT:-$PROJECT/exp}"
FIXED_ROOT="${POINTCEPT_FIXED_ROOT:-$EXP_ROOT/fixed_protocol}"
MANIFEST="$PROJECT/configs/s3dis/eval/kpconvx_fixed_protocol_local_available.json"
RESULT_ROOT="$FIXED_ROOT/results/kpconvx-fixed-server-a"
SCALE04_ROOT="$FIXED_ROOT/results/scale04-baseline-all-checkpoints"
ANALYSIS_ROOT="$FIXED_ROOT/analysis/checkpoint-comparison"
RUN_ID="v17-dual-support_20260706_2303"
RUN_ROOT="$EXP_ROOT/s3dis/$RUN_ID"
WEIGHT="$RUN_ROOT/model/model_last.pth"
CONFIG="$RUN_ROOT/config.py"
DATA_AREA="/root/autodl-tmp/data/s3dis/Area_5"
OUTPUT_DIR="$RESULT_ROOT/screen/v17/$RUN_ID/last"
CACHE_DIR="$OUTPUT_DIR/result"
METRICS="$OUTPUT_DIR/metrics.json"
RUN_META="$OUTPUT_DIR/run_meta.json"
EXPECTED_META="$OUTPUT_DIR/expected_run_meta.json"
LOG_ROOT="$FIXED_ROOT/logs"
TELEMETRY_ROOT="$FIXED_ROOT/telemetry"
LOCK_FILE="$FIXED_ROOT/finish_v17_last.lock"
PID_FILE="$FIXED_ROOT/finish_v17_last.pid"

usage() {
  echo "Usage: $0 check|run"
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1" >&2
    exit 3
  fi
}

check_common() {
  require_file "$PYTHON_BIN"
  require_file "$MANIFEST"
  require_file "$CONFIG"
  require_file "$WEIGHT"
  require_file "$EXPECTED_META"
  if [ ! -d "$DATA_AREA" ]; then
    echo "Missing S3DIS Area_5 data: $DATA_AREA" >&2
    exit 3
  fi

  "$PYTHON_BIN" - "$PROJECT" "$EXP_ROOT" "$MANIFEST" "$EXPECTED_META" <<'PY'
import json
import sys
from pathlib import Path

project, exp_root, manifest_path, actual_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(project))

from tools.s3dis_fixed_protocol import (  # noqa: E402
    build_checkpoint_entries,
    expected_run_metadata,
    load_manifest,
    run_metadata_matches,
)

manifest = load_manifest(manifest_path)
entries = build_checkpoint_entries(manifest, exp_root)
entry = next(
    item
    for item in entries
    if item["run_id"] == "v17-dual-support_20260706_2303"
    and item["checkpoint_kind"] == "last"
)
expected = expected_run_metadata(
    entry,
    protocol="identity",
    point_max=60000,
    fragment_batch_size_test=4,
    grid_size=0.02,
)
actual = json.loads(actual_path.read_text(encoding="utf-8"))
if not run_metadata_matches(actual, expected):
    raise SystemExit("Existing v17 last cache metadata does not match the fixed protocol")
print("Protocol metadata: OK")
PY

  "$PYTHON_BIN" - "$DATA_AREA" "$CACHE_DIR" <<'PY'
import sys
from pathlib import Path

import numpy as np

area_dir = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
rooms = sorted(path for path in area_dir.iterdir() if path.is_dir())
cached = []
invalid = []
for room in rooms:
    pred_path = cache_dir / f"Area_5-{room.name}_pred.npy"
    if not pred_path.is_file():
        continue
    try:
        pred = np.load(pred_path, mmap_mode="r")
        segment = np.load(room / "segment.npy", mmap_mode="r")
        if pred.ndim != 1 or pred.shape[0] != segment.shape[0]:
            invalid.append(
                f"{pred_path.name}: pred={pred.shape}, segment={segment.shape}"
            )
        else:
            cached.append(pred_path.name)
    except (OSError, ValueError, EOFError) as error:
        invalid.append(f"{pred_path.name}: {error}")

if invalid:
    raise SystemExit("Invalid prediction cache:\n" + "\n".join(invalid))
print(f"Dataset rooms: {len(rooms)}")
print(f"Valid cached rooms: {len(cached)}")
print(f"Rooms remaining: {len(rooms) - len(cached)}")
PY

  echo "Weight: $WEIGHT"
  echo "Output: $OUTPUT_DIR"
  echo "Metrics complete: $([ -f "$METRICS" ] && echo yes || echo no)"
}

check_cuda() {
  "$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("CUDA is not available. Start GPU mode before running this task.")
    raise SystemExit(4)
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA devices: {torch.cuda.device_count()}")
PY
}

if [ "$MODE" != "check" ] && [ "$MODE" != "run" ]; then
  usage
  exit 2
fi

cd "$PROJECT"
check_common

if [ "$MODE" = "check" ]; then
  if ! check_cuda; then
    echo "CPU-side preparation is complete; GPU is the only missing prerequisite."
  fi
  exit 0
fi

check_cuda
mkdir -p "$LOG_ROOT" "$TELEMETRY_ROOT" "$ANALYSIS_ROOT"
exec 9> "$LOCK_FILE"
if ! flock -n 9; then
  echo "Another v17 last completion task holds $LOCK_FILE" >&2
  exit 5
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_ROOT/finish_v17_last_${TIMESTAMP}.log"
EXIT_FILE="${RUN_LOG%.log}.exit"
LATEST_LOG="$LOG_ROOT/latest_finish_v17_last.log"
TELEMETRY="$TELEMETRY_ROOT/finish_v17_last_${TIMESTAMP}_gpu.csv"

echo "$RUN_LOG"
ln -sfn "$RUN_LOG" "$LATEST_LOG"
printf '%s\n' "$$" > "$PID_FILE"
exec >> "$RUN_LOG" 2>&1

TELEMETRY_PID=""
cleanup() {
  EXIT_CODE=$?
  if [ -n "$TELEMETRY_PID" ]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "$EXIT_CODE" > "$EXIT_FILE"
  echo "$(date '+%F %T') finish_v17_last_exit=$EXIT_CODE"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "$(date '+%F %T') finish v17 fixed-protocol last evaluation"
echo "run_id=$RUN_ID"
echo "weight=$WEIGHT"
echo "output=$OUTPUT_DIR"
echo "point_max=60000"
echo "fragment_batch_size=4"
echo "num_workers=12"
echo "power_action=none"

bash scripts/record_gpu_telemetry.sh "$$" "$TELEMETRY" 10 &
TELEMETRY_PID=$!

set +e
POINTCEPT_FIXED_MANIFEST="$MANIFEST" \
POINTCEPT_FIXED_RESULT_ROOT="$RESULT_ROOT" \
  bash scripts/run_fixed_eval_safe.sh \
    server-a screen 60000 4 12 last "$RUN_ID"
TASK_EXIT=$?
set -e
if [ "$TASK_EXIT" -ne 0 ]; then
  echo "v17 last evaluation failed with exit code $TASK_EXIT" >&2
  exit "$TASK_EXIT"
fi

# Restore full local inventory and summaries after the single-checkpoint run.
"$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
  --stage discover \
  --manifest "$MANIFEST" \
  --exp-root "$EXP_ROOT" \
  --output-root "$RESULT_ROOT" \
  --point-max 60000 \
  --fragment-batch-size-test 4 \
  --num-worker-test 12 \
  --checkpoint-kinds best last

"$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
  --stage summarize \
  --manifest "$MANIFEST" \
  --output-root "$RESULT_ROOT" \
  --checkpoint-kinds best last

"$PYTHON_BIN" -u tools/eval_s3dis_fixed_protocol.py \
  --stage bundle \
  --manifest "$MANIFEST" \
  --output-root "$RESULT_ROOT" \
  --bundle-path "$FIXED_ROOT/bundles/kpconvx-fixed-server-a-compact.zip"

"$PYTHON_BIN" -u tools/analyze_s3dis_checkpoint_results.py \
  --fixed-root "$RESULT_ROOT" \
  --scale04-root "$SCALE04_ROOT" \
  --output-dir "$ANALYSIS_ROOT"

"$PYTHON_BIN" - "$METRICS" "$RUN_META" "$RESULT_ROOT/screen/completeness.json" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, run_meta_path, completeness_path = map(Path, sys.argv[1:])
for path in (metrics_path, run_meta_path, completeness_path):
    if not path.is_file():
        raise SystemExit(f"Missing completion artifact: {path}")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
if not completeness.get("complete"):
    raise SystemExit(f"Final result set is incomplete: {completeness}")
print(f"v17 last mIoU={metrics['mIoU']:.6f}")
print(
    f"Final completeness={completeness['completed']}/"
    f"{completeness['expected']}"
)
PY

echo "$(date '+%F %T') v17 last evaluation and final analysis complete"
echo "metrics=$METRICS"
echo "summary=$RESULT_ROOT/screen/checkpoint_summary.csv"
echo "analysis=$ANALYSIS_ROOT/checkpoint_analysis.md"
echo "bundle=$FIXED_ROOT/bundles/kpconvx-fixed-server-a-compact.zip"
