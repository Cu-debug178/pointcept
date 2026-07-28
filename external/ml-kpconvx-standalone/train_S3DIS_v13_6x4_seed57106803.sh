#!/usr/bin/env bash
set -euo pipefail

STANDALONE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STANDALONE_ROOT
CONFIG_FILE="${CONFIG_FILE:-$STANDALONE_ROOT/configs/s3dis_v13_6x4_seed57106803.conf}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing launch configuration: $CONFIG_FILE" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable is not available: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN=/path/to/python before launching." >&2
    exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "S3DIS data directory is not available: $DATA_DIR" >&2
    echo "Set DATA_DIR=/path/to/s3dis before launching." >&2
    exit 2
fi

# Never silently overwrite a previous run. Set ALLOW_EXISTING_LOG=1 only when
# deliberately reusing a directory for a controlled continuation/debug run.
if [[ -e "$LOG_PATH/parameters.json" && "${ALLOW_EXISTING_LOG:-0}" != "1" ]]; then
    echo "Log directory already contains parameters.json: $LOG_PATH" >&2
    echo "Choose another LOG_PATH or set ALLOW_EXISTING_LOG=1 deliberately." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="$STANDALONE_ROOT/KPConvX${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$BUILD_EXTENSIONS" == "1" ]]; then
    if ! "$PYTHON_BIN" -c \
        'import cpp_wrappers.cpp_subsampling.cpp_subsampling, cpp_wrappers.cpp_neighbors.cpp_neighbors'; then
        echo "Building Standalone C++ extensions..."
        (cd "$STANDALONE_ROOT/KPConvX/cpp_wrappers/cpp_subsampling" && \
            "$PYTHON_BIN" setup.py build_ext --inplace)
        (cd "$STANDALONE_ROOT/KPConvX/cpp_wrappers/cpp_neighbors" && \
            "$PYTHON_BIN" setup.py build_ext --inplace)
    fi
fi

ARGS=(
    --dataset_path "$DATA_DIR"
    --log_path "$LOG_PATH"
    --seed "$SEED"
    --batch_size "$BATCH_SIZE"
    --accum_batch "$ACCUM_BATCH"
    --num_workers "$NUM_WORKERS"
    --steps_per_epoch "$STEPS_PER_EPOCH"
    --max_epoch "$MAX_EPOCH"
    --enable_da_radius 1
    --da_radius_profile "$DA_RADIUS_PROFILE"
    --da_radius_debug "$DA_RADIUS_DEBUG"
)

echo "Launching Standalone KPConvX V13"
echo "  config: $CONFIG_FILE"
echo "  data:   $DATA_DIR"
echo "  logs:   $LOG_PATH"
echo "  seed:   $SEED"
echo "  batch:  ${BATCH_SIZE} x ${ACCUM_BATCH} (effective ${BATCH_SIZE}*${ACCUM_BATCH})"

exec "$PYTHON_BIN" \
    "$STANDALONE_ROOT/KPConvX/experiments/S3DIS/train_S3DIS.py" \
    "${ARGS[@]}"
