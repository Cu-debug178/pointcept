#!/usr/bin/env bash

# Route-2 launcher for one RTX 4090D. This script never powers off the host.
# Usage:
#   bash scripts/train_route2_scale04.sh baseline
#   bash scripts/train_route2_scale04.sh v17

set -eo pipefail

VARIANT="${1:-baseline}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/exp/route2-scale04/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "${VARIANT}" in
    baseline)
        CONFIG="semseg-kpconvx-base-s3dis-scale04-4090d-area5"
        EXP_NAME="kpconvx-scale04-baseline"
        ;;
    v17)
        CONFIG="semseg-kpconvx-hybrid-v17-scale04-4090d-area5"
        EXP_NAME="kpconvx-v17-scale04"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}. Use baseline or v17." >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${VARIANT}_${TIMESTAMP}.log"
ln -sfn "$(basename "${LOG_PATH}")" "${LOG_DIR}/latest_${VARIANT}.log"
echo "Route-2 log: ${LOG_PATH}"
exec >>"${LOG_PATH}" 2>&1

echo "$(date '+%F %T') route2-scale04 start"
echo "variant=${VARIANT}"
echo "config=${CONFIG}"
echo "project=${PROJECT_DIR}"

cd "${PROJECT_DIR}"

if [[ -f "${PROJECT_DIR}/activate_env.sh" ]]; then
    # Do not enable nounset before sourcing Conda activation scripts.
    source "${PROJECT_DIR}/activate_env.sh"
fi

AVAILABLE_KB="$(df -Pk "${PROJECT_DIR}/exp" | awk 'NR==2 {print $4}')"
REQUIRED_KB=$((4 * 1024 * 1024))
echo "available_disk_kb=${AVAILABLE_KB}"
if [[ -z "${AVAILABLE_KB}" || "${AVAILABLE_KB}" -lt "${REQUIRED_KB}" ]]; then
    echo "Insufficient disk space: at least 4 GiB is required." >&2
    exit 3
fi

CONFIG_PATH="configs/s3dis/${CONFIG}.py"
python -m py_compile "${CONFIG_PATH}"
python - <<PY
from pointcept.models import build_model
from pointcept.utils.config import Config

cfg = Config.fromfile("${CONFIG_PATH}")
backbone_cfg = cfg.model.backbone
assert float(backbone_cfg.subsample_size) == 0.04
assert float(backbone_cfg.kp_radius) == 2.1
assert float(cfg.data.test.test_cfg.voxelize.grid_size) == 0.04
saver = next(hook for hook in cfg.hooks if hook.type == "CheckpointSaver")
assert int(saver.save_freq) == 20
model = build_model(cfg.model)
params = sum(parameter.numel() for parameter in model.parameters())
print(f"config smoke passed: type={backbone_cfg.type}, params={params}")
del model
PY

echo "No shutdown command is configured."
set +e
sh scripts/train_autodl.sh \
    -p python \
    -g 1 \
    -d s3dis \
    -c "${CONFIG}" \
    -n "${EXP_NAME}"
STATUS=$?
set -e

echo "$(date '+%F %T') route2-scale04 exit=${STATUS}"
echo "The server will remain running."
exit "${STATUS}"
