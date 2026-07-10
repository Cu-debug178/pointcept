# KPConvX 固定协议双服务器运行手册

## 1. 两台服务器检查本机 checkpoint

两台服务器分别使用不同输出目录：

```bash
cd /root/autodl-tmp/Pointcept
source /root/autodl-tmp/Pointcept/activate_env.sh

python tools/eval_s3dis_fixed_protocol.py \
  --stage discover \
  --exp-root exp \
  --output-root exp/s3dis/kpconvx-fixed-server-a
```

另一台把输出名改成 `kpconvx-fixed-server-b`。工具会分别生成：

- `available_entries.json`
- `missing_entries.json`
- `protocol.json`

缺少的 checkpoint 不会报错，由另一台服务器补齐。

## 2. v17 所在服务器做显存预检

只在包含 v17 best/last 的服务器运行：

```bash
python tools/eval_s3dis_fixed_protocol.py \
  --stage preflight \
  --exp-root exp \
  --output-root exp/s3dis/kpconvx-fixed-server-b \
  --point-max 60000 \
  --fallback-point-max 40000
```

读取：

```bash
cat exp/s3dis/kpconvx-fixed-server-b/point_max_decision.json
```

后续两台服务器必须使用相同的 `point_max`。

## 3. 两台服务器并行跑 screen

假设预检选择 `60000`：

```bash
LOG=exp/fixed_screen_server_a_$(date +%Y%m%d_%H%M).log
nohup bash -lc '
source /root/autodl-tmp/Pointcept/activate_env.sh
cd /root/autodl-tmp/Pointcept
python tools/eval_s3dis_fixed_protocol.py \
  --stage screen \
  --exp-root exp \
  --output-root exp/s3dis/kpconvx-fixed-server-a \
  --point-max 60000 \
  --num-worker-test 2
' > "$LOG" 2>&1 &
tail -f "$LOG"
```

服务器 B 使用 `kpconvx-fixed-server-b`。已有完整结果会跳过；失败后的同协议结果可以继续。改变协议或 point limit 时必须显式加 `--overwrite`，防止复用旧预测缓存。

如需任务无论成功或失败都在 5 分钟后关机，使用外层包装脚本，不要在激活环境前启用 `set -u`：

```bash
cat > run_fixed_screen_then_shutdown.sh <<'EOF'
#!/usr/bin/env bash
set -o pipefail

SERVER_ID="server-a"
POINT_MAX=60000
PROJECT=/root/autodl-tmp/Pointcept
RUN_LOG="$PROJECT/exp/fixed_screen_${SERVER_ID}_$(date +%Y%m%d_%H%M%S).log"
STATUS_LOG="$PROJECT/exp/fixed_screen_${SERVER_ID}_shutdown_status.log"

mkdir -p "$PROJECT/exp"
cd "$PROJECT" || exit 2

{
  echo "$(date '+%F %T') 开始固定协议 screen: ${SERVER_ID}"
  source "$PROJECT/activate_env.sh"
  python tools/eval_s3dis_fixed_protocol.py \
    --stage screen \
    --exp-root exp \
    --output-root "exp/s3dis/kpconvx-fixed-${SERVER_ID}" \
    --point-max "$POINT_MAX" \
    --num-worker-test 2
} > "$RUN_LOG" 2>&1
EXIT_CODE=$?

{
  echo "$(date '+%F %T') screen 结束，退出码: ${EXIT_CODE}"
  echo "运行日志: ${RUN_LOG}"
  echo "无论成功或失败，5分钟后关机；取消命令: shutdown -c"
} | tee -a "$STATUS_LOG"

sync
shutdown -h +5
exit "$EXIT_CODE"
EOF

chmod +x run_fixed_screen_then_shutdown.sh
nohup bash run_fixed_screen_then_shutdown.sh \
  > exp/fixed_screen_server-a_launcher.log 2>&1 &
```

服务器 B 只需把脚本中的 `SERVER_ID="server-a"` 改成 `SERVER_ID="server-b"`。正式启动前必须根据 v17 预检结果统一设置两台机器的 `POINT_MAX`。

## 4. 生成小体积结果包

两台服务器分别执行：

```bash
python tools/eval_s3dis_fixed_protocol.py \
  --stage bundle \
  --output-root exp/s3dis/kpconvx-fixed-server-a
```

生成：

```text
exp/s3dis/kpconvx-fixed-server-a-compact.zip
```

zip 不包含大体积房间预测，只包含日志、JSON 和 CSV。

## 5. 本地电脑合并 screen

```powershell
Expand-Archive server-a-compact.zip exp/fixed-server-a -Force
Expand-Archive server-b-compact.zip exp/fixed-server-b -Force

python tools/eval_s3dis_fixed_protocol.py `
  --stage merge `
  --manifest configs/s3dis/eval/kpconvx_fixed_protocol_v1.json `
  --output-root exp/fixed-protocol-merged `
  --merge-input exp/fixed-server-a exp/fixed-server-b
```

完整 14 checkpoint 合并后会生成：

```text
exp/fixed-protocol-merged/tta_selection.json
```

如果结果缺失，`screen/completeness.json` 会列出缺少的 checkpoint，不会提前生成有效 TTA 选择。

## 6. 两台服务器并行跑入围模型族 TTA13

把本地生成的 `tta_selection.json` 上传到两台服务器，读取其中的 `selected_family`，然后两台执行：

```bash
python tools/eval_s3dis_fixed_protocol.py \
  --stage tta13 \
  --exp-root exp \
  --output-root exp/s3dis/kpconvx-fixed-server-a \
  --selected-family v16b \
  --point-max 60000 \
  --num-worker-test 2
```

示例中的 `v16b` 必须替换为实际选择结果。每台只运行本机存在的 baseline/入围族 checkpoint。完成后重新 bundle，并在本地对同一 `fixed-protocol-merged` 目录再次执行 merge。

TTA13 完整后生成：

- `decision_report.md`
- `decision.json`
- `tta13/checkpoint_summary.csv`
- `tta13/class_metrics.csv`
- `tta13/room_metrics.csv`

## 7. v17 邻域污染审计

只在包含 v17 best/last 的服务器运行：

```bash
python tools/audit_v17_neighbor_compatibility.py \
  --exp-root exp \
  --output-root exp/s3dis/v17-neighbor-compatibility-audit \
  --point-max 60000 \
  --num-worker-test 2
```

先做两房间 smoke：

```bash
python tools/audit_v17_neighbor_compatibility.py \
  --exp-root exp \
  --output-root exp/s3dis/v17-neighbor-compatibility-smoke \
  --point-max 60000 \
  --num-worker-test 2 \
  --max-rooms 2 \
  --bootstrap 20
```

正式审计以 `audit_report.md` 和 `audit_summary.json` 的 GO/NO-GO 为准。

## 本地电脑能做什么

Windows 本地可以执行：

- compact zip 解压与合并；
- CSV、排名和决策报告生成；
- `python -m unittest tests.test_s3dis_fixed_protocol -v`。

本地当前缺少完整 Pointcept CUDA 扩展，不用于 KPConvX full-scene forward 或邻域审计。
