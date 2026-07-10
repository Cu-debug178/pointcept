# KPConvX 固定协议双服务器运行手册

## 1. 两台服务器检查本机 checkpoint

两台服务器分别使用不同输出目录：

```bash
cd /root/autodl-tmp/Pointcept
source /root/autodl-tmp/Pointcept/activate_env.sh

python tools/eval_s3dis_fixed_protocol.py \
  --stage discover \
  --exp-root exp \
  --output-root exp/fixed_protocol/results/kpconvx-fixed-server-a
```

另一台把输出名改成 `kpconvx-fixed-server-b`。工具会分别生成：

- `available_entries.json`
- `missing_entries.json`
- `protocol.json`

缺少的 checkpoint 不会报错，由另一台服务器补齐。

## 2. v17 所在服务器做显存预检

只在包含 v17 best/last 的服务器运行：

如果上一轮包装任务失败后仍可能处于 300 秒关机倒计时，先执行：

```bash
bash scripts/cancel_fixed_shutdown.sh server-b
```

该脚本会同时写入新旧版本的取消标记、执行 `shutdown -c`，并停止旧的 `watch_train_then_shutdown.sh`。仅执行 `shutdown -c` 不能取消旧包装脚本内部的 `sleep 300; shutdown -h now`。

```bash
python tools/eval_s3dis_fixed_protocol.py \
  --stage preflight \
  --exp-root exp \
  --output-root exp/fixed_protocol/results/kpconvx-fixed-server-b \
  --point-max 60000 \
  --fallback-point-max 40000 \
  --fragment-batch-size-test 4 \
  --fallback-fragment-batch-sizes 2 1 \
  --num-worker-test 6
```

读取：

```bash
cat exp/fixed_protocol/results/kpconvx-fixed-server-b/point_max_decision.json
```

后续两台服务器必须使用相同的 `selected_point_max` 和 `selected_fragment_batch_size`。预检优先保持 `point_max=60000`，fragment batch 按 `4 -> 2 -> 1` 降级；只有同一 point limit 的批量全部 OOM 才尝试 `40000`。

## 3. 两台服务器并行跑 screen

假设预检选择 `60000`：

```bash
mkdir -p exp/fixed_protocol/logs
LOG=exp/fixed_protocol/logs/fixed_screen_server_a_$(date +%Y%m%d_%H%M).log
nohup bash -lc '
source /root/autodl-tmp/Pointcept/activate_env.sh
cd /root/autodl-tmp/Pointcept
python tools/eval_s3dis_fixed_protocol.py \
  --stage screen \
  --exp-root exp \
  --output-root exp/fixed_protocol/results/kpconvx-fixed-server-a \
  --point-max 60000 \
  --fragment-batch-size-test 4 \
  --num-worker-test 6
' > "$LOG" 2>&1 &
tail -f "$LOG"
```

服务器 B 使用 `kpconvx-fixed-server-b`。已有完整结果会跳过；失败后的同协议结果可以继续。改变协议或 point limit 时必须显式加 `--overwrite`，防止复用旧预测缓存。

使用仓库内的无关机包装脚本运行 screen。脚本只执行评估、保存日志和打包结果，成功或失败后服务器都保持运行。启动时会取消旧版本遗留的预约关机，并通过 `flock` 禁止同一 `server-id` 重复运行。

```bash
chmod +x scripts/run_fixed_screen.sh
mkdir -p exp/fixed_protocol/logs
nohup bash scripts/run_fixed_screen.sh server-a 60000 4 6 \
  > exp/fixed_protocol/logs/fixed_screen_server-a_launcher.log 2>&1 &
```

服务器 B 使用 `server-b`。正式启动前必须根据 v17 预检结果统一设置两台机器的 `POINT_MAX` 和 fragment batch。

## 4. 生成小体积结果包

两台服务器分别执行：

```bash
python tools/eval_s3dis_fixed_protocol.py \
  --stage bundle \
  --output-root exp/fixed_protocol/results/kpconvx-fixed-server-a \
  --bundle-path exp/fixed_protocol/bundles/kpconvx-fixed-server-a-compact.zip
```

生成：

```text
exp/fixed_protocol/bundles/kpconvx-fixed-server-a-compact.zip
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
  --output-root exp/fixed_protocol/results/kpconvx-fixed-server-a \
  --selected-family v16b \
  --point-max 60000 \
  --fragment-batch-size-test 4 \
  --num-worker-test 6
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
  --output-root exp/fixed_protocol/audit/v17-neighbor-compatibility-audit \
  --point-max 60000 \
  --num-worker-test 6
```

先做两房间 smoke：

```bash
python tools/audit_v17_neighbor_compatibility.py \
  --exp-root exp \
  --output-root exp/fixed_protocol/audit/v17-neighbor-compatibility-smoke \
  --point-max 60000 \
  --num-worker-test 6 \
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
