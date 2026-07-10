# KPConvX 固定协议双服务器运行手册

## 安全约束

- 两台 AutoDL 服务器的数据盘和进程完全独立，Git 只同步代码。
- 当前复评入口不会执行任何关机、重启或系统电源操作。
- 不再使用历史入口 `run_fixed_screen_then_shutdown.sh`。
- 日志、心跳、PID 和退出码统一保存在 `exp/fixed_protocol/logs/`。
- 启动器直接调用 Pointcept 环境中的 Python，不再 source Conda 激活脚本。

## 1. 同步与取证

两台服务器分别执行：

```bash
cd /root/autodl-tmp/Pointcept
git fetch origin
git pull --ff-only origin codex/v16-boundary-aware-dual-framework

bash scripts/collect_instance_evidence.sh
```

检查当前服务器实际持有的权重：

```bash
find /root/autodl-tmp/Pointcept/exp -type f \
  \( -name 'model_best.pth' -o -name 'model_last.pth' \) | sort
```

服务器 A 和 B 不要求权重重复，最终在本地合并指标。

## 2. 服务器 A 显存预检

只在包含 v17 best/last 的服务器执行：

```bash
cd /root/autodl-tmp/Pointcept
mkdir -p exp/fixed_protocol/logs

nohup bash scripts/run_fixed_eval_safe.sh \
  server-a preflight 60000 4 6 \
  > exp/fixed_protocol/logs/bootstrap_preflight_server-a.log 2>&1 < /dev/null &
```

查看启动器管理的主日志和心跳：

```bash
tail -f exp/fixed_protocol/logs/latest_preflight_server-a.log
ls -lt exp/fixed_protocol/logs/*.heartbeat | head
```

预检结果保存在：

```text
exp/fixed_protocol/results/kpconvx-fixed-server-a/point_max_decision.json
```

## 3. 两台服务器并行 screen

先从预检 JSON 读取统一的 `selected_point_max` 和
`selected_fragment_batch_size`。以下假设选中 `60000 / 4`。

服务器 A：

```bash
nohup bash scripts/run_fixed_eval_safe.sh \
  server-a screen 60000 4 6 \
  > exp/fixed_protocol/logs/bootstrap_screen_server-a.log 2>&1 < /dev/null &
```

服务器 B：

```bash
nohup bash scripts/run_fixed_eval_safe.sh \
  server-b screen 60000 4 6 \
  > exp/fixed_protocol/logs/bootstrap_screen_server-b.log 2>&1 < /dev/null &
```

查看运行状态：

```bash
tail -f exp/fixed_protocol/logs/latest_screen_server-a.log
cat exp/fixed_protocol/fixed_screen_server-a.pid
```

正常退出后会生成 `.exit` 文件，内容 `0` 表示成功。若实例被平台硬停，心跳会中断且通常不会生成 `.exit` 文件，可与脚本正常失败区分。

## 4. 下载结果

安全启动器在 screen 成功后自动生成：

```text
exp/fixed_protocol/bundles/kpconvx-fixed-server-a-compact.zip
exp/fixed_protocol/bundles/kpconvx-fixed-server-b-compact.zip
```

压缩包只包含日志、JSON 和 CSV，不包含大体积预测缓存。

## 5. 本地合并

```powershell
Expand-Archive server-a-compact.zip exp/fixed-server-a -Force
Expand-Archive server-b-compact.zip exp/fixed-server-b -Force

python tools/eval_s3dis_fixed_protocol.py `
  --stage merge `
  --manifest configs/s3dis/eval/kpconvx_fixed_protocol_v1.json `
  --output-root exp/fixed-protocol-merged `
  --merge-input exp/fixed-server-a exp/fixed-server-b
```

完整 14 个 checkpoint 合并后，再根据 `tta_selection.json` 决定是否运行 TTA13。
