# AutoDL 意外关机事故复盘

## 1. 事件概述

- 发生时间：2026-07-13 13:32:26（UTC+8）
- 受影响任务：S3DIS 固定协议评估，`v17-dual-support` 的
  `model_last.pth`
- 中断位置：已完成 46/68 个房间，在第 47 个房间
  `Area_5-office_32` 开始推理后中断
- 结果影响：部分房间预测已保存，但没有生成最终的 `metrics.json` 和
  `run_meta.json`，因此该次评估不能视为完成

## 2. 直接原因

此前的诊断操作为了确认关机命令是否可用，执行了：

```bash
/usr/bin/shutdown --help
```

这是本次事故的直接触发命令。

AutoDL 容器中的 `/usr/bin/shutdown` 不是标准 Linux/systemd 的
`shutdown`。它是 AutoDL 提供的简短包装脚本，不解析参数，也不提供
正常的 `--help`、`--show` 或 `-c` 语义。无论传入什么参数，它都会杀死
容器内的 `supervisord`，进而关闭实例。

事故发生时，该脚本的核心行为相当于：

```bash
ps -ef | grep supervisord | grep -v grep | awk '{print $2}' | xargs kill
```

因此，在 AutoDL 上执行 `shutdown --help` 并不是查看帮助，而是在执行
真实关机。

## 3. 证据链

### 3.1 评估程序没有报错

`v17 last` 的日志在正常推理过程中突然结束：

- 13:32:23 完成第 46/68 个房间
- 13:32:24 开始第 47/68 个房间的第一个 fragment batch
- 日志中没有 Python traceback
- 没有 CUDA OOM
- 没有磁盘写入错误
- 没有生成启动器 `.exit` 文件

这说明程序不是自行失败退出，而是运行环境被外部终止。

### 3.2 时间完全吻合

- `shutdown --help` 的执行时间：13:32:26.732
- `/usr/bin/shutdown` 的文件访问时间：13:32:26.789
- 评估日志和 GPU 遥测在同一时刻停止

三个时间点一致，能够确认直接因果关系。

### 3.3 不是 watcher 正常触发

关机前的进程快照显示以下进程仍在运行：

- 主评估启动器
- “成功才关机”的 watcher
- “无条件关机”的 watcher

两个 watcher 的日志都没有出现：

```text
watched_process_stopped
requesting_shutdown
```

因此，本次关机不是 watcher 检测到任务结束后触发，而是诊断命令直接
调用 `/usr/bin/shutdown` 触发。

## 4. 次要设计问题

虽然 watcher 不是本次关机的直接触发者，但工作区当时存在多个危险的
自动关机入口：

- 同时启动两个 watcher 监控同一个 PID
- watcher 脱离终端并由 PID 1 接管
- 其中一个 watcher 在任务失败、被杀死或没有退出码时也无条件关机
- 工作区已有“禁止自动关机”标记，但 watcher 并不读取该标记
- 原有测试只识别 `shutdown` 和 `/sbin/shutdown`，漏掉了实际使用的
  `/usr/bin/shutdown`
- 原有测试没有扫描仓库根目录的 shell 脚本

这些问题会让异常退出进一步演变成实例关机，也增加了排查难度。

## 5. 已实施的解决方式

### 5.1 禁用系统关机命令

当前实例的 `/usr/bin/shutdown` 已替换为安全桩：

```bash
#!/usr/bin/env bash

echo "AutoDL shutdown command is disabled on this instance; use the AutoDL console." >&2
exit 126
```

现在即使误调用该路径，也只会返回错误，不会关闭实例。

注意：如果 AutoDL 重建了容器或更换了实例，系统文件可能恢复为平台原始
版本。新实例上必须重新以只读方式检查，不能通过执行 `--help` 来检查。

### 5.2 删除自动关机 watcher

已删除活动工作区中的以下入口：

- `scripts/watch_process_then_always_shutdown.sh`
- `scripts/watch_process_then_shutdown.sh`
- `scripts/watch_scale04_screen_then_shutdown.sh`
- `watch_train_then_shutdown.sh`
- `.ipynb_checkpoints/watch_train_then_shutdown-checkpoint.sh`

同时清除了硬关机留下的过期 PID 文件。

### 5.3 增强自动测试

`tests/test_no_auto_poweroff.py` 已增强：

- 扫描仓库根目录、`scripts/` 和 notebook checkpoint 中的 shell 入口
- 识别 `shutdown`、`poweroff` 和 `halt`
- 识别 `/usr/bin/shutdown`、`/sbin/shutdown` 等绝对路径
- 识别分号、`&&`、`||` 和管道后的电源命令
- 增加明确的危险命令回归样例

修复后测试结果为 2/2 通过。

### 5.4 保留任务状态，不操作实例电源

训练和评估脚本只负责：

- 写入运行日志
- 定期写入心跳
- 保存 PID
- 正常退出时写入退出码
- 保存指标和预测结果

无论任务成功、失败还是被中断，都不再从任务脚本执行系统电源操作。

## 6. 后续强制规则

1. 不在 AutoDL 容器中执行 `shutdown` 的任何变体，包括 `--help`、
   `--show` 和 `-c`。
2. 不通过执行命令来探测关机命令是否安全。
3. 不创建自动关机 watcher，不把关机逻辑放进 `trap EXIT`。
4. 任务失败时保留现场、写退出码并告警，不自动关闭实例。
5. 需要停止实例时，只通过 AutoDL 控制台，由用户明确操作。
6. 新增或修改 shell 入口后，必须运行无自动关机测试。
7. 新实例启动后，应重新确认 `/usr/bin/shutdown` 的内容，因为系统文件
   可能随容器重建而恢复。

## 7. 安全检查方法

检查 AutoDL 的关机文件时只能使用只读文件工具，例如：

```bash
ls -l /usr/bin/shutdown
file /usr/bin/shutdown
stat /usr/bin/shutdown
sed -n '1,80p' /usr/bin/shutdown
```

禁止使用以下方式检查：

```bash
shutdown --help
shutdown --show
shutdown -c
/usr/bin/shutdown --help
```

检查仓库中是否存在活动电源命令，应运行：

```bash
python -m unittest tests.test_no_auto_poweroff
```

## 8. 最终结论

本次事故不是模型、评估代码、CUDA 或磁盘故障。直接原因是把 AutoDL 的
自定义 `/usr/bin/shutdown` 错误地当作标准 Linux 命令，并执行了
`--help` 参数。该包装器忽略参数并直接关闭容器。

核心经验是：云平台镜像中的系统命令可能被替换，不能假设其参数语义与
标准发行版一致。涉及关机、重启、删除和资源释放的命令，必须先读取文件
内容或查阅平台文档，不能通过试运行来验证。
