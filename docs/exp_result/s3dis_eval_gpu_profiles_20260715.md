# S3DIS 训练并行与训练后独占评估配置

## 1. 文档目的

本文记录同一张 RTX 4090D 上的两套 S3DIS 全场景评估配置：

1. 训练运行期间，利用训练空闲间隔进行低显存并行评估。
2. 训练完全退出后，独占 GPU 进行高显存、高吞吐评估。

对应实验：

```text
kpconvx-v17-scale04_20260714_1553
```

两套配置使用相同的固定评估协议。资源参数不同，但不丢弃分片、不省略
房间，也不改变最终的点级合并逻辑。

## 2. 配置对比

| 项目 | 训练并行低显存 | 训练后独占高显存 |
|---|---:|---:|
| CUDA 显存比例上限 | 0.17 | 0.80 |
| fragment batch | 1 | 4 |
| 测试 workers | 2 | 6 |
| point max | 60000 | 60000 |
| grid size | 0.04 m | 0.04 m |
| 测试协议 | identity | identity |
| 最低空闲显存保护 | 700 MiB | 1500 MiB |
| CPU 优先级 | nice 10 | 默认 |
| I/O 优先级 | idle | 默认 |
| 适用阶段 | 训练进程存活时 | 训练进程完全退出后 |

固定不变的核心参数：

```text
grid_size=0.04
point_max=60000
protocol=identity
save_predictions=false
```

## 3. 训练期间的低显存评估

### 3.1 文件位置

- 启动脚本：`scripts/run_concurrent_lowmem_eval_sweep.sh`
- 权重清单：`configs/s3dis/eval/v17_scale04_epoch_sweep.json`
- 结果目录：
  `exp/fixed_protocol/results/v17-scale04-concurrent-screening`
- 最新日志：
  `exp/fixed_protocol/logs/latest_v17_scale04_concurrent_sweep.log`
- 最新状态：
  `exp/fixed_protocol/logs/latest_v17_scale04_concurrent_sweep.state`

### 3.2 启动命令

先确认训练 PID，再启动：

```bash
TRAIN_PID=2211 bash scripts/run_concurrent_lowmem_eval_sweep.sh
```

`2211` 只是本次实验的 PID。新实验必须重新查询，不能照抄旧 PID。

### 3.3 核心资源配置

```bash
MEMORY_FRACTION=0.17
MIN_FREE_MIB=700
FRAGMENT_BATCH_SIZE=1
NUM_WORKERS=2
OMP_NUM_THREADS=1
```

评估进程使用较低的 CPU 和磁盘优先级：

```bash
nice -n 10 ionice -c 3
```

这能减少评估的数据准备阶段对训练 dataloader 和 checkpoint 写入的影响。

### 3.4 权重选择规则

训练期间只评估已经固定落盘的 epoch 权重，例如：

```text
epoch_120.pth
epoch_140.pth
epoch_160.pth
```

训练期间不要评估以下动态文件：

```text
model_best.pth
model_last.pth
```

它们可能在评估进程读取元数据和加载权重之间被训练进程覆盖。必须等训练
完全结束后再测。

### 3.5 训练保护逻辑

脚本只对训练进程执行以下只读检查：

```bash
kill -0 "$TRAIN_PID"
/proc/$TRAIN_PID/stat
/proc/$TRAIN_PID/cmdline
```

保护规则：

1. 校验 PID 对应的是指定实验的 `tools/train.py`。
2. 保存进程启动时间，防止 PID 被系统复用。
3. 训练 PID 消失时，只停止评估进程组。
4. 空闲显存低于 700 MiB 时，只停止评估进程组。
5. 绝不向训练 PID 或其子进程发送 TERM、KILL、STOP 等信号。
6. 不包含任何自动关机、重启或 AutoDL 电源操作。

### 3.6 实测资源表现

本次实验观测值：

- 训练显存约 18.6 GiB。
- 评估显存约 3.9 GiB。
- 总显存峰值约 22.53 GiB。
- 最低空闲显存约 1.55 GiB。
- 推理阶段 GPU 利用率可达到 100%。
- 一次完整并行评估期间平均 GPU 利用率约 74.3%。

代价是训练吞吐下降：普通训练批次通常由约 0.35--0.40 秒放慢到
0.5--0.7 秒，数据准备重叠时可能出现数秒尖峰。因此该模式是用更长的
训练墙钟时间换取并行完成固定协议评估，不是零成本计算。

## 4. 训练结束后的高显存评估

### 4.1 文件位置

- 启动脚本：`scripts/run_posttrain_highmem_eval_sweep.sh`
- 完整权重清单：
  `configs/s3dis/eval/v17_scale04_all_saved_checkpoints.json`
- 高显存预检清单：
  `configs/s3dis/eval/v17_scale04_epoch200_highmem_preflight.json`
- 结果目录：
  `exp/fixed_protocol/results/v17-scale04-concurrent-screening`
- 最新日志：
  `exp/fixed_protocol/logs/latest_v17_scale04_posttrain_highmem.log`
- 最新状态：
  `exp/fixed_protocol/logs/latest_v17_scale04_posttrain_highmem.state`

### 4.2 启动前提

启动前必须同时满足：

1. 指定实验的 `tools/train.py` 已完全退出。
2. `nvidia-smi` 中没有残留训练 CUDA 进程。
3. GPU 空闲显存不少于 12000 MiB。
4. 上一个低显存权重已经生成 `metrics.json` 和 `run_meta.json`。
5. 旧评估进程已经退出，不与高显存评估并行。

高显存脚本会主动检查训练进程和空闲显存。条件不满足时直接拒绝启动。

### 4.3 高显存预检

正式运行前，用最大房间之一验证 batch 4：

```bash
/root/autodl-tmp/envs/pointcept/bin/python -u \
  tools/eval_s3dis_fixed_protocol.py \
  --stage preflight \
  --manifest configs/s3dis/eval/v17_scale04_epoch200_highmem_preflight.json \
  --exp-root /root/autodl-tmp/Pointcept/exp \
  --output-root /root/autodl-tmp/Pointcept/exp/fixed_protocol/results/v17-scale04-posttrain-highmem-preflight \
  --grid-size 0.04 \
  --point-max 60000 \
  --fallback-point-max 60000 \
  --fragment-batch-size-test 4 \
  --fallback-fragment-batch-sizes 2 1 \
  --num-worker-test 6 \
  --preflight-room Area_5-hallway_2 \
  --checkpoint-kinds epoch_200 \
  --run-ids kpconvx-v17-scale04_20260714_1553 \
  --cuda-memory-fraction 0.80
```

回退顺序为 batch 4、2、1。只有 batch 4 预检成功后，正式脚本才使用
batch 4。

### 4.4 正式启动命令

```bash
bash scripts/run_posttrain_highmem_eval_sweep.sh
```

可通过环境变量覆盖资源配置：

```bash
MEMORY_FRACTION=0.80 \
FRAGMENT_BATCH_SIZE=4 \
NUM_WORKERS=6 \
bash scripts/run_posttrain_highmem_eval_sweep.sh
```

### 4.5 实测资源表现

`Area_5-hallway_2` 高显存预检结果：

- 360 个分片由 360 个 batch 降为 90 个 batch。
- 评估显存峰值约 10.84 GiB。
- 仍有约 13.23 GiB 空闲显存。
- 推理阶段 GPU 利用率达到 100%。
- 峰值功耗约 250 W。
- batch 4 预检成功，无 CUDA OOM。

显存上限设置为 80% 并不意味着一定分配 80% 显存。它只是 PyTorch
allocator 的最大边界；实际显存由当前房间和 fragment batch 决定。

## 5. 为什么 batch 4 更充分利用 GPU

提高显存占用本身不会自动提高 GPU 利用率。真正起作用的是利用额外显存
并行处理更多分片：

```text
训练并行：每次 1 个 fragment
训练结束：每次最多 4 个 fragment
```

这会减少：

- Python 调度次数；
- kernel 启动间隔；
- 每个房间的 fragment batch 数量；
- GPU 等待 CPU 提交下一小批数据的时间。

因此推理阶段更容易持续保持高 GPU 利用率。继续提高显存上限但不增加
有效 batch，不一定会进一步提速。

## 6. 两种配置的切换流程

从训练并行模式切换到训练后模式时，按以下顺序执行：

1. 等当前权重完成全部 68 个房间。
2. 确认其 `metrics.json` 和 `run_meta.json` 已生成。
3. 停止或暂停旧评估调度器，不能中断正在写结果的 worker。
4. 确认训练 PID 已自然退出。
5. 确认 GPU 空闲显存不少于 12000 MiB。
6. 用 `epoch_200` 和 `Area_5-hallway_2` 执行 batch 4 预检。
7. 预检成功后启动高显存权重队列。
8. 评估工具根据 `metrics.json` 自动跳过已完成权重。

不要为了切换配置删除已完成结果，也不要使用 `--overwrite` 重跑已有
权重。不同资源配置会记录在各自的 `run_meta.json` 中，便于后续审计。

## 7. 结果完整性

fragment batch 从 1 改为 4 只改变一次送入 GPU 的分片数量。固定协议仍会：

1. 对每个房间生成全部测试分片；
2. 对每个分片执行推理；
3. 将分片预测映射回原始点；
4. 汇总全部投票并计算 68 个房间的指标。

因此不会因为提高 batch 而裁掉房间或丢失点。不同 batch 下可能存在正常的
浮点舍入微差，所以必须在 `run_meta.json` 中保留
`fragment_batch_size_test`，分析时不能隐去资源配置。

## 8. 状态检查命令

查看 GPU：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
```

查看高显存实时日志：

```bash
tail -f exp/fixed_protocol/logs/latest_v17_scale04_posttrain_highmem.log
```

查看已经完成的权重：

```bash
find exp/fixed_protocol/results/v17-scale04-concurrent-screening \
  -type f -name metrics.json -print
```

## 9. 最终建议

- 训练期间使用 0.17、batch 1、2 workers，优先保护训练稳定性。
- 训练结束后使用 0.80、batch 4、6 workers，提高推理吞吐。
- 每种新模型或新显卡第一次使用高显存配置时，必须先跑最大房间预检。
- 训练期间只读取不可变 epoch 权重，best 和 last 留到训练结束后评估。
- 任何保护逻辑只能终止评估进程，不能操作训练进程或实例电源。
