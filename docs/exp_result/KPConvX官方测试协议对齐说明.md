# KPConvX 官方测试协议对齐说明

## 1. 当前结论

Apple Standalone KPConvX 的 S3DIS 测试并不是固定 `TTA13`。官方默认使用
10 轮随机整房间投票，并以 `test_momentum=0.95` 对点级概率做 EMA。

当前 Pointcept identity 固定复评已经具备：

- Area5 全部房间；
- `grid_size=0.04`；
- `TestSphereCrop(point_max=60000)` 的完整点覆盖；
- fragment 概率回写到原始点；
- 固定 checkpoint、独立缓存和机器可读指标。

此前缺少的是多视角几何增强和跨视角概率融合。

汇报时必须使用以下三个完整名称：

| 编号 | 完整名称 | 权重 | 测试协议 | mIoU |
| --- | --- | --- | --- | ---: |
| A | 自训练 scale04 identity | 自训练 epoch 200 | Pointcept 单视角 fragment 概率回填 | **69.3926%** |
| B | 官方权重 Standalone 10-vote | Apple epoch 300 | 整房间、10 次随机 vote、完整点云重投影 | **73.47%** |
| C | 官方权重 Pointcept 4-rotation | 与 B 逐位相同的 Apple tensor | 4 个固定旋转、fragment 概率回填 | **68.07%** |

A 是自训练模型；B/C 才是同一官方权重的双协议对照。A 没有使用 TTA13 或 official-like vote10，C 也只有 4 个固定旋转。

A 的完整测试链路为：

```text
自训练 scale04 epoch 200
-> Area5 68 rooms
-> 0.04m GridSample(mode="test")
-> <=60000 点 fragments
-> identity 单视角
-> fragment softmax 按 index 回填
-> mIoU 69.3926%
```

## 2. 自训练单视角与官方 Standalone 的表面差异

| 项目 | A：自训练 identity 固定复评 | B：Apple Standalone S3DIS |
| --- | --- | --- |
| 测试次数 | 1 次 | 10 votes |
| 旋转 | identity | 每 vote 随机垂直旋转 |
| 缩放 | 无 | 各向同性 `0.99-1.01` |
| 翻转 | 无 | X 轴 50% 概率翻转 |
| 概率融合 | fragment softmax 累加 | 跨 vote EMA，momentum `0.95` |
| 房间输入 | 0.04m test voxel partitions + 60000 点分片 | 100m radius 覆盖完整房间 |
| 全分辨率映射 | Pointcept fragment index 回写 | 预下采样点预测后最近邻 projection |
| 中心化 | Pointcept bbox center + floor Z | XY 均值中心 + floor Z |
| 颜色 | Pointcept `NormalizeColor` | 官方训练统计标准化 |

最后三项与训练输入分布和 Standalone 数据管线绑定，不能安全地在旧权重测试时
直接替换。否则测到的是输入预处理失配，而不是 TTA/voting 收益。

这张表只能说明 A 与 B 的整体差异，不能用来计算 TTA 的单独贡献，因为 A/B 不是同一权重、也不是同一模型配置。

## 3. 2026-07-16 官方权重双协议实测

测试对象是同一份 Apple 官方 `S3DIS_KPConvX-L` epoch-300 权重。

| 路线 | 测试协议 | mIoU | mAcc | allAcc |
| --- | --- | ---: | ---: | ---: |
| Apple Standalone | 官方整房间 10-vote | **73.47%** | **78.51%** | **91.74%** |
| Pointcept | 0.04m、4 个固定 Z 轴旋转、fragment 聚合 | 68.07% | 72.99% | 89.91% |

Standalone 的 `73.47%` 四舍五入后为 Apple 公布的 `73.5%`，官方结果已成功复现。Pointcept 比 Standalone 低 `5.40` 个百分点。

两次测试的逐项流程为：

| 对比环节 | B：Apple Standalone | C：Pointcept 对照 |
| --- | --- | --- |
| 权重 | 官方 `current_chkp.tar` | 同一组 864 tensors，仅键加 `backbone.` 前缀 |
| 输入 | 5 维 `[1, RGB, 原始 z]` | 同样构造 5 维 `[1, RGB, 原始 z]` |
| 初始点处理 | 0.04m initial subsampling | 0.04m `GridSample(mode="test")` partitions |
| 房间组织 | `in_radius=100m`，完整房间 | 最多 60000 点 fragments |
| 中心化 | XY 均值中心 + floor Z | bbox 中心 + floor Z，fragment 再 XY CenterShift |
| 几何增强 | 随机垂直旋转、`0.99-1.01` 缩放、X 翻转 | 固定 `0/90/180/270` 度旋转 |
| 视角数量 | 10 votes | 4 views |
| 概率融合 | `momentum=0.95` 跨 vote 累计 | fragment softmax 按 index 累加 |
| 完整点恢复 | nearest projection 回原始完整点云 | fragment 概率直接回填原始 index |

这张 B/C 表才是合法的“同权重协议对照”。它证明全部推理流程合计造成 `5.40` 个百分点差异，但仍不能单独量化 10-vote 的贡献。

这里的 Pointcept `68.07%` 只用于同一官方权重的协议对照。它不能与自训练 scale04 baseline 的 identity `69.3926%` 直接排序，因为模型配置、输入特征、训练预算和测试视角均不相同。

| 模型配置 | A：自训练 scale04 | B/C：Apple 官方 KPConvX-L |
| --- | --- | --- |
| 输入维度 | 9 维 `coord + color + normal` | 5 维 `[1, RGB, z]` |
| shell sizes | `(1, 14, 28)`，43 点 | `(1, 14, 42)`，57 点 |
| influence | `constant` | `linear` |
| `share_kp` | `False` | `True` |
| neighbor limits | `(12,16,20,20,20)` | `(12,16,20,20,20)` |
| 训练 | Pointcept batch 3，无梯度累积 | Standalone batch 4，梯度累积 6 |

因此 `69.3926% > 68.07%` 不是有效的模型优劣结论，也不能由此推断 A 加 vote 后必然达到 `73.47%`。

Pointcept checkpoint 只给 864 个参数键增加 `backbone.` 前缀并移除测试不需要的优化器状态，核验结果为：

```text
source_tensors: 864
converted_tensors: 864
keys_match_after_prefix_removal: True
all_tensor_values_bitwise_equal: True
strict_load missing keys: 0
strict_load unexpected keys: 0
shape mismatches: 0
```

Pointcept 对照显式构造了官方 5 通道输入 `[1, R, G, B, z]`，因此差距不是权重损坏或输入通道顺序错误，而是两套推理/数据聚合流程不等价。官方 ZIP SHA-256 为 `9c89347f26327b1b4bad0f75a912d0cd6344209698147e23e64df447c7bc2487`，官方 checkpoint SHA-256 为 `1cdb6e0c028e6c57ce27665f6460b0e550568dadb46bc38c9565b4d9601a6849`，Pointcept 封装 checkpoint SHA-256 为 `dfc4a6d76d6f28340c7ed8473bf9dac4b62111c5346fdffda5210ebb9485a33e`。

原始报告位于：

```text
D:/虚拟C盘/Edge下载/EVAL_RESULTS_20260716.md
```

报告记录的官方混淆矩阵、Pointcept 预测和转换后 checkpoint 当前仍在云端路径，本地 `exp/` 中尚未同步这些大文件。汇报可使用上述结果和 hash，但完整可复现归档仍需补下载。

报告记录的云端证据路径：

```text
/root/autodl-tmp/ml-kpconvx/Standalone/KPConvX/results/S3DIS_KPConvX-L/test/test_001/report.txt
/root/autodl-tmp/ml-kpconvx/Standalone/KPConvX/results/S3DIS_KPConvX-L/test/test_001/full_conf_009.txt
/root/autodl-tmp/Pointcept-exp/s3dis/official-kpconvx-l-pointcept-area5/test.log
/root/autodl-tmp/Pointcept-exp/s3dis/official-kpconvx-l-pointcept-area5/result/
/root/autodl-tmp/Pointcept-exp/pretrained/kpconvx_official/pointcept/S3DIS_KPConvX-L.pth
```

云端报告使用的入口分别是 `./run_s3dis_kpconvx_l.sh` 和 `./scripts/test_s3dis_official_kpconvx_pointcept.sh`。这两个入口及其专用 Pointcept 配置当前没有出现在本地仓库，应在汇报后同步，避免只剩结果报告而缺少执行代码。

## 4. 本次新增协议

### `tta13`

- 4 个 Z 轴旋转：`0/90/180/270` 度；
- 三个尺度：`1.0/0.95/1.05`；
- 一次 X 轴翻转；
- 共 13 个确定性视角，直接累加 softmax 概率。

原实现最后一次 `RandomFlip(p=1)` 会同时翻转 X/Y，等价于 180 度旋转，和已有
视角重复。本次改为 `RandomFlipAxis(axis="x")`。

### `official_vote10`

- 固定随机种子预采样 10 组官方几何增强；
- 随机垂直旋转；
- `0.99-1.01` 各向同性缩放；
- X 轴 50% 概率翻转；
- 使用 `(1-m) * m^(9-v)` 权重复现从零初始化的 `m=0.95` EMA；
- 输出记录 vote 数、momentum、随机种子和协议签名。

该协议必须标记为 `official-like`。它对齐了官方测试期的几何投票和 EMA，但仍
使用 Pointcept 的 voxel partition、fragment 和 index 回写流程，不是 Standalone
测试代码的逐行复现。

## 5. 自训练基线的首轮测试对象

只测试当前可信主基线：

```text
run: kpconvx-scale04-baseline_20260710_2057
checkpoint: epoch_200.pth
identity mIoU: 0.693926
```

先跑单房间 smoke，再跑 Area5 全量。TTA13 约为 identity 的 13 倍推理量，
official_vote10 约为 10 倍；实际耗时还受数据准备和 fragment batch 影响。

## 6. 结果解释

- identity、TTA13、official-like vote10 必须分别报告，不能混写成官方复现结果。
- 如果两种增强都只提高很少，当前 `0.6939` 的主要差距不在测试投票。
- 如果 vote10 明显优于 TTA13，说明随机小扰动和 EMA 更适合当前模型。
- 如果 TTA13 明显更高，它可以作为论文中的 TTA 结果，但不能称为 Apple 官方
  测试协议。
- 无论测试提升多少，都不能消除 `input_channels=9 vs 5`、`constant vs linear`
  influence、`share_kp=False vs True` 及训练 sampler/优化预算等训练期差异。
