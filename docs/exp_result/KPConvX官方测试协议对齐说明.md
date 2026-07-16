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

## 2. 官方测试与当前测试的差异

| 项目 | 当前 identity 固定复评 | Apple Standalone S3DIS |
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

## 3. 本次新增协议

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

## 4. 首轮测试对象

只测试当前可信主基线：

```text
run: kpconvx-scale04-baseline_20260710_2057
checkpoint: epoch_200.pth
identity mIoU: 0.693926
```

先跑单房间 smoke，再跑 Area5 全量。TTA13 约为 identity 的 13 倍推理量，
official_vote10 约为 10 倍；实际耗时还受数据准备和 fragment batch 影响。

## 5. 结果解释

- identity、TTA13、official-like vote10 必须分别报告，不能混写成官方复现结果。
- 如果两种增强都只提高很少，当前 `0.6939` 的主要差距不在测试投票。
- 如果 vote10 明显优于 TTA13，说明随机小扰动和 EMA 更适合当前模型。
- 如果 TTA13 明显更高，它可以作为论文中的 TTA 结果，但不能称为 Apple 官方
  测试协议。
- 无论测试提升多少，都不能消除 `input_channels=9 vs 5`、`constant vs linear`
  influence、`share_kp=False vs True` 及训练 sampler/优化预算等训练期差异。
