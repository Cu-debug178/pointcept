# KPConvX 官方基线核验与 S3DIS 尺度校准阶段总结

更新时间：2026-07-14

## 1. 当前阶段结论

当前项目使用的 KPConvX backbone 确实来自 Apple 官方仓库
[`apple/ml-kpconvx`](https://github.com/apple/ml-kpconvx) 的 Pointcept wrapper。
本地核验对应官方 commit：

```text
54e644a9f3bddd4c344a58193897a44582b0fea4
```

冻结 baseline 中的 `pointcept/models/kpconvx` 与官方
`Pointcept-wrapper/models/kpconvx` 基本一致：主干结构没有实质改动，差异仅包括无用导入清理、
GPU 初始化辅助函数和本地编译产物。因此，当前 baseline 可以定义为：

> 基于 Apple 官方 KPConvX Pointcept wrapper 的 S3DIS Area5 适配基线。

但它不能定义为论文中 S3DIS `73.5%` 结果的严格复现。官方 README 明确说明：

- KPConvX Pointcept wrapper 是 Standalone backbone 的严格转换；
- backbone 操作相同，但 Pointcept 与 Standalone 训练流程不同；
- 论文 S3DIS 使用 Standalone 流程，ScanNet 使用 Pointcept 流程。

因此必须区分“官方主干代码”和“论文 S3DIS 训练/评估协议”。

## 2. 三类指标不能混用

### 训练期随机验证

原 baseline 的训练期 best mIoU 为 `0.7159`，出现在 epoch 185。该指标来自：

```text
GridSample(mode="train")
+ SphereCrop(point_max=40000, mode="random")
```

它包含随机采样噪声，适合训练过程中观察趋势和保存候选 best checkpoint，不等价于全场景测试。

### 当前固定协议复评

固定协议使用 Area5 全部 68 个房间、单视角、重叠 fragment 聚合到完整房间：

| 模型 | 固定协议 mIoU | 相对 baseline |
| --- | ---: | ---: |
| baseline | 0.655574 | - |
| v13 | 0.658537 | +0.002964 |
| v16-local | 0.652634 | -0.002939 |
| v16b seed 40979289 | 0.667071 | +0.011498 |
| v16b seed 19095314 | 0.655959 | +0.000386 |
| v17 | 0.668531 | +0.012958 |

该结果是当前项目内部最可信的横向比较。服务器 B 上的 v16b seed
`50040149` 仍需补齐，之后才能得到完整的 v16b 三 seed 固定协议均值。

### 论文结果

论文报告的 KPConvX-L S3DIS Area5 best mIoU 为 `73.5%`，10 次实验均值为
`72.4 +/- 0.9%`。该结果来自 Standalone S3DIS 训练流程、全房间测试和 voting，
不能与上述 Pointcept 随机验证或单视角固定协议直接比较。

## 3. 当前 Pointcept 基线与官方 S3DIS 尺度差异

| 项目 | 当前 Pointcept baseline | 官方 Standalone S3DIS 配置 |
| --- | ---: | ---: |
| 初始网格 | 0.02 m | 0.04 m |
| `kp_radius` | 2.3 | 2.1 |
| 第一层物理半径 | 0.046 m | 0.084 m |
| 输入特征 | 9 维：coord + RGB + normal | 5 维：常数 + RGB + z |
| `kp_influence` | constant | linear |
| `share_kp` | False | True |
| 训练 batch | 3 | 4 x 累积 6 |
| 测试 | Pointcept fragment | Standalone 全房间 voting |

第一层物理半径由下式决定：

```text
first_radius = subsample_size * kp_radius
```

当前 `0.02 * 2.3 = 0.046m`，尺度校准后为 `0.04 * 2.1 = 0.084m`。
当前局部物理感受野约为校准配置的 55%。同时，0.02m 网格会保留更多点，
增加显存占用并迫使训练 crop 覆盖更小的物理区域。这可能是 DA-Radius 在部分结构类上出现正信号的原因之一。

## 4. 测试配置对训练结果的影响

- 普通 test/PreciseEvaluator 不参与反向传播，不会改变已训练权重。
- val 配置虽然不产生梯度，但会决定哪个 checkpoint 被标记为 `model_best.pth`。
- 随机 val crop 会造成 best checkpoint 选择噪声，因此最终结论必须使用统一固定协议复评。
- train 中的网格、crop、batch、梯度累积和卷积物理半径会直接改变学习结果。

## 5. 路线二：S3DIS 物理尺度校准

路线二第一轮不是完整复现论文，而是在当前 Pointcept/4090D 条件下隔离验证物理尺度：

### 保持不变

- 模型深度、通道和参数量；
- `neighbor_limits`；
- 输入 9 维特征；
- `kp_influence="constant"`；
- `share_kp=False`；
- batch size 3；
- 单样本最多 40000 点；
- AdamW、OneCycleLR、CE + Lovasz；
- 200 个日志 epoch、训练集 loop 5；
- 不启用 AMP，避免引入额外数值变量。

### 只校准

- train/val/test `GridSample.grid_size: 0.02 -> 0.04`；
- backbone `subsample_size: 0.02 -> 0.04`；
- `kp_radius: 2.3 -> 2.1`；
- `kp_sigma: 2.3 -> 2.1`。

### 4090D 测试策略

- 训练仍使用 batch 3、40000 点；
- 训练结束的 PreciseEvaluator 使用 60000 点重叠 fragment；
- fragment batch size 为 2；
- 默认只跑 identity 单视角，不在训练结束时启动 13-view TTA；
- 13-view 只在模型通过单视角筛选后单独运行。
- 每 20 个日志 epoch 保存一个可恢复 checkpoint：`epoch_20.pth` 到
  `epoch_200.pth`；每个实验额外占用约 1.6GB。

## 6. 实验顺序与判定

1. 先训练 `scale04 baseline`，回答尺度校准本身是否改善固定协议结果。
2. baseline 完成后，再训练同协议的 `scale04 v17`，回答 v17 的固定协议增益能否跨尺度保留。
3. 两个实验都必须比较 fixed identity mIoU、mAcc、OA 和 per-class IoU。
4. 重点观察 `beam/column/window/door/board/clutter`，不能只看总 mIoU。

建议判定规则：

- 如果 scale04 baseline 比当前 baseline `0.6556` 提升至少 `+0.005`，说明尺度失配确实影响主干。
- 如果 scale04 v17 比 scale04 baseline 提升至少 `+0.005`，且边界类没有明显下降，说明 v17 收益具有跨尺度稳定性。
- 如果 scale04 baseline 上升但 v17 增益消失，说明过去的 support/DA 收益主要是在补偿 0.02m 尺度问题。
- 如果二者都没有提升，应停止把物理尺度视为主要瓶颈，再转向训练协议或新全局分支。

## 7. 新配置

```text
configs/s3dis/semseg-kpconvx-base-s3dis-scale04-4090d-area5.py
configs/s3dis/semseg-kpconvx-hybrid-v17-scale04-4090d-area5.py
scripts/train_route2_scale04.sh
```

旧 baseline、v13-v17 配置保持不变，保证历史实验可复现。

固定复评工具已增加显式网格参数。评估 scale04 checkpoint 时必须使用：

```text
--grid-size 0.04
```

默认值仍为 0.02，旧实验和旧缓存保持兼容。复评元数据会记录 `grid_size`，
防止 0.02m 与 0.04m 结果误用同一预测缓存。

## 8. scale04 实际结果

路线二 baseline 已完成训练和 12 个 checkpoint 的统一全场景复评。最终结果：

| 模型/权重 | 固定 mIoU | mAcc | OA |
| --- | ---: | ---: | ---: |
| 旧 0.02m baseline best | 0.6556 | 0.7313 | 0.8790 |
| scale04 model_best（epoch 193） | 0.6863 | 0.7550 | 0.9002 |
| scale04 epoch 200 / model_last | **0.6939** | **0.7608** | **0.9024** |

scale04 epoch 200 相对旧 baseline 提升 `+0.0384` mIoU，远高于预设的
`+0.005` 成功阈值，确认物理尺度失配是此前 Pointcept KPConvX 基线的主要
问题之一。当前不应立即训练 scale04 v17；应先对齐更多官方 S3DIS 主干设置，
并把 scale04 plain KPConvX 作为所有新结构的统一对照。
