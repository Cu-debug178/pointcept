# S3DIS 固定协议复评结果（2026-07-14）

## 1. 数据来源与完整性

本报告核验的数据来自：

```text
exp/s3dis_fixed_protocol_all_results_20260714_1208.zip
```

归档包含 272 个条目，主体包括 `results/`、`analysis/`、`logs/` 和
`telemetry/`。独立一致性检查结果：

- scale04 sweep：12 份 `metrics.json` 和 12 份 `run_meta.json`；
- 旧 0.02m 实验：本机可用的 6 个实验各有 best/last，共 12 份指标；
- 每份指标均包含 Area5 全部 68 个房间和 13 个类别；
- 旧实验协议为 `identity / grid=0.02 / point_max=60000`；
- scale04 协议为 `identity / grid=0.04 / point_max=60000`；
- scale04 的 `epoch_200` 与 `model_last` 指标完全一致。

压缩包中 scale04 的 `failed_entries.json` 是首次磁盘写满后遗留的历史状态，
其中列出的 5 项现在均已有完整 metrics。应以
`completeness.json: complete=true, completed=12, expected=12` 为准。

“旧实验 12/12 完整”仅指该服务器本地存在的 6 个实验。原计划中的
`v16b-controlled-support_20260620_2030`（seed 50040149）best/last 未包含，
因此完整的三 seed v16b 固定协议统计仍缺 2 项。

## 2. scale04 全周期结果

| Checkpoint | Epoch | mIoU | mAcc | OA |
| --- | ---: | ---: | ---: | ---: |
| epoch_20 | 20 | 0.6110 | 0.7063 | 0.8605 |
| epoch_40 | 40 | 0.6541 | 0.7505 | 0.8820 |
| epoch_60 | 60 | 0.6292 | 0.7117 | 0.8828 |
| epoch_80 | 80 | 0.6624 | 0.7420 | 0.8926 |
| epoch_100 | 100 | 0.6496 | 0.7270 | 0.8887 |
| epoch_120 | 120 | 0.6660 | 0.7560 | 0.8845 |
| epoch_140 | 140 | 0.6699 | 0.7378 | 0.8938 |
| epoch_160 | 160 | 0.6739 | 0.7483 | 0.8948 |
| epoch_180 | 180 | 0.6866 | 0.7535 | 0.9005 |
| model_best | 193 | 0.6863 | 0.7550 | 0.9002 |
| epoch_200 / model_last | 200 | **0.6939** | **0.7608** | **0.9024** |

120 到 200 epoch 的固定协议 mIoU 单调上升：

```text
0.6660 -> 0.6699 -> 0.6739 -> 0.6866 -> 0.6939
```

随机裁剪验证选择的 `model_best` 是 epoch 193，但固定全场景结果中
epoch 200 比它高 `+0.0076`。这证明随机 val 没有改变训练参数，却会误导
checkpoint 选择。当前 scale04 的可信主结果应使用 epoch 200，而不是
`model_best.pth`。

## 3. 旧实验 best/last 固定复评

| Family | Best mIoU | Last mIoU | Last - Best |
| --- | ---: | ---: | ---: |
| baseline | 0.6556 | 0.6519 | -0.0037 |
| v13 DA-Radius | 0.6585 | 0.6620 | +0.0035 |
| v16-local | 0.6526 | 0.6569 | +0.0043 |
| v16b seed 40979289 | 0.6671 | 0.6637 | -0.0033 |
| v16b seed 19095314 | 0.6560 | 0.6538 | -0.0022 |
| v17 dual-support | **0.6685** | 0.6677 | -0.0008 |

旧 0.02m 路线中固定协议最高值是 v17 best 的 `0.6685`，但它只比旧
baseline best 高 `0.0130`，且训练成本显著更高。scale04 plain KPConvX
epoch 200 达到 `0.6939`，比旧 baseline best 高 `0.0384`，也比 v17 best
高 `0.0254`。物理尺度校准带来的收益明显大于当前 support 双路径收益。

best/last 没有统一方向：v13、v16-local 的 last 略高，baseline、两条
v16b 和 v17 的 last 略低。因此不能默认 last 总是更好；每个模型都应在
同一固定协议下复评候选 checkpoint。

## 4. scale04 类别变化

相对旧 baseline best，scale04 epoch 200 的主要 IoU 变化：

| Class | Delta IoU |
| --- | ---: |
| door | +0.0933 |
| sofa | +0.0801 |
| window | +0.0754 |
| clutter | +0.0749 |
| column | +0.0688 |
| ceiling | +0.0616 |
| wall | +0.0267 |
| board | +0.0267 |
| table | -0.0205 |
| beam | -0.0056，最终 IoU 为 0 |

尺度校准不仅提升总体 mIoU，也改善了 door/window/column 等此前重点关注
的附着类和边界类。beam 仍然完全失败，table 有明确回落，后续结构不能只
优化总体尺度，还需要处理稀有细长类和家具类之间的取舍。

## 5. 当前结论与剩余工作

1. 当前最强可信结果是 scale04 plain KPConvX epoch 200：`0.6939`。
2. 原始 Pointcept 0.02m 配置存在明显物理尺度失配；过去部分 DA/support
   收益可能是在补偿该失配，不能直接迁移结论。
3. 不能把训练期随机 best `0.7159/0.7204` 与固定全场景结果混用。
4. 120-200 的持续上涨说明 100 epoch 短训不足以评价这个训练协议；但当前
   OneCycleLR 在末期已接近零，不能简单续训并期待同样斜率。
5. 下一轮应按路线二原计划训练同协议 scale04 v17，只改变 dual-support
   机制，直接检验旧尺度下 v17 的固定协议增益能否跨尺度保留。
6. 仍需从另一台服务器补齐 v16b seed 50040149 的 best/last，之后才能形成
   三 seed 固定协议均值。
