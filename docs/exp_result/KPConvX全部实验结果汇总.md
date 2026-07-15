# KPConvX 全部实验结果汇总

更新时间：2026-07-15

## 1. 文档目的

本文汇总当前项目中能够从本地日志、CSV、保存配置、固定协议结果和阶段文档
核验到的全部 KPConvX 改进实验。内容包括：

- 早期 Hybrid、DA-Radius、全局上下文、边界门控、DA-meta、controlled
  support、dual-support 和 scale04 尺度校准；
- 已实际训练的 run、只有配置但没有独立完整结果的版本；
- 训练期随机验证结果和固定全场景复评结果；
- 训练稳定性、类别变化、计算成本和当前结论。

本文不把不同评价协议的 mIoU 混在同一排名中。

## 2. 指标口径

### 2.1 训练期随机验证

旧实验训练期间的验证使用：

```text
GridSample(mode="train")
+ SphereCrop(point_max=40000, mode="random")
```

这类结果适合观察训练趋势和保存 `model_best.pth`，但每个 epoch 都含随机
采样噪声。`best mIoU` 是最多 200 次随机验证中的最高点，不能直接等同于
整房间测试结果，也不能直接与论文结果比较。

### 2.2 固定全场景复评

当前可信复评使用：

- S3DIS Area5 全部 68 个房间；
- identity 单视角；
- `TestSphereCrop(point_max=60000)` 重叠 fragment；
- 旧实验使用匹配模型的 `grid=0.02`；
- scale04 实验使用匹配模型的 `grid=0.04`；
- 分别报告 mIoU、mAcc、OA、13 类 IoU 和房间级结果。

因此，本文将“训练期随机验证”和“固定全场景复评”分栏记录。

## 3. 当前总结果

1. 当前最强可信结果是 **scale04 plain KPConvX epoch 200**：固定全场景
   `mIoU=0.6939`、`mAcc=0.7608`、`OA=0.9024`。
2. 旧 0.02m 配置中固定协议最高的是 **v17 dual-support best**：
   `mIoU=0.6685`，比旧 baseline best 高 `+0.0130`，但训练成本约为
   baseline 的 `1.85x`。
3. scale04 对旧 baseline 的固定提升为 `+0.0384`，明显大于当前
   DA/support 结构带来的提升，说明物理尺度失配是此前的重要瓶颈。
4. v16b controlled support 的训练期三 seed best 均值约为 `0.7064`，低于
   旧 baseline 随机 best `0.7159`，单次 `0.7161` 不能作为稳定提升。
5. v17 expanded residual 在训练后期确实启用，但推理期 alpha 从 0 到 1
   只带来不超过 `0.0008` 的变化，简单调弱 residual 不能修复该路线。
6. **scale04-v17 已完成 22/22 个 checkpoint 的固定复评**。最高点是
   epoch 160 的 `mIoU=0.693936`，scale04 baseline epoch 200 为
   `0.693926`，差值仅 `+0.000010`，远低于预注册的 `+0.005` 门槛，不能
   称为提升。
7. scale04-v17 的 epoch 200 为 `0.681965`，比 baseline epoch 200 低
   `0.011961`；训练期随机选择的 model_best（epoch 191）固定结果也只有
   `0.684036`。这说明 dual-support 带来中期峰值，但没有形成稳定的后期收益。

这里的 scale04 只校准了 Pointcept 管线中的 S3DIS 物理尺度，包括
`grid_size=0.04`、`subsample_size=0.04`、`kp_radius=kp_sigma=2.1`。它仍使用
Pointcept 的数据增强、CE+Lovasz、OneCycleLR 和 4090D 训练预算，不是 Apple
官方 Standalone S3DIS 训练管线的完整复现。

## 4. 全部实际训练实验

表中“随机 best/final”均来自训练期随机裁剪验证；“固定 best/last”来自统一
全场景复评。`-` 表示没有完成该项复评或本地没有对应结果。

| 阶段/实验 | 核心改动 | 状态 | 随机 best / final mIoU | 固定 best / last mIoU | 简要结论 |
| --- | --- | --- | ---: | ---: | --- |
| early hybrid `s3dis-kpconvx-hybrid-training` | DA、SGCA、refine 首次同时加入 | 72/100，失败 | `0.2561 / 0.0034` | - | 出现 CUDA device-side assert，结构和工程均不稳定 |
| early hybrid rerun `20260419_2343` | 早期混合版复跑 | 100/100，训练后报错 | `0.2400 / 0.0034` | - | 完成训练但性能极低，后置阶段还有 OOM |
| early hybrid adjusted `20260421_2323` | 调整后的 DA+SGCA+refine | 200/200，训练后报错 | `0.6602 / 0.5849` | - | 能正常学习，但明显低于 baseline，变量过多 |
| hybrid `20260428` | DA、SGCA、refine，batch 2 调整版 | 161/200，提前结束 | `0.5383 / 0.3534` | - | 组合仍不稳定，不作为主线 |
| 旧 baseline | Pointcept KPConvX，0.02m | 200/200，训练后报错 | `0.7159 / 0.6059` | `0.6556 / 0.6519` | 随机 best 很高，但固定全场景结果明显较低 |
| `kpconvx_stage2_da_only` | Stage2 DA-only 探索 | 200/200，训练后报错 | `0.6939 / 0.6939` | - | DA-only 接近有效区间，但未超过旧 baseline 随机 best |
| `hybrid-old_20260506` | KPConvXStage2 中间混合版 | 146/200，提前结束 | `0.6760 / 0.5884` | - | 未跑满，只作为阶段样例 |
| `da-v1-balanced` | balanced 半径调制 | 139/200，提前结束 | `0.6501 / 0.6056` | - | 半径策略仍不稳定 |
| v12 | Torch DA-Radius mask 参考 | 167/200，提前结束 | `0.6815 / 0.6109` | - | 主要用于语义正确性参考，不是性能主线 |
| v13 | CUDA DA-Radius，stage3/4 | 200/200，后置 OOM | `0.7096 / 0.6386` | `0.6585 / 0.6620` | 最接近成功的真实动态半径版本；固定 last 略高于 best |
| v14 encoder-SGCA | encoder stage4/5 强融合全局上下文 | 103/200，提前止损 | `0.6001 / 0.5448` | - | encoder/skip 特征被强改写，副作用明显 |
| v14 decoder-SGCA | decoder/head 前轻量 SGCA | 200/200，后置 worker 异常 | `0.7004 / 0.6868` | - | 融合位置更安全，但简单全局上下文无稳定净收益 |
| v13b | 更保守 DA-Radius、候选邻居 24、clip 2.0 | 186/200，提前结束 | `0.6841 / 0.6786` | - | 保守半径没有修复问题，反而削弱 v13 收益 |
| v13 saved-code rerun | v13 保存代码复跑，只把 clip 放宽到 2.0 | 191/200，提前结束 | `0.7030 / 0.6115` | - | 单改梯度裁剪不能突破 baseline |
| v15 stage234 | serialized token、stage2/3/4、boundary gate/loss、refine | 107/200，提前止损 | `0.5971 / 0.5545` | - | boundary loss 会下降，但主分割负迁移，组合过重 |
| v16 DA-stat+GC | DA-meta 条件器和 decoder GC 同时启用 | 58/200，提前止损 | `0.5916 / 0.5916` | - | 变量仍未隔离，短训结果不能证明 GC 有效 |
| v16-local | 不改图，只在 stage3/4 注入 DA-meta channel bias | 200/200，完成 | `0.6977 / 0.6693` | `0.6526 / 0.6569` | 统计信号可被主干使用，但不足以复现 v13 |
| v16b seed 40979289 | hard controlled support，stage3/4 | 200/200，完成 | `0.7161 / 0.6873` | `0.6671 / 0.6637` | 单次随机峰值略超 baseline，但不具备多 seed 稳定性 |
| v16b seed 50040149 | 同一 controlled support，不同 seed | 200/200，完成 | `0.7032 / 0.6642` | - | 固定 best/last 尚未补齐 |
| v16b seed 19095314 | 同一 controlled support，不同 seed | 200/200，完成 | `0.7000 / 0.5961` | `0.6560 / 0.6538` | final 回落明显，显示高方差 |
| v17 dual-support | original support 保底，expanded support 做 gated residual | 200/200，完成 | `0.7084 / 0.5861` | `0.6685 / 0.6677` | 旧尺度固定结果最好，但训练慢且随机验证没有净收益 |
| scale04 baseline | 只校准 S3DIS 物理尺度到 0.04m | 200/200，完成 | `0.7204 / 未作为主结果` | `0.6863@best / 0.6939@last` | 当前最强可信基线；epoch 200 高于随机 val 选择的 epoch 193 |
| scale04-v17 | scale04 baseline 上加入同一 v17 dual-support | 200/200，固定复评 22/22 完成 | 原始训练日志待补 | `0.693936@160 / 0.681965@last` | 未达到 `0.6989`；最高点与 baseline 实质持平，后期回落 |

## 5. 配置级实验与未形成独立结果的版本

以下版本存在配置或设计记录，但没有找到可独立核验的完整训练结果，不能写成
“已经验证有效”或“已经验证失败”。

| 版本 | 设计目的 | 当前证据状态 |
| --- | --- | --- |
| v1 | 保守 CUDA DA-Radius | 配置级探索，思想延续到后续版本 |
| v2 | balanced DA-Radius | 配置级探索，没有独立完整日志 |
| v3 | 更强 DA-Radius、更深 stage | 配置级探索，没有独立完整日志 |
| v4 | Torch reference | 参考实现，部分思想延续到 v12 |
| v5 | DA-kernel only | 用于拆解动态 kernel 贡献，未找到独立完整结果 |
| v6 | DA-radius only | 后续 v13 主线的前置消融设计 |
| v7 | global context only | 简单全局上下文配置级消融 |
| v8 | DA-radius + global | 局部和全局组合配置级消融 |
| v9 | DA-kernel + DA-radius + global | 全量组合，归因困难 |
| v10/v11 | constant 与 linear influence 对照 | 底层 influence 配置消融 |
| v15 base/4090D | PTv3-inspired serialized dual attention | 没有找到独立完整主结果；stage234 有明确失败 run |
| v16b superset-only | 只扩大候选邻域，不开 hard mask | 诊断配置，不作为正式结果 |
| v16c light-global | controlled support 后增加轻量 decoder GC | 配置已完成，未找到独立完整训练结果 |
| PTv3 原生 backbone benchmark | 验证完整 serialization backbone | 尚未正式训练 |

## 6. 关键完整训练的稳定性与成本

| 实验 | best epoch | random best | final | tail20 mean/std | 训练时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 baseline | 185 | 0.7159 | 0.6059 | `0.6407 / 0.0336` | 24.10h |
| v13 | 175 | 0.7096 | 0.6386 | `0.6498 / 0.0294` | 24.48h |
| v14 decoder-SGCA | 193 | 0.7004 | 0.6868 | `0.6543 / 0.0316` | 27.57h |
| v16-local | 156 | 0.6977 | 0.6693 | `0.6495 / 0.0287` | 31.77h |
| v16b seed 40979289 | 176 | 0.7161 | 0.6873 | `0.6521 / 0.0338` | 29.32h |
| v16b seed 50040149 | 167 | 0.7032 | 0.6642 | `0.6586 / 0.0287` | 26.55h |
| v16b seed 19095314 | 146 | 0.7000 | 0.5961 | `0.6401 / 0.0308` | 29.39h |
| v17 | 193 | 0.7084 | 0.5861 | `0.6514 / 0.0345` | 44.52h |
| scale04-v17 | fixed peak 160 | 原始随机指标待补 | fixed last 0.6820 | fixed 160-200 波动 | 约 23.7h（checkpoint 时间估算） |

主要判断：

- 多个实验的单点 best 比 tail 均值高很多，存在明显 best-of-200 选择偏差；
- v16b 三 seed best 为 `0.7161/0.7032/0.7000`，均值约 `0.7064`，说明
  hard support 具有高方差；
- v17 参数量只从约 13.55M 增加到约 14.00M，但训练时间增加到 44.52h；
  主要原因是 stage4 original/expanded 两次邻域聚合，不是参数量本身；
- v17 后期 `gamma≈0.706`、`gate_mean≈0.508`、`residual_ratio≈0.13`、
  `extra_util≈0.69`，说明 expanded branch 确实参与了训练。
- scale04-v17 从运行目录时间到 epoch 200 约 23 小时 44 分，epoch 10 到
  epoch 200 的 checkpoint 时间差为 22.49 小时，比旧 v17 的 44.52 小时短
  约 47%。评估配置确认 dual-support 仍启用；准确 batch time 和训练期 gate
  监控仍需原始 `train.log` 核验。

## 7. 固定全场景复评结果

### 7.1 旧 0.02m 实验

| 实验 | fixed best | fixed last | last-best |
| --- | ---: | ---: | ---: |
| baseline | 0.6556 | 0.6519 | -0.0037 |
| v13 DA-Radius | 0.6585 | 0.6620 | +0.0035 |
| v16-local | 0.6526 | 0.6569 | +0.0043 |
| v16b seed 40979289 | 0.6671 | 0.6637 | -0.0033 |
| v16b seed 19095314 | 0.6560 | 0.6538 | -0.0022 |
| v17 dual-support | **0.6685** | 0.6677 | -0.0008 |

v16b seed 50040149 的 fixed best/last 尚未补齐，因此不能形成完整的 v16b
三 seed 固定协议均值。

### 7.2 scale04 baseline 全周期

| Checkpoint | Epoch | fixed mIoU | mAcc | OA |
| --- | ---: | ---: | ---: | ---: |
| epoch 20 | 20 | 0.6110 | 0.7063 | 0.8605 |
| epoch 40 | 40 | 0.6541 | 0.7505 | 0.8820 |
| epoch 60 | 60 | 0.6292 | 0.7117 | 0.8828 |
| epoch 80 | 80 | 0.6624 | 0.7420 | 0.8926 |
| epoch 100 | 100 | 0.6496 | 0.7270 | 0.8887 |
| epoch 120 | 120 | 0.6660 | 0.7560 | 0.8845 |
| epoch 140 | 140 | 0.6699 | 0.7378 | 0.8938 |
| epoch 160 | 160 | 0.6739 | 0.7483 | 0.8948 |
| epoch 180 | 180 | 0.6866 | 0.7535 | 0.9005 |
| model_best | 193 | 0.6863 | 0.7550 | 0.9002 |
| epoch 200 / model_last | 200 | **0.6939** | **0.7608** | **0.9024** |

120 到 200 epoch 的 fixed mIoU 为：

```text
0.6660 -> 0.6699 -> 0.6739 -> 0.6866 -> 0.6939
```

随机验证选择的 epoch 193 并不是固定协议最优点，epoch 200 高 `0.0076`。
这证明训练期随机验证不影响优化过程，但会影响 checkpoint 选择。

### 7.3 scale04-v17 全周期

本次归档包含 epoch 10 到 200、model_best 和 model_last，共 22 个 checkpoint，
`completeness.json` 记录为 `complete=true, completed=22, expected=22`。

| Checkpoint | Epoch | fixed mIoU | mAcc | OA |
| --- | ---: | ---: | ---: | ---: |
| epoch 10 | 10 | 0.4954 | 0.6095 | 0.8165 |
| epoch 20 | 20 | 0.6041 | 0.6706 | 0.8640 |
| epoch 30 | 30 | 0.5810 | 0.6672 | 0.8692 |
| epoch 40 | 40 | 0.6146 | 0.6951 | 0.8593 |
| epoch 50 | 50 | 0.6009 | 0.6807 | 0.8710 |
| epoch 60 | 60 | 0.6345 | 0.7276 | 0.8782 |
| epoch 70 | 70 | 0.6455 | 0.7418 | 0.8741 |
| epoch 80 | 80 | 0.6586 | 0.7524 | 0.8930 |
| epoch 90 | 90 | 0.6484 | 0.7413 | 0.8850 |
| epoch 100 | 100 | 0.6678 | 0.7573 | 0.8903 |
| epoch 110 | 110 | 0.6515 | 0.7313 | 0.8771 |
| epoch 120 | 120 | 0.6455 | 0.7307 | 0.8795 |
| epoch 130 | 130 | 0.6525 | 0.7236 | 0.8857 |
| epoch 140 | 140 | 0.6602 | 0.7232 | 0.8955 |
| epoch 150 | 150 | 0.6615 | 0.7293 | 0.8923 |
| **epoch 160** | **160** | **0.693936** | **0.7538** | **0.9064** |
| epoch 170 | 170 | 0.6827 | 0.7469 | 0.9021 |
| epoch 180 | 180 | 0.6753 | 0.7459 | 0.8997 |
| epoch 190 | 190 | 0.6848 | 0.7493 | 0.9017 |
| model_best | 191 | 0.6840 | 0.7489 | 0.9024 |
| epoch 200 / model_last | 200 | 0.6820 | 0.7487 | 0.9013 |

### 7.4 与 scale04 baseline 的直接裁决

| 比较项 | scale04 baseline | scale04-v17 | 差值 |
| --- | ---: | ---: | ---: |
| 共同 20-epoch 网格内最高 mIoU | 0.693926 @ 200 | 0.693936 @ 160 | **+0.000010** |
| 对应 mAcc | **0.760752** | 0.753759 | -0.006994 |
| 对应 OA | 0.902378 | **0.906361** | +0.003984 |
| model_best 固定 mIoU | **0.686318** | 0.684036 | -0.002282 |
| model_last 固定 mIoU | **0.693926** | 0.681965 | -0.011961 |

v17 没有达到预注册目标 `0.6989`。虽然 epoch 160 的 OA 更高，但 mIoU 只比
baseline 多 `0.000010`，同时 mAcc 更低；这个差异远小于单 seed 和 checkpoint
波动，必须判定为实质持平。160 之后连续回落，说明 dual-support 改变了训练
轨迹或收敛时点，但没有提高最终可信上限。

## 8. 类别级结论

### 8.1 v13 DA-Radius

- beam、board、window、sofa 曾出现正向信号；
- door、column 等类别出现下降；
- 说明真实邻域变化具有结构类价值，但收益与边界风险并存，且不够稳定。

### 8.2 v16b controlled support

- sofa、clutter 在部分 seed 上较强；
- beam、board、window、door、column 对 seed 和 checkpoint 高度敏感；
- hard support mask 不是稳定主线，单个类别峰值不能代表机制成功。

### 8.3 v17 固定协议相对旧 baseline

主要提升：

| 类别 | IoU 变化 |
| --- | ---: |
| ceiling | +0.0508 |
| clutter | +0.0436 |
| beam | +0.0425 |
| window | +0.0383 |
| column | +0.0309 |
| board | +0.0135 |

主要下降：

| 类别 | IoU 变化 |
| --- | ---: |
| sofa | -0.0332 |
| door | -0.0154 |
| bookcase | -0.0109 |

因此，旧判断“v17 系统性污染 wall/window/column/clutter”没有被固定协议
支持。真正明确下降的是 sofa、door 和 bookcase；边界污染仍是待审计假设。

### 8.4 scale04 baseline 相对旧 baseline

| 类别 | IoU 变化 |
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

尺度校准显著改善了 door/window/column 等重点类别，但 beam 仍完全失败，
table 也有回落。后续仍需要专门处理稀有细长类，而不是只扩大整体感受野。

### 8.5 scale04-v17 epoch 160 相对 scale04 baseline epoch 200

主要提升：

| 类别 | IoU 变化 |
| --- | ---: |
| door | +0.0459 |
| table | +0.0369 |
| clutter | +0.0243 |
| ceiling | +0.0133 |
| bookcase | +0.0080 |

主要下降：

| 类别 | IoU 变化 |
| --- | ---: |
| sofa | -0.0480 |
| window | -0.0339 |
| column | -0.0311 |
| board | -0.0145 |

beam 在两者中均为 0。v17 没有满足“door/window/column/board/clutter 不出现
明显下降”的类别约束：door 和 clutter 改善，但 window、column、board
回落，sofa 下降最大。这是类别重分配，不是总体能力提升。

## 9. v17 推理期 alpha 消融

该实验不重新训练，只在同一 checkpoint 上设置：

```text
alpha = 0 / 0.25 / 0.5 / 0.75 / 1.0
```

| alpha | best checkpoint mIoU | last checkpoint mIoU |
| ---: | ---: | ---: |
| 0.00 | 0.6828 | 0.6772 |
| 0.25 | 0.6833 | 0.6777 |
| 0.50 | 0.6834 | 0.6779 |
| 0.75 | 0.6835 | 0.6777 |
| 1.00 | 0.6836 | 0.6775 |

全部差异不超过 `0.0008`。直接结论是：固定当前 v17 权重后，推理期关闭
或缩小 expanded residual 都不能显著改变结果。alpha=0 不等于 baseline，
因为主干已经在 dual-support 训练过程中共同适配。

alpha sweep 的绝对数值来自该诊断工具自己的固定设置，只用于同一 checkpoint
内横向比较，不与第 7 节统一 68 房间固定协议的绝对 mIoU 混合排名。

## 10. 已确认的路线结论

1. 早期把 DA、global、refine 一次性堆叠会造成归因困难和明显负迁移。
2. 真实 CUDA DA-Radius 有结构类信号，但 v13/v13b 证明半径强弱与收益不是
   单调关系。
3. decoder/head 前全局融合比 encoder 强融合安全，但简单 mean token 或
   SGCA 没有稳定超过 baseline。
4. 显式 boundary loss 可以下降，但不保证主分割受益；v15 已证明两者不等价。
5. DA-meta conditioner 能被 KPConvX 使用，但不改邻域图时不足以复现 v13。
6. hard controlled support 高方差，dual-support residual 虽保住原始邻域，
   但计算成本高，旧训练协议下也没有稳定随机验证收益。
7. 物理尺度校准目前是最大、最可信的提升来源，但它属于基线修正，不是结构
   创新。
8. scale04-v17 已完成该隔离变量验证：其最高 fixed mIoU 与 scale04 baseline
   只差 `+0.000010`，final 反而低 `0.011961`。现有证据不支持继续把
   dual-support 作为提升 mIoU 的主线，旧 0.02m 下的正信号更可能包含尺度
   补偿和 checkpoint 选择效应。

## 11. scale04-v17 最终裁决与训练耗时

已完成实验：

```text
semseg-kpconvx-hybrid-v17-scale04-4090d-area5
```

该实验保持 scale04 baseline 的数据、优化器、200 个日志 epoch 和物理尺度，
只增加 v17 dual-support。固定评估已经证明它没有达到 `+0.005` 成功门槛，
且 window/column/board/sofa 存在明显类别下降，因此判定为未成功。

评估日志已确认：

- backbone 为 `KPConvXV17`；
- `enable_dual_support=True`，只开 stage4；
- 参数量为 13,999,028；
- `subsample_size=0.04`，候选邻居上限仍为 24；
- warmup/ramp、gamma 初始化和 gate 初始化与旧 v17 一致。

所以不到 24 小时不是因为误跑成 baseline。当前最合理的速度解释是：训练
GridSample 从 0.02m 改到 0.04m 后，进入 KPConvX 金字塔的活跃点显著减少，
original/expanded 两条 stage4 聚合路径都随之变便宜。保存频率从 20 改为 10
只增加少量 I/O，不会使训练提速。

目前仍缺原始 `train.log`，因此以下项目暂不能精确报告：

- 平均 batch/data time 及其与旧 v17 的倍率；
- 每个 batch 的实际点数分布；
- scale04-v17 后期 `dual_progress/gamma/gate_mean/residual_ratio/extra_util`；
- 训练期随机 best、final 和 tail20 统计。

## 12. 数据来源

本汇总主要依据：

- `exp/csv/experiments_summary.csv`
- `exp/csv/detailed_timeline.csv`
- `exp/csv/all_validation.csv`
- `docs/exp_result/kpconvx_experiment_route_summary.md`
- `docs/exp_result/2026-06-20汇报后KPConvX项目进展总结.md`
- `docs/exp_result/s3dis_fixed_protocol_results_20260714.md`
- `docs/exp_result/kpconvx_official_baseline_scale_calibration_summary.md`
- `docs/exp_result/v17_alpha消融实现记录.md`
- `exp/v16b-controlled-support_*/train.log`
- `exp/v17-dual-support_20260706_2303/train.log`
- `exp/kpconvx-scale04-baseline_20260710_2057/train.log`
- `exp/v17-scale04-concurrent-screening/screen/checkpoint_summary.csv`
- `exp/v17-scale04-concurrent-screening/screen/class_metrics.csv`
- `exp/v17-scale04-concurrent-screening/**/metrics.json`

原始模型、配置、日志和 TensorBoard 文件仍保存在 `exp/` 对应实验目录中。
