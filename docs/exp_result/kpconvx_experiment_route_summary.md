# KPConvX 改进实验总路线

更新时间：2026-07-14

## 2026-07-14 固定协议补充结论

固定协议复评已改变项目的基线判断。训练期随机裁剪 best 只适合观察训练和
保存候选 checkpoint，不能继续作为主结果。当前统一 Area5 全场景 identity
协议下：

- 旧 0.02m baseline best/last 为 `0.6556 / 0.6519`；
- 旧 0.02m 路线最高是 v17 best `0.6685`；
- scale04 plain KPConvX epoch 200 达到 `0.6939`；
- scale04 比旧 baseline best 高 `+0.0384`，比 v17 best 高 `+0.0254`；
- scale04 从 epoch 120 到 200 的 mIoU 为
  `0.6660 -> 0.6699 -> 0.6739 -> 0.6866 -> 0.6939`；
- 随机验证选择的 epoch 193 固定结果为 `0.6863`，低于 epoch 200 的
  `0.6939`，说明随机 val 会影响 checkpoint 选择；
- 旧实验本机可用的 6 组 best/last 已复评完成，但 v16b seed 50040149
  best/last 仍缺失，三 seed 固定协议统计尚不完整。

因此，当前最强可信参考已经从“旧配置训练期 best 0.7159”更新为
“scale04 plain KPConvX epoch 200 固定 mIoU 0.6939”。scale04 是物理尺度
校准后的新基线，不是结构创新。后续 DA-Radius、support 或 global 分支必须
在该基线上重新验证，不能用旧 0.02m 结果直接声称增益。

完整数据见
[`s3dis_fixed_protocol_results_20260714.md`](s3dis_fixed_protocol_results_20260714.md)。

路线二下一步保持原设计：训练同一 scale04 数据与优化协议下的 v17，验证
旧 0.02m 固定协议中 v17 相对 baseline 的 `+0.0130` 是否能够跨尺度保留。
成功门槛是相对 scale04 baseline `0.6939` 至少提高 `+0.005`，即达到
`0.6989`，同时 door/window/column/board/clutter 不出现明显净下降。

## 当前总判断

当前还没有一个改进在稳定性和多次实验上超过原始 KPConvX baseline。已经得到的最重要结论不是“某个参数最好”，而是逐步排除了几类看似合理、实际没有形成净收益的路线：

`真实 DA-Radius 改图 -> 简单全局上下文 -> PTv3-inspired 重型双分支 -> DA-meta 条件器 -> hard controlled support -> dual-support residual -> inference alpha 诊断`

截至 v17，证据支持以下判断：

- baseline 训练期随机裁剪 best mIoU 为 `0.7159`，但不再作为最终可信参考；
  当前固定协议最强参考是 scale04 epoch 200 的 `0.6939`。
- v13 真实 CUDA DA-Radius best 为 `0.7096`，产生过结构类正信号，但没有稳定超过 baseline。
- v13 保存代码复跑并把 `clip_grad` 放宽到 2.0 后 best 为 `0.7030`，说明单改梯度裁剪不能突破。
- v13b 更保守半径、更多候选邻居和 `clip_grad=2.0` 的组合 best 为 `0.6841`，属于策略失败。
- v14 decoder-SGCA best 为 `0.7004`，证明 decoder/head 前融合比 encoder 强融合安全，但简单全局上下文没有形成稳定收益。
- v15 stage234 边界门控双注意力 best 约 `0.5971`；边界辅助损失会下降，但主分割出现负迁移。
- v16-local 不改图、只注入 DA-meta，best 为 `0.6977`；局部统计能够被主干使用，但不足以复现 v13 的结构类收益。
- v16b controlled support 三个 seed 的 best 为 `0.7161 / 0.7032 / 0.7000`，均值约 `0.7064`；唯一超过 baseline 的结果只高 `0.0002`，属于高方差峰值，不能作为稳定提升。
- v17 dual-support best 为 `0.7084`，训练约 `44.52h`，比 baseline 慢约 `1.85x`；没有带来净收益。
- v17 inference alpha sweep 中，`alpha=0~1` 的 mIoU 差异不超过 `0.0008`。这否定了“只要把推理期 residual 调弱就能明显修复 v17”的推测。

因此，下一步不能再把已有路线换名重跑，也不能默认“选择性 support”一定正确。需要先让新的外部判断回答：alpha 几乎无效究竟表示 support 分支没有价值，还是训练期已经让主干与该分支共同适配，导致推理期开关无法分离训练历史。

## 版本时间线

| 阶段 | 版本/实验 | 核心变化 | 结果与结论 |
| --- | --- | --- | --- |
| Baseline | 原始 KPConvX | 标准 KPConvX，S3DIS Area5 | 训练期随机 best `0.7159`；固定 best `0.6556`，作为旧尺度对照 |
| Scale04 baseline | S3DIS 物理尺度校准 | 网格 0.04m，`kp_radius=kp_sigma=2.1`，其余 Pointcept 训练协议保持不变 | 固定协议 epoch 200 `0.6939`，当前新基线 |
| 早期混合 | early hybrid | DA-Kernel、DA-Radius、SGCA、refine 同时加入 | 变量过多、归因困难，转向单模块消融 |
| DA 初探 | v1-v4 | CUDA/Torch DA-Radius、不同 stage 和半径范围 | 证明动态半径路径可运行，但强度与 stage 高度敏感 |
| 模块消融 | v5-v11 | DA-kernel、DA-radius、global、influence mode 的组合与拆分 | 简单叠加模块不能形成可靠主线 |
| CUDA/Torch 对照 | v12 | Torch block mask 对照 CUDA 改图 | best `0.6815`，主要作为语义正确性参考 |
| 真实动态半径 | v13 | CUDA 重建邻域，stage 3/4，strength 0.5，block mask 关闭 | best `0.7096`；beam/board/window/sofa 有信号，door/column 等下降 |
| v13 复跑 | v13 rerun clip2 | 保存代码复跑，仅把 `clip_grad` 从 1.0 放宽到 2.0 | best `0.7030`；单改裁剪没有提升 |
| 保守动态半径 | v13b | stage 3/4，候选邻居 24，范围更窄，strength 0.35，clip 2.0 | best `0.6841`；更保守策略削弱 v13 的结构类收益 |
| 旧全局融合 | v14 encoder-SGCA | encoder stage4/5 直接改写 feature | 提前止损，best 约 `0.6001`；encoder/skip 污染明显 |
| 后置全局融合 | v14 decoder-SGCA | decoder/head 前融合，gamma 零初始化 | best `0.7004`；融合位置更安全，但上下文信息质量不足 |
| PTv3-inspired 双分支 | v15 | serialized patch/token bank + decoder fusion | 没有形成可信正收益 |
| 边界门控重型版 | v15 stage234 | stage2/3/4 global + boundary risk + boundary loss + refine | best 约 `0.5971`；boundary loss 能学，但 segmentation 负迁移 |
| 内化式初版 | v16 DA-stat GC | DA-meta shell bias + decoder GC 同时启用 | 58 epoch 提前止损，best `0.5916`；不能归因到单一模块 |
| 局部条件器 | v16-local | 不改邻域图，stage3/4 DA-meta channel bias，关闭 GC | 完整 200 epoch，best `0.6977`；统计信号有作用但不够强 |
| 候选邻域对照 | v16b superset-only | 只扩大候选邻居，不开 support mask | 用于隔离“大候选邻域本身”的影响，不作为主结果 |
| 受控支持集 | v16b controlled support | superset graph + hard effective-radius mask + DA-meta | 三 seed best 均值 `0.7064`；高方差，不稳定超过 baseline |
| 轻量全局配置 | v16c light-global | 在 controlled support 后加 decoder GC | 配置已设计，但没有找到独立完整训练结果，不能作为已验证路线 |
| 双支持残差 | v17 dual-support | original support 保底，expanded support 作为 gated residual，只开 stage4 | best `0.7084`，final `0.5861`，tail20 `0.6514±0.0345`，训练 `44.52h`；慢且未超过 baseline |
| 零训练诊断 | v17 alpha sweep | 同一 best/last checkpoint，推理期 alpha=`0/0.25/0.5/0.75/1` | best checkpoint 范围 `0.6828~0.6836`；last 范围 `0.6772~0.6779`，差异极小 |

## 关键实验数据

| 实验 | best epoch | best mIoU | final mIoU | tail20 mean/std | 训练时长 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 185 | 0.7159 | 0.6059 | 0.6407 / 0.0336 | 24.10h |
| v13 | 175 | 0.7096 | 0.6386 | 0.6498 / 0.0294 | 24.48h |
| v14 decoder-SGCA | 193 | 0.7004 | 0.6868 | 0.6543 / 0.0316 | 27.57h |
| v16-local | 156 | 0.6977 | 0.6693 | 0.6495 / 0.0287 | 31.77h |
| v16b seed 40979289 | 176 | 0.7161 | 0.6873 | 0.6521 / 0.0338 | 29.32h |
| v16b seed 50040149 | 167 | 0.7032 | 0.6642 | 0.6586 / 0.0287 | 26.55h |
| v16b seed 19095314 | 146 | 0.7000 | 0.5961 | 0.6401 / 0.0308 | 29.39h |
| v17 | 193 | 0.7084 | 0.5861 | 0.6514 / 0.0345 | 44.52h |

注意：训练期 validation 使用 `GridSample(mode="train") + SphereCrop(mode="random")`，单 epoch mIoU 含随机 crop 噪声。best 值不能脱离多 seed、tail 均值和固定协议复评单独使用。

## v17 alpha sweep 结果

### model_best（epoch 193）

| alpha | mIoU | mAcc | allAcc |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.6828 | 0.7680 | 0.8775 |
| 0.25 | 0.6833 | 0.7682 | 0.8778 |
| 0.50 | 0.6834 | 0.7683 | 0.8779 |
| 0.75 | 0.6835 | 0.7683 | 0.8780 |
| 1.00 | 0.6836 | 0.7681 | 0.8779 |

### model_last（epoch 200）

| alpha | mIoU | mAcc | allAcc |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.6772 | 0.7654 | 0.8749 |
| 0.25 | 0.6777 | 0.7658 | 0.8752 |
| 0.50 | 0.6779 | 0.7660 | 0.8753 |
| 0.75 | 0.6777 | 0.7661 | 0.8754 |
| 1.00 | 0.6775 | 0.7660 | 0.8755 |

这些绝对值不能直接与训练日志中的随机 epoch best `0.7084` 比较；alpha sweep 的价值是同一 checkpoint、同一固定 seed 下的横向差异。横向差异接近零，说明推理期缩放 expanded residual 不是主要修复旋钮。

## 已回答的问题

1. **只给 KPConvX 注入 DA-meta 能否复现 v13？**
   - 不能。v16-local 比失败重型版稳定，但 best `0.6977`，没有复现 v13。
2. **把 support 改成 hard controlled mask 能否稳定超过 baseline？**
   - 不能。v16b 三 seed best 均值约 `0.7064`，高于 baseline 的单次结果不可信。
3. **保留 original support，再把 expanded support 作为 residual 能否解决 hard mask 问题？**
   - 没有解决。v17 best `0.7084`，成本显著增加。
4. **v17 是否只是推理期 residual 太强？**
   - 当前证据不支持。alpha 从 0 调到 1 的 mIoU 变化不超过 `0.0008`。
5. **简单 decoder global context 是否已经验证？**
   - 已验证到“更安全但收益不足”。v14 decoder-SGCA 和 v16 初版都没有超过 baseline。
6. **边界辅助监督是否会自动解决全局/邻域污染？**
   - 不会。v15 boundary loss 下降，但 segmentation 变差。

## 仍未回答的问题

1. alpha 几乎无效，是因为 expanded branch 本身贡献很小，还是因为 v17 训练期间主干已经与 expanded branch 共同适配，导致推理期开关无法撤销训练历史？
2. v13、v16b、v17 的结构类正信号有多少是机制收益，有多少是 random crop、seed 和 best-of-200 选择偏差？
3. “选择性 support”是否仍值得实现？目前只有类别现象和文献动机，没有直接实验支持。
4. 下一步应先审计 KPConvX/DA-meta 主干，还是转向 compatibility-aware local aggregation、独立全局 backbone benchmark，或者停止 support 路线？
5. PTv3 的价值是否只能通过完整/近完整 backbone 体现，而不适合作为 KPConvX 的外挂 global branch？

## 明确避免重复的实验

- 不再复跑 `v13 + clip_grad=2.0`；已经有 v13 saved-code rerun。
- 不再复跑“更保守 DA-Radius + 候选邻居 24”；v13b 已失败。
- 不再把 hard support mask 换名重跑；v16b 已有三个 seed。
- 不再只给 v17 改 inference alpha 或简单 gamma；alpha sweep 已表明该旋钮影响极小。
- 不直接给 v17 增加更多 stage、更大 support 或更重 global branch。
- 不把 `boundary loss 能下降` 当作机制有效；v15 已证明两者不等价。
- 不把简单 cloud mean token / decoder GC 当成未验证新方向；v14/v16 已覆盖。

## 上一轮 GPT-5.5Pro 推测的验证状态

| 推测/决策规则 | 当前状态 | 证据 |
| --- | --- | --- |
| `alpha<1` 明显优于 1，则做 strength-constrained v17 | 未触发 | best checkpoint 最大差仅 0.0008，且 alpha=1 略高 |
| `alpha=0` 最好，则 expanded support 为负贡献 | 未触发 | best 上 alpha=0 最低；last 上 alpha=0.5 略高，但差异仅 0.0007 |
| 所有 alpha 差不多，则问题不只是强度，应停止直接扩 support 并审计主干 | 已触发 | 两个 checkpoint 都对 alpha 不敏感 |
| v17 慢主要来自双路径聚合和 candidate support | 已支持 | 训练时长约 baseline 的 1.85x，代码存在两次聚合 |
| v17 后期不是小残差 | 已支持 | gamma 约 0.706、gate_mean 约 0.508、residual_ratio 约 0.13 |
| 训练一个 gamma-cap v17 | 不应直接执行 | 该实验原本以 `alpha<1` 明显更好为前提，前提没有成立 |
| 做 boundary/uncertainty selective support | 尚未验证 | 有文献和类别现象动机，但 alpha 结果没有直接支持它 |

## 当前决策点

上一轮 Pro 的第一阶段诊断已经完成，结果落入“所有 alpha 基本相同”分支。按照它原来的决策规则，不应直接训练 strength-constrained v17。现在真正需要外部模型判断的是：

- 是否应彻底暂停 support 扩展路线；
- 是否需要先做一个更便宜的训练历史/固定验证诊断；
- 如果仍做选择性邻域，怎样保证它不是 v16b/v17 的换名重复；
- 如果转向全局机制，应该先跑 PTv3 backbone benchmark，还是设计与 KPConvX 内部一致的 context operator；
- 在单张 4090D、导师催进度的条件下，哪个单一实验信息增益最高、最可能提升 mIoU。

## 一句话总结

路线已经从“继续堆模块”推进到一个明确的否定性节点：DA-meta、hard controlled support 和 dual-support residual 都没有稳定超过 baseline，v17 推理期 alpha 也不是有效修复旋钮。下一步必须基于这一完整证据重新选方向，而不是继续沿旧 support 路线做低信息量调参。
