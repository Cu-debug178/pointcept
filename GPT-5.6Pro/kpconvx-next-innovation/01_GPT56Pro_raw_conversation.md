```md
问题
--

我正在做 Pointcept/KPConvX 在 S3DIS Area5 上的语义分割改进。附件给出了从 baseline 到 v17 的完整实验路线、v16b 三 seed、v17 类别结果、监控项和刚完成的 inference alpha sweep。

请不要预设我要继续 support、边界或全局分支，也不要把已有实验换一个名字再推荐。请基于证据独立判断下一步最值得做什么。

刚完成的关键诊断是：对 v17 的 `model_best(epoch 193)` 和 `model_last(epoch 200)`，固定验证 seed，分别设置 expanded residual 的 `alpha=0/0.25/0.5/0.75/1.0`。best checkpoint 的 mIoU 仅从 `0.6828` 变化到 `0.6836`，last checkpoint 仅从 `0.6772` 变化到 `0.6779`。也就是说 alpha 几乎不影响结果。

你上一轮给出的决策规则是：

*   alpha<1 明显更好，才做 strength-constrained v17；
*   alpha=0 最好，说明 expanded support 在推理时有害；
*   所有 alpha 都差不多，则问题不只是分支强度，应停止直接扩 support，并转向 backbone/DA-meta 稳定性审计。

现在结果落入第三种。与此同时，DeepSearch 根据论文仍建议 boundary/compatibility selective support。两者存在冲突。

请回答：

1.  上一轮关于 v17 的推测哪些已经被验证、证伪或尚未验证？尤其请区分“推理期分支贡献”和“训练期共同适配造成的不可逆影响”。
2.  alpha 几乎无效，是否足以暂停整个 support 路线？还是只能否定 inference strength tuning？请给出严格因果边界。
3.  DeepSearch 推荐 selective support 是否仍有直接实验依据，还是目前只有文献动机？是否值得再花一次约 30-45 小时训练？
4.  请检查完整路线，列出会与 v13、v13b、v16-local、v16b、v17 重复的实验，禁止推荐这些重复项。
5.  在以下候选中独立排序，不要求一定选其中之一：
    *   停止 support 路线，审计 KPConvX/DA-meta 主干；
    *   先修正随机 validation，做固定协议或多 crop 复评；
    *   compatibility/same-different neighbor aggregation；
    *   boundary/uncertainty selective support；
    *   真实 DA-Radius 的 point-wise 选择机制；
    *   PTv3 完整/近完整 backbone benchmark；
    *   与 KPConvX 内部一致的轻量上下文机制；
    *   放弃当前改造，选择新的总体框架。
6.  最终只推荐一个下一步主实验和一个零训练/低成本前置诊断。不要给三个并行大实验。
7.  对推荐主实验给出：明确假设、与已有实验的非重复点、最小代码改动、训练成本、必须监控的指标、成功/失败阈值和止损 epoch。
8.  目标是提高可信 mIoU，不是制造更高的 random-crop best。请说明如何控制 best-of-200、seed 和 checkpoint 选择偏差。
9.  当前单卡是 4090D，v17 训练约 44.52 小时。请把工程成本和导师催进度纳入决策。

请用批判性视角。如果现有证据不足以支持新的机制训练，请直接说应该先做什么诊断；如果你认为 support 路线应该停止，也请明确说明。

回答
--

根据 2026-07-10 的完整路线文档和 alpha sweep，我的裁决是：**现在应暂停 support 路线作为下一次 30–45 小时主训练方向；但不要把 alpha sweep 解释成“所有 support 机制都被因果证伪”。它直接否定的是 v17 的推理期强度调节，而不是训练期共同适配或尚未实现的语义选择机制。** 当前最值得做的主实验不是新模块训练，而是**固定协议 / 多 crop 复评现有 checkpoint**，先把可信 mIoU 排名锁住。继续烧一次 v17b / selective support 训练，现在证据不够。

1\. 上一轮关于 v17 的推测：已验证、证伪、未验证
----------------------------

**已验证 / 强支持：**

*   **v17 不能算成功。** baseline best 是 `0.7159`，v17 best 是 `0.7084`，final 只有 `0.5861`，训练约 `44.52h`，接近 baseline `24.10h` 的 1.85 倍；v16b 三 seed 均值也只有约 `0.7064`，唯一超过 baseline 的 seed 只高 `0.0002`，不能作为稳定提升。(01\_完整实验路线总结.md)
*   **v17 慢主要来自 dual-support / candidate support。** v17 配置保留 original KPConvX support，又在 stage4 加 expanded candidate graph 的 gated residual；这不是简单 head 小改，而是额外邻域聚合路径。(07\_v17\_config.py)
*   **“所有 alpha 差不多 ⇒ 问题不只是推理期强度”这个分支已触发。** model\_best 的 `alpha=0~1` mIoU 只在 `0.6828~0.6836` 间变化，model\_last 只在 `0.6772~0.6779` 间变化；同一 checkpoint、同一固定 seed 下横向差异接近 0。(01\_完整实验路线总结.md)
*   **v17 后期 residual 不是没开。** 监控项显示 gamma、gate、residual\_ratio 后期都有实际幅度，所以“alpha 无效”不能简单解释为“分支一直是零”。上一轮验证表里记录 gamma 约 `0.706`、gate\_mean 约 `0.508`、residual\_ratio 约 `0.13`。(02\_上一轮Pro推测验证状态.md)

**被证伪 / 不再成立：**

*   **“v17 只是推理期 residual 太强，调小 alpha 就能修好”被否定。** 如果这个假设成立，`alpha<1` 应该有可见提升；实际 best checkpoint 反而 `alpha=1` 略高，但差距只有 `0.0008`，可以视作无效。(01\_完整实验路线总结.md)
*   **“alpha=0 最好 ⇒ expanded branch 推理期是负贡献”没有触发。** best 上 `alpha=0` 最低，last 上 `alpha=0.5` 略高，整体都在噪声量级内。不能说 expanded residual 推理期明显有害。
*   **strength-constrained v17 / gamma-cap v17 不该直接做。** 它原来的前提是 `alpha<1` 明显更好；现在前提没成立。上一轮状态表也明确把该建议标成“原条件不成立”。(02\_上一轮Pro推测验证状态.md)

**尚未验证：**

*   **训练期共同适配是否造成不可逆影响。** alpha sweep 比较的是同一组 v17 权重 `θ_v17` 下的 `f(x; θ_v17, alpha)`，不是比较“从未训练 dual-support 的 θ”与“训练过 dual-support 的 θ”。所以 `alpha=0` 不等价于 baseline，也不等价于 v16-local。附件里也明确指出，v17 的 base path 参数已经在 dual-support 训练中更新，推理时关分支不能恢复“从未启用 dual-support”的模型。(09\_DeepSearch与Pro冲突说明.md)
*   **boundary / compatibility selective support 是否有效。** 目前只有文献动机和类别现象，没有项目内直接训练证据；alpha 结果也没有确认它。(01\_完整实验路线总结.md)
*   **v13 / v16b / v17 的结构类正信号有多少是真机制、有多少是 random crop、seed、best-of-200 偏差。** 路线文档明确提醒训练期 validation 使用 `GridSample(mode="train") + SphereCrop(mode="random")`，单 epoch mIoU 含随机 crop 噪声，best 不能脱离多 seed、tail 和固定协议复评使用。(01\_完整实验路线总结.md)

2\. alpha 几乎无效的严格因果边界
---------------------

**它足以否定：**

*   inference-only alpha tuning；
*   简单把 v17 residual 乘小、gamma-cap、延长 warmup、改 eval alpha 这类“强度调参”；
*   “v17 明显有一个推理期最优 residual 强度，只是没调到”的解释。

**它不足以严格否定：**

*   support 在训练期改变了主干参数；
*   从头训练一个完全不同的 selective support 机制；
*   使用语义兼容性、预测一致性或边界风险来选择邻居的机制。

更严格地说，alpha sweep 估计的是：

    text固定 θ_v17 时，mIoU 对 expanded residual 推理强度 alpha 的局部/横向敏感性

它没有估计：

    text训练得到 θ_v17 的过程是否已经被 expanded support 改写

所以，**科学结论是：只能直接否定 inference strength tuning。工程结论是：结合 v13、v13b、v16-local、v16b、v17 的累计失败和 v17 44.52 小时成本，应该暂停 support 路线作为下一次主训练。** 这不是逻辑上“support 全灭”，而是资源约束下的止损。

3\. DeepSearch 的 selective support：现在只有文献动机，不值得立刻再训 30–45 小时
------------------------------------------------------------

DeepSearch 的想法并非荒谬：v17 类别现象确实像“扩展邻域对部分结构 / 家具有益，对门窗墙柱等边界或附着类有污染”。冲突说明里也写到，两边共同承认 v17 有条件性类别变化，DeepSearch 基于 BoundaryAwareGEM、CBL、CGA-Net 等文献推荐 boundary/compatibility selective support。(09\_DeepSearch与Pro冲突说明.md)

但项目内证据还不够：

*   v15 已经证明“boundary loss 能下降”不等于主分割提升，stage234 边界门控重型版 best 只有约 `0.5971`，属于明显负迁移。(01\_完整实验路线总结.md)
*   v16b hard support 三 seed均值不超过 baseline；v17 把 original support 保底再加 expanded residual，仍低于 baseline且更慢。(01\_完整实验路线总结.md)
*   alpha sweep 没有显示 expanded residual 在推理期有可控强正/负贡献。(01\_完整实验路线总结.md)

所以我的判断是：**selective support 现在只有“文献 + 现象”动机，没有足够的项目内直接依据，不值得马上花 30–45 小时训练。** 它可以保留为未来候选，但必须先通过一个零训练的“邻域污染 / 兼容性可分性诊断”。

4\. 与已有实验重复、禁止再推荐的项
-------------------

这些我会视为重复项，不建议再做：

重复项

已覆盖实验

为什么重复

`v13 + clip_grad=2.0` 或保存代码复跑

v13 rerun

已经证明单改梯度裁剪不能突破。(01\_完整实验路线总结.md)

更保守 DA-Radius、候选邻居 24、窄 range、strength 0.35

v13b

已失败，best `0.6841`。(01\_完整实验路线总结.md)

真实 DA-Radius stage3/4 的另一个半径换皮版本

v13 / v13b

仍是“改邻域图 + 半径强度”的同一因果变量。

只注入 DA-meta channel bias / 局部统计条件器

v16-local

已做完整 200 epoch，best `0.6977`。(01\_完整实验路线总结.md)

superset graph + hard effective-radius mask + DA-meta

v16b

三 seed 已有，均值约 `0.7064`，不稳定。(01\_完整实验路线总结.md)

只扩大 candidate 邻域但不开有效选择

v16b superset-only / v16b 路线

不能回答新问题，只会增加邻域变量噪声。

original support 保底 + expanded residual + gate/gamma/warmup

v17

已做，慢且不超过 baseline。(07\_v17\_config.py)

v17 inference alpha、简单 gamma-cap、单纯 residual 强度调小

alpha sweep

alpha 横向差异小于 `0.001`。(01\_完整实验路线总结.md)

给 v17 加更多 stage、更大 support、更大 range

v17 的直接扩展

与当前否定性节点相反，只会放大成本。

boundary auxiliary loss / boundary risk gate / refine 的重型组合

v15 stage234

boundary loss 能学但 segmentation 负迁移。(01\_完整实验路线总结.md)

简单 cloud mean token、decoder GC、后置 SGCA 换名重做

v14 decoder-SGCA / v16 DA-stat GC

已验证“更安全但收益不足”，简单全局上下文不是未验证空白。(01\_完整实验路线总结.md)

需要特别强调：**compatibility / same-different neighbor aggregation 不完全等同于 v16b/v17，只有在它真的使用“语义兼容性”作为新因果变量时才算新实验。** 但如果只是“expanded support + 一个 gate”，那就是 v17 换名。

5\. 候选方向独立排序
------------

排名

候选

判断

1

**先修正随机 validation，做固定协议或多 crop 复评**

必须先做。当前 best-of-200 + random crop 会污染所有结论；不先锁定可信 mIoU，任何新训练都可能是在追随机峰值。

2

**停止 support 路线，审计 KPConvX/DA-meta 主干**

作为决策方向排第二。不是马上再训 DA-meta，而是审计数据流、val 协议、checkpoint 选择、DA-meta 是否引入不稳定。

3

**与 KPConvX 内部一致的轻量上下文机制**

比继续 support 更有前途，但不能是 v14/v16 的 cloud mean / decoder GC 换名。应放在固定复评之后。

4

**PTv3 完整/近完整 backbone benchmark**

有战略价值，能判断是否应换总框架；但它更像 benchmark / pivot，不是 KPConvX 小步改进。导师催进度时，可作为后续对照，不应压过固定复评。

5

**compatibility/same-different neighbor aggregation**

概念上比 boundary selective support 更干净，因为它直接针对跨类邻居污染；但目前没有项目内直接证据，必须先做邻域兼容性诊断。

6

**boundary/uncertainty selective support**

文献动机有，但项目内最近邻证据不足，而且 v15 的 boundary 辅助路线有负迁移先例。不是下一次主训练。

7

**真实 DA-Radius point-wise 选择机制**

高概率重复 v13/v13b 的变量空间，只是把半径选择再复杂化。当前不推荐。

8

**放弃当前改造，选择新的总体框架**

现在下这个结论过早。先固定复评；如果复评后 baseline 仍稳压全部改造，再考虑 PTv3 或新框架。

6\. 最终只推荐一个主实验 + 一个零训练前置诊断
--------------------------

### 零训练 / 低成本前置诊断：expanded-neighbor contamination / compatibility audit

**目的：** 判断 DeepSearch 的 selective support 有没有项目内直接证据。

做法很简单：在 v17 或 v16b 的 stage4 上，导出 original support 与 expanded-only support 的邻居统计，不训练。对 Area5 val 点计算：

*   expanded-only neighbor 的 GT same-class rate；
*   expanded-only neighbor 的 model-pred same-class rate；
*   original support vs expanded-only support 的 cross-class rate 差值；
*   按类别分组：`beam/board/table/chair/sofa` 与 `door/window/column/wall/clutter`；
*   按边界 / 非边界点分组；
*   用当前模型 entropy 或 prediction margin 看能不能预测“expanded neighbor 是否风险邻居”。

**判据：**

*   如果正信号类的 expanded-only same-class rate 明显高，而风险类 / 边界点明显低，并且 entropy / prediction disagreement 能以 AUC ≥ `0.70` 或 Spearman |ρ| ≥ `0.5` 预测污染，那么 selective support 才有项目内证据。
*   如果 same-class rate、边界污染、uncertainty 与 v17 类别得失没有稳定关系，DeepSearch 路线应继续搁置。

**这一步不使用 GT 做最终模型，只用 GT 判断污染是否真实存在。** 成本应该远低于一次训练，主要是一次 eval forward 加邻居索引统计。

### 下一步主实验：固定协议 / 多 crop 复评现有 checkpoint

我建议把这一步当成主实验，而不是再训练一个新模块。原因是现有证据还不足以支持新的 30–45 小时机制训练。

7\. 推荐主实验细节：固定协议 / 多 crop checkpoint re-ranking
-----------------------------------------------

**明确假设**

当前实验排序被 `random SphereCrop + best-of-200 + 单 seed checkpoint 选择` 放大了随机峰值。固定协议复评后，真正可信的 mIoU 排名可能不同；如果没有任何 support 变体在固定协议下稳定超过 baseline，就应停止 support 训练路线。

**与已有实验的非重复点**

它不改 backbone、不改 support、不改 DA-meta、不改 global。它检验的是**评价协议与 checkpoint 选择偏差**，这是目前路线中尚未被彻底解决的核心不确定性。路线文档已经明确警告当前训练期 validation 使用随机 crop，单 epoch mIoU 含随机噪声，best 不能脱离固定协议复评使用。(01\_完整实验路线总结.md)

**最小代码改动**

*   新建或改造一个 `eval_fixed_area5.py`，优先复用 Pointcept 的 `SemSegTester` / precise eval，而不是当前训练期随机 val。
*   关闭 random `SphereCrop(mode="random")`；使用 deterministic grid / full-scene / multi-crop vote。
*   固定 dataloader seed，但最终指标不能依赖单个 random crop seed。
*   对每个 checkpoint 输出：mIoU、mAcc、allAcc、13 类 IoU、每个 room 的 IoU、耗时。
*   最小 checkpoint 集合：  
    `baseline model_best/model_last`，`v13 model_best/model_last`，`v16-local model_best/model_last`，`v16b 三 seed model_best/model_last`，`v17 model_best/model_last`。如果时间紧，先评 `baseline`、v16b 三 seed、v17；v13/v16-local 随后补齐。

**训练成本**

*   训练成本：`0`。
*   GPU 成本：若按现有 precise eval 量级估计，多个 checkpoint 可能是数小时到一天内的 4090D 推理成本；远低于一次 v17 训练的 `44.52h`。(01\_完整实验路线总结.md)
*   这对导师汇报更有价值：你能说明“我没有继续烧 45 小时做换名 support，而是先把可信排名锁住”。

**必须监控的指标**

*   固定协议 mIoU / mAcc / allAcc；
*   每类 IoU，尤其 `door/window/column/wall/clutter` 与 `beam/board/table/chair/sofa`；
*   每 room IoU，防止某几个 room 偶然拉高；
*   `model_best` vs `model_last` gap；
*   多 crop / 多 deterministic seed 的均值与标准差；
*   推理耗时，尤其 v17 相对 baseline 的成本比。

**成功阈值**

一个改造要被称为“可信提升”，至少满足：

*   固定协议 mIoU 超过 baseline 固定协议结果 ≥ `+0.005`；更稳妥是 ≥ `+0.008`；
*   提升不能只来自 `beam` 这类高波动 rare class；
*   `door/window/column/wall/clutter` 不出现系统性大跌；我会把任一主风险类跌幅超过 `0.03` 视为强警告；
*   `model_last` 或 last-k checkpoint 平均不能明显崩，不能只靠单个 `model_best` 峰值；
*   多 seed 情况下看均值，不接受“一个 seed 高 0.0002”的结论。

**失败阈值**

*   如果 v16b 三 seed固定复评均值仍低于 baseline，且 v17 仍低于 baseline或只在 rare class 上补偿，那么 support 路线停止。
*   如果所有改造与 baseline 差距都落在固定复评噪声内，例如 ±`0.003`，则结论是“没有可信提升”，不能继续包装成新机制成功。
*   如果 v17 固定复评低于 baseline但耗时仍接近 1.8×，直接淘汰 v17 作为主线。

**止损 epoch**

不适用，因为这个主实验不训练。它的止损规则是：**不要在固定协议复评完成前启动任何新的 30–45 小时 support 训练。** 如果必须设置“运行止损”，可以先评 baseline、v16b 三 seed、v17 best/last；若这些已显示 support 变体固定 mIoU均值低于 baseline ≥ `0.005`，就不用再为 support 路线追加新训练，只补 v13/v16-local 作为归档对照。

8\. 如何控制 best-of-200、seed、checkpoint 选择偏差
-----------------------------------------

我建议以后把结果分成两层：

**开发层：** 训练日志里的 random-crop val 只用于观察训练是否崩，不用于宣布 SOTA 或选择论文主结果。

**报告层：** 用固定协议复评输出可信 mIoU。

具体规则：

*   **checkpoint 预注册。** 每个模型只报告 `last`、`best_by_training_val`、以及可选的 `last-k average`；不能在 200 个 epoch 里事后挑一个最漂亮的 random-crop best 作为主结论。
*   **主表以固定协议复评为准。** 训练期 `best mIoU` 只放附录或诊断表。
*   **多 seed 均值优先。** 单 seed 只允许叫 exploratory。v16b 已经证明单 seed `0.7161` 很容易误导，因为三 seed 均值约 `0.7064`。(01\_完整实验路线总结.md)
*   **对每个模型固定相同 eval protocol。** 同样 crop 数、同样 voting、同样 Area5 room 顺序、同样 checkpoint 类型。
*   **不要用 Area5 固定复评反复调参。** 固定复评一旦被用作模型选择，也会变成新的 overfit 通道。新机制最多用一次预注册比较；不合格就止损。
*   **主结论用 effect size，不用名义 best。** 低于 `+0.005` 的 mIoU 差距，在当前随机 crop / seed 背景下不应称为改进。

9\. 工程成本与导师进度压力下的实际建议
---------------------

在单张 4090D 上，v17 已经花了 `44.52h`，但没有超过 baseline；继续做 boundary/compatibility selective support，大概率又是 30–45 小时，而且还可能因为实现复杂度引入新 bug。当前最强的工程策略不是“再赌一个机制”，而是：

**先用 0 训练成本把可信排名固定下来；同时用一个低成本邻域污染诊断判断 selective support 是否有项目内证据。**

我的最终结论很明确：

> **暂停 support 路线。不要现在训练 boundary/compatibility selective support。下一步主实验做固定协议 / 多 crop 复评；前置诊断做 expanded-neighbor contamination / compatibility audit。**

如果固定复评后 baseline 仍稳压全部改造，就可以向导师清楚汇报：support、DA-meta、dual-support residual 已经被系统排除，下一阶段应转向非 support 的 backbone 级审计或独立 PTv3 benchmark，而不是继续把旧实验换名字。

问题
--

2503.16429v1.pdf

Adaptive local neighborhood search and dual attention convolution network.pdf

Hypergraph convolutional network based weakly supervised point cloud.pdf

kP-kpconv.pdf

Thomas\_KPConv\_Flexible\_and\_Deformable\_Convolution\_for\_Point\_Clouds\_ICCV\_2019\_paper.pdf

Thomas\_KPConvX\_Modernizing\_Kernel\_Point\_Convolution\_with\_Kernel\_Attention\_CVPR\_2024\_paper.pdf

Wu\_Point\_Transformer\_V3\_Simpler\_Faster\_Stronger\_CVPR\_2024\_paper.pdf

我正在研究 Pointcept/KPConvX 在 S3DIS Area5 上的语义分割改进。附件包含我看过的主要论文、完整实验进展和论文-项目对照材料。

请新建独立判断，不要延续我过去对 support、双注意力、边界污染或 PTv3 global branch 的预设，也不要为了迎合当前路线而寻找理由。

当前关键事实：

1.  单卡 RTX 4090D 24GB，必须考虑训练时间和显存。
2.  当前主干是 Apple 官方 KPConvX Pointcept wrapper。
3.  训练期 random-crop best：baseline=0.7159，v17=0.7084，但该指标有明显 best-of-200 偏差。
4.  固定 Area5 全场景 identity 复评：baseline=0.6556，v17=0.6685，v17 提升约+0.0130。
5.  已完成 DA-Radius、SGCA、serialized global token、boundary gate、DA-meta、hard controlled support、dual-support residual 和 inference alpha sweep。
6.  当前正在训练0.04m物理尺度校准 baseline，尚无结果。
7.  目标是获得可信、可复现并适合论文表述的 mIoU 提升，不是制造更高的随机峰值。

请完成以下任务：

1.  对所有附件论文做机制级比较，不要只做摘要。
2.  判断哪些论文优点能真正内化进 KPConvX，而不是生硬拼接模块。
3.  最多提出3个创新候选，按科学新颖性、预期收益、工程风险和4090D成本排序。
4.  明确每个候选与我已完成实验的区别，禁止把旧实验换名推荐。
5.  给出首选方案的数据流、模块接口、训练损失、初始化方式和消融顺序。
6.  判断首选方案应继续改KPConvX、使用预训练、引入超图/状态记忆，还是转向独立backbone。
7.  指出最可能失败的原因、受益/受损类别，以及最低成本的Go/No-Go实验。
8.  最终只选择一个主创新方向和一个备用方向。

回答
--

根据 2026-07-10 的项目进展文档，我会把当前局面重新判为：**不是“support 全失败”，也不是“v17 已成功”；而是 baseline 的评价协议与物理尺度都存在强混杂。下一步主创新不应继续扩大 support，也不应把 PTv3 global branch 外挂到 KPConvX 上，而应沿 KPConvX 内部几何算子做“物理尺度一致的 kernel attention 校准”。** 当前最关键事实是：random-crop best 中 v17 低于 baseline，但固定 Area5 全场景 identity 复评中 v17 反而从 `0.6556` 提到 `0.6685`，同时项目已发现当前 Pointcept S3DIS baseline 实际沿用 ScanNet 风格 `0.02m` 尺度，而官方 S3DIS 对应第一层物理半径约为 `0.084m`。这意味着过去 DA/support 的部分正信号可能是在补偿过小物理感受野，而不是证明“扩大 support”本身是正确方向。(2026-06-20汇报后KPConvX项目进展总结.md) (2026-06-20汇报后KPConvX项目进展总结.md)

1\. 附件论文的机制级比较
--------------

论文 / 材料

真正机制

对 KPConvX 的启发

能否内化进 KPConvX

**KPConv 2019**

用欧氏空间 kernel points 定义显式几何卷积，偏好 radius neighborhood + grid subsampling 以处理非均匀密度；deformable KPConv 用可学习 kernel offset 适应局部几何，但需要 fitting / repulsive regularization 防止 kernel 漂空。(Thomas\_KPConv\_Flexible\_and\_Deformable\_Convolution\_for\_Point\_Clouds\_ICCV\_2019\_paper.pdf) (Thomas\_KPConv\_Flexible\_and\_Deformable\_Convolution\_for\_Point\_Clouds\_ICCV\_2019\_paper.pdf)

你的问题首先应从“物理半径、grid size、kernel occupancy”查起，而不是先加新分支。

**强可内化。** KPConvX 本来就是这个体系；最自然的改法是校准半径、kernel attention、kernel occupancy，而不是外挂图网络。

**KPConvX 2024**

把 KPConv 现代化为 depthwise KPConvD，再用中心点特征生成 kernel-point group attention；官方 ablation 中新 architecture、inverted bottleneck、nearest influence、shareKP、attention 都是围绕 kernel 几何算子本体优化，而不是外接大模块。KPConvD-L 与 KPConvX-L 在 S3DIS Area5 10-run 均值上差距很小，但 KPConvD 更快。(Thomas\_KPConvX\_Modernizing\_Kernel\_Point\_Convolution\_with\_Kernel\_Attention\_CVPR\_2024\_paper.pdf) (Thomas\_KPConvX\_Modernizing\_Kernel\_Point\_Convolution\_with\_Kernel\_Attention\_CVPR\_2024\_paper.pdf)

论文真正的路线是“轻量几何算子 + 深网络 + kernel attention”，不是复杂 global / boundary / support 堆叠。

**最强可内化。** 你的创新应作用在 KPConvX 的 kernel attention / kernel projection / physical scale 上。

**PTv3**

核心不是发明更复杂 attention，而是用 serialization 把 receptive field 扩到 1024，同时降低内存和速度成本；它是 backbone 级设计哲学，不是一个可以随手贴到 KPConvX decoder 的 global token。(Wu\_Point\_Transformer\_V3\_Simpler\_Faster\_Stronger\_CVPR\_2024\_paper.pdf)

如果 KPConvX 走不通，PTv3 是独立 backbone benchmark / pivot；但把 PTv3 简化成 KPConvX global branch 会丢掉论文核心。

**不适合内化为小模块。** 适合作为备用独立 backbone，不适合作为 KPConvX 外挂分支。

**Sonata**

解决 3D SSL 的 geometric shortcut：把 SSL loss 放在更粗尺度、扰动 masked points 的空间信息、加强对输入特征的依赖，并用大规模自蒸馏形成可靠表征；它依赖大规模预训练和 PTv3/encoder-only 表征。(2503.16429v1.pdf) (2503.16429v1.pdf)

它告诉你“可靠表征”比随机峰值更重要，也支持用 linear/decoder probing 评估表征质量；但不是 4090D 上从头训练一个 SSL 系统。

**不建议直接内化。** 除非使用已有预训练权重，否则不应作为当前主创新。

**ADNet / ALNS + DCAA/GSA/SPP**

用 adaptive local neighborhood search 处理密度变化，再用 DCAA、GSA、SPP-IPCs 融合局部细节、全局上下文和空间结构；ablation 显示 ALNS 相比 KNN/ball/octree 有收益，DCAA/GSA/SPP 也各自贡献 mIoU。(Adaptive local neighborhood search and dual attention convolution network.pdf) (Adaptive local neighborhood search and dual attention convolution network.pdf)

有价值的是“密度/局部几何状态应进入特征聚合”，但你已经多次做过 support / radius / DA-meta，因此不能再把 ALNS 换名重做。

**部分可内化。** 只应吸收“density / occupancy 作为 kernel attention 条件”，不要再改邻域搜索。

**APGCM + historical state increments / Mamba**

用 PGFE 建局部图特征，用 MDRA 做低秩重构和点/通道注意力，再用 Mamba 预测 node state increment；它关注跨层状态变化，而非单层邻域扩张。(1-s2.0-S003132032501266X-main.pdf) (1-s2.0-S003132032501266X-main.pdf)

可借鉴“block-to-block feature increment”作为轻量状态记忆，但完整 T-Net + KNN + MDRA + Mamba 不适合硬贴到 KPConvX。

**弱到中等可内化。** 适合作为备用轻量 state adapter，不适合作为首选主实验。

**WHCN hypergraph weak supervision**

从 scene-level label 经 CAM 得到 superpoint seeds，再用 weighted hypergraph convolution 做 label propagation；核心是弱监督伪标签传播，不是全监督 backbone 改进。(Hypergraph convolutional network based weakly supervised point cloud.pdf) (Hypergraph convolutional network based weakly supervised point cloud.pdf)

“superpoint / high-order consistency”可用于诊断或后处理，但你有 point-level GT，主瓶颈不在弱监督 label propagation。

**不建议内化为主干。** 引入超图会改变训练问题，工程复杂且论文叙事偏题。

**FA-KPConv**

通过 frame averaging 给 KPConv 引入精确欧氏不变/等变性，不增加可学习参数；主要验证在分类、配准和旋转鲁棒场景。(kP-kpconv.pdf) (kP-kpconv.pdf)

对 S3DIS 语义分割要谨慎：floor/wall/ceiling 等类别依赖重力方向，完整 SO(3) 不变性可能伤害语义。

**不适合作为主方向。** 最多做 z-yaw TTA / consistency 诊断，不能上完整 E(3) frame averaging。

我的独立判断是：**真正能优雅内化进 KPConvX 的，不是 hypergraph、完整 Mamba、PTv3 global branch、ADNet-style ALNS，也不是继续 dual-support；而是“利用 KPConvX 已有 kernel projection / kernel attention，对物理尺度和 kernel occupancy 进行条件化校准”。**

2\. 最多 3 个创新候选排序
----------------

排名

候选

科学新颖性

预期收益

工程风险

4090D 成本

与已做实验的区别

**1**

**Scale-calibrated Occupancy-aware Kernel Attention，简称 SOKA**

中高。把 ADNet 的密度适应思想内化到 KPConvX kernel-point attention，而不是改 support。

中等，目标固定协议 `+0.5~1.5 mIoU`；若 scale04 baseline 证明物理尺度重要，收益更合理。

低到中。只改 KPConvX block 内 attention logit，不新增邻域聚合。

约 baseline `1.05~1.15x`，远低于 v17 `1.85x`。

不改半径、不扩邻域、不做 dual-support、不做 DA-meta channel bias；使用已有 kernel assignment 的 occupancy / influence 分布调制 kernel attention。

**2**

**KPConvX block-level feature-increment state adapter**

中。借鉴 APGCM/Mamba 的 state increment，但不用 Mamba 序列模型。

小到中，可能提升 checkpoint 稳定性而非大幅提升 best。

中。跨 block 状态如果设计不好，可能只是另一种 residual 噪声。

约 baseline `1.05x`。

未做过；不同于 v14 global、v16 DA-meta、v17 support。它调的是同 stage 内 block-to-block feature delta。

**3**

**PTv3 / Sonata 路线作为独立 backbone benchmark 或 pivot**

对你项目的新颖性低，但科学上是强基线。

可能高，但不保证 4090D 训练成本可控；若有官方预训练权重，收益更现实。

中高。会从 KPConvX 改造变成 backbone 切换。

取决于配置；不是免费小改。

不是把 PTv3 global 分支贴到 KPConvX，而是诚实地做独立 backbone 对照。

**首选是候选 1：SOKA。备用是候选 3：PTv3 / Sonata 独立 backbone benchmark。** 候选 2 可以保留为论文 discussion 或后续轻量扩展，但我不建议把它作为当前备用主线，因为 APGCM 论文中 state prediction 的单独增益并不强，且你的项目当前最强证据指向“物理尺度 / kernel 几何校准”，不是“跨层记忆不足”。(1-s2.0-S003132032501266X-main.pdf)

3\. 为什么首选不是继续 support、边界、超图、预训练或新框架
-----------------------------------

**不继续 support：** v17 固定协议 `+0.013` 说明原先不能把 support 信号全盘否定，但 v17 成本约 `44.52h`，训练接近 baseline 的 `1.85x`；alpha sweep 又证明推理期 residual strength 不是有效旋钮。更重要的是，当前 baseline 的第一层物理半径可能偏小，support 正信号可能是尺度补偿。(2026-06-20汇报后KPConvX项目进展总结.md) (2026-06-20汇报后KPConvX项目进展总结.md)

**不继续 boundary：** 你已经做过 boundary gate，且主分割负迁移明显。ADNet/边界类论文只能给“局部结构需要更好条件化”的动机，不能证明再做 boundary loss 有项目内证据。

**不引入超图：** WHCN 的强项是弱监督 scene-level 到 point-level pseudo label propagation；你的任务是全监督 Area5，超图会把问题改成另一个研究方向。(Hypergraph convolutional network based weakly supervised point cloud.pdf)

**不把 PTv3 当外挂 global branch：** PTv3 的核心是 serialization-based backbone scaling，而不是一个 decoder global token。要用就作为独立 backbone，不要拆成一个“global 分支”贴到 KPConvX。(Wu\_Point\_Transformer\_V3\_Simpler\_Faster\_Stronger\_CVPR\_2024\_paper.pdf)

**不把 Sonata 作为当前主线：** Sonata 的价值是大规模可靠预训练；单卡 4090D 上从头复现其 140k point clouds 自蒸馏路线不现实。若有可用预训练权重，可以作为备用 backbone，而不是 KPConvX 小改。(2503.16429v1.pdf)

4\. 首选方案：SOKA
-------------

### 4.1 明确假设

**假设：** 当前 KPConvX 在 S3DIS 上的关键不稳定来源不是“缺一个更大 support 分支”，而是**同一组 kernel points 在不同物理尺度、不同局部密度、不同 kernel occupancy 下被同样使用**。v17 固定协议提升可能来自扩大邻域对过小物理半径的补偿；如果把“邻域是否足够、kernel points 是否空置、不同 shell 的覆盖是否偏斜”直接输入 KPConvX kernel attention，就能获得类似补偿，但避免 v17 的双路径聚合成本和 support 污染。

这与 KPConv/KPConvX 的机制是一致的：KPConv 的信息单元是 kernel point 对空间邻居的几何响应，KPConvX 的新增能力是根据中心特征产生 kernel attention；SOKA 只是把“当前 kernel 实际看到的几何占用状态”作为 attention 条件，而不是新建图或新建分支。(Thomas\_KPConv\_Flexible\_and\_Deformable\_Convolution\_for\_Point\_Clouds\_ICCV\_2019\_paper.pdf) (Thomas\_KPConvX\_Modernizing\_Kernel\_Point\_Convolution\_with\_Kernel\_Attention\_CVPR\_2024\_paper.pdf)

### 4.2 数据流

现有 KPConvX block 大致是：

    textquery points / support points / neighbor indices
            ↓
    kernel projection: neighbor → nearest kernel point / influence
            ↓
    central feature → base kernel group attention
            ↓
    depthwise kernel aggregation
            ↓
    residual block output

SOKA 改成：

    textquery/support/neighbor indices
            ↓
    existing kernel projection，不新增邻域
            ↓
    per-query per-kernel occupancy descriptor
            ↓
    tiny geometry MLP → additive kernel-attention bias
            ↓
    base KPConvX kernel attention + β · occupancy bias
            ↓
    single-path KPConvX aggregation
            ↓
    residual output

每个 query point、每个 kernel point 计算一个很小的 descriptor，例如：

    textocc_count_ratio        = assigned_neighbors_to_kernel / max_neighbors
    mean_influence         = mean(kernel influence)
    empty_flag             = 1 if no assigned neighbor else 0
    mean_normalized_offset = mean(||relative_pos - kernel_pos|| / sigma)
    var_normalized_offset  = var(||relative_pos - kernel_pos|| / sigma)
    local_density_log      = log(1 + valid_neighbors / radius_volume)

这些量都来自 KPConvX 已经算过或容易得到的 neighbor projection，不需要再查一次 KNN / ball query。

### 4.3 模块接口

建议接口保持非常窄：

    pythonclass OccupancyKernelAttention(nn.Module):
        def __init__(
            self,
            num_kernel_points: int,
            num_groups: int = 8,
            desc_dim: int = 5,
            hidden_dim: int = 16,
            init_beta: float = 0.0,
        ):
            ...
    
        def forward(
            self,
            base_attn_logits,      # [Nq, K, G] or equivalent
            kernel_occ_desc,       # [Nq, K, D]
            valid_kernel_mask=None # [Nq, K]
        ):
            # returns calibrated attention logits
            ...

组合方式：

    textcalibrated_logits = base_logits + beta * geom_bias
    attention = sigmoid(calibrated_logits)

其中 `beta` 初始化为 `0`，保证第 0 step 与原 KPConvX 完全等价。这样即使代码有轻微不稳定，也不会像 v17 那样一开始改变训练动力学。

### 4.4 训练损失

第一版不要加复杂新损失：

    textL = CE + Lovasz

可以只加一个很弱的 attention-bias 正则作为保险：

    textL_total = CE + Lovasz + 1e-4 * mean((beta * geom_bias)^2)

但我建议首版先不加正则，靠 `beta=0` 和 warmup 控制。不要加 boundary loss、neighbor compatibility loss、pseudo-label loss 或 hypergraph loss，否则很难解释收益来自哪里。

### 4.5 初始化方式

优先用当前正在训练的 **scale04 baseline** 作为 base。原因是项目已经预注册 scale04：只隔离 `GridSample/subsample_size/kp_radius/kp_sigma/first_radius` 的物理尺度变化，不同时改输入特征、shareKP、influence 或训练 pipeline；且当前状态是“已启动、结果待回收”，不能提前声称有效。(2026-06-20汇报后KPConvX项目进展总结.md)

初始化规则：

    text原 KPConvX 权重：从 scale04 baseline checkpoint 加载
    SOKA MLP 权重：normal/std 很小或 zero-last-layer
    beta：0.0
    base_attn_logits：保持原逻辑

这样 SOKA 的低成本 probe 可以做成 **fine-tune 40~60 epoch**，而不是一上来烧完整 200 epoch。若 probe 过线，再从 scratch 或从 scale04 checkpoint 做正式可报告训练。

### 4.6 消融顺序

只做顺序消融，不并行开三条大实验：

1.  **零训练 occupancy audit**  
    用 baseline、v17、scale04 checkpoint 统计每类的 kernel empty ratio、mean occupancy、shell occupancy imbalance、valid neighbor count。目标是确认 v17 受益类是否确实伴随 occupancy/scale 异常。若 occupancy 与类别得失完全无关，SOKA No-Go。
    
2.  **SOKA stage4-only 40~60 epoch fine-tune probe**  
    只在 stage4 打开 SOKA，`beta=0` 起步。用固定 Area5 全场景 identity 复评，不看 random-crop best。
    
3.  **若 probe 过线，再 stage3+4**  
    不要同时加 stage2。stage2 分辨率高、点数多，显存和噪声都更危险。
    
4.  **正式训练**  
    只在 stage4 或 stage3+4 二选一，跑完整 200 epoch。报告 fixed identity 的 `best / last / last-k mean`，而不是训练期 random-crop best。
    

5\. 成功、失败和止损标准
--------------

### 最低成本 Go / No-Go

**Go 条件：**

*   零训练 audit 显示：固定协议中 v17 受益类与 kernel occupancy / shell imbalance / empty-kernel ratio 有可解释相关性；
*   40~60 epoch SOKA probe 相对同 checkpoint baseline 固定 identity mIoU 提升至少 `+0.004`；
*   batch time 增加不超过 `10~15%`；
*   wall / floor / door / window / column 没有系统性大跌。

**No-Go 条件：**

*   occupancy 统计与类别得失无关；
*   probe 到 60 epoch 固定 identity 不超过 matched baseline `+0.003`；
*   出现和 v17 类似的成本膨胀，超过 baseline `1.2x`；
*   主要提升只来自 beam 这类高波动 rare class，而常见结构类下降。

### 正式成功阈值

正式 200 epoch 后，SOKA 必须相对 scale04 baseline 满足：

    textfixed identity mIoU ≥ +0.006：可继续
    fixed identity mIoU ≥ +0.008：可作为论文主结果候选
    fixed identity mIoU ≥ +0.010 且多 checkpoint 稳定：强结果

如果 scale04 baseline 本身已经比旧 baseline 明显提高，那么所有比较都必须改为 **SOKA vs scale04 baseline**，不能再拿旧 `0.6556` 当参照。项目文档对 scale04 已预注册：scale04 baseline 比旧 baseline 至少 `+0.005` 才支持物理尺度是重要瓶颈，`+0.008` 且多 checkpoint 稳定才升级为新主基线。(2026-06-20汇报后KPConvX项目进展总结.md)

6\. 可能受益 / 受损类别
---------------

**最可能受益：**

*   `chair / table / sofa / bookcase`：家具类局部几何尺度多变、遮挡和密度变化明显；
*   `board / beam`：薄结构和局部稀疏结构可能从 kernel occupancy 校准中受益；
*   `clutter`：若局部 occupancy 描述能避免空 kernel 误响应，可能小幅改善。

**最可能受损：**

*   `wall / floor / ceiling`：大平面本身 occupancy 稳定，过度校准可能破坏原本强类别；
*   `door / window / column`：这些类常贴附在墙面或边界附近，若 occupancy bias 学成“局部平滑器”，可能被墙面吞掉。

所以必须监控两组指标：

    textfurniture/thin classes: chair, table, sofa, bookcase, board, beam
    risk classes: wall, floor, door, window, column

单纯 mIoU 上升但 door/window/column 系统下降，不应算论文级成功。

7\. 与已完成实验的严格区别
---------------

SOKA 不是旧实验换名：

*   不是 **v13 / v13b DA-Radius**：不重建邻域、不改变 radius、不做 CUDA 动态 support。
*   不是 **v16b hard controlled support**：不做 candidate support，不做 effective-radius mask。
*   不是 **v17 dual-support residual**：不执行 original / expanded 两次聚合，没有 `expanded_out - original_out`。
*   不是 **v16-local DA-meta**：v16-local 是局部统计到 channel bias；SOKA 是 per-kernel-point / per-group attention logit 校准，作用位置在 KPConvX kernel attention 内部。
*   不是 **v14 global token / SGCA**：没有 serialized token、没有 decoder global branch。
*   不是 **v15 boundary gate**：没有 boundary loss、没有 boundary prediction、没有边界门控。
*   不是 **ADNet ALNS**：不改 KNN / ball query / octree / support search，只使用已有 kernel projection 的占用统计。
*   不是 **PTv3 branch**：不引入 serialization attention，只保留 KPConvX 主干。

8\. 首选方案应继续改 KPConvX，还是转向预训练 / 超图 / 状态记忆 / 独立 backbone？
-------------------------------------------------------

我的裁决：

    text首选：继续改 KPConvX
    不首选：从头预训练
    不首选：超图
    不首选：完整状态记忆 / Mamba
    备用：独立 PTv3 / Sonata backbone benchmark

理由是：当前最强项目内证据不是“全局语义不够”，也不是“边界污染已被证明”，而是**评价协议改变后 v17 反转、同时官方尺度核验发现 0.02m → 0.04m 的物理尺度失配**。这直接指向 KPConvX 的几何算子和尺度配置，而不是先换框架。(2026-06-20汇报后KPConvX项目进展总结.md)

但如果 scale04 baseline + SOKA 都不能在固定协议下稳定超过 baseline，那么不要再继续发明第 9 个 KPConvX 小模块。那时备用方向应切到 **PTv3 / Sonata 独立 backbone benchmark**：用 Pointcept 官方 PTv3 配置或可用预训练权重，做同一固定 Area5 protocol 下的强基线。这个备用方向的目的不是“创新模块”，而是判断 KPConvX 改造是否还有继续投入价值。

9\. 最终选择
--------

**主创新方向：SOKA — scale-calibrated occupancy-aware kernel attention for KPConvX。**

它最符合三条约束：能从附件论文机制中自然内化进 KPConvX；不重复你已经做完的 support / DA-meta / boundary / global；在 4090D 上成本可控，并且能围绕“可信固定协议 mIoU”形成清晰论文叙事。

**备用方向：PTv3 / Sonata 独立 backbone benchmark。**

若 scale04 + SOKA 的固定协议结果不过线，直接转向独立 backbone 对照，不再继续改 support、边界或 global branch。这样对导师也更好交代：你不是盲目换题，而是在 KPConvX 内部最合理的几何校准方向失败后，用强 backbone 给出上限和路线选择。

问题
--

主创新方向：SOKA — scale-calibrated occupancy-aware kernel attention for KPConvX。

它最符合三条约束：能从附件论文机制中自然内化进 KPConvX；不重复你已经做完的 support / DA-meta / boundary / global；在 4090D 上成本可控，并且能围绕“可信固定协议 mIoU”形成清晰论文叙事。  
给我仔细说说这个

思考
--

**Drafting SOKA proposal and organizing structure**

I’ll begin by stating that SOKA is a proposed solution, not yet validated. The document will include sections like "Core issues," "Exact operators," and "Data flow," with exact equations carefully formulated. I'll also provide citations inline. The SOKA equation will use current constant influence or linear-like influence.

Thought for 2m 26s

回答
--

先把 SOKA 的定位说清楚
--------------

**SOKA：Scale-Calibrated Occupancy-Aware Kernel Attention**，中文可译为：

> **尺度校准的占据感知核点注意力**

它不是再设计一套邻域搜索，也不是再加一条 support 分支，而是直接修改 **KPConvX 原生 kernel attention 的生成方式**：

> 保持邻居集合、核点位置、KPConvX 聚合公式和网络主干不变，只让每个核点的注意力同时感知“中心点语义特征”和“当前邻域在各核点空间单元中的实际占据状态”。

因此它真正研究的问题是：

> **KPConvX 的 kernel attention 仅由中心点特征生成，是否缺少对当前局部几何采样状态的显式感知？**

这是一个尚未被你现有实验直接回答的问题。

原始 KPConvX 已经不是普通邻居注意力。它先把邻居分配到固定顺序的 kernel points，再由中心点特征生成每个 kernel point 的分组 modulation。作者明确强调，这种 attention 作用于空间中的 kernel regions，而不是直接作用于无序邻居；原始 modulation 由一个两层 MLP 从中心特征生成，并在 sigmoid 后参与核点权重调制。(Thomas\_KPConvX\_Modernizing\_Kernel\_Point\_Convolution\_with\_Kernel\_Attention\_CVPR\_2024\_paper.pdf)

* * *

一、为什么提出 SOKA
------------

1\. 当前 KPConvX 的一个潜在盲点
----------------------

设中心点为 qq，邻居为 ii，邻居被分配到最近的 kernel point kik\_i。

KPConvX 的核心形式可以简化为：

yq\=∑ihi,ki(mq,kifeat⊛wki)⊙fiy\_q= \\sum\_i h\_{i,k\_i} \\left( m^{feat}\_{q,k\_i}\\circledast w\_{k\_i} \\right)\\odot f\_i

其中：

*   wkw\_k：第 kk 个 kernel point 的深度卷积权重；
*   hi,kih\_{i,k\_i}：邻居和最近核点之间的 influence；
*   mq,kfeatm^{feat}\_{q,k}：由中心点特征 fqf\_q 生成的 kernel attention；
*   fif\_i：邻居特征。

原始 KPConvX 中：

mqfeat\=σ(α(fq))m^{feat}\_{q}=\\sigma\\big(\\alpha(f\_q)\\big)

也就是说，kernel attention 主要根据中心点当前特征判断：

*   哪个方向的 kernel region 更重要；
*   哪些通道组应当增强；
*   哪些空间区域应当抑制。

问题在于，两个中心点即便具有相似的中心特征，它们的实际邻域可能完全不同：

*   一个点周围的 kernel cells 被均匀占据；
*   一个点位于墙角，邻居只分布在两个平面；
*   一个点位于物体边缘，一半 kernel cells 为空；
*   一个点位于稀疏区域，实际有效邻居很少；
*   一个点受 crop、遮挡或扫描密度影响，邻域严重不完整。

原始 attention 可能通过前序特征**间接推断**这些情况，但没有直接看到“哪个 kernel cell 中真正落了多少点”。

SOKA 就是补这个缺口。

* * *

2\. 为什么你的项目里这个问题值得研究
--------------------

你已经发现当前 Pointcept baseline 的物理尺度并非官方 S3DIS 配置：

配置

原 baseline

官方 S3DIS 风格

初始网格

0.02 m

0.04 m

相对 kernel radius

2.3

2.1

第一层物理半径

0.046 m

0.084 m

原 baseline 第一层只覆盖约 4.6 cm，而 scale04 配置覆盖约 8.4 cm。当前 scale04 baseline 正在训练，因此还不能断言尺度校准有效，但它至少证明了一个重要问题：**KPConvX 的 kernel operator 必须放在正确的物理尺度下理解，而不能只讨论抽象的邻居数和 radius multiplier。** (2026-06-20汇报后KPConvX项目进展总结.md)

与此同时，固定全场景复评显示：

*   baseline：0.6556；
*   v17：0.6685；
*   提升约 +0.0130；
*   68 个房间中 46 个房间提升。

这表明局部几何处理确实可能存在真实改进空间，但 v17 使用双 support，训练时间由约 24.10 小时上升到 44.52 小时。(2026-06-20汇报后KPConvX项目进展总结.md)

SOKA 试图回答一个比 v17 更干净的问题：

> 不扩大邻域、不改变 support，单纯提高原生 kernel attention 对局部几何占据状态的感知，能否获得同类收益？

* * *

二、SOKA 三个部分分别是什么意思
------------------

1\. Scale-Calibrated：尺度校准
-------------------------

这里的“尺度校准”**不等于再次学习 point-wise radius**。

SOKA 不做：

*   动态扩大或缩小 support；
*   每个点预测 radius；
*   重建 KNN 或 radius graph；
*   生成第二套 candidate neighbors。

它做的是两层校准。

### 第一层：物理尺度校准

所有 occupancy 均基于当前层的实际物理 support：

rl\=sl×ρlr\_l = s\_l \\times \\rho\_l

其中：

*   sls\_l：该层的 subsampling/grid size；
*   ρl\\rho\_l：KPConvX 的相对 kernel radius。

因此，同一个 occupancy 数值必须解释为“在多少米的局部空间内形成的占据”。

如果 scale04 成为新主基线，那么 SOKA 应建立在 scale04 上；否则 occupancy 机制和物理尺度失配会混在一起，难以形成清楚的论文结论。

### 第二层：kernel-cell 几何校准

KPConvX 的 kernel cells 并不具有完全相同的空间体积：

*   中心 kernel cell；
*   内层 shell kernel cells；
*   外层 shell kernel cells；

它们的 Voronoi 区域大小并不一致，所以不能直接认为“每个 cell 都应该有相同数量的邻居”。

SOKA 为每个 kernel point 计算一个参考占据先验：

πlk\=P(uniform point assigned to kernel k)\\pi\_{lk}=P(\\text{uniform point assigned to kernel }k)

具体可以在模型初始化时：

1.  在单位球内均匀采样大量虚拟点；
2.  按当前 block 的实际 kernel points 分配最近核点；
3.  统计每个 kernel cell 的理论占据比例；
4.  将结果缓存为 πlk\\pi\_{lk}。

这样，外层 cell 天然较大或较小造成的计数差异不会被误判为局部结构。

这比简单对 neighbor count 做归一化更严谨。

* * *

2\. Occupancy-Aware：占据感知
------------------------

对于中心点 qq，定义第 kk 个 kernel cell 的有效占据量：

cqk\=∑i1(ki\=k) vic\_{qk} = \\sum\_i \\mathbf 1(k\_i=k)\\,v\_i

在当前 `constant influence` 配置下，可以直接取：

vi\=1v\_i=1

因此 cqkc\_{qk} 就是被分配到该 kernel cell 的有效邻居数。

将其归一化：

pqk\=cqk+ε∑jcqj+Kεp\_{qk} = \\frac{c\_{qk}+\\varepsilon} {\\sum\_j c\_{qj}+K\\varepsilon}

然后与核点几何先验比较：

zqk\=clip⁡\[log⁡(pqk+ε)−log⁡(πlk+ε),−3,3\]z\_{qk} = \\operatorname{clip} \\left\[ \\log(p\_{qk}+\\varepsilon) - \\log(\\pi\_{lk}+\\varepsilon), -3,3 \\right\]

解释如下：

*   zqk≈0z\_{qk}\\approx 0：该 kernel cell 的占据接近期望；
*   zqk\>0z\_{qk}>0：该方向过度占据；
*   zqk<0z\_{qk}<0：该方向欠占据或为空。

仅使用 pqkp\_{qk} 会丢失整个邻域有多少有效点，因此还需要一个 support fill ratio：

dq\=HqvalidHmaxd\_q=\\frac{H\_q^{valid}}{H^{max}}

再加入归一化 occupancy entropy：

eq\=−∑kpqklog⁡(pqk+ε)log⁡Ke\_q = -\\frac{\\sum\_k p\_{qk}\\log(p\_{qk}+\\varepsilon)} {\\log K}

三者分别表达：

*   zqkz\_{qk}：每个空间方向相对过密还是过疏；
*   dqd\_q：整个 support 是否完整；
*   eqe\_q：邻居是否集中于少数方向。

ADNet 的 ALNS 论文也指出，室内点云不同区域存在密度不均，固定邻域在稀疏区域可能丢失细节，在密集区域又可能引入冗余；其解决方式是调整邻域尺度。SOKA只吸收这一问题意识，不照搬其动态图构建，而是将密度信息用于 KPConvX 原有 kernel attention。(Adaptive local neighborhood search and dual attention convolution network.pdf)

* * *

3\. Kernel Attention：仍然是 KPConvX 内部机制
-------------------------------------

对每个 kernel point 构造：

uqk\=\[zqk,dq,eq,x~krl\]u\_{qk} = \\left\[ z\_{qk}, d\_q, e\_q, \\frac{\\widetilde{x}\_k}{r\_l} \\right\]

其中 x~k/rl\\widetilde{x}\_k/r\_l 是归一化 kernel coordinate。

使用一个共享的小 MLP：

bqk\=ψ(uqk)b\_{qk}=\\psi(u\_{qk})

最轻量版本只输出一个标量，而不是再输出完整的 CgC\_g 通道向量：

bqk∈Rb\_{qk}\\in \\mathbb R

原始 KPConvX 的 feature attention logits 记为：

aqkgfeata^{feat}\_{qkg}

SOKA 修改为：

aqkgSOKA\=aqkgfeat+γl bqk ηlga^{SOKA}\_{qkg} = a^{feat}\_{qkg} + \\gamma\_l\\, b\_{qk}\\,\\eta\_{lg}

然后：

mqkgSOKA\=σ(aqkgSOKA)m^{SOKA}\_{qkg} = \\sigma(a^{SOKA}\_{qkg})

其中：

*   gg：KPConvX 的 attention group；
*   bqkb\_{qk}：occupancy 给出的 kernel-level spatial bias；
*   ηlg\\eta\_{lg}：不同通道组对 occupancy 的响应；
*   γl\\gamma\_l：每一层的整体 occupancy 强度。

最后仍使用原始 KPConvX 聚合：

yq\=∑ihi,ki(mq,kiSOKA⊛wki)⊙fiy\_q= \\sum\_i h\_{i,k\_i} \\left( m^{SOKA}\_{q,k\_i}\\circledast w\_{k\_i} \\right)\\odot f\_i

所以它不是另一个 feature branch，而是在**原来的 kernel attention logits 内进行条件化**。

* * *

三、完整数据流
-------

    textquery coordinates
    support coordinates
    neighbor indices
            │
            ▼
    原 KPConvX 最近核点分配 k_i
            │
            ├──────────────► 原 KPConvX 聚合所需 k_i / influence
            │
            ▼
    按 kernel cell scatter-count
    c_qk = 每个 kernel cell 的有效邻居数
            │
            ▼
    几何先验校准
    z_qk = log(observed occupancy / expected occupancy)
            │
            ├── valid-neighbor ratio d_q
            ├── occupancy entropy e_q
            └── normalized kernel coordinate
            │
            ▼
    极小共享 MLP ψ
            │
            ▼
    occupancy kernel bias b_qk
            │
    中心特征 f_q ──► 原 KPConvX attention MLP ──► a_feat
            │                                    │
            └────────────────────────────────────┤
                                                 ▼
                                   a_feat + occupancy bias
                                                 │
                                              sigmoid
                                                 │
                                                 ▼
                                  原 KPConvX depthwise aggregation

关键点是：

> **没有第二次 neighbor query，也没有第二次 KPConvX aggregation。**

这正是它与 v17 在工程成本上的根本区别。

* * *

四、它与现有实验到底哪里不同
--------------

已完成方向

改变的对象

SOKA 与其区别

v13/v13b DA-Radius

邻域半径、邻域图

SOKA 固定邻域，不预测 radius

v16b hard controlled support

effective support mask

SOKA 不删除或筛选邻居

v17 dual-support residual

original/expanded 两套聚合

SOKA 只有一次原生聚合

v16-local DA-meta

局部统计生成 channel bias

SOKA 生成 **kernel-index-aligned spatial bias**

SGCA/global token

跨点或全局上下文

SOKA 完全局部，不引入 global token

boundary gate

根据边界风险控制特征

SOKA 不需要边界标签或边界损失

inference alpha sweep

推理时调整已有 residual

SOKA 是训练期内生的 kernel operator

scale04 baseline

全局物理尺度配置

SOKA 在已校准尺度上处理局部 occupancy variation

最容易产生混淆的是 v16-local DA-meta。

二者虽然都使用局部统计，但因果位置不同：

    textDA-meta:
    local statistics
        -> channel bias / gate
        -> 改变整个点特征
    
    SOKA:
    kernel-cell occupancy
        -> 第 k 个核点的 attention logit
        -> 改变该空间核区域的聚合权重

如果最终实现只是把邻居数、平均距离拼到中心特征后再过 MLP，那实际上会重新退化成 DA-meta。

SOKA 必须保留以下特征，才算新实验：

1.  occupancy 是 KK 个 kernel cells 对齐的；
2.  bias 在 sigmoid 前进入 kernel attention；
3.  不改变 support；
4.  不额外执行一次 convolution；
5.  occupancy branch 与 kernel index 存在明确对应关系。

* * *

五、推荐的第一版不是“大而全 SOKA”，而是 SOKA-Lite
---------------------------------

我建议第一版使用：

    text每个 kernel cell 输出 1 个 occupancy bias
    而不是输出 Cg 个完整通道组 bias

即：

ψ:R6→R16→R\\psi:\\mathbb R^6\\rightarrow \\mathbb R^{16}\\rightarrow \\mathbb R

而不是：

ψ:R6→R16→RCg\\psi:\\mathbb R^6\\rightarrow \\mathbb R^{16} \\rightarrow \\mathbb R^{C\_g}

原因有三点：

1.  **因果更干净**  
    occupancy 负责回答“这个空间 cell 是否值得信任”，原中心特征 attention 继续负责通道语义。
    
2.  **成本更小**  
    不需要额外保存完整的 N×K×CgN\\times K\\times C\_g occupancy tensor。
    
3.  **不容易重复 DA-meta**  
    如果 occupancy branch 直接生成大量 channel-wise 参数，它又会变成一个统计条件化通道模块。
    

可以使用一个很小的 group response：

ηlg∈RCg\\eta\_{lg}\\in\\mathbb R^{C\_g}

将单个 kernel occupancy bias 映射到不同 attention groups。

* * *

六、初始化必须保证“初始时严格等于 baseline”
---------------------------

这是 SOKA 是否可信的关键。

occupancy MLP 的最后一层必须：

    pythonweight = 0
    bias = 0

因此初始时：

bqk\=0b\_{qk}=0

从而：

aqkgSOKA\=aqkgfeata^{SOKA}\_{qkg}=a^{feat}\_{qkg}

也就是：

> 加入 SOKA 后，在训练第一步之前，模型输出必须与原 baseline 完全一致。

建议设置单元测试：

    text加载同一 baseline checkpoint
    同一输入
    baseline logits 与 SOKA logits：
    max_abs_diff < 1e-6

不要同时修改：

*   influence mode；
*   `share_kp`；
*   kp radius；
*   input feature；
*   support 数量；
*   loss；
*   pooling；
*   decoder。

否则即使 SOKA 提升，也无法确定是哪一个变量造成的。

* * *

七、训练损失
------

首轮主实验不应增加任何辅助损失：

L\=LCE+LLovasz\\mathcal L = \\mathcal L\_{CE} + \\mathcal L\_{Lovasz}

理由是：

*   SOKA 是 operator 改造，不需要额外监督；
*   occupancy branch 已经通过 segmentation loss 获得梯度；
*   增加 occupancy consistency、attention entropy 或边界损失，会再次引入多个不可分辨变量；
*   你已经有 boundary auxiliary loss 下降但主任务负迁移的先例。

如果出现 attention 饱和，可以通过结构约束处理：

bqk\=2tanh⁡(ψ(uqk))b\_{qk}=2\\tanh(\\psi(u\_{qk}))

而不是马上添加新的正则损失。

* * *

八、训练方案
------

前提：先等 scale04 baseline
----------------------

现在不应立即启动 SOKA 完整训练，因为 scale04 baseline 尚无结果。

建议按以下条件处理。

### scale04 fixed mIoU 提升 ≥ 0.008

把 scale04 升级为 SOKA 主基线。

论文叙事最完整：

    text先校准 S3DIS 的物理 kernel scale
    再处理同一物理 support 内的局部 occupancy variation

### scale04 提升在 0～0.005

尺度可能不是主要瓶颈。

SOKA仍可测试，但“scale-calibrated”的贡献主要来自 kernel-cell prior normalization，而不能声称修复了物理尺度失配。

### scale04 明显下降

先检查：

*   小类是否因 0.04 m 采样损失；
*   有效邻居数量分布；
*   各 stage occupancy entropy；
*   物理 crop 覆盖范围。

不建议直接在失败的 scale04 上继续堆 SOKA。

* * *

最低成本 Go/No-Go
-------------

### A. 零训练 occupancy informativeness audit

用冻结 baseline，在训练区域的固定留出房间中导出：

*   zqkz\_{qk}；
*   fill ratio；
*   occupancy entropy；
*   baseline entropy；
*   baseline 是否预测正确；
*   GT 类别，仅用于分析。

比较两个错误预测器：

    textModel A:
    baseline entropy / confidence
    
    Model B:
    baseline entropy / confidence
    + occupancy descriptors

建议 Go 条件：

*   occupancy 加入后错误预测 AUROC 提高至少 0.03；
*   或错误预测 NLL 降低至少 2%；
*   在至少 3 个独立房间组中方向一致；
*   不是只由 beam 等极少数类驱动。

如果 occupancy 对“baseline 是否出错”没有额外预测力，说明中心特征可能已经包含了这些信息，不值得再训练约 25 小时。

### B. 低成本 branch-only probe

从 scale04 baseline checkpoint 加载 SOKA：

*   冻结原 backbone；
*   只训练 occupancy MLP、η\\eta、分类 head；
*   训练 10～15 个日志 epoch；
*   使用固定训练区 dev rooms 评估。

预计耗时约 1～3 小时。

Go 条件：

*   fixed dev mIoU ≥ baseline +0.003；
*   至少半数 dev rooms 提升；
*   attention 没有大面积饱和；
*   提升不只来自单一 rare class。

这不是最终论文结果，只是决定值不值得跑完整实验。

* * *

九、主实验建议
-------

若 Go/No-Go 通过，首个正式实验：

    textSOKA-Lite
    + 已确认的主 baseline scale
    + 所有 encoder KPConvX blocks
    + 原 support
    + 原 CE + Lovasz
    + 原训练 pipeline

不建议第一版只放 stage4，因为这会重新引入“为什么只选 stage4”的人为假设。

更稳妥的方式是：

*   所有 encoder KPConvX blocks 都接入；
*   每层有独立的 γl\\gamma\_l 或 group response；
*   通过训练后参数判断哪一层真正使用 occupancy；
*   stem 仍保持普通 KPConv；
*   decoder 第一版不改。

* * *

开发期训练方式
-------

从 scale04 baseline checkpoint warm-start：

*   occupancy branch 零初始化；
*   backbone LR 为原 continuation LR 的 0.1 倍；
*   occupancy branch LR 为 1 倍；
*   训练 40～60 个日志 epoch；
*   同时做一个 matched baseline continuation control。

必须有 continuation control，否则无法区分：

    textSOKA 收益

与：

    textbaseline 多训练了 60 epoch 的收益

开发期预计：

*   SOKA continuation：约 7～10 小时；
*   baseline continuation control：约 6～9 小时。

实际时间取决于 scale04 当前训练速度。

* * *

论文确认方式
------

只有开发期 fixed mIoU 提升达到至少 +0.005，才做：

*   从头训练；
*   与 baseline 完全相同的 schedule；
*   至少 3 个 seed；
*   Area5 固定全场景 identity 协议；
*   预注册 checkpoint 规则。

* * *

十、4090D 工程成本
------------

SOKA 的额外操作主要是：

*   一个 `scatter_add`；
*   N×KN\\times K occupancy 统计；
*   一个极小共享 MLP；
*   对原 attention logits 做加法。

它不需要：

*   第二次 neighbor search；
*   第二次 KPConvX；
*   expanded candidate graph；
*   dual-support feature cache。

因此预期成本应该远小于 v17。

保守估计：

项目

预期变化

参数量

增加远低于 0.1M

训练时间

baseline 的 1.05～1.15 倍

峰值显存

增加约 5%～10%

batch size

应继续保持 3

完整训练

若 baseline 24.1h，约 25～29h

这些只是工程估计，必须用前 200～500 iteration 实测。

明确的工程 No-Go：

> 如果 SOKA 导致 batch size 从 3 降为 2，或者训练时间超过 baseline 的 1.2 倍，就应该先优化实现，而不是直接长训。

因为 batch 改变会同时影响 BN 和优化过程，结果不再是干净的模块对照。

* * *

十一、必须监控的机制指标
------------

仅看 mIoU 不足以说明 SOKA 是否真的工作。

每层至少记录：

### 1\. Occupancy 分布

*   valid neighbor ratio；
*   occupied kernel cell 数；
*   occupancy entropy；
*   zqkz\_{qk} 的均值和标准差；
*   空 kernel cell 比例。

### 2\. Occupancy branch 强度

*   bqkb\_{qk} mean/std；
*   occupancy logit residual RMS；
*   ∣Δa∣/∣afeat∣|\\Delta a|/|a^{feat}|；
*   各层 γl\\gamma\_l 或 ηlg\\eta\_{lg}。

### 3\. Attention 状态

*   kernel attention mean/std；
*   小于 0.05 的比例；
*   大于 0.95 的比例；
*   SOKA 与 baseline attention cosine difference。

### 4\. 性能分层

分别统计：

*   low-fill 与 high-fill 点；
*   low-entropy 与 high-entropy occupancy；
*   室内大平面与小物体；
*   房间中心与 fragment/crop 边缘；
*   各类别 IoU。

真正支持 SOKA 假设的结果应当是：

> SOKA 主要改善 occupancy 不完整、方向分布不均或局部采样复杂区域，而不是所有点无差别增益。

* * *

十二、最可能受益和受损的类别
--------------

这只能作为实验前假设，不能提前当作结论。

可能受益
----

### column、beam

这些类具有细长结构，kernel-cell 占据高度不均，普通中心特征 attention 可能难以直接辨别局部几何支撑方向。

### window、door、board

它们通常表现为大平面上的局部中断、凹陷或附着面。occupancy pattern 可能帮助区分：

*   连续墙面；
*   墙中开口；
*   平面附着物。

### clutter

clutter 局部几何和采样分布复杂，显式 occupancy 可能提高 attention 对不规则结构的适应性。

固定协议中 v17 对 ceiling、clutter、beam、window、column 均有明显提升，而真正下降的是 sofa、door 和 bookcase，因此不能再把 window、column 或 clutter 预设为“扩展上下文污染类”。(2026-06-20汇报后KPConvX项目进展总结.md)

可能受损
----

### wall、floor、ceiling

这些大平面本来已容易识别。occupancy branch 如果过强，可能把扫描密度、crop 截断或朝向差异当成语义信号。

### sofa、bookcase

这两类的困难更可能来自：

*   物体级形状；
*   房间语义上下文；
*   材质和颜色；
*   相似家具混淆。

局部 occupancy 未必能解决，甚至可能强化错误的局部几何捷径。

### beam

beam 虽可能受益，但样本极少，occupancy 统计也可能非常噪声化。因此不能让 beam 单类提升主导整体结论。

* * *

十三、最可能失败的原因
-----------

1\. Occupancy 信息已经被中心特征隐式编码
---------------------------

KPConvX 作者认为，从中心特征生成 kernel modulation 会迫使网络在中心特征中保留上下文信息。(Thomas\_KPConvX\_Modernizing\_Kernel\_Point\_Convolution\_with\_Kernel\_Attention\_CVPR\_2024\_paper.pdf)

因此 SOKA 的最大风险是冗余：

> 中心特征已经足够表达邻域完整性，显式 occupancy 不再增加新信息。

这正是为什么必须先做 occupancy informativeness audit。

2\. Occupancy 学到的是采样 pipeline，而不是语义
-----------------------------------

模型可能利用：

*   SphereCrop 边缘；
*   point cap；
*   grid sampling pattern；
*   fragment overlap；
*   固定房间扫描密度；

获得训练指标提升，但在不同采样协议下失效。

因此必须做：

*   identity 全场景；
*   不同 fragment size；
*   至少一个轻微 density perturbation；
*   房间级 bootstrap。

3\. 理论 occupancy prior 不适合表面点云
------------------------------

均匀球体 Monte Carlo prior 是三维体积先验，而室内点云通常采样于二维表面。

解决方法不是马上改成复杂 learned prior，而是做一个消融：

    textSOKA-Calibrated:
    observed / geometric prior
    
    SOKA-Raw:
    仅使用 normalized raw occupancy

如果 raw occupancy 更好，说明均匀球体 prior 并不合理。

4\. 0.04 m 已经消除了大部分问题
---------------------

如果 scale04 baseline 本身显著提升，occupancy variation 可能已经通过更合适的物理 support 得到缓解，SOKA剩余收益可能很小。

这不是坏结果，而是说明：

> 主要瓶颈是 global physical scale，而不是 point-wise occupancy adaptation。

5\. Occupancy branch 过强
-----------------------

若 Δa\\Delta a 很快超过原 attention logits，SOKA就会从“校正 attention”变成“替代 attention”。

建议约束：

bqk\=2tanh⁡(ψ(uqk))b\_{qk}=2\\tanh(\\psi(u\_{qk}))

并监控：

Rl\=RMS⁡(Δal)RMS⁡(alfeat)+εR\_l= \\frac{\\operatorname{RMS}(\\Delta a\_l)} {\\operatorname{RMS}(a^{feat}\_l)+\\varepsilon}

建议训练早期：

Rl<0.1R\_l<0.1

后期最好仍不超过约 0.3～0.5。

* * *

十四、成功与失败判定
----------

成功
--

相对同尺度 baseline：

*   fixed identity mIoU ≥ `+0.005`；
*   `+0.008` 以上可认为有较强价值；
*   房间级 paired bootstrap 95% CI 下界大于 0；
*   至少两个 seed 方向一致，最终三 seed 均值提升；
*   不由单个 rare class 主导；
*   训练成本不超过 baseline 约 1.15 倍；
*   model\_last 或预注册 checkpoint 仍保持提升。

失败
--

满足任一项即可停止：

*   fixed mIoU 差异在 ±0.003 内；
*   random-crop best 提升，但 fixed identity 不提升；
*   收益只来自 beam 等一个稀有类别；
*   large planar classes 系统性下降；
*   occupancy logits 大量饱和；
*   occupancy branch 学到 crop boundary；
*   显存导致 batch size 改变；
*   训练成本接近 v17，但增益没有超过 v17。

* * *

十五、我对 SOKA 的最终判断
----------------

SOKA 不是“必然能提高 mIoU”的方案，它的价值在于**因果问题干净、与 KPConvX 高度一致、成本相对可控**。

它比继续扩展 support 更合理的原因是：

1.  v17 fixed identity 的 +0.0130 表明局部几何适配可能确实有价值；
2.  alpha sweep 说明不能简单依靠推理期 residual 强度解释收益；
3.  当前 support 路线最大的成本来自重复聚合；
4.  KPConvX 的原生 attention 恰好提供了一个更低成本的适配入口；
5.  occupancy 可以直接使用已有 nearest-kernel assignment，不需要重新构图。

我建议把第一版严格限制为：

> **在完成 scale04 baseline 判定后，在固定原始 support 内，用几何先验校准的 kernel-cell occupancy，作为零初始化的标量 bias 加入 KPConvX 原始 attention logits；保持 CE+Lovasz、主干、邻域、decoder 和训练协议全部不变。**

不要在第一版加入：

*   dynamic radius；
*   boundary loss；
*   uncertainty gate；
*   global token；
*   hypergraph；
*   Mamba/state memory；
*   预训练；
*   第二条 convolution 分支。

否则 SOKA 最重要的优势——**它是一个清楚、低成本、可归因的 KPConvX 内部机制实验**——就会被破坏。

问题
--

[

](https://files.chat01.ai/user-uploads/365c7025d12732bfb2b252f2f5d3abc58f27ba37cd14885c545b9cacb5aa93dc)KPConvX全部实验结果汇总.md

[

](https://files.chat01.ai/user-uploads/9ea8b26a66e81e98d17a40eda633e84feeac8501c29b8f41ccaaa5d05226b850)Pointcept-codex-v16-boundary-aware-dual-framework.zip

[

](https://files.chat01.ai/user-uploads/359c98ed563bc12238bbab8fd7aae160a0ca9ce5a78691b94b84359c3f9cc055)Adaptive local neighborhood search and dual attention convolution network.pdf

这是我之前实验结果的汇总，我的来源就是kpconv和kpconvx还有ADNet这3 篇论文。  
压缩包是是我基于pointcept开发的源码，这是基于 Pointcept 里面的源码，里面有一些是它自带的，你可以不用管，重点看我接入的 kpconvx和kpconvx\_hybrid，和kpnext，kpconvx的官方开源地址是：[https://github.com/apple/ml-kpconvx，pdf是和上面发你一样的adnet的论文，你在看看soka的实现可以完成和现在的实现怎么样，还有这篇论文还有SPP-IPCs这个可以加入吗，可以去更改我的kpconvx的数据流和框架，你仔细研究，如果刷要ChatGPT深度研究的辅助或者验证可以在研究的最后指出来](https://github.com/apple/ml-kpconvx%EF%BC%8Cpdf%E6%98%AF%E5%92%8C%E4%B8%8A%E9%9D%A2%E5%8F%91%E4%BD%A0%E4%B8%80%E6%A0%B7%E7%9A%84adnet%E7%9A%84%E8%AE%BA%E6%96%87%EF%BC%8C%E4%BD%A0%E5%9C%A8%E7%9C%8B%E7%9C%8Bsoka%E7%9A%84%E5%AE%9E%E7%8E%B0%E5%8F%AF%E4%BB%A5%E5%AE%8C%E6%88%90%E5%92%8C%E7%8E%B0%E5%9C%A8%E7%9A%84%E5%AE%9E%E7%8E%B0%E6%80%8E%E4%B9%88%E6%A0%B7%EF%BC%8C%E8%BF%98%E6%9C%89%E8%BF%99%E7%AF%87%E8%AE%BA%E6%96%87%E8%BF%98%E6%9C%89SPP-IPCs%E8%BF%99%E4%B8%AA%E5%8F%AF%E4%BB%A5%E5%8A%A0%E5%85%A5%E5%90%97%EF%BC%8C%E5%8F%AF%E4%BB%A5%E5%8E%BB%E6%9B%B4%E6%94%B9%E6%88%91%E7%9A%84kpconvx%E7%9A%84%E6%95%B0%E6%8D%AE%E6%B5%81%E5%92%8C%E6%A1%86%E6%9E%B6%EF%BC%8C%E4%BD%A0%E4%BB%94%E7%BB%86%E7%A0%94%E7%A9%B6%EF%BC%8C%E5%A6%82%E6%9E%9C%E5%88%B7%E8%A6%81ChatGPT%E6%B7%B1%E5%BA%A6%E7%A0%94%E7%A9%B6%E7%9A%84%E8%BE%85%E5%8A%A9%E6%88%96%E8%80%85%E9%AA%8C%E8%AF%81%E5%8F%AF%E4%BB%A5%E5%9C%A8%E7%A0%94%E7%A9%B6%E7%9A%84%E6%9C%80%E5%90%8E%E6%8C%87%E5%87%BA%E6%9D%A5)

回答
--

According to a document from 2026-07-14，结合你的实验汇总、压缩包中的 `kpconvx / kpconvx_hybrid / kpnext` 实现、Apple 官方 Pointcept wrapper，以及 ADNet 的结构图和消融，我的结论是：

> **SOKA 在你当前代码中可以完整、干净地实现，而且适配点非常明确；但需要修正之前“主要依赖邻居计数”的设计。**
> 
> 你的 plain KPConvX 实际使用固定数量 KNN，局部总邻居数通常近似恒定，因此真正有信息量的不是“有多少邻居”，而是：
> 
> *   邻居被分配到了哪些 kernel cells；
> *   每个 kernel cell 内邻居离中心多远；
> *   邻居与最近 kernel point 的匹配误差；
> *   kernel-cell 占据是否集中、空置或偏斜。
> 
> **SPP-IPCs 不建议按论文图示直接移植。** 更合理的做法是把它重新解释为 KPConvX 原生的 **kernel-shell pyramid context**，作为 SOKA 通过后的第二阶段消融，而不是第一版同时加入。

你的最新汇总中，scale04 plain KPConvX 的固定全场景结果已经达到 `0.6939`，是当前最强可信基线；scale04-v17 仍在训练，预注册成功线为 `0.6989`。因此现在可以完成 SOKA 的代码、单元测试和低成本诊断，但不建议在 scale04-v17 结果出来前启动另一轮完整 200 epoch。(KPConvX全部实验结果汇总.md) (KPConvX全部实验结果汇总.md)

* * *

一、先说我对三个源码目录的判断
---------------

Apple 官方仓库明确说明，它提供了两个 Pointcept 模型：

*   `kpnext` 是早期尝试，建立在 PointTransformer V2 架构上，把 attention 替换成 KPConvX；
*   `kpconvx` 是 Standalone KPConvX 主干的严格 Pointcept 转换，主干算子相同，但训练管线不同；官方论文的 S3DIS 结果使用 Standalone 管线，ScanNet 使用 Pointcept 管线。([GitHub](https://github.com/apple/ml-kpconvx))

这与你压缩包中的结构完全一致。

1\. pointcept/models/kpconvx
----------------------------

这是 SOKA 最应该依附的主线。

关键文件：

    textpointcept/models/kpconvx/kpconvx_base.py
    pointcept/models/kpconvx/utils/kpnext_blocks.py
    pointcept/models/kpconvx/utils/torch_pyramid.py

它的优点是：

*   数据流干净；
*   与 Apple 官方 wrapper 高度一致；
*   没有 DA、support、boundary、global 等历史分支；
*   容易证明 SOKA 的增益来自 kernel attention 本身。

2\. pointcept/models/kpconvx\_hybrid
------------------------------------

这里包含你历史实验中积累的：

*   DA-Radius；
*   controlled support；
*   DA-meta；
*   boundary；
*   SGCA；
*   dual-support；
*   v16、v17 监控逻辑。

它对 SOKA 的价值主要是：

> **可以借用其“在 GroupNorm 后、sigmoid 前注入 attention bias”的工程写法，但不要直接继承整个 hybrid 模型。**

如果直接让 SOKA 继承 `KPConvXV16` 或 `KPConvXV17`，会产生三个问题：

1.  代码中存在大量历史开关；
2.  checkpoint key 和监控项复杂；
3.  很难在论文中证明 SOKA 没有暗中依赖 DA-meta 或 support 逻辑。

因此，`kpconvx_hybrid` 应作为参考实现，而不是 SOKA 的基类。

3\. pointcept/models/kpnext
---------------------------

你的 `kpnext` 是 PointTransformer V2 风格的数据流：

    textBlockSequence
      -> 每个 stage 重新 pointops.knn_query
      -> KPConvX 替换原 attention
      -> GridPool / UnpoolWithSkip

相关代码在：

    textpointcept/models/kpnext/kpnext_base.py:53-190

它和当前 scale04 KPConvX baseline 不是同一个主干。

因此：

> **第一版 SOKA 不应在 kpnext 上实现。**

否则一旦结果不同，你无法区分是：

*   SOKA 本身；
*   PTv2 风格 block；
*   重新查询 KNN；
*   GridPool/Unpool 数据流；

造成的。

后续如果 SOKA 在严格 KPConvX 上成功，可以把 kpnext 作为跨架构泛化实验，而不是主实验。

* * *

二、当前 KPConvX 的真实数据流
-------------------

你当前 `KPConvX.forward()` 的关键顺序是：

    text1. gather neighbor features
    2. 取中心/最近点特征 pooled_feats
    3. alpha_mlp(pooled_feats)
    4. GroupNorm
    5. sigmoid
    6. 生成 per-query kernel weights
    7. 计算邻居到 kernel points 的最近分配
    8. gather 对应 kernel weight
    9. depthwise aggregation

源码位置：

    textpointcept/models/kpconvx/utils/kpnext_blocks.py:483-568

尤其是：

    pythonmodulations = self.alpha_mlp(pooled_feats)

在约第 520 行；

    pythonmodulations = self.grpnorm(modulations)
    modulations = self.attention_act(modulations)

在约第 522-529 行；

而几何分配：

    pythoninfluence_weights, neighbors, neighbors_1nn = \
        self.get_neighbors_influences(...)

直到约第 549 行才发生。

`get_neighbors_influences()` 中已经计算了：

    pythondifferences = neighbors.unsqueeze(2) - self.kernel_points
    sq_distances = torch.sum(differences ** 2, dim=3)
    nn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=2)

位置：

    textpointcept/models/kpconvx/utils/kpnext_blocks.py:421-481

这意味着 SOKA 不需要新建邻域，不需要新增 KNN，也不需要新增 convolution。

它只需要：

> 把已有的 `neighbors_1nn` 和 `nn_sq_dists` 提前取出来，构造 occupancy descriptor，再把一个很小的 bias 加到原始 `modulations` 中。

* * *

三、之前 SOKA 设计必须修正的地方
-------------------

1\. 不能只看 valid neighbor count
-----------------------------

你的 plain scale04 路径使用：

    pythonpointops.knn_query(neighbor_limits[i], cur_points, offset)

代码位置：

    textpointcept/models/kpconvx/utils/torch_pyramid.py:230-236

邻居上限是：

    pythonneighbor_limits=(12, 16, 20, 20, 20)

因此对于大部分非异常点：

    textstage1 有 12 个邻居
    stage2 有 16 个邻居
    stage3-5 有 20 个邻居

总邻居数通常固定。

这与 radius ball query 不一样。固定 KNN 下：

    text稀疏区域：邻居更远
    密集区域：邻居更近

但邻居数量可能完全相同。

所以：

    textvalid_count / H

只对以下情况有信息：

*   房间或 fragment 边缘；
*   极小 packed cloud；
*   shadow neighbor；
*   查询失败。

它不能作为 SOKA 的主要密度表征。

2\. 真正应当使用的是 kernel-cell 分布和距离
------------------------------

固定 KNN 下，SOKA 应感知：

邻居被分配到哪个 kernel cell\\text{邻居被分配到哪个 kernel cell}

以及：

该 cell 内邻居的物理扩展程度\\text{该 cell 内邻居的物理扩展程度}

而不是仅仅感知：

总共有几个邻居\\text{总共有几个邻居}

这是这次源码审查后，对原 SOKA 方案最重要的修正。

* * *

四、我建议的正式 SOKA-Lite 描述符
----------------------

对 query point qq，假设有 HH 个 KNN 邻居、K\=43K=43 个 kernel points。

你当前 shell 为：

    pythonshell_sizes = (1, 14, 28)

所以：

    textK = 1 + 14 + 28 = 43

对每个邻居 jj，已有：

*   相对坐标 δqj\=pj−pq\\delta\_{qj}=p\_j-p\_q；
*   最近 kernel cell kqjk\_{qj}；
*   与最近 kernel point 的平方距离 dqj2d^2\_{qj}。

1\. Kernel-cell occupancy fraction
----------------------------------

cqk\=∑j1\[kqj\=k\]1\[j valid\]c\_{qk} = \\sum\_j \\mathbf 1\[k\_{qj}=k\]\\mathbf 1\[j\\text{ valid}\] pqk\=cqk∑tcqt+ϵp\_{qk} = \\frac{c\_{qk}} {\\sum\_t c\_{qt}+\\epsilon}

它表示固定的 16 或 20 个邻居中，有多少比例落入第 kk 个 kernel cell。

这不是普通“局部密度”，而是：

> **局部几何在 kernel lattice 上的占据形状。**

例如：

*   平面点：邻居集中于少数共面 cells；
*   角点：分布于多个方向；
*   细柱：集中于线性方向；
*   fragment 截断点：一侧大量 cells 为空；
*   table 面与 table edge：occupancy pattern 不同。

2\. Empty-cell indicator
------------------------

eqk\=1\[cqk\=0\]e\_{qk}=\\mathbf 1\[c\_{qk}=0\]

不要认为它和 occupancy fraction 完全重复。

对于很小的非零占据和严格空置，语义不同：

*   `0.05`：有一个实际邻居；
*   `0`：该方向完全没有支撑。

3\. Mean normalized radial distance
-----------------------------------

rqk\=∑j:kqj\=k∥δqj∥/Rlcqk+ϵr\_{qk} = \\frac{ \\sum\_{j:k\_{qj}=k} \\|\\delta\_{qj}\\|/R\_l }{ c\_{qk}+\\epsilon }

其中 RlR\_l 是当前层的物理 kernel radius。

这在固定 KNN 下非常关键：

*   稀疏区域的 KNN 会被拉得更远；
*   密集区域邻居更靠近中心；
*   即使总邻居数相同，rqkr\_{qk} 仍有信息。

4\. Mean kernel-assignment error
--------------------------------

aqk\=∑j:kqj\=kdqj2/σlcqk+ϵa\_{qk} = \\frac{ \\sum\_{j:k\_{qj}=k} \\sqrt{d^2\_{qj}}/\\sigma\_l }{ c\_{qk}+\\epsilon }

它直接衡量：

> 邻居和所选 kernel point 的匹配质量。

两个 cell 即使有相同数量的邻居，也可能不同：

*   邻居紧贴 kernel point；
*   邻居只是因为“最近”而被勉强分给该 kernel point。

后者更不应被赋予强 modulation。

5\. Occupancy entropy
---------------------

Hq\=−∑kpqklog⁡(pqk+ϵ)log⁡KH\_q = -\\frac{ \\sum\_k p\_{qk}\\log(p\_{qk}+\\epsilon) }{ \\log K }

解释：

*   低 entropy：邻居集中于少数方向；
*   高 entropy：邻居在多个 kernel cells 中较均匀分布。

对每个 kk 广播 HqH\_q。

6\. Kernel shell radius
-----------------------

sk\=∥x~k∥Rls\_k=\\frac{\\|\\widetilde{x}\_k\\|}{R\_l}

只加入 kernel point 的径向 shell 信息，不直接加入完整 xyz。

原因是：

*   xyz 会加强固定 kernel 朝向记忆；
*   你的训练有 z 轴旋转增强；
*   SOKA 应优先学习“中心/内层/外层的占据差异”，而不是新建方向编码器。

最终：

gqk\=\[pqk,eqk,rqk,aqk,Hq,sk\]g\_{qk} = \[ p\_{qk}, e\_{qk}, r\_{qk}, a\_{qk}, H\_q, s\_k \]

descriptor 维度只有 6。

* * *

五、SOKA 的具体 attention 公式
-----------------------

原 KPConvX：

Aqfeat\=GN⁡(MLP⁡(fq))A^{feat}\_q = \\operatorname{GN} \\left( \\operatorname{MLP}(f\_q) \\right) Mq\=σ(Aqfeat)M\_q=\\sigma(A^{feat}\_q)

SOKA：

Bqk\=ψ(gqk)B\_{qk} = \\psi(g\_{qk})

其中：

ψ:R6→R16→R\\psi:\\mathbb R^6 \\rightarrow \\mathbb R^{16} \\rightarrow \\mathbb R

只输出一个 kernel-cell scalar。

然后：

AqkcSOKA\=Aqkcfeat+BqkA^{SOKA}\_{qkc} = A^{feat}\_{qkc} + B\_{qk} MqkcSOKA\=σ(AqkcSOKA)M^{SOKA}\_{qkc} = \\sigma(A^{SOKA}\_{qkc})

这里 cc 是 KPConvX 的 `ch_per_grp` 维度。

一个 scalar bias 在该 kernel cell 的所有 attention channel groups 中广播。

这很重要，因为它明确分工：

    text原中心特征 attention：
    决定语义通道应该如何响应
    
    SOKA：
    决定这个 kernel spatial cell 当前是否被可靠采样

第一版不要让 SOKA 直接输出完整的 `K × ch_per_grp` bias，否则它很快会变成另一个 DA-meta/channel-conditioning 模块。

* * *

六、正确的代码插入位置
-----------

你已经在 hybrid 代码中验证过一个很合适的模式：

    pythonmodulations = self.alpha_mlp(pooled_feats)
    
    if self.mod_grp_norm:
        modulations = self.grpnorm(...)
    
    da_meta_bias = self._build_da_meta_bias(...)
    modulations = modulations + da_meta_bias
    
    modulations = self.attention_act(modulations)

对应：

    textpointcept/models/kpconvx_hybrid/da_kpnext_blocks.py:737-779

SOKA 应采用同一个**位置**，但不是同一个 meta：

    pythonbase_logits = self.alpha_mlp(pooled_feats)
    
    if self.mod_grp_norm:
        base_logits = apply_group_norm(base_logits)
    
    soka_bias = self.soka_mlp(kernel_cell_desc)   # [M, K, 1]
    
    logits = base_logits.view(M, K, CpG)
    logits = logits + soka_bias
    
    modulations = torch.sigmoid(logits)

即：

> **GroupNorm 后、sigmoid 前。**

不要在 sigmoid 后直接乘一个 gate：

    pythonmodulations *= gate

因为 sigmoid 后相乘会：

*   改变 modulation 的整体尺度；
*   更容易造成全局抑制；
*   与 v17 residual strength 的逻辑过于相似；
*   难以与原始 attention logit 做强度比较。

* * *

七、几何计算顺序需要调整，但不增加第二次计算
----------------------

当前代码先生成 attention，再计算 nearest kernel assignment。

SOKA 需要改为：

    text1. gather neighbor features
    2. 获取/计算 neighbor geometry
    3. 得到 neighbors_1nn 和 nn_sq_dists
    4. 生成 SOKA descriptors
    5. alpha_mlp(center feature)
    6. GroupNorm
    7. 加 SOKA bias
    8. sigmoid
    9. 使用同一个 neighbors_1nn 聚合

注意：

> 这是把已有几何计算前移，不是新增一遍几何计算。

建议把当前：

    pythonget_neighbors_influences()

拆成更清楚的接口：

    pythondef get_neighbor_geometry(
        q_pts,
        s_pts,
        neighb_inds,
    ):
        return (
            influence_weights,
            relative_neighbors,
            neighbors_1nn,
            nn_sq_dists,
            valid_mask,
        )

当前 `get_neighbors_influences()` 没有返回 `nn_sq_dists`，但函数内部已经计算了它。

SOKA 不应为了得到 assignment error，再做一遍：

    pythonneighbors.unsqueeze(2) - kernel_points

否则会重复最重的 M×H×KM\\times H\\times K 距离张量。

* * *

八、shadow neighbor 一定要正确处理
-------------------------

你当前 KNN 查询会把 `-1` 替换成：

    pythoncur_points.shape[0]

即 shadow index：

    textpointcept/models/kpconvx/utils/torch_pyramid.py:233-235

而 `get_neighbors_influences()` 会在支持点尾部拼一个坐标约为 `1e6` 的假点。

如果直接统计 occupancy，它仍会被分配给某个 kernel cell，严重污染：

*   occupancy count；
*   mean radial distance；
*   mean assignment error。

因此必须定义：

    pythonvalid = (neighb_inds >= 0) & (neighb_inds < s_pts.shape[0])

然后所有统计都用 `valid` mask。

不要先算出 shadow 的巨大距离后再简单截断；更安全的是：

    pythonrelative_neighbors = torch.where(
        valid.unsqueeze(-1),
        relative_neighbors,
        torch.zeros_like(relative_neighbors),
    )

* * *

九、不要使用 one-hot 构造 occupancy
---------------------------

不建议：

    pythonF.one_hot(neighbors_1nn, num_classes=K)

因为会产生：

    text[M, H, K]

张量。

在 stage2 假设：

    textM ≈ 10000
    H = 16
    K = 43

这个张量就有约 688 万元素，而且后面还有反向传播和多个统计量。

应使用 flatten + `scatter_add_`：

    pythonflat_index = (
        query_index[:, None] * K + neighbors_1nn
    ).reshape(-1)
    
    count = zeros(M * K)
    count.scatter_add_(0, flat_index, valid.float().reshape(-1))
    count = count.view(M, K)

距离和同样 scatter：

    pythonradial_sum.scatter_add_(...)
    assignment_sum.scatter_add_(...)

这使新增复杂度约为：

O(MH)O(MH)

而不是新增 O(MHK)O(MHK)。

我用纯 PyTorch 做了一个小型原型，已验证：

*   邻居顺序打乱后 descriptor 不变；
*   所有 kernel-cell count 之和等于有效邻居数；
*   空 cell 不产生 NaN；
*   最后一层零初始化时输出与 baseline 完全一致；
*   第一次反向传播时最后一层能获得非零梯度。

但当前环境没有你的 PointOps/CUDA 运行环境，因此这属于算子级验证和静态源码验证，不等于已经完成 4090D 端到端运行。

* * *

十、初始化：不要同时把 beta 和输出都初始化为 0
---------------------------

最安全的初始化是：

    pythonself.soka_mlp = nn.Sequential(
        nn.Linear(6, 16),
        nn.SiLU(),
        nn.Linear(16, 1),
    )
    
    nn.init.zeros_(self.soka_mlp[-1].weight)
    nn.init.zeros_(self.soka_mlp[-1].bias)

然后直接：

    pythonlogits = base_logits + soka_bias

此时：

Bqk\=0B\_{qk}=0

所以：

SOKA initial output\=baseline output\\text{SOKA initial output} = \\text{baseline output}

不要再同时设置：

    pythonbeta = nn.Parameter(torch.tensor(0.0))
    logits = base_logits + beta * soka_bias

并把最后一层也设为 0。

因为：

*   bias 为 0；
*   beta 为 0；
*   beta 梯度也可能为 0；
*   MLP 梯度也为 0；

会形成“零乘零”的死启动。

可以二选一：

    text方案 A，推荐：
    SOKA 最后一层零初始化，固定倍率为 1
    
    方案 B：
    SOKA 最后一层小随机初始化，beta 初始化为 0

方案 A 的 baseline 等价性最容易证明。

必须有单元测试：

    pythonwith torch.no_grad():
        y_base = base_model(batch)
        y_soka = soka_model(batch)
    
    assert (y_base - y_soka).abs().max() < 1e-6

* * *

十一、SOKA 应该放在哪些 stage
--------------------

你的配置：

    pythonfirst_inv_layer = 1
    layer_blocks = (3, 3, 9, 12, 3)

在 `kpconvx_base.py` 中：

    pythonuse_conv = self.first_inv_layer >= layer

因此：

*   stem：普通 KPConv；
*   encoder stage1：KPConvD；
*   encoder stage2-5：真正的 KPConvX attention；
*   decoder：还可能包含 KPConvX block。

所以第一版最清楚的定义不是“随意选 stage4”，而是：

> **SOKA 加到所有 encoder KPConvX blocks，也就是 stage2、3、4、5；stem、stage1 KPConvD 和 decoder 保持不变。**

总共：

3+9+12+3\=273+9+12+3=27

个 KPConvX blocks。

如果每个 block 使用：

    text6 -> 16 -> 1

参数数约为：

(6×16+16)+(16×1+1)\=129(6\\times16+16)+(16\\times1+1)=129

27 个 block 约：

27×129\=348327\\times129=3483

也就是不到 0.004M 参数。

因此参数增长几乎可以忽略。

为什么第一版不改 decoder
----------------

decoder 中的局部邻域已经融合：

*   encoder skip；
*   上采样特征；
*   不同分辨率的信息。

如果同时改 encoder 和 decoder，失败时很难判断：

*   occupancy 对几何编码有害；
*   还是对语义恢复有害。

第一版只改 encoder，更符合“SOKA 改善 kernel geometry encoding”的叙事。

* * *

十二、推荐的源码组织
----------

不建议直接继续往 `kpconvx_hybrid` 塞。

建议新建：

    textpointcept/models/kpconvx_soka/
    ├── __init__.py
    ├── kpconvx_soka.py
    └── soka_blocks.py

soka\_blocks.py
---------------

包含：

    pythonclass SOKAKPConvX(KPConvX):
        ...
    
    class SOKAKPNextMultiShortcutBlock(KPNextMultiShortcutBlock):
        ...

或者更干净地复制一份非常小的 MultiShortcut block，仅把内部：

    pythonself.conv = KPConvX(...)

改成：

    pythonself.conv = SOKAKPConvX(...)

kpconvx\_soka.py
----------------

    python@MODELS.register_module("kpconvx_soka")
    class KPConvXSOKA(KPConvXBase):
        def __init__(
            self,
            soka_enabled=True,
            soka_stages=(2, 3, 4, 5),
            soka_hidden=16,
            soka_use_shell_context=False,
            **kwargs,
        ):
            super().__init__(**kwargs)
            ...
    
        def get_residual_block(...):
            ...

配置
--

继承 scale04：

    python_base_ = [
        "./semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"
    ]
    
    model = dict(
        backbone=dict(
            _delete_=True,
            type="kpconvx_soka",
            # 原 scale04 KPConvX 全部配置
            ...
            soka_enabled=True,
            soka_stages=(2, 3, 4, 5),
            soka_hidden=16,
            soka_use_shell_context=False,
        )
    )

如果 `_delete_` 会要求完整重写 backbone，可以只改：

    pythonmodel = dict(
        backbone=dict(
            type="kpconvx_soka",
            soka_enabled=True,
            ...
        )
    )

具体取决于当前配置合并器对 `type` 替换的行为。

* * *

十三、为什么不能把 SOKA 实现成 DA-meta 换名版
------------------------------

你当前 v16-local 已经做过：

    text局部点级统计
      -> 4 维 meta
      -> channel / shell / point bias

`DensityAdaptiveRadius` 的 4 维 meta 是：

    textscale - 1
    rho_norm
    valid_ratio
    dist_cv

源码：

    textpointcept/models/kpconvx_hybrid/da_kpconvx_block.py:222-243

v16 再通过：

    textpointcept/models/kpconvx_hybrid/kpx_v16.py:436-448

传给 `DAKPConvX`。

因此以下实现会重复 v16-local：

    pythonmeta = [
        mean_neighbor_distance,
        valid_ratio,
        distance_cv,
    ]
    channel_bias = MLP(meta)

SOKA 必须满足以下条件才不重复：

1.  descriptor 形状是 `[M, K, D]`，不是 `[M, D]`；
2.  每个 descriptor 与真实 `neighbors_1nn` 对齐；
3.  输出作用于对应 kernel cell；
4.  不输出普通 point-level channel bias；
5.  不修改 support；
6.  不新增第二次 aggregation。

最核心区别是：

    textv16-local：
    这个点整体处于什么局部统计状态？
    
    SOKA：
    这个点的第 k 个 kernel spatial cell
    当前被怎样占据和采样？

* * *

十四、SPP-IPCs 到底能不能加入
-------------------

结论
--

> **能吸收其思想，但不应直接照论文图实现。**

ADNet 的 SPP-IPCs 在论文中描述为：

*   多个不同大小的 pooling windows；
*   window size sis\_i 大于 stride；
*   overlapping pooling；
*   各尺度结果 concatenate 后经过 MLP。

论文图 5 把它画成沿 point dimension 的多窗口池化。(Adaptive local neighborhood search and dual attention convolution network.pdf)

论文消融也显示，在 ADNet 的完整 ALNS+DCAA+GSA 框架中：

*   用普通 max pooling：`70.7`；
*   用普通 avg pooling：`71.0`；
*   完整 SPP-IPCs：`72.9`；

即相对两个替代项分别高 `2.2` 和 `1.9` mIoU。(Adaptive local neighborhood search and dual attention convolution network.pdf)

但这个结果不能直接外推到你的 KPConvX，原因非常具体。

* * *

1\. 你的点顺序是随机的
-------------

你的 scale04 训练配置明确包含：

    pythondict(type="ShufflePoint")

位于：

    textconfigs/s3dis/semseg-kpconvx-base-s3dis-scale04-4090d-area5.py:89

因此输入点的第 100～200 个，并不一定在空间上相邻。

如果直接做：

    pythonMaxPool1d(kernel_size=s_i, stride=stride)

那么 pooling window 聚合的只是“张量中相邻的点”，而不是空间中相邻的点。

这会导致：

*   不满足点云排列不变性；
*   同一个房间只因点顺序不同就产生不同输出；
*   `ShufflePoint` 成为强随机噪声；
*   identity 测试和训练数据流不一致。

所以：

> **literal 1D SPP-IPCs 在当前 Pointcept 数据流中是不合法的。**

* * *

2\. 论文没有给出足够的可复现细节
------------------

论文提供了示意图和公式，但没有明确说明：

*   点在 pooling 前如何排序；
*   sis\_i 的具体值；
*   stride 的具体值；
*   不同 block 中是否使用相同窗口；
*   variable-length cloud 如何 padding；
*   batch 中多房间如何隔离；
*   pooling 是 feature dimension 还是 point dimension 上实现；
*   是否依赖其 ALNS 图序列。

我也没有检索到该论文的公开官方实现；目前能核验的仍是论文公式和消融，而不是可直接对照的代码。论文自身也承认多尺度搜索和融合会增加时间、参数和过拟合风险。

ADNet 的训练设置也与你不同：

*   S3DIS 切成 `2m × 2m` block；
*   每块随机采样 4096 点；
*   batch size 16；
*   SGD；
*   论文使用 RTX 4090。

而你的 Pointcept 管线是：

*   最多 40000 点 fragment；
*   packed variable-length；
*   batch size 3；
*   AdamW + OneCycle；
*   全场景 fragment 复评。

因此不能拿 ADNet 的 `+2.2` 直接当成 KPConvX 的预期收益。(Adaptive local neighborhood search and dual attention convolution network.pdf)

* * *

十五、最合理的 SPP-IPCs 内化方式：Kernel-Shell Pyramid Context
--------------------------------------------------

KPConvX 自己已经有天然的空间金字塔：

    textshell 0：1 个中心 kernel point
    shell 1：14 个内层 kernel points
    shell 2：28 个外层 kernel points

这比沿随机 point index 做 pooling 更自然。

我建议将 SPP-IPCs 的思想重新实现为：

> **Kernel-Shell Pyramid Context，简称 KSPC。**

它不是全局分支，也不是 serialized token，而是在 SOKA descriptors 上做多尺度 shell pooling。

1\. Cell-level descriptor
-------------------------

已有：

gqk\=\[p,e,r,a,H,s\]g\_{qk} = \[p,e,r,a,H,s\]

2\. Shell-level summary
-----------------------

对每个 shell tt：

uqtmean\=Mean⁡k∈Stgqku\_{qt}^{mean} = \\operatorname{Mean}\_{k\\in\\mathcal S\_t} g\_{qk} uqtmax\=Max⁡k∈Stgqku\_{qt}^{max} = \\operatorname{Max}\_{k\\in\\mathcal S\_t} g\_{qk}

得到：

    textcenter shell context
    inner shell context
    outer shell context

3\. 为每个 cell 拼入所属 shell context
-------------------------------

g^qk\=\[gqk,uq,s(k)mean,uq,s(k)max,uqglobal\]\\widehat g\_{qk} = \[ g\_{qk}, u\_{q,s(k)}^{mean}, u\_{q,s(k)}^{max}, u\_q^{global} \]

再预测：

Bqk\=ψ(g^qk)B\_{qk} = \\psi(\\widehat g\_{qk})

数据流：

    textneighbor geometry
          ↓
    per-kernel-cell descriptors
          ↓
    center / inner / outer shell pooling
          ↓
    cell descriptor + shell context
          ↓
    SOKA bias
          ↓
    原 KPConvX attention logits

这保留了 SPP-IPCs 的本质：

*   多尺度；
*   overlapping/上下文融合；
*   局部和更大范围共同表达；

但满足 KPConvX 的：

*   排列不变性；
*   kernel geometry 对齐；
*   单次邻域聚合；
*   轻量化。

* * *

十六、KSPC 与已有 DA-meta shell bias 是否重复
-----------------------------------

如果你只是做：

    pythonshell_stats -> shell_bias

那确实非常接近 v16 的：

    pythonda_meta_shell_out

因此 KSPC 必须避免直接输出三个 shell gate。

新机制应是：

    textshell pooling 提供上下文
    但最终仍输出 per-kernel-cell bias

即同一个 inner shell 中的 14 个 cells：

*   能看到共同的 inner-shell context；
*   但每个 cell 仍有自己独立的 occupancy、距离和 assignment error；
*   最终 bias 仍然不同。

这与 v16 的 shell-level broadcast bias 有实质区别。

* * *

十七、第一版要不要同时加入 KSPC
------------------

我的答案是：

> **不要。**

第一版实验必须是：

    textscale04 plain KPConvX
    + SOKA-Lite cell descriptors
    + scalar per-cell logit bias

不要同时加入：

*   KSPC；
*   SPP；
*   global pooling；
*   shell gate；
*   boundary；
*   DA-meta；
*   v17 support。

否则如果提高，你无法判断：

*   cell occupancy 有效；
*   还是 shell pyramid 有效；
*   还是更大参数量有效。

正确消融顺序：

    textA0  scale04 plain KPConvX
    
    A1  A0 + SOKA-Lite
        per-cell occupancy/spread/assignment descriptor
    
    A2  A1 + KSPC
        在 SOKA descriptors 上加入 shell pyramid context

只有 A1 已经通过，才值得跑 A2。

* * *

十八、工程成本判断
---------

SOKA-Lite 不增加：

*   KNN；
*   radius search；
*   expanded support；
*   第二次 KPConvX；
*   global token；
*   full attention matrix。

主要新增：

*   数次 `scatter_add_`；
*   `[M, K, 6]` descriptor；
*   一个 `6→16→1` MLP；
*   一次 attention logit 加法。

以 stage2 约 10000 点估算：

10000×43×6×4≈9.8 MB10000\\times43\\times6\\times4 \\approx 9.8\\text{ MB}

后续 stage 更小。

这只是 descriptor 的理论 FP32 大小，实际还可以：

*   使用当前 feature dtype；
*   逐项 scatter；
*   不长期保存中间 descriptor；
*   在 diagnostic 关闭时立即释放。

合理目标：

项目

目标

参数增加

`<0.01M`

峰值显存增加

`<10%`

iteration time 增加

`<12%`

完整训练时间

baseline 的 `1.05–1.15x`

batch size

继续保持 3

你的 v17 训练时间从约 24 小时上升到 44.52 小时，主要来自 original/expanded 两次聚合，而不是参数增加。SOKA 没有第二次聚合，所以不应接近 v17 的成本。(KPConvX全部实验结果汇总.md)

硬性 No-Go：

> 如果 SOKA 使 batch size 从 3 降到 2，或者训练时间超过 plain scale04 的 1.2 倍，先优化实现，不要直接长训。

* * *

十九、当前最合理的实施顺序
-------------

阶段 0：现在就可以做代码和单元测试
------------------

不占用 4090D 长训练。

必须通过：

1.  `soka_enabled=False` 与原 baseline 完全一致；
2.  零初始化 SOKA 与原 checkpoint 输出差 `<1e-6`；
3.  shuffle neighbor order 后 descriptor 不变；
4.  packed batch 中不同 cloud 不串数据；
5.  shadow neighbor 不计入 occupancy；
6.  所有空 cell 无 NaN；
7.  mixed precision 下无溢出；
8.  500 iteration 无持续显存增长。

我已经对你重点源码做过 `py_compile` 静态检查：

    textkpconvx_base.py
    kpnext_blocks.py
    torch_pyramid.py
    da_kpnext_blocks.py
    kpx_v16.py
    kpx_v17.py
    kpnext_base.py
    kpnext_blocks.py

这些文件当前均通过 Python 语法编译。

但还需要你实际 PointOps/CUDA 环境做 forward/backward。

阶段 1：零训练 occupancy audit
------------------------

使用：

    textscale04 baseline model_last / epoch 200

因为它的固定全场景结果 `0.6939` 高于 random-val 选出的 epoch193 `0.6863`，不应再用 random best 作为机制诊断 checkpoint。(KPConvX全部实验结果汇总.md)

导出：

*   occupancy entropy；
*   empty-cell ratio；
*   mean radial distance；
*   mean assignment error；
*   每 stage 分布；
*   正确点和错误点分布；
*   各类别分布。

尤其检查：

    texttable
    beam
    door
    window
    column
    sofa

scale04 对 door、sofa、window、column 提升很大，但 table 回落，beam IoU 仍为 0。(KPConvX全部实验结果汇总.md)

### 对 beam 的重要提醒

如果 beam 在 `grid=0.04` 后本身保留点数过少，或者进入 stage2 后已经几乎消失：

> SOKA、KSPC 和 SPP 都无法从不存在的点中恢复 beam。

因此先统计：

    text每个 room 中 beam 原始点数
    GridSample 后 beam 点数
    stage1/2/3 对应 beam 点覆盖数

这比直接期望 SOKA 修复 beam 更重要。

阶段 2：低成本 continuation probe
---------------------------

从 scale04 epoch200 加载：

    text实验 A：baseline continuation 30-40 epoch
    实验 B：SOKA continuation 30-40 epoch

必须有 matched baseline continuation，否则无法区分：

*   SOKA；
*   额外训练；
*   低学习率微调；

的收益。

训练损失保持：

    textCE + Lovasz

不加辅助损失。

阶段 3：通过后才完整训练
-------------

Go 条件建议：

*   固定内部 dev protocol 提升至少 `+0.003`；
*   多数 dev rooms 方向一致；
*   iteration time `<1.15x`；
*   bias/base-logit RMS 不超过约 0.3；
*   attention 没有大量饱和到 `<0.05` 或 `>0.95`；
*   提升不只由 beam 单类驱动。

正式 200 epoch 成功线：

fixed Area5 mIoU≥0.6939+0.005\=0.6989\\text{fixed Area5 mIoU} \\ge 0.6939+0.005 = 0.6989

强结果线：

≥0.7019\\ge 0.7019

即相对 scale04 至少 `+0.008`。

* * *

二十、不要用 Area5 反复选择 SOKA 结构
-------------------------

这是论文可信度上最需要控制的一点。

你已经基于 Area5 做了很多版本。如果现在继续用 Area5 选择：

*   descriptor 选哪几个；
*   hidden dim；
*   stage；
*   是否加入 KSPC；
*   pooling 类型；

Area5 会逐渐变成开发集。

建议：

    text从 Areas 1/2/3/4/6 固定划出一组 dev rooms

用于：

*   occupancy audit；
*   30-40 epoch probe；
*   A1/A2 选择；
*   止损。

Area5 只在设计冻结后进行一次正式确认。

否则即使 fixed Area5 提升，也可能是研究过程对 Area5 的间接过拟合。

* * *

二十一、我现在对两个方案的最终判断
-----------------

主方案：SOKA-Lite
-------------

**建议实现。**

定义严格限制为：

    text固定 scale04 KNN support
    已有 nearest-kernel assignment
    per-kernel-cell occupancy + distance descriptors
    零初始化 scalar logit bias
    encoder stage2-5
    一次 KPConvX aggregation
    CE + Lovasz

这是：

*   可实现的；
*   与当前代码高度匹配的；
*   不重复 DA-meta 的；
*   不重复 support 的；
*   不需要改整个 Pointcept 管线的；
*   对 4090D 成本友好的。

SPP-IPCs / KSPC
---------------

**不建议直接移植 SPP-IPCs。**

建议把它作为：

    textSOKA 成功后的备用消融：
    Kernel-Shell Pyramid Context

并明确写成：

> 受 SPP-IPCs 多尺度池化思想启发，但针对 KPConvX 的 kernel-shell geometry 重新设计。

不要声称这是 ADNet SPP-IPCs 的直接复现。

* * *

二十二、是否需要 ChatGPT 深度研究辅助
-----------------------

**实现 SOKA 本身不需要再做大范围 Deep Research。** 当前源码已经足以确定：

*   接口；
*   数据流；
*   插入位置；
*   mask；
*   初始化；
*   显存风险；
*   与已有实验的区别。

Deep Research 最值得用于两个狭窄问题：

1.  **新颖性查重**  
    系统检索是否已有工作明确使用：
    
    *   kernel-cell occupancy；
    *   nearest-kernel assignment statistics；
    *   geometry-conditioned additive bias；
    *   kernel-point attention calibration。
    
    这决定论文能否把 SOKA 作为主创新，而不是仅仅工程变体。
    
2.  **SPP-IPCs 可复现性核验**  
    重点寻找：
    
    *   作者代码；
    *   补充材料；
    *   作者学位论文；
    *   窗口大小和 stride；
    *   点排序方式；
    *   packed/variable-length 实现。

目前论文正文不足以支撑 literal SPP-IPCs 的可靠复现，因此在没有找到源码或补充细节前，不应把它加入第一次长训练。

**最终工程建议：先完成独立 `kpconvx_soka` 分支和单元测试；第一轮只跑 SOKA-Lite，不加入 KSPC/SPP-IPCs。**

问题
--

你的意思是让我先放置v17，去完成你现在这个创新吗，之后可以和我现在的合并吗，那你给我讲讲你这个想法和我现在的区别是什么，你这个的核心创新点是什么，你用正常的，小白都可以听懂的大白话和你的书面语，两种方式讲解，也就是正常书面加上一个大白话的解释，让我能够轻松理解，如果是要实现你的创新，你给我实现的大纲，和具体的实现实现步骤，验收的标准，关键的细节，不用全部实现。

回答
--

先给明确结论
------

**不是让你中止正在训练的 scale04-v17。**

正确顺序是：

1.  **让当前 scale04-v17 跑完。**
2.  **暂时冻结 v17 设计**，不要继续做 v17b、改 alpha、改 gamma、换 gate、再扩 support。
3.  在 v17 训练期间，完成 SOKA 的代码框架、单元测试和低成本诊断，但先不启动完整 200 epoch。
4.  scale04-v17 出结果后，先分别判断：
    *   v17 在正确物理尺度下是否仍然有效；
    *   SOKA 单独是否有效。
5.  **只有 v17 和 SOKA 单独都通过，才做一次 SOKA+v17 合并实验。**

这是“暂挂 v17 等待裁决”，不是“直接放弃 v17”。

当前最强可信结果是 scale04 plain KPConvX 的固定全场景 `0.6939`；旧尺度 v17 的固定结果是 `0.6685`，虽然比旧 baseline 高约 `0.0130`，但训练成本约为 baseline 的 `1.85×`。当前正在训练的 scale04-v17，正是用来判断 v17 的收益能否在正确物理尺度下保留。(KPConvX全部实验结果汇总.md)

你已经预先规定了 scale04-v17 的成功线：固定 mIoU 至少达到 `0.6989`，并且重点类别不能明显下降；如果达不到，就不应继续扩展 dual-support 主线。(KPConvX全部实验结果汇总.md)

* * *

一、用一句话区分 v17 和 SOKA
-------------------

书面表述
----

*   **v17 改变的是邻域支持集合和聚合路径。**
*   **SOKA 改变的是原有邻域内部各个 kernel cell 的注意力权重。**

大白话
---

*   **v17：让模型再往外多看一圈，然后把“多看这一圈得到的信息”加回来。**
*   **SOKA：不让模型多看，只告诉模型“你现在看到的这些点，分别落在哪些核点区域里、落得准不准、哪块太挤”，然后重新决定这些区域应该有多大话语权。**

可以把 KPConvX 想象成一个有 43 个格子的圆形会议室：

*   每个邻居点会被安排到距离它最近的一个格子；
*   原始 KPConvX 根据中心点的特征，决定每个格子的发言权；
*   v17 是在会议室外再找一批人进来，另外开一次讨论；
*   SOKA 是不加人，而是检查现有 43 个格子里：
    *   每格坐了多少人；
    *   人是不是坐在格子中心附近；
    *   人是不是只是勉强被分到这个格子；
    *   人群是不是全挤在某几个方向；
*   然后再微调每个格子的发言权。

* * *

二、为什么现在不应该继续改 v17，但也不能马上扔掉它
---------------------------

书面表述
----

v17 当前有两个相反证据：

### 支持 v17 的证据

旧 `0.02m` 配置下，v17 在统一固定全场景协议中：

0.6556→0.66850.6556 \\rightarrow 0.6685

提升约：

+0.0130+0.0130

而且 fixed best 和 fixed last 很接近：

0.6685/0.66770.6685 / 0.6677

说明固定协议结果不像 random-crop best 那样只依赖一个偶然峰值。(KPConvX全部实验结果汇总.md)

### 反对继续盲目扩展 v17 的证据

v17 训练时间达到 `44.52h`，而旧 baseline 是 `24.10h`。参数量只增加了一点，真正的成本来自 stage4 original/expanded 两次邻域聚合。(KPConvX全部实验结果汇总.md)

另外，推理期 alpha 从 0 调到 1，mIoU 变化不超过 `0.0008`，说明“把 expanded residual 调弱一点”不是有效改进方向。(KPConvX全部实验结果汇总.md)

最关键的是，scale04 plain baseline 已经达到 `0.6939`，比旧 baseline 提高了 `0.0384`。这说明旧 v17 的部分收益可能只是在补偿原来物理尺度过小，而不一定来自 dual-support 本身。(KPConvX全部实验结果汇总.md)

大白话
---

旧 baseline 原来可能“近视”，只能看到很小的一块地方。

v17 给它加了一副临时望远镜，所以结果变好了一些。

现在 scale04 相当于直接把 baseline 的正常视力矫正了，而且提升比 v17 更大。

所以现在要等 scale04-v17：

*   如果视力已经矫正后，望远镜仍然有帮助，v17 就有独立价值；
*   如果视力矫正后，望远镜不再有帮助，说明旧 v17 主要是在补救尺度错误。

因此，当前 scale04-v17 必须跑完，但不应该继续发明更多 v17 变体。

* * *

三、KPConvX 原来是怎么工作的
------------------

KPConvX 的论文核心机制是：

1.  在中心点周围放置一组有固定空间顺序的 kernel points；
2.  每个邻居被投影到距离它最近的 kernel point；
3.  中心点特征经过 MLP、归一化和 sigmoid，生成 kernel attention；
4.  kernel attention 调制 kernel weights；
5.  再对邻居特征进行深度卷积聚合。

论文明确说明，KPConvX 使用中心特征生成 kernel attention；同时采用 nearest-kernel projection、双 shell kernel points 和固定邻居数量，以降低计算成本。(Thomas\_KPConvX\_Modernizing\_Kernel\_Point\_Convolution\_with\_Kernel\_Attention\_CVPR\_2024\_paper.pdf) (Thomas\_KPConvX\_Modernizing\_Kernel\_Point\_Convolution\_with\_Kernel\_Attention\_CVPR\_2024\_paper.pdf)

你压缩包里的 plain KPConvX 基本也是这个顺序：

    text中心点特征
        ↓
    alpha_mlp
        ↓
    GroupNorm
        ↓
    sigmoid
        ↓
    kernel modulation
        ↓
    邻居到最近 kernel point 的分配
        ↓
    聚合

对应代码主要在：

    textpointcept/models/kpconvx/utils/kpnext_blocks.py

其中大致是：

    text约 520 行：modulations = self.alpha_mlp(pooled_feats)
    约 529 行：modulations = self.attention_act(modulations)
    约 549 行：get_neighbors_influences(...)

* * *

四、原始 KPConvX 缺少什么
-----------------

书面表述
----

原始 KPConvX 的 kernel attention 主要由中心点特征产生：

Aqfeat\=MLP⁡(fq)A^{feat}\_q=\\operatorname{MLP}(f\_q) Mq\=σ(Aqfeat)M\_q=\\sigma(A^{feat}\_q)

但 attention 并没有直接输入当前邻域在 kernel lattice 上的实际分布状态。

两个中心点的语义特征可能很相似，但邻域采样情况可能不同：

*   一个点周围邻居分布均匀；
*   一个点位于平面边缘，邻居只集中在一侧；
*   一个点位于细柱或横梁上，邻居集中在少数方向；
*   一个点处于稀疏区域，KNN 邻居被拉得很远；
*   一个点的邻居虽然被分配到某个 kernel point，但离该 kernel point 很远，属于“勉强匹配”。

SOKA 把这些信息显式提供给 kernel attention。

大白话
---

原始 KPConvX 会根据中心点的“身份和经验”，判断哪个方向重要。

但它没有直接看座位表。

它不知道：

*   左边这个格子坐了 5 个人还是 1 个人；
*   这些人是不是紧贴格子中心；
*   这些人是不是因为没有更合适的位置，才勉强坐到这个格子里；
*   整个邻域是一个平面、一条细线、一个角，还是杂乱物体。

SOKA 就是给它一张实时座位表。

* * *

五、SOKA 的核心创新点到底是什么
------------------

SOKA 的核心不能简单写成“引入密度信息”。

因为 ADNet 已经明确研究了不均匀密度，并通过 ALNS 动态调整邻域尺度，同时使用 DCAA、GSA 和 SPP-IPCs 做多尺度融合。(Adaptive local neighborhood search and dual attention convolution network.pdf)

SOKA 真正不同的核心是：

> **利用 KPConvX 已经计算的 nearest-kernel assignment，构造与每个 kernel point 一一对应的局部占据状态，并将其作为几何校准偏置注入原生 kernel attention logits。**

可以拆成四点。

* * *

创新点 1：把“邻居到 kernel point 的分配结果”变成显式信息
-------------------------------------

### 书面表述

KPConvX 原本已经知道每个邻居属于哪个 kernel point：

kqj\=arg⁡min⁡k∥(pj−pq)−p~k∥2k\_{qj} = \\arg\\min\_k \\left\\| (p\_j-p\_q)-\\widetilde p\_k \\right\\|^2

但这个分配结果只用于选择 kernel weights，并没有被反馈给 attention 生成过程。

SOKA统计：

cqk\=∑j1(kqj\=k)c\_{qk} = \\sum\_j \\mathbf{1}(k\_{qj}=k)

即第 kk 个 kernel cell 中落入了多少邻居。

### 大白话

KPConvX 本来就已经做了座位分配，但分配完只拿座位号去找权重，没有把整张座位表交给管理人员。

SOKA把座位表也利用起来。

* * *

创新点 2：不是看总邻居数，而是看邻居在各 kernel cells 中怎么分布
----------------------------------------

这是非常关键的实现修正。

你当前 plain KPConvX 主要采用固定数量 KNN：

    textneighbor_limits = 12 / 16 / 20 / 20 / 20

因此，大多数点的邻居总数几乎相同。

只统计：

    text这个点有多少邻居

信息量很小。

真正有用的是：

    text这些邻居分别落在哪些 kernel cells
    每个 cell 中有多少邻居
    邻居离中心有多远
    邻居和对应 kernel point 匹配得好不好

### 大白话

每个班基本都有 20 个学生，所以“班里有多少人”没什么区分度。

有区分度的是：

*   20 个人是不是全挤在左边；
*   是不是分布成一条线；
*   是不是形成两个相交平面；
*   是不是有人离自己的座位特别远。

* * *

创新点 3：尺度归一化
-----------

SOKA 中的距离不能直接使用米或原始坐标。

必须根据每一层的 kernel radius 和 sigma 做归一化：

rqk\=Mean⁡j:kqj\=k∥pj−pq∥Rlr\_{qk} = \\frac{ \\operatorname{Mean}\_{j:k\_{qj}=k} \\|p\_j-p\_q\\| }{ R\_l } aqk\=Mean⁡j:kqj\=kdqj2σla\_{qk} = \\frac{ \\operatorname{Mean}\_{j:k\_{qj}=k} \\sqrt{d^2\_{qj}} }{ \\sigma\_l }

这里的 `Scale-Calibrated` 不表示再次预测动态 radius，而表示：

> 同一个几何数值在不同网络层中，都按照该层真实的物理感受野解释。

### 大白话

5 厘米在第一层可能很远，在第五层可能很近。

所以不能只告诉模型“距离是 5”，而要告诉它：

> 这个距离相当于当前层活动范围的百分之多少。

* * *

创新点 4：直接校准 KPConvX 原生 attention，而不是增加第二条特征分支
--------------------------------------------

原始 attention logits：

AqkcfeatA^{feat}\_{qkc}

SOKA 从几何描述符生成一个 kernel-cell bias：

Bqk\=ψ(gqk)B\_{qk} = \\psi(g\_{qk})

然后：

AqkcSOKA\=Aqkcfeat+BqkA^{SOKA}\_{qkc} = A^{feat}\_{qkc}+B\_{qk}

最后：

MqkcSOKA\=σ(AqkcSOKA)M^{SOKA}\_{qkc} = \\sigma(A^{SOKA}\_{qkc})

其中 BqkB\_{qk} 是一个标量，在该 kernel cell 的通道维度上广播。

### 大白话

原 KPConvX 说：

> 左边这个格子的发言权是 0.6。

SOKA根据实时座位状态补充：

> 左边这个格子虽然有人，但坐得很散、匹配很差，减 0.2。

最后得到：

> 左边格子的发言权是 0.4。

它不是再开一个会议，而是修改原会议里的发言权。

* * *

六、SOKA 和你已经做过的实验有什么本质区别
-----------------------

方案

改变什么

大白话

与 SOKA 的区别

scale04

所有点的全局物理尺度

把整体视力矫正

SOKA处理同一尺度下每个局部区域的实际采样状态

v13 / v13b DA-Radius

每个点的邻域半径或邻域图

有的点看远一点，有的点看近一点

SOKA不改变谁能进入邻域

v16-local DA-meta

点级统计生成 channel/shell bias

根据这个点整体密不密调通道

SOKA是每个 kernel cell 独立的空间状态

v16b controlled support

对候选邻居做 hard mask

把部分邻居踢出去

SOKA一个邻居都不删

v17 dual-support

original 和 expanded 两次聚合

多开一次会议，再把差值加回来

SOKA只有一次聚合

boundary gate

根据边界风险调特征

边界点特殊处理

SOKA不知道 GT 边界，也不需要边界损失

SGCA/global token

加远距离或全局上下文

让全房间的人交流

SOKA只处理当前局部 kernel cells

SPP-IPCs

多尺度池化融合

用大小不同的窗口看特征

第一版 SOKA 不做跨尺度池化

其中最容易混淆的是 v16-local。

v16-local
---------

    text整个点的局部统计
        ↓
    一个点级 meta
        ↓
    channel/shell bias

SOKA
----

    text第 1 个 kernel cell 的统计
    第 2 个 kernel cell 的统计
    ……
    第 43 个 kernel cell 的统计
        ↓
    43 个空间位置对齐的 bias
        ↓
    对应修改 43 个 kernel attention

### 大白话

v16-local 像是在说：

> 这个教室总体比较挤，所以全班都调整一下。

SOKA 像是在说：

> 左边第三排很挤，右边第一排很空，后排的人离座位很远，分别调整。

* * *

七、SOKA 之后能不能和 v17 合并
--------------------

技术上：可以
------

二者作用位置不同：

    textSOKA：
    生成 kernel weights 之前
    修改 attention logits
    
    v17：
    生成 kernel weights 之后
    分别对 original 和 expanded support 聚合
    再计算 residual

未来可以形成：

    textoriginal support
          ↓
    nearest-kernel geometry
          ↓
    SOKA occupancy descriptor
          ↓
    校准后的 kernel attention
          ↓
    校准后的 kernel weights
          ├──────── original aggregation
          └──────── expanded aggregation
                             ↓
                       v17 residual

在你当前 hybrid 代码里，SOKA 的合并位置可以放在：

    textpointcept/models/kpconvx_hybrid/da_kpnext_blocks.py

大致位于：

    pythonmodulations = self.alpha_mlp(pooled_feats)
    
    # GroupNorm 后
    # attention_act 前
    modulations = modulations + soka_bias
    
    modulations = self.attention_act(modulations)

这与现有 DA-meta bias 的插入位置类似，但 SOKA bias 的来源和维度完全不同。

* * *

科学上：现在不能直接合并
------------

正确的比较矩阵应该是：

实验

scale04

SOKA

v17

A

✓

B

✓

✓

C

✓

✓

D

✓

✓

✓

你现在已经有 A，B 正在训练。

下一步完成 C。

只有 B、C 都有效，才跑 D。

这样可以计算交互效应：

I\=mIoUD−mIoUB−mIoUC+mIoUAI = mIoU\_D-mIoU\_B-mIoU\_C+mIoU\_A

解释：

*   I\>0I>0：两者有协同；
*   I≈0I\\approx0：两者基本独立；
*   I<0I<0：两者互相冲突或功能重叠。

大白话
---

先分别确认两种药单独有效，再考虑一起吃。

不能一开始就把两种药混在一起，否则结果变好或变差，都不知道是谁造成的。

* * *

八、当前建议的工作安排
-----------

现在
--

### GPU

继续运行：

    textscale04-v17

不要终止。

### 代码开发

建立干净的：

    textkpconvx_soka

完成：

*   模块接口；
*   descriptor 统计；
*   baseline 等价性；
*   单元测试；
*   小 batch forward/backward。

暂时不启动完整训练。

* * *

scale04-v17 出结果后
----------------

### 情况 1：v17 不到 0.6989

结论：

*   v17 在正确尺度下没有足够独立收益；
*   暂停 support 路线；
*   不做 SOKA+v17；
*   SOKA 在 plain scale04 上单独验证。

### 情况 2：v17 达到 0.6989，但成本仍约 1.8×

结论：

*   v17 有真实机制价值；
*   但成本过高；
*   SOKA仍值得独立测试，因为它可能以更低成本获得相似收益。

### 情况 3：v17 达到 0.6989，SOKA也单独通过

才做：

    textscale04 + SOKA + v17

而且只做一次预注册的合并，不再围绕合并版本连续调参。

* * *

九、SOKA 第一版的正式定义
---------------

第一版建议叫：

> **SOKA-Lite：Scale-Calibrated Occupancy-Aware Kernel Attention**

严格限定为：

    text固定 scale04 数据尺度
    固定原始 KNN support
    使用已有 nearest-kernel assignment
    构造 per-kernel-cell 几何描述符
    生成 scalar kernel bias
    GroupNorm 后、sigmoid 前注入
    一次 KPConvX 聚合
    不增加辅助损失
    只修改 encoder KPConvX blocks

第一版不加入：

*   v17；
*   DA-Radius；
*   DA-meta；
*   boundary；
*   SPP-IPCs；
*   KSPC；
*   global token；
*   Mamba；
*   新损失。

* * *

十、推荐的 SOKA 描述符
--------------

对于每个查询点 qq、每个 kernel cell kk，第一版使用六个量。

1\. Occupancy fraction
----------------------

pqk\=cqkHq+ϵp\_{qk} = \\frac{c\_{qk}} {H\_q+\\epsilon}

表示当前邻居中有多大比例落在第 kk 个 kernel cell。

**大白话：这个格子坐了多少人。**

* * *

2\. Mean normalized radial distance
-----------------------------------

rqk\=Mean⁡∥pj−pq∥Rlr\_{qk} = \\frac{\\operatorname{Mean}\\|p\_j-p\_q\\|} {R\_l}

**大白话：这个格子里的人整体离教室中心有多远。**

* * *

3\. Mean assignment error
-------------------------

aqk\=Mean⁡dqj2σla\_{qk} = \\frac{ \\operatorname{Mean}\\sqrt{d^2\_{qj}} }{ \\sigma\_l }

其中 dqj2d^2\_{qj} 是邻居到最近 kernel point 的平方距离。

**大白话：这些人和分配给他们的座位匹配得准不准。**

* * *

4\. Assignment-error variance
-----------------------------

vqk\=Var⁡(dqj2σl)v\_{qk} = \\operatorname{Var} \\left( \\frac{\\sqrt{d^2\_{qj}}}{\\sigma\_l} \\right)

**大白话：这个格子里的人是不是有的坐得很准、有的坐得很离谱。**

* * *

5\. Occupancy entropy
---------------------

Eq\=−∑kpqklog⁡(pqk+ϵ)log⁡KE\_q = - \\frac{\\sum\_k p\_{qk}\\log(p\_{qk}+\\epsilon)} {\\log K}

对所有 kernel cells 广播。

**大白话：所有人是集中在少数方向，还是分布得比较均匀。**

* * *

6\. Kernel shell radius
-----------------------

sk\=∥p~k∥Rls\_k = \\frac{\\|\\widetilde p\_k\\|}{R\_l}

表示当前 kernel point 位于中心、内层 shell 还是外层 shell。

**大白话：这个座位是在教室中心、内圈还是外圈。**

最终：

gqk\=\[pqk,rqk,aqk,vqk,Eq,sk\]g\_{qk} = \[p\_{qk},r\_{qk},a\_{qk},v\_{qk},E\_q,s\_k\]

形状：

    text[M, K, 6]

* * *

一个重要修正：不要把 empty flag 当成主要输入
----------------------------

因为一个 kernel cell 如果完全没有邻居，当前卷积本来就不会使用它的权重。

所以：

    textempty cell 的 bias

对当前输出通常没有直接作用。

第一版的重点应是：

*   已经有邻居的 cell 中有多少邻居；
*   这些邻居分布和匹配质量怎么样；
*   整个邻域的占据形状怎么样。

空 cell 比例可以监控，但不应当成为核心输入。

* * *

十一、具体代码实现大纲
-----------

目录结构
----

建议从干净的 plain KPConvX 复制最小代码，而不是继承整个 hybrid：

    textpointcept/models/kpconvx_soka/
    ├── __init__.py
    ├── kpconvx_soka.py
    └── soka_blocks.py

原因是 hybrid 已经包含：

*   DA-Radius；
*   support mask；
*   DA-meta；
*   dual-support；
*   boundary；
*   各类 monitor。

直接在 hybrid 上开发，论文归因会非常混乱。

* * *

第一步：扩展几何返回值
-----------

当前：

    pythonget_neighbors_influences(...)

内部已经计算：

    pythonnn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=2)

但只返回：

    pythoninfluence_weights, neighbors, neighbors_1nn

建议改成可选接口：

    pythondef get_neighbors_influences(
        ...,
        return_geometry=False,
    ):
        ...
    
        if return_geometry:
            return (
                influence_weights,
                neighbors,
                neighbors_1nn,
                nn_sq_dists,
                valid_mask,
            )
    
        return influence_weights, neighbors, neighbors_1nn

或者拆成：

    pythonget_neighbor_geometry(...)
    aggregate_with_geometry(...)

第一种改动更小。

* * *

第二步：正确处理 shadow neighbors
-------------------------

你当前 pyramid 中会把 `-1` 替换为：

    pythoncur_points.shape[0]

也就是尾部的 shadow index。

因此必须构造：

    pythonvalid_mask = neighb_inds < s_pts.shape[0]

所有 occupancy、距离均值和方差都必须乘 `valid_mask`。

否则 shadow point 的 `1e6` 坐标会污染：

*   radial distance；
*   assignment error；
*   occupancy cell；
*   entropy。

* * *

第三步：使用 scatter\_add，不要使用 one-hot
--------------------------------

不要写：

    pythonF.one_hot(neighbors_1nn, num_classes=K)

它会建立：

    text[M, H, K]

的大张量。

应使用扁平索引：

    pythonflat_index = (
        torch.arange(M, device=device)[:, None] * K
        + neighbors_1nn
    ).reshape(-1)

然后：

    pythoncounts.scatter_add_(
        0,
        flat_index,
        valid_mask.reshape(-1).to(dtype),
    )

类似地统计：

    pythonradial_sum
    assignment_sum
    assignment_sq_sum

最后 reshape：

    python[M * K] -> [M, K]

### 大白话

不要给每个邻居做一张 43 格的完整表格。

直接告诉程序：

> 第 3 个点属于第 17 格，把第 17 格的计数加一。

这样省显存。

* * *

第四步：加入小型 SOKA MLP
-----------------

    pythonself.soka_mlp = nn.Sequential(
        nn.Linear(6, 16),
        nn.SiLU(),
        nn.Linear(16, 1),
    )

最后一层必须零初始化：

    pythonnn.init.zeros_(self.soka_mlp[-1].weight)
    nn.init.zeros_(self.soka_mlp[-1].bias)

可以使用有界 bias：

    pythonsoka_bias = 2.0 * torch.tanh(self.soka_mlp(desc))

形状：

    text[M, K, 1]

* * *

第五步：调整 forward 数据流
------------------

当前代码顺序是先生成 attention，再计算 geometry。

SOKA改成：

    python# 1. neighbor features
    neighbor_feats = ...
    
    # 2. geometry，只计算一次
    (
        influence_weights,
        relative_neighbors,
        neighbors_1nn,
        nn_sq_dists,
        valid_mask,
    ) = self.get_neighbors_influences(
        q_pts,
        s_pts,
        neighb_inds,
        return_geometry=True,
    )
    
    # 3. geometry descriptor
    soka_desc = self.build_soka_descriptor(
        relative_neighbors,
        neighbors_1nn,
        nn_sq_dists,
        valid_mask,
    )
    
    # 4. 原 KPConvX logits
    modulations = self.alpha_mlp(pooled_feats)
    
    if self.mod_grp_norm:
        modulations = apply_group_norm(modulations)
    
    M = q_pts.shape[0]
    modulations = modulations.view(M, self.K, self.ch_per_grp)
    
    # 5. SOKA bias
    soka_bias = 2.0 * torch.tanh(self.soka_mlp(soka_desc))
    
    # 6. sigmoid 前相加
    modulations = modulations + soka_bias
    modulations = self.attention_act(modulations)
    
    # 7. 后续原 KPConvX 聚合不变
    ...

最重要的是：

> `nn_sq_dists` 和 `neighbors_1nn` 不要重新计算第二次。

KPConvX 中最重的一部分是：

    text[M, H, K, 3]

邻居与所有 kernel points 的距离。

如果为了 SOKA 再算一遍，工程优势就被破坏了。

* * *

第六步：处理 shared kernel geometry cache
-----------------------------------

你的 KPConvX 支持：

    pythonshared_kp_data

同一 stage 内多个 block 可能共享：

*   kernel points；
*   neighbor positions；
*   nearest-kernel assignment；
*   influence weights。

SOKA还需要缓存：

    pythonshared_kp_data["nn_sq_dists"]
    shared_kp_data["valid_mask"]

必须确保：

*   缓存只在相同 query/support/neighbors 下复用；
*   新 batch 不复用旧 shape；
*   stage 切换后不串数据；
*   diagnostic tensor 不长期保留计算图。

* * *

第七步：加入 stage 开关
---------------

模型接口建议：

    pythonsoka_enabled=True
    soka_stages=(2, 3, 4, 5)
    soka_hidden_dim=16
    soka_bias_bound=2.0
    soka_monitor=True

开发时可以使用：

    pythonsoka_stages=(4,)

做 smoke test。

但正式第一版建议应用于：

> 所有真正使用 KPConvX attention 的 encoder blocks。

在你当前 `first_inv_layer=1` 配置下，通常对应后续的 attention stages；stem、前面的 KPConv/KPConvD 和 decoder 第一版保持不变。

* * *

十二、为什么必须零初始化
------------

书面表述
----

初始时：

Bqk\=0B\_{qk}=0

因此：

ASOKA\=AfeatA^{SOKA}=A^{feat}

SOKA模型加载 baseline 权重后，初始输出应与 baseline 完全一致。

这意味着训练结果的变化来自 SOKA 学到的偏置，而不是因为新模块随机初始化破坏原模型。

大白话
---

刚装上 SOKA 时，它必须先闭嘴。

只有训练发现座位表真的有用，它才慢慢开始提出调整意见。

* * *

不要这样做
-----

    pythonbeta = nn.Parameter(torch.tensor(0.0))
    last_layer = zero_init
    
    bias = beta * soka_mlp(desc)

因为：

    textbeta = 0
    MLP 输出 = 0

两边可能同时拿不到有效梯度，形成死启动。

推荐：

    text最后一层零初始化
    倍率固定为 1

或者：

    text最后一层小随机初始化
    beta 初始化为 0

第一种更容易验证 baseline 等价性。

* * *

十三、训练损失
-------

第一版完全不加新损失：

L\=LCE+LLovasz\\mathcal L = \\mathcal L\_{CE} + \\mathcal L\_{Lovasz}

不要加入：

*   occupancy consistency loss；
*   boundary loss；
*   attention entropy loss；
*   compatibility loss；
*   auxiliary head。

### 大白话

先判断“座位表能不能帮助原模型”。

不要同时再加班级纪律分、边界分、邻居关系分，否则结果变了也不知道是谁起作用。

* * *

十四、代码级验收标准
----------

1\. Baseline 等价性
----------------

加载同一个 scale04 baseline checkpoint：

    pythonbase.eval()
    soka.eval()

在零初始化下：

    textmax_abs_diff < 1e-6

至少要求 logits 和最终输出一致。

这是最重要的正确性测试。

* * *

2\. 邻居顺序不变性
-----------

对同一邻域随机打乱 neighbor 顺序：

    textSOKA descriptor 前后差异 < 1e-6
    SOKA output 前后差异 < 1e-6

因为点云邻居没有自然顺序。

* * *

3\. Occupancy 守恒
----------------

对每个 query point：

∑kcqk\=Hqvalid\\sum\_k c\_{qk} = H\_q^{valid}

测试：

    textcounts.sum(dim=1) == valid_mask.sum(dim=1)

* * *

4\. Shadow neighbor 排除
----------------------

人为构造：

    text部分 neighbor index = N

要求：

*   不进入 occupancy count；
*   不进入均值和方差；
*   不产生巨大距离；
*   不产生 NaN。

* * *

5\. 空 cell 数值稳定
---------------

没有邻居的 kernel cell：

    textmean_radial = 0
    mean_assignment = 0
    variance = 0

不能出现：

    textNaN
    Inf
    0 / 0

* * *

6\. 梯度测试
--------

第一次 backward：

*   SOKA 最后一层应有非零梯度；
*   原 KPConvX 参数仍有梯度；
*   第二步后 SOKA 前层开始获得梯度。

* * *

7\. mixed precision 测试
----------------------

在 AMP 下：

*   `sqrt(nn_sq_dists)` 前 clamp；
*   entropy 中加 epsilon；
*   variance 不得因精度误差变成负数。

例如：

    pythonvariance = torch.clamp(second_moment - mean ** 2, min=0.0)

* * *

8\. shared cache 一致性
--------------------

同一 block：

    textcache on
    cache off

输出差异应小于数值误差。

* * *

十五、工程验收标准
---------

在 4090D 上先跑 500～1000 iteration。

指标

通过线

batch size

必须继续保持 3

峰值显存

不超过 baseline 约 10%

iteration time

不超过 baseline 约 15%

参数增加

应远低于 0.01M

显存随 iteration

不持续增长

forward/backward

无 NaN、无 device assert

硬性止损：

> 如果 SOKA 导致 batch size 从 3 降到 2，或 iteration time 超过 baseline 的 1.2×，先优化代码，不启动长训。

SOKA应该远低于 v17 的成本，因为 v17 慢的主要原因是 original/expanded 两次邻域聚合，而不是参数量。(KPConvX全部实验结果汇总.md)

* * *

十六、低成本 Go/No-Go 实验
------------------

第一步：零训练 occupancy audit
-----------------------

使用 scale04 epoch200 checkpoint，导出：

*   每层 occupancy entropy；
*   每层 occupied cell 数；
*   每个 cell 的 occupancy fraction；
*   mean radial distance；
*   mean assignment error；
*   prediction confidence；
*   是否预测正确；
*   类别和房间，仅用于分析。

主要问题是：

> 在原模型预测错误的点上，SOKA描述符是否提供了中心特征之外的额外信息？

可以训练一个很小的离线 logistic regression：

### 模型 A

    textconfidence
    entropy
    margin

预测当前点是否分类错误。

### 模型 B

    textconfidence
    entropy
    margin
    + SOKA descriptors

Go 条件之一：

    text错误预测 AUROC 提升 ≥ 0.03

或者：

    text错误预测 NLL 降低 ≥ 2%

并且不是只在单个房间或 beam 类上成立。

### 大白话

先不训练 SOKA，只检查座位表是不是和错误有关。

如果模型犯错时，座位表和正确点完全没区别，那这个想法很可能没有价值。

* * *

第二步：matched continuation probe
------------------------------

从相同 scale04 epoch200 checkpoint 出发：

    text实验 C0：baseline continuation 30～40 epoch
    实验 C1：SOKA continuation 30～40 epoch

必须有 C0。

否则无法区分：

    textSOKA 提升

和：

    text模型只是多训练了 40 epoch

建议先在 Areas 1/2/3/4/6 中固定一组 dev rooms，用它做开发判断，不要反复使用 Area5 调 descriptor。

Go 条件：

*   fixed dev mIoU ≥ matched baseline `+0.003`；
*   多数 dev rooms 方向一致；
*   提升不是只来自一个稀有类别；
*   iteration time ≤ baseline `1.15×`；
*   没有主要类别突然下降 `0.03` 以上。

预计两组 continuation 合计大约需要数小时到十余小时，仍明显低于一次 v17 的约 44.5 小时。

* * *

十七、正式训练验收标准
-----------

当前 scale04 baseline：

0.69390.6939

因此 SOKA 第一条正式成功线：

mIoU≥0.6989mIoU \\ge 0.6989

即：

+0.005+0.005

较强结果线：

mIoU≥0.7019mIoU \\ge 0.7019

即：

+0.008+0.008

必须同时满足：

1.  固定 Area5 全场景 identity protocol 提升；
2.  `model_last` 或预注册 checkpoint 提升；
3.  不能只提高 random-crop best；
4.  多数房间方向一致；
5.  不能由 beam 单类峰值主导；
6.  door、window、column、table、sofa 等关键类别不能系统性崩溃；
7.  训练成本保持在 baseline 约 `1.05～1.15×`；
8.  第一个正式 seed 过 `+0.005` 后，才追加另外两个 seed。

scale04 的 epoch200 固定结果比 random-val 选出的 epoch193 高 `0.0076`，已经证明 random best 不是可靠的 checkpoint 选择依据。(KPConvX全部实验结果汇总.md)

* * *

十八、SOKA 必须监控什么
--------------

性能指标
----

*   fixed mIoU；
*   mAcc；
*   OA；
*   13 类 IoU；
*   每个房间 IoU；
*   last checkpoint；
*   预注册的 last-k checkpoint；
*   iteration time；
*   peak memory。

机制指标
----

每个 stage 记录：

    textsoka_occ_mean/std
    soka_entropy_mean/std
    soka_radial_mean/std
    soka_assignment_mean/std
    soka_bias_mean/std
    soka_bias_abs
    base_logit_abs
    soka_to_base_ratio
    attention_low_ratio
    attention_high_ratio

定义：

Rl\=RMS⁡(Bl)RMS⁡(Alfeat)+ϵR\_l = \\frac{ \\operatorname{RMS}(B\_l) }{ \\operatorname{RMS}(A\_l^{feat})+\\epsilon }

建议：

*   初期 RlR\_l 接近 0；
*   训练早期不要快速超过 0.1；
*   后期若长期超过 0.5，要警惕 SOKA 已经替代原 attention，而不是校准；
*   sigmoid 输出大量接近 0 或 1，说明 attention 饱和。

* * *

十九、最可能失败的原因
-----------

1\. 中心特征已经隐式编码了 occupancy
-------------------------

原 KPConvX 的中心特征来自前面多层局部聚合，它可能已经知道邻域形状。

此时显式 SOKA只是重复信息。

这就是零训练 audit 必须先做的原因。

* * *

2\. SOKA学到 crop 边界，而不是物体几何
--------------------------

SphereCrop、fragment 边缘或 point cap 可能产生明显不完整 occupancy。

模型可能学会：

> 这种 occupancy 表示快到 crop 边缘了。

而不是：

> 这种 occupancy 表示门、窗或细柱。

因此需要对比：

*   fragment interior；
*   fragment boundary；
*   identity full-scene；
*   不同 fragment size。

* * *

3\. 固定 KNN 拉远邻居造成的伪信号
---------------------

稀疏区域中，KNN会找更远的点。

SOKA可能把扫描稀疏度误当成类别特征。

所以距离必须按 radius/sigma 归一化，并做轻微 density perturbation 测试。

* * *

4\. 43 个 kernel cells、最多 20 个邻居，occupancy 很稀疏
---------------------------------------------

由于：

    textK = 43
    H ≈ 12～24

很多 cells 天然不会被占据。

不能把“空 cell 多”简单解释为邻域质量差。

真正应关注的是：

*   被占据 cells 的数量；
*   occupied cells 的 count 分布；
*   匹配误差；
*   全局 entropy。

* * *

5\. SOKA偏置过强
------------

如果 SOKA很快主导原 attention，会破坏 KPConvX 已经学到的语义 modulation。

所以要：

*   零初始化；
*   使用有界 tanh；
*   监控 bias/base ratio；
*   不额外增加一个自由 beta 和复杂 gate。

* * *

二十、SPP-IPCs 现在怎么处理
------------------

第一版不要加入 SPP-IPCs。

ADNet 中 SPP-IPCs 是多尺度特征融合组件，它与 ALNS、DCAA 和 GSA 一起组成完整框架，而不是一个已经在 KPConvX 上单独验证的即插即用模块。(Adaptive local neighborhood search and dual attention convolution network.pdf)

SOKA通过以后，才考虑一个单独消融：

    textSOKA
        ↓
    kernel-shell pooling
        ↓
    Kernel-Shell Pyramid Context

但这应是：

    textA0 scale04
    A1 scale04 + SOKA
    A2 scale04 + SOKA + KSPC

不能第一版就同时加，否则归因又会回到早期 hybrid 的问题。

* * *

最终执行方案
------

当前不做的事
------

*   不停掉正在运行的 scale04-v17；
*   不继续设计新的 v17 变体；
*   不立即合并 SOKA 和 v17；
*   不同时加入 SPP-IPCs；
*   不在 hybrid 大框架上直接改出 SOKA；
*   不以 random-crop best 作为成功标准。

当前要做的事
------

1.  让 scale04-v17完成；
2.  从 plain `kpconvx` 建立独立 `kpconvx_soka`；
3.  复用已有 nearest-kernel geometry；
4.  构造 per-kernel occupancy、距离和匹配误差；
5.  用 `6→16→1` 零初始化 MLP；
6.  在 GroupNorm 后、sigmoid 前加入 bias；
7.  保持 support、loss、decoder 和训练协议不变；
8.  先通过代码验收和低成本 matched probe；
9.  SOKA单独达到固定协议 `+0.005` 后才跑完整多 seed；
10.  只有 scale04-v17 与 SOKA都单独成功，才做一次二者合并。

**最简单的大白话总结：**

> v17 是“多找一些邻居，再算一次”；SOKA 是“邻居不变，但让模型看懂这些邻居在 43 个核点区域里是怎么分布的”。  
> 当前 v17 不取消，但停止继续改。先让它跑完；SOKA独立开发、独立证明。两者都有效，最后再合并。
```