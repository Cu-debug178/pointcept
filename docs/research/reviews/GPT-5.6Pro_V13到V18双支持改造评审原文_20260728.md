# 直接判断

**V13 现在效果不明显，确实有“参数偏保守”的成分，但更主要的问题不是范围不够大，而是当前机制在固定 \(H\) 邻居约束下，很多点的邻接图根本没有发生变化。**

你当前的 V13：

```text
stage 3 原始范围：0.95～1.20
strength = 0.5
实际范围：0.975～1.10

stage 4 原始范围：0.90～1.30
strength = 0.5
实际范围：0.95～1.15
```

从半径数值上看确实比较温和。但即便把最大值从 1.15 增大到 1.5，只要查询仍然最多取最近的 \(H=20\) 个点，而且原本 20 个最近邻已经都在基础半径内，**扩大半径仍然一个新邻居也加不进来**。

因此我的结论是：

> **不要只把 V13 的 scale range 调大。应当把 V13 从“替换原始邻域”改成“保留原始邻域 + 额外候选邻域残差”，并让额外候选数大于 H。**

你当前实验中 baseline 为 0.65557、V13 为 0.65854，只提升约 0.30 个 mIoU 点；V16b 复跑又从 0.66707 回到 0.65596，说明目前信号很可能没有超过训练方差。相反，保留 identity support 的 V17 结果更有希望，这支持“原始支持域不能被动态分支直接替换”的判断。(prompt.md)

---

# 一、为什么当前 V13 很难产生“一眼可见”的作用

## 1. 固定 H 形成了机制上限

设第 \(H\) 个最近邻的距离为：

\[
d_H(q)
\]

基础半径为：

\[
r_0
\]

动态半径为：

\[
r_q=s_qr_0
\]

当：

\[
d_H(q) < r_0
\]

说明基础半径内已经至少有 \(H\) 个点。此时无论把半径扩大为：

\[
1.1r_0,\quad 1.5r_0,\quad 2r_0
\]

最终仍然只能选择最近的 \(H\) 个点，邻居索引完全相同。

所以当前 V13 实际上主要在三种情况下产生作用：

1. **缩小半径**，把原来的部分 KNN 变成 shadow；
2. 基础半径不足 \(H\) 个点时，扩大半径填补 shadow；
3. CUDA ball query 的点选择顺序与 KNN 排序不完全相同时，邻居集合发生少量改变。

这意味着“扩大稀疏区域感受野”的表述目前并不充分。更准确的表述应该是：

> V13 根据局部密度自适应控制固定 KNN 支持域的有效占用率，对远距离邻居进行动态保留或屏蔽。

KPConvX 原论文已经使用固定邻居上限、nearest-kernel assignment、两层 shell 和 \(r=2.1\times\) grid size，并令 influence sigma 与半径相同；这些都不能作为你的新增贡献重复宣称。(Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf)

---

## 2. 你的深层 stage 本来就已经很大

Pointcept 0.02m 配置中，stage 3、stage 4 的标称支持半径约为：

```text
stage 3：0.4898 m
stage 4：1.0776 m
```

但实际有效感受野不只是一层半径，因为每一阶段有多个连续 block，并且已经经历多次 pooling。

因此最后两个 stage 的特征往往已经覆盖：

- 大型家具；
- 墙体局部；
- 房间级结构；
- 多次卷积传播后的更大区域。

此时再把 stage 4 从 1.08m 扩到 1.24m，语义差异未必明显。**动态几何支持域更应该用在第三、第四分辨率，而不是只用在最后两层。**

建议将：

```python
stages = (3, 4)
```

改为：

```python
stages = (2, 3)  # 0-based
```

即作用在第三、第四个分辨率层。这里仍有足够的局部几何结构，同时计算量又不像前两层那么高。

---

## 3. 只改变邻域半径，没有同步改变 kernel 几何

这是另一个容易被忽视的问题。

KPConvX 不只是聚合邻居，它会将邻居相对坐标分配给最近 kernel point。假设你扩大邻域，却保持：

```text
kernel point 半径不变；
sigma 不变；
相对坐标不归一化。
```

那么新加入的远距离点往往会：

- 集中分配到外层 kernel；
- 与外层 kernel 距离较大；
- 在 linear influence 下受到明显衰减；
- 在 constant influence 下又可能获得过强权重。

所以动态邻域和 kernel geometry 应当联合变化。推荐两种等价思路。

### 方案 A：归一化相对坐标

```python
relative_pos = neighbor_pos - query_pos
relative_pos = relative_pos / query_scale
```

然后仍然使用基础 kernel points 和基础 sigma。

### 方案 B：动态缩放 kernel 和 sigma

```python
kernel_points_q = base_kernel_points * query_scale
sigma_q = base_sigma * query_scale
```

这样动态半径扩大后，新邻居仍然能在完整 kernel shell 中得到合理映射，而不是全部拥挤到最外层。

---

## 4. 每个 fragment 独立做 10/90 percentile，可能不稳定

你的密度归一化是：

```text
每个 packed cloud 内计算 density 的 10/90 percentile
```

这会产生一个问题：

> 同一个局部区域，仅仅因为所在 fragment 的整体密度分布不同，就可能得到不同的动态半径。

Pointcept 测试阶段还会把同一场景经过不同增强、GridSample 和 fragment 切分。不同 TTA 视图的 voxel 分布和 fragment 组成可能改变，导致：

```text
同一个原始点
→ 在不同 TTA 中得到不同 percentile
→ 得到不同 radius scale
→ 邻接图额外波动
```

建议改成与层级网格尺寸无关的无量纲密度：

\[
\rho_q=
\frac{\operatorname{meanKNNdistance}(q)}
{\operatorname{grid\_size}_l}
\]

然后在训练集上预先统计每层固定的：

```text
q10_l
q90_l
```

训练、验证、测试全部使用相同的全局统计量。

更严格的做法是：

```text
整场景坐标
→ 计算一次 density 和 scale
→ 再进行 test fragment
→ 用原始 index 把 scale 分发给每个 fragment
```

不要在每个 fragment 内重新定义“稠密”和“稀疏”。

---

# 二、让模块“一眼看出是否有用”的正确大改方案

我不建议直接做：

```text
V13 scale max 从 1.15 改成 1.8
```

因为 H 仍然是 20，扩大部分可能依旧没有任何拓扑变化。

建议做一个 **强机制诊断版 V18**：

## 1. 原始支持域保持不动

```python
base_neighbors = original_knn_20
base_feat = KPConvX(base_neighbors)
```

这条路径与 baseline 完全一致。

## 2. 建立更大的候选集合

```python
candidate_k = {
    stage2: 40,
    stage3: 48,
}
```

必须满足：

\[
K_{\text{candidate}}>H_{\text{base}}
\]

否则无法真正引入新邻居。

## 3. 额外分支只使用环形邻域

设基础半径为 \(r_0\)，动态扩张半径为 \(r_q\)，残差分支只选择：

\[
r_0<d_i\leq r_q
\]

而不是再次处理全部基础邻居。

```python
ring_mask = (
    (distance > base_radius)
    & (distance <= adaptive_radius)
)
```

这样新增分支明确表示：

> 只编码基础支持域以外，由动态扩张带来的新上下文。

不会把原始邻居计算两次。

## 4. 残差融合

\[
F_{\text{out}}
=
F_{\text{base}}
+
\gamma_qF_{\text{ring}}
\]

诊断版本先不要把 \(\gamma\) 初始化为 \(10^{-3}\)，因为这会让分支初期几乎不可见。

推荐诊断配置：

```python
stages = (2, 3)

candidate_k = (40, 48)
ring_k = (8, 12)

stage2_scale = (0.75, 1.45)
stage3_scale = (0.70, 1.55)

strength = 1.0
ring_only = True

gamma_mode = "fixed"
gamma = 0.25
```

这套参数不是最终论文参数，而是用来回答一个问题：

> 动态附加支持域本身究竟有没有价值？

如果这套强版本仍然几乎不改变结果，那么再微调 0.95、1.10、1.20 没有意义。

如果强版本明显提升或明显下降，说明分支确实在工作，再收缩为正式版本：

```python
stage2_scale = (0.85, 1.30)
stage3_scale = (0.85, 1.40)

gamma_mode = "learnable"
gamma_init = 0.05  # 或 0.1
```

---

# 三、不要先训练，先检查邻接图到底改变了多少

这是你现在最应该做的事情。

针对每个 stage 记录以下指标：

```text
1. 基础半径内的平均有效邻居数；
2. 自适应半径内的平均有效邻居数；
3. valid_count == H 的点所占比例；
4. adaptive 与 base 邻居集合的 Jaccard distance；
5. 每个 query 新增邻居数量；
6. 每个 query 删除邻居数量；
7. 按 density 四分位统计上述指标；
8. d_H / r_base 的分布。
```

定义邻接图变化率：

\[
\operatorname{ChangeRate}
=
1-
\frac{
|\mathcal N_{\text{base}}\cap\mathcal N_{\text{adapt}}|
}{
|\mathcal N_{\text{base}}\cup\mathcal N_{\text{adapt}}|
}
\]

我建议使用以下判断标准：

| 统计结果 | 结论 |
|---|---|
| 超过 85% 的点已经 `valid_count==H` | 扩大半径基本无效 |
| 平均新增邻居数小于 1 | 当前 V13 没有足够机制强度 |
| Jaccard change 小于 10% | 不要继续调 scale，先改候选图 |
| 新增邻居集中在最稀疏 10% 点 | 可以改成稀疏区域专用分支 |
| 所有区域都大量新增远邻居 | 可能引入语义污染，需要 gate |

还应做三个单元测试：

```text
scale=1.0：
输出必须与 baseline 数值等价。

scale=0.5：
有效邻居数应显著下降，输出必须变化。

scale=2.0，candidate_k>H：
新增邻居数必须显著增加，输出必须变化。
```

如果 `scale=2.0` 后新增邻居仍然接近 0，说明当前实现受 H/candidate graph 限制，而不是 scale 不够大。

---

# 四、关于 13 TTA：它真的可能提升，但绝不是必然提升

首先需要区分：

> **13 TTA 是测试时增强和投票，不是训练数据增强。它不会让 checkpoint 本身变强，只会通过多次推理和预测聚合改变最终测试结果。**

当前 Pointcept 的 S3DIS 配置中，常见的 13 个测试视图是：

```text
4 个旋转：
0°、90°、180°、270°

4 个旋转 × scale 0.95

4 个旋转 × scale 1.05

1 个 flip
```

也就是：

\[
4+4+4+1=13
\]

Pointcept 当前官方配置确实仍然保留了这一组 13-view TTA。([GitHub](https://github.com/Pointcept/Pointcept/blob/main/configs/concerto/semseg-ptv3-large-v1m1-3d-s3dis-ppt.py))

Pointcept 会将每个增强版本继续体素化和切成 fragments，分别推理，再按照原始点 index 累加预测，最后取最大类别。([GitHub](https://github.com/Pointcept/Pointcept/issues/300?utm_source=chatgpt.com))

KPConvX 原论文也明确支持 voting，理由是模型面对不同房间方向时结果可能波动；多次投票可以降低方向带来的方差并让预测更平滑。([OpenAccess](https://openaccess.thecvf.com/content/CVPR2024//papers/Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf?utm_source=chatgpt.com))

## 但它可能不提升，甚至下降

出现下降的常见原因包括：

### 1. 模型不具有良好的 scale equivariance

0.95 和 1.05 缩放会改变：

```text
点与固定 GridSample 网格的关系；
voxel 内被保留的代表点；
KNN 距离；
动态密度；
动态 radius；
fragment 数量。
```

对普通模型，这可能只是小扰动；对 V13 来说，它直接改变密度估计和支持域，所以 scale TTA 与 V13 的耦合更强。

### 2. 法向量没有同步旋转或翻转

坐标变换后，normal 必须执行对应的旋转和反射。否则：

```text
coord 表示旋转后的几何；
normal 仍表示旋转前的方向。
```

模型输入产生矛盾。

### 3. 测试 pipeline 与训练 pipeline 不一致

Pointcept 官方社区也出现过测试结果异常偏低，最后通过移除重复 GridSample、让测试流水线与训练预处理一致而恢复正常结果。([GitHub](https://github.com/Pointcept/Pointcept/issues/307?utm_source=chatgpt.com))

### 4. 差的视图被简单平均

假设 identity 得分很好，而某个翻转或缩放视图预测很差，直接累加该视图可能拉低边界类和小类别结果。

所以：

> “官方用了 13 TTA”只能说明这是一种常用评测 recipe，不能证明它一定提升你当前的 V13。

我没有找到公开的、针对你这个 KPConvX wrapper 的严格 `1-view vs 13-view` 对照数据，因此提升幅度必须在你自己的 checkpoint 上测量。

---

# 五、13 TTA 应该怎么验证

无需重新训练。直接对同一组 baseline 和 V13 checkpoint 做以下测试：

| 评测 | 测试视图 |
|---|---|
| T1 | identity |
| T4 | 4 个 z 旋转 |
| T5 | 4 个旋转 + flip |
| T12 | 4 个旋转 × 3 个 scale |
| T13 | T12 + flip |

分别得到：

```text
Baseline-T1
Baseline-T4
Baseline-T5
Baseline-T12
Baseline-T13

V13-T1
V13-T4
V13-T5
V13-T12
V13-T13
```

重点比较四个量：

\[
G_{\text{TTA,base}}
=
\operatorname{Base}_{T13}
-
\operatorname{Base}_{T1}
\]

\[
G_{\text{TTA,V13}}
=
\operatorname{V13}_{T13}
-
\operatorname{V13}_{T1}
\]

\[
\Delta_{T1}
=
\operatorname{V13}_{T1}
-
\operatorname{Base}_{T1}
\]

\[
\Delta_{T13}
=
\operatorname{V13}_{T13}
-
\operatorname{Base}_{T13}
\]

结果可以这样解释：

| 现象 | 含义 |
|---|---|
| baseline 和 V13 都被 TTA 提升 | TTA 本身有效，不证明 V13 有效 |
| V13 在 T1 提升，但 T13 提升消失 | TTA 掩盖了模块优势，或 V13 对增强不稳定 |
| V13 的单视图方差明显高于 baseline | 动态 density/radius 缺乏变换一致性 |
| T4 提升，但加入 scale 后下降 | scale TTA 与动态半径冲突 |
| flip 单独导致下降 | 检查坐标、normal 和特征翻转 |
| T13 比 T4 只高不到 0.15 mIoU | 13 次推理通常不值得，保留 T4 |

这里的 0.15 mIoU 是工程停止阈值，不是论文事实。

还应保存每个单独视图的 mIoU：

```text
R0
R90
R180
R270
R0-S0.95
...
Flip
```

不要只看最终 ensemble。单视图结果会直接告诉你是哪种增强在拖后腿。

---

# 六、13 TTA 对你当前 V13 最可能暴露的问题

我比较怀疑的是：

```text
每个 fragment 内 percentile density normalization
+
scale TTA
+
固定 grid size
```

这三者组合。

全局坐标缩放 0.95/1.05 后：

```text
GridSample 保留的点发生变化
→ fragment 组成变化
→ 10/90 percentile 变化
→ 同一原始点的 normalized density 变化
→ radius scale 变化
→ support graph 再次变化
```

这不是简单的“旋转投票”，而是每个 TTA 视图都在运行一个略有不同的动态图模型。

解决顺序是：

1. 将 density 除以当前 stage grid size；
2. 使用训练集固定的 stage-wise q10/q90；
3. 测试时禁止每个 fragment 重新计算 percentile；
4. 检查同一原始点在 13 个视图下的 scale 标准差；
5. 分别测试 rotation TTA 和 scale TTA。

一个理想的动态半径模块应该满足：

```text
旋转前后：
density scale 几乎一致。

整体缩放后：
经过 grid-size 或物理尺度归一化，scale 分布可预测。

不同 fragment batch size：
同一个点的预测基本一致。
```

---

# 七、推荐你下一轮的明确执行顺序

## 第一步：不训练，检查当前 V13 是不是“死模块”

在 10～20 个训练或测试房间上记录：

```text
stage2/3/4 的 d_H/r_base；
valid neighbor count；
新增/删除邻居；
neighbor Jaccard change；
density 与 scale 分布。
```

停止条件：

```text
平均 graph change < 10%
或
平均新增邻居 < 1
```

满足任一条件，就不要再调当前 V13 的 range。

## 第二步：用现有 checkpoint 测 T1/T4/T5/T12/T13

这一阶段能直接回答：

```text
13 TTA 是否提升；
rotation 是否提升；
scale TTA 是否伤害 V13；
flip 是否存在实现问题。
```

13 TTA 会产生大量增强副本和 fragments，Pointcept 近期也有测试时 CPU 内存过高的报告，因为增强后的所有 fragment 可能同时保存在内存中。测试时需要监控 CPU RAM，而不仅是 GPU 显存。([GitHub](https://github.com/Pointcept/Pointcept/issues/532?utm_source=chatgpt.com))

## 第三步：实现强版双支持残差

核心配置：

```python
stages = (2, 3)
candidate_k = (40, 48)
ring_k = (8, 12)

scale_ranges = (
    (0.75, 1.45),
    (0.70, 1.55),
)

strength = 1.0
ring_only = True
gamma = 0.25
```

同时：

```text
保留原始 H=20 support；
动态分支只处理新增 ring；
相对坐标除以 query scale；
使用固定 stage density 统计；
identity path 不修改。
```

## 第四步：判断是否值得正式做

强版诊断满足以下条件再进入三 seed：

```text
identity fixed test 提升至少 0.8 mIoU；
或者边界类/小类别有清晰一致提升；
graph change 至少 20%～30%；
速度下降不超过 25%；
显存增加不超过 20%。
```

正式三 seed 的保留标准：

```text
平均 identity-test 提升 ≥ 0.5 mIoU；
三个 seed 中至少两个提升；
提升均值大于标准差；
T4/T13 下仍保持正向提升。
```

若三 seed 平均提升小于 0.3 mIoU，或者标准差大于提升均值，当前方向不应再描述为稳定有效。

---

# 最后的明确答案

**你的 V13 数值范围确实偏保守，但单纯放大 scale 大概率解决不了问题。**真正限制它的是：

```text
固定 H；
候选图只有 H 个点；
应用层级太深；
radius 与 kernel geometry 没有同步；
fragment 内 percentile 导致测试不一致。
```

最值得做的大改是：

> **在 stage 2/3 建立大于 H 的候选图，保留原始 support 作为 identity path，只把动态半径新增的 ring 邻域作为可控残差，并同步缩放 kernel 坐标或归一化相对坐标。**

关于 13 TTA：

> **它真实可能提升，而且 KPConvX 和 Pointcept 都使用 voting/TTA；但它不保证提升。对 V13，scale TTA 甚至可能因为体素化和密度自适应的耦合而降低结果。**

因此论文主结果应该先报告：

```text
deterministic identity fixed test
```

然后辅助报告：

```text
T4 rotation voting
T13 full TTA
```

只有 baseline 和改进模型在完全相同的 TTA 协议下比较，才能证明提升来自模块，而不是来自 13 次推理。