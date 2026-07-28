# 审查结论

**当前压缩包中的 V18 不能直接开始训练。源码里有两个确定的 P0 级集成错误，会导致真实 forward 运行失败。**

V13 的实现没有发现同等级的运行阻断问题，但它在 Standalone 中实际实现的是：

> **对已有固定 H-NN 邻居进行密度自适应硬裁剪。**

它并不能真正“扩大稀疏区域的邻域”，因为没有比 H 更大的候选集合。这个结论与审查材料中对 V13 证据边界和 Standalone 邻接图的描述一致。(prompt.md)

我完成了以下检查：

- 解压并比较 before/after 源码；
- `compileall` 语法检查；
- 执行压缩包自带 pytest；
- 手动执行 V13、V18 图构造测试；
- 构造五层 synthetic pyramid，执行模型级 CPU forward；
- 检查官方 KPNeXt 各阶段通道变化；
- 检查 V18 的 identity path、ring path、kernel scaling 和残差融合；
- 检查 `gamma=0` 时是否能与官方基线等价。

没有使用真实 S3DIS 数据和 CUDA 邻居扩展完成整轮训练，因此下面区分“确定错误”“设计风险”和“仍需真实实验确认”。

---

# 一、两个会让 V18 直接崩溃的错误

## P0-1：切片后的 `base_neighbors` 不连续，进入 `.view(-1)` 会报错

位置：

```text
Standalone/KPConvX/models/KPNextV18.py:367-372
```

当前代码：

```python
base_neighbors = [
    neighbors[:, : self.base_neighbor_limits[layer]]
    for layer, neighbors in enumerate(batch.in_dict.neighbors)
]
```

V18 的候选邻居数配置为：

```text
candidate K = [12, 16, 40, 48, 20]
base H      = [12, 16, 20, 20, 20]
```

因此第三、第四层执行的是：

```python
neighbors[:, :20]
```

原 tensor 的行步长分别仍然是 40 和 48。这个切片通常是 **non-contiguous tensor**。

随后官方工具函数：

```text
Standalone/KPConvX/models/generic_blocks.py:57
```

执行：

```python
outputs = inputs.index_select(dim, indices.view(-1))
```

对这种二维非连续切片使用 `view(-1)` 会出现：

```text
RuntimeError:
view size is not compatible with input tensor's size and stride
```

这不是理论风险，我在 synthetic 五层 forward 中复现了这个错误。

### 最小安全修复

只改 V18，不改官方公共工具：

```python
base_neighbors = [
    neighbors[:, : self.base_neighbor_limits[layer]].contiguous()
    for layer, neighbors in enumerate(batch.in_dict.neighbors)
]
```

不建议为了这个问题直接把官方：

```python
indices.view(-1)
```

全局改成：

```python
indices.reshape(-1)
```

因为你的目标是保持官方 baseline 原代码不动。局部 `.contiguous()` 更符合可复现对照原则。

---

## P0-2：V18 ring module 的通道数与实际输入特征不匹配

位置：

```text
Standalone/KPConvX/models/KPNextV18.py:219-241
```

当前代码根据当前 stage 的基础通道构建 ring：

```python
stage_channels = []

for layer in range(self.num_layers):
    target = float(model_cfg.init_channels) * channel_scaling ** layer
    stage_channels.append(int(np.ceil((target - 0.1) / 16)) * 16)
```

得到：

```text
stage 1: 64
stage 2: 96
stage 3: 128
stage 4: 192
stage 5: 256
```

问题在于 V18 ring 不是插在 stage 入口，而是插在该 stage 的所有 encoder blocks 之后：

```python
for block in blocks:
    features, upcut = block(...)

if stage in self.dual_support_stages:
    features = self._apply_ring(...)
```

官方 KPNeXt 在 `grid_pool=True` 时，每个非最终 stage 的最后一个 block 已经把特征投影到下一层通道：

```text
Standalone/KPConvX/models/KPNext.py:112-137
```

例如：

```python
Cout = layer_C[l + 1] if (
    self.grid_pool
    and layer < self.num_layers
    and block_i == self.layer_blocks[l] - 1
) else C
```

因此 ring 实际收到的特征通道是：

| Ring 插入阶段 | 当前代码创建的通道 | 实际输入通道 |
|---|---:|---:|
| stage 3 | 128 | **192** |
| stage 4 | 192 | **256** |

修复第一个 non-contiguous 问题后，我复现的下一个错误就是：

```text
RuntimeError:
mat1 and mat2 shapes cannot be multiplied
(16x192 and 128x128)
```

错误发生在：

```python
modulations = self.alpha_mlp(s_features)
```

### 正确修复

```python
channel_scaling = float(model_cfg.channel_scaling)

layer_channels = []
for layer in range(self.num_layers):
    target = float(model_cfg.init_channels) * channel_scaling ** layer
    layer_channels.append(
        int(np.ceil((target - 0.1) / 16)) * 16
    )

# Ring 插入在 encoder stage 的最后一个 block 之后。
# grid_pool=True 时，非最终 stage 已经升到下一层通道。
stage_channels = [
    layer_channels[layer + 1]
    if self.grid_pool and layer < self.num_layers - 1
    else layer_channels[layer]
    for layer in range(self.num_layers)
]
```

修复后：

```text
stage 3 ring: C = 192
stage 4 ring: C = 256
```

建议再加入显式断言：

```python
ring_module = self.ring_modules[str(stage)]

if features.shape[1] != ring_module.channels:
    raise RuntimeError(
        "Stage {} ring channel mismatch: feature C={}, module C={}".format(
            stage,
            features.shape[1],
            ring_module.channels,
        )
    )
```

这能防止以后改变通道缩放或 stage 插入位置时再次静默出错。

---

# 二、修复后身份路径是否正确

我对上述两个问题做了临时修复，然后进行了 synthetic 五层 forward 测试。

测试条件：

```text
相同模型权重；
相同 points / pools / upsamples；
官方基线使用 H = [12,16,20,20,20]；
V18 使用候选 K = [12,16,40,48,20]；
V18 identity path 截取前 H；
ring gamma = 0。
```

结果：

```text
官方 KPNeXt 输出形状 == V18 输出形状
最大绝对误差 max_abs = 0.0
```

这说明：

> 两个 P0 修复后，V18 的 identity path 在 synthetic forward 中可以保持数值等价。

但这还不是完整的 bitwise 认证，因为没有使用真实 CUDA 邻域构建和真实 S3DIS batch。正式训练前仍应在真实 batch 上完成一次同样测试。

---

# 三、当前文档中的 V18 参数量也算错了

审查包里的说明称：

```text
stage 3 ring: 201,152
stage 4 ring: 446,112
总计: 647,264
```

这个数恰好对应当前错误的通道：

```text
stage 3 C=128
stage 4 C=192
```

修正为真实通道后，在 S3DIS `inv_groups=4` 的设置下应为：

| Stage | 正确通道 | Ring 参数量 |
|---|---:|---:|
| stage 3 | 192 | 446,112 |
| stage 4 | 256 | 787,328 |
| 合计 | — | **1,233,440** |

所以 V18 正确版本会比文档描述多大约：

```text
1,233,440 - 647,264 = 586,176 参数
```

这不一定不可接受，但论文中的参数量、速度、显存报告必须更新。

---

# 四、V13 没有发现 forward 阻断，但机制作用被高估了

## 1. 当前 V13 只能删除邻居，不能新增邻居

V13 在：

```text
Standalone/KPConvX/models/KPNextV13.py:84-117
```

取现有邻接矩阵：

```python
neighbors = in_dict.neighbors[layer]
```

随后调用：

```python
masked = mask_neighbors_by_radius(
    points,
    neighbors,
    scale,
    base_radius=layer_radius,
)
```

而 `mask_neighbors_by_radius` 的行为只是：

```python
masked = neighbors.clone()
masked.masked_fill_(~keep, num_points)
```

即：

```text
现有 H 个邻居
→ 计算距离
→ 超出动态半径的槽替换为 shadow
```

它从来没有查询：

```text
第 H+1、H+2……个候选邻居
```

因此：

- `scale < 1`：可以删除更多已有邻居；
- `scale > 1`：最多保留已有 H 个邻居；
- `scale > 1`：不能加入任何 H 以外的新邻居。

尤其在当前 Standalone 中，官方 baseline 本身没有对这个 H-NN 候选图执行同样的中心距离硬裁剪，所以 V13 相对于 baseline 的净行为基本是：

> **密度自适应地屏蔽部分现有邻居。**

而不是：

> 扩大稀疏区域的实际支持图。

KPConvX 论文中的算子已经包含 nearest-kernel assignment、两层 shell `[1,14,28]`、固定邻居上限、`r=2.1×grid size`、`sigma=r` 和 linear influence。(Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf)

因此不能把这些原有机制重新描述成 V13 的创新。

### 更准确的 V13 名称

当前 Standalone 实现更适合叫：

```text
Density-Adaptive Support Truncation
```

或：

```text
Density-Adaptive Neighbor Masking
```

而不是：

```text
Density-Adaptive Radius Expansion
```

---

## 2. V13 的作用范围和类注释不一致

类注释写的是：

```python
"""KPNeXt with V13 per-query radius filtering on selected encoder stages."""
```

但 forward 实现是：

```python
baseline_neighbors = batch.in_dict.neighbors
batch.in_dict.neighbors = self._adaptive_neighbors(batch.in_dict)

try:
    return super().forward(batch, verbose=verbose)
finally:
    batch.in_dict.neighbors = baseline_neighbors
```

也就是说，在整个 `super().forward()` 期间，邻接图都已被替换。

官方 decoder block 同样使用：

```text
KPNext.py:472-478
```

```python
features, _ = block(
    ...,
    batch.in_dict.neighbors[l],
    ...
)
```

所以当前 V13 实际作用于：

```text
选中分辨率的 encoder blocks
+
对应分辨率的 decoder blocks
```

### 这是不是 bug

取决于你的实验定义：

- 如果目标是“指定分辨率的所有 KPConvX”：当前行为可以保留，但文档必须改；
- 如果目标是“只改 encoder stage”：当前实现不符合定义，需要重写 forward 或增加 encoder-only hook。

论文实验中一定要明确写清楚，否则复现者会误以为只改了 encoder。

---

# 五、V18 的 ring 图实现可以运行后，仍有几个机制问题

下面不是立即崩溃的代码错误，但会影响“模块有没有明显作用”。

## 1. Ring 只使用 `H` 之后且位于基础半径外的邻居

位置：

```text
Standalone/KPConvX/utils/dual_support.py:138-145
```

```python
extra_slot = slot_ids >= base_limit

ring_mask = (
    valid
    & extra_slot
    & (distances > float(base_radius))
    & (distances <= adaptive_radius)
)
```

也就是说 ring 邻居必须同时满足：

```text
候选排序位置 >= H
并且
距离 > base_radius
并且
距离 <= base_radius × adaptive_scale
```

这会丢掉一类可能非常有价值的点：

> 位于前 H 个之外，但仍在基础半径以内的额外候选点。

例如一个高密度边界 voxel：

```text
基础半径内有 40 个点；
identity 只保留最近 20 个；
第 21～40 个点仍在基础半径内；
当前 ring 全部忽略它们。
```

而 FastAdapter 所讨论的几何退化并不只发生在“半径外”，也会发生在：

- 高密度区域；
- 多表面混合区域；
- 边界；
- grid pooling 合并大量细点的位置。

因此当前 V18 更接近：

```text
Sparse-region outer-context expansion
```

而不是完整的：

```text
Geometric degradation repair
```

---

## 2. 建议把两种额外支持拆开消融

建议不要一次混成一个 ring。

### A. In-radius extra support

```python
inside_extra_mask = (
    valid
    & extra_slot
    & (distances <= base_radius)
)
```

作用：

```text
恢复基础半径内被固定 H 截断的额外候选。
```

### B. Outer expansion support

```python
outer_ring_mask = (
    valid
    & extra_slot
    & (distances > base_radius)
    & (distances <= adaptive_radius)
)
```

作用：

```text
在稀疏区域引入基础名义半径外的新上下文。
```

这样可以明确回答：

```text
提升来自 fixed-H truncation repair，
还是来自真正的 radius expansion？
```

论文实验中这两者不能混为同一个贡献。

---

## 3. Standalone 的 `base_radius` 不是当前实现中的严格硬支持边界

论文公式将邻域描述为 radius neighborhood，并以固定 H 截断；同时使用距离最近 kernel point 的 linear influence。(Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf)

但审查材料已核实，当前 Standalone pyramid 更接近先生成固定 H-NN 候选，scalar radius 并未在 helper 中真正把所有超界点裁掉，真正的软抑制发生在 KPConvX 内部的 linear influence。(prompt.md)

而 linear influence 的距离是：

```text
邻居相对坐标
到
最近 kernel point
的距离
```

不是简单的：

```text
邻居到中心点的距离
```

所以某些：

```text
center distance > base_radius
```

的邻居仍可能靠近外层 kernel point，从而拥有非零 influence。

因此当前 V18 的表述：

```text
ring 是官方基础支持域以外的新邻居
```

并不完全严格。

更准确的表述是：

> Ring 使用固定 H 之外、且位于名义 kernel radius 之外的候选邻居。

---

## 4. 更符合 KPConvX 的 ring 判定方式

一个更严格的版本不是按中心半径判定，而是按 influence 判定：

```text
官方尺度下 influence == 0
动态缩放后 influence > 0
```

即：

\[
\mathcal R(q)=
\left\{
i:
h_i^{\text{base}}=0
\land
h_i^{\text{scaled}}>0
\right\}
\]

这样“新增支持”就和 KPConvX 自己的 kernel geometry 定义一致，而不是额外引入一个中心球边界。

推荐至少比较三个版本：

```text
V18-A：所有额外槽 H:K
V18-B：额外槽且 center distance > r
V18-C：base influence=0 且 scaled influence>0
```

---

# 六、V18 BatchNorm 会被大量无 ring 点影响

位置：

```text
KPNextV18.py:159-162
```

当前顺序：

```python
residual = torch.sum(neighbor_features * neighbor_weights, dim=1)
residual = self.activation(self.norm(residual))

has_ring = valid.any(dim=1, keepdim=True).to(dtype=residual.dtype)

return self.gamma * residual * has_ring
```

问题是：

```text
没有任何 ring 邻居的 query
→ residual 为全零
→ 仍然先进入 BatchNorm
→ 最后才乘 has_ring=0
```

如果大多数 query 没有 ring，这些零样本会参与 BatchNorm 均值与方差统计，可能导致：

- ring 特征统计被稀释；
- 训练和推理 running statistics 不稳定；
- 不同 batch 的 ring occupancy 改变输出尺度；
- V18 分支效果被压弱。

### 可选修复

最保守的是先记录 ring occupancy，确认是否严重：

```python
ring_ratio = has_ring.float().mean()
```

如果 ring ratio 很低，可考虑：

1. ring 分支使用 LayerNorm；
2. 使用 GroupNorm；
3. 只对 `has_ring=True` 的行执行 BN，再 scatter 回去；
4. 取消该分支 BN，使用可学习 bias/scale；
5. 最后一层 norm 零初始化配合 learnable gamma。

不建议直接把 `has_ring` 提前乘 residual 后继续普通 BN，因为全零行仍参与统计。

---

# 七、V18 diagnostic 的固定 `gamma=0.25` 适合诊断，但不适合直接当正式模型

当前 diagnostic profile：

```text
gamma_mode = fixed
gamma = 0.25
```

而 ring branch 是随机初始化的。

这意味着训练第一步开始：

```text
官方特征
+
0.25 × 随机 ring 特征
```

优点是：

```text
模块一定有明显作用，适合确认机制是不是“活的”。
```

风险是：

```text
可能明显扰动已经较稳定的官方 KPConvX 表征，
导致训练方差或初期性能下降。
```

建议：

### 强诊断实验

```python
gamma_mode = "fixed"
gamma = 0.25
```

只用于回答：

```text
ring 分支大幅介入后，性能明显升还是明显降？
```

### 正式实验

```python
gamma_mode = "learnable"
gamma_init = 0.0
```

或：

```python
gamma_init = 0.05
```

还可以令 ring 最后一个 norm 的 scale 初始化为 0，使模型从官方 baseline 精确起步。

---

# 八、配置系统会覆盖部分命令行参数

解析顺序是：

```text
1. 读取命令行参数；
2. 写入 cfg；
3. 调用 adjust_config；
4. profile 再次覆盖参数。
```

例如：

```text
train_S3DIS.py:292-314
```

在：

```python
dual_support_profile == "v18_diagnostic"
```

时会强制覆盖：

```python
candidate limits
stages
ring limits
stage ranges
strength
gamma mode
gamma
```

因此即使命令行传入：

```bash
--dual_support_gamma 0.1
--dual_support_strength 0.8
```

只要 profile 仍是 `v18_diagnostic`，就会被改回：

```text
gamma=0.25
strength=1.0
```

只有：

```bash
--dual_support_profile custom
```

才能真正使用部分 CLI 设置。

但以下字典参数又没有 CLI 入口：

```text
dual_support_stage_ranges
dual_support_ring_limits
dual_support_fixed_spacing_bounds
```

这会导致实验配置不方便完整记录。

建议采用：

```text
profile 只负责提供默认值；
显式 CLI 参数最后覆盖 profile。
```

或者每个实验使用独立 YAML/JSON 配置文件，不依赖当前混合覆盖机制。

---

# 九、压缩包自带的测试目前无法直接运行

## Purpose Review 包

测试文件寻找：

```text
purpose/external/ml-kpconvx-standalone/...
```

实际源码位于：

```text
purpose/source/external/ml-kpconvx-standalone/...
```

运行：

```bash
pytest -q
```

在 collection 阶段就报：

```text
FileNotFoundError:
purpose/external/ml-kpconvx-standalone/KPConvX/utils/...
```

正确路径应增加：

```python
/ "source"
```

---

## Before/After 包

测试寻找：

```text
after/external/ml-kpconvx-standalone/...
```

实际源码位于：

```text
after/Standalone/KPConvX/...
```

因此同样在 collection 阶段失败。

应该改为：

```python
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Standalone"
    / "KPConvX"
    / "utils"
    / "dual_support.py"
)
```

---

## 更关键的问题：现有测试只检查工具函数

即使把路径修好，当前测试主要覆盖：

```text
density scale；
radius mask；
ring neighbor selection；
fixed spacing bounds。
```

它们不会发现：

- non-contiguous tensor；
- stage channel mismatch；
- encoder/decoder 作用范围；
- gamma=0 identity 等价；
- ring branch 梯度；
- BatchNorm 问题。

所以“工具单元测试通过”不能说明模型集成正确。

---

# 十、正式训练前必须补的模型级测试

## 1. V18 完整 forward smoke test

```python
def test_v18_full_forward():
    logits = model(batch)
    assert logits.shape == (num_points, num_classes)
    assert torch.isfinite(logits).all()
```

---

## 2. Gamma=0 身份等价测试

加载完全相同的基线权重：

```python
baseline = KPNeXt(cfg_base)
v18 = KPNeXtV18(cfg_v18)
```

把共同参数复制后：

```python
for module in v18.ring_modules.values():
    module.gamma.zero_()
```

断言：

```python
torch.testing.assert_close(
    v18(batch),
    baseline(batch),
    rtol=0,
    atol=0,  # GPU 算子不确定时可改成 1e-6
)
```

这是 V18 最重要的测试。

---

## 3. 通道一致性测试

```python
assert feature_stage3.shape[1] == ring_stage3.channels
assert feature_stage4.shape[1] == ring_stage4.channels
```

---

## 4. 邻接切片连续性测试

```python
for neighbors in base_neighbors:
    assert neighbors.is_contiguous()
```

---

## 5. Ring 梯度测试

构造至少存在一个 ring 邻居的 batch：

```python
loss = model(batch).square().mean()
loss.backward()
```

检查：

```python
assert ring_module.weights.grad is not None
assert ring_module.weights.grad.abs().sum() > 0
assert alpha_mlp_grad > 0
```

---

## 6. V13 作用域测试

分别记录 encoder 和 decoder 收到的 neighbors，明确验证：

```text
encoder-only
```

还是：

```text
encoder + decoder
```

然后让代码、论文和配置名称保持一致。

---

# 十一、V13 和 V18 当前状态的最终评级

| 模块 | 运行正确性 | 机制是否真正生效 | 当前是否可训练 |
|---|---|---|---|
| 官方 KPNeXt baseline | 未发现改动 | 官方机制 | 可以 |
| V13 | synthetic forward 可运行 | 主要是自适应删减 H-NN，不是真扩张 | 可做小规模对照 |
| V18 原压缩包 | 两个 P0 forward 错误 | 未能进入有效 ring 训练 | **不可开始训练** |
| V18 修复两个 P0 后 | synthetic forward 可运行 | 能引入 H 以外候选 | 可进入图审计，不应立即长训 |
| V18 formal | 仍需真实 batch identity、梯度和 BN 验证 | 待验证 | 尚未达到论文级可信状态 |

---

# 十二、建议的执行顺序

## 第一步：修复两个 P0

```text
1. base neighbor slice 加 .contiguous()
2. ring channels 改成 encoder stage 输出后的真实宽度
```

## 第二步：修复测试路径

保证压缩包解压后直接运行：

```bash
pytest -q
```

而不是手工移动目录才能运行。

## 第三步：增加模型级 identity test

先证明：

```text
gamma=0 时 V18 == 官方 KPNeXt
```

再谈性能。

## 第四步：运行真实 S3DIS graph audit

至少输出：

```text
每层平均新增 ring 数；
没有 ring 的 query 比例；
inside-radius extra candidate 数；
outside-radius extra candidate 数；
change rate；
d_H / r；
不同密度分桶的新增邻居数。
```

建议停止条件：

```text
平均 added_count < 1
或
change_rate < 0.10
或
超过 80% query 没有 ring
```

满足其中之一，就不要直接长训，应先修改 ring 定义。

## 第五步：先跑短诊断

```text
20～40 epochs
gamma=0.25
固定 seed
identity test
不使用 TTA
```

主要看：

```text
训练是否稳定；
ring 分支梯度；
BN running stats；
显存和速度；
边界类别是否有方向性变化。
```

## 第六步：再进入三 seed formal

正式版使用：

```text
learnable gamma；
gamma init 0 或 0.05；
固定 spacing bounds；
完整模型级测试；
相同 deterministic identity test。
```

---

# 最终判断

**V18 的总体架构思想是合理的：保留官方 H-NN identity path，用大候选图建立额外残差支持，比 V13 直接替换邻接图更安全。**

但目前上传版本存在：

1. **非连续邻接切片导致 `.view` 崩溃；**
2. **ring 模块通道配置错误；**
3. **自带测试路径失效；**
4. **测试覆盖不到模型集成；**
5. **ring 只处理基础半径外的额外点，遗漏基础半径内被固定 H 截断的点；**
6. **无 ring query 仍参与 ring BatchNorm；**
7. **配置 profile 会覆盖命令行参数。**

因此现在最明确的结论是：

> **V13 可以作为“动态邻居裁剪”对照继续保留；V18 的方向比 V13 更值得推进，但必须先修复两个 P0 错误，再完成真实 batch 的 identity equivalence 和 graph audit。当前原包不应直接提交长时间训练。**

已生成两个 P0 错误的最小源码补丁：

[下载 V18 P0 修复补丁](https://files.chat01.ai/python-generations/c59b313b-283a-4ab6-8fc2-77e32cb6afb4/KPConvX_V18_P0_fixes.patch)

原始审查文件：

[Before/After 源码包](https://files.chat01.ai/python-generations/c59b313b-283a-4ab6-8fc2-77e32cb6afb4/KPConvX_Standalone_V13_V18_Before_After_20260728.zip)

[V18 Purpose Review 包](https://files.chat01.ai/python-generations/c59b313b-283a-4ab6-8fc2-77e32cb6afb4/KPConvX_Standalone_V18_Purpose_Review_20260728.zip)