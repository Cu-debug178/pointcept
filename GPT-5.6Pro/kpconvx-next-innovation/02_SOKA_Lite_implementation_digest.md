# SOKA-Lite 实现摘要

本文只保留原始对话中经过源码审查后形成的最终修正版。早期关于 occupancy prior、完整 channel bias、自由 `beta/gamma`、SPP-IPCs/KSPC 或与 v17 合并的方案，不作为首版实现依据。

## 1. 方法定义

SOKA-Lite（Scale-Calibrated Occupancy-Aware Kernel Attention）解决的问题是：plain KPConvX 的 kernel attention 只由中心特征产生，模型没有显式看到当前固定邻域在各 kernel cells 中的占据和匹配质量。

首版严格限定为：

- 基线为 plain scale04 KPConvX；
- 固定原始 KNN support，不改变 radius、neighbor limits 或邻居集合；
- 复用已有 nearest-kernel assignment；
- 每个 query、每个 kernel cell 构造 `[M, K, 6]` 描述符；
- 用共享于各 cell 的 `6 -> 16 -> 1` MLP 生成 `[M, K, 1]` scalar bias；
- 在原 attention 的 GroupNorm 后、sigmoid/其他 activation 前相加；
- 后续仍执行一次原生 KPConvX aggregation；
- 默认只作用于 encoder stage2-5 的 KPConvX，stem、stage1 KPConvD 和 decoder 不变；
- 损失仍为 CE + Lovasz，不增加辅助监督。

## 2. 六维 kernel-cell descriptor

对 query `q`、kernel cell `k`：

1. `occupancy_fraction`: cell 内有效邻居数除以该 query 的有效邻居总数。
2. `mean_radial`: cell 内邻居到 query 的平均距离除以当前层物理 `radius`。
3. `mean_assignment`: 邻居到最近 kernel point 的平均距离除以当前层 `sigma`。
4. `assignment_variance`: 上述归一化 assignment error 的方差。
5. `occupancy_entropy`: 所有 cells 上 occupancy fraction 的归一化 entropy，广播到每个 cell。
6. `shell_radius`: 当前 kernel point 到中心的距离除以当前层 `radius`。

空 cell 的均值和方差必须为 0。empty flag 只可监控，不作为核心输入。

## 3. Attention 注入

```python
base_logits = alpha_mlp(pooled_feats)
base_logits = optional_group_norm(base_logits)
base_logits = base_logits.reshape(M, K, ch_per_grp)

soka_bias = bias_bound * torch.tanh(soka_mlp(descriptor))
modulations = attention_act(base_logits + soka_bias)
```

默认 `bias_bound=2.0`。scalar bias 沿 `ch_per_grp` 广播，不输出完整 `K x ch_per_grp` bias，避免退化成另一种 channel meta conditioner。

## 4. 初始化与 checkpoint 兼容

```python
soka_mlp = nn.Sequential(
    nn.Linear(6, 16),
    nn.SiLU(),
    nn.Linear(16, 1),
)
nn.init.zeros_(soka_mlp[-1].weight)
nn.init.zeros_(soka_mlp[-1].bias)
```

固定倍率为 1，不再增加零初始化 `beta/gamma`。否则 zero-last 与零倍率相乘会形成死启动。

加载同一 baseline 参数时，SOKA 初始 bias 为 0，logits 最大绝对差必须 `<1e-6`。原 `weights`、`alpha_mlp`、`grpnorm`、`kernel_points` 的参数路径应保持不变，以便 `CheckpointLoader(strict=False)` warm start。

## 5. 几何、shadow 与 cache

- shadow index 满足 `neighb_inds == s_pts.shape[0]`，必须用 `valid_mask = neighb_inds < s_pts.shape[0]` 排除。
- count、radial sum、assignment sum/square sum使用扁平索引 `scatter_add_`，禁止创建 `[M,H,K]` one-hot。
- 不得为了 SOKA 再构造一次 `[M,H,K,3]` 距离张量；可由已有 `neighbors_1nn`、relative neighbors 和 assigned kernel point 以 `O(MH)` 得到 nearest assignment error。
- `sqrt` 前 clamp，variance 使用 `clamp(second_moment - mean**2, min=0)`。
- `shared_kp_data` 可缓存 nearest distance 和 valid mask，但只能在 query/support/neighbors signature 相同时复用；标量监控必须 detach，不能长期保存 point-level 计算图。

## 6. 推荐接口

```python
soka_enabled=True
soka_stages=(2, 3, 4, 5)
soka_hidden_dim=16
soka_bias_bound=2.0
soka_monitor=True
```

开发 smoke test 可先使用 `soka_stages=(4,)`，正式首版为 stage2-5。

## 7. 必须验证

- zero-init baseline 等价：`max_abs_diff < 1e-6`；
- neighbor permutation 前后 descriptor/output 差 `<1e-6`；
- `counts.sum(1) == valid_mask.sum(1)`；
- shadow 不进入统计；
- 空 cell/空邻域无 NaN/Inf；
- 第一次 backward 输出层有梯度，原 KPConvX 有梯度；更新输出层后第二次 backward 前层有梯度；
- shared cache on/off 数值一致；
- AMP 下保持有限值。

## 8. 监控与工程止损

每个 stage 输出 occupancy、entropy、radial、assignment、bias、base-logit RMS、bias/base RMS ratio，以及 attention `<0.05`/`>0.95` 比例。

500-1000 iteration smoke run 的通过线：batch size 仍为 3，峰值显存增幅约 `<=10%`，iteration time 增幅约 `<=15%`，参数增量远低于 `0.01M`。若 batch size 降为 2 或耗时超过 baseline `1.2x`，先优化，不启动长训。

## 9. 实验判据

- 零训练 audit：SOKA descriptors 加入后，错误预测 AUROC 至少 `+0.03` 或 NLL 至少降低 `2%`，且跨房间成立。
- matched continuation：同 checkpoint 的 baseline/SOKA 各 30-40 epoch，fixed dev mIoU 至少 `+0.003`，多数房间同向，耗时 `<=1.15x`。
- 正式 Area5 首 seed：相对 scale04 `0.6939` 至少 `+0.005`，即 `mIoU >= 0.6989`；强结果线为 `0.7019`。
- 结果必须来自 fixed full-scene、last/预注册 checkpoint，不能只提高 random-crop best，也不能由 beam 单类峰值主导。

## 10. 首版禁止项

不加入 v17、DA-Radius、DA-meta、boundary、SPP-IPCs、KSPC、global token、Mamba、预训练、新 loss 或 auxiliary head；不改 support、pooling、decoder 和训练协议；不在 sigmoid 后门控；不立即做 SOKA+v17。
