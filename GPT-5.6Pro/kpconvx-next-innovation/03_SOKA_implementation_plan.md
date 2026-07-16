---
title: 'KPConvX SOKA-Lite 实现'
type: 'feature'
created: '2026-07-16'
status: 'done'
baseline_commit: '0f4f2d5f8bf41d7d79b66b60e5c0caf5a089f8a7'
context:
  - 'GPT-5.6Pro/kpconvx-next-innovation/01_GPT56Pro_raw_conversation.md'
  - 'GPT-5.6Pro/kpconvx-next-innovation/02_SOKA_Lite_implementation_digest.md'
---

<frozen-after-approval reason="用户已明确要求按对话中的最终 SOKA 方案直接实现">

## Intent

**Problem:** plain scale04 KPConvX 的 attention 只由中心特征生成，未显式利用邻域在 43 个 kernel cells 中的占据和 assignment quality；继续扩大 support 会重复 v16/v17 并显著增大成本。

**Approach:** 新建独立 `kpconvx_soka` backbone，在固定 KNN、单次聚合内构造六维 per-cell descriptor，经 zero-last `6 -> 16 -> 1` MLP 生成 scalar bias，在 GroupNorm 后、attention activation 前注入。

## Boundaries & Constraints

**Always:** 以 plain scale04 为基线；保持 support、radius、neighbor limits、influence、pooling、decoder、输入和 CE+Lovasz 不变；descriptor 固定为 occupancy fraction、归一化 radial mean、归一化 assignment mean/variance、occupancy entropy、归一化 shell radius；排除 shadow；使用 `scatter_add_`；最后一层零初始化且倍率固定为 1；默认只改 encoder stage2-5 的 KPConvX。

**Ask First:** 启动 30 小时以上长训；改变 fixed evaluation；加入独立的新机制。

**Never:** 继承 hybrid；混入 v17、DA-Radius/meta、boundary、SPP/KSPC、global、新 loss/head；使用 `[M,H,K]` one-hot；重算完整距离张量；sigmoid 后门控；zero-last 再叠加零 beta/gamma；修改 decoder。

## I/O & Edge-Case Matrix

| State | Expected behavior |
|-------|-------------------|
| 正常邻域 | 输出 `[M,K,6]` descriptor 和 `[M,K,1]` bias，occupancy 按 query 归一化 |
| shadow index `N` | 不进入 count、均值、方差或 entropy |
| 空 cell/空邻域 | 均值、方差、entropy 为 0，无 NaN/Inf |
| shared geometry | cache signature 相同则复用，否则重算 |

</frozen-after-approval>

## Code Map

- `pointcept/models/kpconvx/utils/kpnext_blocks.py` -- 原 attention 与 nearest-kernel geometry。
- `pointcept/models/kpconvx/kpconvx_base.py` -- plain stage/decoder 构建。
- `pointcept/models/kpconvx_soka/` -- 新 geometry、operator 与 backbone。
- `configs/s3dis/semseg-kpconvx-base-s3dis-scale04-4090d-area5.py` -- 唯一基线。
- `tests/test_kpconvx_soka_geometry.py` -- 无扩展依赖的几何测试。
- `tests/test_s3dis_soka_configs.py` -- 实验变量隔离测试。

## Tasks & Acceptance

**Execution:**
- [x] 归档对话并生成只含最终修正版的 SOKA 摘要。
- [x] 实现纯 PyTorch descriptor，覆盖 shadow、空 cell、置换与数值稳定。
- [x] 实现 `SOKAKPConvX`，复用 assignment，只增加 pre-activation scalar bias 和 detached 标量监控。
- [x] 注册 backbone，仅替换目标 encoder stage，保留 baseline 参数键。
- [x] 添加 stage2-5 正式配置、stage4 smoke 配置和测试。

**Acceptance Criteria:**
- Given 相同 baseline 参数，when zero-last SOKA 前向，then logits 最大差 `<1e-6`。
- Given neighbor permutation/shadow/空 cell，when 构造 descriptor，then结果置换不变、count 守恒、无 NaN/Inf。
- Given 首次和第二次反向，when更新 zero-last 输出层，then输出层、原 KPConvX 及随后 SOKA 前层获得梯度。
- Given SOKA 配置，when 与 scale04 baseline 合并，then除 backbone type/SOKA 参数外模型、尺度、loss 和训练协议不变。
- Given 500-1000 iteration smoke run，when比较 baseline，then batch 仍为 3、显存增幅 `<=10%`、耗时增幅 `<=15%`；超过 `1.2x` 不长训。

## Design Notes

scalar bias 按 `ch_per_grp` 广播。模型先原样构建 `KPConvXBase`，再只替换选中 encoder 的 `KPConvX`，复用原 `weights/alpha_mlp/grpnorm/kernel_points` 对象，保证 checkpoint 键和初始行为不变。point-level descriptor 不长期缓存。

## Verification

**Commands:**
- `python -m pytest -q -p no:cacheprovider tests/test_kpconvx_soka_geometry.py tests/test_s3dis_soka_configs.py`
- `python -m compileall pointcept/models/kpconvx_soka configs/s3dis`
- `python -m pytest -q -p no:cacheprovider tests/test_kpconvx_soka_model.py`（完整 Pointcept 依赖环境）

## 云端执行顺序

1. 完整验收：`python -m pytest -q -p no:cacheprovider tests/test_kpconvx_soka_geometry.py tests/test_s3dis_soka_configs.py tests/test_kpconvx_soka_model.py`。
2. stage4 工程 smoke：`sh scripts/train.sh -d s3dis -c semseg-kpconvx-soka-lite-stage4-scale04-4090d-area5 -n soka_stage4_smoke -w /path/to/scale04_epoch200.pth -g 1`，只跑 500-1000 iteration 后人工停止，不用其精度作结论。
3. matched continuation 必须为 C0/C1 预注册相同的 30-40 epoch 与学习率覆盖；C0 使用 baseline 配置，C1 使用 stage4 SOKA 配置。未锁定这组覆盖前，不把上面的 smoke 命令当 continuation 实验。
4. 只有 stage4 smoke 的等价性、显存和速度通过后，才启动 `semseg-kpconvx-soka-lite-scale04-4090d-area5` 的 stage2-5 正式版本。

## Suggested Review Order

**Backbone 入口**

- 从 plain KPConvX 安装 SOKA，只替换选定 encoder attention blocks。
  [`kpconvx_soka.py:11`](../../pointcept/models/kpconvx_soka/kpconvx_soka.py#L11)

- 保留 tensor 基线输出，并通过既有字典契约暴露监控标量。
  [`kpconvx_soka.py:60`](../../pointcept/models/kpconvx_soka/kpconvx_soka.py#L60)

**Operator 与几何**

- 在原 GroupNorm 后、activation 前加入 zero-last scalar bias。
  [`soka_blocks.py:237`](../../pointcept/models/kpconvx_soka/soka_blocks.py#L237)

- 复用原参数对象，保持 baseline checkpoint 键和初始行为。
  [`soka_blocks.py:56`](../../pointcept/models/kpconvx_soka/soka_blocks.py#L56)

- 六维 descriptor 使用 shadow-safe scatter 聚合和稳定方差。
  [`soka_geometry.py:14`](../../pointcept/models/kpconvx_soka/soka_geometry.py#L14)

- stage 首 block 刷新几何，后续 block 按签名共享 cache。
  [`soka_blocks.py:125`](../../pointcept/models/kpconvx_soka/soka_blocks.py#L125)

**验证与实验入口**

- 等价、梯度、cache、state keys、监控契约和 AMP 的完整测试。
  [`test_kpconvx_soka_model.py:51`](../../tests/test_kpconvx_soka_model.py#L51)

- shadow、空邻域、守恒和置换不变性的无扩展依赖测试。
  [`test_kpconvx_soka_geometry.py:46`](../../tests/test_kpconvx_soka_geometry.py#L46)

- 正式 stage2-5 配置只覆盖 backbone type 与 SOKA 参数。
  [`semseg-kpconvx-soka-lite-scale04-4090d-area5.py:5`](../../configs/s3dis/semseg-kpconvx-soka-lite-scale04-4090d-area5.py#L5)

- stage4 配置用于 500-1000 iteration 的工程 smoke。
  [`semseg-kpconvx-soka-lite-stage4-scale04-4090d-area5.py:4`](../../configs/s3dis/semseg-kpconvx-soka-lite-stage4-scale04-4090d-area5.py#L4)
