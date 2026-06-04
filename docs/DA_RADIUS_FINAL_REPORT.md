# DA-Radius Global Context KPConvX 最终报告

## 1. Critical Review

原 DA-Kernel 只改变 kernel point / influence 的空间尺度，没有改变实际 neighbor set，因此更像 radius-aware weighting，而不是严格的 adaptive local neighborhood search。主贡献应放在 CUDA-level Density-Adaptive Radius Search：在邻居构建阶段直接使用 per-point radius。

推荐主方法名：`Density-Adaptive Local-Global KPConvX`。它同时覆盖 local DA-Radius 和 PTv3-style global context，比单独强调 DA-Kernel 更符合论文贡献。

## 2. Code Location Analysis

KPConvX / KPConvXHybrid 的邻居索引生成位于：

- `pointcept/models/kpconvx/utils/torch_pyramid.py`
- `build_full_pyramid(...)`
- `fill_pyramid(...)`
- 原路径主要调用 `pointops.knn_query(neighbor_limits[i], cur_points, offset)`

`neighbor_limits` 当前作为 KNN `nsample` 使用，不是 radius 上限。`radius_scaling`、`subsample_size`、stage pyramid 参数在 pyramid 构建和模型配置中传递。

## 3. CUDA Modification Plan

已完成：

- 新增 `adaptive_ball_query_cuda`
- 新增 `pointops.adaptive_ball_query`
- 新增 `da_radius_backend="cuda"`
- 在 `torch_pyramid.py` 中支持 CUDA per-point radius search
- 保留 `da_radius_backend="torch"` 作为 reference fallback

后续优化计划：

- 用 grid/hash/binning 降低 adaptive ball query 的 batch 内全扫描成本
- 增加 min-neighbor fallback 策略
- 增加 stage-wise neighbor count logging
- 在 Linux 云端用真实 S3DIS batch 记录 forward latency 和 memory

## 4. Modified Files

- `.gitignore`
- `libs/pointops/functions/__init__.py`
- `libs/pointops/functions/query.py`
- `libs/pointops/functions/utils.py`
- `libs/pointops/setup.py`
- `libs/pointops/src/ball_query/ball_query_cuda.cpp`
- `libs/pointops/src/ball_query/ball_query_cuda_kernel.h`
- `libs/pointops/src/pointops_api.cpp`
- `pointcept/models/kpconvx/utils/torch_pyramid.py`
- `pointcept/models/kpconvx_hybrid/da_kpconvx_block.py`
- `pointcept/models/kpconvx_hybrid/da_kpnext_blocks.py`
- `pointcept/models/kpconvx_hybrid/kpconvx_hybrid.py`
- `pointcept/models/kpconvx_hybrid/kpx_stage2.py`
- `pointcept/models/kpconvx_hybrid/kpx_stage1.py`
- `pointcept/models/kpconvx_hybrid/sgca.py`

## 5. New Files

- `libs/pointops/src/ball_query/ball_query_cuda_kernel.cu`
- `libs/pointops/src/knn_query/knn_query_cuda_kernel.cu`
- `libs/pointops/src/missing_pointops_stubs.cpp`
- `tools/check_da_radius.py`
- `tools/benchmark_da_radius.py`
- `docs/DA_RADIUS_STAGE_REPORT.md`
- `docs/DA_RADIUS_FINAL_REPORT.md`
- `configs/s3dis/semseg-kpconvx-hybrid-v1-da-radius-cuda-conservative-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v2-da-radius-cuda-balanced-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v3-da-radius-cuda-stronger-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v4-da-radius-torch-reference-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v5-da-kernel-only-ablation-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v6-da-radius-only-ablation-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v7-global-context-only-ablation-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v8-da-radius-global-context-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v9-da-kernel-radius-global-context-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v10-constant-influence-ablation-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v11-linear-influence-ablation-area5.py`
- `configs/s3dis/semseg-kpconvx-hybrid-v12-cuda-vs-torch-radius-check-area5.py`

## 6. Config Files

| 配置 | 目的 | 风险 | 论文用途 |
| --- | --- | --- | --- |
| V1 conservative | 小幅 DA-Radius | 低 | 稳定性检查 |
| V2 balanced | 平衡 CUDA DA-Radius | 中 | 候选主结果 |
| V3 stronger | 更大 radius range | 中高 | 增强版 |
| V4 torch reference | debug / 对照 | 高 | 正确性参考 |
| V5 DA-Kernel only | 旧贡献消融 | 低 | 消融 |
| V6 DA-Radius only | 验证主贡献 | 中 | 核心消融 |
| V7 Global only | 验证全局上下文 | 中 | 核心消融 |
| V8 DA-Radius + Global | local-global 主组合 | 中高 | 推荐主结果 |
| V9 DA-Kernel + DA-Radius + Global | 最强组合 | 高 | 补充主结果 |
| V10 constant influence | influence 消融 | 低 | 消融 |
| V11 linear influence | influence 消融 | 低 | 消融 |
| V12 cuda vs torch check | CUDA / torch 对齐 | 中 | debug |

## 7. Patch

本次 patch 分为四组：

1. DA-Radius reference：`da_kpconvx_block.py`、`da_kpnext_blocks.py`
2. CUDA neighbor search：`libs/pointops/**`、`torch_pyramid.py`
3. independent ablation switches：`kpconvx_hybrid.py`、`kpx_stage2.py`、`kpx_stage1.py`、`sgca.py`
4. configs / tests / reports：`configs/s3dis/**`、`tools/**`、`docs/**`

## 8. Compile / Test Results

已通过：

```text
python -m compileall tools/check_da_radius.py tools/benchmark_da_radius.py libs/pointops pointcept/models/kpconvx_hybrid pointcept/models/kpconvx configs/s3dis
python tools/check_da_radius.py --device cpu
python tools/check_da_radius.py --device cuda
python tools/benchmark_da_radius.py --num-points 1024 --batches 1 --nsample 16 --device cuda
```

CUDA extension 已成功编译：

```text
libs/pointops/_C.cp314-win_amd64.pyd
```

Linux 云端推荐编译：

```bash
cd libs/pointops
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
python setup.py build_ext --inplace
```

## 9. Benchmark Results

本机 sanity benchmark：

```text
da_radius_benchmark:
  points=512
  batches=1
  nsample=16
  radius_base=0.0800
  radius_scale_min=0.8500
  radius_scale_mean=1.1133
  radius_scale_max=1.3750
  radius_mean=0.0891
  adaptive_ball_query_ms=0.143
  adaptive_peak_mb=8.29
  adaptive_unique_neighbors_min=1.00
  adaptive_unique_neighbors_mean=2.21
  adaptive_unique_neighbors_max=5.00
  adaptive_outside_count=0
  torch_mask_ms=364.287
  torch_mask_peak_mb=10.83
  torch_mask_unique_neighbors_mean=2.21
  torch_mask_outside_count=0
  knn_query_ms=1.838
  knn_peak_mb=8.36
  knn_unique_neighbors_mean=16.00
```

云端建议执行：

```bash
python tools/benchmark_da_radius.py --num-points 8192 --batches 2 --nsample 16 --device cuda
python tools/benchmark_da_radius.py --num-points 32768 --batches 4 --nsample 32 --device cuda
```

## 10. Training Commands

推荐先跑 debug：

```bash
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v12-cuda-vs-torch-radius-check-area5.py
```

主结果：

```bash
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v8-da-radius-global-context-area5.py
```

增强结果：

```bash
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v9-da-kernel-radius-global-context-area5.py
```

核心消融：

```bash
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v5-da-kernel-only-ablation-area5.py
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v6-da-radius-only-ablation-area5.py
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v7-global-context-only-ablation-area5.py
```

## 11. Ablation Table Template

| Method | DA-Kernel | DA-Radius | Global Context | mIoU | mAcc | OA | Latency | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | No | No | No |  |  |  |  |  |
| DA-Kernel only | Yes | No | No |  |  |  |  |  |
| DA-Radius only | No | Yes | No |  |  |  |  |  |
| Global only | No | No | Yes |  |  |  |  |  |
| DA-Radius + Global | No | Yes | Yes |  |  |  |  |  |
| Full | Yes | Yes | Yes |  |  |  |  |  |

## 12. Paper Writing Suggestions

建议论文结构：

1. Motivation：fixed radius 无法适配 indoor point cloud 的非均匀密度。
2. Adaptive Density Estimation：用局部距离 / 邻居统计估计 density。
3. CUDA-level Density-Adaptive Radius Search：在邻居搜索阶段直接使用 per-point radius。
4. Radius-aware KPConvX：将 adaptive radius 传入 KPConvX local convolution。
5. Serialized Global Context Enhancement：用轻量 global context 补充 long-range dependency。
6. Complexity Analysis：分析 torch mask、CUDA direct search、空间索引优化的差异。
7. Ablation Study：DA-Kernel、DA-Radius、Global Context 三者拆分。
8. Limitations：当前功能版 CUDA search 仍需高性能空间索引优化。

推荐主贡献表述：

```text
Kernel point scaling changes local weighting, while DA-Radius changes the effective receptive field at neighborhood construction time.
```

## 13. Risks and Limitations

- 当前 adaptive ball query 是功能正确版，仍是 batch 内直接扫描，论文级速度应继续优化。
- 当前 checkout 缺少旧 pointops 部分 CUDA kernel 源，已用显式 stub 处理链接；若训练路径调用这些 op，需要补齐源码。
- DA-Kernel + DA-Radius + Global Context 同开时效果可能最好，但归因复杂；主论文建议以 DA-Radius + Global 为核心。
- Torch reference 路径适合 debug，不适合最终速度对比。
