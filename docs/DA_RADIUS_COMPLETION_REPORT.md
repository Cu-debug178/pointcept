# DA-Radius Local-Global KPConvX 收尾报告

## 2026-07-20 CUDA 扩展优化

根据 `GPT-5.5Pro/cuda-extension-speed-review` 中保存的 GPT-5.6 Pro 建议及批判性裁决，已完成语义保持的第一、二阶段优化：

- 新增 `adaptive_ball_query_idx`，训练路径不再分配/写回 `dist2`，也不再启动 Python `sqrt` kernel；旧距离接口继续保留。
- adaptive kernel 找满 `nsample` 后提前退出，并删除无输出用途的 nearest-neighbor 临时变量。
- launcher 显式使用 PyTorch current CUDA stream，并在 launch 后调用 `C10_CUDA_KERNEL_LAUNCH_CHECK()`。
- kernel 使用真实 batch 数作为 `new_offset` 扫描上界；C++ binding 增加 device、dtype、shape、contiguous 和同设备校验。
- 保持既有 first-hit 截断与 `-1` shadow padding 语义，没有改成 nearest-K 或 fallback 重复邻居。
- DA-Radius 不再第二次完整执行 `build_full_pyramid()`；复用第一次生成的 points、lengths、pool、upsample 和非 DA stage neighbors，只替换选中的 DA stage neighbor lists。

本机环境：Python 3.14、PyTorch 2.9.1+cu128、CUDA Toolkit 12.8 编译链、GTX 1650。扩展已完整编译并通过：

```text
python tools/check_da_radius.py --device cuda
indices_only_equal_legacy=True
outside_count=0
adaptive_ball_query_idx_ms=0.343
adaptive_ball_query_legacy_ms=0.460

python tools/benchmark_da_radius.py --num-points 1024 --batches 2 \
  --nsample 16 --radius 0.2 --warmup 10 --repeat 50 --device cuda
adaptive_ball_query_idx_ms=0.163
adaptive_ball_query_legacy_ms=0.384
adaptive_indices_equal_legacy=True
adaptive_shadow_ratio=0.1399
adaptive_outside_count=0

python -m pytest tests -q
60 passed, 2 skipped, 11 subtests passed
```

额外检查已覆盖空 query、双 batch 隔离、非默认 CUDA stream、错误 dtype 拒绝，以及 pyramid 结构对象复用。以上时延只代表本机小规模 sanity，不外推为 4090D 或真实 S3DIS batch 的收益。

仍保留的主要限制是 adaptive query 的 batch 内 direct scan 复杂度。candidate reuse、voxel/hash 搜索和手工 buffer 缓存没有实施，因为现有证据不足以证明它们能保持邻居语义或提供端到端收益；应先在真实训练环境测量各 stage 时间与点数。

### 当前 CUDA 路径使用方法

拉取代码或修改 CUDA 源后必须重新编译本地扩展：

```bash
cd libs/pointops
python setup.py build_ext --inplace
cd ../..

python -c "import sys; sys.path.insert(0, 'libs'); import pointops; assert callable(pointops.adaptive_ball_query_idx); print('optimized pointops CUDA ready')"
python tools/check_da_radius.py --device cuda
```

模型配置只需要启用 DA-Radius 并选择 CUDA backend：

```python
model = dict(
    backbone=dict(
        use_da_radius=True,
        da_radius_backend="cuda",
        da_radius_apply_block_mask=False,
    )
)
```

`da_radius_backend="cuda"` 会自动调用 `adaptive_ball_query_idx`，并在 `KPConvXStage2._build_pyramid()` 中复用第一次 pyramid 的结构，不需要增加新的配置开关。旧的 `adaptive_ball_query` 距离接口仅用于兼容和 correctness 对照。

当前干净的 CUDA DA-Radius 训练配置与命令：

```bash
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v13-da-radius-cuda-only-area5.py
```

Torch reference 仍使用 v12 配置，只用于语义检查，不作为速度训练路径：

```bash
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v12-cuda-vs-torch-radius-check-area5.py
```

## 1. 当前完成状态

本轮继续完成的是 Stage 7/8/9 的收尾验证与可运行性修复。此前已经完成的核心内容包括：

- Stage 1：定位 KPConvX / KPConvXHybrid 邻居搜索链路。
- Stage 2：实现 Torch reference DA-Radius fallback。
- Stage 3：实现 CUDA-level per-point adaptive radius search。
- Stage 4：保留 DA-Kernel，并与 DA-Radius / Global Context 独立开关。
- Stage 5：加入轻量 PTv3-style serialized global context。
- Stage 6：生成 V1-V12 S3DIS 配置。

本轮新增修复：

- 从仓库根目录运行工具脚本时，自动把 `libs` 加入 `sys.path`，避免 `import pointops` 失败。
- `pointcept/models/kpconvx/utils/torch_pyramid.py` 同样加入本地 `libs/pointops` fallback，降低训练入口依赖手动 `PYTHONPATH` 的风险。

## 2. 本轮修改文件

- `tools/check_da_radius.py`
- `tools/benchmark_da_radius.py`
- `pointcept/models/kpconvx/utils/torch_pyramid.py`

修改目的：

- 让 `python tools/check_da_radius.py --device cuda` 可以直接从仓库根目录运行。
- 让 `python tools/benchmark_da_radius.py ...` 可以直接从仓库根目录运行。
- 让 KPConvX pyramid 构建阶段能找到本地编译的 `pointops._C`。

## 3. Stage 7 测试结果

### 3.1 语法检查

已通过：

```text
python -m compileall pointcept\models\kpconvx_hybrid pointcept\models\point_transformer_v3 configs tools\check_da_radius.py tools\benchmark_da_radius.py
python -m compileall pointcept\models\kpconvx\utils\torch_pyramid.py tools\check_da_radius.py tools\benchmark_da_radius.py
```

### 3.2 CPU reference sanity

已通过：

```text
python tools\check_da_radius.py --device cpu

torch_reference:
  points=96 k=16
  radius_scale_min=0.8500
  radius_scale_mean=1.0884
  radius_scale_max=1.3750
  valid_ratio=0.1243
```

### 3.3 CUDA sanity

已通过：

```text
python tools\check_da_radius.py --device cuda

torch_reference:
  points=96 k=16
  radius_scale_min=0.8500
  radius_scale_mean=1.1093
  radius_scale_max=1.3750
  valid_ratio=0.0983
cuda_query:
  idx_shape=(128, 16)
  max_distance=0.3187
  outside_count=0
  brute_count_mean=2.17
  cuda_unique_count_mean=2.17
  adaptive_ball_query_ms=0.214
  knn_query_ms=5.142
```

结论：

- CUDA adaptive ball query 没有返回 adaptive radius 外的邻居，`outside_count=0`。
- CUDA 与 brute-force 统计均值一致，`brute_count_mean=2.17`，`cuda_unique_count_mean=2.17`。

### 3.4 小规模 benchmark

已通过：

```text
python tools\benchmark_da_radius.py --num-points 512 --batches 1 --nsample 16 --warmup 2 --repeat 3 --device cuda

adaptive_ball_query_ms=0.185
adaptive_peak_mb=8.29
adaptive_unique_neighbors_mean=2.14
adaptive_outside_count=0
torch_mask_ms=293.548
torch_mask_peak_mb=10.83
torch_mask_unique_neighbors_mean=2.14
torch_mask_outside_count=0
knn_query_ms=1.884
knn_peak_mb=8.36
knn_unique_neighbors_mean=16.00
```

### 3.5 较大规模 benchmark

已通过：

```text
python tools\benchmark_da_radius.py --num-points 8192 --batches 2 --nsample 16 --warmup 1 --repeat 3 --device cuda

adaptive_ball_query_ms=1.311
adaptive_peak_mb=10.78
adaptive_unique_neighbors_mean=9.95
adaptive_outside_count=0
torch_mask_ms=4834.264
torch_mask_peak_mb=236.41
torch_mask_unique_neighbors_mean=9.95
torch_mask_outside_count=0
knn_query_ms=36.286
knn_peak_mb=11.81
knn_unique_neighbors_mean=16.00
```

结论：

- CUDA DA-Radius 明显快于 Torch mask reference。
- CUDA DA-Radius 在 8192 点规模下仍保持 `outside_count=0`。
- 当前 direct-scan CUDA 实现已经可用于功能验证和小中规模 sanity，但论文级速度仍建议继续做 grid/hash/binning 空间索引优化。

## 4. Stage 8 论文方向确认

推荐论文方法名：

```text
Density-Adaptive Local-Global KPConvX
```

论文贡献应这样组织：

1. Fixed-radius / fixed-KNN 难以适配 indoor point cloud 的非均匀密度。
2. DA-Kernel 只改变 kernel point / influence 的空间尺度，不改变实际 neighbor set。
3. DA-Radius 在邻域构建阶段直接使用 per-point adaptive radius，改变 effective receptive field。
4. CUDA-level adaptive ball query 避免只做“大半径候选 + mask”的保守实现。
5. PTv3-style serialized global context 补足 KPConvX local convolution 的长程建模不足。

主结果建议使用：

```text
V8: DA-Radius + Global Context
```

补充最强结果可使用：

```text
V9: DA-Kernel + DA-Radius + Global Context
```

但论文主贡献归因应优先放在 DA-Radius 与 Global Context，不建议把 DA-Kernel 作为主创新点。

## 5. Stage 9 最终实验顺序

云端 Linux 训练前先执行：

```bash
cd libs/pointops
python setup.py build_ext --inplace
cd ../..
python tools/check_da_radius.py --device cuda
python tools/benchmark_da_radius.py --num-points 8192 --batches 2 --nsample 16 --device cuda
```

推荐训练顺序：

```bash
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v12-cuda-vs-torch-radius-check-area5.py
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v6-da-radius-only-ablation-area5.py
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v8-da-radius-global-context-area5.py
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v5-da-kernel-only-ablation-area5.py
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v7-global-context-only-ablation-area5.py
python tools/train.py --config-file configs/s3dis/semseg-kpconvx-hybrid-v9-da-kernel-radius-global-context-area5.py
```

## 6. 剩余风险

- 当前 CUDA adaptive ball query 是 direct-scan 功能版，还不是最终高性能空间索引版。
- 当前 checkout 曾缺少部分旧 pointops kernel 源实现；如果完整训练路径调用 grouping / interpolation / sampling / attention / aggregation 等 stub，需要补齐对应 CUDA kernel 或切换实现。
- Windows 本地验证已经通过 CUDA sanity，但最终论文实验应以 Linux 云端训练日志为准。
- 真实训练仍需要记录 mIoU、mAcc、OA、per-class IoU、forward latency、peak GPU memory、per-stage radius scale 和 neighbor count。

## 7. 当前可进入实验验证

可以开始实验验证。最先跑 V12，确认 CUDA / torch 行为一致；然后跑 V6 验证 DA-Radius 主贡献，再跑 V8 作为主结果候选。
