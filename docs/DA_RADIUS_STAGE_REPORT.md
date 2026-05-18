# DA-Radius KPConvX 阶段报告

## Stage 1：Code Review Only

邻居搜索入口位于 `pointcept/models/kpconvx/utils/torch_pyramid.py`。KPConvX / KPConvXHybrid 的邻居索引主要通过 `pointops.knn_query(...)` 生成，`neighbor_limits` 被作为 KNN 的 `nsample` 使用；当前代码并不是严格 fixed-radius ball query。已有 DA-Kernel 只改变 kernel point 的空间尺度，原始邻居集合没有随 per-point density 重新构建。

可修改位置：

- Python pipeline：`pointcept/models/kpconvx/utils/torch_pyramid.py`
- KPConvXHybrid blocks：`pointcept/models/kpconvx_hybrid/*.py`
- pointops Python wrapper：`libs/pointops/functions/query.py`
- pointops C++ binding：`libs/pointops/src/pointops_api.cpp`
- ball query CUDA/C++：`libs/pointops/src/ball_query/*`

PTv3 审查结论：`pointcept/models/point_transformer_v3` 已有 serialization、patch attention、shuffle order、serialized pooling、CPE/xCPE 类似模块，可作为全局上下文设计参考；不建议把完整 PTv3 backbone 直接复制进 KPConvXHybrid。`kpconvx_hybrid` 已有 SGCA global context，可作为轻量全局模块基础。

## Stage 2：Torch Reference DA-Radius

已实现 torch reference 路径，用于调试和 fallback：

- `DensityAdaptiveRadius`
- DA-Radius mask
- `use_da_kernel`、`use_da_radius`、`use_global_context` 独立开关
- `da_radius_backend="torch"` fallback

核心行为：

- dense region 缩小 effective radius
- sparse region 放大 effective radius
- 关闭 DA-Radius 时尽量恢复原行为
- DA-Kernel 保留为 ablation，不再作为唯一主贡献

## Stage 3：CUDA-level Adaptive Radius Search

已在 `libs/pointops` 增加 per-query radius CUDA op：

- `adaptive_ball_query_cuda`
- `pointops.adaptive_ball_query`
- `da_radius_backend="cuda"`
- `adaptive_radius_search_pack_mode(...)`

本机编译问题已处理：

- 已安装并使用 MSVC BuildTools 2022。
- 已安装 CUDA Toolkit 12.8 到 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`。
- PyTorch 为 `2.9.1+cu128`，与 CUDA 12.8 匹配。
- `setup.py` 已禁用 ninja，避免 Windows 下 `CreateProcess failed`。
- `setup.py` 已修正源码包布局，使扩展生成到 `pointops._C`。
- 补充了简单 CUDA KNN launcher，保证当前 KPConvX 依赖的 `pointops.knn_query` 可导入可运行。

重要限制：

当前 checkout 缺少旧 pointops 的大部分 CUDA kernel 源实现，例如 grouping、interpolation、sampling、attention、aggregation 等 launcher 只有头文件声明，没有 `.cu` 实现。为避免链接失败，已加入显式 stub；这些旧算子如果被调用，会抛出“当前源码缺失 CUDA kernel”的错误。Stage 3 的 `knn_query`、`ball_query`、`adaptive_ball_query` 已可编译和运行。

编译命令：

```powershell
cd E:\Program\python\Pointcept\libs\pointops
$env:CUDA_HOME='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_PATH=$env:CUDA_HOME
$env:PATH="$env:CUDA_HOME\bin;$env:PATH"
python setup.py build_ext --inplace
```

本机实际编译结果：

- `pointops._C` 编译成功。
- 生成物：`libs/pointops/_C.cp314-win_amd64.pyd`
- 编译中仍有 PyTorch/MSVC 编码 warning，但不影响生成扩展。

CUDA sanity 结果：

```text
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
  adaptive_ball_query_ms=0.285
  knn_query_ms=3.139
```

## Stage 4：Ablation 开关

已保留并拆分三类贡献：

- `use_da_kernel`
- `use_da_radius`
- `use_global_context`

建议主结果优先使用 `DA-Radius + Global Context`，DA-Kernel 作为增强项或消融项。DA-Kernel 与 DA-Radius 不完全冗余：前者改变 kernel point/influence 尺度，后者改变实际邻居集合；但二者同时开启时归因会变复杂，因此论文主贡献应突出 DA-Radius。

## Stage 5：PTv3-style Global Context

当前实现方向是复用 `kpconvx_hybrid` 已有 SGCA/global context 框架，而不是直接拷贝 PTv3 backbone。推荐插入 stage 4/5 或 bottleneck，避免低层局部几何被全局模块过早平滑。

## Stage 6：Config Files

已新增 S3DIS 配置，包括：

- V1 conservative
- V2 balanced
- V3 stronger
- V4 torch reference
- V5 DA-Kernel only
- V6 DA-Radius only
- V7 Global Context only
- V8 DA-Radius + Global Context
- V9 DA-Kernel + DA-Radius + Global Context
- V10 constant influence ablation
- V11 linear influence ablation
- V12 cuda vs torch radius check

推荐论文主配置：

```text
V8 DA-Radius + Global Context
V9 DA-Kernel + DA-Radius + Global Context 作为增强版/补充结果
```

## 测试结果

已通过：

```text
python -m compileall tools/check_da_radius.py libs/pointops pointcept/models/kpconvx_hybrid pointcept/models/kpconvx configs/s3dis
python tools/check_da_radius.py --device cpu
python tools/check_da_radius.py --device cuda
```

CPU sanity：

```text
radius_scale_min=0.8500
radius_scale_mean=1.0884
radius_scale_max=1.3750
valid_ratio=0.1243
```

CUDA sanity：

```text
outside_count=0
brute_count_mean=2.17
cuda_unique_count_mean=2.17
adaptive_ball_query_ms=0.285
knn_query_ms=3.139
```

## 后续建议

1. 若要完整训练依赖 `pointops.grouping/interpolation/sampling/attention` 的模型，需要补齐这些旧 CUDA kernel 源，或改用仓库中已有的其他可用 op。
2. 当前 CUDA adaptive ball query 是功能验证版，复杂度是直接扫描 batch 内点；论文级速度还需要空间索引、grid/hash/binning 或复用现有高性能 radius search。
3. 正式实验应记录每 stage 的 radius scale、neighbor count、GPU memory、forward latency、mIoU/mAcc/类别 IoU。

## Linux / 云端训练说明

该实现可以迁移到 Linux 云端环境。新增的 CUDA/C++ 源没有使用 Windows 专有 API，`setup.py` 中的 MSVC 分支只在本机 Windows 下生效，Linux 会自然跳过。

推荐在 Linux 云端执行：

```bash
cd libs/pointops
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
python setup.py build_ext --inplace
```

验证命令：

```bash
python tools/check_da_radius.py --device cuda
python tools/benchmark_da_radius.py --num-points 8192 --batches 2 --nsample 16 --device cuda
```

如果云端环境缺少旧 pointops kernel 源，训练能否直接跑取决于模型是否会调用那些旧算子。`knn_query`、`ball_query` 和 `adaptive_ball_query` 已可用；其余缺失算子需要补源或切换实现路径。
