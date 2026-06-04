# DA-Radius 模型图绘制要求

本文档用于规范 DA-Radius / KPConvXHybrid 相关论文插图的绘制。最终交付物必须是可编辑的 draw.io 文件，不只是一张导出的 PNG。

## 目标

需要绘制一组用于论文、汇报或实验说明的模型结构图，说明当前实现的 `KPConvXHybrid` 如何在 KPConvX 主干上加入：

- DA-Radius per-point adaptive radius 邻域构建
- DA-Kernel 密度自适应核点缩放
- SGCA / serialized patch global context
- GeometryDifficultyRouter 难度路由
- DecoderRefineHead 解码细化头

图件应参考 `docs/image/` 中已有图片的风格，尤其是：

- `KPconvx总框架图.png`
- `框架展开图.png`
- `KPconvx模块前后对比图.png`
- `模块说明图.png`

## 交付格式

主交付文件：

- `docs/DA_RADIUS_MODEL_FIGURES.drawio`

要求：

- 必须是 draw.io 可直接打开和继续编辑的 `.drawio` 文件。
- 每个模块、箭头、公式、说明文字都应保持为可编辑元素。
- 不允许只交付截图、PNG、JPG 或不可编辑图片。
- 如需导出 PNG，可作为附加产物，但不能替代 `.drawio` 源文件。

## 页面规划

建议使用一个多页 draw.io 文件，至少包含以下 5 页。

### Fig1 总体框架

表达 `KPConvXHybrid` 的完整 encoder-decoder 流程。

必须包含：

- 输入点云：`coord / feat / offset`
- Stem：`KPConv / KPConvX`
- Encoder stages 1-5
- Stage 2/3/4 的 DA-Radius + DA-Kernel
- Stage 3/4 的 Router
- Stage 4/5 的 SGCA global context
- Grid Maxpool / Grid Up
- Decoder 上采样与 skip connection
- DecoderRefineHead
- Segmentation Head / logits

重点：

- DA-Radius 是局部邻域构建路径，建议用蓝色突出。
- SGCA 是高层全局上下文路径，建议用绿色突出。
- Decoder 路径建议用紫色。
- Refine 模块建议用红色。

### Fig2 DA-Radius 邻域构建

表达 DA-Radius 的机制，不要只画成普通流程图。

必须包含：

- 基础 `build_full_pyramid()` 初始邻域估计
- `DensityAdaptiveRadius`
- 邻域距离统计：`mean_neighbor_distance`
- 密度估计公式：`rho_i = 1 / (mean_dist_i + eps)`
- 半径缩放：`radius_i = base_radius_l * scale_i`
- dense 区域半径更小，sparse 区域半径更大
- `pointops.adaptive_ball_query`
- 输出新的 `neighbors[l]`

需要标明：

- CUDA 路径：`da_radius_backend="cuda"`
- Torch 路径：fallback / reference / ablation
- Linux GPU 训练前需要编译 `libs/pointops`

### Fig3 算子对比

表达 Baseline KPConvX、DA-Kernel、DA-Radius 三者的区别。

必须包含三栏：

- Baseline KPConvX：固定半径 ball query，邻域不自适应
- DA-Kernel：`p_k_da = s_i * p_k`，改变 kernel points / influence，不改变邻域搜索
- DA-Radius：`r_i = radius_l * scale_i`，改变 local graph construction，重新搜索邻域

叙述重点：

- DA-Radius 是主贡献，因为它改变邻域构建。
- DA-Kernel 保留为独立消融模块，不要把它画成 DA-Radius 的替代品。

### Fig4 SGCA / Router / Refine 子模块

表达三个辅助模块如何接入主干。

SGCA 必须包含：

- stage features + coords
- serialization order
- split patches
- patch attention / MultiheadAttention
- MLP residual

Router 必须包含：

- neighbor stats
- `mean_dist`
- `dist_var`
- `feat_var`
- difficulty score
- global weight
- 融合公式：`f_out = f + global_weight * (SGCA(f) - f)`

Refine 必须包含：

- decoder features
- original coords
- stage-1 neighbors
- boundary / local contrast cues
- lightweight residual refine

### Fig5 图例与实现映射

作为说明图，用于统一颜色、开关和代码位置。

必须包含：

- 图例颜色说明
- 关键开关
- 实现文件映射

关键开关：

- `use_da_radius` / `enable_da_radius`
- `da_radius_backend`
- `da_radius_stages`
- `use_da_kernel` / `enable_da`
- `use_global_context` / `enable_global`
- `global_context_type="serialized_patch"`
- `router_stages`
- `enable_refine`

实现文件映射：

- `pointcept/models/kpconvx_hybrid/kpconvx_hybrid.py`
- `pointcept/models/kpconvx_hybrid/kpx_stage2.py`
- `pointcept/models/kpconvx_hybrid/sgca.py`
- `pointcept/models/kpconvx_hybrid/geometry_router.py`
- `pointcept/models/kpconvx/utils/torch_pyramid.py`
- `libs/pointops/functions/query.py`
- `libs/pointops/src/ball_query/*`

## 视觉风格

整体风格应接近论文插图，而不是软件 UML 图。

要求：

- 白色背景。
- 粗黑主箭头。
- 关键模块使用浅色底板。
- 模块边框清晰，线宽建议 2-3。
- 使用虚线框表达路径范围、展开关系或说明区域。
- 使用圆形、矩形、网格块、并列分栏等论文图常见视觉元素。
- 文字可中英混排，但术语应保持实现一致。
- 公式建议使用可编辑文本，不要贴图。

建议颜色：

- DA-Radius / DA-Kernel：浅蓝 `#e1f5fe`，边框 `#29b6f6`
- SGCA / global context：浅绿 `#e8f5e9`，边框 `#43a047`
- Router / difficulty：浅黄 `#fff8e1`，边框 `#f9a825`
- Decoder：浅紫 `#f3e5f5`，边框 `#8e24aa`
- Refine：浅红 `#ffebee`，边框 `#e53935`
- 普通说明框：白色或浅灰

## 内容准确性要求

图中不能表达成以下错误含义：

- 不要把 DA-Radius 画成只在 KPConvX block 内部改变权重。
- 不要把 DA-Kernel 画成重新搜索邻域。
- 不要把 SGCA 画成所有 stage 都默认使用；当前重点是高层 stage。
- 不要把 Router 画成控制 Fine / Coarse local branches；当前 Router 主要控制 SGCA residual 注入。
- 不要把 DA-Radius 的 CUDA 路径画成强制唯一实现；Torch backend 仍可作为 fallback / reference。

## 验收标准

完成后的 `.drawio` 文件需要满足：

- draw.io 可以直接打开。
- 至少包含 5 个页面。
- 每页标题清楚。
- 每个模块、箭头、说明文字可编辑。
- 总体图能看出完整 KPConvXHybrid forward pipeline。
- DA-Radius 图能明确看出 per-point radius 改变邻域搜索。
- 算子对比图能区分 DA-Kernel 和 DA-Radius。
- 子模块图能表达 SGCA、Router、Refine 的接入方式。
- 图例页能说明颜色、开关和代码文件。

