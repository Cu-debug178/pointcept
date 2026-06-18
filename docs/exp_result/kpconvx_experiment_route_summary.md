# KPConvX 改进实验总路线

## 总路线

当前主线不是简单调参，而是从 KPConvX baseline 出发，逐步尝试：

`局部动态半径 DA-Radius -> 全局上下文 SGCA -> PTv3-inspired 双注意力 -> 边界风险控制 -> v16 内化式 DA-stat + 轻量全局 context mixer`

整体认知已经从“外挂模块叠加”转向“把局部几何统计内化进 KPConvX 主干，再谨慎加入可忽略的深层全局上下文”。

## 版本时间线

| 阶段 | 版本 | 改进内容 | 目的/结论 |
| --- | --- | --- | --- |
| Baseline | 原始 KPConvX | 标准 KPConvX，Area5 S3DIS | best mIoU 约 0.7159，目前仍是最强基线 |
| 初始混合版 | sgca-da-refine-v1m1 | DA-Kernel、SGCA、refine 等模块的早期综合框架 | 变量太多，后续开始拆成单模块消融 |
| DA-Radius 初探 | v1 | CUDA DA-Radius，较保守半径 `(0.9, 1.25)`，stage 2/3/4 | 验证 CUDA 动态半径可跑 |
| DA-Radius 平衡版 | v2 | CUDA DA-Radius，范围 `(0.85, 1.45)`，强度 0.75 | 半径动态更明显 |
| DA-Radius 强版 | v3 | CUDA DA-Radius 扩到 stage 2/3/4/5，范围 `(0.8, 1.6)` | 太激进，作为强扰动测试 |
| Torch 参考版 | v4 | Torch backend DA-Radius，范围 `(0.9, 1.35)` | 用 torch mask 路径对照 CUDA |
| DA-Kernel 消融 | v5 | 只开 DA-Kernel，关 DA-Radius / global / refine | 判断 kernel scaling 本身是否有收益 |
| DA-Radius 消融 | v6 | 只开 DA-Radius torch，关 global / refine | 判断 radius 单模块效果 |
| Global only | v7 | 只开 serialized global context，stage 4/5 | 判断全局上下文本身是否有收益 |
| Radius + Global | v8 | CUDA DA-Radius + global context | 测局部半径和全局上下文是否互补 |
| Kernel + Radius + Global | v9 | DA-Kernel + DA-Radius + global context | 早期“全模块组合”尝试 |
| Influence 消融 | v10 | 基于 v9，`kp_influence="constant"` | 看 KPConvX influence 类型影响 |
| Influence 消融 | v11 | 基于 v9，`kp_influence="linear"` | 与 v10 对照 |
| CUDA/Torch 对照 | v12 | torch DA-Radius mask，stage 2/3/4，关 global/refine | 为 v13 CUDA 版本做干净参照 |
| 稳定 DA-Radius 主线 | v13 | CUDA DA-Radius，只在 stage 3/4，范围更保守，`apply_block_mask=False` | best mIoU 0.7096；整体略低 baseline，但 board/window/sofa/beam 等结构类有价值信号 |
| v13 参数修正 | v13b | `clip_grad=2.0`，neighbor limit 提到 24，半径强度降到 0.35 | best mIoU 0.6841，失败；说明“更保守半径 + 更大邻居上限”没有转化为收益 |
| 旧 v14 | encoder-SGCA | 在 encoder stage 4/5 直接融合 SGCA | 约 90 epoch 提前止损，best 约 0.5883；说明 encoder 强融合会污染主干/skip feature |
| 新 v14 | decoder-SGCA | SGCA 改到 decoder/head 前，`gamma=0` 初始化 | best mIoU 0.7004；明显好于旧 v14，但仍低于 v13/baseline，说明融合位置对了，但 SGCA 信息质量不够 |
| v15 基础版 | dual-attention | PTv3-inspired serialized patch，全局 stage 3/4，decoder fusion | 尝试更接近 PTv3 的全局 token bank |
| v15 4090D | dual-attention-4090d | 增大 token 上限，适配 4090D 显存 | 工程适配版本 |
| v15 stage234 | boundary-gated dual attention | 全局 stage 2/3/4 + boundary risk gate + boundary loss + refine | best 约 0.5971，失败；boundary loss 会下降，但 segmentation 没提升，判断为负迁移/结构失配 |
| v16 初版 | DA-stat GC | 新建 KPConvXV16，DA-Radius 改成局部统计条件器，shell-aware kernel bias，同时打开 decoder 轻量 GC mixer | best 约 0.5916，失败；说明第一版同时打开 shell bias + GC mixer 仍然过强，不能作为主线结论 |
| v16-local | DA-stat local | 不改邻域图，只把 DA-Radius 统计作为 detached DA-meta 注入 KPConvX block；stage 3/4，channel-wise bias，关闭 GC | best 约 0.6977；明显比失败的 v16 初版稳定，说明“局部几何统计内化进 kernel attention”有一定价值，但还没有复现 v13 的动态半径收益 |
| v16b 对照 | superset-only | 仅增大 stage 3/4 的候选邻居上限，不启用 support mask | 用来判断收益是否只是来自更大的候选邻域，而不是 adaptive support 本身 |
| v16b 主实验 | controlled support | 在 v16-local 上加入 stage 3/4 受控 support mask：静态 superset graph + block 内 effective-radius mask + warmup/ramp | 当前局部主线；目标是保留 v13 的结构类收益，同时避免 v13b 那种保守半径削弱结构信号 |
| v16c 新实验 | light-global | 在 v16b controlled support 上，只在 decoder/head 前加入轻量 GC mixer；`gamma=0` 初始化，不启用 token bank、boundary、refine | 当前新制作的全局分支实验；目标不是替代局部主干，而是给 final semantic feature 做低侵入 cloud-level context correction |

## 关键认知变化

### v13 是第一个真正有价值的局部几何版本

v13 没有超过 baseline，但证明 DA-Radius 对部分结构类可能有帮助，尤其是 board、window、sofa、beam 等结构类。

### v13b 说明继续调半径参数不是主线

更保守的半径、更大的邻居上限、更松的梯度裁剪，没有带来收益，反而削掉了结构类信号。

### v14 说明全局上下文的位置非常关键

encoder 强融合失败，decoder/head 前融合明显更稳。这说明全局信息不能过早污染 encoder skip feature。

### v15 说明 PTv3-style 全局分支不能生硬拼接

boundary 辅助任务能学会，但主分割变差，说明问题不是“边界没学到”，而是这个边界/全局机制没有真正服务 segmentation。

### v16 是路线转折

不再把 DA-Radius 当强动态邻域重建器，也不再把 PTv3 当大分支硬接。当前方向改成：

`KPConvX 主干 + DA 局部统计条件化 kernel attention + 轻量 decoder global context`

v16 之后路线被拆成更清楚的消融顺序：

1. `v16-local`：先验证不改邻域图时，DA-meta 是否能让 KPConvX block 感知局部密度、有效邻居比例和距离离散度。
2. `v16b`：如果只给统计不够，就引入受控 support mask，让 stage 3/4 有温和的有效支持集变化。
3. `v16c-light-global`：在局部分支稳定后，只在 decoder/head 前加一个很弱的全局残差，观察它是否能补 door、bookcase、chair、column 等语义类，而不牺牲 board、beam、window 等结构类。

这个拆分的核心价值是隔离变量：先判断局部几何统计有没有用，再判断 support set 自适应是不是必要，最后才判断轻量全局上下文是否能提供额外补偿。

## 一句话总结

当前路线从“外挂模块叠加”逐渐转向“把局部几何统计内化进 KPConvX 主干，再用受控 support mask 恢复局部邻域收益，最后谨慎加入可忽略的深层全局上下文”。这是比 v15 更稳、更工程化的方向。
