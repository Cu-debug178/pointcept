# v17 alpha 消融实现记录

## 背景

GPT-5.5Pro 和 DeepSearch 都认为 v17 dual-support 目前不能算成功，但也不能直接判定 support expansion 完全无效。当前证据更像是：

- expanded support 对 beam、board、table、chair、sofa 等类别存在局部正信号；
- door、window、column、wall、clutter 等边界或墙面附着类出现系统性损失；
- v17 的 residual branch 后期不再是小残差，`gamma`、`gate_mean`、`residual_ratio` 都显示 expanded support 已经有较强影响；
- v17 训练成本很高，直接重训新版本前必须先做低成本诊断。

因此本次优先实现 **inference-only alpha 消融**，先回答一个关键问题：

```text
v17 learned expanded branch 在推理时到底是帮忙，还是拖累？
```

## 实现内容

### 1. v17 backbone 增加推理期 alpha

文件：

- `pointcept/models/kpconvx_hybrid/kpx_v17.py`

新增参数：

```python
dual_support_eval_alpha=None
```

行为：

- 训练时不生效，保持原 v17 训练逻辑；
- `model.eval()` 时，如果 `dual_support_eval_alpha` 不为 `None`，则用它缩放 dual-support residual；
- alpha 小于等于 0 时跳过 expanded support 路径计算，但仍保留 stage4 原始 support 的 base limit，保证 `alpha=0` 表示 v17 的 original-support path，而不是错误地使用完整 candidate support。

新增监控：

- `v17_dual_alpha`
- `v17_s4_dual_alpha`
- `v17_s4_dual_effective_gamma`

### 2. DAKPConvX 支持 dual residual alpha

文件：

- `pointcept/models/kpconvx_hybrid/da_kpnext_blocks.py`

核心公式从：

```text
dual_delta = gamma * progress * gate * residual
```

变为：

```text
dual_delta = alpha * gamma * progress * gate * residual
```

其中 `alpha` 由 v17 backbone 在推理阶段传入。

当 `alpha=0` 或 `progress=0` 时，直接跳过 expanded aggregation，减少无效计算。

### 3. 新增 validation-only alpha sweep 工具

文件：

- `tools/eval_v17_alpha_sweep.py`

用途：

- 加载同一个 v17 checkpoint；
- 在 `data.val` 上依次评估多个 alpha；
- 不跑 `SemSegTester` / precise eval；
- 输出 mIoU、mAcc、allAcc、每类 IoU 和 v17 monitor 到 CSV；
- 每个 alpha 使用同一个 seed 重建 val loader，尽量保证 random SphereCrop 可比。

默认 alpha：

```text
0.0 / 0.25 / 0.5 / 0.75 / 1.0
```

## 推荐运行命令

云端项目路径按当前习惯使用：

```bash
cd /root/autodl-tmp/Pointcept
source /root/autodl-tmp/Pointcept/activate_env.sh

python tools/eval_v17_alpha_sweep.py \
  --config-file configs/s3dis/semseg-kpconvx-hybrid-v17-dual-support-area5.py \
  --weight exp/s3dis/v17-dual-support_20260706_2303/model/model_best.pth \
  --save-path exp/s3dis/v17-alpha-sweep-best \
  --alphas 0 0.25 0.5 0.75 1.0
```

如果云端 checkpoint 实际路径不在 `exp/s3dis/` 下，先用：

```bash
find exp -path '*v17-dual-support*model_best.pth'
```

找到后替换 `--weight`。

## 结果判断

### 情况 A：`alpha < 1.0` 明显更好

说明 expanded support 有用，但 v17 原始强度太大或选择性不足。

下一步：

- 做 strength-constrained v17；
- 限制 effective gamma；
- 延长 warmup/ramp；
- 目标是让 residual ratio 后期控制在更小范围。

### 情况 B：`alpha = 0` 最好

说明 learned expanded branch 在推理时是净负贡献。

下一步：

- 暂停 v17 support expansion 方向；
- 回到 KPConvX/DA-meta 稳定性审计，或改做 boundary/compatibility 选择机制前先证明选择信号有效。

### 情况 C：所有 alpha 都不理想

说明问题不只是 residual 强度，而是 expanded support 选到的邻域本身可能不可靠。

下一步：

- 不再做单纯 gamma/range 调参；
- 优先研究 boundary-selective support 或 same/different support routing。

## 重点观察类别

不要只看 mIoU，需要重点看这些类：

- 正信号类：`beam`、`board`、`table`、`chair`、`sofa`
- 风险类：`door`、`window`、`column`、`wall`、`clutter`

理想结果不是某个 rare class 单点暴涨，而是：

- 风险类恢复；
- 正信号类没有完全消失；
- tail 稳定性和 mIoU 同时改善。

## 为什么这是当前最该实现的

这一步不需要重新训练，可以用很低成本判断 v17 的核心分支是否值得继续。相比直接做 v17b、boundary branch 或 global branch，它能先回答路线是否成立，避免继续烧 40 小时级别的 GPU 时间。
