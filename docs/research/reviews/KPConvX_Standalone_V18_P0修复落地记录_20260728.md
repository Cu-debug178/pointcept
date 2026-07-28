# KPConvX Standalone V18 P0 修复落地记录

日期：2026-07-28

## 输入材料

- 审查原文：`KPConvX_Standalone_V18_P0集成审查结论原文_20260728.md`
- 审查原文 SHA256：
  `947F5DD56C73B583AF5B14C4DED0616ADD50349E5299E2392B0EFCEB673A3034`
- 临时补丁原路径：`D:/Cu/KPConvX_V18_P0_fixes.patch`
- 临时补丁 SHA256：
  `C0DCE8DA8E92C98798A9129AA9BBC0BD5F8580E0011D3316E954CC20624EA6E0`

审查原文按字节原样归档。临时补丁仅作为输入材料，应用和验证完成后按
用户要求删除，不在项目中继续保留。

## 已确认的 P0

1. V18 从 `candidate_k=40/48` 切出前 `H=20` 时得到非连续 tensor，官方
   `index_select` 随后调用 `indices.view(-1)`，真实 forward 会失败。
2. Ring 位于每个 encoder stage 的最后一个 block 之后；在
   `grid_pool=True` 时该 block 已投影到下一 stage 宽度。初版却按当前
   layer 输入宽度构建 ring，stage 3/4 分别出现 `128->192`、`192->256`
   通道不匹配。

## 落地修改

- identity 邻居切片局部调用 `.contiguous()`，不修改官方公共
  `index_select`；
- ring 通道按 post-encoder 实际宽度构建，S3DIS stage 3/4 为
  `192/256`；
- `_apply_ring` 增加显式通道断言，使配置漂移产生可定位错误；
- 新增模型级 synthetic CPU 测试，覆盖完整 forward、参数量和
  `gamma=0` identity equivalence。

## 已验证

- stage 3 ring 参数量：`446,112`；
- stage 4 ring 参数量：`787,328`；
- ring 参数总量：`1,233,440`；
- 五层 synthetic CPU forward 输出形状和有限性检查通过；
- 同权重、同基础 H-NN、`gamma=0` 时，V18 与官方 KPNeXt 最大绝对误差
  为 `0.0`。

## 仍未验证

- 真实 S3DIS batch 和 CUDA 邻居扩展下的 identity equivalence；
- ring 分支梯度、峰值显存、吞吐与长训练稳定性；
- 无 ring query 参与 BatchNorm 对 running statistics 的影响；
- in-radius extra support 与 outer expansion support 的独立消融；
- profile 覆盖命令行参数的配置可追溯性。

因此 P0 修复后可以进入真实 batch smoke 和 graph audit，但不能据此宣称
V18 已达到可直接长训或已提升精度的状态。此前两个审查 ZIP 是修复前快照，
不得作为当前可训练源码使用。
