# GPT-5.6 Pro V18 改造落地记录

日期：2026-07-28

原始评审：`GPT-5.6Pro_V13到V18双支持改造评审原文_20260728.md`

## 已采纳

- 保留官方最近 H 邻居作为 identity path；
- stage 3/4（Standalone 一基编号，对应评审的零基 stage 2/3）建立
  `candidate_k=40/48` 的候选图；
- ring 分支只允许使用第 H 个槽位之后且满足
  `r_base < d <= r_adaptive` 的新增邻居；
- 强诊断版本 `ring_k=8/12`、固定 `gamma=0.25`；
- ring 内的 kernel points 与 influence sigma 同步乘逐点 scale；
- 密度改为 `mean KNN distance / stage grid size` 的无量纲 spacing；
- 提供训练集固定 q10/q90 接口，但在实测前不填写虚构统计量；
- 新增训练前图审计，输出新增邻居数、change rate、`d_H/r`、full-base
  比例和 spacing 分位数；
- V13 保留不覆盖，V18 使用独立模型和启动脚本。

## 暂未执行

- 训练集 stage-wise 固定 q10/q90：需要在真实 S3DIS 数据上先跑 20 个以上
  样本的图审计；
- T1/T4/T5/T12/T13 对照：需要已有 checkpoint，不属于本轮结构代码修改；
- 三 seed 正式训练：只有强诊断达到图变化与精度门槛后才能启动；
- 法向量 TTA 一致性：Standalone S3DIS 使用 `[1, RGB, z]`，不输入 normal，
  该风险主要针对 Pointcept 9D 管线。

## 代码位置

- `external/ml-kpconvx-standalone/KPConvX/utils/dual_support.py`
- `external/ml-kpconvx-standalone/KPConvX/models/KPNextV18.py`
- `external/ml-kpconvx-standalone/KPConvX/experiments/S3DIS/audit_S3DIS_v18.py`
- `external/ml-kpconvx-standalone/train_S3DIS_v18_diagnostic.sh`
- `external/ml-kpconvx-standalone/train_S3DIS_v18_formal.sh`
- `external/ml-kpconvx-standalone/audit_S3DIS_v18.sh`

## 当前验证

- 原始评审附件与归档文件 SHA256 一致；
- V13/V18 纯图算法测试共 7 项通过；
- 新增 Python 文件通过 `py_compile`；
- 在仅屏蔽未编译 C++ 扩展导入的构造 smoke 中，`KPNeXtV18` 成功实例化；
- 初版曾按错误的 stage 输入通道记录 ring 参数量为 `647,264`；P0 修复后
  已按实际 post-encoder 通道重新验证：stage 3 `446,112`、stage 4
  `787,328`，合计 `1,233,440`；
- 尚未在真实 S3DIS batch 上完成 CUDA 前向、显存和吞吐验证。

## P0 集成修正

后续模型级审查确定初版存在两个真实 forward 阻断：候选邻居切片不连续，
以及 `grid_pool=True` 时 ring module 通道仍按 stage 输入宽度构造。修复及
验证记录见：

`KPConvX_Standalone_V18_P0修复落地记录_20260728.md`
