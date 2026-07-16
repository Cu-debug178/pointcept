# GPT-5.6Pro KPConvX 下一创新方向材料包

## 使用方法

1. 新开一个 GPT-5.6Pro 对话，不接续旧 v16b/v17 对话。
2. 将 `00_直接提问_GPT-5.6Pro.md` 正文直接粘贴到对话框。
3. 上传 `upload10-materials` 目录中的全部 10 个文件。
4. 要求模型先核对附件中的最新固定结果，再独立裁决方向。

## 材料原则

- 8 篇唯一论文，不上传重复的 Sonata arXiv 版本。
- 项目事实和创新对照压缩为 2 个 Markdown 文件。
- 不上传完整 `train.log`，避免日志噪声占据上下文。
- scale04 的最新固定结果已经替换旧材料中“尚未产生结果”的过时描述。
- 当前 scale04 checkpoint sweep 为部分完成状态：epoch 200 已有可信结果，120-180 正在补评。

## GPT-5.6Pro 回答与 SOKA 实现

- `01_GPT56Pro_raw_conversation.md`：完整原始对话归档。
- `02_SOKA_Lite_implementation_digest.md`：只保留源码审查后的最终 SOKA-Lite 定义、验收和止损标准。
- `03_SOKA_implementation_plan.md`：当前仓库实现规格与验证清单。

实现时以后半段最终修正版为准：plain scale04、固定 KNN、六维 per-kernel-cell descriptor、zero-last scalar bias、encoder stage2-5 only。不要把前文的早期 occupancy prior、自由 beta/gamma、SPP/KSPC 或 v17 合并建议带入首版。
