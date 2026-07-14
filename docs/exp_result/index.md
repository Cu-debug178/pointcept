# 实验结果索引

这里记录由 `tools/build_exp_result.py` 自动整理过的实验结果。原始日志、模型权重和 TensorBoard 文件仍保留在 `exp/`。

| 实验 | 状态 | epoch | 最新 mIoU | 最高 mIoU | 弱类 | 报告 |
|---|---|---:|---:|---:|---|---|
| v15-stage234-boundary-gated_20260606_2315 | snapshot_incomplete | 107/200 | 0.5545 | 0.5971 | beam,column,door,clutter,window | [report.md](runs/v15-stage234-boundary-gated_20260606_2315/report.md) |
| v13b-da-radius-cuda-balanced_20260529_2139 | snapshot_incomplete | 186/200 | 0.6786 | 0.6841 | beam,door,column,clutter,window | [report.md](runs/v13b-da-radius-cuda-balanced_20260529_2139/report.md) |

## 阶段总结

- [KPConvX 全部实验结果汇总](KPConvX全部实验结果汇总.md)
- [2026 年 6 月 20 日汇报后 KPConvX 项目进展总结](2026-06-20汇报后KPConvX项目进展总结.md)
- [KPConvX 官方基线核验与 S3DIS 尺度校准阶段总结](kpconvx_official_baseline_scale_calibration_summary.md)
- [S3DIS 固定协议复评结果（2026-07-14）](s3dis_fixed_protocol_results_20260714.md)
