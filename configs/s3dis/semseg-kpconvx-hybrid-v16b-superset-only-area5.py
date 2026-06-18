_base_ = ["./semseg-kpconvx-hybrid-v16-local-da-stat-area5.py"]

# v16b 对照实验：
# 只扩大 stage3/4 的候选邻居上限，不做 block 内 effective radius mask。
# 目的不是追求最强结果，而是隔离验证“superset graph 本身”是否会复现 v13b 的退化。
model = dict(
    backbone=dict(
        neighbor_limits=(12, 16, 24, 24, 20),
        enable_support_mask=False,
        enable_v16_monitor=True,
        v16_monitor_stages=(3, 4),
    )
)
