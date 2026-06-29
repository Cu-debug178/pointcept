_base_ = ["./semseg-kpconvx-hybrid-v16-local-da-stat-area5.py"]

# 关闭训练结束后的慢速 precise test，只保留每个 epoch 的验证和 best checkpoint 选择。
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

# v16b 主实验：受控半步支持集自适应。
#
# 和 v13/v13b 的区别：
# - 不启用 CUDA dynamic radius rebuild；
# - 先用略大的 static superset graph 提供候选邻居；
# - 再在 KPConvX block 内用 per-point effective radius mask 选择实际支持集；
# - 保留 v16-local 已验证有效的 DA-meta channel-wise conditioner。
#
# 这版只解决局部分支的 support-set 问题，不混入 GC、boundary、refine 或 shell bias。
model = dict(
    backbone=dict(
        neighbor_limits=(12, 16, 24, 24, 20),
        enable_da_radius=False,
        use_da_radius=False,
        da_radius_backend="torch",
        enable_da_meta=True,
        da_meta_stages=(3, 4),
        da_meta_use_channel_bias=True,
        da_meta_use_shell_bias=False,
        da_meta_use_point_bias=False,
        enable_gc_mixer=False,
        enable_support_mask=True,
        support_mask_stages=(3, 4),
        support_mask_stage_ranges={
            3: (0.98, 1.15),
            4: (1.00, 1.20),
        },
        support_mask_min_keep={
            3: 10,
            4: 8,
        },
        support_mask_strength=1.0,
        support_mask_density_k=16,
        support_mask_norm="percentile",
        support_mask_percentile=(10, 90),
        support_mask_power=1.0,
        # baseline 原始 stage3/4 邻居上限都是 20。
        # 用它来统计新增 4 个候选槽位是否真的被 mask 使用。
        support_mask_base_limits={
            3: 20,
            4: 20,
        },
        # 按 iter warmup/ramp，而不是 epoch。
        # S3DIS 当前约 340 iter/epoch：6800/5100 约等于 20/15 epoch。
        support_mask_warmup_steps=6800,
        support_mask_ramp_steps=5100,
        enable_v16_monitor=True,
        v16_monitor_stages=(3, 4),
    )
)
