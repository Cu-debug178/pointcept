_base_ = ["./semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"]

# v17 under exactly the same scale04 data and optimization protocol as the
# calibrated baseline. The original support remains the identity path and the
# expanded stage4 support is only a gated residual.

# Keep denser periodic weights for deterministic full-scene diagnostics. The
# primary comparison with the baseline still uses the shared 20-epoch grid plus
# model_best/model_last, so the extra checkpoints do not change model selection.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=10),
    dict(type="PreciseEvaluator", test_last=False),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type="KPConvXV17",
        input_channels=9,
        num_classes=13,
        dim=3,
        task="cloud_segmentation",
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=2.1,
        kp_aggregation="nearest",
        kp_influence="constant",
        kp_sigma=2.1,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act="sigmoid",
        inv_grp_norm=True,
        kpx_upcut=False,
        subsample_size=0.04,
        neighbor_limits=(12, 16, 20, 24, 20),
        layer_blocks=(3, 3, 9, 12, 3),
        init_channels=64,
        channel_scaling=1.414,
        radius_scaling=2.2,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0.3,
        norm="batch",
        bn_momentum=0.1,
        smooth_labels=False,
        class_w=(),
        # Disable legacy hybrid paths.
        enable_da=False,
        use_da_kernel=False,
        enable_da_radius=False,
        use_da_radius=False,
        da_radius_backend="torch",
        enable_global=False,
        use_global_context=False,
        enable_router=False,
        global_boundary_gate=False,
        boundary_loss_weight=0.0,
        enable_refine=False,
        # Keep the v16 local statistic conditioner used by v17.
        enable_da_meta=True,
        da_meta_stages=(3, 4),
        da_meta_dim=4,
        da_meta_use_channel_bias=True,
        da_meta_use_shell_bias=False,
        da_meta_use_point_bias=False,
        da_meta_hidden_ratio=0.25,
        da_radius_scale_range=(0.95, 1.25),
        da_radius_stage_ranges={
            3: (0.95, 1.20),
            4: (0.90, 1.30),
        },
        da_radius_density_k=16,
        da_radius_norm="percentile",
        da_radius_percentile=(10, 90),
        da_radius_strength=0.5,
        da_radius_power=1.0,
        enable_gc_mixer=False,
        gc_use_meta_stats=False,
        enable_support_mask=False,
        # Identity-preserving dual-support residual, stage4 only.
        enable_dual_support=True,
        dual_support_stages=(4,),
        dual_support_stage_ranges={
            4: (1.00, 1.20),
        },
        dual_support_min_keep={
            4: 8,
        },
        dual_support_base_limits={
            4: 20,
        },
        dual_support_warmup_steps=3400,
        dual_support_ramp_steps=5100,
        dual_support_gamma_init=1.0e-3,
        dual_support_gate_bias_init=-2.0,
        dual_support_eval_alpha=None,
        enable_v16_monitor=True,
        enable_v17_monitor=True,
        v16_monitor_stages=(3, 4),
        v17_monitor_stages=(3, 4),
    )
)
