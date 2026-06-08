_base_ = ["./semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"]

# v16 low-risk main config:
# - keep the KPConvX local backbone;
# - use DA-Radius only as detached point-wise local geometry statistics;
# - add a lightweight decoder-side global context mixer;
# - keep v15 token-bank, boundary gate, and refine paths disabled.
model = dict(
    backbone=dict(
        _delete_=True,
        type="KPConvXV16",
        input_channels=9,
        num_classes=13,
        dim=3,
        task="cloud_segmentation",
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=2.3,
        kp_aggregation="nearest",
        kp_influence="constant",
        kp_sigma=2.3,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act="sigmoid",
        inv_grp_norm=True,
        kpx_upcut=False,
        subsample_size=0.02,
        neighbor_limits=(12, 16, 20, 20, 20),
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
        # Keep old experimental paths off for v16 isolation.
        enable_da=False,
        use_da_kernel=False,
        enable_da_radius=False,
        use_da_radius=False,
        enable_global=False,
        use_global_context=False,
        enable_router=False,
        global_boundary_gate=False,
        boundary_loss_weight=0.0,
        enable_refine=False,
        # v16 local geometry conditioner.
        enable_da_meta=True,
        da_meta_stages=(3, 4),
        da_meta_dim=4,
        da_meta_use_shell_bias=True,
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
        # v16 lightweight decoder context mixer.
        enable_gc_mixer=True,
        gc_stages=(5,),
        gc_use_meta_stats=False,
        gc_hidden_ratio=1.0,
        gc_dropout=0.0,
        gc_gamma_init=0.0,
    )
)

data = dict(
    test=dict(
        test_cfg=dict(
            crop=dict(_delete_=True, type="TestSphereCrop", point_max=80000),
        )
    )
)
