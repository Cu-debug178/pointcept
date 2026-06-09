_base_ = ["./semseg-kpconvx-hybrid-v16-da-stat-gc-area5.py"]

# v16-local:
# - do not rewrite the neighbor graph;
# - inject detached point-wise geometry statistics into KPConvX blocks;
# - only enable stage3/4 first, matching the low-risk ablation order;
# - keep shell/group-aware bias, decoder GC, boundary, refine, and tail losses off.
model = dict(
    backbone=dict(
        enable_da_radius=False,
        use_da_radius=False,
        da_radius_backend="torch",
        da_radius_stages=(3, 4),
        da_radius_scale_range=(0.95, 1.25),
        da_radius_stage_ranges={
            3: (0.95, 1.20),
            4: (0.90, 1.30),
        },
        da_radius_strength=0.5,
        da_radius_apply_block_mask=False,
        da_radius_debug=False,
        enable_da_meta=True,
        da_meta_stages=(3, 4),
        da_meta_dim=4,
        da_meta_use_channel_bias=True,
        da_meta_use_shell_bias=False,
        da_meta_use_point_bias=False,
        enable_gc_mixer=False,
        gc_use_meta_stats=False,
        enable_v16_monitor=True,
        v16_monitor_stages=(3, 4),
    )
)
