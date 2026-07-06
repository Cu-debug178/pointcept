_base_ = ["./semseg-kpconvx-hybrid-v16-local-da-stat-area5.py"]

# v17 main experiment:
# - keep original KPConvX support as the identity path;
# - use the stage4 expanded candidate graph only as a gated residual branch;
# - do not enable GC, boundary loss, refine, or v16b hard support replacement.
model = dict(
    backbone=dict(
        type="KPConvXV17",
        neighbor_limits=(12, 16, 20, 24, 20),
        enable_da_radius=False,
        use_da_radius=False,
        da_radius_backend="torch",
        enable_da_meta=True,
        da_meta_stages=(3, 4),
        da_meta_dim=4,
        da_meta_use_channel_bias=True,
        da_meta_use_shell_bias=False,
        da_meta_use_point_bias=False,
        enable_gc_mixer=False,
        gc_use_meta_stats=False,
        enable_support_mask=False,
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
        enable_v17_monitor=True,
        v17_monitor_stages=(3, 4),
    )
)
