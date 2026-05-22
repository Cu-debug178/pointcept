_base_ = ["./semseg-kpconvx-hybrid-v13-da-radius-cuda-only-area5.py"]

# v14: add SGCA global context on top of v13 CUDA DA-Radius.
# Keep router/refine/DA-kernel disabled for a clean global-attention ablation.
model = dict(
    backbone=dict(
        enable_global=True,
        use_global_context=True,
        global_context_type="serialized_patch",
        global_stages=(4, 5),
        global_context_stages=(4, 5),
        global_patch_sizes=(192, 320),
        global_num_heads=(8, 16),
        global_context_ratio=1.0,
        global_mlp_ratio=2.0,
        global_dropout=0.0,
        global_context_drop_path=0.0,
        enable_router=False,
        enable_refine=False,
        enable_da=False,
        use_da_kernel=False,
        use_da_radius=True,
        da_radius_backend="cuda",
        da_radius_stages=(3, 4),
        da_radius_scale_range=(0.95, 1.25),
        da_radius_stage_ranges={
            3: (0.95, 1.20),
            4: (0.90, 1.30),
        },
        da_radius_strength=0.5,
        da_radius_apply_block_mask=False,
        da_radius_debug=True,
        da_radius_debug_interval=100,
    )
)

# Keep the same test-time memory guard as v13.
data = dict(
    test=dict(
        test_cfg=dict(
            crop=dict(type="TestSphereCrop", point_max=80000),
        )
    )
)
