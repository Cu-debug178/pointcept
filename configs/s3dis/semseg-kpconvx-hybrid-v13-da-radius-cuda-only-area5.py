_base_ = ["./semseg-kpconvx-hybrid-v12-cuda-vs-torch-radius-check-area5.py"]

# Clean CUDA DA-Radius ablation for cloud training.
# Compared with v12, this changes the neighbor graph with CUDA adaptive radius
# instead of only masking the fixed KNN graph inside KPConvX blocks.
model = dict(
    backbone=dict(
        enable_da=False,
        enable_global=False,
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

# Full-scene precise evaluation can create 300k-400k point fragments after
# voxelization. Split test fragments to avoid KPConvX influence tensor OOM.
data = dict(
    test=dict(
        test_cfg=dict(
            crop=dict(type="TestSphereCrop", point_max=80000),
        )
    )
)
