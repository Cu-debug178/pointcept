_base_ = ["./semseg-kpconvx-hybrid-v13-da-radius-cuda-only-area5.py"]

# Relax gradient clipping for v13b: keep protection against spikes while
# avoiding the frequent update suppression observed with clip_grad=1.0.
clip_grad = 2.0

# v13b: conservative CUDA DA-Radius tuning.
# Goal: keep v13 structural gains while reducing boundary-category regression.
model = dict(
    backbone=dict(
        neighbor_limits=(12, 16, 24, 24, 20),
        da_radius_stages=(3, 4),
        da_radius_stage_ranges={
            3: (0.97, 1.15),
            4: (0.93, 1.20),
        },
        da_radius_strength=0.35,
        da_radius_backend="cuda",
        da_radius_apply_block_mask=False,
        da_radius_debug=True,
        da_radius_debug_interval=100,
    )
)

# Keep precise evaluation safer on large S3DIS Area_5 scenes.
data = dict(
    test=dict(
        test_cfg=dict(
            crop=dict(_delete_=True, type="TestSphereCrop", point_max=60000),
        )
    )
)
