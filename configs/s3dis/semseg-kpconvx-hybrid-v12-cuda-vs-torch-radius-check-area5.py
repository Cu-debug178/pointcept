_base_ = ["./semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"]

model = dict(
    backbone=dict(
        enable_da=False,
        use_da_kernel=False,
        use_da_radius=True,
        da_radius_backend="torch",
        enable_global=False,
        use_global_context=False,
        enable_router=False,
        enable_refine=False,
        da_radius_stages=(2, 3, 4),
        da_radius_scale_range=(0.85, 1.45),
        da_radius_apply_block_mask=True,
    )
)
