_base_ = ["./semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"]

model = dict(
    backbone=dict(
        use_da_kernel=False,
        use_da_radius=True,
        da_radius_backend="torch",
        use_global_context=False,
        enable_router=False,
        enable_refine=False,
        da_radius_stages=(2, 3, 4),
        da_radius_scale_range=(0.85, 1.45),
    )
)
