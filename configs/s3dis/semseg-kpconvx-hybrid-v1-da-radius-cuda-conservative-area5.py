_base_ = ["./semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"]

model = dict(
    backbone=dict(
        use_da_radius=True,
        da_radius_backend="cuda",
        da_radius_stages=(2, 3, 4),
        da_radius_scale_range=(0.9, 1.25),
        da_radius_strength=0.6,
    )
)
