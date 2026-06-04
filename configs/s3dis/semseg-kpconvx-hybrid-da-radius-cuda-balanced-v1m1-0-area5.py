_base_ = ["./semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"]

model = dict(
    backbone=dict(
        enable_da_radius=True,
        da_radius_backend="cuda",
        da_radius_stages=(2, 3, 4),
        da_radius_scale_range=(0.85, 1.45),
        da_radius_density_k=16,
        da_radius_norm="percentile",
        da_radius_percentile=(10, 90),
        da_radius_strength=0.75,
        da_radius_power=1.0,
    )
)
