_base_ = ["./semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"]

model = dict(
    backbone=dict(
        use_da_kernel=True,
        use_da_radius=False,
        use_global_context=False,
        enable_router=False,
        enable_refine=False,
    )
)
