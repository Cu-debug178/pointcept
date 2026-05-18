_base_ = ["./semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"]

model = dict(
    backbone=dict(
        use_da_kernel=False,
        use_da_radius=False,
        use_global_context=True,
        global_context_type="serialized_patch",
        global_stages=(4, 5),
        global_context_stages=(4, 5),
        global_patch_size=(256, 320),
        global_context_ratio=0.25,
        global_context_drop_path=0.1,
        enable_refine=False,
    )
)
