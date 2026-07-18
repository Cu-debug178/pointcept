_base_ = ["./semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"]

# Stronger Pointcept baseline with the main Standalone operator factors and
# training budget aligned. This intentionally keeps the existing 9D input,
# CE + Lovasz objective, Pointcept sampling, and fixed evaluation protocol so
# later SOKA runs can differ only in their backbone operator.
seed = 57106803

# Keep epoch // eval_epoch == 5, hence the same 340 mini-batches per logged
# epoch as scale04. Four hundred logged epochs give about 136k optimizer steps.
epoch = 2000
eval_epoch = 400
batch_size_val = 1

# Keep model_last/model_best resumable, retain lightweight snapshots more
# densely near convergence, and keep a full recovery point every 50 epochs.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(
        type="CheckpointSaver",
        weight_only_save_rules=[
            dict(start=1, end=100, freq=20),
            dict(start=101, end=140, freq=10),
            dict(start=141, end=None, freq=5),
        ],
        resume_save_freq=50,
    ),
    dict(type="PreciseEvaluator", test_last=False),
]

model = dict(
    backbone=dict(
        kp_influence="linear",
        share_kp=True,
    )
)

# Training-time model selection uses the same deterministic center sample from
# every Area5 room on every epoch. Final reporting still uses full-scene test.
data = dict(
    val=dict(
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
                deterministic=True,
            ),
            dict(type="SphereCrop", point_max=40000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    )
)
