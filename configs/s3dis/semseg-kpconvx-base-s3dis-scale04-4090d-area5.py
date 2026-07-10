_base_ = ["./semseg-kpconvx-base.py"]

# S3DIS physical-scale calibration for one RTX 4090D.
# This is an internal Pointcept ablation, not a strict reproduction of the
# paper's Standalone S3DIS pipeline. Keep the training budget and model size
# unchanged, and isolate the 0.04 m grid / 0.084 m first-radius effect.

num_worker = 8
num_worker_test = 6
batch_size = 3
batch_size_val = None
batch_size_test = 1
gradient_accumulation_steps = 1
fragment_batch_size_test = 2
max_input_pts = 40000

sync_bn = False
enable_amp = False
empty_cache = False
clip_grad = None
save_freq = 100

# Save full resumable checkpoints at log epochs 20, 40, ..., 200. The
# historical random validation still selects model_best; periodic weights are
# retained for deterministic full-scene checkpoint selection after training.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=20),
    dict(type="PreciseEvaluator", test_last=False),
]

model = dict(
    backbone=dict(
        kp_radius=2.1,
        kp_sigma=2.1,
        subsample_size=0.04,
    )
)

data = dict(
    train=dict(
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout",
                dropout_ratio=0.2,
                dropout_application_ratio=0.2,
            ),
            dict(
                type="RandomRotateTargetAngle",
                angle=(1 / 2, 1, 3 / 2),
                center=[0, 0, 0],
                axis="z",
                p=0.75,
            ),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.0,
            ),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="ElasticDistortion",
                distortion_params=[[0.2, 0.4], [0.8, 1.6]],
            ),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=max_input_pts, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
        loop=5,
    ),
    val=dict(
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            # Keep the historical random validation protocol for comparable
            # learning curves. Final conclusions use deterministic test data.
            dict(type="SphereCrop", point_max=max_input_pts, mode="random"),
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
    ),
    test=dict(
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="test",
            ),
            crop=dict(
                _delete_=True,
                type="TestSphereCrop",
                point_max=60000,
            ),
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
    ),
)
