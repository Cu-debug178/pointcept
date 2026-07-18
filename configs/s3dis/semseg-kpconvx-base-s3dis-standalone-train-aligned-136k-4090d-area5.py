_base_ = [
    "./semseg-kpconvx-base-s3dis-scale04-136k-linear-sharekp-4090d-area5.py"
]

# Pointcept-compatible alignment of the remaining Standalone model and
# optimization factors. Input features, crop sampling, augmentation, and the
# fixed Pointcept evaluator remain unchanged and must be reported as such.
mix_prob = 0

model = dict(
    backbone=dict(
        channel_scaling=1.41,
        inv_groups=4,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
        )
    ],
)

optimizer = dict(
    type="AdamW",
    lr=0.005,
    weight_decay=0.05,
)

# Standalone starts at 1e-4, rises exponentially to 5e-3 over 30/450 of
# training, holds for 5/450, then loses one decade every 120/450. The custom
# scheduler maps those ratios onto Pointcept's approximately 136k steps.
scheduler = dict(
    _delete_=True,
    type="StandaloneS3DISLR",
    start_lr=1.0e-4,
    peak_lr=5.0e-3,
    reference_epochs=450,
    warmup_epochs=30,
    plateau_epochs=5,
    decay10_epochs=120,
)
