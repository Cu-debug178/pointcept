_base_ = ["./semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"]

# Kernel-cell geometric/topological fusion. The support, physical scale,
# training objective, and evaluation protocol remain identical to scale04.
model = dict(
    backbone=dict(
        type="kpconvx_soka",
        soka_enabled=True,
        soka_stages=(4, 5),
        soka_evidence_dim=16,
        soka_rank=8,
        soka_bias_bound=2.0,
        soka_use_geometry=True,
        soka_use_topology=True,
        soka_use_query=True,
        soka_monitor=True,
    )
)
