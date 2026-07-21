_base_ = [
    "./semseg-kpconvx-base-s3dis-scale04-136k-linear-sharekp-4090d-area5.py"
]

# Main controlled SOKA experiment. Keep the complete Pointcept 136k baseline
# contract unchanged: only replace eligible deep KPConvX operators.
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
