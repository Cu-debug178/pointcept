_base_ = ["./semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"]

# Full-scope kernel-cell geometric/topological fusion. Stage 1 has no KPConvX
# attention block under first_inv_layer=1, so stages 2-5 cover every eligible
# encoder attention stage while retaining the old scale04 protocol.
model = dict(
    backbone=dict(
        type="kpconvx_soka",
        soka_enabled=True,
        soka_stages=(2, 3, 4, 5),
        soka_evidence_dim=16,
        soka_rank=8,
        soka_bias_bound=2.0,
        soka_use_geometry=True,
        soka_use_topology=True,
        soka_use_query=True,
        soka_monitor=True,
    )
)
