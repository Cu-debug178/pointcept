_base_ = ["./semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"]

# Formal SOKA-Lite configuration. The data scale, support, loss, decoder, and
# training protocol are inherited unchanged from the plain scale04 baseline.
model = dict(
    backbone=dict(
        type="kpconvx_soka",
        soka_enabled=True,
        soka_stages=(2, 3, 4, 5),
        soka_hidden_dim=16,
        soka_bias_bound=2.0,
        soka_monitor=True,
    )
)
