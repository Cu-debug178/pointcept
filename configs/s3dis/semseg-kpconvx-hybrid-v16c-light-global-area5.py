_base_ = ["./semseg-kpconvx-hybrid-v16b-controlled-support-area5.py"]

# v16c-light-global:
# - keep v16b local support adaptation unchanged;
# - only add a decoder/head-side lightweight global context residual;
# - no token bank, no boundary branch, no refine branch.
model = dict(
    backbone=dict(
        enable_gc_mixer=True,
        gc_stages=(5,),
        gc_use_meta_stats=False,
        gc_hidden_ratio=0.5,
        gc_dropout=0.0,
        gc_gamma_init=0.0,
        enable_v16_monitor=True,
        v16_monitor_stages=(3, 4),
    )
)
