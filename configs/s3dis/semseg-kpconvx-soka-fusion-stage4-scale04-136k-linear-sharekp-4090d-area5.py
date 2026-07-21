_base_ = [
    "./semseg-kpconvx-soka-fusion-scale04-136k-linear-sharekp-4090d-area5.py"
]

# Lower-cost smoke/profiling run. It differs from the main SOKA configuration
# only by replacing stage 4, so it remains directly interpretable.
model = dict(backbone=dict(soka_stages=(4,)))
