_base_ = ["./semseg-kpconvx-soka-lite-scale04-4090d-area5.py"]

# Low-cost operator and performance probe before enabling all attention stages.
model = dict(backbone=dict(soka_stages=(4,)))
