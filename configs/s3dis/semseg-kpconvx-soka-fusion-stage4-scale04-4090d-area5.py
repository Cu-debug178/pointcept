_base_ = ["./semseg-kpconvx-soka-fusion-scale04-4090d-area5.py"]

# Engineering smoke configuration. It changes only the selected fusion stage.
model = dict(backbone=dict(soka_stages=(4,)))
