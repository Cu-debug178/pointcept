_base_ = ["./semseg-kpconvx-hybrid-v9-da-kernel-radius-global-context-area5.py"]

model = dict(backbone=dict(kp_influence="linear"))
