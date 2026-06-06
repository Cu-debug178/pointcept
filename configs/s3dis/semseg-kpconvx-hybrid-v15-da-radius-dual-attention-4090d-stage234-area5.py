_base_ = ["./semseg-kpconvx-hybrid-v15-da-radius-dual-attention-4090d-area5.py"]

# v15-stage234：边界风险门控版双分支主实验配置。
# 目的：
# 1. 让序列化全局分支观察 stage 2/3/4，补回 stage3/4 过粗时可能丢失的
#    board、beam 等细结构信息；
# 2. 用 point-level boundary risk 软约束全局残差，降低 wall/door/board 等
#    类别边界处的全局上下文污染；
# 3. 保留 v15 base 和普通 4090D 配置作为对照，只覆盖 stage234 消融配置。
model = dict(
    backbone=dict(
        global_stages=(2, 3, 4),
        global_context_stages=(2, 3, 4),
        global_patch_sizes=(256, 192, 128),
        global_num_heads=(8, 8, 8),
        global_context_max_tokens_per_stage=(96, 128, 96),
        global_cross_attention_chunk_size=3072,
        global_boundary_gate=True,
        global_boundary_min_keep=0.10,
        global_boundary_detach=True,
        boundary_loss_weight=0.2,
        boundary_hidden_ratio=0.5,
        boundary_dropout=0.0,
        boundary_ignore_index=-1,
        boundary_dilate_steps=1,
        enable_refine=True,
        refine_hidden_ratio=0.5,
        refine_dropout=0.0,
        refine_use_coords=True,
        refine_use_boundary=True,
    )
)
