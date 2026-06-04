_base_ = ["./semseg-kpconvx-hybrid-v15-da-radius-dual-attention-4090d-area5.py"]

# 更高分辨率的全局上下文风险实验。
# 让序列化分支观察 stage 2/3/4；如果 stage 3/4 过粗，它可能帮助
# board/beam 等细结构，但边界污染和显存风险也更高。建议只作为
# stage3/4 4090D 配置之后的消融实验。
model = dict(
    backbone=dict(
        global_stages=(2, 3, 4),
        global_context_stages=(2, 3, 4),
        global_patch_sizes=(256, 192, 128),
        global_num_heads=(8, 8, 8),
        global_context_max_tokens_per_stage=(96, 128, 96),
        global_cross_attention_chunk_size=3072,
    )
)
