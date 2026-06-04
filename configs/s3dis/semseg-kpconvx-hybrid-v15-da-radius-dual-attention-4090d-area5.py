_base_ = ["./semseg-kpconvx-hybrid-v15-da-radius-dual-attention-area5.py"]

# 4090D 探测配置。
# 目标是在不改变 DA-Radius 局部策略的前提下，测试更大的序列化 token bank
# 是否能让全局分支更接近 PTv3 的大上下文行为。它仍不是原生 PTv3，
# 而是一个可控的 larger-context 双分支消融。
enable_amp = True
amp_dtype = "bfloat16"
batch_size = 2
clip_grad = 2.0

model = dict(
    backbone=dict(
        global_stages=(3, 4),
        global_context_stages=(3, 4),
        global_patch_sizes=(192, 128),
        global_num_heads=(8, 8),
        global_decoder_fusion_type="cross_attention",
        global_context_max_tokens_per_stage=(160, 128),
        global_cross_attention_heads=4,
        global_cross_attention_chunk_size=4096,
        global_use_local_stats=True,
        global_local_stats_dim=8,
    )
)

data = dict(
    test=dict(
        test_cfg=dict(
            crop=dict(_delete_=True, type="TestSphereCrop", point_max=60000),
        )
    )
)
