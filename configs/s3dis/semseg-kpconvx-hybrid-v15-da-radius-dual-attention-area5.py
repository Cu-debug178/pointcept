_base_ = ["./semseg-kpconvx-hybrid-v14-da-radius-global-area5.py"]

# v15：局部-全局双分支。
# Local branch：继续使用 CUDA DA-Radius/KPConvX 作为高精度局部几何路径。
# Global branch：在低分辨率 stage 构建 PTv3-inspired 序列化 patch token bank。
# Fusion：decoder 点特征通过 cross-attention 读取 token bank，再结合 DA-Radius
# 局部统计做门控残差融合，避免全局上下文直接污染 encoder skip feature。
clip_grad = 2.0

model = dict(
    backbone=dict(
        enable_global=True,
        use_global_context=True,
        global_context_type="ptv3_serialized_patch",
        global_context_fusion="decoder",
        global_stages=(3, 4),
        global_context_stages=(3, 4),
        global_patch_sizes=(256, 192),
        global_num_heads=(8, 8),
        global_context_ratio=1.0,
        global_mlp_ratio=2.0,
        global_dropout=0.0,
        global_context_drop_path=0.0,
        global_serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        global_serialization_depth=10,
        global_decoder_fusion_type="cross_attention",
        global_context_max_tokens_per_stage=(64, 48),
        global_cross_attention_heads=4,
        global_cross_attention_chunk_size=8192,
        global_use_local_stats=True,
        global_local_stats_dim=8,
        enable_router=False,
        enable_refine=False,
        enable_da=False,
        use_da_kernel=False,
        use_da_radius=True,
        da_radius_backend="cuda",
        da_radius_stages=(3, 4),
        da_radius_scale_range=(0.95, 1.25),
        da_radius_stage_ranges={
            3: (0.95, 1.20),
            4: (0.90, 1.30),
        },
        da_radius_strength=0.5,
        da_radius_apply_block_mask=False,
        da_radius_debug=True,
        da_radius_debug_interval=100,
    )
)

data = dict(
    test=dict(
        test_cfg=dict(
            crop=dict(_delete_=True, type="TestSphereCrop", point_max=60000),
        )
    )
)
