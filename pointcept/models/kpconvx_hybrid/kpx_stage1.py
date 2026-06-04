import math
import torch
import torch.nn as nn

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.kpconvx_base import KPConvXBase
from pointcept.models.kpconvx.utils.torch_pyramid import build_full_pyramid

from .sgca import SGCALite


@MODELS.register_module()
class KPConvXStage1(KPConvXBase):
    """
    Stage-1 improved KPConvX backbone.

    Main idea:
        - keep KPConvX as the main local geometry backbone
        - insert SGCA-lite only at low-resolution encoder stages
        - keep DefaultSegmentor compatibility (backbone returns logits)

    Stage mapping for default 5-layer KPConvX:
        stage 1 -> high resolution
        stage 2 -> high resolution
        stage 3 -> low resolution
        stage 4 -> lower resolution
        stage 5 -> bottleneck
    """

    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_global=True,
        global_context_type="serialized_patch",
        global_stages=(3, 4, 5),
        global_context_stages=None,
        global_patch_sizes=(128, 256, 384),
        global_patch_size=None,
        global_num_heads=(8, 8, 16),
        global_context_ratio=1.0,
        global_mlp_ratio=2.0,
        global_dropout=0.0,
        global_context_drop_path=0.0,
        global_context_fusion="encoder",
        global_serialization_orders=("linear",),
        global_serialization_depth=10,
        global_use_local_stats=False,
        global_local_stats_dim=0,
        global_decoder_fusion_type="mean",
        global_context_max_tokens_per_stage=0,
        global_cross_attention_heads=4,
        global_cross_attention_chunk_size=8192,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs
    ):
        if "use_global_context" in kwargs:
            enable_global = kwargs.pop("use_global_context")

        # Do not let hybrid-only ablation switches leak into KPConvXBase.
        for key in (
            "enable_da",
            "use_da_kernel",
            "da_stages",
            "da_scale_range",
            "da_density_k",
            "enable_da_radius",
            "use_da_radius",
            "da_radius_stages",
            "da_radius_scale_range",
            "da_radius_stage_ranges",
            "da_radius_density_k",
            "da_radius_norm",
            "da_radius_percentile",
            "da_radius_strength",
            "da_radius_power",
            "da_radius_backend",
            "da_radius_apply_block_mask",
            "da_radius_debug",
            "da_radius_debug_interval",
        ):
            kwargs.pop(key, None)

        if input_channels is None:
            input_channels = in_channels
        if input_channels is None:
            raise ValueError("Either `in_channels` or `input_channels` must be provided.")

        self._stage1_init_channels = init_channels
        self._stage1_channel_scaling = channel_scaling

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs
        )

        if global_context_type not in (
            "sgca",
            "serialized_patch",
            "serialized_patch_context",
            "ptv3_serialized_patch",
        ):
            raise ValueError(f"Unsupported global_context_type: {global_context_type}")

        if global_context_stages is not None:
            global_stages = global_context_stages

        global_context_fusion = str(global_context_fusion).lower()
        if global_context_fusion not in ("encoder", "decoder"):
            raise ValueError(
                "global_context_fusion must be either 'encoder' or 'decoder', "
                f"got {global_context_fusion}"
            )

        if (
            global_context_type == "ptv3_serialized_patch"
            and tuple(global_serialization_orders) == ("linear",)
        ):
            # ptv3_serialized_patch 使用 Pointcept serialization 的多顺序编码。
            # 这只是轻量全局分支，不把 KPConvX 主干替换成完整 PTv3。
            global_serialization_orders = (
                "z",
                "z-trans",
                "hilbert",
                "hilbert-trans",
            )

        if global_patch_size is not None:
            if isinstance(global_patch_size, int):
                global_patch_sizes = tuple(global_patch_size for _ in global_stages)
            else:
                global_patch_sizes = tuple(global_patch_size)

        self.global_context_type = global_context_type
        self.global_context_ratio = global_context_ratio
        self.global_context_drop_path = global_context_drop_path
        self.global_context_fusion = global_context_fusion
        self.global_use_local_stats = bool(global_use_local_stats)
        self.global_local_stats_dim = int(global_local_stats_dim)
        self.global_decoder_fusion_type = str(global_decoder_fusion_type).lower()
        if self.global_decoder_fusion_type not in (
            "mean",
            "token_bank",
            "cross_attention",
        ):
            raise ValueError(
                "global_decoder_fusion_type must be one of "
                "'mean', 'token_bank', or 'cross_attention', "
                f"got {global_decoder_fusion_type}"
            )
        self.global_context_max_tokens_per_stage = global_context_max_tokens_per_stage
        self.global_cross_attention_chunk_size = int(global_cross_attention_chunk_size)
        self.enable_global = enable_global
        self.global_stages = tuple(global_stages) if enable_global else tuple()
        self.global_stage_dims = {}

        layer_channels = self._compute_layer_channels(
            init_channels=self._stage1_init_channels,
            channel_scaling=self._stage1_channel_scaling,
            num_layers=self.num_layers,
        )

        for i, stage in enumerate(self.global_stages):
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid global stage index: {stage}")

            # When grid_pool is enabled, use the next layer's channels
            if self.grid_pool and stage < self.num_layers:
                dim = layer_channels[stage]
            else:
                dim = layer_channels[stage - 1]
            self.global_stage_dims[stage] = dim
            patch_size = global_patch_sizes[min(i, len(global_patch_sizes) - 1)]
            num_heads = self._safe_num_heads(
                dim=dim,
                requested_heads=global_num_heads[min(i, len(global_num_heads) - 1)],
            )

            setattr(
                self,
                f"sgca_{stage}",
                SGCALite(
                    dim=dim,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    context_ratio=global_context_ratio,
                    mlp_ratio=global_mlp_ratio,
                    dropout=global_dropout,
                    drop_path=global_context_drop_path,
                    serialization_orders=global_serialization_orders,
                    serialization_depth=global_serialization_depth,
                ),
            )

        if (
            self.enable_global
            and self.global_context_fusion == "decoder"
            and self.task == "cloud_segmentation"
        ):
            decoder_dim = layer_channels[0]
            self.global_decoder_context_proj = nn.ModuleDict(
                {
                    str(stage): nn.Linear(dim, decoder_dim)
                    for stage, dim in self.global_stage_dims.items()
                }
            )
            self.global_decoder_context_norm = nn.LayerNorm(decoder_dim)
            if self.global_decoder_fusion_type in ("token_bank", "cross_attention"):
                cross_heads = self._safe_num_heads(
                    dim=decoder_dim,
                    requested_heads=global_cross_attention_heads,
                )
                self.global_decoder_query_norm = nn.LayerNorm(decoder_dim)
                self.global_decoder_bank_norm = nn.LayerNorm(decoder_dim)
                self.global_decoder_cross_attn = nn.MultiheadAttention(
                    embed_dim=decoder_dim,
                    num_heads=cross_heads,
                    dropout=global_dropout,
                    batch_first=True,
                )
            gate_in_dim = decoder_dim * 2
            if self.global_use_local_stats and self.global_local_stats_dim > 0:
                self.global_decoder_local_stat_norm = nn.LayerNorm(
                    self.global_local_stats_dim
                )
                self.global_decoder_local_stat_proj = nn.Sequential(
                    nn.Linear(self.global_local_stats_dim, decoder_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(decoder_dim, decoder_dim),
                )
                gate_in_dim += decoder_dim
            self.global_decoder_gate = nn.Sequential(
                nn.Linear(gate_in_dim, decoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(decoder_dim, decoder_dim),
                nn.Sigmoid(),
            )
            self.global_decoder_fuse = nn.Sequential(
                nn.Linear(gate_in_dim, decoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(decoder_dim, decoder_dim),
            )
            # gamma 从 0 开始，让模型初始等价于原局部主干，避免全局分支
            # 在训练初期直接破坏 encoder/decoder 已有的局部边界表征。
            self.global_decoder_gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _compute_layer_channels(init_channels, channel_scaling, num_layers):
        layer_channels = []
        for l in range(num_layers):
            target_c = init_channels * (channel_scaling ** l)
            layer_channels.append(int(math.ceil((target_c - 0.1) / 16)) * 16)
        return layer_channels

    @staticmethod
    def _safe_num_heads(dim, requested_heads):
        heads = max(1, min(dim, int(requested_heads)))
        while dim % heads != 0 and heads > 1:
            heads -= 1
        return heads

    def _apply_sgca_if_needed(self, stage_idx, feats, points, lengths):
        if not self.enable_global:
            return feats
        if self.global_context_fusion != "encoder":
            return feats
        if stage_idx not in self.global_stages:
            return feats
        sgca = getattr(self, f"sgca_{stage_idx}")
        return sgca(feats, points, lengths)

    def _collect_sgca_context_if_needed(self, stage_idx, feats, points, lengths):
        if not self.enable_global:
            return None
        if self.global_context_fusion != "decoder":
            return None
        if stage_idx not in self.global_stages:
            return None
        sgca = getattr(self, f"sgca_{stage_idx}")
        if self.global_decoder_fusion_type in ("token_bank", "cross_attention"):
            tokens, token_lengths = sgca.context_bank(
                feats,
                points,
                lengths,
                max_tokens_per_cloud=self._get_context_max_tokens(stage_idx),
            )
            return dict(
                stage_idx=stage_idx,
                kind="token_bank",
                tokens=tokens,
                lengths=token_lengths,
            )
        return dict(
            stage_idx=stage_idx,
            kind="mean",
            tokens=sgca.context(feats, points, lengths),
        )

    def _get_context_max_tokens(self, stage_idx):
        value = self.global_context_max_tokens_per_stage
        if isinstance(value, dict):
            return int(value.get(stage_idx, value.get(str(stage_idx), 0)))
        if isinstance(value, (list, tuple)):
            stages = list(self.global_stages)
            if stage_idx in stages:
                index = stages.index(stage_idx)
            else:
                index = 0
            return int(value[min(index, len(value) - 1)])
        return int(value or 0)

    @staticmethod
    def _expand_context_to_points(context, lengths):
        parts = []
        for token, length in zip(context, lengths.tolist()):
            parts.append(token.unsqueeze(0).expand(int(length), -1))
        if not parts:
            return context.new_zeros((0, context.shape[-1]))
        return torch.cat(parts, dim=0)

    def _fuse_decoder_context(self, feats, lengths, point_context, local_stats=None):
        gate_parts = [feats, point_context]
        if (
            self.global_use_local_stats
            and self.global_local_stats_dim > 0
        ):
            if local_stats is None:
                local_stats = feats.new_zeros(
                    (int(lengths.numel()), self.global_local_stats_dim)
                )
            local_stats = self.global_decoder_local_stat_norm(local_stats)
            stat_context = self.global_decoder_local_stat_proj(local_stats)
            point_stat_context = self._expand_context_to_points(stat_context, lengths)
            gate_parts.append(point_stat_context)

        # gate 同时看 decoder 点特征、全局上下文和 DA-Radius 局部统计。
        # 目标是让全局残差可控，而不是无差别覆盖 board/beam/window 等边界类。
        gate_input = torch.cat(gate_parts, dim=-1)
        gate = self.global_decoder_gate(gate_input)
        fused = self.global_decoder_fuse(gate_input)
        return feats + self.global_decoder_gamma * gate * fused

    @staticmethod
    def _context_item_stage(item):
        if isinstance(item, dict):
            return item["stage_idx"]
        return item[0]

    @staticmethod
    def _context_item_tokens(item):
        if isinstance(item, dict):
            return item["tokens"]
        return item[1]

    def _apply_decoder_mean_context(
        self, feats, lengths, context_tokens, local_stats=None
    ):
        projected = []
        for item in context_tokens:
            stage_idx = self._context_item_stage(item)
            token = self._context_item_tokens(item)
            projected.append(self.global_decoder_context_proj[str(stage_idx)](token))

        context = torch.stack(projected, dim=0).mean(dim=0)
        context = self.global_decoder_context_norm(context)
        point_context = self._expand_context_to_points(context, lengths)
        return self._fuse_decoder_context(
            feats=feats,
            lengths=lengths,
            point_context=point_context,
            local_stats=local_stats,
        )

    @staticmethod
    def _split_by_lengths(x, lengths):
        parts = []
        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            parts.append(x[start:end])
            start = end
        return parts

    def _apply_decoder_token_bank_context(
        self, feats, lengths, context_tokens, local_stats=None
    ):
        stage_banks = []
        for item in context_tokens:
            if not isinstance(item, dict) or item.get("kind") != "token_bank":
                return self._apply_decoder_mean_context(
                    feats=feats,
                    lengths=lengths,
                    context_tokens=context_tokens,
                    local_stats=local_stats,
                )

            stage_idx = item["stage_idx"]
            tokens = self.global_decoder_context_proj[str(stage_idx)](item["tokens"])
            stage_banks.append(
                self._split_by_lengths(tokens, item["lengths"])
            )

        point_context_parts = []
        point_start = 0
        for cloud_idx, length in enumerate(lengths.tolist()):
            point_end = point_start + int(length)
            query = feats[point_start:point_end]
            banks = [bank[cloud_idx] for bank in stage_banks if bank[cloud_idx].numel()]
            if not banks:
                point_context_parts.append(torch.zeros_like(query))
                point_start = point_end
                continue

            bank = torch.cat(banks, dim=0)
            bank_norm = self.global_decoder_bank_norm(bank).unsqueeze(0)
            chunk_size = max(int(self.global_cross_attention_chunk_size), 1)
            context_chunks = []
            # 按点分块做 cross-attention，控制 4090D 单卡上的显存峰值。
            for chunk_start in range(0, query.shape[0], chunk_size):
                chunk_end = min(chunk_start + chunk_size, query.shape[0])
                query_norm = self.global_decoder_query_norm(
                    query[chunk_start:chunk_end]
                ).unsqueeze(0)
                context_chunks.append(
                    self.global_decoder_cross_attn(
                        query_norm,
                        bank_norm,
                        bank_norm,
                        need_weights=False,
                    )[0].squeeze(0)
                )
            context = torch.cat(context_chunks, dim=0)
            point_context_parts.append(context)
            point_start = point_end

        point_context = torch.cat(point_context_parts, dim=0)
        return self._fuse_decoder_context(
            feats=feats,
            lengths=lengths,
            point_context=point_context,
            local_stats=local_stats,
        )

    def _apply_decoder_global_context(
        self, feats, lengths, context_tokens, local_stats=None
    ):
        if not context_tokens:
            return feats
        if self.global_context_fusion != "decoder":
            return feats
        if self.task != "cloud_segmentation":
            return feats

        if self.global_decoder_fusion_type in ("token_bank", "cross_attention"):
            return self._apply_decoder_token_bank_context(
                feats=feats,
                lengths=lengths,
                context_tokens=context_tokens,
                local_stats=local_stats,
            )

        return self._apply_decoder_mean_context(
            feats=feats,
            lengths=lengths,
            context_tokens=context_tokens,
            local_stats=local_stats,
        )

    def forward(self, data_dict):
        # ------ Init ------
        points = data_dict["coord"]
        feats = data_dict["feat"]
        offset = data_dict["offset"].int()

        # Convert offsets to lengths
        offset = torch.cat(
            [torch.zeros(1, dtype=offset.dtype, device=offset.device), offset],
            dim=0,
        )
        lengths = offset[1:] - offset[:-1]

        in_dict = build_full_pyramid(
            points,
            lengths,
            self.num_layers,
            self.subsample_size,
            self.first_radius,
            self.radius_scaling,
            self.neighbor_limits,
            self.upsample_n,
            sub_mode=self.in_sub_mode,
            grid_pool_mode=self.grid_pool,
        )

        # ------ Stem ------
        feats = self.stem(
            in_dict.points[0],
            in_dict.points[0],
            feats,
            in_dict.neighbors[0],
        )

        # ------ Encoder ------
        skip_feats = []
        context_tokens = []
        for layer in range(1, self.num_layers + 1):
            l = layer - 1
            block_list = getattr(self, f"encoder_{layer}")

            if self.kp_mode in ["kpconv", "kpconvtest"]:
                for block in block_list:
                    feats = block(
                        in_dict.points[l],
                        in_dict.points[l],
                        feats,
                        in_dict.neighbors[l],
                    )
            else:
                upcut = None
                for block in block_list:
                    feats, upcut = block(
                        in_dict.points[l],
                        in_dict.points[l],
                        feats,
                        in_dict.neighbors[l],
                        in_dict.lengths[l],
                        upcut=upcut,
                    )

            context_token = self._collect_sgca_context_if_needed(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                lengths=in_dict.lengths[l],
            )
            if context_token is not None:
                context_tokens.append(context_token)

            # Stage-1 encoder-mode global context insertion
            feats = self._apply_sgca_if_needed(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                lengths=in_dict.lengths[l],
            )

            if layer < self.num_layers:
                skip_feats.append(feats)

                layer_pool = getattr(self, f"pooling_{layer}")
                if self.grid_pool:
                    if isinstance(in_dict.pools[l], tuple):
                        feats = layer_pool(
                            feats,
                            in_dict.pools[l][0],
                            idx_ptr=in_dict.pools[l][1],
                        )
                    else:
                        feats = layer_pool(feats, in_dict.pools[l])
                else:
                    feats = layer_pool(
                        in_dict.points[l + 1],
                        in_dict.points[l],
                        feats,
                        in_dict.pools[l],
                    )

        # ------ Decoder / Classification ------
        if self.task == "classification":
            feats = self.global_pooling(feats, in_dict.lengths[-1])

        elif self.task == "cloud_segmentation":
            for layer in range(self.num_layers - 1, 0, -1):
                l = layer - 1
                upsample = getattr(self, f"upsampling_{layer}")

                if self.grid_pool:
                    feats = upsample(feats, in_dict.upsamples[l])
                else:
                    feats = upsample(
                        feats,
                        in_dict.upsamples[l],
                        in_dict.up_distances[l],
                    )

                feats = torch.cat([feats, skip_feats[l]], dim=1)

                unary = getattr(self, f"decoder_unary_{layer}")
                feats = unary(feats)

                if self.add_decoder_layer:
                    block = getattr(self, f"decoder_layer_{layer}")
                    if self.kp_mode in ["kpconv", "kpconvtest"]:
                        feats = block(
                            in_dict.points[l],
                            in_dict.points[l],
                            feats,
                            in_dict.neighbors[l],
                        )
                    else:
                        feats, _ = block(
                            in_dict.points[l],
                            in_dict.points[l],
                            feats,
                            in_dict.neighbors[l],
                            in_dict.lengths[l],
                        )

            feats = self._apply_decoder_global_context(
                feats=feats,
                lengths=in_dict.lengths[0],
                context_tokens=context_tokens,
            )

        # ------ Head ------
        logits = self.head(feats)
        return logits
