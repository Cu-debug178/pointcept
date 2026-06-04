import math
import os
import torch

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.utils.torch_pyramid import build_full_pyramid
from pointcept.utils.logger import get_root_logger

from .kpx_stage1 import KPConvXStage1
from .da_kpconvx_block import DensityAdaptiveScale, DensityAdaptiveRadius
from .da_kpnext_blocks import DAKPNextMultiShortcutBlock


@MODELS.register_module()
class KPConvXStage2(KPConvXStage1):
    """
    Stage-2 improved KPConvX backbone.

    Stage-2 = Stage-1 + Density-Adaptive KPConvX.

    Main change
    -----------
    Instead of adding a feature-level Fine / Coarse adapter after KPConvX,
    this version computes a density-aware scale s_i for each query point and
    passes it into DA-KPConvX blocks.

    DA-KPConvX kernel scaling
    -------------------------
        rho_i = 1 / (mean_neighbor_distance_i + eps)

        dense region:
            rho_i high -> s_i small

        sparse region:
            rho_i low -> s_i large

        kernel point scaling:
            p_k_da = s_i * p_k

    Pipeline
    --------
    stem
    -> encoder stage with DA-KPConvX kernel scaling
    -> SGCA global context branch from Stage-1
    -> pooling
    -> decoder
    -> head

    Notes
    -----
    - Original baseline KPConvX files are not modified.
    - This class uses DAKPNextMultiShortcutBlock from kpconvx_hybrid.
    - No dynamic graph rebuild.
    - Existing pyramid neighbors from build_full_pyramid() are reused.
    """

    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_da=True,
        use_da_kernel=None,
        da_stages=(2, 3, 4),
        da_scale_range=(0.5, 2.0),
        da_density_k=16,
        enable_da_radius=False,
        use_da_radius=None,
        da_radius_stages=(2, 3, 4),
        da_radius_scale_range=(0.8, 1.5),
        da_radius_stage_ranges=None,
        da_radius_density_k=16,
        da_radius_norm="percentile",
        da_radius_percentile=(10, 90),
        da_radius_strength=0.75,
        da_radius_power=1.0,
        da_radius_backend="torch",
        da_radius_apply_block_mask=None,
        da_radius_debug=False,
        da_radius_debug_interval=100,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs,
    ):
        if input_channels is None:
            input_channels = in_channels

        if input_channels is None:
            raise ValueError("Either `in_channels` or `input_channels` must be provided.")

        self._stage2_init_channels = init_channels
        self._stage2_channel_scaling = channel_scaling

        # DA settings must be assigned before super().__init__().
        # KPConvXStage1 / KPConvXBase may call self.get_residual_block()
        # during initialization, so the DA block factory must already be ready.
        if use_da_kernel is not None:
            enable_da = use_da_kernel
        if use_da_radius is not None:
            enable_da_radius = use_da_radius

        self.enable_da = bool(enable_da)
        self.da_stages = tuple(da_stages) if enable_da else tuple()
        self.da_scale_range = da_scale_range
        self.da_density_k = da_density_k
        self.enable_da_radius = bool(enable_da_radius)
        self.da_radius_stages = tuple(da_radius_stages) if enable_da_radius else tuple()
        self.da_radius_scale_range = da_radius_scale_range
        self.da_radius_stage_ranges = self._normalize_stage_ranges(
            da_radius_stage_ranges
        )
        self.da_radius_density_k = da_radius_density_k
        self.da_radius_norm = da_radius_norm
        self.da_radius_percentile = da_radius_percentile
        self.da_radius_strength = da_radius_strength
        self.da_radius_power = da_radius_power
        self.da_radius_backend = da_radius_backend
        self.da_radius_apply_block_mask = da_radius_apply_block_mask
        self.da_radius_debug = bool(da_radius_debug) or self._env_flag(
            "POINTCEPT_DA_RADIUS_DEBUG"
        )
        self.da_radius_debug_interval = self._env_int(
            "POINTCEPT_DA_RADIUS_DEBUG_INTERVAL",
            da_radius_debug_interval,
        )
        self._da_radius_debug_step = 0
        self._last_da_radius_local_stats = None

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs,
        )

        for stage in self.da_stages:
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid DA stage index: {stage}")

        self.da_scale = DensityAdaptiveScale(
            scale_range=da_scale_range,
            density_k=da_density_k,
        )
        self.da_radius = DensityAdaptiveRadius(
            scale_range=da_radius_scale_range,
            density_k=da_radius_density_k,
            norm=da_radius_norm,
            percentile=da_radius_percentile,
            strength=da_radius_strength,
            power=da_radius_power,
        )

    @staticmethod
    def _env_flag(name):
        value = os.environ.get(name, "")
        return value.lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _env_int(name, default):
        value = os.environ.get(name)
        if value is None:
            return int(default)
        try:
            return int(value)
        except ValueError:
            return int(default)

    @staticmethod
    def _normalize_stage_ranges(stage_ranges):
        if stage_ranges is None:
            return {}

        if isinstance(stage_ranges, dict):
            return {
                int(stage): tuple(scale_range)
                for stage, scale_range in stage_ranges.items()
            }

        normalized = {}
        for item in stage_ranges:
            if isinstance(item, dict):
                stage = item.get("stage")
                scale_range = item.get("range", item.get("scale_range"))
            else:
                stage, scale_range = item
            normalized[int(stage)] = tuple(scale_range)
        return normalized

    def _get_da_radius_scale_range(self, stage_idx):
        return self.da_radius_stage_ranges.get(
            int(stage_idx), self.da_radius_scale_range
        )

    def _should_apply_da_radius_block_mask(self):
        if not self.enable_da_radius:
            return False

        if self.da_radius_apply_block_mask is not None:
            return bool(self.da_radius_apply_block_mask)

        # CUDA backend already changes the neighbor graph with adaptive radius.
        # Applying the block-level mask again over-restricts neighbors.
        return self.da_radius_backend != "cuda"

    @staticmethod
    def _is_main_process():
        if not torch.distributed.is_available():
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _should_log_da_radius_debug(self):
        if not self.da_radius_debug:
            return False
        if not self._is_main_process():
            return False
        interval = max(int(self.da_radius_debug_interval), 1)
        return self._da_radius_debug_step % interval == 0

    @torch.no_grad()
    def _log_da_radius_debug(self, in_dict, radius_scales):
        if not self._should_log_da_radius_debug():
            return

        logger = get_root_logger()
        for layer in self.da_radius_stages:
            l = int(layer) - 1
            if l < 0 or l >= len(in_dict.neighbors):
                continue
            if l >= len(radius_scales) or radius_scales[l] is None:
                continue

            neighbors = in_dict.neighbors[l]
            num_points = in_dict.points[l].shape[0]
            neighbor_limit = neighbors.shape[1]
            valid = (neighbors >= 0) & (neighbors < num_points)
            valid_count = valid.sum(dim=1).float()
            shadow_ratio = (~valid).float().mean()
            full_ratio = (valid_count >= neighbor_limit).float().mean()
            radius_scale = radius_scales[l].detach().reshape(-1).float()

            logger.info(
                "DA-Radius debug "
                f"step={self._da_radius_debug_step} "
                f"stage={layer} "
                f"points={num_points} "
                f"limit={neighbor_limit} "
                f"radius_scale_min={radius_scale.min().item():.4f} "
                f"radius_scale_mean={radius_scale.mean().item():.4f} "
                f"radius_scale_max={radius_scale.max().item():.4f} "
                f"valid_neighbors_min={valid_count.min().item():.2f} "
                f"valid_neighbors_mean={valid_count.mean().item():.2f} "
                f"valid_neighbors_max={valid_count.max().item():.2f} "
                f"shadow_ratio={shadow_ratio.item():.4f} "
                f"full_ratio={full_ratio.item():.4f}"
            )

    @torch.no_grad()
    def _collect_da_radius_local_stats(self, in_dict, radius_scales):
        if not getattr(self, "global_use_local_stats", False):
            return None
        if int(getattr(self, "global_local_stats_dim", 0)) <= 0:
            return None

        # 为 decoder gate 收集每个 cloud 的局部几何状态。
        # 每个选中 stage 提供 4 个统计量：
        # radius_scale_mean, valid_ratio_mean, shadow_ratio_mean, full_ratio_mean。
        # 它们只作为全局融合的条件信号，不参与梯度更新。
        stage_stats = []
        for layer in self.da_radius_stages:
            l = int(layer) - 1
            if l < 0 or l >= len(in_dict.neighbors):
                continue
            if l >= len(radius_scales) or radius_scales[l] is None:
                continue

            neighbors = in_dict.neighbors[l]
            num_points = in_dict.points[l].shape[0]
            neighbor_limit = max(int(neighbors.shape[1]), 1)
            valid = (neighbors >= 0) & (neighbors < num_points)
            valid_count = valid.sum(dim=1).float()
            valid_ratio = valid_count / float(neighbor_limit)
            shadow_ratio = 1.0 - valid_ratio
            full_ratio = (valid_count >= neighbor_limit).float()
            radius_scale = radius_scales[l].detach().reshape(-1).float()

            per_cloud = []
            start = 0
            for length in in_dict.lengths[l].tolist():
                end = start + int(length)
                if end <= start:
                    per_cloud.append(radius_scale.new_zeros((4,)))
                else:
                    per_cloud.append(
                        torch.stack(
                            [
                                radius_scale[start:end].mean(),
                                valid_ratio[start:end].mean(),
                                shadow_ratio[start:end].mean(),
                                full_ratio[start:end].mean(),
                            ],
                            dim=0,
                        )
                    )
                start = end
            if per_cloud:
                stage_stats.append(torch.stack(per_cloud, dim=0))

        if not stage_stats:
            return None

        stats = torch.cat(stage_stats, dim=1)
        target_dim = int(self.global_local_stats_dim)
        if stats.shape[1] < target_dim:
            pad = stats.new_zeros((stats.shape[0], target_dim - stats.shape[1]))
            stats = torch.cat([stats, pad], dim=1)
        elif stats.shape[1] > target_dim:
            stats = stats[:, :target_dim]
        return stats

    def get_residual_block(
        self,
        in_C,
        out_C,
        radius,
        sigma,
        shared_kp_data=None,
        conv_layer=False,
        drop_path=-1,
    ):
        """
        Build encoder residual blocks.

        This overrides the block factory from KPConvXStage1 / KPConvXBase so
        that the improved model uses DAKPNextMultiShortcutBlock, while the
        original KPConvX baseline remains untouched.

        For kpconv / kpconvtest mode, fall back to the original implementation.
        For KPConvX mode, use the DA-compatible block.
        """

        if self.kp_mode in ["kpconv", "kpconvtest"]:
            return super().get_residual_block(
                in_C,
                out_C,
                radius,
                sigma,
                shared_kp_data=shared_kp_data,
                conv_layer=conv_layer,
                drop_path=drop_path,
            )

        attention_groups = self.inv_groups

        if conv_layer or "kpconvd" in self.kp_mode:
            attention_groups = 0

        return DAKPNextMultiShortcutBlock(
            in_C,
            out_C,
            self.shell_sizes,
            radius,
            sigma,
            attention_groups=attention_groups,
            attention_act=self.inv_act,
            mod_grp_norm=self.inv_grp_norm,
            expansion=4,
            drop_path_p=drop_path,
            layer_scale_init_v=-1.0,
            use_upcut=self.kpx_upcut,
            shared_kp_data=shared_kp_data,
            influence_mode=self.kp_influence,
            dimension=self.dim,
            norm_type=self.norm,
            bn_momentum=self.bn_momentum,
        )

    def _get_da_scale_if_needed(self, stage_idx, points, neighbors, lengths):
        """
        Compute density-adaptive scale for selected encoder stages.

        Return:
            None if DA is disabled or this stage is not selected.
            Tensor [N, 1] otherwise.
        """

        if not self.enable_da:
            return None

        if stage_idx not in self.da_stages:
            return None

        return self.da_scale(
            points=points,
            neighbors=neighbors,
            lengths=lengths,
        )

    def _get_da_radius_scale_if_needed(self, stage_idx, points, neighbors, lengths):
        if not self.enable_da_radius:
            return None

        if stage_idx not in self.da_radius_stages:
            return None

        if not self._should_apply_da_radius_block_mask():
            return None

        return self.da_radius(
            points=points,
            neighbors=neighbors,
            lengths=lengths,
            scale_range=self._get_da_radius_scale_range(stage_idx),
        )

    def _build_pyramid(self, points, lengths):
        self._last_da_radius_local_stats = None
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

        if (
            not self.enable_da_radius
            or self.da_radius_backend != "cuda"
            or not points.is_cuda
        ):
            return in_dict

        radius_scales = []
        for layer in range(1, self.num_layers + 1):
            l = layer - 1
            if layer in self.da_radius_stages:
                radius_scales.append(
                    self.da_radius(
                        points=in_dict.points[l],
                        neighbors=in_dict.neighbors[l],
                        lengths=in_dict.lengths[l],
                        scale_range=self._get_da_radius_scale_range(layer),
                    )
                )
            else:
                radius_scales.append(None)

        cuda_in_dict = build_full_pyramid(
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
            da_radius_scales=radius_scales,
            da_radius_backend="cuda",
        )
        self._log_da_radius_debug(cuda_in_dict, radius_scales)
        self._last_da_radius_local_stats = self._collect_da_radius_local_stats(
            cuda_in_dict, radius_scales
        )
        self._da_radius_debug_step += 1
        return cuda_in_dict

    def forward(self, data_dict):
        # ------ Init ------
        points = data_dict["coord"]
        feats = data_dict["feat"]
        offset = data_dict["offset"].int()

        offset = torch.cat(
            [torch.zeros(1, dtype=offset.dtype, device=offset.device), offset],
            dim=0,
        )
        lengths = offset[1:] - offset[:-1]

        in_dict = self._build_pyramid(points, lengths)

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
                da_scale = self._get_da_scale_if_needed(
                    stage_idx=layer,
                    points=in_dict.points[l],
                    neighbors=in_dict.neighbors[l],
                    lengths=in_dict.lengths[l],
                )
                da_radius_scale = self._get_da_radius_scale_if_needed(
                    stage_idx=layer,
                    points=in_dict.points[l],
                    neighbors=in_dict.neighbors[l],
                    lengths=in_dict.lengths[l],
                )

                upcut = None
                for block in block_list:
                    feats, upcut = block(
                        in_dict.points[l],
                        in_dict.points[l],
                        feats,
                        in_dict.neighbors[l],
                        in_dict.lengths[l],
                        upcut=upcut,
                        da_scale=da_scale,
                        da_radius_scale=da_radius_scale,
                    )

            context_token = self._collect_sgca_context_if_needed(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                lengths=in_dict.lengths[l],
            )
            if context_token is not None:
                context_tokens.append(context_token)

            # Encoder-mode global context branch from Stage-1.
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
                        # Decoder remains unchanged for stability.
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
                local_stats=self._last_da_radius_local_stats,
            )

        # ------ Head ------
        logits = self.head(feats)
        return logits
