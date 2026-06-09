import math

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS

from .da_kpnext_blocks import DAKPNextMultiShortcutBlock
from .kpx_stage2 import KPConvXStage2


class GlobalContextMixerLite(nn.Module):
    """Lightweight decoder-side context mixer for v16."""

    def __init__(
        self,
        stage_dims,
        decoder_dim,
        stats_dim=0,
        hidden_ratio=1.0,
        dropout=0.0,
        gamma_init=0.0,
    ):
        super().__init__()
        self.stage_keys = [str(stage) for stage in stage_dims.keys()]
        self.stats_dim = int(stats_dim)
        hidden_dim = max(16, int(decoder_dim * float(hidden_ratio)))

        self.stage_proj = nn.ModuleDict(
            {
                str(stage): nn.Linear(dim, decoder_dim, bias=False)
                for stage, dim in stage_dims.items()
            }
        )
        self.ctx_mix = nn.Sequential(
            nn.Linear(len(stage_dims) * decoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, decoder_dim),
        )
        self.ctx_norm = nn.LayerNorm(decoder_dim)

        gate_in_dim = decoder_dim * 2
        if self.stats_dim > 0:
            self.stats_proj = nn.Sequential(
                nn.Linear(self.stats_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, decoder_dim),
            )
            gate_in_dim += decoder_dim

        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, decoder_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, decoder_dim),
        )
        self.gamma = nn.Parameter(torch.full((1,), float(gamma_init)))

    @staticmethod
    def _cloud_mean(feats, lengths):
        parts = []
        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            if end <= start:
                parts.append(feats.new_zeros((feats.shape[1],)))
            else:
                parts.append(feats[start:end].mean(dim=0))
            start = end
        return torch.stack(parts, dim=0)

    @staticmethod
    def _expand_to_points(context, lengths):
        pieces = []
        for row, length in zip(context, lengths.tolist()):
            pieces.append(row.unsqueeze(0).expand(int(length), -1))
        return torch.cat(pieces, dim=0)

    def forward(self, feats, lengths, stage_contexts, meta_cloud_stats=None):
        if not stage_contexts:
            return feats

        projected = []
        for key in self.stage_keys:
            if key not in stage_contexts:
                return feats
            stage_feats, stage_lengths = stage_contexts[key]
            cloud_ctx = self._cloud_mean(stage_feats, stage_lengths)
            projected.append(self.stage_proj[key](cloud_ctx))

        cloud_ctx = self.ctx_norm(self.ctx_mix(torch.cat(projected, dim=-1)))
        point_ctx = self._expand_to_points(cloud_ctx, lengths)
        gate_parts = [feats, point_ctx]

        if self.stats_dim > 0:
            if meta_cloud_stats is None:
                meta_cloud_stats = feats.new_zeros(
                    (int(lengths.numel()), self.stats_dim)
                )
            else:
                meta_cloud_stats = meta_cloud_stats.to(
                    device=feats.device, dtype=feats.dtype
                )
            meta_ctx = self.stats_proj(meta_cloud_stats)
            gate_parts.append(self._expand_to_points(meta_ctx, lengths))

        gate_input = torch.cat(gate_parts, dim=-1)
        return feats + self.gamma * self.gate(gate_input) * self.fuse(gate_input)


@MODELS.register_module()
class KPConvXV16(KPConvXStage2):
    """Low-risk v16 backbone: DA-stat conditioned KPConvX plus lite GC mixer."""

    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_da_meta=False,
        da_meta_stages=(3, 4),
        da_meta_dim=4,
        da_meta_hidden_ratio=0.25,
        da_meta_use_channel_bias=False,
        da_meta_use_shell_bias=False,
        da_meta_use_point_bias=False,
        enable_gc_mixer=False,
        gc_stages=(5,),
        gc_use_meta_stats=False,
        gc_hidden_ratio=1.0,
        gc_dropout=0.0,
        gc_gamma_init=0.0,
        enable_v16_monitor=False,
        v16_monitor_stages=None,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs,
    ):
        self.enable_da_meta = bool(enable_da_meta)
        self.da_meta_stages = tuple(da_meta_stages) if enable_da_meta else tuple()
        self.da_meta_dim = int(da_meta_dim)
        self.da_meta_hidden_ratio = float(da_meta_hidden_ratio)
        self.da_meta_use_channel_bias = bool(da_meta_use_channel_bias)
        self.da_meta_use_shell_bias = bool(da_meta_use_shell_bias)
        self.da_meta_use_point_bias = bool(da_meta_use_point_bias)

        self.enable_gc_mixer = bool(enable_gc_mixer)
        self.gc_stages = tuple(gc_stages) if enable_gc_mixer else tuple()
        self.gc_use_meta_stats = bool(gc_use_meta_stats)
        self.gc_hidden_ratio = float(gc_hidden_ratio)
        self.gc_dropout = float(gc_dropout)
        self.gc_gamma_init = float(gc_gamma_init)
        self.enable_v16_monitor = bool(enable_v16_monitor)
        if v16_monitor_stages is None:
            v16_monitor_stages = da_meta_stages
        self.v16_monitor_stages = tuple(v16_monitor_stages)

        for key in (
            "enable_router",
            "router_cfg",
            "router_stages",
            "router_hidden_ratio",
            "router_dropout",
            "router_temperature",
            "router_global_boost",
            "fusion_cfg",
            "refine_cfg",
            "enable_refine",
            "refine_hidden_ratio",
            "refine_dropout",
            "refine_use_coords",
            "refine_use_boundary",
            "boundary_loss_weight",
            "boundary_hidden_ratio",
            "boundary_dropout",
            "boundary_ignore_index",
            "boundary_dilate_steps",
            "global_boundary_gate",
            "global_boundary_min_keep",
            "global_boundary_detach",
        ):
            kwargs.pop(key, None)

        kwargs.setdefault("enable_global", False)
        kwargs.setdefault("use_global_context", False)

        super().__init__(
            in_channels=in_channels,
            input_channels=input_channels,
            num_classes=num_classes,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs,
        )

        for stage in self.da_meta_stages:
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid DA-meta stage index: {stage}")
        for stage in self.gc_stages:
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid GC mixer stage index: {stage}")

        layer_channels = self._compute_layer_channels(
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            num_layers=self.num_layers,
        )
        self.gc_stage_dims = {}
        for stage in self.gc_stages:
            if self.grid_pool and stage < self.num_layers:
                dim = layer_channels[stage]
            else:
                dim = layer_channels[stage - 1]
            self.gc_stage_dims[int(stage)] = dim

        self.gc_meta_stats_dim = (
            self.da_meta_dim * len(self.da_meta_stages)
            if self.gc_use_meta_stats
            else 0
        )
        if self.enable_gc_mixer:
            self.gc_mixer = GlobalContextMixerLite(
                stage_dims=self.gc_stage_dims,
                decoder_dim=layer_channels[0],
                stats_dim=self.gc_meta_stats_dim,
                hidden_ratio=self.gc_hidden_ratio,
                dropout=self.gc_dropout,
                gamma_init=self.gc_gamma_init,
            )

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
            da_meta_dim=self.da_meta_dim if self.enable_da_meta else 0,
            da_meta_hidden_ratio=self.da_meta_hidden_ratio,
            da_meta_use_channel_bias=self.da_meta_use_channel_bias,
            da_meta_use_shell_bias=self.da_meta_use_shell_bias,
            da_meta_use_point_bias=self.da_meta_use_point_bias,
        )

    def _get_da_meta_if_needed(self, stage_idx, points, neighbors, lengths):
        if not self.enable_da_meta:
            return None
        if stage_idx not in self.da_meta_stages:
            return None
        _, meta = self.da_radius(
            points=points,
            neighbors=neighbors,
            lengths=lengths,
            scale_range=self._get_da_radius_scale_range(stage_idx),
            return_meta=True,
        )
        return meta

    @staticmethod
    def _cloud_mean(feats, lengths):
        parts = []
        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            if end <= start:
                parts.append(feats.new_zeros((feats.shape[1],)))
            else:
                parts.append(feats[start:end].mean(dim=0))
            start = end
        return torch.stack(parts, dim=0)

    def _collect_meta_cloud_stats(self, da_meta_by_stage, lengths_by_stage, feats):
        if not self.gc_use_meta_stats or self.gc_meta_stats_dim <= 0:
            return None

        pieces = []
        batch_size = int(lengths_by_stage[0].numel())
        for stage in self.da_meta_stages:
            meta = da_meta_by_stage.get(int(stage))
            if meta is None:
                pieces.append(
                    feats.new_zeros((batch_size, self.da_meta_dim), dtype=feats.dtype)
                )
                continue
            meta_feat = meta["feat"].to(device=feats.device, dtype=feats.dtype)
            pieces.append(
                self._cloud_mean(meta_feat, lengths_by_stage[int(stage) - 1])
            )

        return torch.cat(pieces, dim=-1) if pieces else None

    @staticmethod
    def _safe_quantile(values, q):
        if values.numel() == 0:
            return values.new_zeros(())
        return torch.quantile(values.float(), q)

    def _collect_da_meta_diag(self, ref_tensor):
        diag_lists = {}
        for module in self.modules():
            diag = getattr(module, "_last_da_meta_diag", None)
            if not diag:
                continue
            for key, value in diag.items():
                diag_lists.setdefault(key, []).append(
                    value.detach().to(device=ref_tensor.device, dtype=ref_tensor.dtype)
                )

        metrics = {}
        if not diag_lists:
            zero = ref_tensor.new_zeros(())
            metrics["v16_da_alpha_abs"] = zero
            metrics["v16_da_bias_abs"] = zero
            metrics["v16_da_bias_ratio"] = zero
            metrics["v16_da_channel_w_norm"] = zero
            metrics["v16_da_channel_b_norm"] = zero
            metrics["v16_da_shell_w_norm"] = zero
            metrics["v16_da_shell_b_norm"] = zero
            return metrics

        key_map = {
            "alpha_abs": "v16_da_alpha_abs",
            "bias_abs": "v16_da_bias_abs",
            "bias_alpha_ratio": "v16_da_bias_ratio",
            "channel_w_norm": "v16_da_channel_w_norm",
            "channel_b_norm": "v16_da_channel_b_norm",
            "shell_w_norm": "v16_da_shell_w_norm",
            "shell_b_norm": "v16_da_shell_b_norm",
        }
        for src_key, dst_key in key_map.items():
            values = diag_lists.get(src_key)
            if values:
                metrics[dst_key] = torch.stack(values).mean().detach()
            else:
                metrics[dst_key] = ref_tensor.new_zeros(())
        return metrics

    def _collect_stage_monitor(self, da_meta_by_stage, in_dict, ref_tensor):
        metrics = {}
        for stage in self.v16_monitor_stages:
            stage = int(stage)
            meta = da_meta_by_stage.get(stage)
            if meta is None:
                continue

            scale = meta["scale"].reshape(-1).to(
                device=ref_tensor.device, dtype=ref_tensor.dtype
            )
            feat = meta["feat"].to(device=ref_tensor.device, dtype=ref_tensor.dtype)

            prefix = f"v16_s{stage}"
            metrics[f"{prefix}_scale_p10"] = self._safe_quantile(scale, 0.10).to(
                dtype=ref_tensor.dtype
            )
            metrics[f"{prefix}_scale_p50"] = self._safe_quantile(scale, 0.50).to(
                dtype=ref_tensor.dtype
            )
            metrics[f"{prefix}_scale_p90"] = self._safe_quantile(scale, 0.90).to(
                dtype=ref_tensor.dtype
            )
            metrics[f"{prefix}_rho"] = feat[:, 1].mean().detach()
            metrics[f"{prefix}_meta_valid"] = feat[:, 2].mean().detach()
            metrics[f"{prefix}_dist_cv"] = feat[:, 3].mean().detach()

            l = stage - 1
            if l < 0 or l >= len(in_dict.neighbors):
                continue
            neighbors = in_dict.neighbors[l]
            num_points = int(in_dict.points[l].shape[0])
            neighbor_limit = max(int(neighbors.shape[1]), 1)
            valid = (neighbors >= 0) & (neighbors < num_points)
            valid_count = valid.sum(dim=1).to(device=ref_tensor.device).float()
            valid_ratio = valid_count / float(neighbor_limit)
            metrics[f"{prefix}_graph_valid"] = valid_ratio.mean().to(
                dtype=ref_tensor.dtype
            )
            metrics[f"{prefix}_shadow"] = (1.0 - valid_ratio.mean()).to(
                dtype=ref_tensor.dtype
            )
            metrics[f"{prefix}_full"] = (
                valid_count >= float(neighbor_limit)
            ).float().mean().to(dtype=ref_tensor.dtype)
        return metrics

    def _collect_v16_monitor(self, da_meta_by_stage, in_dict, ref_tensor):
        metrics = {}
        metrics.update(self._collect_da_meta_diag(ref_tensor))
        metrics.update(self._collect_stage_monitor(da_meta_by_stage, in_dict, ref_tensor))
        if self.enable_gc_mixer and hasattr(self, "gc_mixer"):
            metrics["v16_gc_gamma"] = self.gc_mixer.gamma.detach().reshape(()).to(
                device=ref_tensor.device, dtype=ref_tensor.dtype
            )
        return metrics

    def forward(self, data_dict):
        points = data_dict["coord"]
        feats = data_dict["feat"]
        offset = data_dict["offset"].int()

        offset = torch.cat(
            [torch.zeros(1, dtype=offset.dtype, device=offset.device), offset],
            dim=0,
        )
        lengths = offset[1:] - offset[:-1]
        in_dict = self._build_pyramid(points, lengths)

        feats = self.stem(
            in_dict.points[0],
            in_dict.points[0],
            feats,
            in_dict.neighbors[0],
        )

        skip_feats = []
        gc_stage_contexts = {}
        da_meta_by_stage = {}

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
                da_meta = self._get_da_meta_if_needed(
                    stage_idx=layer,
                    points=in_dict.points[l],
                    neighbors=in_dict.neighbors[l],
                    lengths=in_dict.lengths[l],
                )
                if da_meta is not None:
                    da_meta_by_stage[int(layer)] = da_meta

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
                        da_meta=da_meta,
                    )

            if self.enable_gc_mixer and layer in self.gc_stages:
                gc_stage_contexts[str(layer)] = (feats, in_dict.lengths[l])

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

            if self.enable_gc_mixer:
                meta_cloud_stats = self._collect_meta_cloud_stats(
                    da_meta_by_stage,
                    in_dict.lengths,
                    feats,
                )
                feats = self.gc_mixer(
                    feats=feats,
                    lengths=in_dict.lengths[0],
                    stage_contexts=gc_stage_contexts,
                    meta_cloud_stats=meta_cloud_stats,
                )

        logits = self.head(feats)
        if self.enable_v16_monitor:
            output = dict(seg_logits=logits)
            output.update(self._collect_v16_monitor(da_meta_by_stage, in_dict, logits))
            return output
        return logits
