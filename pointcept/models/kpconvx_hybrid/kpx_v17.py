import math

import torch

from pointcept.models.builder import MODELS

from .da_kpconvx_block import DensityAdaptiveRadius
from .da_kpnext_blocks import DAKPNextMultiShortcutBlock
from .kpx_v16 import KPConvXV16


@MODELS.register_module()
class KPConvXV17(KPConvXV16):
    """v17: original support as identity path, expanded support as gated residual."""

    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_dual_support=False,
        dual_support_stages=(4,),
        dual_support_scale_range=(1.0, 1.2),
        dual_support_stage_ranges=None,
        dual_support_min_keep=None,
        dual_support_base_limits=None,
        dual_support_warmup_steps=0,
        dual_support_ramp_steps=0,
        dual_support_gamma_init=1.0e-3,
        dual_support_gate_bias_init=-2.0,
        dual_support_hidden_ratio=0.25,
        dual_support_eval_alpha=None,
        enable_v17_monitor=False,
        v17_monitor_stages=None,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs,
    ):
        self.enable_dual_support = bool(enable_dual_support)
        self.dual_support_stages = (
            tuple(dual_support_stages) if enable_dual_support else tuple()
        )
        self.dual_support_scale_range = dual_support_scale_range
        self._dual_support_stage_ranges_raw = dual_support_stage_ranges
        self._dual_support_min_keep_raw = dual_support_min_keep
        self._dual_support_base_limits_raw = dual_support_base_limits
        self.dual_support_warmup_steps = max(int(dual_support_warmup_steps), 0)
        self.dual_support_ramp_steps = max(int(dual_support_ramp_steps), 0)
        self.dual_support_gamma_init = float(dual_support_gamma_init)
        self.dual_support_gate_bias_init = float(dual_support_gate_bias_init)
        self.dual_support_hidden_ratio = float(dual_support_hidden_ratio)
        self.dual_support_eval_alpha = (
            None if dual_support_eval_alpha is None else float(dual_support_eval_alpha)
        )
        self.enable_v17_monitor = bool(enable_v17_monitor)
        if v17_monitor_stages is None:
            v17_monitor_stages = dual_support_stages
        self.v17_monitor_stages = tuple(v17_monitor_stages)

        for key in (
            "enable_support_mask",
            "support_mask_stages",
            "support_mask_scale_range",
            "support_mask_stage_ranges",
            "support_mask_min_keep",
            "support_mask_base_limits",
            "support_mask_density_k",
            "support_mask_norm",
            "support_mask_percentile",
            "support_mask_strength",
            "support_mask_power",
            "support_mask_warmup_steps",
            "support_mask_ramp_steps",
            "enable_v16_monitor",
            "v16_monitor_stages",
        ):
            kwargs.pop(key, None)

        super().__init__(
            in_channels=in_channels,
            input_channels=input_channels,
            num_classes=num_classes,
            enable_support_mask=False,
            enable_v16_monitor=enable_v17_monitor,
            v16_monitor_stages=v17_monitor_stages,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs,
        )

        for stage in self.dual_support_stages:
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid dual-support stage index: {stage}")

        self.dual_support_stage_ranges = self._normalize_stage_ranges(
            self._dual_support_stage_ranges_raw
        )
        self.dual_support_min_keep = self._normalize_stage_int_map(
            self._dual_support_min_keep_raw
        )
        self.dual_support_base_limits = self._normalize_stage_int_map(
            self._dual_support_base_limits_raw
        )
        self.register_buffer(
            "dual_support_step",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        if self.enable_dual_support:
            self.dual_support_radius = DensityAdaptiveRadius(
                scale_range=self.dual_support_scale_range,
                density_k=self.da_radius_density_k,
                norm=self.da_radius_norm,
                percentile=self.da_radius_percentile,
                strength=1.0,
                power=self.da_radius_power,
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
            dual_support_enabled=self.enable_dual_support,
            dual_support_hidden_ratio=self.dual_support_hidden_ratio,
            dual_support_gamma_init=self.dual_support_gamma_init,
            dual_support_gate_bias_init=self.dual_support_gate_bias_init,
        )

    def _get_dual_support_scale_range(self, stage_idx):
        return self.dual_support_stage_ranges.get(
            int(stage_idx), self.dual_support_scale_range
        )

    def _get_dual_support_progress(self):
        if not self.enable_dual_support:
            return 0.0
        step = int(self.dual_support_step.detach().cpu().item())
        if step < self.dual_support_warmup_steps:
            return 0.0
        if self.dual_support_ramp_steps <= 0:
            return 1.0
        progress = (step - self.dual_support_warmup_steps + 1) / float(
            self.dual_support_ramp_steps
        )
        return max(0.0, min(1.0, progress))

    def _get_dual_support_alpha(self):
        if not self.enable_dual_support:
            return 0.0
        if not self.training and self.dual_support_eval_alpha is not None:
            return max(0.0, float(self.dual_support_eval_alpha))
        return 1.0

    def _is_dual_support_stage(self, stage_idx):
        return self.enable_dual_support and int(stage_idx) in self.dual_support_stages

    def _get_dual_support_scale_if_needed(self, stage_idx, points, neighbors, lengths):
        if not self._is_dual_support_stage(stage_idx):
            return None
        if self._get_dual_support_progress() <= 0.0:
            return None
        if self._get_dual_support_alpha() <= 0.0:
            return None
        return self.dual_support_radius(
            points=points,
            neighbors=neighbors,
            lengths=lengths,
            scale_range=self._get_dual_support_scale_range(stage_idx),
        )

    def _get_dual_support_min_keep(self, stage_idx):
        return int(self.dual_support_min_keep.get(int(stage_idx), 0))

    def _get_dual_support_base_limit(self, stage_idx):
        value = self.dual_support_base_limits.get(int(stage_idx))
        return int(value) if value is not None else None

    @staticmethod
    def _collect_dual_support_diag_from_blocks(blocks):
        diag_lists = {}
        for block in blocks:
            for module in block.modules():
                diag = getattr(module, "_last_dual_support_diag", None)
                if not diag:
                    continue
                for key, value in diag.items():
                    diag_lists.setdefault(key, []).append(value.detach())

        return {
            key: torch.stack(values).mean().detach()
            for key, values in diag_lists.items()
            if values
        }

    def _collect_dual_support_monitor(self, dual_support_by_stage, ref_tensor):
        metrics = {}
        if not self.enable_dual_support:
            return metrics

        metrics["v17_dual_progress"] = ref_tensor.new_tensor(
            float(self._get_dual_support_progress())
        )
        metrics["v17_dual_alpha"] = ref_tensor.new_tensor(
            float(self._get_dual_support_alpha())
        )
        key_map = {
            "scale_p10": "scale_p10",
            "scale_p50": "scale_p50",
            "scale_p90": "scale_p90",
            "alpha": "alpha",
            "progress": "progress",
            "keep_ratio_p10": "keep_p10",
            "keep_ratio_p50": "keep_p50",
            "keep_ratio_p90": "keep_p90",
            "valid_ratio": "valid",
            "fallback_hit": "fallback",
            "extra_util": "extra_util",
            "gamma": "gamma",
            "effective_gamma": "effective_gamma",
            "gate_mean": "gate_mean",
            "residual_abs": "residual_abs",
            "residual_ratio": "residual_ratio",
        }
        for stage in self.dual_support_stages:
            stage = int(stage)
            diag = dual_support_by_stage.get(stage, {})
            prefix = f"v17_s{stage}_dual"
            for src_key, dst_key in key_map.items():
                value = diag.get(src_key)
                if value is None:
                    value = ref_tensor.new_zeros(())
                metrics[f"{prefix}_{dst_key}"] = value.to(
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
        dual_support_by_stage = {}

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
                dual_support_scale = self._get_dual_support_scale_if_needed(
                    stage_idx=layer,
                    points=in_dict.points[l],
                    neighbors=in_dict.neighbors[l],
                    lengths=in_dict.lengths[l],
                )
                dual_support_enabled_for_stage = self._is_dual_support_stage(layer)
                dual_support_progress = self._get_dual_support_progress()
                dual_support_alpha = self._get_dual_support_alpha()
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
                        da_radius_min_keep=0,
                        da_radius_base_limit=None,
                        da_meta=da_meta,
                        base_neighbor_limit=(
                            self._get_dual_support_base_limit(layer)
                            if dual_support_enabled_for_stage
                            else None
                        ),
                        dual_radius_scale=dual_support_scale,
                        dual_radius_min_keep=(
                            self._get_dual_support_min_keep(layer)
                            if dual_support_scale is not None
                            else 0
                        ),
                        dual_radius_base_limit=(
                            self._get_dual_support_base_limit(layer)
                            if dual_support_scale is not None
                            else None
                        ),
                        dual_radius_progress=dual_support_progress,
                        dual_radius_alpha=dual_support_alpha,
                    )

                if dual_support_scale is not None:
                    dual_diag = self._collect_dual_support_diag_from_blocks(block_list)
                    dual_scale_flat = dual_support_scale.reshape(-1).detach()
                    dual_diag.update(
                        dict(
                            scale_p10=self._safe_quantile(dual_scale_flat, 0.10),
                            scale_p50=self._safe_quantile(dual_scale_flat, 0.50),
                            scale_p90=self._safe_quantile(dual_scale_flat, 0.90),
                        )
                    )
                    dual_support_by_stage[int(layer)] = dual_diag

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
        if self.enable_v17_monitor:
            output = dict(seg_logits=logits)
            output.update(
                self._collect_v16_monitor(
                    da_meta_by_stage,
                    {},
                    in_dict,
                    logits,
                )
            )
            output.update(self._collect_dual_support_monitor(dual_support_by_stage, logits))
            if self.training and self.enable_dual_support:
                with torch.no_grad():
                    self.dual_support_step.add_(1)
            return output

        if self.training and self.enable_dual_support:
            with torch.no_grad():
                self.dual_support_step.add_(1)
        return logits
