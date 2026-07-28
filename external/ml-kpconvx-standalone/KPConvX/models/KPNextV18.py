"""V18 dual-support diagnostic model for KPConvX Standalone."""

import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_

from kernels.kernel_points import load_kernels
from models.KPNext import KPNeXt
from models.generic_blocks import NormBlock, index_select, mlp_from_list
from utils.dual_support import DimensionlessSpacingScale, build_ring_neighbors
from utils.torch_pyramid import fill_pyramid


class AdaptiveRingKPConvX(nn.Module):
    """KPConvX-style residual that processes only newly added ring neighbors."""

    def __init__(
        self,
        channels,
        shell_sizes,
        radius,
        sigma,
        attention_groups=8,
        attention_act="sigmoid",
        mod_grp_norm=False,
        influence_mode="linear",
        dimension=3,
        norm_type="batch",
        bn_momentum=0.1,
        activation=nn.LeakyReLU(0.1),
        gamma_mode="fixed",
        gamma=0.25,
        inf=1e6,
    ):
        super().__init__()
        if attention_groups > 0:
            if channels % attention_groups != 0:
                raise ValueError("channels must be divisible by attention_groups")
            ch_per_group = channels // attention_groups
        else:
            ch_per_group = -attention_groups
            if channels % ch_per_group != 0:
                raise ValueError("channels must be divisible by channels per group")
            attention_groups = channels // ch_per_group

        self.channels = int(channels)
        self.shell_sizes = tuple(shell_sizes)
        self.K = int(np.sum(shell_sizes))
        self.radius = float(radius)
        self.sigma = float(sigma)
        self.influence_mode = str(influence_mode)
        self.dimension = int(dimension)
        self.ch_per_group = int(ch_per_group)
        self.groups = int(attention_groups)
        self.mod_grp_norm = bool(mod_grp_norm)
        self.inf = float(inf)

        self.weights = nn.Parameter(torch.zeros((self.K, channels)))
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        kernel_points = load_kernels(
            self.radius,
            self.shell_sizes,
            dimension=self.dimension,
            fixed="center",
        )
        self.register_buffer("kernel_points", torch.from_numpy(kernel_points).float())

        attention_out = self.K * self.ch_per_group
        self.alpha_mlp = mlp_from_list(
            channels,
            [channels, "NA", attention_out],
            final_bias=False,
            norm_type="none",
            bn_momentum=-1,
            activation=activation,
        )
        self.group_norm = nn.GroupNorm(self.K, attention_out)
        if attention_act == "sigmoid":
            self.attention_activation = torch.sigmoid
        elif attention_act == "tanh":
            self.attention_activation = torch.tanh
        elif attention_act == "softmax":
            self.attention_activation = nn.Softmax(dim=1)
        else:
            self.attention_activation = nn.Identity()

        self.norm = NormBlock(channels, norm_type, bn_momentum)
        self.activation = activation
        gamma_tensor = torch.tensor(float(gamma))
        if gamma_mode == "learnable":
            self.gamma = nn.Parameter(gamma_tensor)
        elif gamma_mode == "fixed":
            self.register_buffer("gamma", gamma_tensor)
        else:
            raise ValueError("gamma_mode must be 'fixed' or 'learnable'")

    def forward(self, q_points, s_points, s_features, ring_neighbors, radius_scale):
        num_support = int(s_points.shape[0])
        valid = (ring_neighbors >= 0) & (ring_neighbors < num_support)
        padded_points = torch.cat(
            (s_points, torch.zeros_like(s_points[:1]) + self.inf), dim=0
        )
        padded_features = torch.cat(
            (s_features, torch.zeros_like(s_features[:1])), dim=0
        )
        neighbor_points = index_select(padded_points, ring_neighbors, dim=0)
        neighbor_features = index_select(padded_features, ring_neighbors, dim=0)
        relative = neighbor_points - q_points.unsqueeze(1)

        scale = radius_scale.to(device=relative.device, dtype=relative.dtype).view(
            -1, 1, 1
        )
        scaled_kernel_points = self.kernel_points.unsqueeze(0) * scale
        differences = relative.unsqueeze(2) - scaled_kernel_points.unsqueeze(1)
        squared_distances = torch.sum(differences.square(), dim=-1)
        nearest_squared, nearest_kernel = torch.min(squared_distances, dim=2)

        influence = None
        if self.influence_mode == "linear":
            scaled_sigma = self.sigma * scale.reshape(-1, 1)
            influence = torch.clamp(
                1.0 - torch.sqrt(nearest_squared) / scaled_sigma,
                min=0.0,
            )
        elif self.influence_mode == "gaussian":
            scaled_sigma = self.sigma * scale.reshape(-1, 1) * 0.3
            influence = torch.exp(-nearest_squared / (2.0 * scaled_sigma.square()))
        elif self.influence_mode != "constant":
            raise ValueError("Unsupported influence mode: {}".format(self.influence_mode))

        modulations = self.alpha_mlp(s_features)
        if self.mod_grp_norm:
            modulations = modulations.transpose(0, 1).unsqueeze(0)
            modulations = self.group_norm(modulations)
            modulations = modulations.squeeze(0).transpose(0, 1)
        modulations = self.attention_activation(modulations)
        modulations = modulations.view(-1, self.K, self.ch_per_group, 1)
        conv_weights = self.weights.view(
            1, self.K, self.ch_per_group, self.groups
        )
        conv_weights = (conv_weights * modulations).reshape(
            -1, self.K, self.channels
        )
        neighbor_weights = torch.gather(
            conv_weights,
            1,
            nearest_kernel.unsqueeze(2).expand(-1, -1, self.channels),
        )
        if influence is not None:
            neighbor_weights = neighbor_weights * influence.unsqueeze(2)
        neighbor_weights = neighbor_weights * valid.unsqueeze(2).to(
            dtype=neighbor_weights.dtype
        )

        residual = torch.sum(neighbor_features * neighbor_weights, dim=1)
        residual = self.activation(self.norm(residual))
        has_ring = valid.any(dim=1, keepdim=True).to(dtype=residual.dtype)
        return self.gamma.to(dtype=residual.dtype) * residual * has_ring


class KPNeXtV18(KPNeXt):
    """Identity KPNeXt path plus a strong, ring-only adaptive support branch."""

    def __init__(self, cfg):
        super().__init__(cfg)
        model_cfg = cfg.model
        if model_cfg.kp_mode != "kpconvx":
            raise ValueError("KPNeXtV18 currently supports kp_mode='kpconvx' only")
        self.base_neighbor_limits = tuple(
            int(v) for v in model_cfg.base_neighbor_limits
        )
        self.candidate_neighbor_limits = tuple(
            int(v) for v in model_cfg.neighbor_limits
        )
        self.dual_support_stages = tuple(
            int(v) for v in model_cfg.dual_support_stages
        )
        self.dual_support_ring_limits = {
            int(k): int(v) for k, v in dict(model_cfg.dual_support_ring_limits).items()
        }
        self.dual_support_stage_ranges = {
            int(k): tuple(v)
            for k, v in dict(model_cfg.dual_support_stage_ranges).items()
        }
        self.dual_support_fixed_spacing_bounds = {
            int(k): tuple(v)
            for k, v in dict(
                getattr(model_cfg, "dual_support_fixed_spacing_bounds", {})
            ).items()
        }
        self.dual_support_use_fixed_spacing = bool(
            getattr(model_cfg, "dual_support_use_fixed_spacing", False)
        )
        self.dual_support_debug = bool(
            getattr(model_cfg, "dual_support_debug", True)
        )
        self.dual_support_debug_interval = max(
            int(getattr(model_cfg, "dual_support_debug_interval", 100)), 1
        )
        self._dual_support_step = 0
        self.last_dual_support_stats = {}

        if len(self.base_neighbor_limits) != self.num_layers:
            raise ValueError("base_neighbor_limits must cover every layer")
        if len(self.candidate_neighbor_limits) != self.num_layers:
            raise ValueError("candidate neighbor limits must cover every layer")

        self.spacing_scale = DimensionlessSpacingScale(
            density_k=model_cfg.dual_support_density_k,
            percentile=model_cfg.dual_support_percentile,
            strength=model_cfg.dual_support_strength,
            power=model_cfg.dual_support_power,
        )

        channel_scaling = float(model_cfg.channel_scaling)
        stage_channels = []
        for layer in range(self.num_layers):
            target = float(model_cfg.init_channels) * channel_scaling ** layer
            stage_channels.append(int(np.ceil((target - 0.1) / 16)) * 16)

        self.ring_modules = nn.ModuleDict()
        for stage in self.dual_support_stages:
            layer = stage - 1
            if layer < 0 or layer >= self.num_layers:
                raise ValueError("invalid dual-support stage: {}".format(stage))
            if self.candidate_neighbor_limits[layer] <= self.base_neighbor_limits[layer]:
                raise ValueError("dual-support candidate limit must exceed base H")
            ring_limit = self.dual_support_ring_limits[stage]
            if ring_limit > (
                self.candidate_neighbor_limits[layer] - self.base_neighbor_limits[layer]
            ):
                raise ValueError("ring limit does not fit candidate graph")

            radius = self.first_radius * self.radius_scaling ** layer
            sigma = self.first_sigma * self.radius_scaling ** layer
            self.ring_modules[str(stage)] = AdaptiveRingKPConvX(
                stage_channels[layer],
                model_cfg.shell_sizes,
                radius,
                sigma,
                attention_groups=model_cfg.inv_groups,
                attention_act=model_cfg.inv_act,
                mod_grp_norm=model_cfg.inv_grp_norm,
                influence_mode=model_cfg.kp_influence,
                dimension=cfg.data.dim,
                norm_type=model_cfg.norm,
                bn_momentum=model_cfg.bn_momentum,
                gamma_mode=model_cfg.dual_support_gamma_mode,
                gamma=model_cfg.dual_support_gamma,
            )

    def _ensure_pyramid(self, in_dict):
        if len(in_dict.neighbors) > 0:
            return
        fill_pyramid(
            in_dict,
            self.num_layers,
            self.subsample_size,
            self.first_radius,
            self.radius_scaling,
            self.candidate_neighbor_limits,
            self.upsample_n,
            sub_mode=self.in_sub_mode,
            grid_pool_mode=self.grid_pool,
        )

    @staticmethod
    def _finite_mean(values):
        finite = torch.isfinite(values)
        if finite.any():
            return float(values[finite].mean().item())
        return float("nan")

    def _apply_ring(self, stage, in_dict, features):
        layer = stage - 1
        points = in_dict.points[layer]
        candidates = in_dict.neighbors[layer]
        base_limit = self.base_neighbor_limits[layer]
        if candidates.shape[1] < self.candidate_neighbor_limits[layer]:
            raise ValueError("precomputed candidate graph is smaller than configured")

        grid_size = self.subsample_size * self.radius_scaling ** layer
        fixed_bounds = None
        if self.dual_support_use_fixed_spacing:
            if stage not in self.dual_support_fixed_spacing_bounds:
                raise ValueError("missing fixed spacing bounds for stage {}".format(stage))
            fixed_bounds = self.dual_support_fixed_spacing_bounds[stage]

        scale, spacing = self.spacing_scale(
            points,
            candidates[:, :base_limit],
            in_dict.lengths[layer],
            grid_size=grid_size,
            scale_range=self.dual_support_stage_ranges[stage],
            fixed_bounds=fixed_bounds,
        )
        base_radius = self.first_radius * self.radius_scaling ** layer
        ring_neighbors, graph_stats = build_ring_neighbors(
            points,
            candidates,
            scale,
            base_radius=base_radius,
            base_limit=base_limit,
            ring_limit=self.dual_support_ring_limits[stage],
        )
        features = features + self.ring_modules[str(stage)](
            points,
            points,
            features,
            ring_neighbors,
            scale,
        )

        stats = {
            "scale_min": float(scale.min().item()),
            "scale_mean": float(scale.mean().item()),
            "scale_max": float(scale.max().item()),
            "spacing_q10": float(torch.quantile(spacing.float(), 0.10).item()),
            "spacing_q90": float(torch.quantile(spacing.float(), 0.90).item()),
            "base_radius_count": float(graph_stats["base_radius_count"].mean().item()),
            "adaptive_radius_count": float(
                graph_stats["adaptive_radius_count"].mean().item()
            ),
            "added_count": float(graph_stats["added_count"].mean().item()),
            "full_base_ratio": float(
                graph_stats["full_base_radius"].float().mean().item()
            ),
            "change_rate": float(graph_stats["change_rate"].mean().item()),
            "d_h_over_radius": self._finite_mean(graph_stats["d_h_over_radius"]),
        }
        self.last_dual_support_stats[stage] = stats
        return features

    def _log_dual_support(self):
        if not self.dual_support_debug:
            return
        if self._dual_support_step % self.dual_support_debug_interval != 0:
            return
        for stage, stats in self.last_dual_support_stats.items():
            print(
                "DualSupport step={} stage={} scale=[{:.3f},{:.3f},{:.3f}] "
                "spacing_q=[{:.3f},{:.3f}] added={:.2f} change={:.3f} "
                "full_base={:.3f} dH/r={:.3f}".format(
                    self._dual_support_step,
                    stage,
                    stats["scale_min"],
                    stats["scale_mean"],
                    stats["scale_max"],
                    stats["spacing_q10"],
                    stats["spacing_q90"],
                    stats["added_count"],
                    stats["change_rate"],
                    stats["full_base_ratio"],
                    stats["d_h_over_radius"],
                )
            )

    def forward(self, batch, verbose=False):
        if verbose:
            torch.cuda.synchronize(batch.device())
            timings = [time.time()]

        self._ensure_pyramid(batch.in_dict)
        features = batch.in_dict.features.clone().detach()
        base_neighbors = [
            neighbors[:, : self.base_neighbor_limits[layer]]
            for layer, neighbors in enumerate(batch.in_dict.neighbors)
        ]

        features = self.stem(
            batch.in_dict.points[0],
            batch.in_dict.points[0],
            features,
            base_neighbors[0],
        )

        skip_features = []
        self.last_dual_support_stats = {}
        for stage in range(1, self.num_layers + 1):
            layer = stage - 1
            blocks = getattr(self, "encoder_{:d}".format(stage))
            if self.kp_mode in ["kpconv", "kpconvtest"]:
                for block in blocks:
                    features = block(
                        batch.in_dict.points[layer],
                        batch.in_dict.points[layer],
                        features,
                        base_neighbors[layer],
                    )
            else:
                upcut = None
                for block in blocks:
                    features, upcut = block(
                        batch.in_dict.points[layer],
                        batch.in_dict.points[layer],
                        features,
                        base_neighbors[layer],
                        batch.in_dict.lengths[layer],
                        upcut=upcut,
                    )

            if stage in self.dual_support_stages:
                features = self._apply_ring(stage, batch.in_dict, features)

            if stage < self.num_layers:
                skip_features.append(features)
                pooling = getattr(self, "pooling_{:d}".format(stage))
                if self.grid_pool:
                    features = pooling(features, batch.in_dict.pools[layer])
                else:
                    features = pooling(
                        batch.in_dict.points[layer + 1],
                        batch.in_dict.points[layer],
                        features,
                        batch.in_dict.pools[layer],
                    )

        self._log_dual_support()
        self._dual_support_step += 1

        if self.task == "classification":
            features = self.global_pooling(features, batch.in_dict.lengths[-1])
        elif self.task == "cloud_segmentation":
            for stage in range(self.num_layers - 1, 0, -1):
                layer = stage - 1
                upsample = getattr(self, "upsampling_{:d}".format(stage))
                if self.grid_pool:
                    features = upsample(features, batch.in_dict.upsamples[layer])
                else:
                    features = upsample(
                        features,
                        batch.in_dict.upsamples[layer],
                        batch.in_dict.up_distances[layer],
                    )
                features = torch.cat([features, skip_features[layer]], dim=1)
                features = getattr(self, "decoder_unary_{:d}".format(stage))(
                    features
                )
                if self.add_decoder_layer:
                    block = getattr(self, "decoder_layer_{:d}".format(stage))
                    if self.kp_mode in ["kpconv", "kpconvtest"]:
                        features = block(
                            batch.in_dict.points[layer],
                            batch.in_dict.points[layer],
                            features,
                            base_neighbors[layer],
                        )
                    else:
                        features, _ = block(
                            batch.in_dict.points[layer],
                            batch.in_dict.points[layer],
                            features,
                            base_neighbors[layer],
                            batch.in_dict.lengths[layer],
                        )

        logits = self.head(features)
        if verbose:
            torch.cuda.synchronize(batch.device())
            timings.append(time.time())
            print(" " * 75 + "net (ms): {:.1f}".format(1000 * (timings[-1] - timings[0])))
        return logits
