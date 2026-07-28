"""Standalone-native V13 density-adaptive support for KPConvX."""

import torch

from models.KPNext import KPNeXt
from utils.da_radius import DensityAdaptiveRadius, mask_neighbors_by_radius
from utils.torch_pyramid import fill_pyramid


class KPNeXtV13(KPNeXt):
    """KPNeXt with V13 per-query radius filtering on selected encoder stages.

    The official Standalone pyramid already stores the nearest H candidates for
    every query. V13 filters that ordered candidate graph with a per-query
    radius and uses the regular shadow index for rejected slots. The official
    KPNeXt implementation and its default path remain unchanged.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        model_cfg = cfg.model
        self.enable_da_radius = bool(
            getattr(model_cfg, "enable_da_radius", False)
        )
        self.da_radius_stages = tuple(
            int(stage) for stage in getattr(model_cfg, "da_radius_stages", (3, 4))
        )
        self.da_radius_scale_range = tuple(
            getattr(model_cfg, "da_radius_scale_range", (0.95, 1.25))
        )
        self.da_radius_stage_ranges = self._normalize_stage_ranges(
            getattr(model_cfg, "da_radius_stage_ranges", None)
        )
        self.da_radius_debug = bool(getattr(model_cfg, "da_radius_debug", False))
        self.da_radius_debug_interval = max(
            int(getattr(model_cfg, "da_radius_debug_interval", 100)), 1
        )
        self._da_radius_step = 0
        self.last_da_radius_stats = {}

        for stage in self.da_radius_stages:
            if stage < 1 or stage > self.num_layers:
                raise ValueError("Invalid DA-Radius stage: {}".format(stage))

        self.da_radius = DensityAdaptiveRadius(
            scale_range=self.da_radius_scale_range,
            density_k=getattr(model_cfg, "da_radius_density_k", 16),
            norm=getattr(model_cfg, "da_radius_norm", "percentile"),
            percentile=getattr(model_cfg, "da_radius_percentile", (10, 90)),
            strength=getattr(model_cfg, "da_radius_strength", 0.5),
            power=getattr(model_cfg, "da_radius_power", 1.0),
        )

    @staticmethod
    def _normalize_stage_ranges(stage_ranges):
        if stage_ranges is None:
            return {}
        return {
            int(stage): tuple(scale_range)
            for stage, scale_range in dict(stage_ranges).items()
        }

    def _stage_scale_range(self, stage):
        return self.da_radius_stage_ranges.get(
            int(stage), self.da_radius_scale_range
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
            self.neighbor_limits,
            self.upsample_n,
            sub_mode=self.in_sub_mode,
            grid_pool_mode=self.grid_pool,
        )

    @torch.no_grad()
    def _adaptive_neighbors(self, in_dict):
        adapted = list(in_dict.neighbors)
        stats = {}

        for stage in self.da_radius_stages:
            layer = stage - 1
            points = in_dict.points[layer]
            neighbors = in_dict.neighbors[layer]
            lengths = in_dict.lengths[layer]
            if points.shape[0] == 0:
                adapted[layer] = neighbors.clone()
                stats[stage] = {
                    "scale_min": 1.0,
                    "scale_mean": 1.0,
                    "scale_max": 1.0,
                    "valid_ratio": 0.0,
                    "full_ratio": 0.0,
                }
                continue
            scale = self.da_radius(
                points,
                neighbors,
                lengths,
                scale_range=self._stage_scale_range(stage),
            )
            layer_radius = self.first_radius * (self.radius_scaling ** layer)
            masked = mask_neighbors_by_radius(
                points,
                neighbors,
                scale,
                base_radius=layer_radius,
            )
            adapted[layer] = masked

            valid_ratio = (masked < points.shape[0]).float().mean()
            full_ratio = (
                (masked < points.shape[0]).sum(dim=1) >= masked.shape[1]
            ).float().mean()
            if scale.numel() > 0:
                scale_min = float(scale.min().item())
                scale_mean = float(scale.mean().item())
                scale_max = float(scale.max().item())
            else:
                scale_min = scale_mean = scale_max = 1.0
            stats[stage] = {
                "scale_min": scale_min,
                "scale_mean": scale_mean,
                "scale_max": scale_max,
                "valid_ratio": float(valid_ratio.item()),
                "full_ratio": float(full_ratio.item()),
            }

        self.last_da_radius_stats = stats
        if self.da_radius_debug and self._da_radius_step % self.da_radius_debug_interval == 0:
            for stage, stage_stats in stats.items():
                print(
                    "DA-Radius step={} stage={} scale=[{:.4f}, {:.4f}, {:.4f}] "
                    "valid_ratio={:.4f} full_ratio={:.4f}".format(
                        self._da_radius_step,
                        stage,
                        stage_stats["scale_min"],
                        stage_stats["scale_mean"],
                        stage_stats["scale_max"],
                        stage_stats["valid_ratio"],
                        stage_stats["full_ratio"],
                    )
                )
        self._da_radius_step += 1
        return adapted

    def forward(self, batch, verbose=False):
        if not self.enable_da_radius:
            return super().forward(batch, verbose=verbose)

        self._ensure_pyramid(batch.in_dict)
        baseline_neighbors = batch.in_dict.neighbors
        batch.in_dict.neighbors = self._adaptive_neighbors(batch.in_dict)
        try:
            return super().forward(batch, verbose=verbose)
        finally:
            batch.in_dict.neighbors = baseline_neighbors
