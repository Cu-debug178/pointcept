"""Density-adaptive radius utilities for the Standalone V13 experiment."""

import torch
from torch import Tensor


class DensityAdaptiveRadius(torch.nn.Module):
    """Map local neighbor density to a detached per-query radius scale."""

    def __init__(
        self,
        scale_range=(0.8, 1.5),
        density_k=16,
        norm="percentile",
        percentile=(10, 90),
        strength=0.75,
        power=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.s_min = float(scale_range[0])
        self.s_max = float(scale_range[1])
        self.density_k = int(density_k) if density_k is not None else None
        self.norm = str(norm)
        self.percentile = tuple(percentile)
        self.strength = float(strength)
        self.power = float(power)
        self.eps = float(eps)

        if self.s_min <= 0 or self.s_max <= 0 or self.s_min > self.s_max:
            raise ValueError("Invalid DA-Radius scale_range: {}".format(scale_range))

    @staticmethod
    def _sanitize_neighbors(neighbors: Tensor, num_points: int):
        valid = (neighbors >= 0) & (neighbors < num_points)
        safe = neighbors.clamp(min=0, max=max(num_points - 1, 0))
        return safe, valid

    def _normalize_density(self, density: Tensor, lengths: Tensor):
        normalized = torch.empty_like(density)

        def normalize_one(values):
            if values.numel() == 0:
                return values
            if self.norm == "percentile" and values.numel() > 1:
                q = values.new_tensor(self.percentile) / 100.0
                lo, hi = torch.quantile(values.flatten(), q)
            else:
                lo = values.min()
                hi = values.max()
            return ((values - lo) / (hi - lo + self.eps)).clamp(0.0, 1.0)

        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            if end > start:
                normalized[start:end] = normalize_one(density[start:end])
            start = end
        return normalized

    @torch.no_grad()
    def forward(
        self,
        points: Tensor,
        neighbors: Tensor,
        lengths: Tensor,
        scale_range=None,
    ) -> Tensor:
        if points.numel() == 0:
            return points.new_zeros((0, 1))

        s_min = self.s_min if scale_range is None else float(scale_range[0])
        s_max = self.s_max if scale_range is None else float(scale_range[1])
        if s_min <= 0 or s_max <= 0 or s_min > s_max:
            raise ValueError("Invalid DA-Radius scale_range: {}".format(scale_range))

        if self.density_k is not None and neighbors.shape[1] > self.density_k:
            neighbors = neighbors[:, : self.density_k]

        safe_neighbors, valid = self._sanitize_neighbors(neighbors, points.shape[0])
        neighbor_points = points[safe_neighbors]
        distances = torch.norm(neighbor_points - points.unsqueeze(1), dim=-1)

        valid = valid & (distances > self.eps)
        valid_float = valid.to(dtype=distances.dtype)
        valid_count = valid_float.sum(dim=1, keepdim=True)
        mean_distance = (
            (distances * valid_float).sum(dim=1, keepdim=True)
            / valid_count.clamp(min=1.0)
        )

        no_neighbor = valid_count <= 0
        if no_neighbor.any():
            if (~no_neighbor).any():
                fallback = mean_distance[~no_neighbor].mean()
            else:
                fallback = points.new_tensor(1.0)
            mean_distance = torch.where(
                no_neighbor, fallback.expand_as(mean_distance), mean_distance
            )

        density = 1.0 / (mean_distance + self.eps)
        density_norm = self._normalize_density(density, lengths)
        sparse_score = (1.0 - density_norm).clamp(0.0, 1.0).pow(self.power)
        raw_scale = s_min + (s_max - s_min) * sparse_score
        scale = 1.0 + self.strength * (raw_scale - 1.0)
        return scale.clamp(min=s_min, max=s_max).detach()


@torch.no_grad()
def mask_neighbors_by_radius(
    points: Tensor,
    neighbors: Tensor,
    radius_scale: Tensor,
    base_radius: float,
) -> Tensor:
    """Replace neighbors outside each query radius with the shadow index."""

    if points.numel() == 0:
        return neighbors.clone()
    if neighbors.shape[0] != points.shape[0]:
        raise ValueError("DA-Radius currently requires in-place neighborhoods")
    if radius_scale.numel() != points.shape[0]:
        raise ValueError("radius_scale must contain one value per query point")

    num_points = int(points.shape[0])
    valid = (neighbors >= 0) & (neighbors < num_points)
    safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))
    neighbor_points = points[safe_neighbors]
    sq_distances = torch.sum((neighbor_points - points.unsqueeze(1)) ** 2, dim=-1)
    effective_radius = float(base_radius) * radius_scale.reshape(-1)
    keep = valid & (sq_distances <= effective_radius.square().unsqueeze(1))

    masked = neighbors.clone()
    masked.masked_fill_(~keep, num_points)
    return masked
