import torch
import torch.nn as nn


class DensityAdaptiveScale(nn.Module):
    """
    Density-to-scale mapper for DA-KPConvX.

    rho_i = 1 / (mean KNN distance + eps)

    Normalized linear mapping:

        s_i = s_min + (s_max - s_min) * (1 - norm(rho_i))

    Dense area:
        rho high -> s_i close to s_min

    Sparse area:
        rho low -> s_i close to s_max
    """

    def __init__(self, scale_range=(0.5, 2.0), density_k=16, eps=1e-6):
        super().__init__()
        self.s_min = float(scale_range[0])
        self.s_max = float(scale_range[1])
        self.density_k = int(density_k) if density_k is not None else None
        self.eps = float(eps)

        if self.s_min <= 0 or self.s_max <= 0 or self.s_min > self.s_max:
            raise ValueError(f"Invalid scale_range: {scale_range}")

    @staticmethod
    def _sanitize_neighbors(neighbors, num_points):
        valid = (neighbors >= 0) & (neighbors < num_points)
        safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))
        return safe_neighbors, valid.float()

    def _map_density_to_scale(self, rho, lengths=None):
        scale = torch.empty_like(rho)

        if lengths is None:
            rho_min = rho.min()
            rho_max = rho.max()
            rho_norm = (rho - rho_min) / (rho_max - rho_min + self.eps)
            scale = self.s_min + (self.s_max - self.s_min) * (1.0 - rho_norm)
            return scale.clamp(min=self.s_min, max=self.s_max)

        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            if end <= start:
                start = end
                continue

            rho_b = rho[start:end]
            rho_min = rho_b.min()
            rho_max = rho_b.max()
            rho_norm = (rho_b - rho_min) / (rho_max - rho_min + self.eps)

            scale[start:end] = self.s_min + (self.s_max - self.s_min) * (1.0 - rho_norm)
            start = end

        return scale.clamp(min=self.s_min, max=self.s_max)

    @torch.no_grad()
    def forward(self, points, neighbors, lengths=None):
        if points.numel() == 0:
            return points.new_zeros((0, 1))

        if self.density_k is not None and neighbors.shape[1] > self.density_k:
            neighbors = neighbors[:, : self.density_k]

        safe_neighbors, valid = self._sanitize_neighbors(neighbors, points.shape[0])

        neigh_points = points[safe_neighbors]
        dist = torch.norm(neigh_points - points.unsqueeze(1), dim=-1)

        # Remove self-neighbor with zero distance.
        self_mask = dist <= self.eps
        valid = valid * (~self_mask).float()

        valid_count = valid.sum(dim=1, keepdim=True)
        mean_dist = (dist * valid).sum(dim=1, keepdim=True) / valid_count.clamp(min=1.0)

        # If a point has no valid non-self neighbors, use a safe fallback.
        no_neighbor = valid_count <= 0
        if no_neighbor.any():
            if (~no_neighbor).any():
                fallback_dist = mean_dist[~no_neighbor].mean()
            else:
                fallback_dist = points.new_tensor(1.0)

            mean_dist = torch.where(
                no_neighbor,
                fallback_dist.expand_as(mean_dist),
                mean_dist,
            )

        rho = 1.0 / (mean_dist + self.eps)

        da_scale = self._map_density_to_scale(rho, lengths=lengths)
        return da_scale.detach()


class DensityAdaptiveRadius(nn.Module):
    """
    Density-to-radius mapper for DA-Radius.

    This module is intentionally independent from DA-Kernel. It estimates local
    density from the available candidate neighbors and returns per-query radius
    scales used to mask or search neighbors.
    """

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
        self.norm = norm
        self.percentile = tuple(percentile)
        self.strength = float(strength)
        self.power = float(power)
        self.eps = float(eps)

        if self.s_min <= 0 or self.s_max <= 0 or self.s_min > self.s_max:
            raise ValueError(f"Invalid DA-Radius scale_range: {scale_range}")

    @staticmethod
    def _sanitize_neighbors(neighbors, num_points):
        valid = (neighbors >= 0) & (neighbors < num_points)
        safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))
        return safe_neighbors, valid.float()

    def _normalize_density(self, rho, lengths=None):
        rho_norm = torch.empty_like(rho)

        def normalize_one(rho_b):
            if rho_b.numel() == 0:
                return rho_b
            if self.norm == "percentile" and rho_b.numel() > 1:
                q = torch.tensor(
                    self.percentile,
                    dtype=rho_b.dtype,
                    device=rho_b.device,
                ) / 100.0
                lo, hi = torch.quantile(rho_b.flatten(), q)
            else:
                lo = rho_b.min()
                hi = rho_b.max()
            return ((rho_b - lo) / (hi - lo + self.eps)).clamp(0.0, 1.0)

        if lengths is None:
            return normalize_one(rho)

        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            if end > start:
                rho_norm[start:end] = normalize_one(rho[start:end])
            start = end
        return rho_norm

    @torch.no_grad()
    def forward(self, points, neighbors, lengths=None, scale_range=None, return_meta=False):
        if points.numel() == 0:
            scale = points.new_zeros((0, 1))
            if not return_meta:
                return scale
            meta = dict(
                feat=points.new_zeros((0, 4), dtype=torch.float32),
                scale=scale.to(torch.float32),
                rho_norm=scale.to(torch.float32),
                valid_ratio=scale.to(torch.float32),
                dist_cv=scale.to(torch.float32),
            )
            return scale, meta

        s_min = self.s_min if scale_range is None else float(scale_range[0])
        s_max = self.s_max if scale_range is None else float(scale_range[1])

        if self.density_k is not None and neighbors.shape[1] > self.density_k:
            neighbors = neighbors[:, : self.density_k]

        safe_neighbors, valid = self._sanitize_neighbors(neighbors, points.shape[0])
        neigh_points = points[safe_neighbors]
        dist = torch.norm(neigh_points - points.unsqueeze(1), dim=-1)

        self_mask = dist <= self.eps
        valid = valid * (~self_mask).float()

        used_k = max(int(neighbors.shape[1]), 1)
        valid_count = valid.sum(dim=1, keepdim=True)
        mean_dist = (dist * valid).sum(dim=1, keepdim=True) / valid_count.clamp(min=1.0)

        no_neighbor = valid_count <= 0
        if no_neighbor.any():
            if (~no_neighbor).any():
                fallback_dist = mean_dist[~no_neighbor].mean()
            else:
                fallback_dist = points.new_tensor(1.0)
            mean_dist = torch.where(no_neighbor, fallback_dist.expand_as(mean_dist), mean_dist)

        rho = 1.0 / (mean_dist + self.eps)
        rho_norm = self._normalize_density(rho, lengths=lengths)
        sparse_score = (1.0 - rho_norm).clamp(0.0, 1.0).pow(self.power)
        raw_scale = s_min + (s_max - s_min) * sparse_score
        scale = 1.0 + self.strength * (raw_scale - 1.0)
        scale = scale.clamp(min=s_min, max=s_max).detach()

        if not return_meta:
            return scale

        var_dist = ((dist - mean_dist) ** 2 * valid).sum(
            dim=1, keepdim=True
        ) / valid_count.clamp(min=1.0)
        std_dist = torch.sqrt(var_dist + self.eps)
        dist_cv = (std_dist / (mean_dist + self.eps)).clamp(0.0, 2.0) / 2.0
        valid_ratio = (valid_count / float(used_k)).clamp(0.0, 1.0)
        feat = torch.cat(
            [
                scale - 1.0,
                rho_norm.detach(),
                valid_ratio.detach(),
                dist_cv.detach(),
            ],
            dim=1,
        ).to(torch.float32)
        meta = dict(
            feat=feat.detach(),
            scale=scale.to(torch.float32),
            rho_norm=rho_norm.detach().to(torch.float32),
            valid_ratio=valid_ratio.detach().to(torch.float32),
            dist_cv=dist_cv.detach().to(torch.float32),
        )
        return scale, meta
