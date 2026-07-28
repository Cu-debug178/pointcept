"""Graph utilities for the V18 dual-support diagnostic experiment."""

import torch
from torch import Tensor


def _safe_neighbors(points: Tensor, neighbors: Tensor):
    num_points = int(points.shape[0])
    valid = (neighbors >= 0) & (neighbors < num_points)
    safe = neighbors.clamp(min=0, max=max(num_points - 1, 0))
    return safe, valid


class DimensionlessSpacingScale(torch.nn.Module):
    """Map mean KNN spacing divided by grid size to a radius scale."""

    def __init__(
        self,
        density_k=16,
        percentile=(10, 90),
        strength=1.0,
        power=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.density_k = int(density_k)
        self.percentile = tuple(percentile)
        self.strength = float(strength)
        self.power = float(power)
        self.eps = float(eps)

    def _fragment_normalize(self, spacing: Tensor, lengths: Tensor):
        normalized = torch.empty_like(spacing)
        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            values = spacing[start:end]
            if values.numel() > 1:
                q = values.new_tensor(self.percentile) / 100.0
                lo, hi = torch.quantile(values.flatten(), q)
                normalized[start:end] = (
                    (values - lo) / (hi - lo + self.eps)
                ).clamp(0.0, 1.0)
            elif values.numel() == 1:
                normalized[start:end] = 0.5
            start = end
        return normalized

    @torch.no_grad()
    def forward(
        self,
        points: Tensor,
        neighbors: Tensor,
        lengths: Tensor,
        grid_size: float,
        scale_range,
        fixed_bounds=None,
    ):
        if points.numel() == 0:
            empty = points.new_zeros((0, 1))
            return empty, empty
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")

        used_neighbors = neighbors[:, : self.density_k]
        safe, valid = _safe_neighbors(points, used_neighbors)
        relative = points[safe] - points.unsqueeze(1)
        distances = torch.norm(relative, dim=-1)
        valid = valid & (distances > self.eps)
        valid_float = valid.to(dtype=distances.dtype)
        valid_count = valid_float.sum(dim=1, keepdim=True)
        mean_distance = (
            (distances * valid_float).sum(dim=1, keepdim=True)
            / valid_count.clamp(min=1.0)
        )

        missing = valid_count <= 0
        if missing.any():
            fallback = (
                mean_distance[~missing].mean()
                if (~missing).any()
                else points.new_tensor(float(grid_size))
            )
            mean_distance = torch.where(
                missing, fallback.expand_as(mean_distance), mean_distance
            )

        spacing = mean_distance / float(grid_size)
        if fixed_bounds is None:
            spacing_norm = self._fragment_normalize(spacing, lengths)
        else:
            lo, hi = float(fixed_bounds[0]), float(fixed_bounds[1])
            if hi <= lo:
                raise ValueError("fixed spacing bounds must satisfy high > low")
            spacing_norm = ((spacing - lo) / (hi - lo)).clamp(0.0, 1.0)

        s_min, s_max = float(scale_range[0]), float(scale_range[1])
        if s_min <= 0 or s_max < s_min:
            raise ValueError("invalid dual-support scale range")
        raw_scale = s_min + (s_max - s_min) * spacing_norm.pow(self.power)
        scale = 1.0 + self.strength * (raw_scale - 1.0)
        return scale.clamp(min=s_min, max=s_max).detach(), spacing.detach()


@torch.no_grad()
def build_ring_neighbors(
    points: Tensor,
    candidate_neighbors: Tensor,
    radius_scale: Tensor,
    base_radius: float,
    base_limit: int,
    ring_limit: int,
):
    """Select new neighbors in ``base_radius < d <= adaptive_radius``.

    The first ``base_limit`` candidate slots belong exclusively to the identity
    path. The returned ring can only use later slots, so the two supports are
    disjoint by construction when the candidate KNN list has unique indices.
    """

    if candidate_neighbors.shape[0] != points.shape[0]:
        raise ValueError("candidate graph must have one row per point")
    candidate_k = int(candidate_neighbors.shape[1])
    base_limit = int(base_limit)
    ring_limit = int(ring_limit)
    if base_limit <= 0 or base_limit >= candidate_k:
        raise ValueError("candidate_k must be greater than base_limit")
    if ring_limit <= 0 or ring_limit > candidate_k - base_limit:
        raise ValueError("ring_limit must fit in the extra candidate slots")
    if radius_scale.numel() != points.shape[0]:
        raise ValueError("radius_scale must contain one value per point")

    safe, valid = _safe_neighbors(points, candidate_neighbors)
    relative = points[safe] - points.unsqueeze(1)
    distances = torch.norm(relative, dim=-1)
    adaptive_radius = float(base_radius) * radius_scale.reshape(-1, 1)

    slot_ids = torch.arange(candidate_k, device=points.device).unsqueeze(0)
    extra_slot = slot_ids >= base_limit
    ring_mask = (
        valid
        & extra_slot
        & (distances > float(base_radius))
        & (distances <= adaptive_radius)
    )

    inf = torch.full_like(distances, float("inf"))
    ring_distances = torch.where(ring_mask, distances, inf)
    selected_distances, selected_slots = torch.topk(
        ring_distances,
        k=ring_limit,
        dim=1,
        largest=False,
        sorted=True,
    )
    selected = torch.gather(candidate_neighbors, 1, selected_slots)
    selected_valid = torch.isfinite(selected_distances)
    selected = selected.clone()
    selected.masked_fill_(~selected_valid, points.shape[0])

    base_knn = candidate_neighbors[:, :base_limit]
    base_valid = (base_knn >= 0) & (base_knn < points.shape[0])
    base_valid_count = base_valid.sum(dim=1).float()
    base_radius_count = (valid & (distances <= float(base_radius))).sum(dim=1).float()
    adaptive_count = (valid & (distances <= adaptive_radius)).sum(dim=1).float()
    added_count = selected_valid.sum(dim=1).float()
    union_count = base_valid_count + added_count
    change_rate = added_count / union_count.clamp(min=1.0)

    h_distance = distances[:, base_limit - 1]
    h_valid = valid[:, base_limit - 1]
    h_ratio = torch.where(
        h_valid,
        h_distance / max(float(base_radius), 1e-12),
        torch.full_like(h_distance, float("nan")),
    )

    stats = {
        "base_valid_count": base_valid_count,
        "base_radius_count": base_radius_count,
        "adaptive_radius_count": adaptive_count,
        "added_count": added_count,
        "change_rate": change_rate,
        "full_base_radius": base_radius_count >= base_limit,
        "d_h_over_radius": h_ratio,
        "spacing_candidate_distance": distances,
    }
    return selected, stats
