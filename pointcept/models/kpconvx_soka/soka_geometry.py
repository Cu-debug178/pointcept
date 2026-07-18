import math

import torch
from torch import Tensor


def _stats_dtype(tensor: Tensor):
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


def kernel_cell_geometry_dim(dimension: int):
    if int(dimension) < 1:
        raise ValueError("dimension must be positive")
    return 6 + 2 * int(dimension)


def _validate_cell_inputs(
    values: Tensor,
    cell_ids: Tensor,
    valid_mask: Tensor,
    num_cells: int,
):
    if values.ndim != 3:
        raise ValueError("values must have shape [M, H, D]")
    if cell_ids.shape != values.shape[:2]:
        raise ValueError("cell_ids must have shape [M, H]")
    if valid_mask.shape != values.shape[:2]:
        raise ValueError("valid_mask must have shape [M, H]")
    if int(num_cells) < 1:
        raise ValueError("num_cells must be positive")


def scatter_kernel_cell_sum(
    values: Tensor,
    cell_ids: Tensor,
    valid_mask: Tensor,
    num_cells: int,
):
    """Scatter neighbor values into kernel cells without a dense M x H x K tensor."""
    _validate_cell_inputs(values, cell_ids, valid_mask, num_cells)
    num_queries, _, value_dim = values.shape
    work_dtype = _stats_dtype(values)
    safe_ids = cell_ids.clamp(0, int(num_cells) - 1)
    query_offsets = (
        torch.arange(num_queries, device=cell_ids.device, dtype=torch.long).unsqueeze(1)
        * int(num_cells)
    )
    flat_ids = (query_offsets + safe_ids).reshape(-1)
    masked_values = values.to(dtype=work_dtype) * valid_mask.unsqueeze(-1).to(
        dtype=work_dtype
    )
    output = torch.zeros(
        num_queries * int(num_cells),
        value_dim,
        device=values.device,
        dtype=work_dtype,
    )
    output.scatter_add_(
        0,
        flat_ids.unsqueeze(1).expand(-1, value_dim),
        masked_values.reshape(-1, value_dim),
    )
    return output.reshape(num_queries, int(num_cells), value_dim)


def scatter_kernel_cell_mean(
    values: Tensor,
    cell_ids: Tensor,
    valid_mask: Tensor,
    num_cells: int,
):
    sums = scatter_kernel_cell_sum(values, cell_ids, valid_mask, num_cells)
    ones = torch.ones(
        *values.shape[:2],
        1,
        device=values.device,
        dtype=values.dtype,
    )
    counts = scatter_kernel_cell_sum(ones, cell_ids, valid_mask, num_cells)
    means = sums / counts.clamp_min(1.0)
    return means.to(dtype=values.dtype), counts.squeeze(-1)


@torch.no_grad()
def build_kernel_cell_geometry(
    relative_neighbors: Tensor,
    cell_ids: Tensor,
    nn_sq_dists: Tensor,
    valid_mask: Tensor,
    kernel_points: Tensor,
    radius: float,
    sigma: float,
    validate_assignments: bool = False,
):
    """Build direction-aware geometric evidence for each query and kernel cell."""
    if relative_neighbors.ndim != 3:
        raise ValueError("relative_neighbors must have shape [M, H, D]")
    if cell_ids.shape != relative_neighbors.shape[:2]:
        raise ValueError("cell_ids must have shape [M, H]")
    if nn_sq_dists.shape != relative_neighbors.shape[:2]:
        raise ValueError("nn_sq_dists must have shape [M, H]")
    if valid_mask.shape != relative_neighbors.shape[:2]:
        raise ValueError("valid_mask must have shape [M, H]")
    if kernel_points.ndim != 2:
        raise ValueError("kernel_points must have shape [K, D]")
    if kernel_points.shape[1] != relative_neighbors.shape[2]:
        raise ValueError("kernel point dimension does not match neighbor dimension")
    if float(radius) <= 0 or float(sigma) <= 0:
        raise ValueError("radius and sigma must be positive")

    valid_mask = valid_mask.bool()
    num_queries = int(relative_neighbors.shape[0])
    num_cells = int(kernel_points.shape[0])
    if num_cells < 1:
        raise ValueError("at least one kernel point is required")
    if validate_assignments and valid_mask.any():
        valid_ids = cell_ids[valid_mask]
        if (valid_ids < 0).any() or (valid_ids >= num_cells).any():
            raise ValueError("valid assignments must be in [0, K)")

    work_dtype = _stats_dtype(relative_neighbors)
    safe_neighbors = torch.where(
        valid_mask.unsqueeze(-1),
        relative_neighbors,
        torch.zeros_like(relative_neighbors),
    ).to(dtype=work_dtype)
    normalized_neighbors = safe_neighbors / float(radius)
    assignment = torch.sqrt(nn_sq_dists.to(dtype=work_dtype).clamp_min(0.0))
    assignment = assignment / float(sigma)
    assignment = assignment * valid_mask.to(dtype=work_dtype)
    radial_sq = normalized_neighbors.square().sum(dim=-1)

    scalar_values = torch.stack(
        (assignment, assignment.square(), radial_sq),
        dim=-1,
    )
    vector_sums = scatter_kernel_cell_sum(
        normalized_neighbors,
        cell_ids,
        valid_mask,
        num_cells,
    )
    scalar_sums = scatter_kernel_cell_sum(
        scalar_values,
        cell_ids,
        valid_mask,
        num_cells,
    )
    ones = torch.ones(
        *relative_neighbors.shape[:2],
        1,
        device=relative_neighbors.device,
        dtype=work_dtype,
    )
    counts = scatter_kernel_cell_sum(
        ones,
        cell_ids,
        valid_mask,
        num_cells,
    ).squeeze(-1)
    denominator = counts.clamp_min(1.0).unsqueeze(-1)
    mean_relative = vector_sums / denominator
    scalar_means = scalar_sums / denominator
    mean_assignment = scalar_means[..., 0]
    assignment_variance = (
        scalar_means[..., 1] - mean_assignment.square()
    ).clamp_min(0.0)
    mean_radial_sq = scalar_means[..., 2]

    valid_counts = valid_mask.sum(dim=1).to(dtype=work_dtype)
    occupancy = counts / valid_counts.clamp_min(1.0).unsqueeze(1)
    count_scale = torch.log1p(valid_counts).clamp_min(math.log(2.0)).unsqueeze(1)
    log_count = torch.log1p(counts) / count_scale
    occupied_mask = counts > 0
    kernel_coordinates = kernel_points.to(
        device=relative_neighbors.device,
        dtype=work_dtype,
    ) / float(radius)
    kernel_coordinates = kernel_coordinates.unsqueeze(0).expand(num_queries, -1, -1)

    descriptor = torch.cat(
        (
            log_count.unsqueeze(-1),
            occupancy.unsqueeze(-1),
            occupied_mask.to(dtype=work_dtype).unsqueeze(-1),
            mean_relative,
            mean_assignment.unsqueeze(-1),
            assignment_variance.unsqueeze(-1),
            mean_radial_sq.unsqueeze(-1),
            kernel_coordinates,
        ),
        dim=-1,
    )
    expected_dim = kernel_cell_geometry_dim(relative_neighbors.shape[2])
    if descriptor.shape[-1] != expected_dim:
        raise RuntimeError("unexpected kernel-cell geometry dimension")

    auxiliary = {
        "counts": counts,
        "valid_counts": valid_counts,
        "occupied_mask": occupied_mask,
        "occupancy": occupancy,
        "mean_assignment": mean_assignment,
        "assignment_variance": assignment_variance,
        "mean_radial_sq": mean_radial_sq,
    }
    return descriptor, auxiliary
