import math

import torch
from torch import Tensor


def _stats_dtype(tensor: Tensor):
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


@torch.no_grad()
def build_soka_lite_descriptor(
    relative_neighbors: Tensor,
    neighbors_1nn: Tensor,
    nn_sq_dists: Tensor,
    valid_mask: Tensor,
    kernel_points: Tensor,
    radius: float,
    sigma: float,
    eps: float = 1.0e-6,
    validate_assignments: bool = False,
):
    """Build the original six-dimensional SOKA-Lite descriptor."""
    if relative_neighbors.ndim != 3:
        raise ValueError("relative_neighbors must have shape [M, H, D]")
    if neighbors_1nn.shape != relative_neighbors.shape[:2]:
        raise ValueError("neighbors_1nn must have shape [M, H]")
    if nn_sq_dists.shape != relative_neighbors.shape[:2]:
        raise ValueError("nn_sq_dists must have shape [M, H]")
    if valid_mask.shape != relative_neighbors.shape[:2]:
        raise ValueError("valid_mask must have shape [M, H]")
    if kernel_points.ndim != 2:
        raise ValueError("kernel_points must have shape [K, D]")
    if kernel_points.shape[1] != relative_neighbors.shape[2]:
        raise ValueError("kernel point dimension does not match neighbor dimension")
    if radius <= 0 or sigma <= 0:
        raise ValueError("radius and sigma must be positive")

    valid_mask = valid_mask.bool()
    num_queries, num_neighbors = neighbors_1nn.shape
    num_kernels = int(kernel_points.shape[0])
    if num_kernels < 1:
        raise ValueError("at least one kernel point is required")

    if validate_assignments and valid_mask.any():
        valid_assignments = neighbors_1nn[valid_mask]
        if (valid_assignments < 0).any() or (valid_assignments >= num_kernels).any():
            raise ValueError("valid nearest-kernel assignments must be in [0, K)")

    dtype = _stats_dtype(relative_neighbors)
    device = relative_neighbors.device
    safe_assignments = neighbors_1nn.clamp(0, num_kernels - 1)
    query_offsets = (
        torch.arange(num_queries, device=device, dtype=torch.long).unsqueeze(1)
        * num_kernels
    )
    flat_cells = (query_offsets + safe_assignments).reshape(-1)
    flat_valid = valid_mask.reshape(-1).to(dtype=dtype)

    safe_neighbors = torch.where(
        valid_mask.unsqueeze(-1),
        relative_neighbors,
        torch.zeros_like(relative_neighbors),
    ).to(dtype=dtype)
    radial = torch.linalg.vector_norm(safe_neighbors, dim=-1) / float(radius)
    assignment = torch.sqrt(nn_sq_dists.to(dtype=dtype).clamp_min(0.0))
    assignment = assignment / float(sigma)
    radial = radial * valid_mask.to(dtype=dtype)
    assignment = assignment * valid_mask.to(dtype=dtype)

    flat_size = num_queries * num_kernels

    def scatter(values: Tensor):
        output = torch.zeros(flat_size, device=device, dtype=dtype)
        output.scatter_add_(0, flat_cells, values.reshape(-1))
        return output.reshape(num_queries, num_kernels)

    counts = scatter(flat_valid)
    radial_sum = scatter(radial)
    assignment_sum = scatter(assignment)
    assignment_sq_sum = scatter(assignment.square())

    valid_counts = valid_mask.sum(dim=1).to(dtype=dtype)
    occupancy = counts / valid_counts.clamp_min(1.0).unsqueeze(1)
    nonempty_denominator = counts.clamp_min(1.0)
    mean_radial = radial_sum / nonempty_denominator
    mean_assignment = assignment_sum / nonempty_denominator
    assignment_second_moment = assignment_sq_sum / nonempty_denominator
    assignment_variance = (
        assignment_second_moment - mean_assignment.square()
    ).clamp_min(0.0)

    entropy_denominator = math.log(max(num_kernels, 2))
    entropy = -(
        occupancy * torch.log(occupancy.clamp_min(float(eps)))
    ).sum(dim=1)
    entropy = entropy / entropy_denominator
    entropy = torch.where(valid_counts > 0, entropy, torch.zeros_like(entropy))

    shell_radius = torch.linalg.vector_norm(
        kernel_points.to(device=device, dtype=dtype), dim=1
    ) / float(radius)
    shell_radius = shell_radius.unsqueeze(0).expand(num_queries, -1)

    descriptor = torch.stack(
        (
            occupancy,
            mean_radial,
            mean_assignment,
            assignment_variance,
            entropy.unsqueeze(1).expand(-1, num_kernels),
            shell_radius,
        ),
        dim=-1,
    )
    auxiliary = {
        "counts": counts,
        "valid_counts": valid_counts,
        "occupied_mask": counts > 0,
        "occupancy": occupancy,
        "mean_radial": mean_radial,
        "mean_assignment": mean_assignment,
        "assignment_variance": assignment_variance,
        "entropy": entropy,
    }
    return descriptor, auxiliary
