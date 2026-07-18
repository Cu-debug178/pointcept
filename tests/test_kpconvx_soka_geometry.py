import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "pointcept"
    / "models"
    / "kpconvx_soka"
    / "soka_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("soka_geometry", MODULE_PATH)
SOKA_GEOMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOKA_GEOMETRY)
build_kernel_cell_geometry = SOKA_GEOMETRY.build_kernel_cell_geometry
kernel_cell_geometry_dim = SOKA_GEOMETRY.kernel_cell_geometry_dim
scatter_kernel_cell_mean = SOKA_GEOMETRY.scatter_kernel_cell_mean


def _example_geometry():
    relative_neighbors = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1e6, 1e6, 1e6]],
            [[1e6, 1e6, 1e6], [1e6, 1e6, 1e6], [1e6, 1e6, 1e6], [1e6, 1e6, 1e6]],
        ],
        dtype=torch.float32,
    )
    cell_ids = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 0]])
    nn_sq_dists = torch.tensor(
        [[0.0, 0.25, 1.0, 1e12], [1e12, 1e12, 1e12, 1e12]]
    )
    valid_mask = torch.tensor(
        [[True, True, True, False], [False, False, False, False]]
    )
    kernel_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    return relative_neighbors, cell_ids, nn_sq_dists, valid_mask, kernel_points


def test_geometry_is_direction_aware_and_empty_queries_are_finite():
    descriptor, aux = build_kernel_cell_geometry(
        *_example_geometry(), radius=2.0, sigma=1.0
    )

    assert descriptor.shape == (2, 3, kernel_cell_geometry_dim(3))
    assert torch.equal(aux["counts"][0], torch.ones(3))
    assert torch.equal(aux["counts"][1], torch.zeros(3))
    assert torch.equal(aux["valid_counts"], torch.tensor([3.0, 0.0]))
    assert torch.allclose(descriptor[0, :, 1], torch.full((3,), 1.0 / 3.0))
    assert torch.equal(descriptor[0, :, 2], torch.ones(3))
    assert torch.allclose(
        descriptor[0, :, 3:6],
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert torch.allclose(descriptor[0, :, 6], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.all(descriptor[1, :, :-3] == 0)
    assert torch.isfinite(descriptor).all()


def test_geometry_count_is_conserved_and_singletons_have_zero_variance():
    descriptor, aux = build_kernel_cell_geometry(
        *_example_geometry(), radius=2.0, sigma=1.0
    )
    assert torch.equal(aux["counts"].sum(dim=1), aux["valid_counts"])
    assert torch.all(descriptor[0, :, 7] == 0)
    assert torch.all(descriptor[..., 7] >= 0)


def test_geometry_is_invariant_to_neighbor_permutation():
    inputs = _example_geometry()
    descriptor, aux = build_kernel_cell_geometry(*inputs, radius=2.0, sigma=1.0)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = tuple(tensor[:, permutation] for tensor in inputs[:4]) + (inputs[4],)
    descriptor_permuted, aux_permuted = build_kernel_cell_geometry(
        *permuted, radius=2.0, sigma=1.0
    )

    assert torch.allclose(descriptor, descriptor_permuted, atol=1e-6, rtol=0)
    assert torch.equal(aux["counts"], aux_permuted["counts"])


def test_topology_scatter_mean_is_differentiable_and_shadow_safe():
    torch.manual_seed(9)
    values = torch.randn(2, 4, 5, requires_grad=True)
    cell_ids = torch.tensor([[0, 1, 0, 2], [2, 1, 0, 0]])
    valid_mask = torch.tensor(
        [[True, True, True, False], [True, False, False, False]]
    )
    means, counts = scatter_kernel_cell_mean(
        values, cell_ids, valid_mask, num_cells=3
    )

    assert means.shape == (2, 3, 5)
    assert torch.equal(counts, torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    assert torch.all(means[counts == 0] == 0)
    means.square().sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert values.grad[valid_mask].abs().sum() > 0
    assert torch.all(values.grad[~valid_mask] == 0)


def test_invalid_valid_assignment_is_rejected():
    inputs = list(_example_geometry())
    inputs[1] = inputs[1].clone()
    inputs[1][0, 0] = 3
    try:
        build_kernel_cell_geometry(
            *inputs,
            radius=2.0,
            sigma=1.0,
            validate_assignments=True,
        )
    except ValueError as error:
        assert "assignments" in str(error)
    else:
        raise AssertionError("invalid assignment was not rejected")
