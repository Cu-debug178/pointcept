import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "pointcept"
    / "models"
    / "kpconvx_soka"
    / "soka_lite_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("soka_geometry", MODULE_PATH)
SOKA_LITE_GEOMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOKA_LITE_GEOMETRY)
build_soka_lite_descriptor = SOKA_LITE_GEOMETRY.build_soka_lite_descriptor


def _example_geometry():
    relative_neighbors = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1e6, 1e6, 1e6]],
            [[1e6, 1e6, 1e6], [1e6, 1e6, 1e6], [1e6, 1e6, 1e6], [1e6, 1e6, 1e6]],
        ],
        dtype=torch.float32,
    )
    neighbors_1nn = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 0]])
    nn_sq_dists = torch.tensor([[0.0, 0.25, 1.0, 1e12], [1e12, 1e12, 1e12, 1e12]])
    valid_mask = torch.tensor(
        [[True, True, True, False], [False, False, False, False]]
    )
    kernel_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    return (
        relative_neighbors,
        neighbors_1nn,
        nn_sq_dists,
        valid_mask,
        kernel_points,
    )


def test_shadow_neighbors_are_excluded_and_empty_queries_are_finite():
    descriptor, aux = build_soka_lite_descriptor(*_example_geometry(), radius=2.0, sigma=1.0)

    assert descriptor.shape == (2, 3, 6)
    assert torch.equal(aux["counts"][0], torch.ones(3))
    assert torch.equal(aux["counts"][1], torch.zeros(3))
    assert torch.equal(aux["valid_counts"], torch.tensor([3.0, 0.0]))
    assert torch.allclose(descriptor[0, :, 0], torch.full((3,), 1.0 / 3.0))
    assert torch.all(descriptor[1, :, :5] == 0)
    assert torch.isfinite(descriptor).all()


def test_occupancy_count_is_conserved():
    _, aux = build_soka_lite_descriptor(*_example_geometry(), radius=2.0, sigma=1.0)
    assert torch.equal(aux["counts"].sum(dim=1), aux["valid_counts"])


def test_descriptor_is_invariant_to_neighbor_permutation():
    inputs = _example_geometry()
    descriptor, aux = build_soka_lite_descriptor(*inputs, radius=2.0, sigma=1.0)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = tuple(tensor[:, permutation] for tensor in inputs[:4]) + (inputs[4],)
    descriptor_permuted, aux_permuted = build_soka_lite_descriptor(
        *permuted, radius=2.0, sigma=1.0
    )

    assert torch.allclose(descriptor, descriptor_permuted, atol=1e-6, rtol=0)
    assert torch.equal(aux["counts"], aux_permuted["counts"])


def test_assignment_variance_is_clamped_and_empty_cells_are_zero():
    descriptor, _ = build_soka_lite_descriptor(*_example_geometry(), radius=2.0, sigma=1.0)
    assert torch.all(descriptor[..., 3] >= 0)
    assert torch.isfinite(descriptor[..., 3]).all()


def test_invalid_valid_assignment_is_rejected():
    inputs = list(_example_geometry())
    inputs[1] = inputs[1].clone()
    inputs[1][0, 0] = 3
    try:
        build_soka_lite_descriptor(
            *inputs,
            radius=2.0,
            sigma=1.0,
            validate_assignments=True,
        )
    except ValueError as error:
        assert "assignments" in str(error)
    else:
        raise AssertionError("invalid assignment was not rejected")
