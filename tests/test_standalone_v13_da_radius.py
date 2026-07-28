import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "ml-kpconvx-standalone"
    / "KPConvX"
    / "utils"
    / "da_radius.py"
)
SPEC = importlib.util.spec_from_file_location("standalone_da_radius", MODULE_PATH)
DA_RADIUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DA_RADIUS)


def test_sparse_points_receive_larger_radius_scale():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
        ]
    )
    neighbors = torch.tensor(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 0],
            [3, 4, 5],
            [4, 3, 5],
            [5, 4, 3],
        ]
    )
    mapper = DA_RADIUS.DensityAdaptiveRadius(
        scale_range=(0.95, 1.20),
        density_k=3,
        strength=0.5,
    )

    scale = mapper(points, neighbors, torch.tensor([6]))

    assert scale[:3].mean() < scale[3:].mean()
    assert scale.min() >= 0.95
    assert scale.max() <= 1.20


def test_radius_mask_uses_global_shadow_index():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    neighbors = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0]])

    masked = DA_RADIUS.mask_neighbors_by_radius(
        points,
        neighbors,
        radius_scale=torch.ones((3, 1)),
        base_radius=0.15,
    )

    shadow = points.shape[0]
    expected = torch.tensor(
        [[0, 1, shadow], [1, 0, shadow], [2, shadow, shadow]]
    )
    assert torch.equal(masked, expected)


def test_large_radius_preserves_candidate_graph():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    neighbors = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0]])

    masked = DA_RADIUS.mask_neighbors_by_radius(
        points,
        neighbors,
        radius_scale=torch.full((3, 1), 100.0),
        base_radius=0.15,
    )

    assert torch.equal(masked, neighbors)
