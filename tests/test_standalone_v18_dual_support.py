import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "ml-kpconvx-standalone"
    / "KPConvX"
    / "utils"
    / "dual_support.py"
)
SPEC = importlib.util.spec_from_file_location("standalone_dual_support", MODULE_PATH)
DUAL_SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DUAL_SUPPORT)


def _line_cloud():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ]
    )
    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 1, 0, 3, 4],
            [3, 2, 1, 4, 0],
            [4, 3, 2, 1, 0],
        ]
    )
    return points, candidates


def test_scale_one_has_no_expansion_ring():
    points, candidates = _line_cloud()
    ring, stats = DUAL_SUPPORT.build_ring_neighbors(
        points,
        candidates,
        radius_scale=torch.ones((5, 1)),
        base_radius=0.25,
        base_limit=2,
        ring_limit=2,
    )

    assert torch.equal(ring[0], torch.tensor([5, 5]))
    assert stats["added_count"][0].item() == 0


def test_scale_two_adds_neighbor_beyond_base_h_and_radius():
    points, candidates = _line_cloud()
    ring, stats = DUAL_SUPPORT.build_ring_neighbors(
        points,
        candidates,
        radius_scale=torch.full((5, 1), 2.0),
        base_radius=0.25,
        base_limit=2,
        ring_limit=2,
    )

    assert ring[0, 0].item() == 3
    assert ring[0, 1].item() == points.shape[0]
    assert stats["added_count"][0].item() == 1
    assert stats["change_rate"][0].item() > 0
    assert torch.equal(candidates[0, :2], torch.tensor([0, 1]))


def test_candidate_graph_must_be_larger_than_base_h():
    points, candidates = _line_cloud()
    with pytest.raises(ValueError, match="greater than base_limit"):
        DUAL_SUPPORT.build_ring_neighbors(
            points,
            candidates[:, :2],
            radius_scale=torch.ones((5, 1)),
            base_radius=0.25,
            base_limit=2,
            ring_limit=1,
        )


def test_fixed_spacing_bounds_are_fragment_independent():
    points, candidates = _line_cloud()
    mapper = DUAL_SUPPORT.DimensionlessSpacingScale(
        density_k=3,
        strength=1.0,
    )
    scale, spacing = mapper(
        points,
        candidates[:, :3],
        lengths=torch.tensor([5]),
        grid_size=0.1,
        scale_range=(0.75, 1.45),
        fixed_bounds=(1.0, 3.0),
    )

    assert scale.shape == (5, 1)
    assert spacing.shape == (5, 1)
    assert scale.min() >= 0.75
    assert scale.max() <= 1.45
