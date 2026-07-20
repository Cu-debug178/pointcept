import importlib.util
import sys
import types
from pathlib import Path

import torch
from easydict import EasyDict


REPO_ROOT = Path(__file__).resolve().parents[1]
PYRAMID_PATH = (
    REPO_ROOT / "pointcept" / "models" / "kpconvx" / "utils" / "torch_pyramid.py"
)


def _module(**attributes):
    module = types.ModuleType("test_stub")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _load_torch_pyramid(monkeypatch, query_calls):
    def adaptive_ball_query_idx(
        nsample, min_radius, radius, points, offset, new_points, new_offset
    ):
        query_calls.append(radius.clone())
        idx = torch.full(
            (new_points.shape[0], nsample), -1, dtype=torch.int32
        )
        idx[:, 0] = torch.arange(new_points.shape[0], dtype=torch.int32)
        return idx

    stubs = {
        "pointcept.models.kpconvx.utils.gpu_subsampling": _module(
            subsample_pack_batch=lambda *args, **kwargs: None
        ),
        "pointcept.models.kpconvx.utils.gpu_neigbors": _module(
            radius_search_pack_mode=lambda *args, **kwargs: None,
            keops_radius_count=lambda *args, **kwargs: None,
        ),
        "pointcept.models.kpconvx.utils.cpp_funcs": _module(
            batch_grid_partition=lambda *args, **kwargs: None
        ),
        "pointcept.models.kpconvx.cpp_wrappers.cpp_neighbors": _module(
            cpp_neighbors=types.SimpleNamespace()
        ),
        "pointcept.models.utils": _module(
            offset2batch=lambda *args, **kwargs: None,
            batch2offset=lambda *args, **kwargs: None,
        ),
        "torch_geometric": _module(),
        "torch_geometric.nn": _module(),
        "torch_geometric.nn.pool": _module(
            voxel_grid=lambda *args, **kwargs: None
        ),
        "torch_scatter": _module(segment_csr=lambda *args, **kwargs: None),
        "pointops": _module(adaptive_ball_query_idx=adaptive_ball_query_idx),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("torch_pyramid_under_test", PYRAMID_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replace_pyramid_da_neighbors_reuses_structure(monkeypatch):
    query_calls = []
    torch_pyramid = _load_torch_pyramid(monkeypatch, query_calls)

    points = [torch.zeros((4, 3)), torch.zeros((2, 3))]
    lengths = [torch.tensor([4]), torch.tensor([2])]
    base_neighbors = [torch.full((4, 3), 7), torch.full((2, 2), 9)]
    pool = torch.tensor([11])
    upsample = torch.tensor([12])
    pyramid = EasyDict(
        points=points,
        lengths=lengths,
        neighbors=base_neighbors,
        pools=[pool],
        upsamples=[upsample],
        up_distances=[],
    )

    result = torch_pyramid.replace_pyramid_da_neighbors(
        pyramid,
        search_radius=0.2,
        radius_scaling=2.0,
        neighbor_limits=[3, 2],
        da_radius_scales=[None, torch.tensor([1.0, 1.5])],
    )

    assert result is pyramid
    assert result.points[0] is points[0]
    assert result.points[1] is points[1]
    assert result.pools[0] is pool
    assert result.upsamples[0] is upsample
    assert result.neighbors[0] is base_neighbors[0]
    assert torch.equal(result.neighbors[1], torch.tensor([[0, 2], [1, 2]]))
    assert len(query_calls) == 1
    assert torch.allclose(query_calls[0], torch.tensor([0.4, 0.6]))
