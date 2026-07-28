import sys
import types
from pathlib import Path

import torch


KP_ROOT = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "ml-kpconvx-standalone"
    / "KPConvX"
)


def _install_cpp_stubs():
    package_names = [
        "cpp_wrappers",
        "cpp_wrappers.cpp_subsampling",
        "cpp_wrappers.cpp_neighbors",
    ]
    module_names = [
        "cpp_wrappers.cpp_subsampling.cpp_subsampling",
        "cpp_wrappers.cpp_neighbors.cpp_neighbors",
    ]
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules.setdefault(name, module)
    for name in module_names:
        sys.modules.setdefault(name, types.ModuleType(name))


_install_cpp_stubs()
sys.path.insert(0, str(KP_ROOT))

from easydict import EasyDict  # noqa: E402
from models.KPNext import KPNeXt  # noqa: E402
from models.KPNextV18 import KPNeXtV18  # noqa: E402
from utils.config import init_cfg  # noqa: E402


BASE_LIMITS = [12, 16, 20, 20, 20]
CANDIDATE_LIMITS = [12, 16, 40, 48, 20]


def _model_config(candidate_graph, gamma=0.0):
    cfg = init_cfg()
    cfg.data.task = "cloud_segmentation"
    cfg.data.dim = 3
    cfg.data.label_values = [0, 1, 2]
    cfg.data.ignored_labels = []
    cfg.model.kp_mode = "kpconvx"
    cfg.model.shell_sizes = [1, 14, 28]
    cfg.model.kp_radius = 2.1
    cfg.model.kp_sigma = 2.1
    cfg.model.kp_influence = "linear"
    cfg.model.kp_aggregation = "nearest"
    cfg.model.conv_groups = -1
    cfg.model.share_kp = True
    cfg.model.first_inv_layer = 1
    cfg.model.inv_groups = 4
    cfg.model.inv_grp_norm = True
    cfg.model.inv_act = "sigmoid"
    cfg.model.in_sub_size = 0.04
    cfg.model.in_sub_mode = "grid"
    cfg.model.radius_scaling = 2.2
    cfg.model.grid_pool = True
    cfg.model.decoder_layer = True
    cfg.model.drop_path_rate = 0.0
    cfg.model.input_channels = 5
    cfg.model.init_channels = 64
    cfg.model.channel_scaling = 1.41
    cfg.model.layer_blocks = (1, 1, 1, 1, 1)
    cfg.model.neighbor_limits = list(
        CANDIDATE_LIMITS if candidate_graph else BASE_LIMITS
    )
    cfg.model.base_neighbor_limits = list(BASE_LIMITS)
    cfg.model.dual_support_stages = [3, 4]
    cfg.model.dual_support_ring_limits = {"3": 8, "4": 12}
    cfg.model.dual_support_stage_ranges = {
        "3": [0.75, 1.45],
        "4": [0.70, 1.55],
    }
    cfg.model.dual_support_fixed_spacing_bounds = {}
    cfg.model.dual_support_use_fixed_spacing = False
    cfg.model.dual_support_density_k = 16
    cfg.model.dual_support_percentile = [10, 90]
    cfg.model.dual_support_strength = 1.0
    cfg.model.dual_support_power = 1.0
    cfg.model.dual_support_gamma_mode = "fixed"
    cfg.model.dual_support_gamma = gamma
    cfg.model.dual_support_debug = False
    cfg.model.dual_support_debug_interval = 100
    return cfg


def _synthetic_batch(neighbor_limits, features):
    num_points = features.shape[0]
    point_ids = torch.arange(num_points)
    points = torch.stack(
        (
            point_ids.float() * 0.05,
            torch.zeros(num_points),
            torch.zeros(num_points),
        ),
        dim=1,
    )
    neighbors = [
        (point_ids[:, None] + torch.arange(limit)[None, :]) % num_points
        for limit in neighbor_limits
    ]
    identity_map = point_ids[:, None]
    in_dict = EasyDict(
        features=features.clone(),
        points=[points.clone() for _ in neighbor_limits],
        lengths=[torch.tensor([num_points]) for _ in neighbor_limits],
        neighbors=neighbors,
        pools=[identity_map.clone() for _ in neighbor_limits[:-1]],
        upsamples=[identity_map.clone() for _ in neighbor_limits[:-1]],
    )
    return EasyDict(in_dict=in_dict)


def test_v18_full_forward_and_gamma_zero_identity():
    torch.manual_seed(7)
    baseline = KPNeXt(_model_config(candidate_graph=False)).eval()
    v18 = KPNeXtV18(_model_config(candidate_graph=True, gamma=0.0)).eval()

    incompatible = v18.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all("ring_modules" in key for key in incompatible.missing_keys)

    stage_3 = v18.ring_modules["3"]
    stage_4 = v18.ring_modules["4"]
    assert stage_3.channels == 192
    assert stage_4.channels == 256
    assert sum(parameter.numel() for parameter in stage_3.parameters()) == 446_112
    assert sum(parameter.numel() for parameter in stage_4.parameters()) == 787_328

    features = torch.randn(64, 5)
    baseline_batch = _synthetic_batch(BASE_LIMITS, features)
    v18_batch = _synthetic_batch(CANDIDATE_LIMITS, features)
    with torch.no_grad():
        baseline_logits = baseline(baseline_batch)
        v18_logits = v18(v18_batch)

    assert v18_logits.shape == (64, 3)
    assert torch.isfinite(v18_logits).all()
    torch.testing.assert_close(v18_logits, baseline_logits, rtol=0.0, atol=0.0)
