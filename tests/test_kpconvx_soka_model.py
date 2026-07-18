import pytest
import torch


pytest.importorskip("torch_scatter")
pytest.importorskip("torch_geometric")

from pointcept.models.default import DefaultSegmentor
from pointcept.models.kpconvx.kpconvx_base import KPConvXBase
from pointcept.models.kpconvx.utils.kpnext_blocks import KPConvX
from pointcept.models.kpconvx_soka.kpconvx_soka import KPConvXSOKA
from pointcept.models.kpconvx_soka.kpconvx_soka_lite import KPConvXSOKALite
from pointcept.models.kpconvx_soka.soka_blocks import SOKAKPConvX
from pointcept.models.kpconvx_soka.soka_lite_blocks import SOKALiteKPConvX


def _operator(shared_kp_data=None):
    return KPConvX(
        channels=8,
        shell_sizes=(1, 14, 28),
        radius=1.0,
        sigma=1.0,
        attention_groups=2,
        attention_act="sigmoid",
        mod_grp_norm=True,
        shared_kp_data=shared_kp_data,
        influence_mode="constant",
    )


def _inputs(device="cpu"):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.5, 0.0], [0.4, 0.5, 0.0]],
        device=device,
    )
    features = torch.randn(4, 8, device=device)
    neighbors = torch.tensor(
        [[0, 1, 2, 4], [1, 0, 3, 4], [2, 0, 3, 4], [3, 1, 2, 4]],
        device=device,
    )
    return points, features, neighbors


def _upgrade(module, monitor=True):
    return SOKAKPConvX.from_kpconvx(
        module,
        soka_evidence_dim=8,
        soka_rank=4,
        soka_bias_bound=2.0,
        soka_monitor=monitor,
    )


def test_zero_initialized_soka_matches_baseline():
    torch.manual_seed(3)
    baseline = _operator().eval()
    soka = _upgrade(baseline).eval()
    points, features, neighbors = _inputs()

    with torch.no_grad():
        baseline_output = baseline(points, points, features, neighbors)
        soka_output = soka(points, points, features, neighbors)
    assert torch.allclose(baseline_output, soka_output, atol=1e-6, rtol=0)
    assert "base_logit_rms" in soka._last_soka_diag
    assert all(torch.isfinite(value) for value in soka._last_soka_diag.values())


def test_neighbor_permutation_keeps_trained_soka_output_unchanged():
    torch.manual_seed(4)
    soka = _upgrade(_operator()).eval()
    with torch.no_grad():
        soka.soka_correction_out.weight.normal_(std=0.05)
        soka.soka_correction_out.bias.normal_(std=0.05)
    points, features, neighbors = _inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        output = soka(points, points, features, neighbors)
        permuted_output = soka(points, points, features, neighbors[:, permutation])
    assert torch.allclose(output, permuted_output, atol=1e-6, rtol=0)


def test_negative_and_shadow_padding_are_equivalent():
    soka = _upgrade(_operator()).eval()
    points, features, shadow_neighbors = _inputs()
    negative_neighbors = shadow_neighbors.clone()
    negative_neighbors[negative_neighbors == points.shape[0]] = -1
    with torch.no_grad():
        shadow_output = soka(points, points, features, shadow_neighbors)
        negative_output = soka(points, points, features, negative_neighbors)
    assert torch.allclose(shadow_output, negative_output, atol=1e-6, rtol=0)


def test_zero_last_layer_opens_all_evidence_gradients_after_one_update():
    torch.manual_seed(5)
    soka = _upgrade(_operator()).train()
    points, features, neighbors = _inputs()

    first_loss = soka(points, points, features, neighbors).square().mean()
    first_loss.backward()
    assert soka.soka_correction_out.weight.grad.abs().sum() > 0
    assert soka.weights.grad.abs().sum() > 0

    with torch.no_grad():
        soka.soka_correction_out.weight.add_(
            -0.1 * soka.soka_correction_out.weight.grad
        )
    soka.zero_grad(set_to_none=True)
    second_loss = soka(points, points, features, neighbors).square().mean()
    second_loss.backward()
    assert soka.soka_neighbor_proj.weight.grad.abs().sum() > 0
    assert soka.soka_geometry_encoder[0].weight.grad.abs().sum() > 0
    assert soka.soka_query_encoder[0].weight.grad.abs().sum() > 0


def test_shared_geometry_cache_matches_nonshared_operator():
    shared = {}
    first = _upgrade(_operator(shared)).eval()
    second = _upgrade(_operator(shared)).eval()
    standalone = _upgrade(_operator()).eval()
    standalone.load_state_dict(second.state_dict(), strict=True)
    points, features, neighbors = _inputs()

    with torch.no_grad():
        first(points, points, features, neighbors)
        expected_signature = second._geometry_signature(
            points, points, neighbors
        )
        assert shared["soka_geometry_signature"] == expected_signature
        cached_output = second(points, points, features, neighbors)
        standalone_output = standalone(points, points, features, neighbors)
        recomputed_output = second(points, points, features, neighbors.clone())
    assert torch.allclose(cached_output, standalone_output, atol=1e-6, rtol=0)
    assert torch.allclose(recomputed_output, standalone_output, atol=1e-6, rtol=0)

    with torch.no_grad():
        points.add_(0.05)
        first(points, points, features, neighbors)
        refreshed_output = second(points, points, features, neighbors)
        refreshed_standalone = standalone(points, points, features, neighbors)
    assert torch.allclose(
        refreshed_output, refreshed_standalone, atol=1e-6, rtol=0
    )


def test_backbone_replaces_only_selected_encoder_kpconvx_blocks():
    model = KPConvXSOKA(
        input_channels=9,
        num_classes=13,
        layer_blocks=(1, 1, 1, 1, 1),
        init_channels=16,
        channel_scaling=1.0,
        first_inv_layer=1,
        decoder_layer=False,
        soka_stages=(4, 5),
        soka_monitor=False,
    )
    for stage in (1, 2, 3):
        assert not isinstance(getattr(model, f"encoder_{stage}")[0].conv, SOKAKPConvX)
    for stage in (4, 5):
        assert isinstance(getattr(model, f"encoder_{stage}")[0].conv, SOKAKPConvX)
    assert not any(
        isinstance(module, SOKAKPConvX)
        for name, module in model.named_modules()
        if name.startswith("decoder_layer_")
    )


def test_lite_backbone_remains_separately_available():
    model = KPConvXSOKALite(
        input_channels=9,
        num_classes=13,
        layer_blocks=(1, 1, 1, 1, 1),
        init_channels=16,
        channel_scaling=1.0,
        first_inv_layer=1,
        decoder_layer=False,
        soka_stages=(4,),
        soka_monitor=False,
    )
    assert isinstance(model.encoder_4[0].conv, SOKALiteKPConvX)
    assert not isinstance(model.encoder_4[0].conv, SOKAKPConvX)


def test_backbone_preserves_all_baseline_state_keys():
    kwargs = dict(
        input_channels=9,
        num_classes=13,
        layer_blocks=(1, 1, 1, 1, 1),
        init_channels=16,
        channel_scaling=1.0,
        first_inv_layer=1,
        decoder_layer=False,
    )
    baseline = KPConvXBase(**kwargs)
    soka = KPConvXSOKA(**kwargs, soka_monitor=False)
    baseline_keys = set(baseline.state_dict())
    soka_baseline_keys = {
        key for key in soka.state_dict() if ".soka_" not in key
    }
    assert soka_baseline_keys == baseline_keys


def test_monitored_backbone_uses_default_segmentor_output_contract(monkeypatch):
    model = KPConvXSOKA(
        input_channels=9,
        num_classes=13,
        layer_blocks=(1, 1, 1, 1, 1),
        init_channels=16,
        channel_scaling=1.0,
        first_inv_layer=1,
        decoder_layer=False,
        soka_stages=(4,),
        soka_monitor=True,
    )
    stage_module = model._soka_modules_by_stage[4][0]
    stage_module._last_soka_diag = {
        "base_logit_rms": torch.tensor(1.0),
        "soka_to_base_ratio": torch.tensor(0.0),
    }
    logits = torch.randn(5, 13)
    monkeypatch.setattr(KPConvXBase, "forward", lambda self, data: logits)

    output = model({})
    unpacked_logits, aux = DefaultSegmentor._unpack_backbone_output(output)
    assert unpacked_logits is logits
    assert aux["soka_s4_base_logit_rms"].item() == 1.0
    assert aux["soka_s4_soka_to_base_ratio"].item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP")
def test_cuda_amp_forward_is_finite():
    soka = _upgrade(_operator()).cuda().train()
    points, features, neighbors = _inputs("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = soka(points, points, features, neighbors)
        loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(soka.soka_correction_out.weight.grad).all()
