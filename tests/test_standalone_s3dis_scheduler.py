import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pointcept" / "utils" / "scheduler.py"
SPEC = importlib.util.spec_from_file_location(
    "pointcept.utils.scheduler", MODULE_PATH
)
SCHEDULER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCHEDULER_MODULE)
StandaloneS3DISLR = SCHEDULER_MODULE.StandaloneS3DISLR


def test_standalone_schedule_matches_reference_epoch_landmarks():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=5.0e-3)
    scheduler = StandaloneS3DISLR(optimizer, total_steps=450)
    factor = scheduler.lr_lambdas[0]

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert 5.0e-3 * factor(30) == pytest.approx(5.0e-3)
    assert 5.0e-3 * factor(35) == pytest.approx(5.0e-3)
    assert 5.0e-3 * factor(155) == pytest.approx(5.0e-4)
    assert 5.0e-3 * factor(275) == pytest.approx(5.0e-5)
    assert 5.0e-3 * factor(395) == pytest.approx(5.0e-6)


def test_standalone_schedule_scales_landmarks_to_arbitrary_step_budget():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=5.0e-3)
    scheduler = StandaloneS3DISLR(optimizer, total_steps=135_000)
    factor = scheduler.lr_lambdas[0]

    assert 5.0e-3 * factor(9_000) == pytest.approx(5.0e-3)
    assert 5.0e-3 * factor(10_500) == pytest.approx(5.0e-3)
    assert 5.0e-3 * factor(46_500) == pytest.approx(5.0e-4)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"total_steps": 0},
        {"total_steps": 10, "start_lr": 0},
        {"total_steps": 10, "start_lr": 1e-2, "peak_lr": 5e-3},
        {"total_steps": 10, "decay10_epochs": 0},
    ),
)
def test_standalone_schedule_rejects_invalid_parameters(kwargs):
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=5.0e-3)
    with pytest.raises(ValueError):
        StandaloneS3DISLR(optimizer, **kwargs)
