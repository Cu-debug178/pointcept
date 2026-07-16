from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "s3dis"
BASELINE_CONFIG = CONFIG_DIR / "semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"
SOKA_CONFIG = CONFIG_DIR / "semseg-kpconvx-soka-lite-scale04-4090d-area5.py"
SOKA_STAGE4_CONFIG = (
    CONFIG_DIR / "semseg-kpconvx-soka-lite-stage4-scale04-4090d-area5.py"
)
SOKA_KEYS = {
    "soka_enabled",
    "soka_stages",
    "soka_hidden_dim",
    "soka_bias_bound",
    "soka_monitor",
}


def _merge(base, child):
    if isinstance(child, dict) and child.get("_delete_", False):
        return {key: value for key, value in child.items() if key != "_delete_"}
    if isinstance(base, dict) and isinstance(child, dict):
        merged = dict(base)
        for key, value in child.items():
            if key == "_delete_":
                continue
            merged[key] = _merge(merged.get(key), value) if key in merged else value
        return merged
    return child


def _load_config(path):
    path = Path(path).resolve()
    current = {
        key: value
        for key, value in runpy.run_path(str(path)).items()
        if not key.startswith("__")
    }
    bases = current.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    config = {}
    for base in bases:
        config = _merge(config, _load_config(path.parent / base))
    return _merge(config, current)


def _without_soka(backbone):
    return {
        key: value
        for key, value in backbone.items()
        if key not in SOKA_KEYS and key != "type"
    }


def test_formal_soka_config_only_changes_backbone_type_and_soka_options():
    baseline = _load_config(BASELINE_CONFIG)
    soka = _load_config(SOKA_CONFIG)

    baseline_backbone = baseline["model"]["backbone"]
    soka_backbone = soka["model"]["backbone"]
    assert baseline_backbone["type"] == "kpconvx_base"
    assert soka_backbone["type"] == "kpconvx_soka"
    assert _without_soka(soka_backbone) == _without_soka(baseline_backbone)
    assert soka["model"]["criteria"] == baseline["model"]["criteria"]
    assert soka["data"] == baseline["data"]


def test_formal_soka_config_uses_all_encoder_attention_stages():
    backbone = _load_config(SOKA_CONFIG)["model"]["backbone"]
    assert backbone["soka_enabled"] is True
    assert backbone["soka_stages"] == (2, 3, 4, 5)
    assert backbone["soka_hidden_dim"] == 16
    assert backbone["soka_bias_bound"] == 2.0
    assert backbone["soka_monitor"] is True


def test_stage4_probe_changes_only_stage_selection():
    formal = _load_config(SOKA_CONFIG)
    stage4 = _load_config(SOKA_STAGE4_CONFIG)
    formal_backbone = dict(formal["model"]["backbone"])
    stage4_backbone = dict(stage4["model"]["backbone"])
    assert stage4_backbone.pop("soka_stages") == (4,)
    formal_backbone.pop("soka_stages")
    assert stage4_backbone == formal_backbone
    assert stage4["data"] == formal["data"]


def test_scale04_physical_calibration_is_preserved():
    for path in (SOKA_CONFIG, SOKA_STAGE4_CONFIG):
        config = _load_config(path)
        backbone = config["model"]["backbone"]
        assert backbone["subsample_size"] == 0.04
        assert backbone["kp_radius"] == 2.1
        assert backbone["kp_sigma"] == 2.1
        assert config["batch_size"] == 3
