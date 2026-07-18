from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "s3dis"
BASELINE_CONFIG = CONFIG_DIR / "semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"
LITE_CONFIG = CONFIG_DIR / "semseg-kpconvx-soka-lite-scale04-4090d-area5.py"
LITE_STAGE4_CONFIG = (
    CONFIG_DIR / "semseg-kpconvx-soka-lite-stage4-scale04-4090d-area5.py"
)
FUSION_CONFIG = CONFIG_DIR / "semseg-kpconvx-soka-fusion-scale04-4090d-area5.py"
FUSION_STAGE4_CONFIG = (
    CONFIG_DIR / "semseg-kpconvx-soka-fusion-stage4-scale04-4090d-area5.py"
)
COMMON_SOKA_KEYS = {
    "soka_enabled",
    "soka_stages",
    "soka_bias_bound",
    "soka_monitor",
}
LITE_KEYS = COMMON_SOKA_KEYS | {"soka_hidden_dim"}
FUSION_KEYS = COMMON_SOKA_KEYS | {
    "soka_evidence_dim",
    "soka_rank",
    "soka_use_geometry",
    "soka_use_topology",
    "soka_use_query",
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


def _without_soka(backbone, soka_keys):
    return {
        key: value
        for key, value in backbone.items()
        if key not in soka_keys and key != "type"
    }


def _assert_training_contract_is_preserved(config, baseline, soka_keys):
    baseline_backbone = baseline["model"]["backbone"]
    variant_backbone = config["model"]["backbone"]
    assert _without_soka(variant_backbone, soka_keys) == _without_soka(
        baseline_backbone, soka_keys
    )
    assert config["model"]["criteria"] == baseline["model"]["criteria"]
    assert config["data"] == baseline["data"]


def test_lite_config_is_preserved_under_its_new_registry_name():
    baseline = _load_config(BASELINE_CONFIG)
    lite = _load_config(LITE_CONFIG)

    baseline_backbone = baseline["model"]["backbone"]
    assert baseline_backbone["type"] == "kpconvx_base"
    assert lite["model"]["backbone"]["type"] == "kpconvx_soka_lite"
    _assert_training_contract_is_preserved(lite, baseline, LITE_KEYS)


def test_fusion_config_only_changes_backbone_type_and_soka_options():
    baseline = _load_config(BASELINE_CONFIG)
    fusion = _load_config(FUSION_CONFIG)

    assert fusion["model"]["backbone"]["type"] == "kpconvx_soka"
    _assert_training_contract_is_preserved(fusion, baseline, FUSION_KEYS)


def test_lite_formal_config_keeps_all_original_attention_stages():
    backbone = _load_config(LITE_CONFIG)["model"]["backbone"]
    assert backbone["soka_enabled"] is True
    assert backbone["soka_stages"] == (2, 3, 4, 5)
    assert backbone["soka_hidden_dim"] == 16
    assert backbone["soka_bias_bound"] == 2.0
    assert backbone["soka_monitor"] is True


def test_fusion_formal_config_uses_deep_stages_and_all_evidence_branches():
    backbone = _load_config(FUSION_CONFIG)["model"]["backbone"]
    assert backbone["soka_enabled"] is True
    assert backbone["soka_stages"] == (4, 5)
    assert backbone["soka_evidence_dim"] == 16
    assert backbone["soka_rank"] == 8
    assert backbone["soka_bias_bound"] == 2.0
    assert backbone["soka_use_geometry"] is True
    assert backbone["soka_use_topology"] is True
    assert backbone["soka_use_query"] is True
    assert backbone["soka_monitor"] is True


def test_stage4_probes_change_only_stage_selection():
    for formal_path, stage4_path in (
        (LITE_CONFIG, LITE_STAGE4_CONFIG),
        (FUSION_CONFIG, FUSION_STAGE4_CONFIG),
    ):
        formal = _load_config(formal_path)
        stage4 = _load_config(stage4_path)
        formal_backbone = dict(formal["model"]["backbone"])
        stage4_backbone = dict(stage4["model"]["backbone"])
        assert stage4_backbone.pop("soka_stages") == (4,)
        formal_backbone.pop("soka_stages")
        assert stage4_backbone == formal_backbone
        assert stage4["data"] == formal["data"]


def test_scale04_physical_calibration_is_preserved():
    for path in (
        LITE_CONFIG,
        LITE_STAGE4_CONFIG,
        FUSION_CONFIG,
        FUSION_STAGE4_CONFIG,
    ):
        config = _load_config(path)
        backbone = config["model"]["backbone"]
        assert backbone["subsample_size"] == 0.04
        assert backbone["kp_radius"] == 2.1
        assert backbone["kp_sigma"] == 2.1
        assert config["batch_size"] == 3
