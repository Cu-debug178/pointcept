from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = (
    ROOT
    / "configs"
    / "s3dis"
    / "semseg-kpconvx-base-s3dis-scale04-4090d-area5.py"
)
ALIGNED_BASELINE_CONFIG = (
    ROOT
    / "configs"
    / "s3dis"
    / "semseg-kpconvx-base-s3dis-scale04-136k-linear-sharekp-4090d-area5.py"
)
STANDALONE_TRAIN_CONFIG = (
    ROOT
    / "configs"
    / "s3dis"
    / "semseg-kpconvx-base-s3dis-standalone-train-aligned-136k-4090d-area5.py"
)
V17_CONFIG = (
    ROOT
    / "configs"
    / "s3dis"
    / "semseg-kpconvx-hybrid-v17-scale04-4090d-area5.py"
)


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


def _grid_size(config, split):
    if split == "test":
        return config["data"]["test"]["test_cfg"]["voxelize"]["grid_size"]
    return next(
        transform["grid_size"]
        for transform in config["data"][split]["transform"]
        if transform["type"] == "GridSample"
    )


class S3DISScale04ConfigTest(unittest.TestCase):
    def setUp(self):
        self.baseline = _load_config(BASELINE_CONFIG)
        self.aligned_baseline = _load_config(ALIGNED_BASELINE_CONFIG)
        self.standalone_train = _load_config(STANDALONE_TRAIN_CONFIG)
        self.v17 = _load_config(V17_CONFIG)

    def test_physical_scale_is_consistent(self):
        for config in (
            self.baseline,
            self.aligned_baseline,
            self.standalone_train,
            self.v17,
        ):
            backbone = config["model"]["backbone"]
            self.assertEqual(backbone["subsample_size"], 0.04)
            self.assertEqual(backbone["kp_radius"], 2.1)
            self.assertEqual(backbone["kp_sigma"], 2.1)
            self.assertAlmostEqual(
                backbone["subsample_size"] * backbone["kp_radius"], 0.084
            )
            self.assertEqual(
                [_grid_size(config, split) for split in ("train", "val", "test")],
                [0.04, 0.04, 0.04],
            )

    def test_4090d_training_budget_matches(self):
        keys = (
            "batch_size",
            "gradient_accumulation_steps",
            "max_input_pts",
            "enable_amp",
            "fragment_batch_size_test",
            "epoch",
            "eval_epoch",
        )
        for key in keys:
            self.assertEqual(self.baseline[key], self.v17[key], key)
        self.assertEqual(self.baseline["batch_size"], 3)
        self.assertEqual(self.baseline["max_input_pts"], 40000)
        self.assertFalse(self.baseline["enable_amp"])
        self.assertEqual(self.baseline["data"]["train"]["loop"], 5)
        self.assertEqual(self.v17["data"]["train"]["loop"], 5)

    def test_aligned_baseline_changes_only_budget_and_operator_factors(self):
        baseline = self.baseline
        aligned = self.aligned_baseline
        baseline_backbone = dict(baseline["model"]["backbone"])
        aligned_backbone = dict(aligned["model"]["backbone"])

        self.assertEqual(aligned_backbone.pop("kp_influence"), "linear")
        self.assertTrue(aligned_backbone.pop("share_kp"))
        baseline_backbone.pop("kp_influence")
        baseline_backbone.pop("share_kp")
        self.assertEqual(aligned_backbone, baseline_backbone)

        self.assertEqual(aligned["epoch"], 2000)
        self.assertEqual(aligned["eval_epoch"], 400)
        self.assertEqual(aligned["epoch"] // aligned["eval_epoch"], 5)
        self.assertEqual(aligned["seed"], 57106803)
        self.assertEqual(aligned["gradient_accumulation_steps"], 1)
        self.assertEqual(aligned["model"]["criteria"], baseline["model"]["criteria"])
        self.assertEqual(aligned["optimizer"], baseline["optimizer"])
        self.assertEqual(aligned["data"]["train"], baseline["data"]["train"])
        self.assertEqual(aligned["data"]["test"], baseline["data"]["test"])

        self.assertEqual(aligned_backbone["input_channels"], 9)
        self.assertEqual(aligned_backbone["inv_groups"], 8)

    def test_test_protocol_is_single_view_and_fragmented(self):
        for config in (self.baseline, self.v17):
            test_cfg = config["data"]["test"]["test_cfg"]
            self.assertEqual(test_cfg["crop"]["type"], "TestSphereCrop")
            self.assertEqual(test_cfg["crop"]["point_max"], 60000)
            self.assertEqual(len(test_cfg["aug_transform"]), 1)

    def test_periodic_checkpoint_frequency(self):
        baseline_saver = next(
            hook
            for hook in self.baseline["hooks"]
            if hook["type"] == "CheckpointSaver"
        )
        v17_saver = next(
            hook
            for hook in self.v17["hooks"]
            if hook["type"] == "CheckpointSaver"
        )
        self.assertEqual(baseline_saver["save_freq"], 20)
        self.assertEqual(v17_saver["save_freq"], 10)

        aligned_saver = next(
            hook
            for hook in self.aligned_baseline["hooks"]
            if hook["type"] == "CheckpointSaver"
        )
        self.assertEqual(
            aligned_saver["weight_only_save_rules"],
            [
                dict(start=1, end=100, freq=20),
                dict(start=101, end=140, freq=10),
                dict(start=141, end=None, freq=5),
            ],
        )
        self.assertEqual(aligned_saver["resume_save_freq"], 50)

    def test_aligned_validation_is_fixed(self):
        self.assertEqual(self.aligned_baseline["batch_size_val"], 1)
        transforms = self.aligned_baseline["data"]["val"]["transform"]
        grid = next(item for item in transforms if item["type"] == "GridSample")
        crop = next(item for item in transforms if item["type"] == "SphereCrop")
        self.assertTrue(grid["deterministic"])
        self.assertEqual(grid["mode"], "train")
        self.assertEqual(crop["mode"], "center")
        self.assertEqual(crop["point_max"], 40000)

    def test_standalone_training_factors_are_explicitly_aligned(self):
        config = self.standalone_train
        backbone = config["model"]["backbone"]
        self.assertEqual(config["epoch"], 2000)
        self.assertEqual(config["eval_epoch"], 400)
        self.assertEqual(backbone["kp_influence"], "linear")
        self.assertTrue(backbone["share_kp"])
        self.assertEqual(backbone["inv_groups"], 4)
        self.assertEqual(backbone["channel_scaling"], 1.41)
        self.assertEqual(backbone["input_channels"], 9)
        self.assertEqual(config["mix_prob"], 0)
        self.assertEqual(
            config["model"]["criteria"],
            [
                dict(
                    type="CrossEntropyLoss",
                    loss_weight=1.0,
                    ignore_index=-1,
                )
            ],
        )
        self.assertEqual(config["optimizer"]["type"], "AdamW")
        self.assertEqual(config["optimizer"]["lr"], 5.0e-3)
        self.assertEqual(config["optimizer"]["weight_decay"], 0.05)
        self.assertEqual(config["scheduler"]["type"], "StandaloneS3DISLR")
        self.assertNotIn("max_lr", config["scheduler"])
        self.assertNotIn("pct_start", config["scheduler"])
        self.assertNotIn("div_factor", config["scheduler"])
        self.assertEqual(config["scheduler"]["start_lr"], 1.0e-4)
        self.assertEqual(config["scheduler"]["warmup_epochs"], 30)
        self.assertEqual(config["scheduler"]["plateau_epochs"], 5)
        self.assertEqual(config["scheduler"]["decay10_epochs"], 120)

    def test_standalone_training_config_keeps_pointcept_data_protocol(self):
        self.assertEqual(
            self.standalone_train["data"], self.aligned_baseline["data"]
        )
        train_types = [
            transform["type"]
            for transform in self.standalone_train["data"]["train"]["transform"]
        ]
        for transform_type in (
            "RandomDropout",
            "RandomRotateTargetAngle",
            "RandomScale",
            "RandomFlip",
            "RandomJitter",
            "ElasticDistortion",
            "ChromaticAutoContrast",
            "ChromaticTranslation",
            "ChromaticJitter",
        ):
            self.assertIn(transform_type, train_types)

    def test_v17_only_enables_intended_experimental_paths(self):
        baseline = self.baseline["model"]["backbone"]
        v17 = self.v17["model"]["backbone"]
        self.assertEqual(baseline["type"], "kpconvx_base")
        self.assertEqual(v17["type"], "KPConvXV17")
        self.assertTrue(v17["enable_da_meta"])
        self.assertTrue(v17["enable_dual_support"])
        self.assertEqual(v17["dual_support_stages"], (4,))
        self.assertFalse(v17["enable_support_mask"])
        self.assertFalse(v17["enable_gc_mixer"])
        self.assertFalse(v17["enable_refine"])


if __name__ == "__main__":
    unittest.main()
