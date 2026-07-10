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
        self.v17 = _load_config(V17_CONFIG)

    def test_physical_scale_is_consistent(self):
        for config in (self.baseline, self.v17):
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

    def test_test_protocol_is_single_view_and_fragmented(self):
        for config in (self.baseline, self.v17):
            test_cfg = config["data"]["test"]["test_cfg"]
            self.assertEqual(test_cfg["crop"]["type"], "TestSphereCrop")
            self.assertEqual(test_cfg["crop"]["point_max"], 60000)
            self.assertEqual(len(test_cfg["aug_transform"]), 1)

    def test_periodic_checkpoint_frequency(self):
        for config in (self.baseline, self.v17):
            saver = next(
                hook for hook in config["hooks"] if hook["type"] == "CheckpointSaver"
            )
            self.assertEqual(saver["save_freq"], 20)

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
