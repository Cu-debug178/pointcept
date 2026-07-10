import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.audit_v17_neighbor_compatibility import (
    compute_query_stats,
    pool_labels_and_probs,
    roc_auc,
    spearman,
)
from tools.eval_s3dis_fixed_protocol import (
    expected_result_keys,
    fragment_batch_candidates,
    merge_server_results,
)
from tools.s3dis_fixed_protocol import (
    PROTOCOL_VERSION,
    build_checkpoint_entries,
    canonical_test_cfg,
    expected_run_metadata,
    identity_augmentations,
    load_manifest,
    metadata_matches,
    metrics_from_counts,
    rank_families,
    select_tta_family,
    tta13_augmentations,
)
class FixedProtocolTest(unittest.TestCase):
    def test_augmentation_counts(self):
        self.assertEqual(len(identity_augmentations()), 1)
        self.assertEqual(len(tta13_augmentations()), 13)

    def test_test_cfg_supports_attribute_access(self):
        try:
            from pointcept.utils.config import ConfigDict
        except ModuleNotFoundError as error:
            self.skipTest(f"Pointcept config dependency unavailable: {error}")
        cfg = ConfigDict(canonical_test_cfg(60000, "identity"))
        self.assertEqual(cfg.voxelize.grid_size, 0.02)
        self.assertEqual(cfg.crop.point_max, 60000)
        self.assertEqual(len(cfg.aug_transform), 1)

    def test_test_cfg_supports_scale04(self):
        cfg = canonical_test_cfg(60000, "identity", grid_size=0.04)
        self.assertEqual(cfg["voxelize"]["grid_size"], 0.04)
        self.assertEqual(cfg["crop"]["point_max"], 60000)

    def test_fragment_batch_candidates_are_unique_and_ordered(self):
        self.assertEqual(fragment_batch_candidates(4, [2, 1, 2]), [4, 2, 1])
        self.assertEqual(fragment_batch_candidates(0, []), [1])

    def test_expected_result_keys_can_select_best_only(self):
        manifest = dict(
            runs=[
                dict(family="baseline", run_id="base", checkpoints=["best", "last"]),
                dict(family="v17", run_id="v17", checkpoints=["best", "last"]),
            ]
        )
        all_keys = expected_result_keys(manifest)
        best_keys = expected_result_keys(manifest, checkpoint_kinds=["best"])
        self.assertEqual(len(all_keys), 4)
        self.assertEqual(len(best_keys), 2)
        self.assertEqual({key[-1] for key in best_keys}, {"best"})

        selected_keys = expected_result_keys(
            manifest,
            checkpoint_kinds=["best"],
            run_ids=["v17"],
        )
        self.assertEqual(
            selected_keys,
            {("v17", "v17", None, "best")},
        )

    def test_metrics_from_counts(self):
        metrics = metrics_from_counts(
            intersection=[5, 3],
            union=[10, 6],
            target=[8, 4],
        )
        self.assertAlmostEqual(metrics["mIoU"], 0.5)
        self.assertAlmostEqual(metrics["mAcc"], 0.6875)
        self.assertAlmostEqual(metrics["allAcc"], 8.0 / 12.0)

    def test_family_ranking_uses_best_mean(self):
        rows = [
            dict(family="baseline", checkpoint_kind="best", mIoU=0.70),
            dict(family="baseline", checkpoint_kind="last", mIoU=0.50),
            dict(family="v16b", checkpoint_kind="best", mIoU=0.72),
            dict(family="v16b", checkpoint_kind="best", mIoU=0.68),
            dict(family="v17", checkpoint_kind="best", mIoU=0.705),
        ]
        ranking = rank_families(rows)
        self.assertEqual(select_tta_family(ranking), "v17")

    def test_manifest_resolution_supports_both_exp_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run_a = root / "run-a"
            run_b = root / "s3dis" / "run-b"
            for run in (run_a, run_b):
                (run / "model").mkdir(parents=True)
                (run / "config.py").write_text("x = 1\n", encoding="utf-8")
                (run / "model" / "model_best.pth").write_bytes(b"best")
                (run / "model" / "model_last.pth").write_bytes(b"last")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    dict(
                        protocol_version=PROTOCOL_VERSION,
                        runs=[
                            dict(family="a", run_id="run-a"),
                            dict(family="b", run_id="run-b"),
                            dict(family="missing", run_id="run-c"),
                        ],
                    )
                ),
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path)
            entries, missing = build_checkpoint_entries(
                manifest, root, allow_missing=True
            )
            self.assertEqual(len(entries), 4)
            self.assertEqual(len(missing), 2)

    def test_resume_metadata_allows_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            expected = dict(protocol="identity", point_max=60000)
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            (output / "run_meta.json").write_text(
                json.dumps(dict(**expected, runtime_seconds=1.2)), encoding="utf-8"
            )
            self.assertTrue(metadata_matches(output, expected))

    def test_legacy_scale02_metadata_remains_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            legacy = dict(protocol="identity", point_max=60000)
            expected = dict(**legacy, grid_size=0.02)
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            (output / "run_meta.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
            self.assertTrue(metadata_matches(output, expected))

    def test_fragment_batch_size_is_part_of_resume_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.py"
            checkpoint = root / "model_best.pth"
            config.write_text("x = 1\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            entry = dict(
                family="v17",
                run_id="run",
                seed=1,
                checkpoint_kind="best",
                config_path=str(config),
                weight_path=str(checkpoint),
            )
            batch_four = expected_run_metadata(entry, "identity", 60000, 4)
            batch_one = expected_run_metadata(entry, "identity", 60000, 1)
            self.assertEqual(batch_four["fragment_batch_size_test"], 4)
            self.assertNotEqual(batch_four, batch_one)

    def test_grid_size_is_part_of_resume_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.py"
            checkpoint = root / "model_best.pth"
            config.write_text("x = 1\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            entry = dict(
                family="baseline",
                run_id="run",
                seed=None,
                checkpoint_kind="best",
                config_path=str(config),
                weight_path=str(checkpoint),
            )
            scale02 = expected_run_metadata(
                entry, "identity", 60000, 4, grid_size=0.02
            )
            scale04 = expected_run_metadata(
                entry, "identity", 60000, 4, grid_size=0.04
            )
            self.assertEqual(scale04["grid_size"], 0.04)
            self.assertNotEqual(scale02, scale04)

    def test_two_server_compact_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(
                protocol_version=PROTOCOL_VERSION,
                baseline_family="baseline",
                runs=[
                    dict(
                        family="baseline",
                        run_id="baseline-run",
                        seed=None,
                        checkpoints=["best"],
                    ),
                    dict(
                        family="v17",
                        run_id="v17-run",
                        seed=1,
                        checkpoints=["best"],
                    ),
                ],
            )

            def write_result(server_root, family, run_id, miou):
                output = server_root / "screen" / family / run_id / "best"
                output.mkdir(parents=True)
                metadata = dict(
                    protocol_version=PROTOCOL_VERSION,
                    protocol="identity",
                    point_max=60000,
                    fragment_batch_size_test=4,
                    family=family,
                    run_id=run_id,
                    seed=None if family == "baseline" else 1,
                    checkpoint_kind="best",
                    checkpoint_epoch=10,
                    config=dict(path=f"/{run_id}/config.py", size=1, mtime_ns=1),
                    checkpoint=dict(path=f"/{run_id}/best.pth", size=1, mtime_ns=1),
                )
                metrics = dict(
                    mIoU=miou,
                    mAcc=miou,
                    allAcc=miou,
                    num_rooms=1,
                    classes=[
                        dict(
                            index=0,
                            name="wall",
                            intersection=1,
                            union=1,
                            target=1,
                            iou=miou,
                            accuracy=miou,
                        )
                    ],
                    rooms=[
                        dict(
                            name="room",
                            points=1,
                            mIoU_present=miou,
                            mIoU_all=miou,
                            mAcc=miou,
                            allAcc=miou,
                            intersection=[1],
                            union=[1],
                            target=[1],
                        )
                    ],
                )
                (output / "run_meta.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                (output / "metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )

            server_a = root / "server-a"
            server_b = root / "server-b"
            write_result(server_a, "baseline", "baseline-run", 0.70)
            write_result(server_b, "v17", "v17-run", 0.71)
            output_root = root / "merged"
            args = SimpleNamespace(
                merge_input=[str(server_a), str(server_b)],
                output_root=str(output_root),
                selected_family=None,
                manifest=str(root / "manifest.json"),
            )
            merge_server_results(args, manifest)
            selection = json.loads(
                (output_root / "tta_selection.json").read_text(encoding="utf-8")
            )
            completeness = json.loads(
                (output_root / "screen" / "completeness.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selection["selected_family"], "v17")
            self.assertTrue(completeness["complete"])


class NeighborAuditTest(unittest.TestCase):
    def test_pool_labels_and_probs(self):
        labels = torch.tensor([0, 0, 1, 1])
        probs = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]
        )
        pooled_labels, purity, pooled_probs = pool_labels_and_probs(
            labels,
            probs,
            pool_sorting=torch.tensor([0, 1, 2, 3]),
            pool_ptr=torch.tensor([0, 2, 4]),
            num_classes=2,
        )
        self.assertTrue(torch.equal(pooled_labels, torch.tensor([0, 1])))
        self.assertTrue(torch.allclose(purity, torch.ones(2)))
        self.assertTrue(
            torch.allclose(pooled_probs, torch.tensor([[0.85, 0.15], [0.15, 0.85]]))
        )

    def test_expanded_neighbor_contamination(self):
        labels = torch.tensor([0, 0, 1])
        purity = torch.ones(3)
        probs = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]], dtype=torch.float32
        )
        neighbors = torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 2],
                [2, 1, 0],
            ]
        )
        active = torch.ones_like(neighbors, dtype=torch.bool)
        stats = compute_query_stats(
            labels=labels,
            purity=purity,
            probs=probs,
            neighbors=neighbors,
            active_mask=active,
            base_limit=2,
            purity_threshold=0.8,
            radius_scale=torch.ones(3, 1),
            da_meta_feat=torch.zeros(3, 4),
        )
        self.assertTrue(bool(stats["contamination_target"][0]))
        self.assertAlmostEqual(float(stats["base_gt_same_rate"][0]), 1.0)
        self.assertAlmostEqual(float(stats["extra_gt_same_rate"][0]), 0.0)

    def test_auc_and_spearman(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        targets = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc(scores, targets), 1.0)
        self.assertAlmostEqual(spearman(scores, scores), 1.0)


if __name__ == "__main__":
    unittest.main()
