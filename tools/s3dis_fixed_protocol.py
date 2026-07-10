"""Shared helpers for deterministic S3DIS checkpoint evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PROTOCOL_VERSION = "s3dis-fixed-v1"
RISK_CLASSES = ("door", "window", "column", "wall", "clutter")
POSITIVE_CLASSES = ("beam", "board", "table", "chair", "sofa")


def identity_augmentations():
    return [
        [
            dict(
                type="RandomRotateTargetAngle",
                angle=[0],
                axis="z",
                center=[0, 0, 0],
                p=1,
            )
        ]
    ]


def tta13_augmentations():
    augmentations = []
    for scale in (None, 0.95, 1.05):
        for angle in (0, 0.5, 1.0, 1.5):
            transforms = [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[angle],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                )
            ]
            if scale is not None:
                transforms.append(dict(type="RandomScale", scale=[scale, scale]))
            augmentations.append(transforms)
    augmentations.append([dict(type="RandomFlip", p=1)])
    return augmentations


def protocol_augmentations(protocol):
    if protocol == "identity":
        return identity_augmentations()
    if protocol == "tta13":
        return tta13_augmentations()
    raise ValueError(f"Unknown protocol: {protocol}")


def canonical_test_cfg(point_max, protocol):
    return dict(
        voxelize=dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="test",
        ),
        crop=dict(type="TestSphereCrop", point_max=int(point_max)),
        post_transform=[
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "index"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        aug_transform=protocol_augmentations(protocol),
    )


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Manifest protocol_version must be {PROTOCOL_VERSION!r}, got "
            f"{manifest.get('protocol_version')!r}"
        )
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest must contain a non-empty 'runs' list")
    return manifest


def resolve_run_dir(exp_root, run_id):
    exp_root = Path(exp_root)
    direct_candidates = [exp_root / run_id, exp_root / "s3dis" / run_id]
    candidates = [path for path in direct_candidates if path.is_dir()]
    if not candidates and exp_root.is_dir():
        candidates = [path for path in exp_root.rglob(run_id) if path.is_dir()]

    valid = []
    for path in candidates:
        if (path / "config.py").is_file() and (path / "model").is_dir():
            resolved = path.resolve()
            if resolved not in valid:
                valid.append(resolved)
    if not valid:
        raise FileNotFoundError(
            f"Cannot resolve experiment {run_id!r} under {exp_root}"
        )
    if len(valid) > 1:
        raise RuntimeError(
            f"Experiment {run_id!r} is ambiguous: "
            + ", ".join(str(path) for path in valid)
        )
    return valid[0]


def build_checkpoint_entries(manifest, exp_root, allow_missing=False):
    entries = []
    missing = []
    for run in manifest["runs"]:
        run_id = run["run_id"]
        family = run["family"]
        try:
            run_dir = resolve_run_dir(exp_root, run_id)
        except FileNotFoundError as error:
            if not allow_missing:
                raise
            for checkpoint_kind in run.get("checkpoints", ["best", "last"]):
                missing.append(
                    dict(
                        family=family,
                        run_id=run_id,
                        seed=run.get("seed"),
                        checkpoint_kind=checkpoint_kind,
                        reason=str(error),
                    )
                )
            continue
        config_path = run_dir / "config.py"
        checkpoints = run.get("checkpoints", ["best", "last"])
        for checkpoint_kind in checkpoints:
            if checkpoint_kind not in {"best", "last"}:
                raise ValueError(
                    f"Unsupported checkpoint kind {checkpoint_kind!r} for {run_id}"
                )
            weight_path = run_dir / "model" / f"model_{checkpoint_kind}.pth"
            if not weight_path.is_file():
                if not allow_missing:
                    raise FileNotFoundError(weight_path)
                missing.append(
                    dict(
                        family=family,
                        run_id=run_id,
                        seed=run.get("seed"),
                        checkpoint_kind=checkpoint_kind,
                        reason=f"Missing checkpoint: {weight_path}",
                    )
                )
                continue
            entries.append(
                dict(
                    family=family,
                    run_id=run_id,
                    seed=run.get("seed"),
                    checkpoint_kind=checkpoint_kind,
                    run_dir=str(run_dir),
                    config_path=str(config_path.resolve()),
                    weight_path=str(weight_path.resolve()),
                )
            )
    if allow_missing:
        return entries, missing
    return entries


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return value or "item"


def entry_output_dir(stage_dir, entry):
    return (
        Path(stage_dir)
        / safe_name(entry["family"])
        / safe_name(entry["run_id"])
        / safe_name(entry["checkpoint_kind"])
    )


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))


def expected_run_metadata(entry, protocol, point_max):
    return dict(
        protocol_version=PROTOCOL_VERSION,
        protocol=protocol,
        point_max=int(point_max),
        family=entry["family"],
        run_id=entry["run_id"],
        seed=entry.get("seed"),
        checkpoint_kind=entry["checkpoint_kind"],
        config=file_identity(entry["config_path"]),
        checkpoint=file_identity(entry["weight_path"]),
    )


def metadata_matches(output_dir, expected):
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "run_meta.json"
    if not metrics_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
    except (OSError, ValueError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics_from_counts(intersection, union, target):
    intersection = np.asarray(intersection, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    iou = intersection / (union + 1.0e-10)
    accuracy = intersection / (target + 1.0e-10)
    return dict(
        mIoU=float(iou.mean()),
        mAcc=float(accuracy.mean()),
        allAcc=float(intersection.sum() / (target.sum() + 1.0e-10)),
        iou=iou,
        accuracy=accuracy,
    )


def collect_stage_results(stage_dir, entries):
    checkpoint_rows = []
    class_rows = []
    room_rows = []
    for entry in entries:
        output_dir = entry_output_dir(stage_dir, entry)
        metrics_path = output_dir / "metrics.json"
        metadata_path = output_dir / "run_meta.json"
        if not metrics_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete result at {output_dir}")
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        common = dict(
            protocol_version=metadata["protocol_version"],
            protocol=metadata["protocol"],
            point_max=metadata["point_max"],
            family=entry["family"],
            run_id=entry["run_id"],
            seed=entry.get("seed"),
            checkpoint_kind=entry["checkpoint_kind"],
            epoch=metadata.get("checkpoint_epoch"),
        )
        checkpoint_rows.append(
            dict(
                **common,
                mIoU=metrics["mIoU"],
                mAcc=metrics["mAcc"],
                allAcc=metrics["allAcc"],
                num_rooms=metrics["num_rooms"],
                runtime_seconds=metadata.get("runtime_seconds"),
                output_dir=str(output_dir),
            )
        )
        for class_metric in metrics["classes"]:
            class_rows.append(dict(**common, **class_metric))
        for room_metric in metrics["rooms"]:
            room_rows.append(dict(**common, **room_metric))
    return checkpoint_rows, class_rows, room_rows


def discover_completed_entries(stage_dir):
    stage_dir = Path(stage_dir)
    entries = []
    seen = set()
    if not stage_dir.is_dir():
        return entries
    for metadata_path in sorted(stage_dir.rglob("run_meta.json")):
        metrics_path = metadata_path.parent / "metrics.json"
        if not metrics_path.is_file():
            continue
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        key = (
            metadata["family"],
            metadata["run_id"],
            metadata.get("seed"),
            metadata["checkpoint_kind"],
        )
        if key in seen:
            raise RuntimeError(f"Duplicate completed result for {key}")
        seen.add(key)
        entries.append(
            dict(
                family=metadata["family"],
                run_id=metadata["run_id"],
                seed=metadata.get("seed"),
                checkpoint_kind=metadata["checkpoint_kind"],
                config_path=metadata["config"]["path"],
                weight_path=metadata["checkpoint"]["path"],
            )
        )
    return entries


def rank_families(checkpoint_rows, baseline_family="baseline"):
    best_rows = [row for row in checkpoint_rows if row["checkpoint_kind"] == "best"]
    grouped = defaultdict(list)
    for row in best_rows:
        grouped[row["family"]].append(float(row["mIoU"]))

    rows = []
    for family, values in grouped.items():
        rows.append(
            dict(
                family=family,
                is_baseline=family == baseline_family,
                num_best_checkpoints=len(values),
                mean_mIoU=float(np.mean(values)),
                std_mIoU=float(np.std(values)),
                min_mIoU=float(np.min(values)),
                max_mIoU=float(np.max(values)),
            )
        )
    rows.sort(key=lambda row: row["mean_mIoU"], reverse=True)
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def select_tta_family(family_rows, baseline_family="baseline"):
    candidates = [row for row in family_rows if row["family"] != baseline_family]
    if not candidates:
        raise RuntimeError("No modified family is available for TTA selection")
    return max(candidates, key=lambda row: row["mean_mIoU"])["family"]


def select_tta_entries(entries, selected_family, baseline_family="baseline"):
    return [
        entry
        for entry in entries
        if entry["family"] in {baseline_family, selected_family}
    ]


def class_means(class_rows, family, checkpoint_kind="best"):
    grouped = defaultdict(list)
    for row in class_rows:
        if row["family"] == family and row["checkpoint_kind"] == checkpoint_kind:
            grouped[row["name"]].append(float(row["iou"]))
    return {name: float(np.mean(values)) for name, values in grouped.items()}


def build_decision_report(
    checkpoint_rows,
    class_rows,
    selected_family,
    baseline_family="baseline",
):
    baseline_values = [
        float(row["mIoU"])
        for row in checkpoint_rows
        if row["family"] == baseline_family and row["checkpoint_kind"] == "best"
    ]
    selected_values = [
        float(row["mIoU"])
        for row in checkpoint_rows
        if row["family"] == selected_family and row["checkpoint_kind"] == "best"
    ]
    baseline_mean = float(np.mean(baseline_values))
    selected_mean = float(np.mean(selected_values))
    delta = selected_mean - baseline_mean

    baseline_classes = class_means(class_rows, baseline_family)
    selected_classes = class_means(class_rows, selected_family)
    risk_deltas = {
        name: selected_classes.get(name, math.nan) - baseline_classes.get(name, math.nan)
        for name in RISK_CLASSES
    }
    strong_risks = [name for name, value in risk_deltas.items() if value < -0.03]
    credible = delta >= 0.005 and not strong_risks

    lines = [
        "# 固定协议复评决策报告",
        "",
        f"- 入围改进模型族：`{selected_family}`",
        f"- baseline model_best 均值：`{baseline_mean:.4f}`",
        f"- 改进模型族 model_best 均值：`{selected_mean:.4f}`",
        f"- mIoU 差值：`{delta:+.4f}`",
        f"- 可信提升阈值：`+0.005`",
        f"- 判定：`{'通过' if credible else '未通过'}`",
        "",
        "## 风险类别差值",
        "",
        "| 类别 | IoU 差值 |",
        "| --- | ---: |",
    ]
    for name in RISK_CLASSES:
        lines.append(f"| {name} | {risk_deltas[name]:+.4f} |")
    lines.extend(
        [
            "",
            "任何风险类别下降超过 `0.03` 都会触发强风险标记。",
            "",
            "该报告只使用固定协议结果；训练期 random-crop best 不参与判定。",
        ]
    )
    return "\n".join(lines) + "\n", dict(
        selected_family=selected_family,
        baseline_best_mean=baseline_mean,
        selected_best_mean=selected_mean,
        delta=delta,
        strong_risk_classes=strong_risks,
        credible_improvement=credible,
    )
