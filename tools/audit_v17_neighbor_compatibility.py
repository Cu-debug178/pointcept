"""Audit whether v17 expanded-only stage4 neighbors introduce class pollution."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.s3dis_fixed_protocol import (  # noqa: E402
    POSITIVE_CLASSES,
    PROTOCOL_VERSION,
    RISK_CLASSES,
    atomic_write_json,
    build_checkpoint_entries,
    canonical_test_cfg,
    file_identity,
    load_manifest,
    write_csv,
)


DEFAULT_MANIFEST = "configs/s3dis/eval/kpconvx_fixed_protocol_v1.json"
DEFAULT_OUTPUT = "exp/s3dis/v17-neighbor-compatibility-audit"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--exp-root", default="exp")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--point-max", type=int, default=60000)
    parser.add_argument("--num-worker-test", type=int, default=2)
    parser.add_argument("--max-rooms", type=int, default=None)
    parser.add_argument("--purity-threshold", type=float, default=0.8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2435054)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def roc_auc(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & ((targets == 0) | (targets == 1))
    scores = scores[valid]
    targets = targets[valid]
    positives = targets == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(values_a, values_b):
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    if valid.sum() < 2:
        return float("nan")
    rank_a = rankdata(values_a[valid])
    rank_b = rankdata(values_b[valid])
    if np.std(rank_a) <= 0 or np.std(rank_b) <= 0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def pool_labels_and_probs(labels, probs, pool_sorting, pool_ptr, num_classes):
    valid = (labels >= 0) & (labels < num_classes)
    label_counts = probs.new_zeros((labels.shape[0], num_classes))
    if valid.any():
        label_counts[valid] = F.one_hot(
            labels[valid], num_classes=num_classes
        ).to(dtype=probs.dtype)
    pooled_counts, pooled_probs = pool_counts_and_probs(
        label_counts,
        probs,
        pool_sorting,
        pool_ptr,
    )
    valid_counts = pooled_counts.sum(dim=1)
    max_counts, pooled_labels = pooled_counts.max(dim=1)
    pooled_labels = pooled_labels.long()
    pooled_labels[valid_counts <= 0] = -1
    purity = max_counts / valid_counts.clamp(min=1.0)
    return pooled_labels, purity, pooled_probs


def pool_counts_and_probs(label_counts, probs, pool_sorting, pool_ptr):
    pool_sorting = pool_sorting.long()
    pool_ptr = pool_ptr.long()
    counts = pool_ptr[1:] - pool_ptr[:-1]
    num_groups = int(counts.shape[0])
    group_ids = torch.repeat_interleave(
        torch.arange(num_groups, device=label_counts.device), counts
    )
    pooled_counts = label_counts.new_zeros((num_groups, label_counts.shape[1]))
    pooled_counts.index_add_(0, group_ids, label_counts[pool_sorting])
    pooled_probs = probs.new_zeros((num_groups, probs.shape[1]))
    pooled_probs.index_add_(0, group_ids, probs[pool_sorting])
    pooled_probs = pooled_probs / counts.clamp(min=1).to(probs.dtype).unsqueeze(1)
    return pooled_counts, pooled_probs


def propagate_to_stage4(labels, probs, pyramid, num_classes):
    valid = (labels >= 0) & (labels < num_classes)
    label_counts = probs.new_zeros((labels.shape[0], num_classes))
    if valid.any():
        label_counts[valid] = F.one_hot(
            labels[valid], num_classes=num_classes
        ).to(dtype=probs.dtype)
    for pool_index in range(3):
        pool = pyramid.pools[pool_index]
        if not isinstance(pool, tuple):
            raise RuntimeError("The v17 audit requires grid_pool tuple mappings")
        label_counts, probs = pool_counts_and_probs(
            label_counts,
            probs,
            pool_sorting=pool[0],
            pool_ptr=pool[1],
        )
    valid_counts = label_counts.sum(dim=1)
    max_counts, labels = label_counts.max(dim=1)
    labels = labels.long()
    labels[valid_counts <= 0] = -1
    purity = max_counts / valid_counts.clamp(min=1.0)
    return labels, purity, probs


def safe_rate(numerator, denominator):
    result = numerator.float() / denominator.float().clamp(min=1.0)
    return torch.where(denominator > 0, result, torch.full_like(result, float("nan")))


def compute_query_stats(
    labels,
    purity,
    probs,
    neighbors,
    active_mask,
    base_limit,
    purity_threshold,
    radius_scale,
    da_meta_feat,
):
    num_points, neighbor_limit = neighbors.shape
    valid_neighbors = (neighbors >= 0) & (neighbors < num_points)
    query_indices = torch.arange(num_points, device=neighbors.device).unsqueeze(1)
    valid_neighbors = valid_neighbors & (neighbors != query_indices)
    safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))
    neighbor_labels = labels[safe_neighbors]
    neighbor_purity = purity[safe_neighbors]
    neighbor_pred = probs.argmax(dim=1)[safe_neighbors]

    query_valid = (labels >= 0) & (purity >= purity_threshold)
    neighbor_gt_valid = (
        valid_neighbors
        & (neighbor_labels >= 0)
        & (neighbor_purity >= purity_threshold)
    )
    slots = torch.arange(neighbor_limit, device=neighbors.device).unsqueeze(0)
    base_slot = slots < int(base_limit)
    extra_slot = slots >= int(base_limit)
    base_gt_mask = neighbor_gt_valid & base_slot
    extra_gt_mask = neighbor_gt_valid & extra_slot & active_mask
    base_valid_mask = valid_neighbors & base_slot
    extra_active_mask = valid_neighbors & extra_slot & active_mask
    extra_possible_mask = valid_neighbors & extra_slot

    same_gt = neighbor_labels == labels.unsqueeze(1)
    query_pred = probs.argmax(dim=1)
    same_pred = neighbor_pred == query_pred.unsqueeze(1)

    base_gt_count = base_gt_mask.sum(dim=1)
    base_gt_same = (base_gt_mask & same_gt).sum(dim=1)
    extra_gt_count = extra_gt_mask.sum(dim=1)
    extra_gt_same = (extra_gt_mask & same_gt).sum(dim=1)
    base_pred_count = base_valid_mask.sum(dim=1)
    base_pred_same = (base_valid_mask & same_pred).sum(dim=1)
    extra_pred_count = extra_active_mask.sum(dim=1)
    extra_pred_same = (extra_active_mask & same_pred).sum(dim=1)

    base_gt_rate = safe_rate(base_gt_same, base_gt_count)
    extra_gt_rate = safe_rate(extra_gt_same, extra_gt_count)
    base_pred_rate = safe_rate(base_pred_same, base_pred_count)
    extra_pred_rate = safe_rate(extra_pred_same, extra_pred_count)
    boundary = (
        base_gt_mask & (~same_gt) & query_valid.unsqueeze(1)
    ).any(dim=1)
    contamination_target = (
        extra_gt_mask & (~same_gt) & query_valid.unsqueeze(1)
    ).any(dim=1)
    audit_valid = query_valid & (extra_gt_count > 0)

    entropy = -(probs.clamp(min=1.0e-8) * probs.clamp(min=1.0e-8).log()).sum(dim=1)
    entropy = entropy / math.log(probs.shape[1])
    top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]

    result = dict(
        class_id=labels,
        purity=purity,
        ambiguous=(~query_valid),
        boundary=boundary,
        audit_valid=audit_valid,
        contamination_target=contamination_target,
        base_gt_count=base_gt_count,
        base_gt_same=base_gt_same,
        extra_gt_count=extra_gt_count,
        extra_gt_same=extra_gt_same,
        base_gt_same_rate=base_gt_rate,
        extra_gt_same_rate=extra_gt_rate,
        base_pred_count=base_pred_count,
        base_pred_same=base_pred_same,
        extra_pred_count=extra_pred_count,
        extra_pred_same=extra_pred_same,
        base_pred_same_rate=base_pred_rate,
        extra_pred_same_rate=extra_pred_rate,
        extra_active_count=extra_active_mask.sum(dim=1),
        extra_possible_count=extra_possible_mask.sum(dim=1),
        entropy=entropy,
        margin=margin,
        pred_base_disagreement=1.0 - base_pred_rate,
        pred_extra_disagreement=1.0 - extra_pred_rate,
        radius_scale=radius_scale.reshape(-1),
    )
    if da_meta_feat is not None and da_meta_feat.shape[1] >= 4:
        result.update(
            da_scale_delta=da_meta_feat[:, 0],
            rho_norm=da_meta_feat[:, 1],
            valid_ratio=da_meta_feat[:, 2],
            dist_cv=da_meta_feat[:, 3],
        )
    return result


def tensor_dict_to_numpy(data):
    return {
        key: value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
        for key, value in data.items()
    }


def concatenate_query_parts(parts):
    keys = sorted(parts[0])
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def weighted_rate(data, prefix, mask):
    count = np.asarray(data[f"{prefix}_count"])[mask].sum()
    same = np.asarray(data[f"{prefix}_same"])[mask].sum()
    return float(same / count) if count > 0 else float("nan")


def summarize_mask(data, mask):
    mask = np.asarray(mask, dtype=bool)
    queries = int(mask.sum())
    extra_possible = np.asarray(data["extra_possible_count"])[mask].sum()
    extra_active = np.asarray(data["extra_active_count"])[mask].sum()
    base_rate = weighted_rate(data, "base_gt", mask)
    extra_rate = weighted_rate(data, "extra_gt", mask)
    return dict(
        queries=queries,
        audit_valid_queries=int((mask & data["audit_valid"].astype(bool)).sum()),
        ambiguous_rate=(
            float(data["ambiguous"][mask].mean()) if queries else float("nan")
        ),
        boundary_ratio=(
            float(data["boundary"][mask].mean()) if queries else float("nan")
        ),
        extra_active_rate=(
            float(extra_active / extra_possible) if extra_possible > 0 else float("nan")
        ),
        original_gt_same_rate=base_rate,
        expanded_only_gt_same_rate=extra_rate,
        original_cross_class_rate=1.0 - base_rate,
        expanded_only_cross_class_rate=1.0 - extra_rate,
        contamination_gap=base_rate - extra_rate,
        original_pred_same_rate=weighted_rate(data, "base_pred", mask),
        expanded_only_pred_same_rate=weighted_rate(data, "extra_pred", mask),
    )


def bootstrap_auc(scores, targets, room_ids, samples, seed):
    room_ids = np.asarray(room_ids)
    unique_rooms = np.unique(room_ids)
    if unique_rooms.size < 2 or samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    room_indices = {room: np.flatnonzero(room_ids == room) for room in unique_rooms}
    for _ in range(samples):
        sampled_rooms = rng.choice(unique_rooms, size=unique_rooms.size, replace=True)
        indices = np.concatenate([room_indices[room] for room in sampled_rooms])
        auc = roc_auc(scores[indices], targets[indices])
        if np.isfinite(auc):
            values.append(max(auc, 1.0 - auc))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def predictor_rows(data, checkpoint_kind, bootstrap, seed):
    valid = data["audit_valid"].astype(bool)
    targets = data["contamination_target"][valid].astype(np.int64)
    severity = 1.0 - data["extra_gt_same_rate"][valid]
    rooms = data["room_id"][valid]
    predictors = {
        "entropy": data["entropy"][valid],
        "one_minus_margin": 1.0 - data["margin"][valid],
        "pred_base_disagreement": data["pred_base_disagreement"][valid],
        "pred_extra_disagreement": data["pred_extra_disagreement"][valid],
        "abs_radius_delta": np.abs(data["radius_scale"][valid] - 1.0),
        "rho_norm": data.get("rho_norm", np.full(valid.sum(), np.nan))[valid]
        if "rho_norm" in data
        else np.full(valid.sum(), np.nan),
        "one_minus_valid_ratio": 1.0 - data["valid_ratio"][valid]
        if "valid_ratio" in data
        else np.full(valid.sum(), np.nan),
        "dist_cv": data["dist_cv"][valid]
        if "dist_cv" in data
        else np.full(valid.sum(), np.nan),
    }
    rows = []
    for index, (name, scores) in enumerate(predictors.items()):
        raw_auc = roc_auc(scores, targets)
        oriented_auc = max(raw_auc, 1.0 - raw_auc) if np.isfinite(raw_auc) else raw_auc
        direction = "high_is_risk" if not np.isfinite(raw_auc) or raw_auc >= 0.5 else "low_is_risk"
        ci_low, ci_high = bootstrap_auc(
            scores,
            targets,
            rooms,
            samples=bootstrap,
            seed=seed + index,
        )
        rows.append(
            dict(
                checkpoint_kind=checkpoint_kind,
                predictor=name,
                samples=int(targets.shape[0]),
                positive_ratio=float(targets.mean()) if targets.size else float("nan"),
                raw_auc=raw_auc,
                oriented_auc=oriented_auc,
                direction=direction,
                bootstrap_ci_low=ci_low,
                bootstrap_ci_high=ci_high,
                spearman=spearman(scores, severity),
            )
        )
    return rows


def load_checkpoint(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(cleaned, strict=True)
    return checkpoint.get("epoch")


def build_audit_cfg(config_path, weight_path, args):
    from pointcept.engines.defaults import default_setup
    from pointcept.utils.config import Config

    cfg = Config.fromfile(config_path)
    cfg.save_path = str(Path(args.output_root).resolve())
    cfg.weight = str(Path(weight_path).resolve())
    cfg.seed = args.seed
    cfg.batch_size_test = 1
    cfg.num_worker_test = max(int(args.num_worker_test), 0)
    cfg.num_worker_test_per_gpu = max(int(args.num_worker_test), 0)
    if args.data_root:
        cfg.data.test.data_root = args.data_root
    cfg.data.test.transform = [
        dict(type="CenterShift", apply_z=True),
        dict(type="NormalizeColor"),
    ]
    cfg.data.test.test_mode = True
    cfg.data.test.test_cfg = canonical_test_cfg(args.point_max, "identity")
    cfg.data.test.test_cfg["voxel_part_limit"] = 1
    return default_setup(cfg)


def build_test_loader(cfg, max_rooms=None):
    from pointcept.datasets import build_dataset
    from pointcept.engines.test import SemSegTester

    dataset = build_dataset(cfg.data.test)
    if max_rooms is not None:
        dataset.data_list = list(dataset.data_list)[: max(int(max_rooms), 0)]
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_worker_test_per_gpu,
        pin_memory=cfg.num_worker_test_per_gpu > 0,
        collate_fn=SemSegTester.collate_fn,
    )
    if cfg.num_worker_test_per_gpu > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(**loader_kwargs)


def find_dual_conv(backbone):
    for module in backbone.encoder_4.modules():
        if getattr(module, "dual_support_enabled", False) and hasattr(
            module, "get_neighbors_influences"
        ):
            return module
    raise RuntimeError("Cannot find a dual-support DAKPConvX module in encoder_4")


def audit_checkpoint(entry, args):
    from pointcept.datasets import collate_fn
    from pointcept.models import build_model
    from pointcept.utils.env import set_seed

    checkpoint_kind = entry["checkpoint_kind"]
    output_dir = Path(args.output_root) / checkpoint_kind
    query_path = output_dir / "query_stats.npz"
    metadata_path = output_dir / "checkpoint_summary.json"
    expected_metadata = dict(
        protocol_version=PROTOCOL_VERSION,
        checkpoint_kind=checkpoint_kind,
        checkpoint=file_identity(entry["weight_path"]),
        config=file_identity(entry["config_path"]),
        point_max=args.point_max,
        voxel_part_limit=1,
        purity_threshold=args.purity_threshold,
    )
    if query_path.is_file() and metadata_path.is_file() and not args.overwrite:
        with metadata_path.open("r", encoding="utf-8") as f:
            actual_metadata = json.load(f)
        if all(
            actual_metadata.get(key) == value
            for key, value in expected_metadata.items()
        ):
            with np.load(query_path) as stored:
                return {key: stored[key] for key in stored.files}
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    cfg = build_audit_cfg(entry["config_path"], entry["weight_path"], args)
    model = build_model(cfg.model).cuda().eval()
    epoch = load_checkpoint(model, entry["weight_path"])
    backbone = model.backbone
    dual_conv = find_dual_conv(backbone)
    loader = build_test_loader(cfg, args.max_rooms)
    num_classes = int(cfg.data.num_classes)

    room_names = []
    query_parts = []
    start = time.time()
    with torch.no_grad():
        for room_id, batch in enumerate(loader):
            room = batch[0]
            fragments = room["fragment_list"]
            if torch.is_tensor(room["segment"]):
                room_segment = room["segment"].cpu().numpy()
            else:
                room_segment = np.asarray(room["segment"])
            room_name = room["name"]
            room_names.append(room_name)
            for fragment in fragments:
                input_dict = collate_fn([fragment])
                fragment_index = input_dict["index"].long()
                stage0_labels = torch.as_tensor(
                    room_segment,
                    dtype=torch.long,
                    device=fragment_index.device,
                )[fragment_index]
                for key, value in input_dict.items():
                    if torch.is_tensor(value):
                        input_dict[key] = value.cuda(non_blocking=True)
                stage0_labels = stage0_labels.cuda(non_blocking=True)

                output = model(input_dict)
                probs = F.softmax(output["seg_logits"].float(), dim=1)
                offset = input_dict["offset"].int()
                offset = torch.cat(
                    [offset.new_zeros(1), offset],
                    dim=0,
                )
                lengths = offset[1:] - offset[:-1]
                pyramid = backbone._build_pyramid(input_dict["coord"], lengths)
                labels4, purity4, probs4 = propagate_to_stage4(
                    stage0_labels,
                    probs,
                    pyramid,
                    num_classes,
                )
                points4 = pyramid.points[3]
                lengths4 = pyramid.lengths[3]
                neighbors4 = pyramid.neighbors[3]
                radius_scale, radius_meta = backbone.dual_support_radius(
                    points=points4,
                    neighbors=neighbors4,
                    lengths=lengths4,
                    scale_range=backbone._get_dual_support_scale_range(4),
                    return_meta=True,
                )
                influence, _, _ = dual_conv.get_neighbors_influences(
                    points4,
                    points4,
                    neighbors4,
                    da_radius_scale=radius_scale,
                    da_radius_min_keep=backbone._get_dual_support_min_keep(4),
                    da_radius_base_limit=backbone._get_dual_support_base_limit(4),
                )
                active_mask = influence > 0
                stats = compute_query_stats(
                    labels=labels4,
                    purity=purity4,
                    probs=probs4,
                    neighbors=neighbors4,
                    active_mask=active_mask,
                    base_limit=backbone._get_dual_support_base_limit(4),
                    purity_threshold=args.purity_threshold,
                    radius_scale=radius_scale,
                    da_meta_feat=radius_meta["feat"],
                )
                stats["room_id"] = labels4.new_full(labels4.shape, room_id)
                query_parts.append(tensor_dict_to_numpy(stats))
                del input_dict, output, probs, pyramid, influence, stats

    if not query_parts:
        raise RuntimeError("The audit produced no stage4 query samples")
    data = concatenate_query_parts(query_parts)
    np.savez_compressed(query_path, **data)
    atomic_write_json(output_dir / "room_names.json", room_names)
    atomic_write_json(
        metadata_path,
        dict(
            **expected_metadata,
            checkpoint_epoch=epoch,
            num_rooms=len(room_names),
            num_queries=int(data["class_id"].shape[0]),
            runtime_seconds=time.time() - start,
        ),
    )
    del model, loader, backbone, dual_conv
    gc.collect()
    torch.cuda.empty_cache()
    return data


def class_and_room_rows(data, checkpoint_kind, class_names, room_names):
    class_rows = []
    valid_class = (data["class_id"] >= 0) & (data["class_id"] < len(class_names))
    for class_index, class_name in enumerate(class_names):
        class_mask = valid_class & (data["class_id"] == class_index)
        for stratum, stratum_mask in (
            ("all", np.ones_like(class_mask, dtype=bool)),
            ("boundary", data["boundary"].astype(bool)),
            ("interior", ~data["boundary"].astype(bool)),
        ):
            class_rows.append(
                dict(
                    checkpoint_kind=checkpoint_kind,
                    class_index=class_index,
                    class_name=class_name,
                    stratum=stratum,
                    **summarize_mask(data, class_mask & stratum_mask),
                )
            )
    for group_name, group_classes in (
        ("risk_group", RISK_CLASSES),
        ("positive_group", POSITIVE_CLASSES),
    ):
        indices = [class_names.index(name) for name in group_classes]
        group_mask = valid_class & np.isin(data["class_id"], indices)
        for stratum, stratum_mask in (
            ("all", np.ones_like(group_mask, dtype=bool)),
            ("boundary", data["boundary"].astype(bool)),
            ("interior", ~data["boundary"].astype(bool)),
        ):
            class_rows.append(
                dict(
                    checkpoint_kind=checkpoint_kind,
                    class_index=-1,
                    class_name=group_name,
                    stratum=stratum,
                    **summarize_mask(data, group_mask & stratum_mask),
                )
            )

    room_rows = []
    for room_id, room_name in enumerate(room_names):
        room_mask = data["room_id"] == room_id
        for stratum, stratum_mask in (
            ("all", np.ones_like(room_mask, dtype=bool)),
            ("boundary", data["boundary"].astype(bool)),
            ("interior", ~data["boundary"].astype(bool)),
        ):
            room_rows.append(
                dict(
                    checkpoint_kind=checkpoint_kind,
                    room_id=room_id,
                    room_name=room_name,
                    stratum=stratum,
                    **summarize_mask(data, room_mask & stratum_mask),
                )
            )
    return class_rows, room_rows


def checkpoint_gate(class_rows, predictor_rows_for_checkpoint, checkpoint_kind):
    lookup = {
        row["class_name"]: row
        for row in class_rows
        if row["checkpoint_kind"] == checkpoint_kind
        and row["stratum"] == "all"
    }
    risk = lookup["risk_group"]
    positive = lookup["positive_group"]
    qualified_predictors = [
        row["predictor"]
        for row in predictor_rows_for_checkpoint
        if row["checkpoint_kind"] == checkpoint_kind
        and row["oriented_auc"] >= 0.70
        and row["bootstrap_ci_low"] >= 0.65
    ]
    return dict(
        checkpoint_kind=checkpoint_kind,
        risk_gap=risk["contamination_gap"],
        positive_gap=positive["contamination_gap"],
        extra_active_rate=risk["extra_active_rate"],
        risk_audit_valid_queries=risk["audit_valid_queries"],
        risk_condition=bool(risk["contamination_gap"] >= 0.10),
        positive_condition=bool(positive["contamination_gap"] <= 0.03),
        sample_condition=bool(
            risk["extra_active_rate"] >= 0.10
            and risk["audit_valid_queries"] >= 1000
        ),
        qualified_predictors=qualified_predictors,
    )


def build_report(summary):
    lines = [
        "# v17 expanded-neighbor compatibility audit",
        "",
        f"最终判定：`{'GO' if summary['selective_support_go'] else 'NO-GO'}`",
        "",
        "只有 best 和 last 都满足污染分组条件，并存在同一个非 GT predictor "
        "在两者上达到 AUC/CI 阈值，才允许后续 selective support 训练。",
        "",
        "| checkpoint | risk gap | positive gap | extra active | valid queries | 条件通过 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for gate in summary["checkpoint_gates"]:
        passed = (
            gate["risk_condition"]
            and gate["positive_condition"]
            and gate["sample_condition"]
        )
        lines.append(
            "| {checkpoint_kind} | {risk_gap:.4f} | {positive_gap:.4f} | "
            "{extra_active_rate:.4f} | {risk_audit_valid_queries} | {passed} |".format(
                passed="是" if passed else "否", **gate
            )
        )
    lines.extend(
        [
            "",
            "共同达标 predictor："
            + (
                ", ".join(summary["common_qualified_predictors"])
                if summary["common_qualified_predictors"]
                else "无"
            ),
            "",
            "本审计中的 GT 只用于判断污染是否真实存在，不进入模型预测。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    entries, missing = build_checkpoint_entries(
        manifest,
        args.exp_root,
        allow_missing=True,
    )
    v17_entries = [
        entry
        for entry in entries
        if entry["family"] == "v17"
        and entry["checkpoint_kind"] in {"best", "last"}
    ]
    if len(v17_entries) != 2:
        missing_v17 = [item for item in missing if item["family"] == "v17"]
        raise RuntimeError(
            "The audit must run on the server containing v17 best and last. "
            f"Found {len(v17_entries)} checkpoints; missing={missing_v17}"
        )

    from pointcept.utils.config import Config

    cfg = Config.fromfile(v17_entries[0]["config_path"])
    class_names = list(cfg.data.names)
    all_class_rows = []
    all_room_rows = []
    all_predictor_rows = []
    gates = []
    for entry in sorted(v17_entries, key=lambda item: item["checkpoint_kind"]):
        checkpoint_kind = entry["checkpoint_kind"]
        data = audit_checkpoint(entry, args)
        room_names_path = Path(args.output_root) / checkpoint_kind / "room_names.json"
        with room_names_path.open("r", encoding="utf-8") as f:
            room_names = json.load(f)
        class_rows, room_rows = class_and_room_rows(
            data,
            checkpoint_kind,
            class_names,
            room_names,
        )
        predictors = predictor_rows(
            data,
            checkpoint_kind,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        all_class_rows.extend(class_rows)
        all_room_rows.extend(room_rows)
        all_predictor_rows.extend(predictors)
        gates.append(checkpoint_gate(class_rows, predictors, checkpoint_kind))
        del data
        gc.collect()

    output_root = Path(args.output_root)
    write_csv(output_root / "class_contamination.csv", all_class_rows)
    write_csv(output_root / "room_contamination.csv", all_room_rows)
    write_csv(output_root / "predictor_auc.csv", all_predictor_rows)

    predictor_sets = [set(gate["qualified_predictors"]) for gate in gates]
    common_predictors = sorted(set.intersection(*predictor_sets)) if predictor_sets else []
    group_conditions = all(
        gate["risk_condition"]
        and gate["positive_condition"]
        and gate["sample_condition"]
        for gate in gates
    )
    summary = dict(
        protocol_version=PROTOCOL_VERSION,
        checkpoint_gates=gates,
        common_qualified_predictors=common_predictors,
        selective_support_go=bool(group_conditions and common_predictors),
        thresholds=dict(
            risk_gap_min=0.10,
            positive_gap_max=0.03,
            extra_active_rate_min=0.10,
            risk_valid_queries_min=1000,
            predictor_auc_min=0.70,
            predictor_ci_low_min=0.65,
        ),
    )
    atomic_write_json(output_root / "audit_summary.json", summary)
    (output_root / "audit_report.md").write_text(
        build_report(summary), encoding="utf-8"
    )
    print(f"Audit decision: {'GO' if summary['selective_support_go'] else 'NO-GO'}")
    print(f"Saved: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
