"""Audit V18 graph changes on S3DIS before starting a training run."""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader


CURRENT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURRENT))
sys.path.append(ROOT)

from data_handlers.scene_seg import SceneSegCollate, SceneSegSampler
from experiments.S3DIS.S3DIS_rooms import S3DIR_cfg, S3DIRDataset
from experiments.S3DIS.train_S3DIS import adjust_config, my_config
from utils.dual_support import DimensionlessSpacingScale, build_ring_neighbors


def finite_mean(values):
    finite = torch.isfinite(values)
    if finite.any():
        return float(values[finite].float().mean().item())
    return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data/s3dis")
    parser.add_argument("--profile", type=str, default="v18_diagnostic")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = my_config()
    cfg.data.update(S3DIR_cfg(cfg, dataset_path=args.dataset_path).data)
    cfg.model.enable_dual_support = True
    cfg.model.enable_da_radius = False
    cfg.model.dual_support_profile = args.profile
    cfg.model.dual_support_debug = False
    cfg.train.num_workers = 0
    cfg = adjust_config(cfg)

    dataset = S3DIRDataset(
        cfg,
        chosen_set="training",
        precompute_pyramid=True,
    )
    dataset.calib_batch(cfg, update_test=False)
    sampler = SceneSegSampler(dataset)
    sampler.N = max(int(args.samples), 1)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=SceneSegCollate,
        num_workers=0,
        pin_memory=False,
    )

    scale_mapper = DimensionlessSpacingScale(
        density_k=cfg.model.dual_support_density_k,
        percentile=cfg.model.dual_support_percentile,
        strength=cfg.model.dual_support_strength,
        power=cfg.model.dual_support_power,
    )
    stage_ranges = {
        int(stage): tuple(scale_range)
        for stage, scale_range in dict(
            cfg.model.dual_support_stage_ranges
        ).items()
    }
    ring_limits = {
        int(stage): int(limit)
        for stage, limit in dict(cfg.model.dual_support_ring_limits).items()
    }
    collected = {
        int(stage): {
            "spacing": [],
            "scale": [],
            "base_radius_count": [],
            "adaptive_radius_count": [],
            "added_count": [],
            "change_rate": [],
            "full_base_radius": [],
            "d_h_over_radius": [],
        }
        for stage in cfg.model.dual_support_stages
    }

    for batch in loader:
        in_dict = batch.in_dict
        for stage in cfg.model.dual_support_stages:
            stage = int(stage)
            layer = stage - 1
            points = in_dict.points[layer]
            candidates = in_dict.neighbors[layer]
            base_limit = int(cfg.model.base_neighbor_limits[layer])
            grid_size = cfg.model.in_sub_size * cfg.model.radius_scaling ** layer
            scale, spacing = scale_mapper(
                points,
                candidates[:, :base_limit],
                in_dict.lengths[layer],
                grid_size=grid_size,
                scale_range=stage_ranges[stage],
            )
            base_radius = (
                cfg.model.in_sub_size
                * cfg.model.kp_radius
                * cfg.model.radius_scaling ** layer
            )
            _, stats = build_ring_neighbors(
                points,
                candidates,
                scale,
                base_radius=base_radius,
                base_limit=base_limit,
                ring_limit=ring_limits[stage],
            )

            collected[stage]["spacing"].append(spacing.reshape(-1).cpu())
            collected[stage]["scale"].append(scale.reshape(-1).cpu())
            for key in [
                "base_radius_count",
                "adaptive_radius_count",
                "added_count",
                "change_rate",
                "full_base_radius",
                "d_h_over_radius",
            ]:
                collected[stage][key].append(stats[key].reshape(-1).cpu())

    report = {
        "profile": args.profile,
        "samples": max(int(args.samples), 1),
        "stage_numbering": "one_based",
        "stages": {},
    }
    for stage, values in collected.items():
        merged = {key: torch.cat(parts) for key, parts in values.items()}
        spacing = merged["spacing"].float()
        scale = merged["scale"].float()
        added = merged["added_count"].float()
        change = merged["change_rate"].float()
        full_base = merged["full_base_radius"].float()
        report["stages"][str(stage)] = {
            "num_queries": int(spacing.numel()),
            "spacing_q10": float(torch.quantile(spacing, 0.10).item()),
            "spacing_q50": float(torch.quantile(spacing, 0.50).item()),
            "spacing_q90": float(torch.quantile(spacing, 0.90).item()),
            "scale_min": float(scale.min().item()),
            "scale_mean": float(scale.mean().item()),
            "scale_max": float(scale.max().item()),
            "base_radius_count_mean": float(
                merged["base_radius_count"].float().mean().item()
            ),
            "adaptive_radius_count_mean": float(
                merged["adaptive_radius_count"].float().mean().item()
            ),
            "added_count_mean": float(added.mean().item()),
            "added_count_q90": float(torch.quantile(added, 0.90).item()),
            "change_rate_mean": float(change.mean().item()),
            "full_base_radius_ratio": float(full_base.mean().item()),
            "d_h_over_radius_mean": finite_mean(merged["d_h_over_radius"]),
            "mechanism_too_weak": bool(
                added.mean().item() < 1.0 or change.mean().item() < 0.10
            ),
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
