"""
Run validation-only alpha ablation for KPConvXV17 dual-support checkpoints.

This script intentionally uses data.val instead of SemSegTester/data.test, so it
is meant for fast go/no-go diagnosis rather than final S3DIS reporting.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the v17 config file or Pointcept config key.",
    )
    parser.add_argument("--weight", required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Eval-time multipliers for the dual-support residual.",
    )
    parser.add_argument(
        "--save-path",
        default="exp/v17_alpha_sweep",
        help="Directory for the CSV result and copied config.",
    )
    parser.add_argument(
        "--csv-name",
        default="v17_alpha_sweep.csv",
        help="CSV filename under save-path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2435054,
        help="Seed reused before each alpha so random validation crops are comparable.",
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=None,
        help="Override cfg.num_worker for the validation dataloader.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=None,
        help="Extra KEY=VALUE config overrides, same syntax as Pointcept tools.",
    )
    return parser.parse_args()


def parse_options(option_list):
    if not option_list:
        return None
    from pointcept.utils.config import DictAction

    namespace = argparse.Namespace(options=None)
    action = DictAction(option_strings=["--options"], dest="options")
    action(None, namespace, option_list)
    return namespace.options


def load_config(file_path, options):
    from pointcept.utils.config import Config

    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))
    if options is not None:
        cfg.merge_from_dict(options)
    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch
    return cfg


def load_checkpoint(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return checkpoint.get("epoch", None)


def build_val_loader(cfg, seed):
    from pointcept.datasets import build_dataset, collate_fn
    from pointcept.engines.defaults import worker_init_fn

    val_data = build_dataset(cfg.data.val)
    init_fn = (
        partial(
            worker_init_fn,
            num_workers=cfg.num_worker_per_gpu,
            rank=0,
            seed=seed,
        )
        if seed is not None
        else None
    )
    return torch.utils.data.DataLoader(
        val_data,
        batch_size=cfg.batch_size_val_per_gpu,
        shuffle=False,
        num_workers=cfg.num_worker_per_gpu,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
    )


def unwrap_backbone(model):
    if hasattr(model, "module"):
        model = model.module
    return getattr(model, "backbone", model)


def set_eval_alpha(model, alpha):
    backbone = unwrap_backbone(model)
    if not hasattr(backbone, "dual_support_eval_alpha"):
        raise AttributeError(
            "Model backbone does not expose dual_support_eval_alpha; "
            "use KPConvXV17 with the alpha-ablation patch."
        )
    backbone.dual_support_eval_alpha = float(alpha)


def eval_one_alpha(model, cfg, alpha, seed):
    from pointcept.utils.env import set_seed
    from pointcept.utils.misc import intersection_and_union_gpu

    set_seed(seed)
    set_eval_alpha(model, alpha)
    model.eval()

    loader = build_val_loader(cfg, seed)
    intersection_total = np.zeros(cfg.data.num_classes, dtype=np.float64)
    union_total = np.zeros(cfg.data.num_classes, dtype=np.float64)
    target_total = np.zeros(cfg.data.num_classes, dtype=np.float64)
    loss_values = []
    monitor_sums = defaultdict(float)
    monitor_counts = defaultdict(int)

    with torch.no_grad():
        for input_dict in loader:
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)
            output_dict = model(input_dict)
            output = output_dict["seg_logits"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            if "inverse" in input_dict:
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]

            intersection, union, target = intersection_and_union_gpu(
                pred.clone(),
                segment,
                cfg.data.num_classes,
                cfg.data.ignore_index,
            )
            intersection_total += intersection.cpu().numpy()
            union_total += union.cpu().numpy()
            target_total += target.cpu().numpy()

            if "loss" in output_dict:
                loss_values.append(float(output_dict["loss"].detach().cpu()))
            for key, value in output_dict.items():
                if key in {"loss", "seg_logits"}:
                    continue
                if torch.is_tensor(value) and value.numel() == 1:
                    monitor_sums[key] += float(value.detach().cpu())
                    monitor_counts[key] += 1

    iou_class = intersection_total / (union_total + 1.0e-10)
    acc_class = intersection_total / (target_total + 1.0e-10)
    result = {
        "alpha": float(alpha),
        "loss": float(np.mean(loss_values)) if loss_values else float("nan"),
        "mIoU": float(np.mean(iou_class)),
        "mAcc": float(np.mean(acc_class)),
        "allAcc": float(intersection_total.sum() / (target_total.sum() + 1.0e-10)),
    }
    for i, name in enumerate(cfg.data.names):
        result[f"iou_{name}"] = float(iou_class[i])
        result[f"acc_{name}"] = float(acc_class[i])
    for key in sorted(monitor_sums):
        result[key] = monitor_sums[key] / max(monitor_counts[key], 1)
    return result


def main():
    args = parse_args()
    from pointcept.engines.defaults import default_setup
    from pointcept.models import build_model
    from pointcept.utils.env import set_seed

    options = parse_options(args.options)
    cfg = load_config(args.config_file, options)
    cfg.save_path = args.save_path
    cfg.weight = args.weight
    cfg.evaluate = True
    cfg.seed = args.seed
    if args.num_worker is not None:
        cfg.num_worker = int(args.num_worker)
    cfg.model.backbone.dual_support_eval_alpha = None
    cfg = default_setup(cfg)

    Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = build_model(cfg.model).cuda()
    epoch = load_checkpoint(model, args.weight)
    print(f"Loaded checkpoint: {args.weight} (epoch={epoch})")

    rows = []
    for alpha in args.alphas:
        print(f">>> Evaluating alpha={alpha}")
        result = eval_one_alpha(model, cfg, alpha, args.seed)
        rows.append(result)
        print(
            "alpha={alpha:.3f} mIoU={mIoU:.4f} mAcc={mAcc:.4f} allAcc={allAcc:.4f}".format(
                **result
            )
        )

    csv_path = Path(cfg.save_path) / args.csv_name
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
