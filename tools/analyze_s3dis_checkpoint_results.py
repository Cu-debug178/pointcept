"""Analyze fixed-protocol best/last pairs and the scale04 epoch sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FOCUS_CLASSES = ("beam", "column", "window", "door", "board", "clutter")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(metrics):
    return {
        "mIoU": float(metrics["mIoU"]),
        "mAcc": float(metrics["mAcc"]),
        "allAcc": float(metrics["allAcc"]),
    }


def completed_results(stage_dir):
    results = {}
    for metadata_path in sorted(Path(stage_dir).rglob("run_meta.json")):
        metrics_path = metadata_path.parent / "metrics.json"
        if not metrics_path.is_file():
            continue
        metadata = load_json(metadata_path)
        key = (metadata["family"], metadata["run_id"], metadata.get("seed"))
        results.setdefault(key, {})[metadata["checkpoint_kind"]] = {
            "meta": metadata,
            "metrics": load_json(metrics_path),
        }
    return results


def analyze_pairs(stage_dir):
    pair_rows = []
    class_rows = []
    paired = {}
    for key, checkpoints in completed_results(stage_dir).items():
        if "best" not in checkpoints or "last" not in checkpoints:
            continue
        family, run_id, seed = key
        best = checkpoints["best"]
        last = checkpoints["last"]
        best_summary = metric_summary(best["metrics"])
        last_summary = metric_summary(last["metrics"])
        row = {
            "family": family,
            "run_id": run_id,
            "seed": seed,
            "best_epoch": best["meta"].get("checkpoint_epoch"),
            "last_epoch": last["meta"].get("checkpoint_epoch"),
        }
        for metric in ("mIoU", "mAcc", "allAcc"):
            row[f"best_{metric}"] = best_summary[metric]
            row[f"last_{metric}"] = last_summary[metric]
            row[f"delta_{metric}"] = last_summary[metric] - best_summary[metric]
        pair_rows.append(row)

        best_classes = {item["name"]: item for item in best["metrics"]["classes"]}
        last_classes = {item["name"]: item for item in last["metrics"]["classes"]}
        focus = {}
        for name in FOCUS_CLASSES:
            best_iou = float(best_classes[name]["iou"])
            last_iou = float(last_classes[name]["iou"])
            class_row = {
                "family": family,
                "run_id": run_id,
                "seed": seed,
                "class": name,
                "best_iou": best_iou,
                "last_iou": last_iou,
                "delta_iou": last_iou - best_iou,
            }
            class_rows.append(class_row)
            focus[name] = class_row
        paired[run_id] = {"overall": row, "focus_classes": focus}
    return pair_rows, class_rows, paired


def analyze_v16b(pair_rows):
    rows = [row for row in pair_rows if row["family"] == "v16b"]
    output = {"available_seeds": len(rows), "seeds": [row["seed"] for row in rows]}
    for checkpoint in ("best", "last"):
        for metric in ("mIoU", "mAcc", "allAcc"):
            values = [float(row[f"{checkpoint}_{metric}"]) for row in rows]
            if values:
                output[f"{checkpoint}_{metric}_mean"] = statistics.fmean(values)
                output[f"{checkpoint}_{metric}_std"] = statistics.pstdev(values)
    deltas = [float(row["delta_mIoU"]) for row in rows]
    if deltas:
        output["delta_mIoU_mean"] = statistics.fmean(deltas)
        output["delta_mIoU_std"] = statistics.pstdev(deltas)
    return output


def analyze_scale04(stage_dir):
    epoch_rows = []
    results = completed_results(stage_dir)
    for checkpoints in results.values():
        for checkpoint_kind, result in checkpoints.items():
            if not checkpoint_kind.startswith("epoch_"):
                continue
            epoch = int(checkpoint_kind.split("_", 1)[1])
            if epoch < 120:
                continue
            row = {"epoch": epoch, **metric_summary(result["metrics"])}
            epoch_rows.append(row)
    epoch_rows.sort(key=lambda row: row["epoch"])
    for index, row in enumerate(epoch_rows):
        row["delta_mIoU"] = (
            None if index == 0 else row["mIoU"] - epoch_rows[index - 1]["mIoU"]
        )
    monotonic = all(
        epoch_rows[index]["mIoU"] >= epoch_rows[index - 1]["mIoU"]
        for index in range(1, len(epoch_rows))
    )
    slope = None
    if len(epoch_rows) >= 2:
        slope = (epoch_rows[-1]["mIoU"] - epoch_rows[0]["mIoU"]) / (
            epoch_rows[-1]["epoch"] - epoch_rows[0]["epoch"]
        )
    return {"epochs": epoch_rows, "monotonic_mIoU": monotonic, "mIoU_slope": slope}


def render_markdown(pair_rows, class_rows, v16b, scale04):
    lines = [
        "# S3DIS Fixed-Protocol Checkpoint Analysis",
        "",
        "## Best vs Last",
        "",
        "| Family | Run | Best mIoU | Last mIoU | Last-Best | mAcc delta | OA delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(
            "| {family} | {run_id} | {best_mIoU:.4f} | {last_mIoU:.4f} | "
            "{delta_mIoU:+.4f} | {delta_mAcc:+.4f} | {delta_allAcc:+.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Focus-Class IoU Deltas",
            "",
            "| Run | Class | Best | Last | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            "| {run_id} | {class} | {best_iou:.4f} | {last_iou:.4f} | "
            "{delta_iou:+.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## v16b Seeds",
            "",
            f"Available seeds: {v16b.get('available_seeds', 0)}",
        ]
    )
    if "best_mIoU_mean" in v16b:
        lines.extend(
            [
                f"Best mIoU mean/std: {v16b['best_mIoU_mean']:.4f} / "
                f"{v16b['best_mIoU_std']:.4f}",
                f"Last mIoU mean/std: {v16b['last_mIoU_mean']:.4f} / "
                f"{v16b['last_mIoU_std']:.4f}",
                f"Last-Best mIoU mean/std: {v16b['delta_mIoU_mean']:+.4f} / "
                f"{v16b['delta_mIoU_std']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## scale04 Epoch 120-200",
            "",
            "| Epoch | mIoU | mAcc | OA | Delta mIoU |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scale04["epochs"]:
        delta = "-" if row["delta_mIoU"] is None else f"{row['delta_mIoU']:+.4f}"
        lines.append(
            f"| {row['epoch']} | {row['mIoU']:.4f} | {row['mAcc']:.4f} | "
            f"{row['allAcc']:.4f} | {delta} |"
        )
    lines.extend(
        [
            "",
            f"Monotonic mIoU: {scale04['monotonic_mIoU']}",
            f"mIoU slope per epoch: {scale04['mIoU_slope']:.6f}"
            if scale04["mIoU_slope"] is not None
            else "mIoU slope per epoch: unavailable",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--scale04-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pair_rows, class_rows, paired = analyze_pairs(Path(args.fixed_root) / "screen")
    v16b = analyze_v16b(pair_rows)
    scale04 = analyze_scale04(Path(args.scale04_root) / "screen")
    analysis = {
        "best_last_pairs": paired,
        "v16b": v16b,
        "scale04_epoch_120_200": scale04,
    }
    write_json(output_dir / "checkpoint_analysis.json", analysis)
    write_csv(output_dir / "best_last_comparison.csv", pair_rows)
    write_csv(output_dir / "focus_class_comparison.csv", class_rows)
    write_csv(output_dir / "scale04_epoch_120_200.csv", scale04["epochs"])
    report = render_markdown(pair_rows, class_rows, v16b, scale04)
    (output_dir / "checkpoint_analysis.md").write_text(report, encoding="utf-8")
    print(f"Analyzed best/last pairs: {len(pair_rows)}")
    print(f"Analysis saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
