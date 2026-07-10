"""Deterministic two-stage S3DIS evaluation for saved KPConvX experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.s3dis_fixed_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    atomic_write_json,
    build_checkpoint_entries,
    build_decision_report,
    canonical_test_cfg,
    collect_stage_results,
    discover_completed_entries,
    entry_output_dir,
    expected_run_metadata,
    load_manifest,
    metadata_matches,
    rank_families,
    select_tta_entries,
    select_tta_family,
    write_csv,
)


DEFAULT_MANIFEST = "configs/s3dis/eval/kpconvx_fixed_protocol_v1.json"
DEFAULT_OUTPUT = "exp/s3dis/kpconvx-fixed-protocol-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "discover",
            "preflight",
            "screen",
            "tta13",
            "bundle",
            "merge",
            "summarize",
            "all",
        ),
        default="discover",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--exp-root", default="exp")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--point-max", type=int, default=60000)
    parser.add_argument("--fallback-point-max", type=int, default=40000)
    parser.add_argument("--num-worker-test", type=int, default=2)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--selected-family", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-local", action="store_true")
    parser.add_argument("--max-rooms", type=int, default=None)
    parser.add_argument("--preflight-room", default="Area_5-conferenceRoom_3")
    parser.add_argument(
        "--merge-input",
        nargs="+",
        default=None,
        help="Server result roots to merge before local summarization.",
    )
    parser.add_argument(
        "--bundle-path",
        default=None,
        help="Compact zip path; defaults to <output-root>-compact.zip.",
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--weight", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--save-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--protocol", choices=("identity", "tta13"), default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--room-filter", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-meta-json", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def configure_worker_cfg(args):
    from pointcept.engines.defaults import default_setup
    from pointcept.utils.config import Config

    cfg = Config.fromfile(args.config_file)
    cfg.save_path = str(Path(args.save_path).resolve())
    cfg.weight = str(Path(args.weight).resolve())
    cfg.resume = False
    cfg.evaluate = True
    cfg.batch_size_test = 1
    cfg.fragment_batch_size_test = 1
    cfg.num_worker_test = max(int(args.num_worker_test), 0)
    cfg.num_worker_test_per_gpu = max(int(args.num_worker_test), 0)
    cfg.empty_cache = False
    cfg.test.export_metrics = True

    if args.data_root:
        cfg.data.test.data_root = args.data_root
    cfg.data.test.transform = [
        dict(type="CenterShift", apply_z=True),
        dict(type="NormalizeColor"),
    ]
    cfg.data.test.test_mode = True
    cfg.data.test.test_cfg = canonical_test_cfg(args.point_max, args.protocol)
    return default_setup(cfg)


def build_filtered_loader(cfg, room_filter=None, max_rooms=None):
    if not room_filter and not max_rooms:
        return None

    import torch

    from pointcept.datasets import build_dataset
    from pointcept.engines.test import SemSegTester

    dataset = build_dataset(cfg.data.test)
    data_list = list(dataset.data_list)
    if room_filter:
        selected = []
        for index in range(len(data_list)):
            if room_filter in dataset.get_data_name(index):
                selected.append(data_list[index])
        data_list = selected
    if max_rooms is not None:
        data_list = data_list[: max(int(max_rooms), 0)]
    if not data_list:
        raise RuntimeError(
            f"No S3DIS rooms matched room_filter={room_filter!r}, max_rooms={max_rooms!r}"
        )
    dataset.data_list = data_list

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


def run_worker(args):
    import torch

    from pointcept.engines.test import SemSegTester

    cfg = configure_worker_cfg(args)
    Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
    test_loader = build_filtered_loader(cfg, args.room_filter, args.max_rooms)
    tester = SemSegTester(cfg=cfg, test_loader=test_loader, verbose=True)

    start = time.time()
    metrics = tester.test()
    runtime_seconds = time.time() - start
    if metrics is None:
        raise RuntimeError("SemSegTester did not return metrics on the main process")

    checkpoint = torch.load(cfg.weight, map_location="cpu", weights_only=False)
    with open(args.run_meta_json, "r", encoding="utf-8") as f:
        run_meta = json.load(f)
    run_meta.update(
        checkpoint_epoch=checkpoint.get("epoch"),
        runtime_seconds=runtime_seconds,
        evaluated_rooms=metrics["num_rooms"],
        room_filter=args.room_filter,
        max_rooms=args.max_rooms,
    )
    atomic_write_json(Path(cfg.save_path) / "run_meta.json", run_meta)
    return 0


def local_inventory(args):
    manifest = load_manifest(args.manifest)
    entries, missing = build_checkpoint_entries(
        manifest,
        args.exp_root,
        allow_missing=not args.strict_local,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "available_entries.json", entries)
    atomic_write_json(output_root / "missing_entries.json", missing)
    atomic_write_json(
        output_root / "protocol.json",
        dict(
            protocol_version=PROTOCOL_VERSION,
            manifest=str(Path(args.manifest).resolve()),
            exp_root=str(Path(args.exp_root).resolve()),
            requested_point_max=args.point_max,
            fallback_point_max=args.fallback_point_max,
            screen_protocol="identity",
            tta_protocol="tta13",
            screen_expected_checkpoints=14,
            tta_family_limit=1,
            available_checkpoints=len(entries),
            missing_checkpoints=len(missing),
        ),
    )
    print(f"Available checkpoints: {len(entries)}")
    print(f"Missing checkpoints: {len(missing)}")
    for entry in entries:
        print(
            "  FOUND {family}/{run_id}/{checkpoint_kind}".format(**entry)
        )
    for entry in missing:
        print(
            "  MISS  {family}/{run_id}/{checkpoint_kind}: {reason}".format(**entry)
        )
    return manifest, entries, missing


def worker_command(args, entry, output_dir, protocol, point_max, room_filter=None):
    metadata = expected_run_metadata(entry, protocol, point_max)
    metadata_path = output_dir / "expected_run_meta.json"
    atomic_write_json(metadata_path, metadata)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config-file",
        entry["config_path"],
        "--weight",
        entry["weight_path"],
        "--save-path",
        str(output_dir),
        "--protocol",
        protocol,
        "--point-max",
        str(point_max),
        "--num-worker-test",
        str(args.num_worker_test),
        "--run-meta-json",
        str(metadata_path),
    ]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    if room_filter:
        command.extend(["--room-filter", room_filter])
    if args.max_rooms is not None:
        command.extend(["--max-rooms", str(args.max_rooms)])
    return command, metadata


def run_entry(args, entry, stage_dir, protocol, point_max, room_filter=None):
    output_dir = entry_output_dir(stage_dir, entry)
    expected = expected_run_metadata(entry, protocol, point_max)
    if metadata_matches(output_dir, expected) and not args.overwrite:
        print(f"SKIP completed: {output_dir}")
        return True
    expected_path = output_dir / "expected_run_meta.json"
    if output_dir.exists() and not args.overwrite:
        resumable = False
        if expected_path.is_file():
            with expected_path.open("r", encoding="utf-8") as f:
                previous_expected = json.load(f)
            resumable = previous_expected == expected
        if not resumable and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Existing output metadata does not match the requested protocol: "
                f"{output_dir}. Use --overwrite to discard its prediction cache."
            )
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command, _ = worker_command(
        args,
        entry,
        output_dir,
        protocol,
        point_max,
        room_filter=room_filter,
    )
    log_path = output_dir / "worker.log"
    print(f"RUN {entry['run_id']} {entry['checkpoint_kind']} -> {output_dir}")
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        print(f"FAILED ({completed.returncode}): {log_path}")
        return False
    print(f"DONE: {output_dir}")
    return True


def find_entry(entries, family, checkpoint_kind):
    matches = [
        entry
        for entry in entries
        if entry["family"] == family
        and entry["checkpoint_kind"] == checkpoint_kind
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {family}/{checkpoint_kind} checkpoint, found {len(matches)}"
        )
    return matches[0]


def run_preflight(args, entries):
    entry = find_entry(entries, "v17", "best")
    root = Path(args.output_root) / "preflight"
    for point_max in (args.point_max, args.fallback_point_max):
        stage_dir = root / str(point_max)
        success = run_entry(
            args,
            entry,
            stage_dir,
            protocol="identity",
            point_max=point_max,
            room_filter=args.preflight_room,
        )
        if success:
            decision = dict(
                protocol_version=PROTOCOL_VERSION,
                selected_point_max=point_max,
                room=args.preflight_room,
                checkpoint=entry,
            )
            atomic_write_json(Path(args.output_root) / "point_max_decision.json", decision)
            print(f"Preflight selected point_max={point_max}")
            return point_max
        log_path = entry_output_dir(stage_dir, entry) / "worker.log"
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        is_oom = "out of memory" in log_text.lower() or "cuda oom" in log_text.lower()
        if not is_oom:
            raise RuntimeError(f"Preflight failed for a non-OOM reason: {log_path}")
        print(f"OOM at point_max={point_max}; trying the protocol-wide fallback")
    raise RuntimeError("Both preflight point limits failed")


def expected_result_keys(manifest, families=None):
    keys = set()
    for run in manifest["runs"]:
        if families is not None and run["family"] not in families:
            continue
        for checkpoint_kind in run.get("checkpoints", ["best", "last"]):
            keys.add((run["family"], run["run_id"], run.get("seed"), checkpoint_kind))
    return keys


def entry_key(entry):
    return (
        entry["family"],
        entry["run_id"],
        entry.get("seed"),
        entry["checkpoint_kind"],
    )


def summarize_stage(args, manifest, stage_name, selected_family=None):
    stage_dir = Path(args.output_root) / stage_name
    entries = discover_completed_entries(stage_dir)
    if not entries:
        raise RuntimeError(f"No completed results found under {stage_dir}")
    checkpoint_rows, class_rows, room_rows = collect_stage_results(stage_dir, entries)
    protocol_versions = {row["protocol_version"] for row in checkpoint_rows}
    protocols = {row["protocol"] for row in checkpoint_rows}
    point_limits = {int(row["point_max"]) for row in checkpoint_rows}
    expected_protocol = "identity" if stage_name == "screen" else "tta13"
    if protocol_versions != {PROTOCOL_VERSION}:
        raise RuntimeError(f"Mixed protocol versions in {stage_dir}: {protocol_versions}")
    if protocols != {expected_protocol}:
        raise RuntimeError(f"Mixed protocols in {stage_dir}: {protocols}")
    if len(point_limits) != 1:
        raise RuntimeError(f"Mixed point_max values in {stage_dir}: {point_limits}")
    family_rows = rank_families(
        checkpoint_rows,
        baseline_family=manifest.get("baseline_family", "baseline"),
    )
    write_csv(stage_dir / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(stage_dir / "class_metrics.csv", class_rows)
    write_csv(stage_dir / "room_metrics.csv", room_rows)
    write_csv(stage_dir / "family_ranking.csv", family_rows)

    actual_keys = {entry_key(entry) for entry in entries}
    if stage_name == "screen":
        expected_keys = expected_result_keys(manifest)
        missing_keys = sorted(expected_keys - actual_keys, key=str)
        unexpected_keys = sorted(actual_keys - expected_keys, key=str)
        if not missing_keys and not unexpected_keys:
            selected_family = select_tta_family(
                family_rows,
                baseline_family=manifest.get("baseline_family", "baseline"),
            )
            atomic_write_json(
                Path(args.output_root) / "tta_selection.json",
                dict(
                    protocol_version=PROTOCOL_VERSION,
                    selected_family=selected_family,
                    selection_rule=(
                        "highest mean identity mIoU across model_best checkpoints"
                    ),
                    family_ranking=family_rows,
                ),
            )
        else:
            selected_family = None
    else:
        if selected_family is None:
            selection_path = Path(args.output_root) / "tta_selection.json"
            if not selection_path.is_file():
                raise RuntimeError("tta_selection.json is required to summarize tta13")
            with selection_path.open("r", encoding="utf-8") as f:
                selected_family = json.load(f)["selected_family"]
        expected_keys = expected_result_keys(
            manifest,
            families={manifest.get("baseline_family", "baseline"), selected_family},
        )
        missing_keys = sorted(expected_keys - actual_keys, key=str)
        unexpected_keys = sorted(actual_keys - expected_keys, key=str)
        if not missing_keys and not unexpected_keys:
            report, decision = build_decision_report(
                checkpoint_rows,
                class_rows,
                selected_family=selected_family,
                baseline_family=manifest.get("baseline_family", "baseline"),
            )
            (Path(args.output_root) / "decision_report.md").write_text(
                report, encoding="utf-8"
            )
            atomic_write_json(Path(args.output_root) / "decision.json", decision)

    missing_keys = sorted(expected_keys - actual_keys, key=str)
    unexpected_keys = sorted(actual_keys - expected_keys, key=str)
    completeness = dict(
        stage=stage_name,
        completed=len(actual_keys),
        expected=len(expected_keys),
        complete=not missing_keys and not unexpected_keys,
        missing=[list(key) for key in missing_keys],
        unexpected=[list(key) for key in unexpected_keys],
        selected_family=selected_family,
    )
    atomic_write_json(stage_dir / "completeness.json", completeness)
    print(
        f"Summarized {stage_name}: {len(actual_keys)}/{len(expected_keys)} results, "
        f"complete={completeness['complete']}"
    )
    return completeness


def json_files_equal(path_a, path_b):
    with open(path_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)
    return data_a == data_b


def merge_server_results(args, manifest):
    if not args.merge_input:
        raise ValueError("--merge-input is required for --stage merge")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "protocol.json",
        dict(
            protocol_version=PROTOCOL_VERSION,
            manifest=str(Path(args.manifest).resolve()),
            merged_from=[str(Path(path).resolve()) for path in args.merge_input],
            screen_protocol="identity",
            tta_protocol="tta13",
            tta_family_limit=1,
        ),
    )
    copied = 0
    skipped = 0

    point_max_decision = None
    for input_root_value in args.merge_input:
        input_root = Path(input_root_value)
        decision_path = input_root / "point_max_decision.json"
        if decision_path.is_file():
            with decision_path.open("r", encoding="utf-8") as f:
                current_decision = json.load(f)
            current_value = current_decision.get("selected_point_max")
            if point_max_decision is None:
                point_max_decision = current_value
            elif current_value != point_max_decision:
                raise RuntimeError(
                    "Server outputs used different point_max decisions: "
                    f"{point_max_decision} vs {current_value}"
                )

        for stage_name in ("screen", "tta13"):
            source_stage = input_root / stage_name
            for entry in discover_completed_entries(source_stage):
                source_dir = entry_output_dir(source_stage, entry)
                destination_dir = entry_output_dir(output_root / stage_name, entry)
                if destination_dir.exists():
                    same_meta = json_files_equal(
                        source_dir / "run_meta.json",
                        destination_dir / "run_meta.json",
                    )
                    same_metrics = json_files_equal(
                        source_dir / "metrics.json",
                        destination_dir / "metrics.json",
                    )
                    if not same_meta or not same_metrics:
                        raise RuntimeError(
                            f"Conflicting duplicate result: {destination_dir}"
                        )
                    skipped += 1
                    continue
                destination_dir.mkdir(parents=True, exist_ok=False)
                for filename in (
                    "metrics.json",
                    "run_meta.json",
                    "expected_run_meta.json",
                    "worker.log",
                ):
                    source_file = source_dir / filename
                    if source_file.is_file():
                        shutil.copy2(source_file, destination_dir / filename)
                copied += 1

    if point_max_decision is not None:
        atomic_write_json(
            output_root / "point_max_decision.json",
            dict(
                protocol_version=PROTOCOL_VERSION,
                selected_point_max=point_max_decision,
                merged_from=[str(Path(path)) for path in args.merge_input],
            ),
        )
    print(f"Merged results: copied={copied}, identical_skipped={skipped}")

    screen_dir = output_root / "screen"
    if screen_dir.is_dir():
        summarize_stage(args, manifest, "screen")
    tta_dir = output_root / "tta13"
    if tta_dir.is_dir():
        summarize_stage(args, manifest, "tta13", args.selected_family)


def bundle_compact_results(args):
    output_root = Path(args.output_root)
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    bundle_path = (
        Path(args.bundle_path)
        if args.bundle_path
        else output_root.with_name(output_root.name + "-compact.zip")
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_names = {
        "metrics.json",
        "run_meta.json",
        "expected_run_meta.json",
        "worker.log",
        "protocol.json",
        "point_max_decision.json",
        "available_entries.json",
        "missing_entries.json",
        "tta_selection.json",
        "checkpoint_summary.csv",
        "class_metrics.csv",
        "room_metrics.csv",
        "family_ranking.csv",
        "completeness.json",
        "failed_entries.json",
        "decision.json",
        "decision_report.md",
    }
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path.name in allowed_names:
                archive.write(path, arcname=str(path.relative_to(output_root)))
    print(f"Compact bundle saved: {bundle_path}")
    return bundle_path


def run_stage_entries(args, entries, stage_name, protocol, point_max):
    stage_dir = Path(args.output_root) / stage_name
    failures = []
    for entry in entries:
        if not run_entry(args, entry, stage_dir, protocol, point_max):
            failures.append(entry)
    if failures:
        atomic_write_json(stage_dir / "failed_entries.json", failures)
        raise RuntimeError(f"{len(failures)} checkpoint evaluations failed")


def read_selected_family(args):
    if args.selected_family:
        return args.selected_family
    selection_path = Path(args.output_root) / "tta_selection.json"
    if not selection_path.is_file():
        raise RuntimeError(
            "No --selected-family and no tta_selection.json. Merge and summarize "
            "screen results first."
        )
    with selection_path.open("r", encoding="utf-8") as f:
        return json.load(f)["selected_family"]


def main():
    args = parse_args()
    if args.worker:
        return run_worker(args)

    manifest = load_manifest(args.manifest)
    if args.stage == "bundle":
        bundle_compact_results(args)
        return 0
    if args.stage == "merge":
        merge_server_results(args, manifest)
        return 0
    if args.stage == "summarize":
        screen_dir = Path(args.output_root) / "screen"
        tta_dir = Path(args.output_root) / "tta13"
        if screen_dir.is_dir():
            summarize_stage(args, manifest, "screen")
        if tta_dir.is_dir():
            summarize_stage(args, manifest, "tta13", args.selected_family)
        return 0

    manifest, entries, _ = local_inventory(args)
    if args.stage == "discover":
        return 0
    if args.stage == "preflight":
        run_preflight(args, entries)
        return 0
    if args.stage in {"screen", "all"}:
        if not entries:
            print("No local checkpoints are available for screen on this server")
            return 0
        run_stage_entries(
            args,
            entries,
            stage_name="screen",
            protocol="identity",
            point_max=args.point_max,
        )
        # A local shard may be incomplete; summary is still useful for diagnostics.
        screen_completeness = summarize_stage(args, manifest, "screen")
        if args.stage == "all" and not screen_completeness["complete"]:
            raise RuntimeError(
                "Cannot continue to tta13 until screen shards are merged and complete"
            )

    if args.stage in {"tta13", "all"}:
        selected_family = read_selected_family(args)
        local_tta_entries = select_tta_entries(
            entries,
            selected_family,
            baseline_family=manifest.get("baseline_family", "baseline"),
        )
        if not local_tta_entries:
            print(
                "No local baseline or selected-family checkpoints are available "
                "for tta13 on this server"
            )
            return 0
        run_stage_entries(
            args,
            local_tta_entries,
            stage_name="tta13",
            protocol="tta13",
            point_max=args.point_max,
        )
        summarize_stage(args, manifest, "tta13", selected_family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
