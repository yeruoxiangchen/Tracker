#!/usr/bin/env python3
"""Run the frozen 15k/20k/25k official SLat evaluation matrix on five GPUs.

This module is orchestration only.  It delegates every scientific computation
to the existing official GT-support and Native-SS predicted-support evaluators.
It deliberately keeps the predicted-support lane on Dev indices [16, 64): the
first 16 Dev objects were used for Native-SS CFG calibration and are therefore
not part of the frozen held-out Dev48 deployment report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable


FORMAT = "pose_point_depth_mv.proobjaverse_slat_trajectory_eval_5gpu.v1"
DEFAULT_STEPS = (15000, 20000, 25000)
GT_RANGES = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
PREDICTED_RANGES = ((16, 26), (26, 36), (36, 46), (46, 55), (55, 64))
GROUPS = ("train64_gt", "dev64_gt", "dev48_predicted")


@dataclass(frozen=True)
class Job:
    step: int
    group: str
    shard: int
    start: int
    end: int
    gpu: int
    checkpoint: Path
    output_dir: Path
    log_path: Path


def parse_int_csv(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{name} values must be unique")
    return values


def validate_partition(
    ranges: Iterable[tuple[int, int]], *, expected_start: int, expected_end: int
) -> None:
    cursor = expected_start
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError(
                f"non-contiguous partition at [{start},{end}); expected start={cursor}"
            )
        cursor = end
    if cursor != expected_end:
        raise ValueError(f"partition ends at {cursor}, expected {expected_end}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_path(checkpoint_root: Path, step: int) -> Path:
    return checkpoint_root / f"step_{step:06d}.pt"


def group_ranges(group: str) -> tuple[tuple[int, int], ...]:
    if group in {"train64_gt", "dev64_gt"}:
        return GT_RANGES
    if group == "dev48_predicted":
        return PREDICTED_RANGES
    raise ValueError(f"unknown group: {group}")


def build_jobs(
    *,
    steps: tuple[int, ...],
    gpus: tuple[int, ...],
    checkpoint_root: Path,
    output_root: Path,
) -> list[list[Job]]:
    if len(gpus) != 5:
        raise ValueError("exactly five GPUs are required")
    validate_partition(GT_RANGES, expected_start=0, expected_end=64)
    validate_partition(PREDICTED_RANGES, expected_start=16, expected_end=64)
    waves: list[list[Job]] = []
    for step in steps:
        checkpoint = checkpoint_path(checkpoint_root, step)
        for group in GROUPS:
            wave: list[Job] = []
            for shard, ((start, end), gpu) in enumerate(zip(group_ranges(group), gpus)):
                base = output_root / f"step_{step:06d}" / group
                wave.append(
                    Job(
                        step=step,
                        group=group,
                        shard=shard,
                        start=start,
                        end=end,
                        gpu=gpu,
                        checkpoint=checkpoint,
                        output_dir=base / f"shard{shard}_{start}_{end}",
                        log_path=(
                            output_root
                            / "logs"
                            / f"step_{step:06d}_{group}_shard{shard}_gpu{gpu}.log"
                        ),
                    )
                )
            waves.append(wave)
    return waves


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default="/home/zjr/Tracker")
    parser.add_argument(
        "--checkpoint_root",
        default=(
            "/data/zjr/slat_train2000_trajectory_archives/"
            "slat_train2000_trajectory_step10000_25000_strict_fix1_v1/checkpoints"
        ),
    )
    parser.add_argument(
        "--source_root",
        default="/data/zjr/proobjaverse_official_slat_train2000_20260813_v1",
    )
    parser.add_argument(
        "--official_ss_root",
        default="/data/zjr/proobjaverse_official_native_ss_train2000_20260815_v1",
    )
    parser.add_argument(
        "--training_native_ss_report",
        default=(
            "/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/"
            "ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json"
        ),
        help="Native-SS report frozen into the SLat training identity (GT lane only)",
    )
    parser.add_argument(
        "--predicted_native_ss_report",
        default=(
            "/data/zjr/proobjaverse_official_native_ss_train2000_20260815_v1/"
            "dev64_step2000_eval16_64_seed424344_6gpu_v1/aggregate_v1/report.json"
        ),
        help="official Native-SS step2000 EMA/CFG5 held-out Dev48 deployment",
    )
    parser.add_argument(
        "--stock_slat_freeze",
        default=(
            "/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/"
            "stock_slat_freeze_v2.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        default=(
            "/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/"
            "eval_trajectory_step15000_20000_25000_seed424344_5gpu_strict_fix1_v1"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--steps", default="15000,20000,25000")
    parser.add_argument("--gpus", default="3,4,5,6,7")
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--min_free_gpu_mib", type=int, default=22000)
    parser.add_argument("--skip_gpu_free_check", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument(
        "--print_plan_only",
        action="store_true",
        help="print the 9-wave/45-worker plan without reading artifacts or GPUs",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="run artifact preflight and print commands without creating outputs",
    )
    return parser


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).expanduser().resolve()
    official_ss_root = Path(args.official_ss_root).expanduser().resolve()
    steps = parse_int_csv(args.steps, name="steps")
    if steps != tuple(sorted(steps)) or any(step <= 0 for step in steps):
        raise ValueError("steps must be positive and strictly increasing")
    if steps != DEFAULT_STEPS:
        raise ValueError(f"this frozen protocol requires steps={DEFAULT_STEPS}")
    gpus = parse_int_csv(args.gpus, name="gpus")
    if len(gpus) != 5 or any(gpu < 0 for gpu in gpus):
        raise ValueError("gpus must contain exactly five distinct non-negative ids")
    seeds = parse_int_csv(args.joint_seeds, name="joint_seeds")
    if seeds != (42, 43, 44):
        raise ValueError("this frozen protocol requires joint_seeds=42,43,44")
    if int(args.surface_samples) != 20000:
        raise ValueError("this frozen protocol requires surface_samples=20000")
    if int(args.bootstrap_samples) != 5000:
        raise ValueError("this frozen protocol requires bootstrap_samples=5000")
    return {
        "project_root": Path(args.project_root).expanduser().resolve(),
        "checkpoint_root": Path(args.checkpoint_root).expanduser().resolve(),
        "source_root": source_root,
        "official_ss_root": official_ss_root,
        "training_native_ss_report": Path(args.training_native_ss_report)
        .expanduser()
        .resolve(),
        "predicted_native_ss_report": Path(args.predicted_native_ss_report)
        .expanduser()
        .resolve(),
        "stock_slat_freeze": Path(args.stock_slat_freeze).expanduser().resolve(),
        "output_root": Path(args.output_root).expanduser().resolve(),
        "python": Path(args.python).expanduser().resolve(),
        "pretrained": str(args.pretrained),
        "steps": steps,
        "gpus": gpus,
        "seeds": seeds,
        "surface_samples": int(args.surface_samples),
        "bootstrap_samples": int(args.bootstrap_samples),
        "min_free_gpu_mib": int(args.min_free_gpu_mib),
        "train_cache": source_root / "cache_train2000_protocol2128_views8_v1",
        "dev_cache": source_root / "cache_dev64_protocol2128_views8_v1",
    }


def printable_plan(config: dict[str, Any]) -> dict[str, Any]:
    waves = build_jobs(
        steps=config["steps"],
        gpus=config["gpus"],
        checkpoint_root=config["checkpoint_root"],
        output_root=config["output_root"],
    )
    return {
        "format": FORMAT,
        "wave_count": len(waves),
        "worker_count": sum(len(wave) for wave in waves),
        "execution": "9 sequential waves; 5 concurrent single-GPU shards per wave",
        "groups": {
            "train64_gt": {
                "support": "official GT SLat",
                "object_slice": [0, 64],
                "training_overlap": True,
            },
            "dev64_gt": {
                "support": "official GT SLat",
                "object_slice": [0, 64],
                "training_overlap": False,
            },
            "dev48_predicted": {
                "support": "official Native-SS step2000 EMA CFG5 predicted support",
                "parent_split": "Dev64",
                "object_slice": [16, 64],
                "excludes_cfg_calibration_indices": [0, 16],
            },
        },
        "waves": [
            {
                "step": wave[0].step,
                "group": wave[0].group,
                "jobs": [
                    {
                        "shard": job.shard,
                        "object_slice": [job.start, job.end],
                        "gpu": job.gpu,
                        "output_dir": str(job.output_dir),
                    }
                    for job in wave
                ],
            }
            for wave in waves
        ],
    }


def check_gpu_free(gpus: tuple[int, ...], minimum_mib: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gpu in gpus:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=index,name,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row for GPU {gpu}: {result.stdout!r}")
        free_mib = int(fields[2])
        row = {
            "index": int(fields[0]),
            "name": fields[1],
            "free_mib": free_mib,
            "total_mib": int(fields[3]),
            "utilization_percent": int(fields[4]),
        }
        rows.append(row)
        if free_mib < minimum_mib:
            raise RuntimeError(
                f"GPU {gpu} has only {free_mib} MiB free; require {minimum_mib} MiB"
            )
    return rows


def artifact_preflight(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support import (
        _upstream_binding,
    )
    from pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support_cross_host import (
        validate_checkpoint_native_ss_binding_relocation,
    )
    from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset
    from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
    from pose_point_depth_mv.proobjaverse_official_ss import (
        load_official_native_ss_deployment,
    )

    required = {
        "python": config["python"],
        "train_slat_manifest": config["train_cache"] / "slat_manifest.json",
        "train_lifting_manifest": config["train_cache"] / "lifting_manifest.json",
        "dev_slat_manifest": config["dev_cache"] / "slat_manifest.json",
        "dev_lifting_manifest": config["dev_cache"] / "lifting_manifest.json",
        "training_native_ss_report": config["training_native_ss_report"],
        "predicted_native_ss_report": config["predicted_native_ss_report"],
        "stock_slat_freeze": config["stock_slat_freeze"],
    }
    code_files = {
        "orchestrator": config["project_root"]
        / "pose_point_depth_mv/run_proobjaverse_slat_trajectory_eval_5gpu.py",
        "gt_worker": config["project_root"]
        / "pose_point_depth_mv/evaluate_proobjaverse_official_slat_gt_support.py",
        "gt_cross_host": config["project_root"]
        / "pose_point_depth_mv/evaluate_proobjaverse_official_slat_gt_support_cross_host.py",
        "gt_aggregate": config["project_root"]
        / "pose_point_depth_mv/aggregate_proobjaverse_official_slat_gt_support.py",
        "predicted_worker_aggregate": config["project_root"]
        / "pose_point_depth_mv/evaluate_proobjaverse_official_native_ss_stock_slat.py",
    }
    for label, path in {**required, **code_files}.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing {label}: {path}")

    train_dataset = NativeConditionSLatDataset(
        required["train_slat_manifest"], required["train_lifting_manifest"], indices="all"
    )
    dev_dataset = NativeConditionSLatDataset(
        required["dev_slat_manifest"], required["dev_lifting_manifest"], indices="all"
    )
    if len(train_dataset) != 2000 or len(dev_dataset) != 64:
        raise RuntimeError(
            f"dataset sizes differ: train={len(train_dataset)} dev={len(dev_dataset)}"
        )
    if train_dataset.config.get("target_source", {}).get("split") != "train":
        raise RuntimeError("Train2000 cache is not the official train split")
    if dev_dataset.config.get("target_source", {}).get("split") != "dev":
        raise RuntimeError("Dev64 cache is not the official dev split")
    train_protocol = train_dataset.config["target_source"]["protocol_sha256"]
    dev_protocol = dev_dataset.config["target_source"]["protocol_sha256"]
    if train_protocol != dev_protocol:
        raise RuntimeError("Train2000/Dev64 official target protocols differ")

    _, training_deployment = load_no_vggt_ss_evidence(
        required["training_native_ss_report"]
    )
    for label, dataset in (("train", train_dataset), ("dev", dev_dataset)):
        if dataset.config.get("native_ss_deployment") != training_deployment:
            raise RuntimeError(f"{label} cache/training Native-SS deployment differs")
    runtime_upstream = _upstream_binding(training_deployment)

    official_payload, official_binding = load_official_native_ss_deployment(
        required["predicted_native_ss_report"]
    )
    if official_payload.get("passed") is not True:
        raise RuntimeError("official Native-SS held-out deployment did not pass")
    if int(official_payload.get("object_count", -1)) != 48:
        raise RuntimeError("official Native-SS deployment is not Dev48")
    expected_dev48_uids = {
        str(row["object_uid"]) for row in dev_dataset.rows[16:64]
    }
    evidence_uids = {str(value) for value in official_payload.get("object_uids", [])}
    if evidence_uids != expected_dev48_uids:
        raise RuntimeError("official Native-SS evidence is not exactly Dev indices [16,64)")
    expected_official_binding = {
        "checkpoint_step": 2000,
        "weights": "ema",
        "cfg_strength": 5.0,
        "steps": 25,
        "cfg_interval": [0.5, 1.0],
        "amp_dtype": "bf16",
        "false_checks": [],
    }
    for key, expected in expected_official_binding.items():
        if official_binding.get(key) != expected:
            raise RuntimeError(
                f"official Native-SS binding differs at {key}: "
                f"{official_binding.get(key)!r} != {expected!r}"
            )

    checkpoint_rows = []
    for step in config["steps"]:
        path = checkpoint_path(config["checkpoint_root"], step)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing checkpoint: {path}")
        checkpoint_hash = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or int(checkpoint.get("step", -1)) != step:
            raise RuntimeError(f"checkpoint step mismatch: {path}")
        saved = checkpoint.get("data_identity", {}).get("native_ss")
        summary = checkpoint.get("model_summary", {}).get("upstream_native_ss")
        if not isinstance(saved, dict) or saved != summary:
            raise RuntimeError(f"checkpoint Native-SS identity is inconsistent: {path}")
        transition = validate_checkpoint_native_ss_binding_relocation(
            saved,
            runtime_upstream,
            allow_path_relocation=True,
        )
        checkpoint_rows.append(
            {
                "step": step,
                "path": str(path),
                "sha256": checkpoint_hash,
                "format": checkpoint.get("format"),
                "training_native_ss_path_relocation": transition,
            }
        )
        del checkpoint

    return {
        "passed": True,
        "official_protocol_sha256": train_protocol,
        "datasets": {
            "train_objects": len(train_dataset),
            "dev_objects": len(dev_dataset),
            "predicted_support_slice": [16, 64],
            "predicted_support_objects": 48,
        },
        "official_native_ss": {
            "report": str(required["predicted_native_ss_report"]),
            "report_sha256": sha256_file(required["predicted_native_ss_report"]),
            "binding": official_binding,
        },
        "artifacts": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in required.items()
            if label != "python"
        },
        "code": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in code_files.items()
        },
        "checkpoints": checkpoint_rows,
    }


def worker_environment(config: dict[str, Any], gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    project = str(config["project_root"])
    pythonpath = [project, f"{project}/ReconViaGen", f"{project}/ReconViaGen/wheels/vggt"]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "ATTN_BACKEND": "flash_attn",
            "SPCONV_ALGO": "native",
            "MPLCONFIGDIR": "/tmp/matplotlib_slat_trajectory_eval",
            "NUMBA_CACHE_DIR": "/tmp/numba_cache_slat_trajectory_eval",
            "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions_slat_trajectory_eval",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath),
        }
    )
    return environment


def gt_worker_command(config: dict[str, Any], job: Job) -> list[str]:
    cache = config["train_cache"] if job.group == "train64_gt" else config["dev_cache"]
    return [
        str(config["python"]),
        "-u",
        "-m",
        "pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support_cross_host",
        "--arm",
        "condition_lora",
        "--cache_manifest",
        str(cache / "slat_manifest.json"),
        "--lifting_cache_manifest",
        str(cache / "lifting_manifest.json"),
        "--checkpoint",
        str(job.checkpoint),
        "--native_ss_report",
        str(config["training_native_ss_report"]),
        "--stock_slat_freeze",
        str(config["stock_slat_freeze"]),
        "--output_dir",
        str(job.output_dir),
        "--weights",
        "ema",
        "--joint_seeds",
        ",".join(str(value) for value in config["seeds"]),
        "--max_objects",
        "64",
        "--object_start",
        str(job.start),
        "--object_end",
        str(job.end),
        "--surface_samples",
        str(config["surface_samples"]),
        "--bootstrap_samples",
        str(config["bootstrap_samples"]),
        "--amp_dtype",
        "bf16",
        "--allow_checkpoint_data_path_relocation",
    ]


def predicted_worker_command(
    config: dict[str, Any], job: Job, *, resume: bool
) -> list[str]:
    command = [
        str(config["python"]),
        "-u",
        "-m",
        "pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat",
        "worker",
        "--cache_manifest",
        str(config["dev_cache"] / "slat_manifest.json"),
        "--lifting_cache_manifest",
        str(config["dev_cache"] / "lifting_manifest.json"),
        "--native_ss_report",
        str(config["predicted_native_ss_report"]),
        "--stock_slat_freeze",
        str(config["stock_slat_freeze"]),
        "--trained_slat_checkpoint",
        str(job.checkpoint),
        "--trained_slat_weights",
        "ema",
        "--expected_trained_slat_step",
        str(job.step),
        "--output_dir",
        str(job.output_dir),
        "--object_start",
        str(job.start),
        "--object_end",
        str(job.end),
        "--joint_seeds",
        ",".join(str(value) for value in config["seeds"]),
        "--weights",
        "ema",
        "--surface_samples",
        str(config["surface_samples"]),
        "--amp_dtype",
        "bf16",
    ]
    if resume:
        command.append("--resume")
    return command


def validate_worker_report(job: Job) -> dict[str, Any]:
    path = job.output_dir / "report.json"
    report = read_json(path)
    if job.group in {"train64_gt", "dev64_gt"}:
        config = report.get("run_config", {})
        if report.get("passed") is not True:
            raise RuntimeError(f"GT worker did not pass: {path}")
        checks = {
            "checkpoint_step": int(config.get("checkpoint_step", -1)) == job.step,
            "object_start": int(config.get("object_start", -1)) == job.start,
            "object_end": int(config.get("object_end", -1)) == job.end,
            "native_ss_executed": config.get("native_ss_executed") is False,
        }
    else:
        identity = report.get("run_identity", {})
        if report.get("complete") is not True:
            raise RuntimeError(f"predicted-support worker is incomplete: {path}")
        checks = {
            "checkpoint_step": int(identity.get("expected_trained_slat_step", -1))
            == job.step,
            "object_start": int(identity.get("object_start", -1)) == job.start,
            "object_end": int(identity.get("object_end", -1)) == job.end,
            "trained_checkpoint": bool(identity.get("trained_slat_checkpoint")),
        }
    if not all(checks.values()):
        raise RuntimeError(f"worker report identity differs: {path}: {checks}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "passed": bool(report.get("passed")),
        "complete": bool(report.get("complete", report.get("passed"))),
    }


def run_worker_wave(
    config: dict[str, Any], wave: list[Job], *, dry_run: bool
) -> list[dict[str, Any]]:
    label = f"step={wave[0].step} group={wave[0].group}"
    print(f"\n===== WORKER WAVE {label} =====", flush=True)
    results: dict[int, dict[str, Any]] = {}
    pending: list[tuple[Job, list[str]]] = []
    for job in wave:
        report_path = job.output_dir / "report.json"
        if report_path.is_file():
            results[job.shard] = validate_worker_report(job)
            print(f"reuse finalized shard{job.shard}: {report_path}", flush=True)
            continue
        if job.group in {"train64_gt", "dev64_gt"} and job.output_dir.exists():
            raise RuntimeError(
                "GT worker output is partial and not resumable; preserve it and use "
                f"a new versioned --output_root: {job.output_dir}"
            )
        command = (
            gt_worker_command(config, job)
            if job.group in {"train64_gt", "dev64_gt"}
            else predicted_worker_command(config, job, resume=job.output_dir.exists())
        )
        pending.append((job, command))

    if dry_run:
        for job, command in pending:
            print(
                f"GPU{job.gpu} shard{job.shard} [{job.start},{job.end}): "
                + shlex.join(command),
                flush=True,
            )
        return [results[key] for key in sorted(results)]

    processes: list[tuple[Job, subprocess.Popen[bytes], Any]] = []
    try:
        for job, command in pending:
            job.output_dir.parent.mkdir(parents=True, exist_ok=True)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = job.log_path.open("ab", buffering=0)
            header = (
                f"\n===== {datetime.now(timezone.utc).isoformat()} "
                f"step={job.step} group={job.group} shard={job.shard} "
                f"range=[{job.start},{job.end}) gpu={job.gpu} =====\n"
            ).encode("utf-8")
            log_handle.write(header)
            process = subprocess.Popen(
                command,
                cwd=config["project_root"],
                env=worker_environment(config, job.gpu),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((job, process, log_handle))
            print(
                f"started pid={process.pid} GPU{job.gpu} shard{job.shard} "
                f"[{job.start},{job.end}) log={job.log_path}",
                flush=True,
            )
        failures = []
        for job, process, log_handle in processes:
            return_code = process.wait()
            log_handle.close()
            try:
                result = validate_worker_report(job)
            except Exception as error:
                failures.append(
                    f"shard{job.shard} rc={return_code} log={job.log_path}: {error}"
                )
                continue
            result["return_code"] = return_code
            results[job.shard] = result
            if job.group in {"train64_gt", "dev64_gt"} and return_code != 0:
                failures.append(
                    f"GT shard{job.shard} produced a report but exited rc={return_code}: "
                    f"{job.log_path}"
                )
            elif job.group == "dev48_predicted" and return_code not in {0, 2}:
                failures.append(
                    f"predicted shard{job.shard} unexpected rc={return_code}: {job.log_path}"
                )
            print(
                f"finished shard{job.shard} rc={return_code} "
                f"report_passed={result['passed']}",
                flush=True,
            )
        if failures:
            raise RuntimeError("worker wave failed:\n  " + "\n  ".join(failures))
    except BaseException:
        for _, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
            if not log_handle.closed:
                log_handle.close()
        raise
    return [results[index] for index in range(5)]


def aggregate_command(
    config: dict[str, Any], wave: list[Job], report_paths: list[str]
) -> list[str]:
    final = wave[0].output_dir.parent / "aggregate_v1"
    if wave[0].group in {"train64_gt", "dev64_gt"}:
        return [
            str(config["python"]),
            "-u",
            "-m",
            "pose_point_depth_mv.aggregate_proobjaverse_official_slat_gt_support",
            "--shard_reports",
            ",".join(report_paths),
            "--output_dir",
            str(final),
            "--expected_objects",
            "64",
            "--bootstrap_samples",
            str(config["bootstrap_samples"]),
        ]
    return [
        str(config["python"]),
        "-u",
        "-m",
        "pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat",
        "aggregate",
        "--cache_manifest",
        str(config["dev_cache"] / "slat_manifest.json"),
        "--lifting_cache_manifest",
        str(config["dev_cache"] / "lifting_manifest.json"),
        "--shard_reports",
        ",".join(report_paths),
        "--output_dir",
        str(final),
        "--object_start",
        "16",
        "--object_end",
        "64",
        "--expected_objects",
        "48",
        "--joint_seeds",
        ",".join(str(value) for value in config["seeds"]),
        "--bootstrap_samples",
        str(config["bootstrap_samples"]),
        "--chamfer_win_rate_min",
        "0.55",
        "--largest_component_delta_min",
        "-0.02",
    ]


def validate_aggregate_report(wave: list[Job]) -> dict[str, Any]:
    path = wave[0].output_dir.parent / "aggregate_v1/report.json"
    report = read_json(path)
    expected_objects = 64 if wave[0].group in {"train64_gt", "dev64_gt"} else 48
    expected_records = expected_objects * 3
    if int(report.get("object_count", -1)) != expected_objects:
        raise RuntimeError(f"aggregate object_count differs: {path}")
    if int(report.get("record_count", -1)) != expected_records:
        raise RuntimeError(f"aggregate record_count differs: {path}")
    if wave[0].group in {"train64_gt", "dev64_gt"}:
        run_config = report.get("run_config", {})
        if report.get("passed") is not True:
            raise RuntimeError(f"GT aggregate did not pass: {path}")
        if int(run_config.get("checkpoint_step", -1)) != wave[0].step:
            raise RuntimeError(f"GT aggregate checkpoint step differs: {path}")
    else:
        if int(report.get("object_start", -1)) != 16 or int(
            report.get("object_end", -1)
        ) != 64:
            raise RuntimeError(f"predicted aggregate slice differs: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "passed": bool(report.get("passed")),
        "object_count": expected_objects,
        "record_count": expected_records,
    }


def run_aggregate(
    config: dict[str, Any], wave: list[Job], workers: list[dict[str, Any]], *, dry_run: bool
) -> dict[str, Any] | None:
    final = wave[0].output_dir.parent / "aggregate_v1"
    report_path = final / "report.json"
    if report_path.is_file():
        result = validate_aggregate_report(wave)
        print(f"reuse aggregate: {report_path}", flush=True)
        return result
    if final.exists():
        raise RuntimeError(f"partial aggregate output exists: {final}")
    report_paths = [str(job.output_dir / "report.json") for job in wave]
    command = aggregate_command(config, wave, report_paths)
    if dry_run:
        print("aggregate: " + shlex.join(command), flush=True)
        return None
    result = subprocess.run(
        command,
        cwd=config["project_root"],
        env=worker_environment(config, wave[0].gpu),
        text=True,
    )
    if wave[0].group in {"train64_gt", "dev64_gt"}:
        allowed = {0}
    else:
        allowed = {0, 3}
    if result.returncode not in allowed:
        raise RuntimeError(
            f"aggregate failed rc={result.returncode}: {shlex.join(command)}"
        )
    aggregate = validate_aggregate_report(wave)
    aggregate["return_code"] = result.returncode
    print(
        f"aggregate complete step={wave[0].step} group={wave[0].group} "
        f"rc={result.returncode} passed={aggregate['passed']}",
        flush=True,
    )
    return aggregate


def make_contract(
    config: dict[str, Any], preflight: dict[str, Any], gpu_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "steps": list(config["steps"]),
        "gpus": list(config["gpus"]),
        "joint_seeds": list(config["seeds"]),
        "surface_samples": config["surface_samples"],
        "bootstrap_samples": config["bootstrap_samples"],
        "groups": printable_plan(config)["groups"],
        "paths": {
            key: str(config[key])
            for key in (
                "project_root",
                "checkpoint_root",
                "source_root",
                "official_ss_root",
                "training_native_ss_report",
                "predicted_native_ss_report",
                "stock_slat_freeze",
                "output_root",
            )
        },
        "artifact_preflight": preflight,
        # Free memory/utilization are launch-time observations and must not make
        # a scientifically identical resume contract change on the next run.
        "gpu_assignment": [
            {
                "index": row["index"],
                "name": row["name"],
                "total_mib": row["total_mib"],
            }
            for row in gpu_rows
        ],
    }


def initialize_output(config: dict[str, Any], contract: dict[str, Any]) -> None:
    root = config["output_root"]
    manifest = root / "run_manifest.json"
    if manifest.is_file():
        existing = read_json(manifest)
        if existing.get("contract") != contract:
            raise RuntimeError("existing trajectory run_manifest contract differs")
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"nonempty output root lacks run_manifest.json: {root}")
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        manifest,
        {
            "format": FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contract": contract,
        },
    )


def projected_metrics(report: dict[str, Any], group: str) -> dict[str, Any]:
    if group in {"train64_gt", "dev64_gt"}:
        return {
            "passed": report["passed"],
            "formal": report.get("formal"),
            "summary": report["summary"],
        }
    return {
        "runtime_integrity_passed": report["passed"],
        "formal": report.get("formal"),
        "occupancy": report["occupancy"],
        "B_minus_A__new_ss_plus_stock_slat": report["stock_slat_mesh_transfer"],
        "C_minus_A__new_ss_plus_trained_slat": report[
            "trained_slat_end_to_end_transfer"
        ],
        "C_minus_B__trained_slat_increment_on_native_support": report[
            "trained_slat_increment_on_native_support"
        ],
        "decision": report["decision"],
        "integrity": report["integrity"],
    }


def write_final_summary(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    all_runtime_passed = True
    for step in config["steps"]:
        for group in GROUPS:
            path = (
                config["output_root"]
                / f"step_{step:06d}"
                / group
                / "aggregate_v1/report.json"
            )
            report = read_json(path)
            all_runtime_passed = all_runtime_passed and report.get("passed") is True
            rows.append(
                {
                    "step": step,
                    "group": group,
                    "report": str(path),
                    "report_sha256": sha256_file(path),
                    "metrics": projected_metrics(report, group),
                }
            )
    payload = {
        "format": FORMAT,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "all_nine_aggregates_generated": len(rows) == 9,
        "all_runtime_integrity_passed": all_runtime_passed,
        "scientific_note": (
            "Native-SS predicted-support is the frozen held-out Dev48 slice [16,64); "
            "passed=false is preserved as a result and does not mean the unattended "
            "orchestrator failed to finish."
        ),
        "results": rows,
    }
    write_json_atomic(config["output_root"] / "trajectory_summary.json", payload)
    lines = [
        "ProObjaverse SLat 15k/20k/25k trajectory evaluation",
        "=" * 57,
        f"all_nine_aggregates_generated: {payload['all_nine_aggregates_generated']}",
        f"all_runtime_integrity_passed: {payload['all_runtime_integrity_passed']}",
        "",
    ]
    for row in rows:
        metrics = row["metrics"]
        passed = metrics.get("passed", metrics.get("runtime_integrity_passed"))
        lines.append(
            f"step={row['step']} group={row['group']} runtime_passed={passed} "
            f"report={row['report']}"
        )
    lines.extend(["", payload["scientific_note"]])
    (config["output_root"] / "trajectory_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    args = make_parser().parse_args()
    config = resolved_config(args)
    plan = printable_plan(config)
    if args.print_plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    preflight = artifact_preflight(config)
    gpu_rows = []
    if not args.skip_gpu_free_check:
        gpu_rows = check_gpu_free(config["gpus"], config["min_free_gpu_mib"])
    preflight_output = {
        "format": FORMAT,
        "passed": True,
        "artifact_preflight": preflight,
        "gpu_snapshot": gpu_rows,
        "plan_summary": {
            "wave_count": plan["wave_count"],
            "worker_count": plan["worker_count"],
        },
    }
    print(json.dumps(preflight_output, indent=2, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return
    if args.dry_run:
        waves = build_jobs(
            steps=config["steps"],
            gpus=config["gpus"],
            checkpoint_root=config["checkpoint_root"],
            output_root=config["output_root"],
        )
        for wave in waves:
            workers = run_worker_wave(config, wave, dry_run=True)
            run_aggregate(config, wave, workers, dry_run=True)
        return

    contract = make_contract(config, preflight, gpu_rows)
    initialize_output(config, contract)
    waves = build_jobs(
        steps=config["steps"],
        gpus=config["gpus"],
        checkpoint_root=config["checkpoint_root"],
        output_root=config["output_root"],
    )
    completed = []
    for wave_index, wave in enumerate(waves, start=1):
        workers = run_worker_wave(config, wave, dry_run=False)
        aggregate = run_aggregate(config, wave, workers, dry_run=False)
        completed.append(
            {
                "wave": wave_index,
                "step": wave[0].step,
                "group": wave[0].group,
                "workers": workers,
                "aggregate": aggregate,
            }
        )
        write_json_atomic(
            config["output_root"] / "progress.json",
            {
                "format": FORMAT,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed_waves": completed,
                "completed_wave_count": len(completed),
                "total_wave_count": len(waves),
            },
        )
    summary = write_final_summary(config)
    print(
        json.dumps(
            {
                "completed": True,
                "output_root": str(config["output_root"]),
                "all_nine_aggregates_generated": summary[
                    "all_nine_aggregates_generated"
                ],
                "all_runtime_integrity_passed": summary[
                    "all_runtime_integrity_passed"
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
