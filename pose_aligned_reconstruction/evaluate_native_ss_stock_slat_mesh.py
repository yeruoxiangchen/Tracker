#!/usr/bin/env python3
"""Evaluate Native SS support through the frozen Stock SLAT and Mesh decoder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import trimesh

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from ar_ss_flow.shared_object_preprocessing import (
    SHARED_OBJECT_PREPROCESSING_VERSION,
    prepare_shared_object_views,
)
from pose_aligned_reconstruction.eval_direct_flow import (
    bootstrap_mean_ci,
    decode_coords,
    overlap_metrics,
    positive_rate,
    summarize,
)
from pose_aligned_reconstruction.evaluate_native_ss_genrecon import sampling_params
from pose_aligned_reconstruction.export_direct_flow_mesh_pairs import (
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    sample_slat_explicit,
    save_preview,
    shared_noise_audit,
    sparse_exact_diff,
    sparse_from_payload,
    sparse_noise_from_master,
    sparse_payload,
    surface_metrics,
    tensor_sha256,
    tensor_tree_sha256,
    to_cpu_tree,
    to_device_tree,
)
from pose_aligned_reconstruction.native_ss_genrecon import (
    NATIVE_SS_GENRECON_EVAL,
    NativeSSCalibratedCFGFlow,
    NativeSSStockFlow,
    build_native_ss_genrecon_components,
    load_trainable_state_dict,
    require_disjoint_object_uids,
    sha256_file,
    validate_genrecon_cache_contract,
    validate_native_ss_genrecon_checkpoint,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond


REPORT_FORMAT = "pose_point_depth_mv.native_ss_stock_slat_mesh_transfer.v1"
ALLOWED_SS_FALSE_CHECKS = {"count_ratio_lower"}
TRANSFER_METRICS = (
    "chamfer_l1_improvement",
    "fscore_0p02_delta",
    "normal_consistency_delta",
    "largest_component_ratio_delta",
    "mesh_success_delta",
)


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def parse_interval(value: str) -> list[float]:
    result = parse_csv(value, float)
    if len(result) != 2 or not 0.0 <= result[0] <= result[1] <= 1.0:
        raise ValueError("interval must contain two ordered values in [0,1]")
    return result


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def torch_save_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    torch.save(value, temporary)
    os.replace(temporary, path)


def savez_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def pair_id(uid: str, seed: int) -> str:
    digest = hashlib.sha256(f"{uid}|{int(seed)}".encode("utf-8")).hexdigest()
    return digest[:24]


def validate_ss_evidence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("format") != NATIVE_SS_GENRECON_EVAL:
        raise ValueError(f"unexpected Native SS report format={payload.get('format')!r}")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("Native SS report has no checks")
    false_checks = {str(key) for key, value in checks.items() if value is not True}
    if false_checks - ALLOWED_SS_FALSE_CHECKS:
        raise RuntimeError(
            "Native SS report failed quality/integrity checks that cannot be waived for "
            f"SLAT transfer: {sorted(false_checks)}"
        )
    if checks.get("disabled_stock_equivalence") is not True:
        raise RuntimeError("Native SS report lacks disabled Stock equivalence")
    protocol = payload.get("protocol")
    calibrated = payload.get("calibrated_parameters")
    if not isinstance(protocol, dict) or not isinstance(calibrated, dict):
        raise ValueError("Native SS report lacks protocol/calibrated parameters")
    required = {
        "weights": "ema",
        "condition_scale_policy": "learned_projection_only",
        "post_cfg_cap": False,
    }
    mismatch = {
        key: (protocol.get(key), expected)
        for key, expected in required.items()
        if protocol.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"Native SS deployment binding differs: {mismatch}")
    if calibrated.get("condition_scale_policy") != "learned_projection_only" or bool(
        calibrated.get("post_cfg_cap")
    ):
        raise RuntimeError("Native SS calibrated deployment semantics differ")
    cfg_strength = float(calibrated.get("cfg_strength", 0.0))
    if not np.isfinite(cfg_strength) or cfg_strength <= 0.0:
        raise ValueError("Native SS report has invalid CFG")
    return {
        "checkpoint": str(protocol["checkpoint"]),
        "checkpoint_sha256": str(protocol["checkpoint_sha256"]),
        "checkpoint_step": int(protocol["checkpoint_step"]),
        "weights": str(protocol["weights"]),
        "cfg_strength": cfg_strength,
        "steps": int(protocol["steps"]),
        "cfg_interval": [float(value) for value in protocol["cfg_interval"]],
        "guidance_rescale": float(protocol["guidance_rescale"]),
        "rescale_t": float(protocol["rescale_t"]),
        "amp_dtype": str(protocol["amp_dtype"]),
        "false_checks": sorted(false_checks),
    }


def load_ss_evidence(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = Path(path).resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    binding = validate_ss_evidence_payload(payload)
    checkpoint = Path(binding["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != binding["checkpoint_sha256"]:
        raise RuntimeError("Native SS checkpoint hash differs from the V4 report")
    binding["report"] = str(report_path)
    binding["report_sha256"] = sha256_file(report_path)
    binding["checkpoint"] = str(checkpoint)
    return payload, binding


def make_sampling_namespace(binding: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        steps=int(binding["steps"]),
        cfg_interval=",".join(str(value) for value in binding["cfg_interval"]),
        guidance_rescale=float(binding["guidance_rescale"]),
        rescale_t=float(binding["rescale_t"]),
    )


def summary_with_ci(values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "positive_rate": None,
            "bootstrap_mean_95_ci": None,
        }
    return {
        **summarize(values),
        "positive_rate": positive_rate(values),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(
            values, samples=int(samples), seed=int(seed)
        ),
    }


def aggregate_transfer_records(
    records: list[dict[str, Any]],
    *,
    expected_pairs: int,
    seeds: list[int],
    bootstrap_samples: int,
    metadata_by_object: dict[str, dict[str, Any]],
    chamfer_win_rate_min: float,
    lcr_delta_min: float,
) -> dict[str, Any]:
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_pairs: list[dict[str, Any]] = []
    for row in records:
        key = (str(row["object_uid"]), int(row["seed"]))
        branch = str(row["branch"])
        if branch in by_pair[key]:
            duplicate_pairs.append({"object_uid": key[0], "seed": key[1], "branch": branch})
        by_pair[key][branch] = row
    pair_deltas = []
    invalid_pairs = list(duplicate_pairs)
    for (object_uid, seed), branches in sorted(by_pair.items()):
        if set(branches) != {"stock", "native"}:
            invalid_pairs.append(
                {"object_uid": object_uid, "seed": seed, "error": "missing branch"}
            )
            continue
        stock, native = branches["stock"], branches["native"]
        if stock.get("passed") is not True or native.get("passed") is not True:
            invalid_pairs.append(
                {"object_uid": object_uid, "seed": seed, "error": "branch failed"}
            )
            continue
        pair_deltas.append(
            {
                "object_uid": object_uid,
                "seed": seed,
                "chamfer_l1_improvement": float(stock["surface"]["chamfer_l1"])
                - float(native["surface"]["chamfer_l1"]),
                "fscore_0p02_delta": float(native["surface"]["fscore_0p02"])
                - float(stock["surface"]["fscore_0p02"]),
                "normal_consistency_delta": float(
                    native["surface"]["normal_consistency"]
                )
                - float(stock["surface"]["normal_consistency"]),
                "largest_component_ratio_delta": float(
                    native["structure"]["largest_component_ratio"]
                )
                - float(stock["structure"]["largest_component_ratio"]),
                "mesh_success_delta": float(native["structure"]["mesh_success"])
                - float(stock["structure"]["mesh_success"]),
            }
        )
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_deltas:
        by_object[str(row["object_uid"])].append(row)
    object_rows = []
    for object_uid, rows in sorted(by_object.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                **metadata_by_object.get(object_uid, {}),
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in TRANSFER_METRICS
                },
            }
        )
    summary = {
        metric: summary_with_ci(
            [float(row[metric]) for row in object_rows],
            samples=int(bootstrap_samples),
            seed=84000 + position,
        )
        for position, metric in enumerate(TRANSFER_METRICS)
    }
    seed_summary = {}
    for seed in seeds:
        rows = [row for row in pair_deltas if int(row["seed"]) == int(seed)]
        seed_summary[str(seed)] = {
            metric: summary_with_ci(
                [float(row[metric]) for row in rows],
                samples=max(200, min(int(bootstrap_samples), 1000)),
                seed=85000 + int(seed) + position,
            )
            for position, metric in enumerate(TRANSFER_METRICS)
        }

    def grouped(field: str) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in object_rows:
            groups[str(row.get(field, "unknown"))].append(row)
        result = []
        for value, rows in sorted(groups.items()):
            result.append(
                {
                    field: value,
                    "object_count": len(rows),
                    **{
                        metric: {
                            "mean": float(np.mean([float(row[metric]) for row in rows])),
                            "median": float(np.median([float(row[metric]) for row in rows])),
                            "positive_rate": positive_rate(
                                [float(row[metric]) for row in rows]
                            ),
                        }
                        for metric in TRANSFER_METRICS
                    },
                }
            )
        return result

    chamfer = summary["chamfer_l1_improvement"]
    fscore = summary["fscore_0p02_delta"]
    success = summary["mesh_success_delta"]
    lcr = summary["largest_component_ratio_delta"]
    nonnegative_seed_directions = sum(
        row["chamfer_l1_improvement"]["mean"] is not None
        and float(row["chamfer_l1_improvement"]["mean"]) >= 0.0
        for row in seed_summary.values()
    )
    checks = {
        "expected_record_count": len(records) == 2 * int(expected_pairs),
        "expected_pair_count": len(by_pair) == int(expected_pairs),
        "no_invalid_pairs": not invalid_pairs and len(pair_deltas) == int(expected_pairs),
        "chamfer_mean_positive": chamfer["mean"] is not None
        and float(chamfer["mean"]) > 0.0,
        "chamfer_median_positive": chamfer["median"] is not None
        and float(chamfer["median"]) > 0.0,
        "chamfer_object_win": chamfer["positive_rate"] is not None
        and float(chamfer["positive_rate"]) >= float(chamfer_win_rate_min),
        "chamfer_ci_lower_positive": chamfer["bootstrap_mean_95_ci"] is not None
        and float(chamfer["bootstrap_mean_95_ci"][0]) > 0.0,
        "fscore_non_degrading": fscore["mean"] is not None
        and float(fscore["mean"]) >= 0.0,
        "mesh_success_non_degrading": success["mean"] is not None
        and float(success["mean"]) >= 0.0,
        "largest_component_non_degrading": lcr["mean"] is not None
        and float(lcr["mean"]) >= float(lcr_delta_min),
        "minimum_nonnegative_seed_directions": nonnegative_seed_directions
        >= min(2, len(seeds)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_pair_count": int(expected_pairs),
        "valid_pair_count": len(pair_deltas),
        "invalid_pairs": invalid_pairs,
        "summary": summary,
        "seed_summary": seed_summary,
        "by_source": grouped("source"),
        "by_view_count": grouped("view_count"),
        "object_rows": object_rows,
        "pair_deltas": pair_deltas,
    }


def load_split_metadata(
    path: str, phase: str, selected_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    fallback = {
        str(row.get("object_uid", row["uid"])): {
            "source": str(row.get("source", "unknown")),
            "view_count": int(row.get("view_count", 0)),
            "phase_position": position,
        }
        for position, row in enumerate(selected_rows)
    }
    if not path:
        return fallback, None
    split_path = Path(path).resolve()
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    rows = payload.get("phases", {}).get(str(phase))
    if not isinstance(rows, list):
        raise ValueError(f"split audit has no phase={phase!r}")
    available = {
        str(row["object_uid"]): {
            "source": str(row["source"]),
            "view_count": int(row["view_count"]),
            "phase_position": position,
        }
        for position, row in enumerate(rows)
    }
    missing = sorted(set(fallback) - set(available))
    if missing:
        raise RuntimeError(f"selected objects missing from split phase: {missing[:8]}")
    return (
        {object_uid: available[object_uid] for object_uid in fallback},
        {"path": str(split_path), "sha256": sha256_file(split_path), "phase": phase},
    )


def build_stock_conditioning_pipeline(pretrained: str, device: torch.device):
    from pose_aligned_reconstruction.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    required = ("image_cond_model", "sparse_structure_vggt_cond", "slat_vggt_cond")
    for name in required:
        if name not in pipeline.models:
            raise RuntimeError(f"Stock pipeline lacks {name}")
        pipeline.models[name].to(device).eval()
    pipeline.VGGT_model.to(device).eval()
    for module in (pipeline.VGGT_model, *(pipeline.models[name] for name in required)):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return pipeline


@torch.no_grad()
def build_slat_condition(
    pipeline,
    sample: dict[str, Any],
    *,
    device: torch.device,
    condition_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shared = dict(sample.get("preprocessing", {}).get("shared_geometry", {}))
    if shared.get("version") != SHARED_OBJECT_PREPROCESSING_VERSION:
        raise RuntimeError("SLAT transfer requires the shared object preprocessing cache")
    prepared = prepare_shared_object_views(
        sample["image_paths"],
        sample["mask_paths"],
        resolution=int(shared["resolution"]),
        foreground_margin=float(shared["foreground_margin"]),
        alpha_threshold=float(shared["alpha_threshold"]),
    )
    pipeline_resolution = int(getattr(pipeline, "default_image_resolution", 0))
    if pipeline_resolution != int(shared["resolution"]):
        raise RuntimeError(
            f"pipeline/shared resolution differs: {pipeline_resolution}/{shared['resolution']}"
        )
    aggregated, image_tensor = pipeline.vggt_feat(prepared.images)
    raw_image_cond = pipeline.encode_image(image_tensor)
    batch = int(aggregated[0].shape[0])
    views = int(aggregated[0].shape[1])
    image_cond = normalize_image_cond(raw_image_cond, batch=batch, views=views)
    recomputed_ss = pipeline.get_ss_cond(
        image_cond[:, :, 5:], aggregated, num_samples=1
    )["cond"]
    cached_ss = sample["stock_condition"].to(device=device)
    if recomputed_ss.shape != cached_ss.shape:
        raise RuntimeError("recomputed/cached Stock SS condition shape differs")
    difference = float((recomputed_ss.float() - cached_ss.float()).abs().amax().item())
    if difference > float(condition_tolerance):
        raise RuntimeError(
            f"shared preprocessing Stock condition replay differs: {difference} > "
            f"{condition_tolerance}"
        )
    condition = pipeline.get_slat_cond(image_cond, aggregated, num_samples=1)
    audit = {
        "uid": str(sample["uid"]),
        "views": views,
        "shared_preprocessing_version": shared["version"],
        "cached_vs_recomputed_stock_condition_max_abs": difference,
        "condition_tolerance": float(condition_tolerance),
        "slat_condition_sha256": tensor_tree_sha256(condition),
        "passed": True,
    }
    return to_cpu_tree(condition), audit


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--ss_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--split_audit", default="")
    parser.add_argument("--split_phase", default="final")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--slat_steps", type=int, default=25)
    parser.add_argument("--slat_cfg_strength", type=float, default=5.0)
    parser.add_argument("--slat_cfg_interval", default="0.5,1.0")
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--condition_tolerance", type=float, default=1.0e-2)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--fscore_thresholds", default="0.01,0.02,0.05")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--chamfer_win_rate_min", type=float, default=0.55)
    parser.add_argument("--largest_component_delta_min", type=float, default=-0.02)
    parser.add_argument("--slat_repeat_max_abs", type=float, default=0.1)
    parser.add_argument("--render_previews", action="store_true")
    parser.add_argument("--render_frames", type=int, default=36)
    parser.add_argument("--render_resolution", type=int, default=256)
    return parser


def validate_args(args: argparse.Namespace, ss_binding: dict[str, Any]) -> None:
    if int(args.max_objects) < 0:
        raise ValueError("max_objects must be non-negative")
    if int(args.slat_steps) <= 0 or int(args.surface_samples) <= 0:
        raise ValueError("SLAT steps and surface samples must be positive")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 <= float(args.chamfer_win_rate_min) <= 1.0:
        raise ValueError("chamfer win-rate threshold must be in [0,1]")
    if float(args.condition_tolerance) < 0.0 or float(args.slat_repeat_max_abs) < 0.0:
        raise ValueError("audit tolerances must be non-negative")
    parse_interval(args.slat_cfg_interval)
    thresholds = parse_csv(args.fscore_thresholds, float)
    if any(value <= 0.0 for value in thresholds) or not any(
        abs(value - 0.02) <= 1.0e-12 for value in thresholds
    ):
        raise ValueError("F-score thresholds must be positive and include 0.02")
    if str(args.amp_dtype) != str(ss_binding["amp_dtype"]):
        raise ValueError("transfer AMP dtype must match the locked Native SS report")


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    _, ss_binding = load_ss_evidence(args.ss_report)
    validate_args(args, ss_binding)
    output_dir = Path(args.output_dir).resolve()
    cache_manifest = Path(args.cache_manifest).resolve()
    dataset = PoseLiftingCacheDataset(cache_manifest, indices=args.indices)
    if int(args.max_objects) > 0:
        dataset.rows = dataset.rows[: int(args.max_objects)]
    if not dataset.rows:
        raise ValueError("empty transfer selection")
    seeds = parse_csv(args.joint_seeds, int)
    selected_rows = list(dataset.rows)
    metadata_by_object, split_binding = load_split_metadata(
        args.split_audit, args.split_phase, selected_rows
    )
    run_identity = {
        "format": REPORT_FORMAT,
        "cache_manifest": str(cache_manifest),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "indices": str(args.indices),
        "max_objects": int(args.max_objects),
        "selected_uids": [str(row["uid"]) for row in selected_rows],
        "joint_seeds": seeds,
        "ss_binding": ss_binding,
        "pretrained": str(args.pretrained),
        "amp_dtype": str(args.amp_dtype),
        "slat": {
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_cfg_strength),
            "cfg_interval": parse_interval(args.slat_cfg_interval),
            "rescale_t": float(args.slat_rescale_t),
        },
        "mesh": {
            "surface_samples": int(args.surface_samples),
            "fscore_thresholds": parse_csv(args.fscore_thresholds, float),
            "render_previews": bool(args.render_previews),
            "render_frames": int(args.render_frames),
            "render_resolution": int(args.render_resolution),
        },
        "split": split_binding,
    }
    identity_path = output_dir / "run_identity.json"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != run_identity:
            raise RuntimeError("resume arguments differ from the existing run")
    else:
        write_json(identity_path, run_identity)
    completed_report = output_dir / "report.json"
    smoke = int(args.max_objects) > 0
    if completed_report.is_file():
        report = json.loads(completed_report.read_text(encoding="utf-8"))
        print(json.dumps({"reused": True, "passed": report["passed"]}, indent=2))
        raise SystemExit(0 if smoke or report["passed"] else 2)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    checkpoint = torch.load(ss_binding["checkpoint"], map_location="cpu")
    validate_native_ss_genrecon_checkpoint(checkpoint, pretrained=args.pretrained)
    if int(checkpoint["step"]) != int(ss_binding["checkpoint_step"]):
        raise RuntimeError("checkpoint step differs from the Native SS report")
    training_identity = checkpoint.get("data_identity", {})
    training_object_uids = training_identity.get("object_uids")
    if not isinstance(training_object_uids, list):
        raise RuntimeError("Native SS checkpoint lacks training object identities")
    require_disjoint_object_uids(
        (str(row.get("object_uid", row["uid"])) for row in selected_rows),
        training_object_uids,
    )
    validate_genrecon_cache_contract(
        dataset, training_config_hash=str(training_identity.get("config_hash", ""))
    )
    saved = checkpoint["args"]
    ss_sampler, ss_model, ss_decoder, model_summary, ss_defaults = (
        build_native_ss_genrecon_components(
            pretrained=args.pretrained,
            lora_rank=int(saved["lora_rank"]),
            lora_alpha=int(saved["lora_alpha"]),
            condition_channels=int(saved["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if ss_decoder is None:
        raise RuntimeError("Native SS transfer requires the frozen SS decoder")
    load_trainable_state_dict(ss_model, checkpoint["ema_trainable_state"])
    ss_model.eval()
    ss_decoder.eval()
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    ss_params = sampling_params(
        ss_defaults,
        make_sampling_namespace(ss_binding),
        float(ss_binding["cfg_strength"]),
    )
    coord_root = output_dir / "ss_coords"
    ss_rows = []
    for position in range(len(dataset)):
        sample = dataset[position]
        object_uid = str(sample.get("object_uid", sample["uid"]))
        phase_position = int(
            metadata_by_object.get(object_uid, {}).get("phase_position", position)
        )
        positive = sample["stock_condition"].to(device=device)
        negative = torch.zeros_like(positive)
        target_coords = sample["target_coords"].numpy()
        for seed in seeds:
            current_pair = pair_id(str(sample["uid"]), seed)
            coord_path = coord_root / f"{current_pair}.npz"
            audit_path = coord_root / f"{current_pair}.json"
            if coord_path.is_file() and audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("uid") != str(sample["uid"]) or int(audit["seed"]) != seed:
                    raise RuntimeError("resumed SS coordinate identity differs")
                ss_rows.append(audit)
                continue
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + phase_position * 1009
            )
            initial = torch.randn(
                (1, 8, 16, 16, 16), generator=generator, device=device
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                stock_latent = ss_sampler.sample(
                    NativeSSStockFlow(ss_model),
                    initial.clone(),
                    cond=positive,
                    neg_cond=negative,
                    **ss_params,
                    verbose=False,
                ).samples
                native_flow = NativeSSCalibratedCFGFlow(
                    ss_model, positive, sample, enabled=True, projection_mode="correct"
                )
                native_latent = ss_sampler.sample(
                    native_flow,
                    initial.clone(),
                    cond=positive,
                    neg_cond=negative,
                    **ss_params,
                    verbose=False,
                ).samples
            if float(ss_binding["cfg_strength"]) != 1.0 and (
                native_flow.positive_calls == 0 or native_flow.negative_calls == 0
            ):
                raise RuntimeError("Native SS standard CFG did not call both branches")
            coords = {
                "stock": decode_coords(ss_decoder, stock_latent),
                "native": decode_coords(ss_decoder, native_latent),
            }
            branch_metrics = {
                branch: overlap_metrics(value, target_coords)
                for branch, value in coords.items()
            }
            audit = {
                "pair_id": current_pair,
                "uid": str(sample["uid"]),
                "object_uid": str(sample.get("object_uid", sample["uid"])),
                "view_count": int(sample["visual_patch_features"].shape[0]),
                "seed": int(seed),
                "initial_noise_sha256": tensor_sha256(initial),
                "same_ss_initial_noise": True,
                "stock": branch_metrics["stock"],
                "native": branch_metrics["native"],
                "stock_count": int(len(coords["stock"])),
                "native_count": int(len(coords["native"])),
                "native_stock_count_ratio": (
                    float(len(coords["native"]) / len(coords["stock"]))
                    if len(coords["stock"])
                    else None
                ),
                "native_wrapper": native_flow.summary(),
                "passed": bool(len(coords["stock"]) and len(coords["native"])),
            }
            savez_atomic(
                coord_path,
                stock=coords["stock"].astype(np.int32),
                native=coords["native"].astype(np.int32),
            )
            audit["coords_npz_sha256"] = sha256_file(coord_path)
            write_json(audit_path, audit)
            ss_rows.append(audit)
            print(
                f"[native_ss_stock_slat:ss] {position + 1}/{len(dataset)} "
                f"seed={seed} uid={sample['uid']}",
                flush=True,
            )
            del initial, stock_latent, native_latent, native_flow
            torch.cuda.empty_cache()
        del positive, negative
    ss_model.cpu()
    ss_decoder.cpu()
    del ss_model, ss_decoder, ss_sampler, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    pipeline = build_stock_conditioning_pipeline(args.pretrained, device)
    condition_root = output_dir / "slat_conditions"
    for position in range(len(dataset)):
        sample = dataset[position]
        condition_path = condition_root / f"{sample['uid']}.pt"
        audit_path = condition_root / f"{sample['uid']}.json"
        if condition_path.is_file() and audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("uid") != str(sample["uid"]) or audit.get("passed") is not True:
                raise RuntimeError("resumed SLAT condition identity differs")
            continue
        condition, audit = build_slat_condition(
            pipeline,
            sample,
            device=device,
            condition_tolerance=float(args.condition_tolerance),
        )
        torch_save_atomic(condition, condition_path)
        audit["condition_pt_sha256"] = sha256_file(condition_path)
        write_json(audit_path, audit)
        print(
            f"[native_ss_stock_slat:condition] {position + 1}/{len(dataset)} "
            f"uid={sample['uid']}",
            flush=True,
        )
        del condition
        torch.cuda.empty_cache()
    for name in ("image_cond_model", "sparse_structure_vggt_cond", "slat_vggt_cond"):
        pipeline.models[name].cpu()
    pipeline.VGGT_model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    slat_flow = pipeline.models["slat_flow_model"].to(device).eval()
    for parameter in slat_flow.parameters():
        parameter.requires_grad_(False)
    slat_resolution = int(getattr(slat_flow, "resolution", 64))
    slat_channels = int(slat_flow.in_channels)
    if slat_resolution != 64:
        raise RuntimeError(f"unexpected Stock SLAT resolution={slat_resolution}")
    slat_params = dict(pipeline.slat_sampler_params)
    slat_params.update(
        {
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_cfg_strength),
            "cfg_interval": tuple(parse_interval(args.slat_cfg_interval)),
            "rescale_t": float(args.slat_rescale_t),
        }
    )
    slat_root = output_dir / "slat"
    repeat_audit_path = output_dir / "slat_repeat_audit.json"
    repeat_done = repeat_audit_path.is_file()
    for position in range(len(dataset)):
        sample = dataset[position]
        object_uid = str(sample.get("object_uid", sample["uid"]))
        phase_position = int(
            metadata_by_object.get(object_uid, {}).get("phase_position", position)
        )
        condition_cpu = torch.load(
            condition_root / f"{sample['uid']}.pt", map_location="cpu"
        )
        condition = to_device_tree(condition_cpu, device)
        for seed in seeds:
            current_pair = pair_id(str(sample["uid"]), seed)
            pair_dir = slat_root / current_pair
            branch_paths = {
                branch: pair_dir / f"{branch}.pt" for branch in ("stock", "native")
            }
            audit_path = pair_dir / "audit.json"
            if audit_path.is_file() and all(path.is_file() for path in branch_paths.values()):
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("pair_id") != current_pair or audit.get("passed") is not True:
                    raise RuntimeError("resumed SLAT artifact identity differs")
                continue
            with np.load(coord_root / f"{current_pair}.npz") as payload:
                coords = {
                    branch: canonical_coords(payload[branch], resolution=slat_resolution)
                    for branch in ("stock", "native")
                }
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 2000003 + phase_position * 2017 + 7919
            )
            master = torch.randn(
                (slat_resolution, slat_resolution, slat_resolution, slat_channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            initial = {
                branch: sparse_noise_from_master(coords[branch], master, device=device)
                for branch in ("stock", "native")
            }
            noise_audit = shared_noise_audit(
                initial["stock"].coords,
                initial["stock"].feats,
                initial["native"].coords,
                initial["native"].feats,
            )
            if not noise_audit["common_coord_noise_bit_exact"]:
                raise RuntimeError("Stock/Native SLAT common-coordinate noise differs")
            order = (
                ["stock", "native"]
                if int(current_pair[-1], 16) % 2 == 0
                else ["native", "stock"]
            )
            sampled = {}
            for branch in order:
                sampled[branch] = sample_slat_explicit(
                    pipeline=pipeline,
                    slat_condition=condition,
                    initial_noise=initial[branch],
                    params=slat_params,
                )
                torch_save_atomic(sparse_payload(sampled[branch]), branch_paths[branch])
            if not repeat_done:
                repeated = sample_slat_explicit(
                    pipeline=pipeline,
                    slat_condition=condition,
                    initial_noise=initial["stock"],
                    params=slat_params,
                )
                repeat = sparse_exact_diff(sampled["stock"], repeated)
                repeat["limit"] = float(args.slat_repeat_max_abs)
                repeat["passed"] = bool(
                    repeat["coords_equal"]
                    and float(repeat["feats_max_abs"]) <= float(args.slat_repeat_max_abs)
                )
                write_json(
                    repeat_audit_path,
                    {"pair_id": current_pair, "uid": str(sample["uid"]), **repeat},
                )
                if not repeat["passed"]:
                    raise RuntimeError(f"Stock SLAT repeat audit failed: {repeat}")
                repeat_done = True
                del repeated
            audit = {
                "pair_id": current_pair,
                "uid": str(sample["uid"]),
                "object_uid": str(sample.get("object_uid", sample["uid"])),
                "seed": int(seed),
                "master_noise_seed": int(seed) * 2000003
                + phase_position * 2017
                + 7919,
                "master_noise_sha256": tensor_sha256(master),
                "condition_sha256": tensor_tree_sha256(condition_cpu),
                "execution_order": order,
                **noise_audit,
                "branch_payload_sha256": {
                    branch: sha256_file(path) for branch, path in branch_paths.items()
                },
                "passed": True,
            }
            write_json(audit_path, audit)
            print(
                f"[native_ss_stock_slat:slat] {position + 1}/{len(dataset)} "
                f"seed={seed} uid={sample['uid']}",
                flush=True,
            )
            del master, initial, sampled
            torch.cuda.empty_cache()
        del condition_cpu, condition
    slat_flow.cpu()
    del slat_flow
    gc.collect()
    torch.cuda.empty_cache()

    mesh_decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    for parameter in mesh_decoder.parameters():
        parameter.requires_grad_(False)
    mesh_root = output_dir / "mesh_pairs"
    records = []
    thresholds = parse_csv(args.fscore_thresholds, float)
    for position in range(len(dataset)):
        sample = dataset[position]
        object_uid = str(sample.get("object_uid", sample["uid"]))
        phase_position = int(
            metadata_by_object.get(object_uid, {}).get("phase_position", position)
        )
        target_mesh, target_metadata = load_canonical_gt(sample)
        for seed in seeds:
            current_pair = pair_id(str(sample["uid"]), seed)
            pair_dir = mesh_root / current_pair
            pair_record_path = pair_dir / "pair_record.json"
            if pair_record_path.is_file():
                pair_record = json.loads(pair_record_path.read_text(encoding="utf-8"))
                if pair_record.get("pair_id") != current_pair:
                    raise RuntimeError("resumed Mesh pair identity differs")
                records.extend(pair_record["branches"])
                continue
            branch_records = []
            for branch in ("stock", "native"):
                branch_dir = pair_dir / branch
                row: dict[str, Any] = {
                    "pair_id": current_pair,
                    "branch": branch,
                    "uid": str(sample["uid"]),
                    "object_uid": str(sample.get("object_uid", sample["uid"])),
                    "view_count": int(sample["visual_patch_features"].shape[0]),
                    "seed": int(seed),
                    "passed": False,
                }
                try:
                    payload = torch.load(
                        slat_root / current_pair / f"{branch}.pt", map_location="cpu"
                    )
                    slat = sparse_from_payload(payload, device)
                    decoded = mesh_decoder(slat)[0]
                    mesh = decoded.to_trimesh(transform_pose=False)
                    structure = mesh_structure_metrics(mesh)
                    if not structure["mesh_success"]:
                        raise RuntimeError("decoded Mesh is empty or non-finite")
                    obj_path = branch_dir / "mesh_canonical.obj"
                    obj_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = temporary_path(obj_path)
                    mesh.export(temporary)
                    os.replace(temporary, obj_path)
                    reopened = trimesh.load(obj_path, force="mesh", process=False)
                    if not len(reopened.vertices) or not len(reopened.faces):
                        raise RuntimeError("canonical OBJ roundtrip is empty")
                    surface = surface_metrics(
                        mesh,
                        target_mesh,
                        count=int(args.surface_samples),
                        seed=int(seed) * 1009 + phase_position * 9173,
                        thresholds=thresholds,
                    )
                    preview = (
                        save_preview(
                            decoded,
                            branch_dir,
                            frames=int(args.render_frames),
                            resolution=int(args.render_resolution),
                        )
                        if args.render_previews
                        else None
                    )
                    row.update(
                        {
                            "passed": True,
                            "structure": structure,
                            "surface": surface,
                            "canonical_obj": str(obj_path),
                            "canonical_obj_sha256": sha256_file(obj_path),
                            "preview": preview,
                            "target": target_metadata,
                        }
                    )
                    del payload, slat, decoded, mesh, reopened
                except Exception as error:
                    row["error"] = repr(error)
                branch_records.append(row)
                torch.cuda.empty_cache()
            pair_record = {
                "pair_id": current_pair,
                "uid": str(sample["uid"]),
                "object_uid": str(sample.get("object_uid", sample["uid"])),
                "seed": int(seed),
                "passed": all(row["passed"] for row in branch_records),
                "branches": branch_records,
            }
            write_json(pair_record_path, pair_record)
            records.extend(branch_records)
            print(
                f"[native_ss_stock_slat:mesh] {position + 1}/{len(dataset)} "
                f"seed={seed} uid={sample['uid']}",
                flush=True,
            )
        del target_mesh
    mesh_decoder.cpu()
    del mesh_decoder, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    expected_pairs = len(dataset) * len(seeds)
    transfer = aggregate_transfer_records(
        records,
        expected_pairs=expected_pairs,
        seeds=seeds,
        bootstrap_samples=int(args.bootstrap_samples),
        metadata_by_object=metadata_by_object,
        chamfer_win_rate_min=float(args.chamfer_win_rate_min),
        lcr_delta_min=float(args.largest_component_delta_min),
    )
    runtime_passed = bool(
        transfer["checks"]["expected_record_count"]
        and transfer["checks"]["expected_pair_count"]
        and transfer["checks"]["no_invalid_pairs"]
    )
    report = {
        "format": REPORT_FORMAT,
        "scope": "Native SS support through frozen Stock SLAT; no SLAT training",
        "exploratory": bool(smoke or ss_binding["false_checks"]),
        "passed": runtime_passed if smoke else bool(transfer["passed"]),
        "runtime_passed": runtime_passed,
        "transfer_checks_passed": bool(transfer["passed"]),
        "run_identity": run_identity,
        "model_summary": model_summary,
        "ss_support_records": ss_rows,
        "transfer": transfer,
        "decision_guard": (
            "This run may diagnose SS-to-Stock-SLAT transfer. A formal claim requires a "
            "pre-registered gate and a new untouched source-balanced holdout."
        ),
    }
    write_json(completed_report, report)
    chamfer = transfer["summary"]["chamfer_l1_improvement"]
    fscore = transfer["summary"]["fscore_0p02_delta"]
    normal = transfer["summary"]["normal_consistency_delta"]
    lcr = transfer["summary"]["largest_component_ratio_delta"]
    lines = [
        "Native SS -> frozen Stock SLAT -> Mesh transfer",
        "=" * 52,
        f"objects: {len(dataset)}",
        f"seeds: {seeds}",
        f"smoke: {smoke}",
        f"runtime_passed: {runtime_passed}",
        f"chamfer_l1_improvement: {chamfer}",
        f"fscore_0p02_delta: {fscore}",
        f"normal_consistency_delta: {normal}",
        f"largest_component_ratio_delta: {lcr}",
        f"checks: {transfer['checks']}",
        f"PASS: {report['passed']}",
        report["decision_guard"],
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    raise SystemExit(0 if smoke or report["passed"] else 2)


if __name__ == "__main__":
    main()
