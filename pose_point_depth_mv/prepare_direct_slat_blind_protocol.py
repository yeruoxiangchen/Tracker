#!/usr/bin/env python3
"""Freeze a preflight or confirmatory end-to-end Direct-SLAT blind protocol."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
from typing import Any

import torch

from pose_point_depth_mv.direct_slat_blind import (
    HOLDOUT_INTEGRITY_FORMAT,
    PREFLIGHT_REPORT_FORMAT,
    PROTOCOL_FORMAT,
    assert_unseen_holdout,
    bind_file,
    canonical_sha256,
    execution_compatibility_record,
    parse_csv,
    repeat_floors,
    select_object_rows,
    sha256_file,
    target_family_identity,
    validate_binding_tree,
    validate_execution_compatibility_record,
)
from pose_point_depth_mv.direct_slat_flow import (
    DIRECT_SLAT_CACHE_VERSION,
    DIRECT_SLAT_FLOW_VERSION,
    legacy_support_runtime_identity,
    support_generator_identity,
    support_runtime_identity,
)
from pose_point_depth_mv.direct_slat_runtime_repeat import (
    AGGREGATE_REPORT_FORMAT,
)
from pose_point_depth_mv.prepare_direct_flow_mesh_protocol import (
    bind_tree,
    resolve_hf_snapshot,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "confirmatory"), required=True)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--seen_cache_manifests", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target_decoder_audit", required=True)
    parser.add_argument("--holdout_selection_audit", default="")
    parser.add_argument("--preflight_report", default="")
    parser.add_argument("--preflight_protocol", default="")
    parser.add_argument("--holdout_integrity_audit", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--expected_objects", type=int, default=32)
    parser.add_argument("--preflight_objects", type=int, default=3)
    parser.add_argument("--slat_steps", type=int, default=25)
    parser.add_argument("--slat_cfg_strength", type=float, default=5.0)
    parser.add_argument("--slat_cfg_interval", default="0.5,1.0")
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--render_frames", type=int, default=36)
    parser.add_argument("--render_resolution", type=int, default=256)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument(
        "--attention_backend", choices=("flash_attn", "xformers"), default="flash_attn"
    )
    parser.add_argument(
        "--spconv_algo", choices=("native", "implicit_gemm"), default="native"
    )
    parser.add_argument("--cublas_workspace_config", default="")
    parser.add_argument(
        "--deterministic_algorithms", choices=("off", "warn", "on"), default="off"
    )
    parser.add_argument("--allow_tf32", choices=("false", "true"), default="false")
    parser.add_argument("--same_process_repeat_count", type=int, default=2)
    parser.add_argument("--independent_process_count", type=int, default=1)
    parser.add_argument("--runtime_calibration_report", default="")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def resolve_artifact(root: Path, row: dict[str, Any], key: str) -> Path:
    path = Path(str(row[key]))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def bind_expected_file(
    root: Path,
    row: dict[str, Any],
    key: str,
    hash_key: str,
) -> dict[str, str]:
    binding = bind_file(resolve_artifact(root, row, key))
    expected = str(row.get(hash_key, ""))
    if not expected or binding["sha256"] != expected:
        raise RuntimeError(f"manifest artifact hash mismatch: {key}={binding['path']}")
    return binding


def bind_selected_inputs(
    manifest_path: Path,
    manifest: dict[str, Any],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    root = Path(str(manifest.get("output_dir", manifest_path.parent))).resolve()
    lifting_path = Path(str(manifest["source_lifting_manifest"])).resolve()
    lifting = load_json(lifting_path)
    lifting_root = Path(
        str(lifting.get("output_dir", lifting_path.parent))
    ).resolve()
    lifting_by_uid = {
        str(row["uid"]): (index, row)
        for index, row in enumerate(lifting.get("samples", []))
    }
    samples = list(manifest["samples"])
    output = []
    for frozen in selected:
        uid = str(frozen["uid"])
        if uid not in lifting_by_uid:
            raise RuntimeError(f"selected uid is absent from source lifting cache: {uid}")
        source_index, lifting_row = lifting_by_uid[uid]
        cache_file = Path(str(lifting_row["cache_file"]))
        if not cache_file.is_absolute():
            cache_file = lifting_root / cache_file
        lifting_cache_binding = bind_file(cache_file)
        lifting_sample = torch.load(cache_file, map_location="cpu")
        if str(lifting_sample.get("uid")) != uid:
            raise RuntimeError(f"source lifting cache UID mismatch: {uid}")
        images = [bind_file(path) for path in lifting_sample["image_paths"]]
        masks = [bind_file(path) for path in lifting_sample["mask_paths"]]
        per_seed = []
        common: dict[str, Any] | None = None
        for seed_text, cache_index in frozen["cache_indices"].items():
            row = samples[int(cache_index)]
            if (
                str(row["uid"]) != uid
                or str(row["object_uid"]) != str(frozen["object_uid"])
                or int(row["support_seed"]) != int(seed_text)
            ):
                raise RuntimeError("selected direct-SLAT row identity changed")
            current_common = {
                "target_file": bind_expected_file(
                    root, row, "target_file", "target_file_sha256"
                ),
                "physical_file": bind_expected_file(
                    root, row, "physical_file", "physical_file_sha256"
                ),
                "condition_file": bind_expected_file(
                    root, row, "condition_file", "condition_file_sha256"
                ),
                "source_lh_slat": bind_expected_file(
                    root, row, "source_lh_slat", "source_lh_slat_sha256"
                ),
                "source_glb": bind_expected_file(
                    root, row, "source_glb", "source_glb_sha256"
                ),
                "ss_latent": bind_expected_file(
                    root, row, "ss_latent", "ss_latent_sha256"
                ),
            }
            if common is None:
                common = current_common
            elif current_common != common:
                raise RuntimeError(f"uid={uid} common artifacts differ across seeds")
            per_seed.append(
                {
                    "support_seed": int(seed_text),
                    "cache_index": int(cache_index),
                    "support_file": bind_expected_file(
                        root, row, "support_file", "support_file_sha256"
                    ),
                }
            )
        output.append(
            {
                **frozen,
                "source_lifting_index": int(source_index),
                "source_lifting_cache_file": lifting_cache_binding,
                "image_files": images,
                "mask_files": masks,
                "common_artifacts": common,
                "seed_artifacts": per_seed,
            }
        )
    return output


def target_run_config(cache_config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    target_source = cache_config.get("target_source")
    if not isinstance(target_source, dict):
        raise ValueError("direct-SLAT cache has no target_source identity")
    path = Path(str(target_source.get("run_config", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"local target run config is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != str(target_source.get("run_config_sha256", "")):
        raise RuntimeError("local target run config hash differs from cache binding")
    return path, load_json(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if min(
        int(args.expected_objects),
        int(args.preflight_objects),
        int(args.slat_steps),
        int(args.surface_samples),
        int(args.render_frames),
        int(args.render_resolution),
        int(args.bootstrap_samples),
    ) <= 0:
        raise ValueError("protocol counts must be positive")
    seeds = parse_csv(args.joint_seeds, int)
    interval = parse_csv(args.slat_cfg_interval, float)
    if len(interval) != 2 or not 0.0 <= interval[0] <= interval[1] <= 1.0:
        raise ValueError("slat_cfg_interval must contain two ordered values in [0,1]")

    runtime_calibration = None
    runtime_calibration_binding = None
    runtime_calibration_identity = None
    if args.runtime_calibration_report:
        calibration_path = Path(args.runtime_calibration_report).resolve()
        runtime_calibration = load_json(calibration_path)
        calibration_body = dict(runtime_calibration)
        saved_calibration_sha = str(calibration_body.pop("report_sha256", ""))
        selected_runtime = runtime_calibration.get("selected_runtime")
        if (
            runtime_calibration.get("format") != AGGREGATE_REPORT_FORMAT
            or runtime_calibration.get("complete") is not True
            or runtime_calibration.get("passed") is not True
            or not isinstance(selected_runtime, dict)
            or not isinstance(runtime_calibration.get("repeat_policy"), dict)
            or canonical_sha256(calibration_body) != saved_calibration_sha
        ):
            raise RuntimeError("runtime calibration report is invalid or has no winner")
        validate_binding_tree(
            runtime_calibration.get("input_reports"),
            "runtime_calibration.input_reports",
        )
        runtime_config = dict(selected_runtime["runtime"])
        required_runtime_fields = {
            "runtime_id",
            "attention_backend",
            "sparse_attention_backend",
            "spconv_algo",
            "amp_dtype",
            "cublas_workspace_config",
            "deterministic_algorithms",
            "allow_tf32",
            "cuda_version",
            "torch_version",
        }
        if (
            set(runtime_config) != required_runtime_fields
            or str(runtime_config["runtime_id"])
            != str(selected_runtime["runtime_id"])
            or runtime_config["attention_backend"] not in {"flash_attn", "xformers"}
            or runtime_config["sparse_attention_backend"]
            != runtime_config["attention_backend"]
            or runtime_config["spconv_algo"] not in {"native", "implicit_gemm"}
            or runtime_config["amp_dtype"] not in {"bf16", "fp16", "none"}
            or runtime_config["deterministic_algorithms"]
            not in {"off", "warn", "on"}
            or not isinstance(runtime_config["allow_tf32"], bool)
            or runtime_calibration["repeat_policy"].get("mode")
            != "multi_repeat_p95"
        ):
            raise RuntimeError("selected runtime calibration schema is invalid")
        same_process_repeat_count = int(
            selected_runtime["same_process_repeat_count"]
        )
        independent_process_count = int(
            selected_runtime["independent_process_count"]
        )
        repeat_policy = dict(runtime_calibration["repeat_policy"])
        runtime_calibration_binding = bind_file(calibration_path)
        runtime_calibration_identity = {
            "report": runtime_calibration_binding,
            "report_sha256": saved_calibration_sha,
            "selected_runtime_id": str(selected_runtime["runtime_id"]),
            "selected_runtime": runtime_config,
            "repeat_policy_sha256": canonical_sha256(repeat_policy),
        }
    else:
        runtime_config = {
            "runtime_id": "legacy",
            "attention_backend": str(args.attention_backend),
            "sparse_attention_backend": str(args.attention_backend),
            "spconv_algo": str(args.spconv_algo),
            "amp_dtype": str(args.amp_dtype),
            "cublas_workspace_config": str(args.cublas_workspace_config),
            "deterministic_algorithms": str(args.deterministic_algorithms),
            "allow_tf32": args.allow_tf32 == "true",
        }
        same_process_repeat_count = int(args.same_process_repeat_count)
        independent_process_count = int(args.independent_process_count)
        repeat_policy = {"mode": "zero_jitter"}
    if same_process_repeat_count < 2 or independent_process_count < 1:
        raise ValueError("runtime repeat counts are too small")

    manifest_path = Path(args.cache_manifest).resolve()
    manifest = load_json(manifest_path)
    if (
        manifest.get("format") != DIRECT_SLAT_CACHE_VERSION
        or manifest.get("materialized") is not True
    ):
        raise ValueError("blind protocol requires a materialized direct-SLAT cache")
    if str(manifest.get("config", {}).get("pretrained")) != str(args.pretrained):
        raise RuntimeError("holdout cache pretrained identity differs")
    if sorted(int(value) for value in manifest["config"]["ss_seeds"]) != sorted(seeds):
        raise RuntimeError("holdout cache support seeds differ from blind protocol")

    max_objects = int(args.preflight_objects) if args.mode == "preflight" else 0
    selected = select_object_rows(
        list(manifest["samples"]), seeds=seeds, max_objects=max_objects
    )
    if args.mode == "confirmatory":
        if int(manifest.get("object_count", -1)) != int(args.expected_objects):
            raise RuntimeError(
                "confirmatory cache must contain exactly the frozen object count; "
                "do not select a favorable subset"
            )
        if len(selected) != int(args.expected_objects):
            raise RuntimeError("confirmatory selection does not cover every object")

    selection_audit_binding = None
    holdout_render_binding = None
    if args.mode == "confirmatory":
        if not args.holdout_selection_audit:
            raise ValueError(
                "confirmatory mode requires the frozen holdout selection audit"
            )
        selection_audit_path = Path(args.holdout_selection_audit).resolve()
        selection_audit = load_json(selection_audit_path)
        selection_body = dict(selection_audit)
        selection_saved_hash = str(selection_body.pop("audit_sha256", ""))
        if (
            selection_audit.get("format")
            != "pose_point_depth_mv.direct_slat_holdout_manifest.v1"
            or selection_audit.get("passed") is not True
            or selection_audit.get("model_outputs_read") is not False
            or canonical_sha256(selection_body) != selection_saved_hash
            or int(selection_audit.get("selected_object_count", -1))
            != int(args.expected_objects)
            or not isinstance(selection_audit.get("candidate_freeze"), dict)
            or not isinstance(selection_audit.get("eligibility_report"), dict)
            or not isinstance(
                selection_audit.get("input_render_manifests"), list
            )
            or not selection_audit.get("input_render_manifests")
            or selection_audit.get("overlap_counts")
            != {
                "object_uid": 0,
                "source_glb_path": 0,
                "source_glb_sha256": 0,
            }
        ):
            raise RuntimeError("holdout selection audit is not a valid data-only freeze")
        validate_binding_tree(
            selection_audit.get("candidate_freeze"),
            "holdout_selection_audit.candidate_freeze",
        )
        validate_binding_tree(
            selection_audit.get("input_render_manifests"),
            "holdout_selection_audit.input_render_manifests",
        )
        validate_binding_tree(
            selection_audit.get("eligibility_report"),
            "holdout_selection_audit.eligibility_report",
        )
        render_output = dict(selection_audit.get("output_manifest", {}))
        render_path = Path(str(render_output.get("path", ""))).resolve()
        if (
            not render_path.is_file()
            or sha256_file(render_path) != str(render_output.get("sha256", ""))
        ):
            raise RuntimeError("frozen holdout render manifest changed")
        selected_cache_rows = {
            (
                str(row["object_uid"]),
                str(Path(str(row["source_glb"])).resolve()),
                str(row["source_glb_sha256"]),
            )
            for row in manifest.get("objects", [])
        }
        selected_audit_rows = {
            (
                str(row["object_uid"]),
                str(Path(str(row["source_glb"])).resolve()),
                str(row["source_glb_sha256"]),
            )
            for row in selection_audit.get("selected", [])
        }
        if (
            len(selected_cache_rows) != int(args.expected_objects)
            or len(selected_audit_rows) != int(args.expected_objects)
            or selected_cache_rows != selected_audit_rows
        ):
            raise RuntimeError(
                "holdout selection audit and Direct-SLAT cache identities differ"
            )
        selection_audit_binding = bind_file(selection_audit_path)
        holdout_render_binding = bind_file(render_path)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != DIRECT_SLAT_FLOW_VERSION:
        raise ValueError(f"unexpected Direct-SLAT checkpoint={checkpoint.get('format')!r}")
    if int(checkpoint.get("step", -1)) != 800:
        raise RuntimeError("confirmatory candidate is frozen to Direct-SLAT step 800")
    checkpoint_args = dict(checkpoint.get("args", {}))
    if str(checkpoint_args.get("pretrained")) != str(args.pretrained):
        raise RuntimeError("checkpoint pretrained identity differs")
    if float(checkpoint_args.get("support_scale", -1.0)) != 1.0:
        raise RuntimeError("confirmatory full candidate requires support_scale=1.0")
    checkpoint_support = checkpoint.get("model_summary", {}).get("support_generator")
    checkpoint_runtime = legacy_support_runtime_identity(checkpoint_support)
    holdout_runtime = support_runtime_identity(dict(manifest["config"]))
    if checkpoint_runtime != holdout_runtime:
        raise RuntimeError(
            "checkpoint and blind cache use different SS-support runtime settings"
        )
    normalization_hash = str(manifest.get("slat_normalization_hash", ""))
    checkpoint_normalization_hash = str(
        checkpoint.get("model_summary", {}).get("slat_normalization_hash", "")
    )
    if normalization_hash != checkpoint_normalization_hash:
        raise RuntimeError("holdout SLAT normalization differs from checkpoint")

    train_cache_path = Path(str(checkpoint_args.get("cache_manifest", ""))).resolve()
    train_cache = load_json(train_cache_path)
    train_target_path, train_target_run = target_run_config(dict(train_cache["config"]))
    holdout_target_path, holdout_target_run = target_run_config(
        dict(manifest["config"])
    )
    train_family = target_family_identity(train_target_run)
    holdout_family = target_family_identity(holdout_target_run)
    if train_family != holdout_family:
        raise RuntimeError(
            "training and holdout local lh-slats use different encoder/decoder/"
            "DINO/frame families"
        )

    decoder_audit_path = Path(args.target_decoder_audit).resolve()
    decoder_audit = load_json(decoder_audit_path)
    if (
        decoder_audit.get("format")
        != "pose_point_depth_mv.direct_slat_target_decoder_audit.v1"
        or decoder_audit.get("passed") is not True
    ):
        raise RuntimeError("holdout native target-decoder audit did not pass")
    if str(decoder_audit.get("cache_config_hash")) != str(
        manifest.get("config_hash")
    ):
        raise RuntimeError("target-decoder audit cache config differs from holdout")
    if decoder_audit.get("support_generator") != support_generator_identity(
        dict(manifest["config"])
    ):
        raise RuntimeError("target-decoder audit support identity differs from holdout")
    required_audit_objects = (
        int(args.expected_objects)
        if args.mode == "confirmatory"
        else len(selected)
    )
    if int(decoder_audit.get("summary", {}).get("object_count", 0)) < required_audit_objects:
        raise RuntimeError("target-decoder audit covers too few objects")
    manifest_objects = {
        str(row["object_uid"]): row for row in manifest.get("objects", [])
    }
    audited_objects: set[str] = set()
    for row in decoder_audit.get("records", []):
        object_uid = str(row.get("object_uid", ""))
        manifest_row = manifest_objects.get(object_uid)
        if manifest_row is None:
            raise RuntimeError(
                f"decoder-audited object is absent from holdout: {object_uid}"
            )
        for key in (
            "target_file_sha256",
            "source_lh_slat_sha256",
            "ss_latent_sha256",
        ):
            if str(row.get(key, "")) != str(manifest_row.get(key, "")):
                raise RuntimeError(
                    f"decoder audit artifact changed: object={object_uid} key={key}"
                )
        audited_objects.add(object_uid)
    selected_objects = {str(row["object_uid"]) for row in selected}
    missing_audits = sorted(selected_objects - audited_objects)
    if missing_audits:
        raise RuntimeError(
            f"selected objects lack native decoder audit={missing_audits[:8]}"
        )

    seen_paths = (
        [Path(value).resolve() for value in parse_csv(args.seen_cache_manifests, str)]
        if args.seen_cache_manifests
        else []
    )
    seen_payloads = [load_json(path) for path in seen_paths]
    if args.mode == "confirmatory":
        if len(seen_paths) < 2:
            raise ValueError(
                "confirmatory mode requires both seen train and seen validation manifests"
            )
        disjoint_audit = assert_unseen_holdout(manifest, seen_payloads)
    else:
        disjoint_audit = {
            "passed": False,
            "formal_requirement": False,
            "note": "preflight deliberately uses already-seen engineering objects",
        }

    preflight_binding = None
    preflight_protocol_binding = None
    frozen_preflight_protocol = None
    repeat_calibration = None
    if args.mode == "confirmatory":
        if not args.preflight_report or not args.preflight_protocol:
            raise ValueError(
                "confirmatory mode requires the frozen preflight protocol and report"
            )
        preflight_path = Path(args.preflight_report).resolve()
        preflight_protocol_path = Path(args.preflight_protocol).resolve()
        preflight = load_json(preflight_path)
        frozen_preflight_protocol = load_json(preflight_protocol_path)
        preflight_report_body = dict(preflight)
        preflight_report_sha256 = str(
            preflight_report_body.pop("report_sha256", "")
        )
        preflight_body = dict(frozen_preflight_protocol)
        preflight_protocol_sha256 = str(
            preflight_body.pop("protocol_sha256", "")
        )
        if (
            preflight.get("format") != PREFLIGHT_REPORT_FORMAT
            or preflight.get("complete") is not True
            or preflight.get("repeat_calibration", {}).get("passed") is not True
            or canonical_sha256(preflight_report_body)
            != preflight_report_sha256
            or frozen_preflight_protocol.get("format") != PROTOCOL_FORMAT
            or frozen_preflight_protocol.get("mode") != "preflight"
            or frozen_preflight_protocol.get("formal") is not False
            or canonical_sha256(preflight_body) != preflight_protocol_sha256
            or preflight.get("protocol_sha256") != preflight_protocol_sha256
        ):
            raise RuntimeError("same-model preflight repeat calibration did not pass")
        validate_binding_tree(
            frozen_preflight_protocol["bindings"],
            "preflight_protocol.bindings",
        )
        validate_binding_tree(
            frozen_preflight_protocol["runtime_bindings"],
            "preflight_protocol.runtime_bindings",
        )
        validate_binding_tree(
            frozen_preflight_protocol["sample_bindings"],
            "preflight_protocol.sample_bindings",
        )
        validate_binding_tree(
            frozen_preflight_protocol["code_bindings"],
            "preflight_protocol.code_bindings",
        )
        compatibility = validate_execution_compatibility_record(
            frozen_preflight_protocol
        )
        if (
            preflight.get("execution_compatibility_sha256")
            != compatibility["sha256"]
        ):
            raise RuntimeError(
                "preflight report and protocol execution bindings differ"
            )
        frozen_repeat_policy = dict(
            frozen_preflight_protocol.get("statistics", {}).get(
                "repeat_policy", {"mode": "zero_jitter"}
            )
        )
        expected_repeat = repeat_floors(
            preflight.get("repeat_rows", []), policy=frozen_repeat_policy
        )
        repeat_count = int(
            frozen_preflight_protocol.get("runtime", {}).get(
                "same_process_repeat_count", 2
            )
        )
        repeat_pairs_per_side = repeat_count * (repeat_count - 1) // 2
        expected_repeat["passed"] = bool(
            expected_repeat["passed"]
            and preflight.get("all_records_passed") is True
            and len(preflight.get("repeat_rows", []))
            == 2
            * int(preflight.get("expected_pair_count", -1))
            * repeat_pairs_per_side
        )
        if expected_repeat != preflight.get("repeat_calibration"):
            raise RuntimeError(
                "preflight repeat rows and frozen calibration differ"
            )
        preflight_binding = bind_file(preflight_path)
        preflight_protocol_binding = bind_file(preflight_protocol_path)
        repeat_calibration = dict(preflight["repeat_calibration"])
    else:
        repeat_calibration = {
            "passed": False,
            "policy_mode": str(repeat_policy["mode"]),
            "median_abs": {
                "chamfer_l1_abs": 0.0,
                "fscore_0p02_abs": 0.0,
                "largest_component_ratio_abs": 0.0,
                "boundary_edge_count_abs": 0.0,
                "boundary_total_length_abs": 0.0,
                "nonmanifold_edge_count_abs": 0.0,
                "component_count_abs": 0.0,
            },
            "p95_abs": {
                "chamfer_l1_abs": 0.0,
                "fscore_0p02_abs": 0.0,
                "largest_component_ratio_abs": 0.0,
                "boundary_edge_count_abs": 0.0,
                "boundary_total_length_abs": 0.0,
                "nonmanifold_edge_count_abs": 0.0,
                "component_count_abs": 0.0,
            },
            "worst_group_p95_abs": {
                "chamfer_l1_abs": 0.0,
                "fscore_0p02_abs": 0.0,
                "largest_component_ratio_abs": 0.0,
                "boundary_edge_count_abs": 0.0,
                "boundary_total_length_abs": 0.0,
                "nonmanifold_edge_count_abs": 0.0,
                "component_count_abs": 0.0,
            },
            "max_abs": {
                "chamfer_l1_abs": 0.0,
                "fscore_0p02_abs": 0.0,
                "largest_component_ratio_abs": 0.0,
                "boundary_edge_count_abs": 0.0,
                "boundary_total_length_abs": 0.0,
                "nonmanifold_edge_count_abs": 0.0,
                "component_count_abs": 0.0,
            },
            "note": "populated by the preflight exporter",
        }

    sample_bindings = bind_selected_inputs(manifest_path, manifest, selected)
    trellis_snapshot, trellis_ref = resolve_hf_snapshot(args.pretrained)
    code_paths = (
        TRACKER_ROOT / "pose_point_depth_mv" / "direct_slat_blind.py",
        TRACKER_ROOT
        / "pose_point_depth_mv"
        / "prepare_direct_slat_holdout_manifest.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "prepare_direct_slat_blind_protocol.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "export_direct_slat_blind_holdout.py",
        TRACKER_ROOT
        / "pose_point_depth_mv"
        / "freeze_direct_slat_holdout_integrity.py",
        TRACKER_ROOT
        / "pose_point_depth_mv"
        / "package_direct_slat_public_bundle.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "freeze_direct_slat_ratings.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "finalize_direct_slat_blind_holdout.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "direct_slat_flow.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "direct_slat_data.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "direct_flow.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "train_direct_flow.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "train_direct_slat_flow.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "export_direct_flow_mesh_pairs.py",
        TRACKER_ROOT
        / "pose_point_depth_mv"
        / "direct_slat_runtime_repeat.py",
        TRACKER_ROOT
        / "pose_point_depth_mv"
        / "diagnose_direct_slat_runtime_repeat.py",
        TRACKER_ROOT
        / "pose_point_depth_mv"
        / "aggregate_direct_slat_runtime_repeat.py",
        TRACKER_ROOT
        / "pose_point_depth_mv"
        / "launch_direct_slat_blind_runtime.py",
    )
    integrity_binding = None
    if args.mode == "confirmatory":
        if not args.holdout_integrity_audit:
            raise ValueError(
                "confirmatory mode requires the post-selection integrity audit"
            )
        integrity_path = Path(args.holdout_integrity_audit).resolve()
        integrity = load_json(integrity_path)
        integrity_body = dict(integrity)
        integrity_sha256 = str(integrity_body.pop("integrity_sha256", ""))
        if (
            integrity.get("format") != HOLDOUT_INTEGRITY_FORMAT
            or integrity.get("passed") is not True
            or int(integrity.get("object_count", -1))
            != int(args.expected_objects)
            or canonical_sha256(integrity_body) != integrity_sha256
        ):
            raise RuntimeError("post-selection holdout integrity audit is invalid")
        validate_binding_tree(
            integrity.get("bindings"), "holdout_integrity.bindings"
        )
        if (
            integrity["bindings"]["selection_audit"]
            != selection_audit_binding
            or integrity["bindings"]["cache_manifest"]
            != bind_file(manifest_path)
            or integrity["bindings"]["target_decoder_audit"]
            != bind_file(decoder_audit_path)
        ):
            raise RuntimeError(
                "holdout integrity audit and formal protocol inputs differ"
            )
        integrity_binding = bind_file(integrity_path)

    bindings = {
        "holdout_cache_manifest": bind_file(manifest_path),
        "source_lifting_manifest": bind_file(manifest["source_lifting_manifest"]),
        "ss_flow_checkpoint": bind_file(manifest["config"]["ss_flow_checkpoint"]),
        "direct_slat_checkpoint": bind_file(checkpoint_path),
        "training_cache_manifest": bind_file(train_cache_path),
        "target_decoder_audit": bind_file(decoder_audit_path),
        "training_target_run_config": bind_file(train_target_path),
        "holdout_target_run_config": bind_file(holdout_target_path),
        "seen_cache_manifests": [bind_file(path) for path in seen_paths],
        "holdout_selection_audit": selection_audit_binding,
        "holdout_render_manifest": holdout_render_binding,
        "preflight_report": preflight_binding,
        "preflight_protocol": preflight_protocol_binding,
        "holdout_integrity_audit": integrity_binding,
        "runtime_calibration_report": runtime_calibration_binding,
    }
    if (
        bindings["ss_flow_checkpoint"]["sha256"]
        != str(manifest["config"]["ss_flow_checkpoint_sha256"])
    ):
        raise RuntimeError("holdout cache SS checkpoint binding changed")
    runtime_bindings = {
        "trellis_vggt": {
            "repo_id": str(args.pretrained),
            "snapshot_path": str(trellis_snapshot),
            "ref_main": bind_file(trellis_ref),
            "files": bind_tree(trellis_snapshot),
        }
    }
    code_bindings = {
        str(path.relative_to(TRACKER_ROOT)): bind_file(path) for path in code_paths
    }
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=TRACKER_ROOT, text=True
    ).strip()

    blind_key = secrets.token_bytes(32)
    commitment = hashlib.sha256(blind_key).hexdigest()
    protocol = {
        "format": PROTOCOL_FORMAT,
        "mode": args.mode,
        "formal": args.mode == "confirmatory",
        "protocol_name": str(args.protocol_name),
        "purpose": (
            "frozen end-to-end A/C comparison: native stock SS+SLAT versus "
            "corrected SS+full Direct-SLAT step800"
        ),
        "candidate": {
            "ss_checkpoint_step": 900,
            "direct_slat_checkpoint_step": 800,
            "support_scale": 1.0,
            "ablation_status": "LoRA-only and adapter-only paused",
        },
        "bindings": bindings,
        "sample_bindings": sample_bindings,
        "runtime_bindings": runtime_bindings,
        "code_bindings": code_bindings,
        "git_commit": git_commit,
        "pretrained_id": str(args.pretrained),
        "pretrained": str(trellis_snapshot),
        "support_runtime_identity": holdout_runtime,
        "target_family_identity": holdout_family,
        "runtime_calibration_identity": runtime_calibration_identity,
        "holdout_disjoint_audit": disjoint_audit,
        "selection": {
            "method": (
                "all objects in an exactly-sized unseen cache"
                if args.mode == "confirmatory"
                else "deterministic prefix of sorted already-seen objects"
            ),
            "object_count": len(selected),
            "view_counts": dict(
                sorted(Counter(int(row["view_count"]) for row in selected).items())
            ),
            "rows": selected,
        },
        "runtime": {
            "device_type": "cuda",
            **runtime_config,
            "same_process_repeat_count": same_process_repeat_count,
            "independent_process_count": independent_process_count,
        },
        "sampling": {
            "joint_seeds": seeds,
            "branches": {
                "stock": "native stock SS coords + native stock SLAT Flow",
                "full": "cached corrected SS coords + full Direct-SLAT step800",
            },
            "ss": {
                "steps": int(manifest["config"]["ss_steps"]),
                "cfg_strength": float(manifest["config"]["cfg_strength"]),
                "guidance_rescale": float(
                    manifest["config"]["guidance_rescale"]
                ),
                "rescale_t": float(manifest["config"]["rescale_t"]),
                "noise_seed_formula": (
                    "joint_seed*1000003 + source_lifting_dataset_index*1009"
                ),
                "shared_noise": (
                    "stock SS is replayed from the same seed used by cached corrected SS"
                ),
            },
            "slat": {
                "steps": int(args.slat_steps),
                "cfg_strength": float(args.slat_cfg_strength),
                "cfg_interval": [float(value) for value in interval],
                "rescale_t": float(args.slat_rescale_t),
                "noise_field": "one FP32 [64,64,64,8] coordinate-keyed field per object/seed",
                "noise_seed_formula": (
                    "joint_seed*2000003 + object_position*2017 + 7919"
                ),
            },
        },
        "mesh": {
            "surface_samples": int(args.surface_samples),
            "fscore_thresholds": [0.01, 0.02, 0.05],
            "render_frames": int(args.render_frames),
            "render_resolution": int(args.render_resolution),
            "canonical_frame": (
                "raw decoder mesh with transform_pose=false; no ICP, cleanup, "
                "per-branch normalization, exclusion, or best-of-seed"
            ),
        },
        "statistics": {
            "unit": "average paired seeds within object, then bootstrap objects",
            "bootstrap_samples": int(args.bootstrap_samples),
            "repeat_policy": repeat_policy,
            "repeat_floors": dict(
                repeat_calibration[
                    "worst_group_p95_abs"
                    if repeat_policy["mode"] == "multi_repeat_p95"
                    else "max_abs"
                ]
            ),
            "preflight_repeat_calibration": repeat_calibration,
            "checks": {
                "chamfer_object_win_rate_min": 0.55,
                "minimum_nonnegative_seed_directions": 2,
                "per_seed_chamfer_mean_min": -0.002,
                "largest_component_ratio_mean_delta_min": -0.02,
                "largest_component_ratio_object_delta_min": -0.10,
                "largest_component_ratio_pair_delta_min": -0.15,
                "boundary_edge_count_mean_increase_max": 32.0,
                "boundary_edge_count_object_increase_max": 256.0,
                "boundary_edge_count_pair_increase_max": 512.0,
                "boundary_total_length_mean_increase_max": 0.02,
                "boundary_total_length_object_increase_max": 0.25,
                "boundary_total_length_pair_increase_max": 0.50,
                "nonmanifold_edge_count_mean_increase_max": 8.0,
                "nonmanifold_edge_count_object_increase_max": 64.0,
                "nonmanifold_edge_count_pair_increase_max": 128.0,
                "component_count_mean_increase_max": 1.0,
                "component_count_object_increase_max": 10.0,
                "component_count_pair_increase_max": 20.0,
                "topology_object_worsening_rate_max": 0.10,
                "watertight_rate_delta_min": -0.05,
                "zero_boundary_rate_delta_min": -0.05,
                "nonmanifold_free_rate_delta_min": 0.0,
            },
            "rating_checks": {
                "main_structure_mean_delta_min": 0.0,
                "main_structure_ci_lower_min": -0.25,
                "defect_mean_delta_max": 0.0,
                "severe_defect_rate_delta_max": 0.05,
                "severe_defect_rate_ci_upper_max": 0.10,
                "overall_score_mean_min_exclusive": 0.0,
                "overall_score_ci_lower_min": 0.0,
                "overall_preference_mean_min_exclusive": 0.0,
                "overall_preference_ci_lower_min": 0.0,
            },
        },
        "blinding": {
            "pair_id": "SHA256(protocol_name|uid|seed) prefix",
            "side_assignment": (
                "HMAC-SHA256(private blind key, protocol_name|uid|seed) parity"
            ),
            "blind_key_sha256_commitment": commitment,
            "public_bundle": (
                "blind_pairs/, blind_manifest.csv, and score_templates/ only"
            ),
            "minimum_independent_raters": 3,
            "rating_schema": {
                "main_structure_and_overall_score": "integer 1..5, higher is better",
                "defect_dimensions": (
                    "missing_parts, floating_fragments, thin_spikes, and "
                    "holes_open_boundaries are integer 0..3, lower is better; "
                    "severity >=2 is severe"
                ),
                "overall_preference": "A, B, or tie",
            },
            "unblinding": (
                "one finalizer run after complete score files are frozen by SHA"
            ),
        },
        "failure_policy": (
            "retain every frozen object/seed/side; failed, empty, or nonfinite "
            "outputs fail the protocol and cannot be excluded or selectively rerun"
        ),
        "decision_timing": (
            "confirmatory exporter emits no science decision and no branch mapping; "
            "automatic and blind-review gates are evaluated only by the finalizer"
        ),
    }
    protocol["execution_compatibility"] = execution_compatibility_record(protocol)
    if frozen_preflight_protocol is not None:
        frozen_compatibility = validate_execution_compatibility_record(
            frozen_preflight_protocol
        )
        if protocol["execution_compatibility"] != frozen_compatibility:
            raise RuntimeError(
                "confirmatory execution differs from the frozen preflight; "
                "code, runtime, checkpoint, sampling, Mesh, and amp bindings "
                "must match exactly"
            )
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    key_path = output_dir / "blind_key.txt"
    key_path.write_text(blind_key.hex() + "\n", encoding="ascii")
    key_path.chmod(0o600)
    print(
        json.dumps(
            {
                "protocol": str(output_dir / "protocol.json"),
                "protocol_sha256": protocol["protocol_sha256"],
                "mode": args.mode,
                "objects": len(selected),
                "pairs": len(selected) * len(seeds),
                "blind_key_commitment": commitment,
                "target_family_hash": holdout_family["hash"],
                "support_runtime_hash": holdout_runtime["hash"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
