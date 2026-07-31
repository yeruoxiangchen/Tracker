#!/usr/bin/env python3
"""Freeze the Stage-B paired Mesh protocol from a passed Stage-A rollout."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any

import numpy as np
import torch


PROTOCOL_FORMAT = "pose_point_depth_mv.direct_flow_mesh_protocol.v1"
TRACKER_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_int_csv(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("seed CSV must be non-empty and unique")
    return values


def resolve_recorded_path(text: str, *, base: Path) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def bind_file(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def resolve_hf_snapshot(repo_id: str) -> tuple[Path, Path]:
    direct = Path(repo_id)
    if direct.is_dir() and (direct / "pipeline.json").is_file():
        return direct.resolve(), direct.resolve()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_root = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref_path = repo_root / "refs" / "main"
    if not ref_path.is_file():
        raise FileNotFoundError(f"missing local HF ref for {repo_id}: {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot = repo_root / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing local HF snapshot for {repo_id}: {snapshot}")
    return snapshot.resolve(), ref_path.resolve()


def bind_tree(root: Path, *, suffixes: set[str] | None = None) -> list[dict[str, str]]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes)
    )
    if not paths:
        raise FileNotFoundError(f"no bindable files below {root}")
    return [
        {
            # Hugging Face snapshots are symlink trees.  Preserve the logical
            # filename that the loader opens as well as its resolved target.
            "path": str(path.absolute()),
            "resolved_path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def bind_selected_sample_inputs(
    cache_manifest: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("cache manifest has no samples")
    root = Path(payload.get("output_dir", cache_manifest.parent))
    if not root.is_absolute():
        root = cache_manifest.parent / root
    output = []
    for frozen in rows:
        manifest_row = samples[int(frozen["sample_index"])]
        if str(manifest_row["uid"]) != str(frozen["uid"]):
            raise RuntimeError("frozen sample index/UID differs from cache manifest")
        cache_file = Path(manifest_row["cache_file"])
        if not cache_file.is_absolute():
            cache_file = root / cache_file
        sample = torch.load(cache_file, map_location="cpu")
        if str(sample.get("uid")) != str(frozen["uid"]):
            raise RuntimeError("sample cache UID differs from frozen protocol")
        latent_path = Path(sample["ss_latent"]).resolve()
        with np.load(latent_path) as latent:
            source_glb = Path(str(latent["source_glb"])).resolve()
        images = [bind_file(path) for path in sample["image_paths"]]
        masks = [bind_file(path) for path in sample["mask_paths"]]
        output.append(
            {
                "sample_index": int(frozen["sample_index"]),
                "rollout_position": int(frozen["rollout_position"]),
                "uid": str(frozen["uid"]),
                "object_uid": str(frozen["object_uid"]),
                "cache_file": bind_file(cache_file),
                "ss_latent": bind_file(latent_path),
                "source_glb": bind_file(source_glb),
                "image_paths": images,
                "mask_paths": masks,
            }
        )
    return output


def rollout_gate(report: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    rollout = report.get("rollout")
    if not isinstance(rollout, dict):
        raise ValueError("source report has no rollout")
    correct = rollout["delta_vs_stock"]["correct"]
    checks = {
        "iou_mean_positive": float(correct["iou"]["mean"])
        > float(thresholds["iou_mean_delta_min_exclusive"]),
        "iou_median_positive": float(correct["iou"]["median"])
        > float(thresholds["iou_median_delta_min_exclusive"]),
        "iou_object_win": float(correct["iou"]["object_win_rate"])
        >= float(thresholds["iou_object_win_rate_min"]),
        "precision_mean_nonnegative": float(correct["precision"]["mean"])
        >= float(thresholds["precision_mean_delta_min"]),
        "precision_object_win": float(correct["precision"]["object_win_rate"])
        >= float(thresholds["precision_object_win_rate_min"]),
        "largest_component_non_degrading": float(
            correct["largest_component_ratio"]["mean"]
        )
        >= float(thresholds["largest_component_ratio_mean_delta_min"]),
    }
    strong_required = bool(
        thresholds.get("strong_pass_requires_iou_bootstrap_ci_lower_positive", True)
    )
    ci_lower = float(correct["iou"]["bootstrap_mean_95_ci"][0])
    checks["iou_bootstrap_ci_lower_positive"] = (not strong_required) or ci_lower > 0.0
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "correct_delta_vs_stock": correct,
        "note": (
            "This gate is recomputed from the external frozen thresholds; "
            "report_only selected_passed is deliberately ignored."
        ),
    }


def ordered_rollout_uids(report: dict[str, Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for row in report["rollout"]["records"]:
        if str(row["branch"]) != "stock":
            continue
        uid = str(row["uid"])
        if uid not in seen:
            seen.add(uid)
            output.append(uid)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_report", required=True)
    parser.add_argument("--rollout_protocol_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument(
        "--max_objects",
        type=int,
        default=0,
        help="0 freezes all Stage-A objects; a positive value takes the existing order.",
    )
    parser.add_argument("--slat_steps", type=int, default=25)
    parser.add_argument("--slat_cfg_strength", type=float, default=5.0)
    parser.add_argument("--slat_cfg_interval", default="0.5,1.0")
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--fscore_thresholds", default="0.01,0.02,0.05")
    parser.add_argument("--render_frames", type=int, default=36)
    parser.add_argument("--render_resolution", type=int, default=256)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument(
        "--deterministic_repeat_slat_max_abs",
        type=float,
        default=0.1,
        help="Maximum native-spconv repeat jitter in decoded SLAT features.",
    )
    parser.add_argument(
        "--deterministic_repeat_mesh_vertex_chamfer_l1_max",
        type=float,
        default=1.0e-4,
        help="Maximum all-vertex Chamfer-L1 between repeated native Mesh runs.",
    )
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.rollout_report).resolve()
    protocol_dir = Path(args.rollout_protocol_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if int(args.max_objects) < 0:
        raise ValueError("max_objects must be non-negative")
    if int(args.surface_samples) <= 0 or int(args.bootstrap_samples) <= 0:
        raise ValueError("surface/bootstrap samples must be positive")
    if (
        float(args.deterministic_repeat_slat_max_abs) < 0.0
        or float(args.deterministic_repeat_mesh_vertex_chamfer_l1_max) < 0.0
    ):
        raise ValueError("deterministic repeat tolerances must be non-negative")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection_path = protocol_dir / "selection.json"
    indices_path = protocol_dir / "indices.txt"
    thresholds_path = protocol_dir / "thresholds.json"
    for path in (selection_path, indices_path, thresholds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rollout_thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    gate = rollout_gate(report, rollout_thresholds)
    if not gate["passed"]:
        raise RuntimeError(f"Stage-A rollout did not pass the frozen strong gate: {gate}")

    source_rows = list(source_selection["selected"])
    if len(source_rows) != 32 or len({str(row["object_uid"]) for row in source_rows}) != 32:
        raise ValueError("expected the frozen Stage-A 32-object selection")
    rollout_uids = ordered_rollout_uids(report)
    selected_uids = [str(row["uid"]) for row in source_rows]
    if rollout_uids != selected_uids:
        raise RuntimeError("Stage-A report order differs from frozen selection order")
    if int(report["rollout"]["sample_count"]) != len(source_rows):
        raise RuntimeError("Stage-A report sample count differs from selection")

    selected_count = len(source_rows)
    if int(args.max_objects) > 0:
        selected_count = min(int(args.max_objects), len(source_rows))
    chosen = []
    for rollout_position, row in enumerate(source_rows[:selected_count]):
        chosen.append(
            {
                "rollout_position": int(rollout_position),
                "sample_index": int(row["sample_index"]),
                "uid": str(row["uid"]),
                "object_uid": str(row["object_uid"]),
                "views": int(row["views"]),
            }
        )
    if len({row["object_uid"] for row in chosen}) != len(chosen):
        raise RuntimeError("Mesh selection is not one sequence per object")

    seeds = parse_int_csv(args.joint_seeds)
    rollout_seeds = [int(value) for value in report["rollout"]["seeds"]]
    if any(seed not in rollout_seeds for seed in seeds):
        raise ValueError(
            f"joint seeds must be Stage-A rollout seeds: {seeds} vs {rollout_seeds}"
        )
    interval = [float(value) for value in args.slat_cfg_interval.split(",")]
    if len(interval) != 2 or not 0.0 <= interval[0] <= interval[1] <= 1.0:
        raise ValueError("slat_cfg_interval must be two ordered values in [0,1]")
    fscore_thresholds = [
        float(value) for value in args.fscore_thresholds.split(",") if value.strip()
    ]
    if not fscore_thresholds or any(value <= 0.0 for value in fscore_thresholds):
        raise ValueError("F-score thresholds must be positive")
    if not any(abs(value - 0.02) <= 1.0e-12 for value in fscore_thresholds):
        raise ValueError("formal Stage-B protocol requires F-score threshold 0.02")

    report_args = report["args"]
    cache_manifest_path = resolve_recorded_path(
        report_args["cache_manifest"], base=TRACKER_ROOT
    )
    bindings = {
        "rollout_report": bind_file(report_path),
        "rollout_selection": bind_file(selection_path),
        "rollout_indices": bind_file(indices_path),
        "rollout_thresholds": bind_file(thresholds_path),
        "cache_manifest": bind_file(cache_manifest_path),
        "flow_checkpoint": bind_file(
            resolve_recorded_path(report_args["flow_checkpoint"], base=TRACKER_ROOT)
        ),
        "correspondence_checkpoint": bind_file(
            resolve_recorded_path(
                report_args["correspondence_checkpoint"], base=TRACKER_ROOT
            )
        ),
        "n3_report": bind_file(
            resolve_recorded_path(report_args["n3_report"], base=TRACKER_ROOT)
        ),
    }
    sample_bindings = bind_selected_sample_inputs(cache_manifest_path, chosen)

    pretrained_id = str(report_args["pretrained"])
    trellis_snapshot, trellis_ref = resolve_hf_snapshot(pretrained_id)
    vggt_snapshot, vggt_ref = resolve_hf_snapshot("Stable-X/vggt-object-v0-1")
    birefnet_snapshot, birefnet_ref = resolve_hf_snapshot("ZhengPeng7/BiRefNet")
    dino_hub_root = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    dino_checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / "dinov2_vitl14_reg4_pretrain.pth"
    )
    dreamsim_root = TRACKER_ROOT / "weights" / "dreamsim"
    runtime_bindings = {
        "trellis_vggt": {
            "repo_id": pretrained_id,
            "snapshot_path": str(trellis_snapshot),
            "ref_main": bind_file(trellis_ref),
            "files": bind_tree(trellis_snapshot),
        },
        "vggt_object": {
            "repo_id": "Stable-X/vggt-object-v0-1",
            "snapshot_path": str(vggt_snapshot),
            "ref_main": bind_file(vggt_ref),
            "files": bind_tree(vggt_snapshot),
        },
        "birefnet": {
            "repo_id": "ZhengPeng7/BiRefNet",
            "snapshot_path": str(birefnet_snapshot),
            "ref_main": bind_file(birefnet_ref),
            "files": bind_tree(birefnet_snapshot),
        },
        "dinov2": {
            "hub_source": str(dino_hub_root.resolve()),
            "python_files": bind_tree(dino_hub_root, suffixes={".py"}),
            "checkpoint": bind_file(dino_checkpoint),
        },
        "dreamsim": {
            "root": str(dreamsim_root.resolve()),
            "files": bind_tree(dreamsim_root),
        },
        "local_runtime_python": {
            "trellis": bind_tree(
                TRACKER_ROOT / "ReconViaGen" / "trellis", suffixes={".py"}
            ),
            "vggt": bind_tree(
                TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt", suffixes={".py"}
            ),
        },
    }
    code_paths = (
        TRACKER_ROOT / "pose_point_depth_mv" / "prepare_direct_flow_mesh_protocol.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "export_direct_flow_mesh_pairs.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "eval_direct_flow.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "direct_flow.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "train_direct_flow.py",
        TRACKER_ROOT / "pose_point_depth_mv" / "tools" / "ninja",
        TRACKER_ROOT
        / "ReconViaGen"
        / "trellis"
        / "pipelines"
        / "trellis_image_to_3d.py",
        TRACKER_ROOT
        / "ReconViaGen"
        / "trellis"
        / "pipelines"
        / "samplers"
        / "flow_euler.py",
        TRACKER_ROOT
        / "ReconViaGen"
        / "trellis"
        / "modules"
        / "sparse"
        / "basic.py",
        TRACKER_ROOT
        / "ReconViaGen"
        / "trellis"
        / "representations"
        / "mesh"
        / "cube2mesh.py",
        TRACKER_ROOT
        / "ReconViaGen"
        / "trellis"
        / "models"
        / "structured_latent_vae"
        / "decoder_mesh.py",
        TRACKER_ROOT / "ReconViaGen" / "trellis" / "utils" / "render_utils.py",
        TRACKER_ROOT
        / "reconvggt_ar_adapter_a"
        / "train_pointpose_ss_lora.py",
        TRACKER_ROOT / "reconvggt_ar_adapter_a" / "inspect_and_sanity.py",
    )
    code_bindings = {str(path.relative_to(TRACKER_ROOT)): bind_file(path) for path in code_paths}
    ninja_label = "pose_point_depth_mv/tools/ninja"
    code_bindings[ninja_label]["mode"] = oct(
        Path(code_bindings[ninja_label]["path"]).stat().st_mode & 0o777
    )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=TRACKER_ROOT, text=True
    ).strip()

    blind_key = secrets.token_bytes(32)
    blind_key_sha256 = hashlib.sha256(blind_key).hexdigest()
    protocol = {
        "format": PROTOCOL_FORMAT,
        "protocol_name": str(args.protocol_name),
        "purpose": "Stage-B frozen same-SS-noise/same-coordinate-SLAT-noise Mesh comparison",
        "claim_scope": (
            "stock/base versus correct direct-Flow configuration; no pose/depth-specific "
            "mechanism claim because Stage-A controls tracked correct"
        ),
        "source_rollout_gate": gate,
        "bindings": bindings,
        "sample_bindings": sample_bindings,
        "runtime_bindings": runtime_bindings,
        "code_bindings": code_bindings,
        "git_commit": git_commit,
        "pretrained_id": pretrained_id,
        # Use the immutable local snapshot path instead of a mutable Hub ref.
        "pretrained": str(trellis_snapshot),
        "runtime": {
            "device_type": "cuda",
            "amp_dtype": str(args.amp_dtype),
        },
        "selection": {
            "method": "prefix of the pre-existing frozen Stage-A order; no result-based selection",
            "sample_count": len(chosen),
            "object_count": len(chosen),
            "view_counts": dict(sorted(Counter(row["views"] for row in chosen).items())),
            "rows": chosen,
        },
        "sampling": {
            "joint_seeds": seeds,
            "branches": ["stock", "correct"],
            "ss": {
                "steps": int(report_args["rollout_steps"]),
                "cfg_strength": float(report_args["cfg_strength"]),
                "guidance_rescale": float(report_args["guidance_rescale"]),
                "rescale_t": float(report_args["rescale_t"]),
                "seed_formula": "joint_seed*1000003 + rollout_position*1009",
            },
            "slat": {
                "steps": int(args.slat_steps),
                "cfg_strength": float(args.slat_cfg_strength),
                "cfg_interval": interval,
                "rescale_t": float(args.slat_rescale_t),
                "noise_field": "one FP32 canonical [64,64,64,C] field per uid/seed",
                "seed_formula": "joint_seed*2000003 + rollout_position*2017 + 7919",
            },
        },
        "mesh": {
            "decoder": "frozen native Stable-X/trellis-vggt-v0-2 slat_decoder_mesh",
            "canonical_export": "raw MeshExtractResult, transform_pose=false, no simplification/ICP/autoframe",
            "view_export": "raw vertex-color GLB, transform_pose=true, no texture baking",
            "surface_samples": int(args.surface_samples),
            "fscore_thresholds": fscore_thresholds,
            "primary_fscore_threshold": 0.02,
            "render_frames": int(args.render_frames),
            "render_resolution": int(args.render_resolution),
        },
        "audits": {
            "rollout_metric_max_abs_diff": 1.0e-9,
            "cached_vs_recomputed_stock_condition_max_abs": 1.0e-2,
            "shared_slat_noise_common_coord_max_abs": 0.0,
            "deterministic_repeat_slat_max_abs": float(
                args.deterministic_repeat_slat_max_abs
            ),
            "deterministic_repeat_mesh_vertex_chamfer_l1_max": float(
                args.deterministic_repeat_mesh_vertex_chamfer_l1_max
            ),
        },
        "statistics": {
            "unit": "average paired seeds within object, then summarize/bootstrap objects",
            "bootstrap_samples": int(args.bootstrap_samples),
            "primary_metric": "canonical_gt_surface_chamfer_l1_improvement_stock_minus_correct",
            "checks": {
                "chamfer_mean_improvement_positive": True,
                "chamfer_median_improvement_positive": True,
                "chamfer_object_win_rate_min": 0.55,
                "strong_chamfer_bootstrap_ci_lower_positive": True,
                "fscore_0p02_mean_delta_min": 0.0,
                "mesh_success_rate_delta_min": 0.0,
                "largest_component_ratio_mean_delta_min": -0.02,
                "minimum_nonnegative_seed_directions": 2,
            },
        },
        "failure_policy": (
            "Every frozen pair is retained. Empty/nonfinite/export-failed meshes are failures; "
            "no post-unblinding exclusion or best-of-N rerun."
        ),
        "blinding": {
            "pair_id": "SHA256(protocol_name|uid|seed) prefix",
            "side_assignment": "HMAC-SHA256(private blind key, protocol_name|uid|seed) parity",
            "blind_key_sha256_commitment": blind_key_sha256,
            "private_key_file": "blind_key.txt (not copied into Mesh output; keep from raters)",
            "mapping_file": "unblinding_key.json; keep hidden from raters",
            "rater_bundle": (
                "blind_pairs/ plus blind_manifest.csv only; all branch-labelled metrics "
                "and pair records live under private_records/"
            ),
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "indices.txt").write_text(
        ",".join(str(row["sample_index"]) for row in chosen) + "\n",
        encoding="utf-8",
    )
    blind_key_path = output_dir / "blind_key.txt"
    blind_key_path.write_text(blind_key.hex() + "\n", encoding="ascii")
    blind_key_path.chmod(0o600)
    print(json.dumps({
        "protocol": str(output_dir / "protocol.json"),
        "protocol_sha256": protocol["protocol_sha256"],
        "objects": len(chosen),
        "view_counts": protocol["selection"]["view_counts"],
        "seeds": seeds,
        "pair_count": len(chosen) * len(seeds),
        "passed_source_rollout_gate": gate["passed"],
        "blind_key_sha256_commitment": blind_key_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
