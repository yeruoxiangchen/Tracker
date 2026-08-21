#!/usr/bin/env python3
"""Build frozen-SS support and native-condition caches for direct SLAT Flow.

This builder deliberately requires pre-encoded, ground-truth-derived SLAT
latents in the ProObjaverse ``lh-slats`` file schema.  They may come from the
official release or from the separately audited local multi-view encoder
pipeline.  Stock SLAT samples are never accepted as training targets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from ar_ss_flow.shared_object_preprocessing import (  # noqa: E402
    SHARED_OBJECT_PREPROCESSING_VERSION,
    prepare_shared_object_views,
)
from pose_point_depth_mv.direct_flow import (  # noqa: E402
    DIRECT_FLOW_VERSION,
    PositivePhysicalRolloutFlow,
    load_frozen_correspondence_head,
    make_direct_evidence_bundle,
    validate_n3_checkpoint,
)
from pose_point_depth_mv.direct_slat_flow import (  # noqa: E402
    DIRECT_SLAT_CACHE_VERSION,
    SLAT_SUPPORT_MAPPING_VERSION,
    canonical_json_sha256,
    validate_sparse_target_alignment,
)
from pose_point_depth_mv.train_direct_flow import build_direct_components  # noqa: E402
from pose_point_depth_mv.native_3d_condition import (  # noqa: E402
    NATIVE_SS_FLOW_VERSION,
    PositiveNativeSSRolloutFlow,
    build_native_ss_components,
    load_trainable_state_dict as load_native_trainable_state,
)
from pose_point_depth_mv.evaluate_native_ss_genrecon import (  # noqa: E402
    sampling_params as native_ss_sampling_params,
)
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import (  # noqa: E402
    load_ss_evidence,
    make_sampling_namespace,
)
from pose_point_depth_mv.native_ss_genrecon import (  # noqa: E402
    NATIVE_SS_GENRECON_VERSION,
    NativeSSCalibratedCFGFlow,
    build_native_ss_genrecon_components,
    load_trainable_state_dict as load_native_ss_genrecon_state,
    validate_native_ss_genrecon_checkpoint,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond  # noqa: E402
from reconvggt_ar_adapter_a.pointpose_ss_condition import load_partial_state  # noqa: E402
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import rgba_images  # noqa: E402


LEGACY_CONDITION_PREPROCESSING_VERSION = (
    "reconvggt_ar_adapter_a.rgba_images_pipeline_preprocess.v1"
)
PRECOMPUTED_NATIVE_V2_CONDITION_VERSION = (
    "pose_point_depth_mv.precomputed_native_v2_slat_condition.v1"
)
DIRECT_DINO_ONLY_PROVENANCE_VERSION = (
    "pose_point_depth_mv.direct_dino_only_lifting_builder.v1"
)


def prepare_native_condition_images(
    sample: dict[str, Any], pipeline: Any
) -> tuple[list[Any], dict[str, Any]]:
    """Replay the image geometry frozen by the lifting cache.

    New Native-SS caches bind a foreground-square crop in
    ``preprocessing.shared_geometry``.  Falling back to the historical
    ``pipeline.preprocess_image`` path for those samples changes the image
    tokens substantially even though the source files are identical.  Legacy
    lifting caches have no shared-geometry record and retain the old path.
    """

    preprocessing = dict(sample.get("preprocessing", {}))
    shared = dict(preprocessing.get("shared_geometry", {}))
    if not shared:
        return (
            rgba_images(sample["image_paths"], sample["mask_paths"], pipeline),
            {
                "version": LEGACY_CONDITION_PREPROCESSING_VERSION,
                "source": "legacy lifting cache without shared_geometry",
            },
        )
    if shared.get("version") != SHARED_OBJECT_PREPROCESSING_VERSION:
        raise RuntimeError(
            "unsupported shared condition preprocessing version: "
            f"{shared.get('version')!r}"
        )
    resolution = int(shared["resolution"])
    pipeline_resolution = int(getattr(pipeline, "default_image_resolution", 0))
    if pipeline_resolution != resolution:
        raise RuntimeError(
            "pipeline/shared condition resolution differs: "
            f"{pipeline_resolution}/{resolution}"
        )
    prepared = prepare_shared_object_views(
        sample["image_paths"],
        sample["mask_paths"],
        resolution=resolution,
        foreground_margin=float(shared["foreground_margin"]),
        alpha_threshold=float(shared["alpha_threshold"]),
    )
    if prepared.contract != shared:
        raise RuntimeError("replayed shared condition contract differs from cache")
    geometry = prepared.geometry_record()
    expected_geometry_hash = str(preprocessing.get("shared_geometry_hash", ""))
    if not expected_geometry_hash or geometry["geometry_hash"] != expected_geometry_hash:
        raise RuntimeError(
            "replayed shared condition geometry hash differs from lifting cache"
        )
    return (
        prepared.images,
        {
            "version": SHARED_OBJECT_PREPROCESSING_VERSION,
            "geometry_hash": geometry["geometry_hash"],
            "contract_hash": geometry["contract_hash"],
            "resolution": resolution,
        },
    )


def expected_condition_preprocessing_version(sample: dict[str, Any]) -> str:
    if isinstance(sample.get("slat_condition"), dict):
        return PRECOMPUTED_NATIVE_V2_CONDITION_VERSION
    shared = dict(
        sample.get("preprocessing", {}).get("shared_geometry", {})
    )
    return (
        str(shared.get("version"))
        if shared
        else LEGACY_CONDITION_PREPROCESSING_VERSION
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


def tensor_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(tensor_sha256(item).encode("ascii"))
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def validate_precomputed_slat_condition(
    sample: dict[str, Any],
    condition: dict[str, Any],
    *,
    manifest_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate old model-input or new Direct-DINO condition provenance."""

    provenance = dict(sample.get("slat_condition_provenance", {}))
    tree_hash = str(provenance.get("condition_tree_sha256", ""))
    runtime_hash = str(sample.get("runtime_condition_sha256", ""))
    if not tree_hash or tensor_tree_sha256(condition) != tree_hash:
        raise RuntimeError("precomputed Native-SLAT condition tree changed")

    legacy_required = {
        "model_input",
        "model_input_sha256",
        "condition_sha256",
        "condition_tree_sha256",
    }
    if legacy_required.issubset(provenance):
        if str(provenance["condition_sha256"]) != runtime_hash:
            raise RuntimeError("precomputed Native-SLAT condition identity differs")
        return {
            "version": PRECOMPUTED_NATIVE_V2_CONDITION_VERSION,
            "source": "frozen prepare_omni_real_model_inputs.v2 artifact",
            "condition_sha256": str(provenance["condition_sha256"]),
            "condition_tree_sha256": tree_hash,
            "model_input": str(provenance["model_input"]),
            "model_input_sha256": str(provenance["model_input_sha256"]),
        }

    direct = dict(sample.get("dino_only_direct_build", {}))
    context = dict(sample.get("dino_only_context_contract", {}))
    binding_hash = str(provenance.get("sample_input_binding_sha256", ""))
    row_binding_hash = str((manifest_row or {}).get("source_input_binding_sha256", ""))
    direct_flags = (
        provenance.get("vggt_model_loaded"),
        provenance.get("vggt_model_executed"),
        direct.get("vggt_model_loaded"),
        direct.get("vggt_model_executed"),
        context.get("vggt_model_executed"),
    )
    if (
        provenance.get("source") != DIRECT_DINO_ONLY_PROVENANCE_VERSION
        or direct.get("version") != DIRECT_DINO_ONLY_PROVENANCE_VERSION
        or not binding_hash
        or binding_hash != str(direct.get("sample_input_binding_sha256", ""))
        or binding_hash != row_binding_hash
        or any(flag is not False for flag in direct_flags)
    ):
        raise RuntimeError("precomputed Direct-DINO condition provenance differs")
    if runtime_hash != tree_hash:
        raise RuntimeError("precomputed Direct-DINO condition identity differs")
    return {
        "version": PRECOMPUTED_NATIVE_V2_CONDITION_VERSION,
        "source": "precomputed direct DINO-only lifting artifact",
        "condition_sha256": runtime_hash,
        "condition_tree_sha256": tree_hash,
        "sample_input_binding_sha256": binding_hash,
        "vggt_model_executed": False,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def to_cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(child) for child in value)
    return value


def parse_int_csv(value: str) -> list[int]:
    result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("SS seeds must be non-empty and unique")
    return result


def index_lh_slats(root: str | Path) -> dict[str, Path]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"ground-truth SLAT root does not exist: {root_path}; expected "
            "ProObjaverse-300K/lh-slats/shard-*/*.npz"
        )
    output: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted(root_path.rglob("*.npz")):
        uid = path.stem
        if uid in output:
            duplicates.setdefault(uid, [str(output[uid])]).append(str(path))
        else:
            output[uid] = path.resolve()
    if duplicates:
        raise RuntimeError(f"duplicate lh-slat object IDs: {dict(list(duplicates.items())[:8])}")
    if not output:
        raise RuntimeError(f"no .npz targets found below {root_path}")
    return output


def sparse_frame_audit(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_set = {tuple(map(int, row[-3:])) for row in np.asarray(left)}
    right_set = {tuple(map(int, row[-3:])) for row in np.asarray(right)}
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)

    def bounds(values: set[tuple[int, int, int]]) -> dict[str, list[int]]:
        if not values:
            return {"min": [], "max": []}
        array = np.asarray(sorted(values), dtype=np.int64)
        return {"min": array.min(0).tolist(), "max": array.max(0).tolist()}

    return {
        "local_ss_count": len(left_set),
        "target_slat_count": len(right_set),
        "intersection": intersection,
        "union": union,
        "iou": float(intersection / union) if union else 1.0,
        "local_ss_bounds": bounds(left_set),
        "target_slat_bounds": bounds(right_set),
        "interpretation": "frame audit only; target support is not cropped",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifting_manifest", required=True)
    parser.add_argument("--slat_root", required=True)
    parser.add_argument("--flow_checkpoint", required=True)
    parser.add_argument("--correspondence_checkpoint", default="")
    parser.add_argument("--n3_report", default="")
    parser.add_argument(
        "--condition_arch",
        choices=("legacy", "native_every_block_v1", "native_ss_genrecon_v2"),
        default="legacy",
        help=(
            "legacy preserves Direct-SS v3 support generation; "
            "native_every_block_v1 rebuilds support with the historical adapter-only "
            "Native SS checkpoint; native_ss_genrecon_v2 binds the audited Native SS "
            "EMA/standard-CFG deployment report"
        ),
    )
    parser.add_argument(
        "--native_ss_report",
        default="",
        help="required V4 Native SS deployment report for native_ss_genrecon_v2",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--ss_seeds", default="42,43,44")
    parser.add_argument("--ss_steps", type=int, default=30)
    parser.add_argument("--cfg_strength", type=float, default=7.5)
    parser.add_argument("--guidance_rescale", type=float, default=0.5)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--physical_scale", type=float, default=None)
    parser.add_argument("--expected_ss_step", type=int, default=900)
    parser.add_argument("--min_frame_iou", type=float, default=0.90)
    parser.add_argument("--condition_replay_max_abs", type=float, default=1.0e-3)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require_all_objects", action="store_true")
    parser.add_argument(
        "--target_only",
        action="store_true",
        help="Only index/copy true SLAT targets and report coverage; no GPU work.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seeds = parse_int_csv(args.ss_seeds)
    if int(args.ss_steps) <= 0:
        raise ValueError("ss_steps must be positive")
    device = torch.device(args.device)
    if not args.target_only and device.type != "cuda":
        raise ValueError("full direct SLAT cache materialization requires CUDA")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PoseLiftingCacheDataset(args.lifting_manifest, indices=args.indices)
    ss_report_payload = None
    ss_report_binding = None
    if args.condition_arch == "native_ss_genrecon_v2":
        if not args.native_ss_report:
            raise ValueError(
                "native_ss_genrecon_v2 requires --native_ss_report; CLI sampling "
                "overrides are not accepted as the Native SS deployment identity"
            )
        ss_report_payload, ss_report_binding = load_ss_evidence(
            args.native_ss_report
        )
        if Path(args.flow_checkpoint).resolve() != Path(
            ss_report_binding["checkpoint"]
        ).resolve():
            raise RuntimeError(
                "--flow_checkpoint differs from the checkpoint frozen by --native_ss_report"
            )
    checkpoint = torch.load(args.flow_checkpoint, map_location="cpu")
    expected_checkpoint_format = {
        "legacy": DIRECT_FLOW_VERSION,
        "native_every_block_v1": NATIVE_SS_FLOW_VERSION,
        "native_ss_genrecon_v2": NATIVE_SS_GENRECON_VERSION,
    }[args.condition_arch]
    if checkpoint.get("format") != expected_checkpoint_format:
        raise ValueError(
            f"unexpected frozen SS checkpoint format={checkpoint.get('format')!r}"
        )
    expected_ss_step = (
        int(ss_report_binding["checkpoint_step"])
        if ss_report_binding is not None
        else int(args.expected_ss_step)
    )
    if int(checkpoint.get("step", -1)) != expected_ss_step:
        raise RuntimeError(
            f"frozen SS checkpoint step={checkpoint.get('step')} != "
            f"expected={expected_ss_step}"
        )
    saved_args = checkpoint["args"]
    if args.condition_arch == "legacy":
        if not args.correspondence_checkpoint or not args.n3_report:
            raise ValueError(
                "legacy support generation requires correspondence_checkpoint and n3_report"
            )
        physical_scale = (
            float(saved_args.get("physical_scale", 1.0))
            if args.physical_scale is None
            else float(args.physical_scale)
        )
    elif args.condition_arch == "native_every_block_v1":
        physical_scale = (
            float(saved_args.get("condition_scale", 1.0))
            if args.physical_scale is None
            else float(args.physical_scale)
        )
    else:
        if args.physical_scale is not None:
            raise ValueError(
                "native_ss_genrecon_v2 has learned_projection_only semantics; "
                "--physical_scale is forbidden"
            )
        validate_native_ss_genrecon_checkpoint(
            checkpoint, pretrained=args.pretrained
        )
        physical_scale = None
    if not 0.0 <= float(args.min_frame_iou) <= 1.0:
        raise ValueError("min_frame_iou must lie in [0,1]")
    slat_root = Path(args.slat_root).resolve()
    local_lh_binding_path = slat_root / "run_config.json"
    if local_lh_binding_path.is_file():
        local_lh_binding = json.loads(
            local_lh_binding_path.read_text(encoding="utf-8")
        )
        local_format = local_lh_binding.get("config", {}).get("format")
        if local_format != "pose_point_depth_mv.local_lh_slats.v2":
            raise RuntimeError(
                f"unsupported local lh-slats provenance format={local_format!r}"
            )
        target_source: dict[str, Any] = {
            "kind": "audited local limited-view TRELLIS SLatEncoder rebuild",
            "run_config": str(local_lh_binding_path),
            "run_config_sha256": sha256_file(local_lh_binding_path),
            "config_hash": str(local_lh_binding.get("config_hash", "")),
            "source_kind": str(
                local_lh_binding.get("config", {}).get("source_kind", "")
            ),
        }
    else:
        target_source = {
            "kind": "external ProObjaverse lh-slats",
            "run_config": "",
            "run_config_sha256": "",
        }
    if ss_report_binding is not None:
        resolved_ss_steps = int(ss_report_binding["steps"])
        resolved_cfg_strength = float(ss_report_binding["cfg_strength"])
        resolved_guidance_rescale = float(ss_report_binding["guidance_rescale"])
        resolved_rescale_t = float(ss_report_binding["rescale_t"])
        resolved_cfg_interval = [
            float(value) for value in ss_report_binding["cfg_interval"]
        ]
    else:
        resolved_ss_steps = int(args.ss_steps)
        resolved_cfg_strength = float(args.cfg_strength)
        resolved_guidance_rescale = float(args.guidance_rescale)
        resolved_rescale_t = float(args.rescale_t)
        resolved_cfg_interval = None
    config = {
        "pretrained": args.pretrained,
        "source_lifting_manifest": str(Path(args.lifting_manifest).resolve()),
        "source_lifting_manifest_sha256": sha256_file(args.lifting_manifest),
        "slat_root": str(slat_root),
        "ss_flow_checkpoint": str(Path(args.flow_checkpoint).resolve()),
        "ss_flow_checkpoint_sha256": sha256_file(args.flow_checkpoint),
        "expected_ss_step": int(expected_ss_step),
        "correspondence_checkpoint": (
            str(Path(args.correspondence_checkpoint).resolve())
            if args.correspondence_checkpoint
            else ""
        ),
        "correspondence_checkpoint_sha256": (
            sha256_file(args.correspondence_checkpoint)
            if args.correspondence_checkpoint
            else ""
        ),
        "n3_report": (
            str(Path(args.n3_report).resolve()) if args.n3_report else ""
        ),
        "n3_report_sha256": (
            sha256_file(args.n3_report) if args.n3_report else ""
        ),
        "ss_seeds": seeds,
        "ss_steps": resolved_ss_steps,
        "cfg_strength": resolved_cfg_strength,
        "guidance_rescale": resolved_guidance_rescale,
        "rescale_t": resolved_rescale_t,
        "physical_scale": physical_scale,
        "condition_replay_max_abs": float(args.condition_replay_max_abs),
        "min_frame_iou": float(args.min_frame_iou),
        "amp_dtype": args.amp_dtype,
        "mapping_version": (
            SLAT_SUPPORT_MAPPING_VERSION
            if args.condition_arch == "legacy"
            else (
                "direct_image_active32_every_block.v1"
                if args.condition_arch == "native_every_block_v1"
                else "native_ss_genrecon_active32_every_block.v2"
            )
        ),
        "target_source": target_source,
    }
    if ss_report_binding is not None:
        config["cfg_interval"] = resolved_cfg_interval
        config["native_ss_deployment"] = dict(ss_report_binding)
    if args.condition_arch != "legacy":
        config["condition_arch"] = str(args.condition_arch)
    config_hash = canonical_json_sha256(config)
    run_binding = {
        "format": DIRECT_SLAT_CACHE_VERSION,
        "config": config,
        "config_hash": config_hash,
    }
    binding_path = output_dir / "run_config.json"
    if binding_path.is_file():
        existing_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing_binding != run_binding:
            raise RuntimeError(
                "resume cache arguments/source bindings differ from run_config.json"
            )
    else:
        atomic_json(binding_path, run_binding)
    object_uids = sorted(
        {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
    )
    target_index = index_lh_slats(args.slat_root)
    matched = [uid for uid in object_uids if uid in target_index]
    missing = [uid for uid in object_uids if uid not in target_index]
    coverage = {
        "lifting_manifest": str(Path(args.lifting_manifest).resolve()),
        "slat_root": str(Path(args.slat_root).resolve()),
        "candidate_object_count": len(object_uids),
        "matched_object_count": len(matched),
        "missing_object_count": len(missing),
        "coverage": float(len(matched) / len(object_uids)),
        "missing_objects": missing,
        "passed": not missing,
        "hard_guard": (
            "targets must contain external ground-truth feats[N,8]/coords[N,3]; "
            "stock SLAT samples are forbidden as supervision"
        ),
    }
    atomic_json(output_dir / "target_coverage.json", coverage)
    if missing and args.require_all_objects:
        raise RuntimeError(
            f"ground-truth SLAT coverage is incomplete: {len(missing)}/{len(object_uids)} "
            f"objects missing; see {output_dir / 'target_coverage.json'}"
        )
    if not matched:
        raise RuntimeError("none of the lifting-cache objects has a true SLAT target")

    selected_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if str(row.get("object_uid", row["uid"])) in target_index
    ]
    selected_rows = [dataset.rows[index] for index in selected_indices]
    object_target_rows: dict[str, dict[str, Any]] = {}
    first_index_by_object: dict[str, int] = {}
    for dataset_index in selected_indices:
        row = dataset.rows[dataset_index]
        object_uid = str(row.get("object_uid", row["uid"]))
        first_index_by_object.setdefault(object_uid, dataset_index)
    for position, object_uid in enumerate(matched, start=1):
        source = target_index[object_uid]
        with np.load(source) as payload:
            if "coords" not in payload or "feats" not in payload:
                raise ValueError(f"{source} lacks coords/feats")
            coords = np.asarray(payload["coords"], dtype=np.int32)
            feats = np.asarray(payload["feats"], dtype=np.float32)
        coords4 = torch.cat(
            [torch.zeros((len(coords), 1), dtype=torch.int32), torch.from_numpy(coords)],
            dim=1,
        )
        validate_sparse_target_alignment(
            coords4, torch.from_numpy(feats), require_single_batch=True
        )
        sample = dataset[first_index_by_object[object_uid]]
        ss_latent_path = Path(sample["ss_latent"]).resolve()
        with np.load(ss_latent_path) as ss_payload:
            source_glb = str(Path(str(ss_payload["source_glb"])).resolve())
        frame_audit = sparse_frame_audit(sample["target_coords"].numpy(), coords)
        frame_audit["minimum_iou"] = float(args.min_frame_iou)
        frame_audit["passed"] = frame_audit["iou"] >= float(args.min_frame_iou)
        if not frame_audit["passed"]:
            raise RuntimeError(
                f"SLAT/SS coordinate-frame audit failed object={object_uid}: "
                f"IoU={frame_audit['iou']:.6f} < {args.min_frame_iou}; "
                "do not train until the encoder/canonical frame is reconciled"
            )
        target_path = output_dir / "targets" / object_uid[:2] / f"{object_uid}.npz"
        if target_path.is_file():
            with np.load(target_path) as existing:
                same = (
                    np.array_equal(np.asarray(existing["coords"]), coords)
                    and np.array_equal(np.asarray(existing["feats"]), feats)
                    and np.array_equal(
                        np.asarray(existing["local_ss_coords"]),
                        sample["target_coords"].numpy().astype(np.int32),
                    )
                )
            if not same:
                raise RuntimeError(f"resumed target artifact differs from source: {target_path}")
        else:
            atomic_savez(
                target_path,
                coords=coords,
                feats=feats,
                local_ss_coords=sample["target_coords"].numpy().astype(np.int32),
            )
        object_target_rows[object_uid] = {
            "object_uid": object_uid,
            "target_file": str(target_path.relative_to(output_dir)),
            "target_file_sha256": sha256_file(target_path),
            "source_lh_slat": str(source),
            "source_lh_slat_sha256": sha256_file(source),
            "source_glb": source_glb,
            "source_glb_sha256": sha256_file(source_glb),
            "ss_latent": str(ss_latent_path),
            "ss_latent_sha256": sha256_file(ss_latent_path),
            "target_point_count": int(len(coords)),
            "frame_audit": frame_audit,
        }
        del sample
        print(f"[direct_slat_cache:target] {position}/{len(matched)} {object_uid}", flush=True)

    if args.target_only:
        manifest = {
            "format": DIRECT_SLAT_CACHE_VERSION,
            "materialized": False,
            "output_dir": str(output_dir),
            "source_lifting_manifest": str(Path(args.lifting_manifest).resolve()),
            "source_lifting_manifest_sha256": sha256_file(args.lifting_manifest),
            "slat_root": str(Path(args.slat_root).resolve()),
            "config": config,
            "config_hash": config_hash,
            "sample_count": len(selected_rows),
            "object_count": len(matched),
            "objects": [object_target_rows[uid] for uid in matched],
            "samples": [],
            "target_coverage": coverage,
        }
        atomic_json(output_dir / "manifest.json", manifest)
        print(json.dumps({"target_only": True, **coverage}, indent=2), flush=True)
        return

    torch.cuda.set_device(0 if device.index is None else int(device.index))
    if args.condition_arch == "legacy":
        n3_audit = validate_n3_checkpoint(
            args.n3_report, args.correspondence_checkpoint
        )
        correspondence_head, _, correspondence_runtime = (
            load_frozen_correspondence_head(
                args.correspondence_checkpoint,
                device=device,
                visual_channels=dataset.visual_feature_dim,
            )
        )
        if correspondence_runtime["checkpoint_sha256"] != n3_audit["checkpoint_sha256"]:
            raise RuntimeError("runtime correspondence checkpoint differs from N3")
        built = build_direct_components(
            pretrained=args.pretrained,
            visual_channels=dataset.visual_feature_dim,
            physical_hidden_dim=int(saved_args["physical_hidden_dim"]),
            lora_rank=int(saved_args["lora_rank"]),
            lora_alpha=int(saved_args["lora_alpha"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
            retain_pipeline=True,
        )
        sampler, ss_model, decoder, _, sampler_defaults, pipeline = built
        load_partial_state(
            ss_model, checkpoint["model_trainable_state"], require_all_trainable=True
        )
    elif args.condition_arch == "native_every_block_v1":
        correspondence_head = None
        correspondence_runtime = {
            "kind": "not_used_by_native_every_block_v1",
            "checkpoint_sha256": "",
        }
        n3_audit = {
            "kind": "not_used_by_native_every_block_v1",
            "checkpoint_sha256": "",
        }
        built = build_native_ss_components(
            pretrained=args.pretrained,
            hidden_dim=int(saved_args["hidden_dim"]),
            feature_source=str(saved_args["feature_source"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
            retain_pipeline=True,
        )
        sampler, ss_model, decoder, _, sampler_defaults, pipeline = built
        load_native_trainable_state(
            ss_model, checkpoint["model_trainable_state"]
        )
    else:
        correspondence_head = None
        correspondence_runtime = {
            "kind": "not_used_by_native_ss_genrecon_v2",
            "checkpoint_sha256": "",
        }
        n3_audit = {
            "kind": "not_used_by_native_ss_genrecon_v2",
            "checkpoint_sha256": "",
        }
        if ss_report_binding is None:
            raise RuntimeError("Native SS deployment binding disappeared")
        built = build_native_ss_genrecon_components(
            pretrained=args.pretrained,
            lora_rank=int(saved_args["lora_rank"]),
            lora_alpha=int(saved_args["lora_alpha"]),
            condition_channels=int(saved_args["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
        sampler, ss_model, decoder, _, sampler_defaults = built
        load_native_ss_genrecon_state(
            ss_model, checkpoint["ema_trainable_state"]
        )
        # SLAT-condition construction later still needs the original pipeline.
        from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
        from trellis.pipelines import TrellisVGGTTo3DPipeline

        install_unused_model_stubs()
        pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
        pipeline._device = device
        pipeline.low_vram = False
    ss_model.eval()
    for parameter in ss_model.parameters():
        parameter.requires_grad_(False)
    if correspondence_head is not None:
        correspondence_head.eval()
    if ss_report_binding is not None:
        ss_params = native_ss_sampling_params(
            dict(sampler_defaults),
            make_sampling_namespace(ss_report_binding),
            float(ss_report_binding["cfg_strength"]),
        )
    else:
        ss_params = dict(sampler_defaults)
        ss_params.update(
            {
                "steps": int(args.ss_steps),
                "cfg_strength": float(args.cfg_strength),
                "guidance_rescale": float(args.guidance_rescale),
                "rescale_t": float(args.rescale_t),
            }
        )
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    sample_artifacts: dict[tuple[str, int], dict[str, Any]] = {}
    physical_artifacts: dict[str, dict[str, Any]] = {}
    for output_position, dataset_index in enumerate(selected_indices, start=1):
        sample = dataset[dataset_index]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        evidence = (
            make_direct_evidence_bundle(
                sample,
                modes=(),
                device=device,
                correspondence_head=correspondence_head,
                correspondence_runtime=correspondence_runtime,
            )["correct"]
            if args.condition_arch == "legacy"
            else None
        )
        physical_path = output_dir / "physical" / uid[:2] / f"{uid}.pt"
        if physical_path.is_file():
            physical_payload = torch.load(physical_path, map_location="cpu")
            if (
                physical_payload.get("format") != DIRECT_SLAT_CACHE_VERSION
                or str(physical_payload.get("uid")) != uid
                or str(physical_payload.get("config_hash")) != config_hash
            ):
                raise RuntimeError(f"invalid resumed physical artifact: {physical_path}")
        else:
            if args.condition_arch == "legacy":
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    physical_tokens, physical_stats = ss_model.physical_encoder(
                        *evidence[:4]
                    )
            else:
                # Direct 32^3 image lifting in native SLAT does not consume this
                # legacy field. Store only a marker; the native-aware loader
                # supplies a transient compatibility zero tensor on demand.
                physical_tokens = None
                zero = torch.zeros((), device=device, dtype=torch.float32)
                physical_stats = {
                    "unused_native_placeholder": zero,
                    "physical_token_rms": zero,
                }
            physical_payload = {
                "format": DIRECT_SLAT_CACHE_VERSION,
                "uid": uid,
                "object_uid": object_uid,
                "config_hash": config_hash,
                "physical_tokens16": (
                    physical_tokens.to(torch.float16).cpu()
                    if physical_tokens is not None
                    else None
                ),
                "unused_native_placeholder": args.condition_arch
                in {"native_every_block_v1", "native_ss_genrecon_v2"},
                "stats": {
                    key: float(value.detach().float().cpu().item())
                    for key, value in physical_stats.items()
                },
            }
            atomic_torch_save(physical_payload, physical_path)
        physical_artifacts[uid] = {
            "physical_file": str(physical_path.relative_to(output_dir)),
            "physical_file_sha256": sha256_file(physical_path),
            "physical_tokens_sha256": (
                tensor_sha256(physical_payload["physical_tokens16"])
                if torch.is_tensor(physical_payload.get("physical_tokens16"))
                else canonical_json_sha256(
                    {"unused_native_placeholder": True}
                )
            ),
        }
        condition = sample["stock_condition"].to(device=device)
        negative_condition = torch.zeros_like(condition)
        for seed in seeds:
            support_path = (
                output_dir
                / "support"
                / uid[:2]
                / uid
                / f"seed_{int(seed)}.pt"
            )
            if support_path.is_file():
                support_payload = torch.load(support_path, map_location="cpu")
                if (
                    support_payload.get("format") != DIRECT_SLAT_CACHE_VERSION
                    or str(support_payload.get("uid")) != uid
                    or int(support_payload.get("seed", -1)) != int(seed)
                    or str(support_payload.get("config_hash")) != config_hash
                ):
                    raise RuntimeError(f"invalid resumed support artifact: {support_path}")
            else:
                generator = torch.Generator(device=device).manual_seed(
                    int(seed) * 1000003 + int(dataset_index) * 1009
                )
                noise = torch.randn(
                    (1, 8, 16, 16, 16), generator=generator, device=device
                )
                if args.condition_arch == "legacy":
                    wrapper = PositivePhysicalRolloutFlow(
                        ss_model,
                        condition,
                        evidence[:4],
                        physical_scale=physical_scale,
                    )
                elif args.condition_arch == "native_every_block_v1":
                    wrapper = PositiveNativeSSRolloutFlow(
                        ss_model,
                        condition,
                        sample,
                        condition_scale=physical_scale,
                    )
                else:
                    wrapper = NativeSSCalibratedCFGFlow(
                        ss_model,
                        condition,
                        sample,
                        enabled=True,
                        projection_mode="correct",
                    )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    corrected_ss = sampler.sample(
                        wrapper,
                        noise,
                        cond=condition,
                        neg_cond=negative_condition,
                        **ss_params,
                        verbose=False,
                    ).samples
                    decoder_dtype = next(decoder.parameters()).dtype
                    occupancy_logits = decoder(
                        corrected_ss.to(dtype=decoder_dtype)
                    ).float()
                if wrapper.positive_calls <= 0:
                    raise RuntimeError(
                        "corrected SS rollout never used its positive condition path"
                    )
                if (
                    args.condition_arch == "native_ss_genrecon_v2"
                    and float(ss_params.get("cfg_strength", 1.0)) != 1.0
                    and wrapper.negative_calls <= 0
                ):
                    raise RuntimeError(
                        "Native SS standard CFG never used its adapted unconditional path"
                    )
                corrected_coords = torch.argwhere(occupancy_logits > 0)[:, [0, 2, 3, 4]]
                support_payload = {
                    "format": DIRECT_SLAT_CACHE_VERSION,
                    "uid": uid,
                    "object_uid": object_uid,
                    "seed": int(seed),
                    "config_hash": config_hash,
                    "noise_seed_formula": "seed*1000003 + selected_dataset_index*1009",
                    "corrected_ss": corrected_ss.to(torch.float16).cpu(),
                    "occupancy_logits64": occupancy_logits.to(torch.float16).cpu(),
                    "corrected_coords64": corrected_coords.to(torch.int32).cpu(),
                    "positive_cfg_calls": int(wrapper.positive_calls),
                    "negative_cfg_calls": int(wrapper.negative_calls),
                    "native_ss_deployment": (
                        dict(ss_report_binding)
                        if ss_report_binding is not None
                        else None
                    ),
                }
                atomic_torch_save(support_payload, support_path)
            sample_artifacts[(uid, int(seed))] = {
                "support_file": str(support_path.relative_to(output_dir)),
                "support_file_sha256": sha256_file(support_path),
                "corrected_coord_count": int(
                    support_payload["corrected_coords64"].shape[0]
                ),
            }
        del evidence, condition, negative_condition, physical_payload
        torch.cuda.empty_cache()
        print(
            f"[direct_slat_cache:ss] {output_position}/{len(selected_indices)} {uid}",
            flush=True,
        )

    ss_model.cpu()
    decoder.cpu()
    if correspondence_head is not None:
        correspondence_head.cpu()
    for name in ("sparse_structure_flow_model", "sparse_structure_decoder"):
        pipeline.models[name].cpu()
    del ss_model, decoder, correspondence_head
    gc.collect()
    torch.cuda.empty_cache()

    pipeline._device = device
    pipeline.low_vram = True
    condition_artifacts: dict[str, dict[str, Any]] = {}
    for output_position, dataset_index in enumerate(selected_indices, start=1):
        sample = dataset[dataset_index]
        uid = str(sample["uid"])
        condition_path = output_dir / "conditions" / uid[:2] / f"{uid}.pt"
        if condition_path.is_file():
            condition_payload = torch.load(condition_path, map_location="cpu")
            saved_preprocessing = dict(
                condition_payload.get("condition_preprocessing", {})
            )
            if (
                condition_payload.get("format") != DIRECT_SLAT_CACHE_VERSION
                or str(condition_payload.get("uid")) != uid
                or str(condition_payload.get("config_hash")) != config_hash
                or str(saved_preprocessing.get("version", ""))
                != expected_condition_preprocessing_version(sample)
            ):
                raise RuntimeError(f"invalid resumed condition artifact: {condition_path}")
        else:
            precomputed = sample.get("slat_condition")
            if isinstance(precomputed, dict):
                native_slat = to_cpu_tree(precomputed)
                views = len(native_slat.get("cond", []))
                if views <= 0 or views != len(native_slat.get("neg_cond", [])):
                    raise RuntimeError(
                        f"precomputed Native-SLAT view contract is invalid uid={uid}"
                    )
                replay_diff = 0.0
                try:
                    condition_preprocessing = validate_precomputed_slat_condition(
                        sample,
                        native_slat,
                        manifest_row=dataset.rows[dataset_index],
                    )
                except RuntimeError as error:
                    raise RuntimeError(f"{error} uid={uid}") from error
            else:
                images, condition_preprocessing = prepare_native_condition_images(
                    sample, pipeline
                )
                aggregated, image_tensor = pipeline.vggt_feat(images)
                raw_image_cond = pipeline.encode_image(image_tensor)
                batch = int(aggregated[0].shape[0])
                views = int(aggregated[0].shape[1])
                image_cond = normalize_image_cond(
                    raw_image_cond, batch=batch, views=views
                )
                recomputed_ss = pipeline.get_ss_cond(
                    image_cond[:, :, 5:], aggregated, num_samples=1
                )["cond"]
                cached_ss = sample["stock_condition"].to(device=device)
                replay_diff = float(
                    (recomputed_ss.float() - cached_ss.float()).abs().max().item()
                )
                if replay_diff > float(args.condition_replay_max_abs):
                    raise RuntimeError(
                        f"native condition replay mismatch uid={uid}: {replay_diff} > "
                        f"{args.condition_replay_max_abs}"
                    )
                native_slat = pipeline.get_slat_cond(
                    image_cond, aggregated, num_samples=1
                )
            condition_payload = {
                "format": DIRECT_SLAT_CACHE_VERSION,
                "uid": uid,
                "object_uid": str(sample.get("object_uid", uid)),
                "config_hash": config_hash,
                "views": views,
                "condition": to_cpu_tree(native_slat),
                "cached_vs_recomputed_ss_max_abs": replay_diff,
                "condition_preprocessing": condition_preprocessing,
            }
            atomic_torch_save(condition_payload, condition_path)
            if not isinstance(precomputed, dict):
                del images, aggregated, image_tensor, raw_image_cond, image_cond
                del recomputed_ss, cached_ss
            del native_slat
        condition_artifacts[uid] = {
            "condition_file": str(condition_path.relative_to(output_dir)),
            "condition_file_sha256": sha256_file(condition_path),
            "condition_tree_sha256": tensor_tree_sha256(
                condition_payload["condition"]
            ),
            "views": int(condition_payload["views"]),
            "cached_vs_recomputed_ss_max_abs": float(
                condition_payload["cached_vs_recomputed_ss_max_abs"]
            ),
            "condition_preprocessing": dict(
                condition_payload.get("condition_preprocessing", {})
            ),
        }
        torch.cuda.empty_cache()
        print(
            f"[direct_slat_cache:cond] {output_position}/{len(selected_indices)} {uid}",
            flush=True,
        )

    normalization = {
        key: [float(item) for item in value]
        for key, value in pipeline.slat_normalization.items()
    }
    normalization_hash = canonical_json_sha256(normalization)
    manifest_samples = []
    for dataset_index in selected_indices:
        row = dataset.rows[dataset_index]
        uid = str(row["uid"])
        object_uid = str(row.get("object_uid", uid))
        source_glb = str(object_target_rows[object_uid]["source_glb"])
        for seed in seeds:
            manifest_samples.append(
                {
                    "uid": uid,
                    "object_uid": object_uid,
                    "support_seed": int(seed),
                    "view_count": int(row.get("view_count", condition_artifacts[uid]["views"])),
                    "source_glb": source_glb,
                    "source_glb_sha256": str(
                        object_target_rows[object_uid]["source_glb_sha256"]
                    ),
                    "ss_latent": str(object_target_rows[object_uid]["ss_latent"]),
                    "ss_latent_sha256": str(
                        object_target_rows[object_uid]["ss_latent_sha256"]
                    ),
                    **{
                        key: value
                        for key, value in object_target_rows[object_uid].items()
                        if key
                        in {
                            "target_file",
                            "target_file_sha256",
                            "source_lh_slat",
                            "source_lh_slat_sha256",
                            "target_point_count",
                        }
                    },
                    **physical_artifacts[uid],
                    **sample_artifacts[(uid, int(seed))],
                    **condition_artifacts[uid],
                }
            )
    manifest = {
        "format": DIRECT_SLAT_CACHE_VERSION,
        "materialized": True,
        "output_dir": str(output_dir),
        "source_lifting_manifest": str(Path(args.lifting_manifest).resolve()),
        "source_lifting_manifest_sha256": sha256_file(args.lifting_manifest),
        "slat_root": str(Path(args.slat_root).resolve()),
        "config": config,
        "config_hash": config_hash,
        "slat_normalization": normalization,
        "slat_normalization_hash": normalization_hash,
        "sample_count": len(manifest_samples),
        "sequence_count": len(selected_indices),
        "object_count": len(matched),
        "uid_hash": canonical_json_sha256(
            [f"{row['uid']}@{row['support_seed']}" for row in manifest_samples]
        ),
        "object_uid_hash": canonical_json_sha256(sorted(matched)),
        "objects": [object_target_rows[uid] for uid in matched],
        "samples": manifest_samples,
        "target_coverage": coverage,
        "frozen_ss": {
            "checkpoint": str(Path(args.flow_checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.flow_checkpoint),
            "checkpoint_step": int(checkpoint.get("step", -1)),
            "physical_scale": physical_scale,
            "correspondence": correspondence_runtime,
            "n3_audit": n3_audit,
        },
    }
    if args.condition_arch != "legacy":
        manifest["frozen_ss"]["condition_arch"] = str(args.condition_arch)
    if ss_report_binding is not None:
        manifest["frozen_ss"]["native_ss_deployment"] = dict(
            ss_report_binding
        )
        manifest["frozen_ss"]["weights"] = "ema"
    atomic_json(output_dir / "manifest.json", manifest)
    audit = {
        "passed": True,
        "format": manifest["format"],
        "materialized": True,
        "sample_count": manifest["sample_count"],
        "sequence_count": manifest["sequence_count"],
        "object_count": manifest["object_count"],
        "all_true_targets": True,
        "target_coverage": coverage,
        "condition_replay_max_abs": max(
            row["cached_vs_recomputed_ss_max_abs"]
            for row in condition_artifacts.values()
        ),
        "condition_preprocessing_versions": sorted(
            {
                str(row["condition_preprocessing"].get("version", ""))
                for row in condition_artifacts.values()
            }
        ),
        "stock_slat_used_as_target": False,
    }
    atomic_json(output_dir / "cache_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
