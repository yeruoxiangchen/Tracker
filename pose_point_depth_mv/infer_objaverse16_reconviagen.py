#!/usr/bin/env python3
"""Run stock ReconViaGen on the frozen Objaverse16 RGB/mask views."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import torch


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for dependency in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from pose_point_depth_mv.bunny_review.common import (  # noqa: E402
    binding,
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (  # noqa: E402
    mesh_structure_metrics,
)
from pose_point_depth_mv.omni_real_benchmark_common import (  # noqa: E402
    atomic_json,
    load_json,
    resolve_torch_device,
)
from pose_point_depth_mv.training_overlap_objaverse import (  # noqa: E402
    expected_view_count,
    validate_selection,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse16_reconviagen_inference.v1"
MANIFEST_FORMAT = (
    "pose_point_depth_mv.objaverse16_reconviagen_inference_manifest.v1"
)


def parse_csv_int(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return values


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def export_mesh_atomic(mesh: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    mesh.export(temporary, file_type="obj")
    os.replace(temporary, path)


def _tensor_int_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]


def _load_visible_inputs(
    *,
    lifting_path: Path,
    lifting_row: dict[str, Any],
    selected_row: dict[str, Any],
) -> dict[str, Any]:
    cache_path = resolve_from(lifting_path.parent, lifting_row["cache_file"])
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    uid = str(selected_row["uid"])
    object_uid = str(selected_row["object_uid"])
    if (
        not isinstance(cache, dict)
        or str(cache.get("uid")) != uid
        or str(cache.get("object_uid")) != object_uid
    ):
        raise RuntimeError(f"lifting cache identity differs for uid={uid}")

    image_paths = [Path(value).resolve() for value in cache.get("image_paths", [])]
    mask_paths = [Path(value).resolve() for value in cache.get("mask_paths", [])]
    view_ids = _tensor_int_list(cache.get("view_ids", []))
    expected_views = expected_view_count(selected_row)
    if (
        len(image_paths) != expected_views
        or len(mask_paths) != expected_views
        or len(view_ids) != expected_views
        or len(set(view_ids)) != expected_views
    ):
        raise RuntimeError(f"visible input view counts differ for uid={uid}")
    for path in [*image_paths, *mask_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    embedded_visible = selected_row.get("visible_inputs")
    if isinstance(embedded_visible, dict):
        if [int(value) for value in embedded_visible.get("view_ids", [])] != view_ids:
            raise RuntimeError(f"lifting view IDs differ from overlap selection for uid={uid}")
        expected_images = [
            Path(value).resolve() for value in embedded_visible.get("image_paths", [])
        ]
        expected_masks = [
            Path(value).resolve() for value in embedded_visible.get("mask_paths", [])
        ]
        source_view_indices = [
            int(value) for value in embedded_visible.get("source_view_indices", view_ids)
        ]
    else:
        selected_frames = list(selected_row["frames"])
        if max(view_ids) >= len(selected_frames):
            raise RuntimeError(f"lifting view ID is outside frozen frames for uid={uid}")
        expected_images = [
            Path(selected_frames[index]["image"]).resolve() for index in view_ids
        ]
        expected_masks = [
            Path(selected_frames[index]["mask"]).resolve() for index in view_ids
        ]
        source_view_indices = [
            int(selected_frames[index]["source_view_index"]) for index in view_ids
        ]
    if image_paths != expected_images or mask_paths != expected_masks:
        raise RuntimeError(f"lifting visible inputs differ from frozen selection for uid={uid}")

    visible = {
        "uid": uid,
        "object_uid": object_uid,
        "source_group": str(selected_row["source_group"]),
        "view_count": expected_views,
        "view_ids": view_ids,
        "source_view_indices": source_view_indices,
        "source_cache": binding(cache_path),
        "input_images": [binding(path) for path in image_paths],
        "input_masks": [binding(path) for path in mask_paths],
        "image_paths": image_paths,
        "mask_paths": mask_paths,
    }
    visible["visible_input_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in visible.items()
            if key not in {"image_paths", "mask_paths", "visible_input_sha256"}
        }
    )
    return visible


def _result_paths(output_dir: Path, visible: dict[str, Any], seed: int) -> tuple[Path, Path]:
    root = output_dir / "objects" / str(visible["object_uid"]) / f"seed_{seed}"
    return root / "mesh_decoder_canonical.obj", root / "result.json"


def _expected_identity(
    *,
    visible: dict[str, Any],
    seed: int,
    selection_sha256: str,
    lifting_sha256: str,
    pretrained: str,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": REPORT_FORMAT,
        "method": "reconviagen_original",
        "uid": str(visible["uid"]),
        "object_uid": str(visible["object_uid"]),
        "seed": int(seed),
        "view_count": int(visible["view_count"]),
        "visible_input_sha256": str(visible["visible_input_sha256"]),
        "selection_manifest_sha256": selection_sha256,
        "source_lifting_manifest_sha256": lifting_sha256,
        "pretrained": str(pretrained),
        "sampling_sha256": canonical_sha256(sampling),
    }


def _reuse_result(
    result_path: Path,
    mesh_path: Path,
    *,
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    if not result_path.is_file() and not mesh_path.is_file():
        return None
    if not result_path.is_file() or not mesh_path.is_file():
        raise RuntimeError(f"partial ReconViaGen output: {result_path.parent}")
    result = load_json(result_path)
    mismatch = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if result.get("passed") is not True:
        mismatch["passed"] = (result.get("passed"), True)
    if result.get("mesh_sha256") != sha256_file(mesh_path):
        mismatch["mesh_sha256"] = (result.get("mesh_sha256"), sha256_file(mesh_path))
    if mismatch:
        raise RuntimeError(f"stale ReconViaGen result={mismatch}: {result_path}")
    return result


def _build_pipeline(pretrained: str, device: torch.device, low_vram: bool):
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = bool(low_vram)
    if not low_vram:
        pipeline.VGGT_model.to(device).eval()
        for module in pipeline.models.values():
            module.to(device).eval()
    return pipeline


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--source_lifting_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--seeds", type=parse_csv_int, default=[42])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--ss_steps", type=int, default=30)
    parser.add_argument("--ss_guidance", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    parser.add_argument("--ss_rescale_t", type=float, default=5.0)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_guidance", type=float, default=7.5)
    parser.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument(
        "--multiimage_algo",
        choices=("multidiffusion", "stochastic"),
        default="multidiffusion",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.num_workers) <= 0 or not 0 <= int(args.worker_index) < int(args.num_workers):
        raise ValueError("worker_index must be in [0, num_workers)")
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    lifting_path = Path(args.source_lifting_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    selection = load_json(selection_path)
    lifting = load_json(lifting_path)
    selection_contract = validate_selection(selection)

    all_selected = list(selection.get("samples", []))
    lifting_rows = {str(row["uid"]): row for row in lifting.get("samples", [])}
    selected_uids = {str(row["uid"]) for row in all_selected}
    if len(all_selected) != selection_contract.object_count or set(lifting_rows) != selected_uids:
        raise RuntimeError("selection and source lifting object sets differ")
    selected = [
        row
        for position, row in enumerate(all_selected)
        if position % int(args.num_workers) == int(args.worker_index)
    ]
    if not selected:
        raise RuntimeError("ReconViaGen worker shard contains no selected objects")

    selection_sha = sha256_file(selection_path)
    lifting_sha = sha256_file(lifting_path)
    sampling = {
        "sparse_structure": {
            "steps": int(args.ss_steps),
            "cfg_strength": float(args.ss_guidance),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        "slat": {
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_guidance),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.slat_guidance_rescale),
            "rescale_t": float(args.slat_rescale_t),
        },
        "multiimage_algo": str(args.multiimage_algo),
    }
    visible_inputs = [
        _load_visible_inputs(
            lifting_path=lifting_path,
            lifting_row=lifting_rows[str(row["uid"])],
            selected_row=row,
        )
        for row in selected
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any], int, Path, Path, dict[str, Any]]] = []
    for position, visible in enumerate(visible_inputs, start=1):
        for seed in args.seeds:
            mesh_path, result_path = _result_paths(output_dir, visible, seed)
            expected = _expected_identity(
                visible=visible,
                seed=seed,
                selection_sha256=selection_sha,
                lifting_sha256=lifting_sha,
                pretrained=args.pretrained,
                sampling=sampling,
            )
            reused = _reuse_result(result_path, mesh_path, expected=expected)
            if reused is not None:
                if not args.resume:
                    raise FileExistsError(result_path)
                records.append(reused)
            else:
                if mesh_path.parent.exists() and any(mesh_path.parent.iterdir()):
                    raise RuntimeError(f"partial ReconViaGen output: {mesh_path.parent}")
                pending.append((position, visible, seed, mesh_path, result_path, expected))

    device = resolve_torch_device(args.device)
    pipeline = None
    if pending:
        from reconvggt_ar_adapter_a.train_pointpose_ss_lora import rgba_images

        pipeline = _build_pipeline(args.pretrained, device, bool(args.low_vram))
        try:
            for position, visible, seed, mesh_path, result_path, expected in pending:
                images = rgba_images(
                    [str(path) for path in visible["image_paths"]],
                    [str(path) for path in visible["mask_paths"]],
                    pipeline,
                )
                print(
                    f"[objaverse_reconviagen] {position}/{len(selected)} "
                    f"uid={visible['uid']} views={len(images)} seed={seed}",
                    flush=True,
                )
                outputs, coords, ss_noise = pipeline.run(
                    image=images,
                    seed=int(seed),
                    formats=["mesh"],
                    preprocess_image=False,
                    sparse_structure_sampler_params=sampling["sparse_structure"],
                    slat_sampler_params=sampling["slat"],
                    mode=str(args.multiimage_algo),
                )
                decoded = outputs["mesh"][0]
                mesh = decoded.to_trimesh(transform_pose=False)
                structure = mesh_structure_metrics(mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(
                        f"ReconViaGen decoded empty Mesh: {visible['uid']}"
                    )
                mesh_path.parent.mkdir(parents=True, exist_ok=False)
                export_mesh_atomic(mesh, mesh_path)
                result = {
                    **expected,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_group": str(visible["source_group"]),
                    "view_ids": list(visible["view_ids"]),
                    "source_view_indices": list(visible["source_view_indices"]),
                    "source_cache": visible["source_cache"],
                    "input_images": visible["input_images"],
                    "input_masks": visible["input_masks"],
                    "sampling": sampling,
                    "mesh": str(mesh_path),
                    "mesh_sha256": sha256_file(mesh_path),
                    "structure": structure,
                    "coord_count": int(coords.shape[0]),
                    "ss_noise_shape": list(ss_noise.shape),
                    "output_frame": "latent decoder canonical; transform_pose=False",
                    "decoder_to_source_axis_transform_applied": False,
                    "preprocessing": (
                        "frozen GT mask as RGBA alpha, then official ReconViaGen "
                        "1.10 foreground crop and 518x518 resize"
                    ),
                    "explicit_camera_pose_consumed": False,
                    "point_cloud_tensor_consumed": False,
                    "target_or_metric_consumed": False,
                    "vggt_model_loaded": True,
                    "vggt_model_executed": True,
                    "passed": True,
                }
                atomic_json(result_path, result)
                records.append(result)
                del images, outputs, coords, ss_noise, decoded, mesh
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            del pipeline
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    records.sort(
        key=lambda row: (
            selection_contract.selected_uids.index(str(row["uid"])),
            int(row["seed"]),
        )
    )
    expected_records = len(selected) * len(args.seeds)
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": len(records) == expected_records,
        "formal": False,
        "protocol_scope": selection_contract.scope,
        "training_overlap": selection_contract.training_overlap,
        "source_scope": selection_contract.source_scope,
        "selection_object_count": selection_contract.object_count,
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "method": "reconviagen_original",
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha,
        "source_lifting_manifest": str(lifting_path),
        "source_lifting_manifest_sha256": lifting_sha,
        "pretrained": str(args.pretrained),
        "seeds": list(args.seeds),
        "sampling": sampling,
        "sampling_sha256": canonical_sha256(sampling),
        "object_count": len({str(row["object_uid"]) for row in records}),
        "record_count": len(records),
        "objects": records,
        "input_contract": (
            "same frozen O3 RGB/mask views as current method; official ReconViaGen "
            "image preprocessing; camera matrices and point tensors are not passed"
        ),
        "explicit_camera_pose_consumed": False,
        "point_cloud_tensor_consumed": False,
        "target_or_metric_consumed": False,
        "vggt_model_loaded": True,
        "vggt_model_executed": True,
        "training_object_disjoint": selection_contract.training_object_disjoint,
        "source_mesh_disjoint": selection_contract.source_mesh_disjoint,
    }
    manifest_path = output_dir / "inference_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "object_count": manifest["object_count"],
                "record_count": manifest["record_count"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
