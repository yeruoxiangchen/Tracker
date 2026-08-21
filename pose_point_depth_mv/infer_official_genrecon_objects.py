#!/usr/bin/env python3
"""Run the official GenRecon scene pipeline on frozen canonical object cases.

This is deliberately a cross-domain diagnostic: GenRecon is a scene-scale
method, while the frozen protocol contains centred single objects.  A single
unit chunk is used, all posed views form the official global 3D condition, and
the protocol's score-independent largest-mask view forms the 2D condition.
No target mesh or metric is read during inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


FORMAT = "pose_point_depth_mv.official_genrecon_object_inference.v1"
CAMERA_CONVERSION = "opencv_w2c=diag(1,-1,-1,1)@inverse(blender_c2w);identity_chunk0"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def binding(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = str(protocol.get("protocol_sha256", ""))
    body = dict(protocol)
    body.pop("protocol_sha256", None)
    if not expected or canonical_sha256(body) != expected:
        raise RuntimeError("frozen protocol canonical SHA-256 mismatch")
    comparison = protocol.get("comparison", {})
    if comparison.get("selection_mode") not in {
        "source_view_balanced_sha256_v1",
        "t3_exact_objects_matched_seed_v1",
    }:
        raise RuntimeError("protocol is not a supported frozen Native SS benchmark")
    for case in protocol.get("cases", []):
        for key in ("cache_payload", "target_mesh"):
            item = case.get(key, {})
            source = Path(str(item.get("path", "")))
            if not source.is_file() or sha256_file(source) != item.get("sha256"):
                raise RuntimeError(
                    f"frozen case binding changed: {case.get('case_id')}.{key}"
                )
    if not protocol.get("cases"):
        raise RuntimeError("protocol contains no cases")
    protocol["_protocol_path"] = str(path.resolve())
    return protocol


def _masked_square_inputs(
    image_paths: list[str],
    mask_paths: list[str],
    pixel_intrinsics,
):
    """Match GenRecon's official square crop and alpha-masking convention."""

    import torch
    import torch.nn.functional as functional

    normalized_intrinsics = []
    images_512 = []
    images_1024 = []
    preprocessing = []
    for image_value, mask_value, intrinsic_value in zip(
        image_paths, mask_paths, pixel_intrinsics
    ):
        image_path = Path(image_value).resolve()
        mask_path = Path(mask_value).resolve()
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if image.size != mask.size:
            raise RuntimeError(f"image/mask size mismatch: {image_path}")
        width, height = image.size
        intrinsic = torch.as_tensor(intrinsic_value, dtype=torch.float32).clone()
        if intrinsic.shape != (3, 3):
            raise RuntimeError(f"invalid source intrinsic for {image_path}")

        left, top, crop_size = 0, 0, min(width, height)
        if width != height:
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            max_half = min(cx, width - cx, cy, height - cy)
            crop_size = int(2 * max_half)
            if crop_size <= 0:
                raise RuntimeError(f"invalid principal-point crop for {image_path}")
            left = max(
                0,
                min(int(round(cx - 0.5 * crop_size)), width - crop_size),
            )
            top = max(
                0,
                min(int(round(cy - 0.5 * crop_size)), height - crop_size),
            )
            image = image.crop((left, top, left + crop_size, top + crop_size))
            mask = mask.crop((left, top, left + crop_size, top + crop_size))
            intrinsic[0, 2] -= float(left)
            intrinsic[1, 2] -= float(top)
        elif width == height:
            crop_size = width

        intrinsic[0, :] /= float(crop_size)
        intrinsic[1, :] /= float(crop_size)
        normalized_intrinsics.append(intrinsic)

        mask_array = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)
        for size, output in ((512, images_512), (1024, images_1024)):
            rgb = np.asarray(
                image.resize((size, size), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            tensor = torch.from_numpy(rgb).permute(2, 0, 1) / 255.0
            alpha = functional.interpolate(
                mask_array[None, None],
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )[0]
            output.append(tensor * alpha)
        preprocessing.append(
            {
                "image": binding(image_path),
                "mask": binding(mask_path),
                "source_size_wh": [int(width), int(height)],
                "crop_left_top_size": [int(left), int(top), int(crop_size)],
                "normalized_intrinsic": intrinsic.tolist(),
            }
        )
    return (
        torch.stack(normalized_intrinsics),
        torch.stack(images_512),
        torch.stack(images_1024),
        preprocessing,
    )


def _validate_existing(
    result_path: Path,
    mesh_path: Path,
    *,
    protocol_sha256: str,
    case_id: str,
) -> bool:
    if not result_path.is_file() or not mesh_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return bool(
        result.get("format") == FORMAT
        and result.get("complete") is True
        and result.get("protocol_sha256") == protocol_sha256
        and result.get("case_id") == case_id
        and result.get("mesh", {}).get("sha256") == sha256_file(mesh_path)
    )


def _run_official_geometry(pipeline, selected, *, seed: int, occ_threshold: float):
    """Official FullScene SS + Shape-SLAT path, stopping before texture.

    This mirrors ``FullSceneImagesTo3DPipeline.run`` through joint shape decode.
    Texture-SLAT cannot affect mesh vertices/faces and is intentionally omitted
    from a geometry-only benchmark.
    """

    from contextlib import nullcontext

    import torch

    from genrecon.modules.sparse import SparseTensor
    from genrecon.pipelines.samplers.multi_diff_orchestrator import (
        MultiDiffusionOrchestrator,
    )

    pipeline._require_components(
        required_model_keys=(
            "sparse_structure_flow_model",
            "sparse_structure_decoder",
            "shape_slat_flow_model_512",
            "shape_slat_decoder",
        ),
        required_attrs=(
            "image_cond_model",
            "sparse_structure_sampler",
            "sparse_structure_sampler_params",
            "shape_slat_sampler",
            "shape_slat_sampler_params",
            "shape_slat_normalization",
        ),
    )
    torch.manual_seed(int(seed))

    def amp_context(model):
        if pipeline.device.type == "cuda" and getattr(model, "dtype", None) in {
            torch.float16,
            torch.bfloat16,
        }:
            return torch.autocast(device_type="cuda", dtype=model.dtype)
        return nullcontext()

    scene_features = pipeline._encode_scene(selected.scene_images_512, 512)
    cond_features = pipeline._encode_scene(
        torch.stack(selected.cond2d_images_512), 512
    )
    relative_translations = [torch.zeros(3, dtype=torch.float32)]

    ss_model = pipeline.models["sparse_structure_flow_model"]
    scene_ext, scene_intr = pipeline._prep_extrinsics_intrinsics(
        selected.scene_extrinsics_c0,
        selected.scene_intrinsics,
        ss_model.dtype,
    )
    ss_conditions = pipeline._build_global_dense_cond(
        ss_model,
        scene_features.to(ss_model.dtype),
        cond_features.to(ss_model.dtype),
        scene_ext,
        scene_intr,
        relative_translations,
        ss_model.resolution,
    )
    ss_noise = [
        torch.randn(
            1,
            ss_model.in_channels,
            32,
            32,
            32,
            dtype=ss_model.dtype,
            device=pipeline.device,
        )
    ]
    if pipeline.low_vram:
        ss_model.to(pipeline.device)
    with amp_context(ss_model):
        ss_latents = MultiDiffusionOrchestrator(
            pipeline.sparse_structure_sampler
        ).sample(
            ss_model,
            ss_noise,
            [item["cond"] for item in ss_conditions],
            [item["neg_cond"] for item in ss_conditions],
            relative_translations,
            **pipeline.sparse_structure_sampler_params,
            tqdm_desc="Sampling sparse structure",
        )
    if pipeline.low_vram:
        ss_model.cpu()
    del ss_noise, ss_conditions
    occ_logits = pipeline.joint_decode_sparse_structure(
        ss_latents, relative_translations, 32
    )
    del ss_latents

    shape_model = pipeline.models["shape_slat_flow_model_512"]
    coordinates = pipeline.extract_coords_from_occ_logits(
        occ_logits,
        32,
        shape_model.resolution,
        threshold=float(occ_threshold),
    )
    del occ_logits
    std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(
        pipeline.device
    )
    mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(
        pipeline.device
    )
    shape_ext, shape_intr = pipeline._prep_extrinsics_intrinsics(
        selected.scene_extrinsics_c0,
        selected.scene_intrinsics,
        shape_model.dtype,
    )
    shape_conditions = pipeline._build_global_sparse_cond(
        shape_model,
        scene_features.to(shape_model.dtype),
        cond_features.to(shape_model.dtype),
        shape_ext,
        shape_intr,
        coordinates,
        relative_translations,
        shape_model.resolution,
    )
    shape_noise = [
        SparseTensor(
            feats=torch.randn(
                coordinates[0].shape[0],
                shape_model.in_channels,
                device=pipeline.device,
            ),
            coords=coordinates[0],
        )
    ]
    if pipeline.low_vram:
        shape_model.to(pipeline.device)
    with amp_context(shape_model):
        shape_raw = MultiDiffusionOrchestrator(pipeline.shape_slat_sampler).sample(
            shape_model,
            shape_noise,
            [item["cond"] for item in shape_conditions],
            [item["neg_cond"] for item in shape_conditions],
            relative_translations,
            **pipeline.shape_slat_sampler_params,
            tqdm_desc="Sampling shape SLat",
        )
    if pipeline.low_vram:
        shape_model.cpu()
    del shape_noise, shape_conditions, scene_features, cond_features
    shape_latents = [item * std + mean for item in shape_raw]
    del shape_raw
    torch.cuda.empty_cache()
    scene_mesh, _, _, _, _ = pipeline.joint_decode_shape_slats(
        shape_latents,
        512,
        relative_translations,
    )
    del shape_latents
    scene_mesh.fill_holes()
    return scene_mesh, coordinates


def run_case(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    import torch
    import trimesh

    case_by_id = {str(item["case_id"]): item for item in protocol["cases"]}
    if args.case_id not in case_by_id:
        raise KeyError(f"case_id absent from frozen protocol: {args.case_id}")
    case = case_by_id[args.case_id]
    output_dir = args.output_root.resolve() / args.case_id
    mesh_path = output_dir / "mesh_canonical.obj"
    result_path = output_dir / "result.json"
    if _validate_existing(
        result_path,
        mesh_path,
        protocol_sha256=protocol["protocol_sha256"],
        case_id=args.case_id,
    ):
        print(f"reuse immutable official GenRecon result: {result_path}")
        return
    if output_dir.exists():
        raise FileExistsError(
            f"partial official GenRecon output exists; preserve and inspect: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    payload_path = Path(case["cache_payload"]["path"]).resolve()
    payload = torch.load(payload_path, map_location="cpu")
    if str(payload.get("uid")) != str(case["uid"]):
        raise RuntimeError("cache payload UID differs from frozen case")
    if payload.get("extrinsics_type") != "c2w":
        raise RuntimeError("official GenRecon adapter requires Blender c2w inputs")
    image_paths = [str(value) for value in payload["image_paths"]]
    mask_paths = [str(value) for value in payload["mask_paths"]]
    intrinsics_px = torch.as_tensor(payload["source_intrinsics"], dtype=torch.float32)
    c2w = torch.as_tensor(payload["extrinsics"], dtype=torch.float32)
    count = len(image_paths)
    if not (
        count == int(case["view_count"])
        == len(mask_paths)
        == len(intrinsics_px)
        == len(c2w)
    ):
        raise RuntimeError("frozen case input-view cardinality mismatch")
    intrinsics, images_512, images_1024, preprocessing = _masked_square_inputs(
        image_paths, mask_paths, intrinsics_px
    )
    flip_y_z = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    extrinsics_c0 = torch.stack([flip_y_z @ torch.linalg.inv(item) for item in c2w])
    cond_index = int(case["selected_input_position"])
    if not 0 <= cond_index < count:
        raise RuntimeError("frozen selected_input_position is outside input set")

    genrecon_root = args.genrecon_root.resolve()
    sys.path.insert(0, str(genrecon_root))
    from genrecon.pipelines.full_scene_images_to_3d import (  # noqa: PLC0415
        FullSceneImagesTo3DPipeline,
    )
    from genrecon.pipelines.types import SelectedImages  # noqa: PLC0415

    selected = SelectedImages(
        scene_images_1024=images_1024,
        scene_images_512=images_512,
        scene_intrinsics=intrinsics,
        scene_extrinsics_c0=extrinsics_c0,
        cond2d_images_1024=[images_1024[cond_index]],
        cond2d_images_512=[images_512[cond_index]],
        cond2d_intrinsics=[intrinsics[cond_index]],
        cond2d_extrinsics_c0=[extrinsics_c0[cond_index]],
        chunk_indices=[0],
    )
    checkpoints = {
        "sparse_structure_flow_model": str(args.ss_checkpoint.resolve()),
        "shape_slat_flow_model_512": str(args.shape_checkpoint.resolve()),
    }
    stage_train_configs = {
        "sparse_structure_flow_model": str(args.ss_train_config.resolve()),
        "shape_slat_flow_model_512": str(args.shape_train_config.resolve()),
    }
    official_config = json.loads(
        args.pipeline_config.resolve().read_text(encoding="utf-8")
    )
    runtime_config = json.loads(json.dumps(official_config))
    if args.dino_model is not None:
        dino_model = args.dino_model.resolve()
        if not (dino_model / "config.json").is_file():
            raise FileNotFoundError(f"local DINO model is incomplete: {dino_model}")
        runtime_config["args"]["image_cond_model"]["args"]["model_name"] = str(
            dino_model
        )
    runtime_config_path = output_dir / "pipeline_runtime.json"
    atomic_json(runtime_config_path, runtime_config)
    started = time.time()
    pipeline = FullSceneImagesTo3DPipeline.from_finetuned(
        stage_models=checkpoints,
        pipeline_config_file=str(runtime_config_path),
        stage_train_configs=stage_train_configs,
    )
    pipeline.proj_batch_voxels = int(args.proj_batch_voxels)
    device = torch.device(args.device)
    pipeline.to(device)
    scene, coords_list = _run_official_geometry(
        pipeline,
        selected,
        seed=int(args.seed),
        occ_threshold=float(args.occ_threshold),
    )
    vertices = scene.vertices.detach().float().cpu().numpy()
    faces = scene.faces.detach().long().cpu().numpy()
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise RuntimeError("official GenRecon returned an empty/non-finite mesh")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    temporary_mesh = mesh_path.with_name(mesh_path.name + f".tmp.{os.getpid()}")
    mesh.export(temporary_mesh, file_type="obj")
    os.replace(temporary_mesh, mesh_path)

    result = {
        "format": FORMAT,
        "complete": True,
        "scope": "official GenRecon scene model on centred single-object OOD input",
        "protocol": binding(Path(protocol["_protocol_path"])),
        "protocol_sha256": protocol["protocol_sha256"],
        "case_id": args.case_id,
        "uid": case["uid"],
        "source": case["source"],
        "view_count": count,
        "seed": int(args.seed),
        "pipeline_type": "512_geometry_only",
        "stages": [
            "official FullScene sparse structure",
            "official FullScene shape SLat",
            "official joint shape decoder",
        ],
        "texture_stage": "not run; texture cannot change geometry metrics",
        "occ_threshold": float(args.occ_threshold),
        "camera_conversion": CAMERA_CONVERSION,
        "chunk_policy": "one identity unit chunk [-0.5,0.5]^3",
        "cond2d_policy": "frozen largest-mask selected_input_position",
        "cond2d_input_position": cond_index,
        "preprocessing": preprocessing,
        "cache_payload": binding(payload_path),
        "checkpoints": {key: binding(value) for key, value in checkpoints.items()},
        "stage_train_configs": {
            key: binding(value) for key, value in stage_train_configs.items()
        },
        "official_pipeline_config": binding(args.pipeline_config),
        "runtime_pipeline_config": binding(runtime_config_path),
        "dino_model_override": (
            {
                "root": str(args.dino_model.resolve()),
                "config": binding(args.dino_model.resolve() / "config.json"),
                "weights": binding(
                    args.dino_model.resolve() / "model.safetensors"
                ),
                "upstream_sha_verified": False,
            }
            if args.dino_model is not None
            else None
        ),
        "genrecon_code": binding(
            genrecon_root / "genrecon/pipelines/full_scene_images_to_3d.py"
        ),
        "adapter_code": binding(Path(__file__).resolve()),
        "mesh": binding(mesh_path),
        "mesh_coordinate_frame": "canonical chunk0; identity to reference",
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces)),
        "occupied_coordinate_count": int(coords_list[0].shape[0]),
        "elapsed_seconds": float(time.time() - started),
        "max_cuda_memory_allocated_mib": (
            float(torch.cuda.max_memory_allocated() / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--genrecon_root", type=Path, default=Path("GenRecon"))
    parser.add_argument(
        "--pipeline_config",
        type=Path,
        default=Path("GenRecon/configs/pipelines/original.json"),
    )
    parser.add_argument("--ss_checkpoint", type=Path, required=True)
    parser.add_argument("--shape_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ss_train_config",
        type=Path,
        required=True,
        help="Training config that defines the sparse-structure checkpoint architecture.",
    )
    parser.add_argument(
        "--shape_train_config",
        type=Path,
        required=True,
        help="Training config that defines the 512 shape-SLat checkpoint architecture.",
    )
    parser.add_argument(
        "--dino_model",
        type=Path,
        default=None,
        help="Optional local mirror of the official DINOv3 ViT-L/16 weights.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occ_threshold", type=float, default=-1.0)
    parser.add_argument("--proj_batch_voxels", type=int, default=4096)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol_path = args.protocol.resolve()
    for required in (
        args.ss_checkpoint,
        args.shape_checkpoint,
        args.ss_train_config,
        args.shape_train_config,
        args.pipeline_config,
    ):
        if not required.resolve().is_file():
            raise FileNotFoundError(required.resolve())
    protocol = validate_protocol(protocol_path)
    run_case(args, protocol)


if __name__ == "__main__":
    main()
