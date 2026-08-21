#!/usr/bin/env python3
"""Adapter for color+mask datasets with reusable or freshly built COLMAP."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from manual_mesh_reconstruction.common import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from manual_mesh_reconstruction.data_adapters.common import (
    CameraFrame,
    deferred_selection_request,
    find_image_directory,
    find_mask_directory,
    indexed_media,
    materialize_raw_cache,
    natural_key,
    safe_id,
    utc_now,
)
from manual_mesh_reconstruction.raw_cache import (
    camera_intrinsics,
    parse_cameras,
    parse_registered_images,
    qvec_to_rotation,
)


COLMAP_RUN_FORMAT = "manual_mesh_reconstruction.colmap_run.v1"
COLMAP_INVENTORY_FORMAT = "manual_mesh_reconstruction.colmap_input_inventory.v1"


def _text_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt"))


def _binary_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))


def _model_candidates(root: Path) -> list[Path]:
    root = root.resolve()
    candidates = []
    if _text_complete(root) or _binary_complete(root):
        candidates.append(root)
    if root.is_dir():
        candidates.extend(
            path
            for path in root.iterdir()
            if path.is_dir() and (_text_complete(path) or _binary_complete(path))
        )
    unique = {str(path.resolve()): path.resolve() for path in candidates}
    return sorted(unique.values(), key=lambda path: natural_key(path.name))


def _model_file_hashes(path: Path) -> dict[str, str]:
    names = (
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    )
    return {name: sha256_file(path / name) for name in names if (path / name).is_file()}


def discover_existing_sparse(dataset: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        candidates = _model_candidates(explicit.expanduser().resolve(strict=True))
        if not candidates:
            raise FileNotFoundError(f"explicit COLMAP model is incomplete: {explicit}")
        return candidates
    roots = [
        dataset / "sparse/0",
        dataset / "sparse",
        dataset / "colmap/sparse/0",
        dataset / "colmap/sparse",
        dataset / "sparse_text/0",
        dataset / "sparse_text",
    ]
    output: dict[str, Path] = {}
    for root in roots:
        if root.exists():
            for candidate in _model_candidates(root):
                output[str(candidate)] = candidate
    return list(output.values())


def _resolve_colmap_bin(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(f"COLMAP executable is unavailable: {value}")
    return Path(resolved).resolve()


def _run_logged(
    *,
    stage: str,
    command: Sequence[str],
    workspace: Path,
    environment: dict[str, str],
    resume: bool,
    complete: Callable[[], bool],
) -> None:
    marker = workspace / "stages" / f"{stage}.json"
    if marker.is_file():
        payload = load_json(marker)
        if payload.get("passed") is not True or payload.get("command") != list(command):
            raise RuntimeError(f"stale COLMAP stage marker: {marker}")
        if not complete():
            raise RuntimeError(f"COLMAP stage marker has incomplete outputs: {marker}")
        if not resume:
            raise FileExistsError(f"COLMAP stage exists; pass --resume: {marker}")
        print(f"[dataset_adapter:colmap] reuse stage={stage}", flush=True)
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "colmap.log"
    header = f"\n[{utc_now()}] stage={stage} command={shlex.join(command)}\n"
    print(header.rstrip(), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(workspace),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[dataset_adapter:colmap:{stage}] {line}", end="", flush=True)
            log.write(line)
            log.flush()
        return_code = int(process.wait())
    if return_code != 0:
        raise RuntimeError(f"COLMAP stage={stage} failed rc={return_code}; log={log_path}")
    if not complete():
        raise RuntimeError(f"COLMAP stage={stage} produced incomplete outputs")
    atomic_json(
        marker,
        {
            "format": COLMAP_RUN_FORMAT,
            "created_at_utc": utc_now(),
            "stage": stage,
            "command": list(command),
            "log": str(log_path.resolve()),
            "passed": True,
        },
    )


def _paired_source(dataset: Path) -> tuple[Path, Path, list[tuple[str, Path, Path]]]:
    image_dir = find_image_directory(dataset)
    mask_dir = find_mask_directory(dataset)
    images = indexed_media(image_dir)
    masks = indexed_media(mask_dir)
    stems = sorted(set(images) & set(masks), key=natural_key)
    if len(stems) < 2:
        raise RuntimeError(f"dataset has fewer than two paired color/mask frames: {dataset}")
    return image_dir, mask_dir, [(stem, images[stem], masks[stem]) for stem in stems]


def _prepare_rebuild_inputs(
    dataset: Path, workspace: Path
) -> tuple[list[dict[str, Any]], str]:
    _image_dir, _mask_dir, pairs = _paired_source(dataset)
    images_out = workspace / "images"
    masks_out = workspace / "masks"
    colmap_masks = workspace / "colmap_masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    colmap_masks.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (stem, image_path, mask_path) in enumerate(pairs):
        name = f"frame_{index:06d}.png"
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            size = image.size
            image.save(images_out / name, format="PNG")
        with Image.open(mask_path) as handle:
            mask = handle.convert("L")
            if mask.size != size:
                mask = mask.resize(size, Image.Resampling.NEAREST)
            mask.save(masks_out / name, format="PNG")
            # COLMAP appends .png to the complete image filename.
            mask.save(colmap_masks / f"{name}.png", format="PNG")
        rows.append(
            {
                "index": int(index),
                "stem": stem,
                "normalized_name": name,
                "source_image": str(image_path),
                "source_image_sha256": sha256_file(image_path),
                "source_mask": str(mask_path),
                "source_mask_sha256": sha256_file(mask_path),
                "normalized_image_sha256": sha256_file(images_out / name),
                "normalized_mask_sha256": sha256_file(masks_out / name),
            }
        )
    inventory = {
        "format": COLMAP_INVENTORY_FORMAT,
        "created_at_utc": utc_now(),
        "dataset": str(dataset.resolve()),
        "frame_count": len(rows),
        "frames": rows,
    }
    inventory["identity_sha256"] = canonical_sha256(inventory)
    inventory_path = workspace / "input_inventory.json"
    if inventory_path.is_file():
        old = load_json(inventory_path)
        if old.get("identity_sha256") != inventory["identity_sha256"]:
            raise RuntimeError(f"COLMAP input inventory changed: {inventory_path}")
    else:
        atomic_json(inventory_path, inventory)
    return rows, str(inventory["identity_sha256"])


def _convert_to_text(
    *,
    source: Path,
    destination: Path,
    colmap_bin: Path,
    workspace: Path,
    environment: dict[str, str],
    resume: bool,
    stage: str,
) -> Path:
    if _text_complete(source):
        return source
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(colmap_bin),
        "model_converter",
        "--input_path",
        str(source),
        "--output_path",
        str(destination),
        "--output_type",
        "TXT",
    ]
    _run_logged(
        stage=stage,
        command=command,
        workspace=workspace,
        environment=environment,
        resume=resume,
        complete=lambda: _text_complete(destination),
    )
    return destination


def _select_largest_text_model(
    *,
    candidates: Sequence[Path],
    workspace: Path,
    colmap_bin: Path,
    environment: dict[str, str],
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    rows = []
    for index, candidate in enumerate(candidates):
        text = _convert_to_text(
            source=candidate,
            destination=workspace / "converted_models" / f"model_{index:03d}",
            colmap_bin=colmap_bin,
            workspace=workspace,
            environment=environment,
            resume=resume,
            stage=f"convert_model_{index:03d}",
        )
        registered = parse_registered_images(text / "images.txt")
        rows.append(
            {
                "source": candidate,
                "text": text,
                "source_file_sha256": _model_file_hashes(candidate),
                "registered_count": len(registered),
            }
        )
    if not rows:
        raise RuntimeError("COLMAP produced no candidate model")
    selected = min(
        rows,
        key=lambda row: (-int(row["registered_count"]), natural_key(row["source"].name)),
    )
    return Path(selected["text"]), {
        "selection": "maximum_registered_images_then_lexical_model_path",
        "selected_source_model": str(Path(selected["source"]).resolve()),
        "selected_text_model": str(Path(selected["text"]).resolve()),
        "registered_count": int(selected["registered_count"]),
        "candidates": [
            {
                "source": str(Path(row["source"]).resolve()),
                "text": str(Path(row["text"]).resolve()),
                "source_file_sha256": row["source_file_sha256"],
                "registered_count": int(row["registered_count"]),
            }
            for row in rows
        ],
    }


def _run_rebuild(
    *,
    dataset: Path,
    workspace: Path,
    colmap_bin: Path,
    matcher: str,
    use_foreground_masks: bool,
    use_gpu: bool,
    resume: bool,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    inventory, inventory_identity = _prepare_rebuild_inputs(dataset, workspace)
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    use_gpu_value = "1" if use_gpu else "0"
    feature = [
        str(colmap_bin),
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(workspace / "images"),
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        use_gpu_value,
        "--SiftExtraction.max_num_features",
        "16384",
        "--SiftExtraction.peak_threshold",
        "0.003",
    ]
    if use_gpu:
        feature.extend(["--SiftExtraction.gpu_index", "0"])
    if use_foreground_masks:
        feature.extend(["--ImageReader.mask_path", str(workspace / "colmap_masks")])
    matcher_command = [
        str(colmap_bin),
        "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher",
        "--database_path",
        str(database),
        "--SiftMatching.use_gpu",
        use_gpu_value,
        "--SiftMatching.guided_matching",
        "1",
    ]
    if use_gpu:
        matcher_command.extend(["--SiftMatching.gpu_index", "0"])
    mapper = [
        str(colmap_bin),
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(workspace / "images"),
        "--output_path",
        str(sparse),
        "--Mapper.min_model_size",
        "8",
        "--Mapper.ba_refine_principal_point",
        "0",
    ]
    stages = [
        ("feature_extractor", feature, lambda: database.is_file() and database.stat().st_size > 0),
        ("matcher", matcher_command, lambda: database.is_file() and database.stat().st_size > 0),
        (
            "mapper",
            mapper,
            lambda: any(_binary_complete(path) for path in _model_candidates(sparse)),
        ),
    ]
    for name, command, complete in stages:
        _run_logged(
            stage=name,
            command=command,
            workspace=workspace,
            environment=environment,
            resume=resume,
            complete=complete,
        )
    model, model_selection = _select_largest_text_model(
        candidates=_model_candidates(sparse),
        workspace=workspace,
        colmap_bin=colmap_bin,
        environment=environment,
        resume=resume,
    )
    return model, {
        "mode": "rebuild",
        "colmap_executed": True,
        "colmap_model_reconstruction_executed": True,
        "colmap_model_converter_executed": True,
        "colmap_binary": str(colmap_bin),
        "matcher": matcher,
        "foreground_masks_used_for_features": bool(use_foreground_masks),
        "gpu_used": bool(use_gpu),
        "input_inventory_identity": inventory_identity,
        "model_selection": model_selection,
    }, inventory


def _frames_from_model(
    *,
    sparse: Path,
    image_dir: Path,
    mask_dir: Path,
    source_name_by_registered_name: dict[str, str] | None,
    pose_source: str,
) -> list[CameraFrame]:
    cameras = parse_cameras(sparse / "cameras.txt")
    registered = parse_registered_images(sparse / "images.txt")
    images = {path.name: path for path in image_dir.iterdir() if path.is_file()}
    masks_by_stem = indexed_media(mask_dir)
    output = []
    for row in sorted(registered, key=lambda value: natural_key(str(value["name"]))):
        name = str(row["name"])
        image = images.get(name)
        mask = masks_by_stem.get(Path(name).stem)
        if image is None or mask is None:
            continue
        camera = cameras[int(row["camera_id"])]
        K, distortion = camera_intrinsics(camera)
        with Image.open(image) as handle:
            width, height = handle.size
        if int(camera["width"]) != width or int(camera["height"]) != height:
            K[0, :] *= width / float(camera["width"])
            K[1, :] *= height / float(camera["height"])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = qvec_to_rotation(row["qvec"])
        T[:3, 3] = np.asarray(row["tvec"], dtype=np.float64)
        output.append(
            CameraFrame(
                source_index=len(output),
                source_name=(source_name_by_registered_name or {}).get(name, name),
                image_path=image.resolve(),
                mask_path=mask.resolve(),
                K=K,
                T_W2C=T,
                camera_model=str(camera["model"]),
                distortion=tuple(distortion),
                pose_source=pose_source,
            )
        )
    if not output:
        raise RuntimeError("COLMAP model has no registered RGB/mask pairs")
    return output


def adapt(
    *,
    input_path: Path,
    output_dir: Path,
    selected_view_count: int,
    selection_policy: str,
    random_seed: int,
    colmap_mode: str,
    colmap_sparse: Path | None,
    colmap_bin: str,
    matcher: str,
    use_foreground_masks: bool,
    use_gpu: bool,
    resume: bool,
) -> dict[str, Any]:
    dataset = input_path.expanduser().resolve(strict=True)
    if not dataset.is_dir():
        raise NotADirectoryError(dataset)
    output_dir = output_dir.resolve()
    workspace = output_dir.parent / "colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    existing = discover_existing_sparse(dataset, colmap_sparse)
    if colmap_mode == "reuse" and not existing:
        raise FileNotFoundError(
            "--colmap-mode reuse requested, but no complete existing model was found; "
            "pass --colmap-sparse explicitly or use --colmap-mode rebuild"
        )
    effective = "reuse" if colmap_mode == "reuse" or (colmap_mode == "auto" and existing) else "rebuild"
    needs_converter = effective == "rebuild" or any(
        not _text_complete(candidate) for candidate in existing
    )
    binary = (
        _resolve_colmap_bin(colmap_bin)
        if needs_converter
        else Path(shutil.which(colmap_bin) or colmap_bin)
    )
    if effective == "reuse":
        environment = dict(os.environ)
        sparse, model_selection = _select_largest_text_model(
            candidates=existing,
            workspace=workspace,
            colmap_bin=binary,
            environment=environment,
            resume=resume,
        )
        image_dir = find_image_directory(dataset)
        mask_dir = find_mask_directory(dataset)
        source_names = None
        colmap_record = {
            "requested_mode": colmap_mode,
            "effective_mode": "reuse",
            "colmap_executed": False,
            "colmap_model_reconstruction_executed": False,
            "colmap_model_converter_executed": any(
                not _text_complete(candidate) for candidate in existing
            ),
            "existing_model_reused": True,
            "model_selection": model_selection,
        }
        inventory = None
    else:
        sparse, colmap_record, inventory = _run_rebuild(
            dataset=dataset,
            workspace=workspace,
            colmap_bin=binary,
            matcher=matcher,
            use_foreground_masks=use_foreground_masks,
            use_gpu=use_gpu,
            resume=resume,
        )
        colmap_record["requested_mode"] = colmap_mode
        colmap_record["effective_mode"] = "rebuild"
        image_dir = workspace / "images"
        mask_dir = workspace / "masks"
        source_names = {str(row["normalized_name"]): str(row["source_image"]) for row in inventory}
    frames = _frames_from_model(
        sparse=sparse,
        image_dir=image_dir,
        mask_dir=mask_dir,
        source_name_by_registered_name=source_names,
        pose_source=f"colmap_{effective}",
    )
    selection = deferred_selection_request(
        len(frames),
        int(selected_view_count),
        policy=selection_policy,
        random_seed=int(random_seed),
    )
    selection.update(
        {
            "eligible_registered_rgb_mask_count": len(frames),
            "selection_domain": "successfully COLMAP-registered frames with masks",
        }
    )
    source_binding = {
        "colmap": colmap_record,
        "sparse_text_model": str(sparse.resolve()),
        "sparse_text_sha256": {
            name: sha256_file(sparse / name)
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        },
        "image_dir": str(image_dir.resolve()),
        "mask_dir": str(mask_dir.resolve()),
    }
    raw, row = materialize_raw_cache(
        output_dir=output_dir,
        dataset_type="colmap_color_mask",
        source_path=dataset,
        category="colmap_real",
        object_id=safe_id(dataset.name, label="dataset name"),
        input_frames=frames,
        selection_request=selection,
        source_binding=source_binding,
        extra_report={
            "colmap": colmap_record,
            "gravity_up_W": None,
        },
    )
    return {
        "raw_cache_report": str(raw.resolve()),
        "raw_cache_report_sha256": sha256_file(raw),
        "runtime_input_manifest": None,
        "runtime_input_manifest_sha256": None,
        "object_key": row["object_key"],
        "geometry_mode": "pose_mask",
        "gravity_up_W": None,
        "selection": row["view_selection"],
        "source_binding": source_binding,
    }
