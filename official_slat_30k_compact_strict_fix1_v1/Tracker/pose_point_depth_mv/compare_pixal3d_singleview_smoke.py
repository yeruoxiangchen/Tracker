#!/usr/bin/env python3
"""Exploratory original-Pixal3D single-view versus current multi-view smoke.

The protocol intentionally reuses already exported Direct-SS + Direct-SLAT
meshes.  It has three stages:

1. ``prepare`` freezes a small, performance-independent subset and creates
   masked RGBA single-view inputs for the original Pixal3D pipeline.
2. ``infer`` exports the official postprocessed GLB for all or an exact subset
   of frozen cases.  The CLI's ``--case_ids`` selector permits a shell runner
   to use one fresh Python/CUDA process per case.  The export exactly follows
   ``Pixal3D/inference.py``: ``o_voxel.postprocess.to_glb`` with remeshing,
   followed by the official output rotation.
3. ``evaluate`` applies the same proper-rotation similarity ICP to both
   methods, then measures both meshes against the same canonical GT mesh.

This is an exploratory shape comparison.  Similarity alignment deliberately
removes global pose/scale error and therefore is not a pose-aware AR metric.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable


FORMAT = "pose_point_depth_mv.pixal3d_singleview_smoke.v1"
REPORT_FORMAT = "pose_point_depth_mv.pixal3d_singleview_smoke_report.v2"
INFERENCE_RESULT_FORMAT = (
    "pose_point_depth_mv.pixal3d_singleview_official_inference.v2"
)
OFFICIAL_GEOMETRY_EXPORT = (
    "official Pixal3D final GLB after o_voxel.postprocess.to_glb "
    "remesh and inference.py output rotation"
)
OFFICIAL_POSTPROCESS = {
    "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    "decimation_target": 1_000_000,
    "texture_size": 4096,
    "remesh": True,
    "remesh_band": 1.0,
    "remesh_project": 0.0,
}
TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
NAF_COMMIT = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"
NAF_HUBCONF_SHA256 = "b81540828672fe56214a93705e4484a92cbc5569caf95b07fb9e49cb3f5adb69"
NAF_CHECKPOINT_SHA256 = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"
NAF_SOURCE_MANIFEST_SHA256 = "f479adf2f915defce20ac6912b3f1d991c37ae77c9b28966f5f57c7ea5cb9082"
CUMESH_FILL_HOLES_GUARD_VERSION = "boundary_loop_empty_noop.v1"


def _guarded_fill_holes(
    mesh: Any,
    original_fill_holes: Any,
    max_hole_perimeter: float = 3e-2,
) -> Any:
    """Match Pixal3D's boundary guards before entering the CuMesh CUDA op.

    ``o_voxel.postprocess.to_glb`` calls ``CuMesh.fill_holes`` directly even
    though Pixal3D's decoded mesh has already gone through the guarded
    ``Mesh.fill_holes`` path.  Some closed meshes therefore reach the CUDA
    extension with no boundary work items, which can launch an invalid empty
    kernel configuration.  Empty boundary sets are a semantic no-op; meshes
    with real boundary loops still call the untouched CuMesh implementation.
    """

    mesh.get_edges()
    mesh.get_boundary_info()
    if int(mesh.num_boundaries) == 0:
        print(
            "[pixal3d_smoke] CuMesh fill_holes skipped: no boundaries",
            flush=True,
        )
        return None

    mesh.get_vertex_edge_adjacency()
    mesh.get_vertex_boundary_adjacency()
    mesh.get_manifold_boundary_adjacency()
    mesh.read_manifold_boundary_adjacency()
    mesh.get_boundary_connected_components()
    mesh.get_boundary_loops()
    if int(mesh.num_boundary_loops) == 0:
        print(
            "[pixal3d_smoke] CuMesh fill_holes skipped: no boundary loops",
            flush=True,
        )
        return None
    return original_fill_holes(
        mesh, max_hole_perimeter=max_hole_perimeter
    )


@contextmanager
def guarded_cumesh_fill_holes(cumesh_module: Any):
    """Temporarily guard direct CuMesh hole filling inside official to_glb."""

    mesh_type = cumesh_module.CuMesh
    original_fill_holes = mesh_type.fill_holes

    def guarded(
        mesh: Any, max_hole_perimeter: float = 3e-2
    ) -> Any:
        return _guarded_fill_holes(
            mesh,
            original_fill_holes,
            max_hole_perimeter=max_hole_perimeter,
        )

    mesh_type.fill_holes = guarded
    try:
        yield
    finally:
        mesh_type.fill_holes = original_fill_holes


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def binding(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def model_snapshot_identity(model_path: str | Path) -> dict[str, Any]:
    requested = str(model_path)
    local = Path(requested)
    if local.is_dir():
        snapshot = local.resolve()
        revision = snapshot.name
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(requested, local_files_only=True)
        ).resolve()
        revision = snapshot.name
    pipeline_json = snapshot / "pipeline.json"
    if not pipeline_json.is_file():
        raise FileNotFoundError(
            f"Pixal3D snapshot lacks pipeline.json: {snapshot}"
        )
    return {
        "requested": requested,
        "snapshot_path": str(snapshot),
        "revision": revision,
        "pipeline_json": binding(pipeline_json),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def verify_source_manifest(repo: Path, manifest: Path) -> int:
    lines = manifest.read_text(encoding="utf-8").splitlines()
    expected: dict[str, str] = {}
    for index, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise RuntimeError(f"invalid NAF source manifest line {index}")
        digest = line[:64]
        relative_text = line[66:]
        if relative_text.startswith("./"):
            relative_text = relative_text[2:]
        relative = Path(relative_text)
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in expected
        ):
            raise RuntimeError(f"unsafe NAF source manifest line {index}")
        expected[relative.as_posix()] = digest
    actual = {
        path.relative_to(repo).as_posix()
        for path in repo.rglob("*")
        if path.is_file()
    }
    if actual != set(expected):
        raise RuntimeError(
            "NAF source manifest coverage mismatch: "
            f"missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )
    for relative, digest in expected.items():
        path = repo / relative
        if sha256_file(path) != digest:
            raise RuntimeError(f"NAF source file SHA mismatch: {relative}")
    return len(expected)


def configure_local_naf(
    repo: Path,
    checkpoint: Path,
    source_manifest: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    checkpoint = checkpoint.resolve()
    source_manifest = source_manifest.resolve()
    hubconf = repo / "hubconf.py"
    if not hubconf.is_file():
        raise FileNotFoundError(f"NAF repository lacks hubconf.py: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"NAF checkpoint is missing: {checkpoint}")
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"NAF source manifest is missing: {source_manifest}"
        )
    hubconf_sha256 = sha256_file(hubconf)
    checkpoint_sha256 = sha256_file(checkpoint)
    source_manifest_sha256 = sha256_file(source_manifest)
    if hubconf_sha256 != NAF_HUBCONF_SHA256:
        raise RuntimeError(
            f"NAF hubconf SHA mismatch: {hubconf_sha256} != {NAF_HUBCONF_SHA256}"
        )
    if checkpoint_sha256 != NAF_CHECKPOINT_SHA256:
        raise RuntimeError(
            "NAF checkpoint SHA mismatch: "
            f"{checkpoint_sha256} != {NAF_CHECKPOINT_SHA256}"
        )
    if source_manifest_sha256 != NAF_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "NAF source manifest SHA mismatch: "
            f"{source_manifest_sha256} != {NAF_SOURCE_MANIFEST_SHA256}"
        )
    source_file_count = verify_source_manifest(repo, source_manifest)
    os.environ["PIXAL3D_NAF_REPO"] = str(repo)
    os.environ["PIXAL3D_NAF_CHECKPOINT"] = str(checkpoint)
    return {
        "repository": str(repo),
        "commit": NAF_COMMIT,
        "hubconf": binding(hubconf),
        "source_manifest": binding(source_manifest),
        "source_file_count": source_file_count,
        "checkpoint": binding(checkpoint),
        "loading": "torch.hub source=local with pretrained=False, then strict local state_dict",
    }


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_mesh(path: str | Path):
    import trimesh

    loaded = trimesh.load(Path(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [
            item
            for item in loaded.dump(concatenate=False)
            if isinstance(item, trimesh.Trimesh)
            and len(item.vertices)
            and len(item.faces)
        ]
        if not pieces:
            raise ValueError(f"mesh scene contains no triangles: {path}")
        return trimesh.util.concatenate(pieces)
    if isinstance(loaded, trimesh.Trimesh):
        if not len(loaded.vertices) or not len(loaded.faces):
            raise ValueError(f"mesh is empty: {path}")
        return loaded
    raise TypeError(f"unsupported mesh type={type(loaded)} for {path}")


def load_canonical_target(ss_latent: str | Path):
    import numpy as np

    latent_path = Path(ss_latent).resolve()
    with np.load(latent_path) as payload:
        source_glb = Path(str(payload["source_glb"])).resolve()
        center = np.asarray(payload["normalize_center"], dtype=np.float64)
        scale = float(payload["normalize_scale"])
        margin = float(payload["canonical_margin"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid normalization scale={scale} in {latent_path}")
    mesh = load_mesh(source_glb)
    mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64) - center[None]
    ) / scale * margin
    return mesh, {
        "source_glb": str(source_glb),
        "source_glb_sha256": sha256_file(source_glb),
        "ss_latent": str(latent_path),
        "ss_latent_sha256": sha256_file(latent_path),
        "normalize_center": center.tolist(),
        "normalize_scale": scale,
        "canonical_margin": margin,
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT:
        raise ValueError(f"unexpected protocol format={protocol.get('format')!r}")
    expected = str(protocol.get("protocol_sha256", ""))
    body = dict(protocol)
    body.pop("protocol_sha256", None)
    if canonical_sha256(body) != expected:
        raise RuntimeError("protocol canonical SHA-256 mismatch")
    for label, item in protocol["bindings"].items():
        bound = Path(item["path"])
        if not bound.is_file() or sha256_file(bound) != str(item["sha256"]):
            raise RuntimeError(f"frozen binding changed: {label}={bound}")
    for case in protocol["cases"]:
        for key in (
            "input_rgba",
            "source_image",
            "source_mask",
            "current_mesh",
            "target_mesh",
        ):
            item = case[key]
            bound = Path(item["path"])
            if not bound.is_file() or sha256_file(bound) != str(item["sha256"]):
                raise RuntimeError(
                    f"frozen case binding changed: {case['case_id']}.{key}={bound}"
                )
    return protocol


def mask_array(path: Path):
    import numpy as np
    from PIL import Image

    value = np.asarray(Image.open(path))
    if value.ndim == 3:
        if value.shape[2] == 4:
            value = value[:, :, 3]
        else:
            value = np.max(value[:, :, :3], axis=2)
    return value.astype(np.uint8)


def source_frame_metadata(source_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in source_manifest.get("samples", []):
        uid = str(row["uid"])
        for frame in row.get("frames", []):
            name = Path(str(frame["image"])).name
            output[f"{uid}/{name}"] = frame
    return output


def parse_unique_csv(value: str, cast) -> list[Any]:
    values = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV values must be non-empty and unique")
    return values


def select_protocol_cases(
    protocol: dict[str, Any],
    case_ids_csv: str,
) -> list[dict[str, Any]]:
    """Select exact frozen cases without changing their protocol bindings."""

    cases = list(protocol["cases"])
    if not str(case_ids_csv).strip():
        return cases
    requested = parse_unique_csv(case_ids_csv, str)
    by_id = {str(case["case_id"]): case for case in cases}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(
            "requested Pixal3D case IDs are absent from the frozen protocol: "
            f"{missing}"
        )
    return [by_id[case_id] for case_id in requested]


def command_prepare(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    output_dir = args.output_dir.resolve()
    protocol_path = output_dir / "protocol.json"
    utility_mode = args.selection_mode == "report_objects_v1"
    joint_seeds = (
        parse_unique_csv(args.joint_seeds, int)
        if utility_mode
        else [int(args.current_seed)]
    )
    requested_uids = (
        set(parse_unique_csv(args.uids, str)) if str(args.uids).strip() else set()
    )
    if requested_uids and not utility_mode:
        raise ValueError("--uids requires --selection_mode report_objects_v1")
    if protocol_path.is_file():
        protocol = validate_protocol(protocol_path)
        if utility_mode:
            expected_report = binding(args.current_report.resolve())
            expected_cache = binding(args.cache_manifest.resolve())
            comparison = protocol.get("comparison", {})
            actual_uids = {str(case["uid"]) for case in protocol["cases"]}
            actual_objects = {
                str(case["object_uid"]) for case in protocol["cases"]
            }
            checks = {
                "current_report": (
                    protocol["bindings"].get("current_report") == expected_report
                ),
                "cache_manifest": (
                    protocol["bindings"].get("direct_slat_cache_manifest")
                    == expected_cache
                ),
                "selection_mode": (
                    comparison.get("selection_mode") == args.selection_mode
                ),
                "joint_seeds": (
                    comparison.get("joint_seeds") == joint_seeds
                ),
                "uids": not requested_uids or actual_uids == requested_uids,
                "max_objects": (
                    int(args.max_objects) <= 0
                    or len(actual_objects) == int(args.max_objects)
                ),
                "selected_c2w": all(
                    isinstance(case.get("selected_frame"), dict)
                    and case["selected_frame"].get("extrinsics_type") == "c2w"
                    for case in protocol["cases"]
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(
                    "existing utility protocol arguments/bindings changed: "
                    f"{checks}; choose a fresh output directory"
                )
        print(
            json.dumps(
                {
                    "status": "reused",
                    "protocol": str(protocol_path),
                    "cases": len(protocol["cases"]),
                    "protocol_sha256": protocol["protocol_sha256"],
                },
                indent=2,
            )
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"non-empty output has no reusable protocol: {output_dir}; "
            "choose a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    current_report_path = args.current_report.resolve()
    direct_manifest_path = args.cache_manifest.resolve()
    current_report = json.loads(current_report_path.read_text(encoding="utf-8"))
    direct_manifest = json.loads(direct_manifest_path.read_text(encoding="utf-8"))
    direct_rows = direct_manifest["samples"]

    lifting_manifest_path = Path(
        direct_manifest["source_lifting_manifest"]
    ).resolve()
    lifting_manifest = json.loads(lifting_manifest_path.read_text(encoding="utf-8"))
    source_cache_manifest_path = Path(
        lifting_manifest["source_cache_manifest"]
    ).resolve()
    source_cache_manifest = json.loads(
        source_cache_manifest_path.read_text(encoding="utf-8")
    )
    source_cache_by_uid = {
        str(row["uid"]): row for row in source_cache_manifest["samples"]
    }
    source_manifest_path = Path(source_cache_manifest["source_manifest"]).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    frame_metadata = source_frame_metadata(source_manifest)

    candidate_by_views: dict[int, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_object: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    current_root = current_report_path.parent
    for record in current_report["records"]:
        record_seed = int(record["joint_seed"])
        if record_seed not in joint_seeds:
            continue
        dataset_index = int(record["dataset_index"])
        row = direct_rows[dataset_index]
        if (
            str(row["uid"]) != str(record["uid"])
            or int(row["support_seed"]) != record_seed
        ):
            raise RuntimeError(
                f"report/cache identity mismatch at dataset_index={dataset_index}"
            )
        views = int(row["view_count"])
        if views not in args.view_counts:
            continue
        current_mesh = (
            current_root
            / "mesh_pairs"
            / str(record["pair_id"])
            / "full"
            / "mesh_canonical.obj"
        ).resolve()
        if not current_mesh.is_file():
            continue
        source_row = source_cache_by_uid.get(str(row["uid"]))
        if source_row is None:
            raise KeyError(f"missing source cache row for uid={row['uid']}")
        image_paths = [Path(value).resolve() for value in source_row["image_paths"]]
        mask_paths = [Path(value).resolve() for value in source_row["mask_paths"]]
        if len(image_paths) != views or len(mask_paths) != views:
            raise RuntimeError(
                f"uid={row['uid']} expected {views} input views, "
                f"got {len(image_paths)}/{len(mask_paths)}"
            )
        areas = [int(np.count_nonzero(mask_array(path) > 127)) for path in mask_paths]
        best_frame = max(range(len(areas)), key=lambda index: (areas[index], -index))
        rank = hashlib.sha256(
            f"{args.selection_seed}|{views}|{row['object_uid']}|{row['uid']}".encode(
                "utf-8"
            )
        ).hexdigest()
        candidate_by_views[views].append(
            candidate := {
                "rank": rank,
                "record": record,
                "row": row,
                "source_row": source_row,
                "image": image_paths[best_frame],
                "mask": mask_paths[best_frame],
                "mask_area": areas[best_frame],
                "selected_input_position": best_frame,
                "current_mesh": current_mesh,
            }
        )
        object_uid = str(row["object_uid"])
        if record_seed in candidate_by_object[object_uid]:
            raise RuntimeError(
                f"duplicate object/seed in current report: {object_uid}/{record_seed}"
            )
        candidate_by_object[object_uid][record_seed] = candidate

    selected: list[dict[str, Any]] = []
    if utility_mode:
        eligible_objects = []
        for object_uid, seed_map in candidate_by_object.items():
            if requested_uids and str(next(iter(seed_map.values()))["row"]["uid"]) not in requested_uids:
                continue
            if set(seed_map) != set(joint_seeds):
                raise RuntimeError(
                    f"object={object_uid} lacks exact joint seed coverage: "
                    f"actual={sorted(seed_map)} expected={joint_seeds}"
                )
            uids = {str(item["row"]["uid"]) for item in seed_map.values()}
            if len(uids) != 1:
                raise RuntimeError(
                    f"object={object_uid} changes sequence UID across seeds: {sorted(uids)}"
                )
            object_position = min(
                int(item["record"].get("object_position", 1 << 30))
                for item in seed_map.values()
            )
            eligible_objects.append((object_position, object_uid, seed_map))
        eligible_objects.sort(key=lambda item: (item[0], item[1]))
        selected_uids = {
            str(next(iter(seed_map.values()))["row"]["uid"])
            for _, _, seed_map in eligible_objects
        }
        if requested_uids and selected_uids != requested_uids:
            raise RuntimeError(
                "requested utility UIDs are not exactly covered: "
                f"missing={sorted(requested_uids - selected_uids)} "
                f"unexpected={sorted(selected_uids - requested_uids)}"
            )
        if int(args.max_objects) > 0:
            eligible_objects = eligible_objects[: int(args.max_objects)]
        if not eligible_objects:
            raise RuntimeError("utility protocol selected no objects")
        for _, _, seed_map in eligible_objects:
            selected.extend(seed_map[seed] for seed in joint_seeds)
    else:
        for views in args.view_counts:
            bucket = sorted(
                candidate_by_views.get(int(views), []),
                key=lambda item: item["rank"],
            )
            if len(bucket) < args.objects_per_view_count:
                raise RuntimeError(
                    f"view_count={views} has {len(bucket)} usable cases, "
                    f"needs {args.objects_per_view_count}"
                )
            selected.extend(bucket[: args.objects_per_view_count])

    cases = []
    for bucket_position, item in enumerate(selected):
        row = item["row"]
        record = item["record"]
        views = int(row["view_count"])
        record_seed = int(record["joint_seed"])
        if utility_mode:
            case_id = (
                f"u{bucket_position:04d}_{str(row['object_uid'])[:12]}_"
                f"{str(row['uid']).split('_')[-1]}_seed{record_seed}"
            )
        else:
            case_id = (
                f"v{views:02d}_{bucket_position:02d}_"
                f"{str(row['object_uid'])[:12]}_{str(row['uid']).split('_')[-1]}"
            )
        rgba_path = (output_dir / "inputs" / f"{case_id}.png").resolve()
        rgba_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = Image.open(item["image"]).convert("RGB")
        alpha = Image.fromarray(mask_array(item["mask"]), mode="L")
        if alpha.size != rgb.size:
            alpha = alpha.resize(rgb.size, Image.Resampling.NEAREST)
        rgba = rgb.convert("RGBA")
        rgba.putalpha(alpha)
        rgba.save(rgba_path)

        target_path = (output_dir / "targets" / f"{case_id}.obj").resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target, target_metadata = load_canonical_target(row["ss_latent"])
        target.export(
            target_path,
            file_type="obj",
            include_color=False,
            include_texture=False,
        )

        metadata_key = f"{row['uid']}/{item['image'].name}"
        frame = frame_metadata.get(metadata_key, {})
        intrinsic = frame.get("intrinsic")
        extrinsic = frame.get("extrinsic")
        if utility_mode:
            if source_manifest.get("extrinsics_type") != "c2w":
                raise RuntimeError("utility protocol requires c2w source extrinsics")
            if (
                not isinstance(extrinsic, list)
                or len(extrinsic) != 4
                or any(not isinstance(row_value, list) or len(row_value) != 4 for row_value in extrinsic)
            ):
                raise RuntimeError(
                    f"utility protocol lacks selected c2w for uid={row['uid']}"
                )
        fx = (
            float(intrinsic[0][0])
            if isinstance(intrinsic, list) and len(intrinsic) >= 1
            else float(args.fallback_fx)
        )
        input_position = int(item["selected_input_position"])
        view_ids = item["source_row"].get("view_ids", list(range(views)))
        cases.append(
            {
                "case_id": case_id,
                "uid": str(row["uid"]),
                "object_uid": str(row["object_uid"]),
                "view_count": views,
                "current_seed": record_seed,
                "pixal3d_seed": (
                    record_seed if utility_mode else int(args.pixal3d_seed)
                ),
                "dataset_index": int(record["dataset_index"]),
                "pair_id": str(record["pair_id"]),
                "selection_rank": item["rank"],
                "single_view_policy": "largest foreground mask among current method inputs",
                "selected_input_position": input_position,
                "selected_view_id": int(view_ids[input_position]),
                "selected_frame": (
                    {
                        "extrinsics_type": "c2w",
                        "extrinsic": extrinsic,
                        "source_view_index": int(
                            frame.get("source_view_index", view_ids[input_position])
                        ),
                    }
                    if utility_mode
                    else None
                ),
                "mask_foreground_pixels": int(item["mask_area"]),
                "source_width": int(rgb.width),
                "source_height": int(rgb.height),
                "source_fx_pixels": fx,
                "input_rgba": binding(rgba_path),
                "source_image": binding(item["image"]),
                "source_mask": binding(item["mask"]),
                "current_mesh": binding(item["current_mesh"]),
                "target_mesh": binding(target_path),
                "target_metadata": target_metadata,
                "current_canonical_surface_from_frozen_report": record["branches"]["full"][
                    "surface"
                ],
            }
        )

    body = {
        "format": FORMAT,
        "formal": False,
        "purpose": (
            "engineering utility protocol for native single-view Pixal3D, "
            "Direct-SLAT stock, and Direct-SLAT Full"
            if utility_mode
            else (
                "exploratory original single-view Pixal3D versus frozen current "
                "Direct-SS step900 + Direct-SLAT step800 full branch"
            )
        ),
        "comparison": {
            "current_input_budgets": [int(value) for value in args.view_counts],
            "pixal3d_input_budget": 1,
            "selection_mode": args.selection_mode,
            "objects_per_view_count": (
                None if utility_mode else int(args.objects_per_view_count)
            ),
            "object_count": len({str(case["object_uid"]) for case in cases}),
            "joint_seeds": joint_seeds,
            "selection_seed": int(args.selection_seed),
            "selection_is_performance_independent": True,
            "single_view_policy": (
                "largest foreground mask among the exact views consumed by the "
                "current method; this intentionally gives the single-view baseline "
                "a favorable deterministic view"
            ),
            "primary_metric_frame": (
                "fixed score-independent native-Pixal3D-to-Direct canonical "
                "transform; no per-object normalization, ICP, or best transform"
                if utility_mode
                else "same proper-rotation isotropic-similarity ICP for both methods"
            ),
            "scope": (
                "utility-track protocol; evaluate with "
                "evaluate_direct_slat_pixal3d_utility.py"
                if utility_mode
                else (
                    "shape-only smoke; not a pose-aware AR metric and not a formal "
                    "checkpoint-selection result"
                )
            ),
        },
        "bindings": {
            "current_report": binding(current_report_path),
            "direct_slat_cache_manifest": binding(direct_manifest_path),
            "lifting_manifest": binding(lifting_manifest_path),
            "source_cache_manifest": binding(source_cache_manifest_path),
            "source_manifest": binding(source_manifest_path),
            "prepare_code": binding(Path(__file__).resolve()),
        },
        "cases": cases,
    }
    body["protocol_sha256"] = canonical_sha256(body)
    atomic_json(protocol_path, body)
    print(
        json.dumps(
            {
                "status": "prepared",
                "protocol": str(protocol_path),
                "cases": len(cases),
                "view_counts": {
                    str(value): sum(case["view_count"] == value for case in cases)
                    for value in args.view_counts
                },
                "protocol_sha256": body["protocol_sha256"],
            },
            indent=2,
        )
    )


def pixal3d_mesh_path(protocol_path: Path, case_id: str) -> Path:
    return (
        protocol_path.parent
        / "pixal3d"
        / case_id
        / "mesh_official_postprocessed.glb"
    ).resolve()


def pixal3d_result_path(protocol_path: Path, case_id: str) -> Path:
    return (
        protocol_path.parent
        / "pixal3d"
        / case_id
        / "result_official_postprocessed.json"
    ).resolve()


def validate_official_inference_result(
    result: dict[str, Any],
    *,
    protocol: dict[str, Any],
    case: dict[str, Any],
    mesh_path: Path,
) -> None:
    if result.get("format") != INFERENCE_RESULT_FORMAT:
        raise RuntimeError(
            f"unsupported Pixal3D inference result for {case['case_id']}: "
            f"{result.get('format')!r}"
        )
    if result.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError(
            f"Pixal3D protocol mismatch for {case['case_id']}"
        )
    if Path(str(result.get("mesh", ""))).resolve() != mesh_path.resolve():
        raise RuntimeError(
            f"Pixal3D result path mismatch for {case['case_id']}"
        )
    if result.get("mesh_sha256") != sha256_file(mesh_path):
        raise RuntimeError(
            f"Pixal3D mesh SHA mismatch for {case['case_id']}"
        )
    if result.get("geometry_export") != OFFICIAL_GEOMETRY_EXPORT:
        raise RuntimeError(
            f"Pixal3D result is not the official final export for "
            f"{case['case_id']}"
        )
    if result.get("postprocess") != OFFICIAL_POSTPROCESS:
        raise RuntimeError(
            f"Pixal3D official postprocess changed for {case['case_id']}"
        )


def command_verify_naf(args: argparse.Namespace) -> None:
    import importlib.metadata
    import torch

    metadata = configure_local_naf(
        args.naf_repo,
        args.naf_checkpoint,
        args.naf_source_manifest,
    )
    model = torch.hub.load(
        metadata["repository"],
        "naf",
        pretrained=False,
        device="cpu",
        source="local",
    )
    state = torch.load(
        metadata["checkpoint"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    incompatible = model.load_state_dict(state, strict=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "naf": metadata,
                "einops_version": importlib.metadata.version("einops"),
                "state_key_count": len(state),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            },
            indent=2,
        )
    )


def command_infer(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image
    import torch
    import trimesh

    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(protocol_path)
    selected_cases = select_protocol_cases(protocol, args.case_ids)
    naf_binding = configure_local_naf(
        args.naf_repo,
        args.naf_checkpoint,
        args.naf_source_manifest,
    )
    print(
        "[NAF] Verified pinned local dependency "
        f"commit={naf_binding['commit']} repo={naf_binding['repository']}"
    )

    completed = 0
    pending_cases: list[dict[str, Any]] = []
    selected_positions = {
        str(case["case_id"]): position
        for position, case in enumerate(selected_cases, start=1)
    }
    for case in selected_cases:
        destination = pixal3d_mesh_path(protocol_path, case["case_id"])
        result_path = pixal3d_result_path(protocol_path, case["case_id"])
        if destination.is_file() and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            try:
                validate_official_inference_result(
                    result,
                    protocol=protocol,
                    case=case,
                    mesh_path=destination,
                )
            except Exception:
                raise RuntimeError(
                    "existing official Pixal3D output failed binding; "
                    f"preserve and inspect it, then use a fresh protocol root: "
                    f"{destination.parent}"
                )
            print(
                "[pixal3d_smoke] "
                f"{selected_positions[str(case['case_id'])]}/"
                f"{len(selected_cases)} {case['case_id']} "
                "reuse official final GLB before pipeline initialization"
            )
            completed += 1
            continue
        if destination.exists() or result_path.exists():
            raise RuntimeError(
                "partial official Pixal3D output exists; preserve and inspect "
                f"it, then use a fresh protocol root: {destination.parent}"
            )
        pending_cases.append(case)

    if not pending_cases:
        print(
            json.dumps(
                {
                    "status": "complete",
                    "protocol": str(protocol_path),
                    "completed": completed,
                    "total": len(selected_cases),
                    "protocol_total": len(protocol["cases"]),
                    "selected_case_ids": [
                        str(case["case_id"]) for case in selected_cases
                    ],
                    "pipeline_initialized": False,
                },
                indent=2,
            )
        )
        return

    if args.skip_rembg:
        for case in pending_cases:
            image_path = Path(case["input_rgba"]["path"])
            with Image.open(image_path) as candidate:
                if candidate.mode != "RGBA":
                    raise RuntimeError(
                        f"--skip_rembg requires RGBA input: "
                        f"{case['case_id']}={image_path}"
                    )
                alpha = np.asarray(candidate)[:, :, 3]
            if not np.any(alpha > int(0.8 * 255)) or not np.any(alpha < 255):
                raise RuntimeError(
                    f"--skip_rembg requires both foreground and transparent "
                    f"pixels: {case['case_id']}={image_path}"
                )
    if not PIXAL3D_ROOT.is_dir():
        raise FileNotFoundError(PIXAL3D_ROOT)
    sys.path.insert(0, str(PIXAL3D_ROOT))
    from inference import (  # type: ignore
        distance_from_fov,
        init_pipeline,
        o_voxel,
    )

    pipeline = init_pipeline(
        model_path=args.model_path,
        device=args.device,
        low_vram=bool(args.low_vram),
        load_rembg=not bool(args.skip_rembg),
    )
    model_snapshot = model_snapshot_identity(args.model_path)
    sampler_ss = {
        "steps": int(args.sampling_steps),
        "guidance_strength": float(args.ss_guidance),
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    sampler_shape = {
        "steps": int(args.sampling_steps),
        "guidance_strength": float(args.shape_guidance),
        "guidance_rescale": float(args.shape_guidance_rescale),
        "rescale_t": float(args.shape_rescale_t),
    }
    sampler_texture = {
        "steps": int(args.sampling_steps),
        "guidance_strength": float(args.texture_guidance),
        "guidance_rescale": float(args.texture_guidance_rescale),
        "rescale_t": float(args.texture_rescale_t),
    }
    official_rotation = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    for case in pending_cases:
        position = selected_positions[str(case["case_id"])]
        destination = pixal3d_mesh_path(protocol_path, case["case_id"])
        result_path = pixal3d_result_path(protocol_path, case["case_id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(case["input_rgba"]["path"]) as image:
            preprocessed = pipeline.preprocess_image(image)
        if args.camera_mode == "known_crop_fov":
            initial_resize = min(1.0, 1024.0 / max(case["source_width"], case["source_height"]))
            focal = float(case["source_fx_pixels"]) * initial_resize
            camera_angle_x = 2.0 * math.atan(float(preprocessed.width) / (2.0 * focal))
        else:
            camera_angle_x = float(args.fixed_fov)
        distance = distance_from_fov(
            camera_angle_x,
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([0.0, 511.0]),
            float(args.mesh_scale),
            512,
        )["distance_from_x"]
        camera_params = {
            "camera_angle_x": camera_angle_x,
            "distance": distance,
            "mesh_scale": float(args.mesh_scale),
        }
        seed = int(case["pixal3d_seed"])
        print(
            f"[pixal3d_smoke] {position}/{len(selected_cases)} "
            f"{case['case_id']} seed={seed} fov={camera_angle_x:.6f}"
        )
        torch.manual_seed(seed)
        mesh_list, latent = pipeline.run(
            preprocessed,
            camera_params=camera_params,
            seed=seed,
            sparse_structure_sampler_params=sampler_ss,
            shape_slat_sampler_params=sampler_shape,
            tex_slat_sampler_params=sampler_texture,
            preprocess_image=False,
            return_latent=True,
            pipeline_type=f"{int(args.resolution)}_cascade",
            max_num_tokens=int(args.max_num_tokens),
        )
        decoded = mesh_list[0]
        _, _, actual_resolution = latent
        decoded_vertices = decoded.vertices
        decoded_faces = decoded.faces
        decoded_vertex_count = int(len(decoded_vertices))
        decoded_face_count = int(len(decoded_faces))
        if torch.is_tensor(decoded_vertices):
            decoded_finite = bool(
                torch.isfinite(decoded_vertices).all().item()
            )
        else:
            decoded_finite = bool(
                np.isfinite(np.asarray(decoded_vertices)).all()
            )
        if not decoded_finite:
            raise RuntimeError(
                f"Pixal3D decoded vertices are non-finite: {case['case_id']}"
            )

        print(
            f"[pixal3d_smoke] {case['case_id']} official postprocess "
            f"remesh=True texture_size="
            f"{OFFICIAL_POSTPROCESS['texture_size']}"
        )
        with guarded_cumesh_fill_holes(o_voxel.postprocess.cumesh):
            official_mesh = o_voxel.postprocess.to_glb(
                vertices=decoded.vertices,
                faces=decoded.faces,
                attr_volume=decoded.attrs,
                coords=decoded.coords,
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=actual_resolution,
                aabb=OFFICIAL_POSTPROCESS["aabb"],
                decimation_target=OFFICIAL_POSTPROCESS["decimation_target"],
                texture_size=OFFICIAL_POSTPROCESS["texture_size"],
                remesh=OFFICIAL_POSTPROCESS["remesh"],
                remesh_band=OFFICIAL_POSTPROCESS["remesh_band"],
                remesh_project=OFFICIAL_POSTPROCESS["remesh_project"],
                use_tqdm=True,
            )
        official_mesh.apply_transform(official_rotation)
        temporary = destination.with_name(
            f".{destination.stem}.tmp-{os.getpid()}.glb"
        )
        try:
            official_mesh.export(str(temporary), extension_webp=True)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError(
                    f"Pixal3D official GLB export is empty: {temporary}"
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        final_mesh = load_mesh(destination)
        final_vertices = np.asarray(final_mesh.vertices, dtype=np.float64)
        final_geometry = {
            "vertex_count": int(len(final_mesh.vertices)),
            "face_count": int(len(final_mesh.faces)),
            "vertices_finite": bool(np.isfinite(final_vertices).all()),
            "bounds": np.asarray(
                final_mesh.bounds, dtype=np.float64
            ).tolist(),
            "extents": np.asarray(
                final_mesh.extents, dtype=np.float64
            ).tolist(),
            "watertight": bool(final_mesh.is_watertight),
            "winding_consistent": bool(final_mesh.is_winding_consistent),
            "component_count": int(final_mesh.body_count),
        }
        if not final_geometry["vertices_finite"]:
            raise RuntimeError(
                f"Pixal3D official GLB contains non-finite vertices: "
                f"{case['case_id']}"
            )
        result = {
            "format": INFERENCE_RESULT_FORMAT,
            "protocol_sha256": protocol["protocol_sha256"],
            "case_id": case["case_id"],
            "model_path": args.model_path,
            "model_snapshot": model_snapshot,
            "original_pixal3d_code": binding(PIXAL3D_ROOT / "inference.py"),
            "pixal3d_pipeline_code": binding(
                PIXAL3D_ROOT
                / "pixal3d"
                / "pipelines"
                / "pixal3d_image_to_3d.py"
            ),
            "naf_loader_code": binding(
                PIXAL3D_ROOT
                / "pixal3d"
                / "trainers"
                / "flow_matching"
                / "mixins"
                / "image_conditioned_proj.py"
            ),
            "naf": naf_binding,
            "background_removal": (
                "skipped; frozen RGBA alpha used"
                if args.skip_rembg
                else "Pixal3D configured rembg model"
            ),
            "seed": seed,
            "camera_mode": args.camera_mode,
            "camera_params": camera_params,
            "preprocessed_size": list(preprocessed.size),
            "pipeline_type": f"{int(args.resolution)}_cascade",
            "sampling_steps": int(args.sampling_steps),
            "geometry_export": OFFICIAL_GEOMETRY_EXPORT,
            "postprocess": OFFICIAL_POSTPROCESS,
            "postprocess_code": binding(Path(o_voxel.postprocess.__file__)),
            "cumesh_fill_holes_guard": CUMESH_FILL_HOLES_GUARD_VERSION,
            "decoded_geometry": {
                "vertex_count": decoded_vertex_count,
                "face_count": decoded_face_count,
                "vertices_finite": decoded_finite,
                "saved": False,
            },
            "final_geometry": final_geometry,
            "vertex_count": final_geometry["vertex_count"],
            "face_count": final_geometry["face_count"],
            "mesh": str(destination),
            "mesh_sha256": sha256_file(destination),
        }
        atomic_json(result_path, result)
        completed += 1
        del final_mesh, official_mesh, decoded, mesh_list, latent, preprocessed
        gc.collect()
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "complete",
                "protocol": str(protocol_path),
                "completed": completed,
                "total": len(selected_cases),
                "protocol_total": len(protocol["cases"]),
                "selected_case_ids": [
                    str(case["case_id"]) for case in selected_cases
                ],
                "pipeline_initialized": True,
            },
            indent=2,
        )
    )


def deterministic_surface_sample(mesh, count: int, seed: int):
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1.0e-15)
    if not np.any(valid):
        raise ValueError("mesh contains no finite non-degenerate triangles")
    valid_ids = np.flatnonzero(valid)
    probability = double_area[valid] / double_area[valid].sum()
    rng = np.random.default_rng(int(seed))
    face_ids = rng.choice(valid_ids, size=int(count), replace=True, p=probability)
    u = rng.random(int(count))
    v = rng.random(int(count))
    sqrt_u = np.sqrt(u)
    weights = np.stack(
        (1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v), axis=1
    )
    points = np.sum(triangles[face_ids] * weights[:, :, None], axis=1)
    normals = cross[face_ids]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15)
    return points, normals


def proper_cube_rotations():
    import numpy as np

    rotations = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[list(permutation)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.diag(signs) @ base
            if np.linalg.det(rotation) > 0.5:
                rotations.append(rotation)
    if len(rotations) != 24:
        raise RuntimeError(f"expected 24 proper cube rotations, got {len(rotations)}")
    return rotations


def similarity_icp(
    source,
    target,
    *,
    seed: int,
    candidate_samples: int,
    final_samples: int,
    candidate_iterations: int,
    final_iterations: int,
):
    import numpy as np
    import trimesh

    source_candidate, _ = deterministic_surface_sample(
        source, candidate_samples, seed
    )
    target_candidate, _ = deterministic_surface_sample(
        target, candidate_samples * 2, seed + 1
    )
    source_min, source_max = source_candidate.min(axis=0), source_candidate.max(axis=0)
    target_min, target_max = target_candidate.min(axis=0), target_candidate.max(axis=0)
    source_center = 0.5 * (source_min + source_max)
    target_center = 0.5 * (target_min + target_max)
    source_diag = float(np.linalg.norm(source_max - source_min))
    target_diag = float(np.linalg.norm(target_max - target_min))
    if source_diag <= 1.0e-12 or target_diag <= 1.0e-12:
        raise ValueError("cannot align degenerate mesh bounds")
    initial_scale = target_diag / source_diag
    candidates = []
    for rotation in proper_cube_rotations():
        initial = np.eye(4, dtype=np.float64)
        initial[:3, :3] = initial_scale * rotation
        initial[:3, 3] = target_center - initial[:3, :3] @ source_center
        matrix, _, cost = trimesh.registration.icp(
            source_candidate,
            target_candidate,
            initial=initial,
            max_iterations=int(candidate_iterations),
            reflection=False,
            translation=True,
            scale=True,
        )
        candidates.append((float(cost), matrix))
    _, best = min(candidates, key=lambda item: item[0])
    source_final, _ = deterministic_surface_sample(source, final_samples, seed + 2)
    target_final, _ = deterministic_surface_sample(
        target, final_samples * 2, seed + 3
    )
    matrix, _, cost = trimesh.registration.icp(
        source_final,
        target_final,
        initial=best,
        max_iterations=int(final_iterations),
        reflection=False,
        translation=True,
        scale=True,
    )
    linear = matrix[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    if determinant <= 0:
        raise RuntimeError("similarity ICP produced a reflected transform")
    aligned = source.copy()
    aligned.apply_transform(matrix)
    return aligned, {
        "matrix": matrix.tolist(),
        "cost": float(cost),
        "determinant": determinant,
        "singular_values": singular.tolist(),
        "anisotropy_ratio": float(singular.max() / max(singular.min(), 1.0e-15)),
        "proper_rotation_only": True,
        "isotropic_scale": True,
    }


def surface_metrics(
    predicted,
    target,
    *,
    count: int,
    seed: int,
    thresholds: Iterable[float],
) -> dict[str, float]:
    import numpy as np
    from scipy.spatial import cKDTree

    pred_points, pred_normals = deterministic_surface_sample(predicted, count, seed)
    target_points, target_normals = deterministic_surface_sample(target, count, seed)
    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)
    pred_distance, pred_index = target_tree.query(pred_points, k=1, workers=-1)
    target_distance, target_index = pred_tree.query(target_points, k=1, workers=-1)
    output = {
        "pred_to_gt_mean": float(np.mean(pred_distance)),
        "gt_to_pred_mean": float(np.mean(target_distance)),
        "chamfer_l1": float(
            0.5 * (np.mean(pred_distance) + np.mean(target_distance))
        ),
        "chamfer_l2": float(
            0.5 * (np.mean(pred_distance**2) + np.mean(target_distance**2))
        ),
        "normal_consistency": float(
            0.5
            * (
                np.mean(
                    np.abs(np.sum(pred_normals * target_normals[pred_index], axis=1))
                )
                + np.mean(
                    np.abs(
                        np.sum(target_normals * pred_normals[target_index], axis=1)
                    )
                )
            )
        ),
    }
    for threshold in thresholds:
        key = str(float(threshold)).replace(".", "p")
        precision = float(np.mean(pred_distance < float(threshold)))
        recall = float(np.mean(target_distance < float(threshold)))
        output[f"precision_{key}"] = precision
        output[f"recall_{key}"] = recall
        output[f"fscore_{key}"] = (
            0.0
            if precision + recall <= 1.0e-12
            else float(2.0 * precision * recall / (precision + recall))
        )
    return output


def numeric_summary(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"case_count": len(records)}
    for method in ("current", "pixal3d"):
        output[method] = {}
        for metric in ("chamfer_l1", "fscore_0p02", "normal_consistency"):
            output[method][metric] = numeric_summary(
                [float(row[method]["surface"][metric]) for row in records]
            )
    output["paired"] = {
        "current_chamfer_improvement": numeric_summary(
            [float(row["paired"]["chamfer_improvement"]) for row in records]
        ),
        "current_fscore_delta": numeric_summary(
            [float(row["paired"]["fscore_0p02_delta"]) for row in records]
        ),
        "current_chamfer_win_rate": sum(
            float(row["paired"]["chamfer_improvement"]) > 0 for row in records
        )
        / len(records),
        "current_fscore_win_rate": sum(
            float(row["paired"]["fscore_0p02_delta"]) > 0 for row in records
        )
        / len(records),
    }
    return output


def command_evaluate(args: argparse.Namespace) -> None:
    import numpy as np

    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(protocol_path)
    records = []
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else protocol_path.parent
    )
    if args.output_dir is not None:
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite evaluation output: {output_dir}"
            )
        output_dir.mkdir(parents=True)
    aligned_root = output_dir / "aligned_pixal3d"
    for position, case in enumerate(protocol["cases"], start=1):
        pixal_path = pixal3d_mesh_path(protocol_path, case["case_id"])
        result_path = pixal3d_result_path(protocol_path, case["case_id"])
        if not pixal_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(
                f"missing Pixal3D result for {case['case_id']}: run infer first"
            )
        pixal_result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_official_inference_result(
            pixal_result,
            protocol=protocol,
            case=case,
            mesh_path=pixal_path,
        )
        print(
            f"[pixal3d_smoke_eval] {position}/{len(protocol['cases'])} "
            f"{case['case_id']}"
        )
        target = load_mesh(case["target_mesh"]["path"])
        current = load_mesh(case["current_mesh"]["path"])
        pixal = load_mesh(pixal_path)
        current_aligned, current_alignment = similarity_icp(
            current,
            target,
            seed=int(args.seed) + position * 100,
            candidate_samples=int(args.candidate_samples),
            final_samples=int(args.alignment_samples),
            candidate_iterations=int(args.candidate_iterations),
            final_iterations=int(args.final_iterations),
        )
        pixal_aligned, pixal_alignment = similarity_icp(
            pixal,
            target,
            seed=int(args.seed) + position * 100 + 10,
            candidate_samples=int(args.candidate_samples),
            final_samples=int(args.alignment_samples),
            candidate_iterations=int(args.candidate_iterations),
            final_iterations=int(args.final_iterations),
        )
        current_surface = surface_metrics(
            current_aligned,
            target,
            count=int(args.surface_samples),
            seed=int(args.seed) + position,
            thresholds=(0.01, 0.02, 0.05),
        )
        pixal_surface = surface_metrics(
            pixal_aligned,
            target,
            count=int(args.surface_samples),
            seed=int(args.seed) + position,
            thresholds=(0.01, 0.02, 0.05),
        )
        aligned_pixal_path = aligned_root / case["case_id"] / "mesh.obj"
        aligned_pixal_path.parent.mkdir(parents=True, exist_ok=True)
        pixal_aligned.export(aligned_pixal_path)
        row = {
            "case_id": case["case_id"],
            "uid": case["uid"],
            "object_uid": case["object_uid"],
            "view_count": int(case["view_count"]),
            "current": {
                "mesh": case["current_mesh"],
                "surface": current_surface,
                "alignment": current_alignment,
                "canonical_surface_from_frozen_report": case[
                    "current_canonical_surface_from_frozen_report"
                ],
            },
            "pixal3d": {
                "mesh": binding(pixal_path),
                "aligned_mesh": binding(aligned_pixal_path),
                "surface": pixal_surface,
                "alignment": pixal_alignment,
                "inference": pixal_result,
            },
            "target_mesh": case["target_mesh"],
            "paired": {
                "chamfer_improvement": float(
                    pixal_surface["chamfer_l1"] - current_surface["chamfer_l1"]
                ),
                "fscore_0p02_delta": float(
                    current_surface["fscore_0p02"] - pixal_surface["fscore_0p02"]
                ),
                "normal_consistency_delta": float(
                    current_surface["normal_consistency"]
                    - pixal_surface["normal_consistency"]
                ),
            },
        }
        records.append(row)

    by_view_count = {
        str(views): summarize_records(
            [row for row in records if int(row["view_count"]) == int(views)]
        )
        for views in sorted({int(row["view_count"]) for row in records})
    }
    report = {
        "format": REPORT_FORMAT,
        "formal": False,
        "passed": True,
        "protocol": binding(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation": {
            "surface_samples": int(args.surface_samples),
            "alignment": (
                "24 proper cube-rotation initializations followed by isotropic "
                "similarity ICP; reflection forbidden; applied identically to both methods"
            ),
            "primary_interpretation": (
                "shape quality after removing global rigid pose and isotropic scale"
            ),
        },
        "summary": summarize_records(records),
        "by_view_count": by_view_count,
        "records": records,
        "guardrail": (
            "Exploratory smoke only: few Objaverse objects, reused current outputs, "
            "different input budgets by design, and similarity-aligned metrics. "
            "Do not use this report as a formal checkpoint-selection gate."
        ),
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)

    csv_path = output_dir / "metrics.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "uid",
                "object_uid",
                "current_view_count",
                "current_chamfer_l1",
                "pixal3d_chamfer_l1",
                "current_chamfer_improvement",
                "current_fscore_0p02",
                "pixal3d_fscore_0p02",
                "current_fscore_0p02_delta",
                "current_normal_consistency",
                "pixal3d_normal_consistency",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "uid": row["uid"],
                    "object_uid": row["object_uid"],
                    "current_view_count": row["view_count"],
                    "current_chamfer_l1": row["current"]["surface"]["chamfer_l1"],
                    "pixal3d_chamfer_l1": row["pixal3d"]["surface"]["chamfer_l1"],
                    "current_chamfer_improvement": row["paired"][
                        "chamfer_improvement"
                    ],
                    "current_fscore_0p02": row["current"]["surface"][
                        "fscore_0p02"
                    ],
                    "pixal3d_fscore_0p02": row["pixal3d"]["surface"][
                        "fscore_0p02"
                    ],
                    "current_fscore_0p02_delta": row["paired"][
                        "fscore_0p02_delta"
                    ],
                    "current_normal_consistency": row["current"]["surface"][
                        "normal_consistency"
                    ],
                    "pixal3d_normal_consistency": row["pixal3d"]["surface"][
                        "normal_consistency"
                    ],
                }
            )
    os.replace(temporary_csv, csv_path)

    summary = report["summary"]
    lines = [
        "Original single-view Pixal3D vs current multi-view smoke",
        "========================================================",
        "",
        "FORMAL: false",
        f"cases: {len(records)}",
        "primary frame: proper-rotation isotropic-similarity aligned shape",
        "",
        "[all cases]",
        (
            "chamfer_l1 mean: "
            f"current={summary['current']['chamfer_l1']['mean']:.8f} "
            f"pixal3d={summary['pixal3d']['chamfer_l1']['mean']:.8f} "
            f"current_win={summary['paired']['current_chamfer_win_rate']:.3f}"
        ),
        (
            "fscore_0p02 mean: "
            f"current={summary['current']['fscore_0p02']['mean']:.8f} "
            f"pixal3d={summary['pixal3d']['fscore_0p02']['mean']:.8f} "
            f"current_win={summary['paired']['current_fscore_win_rate']:.3f}"
        ),
        (
            "normal_consistency mean: "
            f"current={summary['current']['normal_consistency']['mean']:.8f} "
            f"pixal3d={summary['pixal3d']['normal_consistency']['mean']:.8f}"
        ),
        "",
    ]
    for views, subset in by_view_count.items():
        lines.extend(
            [
                f"[current input views={views}]",
                (
                    "chamfer_l1 mean: "
                    f"current={subset['current']['chamfer_l1']['mean']:.8f} "
                    f"pixal3d={subset['pixal3d']['chamfer_l1']['mean']:.8f}"
                ),
                (
                    "fscore_0p02 mean: "
                    f"current={subset['current']['fscore_0p02']['mean']:.8f} "
                    f"pixal3d={subset['pixal3d']['fscore_0p02']['mean']:.8f}"
                ),
                "",
            ]
        )
    lines.extend(["Guardrail:", report["guardrail"], ""])
    summary_path = output_dir / "summary.txt"
    atomic_text(summary_path, "\n".join(lines))
    print("\n".join(lines))
    print(f"report: {report_path}")
    print(f"csv: {csv_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_naf = subparsers.add_parser(
        "verify-naf", help="verify and CPU-load the pinned local NAF dependency"
    )
    verify_naf.add_argument("--naf_repo", type=Path, required=True)
    verify_naf.add_argument("--naf_checkpoint", type=Path, required=True)
    verify_naf.add_argument("--naf_source_manifest", type=Path, required=True)
    verify_naf.set_defaults(handler=command_verify_naf)

    prepare = subparsers.add_parser("prepare", help="freeze cases and create RGBA inputs")
    prepare.add_argument("--current_report", type=Path, required=True)
    prepare.add_argument("--cache_manifest", type=Path, required=True)
    prepare.add_argument("--output_dir", type=Path, required=True)
    prepare.add_argument("--view_counts", type=int, nargs="+", default=[2, 4, 8])
    prepare.add_argument("--objects_per_view_count", type=int, default=2)
    prepare.add_argument(
        "--selection_mode",
        choices=("view_stratified_v1", "report_objects_v1"),
        default="view_stratified_v1",
    )
    prepare.add_argument(
        "--joint_seeds",
        default="42",
        help=(
            "Comma-separated current/Pixal3D seed IDs in report_objects_v1 mode."
        ),
    )
    prepare.add_argument(
        "--uids",
        default="",
        help="Optional exact sequence UIDs in report_objects_v1 mode.",
    )
    prepare.add_argument(
        "--max_objects",
        type=int,
        default=0,
        help="Optional object cap in report_objects_v1 mode.",
    )
    prepare.add_argument("--current_seed", type=int, default=42)
    prepare.add_argument("--pixal3d_seed", type=int, default=42)
    prepare.add_argument("--selection_seed", type=int, default=20260727)
    prepare.add_argument("--fallback_fx", type=float, default=486.4)
    prepare.set_defaults(handler=command_prepare)

    infer = subparsers.add_parser("infer", help="run original Pixal3D once per case")
    infer.add_argument("--protocol", type=Path, required=True)
    infer.add_argument(
        "--case_ids",
        default="",
        help=(
            "Optional comma-separated exact frozen case IDs. This selector "
            "does not alter protocol bindings and enables one-process-per-case "
            "execution for deterministic CUDA-memory isolation."
        ),
    )
    infer.add_argument("--model_path", default="TencentARC/Pixal3D")
    infer.add_argument(
        "--naf_repo",
        type=Path,
        required=True,
        help=f"local valeoai/NAF repository at pinned commit {NAF_COMMIT}",
    )
    infer.add_argument(
        "--naf_checkpoint",
        type=Path,
        required=True,
        help="local naf_release.pth with the frozen SHA-256",
    )
    infer.add_argument(
        "--naf_source_manifest",
        type=Path,
        required=True,
        help="frozen per-file SHA-256 manifest for the local NAF source tree",
    )
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--low_vram", action="store_true")
    infer.add_argument(
        "--skip_rembg",
        action="store_true",
        help=(
            "do not load the optional gated background-removal model; every "
            "frozen input must be RGBA with foreground and transparent pixels"
        ),
    )
    infer.add_argument(
        "--camera_mode",
        choices=("known_crop_fov", "fixed_fov"),
        default="known_crop_fov",
    )
    infer.add_argument("--fixed_fov", type=float, default=0.857556)
    infer.add_argument("--mesh_scale", type=float, default=1.0)
    infer.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    infer.add_argument("--max_num_tokens", type=int, default=49152)
    infer.add_argument("--sampling_steps", type=int, default=12)
    infer.add_argument("--ss_guidance", type=float, default=7.5)
    infer.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    infer.add_argument("--ss_rescale_t", type=float, default=5.0)
    infer.add_argument("--shape_guidance", type=float, default=7.5)
    infer.add_argument("--shape_guidance_rescale", type=float, default=0.5)
    infer.add_argument("--shape_rescale_t", type=float, default=3.0)
    infer.add_argument("--texture_guidance", type=float, default=1.0)
    infer.add_argument("--texture_guidance_rescale", type=float, default=0.0)
    infer.add_argument("--texture_rescale_t", type=float, default=3.0)
    infer.set_defaults(handler=command_infer)

    evaluate = subparsers.add_parser(
        "evaluate", help="align both methods and compute paired shape metrics"
    )
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=20260727)
    evaluate.add_argument("--candidate_samples", type=int, default=1000)
    evaluate.add_argument("--alignment_samples", type=int, default=4000)
    evaluate.add_argument("--candidate_iterations", type=int, default=8)
    evaluate.add_argument("--final_iterations", type=int, default=30)
    evaluate.add_argument("--surface_samples", type=int, default=20000)
    evaluate.add_argument(
        "--output_dir",
        type=Path,
        help=(
            "fresh directory for metrics/aligned meshes; use this when "
            "preserving an earlier decoded-geometry report"
        ),
    )
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
