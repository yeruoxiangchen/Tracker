#!/usr/bin/env python3
"""Replay one calibrated Omni Holdout object through the phone capture contract.

The tool is deliberately input-only.  It converts calibrated COLMAP cameras to
the Unity pose convention consumed by ``collect_ar_object_server.py``, writes
lossless RGB frames using the phone ``frame_XXXX.jpg`` naming contract, and
records an explicit pose round-trip audit.  No GT mesh is used to form poses,
masks, runtime-O, or posed-DINO inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
from PIL import Image

from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    evenly_spaced_frame_indices,
)
from pose_point_depth_mv.real_object_canonicalization import (
    undistort_rgb_mask_views,
)
from trellis_point_prior_mv.build_ar_session_smoke_dataset import (
    intrinsics_for_pose,
    read_phone_poses,
    rotmat_to_qvec,
    unity_pose_to_colmap_w2c,
)


FORMAT = "pose_point_depth_mv.omni_holdout_ar_capture_replay.v1"
DEFAULT_ROOT = Path(
    "/data/zjr/omni_real_video500_download_20260804_v2"
)
DEFAULT_RAW = DEFAULT_ROOT / "M11B_holdout64_raw_cache_v1/raw_cache_report.json"
DEFAULT_REFERENCE_RUNTIME = (
    DEFAULT_ROOT
    / "M11N_holdout64_pose_mask_runtime_o_blind_v1/runtime_input_manifest.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs/可视AR/OmniHoldout64复杂样本真实采集流程回放_20260811_v1"
)


def _object_key(row: dict[str, Any]) -> str:
    return str(row.get("object_key") or f"{row['category']}:{row['object_id']}")


def _find_object(report: dict[str, Any], key: str) -> dict[str, Any]:
    matches = [row for row in report.get("objects", []) if _object_key(row) == key]
    if len(matches) != 1:
        raise RuntimeError(f"object selector {key!r} matched {len(matches)} rows")
    return matches[0]


def _align_vector_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a proper rotation Q with Q @ source == target."""

    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if cosine > 1.0 - 1.0e-12:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1.0e-12:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        axis = np.cross(first, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    skew = np.asarray(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def colmap_w2c_to_unity_pose(T_W2C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert the repository's Unity-to-COLMAP camera conversion at Rz(0)."""

    transform = np.asarray(T_W2C, dtype=np.float64)
    rotation_w2c = transform[:3, :3]
    translation_w2c = transform[:3, 3]
    center_cv_world = -rotation_w2c.T @ translation_w2c
    cv_c2w = rotation_w2c.T
    unity_to_cv_world = np.diag([1.0, 1.0, -1.0])
    unity_camera_to_cv_camera = np.diag([1.0, -1.0, 1.0])
    unity_position = unity_to_cv_world @ center_cv_world
    unity_c2w = unity_to_cv_world @ cv_c2w @ unity_camera_to_cv_camera
    q_wxyz = rotmat_to_qvec(unity_c2w)
    quaternion_xyzw = np.asarray(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64
    )
    return unity_position, quaternion_xyzw


def _selected_source_names(
    *, mode: str, count: int, raw_names: Sequence[str], reference: dict[str, Any]
) -> list[str]:
    if mode == "reference8":
        names = [str(value) for value in reference["selected_frame_names"]]
        if len(names) != 8:
            raise RuntimeError("reference runtime does not bind exactly eight views")
        return names
    indices = evenly_spaced_frame_indices(raw_names, int(count))
    return [str(raw_names[int(index)]) for index in indices]


def _mesh_sources(root: Path, object_key: str) -> dict[str, Path]:
    category, object_id = object_key.split(":", 1)
    return {
        "00_GT_runtime_O.obj": root / f"M11E_holdout64_mesh_o_labels_v1/objects/{category}/{object_id}/Scan_in_runtime_O.obj",
        "01_当前Mixed_NoVGGT_PointMask.obj": root / f"M11H_holdout64_native_no_vggt_mixed1244_seed42_v1/meshes/{category}/{object_id}/seed_42/mesh_o.obj",
        "02_当前Mixed_NoVGGT_PoseMask.obj": root / f"M11P_holdout64_pose_mask_no_vggt_mixed1244_seed42_blind_v1/meshes/{category}/{object_id}/seed_42/mesh_o.obj",
        "03_当前PoseMask重表达到ReferenceO.obj": root / f"M11Q_holdout64_pose_mask_reference_o_seed42_blind_v1/meshes/{category}/{object_id}/seed_42/mesh_reference_o.obj",
        "04_NativeV2_RealAdapt_Full.obj": root / f"M11I_holdout64_native_v2_realadapt_step1000_seed42_v1/meshes/{category}/{object_id}/seed_42/mesh_o.obj",
        "05_NativeV2_Parent.obj": root / f"M11J_holdout64_native_v2_parent_seed42_v1/meshes/{category}/{object_id}/seed_42/mesh_o.obj",
        "06_ReconViaGen.obj": root / f"M11K_holdout64_reconviagen_original_seed42_v1/meshes/{category}/{object_id}/seed_42/mesh_reference_o.obj",
        "07_Pixal3D.glb": root / f"M11L_holdout64_pixal3d_official_seed42_v1/pixal3d/{category}/{object_id}/seed_42/mesh_official_postprocessed.glb",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    raw_report_path = args.raw_cache_report.expanduser().resolve()
    reference_path = args.reference_runtime_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    reference_manifest = json.loads(reference_path.read_text(encoding="utf-8"))
    raw = _find_object(raw_report, args.object)
    reference = _find_object(reference_manifest, args.object)

    session_id = args.session_id or (
        f"omni_{raw['object_id']}_{args.mode}_{args.frame_count}_v1"
        if args.mode != "reference8"
        else f"omni_{raw['object_id']}_reference8_v1"
    )
    data_dir = output_root / "runtime/data" / session_id
    mask_dir = output_root / "runtime/masks" / session_id
    preview_dir = output_root / "runtime/previews" / session_id
    report_path = output_root / "replay_inputs" / session_id / "replay_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed") is True:
            return report
    if any(path.exists() for path in (data_dir, mask_dir, preview_dir)):
        raise RuntimeError(
            f"partial replay exists; preserve and inspect before retry: {session_id}"
        )
    data_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True)

    camera_by_name = {
        str(row["frame_name"]): row for row in raw.get("cameras", [])
    }
    source_cache = Path(raw["cache_npz"])
    with np.load(source_cache, allow_pickle=False) as source:
        raw_names = [str(value) for value in source["frame_name"].tolist()]
        k_by_name = {
            name: np.asarray(source["K"][index], dtype=np.float64)
            for index, name in enumerate(raw_names)
        }
        pose_by_name = {
            name: np.asarray(source["T_W2C"][index], dtype=np.float64)
            for index, name in enumerate(raw_names)
        }

    source_names = _selected_source_names(
        mode=args.mode,
        count=args.frame_count,
        raw_names=raw_names,
        reference=reference,
    )
    # A phone world supplies *physical gravity*, which is the anchor that the
    # runtime frame code blends once with the camera-orbit normal.  Do not use
    # reference_cache.T_O2W[:, 1] here: that axis has already been blended with
    # the orbit normal, and feeding it back as gravity would blend the same
    # evidence twice.  The calibrated capture has no IMU, so the observable
    # equivalent is the selected cameras' mean physical image-up direction --
    # exactly the anchor used by the original Holdout pose-mask runtime.
    selected_c2w = np.stack(
        [np.linalg.inv(pose_by_name[name]) for name in source_names], axis=0
    )
    source_up_w = np.mean(-selected_c2w[:, :3, 1], axis=0)
    source_up_w /= max(float(np.linalg.norm(source_up_w)), 1.0e-12)
    source_to_replay_rotation = _align_vector_rotation(
        source_up_w, np.asarray([0.0, 1.0, 0.0])
    )
    replay_to_source = np.eye(4, dtype=np.float64)
    replay_to_source[:3, :3] = source_to_replay_rotation.T

    image_root = Path(raw["images_dir"])
    source_mask_root = Path(raw["masks_dir"])
    pose_lines: list[str] = []
    metadata_lines: list[str] = []
    mappings: list[dict[str, Any]] = []
    expected_w2c: dict[str, np.ndarray] = {}
    expected_k: dict[str, np.ndarray] = {}
    identity_matrix = " ".join(str(value) for value in np.eye(4).reshape(-1))

    for index, source_name in enumerate(source_names):
        destination_name = f"frame_{index:04d}.jpg"
        camera = camera_by_name[source_name]
        K = k_by_name[source_name]
        raw_image = np.asarray(Image.open(image_root / source_name).convert("RGB"))
        raw_mask = np.asarray(Image.open(source_mask_root / source_name).convert("L"))
        images, masks, _k, undistortion = undistort_rgb_mask_views(
            [raw_image],
            [raw_mask],
            K[None],
            camera_models=[str(camera["model"])],
            distortion_coefficients=[camera.get("distortion", [])],
        )
        image = images[0]
        mask = masks[0]
        height, width = image.shape[:2]
        # The .jpg name matches the phone protocol; a PNG payload avoids adding a
        # second lossy encode during the numerical parity audit.
        Image.fromarray(image).save(data_dir / destination_name, format="PNG")
        Image.fromarray(mask).save(mask_dir / f"frame_{index:04d}.png")
        rgba = np.concatenate([image, mask[..., None]], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(preview_dir / f"{index}.png")

        replay_w2c = pose_by_name[source_name] @ replay_to_source
        position, quaternion = colmap_w2c_to_unity_pose(replay_w2c)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        timestamp = 1000.0 + index * 0.5
        fields = [
            destination_name,
            *[f"{value:.12g}" for value in position],
            "0", "0", "0",
            *[f"{value:.12g}" for value in quaternion],
            f"{fx:.12g}", f"{fy:.12g}", f"{cx:.12g}", f"{cy:.12g}",
            str(width), str(height), str(width), str(height), str(width), str(height),
            "None",
            f"{timestamp:.9f}", str(int(round(timestamp * 1.0e9))), f"{timestamp:.9f}", "0",
            "camera_frame_received", "LandscapeLeft", "SessionTracking",
            identity_matrix, identity_matrix,
        ]
        pose_lines.append(",".join(fields))
        metadata_lines.append(
            json.dumps(
                {
                    "schema": "omni_holdout_ar_replay_frame_metadata_v1",
                    "frame_name": destination_name,
                    "frame_index": index,
                    "source_frame_name": source_name,
                    "cpu_image_timestamp_s": timestamp,
                    "camera_frame_timestamp_ns": int(round(timestamp * 1.0e9)),
                    "pose_sample_realtime_s": timestamp,
                    "camera_frame_timestamp_delta_s": 0.0,
                    "pose_binding": "camera_frame_received",
                    "screen_orientation": "LandscapeLeft",
                    "tracking_state": "SessionTracking",
                    "image_transform": "None",
                    "display_matrix": identity_matrix,
                    "projection_matrix": identity_matrix,
                    "uploaded_image_size": [width, height],
                    "cpu_image_size": [width, height],
                    "intrinsics_resolution": [width, height],
                },
                ensure_ascii=False,
            )
        )
        mappings.append(
            {
                "runtime_frame_name": destination_name,
                "source_frame_name": source_name,
                "source_camera_model": camera["model"],
                "source_distortion": camera.get("distortion", []),
                "undistortion": undistortion[0],
            }
        )
        expected_w2c[destination_name] = replay_w2c
        expected_k[destination_name] = K

    (data_dir / "poses.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    (data_dir / "frame_metadata.jsonl").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8"
    )

    parsed = read_phone_poses(data_dir / "poses.txt")
    pose_errors = []
    intrinsic_errors = []
    for name in expected_w2c:
        rotation, translation = unity_pose_to_colmap_w2c(
            parsed[name], image_camera_rotation_degrees=0.0
        )
        actual = np.eye(4, dtype=np.float64)
        actual[:3, :3] = rotation
        actual[:3, 3] = translation
        pose_errors.append(float(np.max(np.abs(actual - expected_w2c[name]))))
        width, height = Image.open(data_dir / name).size
        fx, fy, cx, cy, _source = intrinsics_for_pose(parsed[name], width, height)
        actual_k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        intrinsic_errors.append(float(np.max(np.abs(actual_k - expected_k[name]))))

    comparison_dir = output_root / "模型Mesh对比"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    copied_meshes = {}
    for destination_name, source in _mesh_sources(DEFAULT_ROOT, args.object).items():
        if not source.is_file():
            continue
        destination = comparison_dir / destination_name
        shutil.copy2(source, destination)
        copied_meshes[destination_name] = {
            "source": str(source.resolve()),
            "copy": str(destination.resolve()),
        }

    report = {
        "format": FORMAT,
        "passed": max(pose_errors) <= 1.0e-9 and max(intrinsic_errors) <= 1.0e-8,
        "object_key": args.object,
        "selection_reason": (
            "thin layered leaves, self-occlusion and asymmetric geometry; 200 "
            "registered cameras and clean masks make it complex but auditable"
        ),
        "session_id": session_id,
        "mode": args.mode,
        "frame_count": len(source_names),
        "source_frame_names": source_names,
        "source_to_replay_world_rotation": source_to_replay_rotation.tolist(),
        "gravity_proxy": "selected_camera_mean_image_up_before_orbit_blending",
        "gravity_up_replay_W": [0.0, 1.0, 0.0],
        "data_dir": str(data_dir.resolve()),
        "mask_dir": str(mask_dir.resolve()),
        "preview_dir": str(preview_dir.resolve()),
        "frame_mapping": mappings,
        "phone_roundtrip": {
            "T_W2C_max_abs": max(pose_errors),
            "K_max_abs": max(intrinsic_errors),
            "image_camera_rotation_degrees": 0.0,
            "passed": max(pose_errors) <= 1.0e-9 and max(intrinsic_errors) <= 1.0e-8,
        },
        "copied_meshes": copied_meshes,
        "scope_guard": (
            "Input-only Omni-to-phone replay. GT and comparison meshes are copied "
            "after input materialization and are never consumed by runtime-O or DINO."
        ),
    }
    _write_json(report_path, report)
    _write_json(output_root / "模型Mesh对比/来源清单.json", copied_meshes)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw_cache_report", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--reference_runtime_manifest", type=Path, default=DEFAULT_REFERENCE_RUNTIME
    )
    parser.add_argument("--object", default="plant:plant_012")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("uniform", "reference8"), default="uniform")
    parser.add_argument("--frame_count", type=int, default=64)
    parser.add_argument("--session_id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "uniform" and args.frame_count < 8:
        raise ValueError("uniform replay requires at least eight frames")
    prepare(args)


if __name__ == "__main__":
    main()
