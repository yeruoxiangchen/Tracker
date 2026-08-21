#!/usr/bin/env python3
"""Render Reference/official-final Pixal3D/ReconViaGen/Direct-SLAT review."""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

from .bunny_review.common import (
    atomic_json,
    atomic_text,
    binding,
    canonical_sha256,
    code_bindings,
    sha256_file,
    validate_binding,
)
from .bunny_review.finalize import (
    comparison_contact_sheet,
    comparison_frames,
    contact_sheet,
    display_transform_from_mesh,
    load_mesh,
    mesh_stats,
    render_method,
    save_video_atomic,
)
from .compare_pixal3d_singleview_smoke import (
    deterministic_surface_sample,
    pixal3d_mesh_path,
    proper_cube_rotations,
    similarity_icp,
    validate_protocol,
)
from .evaluate_direct_slat_pixal3d_utility import (
    case_canonical_transform,
    load_pixal_result,
    validate_transform,
)


FORMAT = "pose_point_depth_mv.direct_slat_fourway_normal_review.v5"
METHOD_SELECTORS = (
    "reference",
    "pixal3d_native",
    "reconviagen_stock",
    "direct",
)
PIXAL_POSE_POLICIES = (
    "metadata",
    "reference_rigid_icp",
)
DIRECT_REPORT_FORMATS = frozenset(
    {
        "pose_point_depth_mv.direct_slat_mesh_exploratory.v1",
        "pose_point_depth_mv.direct_slat_mesh_exploratory.v2",
    }
)
RECON_REPORT_FORMAT = "pose_point_depth_mv.reconviagen_stock_from_direct_cache.v1"

# ReconViaGen/TRELLIS MeshExtractResult.to_trimesh(transform_pose=True) applies
# row-vector ``vertices @ [[1,0,0],[0,0,-1],[0,1,0]]``.  In the column-vector
# convention used by trimesh.apply_transform this is (x, y, z) -> (x, z, -y).
# Decoder ``mesh_canonical.obj`` files were exported with transform_pose=False,
# while the normalized source GLB Reference already uses the view/source frame.
LATENT_DECODER_TO_REFERENCE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def direct_record(
    report: dict[str, Any],
    *,
    uid: str,
    seed: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in report.get("records", [])
        if str(row.get("uid")) == uid and int(row.get("joint_seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Direct row for uid={uid!r}, seed={seed}; "
            f"found {len(matches)}"
        )
    return matches[0]


def direct_method_identity(
    report: dict[str, Any],
    *,
    view_count: int,
) -> dict[str, Any]:
    step = int(report.get("checkpoint_step", -1))
    if step <= 0:
        raise RuntimeError("Direct report lacks a positive checkpoint_step")
    return {
        "method_id": f"direct_ss900_slat_step{step:06d}",
        "label": f"Direct SS900 + SLAT step{step} ({int(view_count)} views)",
        "checkpoint_step": step,
        "report_format": str(report.get("format")),
    }


def parse_method_ids(value: str) -> list[str]:
    requested = [item.strip() for item in str(value).split(",") if item.strip()]
    if not requested:
        raise ValueError("--method_ids must select at least one method")
    if len(requested) != len(set(requested)):
        raise ValueError("--method_ids contains duplicate selectors")
    unknown = [item for item in requested if item not in METHOD_SELECTORS]
    if unknown:
        raise ValueError(
            f"unsupported --method_ids selectors={unknown}; "
            f"allowed={list(METHOD_SELECTORS)}"
        )
    requested_set = set(requested)
    # The renderer owns the scientifically meaningful, stable column order.
    return [item for item in METHOD_SELECTORS if item in requested_set]


def review_mode_ids(pixal_pose_policy: str) -> list[str]:
    if pixal_pose_policy == "metadata":
        return ["canonical_pose", "shape_aligned"]
    if pixal_pose_policy == "reference_rigid_icp":
        return ["pixal_pose_aligned"]
    raise ValueError(f"unsupported Pixal pose policy={pixal_pose_policy!r}")


def output_summary(
    *,
    status: str,
    report_path: Path,
    comparisons: dict[str, dict[str, Any]],
    selected_method_ids: list[str],
    index_path: Path | None = None,
) -> dict[str, Any]:
    result = {
        "status": status,
        "report": str(report_path),
        "selected_method_ids": selected_method_ids,
        "contact_sheets": {
            mode_id: comparison["normal_contact_sheet"]["path"]
            for mode_id, comparison in comparisons.items()
        },
        "turntables": {
            mode_id: comparison["normal_turntable"]["path"]
            for mode_id, comparison in comparisons.items()
        },
    }
    if index_path is not None:
        result["html"] = str(index_path)
    return result


def make_html(
    *,
    output_dir: Path,
    comparisons: dict[str, dict[str, Any]],
    complete: bool,
) -> str:
    def relative(path: str | Path) -> str:
        return os.path.relpath(Path(path), output_dir).replace(os.sep, "/")

    sections = []
    for mode_id, comparison in comparisons.items():
        links = "\n".join(
            (
                f"<li><strong>{html.escape(row['label'])}</strong>: "
                f"<a href=\"{html.escape(relative(row['mesh']['path']))}\">"
                "rendered-coordinate Mesh</a> · "
                f"<a href=\"{html.escape(relative(row['contact_sheet']['path']))}\">"
                "turntable sheet</a></li>"
            )
            for row in comparison["methods"]
        )
        if mode_id == "canonical_pose":
            warning = (
                "保留预测位姿、尺度和平移误差；四列只使用 Reference 派生的一份"
                "显示矩阵。"
            )
        elif mode_id == "pixal_pose_aligned":
            warning = (
                "GT-assisted pose-only：只对 Pixal 应用 proper SE(3) 旋转和平移；"
                "禁止 reflection/scale，其他三列保持 canonical pose。"
            )
        else:
            warning = (
                "GT-assisted shape-only：预测 Mesh 经过 proper isotropic Sim(3) ICP；"
                "不能用于证明世界位姿或模型排名。"
            )
        sections.append(
            "<section>"
            f"<h2>{html.escape(comparison['label'])}</h2>"
            f"<p>{html.escape(warning)}</p>"
            f"<p><a href=\"{html.escape(relative(comparison['normal_turntable']['path']))}\">"
            "并排视频</a></p>"
            f"<img src=\"{html.escape(relative(comparison['normal_contact_sheet']['path']))}\">"
            f"<ul>{links}</ul>"
            "</section>"
        )
    pose_only = list(comparisons) == ["pixal_pose_aligned"]
    title = (
        "Pixal pose-only corrected four-way review"
        if pose_only
        else (
            "Calibrated object four-way canonical/shape review"
            if complete
            else "Calibrated object available-method canonical/shape preview"
        )
    )
    status = (
        "四种 Mesh 均已完成。"
        if complete
        else "这是先行预览：只包含当前已有 Mesh；缺失方法完成后另行生成严格四路结果。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ background:#151515; color:#eee; font-family:sans-serif; margin:24px; }}
a {{ color:#7cc7ff; }} img {{ max-width:100%; border:1px solid #555; }}
section {{ margin:30px 0; padding:16px; background:#222; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p>{html.escape(status)}</p>
<p>canonical_pose 检查原始模型坐标与尺度；pixal_pose_aligned 只修正 Pixal
的 rigid pose；shape_aligned 去除每个预测的 Sim(3)。三种口径不可混用。</p>
{''.join(sections)}
</body>
</html>
"""


def affine_audit(matrix: np.ndarray) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(
        value[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("transform must be affine")
    linear = value[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise ValueError("reflected or degenerate transforms are forbidden")
    return {
        "matrix": value.tolist(),
        "determinant": determinant,
        "singular_values": singular.tolist(),
        "anisotropy_ratio": float(
            singular.max() / max(singular.min(), 1.0e-15)
        ),
        "proper": True,
    }


def identity_alignment(*, policy: str) -> dict[str, Any]:
    audit = affine_audit(np.eye(4, dtype=np.float64))
    audit.update(
        {
            "policy": policy,
            "gt_assisted": False,
            "cost": 0.0,
            "proper_rotation_only": True,
            "isotropic_scale": True,
        }
    )
    return audit


def rigid_icp(
    source,
    target,
    *,
    seed: int,
    candidate_samples: int,
    final_samples: int,
    candidate_iterations: int,
    final_iterations: int,
):
    """Align source to target with a proper SE(3), never scale or reflection."""

    source_candidate, _ = deterministic_surface_sample(
        source, candidate_samples, seed
    )
    target_candidate, _ = deterministic_surface_sample(
        target, candidate_samples * 2, seed + 1
    )
    source_center = 0.5 * (
        source_candidate.min(axis=0) + source_candidate.max(axis=0)
    )
    target_center = 0.5 * (
        target_candidate.min(axis=0) + target_candidate.max(axis=0)
    )
    candidates = []
    for rotation in proper_cube_rotations():
        initial = np.eye(4, dtype=np.float64)
        initial[:3, :3] = rotation
        initial[:3, 3] = target_center - rotation @ source_center
        matrix, _, cost = trimesh.registration.icp(
            source_candidate,
            target_candidate,
            initial=initial,
            max_iterations=int(candidate_iterations),
            reflection=False,
            translation=True,
            scale=False,
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
        scale=False,
    )
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    singular = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise RuntimeError("rigid ICP produced a reflected transform")
    if not np.allclose(singular, np.ones(3), rtol=0.0, atol=1.0e-6):
        raise RuntimeError(
            f"rigid ICP unexpectedly changed scale: singular_values={singular}"
        )
    aligned = source.copy()
    aligned.apply_transform(matrix)
    return aligned, {
        "matrix": matrix.tolist(),
        "cost": float(cost),
        "determinant": determinant,
        "singular_values": singular.tolist(),
        "anisotropy_ratio": float(singular.max() / singular.min()),
        "proper_rotation_only": True,
        "translation": True,
        "rigid": True,
        "scale_applied": False,
        "isotropic_scale": False,
    }


def normalized_symmetric_rigid_icp(
    source,
    target,
    *,
    seed: int,
    candidate_samples: int,
    final_samples: int,
    candidate_iterations: int,
    final_iterations: int,
):
    """Robust pose-only fit for incomplete predictions with different scale.

    Scale is removed only while selecting the rotation.  The returned Mesh is
    transformed by a proper SE(3): the original scale is never changed.
    """

    from scipy.spatial import cKDTree

    def normalized(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        centered = points - center[None]
        rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        if not np.isfinite(rms) or rms <= 1.0e-12:
            raise RuntimeError("cannot align a degenerate surface sample")
        return centered / rms, center, rms

    def symmetric_cost(
        source_points: np.ndarray,
        target_points: np.ndarray,
        *,
        keep_fraction: float = 0.9,
    ) -> float:
        source_distance = cKDTree(target_points).query(source_points, k=1)[0]
        target_distance = cKDTree(source_points).query(target_points, k=1)[0]

        def trimmed_square_mean(distance: np.ndarray) -> float:
            keep = max(1, int(np.floor(len(distance) * keep_fraction)))
            selected = np.partition(distance, keep - 1)[:keep]
            return float(np.mean(selected * selected))

        return 0.5 * (
            trimmed_square_mean(source_distance)
            + trimmed_square_mean(target_distance)
        )

    source_candidate, _ = deterministic_surface_sample(
        source, candidate_samples, seed
    )
    target_candidate, _ = deterministic_surface_sample(
        target, candidate_samples * 2, seed + 1
    )
    source_normalized, _, _ = normalized(source_candidate)
    target_normalized, _, _ = normalized(target_candidate)
    candidates = []
    for rotation in proper_cube_rotations():
        initial = np.eye(4, dtype=np.float64)
        initial[:3, :3] = rotation
        matrix, _, _ = trimesh.registration.icp(
            source_normalized,
            target_normalized,
            initial=initial,
            max_iterations=int(candidate_iterations),
            reflection=False,
            translation=True,
            scale=False,
        )
        transformed = trimesh.transform_points(source_normalized, matrix)
        candidates.append(
            (symmetric_cost(transformed, target_normalized), matrix)
        )

    source_final, _ = deterministic_surface_sample(source, final_samples, seed + 2)
    target_final, _ = deterministic_surface_sample(
        target, final_samples * 2, seed + 3
    )
    source_final_normalized, source_center, source_rms = normalized(source_final)
    target_final_normalized, target_center, target_rms = normalized(target_final)
    finalists = []
    for _, initial in sorted(candidates, key=lambda item: item[0])[:4]:
        matrix, _, _ = trimesh.registration.icp(
            source_final_normalized,
            target_final_normalized,
            initial=initial,
            max_iterations=int(final_iterations),
            reflection=False,
            translation=True,
            scale=False,
        )
        transformed = trimesh.transform_points(source_final_normalized, matrix)
        finalists.append(
            (symmetric_cost(transformed, target_final_normalized), matrix)
        )
    score, normalized_matrix = min(finalists, key=lambda item: item[0])

    u, _, vh = np.linalg.svd(normalized_matrix[:3, :3])
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = target_center - rotation @ source_center
    singular = np.linalg.svd(rotation, compute_uv=False)
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not np.allclose(
        singular, np.ones(3), rtol=0.0, atol=1.0e-6
    ):
        raise RuntimeError("normalized symmetric fit did not produce proper SE(3)")

    aligned = source.copy()
    aligned.apply_transform(matrix)
    return aligned, {
        "matrix": matrix.tolist(),
        "cost": float(score),
        "cost_space": "independently RMS-normalized point clouds",
        "cost_policy": "90%-trimmed symmetric bidirectional squared NN distance",
        "source_selection_rms": source_rms,
        "target_selection_rms": target_rms,
        "determinant": determinant,
        "singular_values": singular.tolist(),
        "anisotropy_ratio": float(singular.max() / singular.min()),
        "proper_rotation_only": True,
        "translation": True,
        "rigid": True,
        "scale_applied": False,
        "isotropic_scale": False,
    }


def export_mesh_atomic(mesh: trimesh.Trimesh, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    mesh.export(temporary, file_type=destination.suffix.lstrip("."))
    os.replace(temporary, destination)


def method_render_config(
    *,
    mode_id: str,
    spec: dict[str, Any],
    target_path: Path,
    shared_display_transform: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    method_position = int(spec["method_position"])
    config = {
        "mode_id": mode_id,
        "selector": str(spec["selector"]),
        "method_position": method_position,
        "method_id": str(spec["method_id"]),
        "label": str(spec["label"]),
        "pose_status": str(spec["pose_status"]),
        "source_mesh": binding(Path(spec["mesh_path"]).resolve()),
        "source_to_canonical": np.asarray(
            spec["source_to_canonical"], dtype=np.float64
        ).tolist(),
        "shared_display_transform": shared_display_transform,
        "render_frames": int(args.render_frames),
        "render_resolution": int(args.render_resolution),
        "contact_frames": int(args.contact_frames),
        "fps": int(args.fps),
        "display_margin": float(args.display_margin),
        "alignment": {
            "enabled": mode_id in {"rigid_pose_gt", "shape_aligned"}
            and str(spec["method_id"]) != "reference",
            "target_mesh": (
                binding(target_path)
                if mode_id in {"rigid_pose_gt", "shape_aligned"}
                else None
            ),
            "seed": int(args.alignment_seed) + method_position * 100,
            "candidate_samples": int(args.candidate_samples),
            "alignment_samples": int(args.alignment_samples),
            "candidate_iterations": int(args.candidate_iterations),
            "final_iterations": int(args.final_iterations),
            "rigid_alignment_policy": str(
                spec.get("rigid_alignment_policy", "standard_surface_icp")
            ),
        },
        "code_bindings": code_bindings(
            {
                "runner": Path(__file__).resolve(),
                "render_helpers": (
                    Path(__file__).resolve().parent / "bunny_review" / "finalize.py"
                ),
                "similarity_icp": (
                    Path(__file__).resolve().parent
                    / "compare_pixal3d_singleview_smoke.py"
                ),
            }
        ),
    }
    config["config_sha256"] = canonical_sha256(config)
    return config


def load_completed_method(
    *,
    result_path: Path,
    expected_config: dict[str, Any],
    expected_frames: int,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != f"{FORMAT}.method.v1":
        raise RuntimeError(f"unexpected method render format: {result_path}")
    if result.get("config") != expected_config:
        raise RuntimeError(f"method render config changed: {result_path}")
    row = result["method"]
    for key in ("mesh", "source_mesh", "normal_turntable", "contact_sheet"):
        validate_binding(row[key], label=f"{result_path}:{key}")
    frames = [
        np.asarray(frame)
        for frame in imageio.mimread(row["normal_turntable"]["path"])
    ]
    if len(frames) != int(expected_frames):
        raise RuntimeError(
            f"method video frame count changed: {result_path}, "
            f"expected={expected_frames}, actual={len(frames)}"
        )
    return row, frames


def render_review_mode(
    *,
    mode_id: str,
    mode_label: str,
    output_dir: Path,
    method_specs: list[dict[str, Any]],
    target_path: Path,
    shared_display_transform: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if mode_id not in {
        "canonical_pose",
        "pixal_pose_aligned",
        "rigid_pose_gt",
        "shape_aligned",
    }:
        raise ValueError(f"unsupported review mode={mode_id!r}")
    target = (
        load_mesh(target_path)
        if mode_id in {"rigid_pose_gt", "shape_aligned"}
        else None
    )
    rows: list[dict[str, Any]] = []
    all_frames: list[tuple[str, list[np.ndarray]]] = []
    mode_dir = output_dir / mode_id
    for progress_position, spec in enumerate(method_specs, start=1):
        method_id = str(spec["method_id"])
        label = str(spec["label"])
        method_position = int(spec["method_position"])
        method_dir = mode_dir / "methods" / method_id
        method_result_path = method_dir / "method_result.json"
        method_config = method_render_config(
            mode_id=mode_id,
            spec=spec,
            target_path=target_path,
            shared_display_transform=shared_display_transform,
            args=args,
        )
        if args.resume and method_result_path.is_file():
            row, frames = load_completed_method(
                result_path=method_result_path,
                expected_config=method_config,
                expected_frames=int(args.render_frames),
            )
            print(
                f"[fourway_render:{mode_id}] {progress_position}/"
                f"{len(method_specs)} {method_id} reused",
                flush=True,
            )
            rows.append(row)
            all_frames.append((label, frames))
            continue
        print(
            f"[fourway_render:{mode_id}] {progress_position}/{len(method_specs)} "
            f"{method_id}",
            flush=True,
        )
        source_path = Path(spec["mesh_path"]).resolve()
        source_to_canonical = np.asarray(
            spec["source_to_canonical"],
            dtype=np.float64,
        )
        source_to_canonical_audit = affine_audit(source_to_canonical)
        canonical = load_mesh(source_path)
        canonical.apply_transform(source_to_canonical)
        canonical_stats = mesh_stats(canonical)

        if mode_id in {"rigid_pose_gt", "shape_aligned"} and method_id != "reference":
            assert target is not None
            if mode_id == "rigid_pose_gt":
                rigid_policy = str(
                    spec.get("rigid_alignment_policy", "standard_surface_icp")
                )
                aligner = (
                    normalized_symmetric_rigid_icp
                    if rigid_policy == "normalized_symmetric"
                    else rigid_icp
                )
            else:
                rigid_policy = "not_applicable"
                aligner = similarity_icp
            mode_mesh, alignment = aligner(
                canonical,
                target,
                seed=int(args.alignment_seed) + method_position * 100,
                candidate_samples=int(args.candidate_samples),
                final_samples=int(args.alignment_samples),
                candidate_iterations=int(args.candidate_iterations),
                final_iterations=int(args.final_iterations),
            )
            alignment = dict(alignment)
            alignment.update(
                {
                    "policy": (
                        "scale-normalized symmetric bidirectional rotation selection; "
                        "returned transform is proper SE(3), with scale and reflection "
                        "forbidden"
                        if mode_id == "rigid_pose_gt"
                        and rigid_policy == "normalized_symmetric"
                        else "24 proper cube rotations plus proper SE(3) ICP against "
                        "Reference; scale and reflection forbidden"
                        if mode_id == "rigid_pose_gt"
                        else "24 proper cube rotations plus proper isotropic-similarity "
                        "ICP against Reference"
                    ),
                    "gt_assisted": True,
                    "seed": int(args.alignment_seed) + method_position * 100,
                }
            )
            canonical_to_mode = np.asarray(alignment["matrix"], dtype=np.float64)
        else:
            mode_mesh = canonical
            alignment = identity_alignment(
                policy=(
                    "Reference remains fixed during GT-assisted alignment"
                    if mode_id in {"rigid_pose_gt", "shape_aligned"}
                    else "no per-object alignment in canonical-pose review"
                )
            )
            canonical_to_mode = np.eye(4, dtype=np.float64)
        alignment_audit = affine_audit(canonical_to_mode)
        if alignment_audit["anisotropy_ratio"] > 1.0 + 1.0e-5:
            raise RuntimeError(
                f"{method_id} alignment is unexpectedly anisotropic"
            )
        mode_stats = dict(canonical_stats)
        mode_stats["bounds"] = np.asarray(
            mode_mesh.bounds,
            dtype=np.float64,
        ).tolist()
        mode_stats["extent"] = np.asarray(
            mode_mesh.extents,
            dtype=np.float64,
        ).tolist()
        mode_stats["topology_invariant_under_affine_transform"] = True
        frames, display = render_method(
            mode_mesh,
            device=device,
            frames=int(args.render_frames),
            resolution=int(args.render_resolution),
            display_margin=float(args.display_margin),
            shared_display_transform=shared_display_transform,
        )
        video_path = method_dir / "normal_turntable.mp4"
        sheet_path = method_dir / "normal_contact_sheet.png"
        save_video_atomic(frames, video_path, int(args.fps))
        contact_sheet(
            frames,
            label,
            sheet_path,
            count=int(args.contact_frames),
        )

        transform_is_identity = np.allclose(
            canonical_to_mode @ source_to_canonical,
            np.eye(4, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        )
        rendered_mesh_path = source_path
        if not transform_is_identity:
            mesh_names = {
                "shape_aligned": "mesh_shape_aligned.obj",
                "rigid_pose_gt": "mesh_rigid_pose_gt.obj",
            }
            rendered_mesh_path = method_dir / mesh_names.get(
                mode_id, "mesh_direct_canonical.obj"
            )
            export_mesh_atomic(mode_mesh, rendered_mesh_path)
        display_matrix = np.asarray(
            shared_display_transform["matrix"],
            dtype=np.float64,
        )
        source_to_display = (
            display_matrix @ canonical_to_mode @ source_to_canonical
        )
        row = {
            "method_id": method_id,
            "label": label,
            "pose_status": spec["pose_status"],
            "mesh": binding(rendered_mesh_path),
            "source_mesh": binding(source_path),
            "canonical_mesh_stats": canonical_stats,
            "mode_mesh_stats": mode_stats,
            "source_to_canonical": source_to_canonical_audit,
            "canonical_to_mode": alignment,
            "source_to_display": affine_audit(source_to_display),
            "display_transform": display,
            "normal_turntable": binding(video_path),
            "contact_sheet": binding(sheet_path),
        }
        atomic_json(
            method_result_path,
            {
                "format": f"{FORMAT}.method.v1",
                "complete": True,
                "config": method_config,
                "method": row,
            },
        )
        rows.append(row)
        all_frames.append((label, frames))
        del canonical, mode_mesh, frames
        gc.collect()
        torch.cuda.empty_cache()

    combined = comparison_frames(all_frames)
    comparison_video = mode_dir / "normal_side_by_side.mp4"
    comparison_sheet = mode_dir / "normal_contact_sheet.png"
    save_video_atomic(combined, comparison_video, int(args.fps))
    comparison_contact_sheet(
        combined,
        comparison_sheet,
        count=int(args.contact_frames),
    )
    if target is not None:
        del target
    return {
        "mode_id": mode_id,
        "label": mode_label,
        "methods": rows,
        "normal_contact_sheet": binding(comparison_sheet),
        "normal_turntable": binding(comparison_video),
        "shared_reference_display_transform": shared_display_transform,
        "gt_assisted": mode_id in {"rigid_pose_gt", "shape_aligned"},
    }


def run(args: argparse.Namespace) -> None:
    for name in (
        "render_frames",
        "render_resolution",
        "contact_frames",
        "fps",
        "candidate_samples",
        "alignment_samples",
        "candidate_iterations",
        "final_iterations",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not 0.0 < float(args.display_margin) <= 1.0:
        raise ValueError("--display_margin must be in (0, 1]")
    selected_method_ids = parse_method_ids(args.method_ids)
    if args.pixal_pose_policy not in PIXAL_POSE_POLICIES:
        raise ValueError(
            f"unsupported --pixal_pose_policy={args.pixal_pose_policy!r}"
        )
    if (
        args.pixal_pose_policy == "reference_rigid_icp"
        and "pixal3d_native" not in selected_method_ids
    ):
        raise ValueError(
            "reference_rigid_icp requires pixal3d_native in --method_ids"
        )
    full_method_set = selected_method_ids == list(METHOD_SELECTORS)
    selected_review_modes = review_mode_ids(args.pixal_pose_policy)
    protocol_path = args.pixal_protocol.resolve()
    protocol = validate_protocol(protocol_path)
    protocol["_protocol_path"] = str(protocol_path)
    if len(protocol["cases"]) != 1:
        raise RuntimeError("four-way review requires exactly one Pixal3D case")
    case = protocol["cases"][0]
    if str(case["uid"]) != args.uid or int(case["current_seed"]) != args.seed:
        raise RuntimeError("Pixal3D protocol UID/seed differs from requested review")

    direct_report_path = args.direct_report.resolve()
    direct_report = json.loads(direct_report_path.read_text(encoding="utf-8"))
    if direct_report.get("format") not in DIRECT_REPORT_FORMATS:
        raise RuntimeError("unexpected Direct-SLAT report format")
    direct_identity = direct_method_identity(
        direct_report,
        view_count=int(case["view_count"]),
    )
    record = direct_record(direct_report, uid=args.uid, seed=args.seed)
    if str(case["pair_id"]) != str(record["pair_id"]):
        raise RuntimeError("Pixal3D protocol pair ID differs from Direct report")
    direct_root = direct_report_path.parent
    full_path = (
        direct_root
        / "mesh_pairs"
        / str(record["pair_id"])
        / "full"
        / "mesh_canonical.obj"
    ).resolve()
    if full_path != Path(case["current_mesh"]["path"]).resolve():
        raise RuntimeError("Pixal3D protocol Full mesh path differs from Direct report")
    if sha256_file(full_path) != case["current_mesh"]["sha256"]:
        raise RuntimeError("Direct Full mesh changed after Pixal3D protocol freeze")

    target_path = Path(case["target_mesh"]["path"]).resolve()
    if sha256_file(target_path) != case["target_mesh"]["sha256"]:
        raise RuntimeError("canonical target mesh changed after protocol freeze")
    transform_path = args.pixal_transform.resolve()
    transform = validate_transform(transform_path)
    pixal_matrix = case_canonical_transform(case, transform)
    pixal_path = pixal3d_mesh_path(protocol_path, str(case["case_id"])).resolve()
    if "pixal3d_native" in selected_method_ids:
        pixal_result = load_pixal_result(protocol, case)
        if Path(pixal_result["mesh"]).resolve() != pixal_path:
            raise RuntimeError("Pixal3D result mesh path is inconsistent")

    recon_report_path = args.reconviagen_report.resolve()
    recon_report = json.loads(recon_report_path.read_text(encoding="utf-8"))
    if (
        recon_report.get("format") != RECON_REPORT_FORMAT
        or recon_report.get("complete") is not True
    ):
        raise RuntimeError("ReconViaGen stock report is incomplete or unsupported")
    if str(recon_report["run_config"]["uid"]) != args.uid:
        raise RuntimeError("ReconViaGen stock report UID differs from review")
    recon_path = validate_binding(
        recon_report["mesh_canonical"],
        label="ReconViaGen stock canonical mesh",
    )

    source_bindings = {
        "direct_report": binding(direct_report_path),
        "pixal_protocol": binding(protocol_path),
        "pixal_transform": binding(transform_path),
        "reconviagen_report": binding(recon_report_path),
        "reference_mesh": binding(target_path),
        "reconviagen_stock_mesh": binding(recon_path),
        "direct_full_mesh": binding(full_path),
    }
    if "pixal3d_native" in selected_method_ids:
        source_bindings["pixal3d_official_final_mesh"] = binding(pixal_path)
    pixal_base_matrix = (
        LATENT_DECODER_TO_REFERENCE
        @ np.asarray(pixal_matrix, dtype=np.float64)
    )
    pixal_source_to_canonical = pixal_base_matrix
    pixal_pose_alignment = identity_alignment(
        policy=(
            "metadata-only Pixal canonical transform; no Reference fitting"
        )
    )
    pixal_pose_alignment.update(
        {
            "pose_policy": "metadata",
            "scale_applied": False,
        }
    )
    if args.pixal_pose_policy == "reference_rigid_icp":
        pixal_canonical = load_mesh(pixal_path)
        pixal_canonical.apply_transform(pixal_base_matrix)
        reference_for_pose = load_mesh(target_path)
        _, pixal_pose_alignment = rigid_icp(
            pixal_canonical,
            reference_for_pose,
            seed=int(args.alignment_seed) + 200,
            candidate_samples=int(args.candidate_samples),
            final_samples=int(args.alignment_samples),
            candidate_iterations=int(args.candidate_iterations),
            final_iterations=int(args.final_iterations),
        )
        pixal_pose_alignment = dict(pixal_pose_alignment)
        pixal_pose_alignment.update(
            {
                "policy": (
                    "GT-assisted Reference rigid ICP for Pixal only; proper "
                    "SE(3), reflection and scale forbidden"
                ),
                "pose_policy": "reference_rigid_icp",
                "gt_assisted": True,
                "seed": int(args.alignment_seed) + 200,
            }
        )
        pixal_pose_matrix = np.asarray(
            pixal_pose_alignment["matrix"], dtype=np.float64
        )
        pixal_source_to_canonical = pixal_pose_matrix @ pixal_base_matrix
        del pixal_canonical, reference_for_pose

    run_config = {
        "uid": args.uid,
        "object_uid": str(case["object_uid"]),
        "seed": int(args.seed),
        "render_frames": int(args.render_frames),
        "render_resolution": int(args.render_resolution),
        "contact_frames": int(args.contact_frames),
        "fps": int(args.fps),
        "display_margin": float(args.display_margin),
        "review_modes": selected_review_modes,
        "selected_method_ids": selected_method_ids,
        "full_method_set": full_method_set,
        "shape_alignment": {
            "alignment_seed": int(args.alignment_seed),
            "candidate_samples": int(args.candidate_samples),
            "alignment_samples": int(args.alignment_samples),
            "candidate_iterations": int(args.candidate_iterations),
            "final_iterations": int(args.final_iterations),
            "proper_cube_rotation_initializations": 24,
            "reflection": False,
            "isotropic_scale": True,
        },
        "sources": source_bindings,
        "direct_method": direct_identity,
        "pixal3d_case_canonical_matrix": np.asarray(
            pixal_matrix, dtype=np.float64
        ).tolist(),
        "pixal_pose_policy": args.pixal_pose_policy,
        "pixal_pose_alignment": pixal_pose_alignment,
        "latent_decoder_to_reference_matrix": (
            LATENT_DECODER_TO_REFERENCE.tolist()
        ),
        "reference_frame": (
            "normalized source-GLB/view frame; decoder meshes exported with "
            "transform_pose=False are converted using the exact vendored "
            "MeshExtractResult transform_pose=True axis convention"
        ),
    }
    run_config["config_sha256"] = canonical_sha256(run_config)

    output_dir = args.output_dir.resolve()
    report_path = output_dir / (
        "report.json" if full_method_set else "partial_report.json"
    )
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("format") != FORMAT or report.get("run_config") != run_config:
            raise RuntimeError("existing Mesh review uses a different config")
        for mode_id in run_config["review_modes"]:
            validate_binding(
                report["comparisons"][mode_id]["normal_contact_sheet"],
                label=f"{mode_id}.normal_contact_sheet",
            )
            validate_binding(
                report["comparisons"][mode_id]["normal_turntable"],
                label=f"{mode_id}.normal_turntable",
            )
        print(
            json.dumps(
                output_summary(
                    status="reused",
                    report_path=report_path,
                    comparisons=report["comparisons"],
                    selected_method_ids=selected_method_ids,
                    index_path=Path(report["html"]["path"]),
                ),
                indent=2,
            )
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise RuntimeError(
            f"partial Mesh review exists; rerun with --resume after inspection: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("shared nvdiffrast renderer requires --device cuda")
    identity = np.eye(4, dtype=np.float64)
    all_method_specs = [
        {
            "selector": "reference",
            "method_position": 1,
            "method_id": "reference",
            "label": "Reference",
            "mesh_path": target_path,
            "source_to_canonical": identity,
            "pose_status": (
                "normalized source-GLB/view-frame Reference; authoritative "
                "display-frame owner"
            ),
        },
        {
            "selector": "pixal3d_native",
            "method_position": 2,
            "method_id": "pixal3d_native",
            "label": "Pixal3D official final (1 view)",
            "mesh_path": pixal_path,
            "source_to_canonical": pixal_source_to_canonical,
            "pose_status": (
                (
                    "official postprocessed GLB; GT-assisted proper rigid SE(3) "
                    "alignment to Reference; scale and geometry unchanged"
                )
                if args.pixal_pose_policy == "reference_rigid_icp"
                else (
                    "official postprocessed GLB; score-independent selected-c2w "
                    "metadata transform into decoder latent frame, followed by "
                    "the fixed decoder-to-Reference axis conversion; pose-aware "
                    "but not GT-fitted"
                )
            ),
        },
        {
            "selector": "reconviagen_stock",
            "method_position": 3,
            "method_id": "reconviagen_stock",
            "label": (
                f"ReconViaGen stock "
                f"({len(recon_report['run_config']['view_ids'])} views)"
            ),
            "mesh_path": recon_path,
            "source_to_canonical": LATENT_DECODER_TO_REFERENCE,
            "pose_status": (
                "fixed vendored decoder-to-Reference axis conversion applied; "
                "original inference did not consume camera extrinsics, so residual "
                "AR/world yaw remains unanchored"
            ),
        },
        {
            "selector": "direct",
            "method_position": 4,
            "method_id": direct_identity["method_id"],
            "label": direct_identity["label"],
            "mesh_path": full_path,
            "source_to_canonical": LATENT_DECODER_TO_REFERENCE,
            "pose_status": (
                "fixed vendored decoder-to-Reference axis conversion; no per-object "
                "or GT fitting"
            ),
        },
    ]
    method_specs = [
        spec
        for spec in all_method_specs
        if str(spec["selector"]) in selected_method_ids
    ]
    reference_mesh = load_mesh(target_path)
    shared_display_transform = display_transform_from_mesh(
        reference_mesh,
        float(args.display_margin),
        owner="reference",
    )
    del reference_mesh
    comparison_builders = {
        "canonical_pose": lambda: render_review_mode(
            mode_id="canonical_pose",
            mode_label="Canonical pose / shared Reference display frame",
            output_dir=output_dir,
            method_specs=method_specs,
            target_path=target_path,
            shared_display_transform=shared_display_transform,
            device=device,
            args=args,
        ),
        "pixal_pose_aligned": lambda: render_review_mode(
            mode_id="pixal_pose_aligned",
            mode_label=(
                "GT-assisted Pixal-only rigid pose alignment / shared "
                "Reference display frame"
            ),
            output_dir=output_dir,
            method_specs=method_specs,
            target_path=target_path,
            shared_display_transform=shared_display_transform,
            device=device,
            args=args,
        ),
        "shape_aligned": lambda: render_review_mode(
            mode_id="shape_aligned",
            mode_label="GT-assisted shape-only Sim(3) alignment",
            output_dir=output_dir,
            method_specs=method_specs,
            target_path=target_path,
            shared_display_transform=shared_display_transform,
            device=device,
            args=args,
        ),
    }
    comparisons = {
        mode_id: comparison_builders[mode_id]()
        for mode_id in selected_review_modes
    }
    index_path = output_dir / "index.html"
    atomic_text(
        index_path,
        make_html(
            output_dir=output_dir,
            comparisons=comparisons,
            complete=full_method_set,
        ),
    )
    report = {
        "format": FORMAT,
        "complete": full_method_set,
        "partial": not full_method_set,
        "formal": False,
        "purpose": (
            "single-object GT-assisted Pixal pose-only human geometry review; "
            "not pose evaluation or checkpoint selection"
            if args.pixal_pose_policy == "reference_rigid_icp"
            else "single-object human geometry review; not checkpoint selection"
        ),
        "run_config": run_config,
        "comparisons": comparisons,
        "html": binding(index_path),
        "transform_gates": {
            "passed": True,
            "shared_display_matrix_owner": "reference",
            "shared_display_matrix_reused_for_every_method_and_mode": True,
            "reference_source_to_canonical_identity": True,
            "decoder_to_reference_axis_conversion_fixed": True,
            "decoder_to_reference_axis_conversion_gt_fitted": False,
            "decoder_to_reference_axis_conversion": (
                "exact vendored MeshExtractResult.to_trimesh(transform_pose=True) "
                "(x,y,z)->(x,z,-y)"
            ),
            "pixal3d_metadata_transform_score_independent": True,
            "pixal3d_render_transform_score_independent": (
                args.pixal_pose_policy == "metadata"
            ),
            "pixal_pose_policy": args.pixal_pose_policy,
            "pixal_pose_alignment_gt_assisted": (
                args.pixal_pose_policy == "reference_rigid_icp"
            ),
            "pixal_pose_alignment_rigid_no_scale": (
                args.pixal_pose_policy == "reference_rigid_icp"
            ),
            "shape_alignment_proper_no_reflection": True,
            "shape_alignment_isotropic_scale": True,
            "shape_alignment_run": "shape_aligned" in selected_review_modes,
            "silhouette_reprojection_gate": {
                "status": "not_run",
                "reason": (
                    "this renderer audits shared Mesh coordinates; arbitrary-camera "
                    "mask replay remains a separate future calibration test"
                ),
            },
        },
        "render_policy": {
            "camera_path": "shared yaw turntable with fixed pitch/radius/FOV",
            "canonical_pose_display_normalization": (
                "one Reference-derived bbox centering/isotropic-scale matrix is "
                "applied unchanged to every method; per-method normalization forbidden"
            ),
            "decoder_mesh_frame": (
                "ReconViaGen and Direct transform_pose=False decoder Meshes receive "
                "the fixed vendored decoder-to-source/view axis conversion before "
                "rendering"
            ),
            "shape_aligned": (
                "GT-assisted proper isotropic Sim(3) alignment using 24 proper cube "
                "rotation initializations; shared Reference display matrix afterward"
            ),
            "pixal3d_orientation": (
                (
                    "GT-assisted proper rigid SE(3) against Reference for the "
                    "pose-only review; scale and geometry unchanged"
                )
                if args.pixal_pose_policy == "reference_rigid_icp"
                else (
                    "frozen metadata-only canonical transform from the audited "
                    "single-view camera"
                )
            ),
            "reconviagen_orientation": (
                "native ReconViaGen canonical output; no ICP or per-object "
                "orientation fitting"
            ),
        },
        "guardrails": [
            "ReconViaGen stock is an original full-chain run, not the Direct exporter stock branch.",
            (
                "Direct checkpoint identity is report-bound: "
                f"step={direct_identity['checkpoint_step']}, "
                f"format={direct_identity['report_format']}."
            ),
            (
                "Pixal receives the Reference only for GT-assisted rigid pose "
                "alignment; scale and geometry are unchanged. Other methods do "
                "not receive the target."
                if args.pixal_pose_policy == "reference_rigid_icp"
                else (
                    "Normal renders compare geometry only; no method receives "
                    "target Mesh as input in canonical_pose."
                )
            ),
            "Pixal3D uses the official final GLB after o_voxel remesh and the inference.py output rotation; decoded pre-remesh geometry is forbidden.",
            "Reference source GLB and transform_pose=False decoder Meshes do not share axes; the exact vendored transform_pose=True convention is applied before comparison.",
            "canonical_pose is the only mode that preserves pose/translation/scale errors.",
            (
                "pixal_pose_aligned is GT-assisted and changes only Pixal rotation/"
                "translation; it cannot support a pose-estimation claim."
            ),
            "shape_aligned reads the Reference Mesh and is human-review-only; it cannot support an AR/world-pose claim.",
            "ReconViaGen native output is explicitly pose-unanchored because its original inference did not consume c2w.",
        ],
        "code_bindings": code_bindings(
            {
                "runner": Path(__file__).resolve(),
                "render_helpers": (
                    Path(__file__).resolve().parent / "bunny_review" / "finalize.py"
                ),
                "similarity_icp": (
                    Path(__file__).resolve().parent
                    / "compare_pixal3d_singleview_smoke.py"
                ),
                "mesh_coordinate_convention": (
                    Path(__file__).resolve().parents[1]
                    / "ReconViaGen"
                    / "trellis"
                    / "representations"
                    / "mesh"
                    / "cube2mesh.py"
                ),
            }
        ),
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            output_summary(
                status="complete" if full_method_set else "partial_complete",
                report_path=report_path,
                comparisons=comparisons,
                selected_method_ids=selected_method_ids,
                index_path=index_path,
            ),
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct_report", type=Path, required=True)
    parser.add_argument("--pixal_protocol", type=Path, required=True)
    parser.add_argument("--pixal_transform", type=Path, required=True)
    parser.add_argument("--reconviagen_report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_frames", type=int, default=72)
    parser.add_argument("--render_resolution", type=int, default=320)
    parser.add_argument("--contact_frames", type=int, default=6)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--display_margin", type=float, default=0.9)
    parser.add_argument("--alignment_seed", type=int, default=20260729)
    parser.add_argument("--candidate_samples", type=int, default=1000)
    parser.add_argument("--alignment_samples", type=int, default=4000)
    parser.add_argument("--candidate_iterations", type=int, default=8)
    parser.add_argument("--final_iterations", type=int, default=30)
    parser.add_argument(
        "--method_ids",
        default=",".join(METHOD_SELECTORS),
        help=(
            "Comma-separated subset of reference,pixal3d_native,"
            "reconviagen_stock,direct. Subsets write partial_report.json and "
            "do not require an already-generated Pixal Mesh when Pixal is omitted."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an inspected partial output. Completed per-method renders "
            "with identical bindings/config/code are reused atomically."
        ),
    )
    parser.add_argument(
        "--pixal_pose_policy",
        choices=PIXAL_POSE_POLICIES,
        default="metadata",
        help=(
            "metadata preserves the frozen Pixal pose and renders canonical plus "
            "shape-aligned reviews. reference_rigid_icp renders one GT-assisted "
            "Pixal-only pose-corrected review using proper SE(3), with no scale."
        ),
    )
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
