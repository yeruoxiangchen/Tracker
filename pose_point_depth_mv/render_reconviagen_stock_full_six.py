#!/usr/bin/env python3
"""Render six matched GT/ReconViaGen/Direct-stock/Direct-Full Mesh cases.

The input is the frozen report written by
``evaluate_reconviagen_stock_full_mesh``.  Every prediction receives the
coordinate transforms recorded in that report.  Within one object, all four
columns then share a display transform derived only from normalized GT and the
same turntable camera path.  No prediction is independently centered or scaled.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    save_image_atomic,
    save_video_atomic,
)
from .evaluate_reconviagen_stock_full_mesh import FORMAT as INPUT_REPORT_FORMAT
from .render_direct_slat_fourway import rigid_icp


FORMAT = "pose_point_depth_mv.reconviagen_stock_full_six_render.v1"
MODE = "canonical_pose"
METHOD_IDS = (
    "gt",
    "reconviagen_original",
    "direct_stock",
    "direct_full",
)
ALIGNMENT_MODES = (
    "canonical_pose",
    "rigid_pose_gt",
)


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {resolved}")
    return value


def finite_affine(value: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{label} must be affine")
    return matrix


def validate_input_report(
    report_path: str | Path,
    *,
    expected_objects: int,
    verify_meshes: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(report_path).resolve()
    report = load_json(path)
    if report.get("format") != INPUT_REPORT_FORMAT:
        raise ValueError(f"unsupported input report format={report.get('format')!r}")
    if report.get("complete") is not True:
        raise RuntimeError(f"input report is incomplete: {path}")
    body = dict(report)
    saved_hash = str(body.pop("report_sha256", ""))
    actual_hash = canonical_sha256(body)
    if not saved_hash or saved_hash != actual_hash:
        raise RuntimeError(
            f"input report hash mismatch: {actual_hash} != {saved_hash}"
        )
    if report.get("same_input_audit", {}).get("passed") is not True:
        raise RuntimeError("input report same-input audit did not pass")
    mode = report.get("modes", {}).get(MODE)
    if not isinstance(mode, dict) or mode.get("primary") is not True:
        raise RuntimeError(f"input report lacks primary mode={MODE!r}")
    records = list(mode.get("records", []))
    if len(records) != int(expected_objects):
        raise RuntimeError(
            f"expected {expected_objects} records, found {len(records)}"
        )
    pair_ids = [str(row.get("pair_id", "")) for row in records]
    if any(not value for value in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("input report pair_id values are empty or duplicated")
    for record_index, record in enumerate(records):
        target = record.get("target", {})
        if verify_meshes:
            validate_binding(
                target.get("source_glb", {}),
                label=f"records[{record_index}].target.source_glb",
            )
        for method_id in METHOD_IDS[1:]:
            method = record.get("methods", {}).get(method_id)
            if not isinstance(method, dict):
                raise RuntimeError(
                    f"records[{record_index}] lacks method={method_id!r}"
                )
            if verify_meshes:
                validate_binding(
                    method.get("mesh", {}),
                    label=f"records[{record_index}].methods.{method_id}.mesh",
                )
            finite_affine(
                method.get("source_to_reference", {}).get("matrix"),
                label=f"{method_id}.source_to_reference",
            )
            finite_affine(
                method.get("alignment", {}).get("matrix"),
                label=f"{method_id}.alignment",
            )
            if method.get("alignment", {}).get("gt_assisted") is not False:
                raise RuntimeError(
                    f"{method_id} canonical alignment unexpectedly uses GT"
                )
    return report, records


def normalized_target_mesh(target: dict[str, Any]) -> trimesh.Trimesh:
    source_path = validate_binding(target["source_glb"], label="target.source_glb")
    center = np.asarray(target["normalize_center"], dtype=np.float64)
    scale = float(target["normalize_scale"])
    margin = float(target["canonical_margin"])
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("target normalize_center is invalid")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("target normalize_scale is invalid")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("target canonical_margin is invalid")
    mesh = load_mesh(source_path)
    mesh.vertices = (
        (np.asarray(mesh.vertices, dtype=np.float64) - center[None]) / scale * margin
    )
    return mesh


def canonical_prediction_mesh(method: dict[str, Any], *, label: str) -> trimesh.Trimesh:
    path = validate_binding(method["mesh"], label=f"{label}.mesh")
    mesh = load_mesh(path)
    source_to_reference = finite_affine(
        method["source_to_reference"]["matrix"],
        label=f"{label}.source_to_reference",
    )
    alignment = finite_affine(
        method["alignment"]["matrix"],
        label=f"{label}.alignment",
    )
    mesh.apply_transform(source_to_reference)
    mesh.apply_transform(alignment)
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError(f"{label} contains non-finite transformed vertices")
    return mesh


def method_label(method_id: str, *, view_count: int) -> str:
    labels = {
        "gt": "GT / Reference",
        "reconviagen_original": f"ReconViaGen original ({view_count} views)",
        "direct_stock": f"Direct stock ({view_count} views)",
        "direct_full": f"Direct Full step100 ({view_count} views)",
    }
    return labels[method_id]


def identity_visual_alignment(*, policy: str) -> dict[str, Any]:
    return {
        "matrix": np.eye(4, dtype=np.float64).tolist(),
        "determinant": 1.0,
        "singular_values": [1.0, 1.0, 1.0],
        "anisotropy_ratio": 1.0,
        "proper_rotation_only": True,
        "translation": False,
        "rigid": True,
        "scale_applied": False,
        "gt_assisted": False,
        "policy": policy,
    }


def align_meshes_for_review(
    meshes: dict[str, trimesh.Trimesh],
    *,
    alignment_mode: str,
    alignment_seed: int,
    candidate_samples: int,
    alignment_samples: int,
    candidate_iterations: int,
    final_iterations: int,
) -> tuple[dict[str, trimesh.Trimesh], dict[str, dict[str, Any]]]:
    if alignment_mode == "canonical_pose":
        return meshes, {
            method_id: identity_visual_alignment(
                policy="canonical pose retained; no GT-assisted visual alignment"
            )
            for method_id in METHOD_IDS
        }
    if alignment_mode != "rigid_pose_gt":
        raise ValueError(f"unsupported alignment_mode={alignment_mode!r}")

    target = meshes["gt"]
    aligned_recon, recon_alignment = rigid_icp(
        meshes["reconviagen_original"],
        target,
        seed=int(alignment_seed) + 101,
        candidate_samples=int(candidate_samples),
        final_samples=int(alignment_samples),
        candidate_iterations=int(candidate_iterations),
        final_iterations=int(final_iterations),
    )
    # Stock and Full originate from the same Direct-SS/SLAT coordinate branch.
    # Estimate one rigid pose from Stock and apply that exact matrix to Full so
    # Full-vs-Stock geometry is not contaminated by two independent ICP fits.
    aligned_stock, direct_alignment = rigid_icp(
        meshes["direct_stock"],
        target,
        seed=int(alignment_seed) + 202,
        candidate_samples=int(candidate_samples),
        final_samples=int(alignment_samples),
        candidate_iterations=int(candidate_iterations),
        final_iterations=int(final_iterations),
    )
    direct_matrix = finite_affine(
        direct_alignment["matrix"],
        label="direct shared rigid alignment",
    )
    aligned_full = meshes["direct_full"].copy()
    aligned_full.apply_transform(direct_matrix)
    recon_alignment = {
        **recon_alignment,
        "gt_assisted": True,
        "policy": (
            "proper SE(3) ICP against GT for visual pose alignment only; "
            "reflection and scale are forbidden"
        ),
        "alignment_group": "reconviagen_independent",
    }
    direct_alignment = {
        **direct_alignment,
        "gt_assisted": True,
        "policy": (
            "one proper SE(3) ICP estimated from Direct stock against GT and "
            "shared unchanged by Direct stock and Direct Full; reflection and "
            "scale are forbidden"
        ),
        "alignment_group": "direct_stock_full_shared",
    }
    full_alignment = {
        **direct_alignment,
        "matrix": direct_matrix.tolist(),
        "derived_from": "direct_stock",
    }
    return {
        "gt": target,
        "reconviagen_original": aligned_recon,
        "direct_stock": aligned_stock,
        "direct_full": aligned_full,
    }, {
        "gt": identity_visual_alignment(
            policy="GT remains fixed during GT-assisted rigid visual alignment"
        ),
        "reconviagen_original": recon_alignment,
        "direct_stock": direct_alignment,
        "direct_full": full_alignment,
    }


def metric_excerpt(record: dict[str, Any], method_id: str) -> dict[str, float] | None:
    if method_id == "gt":
        return None
    surface = record["methods"][method_id]["surface"]
    structure = record["methods"][method_id]["structure"]
    return {
        "chamfer_l1": float(surface["chamfer_l1"]),
        "fscore_0p02": float(surface["fscore_0p02"]),
        "normal_consistency": float(surface["normal_consistency"]),
        "largest_component_ratio": float(structure["largest_component_ratio"]),
    }


def case_complete(case_dir: Path, *, config_sha256: str) -> dict[str, Any] | None:
    path = case_dir / "case_report.json"
    if not path.is_file():
        return None
    report = load_json(path)
    body = dict(report)
    saved = str(body.pop("case_report_sha256", ""))
    if canonical_sha256(body) != saved:
        raise RuntimeError(f"case report hash mismatch: {path}")
    if (
        report.get("format") != FORMAT
        or report.get("complete") is not True
        or report.get("run_config_sha256") != config_sha256
    ):
        raise RuntimeError(f"incompatible completed case: {path}")
    for key, value in report.get("outputs", {}).items():
        validate_binding(value, label=f"{case_dir.name}.outputs.{key}")
    return report


def case_contact_overview(
    case_rows: list[tuple[str, np.ndarray]],
    destination: Path,
) -> None:
    images: list[Image.Image] = []
    font = ImageFont.load_default()
    for pair_id, frame in case_rows:
        image = Image.fromarray(frame)
        header = 28
        panel = Image.new("RGB", (image.width, image.height + header), (18, 18, 18))
        panel.paste(image, (0, header))
        ImageDraw.Draw(panel).text(
            (8, 9),
            pair_id,
            fill=(245, 245, 245),
            font=font,
        )
        images.append(panel)
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), (18, 18, 18))
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    save_image_atomic(sheet, destination)


def html_index(
    *,
    output_dir: Path,
    cases: list[dict[str, Any]],
    overview: Path,
    input_report: Path,
    alignment_mode: str,
) -> str:
    def relative(path: str | Path) -> str:
        return os.path.relpath(Path(path), output_dir).replace(os.sep, "/")

    sections = []
    for case in cases:
        metrics = {
            key: value["metrics"]
            for key, value in case["methods"].items()
            if value["metrics"] is not None
        }
        sections.append(
            "<section>"
            f"<h2>{html.escape(case['pair_id'])} · "
            f"{html.escape(case['uid'])} · {case['view_count']} views</h2>"
            f"<img src=\"{html.escape(relative(case['outputs']['comparison_sheet']['path']))}\">"
            f"<p><a href=\"{html.escape(relative(case['outputs']['comparison_video']['path']))}\">"
            "four-column turntable video</a></p>"
            f"<pre>{html.escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>"
            "</section>"
        )
    alignment_note = (
        "原始 canonical 位姿被保留，没有 GT-assisted 对齐。"
        if alignment_mode == "canonical_pose"
        else (
            "仅为观察形状而做 GT-assisted proper SE(3) 刚体对齐："
            "ReconViaGen 单独求解；Direct stock 求解一份旋转/平移并与 Full 共享。"
            "禁止缩放、镜像和非刚性变形，不能用于评价世界位姿。"
        )
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>GT / ReconViaGen / Direct stock / Direct Full — six-case review</title>
<style>
body {{ background:#141414; color:#eee; font-family:sans-serif; margin:24px; }}
a {{ color:#8cc8ff; }}
img {{ max-width:100%; height:auto; border:1px solid #444; }}
section {{ margin:28px 0 44px; }}
pre {{ background:#202020; padding:12px; overflow:auto; }}
</style>
</head>
<body>
<h1>六例公共相机 Mesh 对比</h1>
<p>列顺序固定为 GT / ReconViaGen original / Direct stock / Direct Full。
同一对象四列共享由 GT 推导的显示变换和完全相同的相机轨迹；没有逐方法 bbox 对齐。</p>
<p>{html.escape(alignment_note)}</p>
<p>输入评估报告：<code>{html.escape(str(input_report))}</code></p>
<img src="{html.escape(relative(overview))}">
{''.join(sections)}
</body>
</html>
"""


def render_case(
    *,
    record: dict[str, Any],
    case_dir: Path,
    device: torch.device,
    frames: int,
    resolution: int,
    fps: int,
    contact_frames: int,
    display_margin: float,
    run_config_sha256: str,
    alignment_mode: str,
    alignment_seed: int,
    candidate_samples: int,
    alignment_samples: int,
    candidate_iterations: int,
    final_iterations: int,
) -> tuple[dict[str, Any], np.ndarray]:
    pair_id = str(record["pair_id"])
    view_count = int(record["view_count"])
    meshes = {
        "gt": normalized_target_mesh(record["target"]),
        **{
            method_id: canonical_prediction_mesh(
                record["methods"][method_id],
                label=f"{pair_id}.{method_id}",
            )
            for method_id in METHOD_IDS[1:]
        },
    }
    meshes, visual_alignments = align_meshes_for_review(
        meshes,
        alignment_mode=alignment_mode,
        alignment_seed=int(alignment_seed),
        candidate_samples=int(candidate_samples),
        alignment_samples=int(alignment_samples),
        candidate_iterations=int(candidate_iterations),
        final_iterations=int(final_iterations),
    )
    shared_transform = display_transform_from_mesh(
        meshes["gt"],
        float(display_margin),
        owner="normalized_gt",
    )
    method_frames: list[tuple[str, list[np.ndarray]]] = []
    method_reports: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        label = method_label(method_id, view_count=view_count)
        print(f"[six_render] {pair_id} method={method_id}", flush=True)
        arrays, display_audit = render_method(
            meshes[method_id],
            device=device,
            frames=int(frames),
            resolution=int(resolution),
            display_margin=float(display_margin),
            shared_display_transform=shared_transform,
        )
        method_dir = case_dir / "methods" / method_id
        video = method_dir / "normal_turntable.mp4"
        sheet = method_dir / "normal_contact_sheet.png"
        save_video_atomic(arrays, video, fps=int(fps))
        contact_sheet(
            arrays,
            label,
            sheet,
            count=int(contact_frames),
        )
        method_frames.append((label, arrays))
        method_reports[method_id] = {
            "label": label,
            "input_mesh": (
                record["target"]["source_glb"]
                if method_id == "gt"
                else record["methods"][method_id]["mesh"]
            ),
            "mesh_stats_canonical": mesh_stats(meshes[method_id]),
            "visual_alignment": visual_alignments[method_id],
            "display_transform": display_audit,
            "metrics": metric_excerpt(record, method_id),
            "outputs": {
                "video": binding(video),
                "contact_sheet": binding(sheet),
            },
        }
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined = comparison_frames(method_frames)
    comparison_video = case_dir / "normal_side_by_side.mp4"
    comparison_sheet = case_dir / "normal_contact_sheet.png"
    save_video_atomic(combined, comparison_video, fps=int(fps))
    comparison_contact_sheet(
        combined,
        comparison_sheet,
        count=int(contact_frames),
    )
    representative = combined[0]
    outputs = {
        "comparison_video": binding(comparison_video),
        "comparison_sheet": binding(comparison_sheet),
    }
    case_report = {
        "format": FORMAT,
        "complete": True,
        "pair_id": pair_id,
        "uid": str(record["uid"]),
        "object_uid": str(record["object_uid"]),
        "joint_seed": int(record["joint_seed"]),
        "view_count": view_count,
        "alignment_mode": alignment_mode,
        "column_order": list(METHOD_IDS),
        "run_config_sha256": run_config_sha256,
        "shared_display_transform": shared_transform,
        "methods": method_reports,
        "outputs": outputs,
    }
    case_report["case_report_sha256"] = canonical_sha256(case_report)
    atomic_json(case_dir / "case_report.json", case_report)
    del meshes, method_frames, combined
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return case_report, representative


def run(args: argparse.Namespace) -> None:
    input_report_path = Path(args.input_report).resolve()
    output_dir = Path(args.output_dir).resolve()
    input_report, records = validate_input_report(
        input_report_path,
        expected_objects=int(args.expected_objects),
        verify_meshes=True,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "input_report": str(input_report_path),
                    "objects": len(records),
                    "pair_ids": [row["pair_id"] for row in records],
                    "column_order": list(METHOD_IDS),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA rendering was requested but CUDA is unavailable")
    device = torch.device(args.device)
    run_config = {
        "input_report": binding(input_report_path),
        "input_report_sha256": str(input_report["report_sha256"]),
        "expected_objects": int(args.expected_objects),
        "frames": int(args.frames),
        "resolution": int(args.resolution),
        "fps": int(args.fps),
        "contact_frames": int(args.contact_frames),
        "display_margin": float(args.display_margin),
        "device_type": device.type,
        "input_mode": MODE,
        "alignment_mode": str(args.alignment_mode),
        "alignment_seed": int(args.alignment_seed),
        "candidate_samples": int(args.candidate_samples),
        "alignment_samples": int(args.alignment_samples),
        "candidate_iterations": int(args.candidate_iterations),
        "final_iterations": int(args.final_iterations),
        "column_order": list(METHOD_IDS),
        "camera_policy": (
            "one vendored TRELLIS normal turntable path shared by all methods"
        ),
        "display_policy": (
            "one normalized-GT-derived centering/isotropic scale shared by all "
            "four methods within each object; no per-prediction normalization"
        ),
    }
    run_config_sha256 = canonical_sha256(run_config)
    run_config["run_config_sha256"] = run_config_sha256
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.is_file():
        existing = load_json(run_config_path)
        if existing != run_config:
            raise RuntimeError(
                f"existing render run_config differs; preserve output: {output_dir}"
            )
    else:
        atomic_json(run_config_path, run_config)

    if (output_dir / "report.json").is_file():
        report = load_json(output_dir / "report.json")
        if (
            report.get("format") == FORMAT
            and report.get("complete") is True
            and report.get("run_config_sha256") == run_config_sha256
        ):
            print(f"reuse completed render: {output_dir / 'report.json'}")
            return
        raise RuntimeError(f"incompatible report already exists: {output_dir}")

    case_reports: list[dict[str, Any]] = []
    overview_rows: list[tuple[str, np.ndarray]] = []
    for index, record in enumerate(records, start=1):
        pair_id = str(record["pair_id"])
        case_dir = output_dir / "cases" / pair_id
        print(f"[six_render] case {index}/{len(records)} {pair_id}", flush=True)
        existing = case_complete(case_dir, config_sha256=run_config_sha256)
        if existing is None:
            case_report, representative = render_case(
                record=record,
                case_dir=case_dir,
                device=device,
                frames=int(args.frames),
                resolution=int(args.resolution),
                fps=int(args.fps),
                contact_frames=int(args.contact_frames),
                display_margin=float(args.display_margin),
                run_config_sha256=run_config_sha256,
                alignment_mode=str(args.alignment_mode),
                alignment_seed=int(args.alignment_seed) + index * 1000,
                candidate_samples=int(args.candidate_samples),
                alignment_samples=int(args.alignment_samples),
                candidate_iterations=int(args.candidate_iterations),
                final_iterations=int(args.final_iterations),
            )
        else:
            case_report = existing
            representative = np.asarray(
                Image.open(
                    Path(case_report["outputs"]["comparison_sheet"]["path"])
                ).convert("RGB")
            )
            # A resumed case stores a multi-angle sheet.  Crop the first row for
            # the compact all-object overview.
            representative = representative[
                : int(args.resolution) + 34,
                :,
            ]
        case_reports.append(case_report)
        overview_rows.append((pair_id, representative))

    overview = output_dir / "all_six_normal_contact_sheet.png"
    case_contact_overview(overview_rows, overview)
    html_path = output_dir / "index.html"
    atomic_text(
        html_path,
        html_index(
            output_dir=output_dir,
            cases=case_reports,
            overview=overview,
            input_report=input_report_path,
            alignment_mode=str(args.alignment_mode),
        ),
    )
    summary_path = output_dir / "summary.txt"
    summary_lines = [
        "GT / ReconViaGen original / Direct stock / Direct Full six-case render",
        "=" * 75,
        f"objects: {len(case_reports)}",
        f"column order: {' | '.join(METHOD_IDS)}",
        f"visual alignment: {args.alignment_mode}",
        "pose/display: canonical coordinates; one GT-derived display transform "
        "per object shared by all four columns",
        f"overview: {overview}",
        f"HTML: {html_path}",
        "",
    ]
    for case in case_reports:
        summary_lines.append(
            f"{case['pair_id']}: {case['view_count']} views -> "
            f"{case['outputs']['comparison_sheet']['path']}"
        )
    atomic_text(summary_path, "\n".join(summary_lines) + "\n")

    report = {
        "format": FORMAT,
        "complete": True,
        "formal": False,
        "purpose": "six-case human geometry review; no Pixal3D column",
        "input_report": binding(input_report_path),
        "run_config": binding(run_config_path),
        "run_config_sha256": run_config_sha256,
        "object_count": len(case_reports),
        "column_order": list(METHOD_IDS),
        "cases": [
            {
                "pair_id": case["pair_id"],
                "uid": case["uid"],
                "view_count": case["view_count"],
                "case_report": binding(
                    output_dir / "cases" / case["pair_id"] / "case_report.json"
                ),
                "outputs": case["outputs"],
            }
            for case in case_reports
        ],
        "outputs": {
            "overview": binding(overview),
            "html": binding(html_path),
            "summary": binding(summary_path),
        },
        "code_bindings": code_bindings(
            {
                "renderer": Path(__file__).resolve(),
                "shared_render": Path(__file__).resolve().parent
                / "bunny_review"
                / "finalize.py",
            }
        ),
        "scope_guard": (
            "exploratory visual review only; rigid_pose_gt is GT-assisted "
            "rotation/translation for shape inspection and cannot support a "
            "world-pose claim or replace the canonical-pose metrics"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "passed": True,
                "objects": len(case_reports),
                "column_order": list(METHOD_IDS),
                "overview": str(overview),
                "html": str(html_path),
                "report": str(output_dir / "report.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_objects", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--contact_frames", type=int, default=6)
    parser.add_argument("--display_margin", type=float, default=0.9)
    parser.add_argument(
        "--alignment_mode",
        choices=ALIGNMENT_MODES,
        default="canonical_pose",
    )
    parser.add_argument("--alignment_seed", type=int, default=20260730)
    parser.add_argument("--candidate_samples", type=int, default=1000)
    parser.add_argument("--alignment_samples", type=int, default=4000)
    parser.add_argument("--candidate_iterations", type=int, default=8)
    parser.add_argument("--final_iterations", type=int, default=30)
    parser.add_argument("--preflight_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_objects <= 0:
        raise ValueError("--expected_objects must be positive")
    if args.frames <= 0 or args.resolution <= 0 or args.fps <= 0:
        raise ValueError("render frame/resolution/fps values must be positive")
    if args.contact_frames <= 0:
        raise ValueError("--contact_frames must be positive")
    for name in (
        "candidate_samples",
        "alignment_samples",
        "candidate_iterations",
        "final_iterations",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if (
        not math.isfinite(args.display_margin)
        or args.display_margin <= 0
        or args.display_margin > 1.0
    ):
        raise ValueError("--display_margin must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
