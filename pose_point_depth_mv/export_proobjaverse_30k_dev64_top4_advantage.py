#!/usr/bin/env python3
"""Materialize the four strongest held-out Dev64 C-minus-R examples.

The object ranking is frozen by the three-seed mean Chamfer-L1 improvement in
the passed SS30K+SLat30K versus strict ReconViaGen aggregate.  This is an
explicit best-case qualitative export, not a representative sample.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    pair_id,
)
from pose_point_depth_mv.evaluate_proobjaverse_official_reconviagen import (
    _load_contract,
)
from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics
from pose_point_depth_mv.prepare_proobjaverse_official_slat_dino_cache import (
    _load_views_with_audit,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    atomic_json,
    canonical_sha256,
    sha256_file,
)


FORMAT = "pose_point_depth_mv.proobjaverse_30k_dev64_top4_advantage_export.v1"
DEFAULT_OUTPUT = Path(
    "/home/zjr/Tracker/pose_point_depth_mv/outputs2/"
    "ProObjaverse30K_Dev64_CminusR优势最明显Top4_"
    "GT_ReconViaGen_SS30K_SLat30K_输入Mask_20260819_v1"
)
EVAL_ROOT = Path(
    "/data/zjr/proobjaverse_official_30k_heldout_dev64_"
    "ss30k_slat30k_20260818_v1"
)
AGGREGATE = EVAL_ROOT / "abc_r_dev64_aggregate/report.json"
DEV_SPLIT = Path(
    "/data/zjr/proobjaverse_official_slat_protocol30k_"
    "seed20260813_source_relocated_v1/dev.json"
)
CACHE_REPORT = EVAL_ROOT / "cache_dev64_compact_v2/report.json"
ABC_ROOT = EVAL_ROOT / "abc_dev64"
RECON_ROOT = EVAL_ROOT / "strict_reconviagen_dev64"
TARGET_ROOTS = [
    ABC_ROOT / "shard0_0_16/target_mesh_cache",
    ABC_ROOT / "shard1_16_32/target_mesh_cache",
    ABC_ROOT / "shard2_32_48/target_mesh_cache",
    ABC_ROOT / "shard3_48_64/target_mesh_cache",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def export_mesh(path: Path, mesh: trimesh.Trimesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    mesh.export(temporary, file_type="obj")
    os.replace(temporary, path)


def contract_rows() -> list[dict[str, Any]]:
    namespace = argparse.Namespace(
        dev_split=str(DEV_SPLIT),
        cache_report=str(CACHE_REPORT),
        target_report="",
        target_mesh_root="",
        paired_target_cache_roots=",".join(str(path) for path in TARGET_ROOTS),
        paired_targets_cover_all_objects=True,
    )
    return list(_load_contract(namespace)["rows"])


def original_pair_path(index: int, uid: str, seed: int) -> Path:
    shard = int(index) // 16
    start = shard * 16
    end = start + 16
    return (
        ABC_ROOT
        / f"shard{shard}_{start}_{end}"
        / "mesh_pairs"
        / pair_id(uid, seed)
        / "pair_record.json"
    )


def original_recon_path(index: int, uid: str, seed: int) -> Path:
    return (
        RECON_ROOT
        / f"worker_{int(index) % 4:02d}_of_04"
        / "records"
        / uid
        / f"seed_{int(seed)}.json"
    )


def input_contact_sheet(rgb: list[Image.Image], masks: list[Image.Image]) -> Image.Image:
    thumb_w, thumb_h = 224, 224
    label_h = 24
    count = len(rgb)
    canvas = Image.new("RGB", (count * thumb_w, 2 * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for position, (image, mask) in enumerate(zip(rgb, masks)):
        for row, item in enumerate((image.convert("RGB"), mask.convert("RGB"))):
            copy = item.copy()
            copy.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = position * thumb_w + (thumb_w - copy.width) // 2
            y = row * (thumb_h + label_h) + label_h + (thumb_h - copy.height) // 2
            canvas.paste(copy, (x, y))
        draw.text((position * thumb_w + 5, 5), f"input {position:02d}", fill="black")
        draw.text(
            (position * thumb_w + 5, thumb_h + label_h + 5),
            f"mask {position:02d}",
            fill="black",
        )
    return canvas


def prepare(output: Path) -> None:
    aggregate = load_json(AGGREGATE)
    if (
        aggregate.get("format")
        != "pose_point_depth_mv.proobjaverse_official_ss_slat_vs_reconviagen.v1"
        or aggregate.get("passed") is not True
        or aggregate.get("runtime_integrity_passed") is not True
        or int(aggregate.get("common_complete_object_count", -1)) != 64
    ):
        raise RuntimeError("frozen Dev64 A/B/C/R aggregate did not pass")
    comparison = aggregate["comparisons"]["current_full_vs_strict_reconviagen"]
    if (
        comparison.get("comparison_kind") != "C_minus_R_complete_endpoint_comparison"
        or int(comparison.get("object_count", -1)) != 64
        or comparison.get("positive_definition")
        != "positive means candidate is better than baseline"
    ):
        raise RuntimeError("C-minus-R comparison contract differs")
    ranked = sorted(
        comparison["per_object_deltas"].items(),
        key=lambda item: (-float(item[1]["chamfer_l1"]), str(item[0])),
    )
    selected = ranked[:4]
    if len(selected) != 4 or any(float(row[1]["chamfer_l1"]) <= 0 for row in selected):
        raise RuntimeError("top-four C-minus-R ranking is invalid")

    rows = contract_rows()
    by_uid = {str(row["uid"]): row for row in rows}
    if len(by_uid) != 64:
        raise RuntimeError("Dev64 contract UID coverage differs")
    if output.exists():
        plan = load_json(output / "selection.json")
        expected = [uid for uid, _ in selected]
        if plan.get("selected_object_uids") != expected:
            raise RuntimeError("existing best-case selection differs")
        print(json.dumps({"passed": True, "reused": True, "output": str(output)}))
        return
    output.mkdir(parents=True, exist_ok=False)

    result_rows = []
    for rank, (uid, deltas) in enumerate(selected, start=1):
        row = by_uid[uid]
        index = int(row["index"])
        object_dir = output / f"rank_{rank:02d}_{uid[:12]}"
        images_dir = object_dir / "输入图像"
        masks_dir = object_dir / "输入Mask"
        rgba_dir = object_dir / "输入RGBA"
        camera_dir = object_dir / "输入相机元数据"
        mesh_dir = object_dir / "Mesh"
        for path in (images_dir, masks_dir, rgba_dir, camera_dir, mesh_dir):
            path.mkdir(parents=True, exist_ok=False)

        views, archive_audit = _load_views_with_audit(row["render_tar"], uid)
        by_id = {int(view["id"]): view for view in views}
        selected_ids = [int(value) for value in row["selected_view_ids"]]
        if len(selected_ids) != 8 or any(value not in by_id for value in selected_ids):
            raise RuntimeError(f"frozen selected views differ: {uid}")
        rgb_images: list[Image.Image] = []
        mask_images: list[Image.Image] = []
        input_records = []
        with tarfile.open(row["render_tar"], "r") as archive:
            for position, view_id in enumerate(selected_ids):
                rgba_array = np.ascontiguousarray(by_id[view_id]["rgba"], dtype=np.uint8)
                if rgba_array.ndim != 3 or rgba_array.shape[-1] != 4:
                    raise RuntimeError(f"official selected input is not RGBA: {uid}/{view_id}")
                rgba = Image.fromarray(rgba_array, mode="RGBA")
                rgb = Image.fromarray(rgba_array[..., :3], mode="RGB")
                mask = Image.fromarray(rgba_array[..., 3], mode="L")
                stem = f"input_{position:02d}_view_{view_id:03d}"
                rgba_path = rgba_dir / f"{stem}_rgba.png"
                rgb_path = images_dir / f"{stem}_rgb.png"
                mask_path = masks_dir / f"{stem}_mask.png"
                save_image(rgba_path, rgba)
                save_image(rgb_path, rgb)
                save_image(mask_path, mask)
                metadata_member = archive.getmember(f"{uid}/{view_id:03d}.json")
                metadata_bytes = archive.extractfile(metadata_member).read()
                metadata_path = camera_dir / f"{stem}.json"
                metadata_path.write_bytes(metadata_bytes)
                input_records.append(
                    {
                        "position": position,
                        "view_id": view_id,
                        "rgba": str(rgba_path),
                        "rgba_sha256": sha256_file(rgba_path),
                        "rgb": str(rgb_path),
                        "rgb_sha256": sha256_file(rgb_path),
                        "mask": str(mask_path),
                        "mask_sha256": sha256_file(mask_path),
                        "camera_metadata": str(metadata_path),
                        "camera_metadata_sha256": sha256_file(metadata_path),
                    }
                )
                rgb_images.append(rgb)
                mask_images.append(mask)
        sheet_path = object_dir / "输入图像与Mask总览.png"
        save_image(sheet_path, input_contact_sheet(rgb_images, mask_images))

        target_npz = Path(row["target_mesh"]).resolve(strict=True)
        with np.load(target_npz, allow_pickle=False) as payload:
            target_mesh = trimesh.Trimesh(
                vertices=np.asarray(payload["vertices"]),
                faces=np.asarray(payload["faces"]),
                process=False,
            )
        target_structure = mesh_structure_metrics(target_mesh)
        if target_structure.get("mesh_success") is not True:
            raise RuntimeError(f"official GT Mesh is invalid: {uid}")
        target_obj = mesh_dir / "GT_official_StockDecoder.obj"
        export_mesh(target_obj, target_mesh)

        pair_path = original_pair_path(index, uid, 42)
        pair = load_json(pair_path)
        current_rows = [
            value
            for value in pair.get("branches", [])
            if value.get("branch") == "native_trained"
        ]
        if len(current_rows) != 1 or current_rows[0].get("passed") is not True:
            raise RuntimeError(f"original current seed42 record is invalid: {uid}")
        recon_path = original_recon_path(index, uid, 42)
        recon = load_json(recon_path)
        if recon.get("passed") is not True or int(recon.get("seed", -1)) != 42:
            raise RuntimeError(f"original ReconViaGen seed42 record is invalid: {uid}")

        metrics = {
            "format": f"{FORMAT}.object_metrics",
            "rank": rank,
            "selection_rule": (
                "descending three-seed mean C-minus-R Chamfer-L1 improvement; "
                "ties broken by object UID"
            ),
            "explicit_best_case_selection": True,
            "object_index": index,
            "object_uid": uid,
            "selected_view_ids": selected_ids,
            "c_minus_r_three_seed_mean_deltas": deltas,
            "seed42": {
                "current_ss30k_slat30k": current_rows[0]["surface"],
                "strict_reconviagen": recon["surface"],
            },
            "source_records": {
                "current_pair_record": str(pair_path),
                "current_pair_record_sha256": sha256_file(pair_path),
                "strict_reconviagen_record": str(recon_path),
                "strict_reconviagen_record_sha256": sha256_file(recon_path),
            },
            "target": {
                "source_npz": str(target_npz),
                "source_npz_sha256": sha256_file(target_npz),
                "exported_obj": str(target_obj),
                "exported_obj_sha256": sha256_file(target_obj),
                "structure": target_structure,
            },
            "inputs": input_records,
            "input_contact_sheet": str(sheet_path),
            "render_archive_audit": archive_audit,
        }
        metrics_path = object_dir / "metrics.json"
        atomic_json(metrics_path, metrics)
        result_rows.append(
            {
                "rank": rank,
                "object_index": index,
                "object_uid": uid,
                "directory": str(object_dir),
                "c_minus_r_chamfer_l1_improvement": float(deltas["chamfer_l1"]),
                "c_minus_r_fscore_0p02_delta": float(deltas["fscore_0p02"]),
                "c_minus_r_normal_consistency_delta": float(
                    deltas["normal_consistency"]
                ),
                "metrics": str(metrics_path),
                "gt_mesh": str(target_obj),
                "input_count": len(input_records),
            }
        )

    selection = {
        "format": f"{FORMAT}.selection",
        "created_at_utc": utc_now(),
        "passed": True,
        "explicit_best_case_selection": True,
        "representative_or_random_sample": False,
        "selection_rule": (
            "top four object-wise three-seed mean Chamfer-L1 improvements from "
            "C-minus-R complete endpoint comparison"
        ),
        "seed_for_mesh_export": 42,
        "source_aggregate": str(AGGREGATE),
        "source_aggregate_sha256": sha256_file(AGGREGATE),
        "official_protocol_sha256": aggregate["official_protocol_sha256"],
        "selected_object_uids": [uid for uid, _ in selected],
        "objects": result_rows,
    }
    selection["selection_identity_sha256"] = canonical_sha256(selection)
    atomic_json(output / "selection.json", selection)
    print(
        json.dumps(
            {
                "passed": True,
                "reused": False,
                "output": str(output),
                "selected": [
                    [uid, float(deltas["chamfer_l1"])]
                    for uid, deltas in selected
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def copy_checked(source: Path, target: Path, expected_sha256: str) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if sha256_file(source) != expected_sha256:
        raise RuntimeError(f"source Mesh hash differs from runtime record: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    if sha256_file(target) != expected_sha256:
        raise RuntimeError(f"copied Mesh hash differs: {target}")
    mesh = trimesh.load_mesh(target, process=False)
    structure = mesh_structure_metrics(mesh)
    if structure.get("mesh_success") is not True:
        raise RuntimeError(f"copied Mesh is invalid: {target}")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": expected_sha256,
        "structure": structure,
    }


def finalize(output: Path) -> None:
    selection = load_json(output / "selection.json")
    if selection.get("passed") is not True or len(selection.get("objects", [])) != 4:
        raise RuntimeError("best-case selection is incomplete")
    results = []
    for selected in selection["objects"]:
        rank = int(selected["rank"])
        uid = str(selected["object_uid"])
        object_dir = Path(selected["directory"])
        metrics = load_json(Path(selected["metrics"]))
        current_root = object_dir / "runtime/current_seed42"
        current_pair_path = (
            current_root / "mesh_pairs" / pair_id(uid, 42) / "pair_record.json"
        )
        current_pair = load_json(current_pair_path)
        current_rows = [
            row
            for row in current_pair.get("branches", [])
            if row.get("branch") == "native_trained"
        ]
        if len(current_rows) != 1 or current_rows[0].get("passed") is not True:
            raise RuntimeError(f"exported current seed42 branch is invalid: {uid}")
        current_row = current_rows[0]
        if int(current_row.get("seed", -1)) != 42 or not current_row.get("mesh"):
            raise RuntimeError(f"exported current seed42 Mesh is absent: {uid}")

        recon_record_path = (
            object_dir
            / "runtime/reconviagen_all_seeds/records"
            / uid
            / "seed_42.json"
        )
        recon_row = load_json(recon_record_path)
        if (
            recon_row.get("passed") is not True
            or int(recon_row.get("seed", -1)) != 42
            or recon_row.get("predicted_mesh_saved") is not True
            or not recon_row.get("mesh")
        ):
            raise RuntimeError(f"exported ReconViaGen seed42 Mesh is invalid: {uid}")

        original_current = metrics["seed42"]["current_ss30k_slat30k"]
        original_recon = metrics["seed42"]["strict_reconviagen"]
        rerun_drift = {}
        for name, observed, expected in (
            ("current", current_row["surface"], original_current),
            ("reconviagen", recon_row["surface"], original_recon),
        ):
            differences = {
                key: abs(float(observed[key]) - float(expected[key]))
                for key in expected
                if key in observed
            }
            if not differences:
                raise RuntimeError(f"{name} seed42 rerun metrics are absent: {uid}")
            rerun_drift[name] = {
                "absolute_difference_by_metric": differences,
                "max_absolute_difference": max(differences.values()),
                "bit_exact": max(differences.values()) == 0.0,
            }

        rerun_c_minus_r_chamfer = float(recon_row["surface"]["chamfer_l1"]) - float(
            current_row["surface"]["chamfer_l1"]
        )
        if rerun_c_minus_r_chamfer <= 0.0:
            raise RuntimeError(
                "qualitative seed42 replay no longer has positive C-minus-R "
                f"Chamfer-L1 improvement: uid={uid} value={rerun_c_minus_r_chamfer}"
            )

        mesh_dir = object_dir / "Mesh"
        current = copy_checked(
            Path(current_row["mesh"]),
            mesh_dir / "当前模型_SS30K_SLat30K_seed42.obj",
            str(current_row["mesh_sha256"]),
        )
        recon = copy_checked(
            Path(recon_row["mesh"]),
            mesh_dir / "ReconViaGen原版_seed42.obj",
            str(recon_row["mesh_sha256"]),
        )
        gt = Path(selected["gt_mesh"]).resolve(strict=True)
        results.append(
            {
                **selected,
                "gt_mesh": {
                    "path": str(gt),
                    "bytes": gt.stat().st_size,
                    "sha256": sha256_file(gt),
                },
                "current_mesh": current,
                "reconviagen_mesh": recon,
                "seed42_replay_metrics": {
                    "current_ss30k_slat30k": current_row["surface"],
                    "strict_reconviagen": recon_row["surface"],
                },
                "seed42_replay_c_minus_r": {
                    "chamfer_l1_improvement": rerun_c_minus_r_chamfer,
                    "fscore_0p02_delta": float(
                        current_row["surface"]["fscore_0p02"]
                    )
                    - float(recon_row["surface"]["fscore_0p02"]),
                    "normal_consistency_delta": float(
                        current_row["surface"]["normal_consistency"]
                    )
                    - float(recon_row["surface"]["normal_consistency"]),
                },
                "seed42_replay_drift_from_original_quantitative_record": rerun_drift,
                "seed42_replay_identity_locked_but_bit_exact_not_required": True,
            }
        )

    report = {
        "format": FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_count": 4,
        "seed": 42,
        "selection": {
            "explicit_best_case_selection": True,
            "representative_or_random_sample": False,
            "ranking_metric": "three-seed mean C-minus-R Chamfer-L1 improvement",
            "selection_report": str(output / "selection.json"),
            "selection_report_sha256": sha256_file(output / "selection.json"),
        },
        "current_endpoint": (
            "posed-DINO official Native-SS step30000 EMA -> "
            "Native-SLat step30000 EMA -> Stock Mesh decoder"
        ),
        "reconviagen_endpoint": (
            "strict original VGGT -> Stock SS -> Stock SLat -> Stock Mesh decoder"
        ),
        "inputs_per_object": 8,
        "includes": [
            "eight frozen official input RGB images",
            "eight exact alpha masks",
            "eight original RGBA inputs",
            "official GT Mesh decoded by the frozen Stock decoder",
            "strict ReconViaGen seed42 Mesh",
            "SS30K+SLat30K seed42 Mesh",
        ],
        "objects": results,
        "scope_guard": (
            "These four objects were deliberately selected as the largest C-minus-R "
            "Chamfer-L1 gains. They are a best-case qualitative exhibit and must not "
            "be presented as a random, representative, or unbiased Dev64 subset."
        ),
        "replay_numerics_guard": (
            "Exported Meshes are fresh seed42 replays under the exact frozen object, "
            "input, protocol, checkpoint and sampling identities. Sparse CUDA and "
            "FlexiCubes execution is not bit-exact; replay metric drift from the "
            "original quantitative record is reported explicitly. Each exported "
            "replay is required to retain positive C-minus-R Chamfer-L1 improvement."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output / "report.json", report)
    lines = [
        "# ProObjaverse 30K held-out Dev64：C−R 优势最明显 Top4",
        "",
        "这四组按完整端点 C−R 的三种子平均 Chamfer-L1 improvement 降序选择。",
        "它们是明确的 best-case 展示，不是随机样本，也不代表 Dev64 总体分布。",
        "每组均包含8张冻结输入图像、8张mask、GT Mesh、ReconViaGen seed42 Mesh",
        "以及 SS30K+SLat30K seed42 Mesh。",
        "",
    ]
    for row in results:
        lines.extend(
            [
                f"- rank {row['rank']}: `{row['object_uid']}`",
                f"  - C−R Chamfer-L1 improvement: "
                f"{row['c_minus_r_chamfer_l1_improvement']:+.8f}",
                f"  - 目录：`{row['directory']}`",
            ]
        )
    atomic_text(output / "README.md", "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "object_count": 4,
                "report": str(output / "report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "finalize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output = args.output.expanduser().resolve()
    if args.command == "prepare":
        prepare(output)
    else:
        finalize(output)


if __name__ == "__main__":
    main()
