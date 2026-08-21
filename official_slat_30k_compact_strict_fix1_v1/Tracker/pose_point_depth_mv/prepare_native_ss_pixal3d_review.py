#!/usr/bin/env python3
"""Freeze source-balanced Native-SS/Stock-SLAT cases for Pixal3D review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .compare_pixal3d_singleview_smoke import (
    FORMAT,
    atomic_json,
    binding,
    canonical_sha256,
    load_canonical_target,
    mask_array,
    sha256_file,
    validate_protocol,
)


T3_FORMAT = "pose_point_depth_mv.native_ss_stock_slat_mesh_transfer.v1"
SPLIT_FORMAT = "pose_point_depth_mv.native_ss_sourcebalanced_split.v1"
SOURCE_ORDER = (
    "legacy_objaverse",
    "gap_objaverse",
    "pilot_objaverse",
    "omni",
)
DEFAULT_SOURCE_VIEW_TARGETS = (
    ("legacy_objaverse", 2),
    ("gap_objaverse", 8),
    ("pilot_objaverse", 4),
    ("omni", 4),
)


def parse_source_view_targets(value: str) -> list[tuple[str, int]]:
    output: list[tuple[str, int]] = []
    for item in str(value).split(","):
        source, separator, views_text = item.strip().partition(":")
        if not separator:
            raise ValueError(
                "--source_view_targets entries must use source:view_count"
            )
        views = int(views_text)
        if source not in SOURCE_ORDER or views not in {2, 4, 8}:
            raise ValueError(f"unsupported source/view target: {item!r}")
        output.append((source, views))
    if not output or len(output) != len(set(output)):
        raise ValueError("source/view targets must be non-empty and unique")
    sources = [source for source, _ in output]
    if len(sources) != len(set(sources)):
        raise ValueError("each source may appear only once")
    return sorted(output, key=lambda item: SOURCE_ORDER.index(item[0]))


def selection_rank(seed: int, source: str, view_count: int, uid: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}|{source}|{int(view_count)}|{uid}".encode("utf-8")
    ).hexdigest()


def resolve_cache_file(manifest: dict[str, Any], row: dict[str, Any]) -> Path:
    path = Path(str(row["cache_file"]))
    if not path.is_absolute():
        path = Path(str(manifest["output_dir"])) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def tensor_list(value: Any) -> list[Any]:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).tolist()


def save_rgba_atomic(rgb_path: Path, mask_path: Path, output: Path) -> dict[str, int]:
    rgb = Image.open(rgb_path).convert("RGB")
    alpha_array = mask_array(mask_path)
    alpha = Image.fromarray(alpha_array, mode="L")
    if alpha.size != rgb.size:
        alpha = alpha.resize(rgb.size, Image.Resampling.NEAREST)
        alpha_array = np.asarray(alpha)
    if not np.any(alpha_array > 204) or not np.any(alpha_array < 255):
        raise RuntimeError(
            f"Pixal3D --skip_rembg input needs foreground and transparency: {mask_path}"
        )
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp-{os.getpid()}.png")
    rgba.save(temporary)
    os.replace(temporary, output)
    return {
        "width": int(rgb.width),
        "height": int(rgb.height),
        "foreground_pixels": int(np.count_nonzero(alpha_array > 127)),
    }


def export_target_atomic(ss_latent: str | Path, output: Path) -> dict[str, Any]:
    mesh, metadata = load_canonical_target(ss_latent)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp-{os.getpid()}.obj")
    try:
        mesh.export(
            temporary,
            file_type="obj",
            include_color=False,
            include_texture=False,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"empty canonical GT export: {temporary}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def validate_native_protocol(path: Path) -> dict[str, Any]:
    protocol = validate_protocol(path.resolve())
    comparison = protocol.get("comparison", {})
    if comparison.get("selection_mode") != "source_view_balanced_sha256_v1":
        raise RuntimeError("unexpected Native review selection mode")
    if comparison.get("mesh_branches") != {"full": "native", "stock": "stock"}:
        raise RuntimeError("unexpected Native/Stock branch mapping")
    for case in protocol["cases"]:
        for key in (
            "native_full_mesh",
            "stock_mesh",
            "pair_record",
            "cache_payload",
        ):
            item = case.get(key)
            if not isinstance(item, dict):
                raise RuntimeError(f"case={case['case_id']} lacks {key} binding")
            source = Path(item["path"])
            if not source.is_file() or sha256_file(source) != item["sha256"]:
                raise RuntimeError(
                    f"frozen Native case binding changed: {case['case_id']}.{key}"
                )
        if case["current_mesh"] != case["native_full_mesh"]:
            raise RuntimeError("Pixal protocol current_mesh must bind Native Full")
    return protocol


def selected_rows(
    *,
    split: dict[str, Any],
    manifest_by_uid: dict[str, dict[str, Any]],
    targets: list[tuple[str, int]],
    selection_seed: int,
    max_gt_source_bytes: int,
) -> list[dict[str, Any]]:
    rows = list(split.get("phases", {}).get("final", []))
    output = []
    for source, view_count in targets:
        candidates = [
            dict(row)
            for row in rows
            if str(row.get("source")) == source
            and int(row.get("view_count", -1)) == view_count
        ]
        eligible = []
        for row in candidates:
            manifest_row = manifest_by_uid.get(str(row["uid"]))
            if manifest_row is None:
                raise KeyError(f"split UID absent from cache manifest: {row['uid']}")
            with np.load(Path(str(manifest_row["ss_latent"])).resolve()) as latent:
                source_mesh = Path(str(latent["source_glb"])).resolve()
            if not source_mesh.is_file():
                raise FileNotFoundError(source_mesh)
            row["gt_source_mesh_size_bytes"] = int(source_mesh.stat().st_size)
            row["gt_source_mesh"] = str(source_mesh)
            if row["gt_source_mesh_size_bytes"] <= int(max_gt_source_bytes):
                eligible.append(row)
        if not eligible:
            raise RuntimeError(
                f"final split has no render-eligible source={source}, "
                f"view_count={view_count} candidate under "
                f"max_gt_source_bytes={max_gt_source_bytes}"
            )
        for row in eligible:
            row["selection_rank"] = selection_rank(
                selection_seed, source, view_count, str(row["uid"])
            )
        output.append(min(eligible, key=lambda row: row["selection_rank"]))
    return output


def command_prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    protocol_path = output_dir / "protocol.json"
    targets = parse_source_view_targets(args.source_view_targets)
    if int(args.max_gt_source_mib) <= 0:
        raise ValueError("--max_gt_source_mib must be positive")
    if protocol_path.is_file():
        protocol = validate_native_protocol(protocol_path)
        expected = {
            "t3_report": binding(args.t3_report.resolve()),
            "cache_manifest": binding(args.cache_manifest.resolve()),
            "split_audit": binding(args.split_audit.resolve()),
        }
        checks = {
            key: protocol["bindings"].get(key) == value
            for key, value in expected.items()
        }
        checks.update(
            {
                "seed": protocol["comparison"].get("mesh_seed") == int(args.seed),
                "selection_seed": protocol["comparison"].get("selection_seed")
                == int(args.selection_seed),
                "source_view_targets": [
                    [source, views] for source, views in targets
                ]
                == protocol["comparison"].get("source_view_targets"),
                "max_gt_source_mib": protocol["comparison"].get(
                    "max_gt_source_mib"
                )
                == int(args.max_gt_source_mib),
            }
        )
        if not all(checks.values()):
            raise RuntimeError(
                f"existing protocol arguments changed: {checks}; use a fresh output"
            )
        print(
            json.dumps(
                {
                    "status": "reused",
                    "protocol": str(protocol_path),
                    "protocol_sha256": protocol["protocol_sha256"],
                    "cases": [
                        {
                            key: case[key]
                            for key in ("case_id", "source", "uid", "view_count")
                        }
                        for case in protocol["cases"]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"non-empty output has no reusable protocol: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    t3_path = args.t3_report.resolve()
    manifest_path = args.cache_manifest.resolve()
    split_path = args.split_audit.resolve()
    t3 = json.loads(t3_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if t3.get("format") != T3_FORMAT or t3.get("runtime_passed") is not True:
        raise RuntimeError("T3 report is unsupported or runtime-incomplete")
    if split.get("format") != SPLIT_FORMAT or split.get("passed") is not True:
        raise RuntimeError("source-balanced split audit is unsupported or failed")
    run_identity = t3.get("run_identity", {})
    if (
        Path(str(run_identity.get("cache_manifest", ""))).resolve() != manifest_path
        or run_identity.get("cache_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("T3 report is not bound to the requested cache manifest")
    split_identity = run_identity.get("split", {})
    if (
        Path(str(split_identity.get("path", ""))).resolve() != split_path
        or split_identity.get("sha256") != sha256_file(split_path)
        or split_identity.get("phase") != "final"
    ):
        raise RuntimeError("T3 report is not bound to the requested final split")

    manifest_by_uid = {str(row["uid"]): row for row in manifest["samples"]}
    support_by_key = {
        (str(row["uid"]), int(row["seed"])): row
        for row in t3.get("ss_support_records", [])
    }
    chosen = selected_rows(
        split=split,
        manifest_by_uid=manifest_by_uid,
        targets=targets,
        selection_seed=int(args.selection_seed),
        max_gt_source_bytes=int(args.max_gt_source_mib) * 1024 * 1024,
    )
    t3_root = t3_path.parent
    cases = []
    for position, selected in enumerate(chosen):
        uid = str(selected["uid"])
        source = str(selected["source"])
        view_count = int(selected["view_count"])
        manifest_row = manifest_by_uid.get(uid)
        if manifest_row is None:
            raise KeyError(f"selected UID absent from cache manifest: {uid}")
        support = support_by_key.get((uid, int(args.seed)))
        if support is None or support.get("passed") is not True:
            raise RuntimeError(f"T3 lacks a passing seed={args.seed} row for {uid}")
        pair_id = str(support["pair_id"])
        pair_path = (t3_root / "mesh_pairs" / pair_id / "pair_record.json").resolve()
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        if (
            pair.get("passed") is not True
            or str(pair.get("uid")) != uid
            or int(pair.get("seed", -1)) != int(args.seed)
        ):
            raise RuntimeError(f"T3 pair record identity failed: {pair_path}")
        branches = {str(row["branch"]): row for row in pair.get("branches", [])}
        if set(branches) != {"stock", "native"}:
            raise RuntimeError(f"T3 pair lacks exact stock/native branches: {pair_path}")
        native_path = Path(str(branches["native"]["canonical_obj"])).resolve()
        stock_path = Path(str(branches["stock"]["canonical_obj"])).resolve()
        for branch, mesh_path in (("native", native_path), ("stock", stock_path)):
            if (
                not mesh_path.is_file()
                or branches[branch].get("canonical_obj_sha256")
                != sha256_file(mesh_path)
            ):
                raise RuntimeError(f"T3 {branch} Mesh binding failed: {mesh_path}")

        cache_path = resolve_cache_file(manifest, manifest_row)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            str(payload.get("uid")) != uid
            or str(payload.get("object_uid")) != str(selected["object_uid"])
            or str(payload.get("extrinsics_type")) != "c2w"
        ):
            raise RuntimeError(f"lifting cache identity/frame failed: {cache_path}")
        image_paths = [Path(str(value)).resolve() for value in payload["image_paths"]]
        mask_paths = [Path(str(value)).resolve() for value in payload["mask_paths"]]
        if len(image_paths) != view_count or len(mask_paths) != view_count:
            raise RuntimeError(f"cache view count differs for uid={uid}")
        mask_areas = [
            int(np.count_nonzero(mask_array(mask_path) > 127))
            for mask_path in mask_paths
        ]
        selected_input_position = max(
            range(view_count), key=lambda index: (mask_areas[index], -index)
        )
        image_path = image_paths[selected_input_position]
        mask_path = mask_paths[selected_input_position]
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"selected source image/mask missing for uid={uid}")

        short_uid = str(selected["object_uid"]).replace("objaverse_", "")[:12]
        case_id = f"n{position:02d}_{source}_v{view_count:02d}_{short_uid}"
        rgba_path = (output_dir / "inputs" / f"{case_id}.png").resolve()
        image_metadata = save_rgba_atomic(image_path, mask_path, rgba_path)
        target_path = (output_dir / "targets" / f"{case_id}.obj").resolve()
        target_metadata = export_target_atomic(payload["ss_latent"], target_path)

        extrinsics = tensor_list(payload["extrinsics"])
        intrinsics = tensor_list(payload["source_intrinsics"])
        view_ids = [int(value) for value in tensor_list(payload["view_ids"])]
        selected_extrinsic = extrinsics[selected_input_position]
        selected_intrinsic = intrinsics[selected_input_position]
        if np.asarray(selected_extrinsic).shape != (4, 4):
            raise RuntimeError(f"invalid selected c2w for uid={uid}")
        if np.asarray(selected_intrinsic).shape != (3, 3):
            raise RuntimeError(f"invalid selected intrinsic for uid={uid}")
        native_binding = binding(native_path)
        cases.append(
            {
                "case_id": case_id,
                "uid": uid,
                "object_uid": str(selected["object_uid"]),
                "source": source,
                "view_count": view_count,
                "current_seed": int(args.seed),
                "pixal3d_seed": int(args.seed),
                "dataset_index": int(manifest["samples"].index(manifest_row)),
                "pair_id": pair_id,
                "selection_rank": str(selected["selection_rank"]),
                "gt_source_mesh_size_bytes": int(
                    selected["gt_source_mesh_size_bytes"]
                ),
                "single_view_policy": "largest foreground mask among Native inputs",
                "selected_input_position": int(selected_input_position),
                "selected_view_id": int(view_ids[selected_input_position]),
                "selected_frame": {
                    "extrinsics_type": "c2w",
                    "extrinsic": selected_extrinsic,
                    "source_view_index": int(view_ids[selected_input_position]),
                },
                "selected_intrinsic": selected_intrinsic,
                "mask_foreground_pixels": image_metadata["foreground_pixels"],
                "source_width": image_metadata["width"],
                "source_height": image_metadata["height"],
                "source_fx_pixels": float(selected_intrinsic[0][0]),
                "input_rgba": binding(rgba_path),
                "source_image": binding(image_path),
                "source_mask": binding(mask_path),
                # Existing official Pixal inference validates this compatibility key.
                "current_mesh": native_binding,
                "native_full_mesh": native_binding,
                "stock_mesh": binding(stock_path),
                "target_mesh": binding(target_path),
                "target_metadata": target_metadata,
                "pair_record": binding(pair_path),
                "cache_payload": binding(cache_path),
                "t3_ss_support": support,
                "t3_mesh_surface": {
                    branch: branches[branch].get("surface")
                    for branch in ("native", "stock")
                },
            }
        )

    body = {
        "format": FORMAT,
        "formal": False,
        "purpose": (
            "source-balanced human Mesh review of canonical GT, Native SS + frozen "
            "Stock SLAT Full, Stock SS + Stock SLAT, and official-final Pixal3D"
        ),
        "comparison": {
            "selection_mode": "source_view_balanced_sha256_v1",
            "selection_seed": int(args.selection_seed),
            "selection_is_performance_independent": True,
            "selection_pool": "untouched source-balanced final32 T3 objects",
            "source_view_targets": [
                [source, views] for source, views in targets
            ],
            "max_gt_source_mib": int(args.max_gt_source_mib),
            "gt_size_filter": (
                "source Mesh file-size operational guard applied before SHA256 "
                "ranking; no model output or metric is read"
            ),
            "object_count": len(cases),
            "mesh_seed": int(args.seed),
            "pixal3d_input_budget": 1,
            "single_view_policy": (
                "largest foreground mask among the exact Native input views; "
                "deterministic and favorable to the single-view baseline"
            ),
            "mesh_branches": {"full": "native", "stock": "stock"},
            "primary_review": (
                "canonical_pose with one GT-owned display transform; shape_aligned "
                "is separately labeled GT-assisted inspection"
            ),
        },
        "bindings": {
            "t3_report": binding(t3_path),
            "cache_manifest": binding(manifest_path),
            "split_audit": binding(split_path),
            "prepare_code": binding(Path(__file__).resolve()),
        },
        "cases": cases,
    }
    body["protocol_sha256"] = canonical_sha256(body)
    atomic_json(protocol_path, body)
    protocol = validate_native_protocol(protocol_path)
    print(
        json.dumps(
            {
                "status": "created",
                "protocol": str(protocol_path),
                "protocol_sha256": protocol["protocol_sha256"],
                "cases": [
                    {
                        key: case[key]
                        for key in ("case_id", "source", "uid", "view_count")
                    }
                    for case in protocol["cases"]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def command_verify(args: argparse.Namespace) -> None:
    protocol = validate_native_protocol(args.protocol.resolve())
    print(
        json.dumps(
            {
                "passed": True,
                "protocol": str(args.protocol.resolve()),
                "protocol_sha256": protocol["protocol_sha256"],
                "cases": [
                    {
                        key: case[key]
                        for key in ("case_id", "source", "uid", "view_count")
                    }
                    for case in protocol["cases"]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="freeze four review cases")
    prepare.add_argument("--t3_report", type=Path, required=True)
    prepare.add_argument("--cache_manifest", type=Path, required=True)
    prepare.add_argument("--split_audit", type=Path, required=True)
    prepare.add_argument("--output_dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--selection_seed", type=int, default=20260802)
    prepare.add_argument("--max_gt_source_mib", type=int, default=128)
    prepare.add_argument(
        "--source_view_targets",
        default=",".join(
            f"{source}:{views}" for source, views in DEFAULT_SOURCE_VIEW_TARGETS
        ),
    )
    prepare.set_defaults(handler=command_prepare)
    verify = subparsers.add_parser("verify", help="verify every frozen binding")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.set_defaults(handler=command_verify)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
