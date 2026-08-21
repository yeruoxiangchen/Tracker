#!/usr/bin/env python3
"""Convert a trusted official v1 cache into lossless compact cache v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv.dino_only_condition import build_dino_only_contexts
from pose_point_depth_mv.direct_slat_flow import DIRECT_SLAT_CACHE_VERSION
from pose_point_depth_mv.prepare_proobjaverse_official_slat_compact_cache import (
    COMPACT_CACHE_REPORT_FORMAT,
    atomic_torch_save,
    make_rows,
    object_paths,
)
from pose_point_depth_mv.proobjaverse_official_slat_compact import (
    COMPACT_LAYOUT_VERSION,
    COMPACT_LIFTING_MANIFEST_FORMAT,
    COMPACT_OBJECT_FORMAT,
    COMPACT_SLAT_MANIFEST_FORMAT,
    sha256_file,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    atomic_json,
    canonical_sha256,
    load_json,
)
from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION


CONVERSION_REPORT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_cache_v1_to_v2.v1"
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slat_manifest_v1", required=True)
    parser.add_argument("--lifting_manifest_v1", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_objects", type=int, default=0)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    slat_path = Path(args.slat_manifest_v1).expanduser().resolve()
    lifting_path = Path(args.lifting_manifest_v1).expanduser().resolve()
    slat = load_json(slat_path)
    lifting = load_json(lifting_path)
    if slat.get("format") != DIRECT_SLAT_CACHE_VERSION:
        raise ValueError(f"expected v1 direct SLat cache, got {slat.get('format')!r}")
    if lifting.get("format") != LIFTING_CACHE_VERSION:
        raise ValueError(f"expected v1 lifting cache, got {lifting.get('format')!r}")
    rows = list(slat["samples"])
    if int(args.max_objects) > 0:
        rows = rows[: int(args.max_objects)]
    lifting_by_uid = {str(row["uid"]): row for row in lifting["samples"]}
    if len(lifting_by_uid) != len(lifting["samples"]):
        raise ValueError("v1 lifting manifest has duplicate UIDs")
    missing = [str(row["uid"]) for row in rows if str(row["uid"]) not in lifting_by_uid]
    if missing:
        raise ValueError(f"v1 SLat/lifting join incomplete: {missing[:8]}")
    slat_root = Path(slat.get("output_dir", slat_path.parent)).resolve()
    lifting_root = Path(lifting.get("output_dir", lifting_path.parent)).resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"converter output must be a new empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    slat_config_hash = str(slat["config_hash"])
    lifting_config_hash = str(lifting["config_hash"])
    new_slat_rows: list[dict[str, Any]] = []
    new_lifting_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        uid = str(row["uid"])
        object_uid = str(row["object_uid"])
        if uid != object_uid:
            raise ValueError(f"official v1 row is not one-object/one-UID: {uid}")
        lifting_row = lifting_by_uid[uid]
        lifting_file = _resolve(lifting_root, lifting_row["cache_file"])
        condition_file = _resolve(slat_root, row["condition_file"])
        support_file = _resolve(slat_root, row["support_file"])
        physical_file = _resolve(slat_root, row["physical_file"])
        old_lifting = torch.load(
            lifting_file, map_location="cpu", weights_only=False
        )
        old_condition = torch.load(
            condition_file, map_location="cpu", weights_only=False
        )
        old_support = torch.load(support_file, map_location="cpu", weights_only=False)
        old_physical = torch.load(
            physical_file, map_location="cpu", weights_only=False
        )
        if old_support.get("unused_native_ss_placeholder") is not True:
            raise ValueError(f"uid={uid} v1 support is not official GT placeholder")
        if old_physical.get("unused_native_placeholder") is not True:
            raise ValueError(f"uid={uid} v1 physical payload is not a placeholder")
        visual = old_lifting["visual_patch_features"]
        contexts = build_dino_only_contexts(visual, ss_context_tokens=4096)
        condition = old_condition["condition"]
        if not all(
            torch.equal(left, right)
            for left, right in zip(
                contexts["slat_condition"]["cond"], condition["cond"]
            )
        ):
            raise RuntimeError(f"uid={uid} v1 positive context is not reconstructible")
        if not all(
            torch.equal(left, right)
            for left, right in zip(
                contexts["slat_condition"]["neg_cond"], condition["neg_cond"]
            )
        ):
            raise RuntimeError(f"uid={uid} v1 negative context is not reconstructible")
        if not torch.equal(contexts["stock_condition"], old_lifting["stock_condition"]):
            raise RuntimeError(f"uid={uid} v1 Stock condition is not reconstructible")
        if torch.count_nonzero(old_lifting["predicted_depth"]).item() != 0:
            raise RuntimeError(f"uid={uid} v1 official depth is not all-zero")
        if torch.count_nonzero(old_lifting["depth_confidence"]).item() != 0:
            raise RuntimeError(f"uid={uid} v1 official confidence is not all-zero")
        if torch.count_nonzero(old_lifting["prior_confidence"]).item() != 0:
            raise RuntimeError(f"uid={uid} v1 official prior confidence is not all-zero")
        paths = object_paths(output, uid)
        payload = {
            "format": COMPACT_OBJECT_FORMAT,
            "layout": COMPACT_LAYOUT_VERSION,
            "uid": uid,
            "object_uid": object_uid,
            "slat_config_hash": slat_config_hash,
            "lifting_config_hash": lifting_config_hash,
            "visual_patch_features": visual,
            "intrinsics": old_lifting["intrinsics"],
            "extrinsics": old_lifting["extrinsics"],
            "image_size": list(map(int, old_lifting["predicted_depth"].shape[-2:])),
            "grid_transform": str(old_lifting["grid_transform"]),
            "extrinsics_type": str(old_lifting["extrinsics_type"]),
            "camera_forward_sign": float(old_lifting["camera_forward_sign"]),
            "view_ids": old_lifting["view_ids"],
            "source_intrinsics": old_lifting["source_intrinsics"],
            "source_to_feature_affines": old_lifting[
                "source_to_feature_affines"
            ],
            "preprocessing": dict(old_lifting["preprocessing"]),
            "context_contract": contexts["context_contract"],
            "official_gt_support_only": True,
        }
        atomic_torch_save(paths["compact"], payload)
        target_file = _resolve(slat_root, row["target_file"])
        slat_row, compact_lifting_row = make_rows(
            output=output,
            uid=uid,
            target_path=target_file,
            compact_path=paths["compact"],
        )
        record = {
            "uid": uid,
            "candidate_view_count": None,
            "selected_view_ids": old_lifting["view_ids"].to(torch.int64).tolist(),
            "camera_forward_sign": float(old_lifting["camera_forward_sign"]),
            "official_camera_conversion": "preserved_from_v1",
            "coord_count": int(old_support["corrected_coords64"].shape[0]),
            "visual_shape": list(visual.shape),
            "compact_bytes": paths["compact"].stat().st_size,
            "render_archive_complete": None,
            "render_archive_recovered": None,
            "render_archive_read_error": "converter did not re-read render tar",
            "cache_reused_after_interruption": False,
        }
        atomic_json(
            paths["record"],
            {
                "format": (
                    "pose_point_depth_mv.proobjaverse_official_slat_compact_record.v2"
                ),
                "uid": uid,
                "slat_config_hash": slat_config_hash,
                "lifting_config_hash": lifting_config_hash,
                "slat_row": slat_row,
                "lifting_row": compact_lifting_row,
                "record": record,
            },
        )
        new_slat_rows.append(slat_row)
        new_lifting_rows.append(compact_lifting_row)
        records.append(record)
        print(f"[official_slat_v1_to_v2] {position}/{len(rows)} {uid}", flush=True)
    new_slat = {
        **slat,
        "format": COMPACT_SLAT_MANIFEST_FORMAT,
        "layout": COMPACT_LAYOUT_VERSION,
        "output_dir": str(output),
        "samples": new_slat_rows,
        "sample_count": len(new_slat_rows),
        "object_count": len(new_slat_rows),
    }
    new_lifting = {
        **lifting,
        "format": COMPACT_LIFTING_MANIFEST_FORMAT,
        "layout": COMPACT_LAYOUT_VERSION,
        "output_dir": str(output),
        "stock_condition_source": "runtime_deterministic_from_single_fp16_dino",
        "lifting_feature_source": "official_pose_raw_dino_compact_v2",
        "samples": new_lifting_rows,
        "sample_count": len(new_lifting_rows),
        "object_count": len(new_lifting_rows),
    }
    new_slat_path = output / "slat_manifest.json"
    new_lifting_path = output / "lifting_manifest.json"
    atomic_json(new_slat_path, new_slat)
    atomic_json(new_lifting_path, new_lifting)
    report = {
        "format": CONVERSION_REPORT_FORMAT,
        "passed": True,
        "source_slat_manifest": str(slat_path),
        "source_slat_manifest_sha256": sha256_file(slat_path),
        "source_lifting_manifest": str(lifting_path),
        "source_lifting_manifest_sha256": sha256_file(lifting_path),
        "slat_manifest": str(new_slat_path),
        "slat_manifest_sha256": sha256_file(new_slat_path),
        "lifting_manifest": str(new_lifting_path),
        "lifting_manifest_sha256": sha256_file(new_lifting_path),
        "object_count": len(rows),
        "compact_payload_bytes": sum(row["compact_bytes"] for row in records),
        "exact_reconstruction_checks": {
            "positive_slat_condition": True,
            "negative_slat_condition": True,
            "stock_ss_condition": True,
            "projection_intrinsics_extrinsics": True,
            "target_lh_slat_referenced_unchanged": True,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
