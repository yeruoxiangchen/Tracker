#!/usr/bin/env python3
"""Build the lossless compact-v2 official no-VGGT Native-SLat cache.

Unlike the legacy builder, each object stores exactly one fp16 DINO tensor plus
camera/projection metadata.  Positive/negative SLat contexts and the Stock-SS
context are reconstructed by the shared deterministic helper at runtime.
Large all-zero depth/confidence/support placeholders are not serialized.

The builder supports independent GPU workers.  Workers write disjoint object
artifacts; a final CPU-only invocation validates every object and writes the
two manifests.  No worker is allowed to publish a partial manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ar_ss_flow.pose_lifting import LIFTING_METADATA_NAMES, schema_hash
from ar_ss_flow.shared_object_preprocessing import (
    canonical_json_sha256 as preprocessing_sha256,
    prepare_shared_object_arrays,
    transform_intrinsics,
)
from pose_aligned_reconstruction.dataset_tools.prepare_omni_real_dino_only_model_inputs import (
    FEATURE_CONTRACT,
    _build_dino_pipeline,
)
from pose_aligned_reconstruction.dino_only_condition import (
    build_dino_only_contexts,
    dino_only_feature_metadata,
)
from pose_aligned_reconstruction.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_aligned_reconstruction.prepare_proobjaverse_official_slat_dino_cache import (
    _front_sign,
    _load_views_with_audit,
    _make_config,
    _pose_diverse_indices,
)
from pose_aligned_reconstruction.proobjaverse_official_slat_compact import (
    COMPACT_LAYOUT_VERSION,
    COMPACT_LIFTING_MANIFEST_FORMAT,
    COMPACT_OBJECT_FORMAT,
    COMPACT_SLAT_MANIFEST_FORMAT,
    load_compact_object,
    sha256_file,
)
from pose_aligned_reconstruction.proobjaverse_official_slat_protocol import (
    SPLIT_FORMAT,
    atomic_json,
    canonical_sha256,
    load_json,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond


COMPACT_CACHE_REPORT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_compact_cache.v2"
)
COMPACT_OBJECT_RECORD_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_compact_record.v2"
)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def object_paths(output: Path, uid: str) -> dict[str, Path]:
    root = output / "objects" / uid[:2] / uid
    return {
        "root": root,
        "compact": root / "dino_geometry.pt",
        "record": root / "cache_record.json",
    }


def make_rows(
    *, output: Path, uid: str, target_path: Path, compact_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_hash = sha256_file(target_path)
    compact_hash = sha256_file(compact_path)
    compact_relative = str(compact_path.relative_to(output))
    slat_row = {
        "uid": uid,
        "object_uid": uid,
        "support_seed": 42,
        "target_file": str(target_path),
        "target_file_sha256": target_hash,
        "source_lh_slat": str(target_path),
        "source_lh_slat_sha256": target_hash,
        "compact_file": compact_relative,
        "compact_file_sha256": compact_hash,
    }
    lifting_row = {
        "uid": uid,
        "object_uid": uid,
        "compact_file": compact_relative,
        "compact_file_sha256": compact_hash,
    }
    return slat_row, lifting_row


def validate_cached_object(
    *,
    output: Path,
    source: dict[str, Any],
    slat_config_hash: str,
    lifting_config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    uid = str(source["uid"])
    target_path = Path(source["slat_npz"]).expanduser().resolve()
    paths = object_paths(output, uid)
    if not paths["compact"].is_file() and not paths["record"].is_file():
        return None
    if not paths["compact"].is_file() or not paths["record"].is_file():
        raise RuntimeError(f"uid={uid} compact cache is partially materialized")
    record_payload = load_json(paths["record"])
    expected = {
        "format": COMPACT_OBJECT_RECORD_FORMAT,
        "uid": uid,
        "slat_config_hash": slat_config_hash,
        "lifting_config_hash": lifting_config_hash,
    }
    actual = {key: record_payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"uid={uid} compact sidecar identity mismatch")
    compact = load_compact_object(
        paths["compact"],
        uid=uid,
        object_uid=uid,
        slat_config_hash=slat_config_hash,
        lifting_config_hash=lifting_config_hash,
    )
    slat_row, lifting_row = make_rows(
        output=output,
        uid=uid,
        target_path=target_path,
        compact_path=paths["compact"],
    )
    if record_payload.get("slat_row") != slat_row:
        raise RuntimeError(f"uid={uid} compact SLat row/hash changed")
    if record_payload.get("lifting_row") != lifting_row:
        raise RuntimeError(f"uid={uid} compact lifting row/hash changed")
    record = dict(record_payload["record"])
    if list(compact["view_ids"].to(torch.int64).tolist()) != record[
        "selected_view_ids"
    ]:
        raise RuntimeError(f"uid={uid} compact view identity changed")
    return slat_row, lifting_row, record


def build_one(
    *,
    output: Path,
    source: dict[str, Any],
    pipeline: Any,
    selected_views: int,
    ss_context_tokens: int,
    slat_config_hash: str,
    lifting_config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    uid = str(source["uid"])
    target_path = Path(source["slat_npz"]).expanduser().resolve()
    with np.load(target_path, allow_pickle=False) as target:
        coords3 = np.asarray(target["coords"], dtype=np.int32)
    views, archive_audit = _load_views_with_audit(
        Path(source["render_tar"]).expanduser().resolve(), uid
    )
    if len(views) < int(selected_views):
        raise RuntimeError(
            f"uid={uid} has {len(views)} complete views; need {selected_views}"
        )
    selected_indices = _pose_diverse_indices(views, int(selected_views), uid)
    selected = [views[index] for index in selected_indices]
    rgba = [row["rgba"] for row in selected]
    shared = prepare_shared_object_arrays(
        [value[..., :3] for value in rgba],
        [value[..., 3] for value in rgba],
        resolution=518,
        foreground_margin=1.10,
        alpha_threshold=0.80,
    )
    source_intrinsics = np.stack([row["intrinsic"] for row in selected])
    intrinsics = transform_intrinsics(
        source_intrinsics, shared.source_to_feature_affines
    )
    extrinsics = np.stack([row["extrinsic"] for row in selected])
    forward_sign = _front_sign(extrinsics)
    raw = pipeline.encode_image(shared.images)
    encoded = normalize_image_cond(raw, batch=1, views=len(selected))
    dino_patch = encoded[0, :, int(FEATURE_CONTRACT["patch_start_idx"]) :]
    contexts = build_dino_only_contexts(
        dino_patch, ss_context_tokens=int(ss_context_tokens)
    )
    visual = contexts["visual_patch_features"].to(torch.float16).cpu()
    # Rebuild once after fp16 conversion.  This is the exact runtime source;
    # the contract is stored so loader-side reconstruction is auditable.
    contexts = build_dino_only_contexts(
        visual, ss_context_tokens=int(ss_context_tokens)
    )
    shared_geometry = shared.contract
    geometry_identity = {
        "shared_geometry_hash": preprocessing_sha256(shared_geometry),
        "view_ids": [int(row["id"]) for row in selected],
        "source_intrinsics": source_intrinsics.astype(np.float32).tolist(),
        "feature_intrinsics": intrinsics.astype(np.float32).tolist(),
    }
    paths = object_paths(output, uid)
    payload = {
        "format": COMPACT_OBJECT_FORMAT,
        "layout": COMPACT_LAYOUT_VERSION,
        "uid": uid,
        "object_uid": uid,
        "slat_config_hash": slat_config_hash,
        "lifting_config_hash": lifting_config_hash,
        "visual_patch_features": visual,
        "intrinsics": torch.from_numpy(intrinsics.astype(np.float32)),
        "extrinsics": torch.from_numpy(extrinsics.astype(np.float32)),
        "image_size": [518, 518],
        "grid_transform": "identity",
        "extrinsics_type": "w2c",
        "camera_forward_sign": float(forward_sign),
        "view_ids": torch.tensor(
            [int(row["id"]) for row in selected], dtype=torch.int64
        ),
        "source_intrinsics": torch.from_numpy(source_intrinsics.astype(np.float32)),
        "source_to_feature_affines": torch.from_numpy(
            shared.source_to_feature_affines.astype(np.float32)
        ),
        "preprocessing": {
            "shared_geometry": shared_geometry,
            "shared_geometry_hash": preprocessing_sha256(shared_geometry),
            "sample_geometry_identity_hash": preprocessing_sha256(
                geometry_identity
            ),
        },
        "context_contract": contexts["context_contract"],
        "official_gt_support_only": True,
    }
    atomic_torch_save(paths["compact"], payload)
    slat_row, lifting_row = make_rows(
        output=output,
        uid=uid,
        target_path=target_path,
        compact_path=paths["compact"],
    )
    record = {
        "uid": uid,
        "candidate_view_count": len(views),
        "selected_view_ids": [int(row["id"]) for row in selected],
        "camera_forward_sign": float(forward_sign),
        "official_camera_conversion": "camera_to_world_to_world_to_camera",
        "coord_count": len(coords3),
        "visual_shape": list(visual.shape),
        "compact_bytes": paths["compact"].stat().st_size,
        "render_archive_complete": archive_audit["archive_complete"],
        "render_archive_recovered": archive_audit["archive_recovered"],
        "render_archive_read_error": archive_audit["archive_read_error"],
        "cache_reused_after_interruption": False,
    }
    atomic_json(
        paths["record"],
        {
            "format": COMPACT_OBJECT_RECORD_FORMAT,
            "uid": uid,
            "slat_config_hash": slat_config_hash,
            "lifting_config_hash": lifting_config_hash,
            "slat_row": slat_row,
            "lifting_row": lifting_row,
            "record": record,
        },
    )
    return slat_row, lifting_row, record


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--dino_model", default="dinov2_vitl14_reg")
    parser.add_argument("--selected_views", type=int, default=8)
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--ss_context_tokens", type=int, default=4096)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--worker_count", type=int, default=1)
    parser.add_argument("--materialize_only", action="store_true")
    parser.add_argument("--finalize_only", action="store_true")
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if args.materialize_only and args.finalize_only:
        raise ValueError("materialize_only/finalize_only are mutually exclusive")
    if int(args.selected_views) <= 0 or int(args.ss_context_tokens) <= 0:
        raise ValueError("selected_views/ss_context_tokens must be positive")
    if int(args.worker_count) <= 0 or not 0 <= int(args.worker_index) < int(
        args.worker_count
    ):
        raise ValueError("worker_index must satisfy 0 <= index < worker_count")
    split_path = Path(args.split_manifest).expanduser().resolve()
    split = load_json(split_path)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError(f"unexpected split format={split.get('format')!r}")
    rows = list(split["rows"])
    if int(args.max_objects) > 0:
        rows = rows[: int(args.max_objects)]
    if not rows:
        raise ValueError("official compact cache selection is empty")
    _, ss_binding = load_no_vggt_ss_evidence(args.native_ss_report)
    stock_freeze = load_json(args.stock_slat_freeze)
    normalization = {
        key: [float(value) for value in values]
        for key, values in stock_freeze["slat_normalization"].items()
    }
    slat_config, lifting_config = _make_config(
        args=args, split=split, ss_binding=ss_binding
    )
    slat_config_hash = canonical_sha256(slat_config)
    lifting_config_hash = canonical_sha256(lifting_config)
    normalization_hash = canonical_sha256(normalization)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    work = [
        (position, source)
        for position, source in enumerate(rows)
        if position % int(args.worker_count) == int(args.worker_index)
    ]
    reused_count = 0
    built_count = 0
    pipeline = None
    if not args.finalize_only:
        for local_position, (global_position, source) in enumerate(work, start=1):
            reused = validate_cached_object(
                output=output,
                source=source,
                slat_config_hash=slat_config_hash,
                lifting_config_hash=lifting_config_hash,
            )
            if reused is not None:
                reused_count += 1
                print(
                    f"[official_slat_compact:reuse] worker={args.worker_index}/"
                    f"{args.worker_count} local={local_position}/{len(work)} "
                    f"global={global_position + 1}/{len(rows)} uid={source['uid']}",
                    flush=True,
                )
                continue
            if pipeline is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("compact materialization requires CUDA for DINO")
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
                pipeline = _build_dino_pipeline(args.dino_model, device)
            build_one(
                output=output,
                source=source,
                pipeline=pipeline,
                selected_views=int(args.selected_views),
                ss_context_tokens=int(args.ss_context_tokens),
                slat_config_hash=slat_config_hash,
                lifting_config_hash=lifting_config_hash,
            )
            built_count += 1
            print(
                f"[official_slat_compact] worker={args.worker_index}/"
                f"{args.worker_count} local={local_position}/{len(work)} "
                f"global={global_position + 1}/{len(rows)} uid={source['uid']}",
                flush=True,
            )
        worker_report = {
            "format": COMPACT_CACHE_REPORT_FORMAT,
            "stage": "materialize_worker",
            "passed": True,
            "worker_index": int(args.worker_index),
            "worker_count": int(args.worker_count),
            "assigned_object_count": len(work),
            "built_object_count": built_count,
            "reused_object_count": reused_count,
            "protocol_sha256": split["protocol_sha256"],
            "split": split["name"],
        }
        atomic_json(
            output
            / f"worker_{int(args.worker_index):03d}_of_{int(args.worker_count):03d}.json",
            worker_report,
        )
        if args.materialize_only or int(args.worker_count) > 1:
            print(json.dumps(worker_report, indent=2, ensure_ascii=False))
            return

    slat_rows: list[dict[str, Any]] = []
    lifting_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in rows:
        cached = validate_cached_object(
            output=output,
            source=source,
            slat_config_hash=slat_config_hash,
            lifting_config_hash=lifting_config_hash,
        )
        if cached is None:
            missing.append(str(source["uid"]))
            continue
        slat_row, lifting_row, record = cached
        slat_rows.append(slat_row)
        lifting_rows.append(lifting_row)
        records.append(record)
    if missing:
        raise RuntimeError(
            f"compact finalize found missing objects count={len(missing)} first={missing[:8]}"
        )
    slat_manifest = {
        "format": COMPACT_SLAT_MANIFEST_FORMAT,
        "layout": COMPACT_LAYOUT_VERSION,
        "materialized": True,
        "output_dir": str(output),
        "config": slat_config,
        "config_hash": slat_config_hash,
        "slat_normalization": normalization,
        "slat_normalization_hash": normalization_hash,
        "samples": slat_rows,
        "sample_count": len(slat_rows),
        "object_count": len(slat_rows),
        "official_gt_support_only": True,
    }
    lifting_manifest = {
        "format": COMPACT_LIFTING_MANIFEST_FORMAT,
        "layout": COMPACT_LAYOUT_VERSION,
        "output_dir": str(output),
        "source_cache_manifest": str(split_path),
        "stock_condition_source": "runtime_deterministic_from_single_fp16_dino",
        "lifting_feature_source": "official_pose_raw_dino_compact_v2",
        "selection": {
            "mode": "frozen_official_slat_split_prefix",
            "split": split["name"],
            "count": len(rows),
        },
        "samples": lifting_rows,
        "sample_count": len(lifting_rows),
        "object_count": len(lifting_rows),
        "failure_count": 0,
        "feature_metadata": {
            **dino_only_feature_metadata(
                patch_count=int(records[0]["visual_shape"][1])
            ),
            "vggt_model_executed": False,
        },
        "visual_feature_dim": 1024,
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": lifting_config,
        "config_hash": lifting_config_hash,
        "depth_calibration_enabled_count": 0,
        "split_identity": f"official_slat.{split['protocol_sha256']}.{split['name']}",
    }
    slat_manifest_path = output / "slat_manifest.json"
    lifting_manifest_path = output / "lifting_manifest.json"
    atomic_json(slat_manifest_path, slat_manifest)
    atomic_json(lifting_manifest_path, lifting_manifest)
    report = {
        "format": COMPACT_CACHE_REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "protocol_sha256": split["protocol_sha256"],
        "split": split["name"],
        "object_count": len(rows),
        "selected_views": int(args.selected_views),
        "slat_manifest": str(slat_manifest_path.resolve()),
        "slat_manifest_sha256": sha256_file(slat_manifest_path),
        "lifting_manifest": str(lifting_manifest_path.resolve()),
        "lifting_manifest_sha256": sha256_file(lifting_manifest_path),
        "native_ss_binding_only": ss_binding,
        "native_ss_executed": False,
        "vggt_executed": False,
        "stored_tensor_policy": [
            "single fp16 visual_patch_features",
            "intrinsics/extrinsics/view ids/projection metadata",
        ],
        "runtime_reconstructed_policy": [
            "positive per-view SLat contexts as views of stored DINO",
            "all-zero negative SLat contexts",
            "deterministic Stock-SS context",
            "legacy all-zero support placeholders when compatibility requires them",
        ],
        "serialized_zero_depth_confidence_prior": False,
        "serialized_duplicate_condition_dino": False,
        "compact_payload_bytes": int(sum(row["compact_bytes"] for row in records)),
        "records": records,
        "scope_guard": (
            "lossless official GT-support SLat cache compaction only; no feature "
            "quantization and no predicted-only target semantics"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "passed": True,
                "split": split["name"],
                "object_count": len(rows),
                "compact_payload_bytes": report["compact_payload_bytes"],
                "slat_manifest": report["slat_manifest"],
                "lifting_manifest": report["lifting_manifest"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
