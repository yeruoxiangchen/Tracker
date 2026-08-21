"""Build native ReconViaGen SS-context sidecars for official SS caches.

No view is selected here.  The exact ordered ``view_ids`` from the immutable
official lifting cache are replayed.  VGGT cameras/depth are not consumed;
known K/T and posed-DINO tensors remain in the base cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from ar_ss_flow.shared_object_preprocessing import (
    prepare_shared_object_arrays,
    transform_intrinsics,
)
from official_ss_with_vggt_perf_v1.cache import (
    CACHE_REPORT_FORMAT,
    CONTEXT_VERSION,
    MANIFEST_FORMAT,
    MODEL_CONTEXT_CONTRACT,
    SIDECAR_FORMAT,
    validate_native_ss_vggt_context_tensor,
)
from pose_point_depth_mv.native_slat_genrecon import module_schema
from pose_point_depth_mv.prepare_proobjaverse_official_slat_dino_cache import (
    _load_views_with_audit,
)
from pose_point_depth_mv.prepare_proobjaverse_official_slat_with_vggt_sidecar import (
    BASE_DINO_PATCH_MAX_ABS_TOLERANCE,
    BASE_DINO_PATCH_MEAN_ABS_TOLERANCE,
    EXPECTED_DINO_PREFIX_TOKENS,
    VGGT_REPO_ID,
    _array_sha256,
    _atomic_json_once_or_exact,
    _atomic_torch_save,
    _encoder_asset_identity as _slat_encoder_asset_identity,
    _python_tree_sha256,
    _relative_reference,
    _resolve_pretrained_asset,
    _validate_base_dino_patch_replay,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    SPLIT_FORMAT,
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
BUILDER_VERSION = "official_ss_with_vggt_perf_v1.build_cache.v3"


def _encoder_asset_identity(
    pretrained: str, vggt_repo: str
) -> tuple[dict[str, Any], dict[str, str]]:
    common, diagnostics = _slat_encoder_asset_identity(pretrained, vggt_repo)
    ss_config, ss_revision = _resolve_pretrained_asset(
        pretrained, "ckpts/ss_vggt_cond.json"
    )
    ss_weights, weights_revision = _resolve_pretrained_asset(
        pretrained, "ckpts/ss_vggt_cond.safetensors"
    )
    if ss_revision != weights_revision or ss_revision != common["trellis_snapshot_revision"]:
        raise RuntimeError("SS condition/pipeline assets resolve to different revisions")
    assets = {
        key: value
        for key, value in common.items()
        if not key.startswith("slat_vggt_cond_") and key != "builder_source_sha256"
    }
    assets.update(
        {
            "ss_vggt_cond_config_sha256": sha256_file(ss_config),
            "ss_vggt_cond_weights_sha256": sha256_file(ss_weights),
            "shared_slat_sidecar_builder_source_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "pose_point_depth_mv/prepare_proobjaverse_official_slat_with_vggt_sidecar.py"
            ),
            "builder_source_sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    diagnostics = {
        key: value
        for key, value in diagnostics.items()
        if not key.startswith("slat_vggt_cond")
    }
    diagnostics.update(
        {
            "ss_vggt_cond_config": str(ss_config),
            "ss_vggt_cond_weights": str(ss_weights),
        }
    )
    return assets, diagnostics


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    split_path = Path(args.split_manifest).expanduser().resolve(strict=True)
    split = load_json(split_path)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError(f"unexpected official split={split.get('format')!r}")
    base_path = Path(args.base_manifest).expanduser().resolve(strict=True)
    base = PoseLiftingCacheDataset(base_path, indices="all")
    config = dict(base.config)
    target_binding = config.get("official_ss_targets")
    if not isinstance(target_binding, dict):
        raise RuntimeError("base cache is not an official SS target cache")
    if str(target_binding.get("split")) != str(split.get("name")):
        raise RuntimeError("base official SS split differs from split manifest")
    if str(config.get("official_slat_protocol_sha256")) != str(
        split.get("protocol_sha256")
    ):
        raise RuntimeError("base official SS protocol differs")
    split_rows = split.get("rows")
    if not isinstance(split_rows, list) or not split_rows:
        raise ValueError("official split has no rows")
    split_by_uid = {str(row["uid"]): row for row in split_rows}
    if len(split_by_uid) != len(split_rows):
        raise ValueError("official split contains duplicate UIDs")
    selected = []
    for base_index, row in enumerate(base.rows):
        uid = str(row.get("uid", ""))
        source = split_by_uid.get(uid)
        if source is None:
            raise RuntimeError(f"base official SS UID absent from split: {uid}")
        if str(row.get("object_uid", uid)) != uid:
            raise RuntimeError(f"official SS object UID differs: {uid}")
        if not str(row.get("ss_latent", "")) or not str(
            row.get("ss_latent_sha256", "")
        ):
            raise RuntimeError(f"official SS target identity missing: {uid}")
        render = Path(source["render_tar"]).expanduser().resolve(strict=True)
        if render.stat().st_size != int(source["render_size"]):
            raise RuntimeError(f"official render size changed: {uid}")
        selected.append(
            {
                "uid": uid,
                "object_uid": uid,
                "base_index": base_index,
                "source": source,
            }
        )
    if [row["uid"] for row in selected] != [str(row["uid"]) for row in split_rows]:
        raise RuntimeError("base official SS order differs from frozen split")
    if int(args.max_objects) > 0:
        selected = selected[: int(args.max_objects)]
    if not selected:
        raise ValueError("with-VGGT SS cache selection is empty")
    return {
        "split_path": split_path,
        "split": split,
        "base_path": base_path,
        "base": base,
        "selected": selected,
    }


def _contract(
    args: argparse.Namespace,
    plan: dict[str, Any],
    encoder_assets: dict[str, Any],
) -> dict[str, Any]:
    base = plan["base"]
    return {
        "version": CONTEXT_VERSION,
        "builder_version": BUILDER_VERSION,
        "protocol_sha256": str(plan["split"]["protocol_sha256"]),
        "split": str(plan["split"]["name"]),
        "base_cache": {
            "manifest_sha256": sha256_file(plan["base_path"]),
            "config_hash": str(base.config_hash),
        },
        "selected_view_policy": (
            "exact ordered view_ids from immutable official lifting cache; "
            "no view selection is executed"
        ),
        "selected_view_count": int(args.expected_selected_views),
        "shared_geometric_preprocessing": copy.deepcopy(
            base.config.get("geometric_preprocessing", {})
        ),
        "shared_geometric_preprocessing_hash": base.config.get(
            "geometric_preprocessing_hash"
        ),
        "context_semantics": {
            "producer": (
                "ar_ss_flow.build_pose_lifting_cache.build_native_stock_pipeline "
                "+ extract_stock_condition -> native get_ss_cond"
            ),
            "historical_native_ss_v2_production_path_reused": True,
            "native_pipeline_loader": (
                "TrellisVGGTTo3DPipeline.from_pretrained; low_vram=false"
            ),
            "vggt_layers": [4, 11, 17, 23],
            "dino_input": "patch_tokens_only_after_5_prefix_tokens",
            "output_layout": "[1,4096,1024]",
            "positive_context_materialized": True,
            "negative_context_policy": "runtime_zeros_like_positive",
            "all_frozen_views_fused_by_native_ss_vggt_cond": True,
            "posed_dino_random_view_subset_unchanged": True,
            "base_dino_patch_replay_tolerances": {
                "comparison_dtype": "torch.float16",
                "max_abs": BASE_DINO_PATCH_MAX_ABS_TOLERANCE,
                "mean_abs": BASE_DINO_PATCH_MEAN_ABS_TOLERANCE,
            },
        },
        "model_context": copy.deepcopy(MODEL_CONTEXT_CONTRACT),
        "encoder_assets": encoder_assets,
    }


def _load_pipeline(
    *, pretrained: str, vggt_repo: str, device: torch.device
) -> Any:
    """Reuse the CUDA production loader from the proven Native-SS v2 cache.

    The failed v1 sidecar builder manually reconstructed a lightweight subset
    of the pipeline and enabled low-VRAM module swapping.  Although its frozen
    weights were identical, that was a new CUDA execution path and therefore
    not the historical Native-SS v2 implementation the experiment intends to
    hold fixed.  Only the official dataset/view adapter belongs in this module.
    """

    from ar_ss_flow.build_pose_lifting_cache import build_native_stock_pipeline

    if str(vggt_repo) != VGGT_REPO_ID:
        raise ValueError(f"native pipeline freezes vggt_repo={VGGT_REPO_ID!r}")
    torch.cuda.set_device(device)
    pipeline = build_native_stock_pipeline(pretrained, device)
    if getattr(pipeline, "low_vram", None) is not False:
        raise RuntimeError("historical Native-SS v2 pipeline must keep low_vram=false")
    return pipeline


def _extract_historical_native_ss_condition(
    pipeline: Any,
    images: list[Any],
    cached_visual_patch_features: torch.Tensor,
    *,
    uid: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the historical v2 extractor and retain its exact DINO input audit.

    ``extract_stock_condition`` intentionally returns only the final Stock SS
    context.  A temporary wrapper observes (without modifying) the patch-only
    DINO tensor passed to the native ``get_ss_cond`` call so the official
    sidecar can still prove equality to the immutable posed-DINO cache.
    """

    from ar_ss_flow.build_pose_lifting_cache import extract_stock_condition

    original = pipeline.get_ss_cond
    captured: dict[str, Any] = {}

    def capture(image_cond, aggregated_tokens_list, num_samples):
        result = original(image_cond, aggregated_tokens_list, num_samples)
        captured["image_cond"] = image_cond.detach()
        captured["negative"] = result.get("neg_cond")
        return result

    pipeline.get_ss_cond = capture
    try:
        context = extract_stock_condition(pipeline, images)
    finally:
        pipeline.get_ss_cond = original
    patches = captured.get("image_cond")
    if not torch.is_tensor(patches) or patches.ndim != 4:
        raise RuntimeError(f"uid={uid} historical Native-SS v2 DINO input missing")
    if int(patches.shape[0]) != 1 or int(patches.shape[2]) != int(
        cached_visual_patch_features.shape[1]
    ):
        raise RuntimeError(
            f"uid={uid} historical Native-SS v2 DINO patch shape differs"
        )
    prefix = patches.new_zeros(
        (
            int(patches.shape[0]),
            int(patches.shape[1]),
            EXPECTED_DINO_PREFIX_TOKENS,
            int(patches.shape[3]),
        )
    )
    replay = _validate_base_dino_patch_replay(
        torch.cat((prefix, patches), dim=2),
        cached_visual_patch_features,
        uid=uid,
    )
    negative = captured.get("negative")
    if not torch.is_tensor(negative) or int(torch.count_nonzero(negative).item()) != 0:
        raise RuntimeError(f"uid={uid} native negative context is not zero")
    return context, replay


def _paths(output: Path, uid: str) -> tuple[Path, Path]:
    return (
        output / "sidecars" / uid[:2] / f"{uid}.pt",
        output / "records" / uid[:2] / f"{uid}.json",
    )


def _load_completed(
    *, output: Path, row: dict[str, Any], contract_hash: str
) -> dict[str, Any] | None:
    sidecar, record_path = _paths(output, row["uid"])
    if not sidecar.exists() and not record_path.exists():
        return None
    if sidecar.is_file() and not record_path.exists():
        return None
    if not sidecar.is_file() or not record_path.is_file():
        raise RuntimeError(f"incomplete with-VGGT SS object: {sidecar}")
    record = load_json(record_path)
    expected = {
        "format": SIDECAR_FORMAT,
        "uid": row["uid"],
        "object_uid": row["object_uid"],
        "base_index": int(row["base_index"]),
        "sidecar_contract_hash": contract_hash,
        "sidecar_file": _relative_reference(sidecar, output),
        "sidecar_file_sha256": sha256_file(sidecar),
        "sidecar_file_size": sidecar.stat().st_size,
        "source_render_tar_sha256": sha256_file(
            Path(row["source"]["render_tar"]).expanduser().resolve(strict=True)
        ),
    }
    mismatch = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"resumed with-VGGT SS record differs={mismatch}")
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    context = validate_native_ss_vggt_context_tensor(
        payload.get("native_ss_vggt_cond"), uid=row["uid"]
    )
    payload_view_ids = payload.get("view_ids")
    if (
        payload.get("format") != SIDECAR_FORMAT
        or payload.get("uid") != row["uid"]
        or payload.get("object_uid") != row["object_uid"]
        or payload.get("sidecar_contract_hash") != contract_hash
        or not torch.is_tensor(payload_view_ids)
        or [int(value) for value in payload_view_ids.tolist()]
        != record.get("view_ids")
        or list(context.shape) != record.get("native_context_shape")
        or str(context.dtype) != record.get("native_context_dtype")
        or payload.get("negative_context_policy")
        != "runtime_zeros_like_positive"
        or payload.get("vggt_camera_consumed") is not False
        or payload.get("known_K_T_replaced") is not False
        or record.get("vggt_forward_executed") is not True
        or record.get("vggt_camera_consumed") is not False
        or record.get("known_K_T_replaced") is not False
        or record.get("base_dino_patch_replay", {}).get("passed") is not True
    ):
        raise RuntimeError(f"resumed with-VGGT SS payload differs: {sidecar}")
    return record


@torch.no_grad()
def _materialize(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    if int(args.worker_count) <= 0 or not 0 <= int(args.worker_index) < int(
        args.worker_count
    ):
        raise ValueError("worker_index must lie in [0,worker_count)")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets, diagnostics = _encoder_asset_identity(args.pretrained, args.vggt_repo)
    contract = _contract(args, plan, assets)
    contract_hash = canonical_sha256(contract)
    assigned = [
        row
        for position, row in enumerate(plan["selected"])
        if position % int(args.worker_count) == int(args.worker_index)
    ]
    if not assigned:
        raise ValueError("with-VGGT SS worker has no assigned objects")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("with-VGGT SS materialization requires CUDA")
    pipeline = _load_pipeline(
        pretrained=args.pretrained, vggt_repo=args.vggt_repo, device=device
    )
    runtime_schema = {
        "vggt_aggregator": module_schema(pipeline.VGGT_model.aggregator),
        "dino_image_condition_model": module_schema(
            pipeline.models["image_cond_model"]
        ),
        "ss_vggt_condition": module_schema(
            pipeline.models["sparse_structure_vggt_cond"]
        ),
        "historical_native_ss_v2_loader_reused": True,
        "low_vram": bool(pipeline.low_vram),
        "unexecuted_vggt_heads": [
            name
            for name in ("depth_head", "track_head", "camera_head", "point_head")
            if hasattr(pipeline.VGGT_model, name)
        ],
    }
    records = []
    executed = 0
    reused = 0
    for position, row in enumerate(assigned, start=1):
        completed = _load_completed(
            output=output, row=row, contract_hash=contract_hash
        )
        if completed is not None:
            records.append(completed)
            reused += 1
            print(
                f"[official_ss_with_vggt:reuse] {position}/{len(assigned)} {row['uid']}",
                flush=True,
            )
            continue
        sample = plan["base"][row["base_index"]]
        view_ids = [int(value) for value in sample["view_ids"].tolist()]
        if len(view_ids) != int(args.expected_selected_views):
            raise RuntimeError(f"uid={row['uid']} frozen view count differs")
        render_path = Path(row["source"]["render_tar"]).expanduser().resolve(
            strict=True
        )
        render_sha = sha256_file(render_path)
        views, archive_audit = _load_views_with_audit(render_path, row["uid"])
        by_id = {int(value["id"]): value for value in views}
        if len(by_id) != len(views):
            raise RuntimeError(f"uid={row['uid']} render has duplicate views")
        missing = [view_id for view_id in view_ids if view_id not in by_id]
        if missing:
            raise RuntimeError(f"uid={row['uid']} frozen views missing={missing}")
        selected = [by_id[view_id] for view_id in view_ids]
        rgba = [value["rgba"] for value in selected]
        shared = prepare_shared_object_arrays(
            [value[..., :3] for value in rgba],
            [value[..., 3] for value in rgba],
            resolution=518,
            foreground_margin=1.10,
            alpha_threshold=0.80,
        )
        if shared.contract != sample["preprocessing"]["shared_geometry"]:
            raise RuntimeError(f"uid={row['uid']} preprocessing contract changed")
        if not np.array_equal(
            shared.source_to_feature_affines,
            sample["source_to_feature_affines"].numpy(),
        ):
            raise RuntimeError(f"uid={row['uid']} preprocessing affine changed")
        source_k = np.stack([value["intrinsic"] for value in selected])
        feature_k = transform_intrinsics(source_k, shared.source_to_feature_affines)
        source_t = np.stack([value["extrinsic"] for value in selected])
        if not np.allclose(
            source_k, sample["source_intrinsics"].numpy(), rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"uid={row['uid']} known source K changed")
        if not np.allclose(
            feature_k, sample["intrinsics"].numpy(), rtol=1.0e-6, atol=1.0e-5
        ):
            raise RuntimeError(f"uid={row['uid']} known feature K changed")
        if not np.allclose(
            source_t, sample["extrinsics"].numpy(), rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"uid={row['uid']} known T changed")
        native_context, replay = _extract_historical_native_ss_condition(
            pipeline,
            shared.images,
            sample["visual_patch_features"],
            uid=row["uid"],
        )
        context = validate_native_ss_vggt_context_tensor(
            native_context.detach().cpu(), uid=row["uid"]
        )
        decoded_hashes = [_array_sha256(value) for value in rgba]
        processed_hashes = [
            _array_sha256(np.asarray(image, dtype=np.uint8)) for image in shared.images
        ]
        sidecar, record_path = _paths(output, row["uid"])
        payload = {
            "format": SIDECAR_FORMAT,
            "uid": row["uid"],
            "object_uid": row["object_uid"],
            "sidecar_contract_hash": contract_hash,
            "view_ids": torch.tensor(view_ids, dtype=torch.int64),
            "native_ss_vggt_cond": context,
            "negative_context_policy": "runtime_zeros_like_positive",
            "decoded_source_rgba_sha256": decoded_hashes,
            "processed_input_rgb_sha256": processed_hashes,
            "vggt_camera_consumed": False,
            "known_K_T_replaced": False,
        }
        _atomic_torch_save(sidecar, payload)
        record = {
            "format": SIDECAR_FORMAT,
            "uid": row["uid"],
            "object_uid": row["object_uid"],
            "base_index": int(row["base_index"]),
            "sidecar_contract_hash": contract_hash,
            "sidecar_file": _relative_reference(sidecar, output),
            "sidecar_file_sha256": sha256_file(sidecar),
            "sidecar_file_size": sidecar.stat().st_size,
            "view_ids": view_ids,
            "native_context_shape": list(context.shape),
            "native_context_dtype": str(context.dtype),
            "decoded_source_rgba_sha256": decoded_hashes,
            "processed_input_rgb_sha256": processed_hashes,
            "source_render_tar_sha256": render_sha,
            "render_archive_audit": archive_audit,
            "base_dino_patch_replay": replay,
            "exact_base_affines": True,
            "exact_base_source_intrinsics": True,
            "base_feature_intrinsics_within_tolerance": True,
            "exact_base_extrinsics": True,
            "vggt_forward_executed": True,
            "vggt_camera_consumed": False,
            "known_K_T_replaced": False,
        }
        atomic_json(record_path, record)
        records.append(record)
        executed += 1
        # The historical extractor owns its VGGT/DINO temporaries.  Only
        # release names that are local to this materialization scope; stale
        # names from the retired inline implementation caused the first fix1
        # CUDA smoke to fail *after* its first valid sidecar was committed.
        del sample, views, selected, rgba, shared, native_context, context, payload
        torch.cuda.empty_cache()
        print(
            f"[official_ss_with_vggt] {position}/{len(assigned)} {row['uid']}",
            flush=True,
        )
    report = {
        "format": BUILDER_VERSION,
        "passed": True,
        "mode": "materialize",
        "worker_index": int(args.worker_index),
        "worker_count": int(args.worker_count),
        "assigned_object_count": len(assigned),
        "materialized_object_count": executed,
        "reused_object_count": reused,
        "vggt_forward_call_count": executed,
        "sidecar_contract": contract,
        "sidecar_contract_hash": contract_hash,
        "runtime_encoder_schema": runtime_schema,
        "encoder_asset_locations_diagnostic_only": diagnostics,
        "records": records,
    }
    shard = output / "shards" / (
        f"worker_{int(args.worker_index):02d}_of_{int(args.worker_count):02d}.json"
    )
    atomic_json(shard, report)
    return report


def _finalize(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve(strict=True)
    assets, diagnostics = _encoder_asset_identity(args.pretrained, args.vggt_repo)
    contract = _contract(args, plan, assets)
    contract_hash = canonical_sha256(contract)
    records = []
    for row in plan["selected"]:
        record = _load_completed(output=output, row=row, contract_hash=contract_hash)
        if record is None:
            raise RuntimeError(f"missing with-VGGT SS sidecar uid={row['uid']}")
        records.append(record)
    samples = [
        {
            key: record[key]
            for key in (
                "uid",
                "object_uid",
                "base_index",
                "sidecar_file",
                "sidecar_file_sha256",
                "sidecar_file_size",
                "view_ids",
                "native_context_shape",
                "native_context_dtype",
                "source_render_tar_sha256",
            )
        }
        for record in records
    ]
    base_cache = {
        "manifest": _relative_reference(plan["base_path"], output),
        "manifest_sha256": sha256_file(plan["base_path"]),
        "config_hash": str(plan["base"].config_hash),
    }
    binding = {
        "version": CONTEXT_VERSION,
        "base_cache": base_cache,
        "sidecar_contract": contract,
        "sidecar_contract_hash": contract_hash,
        "sidecar_index_sha256": canonical_sha256(samples),
        "sample_count": len(samples),
        "ordered_uid_sha256": canonical_sha256([row["uid"] for row in samples]),
    }
    pair_identity = canonical_sha256(binding)
    config = copy.deepcopy(plan["base"].config)
    config["base_no_vggt_identity_preserved"] = copy.deepcopy(config.pop("no_vggt"))
    config["ss_input_context"] = {
        "version": CONTEXT_VERSION,
        "stock_floor": "VSS0",
        "source": "native_reconviagen_vggt_plus_dinov2_ss_vggt_cond",
        "pair_identity": pair_identity,
        "sidecar_contract_hash": contract_hash,
        "selected_views": "exact ordered base lifting view_ids",
        "vggt_model_executed": True,
        "vggt_camera_consumed": False,
        "known_pose_dino_branch_unchanged": True,
        "official_ss_target_unchanged": True,
    }
    manifest = {
        "format": MANIFEST_FORMAT,
        "pair_identity": pair_identity,
        "pair_binding": binding,
        "output_dir": ".",
        "config": config,
        "config_hash": canonical_sha256(config),
        "samples": samples,
        "sample_count": len(samples),
        "object_count": len(samples),
        "vggt_model_executed": True,
        "vggt_camera_consumed": False,
        "known_K_T_replaced": False,
    }
    manifest_path = output / "with_vggt_ss_manifest.json"
    _atomic_json_once_or_exact(manifest_path, manifest)
    report = {
        "format": CACHE_REPORT_FORMAT,
        "passed": True,
        "complete": True,
        "split": plan["split"]["name"],
        "protocol_sha256": plan["split"]["protocol_sha256"],
        "object_count": len(samples),
        "selected_views": int(args.expected_selected_views),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "pair_identity": pair_identity,
        "sidecar_contract": contract,
        "sidecar_contract_hash": contract_hash,
        "sidecar_bytes": int(sum(row["sidecar_file_size"] for row in records)),
        "vggt_forward_call_count": len(records),
        "vggt_model_executed": True,
        "vggt_camera_consumed": False,
        "known_K_T_replaced": False,
        "same_frozen_view_ids_as_base": True,
        "official_ss_target_changed": False,
        "base_cache_rewritten": False,
        "encoder_asset_locations_diagnostic_only": diagnostics,
        "scope_guard": (
            "official with-VGGT Native-SS input isolation; target, posed-DINO, "
            "known K/T, Flow architecture, LoRA recipe and decoder unchanged"
        ),
    }
    body = dict(report)
    report["report_sha256"] = canonical_sha256(body)
    _atomic_json_once_or_exact(output / "report.json", report)
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "materialize", "finalize", "both"),
        default="preflight",
    )
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--base_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--vggt_repo", default=VGGT_REPO_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected_selected_views", type=int, default=8)
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--worker_count", type=int, default=1)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.expected_selected_views) <= 0:
        raise ValueError("expected_selected_views must be positive")
    if args.mode == "both" and (
        int(args.worker_index) != 0 or int(args.worker_count) != 1
    ):
        raise ValueError("--mode both supports only one worker")
    plan = _plan(args)
    assets, diagnostics = _encoder_asset_identity(args.pretrained, args.vggt_repo)
    preflight = {
        "passed": True,
        "mode": "preflight",
        "split": plan["split"]["name"],
        "protocol_sha256": plan["split"]["protocol_sha256"],
        "object_count": len(plan["selected"]),
        "base_manifest": str(plan["base_path"]),
        "base_manifest_sha256": sha256_file(plan["base_path"]),
        "base_config_hash": plan["base"].config_hash,
        "exact_base_view_ids_only": True,
        "vggt_camera_consumed": False,
        "model_context": MODEL_CONTEXT_CONTRACT,
        "encoder_assets": assets,
        "encoder_asset_locations_diagnostic_only": diagnostics,
    }
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return
    if args.mode in ("materialize", "both"):
        result = _materialize(args, plan)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "passed",
                        "worker_index",
                        "worker_count",
                        "assigned_object_count",
                        "materialized_object_count",
                        "reused_object_count",
                    )
                },
                indent=2,
            )
        )
    if args.mode in ("finalize", "both"):
        report = _finalize(args, plan)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "passed",
                        "split",
                        "object_count",
                        "manifest",
                        "pair_identity",
                        "sidecar_bytes",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
