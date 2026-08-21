#!/usr/bin/env python3
"""Build native ReconViaGen VGGT SLat-context sidecars for official caches.

The builder never chooses views.  It reads the exact ``view_ids`` frozen in an
existing official no-VGGT lifting cache, replays the same shared RGB/alpha
geometry, and materializes only the native positive ``slat_vggt_cond`` tensor.
The all-zero negative context is reconstructed at dataset load time.
"""

from __future__ import annotations

import argparse
import copy
import functools
import gc
import hashlib
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
from pose_point_depth_mv.direct_slat_data import DirectSLatCacheDataset
from pose_point_depth_mv.native_slat_genrecon import module_schema
from pose_point_depth_mv.prepare_proobjaverse_official_slat_dino_cache import (
    _load_views_with_audit,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    SPLIT_FORMAT,
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache import (
    WITH_VGGT_CACHE_REPORT_FORMAT,
    WITH_VGGT_CONTEXT_VERSION,
    WITH_VGGT_LIFTING_MANIFEST_FORMAT,
    WITH_VGGT_SIDECAR_FORMAT,
    WITH_VGGT_SLAT_MANIFEST_FORMAT,
    validate_native_slat_vggt_context_tensor,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond


BUILDER_VERSION = (
    "pose_point_depth_mv.prepare_proobjaverse_official_slat_with_vggt_sidecar.v1"
)
VGGT_REPO_ID = "Stable-X/vggt-object-v0-1"
EXPECTED_DINO_PREFIX_TOKENS = 5
BASE_DINO_PATCH_MAX_ABS_TOLERANCE = 0.01
BASE_DINO_PATCH_MEAN_ABS_TOLERANCE = 0.0005

_RECORD_KEYS = {
    "format",
    "uid",
    "object_uid",
    "support_seed",
    "base_index",
    "sidecar_contract_hash",
    "sidecar_file",
    "sidecar_file_sha256",
    "sidecar_file_size",
    "view_ids",
    "native_context_shape",
    "native_context_dtype",
    "dino_patch_tokens",
    "native_prefix_tokens",
    "decoded_source_rgba_sha256",
    "processed_input_rgb_sha256",
    "source_render_tar_sha256",
    "render_archive_audit",
    "base_dino_patch_replay",
    "exact_base_affines",
    "exact_base_source_intrinsics",
    "base_feature_intrinsics_within_tolerance",
    "exact_base_extrinsics",
    "vggt_forward_executed",
    "vggt_camera_consumed",
    "known_K_T_replaced",
}


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_once_or_exact(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if not path.is_file() or load_json(path) != value:
            raise RuntimeError(f"refusing to overwrite changed with-VGGT artifact: {path}")
        return
    atomic_json(path, value)


def _relative_reference(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), start=root.resolve())


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise RuntimeError(f"no Python sources found under {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _hf_asset(repo_id: str, filename: str) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    # Keep the snapshot symlink path long enough to retain its immutable
    # revision component.  ``Path.resolve`` would collapse it to ``blobs/...``
    # and erase the revision even though hashing through the symlink is safe.
    path = Path(hf_hub_download(repo_id, filename, local_files_only=True)).absolute()
    if not path.is_file():
        raise FileNotFoundError(path)
    parts = path.parts
    revision = ""
    if "snapshots" in parts:
        position = parts.index("snapshots")
        if position + 1 < len(parts):
            revision = parts[position + 1]
    if not revision:
        raise RuntimeError(f"cannot identify cached HF snapshot revision: {path}")
    return path, revision


def _local_pretrained_asset(root: Path, filename: str) -> tuple[Path, str]:
    path = (root / filename).resolve(strict=True)
    return path, f"local:{sha256_file(root / 'pipeline.json')}"


def _resolve_pretrained_asset(pretrained: str, filename: str) -> tuple[Path, str]:
    root = Path(pretrained).expanduser()
    if root.is_dir():
        return _local_pretrained_asset(root, filename)
    return _hf_asset(str(pretrained), filename)


def _resolve_vggt_asset(vggt_repo: str, filename: str) -> tuple[Path, str]:
    root = Path(vggt_repo).expanduser()
    if root.is_dir():
        path = (root / filename).resolve(strict=True)
        return path, f"local:{sha256_file(root / 'config.json')}"
    return _hf_asset(str(vggt_repo), filename)


@functools.lru_cache(maxsize=None)
def _encoder_asset_identity(pretrained: str, vggt_repo: str) -> tuple[dict[str, Any], dict[str, str]]:
    pipeline_json, pipeline_revision = _resolve_pretrained_asset(
        pretrained, "pipeline.json"
    )
    slat_config, slat_revision = _resolve_pretrained_asset(
        pretrained, "ckpts/slat_vggt_cond.json"
    )
    slat_weights, weights_revision = _resolve_pretrained_asset(
        pretrained, "ckpts/slat_vggt_cond.safetensors"
    )
    if len({pipeline_revision, slat_revision, weights_revision}) != 1:
        raise RuntimeError("Trellis pipeline/condition assets resolve to different revisions")
    vggt_config, vggt_revision = _resolve_vggt_asset(vggt_repo, "config.json")
    vggt_weights, vggt_weights_revision = _resolve_vggt_asset(
        vggt_repo, "model.safetensors"
    )
    if vggt_revision != vggt_weights_revision:
        raise RuntimeError("VGGT config/weights resolve to different revisions")

    hub_root = Path(torch.hub.get_dir()).resolve(strict=True)
    dino_source = (hub_root / "facebookresearch_dinov2_main").resolve(strict=True)
    dino_candidates = sorted(
        (hub_root / "checkpoints").glob("dinov2_vitl14_reg*_pretrain.pth")
    )
    if len(dino_candidates) != 1:
        raise RuntimeError(
            "expected exactly one cached dinov2_vitl14_reg checkpoint, got "
            f"{[str(path) for path in dino_candidates]}"
        )
    dino_weights = dino_candidates[0].resolve(strict=True)
    tracker = Path(__file__).resolve().parents[1]
    pipeline_source = (
        tracker / "ReconViaGen/trellis/pipelines/trellis_image_to_3d.py"
    ).resolve(strict=True)
    slat_condition_source = (
        tracker / "ReconViaGen/trellis/models/structured_latent_flow.py"
    ).resolve(strict=True)
    vggt_source_root = (tracker / "ReconViaGen/wheels/vggt/vggt").resolve(strict=True)
    builder_source = Path(__file__).resolve(strict=True)
    preprocessing_source = (
        tracker / "ar_ss_flow/shared_object_preprocessing.py"
    ).resolve(strict=True)
    contract = {
        "pretrained": str(pretrained),
        "trellis_snapshot_revision": pipeline_revision,
        "pipeline_json_sha256": sha256_file(pipeline_json),
        "slat_vggt_cond_config_sha256": sha256_file(slat_config),
        "slat_vggt_cond_weights_sha256": sha256_file(slat_weights),
        "vggt_repo": str(vggt_repo),
        "vggt_snapshot_revision": vggt_revision,
        "vggt_config_sha256": sha256_file(vggt_config),
        "vggt_weights_sha256": sha256_file(vggt_weights),
        "dino_model": "dinov2_vitl14_reg",
        "dino_weights_filename": dino_weights.name,
        "dino_weights_sha256": sha256_file(dino_weights),
        "dino_source_tree_sha256": _python_tree_sha256(dino_source),
        "vggt_source_tree_sha256": _python_tree_sha256(vggt_source_root),
        "trellis_pipeline_source_sha256": sha256_file(pipeline_source),
        "slat_condition_source_sha256": sha256_file(slat_condition_source),
        "shared_preprocessing_source_sha256": sha256_file(preprocessing_source),
        "builder_source_sha256": sha256_file(builder_source),
    }
    diagnostics = {
        "pipeline_json": str(pipeline_json),
        "slat_vggt_cond_config": str(slat_config),
        "slat_vggt_cond_weights": str(slat_weights),
        "vggt_config": str(vggt_config),
        "vggt_weights": str(vggt_weights),
        "dino_weights": str(dino_weights),
        "dino_source": str(dino_source),
        "vggt_source": str(vggt_source_root),
    }
    return contract, diagnostics


def _base_plan(args: argparse.Namespace) -> dict[str, Any]:
    split_path = Path(args.split_manifest).expanduser().resolve(strict=True)
    split = load_json(split_path)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError(f"unexpected split format={split.get('format')!r}")
    base_slat_path = Path(args.base_slat_manifest).expanduser().resolve(strict=True)
    base_lifting_path = (
        Path(args.base_lifting_manifest).expanduser().resolve(strict=True)
    )
    slat = DirectSLatCacheDataset(base_slat_path, indices="all", verify_hashes=False)
    lifting = PoseLiftingCacheDataset(base_lifting_path, indices="all")
    if str(slat.config.get("pretrained")) != str(args.pretrained):
        raise RuntimeError("base cache/pretrained binding differs")
    if slat.config.get("condition_arch") != "native_ss_genrecon_v2":
        raise RuntimeError("base cache is not the official Native-SLat v2 contract")
    no_vggt = dict(lifting.config.get("no_vggt", {}))
    if no_vggt.get("vggt_model_executed", False) is not False:
        raise RuntimeError("base lifting cache is not the frozen no-VGGT cache")
    split_by_uid = {str(row["uid"]): row for row in split.get("rows", [])}
    if len(split_by_uid) != len(split.get("rows", [])):
        raise ValueError("official split contains duplicate UID values")
    lifting_by_uid = {
        str(row["uid"]): index for index, row in enumerate(lifting.rows)
    }
    if len(lifting_by_uid) != len(lifting.rows):
        raise ValueError("base lifting manifest contains duplicate UID values")
    base_uids = [str(row["uid"]) for row in slat.rows]
    missing_split = [uid for uid in base_uids if uid not in split_by_uid]
    missing_lifting = [uid for uid in base_uids if uid not in lifting_by_uid]
    if missing_split or missing_lifting:
        raise RuntimeError(
            "base/split/lifting join incomplete: "
            f"split={missing_split[:8]} lifting={missing_lifting[:8]}"
        )
    if len(base_uids) == len(split["rows"]):
        split_uids = [str(row["uid"]) for row in split["rows"]]
        if base_uids != split_uids:
            raise RuntimeError("base cache order differs from the frozen official split")
    count = len(base_uids)
    if int(args.max_objects) > 0:
        count = min(count, int(args.max_objects))
    if count <= 0:
        raise ValueError("with-VGGT cache selection is empty")
    selected = []
    for base_index, row in enumerate(slat.rows[:count]):
        uid = str(row["uid"])
        source = split_by_uid[uid]
        render_path = Path(source["render_tar"]).expanduser().resolve(strict=True)
        target_path = Path(source["slat_npz"]).expanduser().resolve(strict=True)
        if render_path.stat().st_size != int(source["render_size"]):
            raise RuntimeError(f"official render size changed uid={uid}: {render_path}")
        if target_path.stat().st_size != int(source["slat_size"]):
            raise RuntimeError(f"official lh-slat size changed uid={uid}: {target_path}")
        base_target_path = Path(row["target_file"]).expanduser().resolve(strict=True)
        if base_target_path != target_path:
            raise RuntimeError(
                f"base cache/official split target path differs uid={uid}: "
                f"{base_target_path} != {target_path}"
            )
        selected.append(
            {
                "uid": uid,
                "object_uid": str(row["object_uid"]),
                "support_seed": int(row.get("support_seed", 42)),
                "base_index": base_index,
                "lifting_index": lifting_by_uid[uid],
                "source": source,
            }
        )
    if any(
        str(lifting.rows[row["lifting_index"]]["object_uid"]) != row["object_uid"]
        for row in selected
    ):
        raise RuntimeError("base SLat/lifting object identities differ")
    return {
        "split_path": split_path,
        "split": split,
        "base_slat_path": base_slat_path,
        "base_lifting_path": base_lifting_path,
        "slat": slat,
        "lifting": lifting,
        "selected": selected,
    }


def _sidecar_contract(
    args: argparse.Namespace,
    plan: dict[str, Any],
    encoder_assets: dict[str, Any],
) -> dict[str, Any]:
    lifting = plan["lifting"]
    return {
        "version": WITH_VGGT_CONTEXT_VERSION,
        "builder_version": BUILDER_VERSION,
        "protocol_sha256": str(plan["split"]["protocol_sha256"]),
        "split": str(plan["split"]["name"]),
        "base_cache": {
            "slat_manifest_sha256": sha256_file(plan["base_slat_path"]),
            "lifting_manifest_sha256": sha256_file(plan["base_lifting_path"]),
            "slat_config_hash": str(plan["slat"].config_hash),
            "lifting_config_hash": str(lifting.config_hash),
            "slat_normalization_hash": str(plan["slat"].slat_normalization_hash),
        },
        "selected_view_policy": (
            "read exact ordered view_ids from immutable base lifting cache; "
            "no view selection is executed"
        ),
        "selected_view_count": int(args.expected_selected_views),
        "shared_geometric_preprocessing": copy.deepcopy(
            lifting.config.get("geometric_preprocessing", {})
        ),
        "shared_geometric_preprocessing_hash": lifting.config.get(
            "geometric_preprocessing_hash"
        ),
        "native_condition": {
            "producer": (
                "TrellisVGGTTo3DPipeline.vggt_feat + encode_image + get_slat_cond"
            ),
            "vggt_layers": [4, 11, 17, 23],
            "dino_sequence": "full_cls_register_patch_sequence",
            "expected_dino_prefix_tokens": EXPECTED_DINO_PREFIX_TOKENS,
            "output_layout": "[views,tokens,1024]",
            "positive_context_materialized": True,
            "negative_context_policy": "runtime_zeros_like_positive",
            "base_dino_patch_replay": {
                "source": "same encode_image result used by native slat_vggt_cond",
                "reference": "immutable base lifting visual_patch_features",
                "comparison_dtype": "torch.float16",
                "max_abs_tolerance": BASE_DINO_PATCH_MAX_ABS_TOLERANCE,
                "mean_abs_tolerance": BASE_DINO_PATCH_MEAN_ABS_TOLERANCE,
            },
        },
        "camera_contract": {
            "vggt_camera_consumed": False,
            "vggt_depth_consumed": False,
            "posed_dino_uses_base_known_K_T": True,
            "known_K_T_replaced": False,
        },
        "native_ss_contract": (
            "unchanged base official no-VGGT Native-SS deployment binding; "
            "VGGT is used only for SLat cross-attention context"
        ),
        "encoder_assets": encoder_assets,
    }


def _load_condition_pipeline(
    *, pretrained: str, vggt_repo: str, device: torch.device
) -> Any:
    from trellis import models as trellis_models
    from trellis.pipelines.trellis_image_to_3d import (
        TrellisVGGTTo3DPipeline,
        VGGT,
    )

    if str(vggt_repo) != VGGT_REPO_ID:
        # ``from_pretrained`` currently freezes this repository internally.
        # Refuse a misleading CLI identity until the upstream constructor is
        # parameterized rather than pretending it loaded another repository.
        raise ValueError(
            f"current native pipeline freezes vggt_repo={VGGT_REPO_ID!r}, "
            f"got {vggt_repo!r}"
        )
    torch.cuda.set_device(device)
    # Load only the three modules that actually execute during sidecar build.
    # ``TrellisVGGTTo3DPipeline.from_pretrained`` first materializes Stock SS,
    # SLat Flow and every decoder even though this builder never calls them;
    # doing that in eight workers creates a large, avoidable host-RAM peak.
    # The exact same upstream constructors and native pipeline methods remain
    # in use for VGGT, DINO and slat_vggt_cond.
    pipeline = TrellisVGGTTo3DPipeline()
    slat_vggt_cond = trellis_models.from_pretrained(
        f"{pretrained}/ckpts/slat_vggt_cond"
    )
    pipeline.models = {"slat_vggt_cond": slat_vggt_cond}
    pipeline.slat_vggt_cond = slat_vggt_cond
    pipeline._init_image_cond_model("dinov2_vitl14_reg")
    pipeline.VGGT_dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    pipeline.VGGT_model = VGGT.from_pretrained(vggt_repo)
    for name in ("depth_head", "track_head", "camera_head", "point_head"):
        if hasattr(pipeline.VGGT_model, name):
            delattr(pipeline.VGGT_model, name)
    pipeline._device = device
    pipeline.low_vram = True
    for module in (
        pipeline.VGGT_model,
        pipeline.models["image_cond_model"],
        pipeline.slat_vggt_cond,
    ):
        module.eval()
        module.requires_grad_(False)
        module.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return pipeline


def _runtime_encoder_schema(pipeline: Any) -> dict[str, Any]:
    return {
        "vggt_aggregator": module_schema(pipeline.VGGT_model.aggregator),
        "dino_image_condition_model": module_schema(
            pipeline.models["image_cond_model"]
        ),
        "slat_vggt_condition": module_schema(pipeline.slat_vggt_cond),
        "deleted_unexecuted_vggt_heads": [
            "depth_head",
            "track_head",
            "camera_head",
            "point_head",
        ],
    }


def _validate_base_dino_patch_replay(
    image_cond: torch.Tensor,
    cached_visual_patch_features: Any,
    *,
    uid: str,
) -> dict[str, Any]:
    """Prove that the native context reused the frozen no-VGGT DINO input.

    The old official cache stores fp16 patch tokens.  Compare in that same
    representation so the check is meaningful across CUDA architectures while
    retaining tight, pre-registered bounds for normal kernel-level variation.
    """

    if not torch.is_tensor(image_cond) or image_cond.ndim != 4:
        raise ValueError(f"uid={uid} native DINO context must be [B,V,T,C]")
    if int(image_cond.shape[0]) != 1 or int(image_cond.shape[-1]) != 1024:
        raise ValueError(f"uid={uid} native DINO context batch/channel differs")
    if not torch.is_tensor(cached_visual_patch_features):
        raise ValueError(f"uid={uid} base DINO patch cache is not a tensor")
    replay = image_cond[0, :, EXPECTED_DINO_PREFIX_TOKENS:].detach().to(
        device="cpu", dtype=torch.float16
    )
    frozen = cached_visual_patch_features.detach().to(
        device="cpu", dtype=torch.float16
    )
    if replay.shape != frozen.shape:
        raise RuntimeError(
            f"uid={uid} native/base DINO patch shapes differ: "
            f"{tuple(replay.shape)} != {tuple(frozen.shape)}"
        )
    if not bool(torch.isfinite(replay.float()).all().item()) or not bool(
        torch.isfinite(frozen.float()).all().item()
    ):
        raise RuntimeError(f"uid={uid} native/base DINO patch contains non-finite values")
    difference = (replay.float() - frozen.float()).abs()
    maximum = float(difference.amax().item()) if difference.numel() else 0.0
    mean = float(difference.mean().item()) if difference.numel() else 0.0
    passed = bool(
        maximum <= BASE_DINO_PATCH_MAX_ABS_TOLERANCE
        and mean <= BASE_DINO_PATCH_MEAN_ABS_TOLERANCE
    )
    report = {
        "passed": passed,
        "comparison_dtype": "torch.float16",
        "shape": list(replay.shape),
        "max_abs": maximum,
        "mean_abs": mean,
        "max_abs_tolerance": BASE_DINO_PATCH_MAX_ABS_TOLERANCE,
        "mean_abs_tolerance": BASE_DINO_PATCH_MEAN_ABS_TOLERANCE,
    }
    if not passed:
        raise RuntimeError(f"uid={uid} native/base DINO patch replay differs: {report}")
    return report


def _paths(output: Path, uid: str) -> tuple[Path, Path]:
    return (
        output / "sidecars" / uid[:2] / f"{uid}.pt",
        output / "records" / uid[:2] / f"{uid}.json",
    )


def _validate_resumed_record(
    *, output: Path, row: dict[str, Any], sidecar_contract_hash: str
) -> dict[str, Any] | None:
    sidecar_path, record_path = _paths(output, row["uid"])
    if not sidecar_path.exists() and not record_path.exists():
        return None
    if sidecar_path.is_file() and not record_path.exists():
        # ``sidecar`` is committed first and ``record`` second.  A process kill
        # in that very small window leaves no completed identity record, so the
        # object is safe to recompute atomically on the same resume command.
        print(
            f"[official_slat_with_vggt:resume_incomplete] recompute uid={row['uid']}",
            flush=True,
        )
        return None
    if not sidecar_path.is_file() or not record_path.is_file():
        raise RuntimeError(
            f"incomplete with-VGGT object must be inspected: {sidecar_path} / {record_path}"
        )
    record = load_json(record_path)
    if set(record) != _RECORD_KEYS:
        raise RuntimeError(
            f"resumed with-VGGT record schema differs: "
            f"missing={sorted(_RECORD_KEYS - set(record))} "
            f"unexpected={sorted(set(record) - _RECORD_KEYS)}"
        )
    source_render = Path(row["source"]["render_tar"]).expanduser().resolve(strict=True)
    expected = {
        "format": WITH_VGGT_SIDECAR_FORMAT,
        "uid": row["uid"],
        "object_uid": row["object_uid"],
        "support_seed": int(row["support_seed"]),
        "base_index": int(row["base_index"]),
        "sidecar_contract_hash": sidecar_contract_hash,
        "sidecar_file": _relative_reference(sidecar_path, output),
        "sidecar_file_sha256": sha256_file(sidecar_path),
        "sidecar_file_size": sidecar_path.stat().st_size,
        "source_render_tar_sha256": sha256_file(source_render),
    }
    mismatch = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"resumed with-VGGT record mismatch={mismatch}")
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    required_payload_keys = {
        "format",
        "uid",
        "object_uid",
        "support_seed",
        "sidecar_contract_hash",
        "view_ids",
        "native_slat_vggt_cond",
        "negative_context_policy",
        "decoded_source_rgba_sha256",
        "processed_input_rgb_sha256",
        "vggt_camera_consumed",
        "known_K_T_replaced",
    }
    if not isinstance(payload, dict) or set(payload) != required_payload_keys:
        raise RuntimeError(f"resumed with-VGGT payload schema differs: {sidecar_path}")
    if (
        payload.get("format") != WITH_VGGT_SIDECAR_FORMAT
        or payload.get("uid") != row["uid"]
        or payload.get("object_uid") != row["object_uid"]
        or int(payload.get("support_seed", -1)) != int(row["support_seed"])
        or payload.get("sidecar_contract_hash") != sidecar_contract_hash
        or payload.get("negative_context_policy")
        != "runtime_zeros_like_positive"
        or payload.get("vggt_camera_consumed") is not False
        or payload.get("known_K_T_replaced") is not False
    ):
        raise RuntimeError(f"resumed with-VGGT payload semantics differ: {sidecar_path}")
    context = validate_native_slat_vggt_context_tensor(
        payload.get("native_slat_vggt_cond"),
        views=len(payload.get("view_ids", [])),
        uid=row["uid"],
    )
    view_ids = payload["view_ids"].to(torch.int64).tolist()
    if (
        view_ids != record.get("view_ids")
        or list(context.shape) != record.get("native_context_shape")
        or str(context.dtype) != record.get("native_context_dtype")
        or payload["decoded_source_rgba_sha256"]
        != record.get("decoded_source_rgba_sha256")
        or payload["processed_input_rgb_sha256"]
        != record.get("processed_input_rgb_sha256")
        or record.get("base_dino_patch_replay", {}).get("passed") is not True
        or record.get("vggt_forward_executed") is not True
        or record.get("vggt_camera_consumed") is not False
        or record.get("known_K_T_replaced") is not False
    ):
        raise RuntimeError(f"resumed with-VGGT payload/record differs: {sidecar_path}")
    return record


@torch.no_grad()
def _materialize(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    if int(args.worker_count) <= 0 or not 0 <= int(args.worker_index) < int(
        args.worker_count
    ):
        raise ValueError("worker_index must lie in [0,worker_count)")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    encoder_assets, encoder_diagnostics = _encoder_asset_identity(
        args.pretrained, args.vggt_repo
    )
    contract = _sidecar_contract(args, plan, encoder_assets)
    contract_hash = canonical_sha256(contract)
    assigned = [
        row
        for position, row in enumerate(plan["selected"])
        if position % int(args.worker_count) == int(args.worker_index)
    ]
    if not assigned:
        raise ValueError("with-VGGT worker has no assigned objects")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("native VGGT sidecar materialization requires CUDA")
    pipeline = _load_condition_pipeline(
        pretrained=args.pretrained,
        vggt_repo=args.vggt_repo,
        device=device,
    )
    runtime_schema = _runtime_encoder_schema(pipeline)
    records: list[dict[str, Any]] = []
    executed = 0
    reused = 0
    for position, row in enumerate(assigned, start=1):
        resumed = _validate_resumed_record(
            output=output,
            row=row,
            sidecar_contract_hash=contract_hash,
        )
        if resumed is not None:
            records.append(resumed)
            reused += 1
            print(
                f"[official_slat_with_vggt:reuse] {position}/{len(assigned)} {row['uid']}",
                flush=True,
            )
            continue
        lifting_sample = plan["lifting"][row["lifting_index"]]
        view_ids = [int(value) for value in lifting_sample["view_ids"].tolist()]
        if len(view_ids) != int(args.expected_selected_views):
            raise RuntimeError(
                f"uid={row['uid']} base selected views={len(view_ids)}; "
                f"expected={args.expected_selected_views}"
            )
        render_path = Path(row["source"]["render_tar"]).expanduser().resolve(
            strict=True
        )
        source_render_tar_sha256 = sha256_file(render_path)
        views, archive_audit = _load_views_with_audit(render_path, row["uid"])
        by_id = {int(value["id"]): value for value in views}
        if len(by_id) != len(views):
            raise RuntimeError(f"uid={row['uid']} render archive has duplicate view IDs")
        missing = [view_id for view_id in view_ids if view_id not in by_id]
        if missing:
            raise RuntimeError(
                f"uid={row['uid']} frozen views missing from render archive: {missing}"
            )
        selected = [by_id[view_id] for view_id in view_ids]
        rgba = [value["rgba"] for value in selected]
        shared = prepare_shared_object_arrays(
            [value[..., :3] for value in rgba],
            [value[..., 3] for value in rgba],
            resolution=518,
            foreground_margin=1.10,
            alpha_threshold=0.80,
        )
        if shared.contract != lifting_sample["preprocessing"]["shared_geometry"]:
            raise RuntimeError(f"uid={row['uid']} shared preprocessing contract changed")
        cached_affines = lifting_sample["source_to_feature_affines"].numpy()
        if not np.array_equal(shared.source_to_feature_affines, cached_affines):
            raise RuntimeError(f"uid={row['uid']} shared preprocessing affine changed")
        source_intrinsics = np.stack([value["intrinsic"] for value in selected])
        feature_intrinsics = transform_intrinsics(
            source_intrinsics, shared.source_to_feature_affines
        )
        if not np.allclose(
            source_intrinsics,
            lifting_sample["source_intrinsics"].numpy(),
            rtol=0.0,
            atol=0.0,
        ) or not np.allclose(
            feature_intrinsics,
            lifting_sample["intrinsics"].numpy(),
            rtol=1.0e-6,
            atol=1.0e-5,
        ):
            raise RuntimeError(f"uid={row['uid']} known K tensors changed")
        source_extrinsics = np.stack([value["extrinsic"] for value in selected])
        if not np.allclose(
            source_extrinsics,
            lifting_sample["extrinsics"].numpy(),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(f"uid={row['uid']} known T tensors changed")

        aggregated, image_tensor = pipeline.vggt_feat(shared.images)
        raw_image_cond = pipeline.encode_image(image_tensor)
        image_cond = normalize_image_cond(
            raw_image_cond, batch=1, views=len(view_ids)
        )
        base_dino_patch_replay = _validate_base_dino_patch_replay(
            image_cond,
            lifting_sample["visual_patch_features"],
            uid=row["uid"],
        )
        native = pipeline.get_slat_cond(
            image_cond, aggregated, num_samples=1
        )
        if len(native.get("cond", [])) != len(view_ids) or len(
            native.get("neg_cond", [])
        ) != len(view_ids):
            raise RuntimeError(f"uid={row['uid']} native SLat condition view mismatch")
        if not all(
            int(torch.count_nonzero(value).item()) == 0
            for value in native["neg_cond"]
        ):
            raise RuntimeError(f"uid={row['uid']} native negative context is not zero")
        context = torch.cat(
            [value.detach().cpu() for value in native["cond"]], dim=0
        ).contiguous()
        context = validate_native_slat_vggt_context_tensor(
            context, views=len(view_ids), uid=row["uid"]
        )
        dino_patch_count = int(lifting_sample["visual_patch_features"].shape[1])
        prefix_tokens = int(context.shape[1]) - dino_patch_count
        if prefix_tokens != EXPECTED_DINO_PREFIX_TOKENS:
            raise RuntimeError(
                f"uid={row['uid']} native context prefix tokens={prefix_tokens}; "
                f"expected={EXPECTED_DINO_PREFIX_TOKENS}"
            )
        decoded_rgba_hashes = [_array_sha256(value) for value in rgba]
        processed_rgb_hashes = [
            _array_sha256(np.asarray(image, dtype=np.uint8)) for image in shared.images
        ]
        sidecar_path, record_path = _paths(output, row["uid"])
        payload = {
            "format": WITH_VGGT_SIDECAR_FORMAT,
            "uid": row["uid"],
            "object_uid": row["object_uid"],
            "support_seed": int(row["support_seed"]),
            "sidecar_contract_hash": contract_hash,
            "view_ids": torch.tensor(view_ids, dtype=torch.int64),
            "native_slat_vggt_cond": context,
            "negative_context_policy": "runtime_zeros_like_positive",
            "decoded_source_rgba_sha256": decoded_rgba_hashes,
            "processed_input_rgb_sha256": processed_rgb_hashes,
            "vggt_camera_consumed": False,
            "known_K_T_replaced": False,
        }
        _atomic_torch_save(sidecar_path, payload)
        record = {
            "format": WITH_VGGT_SIDECAR_FORMAT,
            "uid": row["uid"],
            "object_uid": row["object_uid"],
            "support_seed": int(row["support_seed"]),
            "base_index": int(row["base_index"]),
            "sidecar_contract_hash": contract_hash,
            "sidecar_file": _relative_reference(sidecar_path, output),
            "sidecar_file_sha256": sha256_file(sidecar_path),
            "sidecar_file_size": sidecar_path.stat().st_size,
            "view_ids": view_ids,
            "native_context_shape": list(context.shape),
            "native_context_dtype": str(context.dtype),
            "dino_patch_tokens": dino_patch_count,
            "native_prefix_tokens": prefix_tokens,
            "decoded_source_rgba_sha256": decoded_rgba_hashes,
            "processed_input_rgb_sha256": processed_rgb_hashes,
            "source_render_tar_sha256": source_render_tar_sha256,
            "render_archive_audit": archive_audit,
            "base_dino_patch_replay": base_dino_patch_replay,
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
        del (
            lifting_sample,
            views,
            selected,
            rgba,
            shared,
            aggregated,
            image_tensor,
            raw_image_cond,
            image_cond,
            native,
            context,
        )
        torch.cuda.empty_cache()
        print(
            f"[official_slat_with_vggt] {position}/{len(assigned)} {row['uid']}",
            flush=True,
        )
    shard_report = {
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
        "encoder_asset_locations_diagnostic_only": encoder_diagnostics,
        "records": records,
    }
    shard_path = output / "shards" / (
        f"worker_{int(args.worker_index):02d}_of_{int(args.worker_count):02d}.json"
    )
    atomic_json(shard_path, shard_report)
    return shard_report


def _finalize(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve(strict=True)
    encoder_assets, encoder_diagnostics = _encoder_asset_identity(
        args.pretrained, args.vggt_repo
    )
    contract = _sidecar_contract(args, plan, encoder_assets)
    contract_hash = canonical_sha256(contract)
    records = []
    for row in plan["selected"]:
        record = _validate_resumed_record(
            output=output,
            row=row,
            sidecar_contract_hash=contract_hash,
        )
        if record is None:
            raise RuntimeError(f"missing with-VGGT sidecar uid={row['uid']}")
        records.append(record)
    samples = [
        {
            key: record[key]
            for key in (
                "uid",
                "object_uid",
                "support_seed",
                "base_index",
                "sidecar_file",
                "sidecar_file_sha256",
                "sidecar_file_size",
                "view_ids",
                "native_context_shape",
                "native_context_dtype",
                "decoded_source_rgba_sha256",
                "processed_input_rgb_sha256",
                "source_render_tar_sha256",
            )
        }
        for record in records
    ]
    base_cache = {
        "slat_manifest": _relative_reference(plan["base_slat_path"], output),
        "slat_manifest_sha256": sha256_file(plan["base_slat_path"]),
        "lifting_manifest": _relative_reference(plan["base_lifting_path"], output),
        "lifting_manifest_sha256": sha256_file(plan["base_lifting_path"]),
        "slat_config_hash": str(plan["slat"].config_hash),
        "lifting_config_hash": str(plan["lifting"].config_hash),
        "slat_normalization_hash": str(plan["slat"].slat_normalization_hash),
    }
    pair_binding = {
        "version": WITH_VGGT_CONTEXT_VERSION,
        "base_cache": base_cache,
        "sidecar_contract": contract,
        "sidecar_contract_hash": contract_hash,
        "sidecar_index_sha256": canonical_sha256(samples),
        "sample_count": len(samples),
        "ordered_uid_sha256": canonical_sha256([row["uid"] for row in samples]),
    }
    pair_identity = canonical_sha256(pair_binding)
    config = copy.deepcopy(plan["slat"].config)
    config["slat_input_context"] = {
        "version": WITH_VGGT_CONTEXT_VERSION,
        "stock_floor": "V0",
        "source": "native_reconviagen_vggt_plus_dinov2_slat_vggt_cond",
        "sidecar_contract_hash": contract_hash,
        "base_no_vggt_slat_config_hash": str(plan["slat"].config_hash),
        "selected_views": "exact ordered base lifting view_ids",
        "native_full_dino_sequence": True,
        "vggt_model_executed": True,
        "vggt_camera_consumed": False,
        "known_pose_dino_branch_unchanged": True,
        "negative_context_policy": "runtime_zeros_like_positive",
    }
    config_hash = canonical_sha256(config)
    common = {
        "pair_identity": pair_identity,
        "pair_binding": pair_binding,
        "output_dir": ".",
        "config": config,
        "config_hash": config_hash,
        "samples": samples,
        "sample_count": len(samples),
        "object_count": len({row["object_uid"] for row in samples}),
        "official_gt_support_only": True,
        "vggt_model_executed": True,
        "vggt_camera_consumed": False,
    }
    slat_manifest = {
        "format": WITH_VGGT_SLAT_MANIFEST_FORMAT,
        **common,
        "slat_normalization": copy.deepcopy(plan["slat"].slat_normalization),
        "slat_normalization_hash": str(plan["slat"].slat_normalization_hash),
    }
    lifting_manifest = {
        "format": WITH_VGGT_LIFTING_MANIFEST_FORMAT,
        **common,
        "lifting_role": (
            "immutable base known-K/T posed-DINO tensors; only Stock SLat "
            "cross-attention context is replaced by the sidecar"
        ),
    }
    slat_path = output / "with_vggt_slat_manifest.json"
    lifting_path = output / "with_vggt_lifting_manifest.json"
    _atomic_json_once_or_exact(slat_path, slat_manifest)
    _atomic_json_once_or_exact(lifting_path, lifting_manifest)
    report = {
        "format": WITH_VGGT_CACHE_REPORT_FORMAT,
        "passed": True,
        "complete": True,
        "protocol_sha256": plan["split"]["protocol_sha256"],
        "split": plan["split"]["name"],
        "object_count": len(samples),
        "selected_views": int(args.expected_selected_views),
        "slat_manifest": str(slat_path),
        "slat_manifest_sha256": sha256_file(slat_path),
        "lifting_manifest": str(lifting_path),
        "lifting_manifest_sha256": sha256_file(lifting_path),
        "pair_identity": pair_identity,
        "sidecar_contract": contract,
        "sidecar_contract_hash": contract_hash,
        "sidecar_bytes": int(sum(row["sidecar_file_size"] for row in records)),
        "vggt_forward_call_count": len(records),
        "vggt_model_executed": True,
        "vggt_camera_consumed": False,
        "known_K_T_replaced": False,
        "same_frozen_view_ids_as_base": True,
        "native_ss_changed": False,
        "base_cache_rewritten": False,
        "encoder_asset_locations_diagnostic_only": encoder_diagnostics,
        "scope_guard": (
            "official Train2000 with-VGGT SLat condition isolation; Native-SS, "
            "GT support, target SLat, known K/T, decoder and no-VGGT base cache "
            "remain unchanged"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    _atomic_json_once_or_exact(output / "report.json", report)
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preflight", "materialize", "finalize", "both"), default="preflight"
    )
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--base_slat_manifest", required=True)
    parser.add_argument("--base_lifting_manifest", required=True)
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
        raise ValueError("--mode both is only valid for a single worker")
    plan = _base_plan(args)
    encoder_assets, encoder_diagnostics = _encoder_asset_identity(
        args.pretrained, args.vggt_repo
    )
    preflight = {
        "passed": True,
        "mode": "preflight",
        "split": plan["split"]["name"],
        "protocol_sha256": plan["split"]["protocol_sha256"],
        "object_count": len(plan["selected"]),
        "base_slat_manifest": str(plan["base_slat_path"]),
        "base_slat_manifest_sha256": sha256_file(plan["base_slat_path"]),
        "base_lifting_manifest": str(plan["base_lifting_path"]),
        "base_lifting_manifest_sha256": sha256_file(plan["base_lifting_path"]),
        "exact_base_view_ids_only": True,
        "vggt_camera_consumed": False,
        "encoder_assets": encoder_assets,
        "encoder_asset_locations_diagnostic_only": encoder_diagnostics,
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
                        "slat_manifest",
                        "lifting_manifest",
                        "pair_identity",
                        "sidecar_bytes",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
