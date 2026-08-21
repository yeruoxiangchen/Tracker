#!/usr/bin/env python3
"""Four-worker frozen-SS support audit for the staged Objaverse cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION
from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.dino_only_condition import (
    validate_dino_only_lifting_contract,
)
from pose_point_depth_mv.eval_direct_flow import (
    bootstrap_mean_ci,
    positive_rate,
    summarize,
)
from pose_point_depth_mv.evaluate_native_ss_genrecon import (
    METRICS,
    aggregate_records,
    run_candidate,
)
from pose_point_depth_mv.native_ss_genrecon import (
    canonical_json_sha256,
    load_trainable_state_dict,
    require_disjoint_object_uids,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_EVAL,
    build_native_ss_no_vggt_components,
    select_manifest_order_object_indices,
    validate_native_ss_no_vggt_checkpoint,
    validate_no_vggt_cache_contract,
)


SELECTION_FORMAT = "pose_point_depth_mv.objaverse2k_ss_audit_selection.v1"
SELECTION_MARKER_FORMAT = (
    "pose_point_depth_mv.objaverse2k_ss_audit_selection_marker.v1"
)
SHARD_REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_ss_support_audit_shard.v1"
AGGREGATE_REPORT_FORMAT = "pose_point_depth_mv.objaverse2k_ss_support_audit.v1"
SELECTION_MARKER = "_OBJAVERSE2K_SS_AUDIT_SELECTION_COMPLETE.json"
DEFAULT_REFERENCE_REPORT = Path(
    "/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/"
    "ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json"
)
DEFAULT_CHECKPOINT = Path(
    "/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/"
    "ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt"
)
ABSOLUTE_METRICS = ("iou", "precision", "recall", "coord_count_ratio")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stable_score(seed: int, *parts: object) -> str:
    text = ":".join([str(int(seed)), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_shard_position(row: dict[str, Any]) -> int:
    value = row.get("source_shard_position")
    if value is None:
        raise ValueError("merged lifting row lacks source_shard_position")
    position = int(value)
    if position < 0:
        raise ValueError("source_shard_position must be nonnegative")
    return position


def select_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    expected_shards: int,
    objects_per_shard: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one sequence for stable-hash-ranked objects in every shard."""

    by_shard_object: dict[int, dict[str, list[tuple[int, dict[str, Any]]]]] = {}
    for source_index, row in enumerate(rows):
        shard = _source_shard_position(row)
        object_uid = str(row.get("object_uid", row.get("uid", "")))
        uid = str(row.get("uid", ""))
        if not object_uid or not uid:
            raise ValueError("lifting row lacks uid/object_uid")
        by_shard_object.setdefault(shard, {}).setdefault(object_uid, []).append(
            (source_index, row)
        )
    expected = list(range(int(expected_shards)))
    if sorted(by_shard_object) != expected:
        raise RuntimeError(
            f"source shards differ: observed={sorted(by_shard_object)} expected={expected}"
        )

    selected_rows: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    seen_objects: set[str] = set()
    for shard in expected:
        objects = by_shard_object[shard]
        if len(objects) < int(objects_per_shard):
            raise RuntimeError(
                f"shard {shard:03d} has {len(objects)} objects, needs {objects_per_shard}"
            )
        ranked_objects = sorted(
            objects,
            key=lambda object_uid: (stable_score(seed, "object", shard, object_uid), object_uid),
        )[: int(objects_per_shard)]
        for object_uid in ranked_objects:
            candidates = objects[object_uid]
            source_index, source_row = min(
                candidates,
                key=lambda item: (
                    stable_score(seed, "sequence", object_uid, item[1]["uid"]),
                    str(item[1]["uid"]),
                ),
            )
            if object_uid in seen_objects:
                raise RuntimeError(f"object occurs across source shards: {object_uid}")
            seen_objects.add(object_uid)
            selected_rows.append(dict(source_row))
            selection.append(
                {
                    "selection_position": len(selection),
                    "source_shard_position": shard,
                    "source_sample_index": int(source_index),
                    "object_uid": object_uid,
                    "uid": str(source_row["uid"]),
                    "available_sequence_count": len(candidates),
                    "object_rank_sha256": stable_score(
                        seed, "object", shard, object_uid
                    ),
                    "sequence_rank_sha256": stable_score(
                        seed, "sequence", object_uid, source_row["uid"]
                    ),
                }
            )
    return selected_rows, selection


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path).expanduser().resolve(), map_location="cpu")


def _selection_reusable(
    output_dir: Path,
    *,
    source_manifest_sha256: str,
    checkpoint_sha256: str,
    seed: int,
    expected_shards: int,
    objects_per_shard: int,
) -> bool:
    marker_path = output_dir / SELECTION_MARKER
    manifest_path = output_dir / "lifting_manifest.json"
    selection_path = output_dir / "selection.json"
    if not all(path.is_file() for path in (marker_path, manifest_path, selection_path)):
        return False
    marker = load_json(marker_path)
    expected = {
        "source_manifest_sha256": source_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": int(seed),
        "expected_shards": int(expected_shards),
        "objects_per_shard": int(objects_per_shard),
    }
    return bool(
        marker.get("format") == SELECTION_MARKER_FORMAT
        and marker.get("passed") is True
        and all(marker.get(key) == value for key, value in expected.items())
        and marker.get("manifest_sha256") == sha256_file(manifest_path)
        and marker.get("selection_sha256") == sha256_file(selection_path)
    )


def prepare_selection(args: argparse.Namespace) -> None:
    source_path = Path(args.source_manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_sha = sha256_file(source_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    if output_dir.exists():
        if _selection_reusable(
            output_dir,
            source_manifest_sha256=source_sha,
            checkpoint_sha256=checkpoint_sha,
            seed=int(args.seed),
            expected_shards=int(args.expected_shards),
            objects_per_shard=int(args.objects_per_shard),
        ):
            print(
                json.dumps(
                    {
                        "reused": True,
                        "manifest": str(output_dir / "lifting_manifest.json"),
                        "selection": str(output_dir / "selection.json"),
                    },
                    indent=2,
                )
            )
            return
        raise RuntimeError(f"immutable selection output already exists: {output_dir}")

    source = load_json(source_path)
    if (
        source.get("format") != LIFTING_CACHE_VERSION
        or source.get("passed") is not True
        or source.get("training_ready") is not True
        or not isinstance(source.get("samples"), list)
    ):
        raise RuntimeError("source lifting manifest is not training-ready")
    merge = source.get("shard_merge")
    if not isinstance(merge, dict) or int(merge.get("source_shard_count", -1)) != int(
        args.expected_shards
    ):
        raise RuntimeError("source manifest is not the expected shard merge")
    rows, selection_rows = select_balanced_rows(
        source["samples"],
        expected_shards=int(args.expected_shards),
        objects_per_shard=int(args.objects_per_shard),
        seed=int(args.seed),
    )
    expected_objects = int(args.expected_shards) * int(args.objects_per_shard)
    if len(rows) != expected_objects:
        raise AssertionError("balanced selection produced the wrong object count")

    checkpoint = _load_checkpoint(checkpoint_path)
    validate_native_ss_no_vggt_checkpoint(
        checkpoint,
        pretrained=str(args.pretrained),
        allow_v2_parent=False,
    )
    training_objects = checkpoint.get("data_identity", {}).get("object_uids")
    if not isinstance(training_objects, list):
        raise RuntimeError("checkpoint lacks training object identities")
    selected_objects = [str(row["object_uid"]) for row in selection_rows]
    require_disjoint_object_uids(selected_objects, training_objects)

    source_root = Path(str(source.get("output_dir", source_path.parent))).resolve()
    for row in rows:
        cache_file = Path(str(row["cache_file"]))
        cache_file = cache_file if cache_file.is_absolute() else source_root / cache_file
        if not cache_file.is_file():
            raise FileNotFoundError(cache_file)
        if str(row.get("cache_file_sha256", "")) != sha256_file(cache_file):
            raise RuntimeError(f"selected cache sample hash differs: {cache_file}")

    output_dir.mkdir(parents=True)
    selection_payload = {
        "format": SELECTION_FORMAT,
        "created_at_utc": utc_now(),
        "formal": False,
        "purpose": "frozen-SS training-distribution support reliability audit",
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_sha,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "selection_policy": "stable_sha256_rank_per_shard_then_one_sequence_per_object",
        "seed": int(args.seed),
        "expected_shards": int(args.expected_shards),
        "objects_per_shard": int(args.objects_per_shard),
        "object_count": expected_objects,
        "object_uid_hash": canonical_json_sha256(sorted(selected_objects)),
        "future_slat_training_overlap": True,
        "training_object_disjoint_from_frozen_ss": True,
        "rows": selection_rows,
    }
    selection_path = output_dir / "selection.json"
    atomic_json(selection_path, selection_payload)
    manifest = {
        "format": LIFTING_CACHE_VERSION,
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "source_cache_manifest": str(selection_path),
        "source_cache_manifest_sha256": sha256_file(selection_path),
        "source_lifting_manifest": str(source_path),
        "source_lifting_manifest_sha256": source_sha,
        "stock_condition_source": source.get("stock_condition_source"),
        "lifting_feature_source": (
            "frozen 64-object audit subset of direct DINO-only shard merge"
        ),
        "sample_count": expected_objects,
        "object_count": expected_objects,
        "failure_count": 0,
        "feature_metadata": source["feature_metadata"],
        "visual_feature_dim": source["visual_feature_dim"],
        "metadata_names": source["metadata_names"],
        "metadata_schema_hash": source["metadata_schema_hash"],
        "config": source["config"],
        "config_hash": source["config_hash"],
        "samples": rows,
        "passed": True,
        "training_ready": True,
        "audit_selection": {
            "format": SELECTION_FORMAT,
            "selection_sha256": sha256_file(selection_path),
            "selection_seed": int(args.seed),
            "expected_shards": int(args.expected_shards),
            "objects_per_shard": int(args.objects_per_shard),
        },
    }
    proxy = SimpleNamespace(
        visual_feature_dim=manifest["visual_feature_dim"],
        feature_metadata=manifest["feature_metadata"],
        config=manifest["config"],
        config_hash=manifest["config_hash"],
    )
    manifest["no_vggt_contract"] = validate_dino_only_lifting_contract(proxy)
    manifest_path = output_dir / "lifting_manifest.json"
    atomic_json(manifest_path, manifest)
    marker = {
        "format": SELECTION_MARKER_FORMAT,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "source_manifest_sha256": source_sha,
        "checkpoint_sha256": checkpoint_sha,
        "seed": int(args.seed),
        "expected_shards": int(args.expected_shards),
        "objects_per_shard": int(args.objects_per_shard),
        "object_count": expected_objects,
        "passed": True,
    }
    atomic_json(output_dir / SELECTION_MARKER, marker)
    print(json.dumps(marker, indent=2, ensure_ascii=False))


def validate_selection_bundle(
    manifest_path: str | Path,
    *,
    checkpoint_sha256: str,
    expected_objects: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify that a worker consumes the completed immutable Audit64 subset."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = load_json(path)
    audit = manifest.get("audit_selection")
    samples = manifest.get("samples")
    if (
        manifest.get("format") != LIFTING_CACHE_VERSION
        or manifest.get("passed") is not True
        or manifest.get("training_ready") is not True
        or not isinstance(audit, dict)
        or audit.get("format") != SELECTION_FORMAT
        or int(manifest.get("sample_count", -1)) != int(expected_objects)
        or int(manifest.get("object_count", -1)) != int(expected_objects)
        or not isinstance(samples, list)
        or len(samples) != int(expected_objects)
    ):
        raise RuntimeError("cache is not a completed frozen Audit64 selection")

    selection_path = Path(str(manifest.get("source_cache_manifest", ""))).expanduser()
    if not selection_path.is_absolute():
        selection_path = (path.parent / selection_path).resolve()
    marker_path = path.parent / SELECTION_MARKER
    if not selection_path.is_file() or not marker_path.is_file():
        raise RuntimeError("Audit64 selection payload or completion marker is missing")
    selection = load_json(selection_path)
    marker = load_json(marker_path)
    manifest_sha = sha256_file(path)
    selection_sha = sha256_file(selection_path)
    if (
        selection.get("format") != SELECTION_FORMAT
        or selection.get("formal") is not False
        or selection.get("future_slat_training_overlap") is not True
        or selection.get("training_object_disjoint_from_frozen_ss") is not True
        or int(selection.get("object_count", -1)) != int(expected_objects)
        or not isinstance(selection.get("rows"), list)
        or len(selection["rows"]) != int(expected_objects)
        or audit.get("selection_sha256") != selection_sha
        or marker.get("format") != SELECTION_MARKER_FORMAT
        or marker.get("passed") is not True
        or marker.get("manifest_sha256") != manifest_sha
        or marker.get("selection_sha256") != selection_sha
        or marker.get("checkpoint_sha256") != str(checkpoint_sha256)
        or int(marker.get("object_count", -1)) != int(expected_objects)
    ):
        raise RuntimeError("Audit64 selection identity or completion marker differs")

    sample_identity = [
        (
            str(row.get("uid", "")),
            str(row.get("object_uid", row.get("uid", ""))),
            _source_shard_position(row),
        )
        for row in samples
    ]
    selection_identity = [
        (
            str(row.get("uid", "")),
            str(row.get("object_uid", "")),
            int(row.get("source_shard_position", -1)),
        )
        for row in selection["rows"]
    ]
    object_uids = [row[1] for row in sample_identity]
    if (
        sample_identity != selection_identity
        or any(not uid or not object_uid for uid, object_uid, _ in sample_identity)
        or len(set(object_uids)) != int(expected_objects)
        or selection.get("object_uid_hash")
        != canonical_json_sha256(sorted(object_uids))
    ):
        raise RuntimeError("Audit64 manifest rows differ from the frozen selection")
    return manifest, selection


def _runtime_contract_without_source_hash(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "config_hash"}


def validate_semantic_cache_compatibility(
    dataset: PoseLiftingCacheDataset, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    observed_all = validate_no_vggt_cache_contract(dataset)
    observed = dict(observed_all["no_vggt"])
    feature_contract = checkpoint.get("data_identity", {}).get("feature_contract")
    if not isinstance(feature_contract, dict):
        raise RuntimeError("checkpoint lacks its no-VGGT feature contract")
    domains = feature_contract.get("mixed_domains")
    synthetic = domains.get("synthetic") if isinstance(domains, dict) else None
    frozen = synthetic.get("contract") if isinstance(synthetic, dict) else None
    if not isinstance(frozen, dict):
        raise RuntimeError("checkpoint lacks frozen synthetic cache contract")
    observed_runtime = _runtime_contract_without_source_hash(observed)
    frozen_runtime = _runtime_contract_without_source_hash(frozen)
    if observed_runtime != frozen_runtime:
        mismatch = {
            key: (frozen_runtime.get(key), observed_runtime.get(key))
            for key in sorted(set(frozen_runtime) | set(observed_runtime))
            if frozen_runtime.get(key) != observed_runtime.get(key)
        }
        raise RuntimeError(f"Objaverse cache runtime contract differs: {mismatch}")
    if observed["config_hash"] == frozen["config_hash"]:
        mode = "exact_training_component_config"
    else:
        mode = "runtime_semantics_equal_source_config_distinct"
    return {
        "mode": mode,
        "observed_config_hash": observed["config_hash"],
        "frozen_synthetic_config_hash": frozen["config_hash"],
        "runtime_contract": observed_runtime,
        "explicitly_ignored_for_runtime_equivalence": ["config_hash"],
        "formal": False,
    }


def load_reference_contract(
    path: str | Path, checkpoint_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = Path(path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = load_json(report_path)
    if payload.get("format") != NATIVE_SS_NO_VGGT_EVAL or payload.get("passed") is not True:
        raise RuntimeError("reference SS report is not a passed no-VGGT evaluation")
    protocol = payload.get("protocol")
    calibrated = payload.get("calibrated_parameters")
    if not isinstance(protocol, dict) or not isinstance(calibrated, dict):
        raise RuntimeError("reference SS report is malformed")
    if (
        str(protocol.get("checkpoint_sha256", "")) != sha256_file(checkpoint)
        or Path(str(protocol.get("checkpoint", ""))).resolve() != checkpoint
    ):
        raise RuntimeError("reference report and audit checkpoint differ")
    frozen = {
        "weights": str(protocol["weights"]),
        "steps": int(protocol["steps"]),
        "joint_seeds": [int(value) for value in protocol["joint_seeds"]],
        "cfg_strength": float(calibrated["cfg_strength"]),
        "cfg_interval": [float(value) for value in protocol["cfg_interval"]],
        "guidance_rescale": float(protocol["guidance_rescale"]),
        "rescale_t": float(protocol["rescale_t"]),
        "amp_dtype": str(protocol["amp_dtype"]),
        "condition_scale_policy": str(protocol["condition_scale_policy"]),
        "post_cfg_cap": bool(protocol["post_cfg_cap"]),
    }
    expected = {
        "weights": "ema",
        "steps": 25,
        "joint_seeds": [42, 43, 44],
        "cfg_strength": 3.0,
        "cfg_interval": [0.5, 1.0],
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
        "amp_dtype": "bf16",
        "condition_scale_policy": "learned_projection_only",
        "post_cfg_cap": False,
    }
    if frozen != expected:
        raise RuntimeError(f"reference deployment differs from frozen audit contract: {frozen}")
    return payload, frozen


def _candidate_namespace(contract: dict[str, Any], bootstrap_samples: int) -> argparse.Namespace:
    return argparse.Namespace(
        mode="objaverse2k_support_audit",
        steps=int(contract["steps"]),
        cfg_interval=",".join(str(value) for value in contract["cfg_interval"]),
        guidance_rescale=float(contract["guidance_rescale"]),
        rescale_t=float(contract["rescale_t"]),
        amp_dtype=str(contract["amp_dtype"]),
        weights=str(contract["weights"]),
        bootstrap_samples=int(bootstrap_samples),
    )


def _pose_advantage(
    correct_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    control = {
        (str(row["object_uid"]), int(row["seed"])): row for row in control_records
    }
    expected_keys = {
        (str(row["object_uid"]), int(row["seed"])) for row in correct_records
    }
    if set(control) != expected_keys:
        raise RuntimeError("correct/pose-control record identities differ")
    by_object: dict[str, list[float]] = defaultdict(list)
    for row in correct_records:
        key = (str(row["object_uid"]), int(row["seed"]))
        by_object[key[0]].append(
            float(row["full"]["iou"] - control[key]["full"]["iou"])
        )
    values = [float(np.mean(rows)) for _, rows in sorted(by_object.items())]
    return {
        **summarize(values),
        "positive_rate": positive_rate(values),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(
            values, samples=int(bootstrap_samples), seed=99001
        ),
    }


def validate_candidate_record_identity(
    candidate: dict[str, Any],
    *,
    expected_objects: list[str],
    expected_seeds: list[int],
    projection_mode: str,
) -> None:
    records = candidate.get("records")
    if not isinstance(records, list):
        raise RuntimeError("candidate report lacks records")
    expected_keys = {
        (str(object_uid), int(seed))
        for object_uid in expected_objects
        for seed in expected_seeds
    }
    actual_keys = [
        (str(row.get("object_uid", "")), int(row.get("seed", -1)))
        for row in records
    ]
    if (
        len(actual_keys) != len(expected_keys)
        or len(set(actual_keys)) != len(actual_keys)
        or set(actual_keys) != expected_keys
        or any(str(row.get("projection_mode", "")) != projection_mode for row in records)
        or int(candidate.get("object_count", -1)) != len(expected_objects)
        or int(candidate.get("record_count", -1)) != len(expected_keys)
    ):
        raise RuntimeError(
            f"{projection_mode} candidate object/seed record identities differ"
        )


def run_worker(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache_manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    reference_path = Path(args.reference_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"immutable worker output already exists: {output_dir}")
    if not 0 <= int(args.worker_index) < int(args.num_workers):
        raise ValueError("worker_index must lie in [0,num_workers)")
    if int(args.expected_objects) % int(args.num_workers) != 0:
        raise ValueError("expected_objects must divide evenly across workers")

    checkpoint_sha = sha256_file(checkpoint_path)
    validate_selection_bundle(
        cache_path,
        checkpoint_sha256=checkpoint_sha,
        expected_objects=int(args.expected_objects),
    )
    reference, frozen = load_reference_contract(reference_path, checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    validate_native_ss_no_vggt_checkpoint(
        checkpoint,
        pretrained=str(args.pretrained),
        allow_v2_parent=False,
    )
    dataset = PoseLiftingCacheDataset(cache_path, indices="all")
    if len({str(row.get("object_uid", row["uid"])) for row in dataset.rows}) != int(
        args.expected_objects
    ):
        raise RuntimeError("audit cache does not contain the expected object count")
    per_worker = int(args.expected_objects) // int(args.num_workers)
    start = int(args.worker_index) * per_worker
    end = start + per_worker
    selected = select_manifest_order_object_indices(dataset.rows, start=start, end=end)
    if len(selected) != per_worker:
        raise RuntimeError("worker object slice is incomplete")
    selected_objects = [
        str(dataset.rows[index].get("object_uid", dataset.rows[index]["uid"]))
        for index in selected
    ]
    training_objects = checkpoint.get("data_identity", {}).get("object_uids")
    if not isinstance(training_objects, list):
        raise RuntimeError("checkpoint lacks training object identities")
    require_disjoint_object_uids(selected_objects, training_objects)
    compatibility = validate_semantic_cache_compatibility(dataset, checkpoint)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    saved = checkpoint["args"]
    model_sampler, model, decoder, model_summary, defaults = (
        build_native_ss_no_vggt_components(
            pretrained=str(args.pretrained),
            lora_rank=int(saved["lora_rank"]),
            lora_alpha=int(saved["lora_alpha"]),
            condition_channels=int(saved["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("SS support audit requires the frozen decoder")
    load_trainable_state_dict(model, checkpoint["ema_trainable_state"])
    model.eval()
    decoder.eval()
    runtime_args = _candidate_namespace(frozen, int(args.bootstrap_samples))
    amp_dtype = torch.bfloat16
    baseline_cache: dict[tuple[int, int, float], dict[str, Any]] = {}
    correct = run_candidate(
        dataset=dataset,
        selected=selected,
        seeds=list(frozen["joint_seeds"]),
        model=model,
        decoder=decoder,
        model_sampler=model_sampler,
        sampler_defaults=defaults,
        args=runtime_args,
        cfg_strength=float(frozen["cfg_strength"]),
        projection_mode="correct",
        device=device,
        use_amp=True,
        amp_dtype=amp_dtype,
        audit_disabled=True,
        baseline_cache=baseline_cache,
    )
    control = run_candidate(
        dataset=dataset,
        selected=selected,
        seeds=list(frozen["joint_seeds"]),
        model=model,
        decoder=decoder,
        model_sampler=model_sampler,
        sampler_defaults=defaults,
        args=runtime_args,
        cfg_strength=float(frozen["cfg_strength"]),
        projection_mode="pose_cyclic1",
        device=device,
        use_amp=True,
        amp_dtype=amp_dtype,
        audit_disabled=False,
        baseline_cache=baseline_cache,
    )
    validate_candidate_record_identity(
        correct,
        expected_objects=selected_objects,
        expected_seeds=list(frozen["joint_seeds"]),
        projection_mode="correct",
    )
    validate_candidate_record_identity(
        control,
        expected_objects=selected_objects,
        expected_seeds=list(frozen["joint_seeds"]),
        projection_mode="pose_cyclic1",
    )
    pose_summary = _pose_advantage(
        correct["records"],
        control["records"],
        bootstrap_samples=int(args.bootstrap_samples),
    )
    integrity_checks = {
        "object_count": int(correct["object_count"]) == per_worker,
        "correct_record_count": int(correct["record_count"])
        == per_worker * len(frozen["joint_seeds"]),
        "control_record_count": int(control["record_count"])
        == per_worker * len(frozen["joint_seeds"]),
        "same_initial_noise": all(
            row.get("same_initial_noise") is True
            for row in [*correct["records"], *control["records"]]
        ),
        "disabled_stock_equivalence": bool(
            correct.get("disabled_stock_equivalence", {}).get("passed") is True
        ),
        "ss_training_object_disjoint": True,
        "semantic_cache_compatibility": True,
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(f"worker integrity checks failed: {integrity_checks}")
    report = {
        "format": SHARD_REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "complete": True,
        "formal": False,
        "quality_decision": "deferred_to_four_worker_aggregate",
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "object_start": start,
        "object_end": end,
        "object_uids": sorted(selected_objects),
        "object_uid_hash": canonical_json_sha256(sorted(selected_objects)),
        "cache_manifest": str(cache_path),
        "cache_manifest_sha256": sha256_file(cache_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "reference_report": str(reference_path),
        "reference_report_sha256": sha256_file(reference_path),
        "frozen_deployment": frozen,
        "cache_compatibility": compatibility,
        "future_slat_training_overlap": True,
        "ss_training_object_disjoint": True,
        "integrity_checks": integrity_checks,
        "correct": correct,
        "pose_cyclic_control": control,
        "correct_over_pose_control_iou": pose_summary,
        "model_summary": model_summary,
        "scope_guard": (
            "training-distribution frozen-SS support audit only; not a held-out "
            "SLat, Mesh, or ReconViaGen quality claim"
        ),
    }
    report["report_identity_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "created_at_utc"}
    )
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "report.json", report)
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"worker: {args.worker_index}/{args.num_workers}",
                f"objects: {correct['object_count']}",
                f"records per branch: {correct['record_count']}",
                f"iou gain mean: {correct['summary']['iou_gain']['mean']:+.8f}",
                f"pose advantage mean: {pose_summary['mean']:+.8f}",
                "complete: true",
                report["scope_guard"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"complete": True, "report": str(output_dir / "report.json")}))


def aggregate_absolute_records(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_object[str(row["object_uid"])].append(row)
    summaries: dict[str, Any] = {}
    for position, metric in enumerate(ABSOLUTE_METRICS):
        values = [
            float(np.mean([float(row["full"][metric]) for row in object_rows]))
            for _, object_rows in sorted(by_object.items())
        ]
        summaries[metric] = {
            **summarize(values),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=int(seed) + position,
            ),
        }
    summaries["full_empty_record_count"] = sum(
        int(int(row["full_count"]) <= 0) for row in records
    )
    summaries["full_empty_record_rate"] = float(
        summaries["full_empty_record_count"] / max(len(records), 1)
    )
    return summaries


def _reference_absolute(report: dict[str, Any], bootstrap_samples: int) -> dict[str, Any]:
    records = report.get("correct", {}).get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("reference report lacks correct-branch records")
    return aggregate_absolute_records(
        records, bootstrap_samples=int(bootstrap_samples), seed=88001
    )


def aggregate_reports(
    reports: list[dict[str, Any]],
    *,
    selection_manifest: dict[str, Any],
    reference_report: dict[str, Any],
    bootstrap_samples: int,
    min_absolute_retention: float,
    min_count_ratio: float,
    max_count_ratio: float,
    min_iou_win_rate: float,
) -> dict[str, Any]:
    worker_indices = sorted(int(report.get("worker_index", -1)) for report in reports)
    if worker_indices != list(range(len(reports))):
        raise RuntimeError(f"worker reports are incomplete/duplicated: {worker_indices}")
    if not reports or len(reports) != int(reports[0].get("num_workers", -1)):
        raise RuntimeError("report count differs from frozen num_workers")
    invariant_fields = (
        "num_workers",
        "cache_manifest_sha256",
        "checkpoint_sha256",
        "reference_report_sha256",
        "frozen_deployment",
        "cache_compatibility",
    )
    first = reports[0]
    for report in reports:
        if report.get("format") != SHARD_REPORT_FORMAT or report.get("complete") is not True:
            raise RuntimeError("input is not a completed SS audit shard report")
        mismatch = {
            field: (first.get(field), report.get(field))
            for field in invariant_fields
            if report.get(field) != first.get(field)
        }
        if mismatch:
            raise RuntimeError(f"worker protocols differ: {mismatch}")
        if not all(report.get("integrity_checks", {}).values()):
            raise RuntimeError("worker report has failed integrity checks")

    ordered_expected_objects = [
        str(row.get("object_uid", row.get("uid", "")))
        for row in selection_manifest["samples"]
    ]
    expected_objects = set(ordered_expected_objects)
    if len(expected_objects) != len(ordered_expected_objects):
        raise RuntimeError("frozen selection contains duplicate objects")
    report_objects: set[str] = set()
    correct_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    ranges = []
    for report in sorted(reports, key=lambda row: int(row["worker_index"])):
        objects = set(str(value) for value in report["object_uids"])
        start = int(report["object_start"])
        end = int(report["object_end"])
        expected_slice = set(ordered_expected_objects[start:end])
        if objects != expected_slice:
            raise RuntimeError("worker objects differ from its frozen selection slice")
        overlap = report_objects.intersection(objects)
        if overlap:
            raise RuntimeError(f"worker object overlap: {sorted(overlap)[:8]}")
        report_objects.update(objects)
        ranges.append((start, end))
        seeds = [int(value) for value in report["frozen_deployment"]["joint_seeds"]]
        validate_candidate_record_identity(
            report["correct"],
            expected_objects=sorted(objects),
            expected_seeds=seeds,
            projection_mode="correct",
        )
        validate_candidate_record_identity(
            report["pose_cyclic_control"],
            expected_objects=sorted(objects),
            expected_seeds=seeds,
            projection_mode="pose_cyclic1",
        )
        correct_records.extend(report["correct"]["records"])
        control_records.extend(report["pose_cyclic_control"]["records"])
    if report_objects != expected_objects:
        raise RuntimeError("worker object union differs from frozen selection")
    expected_ranges = [
        (
            index * len(expected_objects) // len(reports),
            (index + 1) * len(expected_objects) // len(reports),
        )
        for index in range(len(reports))
    ]
    if ranges != expected_ranges:
        raise RuntimeError(f"worker object ranges differ: {ranges} != {expected_ranges}")

    object_rows, summaries, count_summary = aggregate_records(
        correct_records,
        bootstrap_samples=int(bootstrap_samples),
        seed=91300,
    )
    control_object_rows, control_summaries, control_count_summary = aggregate_records(
        control_records,
        bootstrap_samples=int(bootstrap_samples),
        seed=92300,
    )
    pose_summary = _pose_advantage(
        correct_records,
        control_records,
        bootstrap_samples=int(bootstrap_samples),
    )
    absolute = aggregate_absolute_records(
        correct_records, bootstrap_samples=int(bootstrap_samples), seed=87001
    )
    reference_absolute = _reference_absolute(reference_report, int(bootstrap_samples))
    retention = {
        metric: float(absolute[metric]["mean"] / reference_absolute[metric]["mean"])
        for metric in ("iou", "precision", "recall")
    }
    checks = {
        "object_count_64": len(object_rows) == 64,
        "record_count_192": len(correct_records) == 64 * 3,
        "full_support_nonempty": int(absolute["full_empty_record_count"]) == 0,
        "stock_support_nonempty": int(count_summary["stock_empty_record_count"]) == 0,
        "absolute_iou_retention": retention["iou"] >= float(min_absolute_retention),
        "absolute_precision_retention": retention["precision"]
        >= float(min_absolute_retention),
        "absolute_recall_retention": retention["recall"]
        >= float(min_absolute_retention),
        "absolute_target_count_ratio_lower": absolute["coord_count_ratio"]["mean"]
        >= 0.75,
        "absolute_target_count_ratio_upper": absolute["coord_count_ratio"]["mean"]
        <= 1.50,
        "iou_gain_mean": summaries["iou_gain"]["mean"] >= 0.0,
        "recall_gain_mean": summaries["recall_gain"]["mean"] >= 0.0,
        "latent_mse_gain_mean": summaries["latent_mse_gain"]["mean"] >= 0.0,
        "iou_object_win_rate": summaries["iou_gain"]["positive_rate"]
        >= float(min_iou_win_rate),
        "full_stock_count_ratio_lower": count_summary["full_stock_count_ratio"]["mean"]
        >= float(min_count_ratio),
        "full_stock_count_ratio_upper": count_summary["full_stock_count_ratio"]["mean"]
        <= float(max_count_ratio),
        "correct_pose_advantage": pose_summary["mean"] > 0.0,
        "disabled_stock_equivalence_all_workers": all(
            report["correct"]["disabled_stock_equivalence"]["passed"] is True
            for report in reports
        ),
    }
    passed = all(checks.values())
    return {
        "format": AGGREGATE_REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": passed,
        "formal": False,
        "training_distribution_audit": True,
        "future_slat_training_overlap": True,
        "object_count": len(object_rows),
        "record_count_per_branch": len(correct_records),
        "worker_count": len(reports),
        "frozen_deployment": first["frozen_deployment"],
        "cache_compatibility": first["cache_compatibility"],
        "thresholds": {
            "min_absolute_retention_vs_old_dev32": float(min_absolute_retention),
            "absolute_target_count_ratio": [0.75, 1.50],
            "full_stock_count_ratio": [float(min_count_ratio), float(max_count_ratio)],
            "min_iou_win_rate": float(min_iou_win_rate),
            "relative_gain_means": "IoU/recall/latent-MSE >= 0",
            "correct_pose_iou_advantage_mean": "> 0",
        },
        "checks": checks,
        "correct": {
            "object_count": len(object_rows),
            "record_count": len(correct_records),
            "summary": summaries,
            "count_summary": count_summary,
            "absolute_summary": absolute,
            "object_rows": object_rows,
            "records": correct_records,
        },
        "pose_cyclic_control": {
            "object_count": len(control_object_rows),
            "record_count": len(control_records),
            "summary": control_summaries,
            "count_summary": control_count_summary,
            "object_rows": control_object_rows,
            "records": control_records,
        },
        "correct_over_pose_control_iou": pose_summary,
        "old_dev32_reference_absolute_summary": reference_absolute,
        "absolute_retention_vs_old_dev32": retention,
        "scope_guard": (
            "This audit only decides whether the frozen SS support is non-degenerate "
            "for the staged Objaverse SLat experiment. It is not held-out "
            "generalization or a SLat/Mesh/ReconViaGen result."
        ),
    }


def run_aggregate(args: argparse.Namespace) -> None:
    selection_path = Path(args.cache_manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    reference_path = Path(args.reference_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"immutable aggregate output already exists: {output_dir}")
    checkpoint_sha = sha256_file(checkpoint_path)
    selection, _ = validate_selection_bundle(
        selection_path,
        checkpoint_sha256=checkpoint_sha,
        expected_objects=64,
    )
    reference, _ = load_reference_contract(reference_path, checkpoint_path)
    report_paths = [Path(path).expanduser().resolve() for path in args.shard_report]
    if len(report_paths) != 4 or len(set(report_paths)) != 4:
        raise RuntimeError("aggregate requires exactly four unique shard reports")
    reports = [load_json(path) for path in report_paths]
    expected_identity = {
        "cache_manifest_sha256": sha256_file(selection_path),
        "checkpoint_sha256": checkpoint_sha,
        "reference_report_sha256": sha256_file(reference_path),
    }
    for report in reports:
        mismatch = {
            key: (value, report.get(key))
            for key, value in expected_identity.items()
            if report.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"worker report is bound to different inputs: {mismatch}")
    aggregate = aggregate_reports(
        reports,
        selection_manifest=selection,
        reference_report=reference,
        bootstrap_samples=int(args.bootstrap_samples),
        min_absolute_retention=float(args.min_absolute_retention),
        min_count_ratio=float(args.min_count_ratio),
        max_count_ratio=float(args.max_count_ratio),
        min_iou_win_rate=float(args.min_iou_win_rate),
    )
    aggregate.update(
        {
            "cache_manifest": str(selection_path),
            "cache_manifest_sha256": sha256_file(selection_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "reference_report": str(reference_path),
            "reference_report_sha256": sha256_file(reference_path),
            "input_reports": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in report_paths
            ],
        }
    )
    aggregate["report_identity_sha256"] = canonical_json_sha256(
        {key: value for key, value in aggregate.items() if key != "created_at_utc"}
    )
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "report.json", aggregate)
    correct = aggregate["correct"]
    absolute = correct["absolute_summary"]
    lines = [
        "Objaverse2K frozen-SS support audit64",
        "=" * 42,
        f"passed: {str(aggregate['passed']).lower()}",
        "formal: false",
        f"objects: {aggregate['object_count']}",
        f"records per branch: {aggregate['record_count_per_branch']}",
        f"full absolute IoU: {absolute['iou']['mean']:.8f}",
        f"full absolute precision: {absolute['precision']['mean']:.8f}",
        f"full absolute recall: {absolute['recall']['mean']:.8f}",
        (
            "absolute retention vs old dev32: "
            f"iou={aggregate['absolute_retention_vs_old_dev32']['iou']:.6f} "
            f"precision={aggregate['absolute_retention_vs_old_dev32']['precision']:.6f} "
            f"recall={aggregate['absolute_retention_vs_old_dev32']['recall']:.6f}"
        ),
    ]
    for metric in METRICS:
        row = correct["summary"][metric]
        lines.append(
            f"{metric}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} CI={row['bootstrap_mean_95_ci']}"
        )
    lines.extend(
        [
            f"correct-over-cyclic IoU: {aggregate['correct_over_pose_control_iou']}",
            f"checks: {aggregate['checks']}",
            aggregate["scope_guard"],
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    raise SystemExit(0 if aggregate["passed"] else 2)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze a shard-balanced audit64")
    prepare.add_argument("--source_manifest", required=True)
    prepare.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--expected_shards", type=int, default=8)
    prepare.add_argument("--objects_per_shard", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=20260811)
    prepare.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    prepare.set_defaults(func=prepare_selection)

    worker = subparsers.add_parser("worker", help="run one GPU audit shard")
    worker.add_argument("--cache_manifest", required=True)
    worker.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    worker.add_argument("--reference_report", default=str(DEFAULT_REFERENCE_REPORT))
    worker.add_argument("--output_dir", required=True)
    worker.add_argument("--worker_index", type=int, required=True)
    worker.add_argument("--num_workers", type=int, default=4)
    worker.add_argument("--expected_objects", type=int, default=64)
    worker.add_argument("--bootstrap_samples", type=int, default=5000)
    worker.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    worker.set_defaults(func=run_worker)

    aggregate = subparsers.add_parser("aggregate", help="combine four audit shards")
    aggregate.add_argument("--cache_manifest", required=True)
    aggregate.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    aggregate.add_argument("--reference_report", default=str(DEFAULT_REFERENCE_REPORT))
    aggregate.add_argument("--shard_report", action="append", required=True)
    aggregate.add_argument("--output_dir", required=True)
    aggregate.add_argument("--bootstrap_samples", type=int, default=10000)
    aggregate.add_argument("--min_absolute_retention", type=float, default=0.80)
    aggregate.add_argument("--min_count_ratio", type=float, default=0.85)
    aggregate.add_argument("--max_count_ratio", type=float, default=1.25)
    aggregate.add_argument("--min_iou_win_rate", type=float, default=0.50)
    aggregate.set_defaults(func=run_aggregate)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
