"""One-pass acceptance audit for the new official with-VGGT cache delta.

The historical official no-VGGT cache is bound by its two frozen manifest
SHA256 values.  This audit deliberately does not re-hash every historical
base artifact.  It reads every newly materialized with-VGGT sidecar exactly
once, using the same bytes for SHA256 verification and ``torch.load`` schema
validation, and performs deterministic spot checks of base lifting view IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any

import torch

from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache import (
    WITH_VGGT_CACHE_REPORT_FORMAT,
    WITH_VGGT_SIDECAR_FORMAT,
    WithVGGTNativeConditionSLatDataset,
    validate_native_slat_vggt_context_tensor,
)


_PAYLOAD_KEYS = {
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


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(parent: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    return path.resolve(strict=True)


def _spot_indices(count: int, requested: int) -> list[int]:
    take = min(max(int(requested), 0), int(count))
    if take == 0:
        return []
    if take == 1:
        return [0]
    return sorted(
        {round(index * (count - 1) / (take - 1)) for index in range(take)}
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--expected_objects", type=int, default=2000)
    parser.add_argument("--expected_workers", type=int, default=8)
    parser.add_argument("--base_lifting_spot_checks", type=int, default=64)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument(
        "--max_sidecars",
        type=int,
        default=0,
        help="diagnostic-only prefix limit; a limited run is not P6-complete",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    slat_path = Path(args.cache_manifest).expanduser().resolve(strict=True)
    lifting_path = Path(args.lifting_cache_manifest).expanduser().resolve(strict=True)
    dataset = WithVGGTNativeConditionSLatDataset(
        slat_path,
        lifting_path,
        verify_hashes=False,
    )
    count = len(dataset)
    if count != int(args.expected_objects):
        raise RuntimeError(
            f"with-VGGT object count differs: {count} != {args.expected_objects}"
        )

    report_path = slat_path.parent / "report.json"
    report = _json(report_path)
    report_without_hash = dict(report)
    saved_report_hash = report_without_hash.pop("report_sha256", None)
    if (
        report.get("format") != WITH_VGGT_CACHE_REPORT_FORMAT
        or report.get("passed") is not True
        or report.get("complete") is not True
        or int(report.get("object_count", -1)) != count
        or report.get("pair_identity") != dataset.pair_identity
        or report.get("sidecar_contract_hash") != dataset.sidecar_contract_hash
        or saved_report_hash != canonical_sha256(report_without_hash)
        or report.get("slat_manifest_sha256") != sha256_file(slat_path)
        or report.get("lifting_manifest_sha256") != sha256_file(lifting_path)
        or report.get("same_frozen_view_ids_as_base") is not True
        or report.get("vggt_model_executed") is not True
        or report.get("vggt_camera_consumed") is not False
        or report.get("known_K_T_replaced") is not False
        or report.get("native_ss_changed") is not False
        or report.get("base_cache_rewritten") is not False
    ):
        raise RuntimeError("with-VGGT final builder report contract differs")

    shard_paths = sorted((slat_path.parent / "shards").glob("worker_*_of_*.json"))
    if len(shard_paths) != int(args.expected_workers):
        raise RuntimeError(
            f"with-VGGT worker report count differs: {len(shard_paths)}"
        )
    shard_records: dict[tuple[str, int], dict[str, Any]] = {}
    assigned = 0
    for expected_index, path in enumerate(shard_paths):
        shard = _json(path)
        records = shard.get("records")
        if (
            shard.get("passed") is not True
            or shard.get("mode") != "materialize"
            or int(shard.get("worker_index", -1)) != expected_index
            or int(shard.get("worker_count", -1)) != int(args.expected_workers)
            or shard.get("sidecar_contract_hash") != dataset.sidecar_contract_hash
            or not isinstance(records, list)
            or len(records) != int(shard.get("assigned_object_count", -1))
            or int(shard.get("materialized_object_count", 0))
            + int(shard.get("reused_object_count", 0))
            != len(records)
        ):
            raise RuntimeError(f"invalid with-VGGT worker report: {path}")
        assigned += len(records)
        for record in records:
            identity = (str(record.get("uid", "")), int(record.get("support_seed", -1)))
            if not identity[0] or identity in shard_records:
                raise RuntimeError(f"duplicate worker identity={identity}")
            shard_records[identity] = record
    if assigned != count:
        raise RuntimeError(f"worker coverage differs: {assigned} != {count}")

    limit = count if int(args.max_sidecars) <= 0 else min(
        count, int(args.max_sidecars)
    )
    rows = dataset.sidecar_rows[:limit]
    total_bytes = sum(int(row["sidecar_file_size"]) for row in rows)
    verified_bytes = 0
    started = time.monotonic()
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    for position, row in enumerate(rows, start=1):
        path = _resolve(slat_path.parent, str(row["sidecar_file"]))
        expected_size = int(row["sidecar_file_size"])
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"sidecar size changed: {path}")
        raw = path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != row["sidecar_file_sha256"]:
            raise RuntimeError(f"sidecar SHA256 changed: {path}")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            raise RuntimeError(f"sidecar payload schema differs: {path}")
        identity = (str(row["uid"]), int(row["support_seed"]))
        worker_record = shard_records.get(identity)
        view_ids = payload.get("view_ids")
        if (
            payload.get("format") != WITH_VGGT_SIDECAR_FORMAT
            or str(payload.get("uid")) != identity[0]
            or str(payload.get("object_uid")) != str(row["object_uid"])
            or int(payload.get("support_seed", -1)) != identity[1]
            or payload.get("sidecar_contract_hash") != dataset.sidecar_contract_hash
            or payload.get("negative_context_policy")
            != "runtime_zeros_like_positive"
            or payload.get("vggt_camera_consumed") is not False
            or payload.get("known_K_T_replaced") is not False
            or not torch.is_tensor(view_ids)
            or view_ids.ndim != 1
            or view_ids.to(torch.int64).tolist() != list(row["view_ids"])
            or payload.get("decoded_source_rgba_sha256")
            != row["decoded_source_rgba_sha256"]
            or payload.get("processed_input_rgb_sha256")
            != row["processed_input_rgb_sha256"]
            or worker_record is None
            or worker_record.get("sidecar_file_sha256")
            != row["sidecar_file_sha256"]
        ):
            raise RuntimeError(f"sidecar semantic identity differs: {path}")
        context = validate_native_slat_vggt_context_tensor(
            payload.get("native_slat_vggt_cond"),
            views=len(row["view_ids"]),
            uid=identity[0],
        )
        if (
            list(context.shape) != list(row["native_context_shape"])
            or str(context.dtype) != str(row["native_context_dtype"])
        ):
            raise RuntimeError(f"sidecar tensor metadata differs: {path}")
        shapes["x".join(map(str, context.shape))] += 1
        dtypes[str(context.dtype)] += 1
        verified_bytes += len(raw)
        del raw, payload, context
        if (
            position == 1
            or position == limit
            or position % max(int(args.progress_every), 1) == 0
        ):
            elapsed = max(time.monotonic() - started, 1.0e-9)
            rate = verified_bytes / elapsed
            eta = (total_bytes - verified_bytes) / rate if rate > 0 else None
            print(
                json.dumps(
                    {
                        "phase": "new_sidecars_one_pass",
                        "objects": f"{position}/{limit}",
                        "bytes": f"{verified_bytes}/{total_bytes}",
                        "percent": 100.0 * verified_bytes / max(total_bytes, 1),
                        "MiB_per_second": rate / (1024**2),
                        "eta_seconds": eta,
                    }
                ),
                flush=True,
            )

    spot_indices = _spot_indices(count, int(args.base_lifting_spot_checks))
    for position, index in enumerate(spot_indices, start=1):
        row = dataset.sidecar_rows[index]
        base_index = dataset.base_indices[index]
        lifting_index = dataset.base.lifting_indices[base_index]
        lifting_row = dataset.base.lifting.rows[lifting_index]
        path = _resolve(dataset.base.lifting.root, str(lifting_row["cache_file"]))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        view_ids = payload.get("view_ids")
        if (
            str(payload.get("uid")) != str(row["uid"])
            or not torch.is_tensor(view_ids)
            or view_ids.to(torch.int64).tolist() != list(row["view_ids"])
        ):
            raise RuntimeError(f"base lifting/view-ID spot check differs: {path}")
        if position == len(spot_indices) or position % 16 == 0:
            print(
                json.dumps(
                    {
                        "phase": "base_lifting_view_id_spot_checks",
                        "objects": f"{position}/{len(spot_indices)}",
                    }
                ),
                flush=True,
            )

    complete = limit == count
    return {
        "passed": bool(complete),
        "p6_acceptance_complete": bool(complete),
        "format": "official_slat_with_vggt_perf_v1.light_cache_audit.v1",
        "object_count": count,
        "audited_sidecar_count": limit,
        "sidecar_bytes_verified_once": verified_bytes,
        "sidecar_sha256_all_verified": bool(complete),
        "sidecar_payload_schema_all_verified": bool(complete),
        "sidecar_tensor_finite_all_verified": bool(complete),
        "worker_report_count": len(shard_paths),
        "worker_coverage_exact": assigned == count,
        "pair_identity": dataset.pair_identity,
        "sidecar_contract_hash": dataset.sidecar_contract_hash,
        "base_manifest_identity_exact": True,
        "historical_base_artifacts_rehashed": False,
        "base_lifting_view_id_spot_check_count": len(spot_indices),
        "native_context_shapes": dict(sorted(shapes.items())),
        "native_context_dtypes": dict(sorted(dtypes.items())),
        "audit_policy": (
            "frozen historical base manifests; all newly materialized sidecars "
            "read exactly once for SHA256 plus payload validation"
        ),
    }


def main() -> None:
    report = run(make_parser().parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["p6_acceptance_complete"] else 2)


if __name__ == "__main__":
    main()
