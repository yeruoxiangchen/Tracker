#!/usr/bin/env python3
"""Materialize real Native-SS targets from official ProObjaverse SLat support.

The existing official SLat/DINO cache intentionally contains an all-zero SS
placeholder because it was built for GT-support SLat experiments.  This tool
keeps the large DINO/pose/mask lifting tensors immutable, encodes the official
64^3 SLat coordinates with the frozen TRELLIS SS VAE, audits the decoder
round-trip, and emits a new lifting manifest whose rows bind the real targets.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION
from pose_aligned_reconstruction.direct_slat_flow import DIRECT_SLAT_CACHE_VERSION
from pose_aligned_reconstruction.native_ss_genrecon import canonical_json_sha256, sha256_file
from pose_aligned_reconstruction.proobjaverse_official_slat_compact import (
    is_compact_manifest_pair,
    validate_compact_manifest_pair_payloads,
)
from pose_aligned_reconstruction.proobjaverse_official_ss_compact import (
    build_official_ss_compact_manifest,
)
from pose_aligned_reconstruction.proobjaverse_official_ss import (
    OFFICIAL_SS_CACHE_FORMAT,
    OFFICIAL_SS_TARGET_FORMAT,
    official_domain_contract,
)


DEFAULT_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
DEFAULT_DECODER = "Stable-X/trellis-vggt-v0-2"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def sparse_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_set = {tuple(map(int, row[-3:])) for row in np.asarray(left)}
    right_set = {tuple(map(int, row[-3:])) for row in np.asarray(right)}
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def load_manifest(path: Path, expected_format: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != expected_format:
        raise ValueError(f"unexpected manifest format: {path}")
    rows = value.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest has no samples: {path}")
    return value


def load_json_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    rows = value.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest has no samples: {path}")
    return value


def resolve_manifest_row_file(
    manifest_path: Path, manifest: dict[str, Any], value: str | Path
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path(manifest.get("output_dir", manifest_path.parent)).expanduser().resolve()
    return (root / path).resolve()


def unique_coords(value: np.ndarray, *, uid: str) -> np.ndarray:
    coords = np.asarray(value, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError(f"uid={uid} invalid official coords shape={coords.shape}")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"uid={uid} official coords leave [0,63]")
    return np.unique(coords, axis=0).astype(np.int32)


def join_source_rows(
    slat: dict[str, Any], lifting: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    lifting_by_uid = {str(row.get("uid", "")): row for row in lifting["samples"]}
    if "" in lifting_by_uid or len(lifting_by_uid) != len(lifting["samples"]):
        raise ValueError("lifting manifest contains empty/duplicate UIDs")
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for slat_row in slat["samples"]:
        uid = str(slat_row.get("uid", ""))
        if not uid or uid in seen or uid not in lifting_by_uid:
            raise ValueError(f"cannot join official cache uid={uid!r}")
        seen.add(uid)
        lifting_row = lifting_by_uid[uid]
        if str(lifting_row.get("object_uid", uid)) != str(
            slat_row.get("object_uid", uid)
        ):
            raise ValueError(f"uid={uid} object identity differs")
        joined.append((slat_row, lifting_row))
    if len(joined) != len(lifting_by_uid):
        raise ValueError("SLat/lifting manifest UID sets differ")
    return joined


def build_rebound_lifting_manifest(
    *,
    source: dict[str, Any],
    source_manifest: Path,
    source_slat_manifest: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    domain_contract: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    config = dict(source.get("config", {}))
    config["official_ss_targets"] = {
        "format": OFFICIAL_SS_CACHE_FORMAT,
        "split": str(split),
        "object_count": len(rows),
        "source_lifting_manifest": str(source_manifest.resolve()),
        "source_lifting_manifest_sha256": sha256_file(source_manifest),
        "source_slat_manifest": str(source_slat_manifest.resolve()),
        "source_slat_manifest_sha256": sha256_file(source_slat_manifest),
        "domain_contract": domain_contract,
        "placeholder_targets_consumed": False,
        "row_level_target_override": True,
    }
    result = dict(source)
    result.update(
        {
            "output_dir": str(output_dir.resolve()),
            "samples": rows,
            "sample_count": len(rows),
            "object_count": len(rows),
            "config": config,
            "config_hash": canonical_json_sha256(config),
            "official_gt_support_only": False,
            "source_lifting_manifest": str(source_manifest.resolve()),
            "source_lifting_manifest_sha256": sha256_file(source_manifest),
            "source_slat_manifest": str(source_slat_manifest.resolve()),
            "source_slat_manifest_sha256": sha256_file(source_slat_manifest),
        }
    )
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slat_manifest", required=True)
    parser.add_argument("--lifting_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ss_encoder_pretrained", default=DEFAULT_ENCODER)
    parser.add_argument("--ss_decoder_pretrained", default=DEFAULT_DECODER)
    parser.add_argument(
        "--ss_encoder_identity",
        default="",
        help=(
            "optional frozen training-time encoder identity; permits only path "
            "relocation while ss_encoder_pretrained names the local byte-identical asset"
        ),
    )
    parser.add_argument(
        "--ss_decoder_identity",
        default="",
        help="optional frozen training-time decoder identity for path relocation",
    )
    parser.add_argument("--latent_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--min_roundtrip_iou", type=float, default=0.90)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if not 0.0 <= float(args.min_roundtrip_iou) <= 1.0:
        raise ValueError("min_roundtrip_iou must be in [0,1]")
    if int(args.max_objects) < 0 or int(args.log_every) <= 0:
        raise ValueError("max_objects/log_every are invalid")
    slat_path = Path(args.slat_manifest).expanduser().resolve()
    lifting_path = Path(args.lifting_manifest).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    compact_mode = is_compact_manifest_pair(slat_path, lifting_path)
    if compact_mode:
        slat = load_json_manifest(slat_path)
        lifting = load_json_manifest(lifting_path)
        pair = validate_compact_manifest_pair_payloads(slat, lifting)
        lifting_by_uid = pair["lifting_by_uid"]
        joined = [
            (slat_row, lifting_by_uid[str(slat_row["uid"])])
            for slat_row in slat["samples"]
        ]
    else:
        slat = load_manifest(slat_path, DIRECT_SLAT_CACHE_VERSION)
        lifting = load_manifest(lifting_path, LIFTING_CACHE_VERSION)
        joined = join_source_rows(slat, lifting)
    if int(args.max_objects) > 0:
        joined = joined[: int(args.max_objects)]
    target_source = dict(slat.get("config", {}).get("target_source", {}))
    protocol_sha256 = str(target_source.get("protocol_sha256", ""))
    split = str(target_source.get("split", ""))
    if not protocol_sha256 or split not in {
        "train",
        "dev",
        "val",
        "holdout",
        "predicted_support_bridge",
    }:
        raise RuntimeError("official SLat source lacks frozen protocol/split")
    domain = official_domain_contract(
        protocol_sha256=protocol_sha256,
        encoder_pretrained=str(
            args.ss_encoder_identity or args.ss_encoder_pretrained
        ),
        decoder_pretrained=str(
            args.ss_decoder_identity or args.ss_decoder_pretrained
        ),
        latent_dtype=str(args.latent_dtype),
        minimum_roundtrip_iou=float(args.min_roundtrip_iou),
    )
    run_config = {
        "format": OFFICIAL_SS_CACHE_FORMAT,
        "slat_manifest": str(slat_path),
        "slat_manifest_sha256": sha256_file(slat_path),
        "lifting_manifest": str(lifting_path),
        "lifting_manifest_sha256": sha256_file(lifting_path),
        "selected_object_uids": [str(row[0]["uid"]) for row in joined],
        "domain_contract": domain,
        "split": split,
        "storage_mode": "compact_v2" if compact_mode else "legacy",
        "model_asset_runtime_relocation": {
            "ss_encoder_identity": str(
                args.ss_encoder_identity or args.ss_encoder_pretrained
            ),
            "ss_encoder_runtime": str(args.ss_encoder_pretrained),
            "ss_decoder_identity": str(
                args.ss_decoder_identity or args.ss_decoder_pretrained
            ),
            "ss_decoder_runtime": str(args.ss_decoder_pretrained),
            "numerical_implementation_changed": False,
        },
    }
    run_config["run_config_sha256"] = canonical_json_sha256(run_config)
    run_config_path = output / "run_config.json"
    if run_config_path.is_file():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise RuntimeError("official SS target resume binding changed")
        if not args.resume:
            raise FileExistsError("output exists; pass --resume")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("unbound non-empty official SS output directory")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(run_config_path, run_config)

    from pixal3d_multiview.dataset_tools.repair_object_level_ss_dataset import (
        decode_threshold_coords,
        encode_sparse_latent,
        load_decoder,
        load_encoder,
    )

    device = torch.device(str(args.device))
    encoder = load_encoder(str(args.ss_encoder_pretrained), device)
    decoder = load_decoder(str(args.ss_decoder_pretrained), device)
    output_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for position, (slat_row, lifting_row) in enumerate(joined, start=1):
        uid = str(slat_row["uid"])
        source_target = resolve_manifest_row_file(
            slat_path, slat, slat_row["target_file"]
        )
        expected_source_sha = str(slat_row.get("target_file_sha256", ""))
        if not source_target.is_file() or sha256_file(source_target) != expected_source_sha:
            raise RuntimeError(f"uid={uid} official lh-slat changed")
        target_path = output / "ss_latents" / uid[:2] / f"{uid}.npz"
        if target_path.is_file():
            if not args.resume:
                raise FileExistsError(f"target exists without --resume: {target_path}")
            with np.load(target_path, allow_pickle=False) as cached:
                if (
                    str(np.asarray(cached["uid"]).item()) != uid
                    or str(np.asarray(cached["format"]).item())
                    != OFFICIAL_SS_TARGET_FORMAT
                    or str(np.asarray(cached["domain_contract_sha256"]).item())
                    != str(domain["domain_contract_sha256"])
                    or str(np.asarray(cached["source_lh_slat_sha256"]).item())
                    != expected_source_sha
                ):
                    raise RuntimeError(f"uid={uid} stale official SS target")
                roundtrip_iou = float(np.asarray(cached["roundtrip_iou"]).item())
                official_count = int(len(cached["official_coords"]))
                decoded_count = int(len(cached["target_coords"]))
        else:
            with np.load(source_target, allow_pickle=False) as source:
                if "coords" not in source.files:
                    raise RuntimeError(f"uid={uid} official lh-slat lacks coords")
                official_coords = unique_coords(source["coords"], uid=uid)
            z = encode_sparse_latent(
                encoder,
                official_coords,
                64,
                device,
                str(args.latent_dtype),
            )
            decoded_coords = unique_coords(
                decode_threshold_coords(decoder, z, device, 64), uid=uid
            )
            roundtrip_iou = sparse_iou(official_coords, decoded_coords)
            if roundtrip_iou < float(args.min_roundtrip_iou):
                raise RuntimeError(
                    f"uid={uid} official SS roundtrip IoU {roundtrip_iou:.6f} "
                    f"< {float(args.min_roundtrip_iou):.6f}"
                )
            atomic_npz(
                target_path,
                z=z,
                target_coords=decoded_coords.astype(np.int32),
                official_coords=official_coords.astype(np.int32),
                uid=np.asarray(uid),
                format=np.asarray(OFFICIAL_SS_TARGET_FORMAT),
                repair_format=np.asarray(OFFICIAL_SS_TARGET_FORMAT),
                repair_target_mode=np.asarray("decoder_projected"),
                coordinate_frame=np.asarray("official_lh_slat_64"),
                domain_contract_sha256=np.asarray(domain["domain_contract_sha256"]),
                source_lh_slat=np.asarray(str(source_target)),
                source_lh_slat_sha256=np.asarray(expected_source_sha),
                roundtrip_iou=np.asarray(roundtrip_iou, dtype=np.float64),
            )
            official_count = len(official_coords)
            decoded_count = len(decoded_coords)
        target_sha = sha256_file(target_path)
        output_rows.append(
            {
                **lifting_row,
                "ss_latent": str(target_path),
                "ss_latent_sha256": target_sha,
                "official_lh_slat": str(source_target),
                "official_lh_slat_sha256": expected_source_sha,
                "official_ss_roundtrip_iou": roundtrip_iou,
            }
        )
        records.append(
            {
                "uid": uid,
                "ss_latent": str(target_path),
                "ss_latent_sha256": target_sha,
                "official_coord_count": official_count,
                "decoded_coord_count": decoded_count,
                "roundtrip_iou": roundtrip_iou,
            }
        )
        if position == 1 or position % int(args.log_every) == 0 or position == len(joined):
            print(
                f"[official_ss_targets] {position}/{len(joined)} uid={uid} "
                f"official={official_count} decoded={decoded_count} "
                f"iou={roundtrip_iou:.6f}",
                flush=True,
            )
    del encoder, decoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if compact_mode:
        official_binding = {
            "format": OFFICIAL_SS_CACHE_FORMAT,
            "split": str(split),
            "object_count": len(output_rows),
            "source_lifting_manifest": str(lifting_path.resolve()),
            "source_lifting_manifest_sha256": sha256_file(lifting_path),
            "source_slat_manifest": str(slat_path.resolve()),
            "source_slat_manifest_sha256": sha256_file(slat_path),
            "domain_contract": domain,
            "placeholder_targets_consumed": False,
            "row_level_target_override": True,
            "storage_mode": "compact_v2",
        }
        rebound = build_official_ss_compact_manifest(
            source_slat=slat,
            source_lifting=lifting,
            source_slat_manifest=slat_path,
            source_lifting_manifest=lifting_path,
            output_dir=output,
            rows=output_rows,
            official_ss_targets=official_binding,
        )
    else:
        rebound = build_rebound_lifting_manifest(
            source=lifting,
            source_manifest=lifting_path,
            source_slat_manifest=slat_path,
            output_dir=output,
            rows=output_rows,
            domain_contract=domain,
            split=split,
        )
    manifest_path = output / "lifting_manifest.json"
    atomic_json(manifest_path, rebound)
    ious = [float(row["roundtrip_iou"]) for row in records]
    report = {
        "format": OFFICIAL_SS_CACHE_FORMAT,
        "passed": len(records) == len(joined) and min(ious) >= float(args.min_roundtrip_iou),
        "split": split,
        "object_count": len(records),
        "lifting_manifest": str(manifest_path),
        "lifting_manifest_sha256": sha256_file(manifest_path),
        "domain_contract": domain,
        "roundtrip_iou": {
            "minimum": min(ious),
            "mean": float(np.mean(ious)),
            "median": float(np.median(ious)),
            "maximum": max(ious),
        },
        "records": records,
        "placeholder_targets_consumed": False,
        "scope_guard": (
            "Native-SS target materialization only; no model was trained or selected"
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    atomic_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "split": split,
                "objects": len(records),
                "roundtrip_iou": report["roundtrip_iou"],
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
