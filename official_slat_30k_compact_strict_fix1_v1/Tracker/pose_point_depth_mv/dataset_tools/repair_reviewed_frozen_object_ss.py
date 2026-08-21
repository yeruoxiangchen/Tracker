#!/usr/bin/env python3
"""Rebind a reviewed frozen dataset to deterministic object-level SS targets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
import trimesh

from pixal3d_multiview.dataset_tools.repair_object_level_ss_dataset import (
    decode_threshold_coords,
    deterministic_surface_points,
    encode_sparse_latent,
    load_decoder,
    load_encoder,
    load_meshes,
    stable_object_seed,
)


FORMAT = "pose_point_depth_mv.reviewed_object_ss_repair.v1"
REPAIR_FORMAT = "object_level_ss_repair.v1"
SPLITS = ("train", "val", "test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.ndim == 0 else str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.array(payload[key], copy=True) for key in payload.files}


def validate_object_metadata(
    object_uid: str, packs: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, float, str, np.ndarray]:
    required = (
        "normalize_center",
        "normalize_scale",
        "source_glb",
        "pixal3d_rotation",
    )
    for pack in packs:
        missing = [key for key in required if key not in pack]
        if missing:
            raise KeyError(f"object={object_uid} latent missing fields {missing}")
    center = np.asarray(packs[0]["normalize_center"], dtype=np.float32)
    scale = float(np.asarray(packs[0]["normalize_scale"]).item())
    source_glb = scalar_text(packs[0]["source_glb"])
    rotation = np.asarray(packs[0]["pixal3d_rotation"], dtype=np.float32)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError(f"object={object_uid} has invalid normalize_center")
    if not np.isfinite(scale) or scale <= 1.0e-8:
        raise ValueError(f"object={object_uid} has invalid normalize_scale={scale}")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"object={object_uid} has invalid pixal3d_rotation")
    for pack in packs[1:]:
        if not np.array_equal(
            np.asarray(pack["normalize_center"], dtype=np.float32), center
        ):
            raise ValueError(f"object={object_uid} normalize_center differs")
        if float(np.asarray(pack["normalize_scale"]).item()) != scale:
            raise ValueError(f"object={object_uid} normalize_scale differs")
        if scalar_text(pack["source_glb"]) != source_glb:
            raise ValueError(f"object={object_uid} source_glb differs")
        if not np.array_equal(
            np.asarray(pack["pixal3d_rotation"], dtype=np.float32), rotation
        ):
            raise ValueError(f"object={object_uid} pixal3d_rotation differs")
    return center, scale, source_glb, rotation


def is_reusable_object_repair(
    packs: list[dict[str, np.ndarray]], *, target_mode: str
) -> bool:
    if not packs:
        return False
    required = (
        "z",
        "target_coords",
        "mesh_target_coords",
        "repair_format",
        "repair_target_mode",
        "surface_seed",
    )
    if any(any(key not in pack for key in required) for pack in packs):
        return False
    if any(scalar_text(pack["repair_format"]) != REPAIR_FORMAT for pack in packs):
        return False
    if any(
        scalar_text(pack["repair_target_mode"]) != target_mode for pack in packs
    ):
        return False
    reference_coords = np.asarray(packs[0]["target_coords"], dtype=np.int32)
    reference_z = np.asarray(packs[0]["z"])
    return all(
        np.array_equal(np.asarray(pack["target_coords"], dtype=np.int32), reference_coords)
        and np.array_equal(np.asarray(pack["z"]), reference_z)
        for pack in packs[1:]
    )


def output_latent_path(output_root: Path, uid: str) -> Path:
    return output_root / "ss_latents" / uid[:2] / f"{uid}.npz"


def load_source(source_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    documents: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_objects: dict[str, str] = {}
    for split in SPLITS:
        path = source_root / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("samples"), list):
            raise ValueError(f"{path} has no samples list")
        documents[split] = payload
        for source_row in payload["samples"]:
            row = copy.deepcopy(source_row)
            uid = str(row["uid"])
            object_uid = str(row.get("object_uid", uid))
            if uid in seen_uids:
                raise ValueError(f"duplicate sample uid={uid}")
            previous_split = seen_objects.setdefault(object_uid, split)
            if previous_split != split:
                raise ValueError(f"object={object_uid} crosses splits")
            latent_path = Path(str(row["ss_latent"])).resolve()
            if not latent_path.is_file():
                raise FileNotFoundError(latent_path)
            row["_split"] = split
            row["_source_latent"] = str(latent_path)
            rows.append(row)
            seen_uids.add(uid)
    return documents, rows


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--encoder_pretrained",
        default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
    )
    parser.add_argument(
        "--decoder_pretrained", default="Stable-X/trellis-vggt-v0-2"
    )
    parser.add_argument(
        "--target_mode", choices=("decoder_projected",), default="decoder_projected"
    )
    parser.add_argument("--surface_points", type=int, default=160000)
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--canonical_margin", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    return parser


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    if source_root == output_root:
        raise ValueError("source_root and output_root must differ")
    source_report = source_root / "report.json"
    if not source_report.is_file():
        raise FileNotFoundError(source_report)
    documents, rows = load_source(source_root)
    source_bindings = {
        name: {
            "path": str(source_root / f"{name}.json"),
            "sha256": sha256_file(source_root / f"{name}.json"),
        }
        for name in SPLITS
    }
    config = {
        "format": FORMAT,
        "source_root": str(source_root),
        "source_report_sha256": sha256_file(source_report),
        "source_manifests": source_bindings,
        "encoder_pretrained": args.encoder_pretrained,
        "decoder_pretrained": args.decoder_pretrained,
        "target_mode": args.target_mode,
        "surface_points": int(args.surface_points),
        "voxel_resolution": int(args.voxel_resolution),
        "canonical_margin": float(args.canonical_margin),
        "seed": int(args.seed),
        "latent_dtype": args.latent_dtype,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_config = {"config": config, "config_hash": config_hash}
    run_config_path = output_root / "run_config.json"
    report_path = output_root / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed") is True and report.get("config_hash") == config_hash:
            print(json.dumps({"reused": True, **report["summary"]}, indent=2))
            return
        raise FileExistsError(f"incompatible repair report exists: {report_path}")
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"partial repair output exists; preserve and pass --resume: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if run_config_path.is_file():
        if json.loads(run_config_path.read_text(encoding="utf-8")) != run_config:
            raise RuntimeError("repair resume config/source binding changed")
    else:
        atomic_json(run_config_path, run_config)

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row.get("object_uid", row["uid"]))].append(row)
    for object_rows in by_object.values():
        object_rows.sort(key=lambda row: str(row["uid"]))

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("object-level SS repair requires CUDA")
    encoder = load_encoder(args.encoder_pretrained, device)
    decoder = load_decoder(args.decoder_pretrained, device)
    counters: Counter[str] = Counter()
    identities: dict[str, dict[str, Any]] = {}
    for ordinal, (object_uid, object_rows) in enumerate(
        sorted(by_object.items()), start=1
    ):
        source_paths = [Path(row["_source_latent"]) for row in object_rows]
        packs = [load_npz(path) for path in source_paths]
        center, scale, source_glb, _ = validate_object_metadata(object_uid, packs)
        if is_reusable_object_repair(packs, target_mode=args.target_mode):
            target_coords = np.asarray(packs[0]["target_coords"], dtype=np.int32)
            mesh_target_coords = np.asarray(
                packs[0]["mesh_target_coords"], dtype=np.int32
            )
            z = np.asarray(packs[0]["z"])
            surface_seed = int(np.asarray(packs[0]["surface_seed"]).item())
            counters["reused_repaired_objects"] += 1
        else:
            source_path = Path(source_glb)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            mesh = trimesh.util.concatenate(load_meshes(str(source_path)))
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            vertices_latent = (
                (vertices - center[None]) / scale * float(args.canonical_margin)
            ).astype(np.float32)
            surface_seed = stable_object_seed(args.seed, object_uid)
            points = deterministic_surface_points(
                mesh, vertices_latent, args.surface_points, surface_seed
            )
            mesh_target_coords = np.unique(
                np.clip(
                    np.floor(
                        (points + 0.5) * int(args.voxel_resolution)
                    ).astype(np.int32),
                    0,
                    int(args.voxel_resolution) - 1,
                ),
                axis=0,
            ).astype(np.int32)
            z = encode_sparse_latent(
                encoder,
                mesh_target_coords,
                args.voxel_resolution,
                device,
                args.latent_dtype,
            )
            target_coords = decode_threshold_coords(
                decoder, z, device, args.voxel_resolution
            )
            counters["rebuilt_objects"] += 1
        target_hash = sha256_array(target_coords)
        z_hash = sha256_array(z)
        for row, source_path, old_pack in zip(object_rows, source_paths, packs):
            uid = str(row["uid"])
            new_pack = {
                key: value
                for key, value in old_pack.items()
                if key
                not in {
                    "z",
                    "target_coords",
                    "mesh_target_coords",
                    "surface_seed",
                    "surface_points",
                    "voxel_resolution",
                    "canonical_margin",
                    "repair_format",
                    "repair_target_mode",
                    "repair_encoder_pretrained",
                    "repair_decoder_pretrained",
                    "source_latent_sha256",
                }
            }
            new_pack.update(
                {
                    "z": np.asarray(z),
                    "target_coords": target_coords.astype(np.int32),
                    "mesh_target_coords": mesh_target_coords.astype(np.int32),
                    "surface_seed": np.array(surface_seed, dtype=np.uint32),
                    "surface_points": np.array(args.surface_points, dtype=np.int32),
                    "voxel_resolution": np.array(
                        args.voxel_resolution, dtype=np.int32
                    ),
                    "canonical_margin": np.array(
                        args.canonical_margin, dtype=np.float32
                    ),
                    "repair_format": np.array(REPAIR_FORMAT),
                    "repair_target_mode": np.array(args.target_mode),
                    "repair_encoder_pretrained": np.array(args.encoder_pretrained),
                    "repair_decoder_pretrained": np.array(args.decoder_pretrained),
                    "source_latent_sha256": np.array(sha256_file(source_path)),
                }
            )
            output_path = output_latent_path(output_root, uid)
            if not output_path.is_file():
                atomic_savez(output_path, new_pack)
            counters["samples"] += 1
            identities[uid] = {
                "path": str(output_path),
                "target_hash": target_hash,
                "z_hash": z_hash,
                "target_count": int(len(target_coords)),
            }
        if ordinal == 1 or ordinal % max(1, int(args.log_every)) == 0:
            print(
                f"[reviewed_ss_repair] {ordinal}/{len(by_object)} "
                f"{object_uid} target={len(target_coords)}",
                flush=True,
            )

    def repaired_document(payload: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(payload)
        for row in result["samples"]:
            identity = identities[str(row["uid"])]
            row["ss_latent"] = identity["path"]
            row["num_voxels"] = identity["target_count"]
            if isinstance(row.get("shape_stats"), dict):
                row["shape_stats"]["num_voxels"] = identity["target_count"]
            row["ss_repair_target_sha256"] = identity["target_hash"]
            row["ss_repair_z_sha256"] = identity["z_hash"]
            row["ss_repair_target_mode"] = args.target_mode
        result["latent_root"] = "/"
        result["ss_repair"] = {
            "format": FORMAT,
            "repair_format": REPAIR_FORMAT,
            "config_hash": config_hash,
            "target_mode": args.target_mode,
            "source_root": str(source_root),
        }
        return result

    for split, payload in documents.items():
        atomic_json(output_root / f"{split}.json", repaired_document(payload))
    source_all = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    atomic_json(output_root / "manifest.json", repaired_document(source_all))

    failures: list[str] = []
    for object_uid, object_rows in sorted(by_object.items()):
        reference_coords = None
        reference_z = None
        for row in object_rows:
            identity = identities[str(row["uid"])]
            pack = load_npz(Path(identity["path"]))
            coords = np.asarray(pack["target_coords"], dtype=np.int32)
            z = np.asarray(pack["z"])
            if scalar_text(pack["repair_format"]) != REPAIR_FORMAT:
                failures.append(f"{row['uid']}: repair_format")
            if sha256_array(coords) != identity["target_hash"]:
                failures.append(f"{row['uid']}: target_hash")
            if sha256_array(z) != identity["z_hash"]:
                failures.append(f"{row['uid']}: z_hash")
            if reference_coords is None:
                reference_coords, reference_z = coords, z
            elif not np.array_equal(coords, reference_coords):
                failures.append(f"{object_uid}: target_coords differ")
            elif not np.array_equal(z, reference_z):
                failures.append(f"{object_uid}: z differs")
    summary = {
        "object_count": len(by_object),
        "sample_count": len(rows),
        "reused_repaired_objects": counters["reused_repaired_objects"],
        "rebuilt_objects": counters["rebuilt_objects"],
        "failure_count": len(failures),
    }
    report = {
        "format": FORMAT,
        "passed": not failures,
        "config_hash": config_hash,
        "summary": summary,
        "failures": failures[:100],
        "output_manifests": {
            name: {
                "path": str(output_root / f"{name}.json"),
                "sha256": sha256_file(output_root / f"{name}.json"),
            }
            for name in ("manifest", *SPLITS)
        },
    }
    atomic_json(report_path, report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"object-level SS repair audit failed: {failures[:8]}")


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
