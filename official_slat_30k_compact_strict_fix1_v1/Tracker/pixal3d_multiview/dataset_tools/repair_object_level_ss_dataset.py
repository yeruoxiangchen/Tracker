#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import trimesh


TRACKER_ROOT = Path(__file__).resolve().parents[2]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
for path in (TRACKER_ROOT, PIXAL3D_ROOT, RECONVIAGEN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-destructive object-level SS repair dataset. "
            "Observations are linked/copied from the source dataset; SS latents "
            "and manifests are written only under new roots."
        )
    )
    parser.add_argument("--source_dataset_root", required=True)
    parser.add_argument("--output_dataset_root", required=True)
    parser.add_argument("--source_experiment_root", required=True)
    parser.add_argument("--output_experiment_root", required=True)
    parser.add_argument(
        "--dataset_manifests",
        default="train.json,val.json",
        help="Comma-separated manifest names under source_dataset_root.",
    )
    parser.add_argument(
        "--split_names",
        default="train,val,holdout",
        help="Comma-separated split names under source_experiment_root/manifests.",
    )
    parser.add_argument(
        "--encoder_pretrained",
        default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
    )
    parser.add_argument(
        "--decoder_pretrained",
        default="Stable-X/trellis-vggt-v0-2",
    )
    parser.add_argument(
        "--target_mode",
        choices=("decoder_projected", "mesh_sampled"),
        default="decoder_projected",
        help=(
            "decoder_projected: save decoder(z)>0 as target_coords, while also "
            "saving deterministic mesh_target_coords. This matches the strict "
            "decoder round-trip audit. mesh_sampled: save deterministic mesh "
            "voxelization directly as target_coords."
        ),
    )
    parser.add_argument("--surface_points", type=int, default=160000)
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--canonical_margin", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--observation_mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to expose images/ and masks/ under the new dataset root.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def comma_items(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def scalar_text(value: np.ndarray | Any) -> str:
    arr = np.asarray(value)
    return str(arr.item()) if arr.shape == () else str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_object_seed(base_seed: int, object_uid: str) -> int:
    payload = f"{int(base_seed)}:{object_uid}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def load_meshes(path: str) -> list[trimesh.Trimesh]:
    obj = trimesh.load(path, force="scene", process=False)
    if isinstance(obj, trimesh.Scene):
        meshes = [
            mesh
            for mesh in obj.dump(concatenate=False)
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0
        ]
    elif isinstance(obj, trimesh.Trimesh):
        meshes = [obj]
    else:
        raise ValueError(f"unsupported object type: {type(obj)}")
    if not meshes:
        raise ValueError(f"scene has no mesh geometry: {path}")
    for mesh in meshes:
        if mesh.faces is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            raise ValueError(f"empty mesh in {path}")
        mesh.remove_unreferenced_vertices()
    return meshes


def deterministic_surface_points(
    mesh: trimesh.Trimesh,
    vertices_latent: np.ndarray,
    num_points: int,
    seed: int,
) -> np.ndarray:
    local_mesh = mesh.copy()
    local_mesh.vertices = np.asarray(vertices_latent, dtype=np.float32)

    try:
        points, _ = trimesh.sample.sample_surface(
            local_mesh,
            int(num_points),
            seed=int(seed),
        )
    except TypeError:
        state = np.random.get_state()
        np.random.seed(int(seed))
        try:
            points, _ = trimesh.sample.sample_surface(local_mesh, int(num_points))
        finally:
            np.random.set_state(state)
    return np.asarray(points, dtype=np.float32)


def coords_from_points(points: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.floor((np.asarray(points, dtype=np.float32) + 0.5) * int(resolution))
    coords = np.clip(coords.astype(np.int32), 0, int(resolution) - 1)
    return np.unique(coords, axis=0).astype(np.int32)


def load_encoder(pretrained: str, device: torch.device):
    import pixal3d.models as models

    encoder = models.from_pretrained(pretrained).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def load_decoder(pretrained: str, device: torch.device):
    from reconvggt_ar_adapter_a.audit_pointpose_ss_cache import load_decoder as _load_decoder

    return _load_decoder(pretrained, device)


@torch.no_grad()
def encode_sparse_latent(
    encoder,
    coords: np.ndarray,
    resolution: int,
    device: torch.device,
    save_dtype: str,
) -> np.ndarray:
    occupancy = torch.zeros(
        1,
        int(resolution),
        int(resolution),
        int(resolution),
        dtype=torch.float32,
    )
    coords_t = torch.from_numpy(np.asarray(coords, dtype=np.int64))
    occupancy[0, coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = 1.0
    z = encoder(occupancy[None].to(device), sample_posterior=False)
    result = z[0].detach().cpu().numpy()
    return result.astype(np.float16 if save_dtype == "float16" else np.float32)


@torch.no_grad()
def decode_threshold_coords(
    decoder,
    z: np.ndarray,
    device: torch.device,
    resolution: int,
) -> np.ndarray:
    if int(resolution) != 64:
        raise ValueError("current TRELLIS SS decoder audit assumes resolution=64")
    dtype = next(decoder.parameters()).dtype
    latent = torch.from_numpy(np.asarray(z))[None].to(device=device, dtype=dtype)
    logits = decoder(latent).float()[0, 0]
    coords = torch.nonzero(logits > 0, as_tuple=False)
    result = coords.detach().cpu().numpy().astype(np.int32)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] == 0:
        raise RuntimeError(f"decoder produced invalid threshold coords: {result.shape}")
    return result


def load_manifest(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"{path}: dict manifest has no samples list")
        return payload, samples
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path}: list manifest must contain objects")
        return payload, payload
    raise ValueError(f"{path}: unsupported manifest type {type(payload)}")


def object_uid_from_sample(sample: dict[str, Any]) -> str:
    uid = str(sample["uid"])
    return str(sample.get("object_uid") or uid.split("_seq", 1)[0])


def resolve_old_latent(
    sample: dict[str, Any],
    manifest_path: Path,
    source_dataset_root: Path,
) -> Path:
    raw = sample.get("ss_latent", sample.get("latent_path"))
    candidates: list[Path] = []
    if raw:
        path = Path(str(raw))
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend(
                [
                    manifest_path.parent / path,
                    source_dataset_root / path,
                    source_dataset_root / "ss_latents" / path,
                ]
            )
    uid = str(sample["uid"])
    candidates.append(source_dataset_root / "ss_latents" / uid[:2] / f"{uid}.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot resolve old latent for uid={uid}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def atomic_save_npz(path: Path, pack: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **pack)
    os.replace(tmp, path)


def expose_observations(source_root: Path, output_root: Path, mode: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("images", "masks"):
        source = (source_root / name).resolve()
        destination = output_root / name
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists() or destination.is_symlink():
            if mode == "symlink" and destination.is_symlink() and destination.resolve() == source:
                continue
            raise FileExistsError(
                f"{destination} already exists and is not the expected source link"
            )
        if mode == "symlink":
            destination.symlink_to(source, target_is_directory=True)
        else:
            shutil.copytree(source, destination, copy_function=shutil.copy2)


def recursive_replace_paths(
    value: Any,
    source_dataset_root: Path,
    output_dataset_root: Path,
    source_experiment_root: Path,
    output_experiment_root: Path,
) -> Any:
    if isinstance(value, dict):
        return {
            key: recursive_replace_paths(
                item,
                source_dataset_root,
                output_dataset_root,
                source_experiment_root,
                output_experiment_root,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            recursive_replace_paths(
                item,
                source_dataset_root,
                output_dataset_root,
                source_experiment_root,
                output_experiment_root,
            )
            for item in value
        ]
    if isinstance(value, str):
        replacements = (
            (str(source_dataset_root), str(output_dataset_root)),
            (str(source_experiment_root), str(output_experiment_root)),
        )
        for old, new in replacements:
            if value == old:
                return new
            if value.startswith(old + os.sep):
                return new + value[len(old) :]
    return value


def manifest_output_path(
    source_path: Path,
    source_dataset_root: Path,
    output_dataset_root: Path,
    source_experiment_root: Path,
    output_experiment_root: Path,
) -> Path:
    try:
        relative = source_path.resolve().relative_to(source_dataset_root.resolve())
        return output_dataset_root / relative
    except ValueError:
        relative = source_path.resolve().relative_to(source_experiment_root.resolve())
        return output_experiment_root / relative


def main() -> None:
    args = parse_args()
    source_dataset_root = Path(args.source_dataset_root).resolve()
    output_dataset_root = Path(args.output_dataset_root).resolve()
    source_experiment_root = Path(args.source_experiment_root).resolve()
    output_experiment_root = Path(args.output_experiment_root).resolve()

    if source_dataset_root == output_dataset_root:
        raise ValueError("source_dataset_root and output_dataset_root must differ")
    if source_experiment_root == output_experiment_root:
        raise ValueError("source_experiment_root and output_experiment_root must differ")
    if not source_dataset_root.exists():
        raise FileNotFoundError(source_dataset_root)
    if not source_experiment_root.exists():
        raise FileNotFoundError(source_experiment_root)

    output_latent_root = output_dataset_root / "ss_latents"
    report_path = output_experiment_root / "ss_repair_report.json"
    if not args.resume:
        if output_latent_root.exists() and any(output_latent_root.rglob("*.npz")):
            raise FileExistsError(
                f"{output_latent_root} already contains NPZ files; use a new root or --resume"
            )
        if report_path.exists():
            raise FileExistsError(f"{report_path} already exists; use a new root or --resume")

    expose_observations(
        source_dataset_root,
        output_dataset_root,
        args.observation_mode,
    )
    output_latent_root.mkdir(parents=True, exist_ok=True)
    (output_experiment_root / "manifests").mkdir(parents=True, exist_ok=True)

    manifest_paths: list[Path] = []
    for name in comma_items(args.dataset_manifests):
        path = source_dataset_root / name
        if path.exists():
            manifest_paths.append(path)
    for split in comma_items(args.split_names):
        path = source_experiment_root / "manifests" / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        manifest_paths.append(path)
    if not manifest_paths:
        raise RuntimeError("no source manifests found")

    docs: list[dict[str, Any]] = []
    uid_records: dict[str, dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        payload, samples = load_manifest(manifest_path)
        docs.append(
            {
                "source_path": manifest_path,
                "payload": payload,
                "samples": samples,
            }
        )
        for sample in samples:
            uid = str(sample["uid"])
            object_uid = object_uid_from_sample(sample)
            old_latent = resolve_old_latent(sample, manifest_path, source_dataset_root)
            existing = uid_records.get(uid)
            record = {
                "uid": uid,
                "object_uid": object_uid,
                "old_latent": old_latent,
            }
            if existing is not None:
                if existing["object_uid"] != object_uid or existing["old_latent"] != old_latent:
                    raise ValueError(f"inconsistent duplicated uid across manifests: {uid}")
            else:
                uid_records[uid] = record

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in uid_records.values():
        by_object[record["object_uid"]].append(record)
    for records in by_object.values():
        records.sort(key=lambda row: row["uid"])

    device = torch.device(args.device)
    encoder = load_encoder(args.encoder_pretrained, device)
    decoder = (
        load_decoder(args.decoder_pretrained, device)
        if args.target_mode == "decoder_projected"
        else None
    )

    object_rows: list[dict[str, Any]] = []
    voxel_count_by_uid: dict[str, int] = {}
    target_hash_by_uid: dict[str, str] = {}
    z_hash_by_uid: dict[str, str] = {}

    objects = sorted(by_object.items())
    for object_index, (object_uid, records) in enumerate(objects):
        output_paths = {
            record["uid"]: output_latent_root
            / record["uid"][:2]
            / f'{record["uid"]}.npz'
            for record in records
        }
        if args.resume and all(path.exists() for path in output_paths.values()):
            with np.load(next(iter(output_paths.values())), allow_pickle=False) as data:
                target_coords = np.asarray(data["target_coords"], dtype=np.int32)
                z = np.asarray(data["z"])
                mesh_target_coords = np.asarray(
                    data["mesh_target_coords"]
                    if "mesh_target_coords" in data
                    else data["target_coords"],
                    dtype=np.int32,
                )
                surface_seed = int(np.asarray(data["surface_seed"]).item())
            for uid, path in output_paths.items():
                with np.load(path, allow_pickle=False) as data:
                    coords_i = np.asarray(data["target_coords"], dtype=np.int32)
                    z_i = np.asarray(data["z"])
                if not np.array_equal(coords_i, target_coords) or not np.array_equal(z_i, z):
                    raise RuntimeError(f"resume consistency failure for object={object_uid}")
                voxel_count_by_uid[uid] = int(len(coords_i))
                target_hash_by_uid[uid] = hashlib.sha256(coords_i.tobytes()).hexdigest()
                z_hash_by_uid[uid] = hashlib.sha256(z_i.tobytes()).hexdigest()
            object_rows.append(
                {
                    "object_uid": object_uid,
                    "sequence_count": len(records),
                    "surface_seed": surface_seed,
                    "mesh_voxel_count": int(len(mesh_target_coords)),
                    "target_voxel_count": int(len(target_coords)),
                    "resumed": True,
                }
            )
            continue

        metadata_rows: list[dict[str, Any]] = []
        old_packs: dict[str, dict[str, Any]] = {}
        for record in records:
            with np.load(record["old_latent"], allow_pickle=False) as data:
                pack = {key: np.array(data[key], copy=True) for key in data.files}
            old_packs[record["uid"]] = pack
            required = (
                "normalize_center",
                "normalize_scale",
                "source_glb",
                "pixal3d_rotation",
            )
            missing = [key for key in required if key not in pack]
            if missing:
                raise KeyError(f'{record["old_latent"]} missing keys {missing}')
            metadata_rows.append(
                {
                    "center": np.asarray(pack["normalize_center"], dtype=np.float32),
                    "scale": float(np.asarray(pack["normalize_scale"]).item()),
                    "source_glb": scalar_text(pack["source_glb"]),
                    "rotation": np.asarray(pack["pixal3d_rotation"], dtype=np.float32),
                }
            )

        reference = metadata_rows[0]
        for row in metadata_rows[1:]:
            if not np.allclose(row["center"], reference["center"]):
                raise ValueError(f"normalize_center mismatch within object={object_uid}")
            if not np.isclose(row["scale"], reference["scale"]):
                raise ValueError(f"normalize_scale mismatch within object={object_uid}")
            if row["source_glb"] != reference["source_glb"]:
                raise ValueError(f"source_glb mismatch within object={object_uid}")
            if not np.allclose(row["rotation"], reference["rotation"]):
                raise ValueError(f"pixal3d_rotation mismatch within object={object_uid}")

        source_glb = Path(reference["source_glb"])
        if not source_glb.exists():
            raise FileNotFoundError(source_glb)
        meshes = load_meshes(str(source_glb))
        mesh = trimesh.util.concatenate(meshes)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        center = reference["center"]
        scale = float(reference["scale"])
        if not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError(f"invalid normalize_scale for object={object_uid}: {scale}")
        vertices_latent = (
            (vertices - center[None]) / scale * float(args.canonical_margin)
        ).astype(np.float32)

        surface_seed = stable_object_seed(args.seed, object_uid)
        points = deterministic_surface_points(
            mesh,
            vertices_latent,
            args.surface_points,
            surface_seed,
        )
        mesh_target_coords = coords_from_points(points, args.voxel_resolution)
        if mesh_target_coords.shape[0] == 0:
            raise RuntimeError(f"empty deterministic mesh target for object={object_uid}")

        z = encode_sparse_latent(
            encoder,
            mesh_target_coords,
            args.voxel_resolution,
            device,
            args.latent_dtype,
        )
        if args.target_mode == "decoder_projected":
            assert decoder is not None
            target_coords = decode_threshold_coords(
                decoder,
                z,
                device,
                args.voxel_resolution,
            )
        else:
            target_coords = mesh_target_coords

        target_hash = hashlib.sha256(target_coords.tobytes()).hexdigest()
        z_hash = hashlib.sha256(z.tobytes()).hexdigest()
        for record in records:
            uid = record["uid"]
            old_pack = old_packs[uid]
            new_pack = {
                key: value
                for key, value in old_pack.items()
                if key not in {
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
                    "z": z,
                    "target_coords": target_coords.astype(np.int32),
                    "mesh_target_coords": mesh_target_coords.astype(np.int32),
                    "surface_seed": np.array(surface_seed, dtype=np.uint32),
                    "surface_points": np.array(args.surface_points, dtype=np.int32),
                    "voxel_resolution": np.array(args.voxel_resolution, dtype=np.int32),
                    "canonical_margin": np.array(args.canonical_margin, dtype=np.float32),
                    "repair_format": np.array("object_level_ss_repair.v1"),
                    "repair_target_mode": np.array(args.target_mode),
                    "repair_encoder_pretrained": np.array(args.encoder_pretrained),
                    "repair_decoder_pretrained": np.array(
                        args.decoder_pretrained
                        if args.target_mode == "decoder_projected"
                        else ""
                    ),
                    "source_latent_sha256": np.array(
                        sha256_file(record["old_latent"])
                    ),
                }
            )
            atomic_save_npz(output_paths[uid], new_pack)
            voxel_count_by_uid[uid] = int(len(target_coords))
            target_hash_by_uid[uid] = target_hash
            z_hash_by_uid[uid] = z_hash

        object_rows.append(
            {
                "object_uid": object_uid,
                "sequence_count": len(records),
                "surface_seed": int(surface_seed),
                "mesh_voxel_count": int(len(mesh_target_coords)),
                "target_voxel_count": int(len(target_coords)),
                "target_sha256": target_hash,
                "z_sha256": z_hash,
                "resumed": False,
            }
        )
        if (object_index + 1) % max(1, int(args.log_every)) == 0 or object_index == 0:
            print(
                f"[repair] objects={object_index + 1}/{len(objects)} "
                f"object={object_uid} mesh={len(mesh_target_coords)} "
                f"target={len(target_coords)} seqs={len(records)}",
                flush=True,
            )

    written_manifests: list[str] = []
    for doc in docs:
        source_path: Path = doc["source_path"]
        output_path = manifest_output_path(
            source_path,
            source_dataset_root,
            output_dataset_root,
            source_experiment_root,
            output_experiment_root,
        )
        payload = recursive_replace_paths(
            copy.deepcopy(doc["payload"]),
            source_dataset_root,
            output_dataset_root,
            source_experiment_root,
            output_experiment_root,
        )
        samples = payload["samples"] if isinstance(payload, dict) else payload
        for sample in samples:
            uid = str(sample["uid"])
            if uid not in voxel_count_by_uid:
                raise KeyError(f"manifest uid missing repaired latent: {uid}")
            new_latent = output_latent_root / uid[:2] / f"{uid}.npz"
            sample["ss_latent"] = str(new_latent)
            if "latent_path" in sample:
                sample["latent_path"] = str(new_latent)
            sample["num_voxels"] = int(voxel_count_by_uid[uid])
            if isinstance(sample.get("shape_stats"), dict):
                sample["shape_stats"]["num_voxels"] = int(voxel_count_by_uid[uid])
            sample["ss_repair_target_sha256"] = target_hash_by_uid[uid]
            sample["ss_repair_z_sha256"] = z_hash_by_uid[uid]
            sample["ss_repair_target_mode"] = args.target_mode
        if isinstance(payload, dict):
            payload["latent_root"] = str(output_latent_root)
            payload["ss_repair"] = {
                "format": "object_level_ss_repair.v1",
                "source_manifest": str(source_path),
                "source_dataset_root": str(source_dataset_root),
                "output_dataset_root": str(output_dataset_root),
                "target_mode": args.target_mode,
                "surface_points": int(args.surface_points),
                "voxel_resolution": int(args.voxel_resolution),
                "canonical_margin": float(args.canonical_margin),
                "seed": int(args.seed),
                "encoder_pretrained": args.encoder_pretrained,
                "decoder_pretrained": (
                    args.decoder_pretrained
                    if args.target_mode == "decoder_projected"
                    else None
                ),
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written_manifests.append(str(output_path))

    consistency_failures: list[str] = []
    for object_uid, records in objects:
        reference_coords = None
        reference_z = None
        for record in records:
            uid = record["uid"]
            path = output_latent_root / uid[:2] / f"{uid}.npz"
            with np.load(path, allow_pickle=False) as data:
                coords = np.asarray(data["target_coords"], dtype=np.int32)
                z = np.asarray(data["z"])
                stored_uid = scalar_text(data["uid"]) if "uid" in data else uid
                stored_object_uid = (
                    scalar_text(data["object_uid"])
                    if "object_uid" in data
                    else object_uid
                )
            if stored_uid != uid:
                consistency_failures.append(f"{uid}: stored uid={stored_uid}")
            if stored_object_uid != object_uid:
                consistency_failures.append(
                    f"{uid}: stored object_uid={stored_object_uid}"
                )
            if reference_coords is None:
                reference_coords = coords
                reference_z = z
            elif not np.array_equal(coords, reference_coords):
                consistency_failures.append(
                    f"{object_uid}: target_coords differ across sequences"
                )
            elif not np.array_equal(z, reference_z):
                consistency_failures.append(f"{object_uid}: z differs across sequences")

    mesh_counts = np.asarray(
        [row["mesh_voxel_count"] for row in object_rows], dtype=np.float64
    )
    target_counts = np.asarray(
        [row["target_voxel_count"] for row in object_rows], dtype=np.float64
    )
    report = {
        "format": "object_level_ss_repair_report.v1",
        "passed": not consistency_failures,
        "source_dataset_root": str(source_dataset_root),
        "output_dataset_root": str(output_dataset_root),
        "source_experiment_root": str(source_experiment_root),
        "output_experiment_root": str(output_experiment_root),
        "target_mode": args.target_mode,
        "object_count": len(objects),
        "sample_count": len(uid_records),
        "manifest_count": len(written_manifests),
        "written_manifests": written_manifests,
        "observation_mode": args.observation_mode,
        "surface_points": int(args.surface_points),
        "voxel_resolution": int(args.voxel_resolution),
        "canonical_margin": float(args.canonical_margin),
        "seed": int(args.seed),
        "encoder_pretrained": args.encoder_pretrained,
        "decoder_pretrained": (
            args.decoder_pretrained
            if args.target_mode == "decoder_projected"
            else None
        ),
        "mesh_voxel_count": {
            "min": int(mesh_counts.min()),
            "median": float(np.median(mesh_counts)),
            "mean": float(mesh_counts.mean()),
            "max": int(mesh_counts.max()),
        },
        "target_voxel_count": {
            "min": int(target_counts.min()),
            "median": float(np.median(target_counts)),
            "mean": float(target_counts.mean()),
            "max": int(target_counts.max()),
        },
        "same_object_consistency_failures": consistency_failures,
        "objects": object_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "passed",
        "object_count",
        "sample_count",
        "manifest_count",
        "target_mode",
        "mesh_voxel_count",
        "target_voxel_count",
    )}, indent=2), flush=True)
    print(f"[repair] wrote report: {report_path}", flush=True)
    if consistency_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()