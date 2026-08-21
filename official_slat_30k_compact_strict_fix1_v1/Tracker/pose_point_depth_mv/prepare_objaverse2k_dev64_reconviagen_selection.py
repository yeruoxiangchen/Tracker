#!/usr/bin/env python3
"""Freeze the existing Objaverse2K dev64 inputs for ReconViaGen evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat import (
    WORKER_REPORT_FORMAT,
)
from pose_point_depth_mv.export_native_slat_mesh_pairs import select_matrix
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.training_overlap_objaverse import (
    OBJAVERSE2K_DEV64_PROTOCOL_FORMAT,
    OBJAVERSE2K_DEV64_SCOPE,
    validate_selection,
)


SELECTION_FORMAT = "pose_point_depth_mv.objaverse2k_dev64_selection.v1"
LIFTING_FORMAT = "ar_ss_flow.pose_lifting_cache.v1"
SLAT_FORMAT = "pose_point_depth_mv.direct_slat_cache.v1"
JOINT_SEEDS = (42, 43, 44)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_int_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def selected_dev_rows(slat: dict[str, Any]) -> list[tuple[str, str, dict[int, int]]]:
    rows = list(slat.get("samples", []))
    selected = select_matrix(
        rows,
        joint_seeds=list(JOINT_SEEDS),
        max_objects=64,
        object_offset=0,
        require_complete=True,
    )
    if len({object_uid for object_uid, _, _ in selected}) != 64:
        raise RuntimeError("dev64 selection does not contain 64 unique objects")
    return selected


def validate_native_reports(
    paths: list[Path], *, slat_sha: str
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    if len(paths) != 4:
        raise ValueError("exactly four Objaverse2K worker reports are required")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    checkpoint_binding: dict[str, Any] | None = None
    seen_workers: set[int] = set()
    report_bindings = []
    for path in paths:
        report = load_json(path)
        if (
            report.get("format") != WORKER_REPORT_FORMAT
            or report.get("passed") is not True
            or report.get("formal") is not False
            or report.get("model_label") != "objaverse2k"
            or int(report.get("num_workers", -1)) != 4
        ):
            raise RuntimeError(f"invalid Objaverse2K dev64 worker report: {path}")
        worker = int(report["worker_index"])
        if worker in seen_workers:
            raise RuntimeError(f"duplicate Objaverse2K worker={worker}")
        seen_workers.add(worker)
        run = dict(report["run_config"])
        if run.get("cache_manifest_sha256") != slat_sha:
            raise RuntimeError("Objaverse2K worker dev-cache binding differs")
        current_checkpoint = {
            "path": str(Path(run["checkpoint"]).resolve()),
            "sha256": str(run["checkpoint_sha256"]),
            "step": int(run["checkpoint_step"]),
            "weights": str(run["weights"]),
        }
        if current_checkpoint["step"] != 2000 or current_checkpoint["weights"] != "ema":
            raise RuntimeError("Objaverse2K dev64 is not step2000 EMA")
        if checkpoint_binding is None:
            checkpoint_binding = current_checkpoint
        elif checkpoint_binding != current_checkpoint:
            raise RuntimeError("Objaverse2K worker checkpoints differ")
        for record in report.get("records", []):
            identity = record["identity"]
            key = (
                str(identity["object_uid"]),
                str(identity["uid"]),
                int(identity["support_seed"]),
            )
            if key in records:
                raise RuntimeError(f"duplicate Native dev64 record={key}")
            branch = record["branches"]["full"]
            mesh_path = Path(branch["mesh"]).resolve()
            if sha256_file(mesh_path) != branch["mesh_sha256"]:
                raise RuntimeError(f"Native dev64 mesh changed: {mesh_path}")
            records[key] = record
        report_bindings.append({"path": str(path), "sha256": sha256_file(path)})
    if seen_workers != set(range(4)) or len(records) != 64 * len(JOINT_SEEDS):
        raise RuntimeError("Objaverse2K dev64 Native matrix is incomplete")
    if checkpoint_binding is None:
        raise RuntimeError("Objaverse2K checkpoint binding is absent")
    return records, {"checkpoint": checkpoint_binding, "worker_reports": report_bindings}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifting_manifest", required=True)
    parser.add_argument("--slat_manifest", required=True)
    parser.add_argument("--native_report", action="append", required=True)
    parser.add_argument("--output_selection", required=True)
    parser.add_argument("--output_lifting_subset", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    lifting_path = Path(args.lifting_manifest).expanduser().resolve()
    slat_path = Path(args.slat_manifest).expanduser().resolve()
    report_paths = [Path(value).expanduser().resolve() for value in args.native_report]
    output_selection = Path(args.output_selection).expanduser().resolve()
    output_lifting = Path(args.output_lifting_subset).expanduser().resolve()
    lifting = load_json(lifting_path)
    slat = load_json(slat_path)
    if lifting.get("format") != LIFTING_FORMAT or lifting.get("passed") is not True:
        raise RuntimeError("dev64 lifting manifest is not passed")
    if slat.get("format") != SLAT_FORMAT or int(slat.get("object_count", -1)) != 64:
        raise RuntimeError("dev64 SLat manifest contract differs")
    split = dict(lifting.get("objaverse2k_split", {}))
    if (
        split.get("name") != "dev"
        or split.get("train_dev_object_disjoint") is not True
        or int(split.get("object_count", -1)) != 64
    ):
        raise RuntimeError("lifting manifest is not frozen object-disjoint dev64")
    lifting_sha = sha256_file(lifting_path)
    slat_sha = sha256_file(slat_path)
    native_records, native_binding = validate_native_reports(report_paths, slat_sha=slat_sha)
    lifting_rows = {str(row["uid"]): row for row in lifting.get("samples", [])}
    selected = selected_dev_rows(slat)
    train_objects = set()
    checkpoint_path = Path(native_binding["checkpoint"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if sha256_file(checkpoint_path) != native_binding["checkpoint"]["sha256"]:
        raise RuntimeError("Objaverse2K checkpoint changed")
    train_objects.update(map(str, checkpoint.get("data_identity", {}).get("object_uids", [])))
    training_cache_path = Path(checkpoint["data_identity"]["cache_manifest"]).resolve()
    if sha256_file(training_cache_path) != checkpoint["data_identity"]["cache_manifest_sha256"]:
        raise RuntimeError("Objaverse2K training cache changed")
    training_cache = load_json(training_cache_path)
    training_meshes = {
        str(row.get("source_glb_sha256", ""))
        for row in training_cache.get("samples", [])
        if row.get("source_glb_sha256")
    }

    samples: list[dict[str, Any]] = []
    lifting_subset_rows: list[dict[str, Any]] = []
    for position, (object_uid, uid, seed_map) in enumerate(selected):
        if object_uid in train_objects:
            raise RuntimeError(f"dev64 object overlaps Objaverse2K training: {object_uid}")
        lifting_row = lifting_rows.get(uid)
        if lifting_row is None or str(lifting_row.get("object_uid")) != object_uid:
            raise RuntimeError(f"dev64 lifting identity differs: {uid}")
        cache_path = resolve_from(lifting_path.parent, lifting_row["cache_file"])
        if sha256_file(cache_path) != lifting_row.get("cache_file_sha256"):
            raise RuntimeError(f"dev64 lifting cache changed: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        image_paths = [Path(value).resolve() for value in cache.get("image_paths", [])]
        mask_paths = [Path(value).resolve() for value in cache.get("mask_paths", [])]
        view_ids = tensor_int_list(cache.get("view_ids", []))
        if not image_paths or len(image_paths) != len(mask_paths) or len(image_paths) != len(view_ids):
            raise RuntimeError(f"dev64 visible view count differs: {uid}")
        for path in [*image_paths, *mask_paths]:
            if not path.is_file():
                raise FileNotFoundError(path)
        seed42 = slat["samples"][seed_map[42]]
        latent_path = Path(seed42["ss_latent"]).resolve()
        with np.load(latent_path) as latent:
            source_glb = Path(str(latent["source_glb"])).resolve()
        source_sha = sha256_file(source_glb)
        if source_sha != str(seed42["source_glb_sha256"]):
            raise RuntimeError(f"dev64 source GLB changed: {source_glb}")
        if source_sha in training_meshes:
            raise RuntimeError(f"dev64 source mesh overlaps Objaverse2K training: {source_glb}")
        native_meshes = {}
        for seed in JOINT_SEEDS:
            record = native_records[(object_uid, uid, seed)]
            branch = record["branches"]["full"]
            native_meshes[str(seed)] = {
                "mesh": str(Path(branch["mesh"]).resolve()),
                "mesh_sha256": str(branch["mesh_sha256"]),
            }
        visible = {
            "view_count": len(view_ids),
            "view_ids": view_ids,
            "source_view_indices": view_ids,
            "image_paths": [str(path) for path in image_paths],
            "mask_paths": [str(path) for path in mask_paths],
            "source_cache": str(cache_path),
            "source_cache_sha256": sha256_file(cache_path),
        }
        samples.append(
            {
                **dict(seed42),
                "dataset_source": "objaverse",
                "source_group": "objaverse2k_dev64",
                "visible_inputs": visible,
                "native_objaverse2k_meshes": native_meshes,
                "objaverse2k_dev64_selection": {
                    "position": position,
                    "expected_view_count": len(view_ids),
                    "support_seeds": list(JOINT_SEEDS),
                },
            }
        )
        lifting_subset_rows.append(dict(lifting_row))

    selected_uids = [str(row["uid"]) for row in samples]
    selected_objects = [str(row["object_uid"]) for row in samples]
    protocol = {
        "format": OBJAVERSE2K_DEV64_PROTOCOL_FORMAT,
        "scope": OBJAVERSE2K_DEV64_SCOPE,
        "formal": False,
        "training_overlap": False,
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "object_count": 64,
        "selected_uids": selected_uids,
        "selected_object_uids": selected_objects,
        "joint_seeds": list(JOINT_SEEDS),
        "source_lifting_manifest": str(lifting_path),
        "source_lifting_manifest_sha256": lifting_sha,
        "source_slat_manifest": str(slat_path),
        "source_slat_manifest_sha256": slat_sha,
        "native_binding": native_binding,
        "selection_rule": "existing frozen dev64 complete 42/43/44 object matrix",
        "limitations": "development set already used for checkpoint diagnostics; formal=false",
        "passed": True,
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    selection_payload = {
        "format": SELECTION_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "objaverse2k_dev64_protocol": protocol,
        "samples": samples,
    }
    subset_payload = dict(lifting)
    subset_payload.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_lifting_manifest": str(lifting_path),
            "source_lifting_manifest_sha256": lifting_sha,
            "output_dir": str(output_lifting.parent),
            "sample_count": len(lifting_subset_rows),
            "object_count": len(lifting_subset_rows),
            "samples": lifting_subset_rows,
            "passed": len(lifting_subset_rows) == 64,
        }
    )
    if output_selection.exists() or output_lifting.exists():
        if not args.resume:
            raise FileExistsError("dev64 selection outputs already exist")
        old_selection = load_json(output_selection)
        old_lifting = load_json(output_lifting)
        if (
            old_selection.get("objaverse2k_dev64_protocol", {}).get("protocol_sha256")
            != protocol["protocol_sha256"]
            or old_lifting.get("source_lifting_manifest_sha256") != lifting_sha
        ):
            raise RuntimeError("stale dev64 selection output")
    else:
        atomic_json(output_selection, selection_payload)
        atomic_json(output_lifting, subset_payload)
    validate_selection(selection_payload)
    print(
        json.dumps(
            {
                "passed": True,
                "formal": False,
                "training_object_disjoint": True,
                "objects": 64,
                "selection": str(output_selection),
                "lifting_subset": str(output_lifting),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
