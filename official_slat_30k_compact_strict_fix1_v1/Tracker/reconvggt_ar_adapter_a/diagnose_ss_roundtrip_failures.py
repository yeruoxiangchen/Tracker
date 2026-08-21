#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from reconvggt_ar_adapter_a.audit_pointpose_ss_cache import load_decoder


GRID_SHAPE = (64, 64, 64)
GRID_NUMEL = 64 ** 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coord_linear(coords: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coords, dtype=np.int64)[:, -3:]
    return xyz[:, 0] * (64 * 64) + xyz[:, 1] * 64 + xyz[:, 2]


def linear_coords(indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    x = indices // (64 * 64)
    rem = indices % (64 * 64)
    y = rem // 64
    z = rem % 64
    return np.stack((x, y, z), axis=-1).astype(np.int32)


def occupancy_from_coords(coords: np.ndarray) -> np.ndarray:
    occupancy = np.zeros((GRID_NUMEL,), dtype=bool)
    if len(coords):
        occupancy[np.unique(coord_linear(coords))] = True
    return occupancy


def occupancy_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    intersection = int(np.logical_and(pred, target).sum())
    pred_count = int(pred.sum())
    target_count = int(target.sum())
    union = pred_count + target_count - intersection
    return {
        "pred_count": pred_count,
        "target_count": target_count,
        "intersection": intersection,
        "iou": float(intersection / union) if union else 1.0,
        "precision": float(intersection / pred_count) if pred_count else 1.0,
        "recall": float(intersection / target_count) if target_count else 1.0,
    }


def quantile_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def nearest_distance_stats(source: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if not len(source) or not len(target):
        return {"count": int(len(source))}

    distance, _ = cKDTree(target).query(source, k=1)
    return {
        "count": int(len(distance)),
        "mean": float(distance.mean()),
        "median": float(np.median(distance)),
        "p90": float(np.quantile(distance, 0.90)),
        "p99": float(np.quantile(distance, 0.99)),
        "max": float(distance.max()),
        "within_1_voxel_ratio": float((distance <= 1.01).mean()),
        "within_2_voxel_ratio": float((distance <= 2.01).mean()),
    }


def best_rank_cut(logits: np.ndarray, target_occ: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    target = np.asarray(target_occ, dtype=bool).reshape(-1)

    order = np.argsort(scores)[::-1]
    target_sorted = target[order].astype(np.int64)
    true_positive = np.cumsum(target_sorted)
    k = np.arange(1, len(scores) + 1, dtype=np.int64)
    target_count = int(target.sum())
    union = target_count + k - true_positive
    iou = true_positive / np.maximum(union, 1)

    best_index = int(np.argmax(iou))
    oracle_index = max(0, min(target_count - 1, len(scores) - 1))

    return {
        "best_rank_iou": float(iou[best_index]),
        "best_rank_k": int(best_index + 1),
        "best_rank_score": float(scores[order[best_index]]),
        "oracle_count_iou": float(iou[oracle_index]),
        "oracle_count_k": int(oracle_index + 1),
        "oracle_count_score": float(scores[order[oracle_index]]),
    }


def transformed_variants(coords: np.ndarray) -> dict[str, np.ndarray]:
    coords = np.asarray(coords, dtype=np.int32)
    return {
        "identity": coords,
        "swap_xy": coords[:, [1, 0, 2]],
        "swap_xz": coords[:, [2, 1, 0]],
        "swap_yz": coords[:, [0, 2, 1]],
        "flip_x": np.stack((63 - coords[:, 0], coords[:, 1], coords[:, 2]), axis=-1),
        "flip_y": np.stack((coords[:, 0], 63 - coords[:, 1], coords[:, 2]), axis=-1),
        "flip_z": np.stack((coords[:, 0], coords[:, 1], 63 - coords[:, 2]), axis=-1),
        "flip_xyz": 63 - coords,
    }


def transform_audit(pred_coords: np.ndarray, target_occ: np.ndarray) -> dict[str, Any]:
    rows = {}
    for name, transformed in transformed_variants(pred_coords).items():
        rows[name] = occupancy_metrics(occupancy_from_coords(transformed), target_occ)
    best_name = max(rows, key=lambda key: float(rows[key]["iou"]))
    return {
        "best_transform": best_name,
        "best_iou": float(rows[best_name]["iou"]),
        "rows": rows,
    }


def translation_audit(pred_coords: np.ndarray, target_occ: np.ndarray, radius: int = 2) -> dict[str, Any]:
    best = {"shift": [0, 0, 0], "iou": -1.0}
    for dx, dy, dz in itertools.product(
        range(-radius, radius + 1),
        range(-radius, radius + 1),
        range(-radius, radius + 1),
    ):
        shifted = pred_coords + np.asarray([dx, dy, dz], dtype=np.int32)
        valid = ((shifted >= 0) & (shifted < 64)).all(axis=1)
        metrics = occupancy_metrics(occupancy_from_coords(shifted[valid]), target_occ)
        if float(metrics["iou"]) > float(best["iou"]):
            best = {
                "shift": [int(dx), int(dy), int(dz)],
                **metrics,
            }
    return best


def load_manifest_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        root = Path(payload.get("output_dir", path.parent))
        for row in payload.get("samples", []):
            uid = str(row["uid"])
            if uid in rows:
                raise ValueError(f"duplicate UID across cache manifests: {uid}")
            rows[uid] = {
                **row,
                "_cache_manifest": str(path),
                "_cache_root": str(root),
            }
    return rows


def npz_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    result = {}
    for key in data.files:
        array = np.asarray(data[key])
        entry: dict[str, Any] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if array.ndim == 0:
            try:
                entry["value"] = array.item()
            except Exception:
                pass
        result[key] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_manifests", nargs="+", required=True)
    parser.add_argument("--uids_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check_fp32", action="store_true")
    args = parser.parse_args()

    cache_rows = load_manifest_rows([Path(path) for path in args.cache_manifests])
    uids = [
        line.strip()
        for line in Path(args.uids_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = sorted(set(uids) - set(cache_rows))
    if missing:
        raise KeyError(f"UIDs absent from supplied manifests: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    decoder = load_decoder(args.pretrained, device)
    native_dtype = next(decoder.parameters()).dtype

    reports: list[dict[str, Any]] = []
    retained: dict[str, dict[str, Any]] = {}

    for uid in uids:
        row = cache_rows[uid]
        latent_path = Path(row["ss_latent"])
        if not latent_path.is_file():
            raise FileNotFoundError(latent_path)

        with np.load(latent_path, allow_pickle=False) as data:
            metadata = npz_metadata(data)
            z_original = np.asarray(data["z"])
            target_original = np.asarray(data["target_coords"])

        z = z_original
        if z.ndim == 5 and z.shape[0] == 1:
            z = z[0]
        z = np.asarray(z, dtype=np.float32)
        target_coords = np.asarray(target_original, dtype=np.int32)[:, -3:]

        target_occ = occupancy_from_coords(target_coords)
        latent = torch.from_numpy(z[None]).to(device=device, dtype=native_dtype)

        with torch.no_grad():
            logits_tensor = decoder(latent).float()[0, 0]
        logits = logits_tensor.detach().cpu().numpy()
        pred_occ = logits.reshape(-1) > 0
        pred_coords = linear_coords(np.flatnonzero(pred_occ))

        false_positive_occ = pred_occ & ~target_occ
        false_negative_occ = target_occ & ~pred_occ
        false_positive_coords = linear_coords(np.flatnonzero(false_positive_occ))
        false_negative_coords = linear_coords(np.flatnonzero(false_negative_occ))

        report: dict[str, Any] = {
            "uid": uid,
            "object_uid": str(row.get("object_uid", "")),
            "cache_manifest": row["_cache_manifest"],
            "latent_path": str(latent_path),
            "latent_sha256": sha256_file(latent_path),
            "npz": metadata,
            "z_loaded_shape": list(z.shape),
            "z_loaded_dtype": str(z_original.dtype),
            "target_coord_array_count": int(len(target_coords)),
            "target_coord_unique_count": int(target_occ.sum()),
            "target_coord_duplicate_count": int(len(target_coords) - target_occ.sum()),
            "decoder_native_dtype": str(native_dtype),
            "threshold_0": occupancy_metrics(pred_occ, target_occ),
            "rank_cut": best_rank_cut(logits, target_occ),
            "logit_stats": {
                "all": quantile_stats(logits),
                "target": quantile_stats(logits.reshape(-1)[target_occ]),
                "non_target": quantile_stats(logits.reshape(-1)[~target_occ]),
                "false_positive": quantile_stats(logits.reshape(-1)[false_positive_occ]),
                "false_negative": quantile_stats(logits.reshape(-1)[false_negative_occ]),
            },
            "spatial_error": {
                "false_positive_to_target": nearest_distance_stats(
                    false_positive_coords, target_coords
                ),
                "false_negative_to_prediction": nearest_distance_stats(
                    false_negative_coords, pred_coords
                ),
                "named_transform": transform_audit(pred_coords, target_occ),
                "best_translation_radius2": translation_audit(
                    pred_coords, target_occ, radius=2
                ),
            },
        }

        np.savez_compressed(
            output_dir / f"{uid}_diagnostic.npz",
            logits=logits.astype(np.float16),
            target_coords=target_coords.astype(np.int16),
            decoded_coords=pred_coords.astype(np.int16),
            false_positive_coords=false_positive_coords.astype(np.int16),
            false_negative_coords=false_negative_coords.astype(np.int16),
        )

        reports.append(report)
        retained[uid] = {
            "object_uid": report["object_uid"],
            "z": z,
            "target_coords": target_coords,
            "decoded_coords": pred_coords,
            "native_logits": logits,
        }

        print(
            f"[diagnose] {uid} "
            f"iou0={report['threshold_0']['iou']:.6f} "
            f"best_rank={report['rank_cut']['best_rank_iou']:.6f} "
            f"best_transform={report['spatial_error']['named_transform']['best_transform']} "
            f"best_shift={report['spatial_error']['best_translation_radius2']['shift']}",
            flush=True,
        )

    pair_reports = []
    by_object: dict[str, list[str]] = defaultdict(list)
    for uid, state in retained.items():
        by_object[str(state["object_uid"])].append(uid)

    for object_uid, object_uids in by_object.items():
        if len(object_uids) < 2:
            continue
        for left_uid, right_uid in itertools.combinations(sorted(object_uids), 2):
            left = retained[left_uid]
            right = retained[right_uid]
            target_left = occupancy_from_coords(left["target_coords"])
            target_right = occupancy_from_coords(right["target_coords"])
            decoded_left = occupancy_from_coords(left["decoded_coords"])
            decoded_right = occupancy_from_coords(right["decoded_coords"])
            pair_reports.append(
                {
                    "object_uid": object_uid,
                    "left_uid": left_uid,
                    "right_uid": right_uid,
                    "z_max_abs_diff": float(
                        np.max(np.abs(left["z"] - right["z"]))
                    ),
                    "z_mean_abs_diff": float(
                        np.mean(np.abs(left["z"] - right["z"]))
                    ),
                    "target_coord_iou": occupancy_metrics(
                        target_left, target_right
                    )["iou"],
                    "decoded_coord_iou": occupancy_metrics(
                        decoded_left, decoded_right
                    )["iou"],
                }
            )

    if args.check_fp32:
        try:
            decoder.float()
            for report in reports:
                uid = str(report["uid"])
                state = retained[uid]
                latent = torch.from_numpy(state["z"][None]).to(
                    device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    logits_fp32 = decoder(latent).float()[0, 0].cpu().numpy()
                target_occ = occupancy_from_coords(state["target_coords"])
                pred_fp32 = logits_fp32.reshape(-1) > 0
                report["fp32_decoder"] = {
                    "threshold_0": occupancy_metrics(pred_fp32, target_occ),
                    "native_vs_fp32_logit_max_abs": float(
                        np.max(np.abs(state["native_logits"] - logits_fp32))
                    ),
                    "native_vs_fp32_logit_mean_abs": float(
                        np.mean(np.abs(state["native_logits"] - logits_fp32))
                    ),
                }
        except Exception as exc:
            for report in reports:
                report["fp32_decoder_error"] = f"{type(exc).__name__}: {exc}"

    final_report = {
        "pretrained": args.pretrained,
        "uids": uids,
        "samples": reports,
        "same_object_pairs": pair_reports,
    }
    (output_dir / "report.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[diagnose] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()